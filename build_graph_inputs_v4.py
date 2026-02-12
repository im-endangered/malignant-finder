#!/usr/bin/env python3
"""
build_graph_inputs_v4.py

V4 (fixed): CNA edges + RNA signature node features with proper normalization.

Edges:
  - CNA kNN (cosine distance) -> heat kernel weights (sigma)

Node features:
  - RNA: library-size normalize to target_sum (CP10K) -> log1p
  - subset to signature genes (preferred) or HVGs fallback

Outputs (.npz):
  - cell_names (N,)
  - edge_index (2, E)
  - edge_weight (E,)
  - rna_feat (N, G)  <-- node features (NOT NxN)
  - used_genes (G,)
  - params (json)
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

DTYPE = np.float32


# ----------------------------
# CNA graph utilities
# ----------------------------
def cosine_knn_heat_kernel(X, k: int, sigma: float, n_jobs: int = 1):
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=n_jobs)
    nbrs.fit(X)
    dists, idxs = nbrs.kneighbors(X, return_distance=True)

    # drop self
    dists = dists[:, 1:]
    idxs = idxs[:, 1:]

    n = X.shape[0]
    src = np.repeat(np.arange(n, dtype=np.int64), k)
    dst = idxs.reshape(-1).astype(np.int64)
    dist = dists.reshape(-1).astype(np.float32)

    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    w = np.exp(-(dist ** 2) / (2.0 * (sigma ** 2))).astype(DTYPE)

    edge_index = np.vstack([src, dst])
    edge_weight = w
    return edge_index, edge_weight


def symmetrize_and_coalesce(edge_index, edge_weight, n_nodes: int):
    src, dst = edge_index
    ei = np.hstack([edge_index, np.vstack([dst, src])])
    ew = np.hstack([edge_weight, edge_weight.copy()])

    key = ei[0].astype(np.int64) * n_nodes + ei[1].astype(np.int64)
    order = np.argsort(key)
    key = key[order]
    ei = ei[:, order]
    ew = ew[order]

    _, idx_start = np.unique(key, return_index=True)
    ew_max = np.maximum.reduceat(ew, idx_start)
    ei_uniq = ei[:, idx_start]
    return ei_uniq.astype(np.int64), ew_max.astype(DTYPE)


def load_cna_csv(cna_csv_path: str, cell_order: np.ndarray):
    cna = pd.read_csv(cna_csv_path)
    if "cell_name" not in cna.columns:
        raise ValueError("CNA CSV must contain 'cell_name' column")

    cna_index = pd.Index(cna["cell_name"].astype(str).values)
    wanted = pd.Index(cell_order.astype(str))
    if not wanted.isin(cna_index).all():
        missing = wanted[~wanted.isin(cna_index)].tolist()[:10]
        raise ValueError(f"Some RNA cells missing in CNA: example {missing}")

    row_pos = cna_index.get_indexer(wanted)
    X = cna.drop(columns=["cell_name"]).to_numpy(dtype=DTYPE, copy=False)
    X = X[row_pos, :]
    return X


# ----------------------------
# RNA utilities
# ----------------------------
def load_rna_mtx(genes_txt: str, cells_csv: str, mtx_path: str):
    cells_df = pd.read_csv(cells_csv)
    cell_names = cells_df["cell_name"].astype(str).values

    genes = (
        pd.read_csv(genes_txt, header=None)[0]
        .astype(str)
        .str.replace('"', "", regex=False)
        .tolist()
    )

    Xgxc = mmread(mtx_path).tocsr()  # genes x cells
    G, N = Xgxc.shape
    if G != len(genes):
        raise ValueError(f"Genes mismatch: mtx rows={G} but Genes.txt={len(genes)}")
    if N != len(cell_names):
        raise ValueError(f"Cells mismatch: mtx cols={N} but Cells.csv rows={len(cell_names)}")

    X = Xgxc.transpose().tocsr()  # cells x genes
    return X, genes, cells_df, cell_names


def cp10k_normalize_sparse(X_csr: sparse.csr_matrix, target_sum: float = 1e4, eps: float = 1e-12):
    """
    Scale each row to sum to target_sum (like Scanpy normalize_total).
    """
    X = X_csr.tocsr(copy=True)
    rs = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    scale = (target_sum / (rs + eps)).astype(np.float32)
    X = sparse.diags(scale) @ X
    return X.astype(DTYPE)


def log1p_sparse(X_csr: sparse.csr_matrix):
    X = X_csr.copy()
    X.data = np.log1p(X.data).astype(DTYPE, copy=False)
    return X


def read_gene_list(path: str):
    genes = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            genes.append(s)
    return genes


def subset_genes_sparse(X_csr: sparse.csr_matrix, all_genes: list[str], wanted_genes: list[str]):
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    idx = []
    used = []
    for g in wanted_genes:
        j = gene_to_idx.get(g)
        if j is not None:
            idx.append(j)
            used.append(g)
    if len(idx) == 0:
        return X_csr[:, []], [], 0, len(wanted_genes)
    X_sub = X_csr[:, np.array(idx, dtype=np.int64)].tocsr()
    return X_sub, used, len(idx), len(wanted_genes)


def select_hvgs_variance(X_csr: sparse.csr_matrix, genes: list[str], n_hvgs: int):
    mean = np.asarray(X_csr.mean(axis=0)).ravel()
    X_sq = X_csr.copy()
    X_sq.data = X_sq.data ** 2
    mean_sq = np.asarray(X_sq.mean(axis=0)).ravel()
    var = mean_sq - mean ** 2

    if n_hvgs >= X_csr.shape[1]:
        idx = np.arange(X_csr.shape[1])
    else:
        idx = np.argpartition(-var, n_hvgs)[:n_hvgs]
        idx = idx[np.argsort(-var[idx])]

    hvg_genes = [genes[i] for i in idx.tolist()]
    X_hvg = X_csr[:, idx].tocsr()
    return X_hvg, hvg_genes


def zscore_dense(X: np.ndarray, eps: float = 1e-6):
    """
    Z-score each gene (column) across cells.
    """
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    return (X - mu) / (sd + eps)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cna_csv", required=True)
    ap.add_argument("--cells_csv", required=True)
    ap.add_argument("--genes_txt", required=True)
    ap.add_argument("--mtx", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--n_jobs", type=int, default=1)

    ap.add_argument("--sig_genes_txt", default=None)
    ap.add_argument("--hvgs", type=int, default=2000)
    ap.add_argument("--min_sig_found", type=int, default=50)

    # NEW: proper RNA normalization
    ap.add_argument("--target_sum", type=float, default=1e4, help="CP10K target sum (per cell)")
    ap.add_argument("--no_log1p", action="store_true")
    ap.add_argument("--zscore", action="store_true", help="Z-score genes after log1p (recommended to prevent depth effects)")

    args = ap.parse_args()

    # ----- RNA -----
    X_rna, genes, _, cell_names = load_rna_mtx(args.genes_txt, args.cells_csv, args.mtx)
    N = X_rna.shape[0]
    print(f"[RNA] raw counts shape: {X_rna.shape}")

    X = cp10k_normalize_sparse(X_rna, target_sum=args.target_sum)
    print(f"[RNA] CP{int(args.target_sum)} normalized. row-sum median={float(np.median(np.asarray(X.sum(axis=1)).ravel())):.2f}")

    if not args.no_log1p:
        X = log1p_sparse(X)
        print("[RNA] log1p done.")

    used_genes = None
    X_used = None
    feature_mode = None

    if args.sig_genes_txt is not None:
        sig_list = read_gene_list(args.sig_genes_txt)
        X_sig, sig_used, n_found, n_total = subset_genes_sparse(X, genes, sig_list)
        print(f"[RNA] Using signatures: found {n_found}/{n_total}. X_used={X_sig.shape}")
        if n_found >= args.min_sig_found:
            X_used = X_sig
            used_genes = sig_used
            feature_mode = "signatures_cp10k_log1p"
        else:
            print(f"[RNA] WARNING: only {n_found} signatures found (<{args.min_sig_found}) -> fallback HVGs")

    if X_used is None:
        X_hvg, hvg_genes = select_hvgs_variance(X, genes, args.hvgs)
        print(f"[RNA] Using HVGs: {len(hvg_genes)}. X_used={X_hvg.shape}")
        X_used = X_hvg
        used_genes = hvg_genes
        feature_mode = "hvgs_cp10k_log1p"

    rna_feat = X_used.toarray().astype(DTYPE, copy=False)
    if args.zscore:
        rna_feat = zscore_dense(rna_feat).astype(DTYPE, copy=False)
        feature_mode += "_zscore"

    print(f"[RNA] rna_feat (N x G): {rna_feat.shape} mode={feature_mode}")
    print("[RNA] feat min/median/max:",
          float(rna_feat.min()), float(np.median(rna_feat)), float(rna_feat.max()))

    # ----- CNA edges -----
    X_cna = load_cna_csv(args.cna_csv, cell_order=cell_names)
    print(f"[CNA] shape: {X_cna.shape}")
    edge_index, edge_weight = cosine_knn_heat_kernel(X_cna, k=args.k, sigma=args.sigma, n_jobs=args.n_jobs)
    edge_index, edge_weight = symmetrize_and_coalesce(edge_index, edge_weight, n_nodes=N)
    print(f"[CNA] edges: {edge_index.shape[1]} weight min/max={float(edge_weight.min()):.4f}/{float(edge_weight.max()):.4f}")

    params = {
        "k": args.k,
        "sigma": args.sigma,
        "metric_edges": "cosine",
        "kernel": "heat",
        "rna_norm": f"CP{int(args.target_sum)}",
        "rna_log1p": (not args.no_log1p),
        "rna_zscore": args.zscore,
        "rna_feat_mode": feature_mode,
        "sig_genes_txt": args.sig_genes_txt,
        "hvgs": args.hvgs,
        "canonical_order": "Cells.csv order (RNA), CNA reordered to match",
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        cell_names=cell_names.astype(object),
        edge_index=edge_index.astype(np.int64),
        edge_weight=edge_weight.astype(DTYPE),
        rna_feat=rna_feat.astype(DTYPE),
        used_genes=np.array(used_genes, dtype=object),
        params=json.dumps(params),
    )
    print("[DONE] wrote:", args.out)


if __name__ == "__main__":
    main()
