#!/usr/bin/env python3
"""
build_graph_inputs_v4_rawcos.py

Same as your v4 build, but CNA edge weights use RAW cosine similarity
instead of heat-kernel(exp(-d^2/2sigma^2)) which tends to saturate near 1.

Edges:
  - kNN on CNA with cosine distance
  - edge_weight = cosine_similarity = 1 - cosine_distance
  - optional sharpening: (1 - d)^p  (p>=1). p=1 is pure raw similarity.

Node features:
  - signatures (CP10K -> log1p) matrix: (cells x genes_used)
  - NOTE: unlike older versions, rna_feat is NOT (N x N). It is (N x G).

Outputs NPZ:
  - cell_names (N,)
  - edge_index (2, E)
  - edge_weight (E,)
  - rna_feat (N, G) float32
  - used_genes (G,) object strings
  - params (json string)
"""

from __future__ import annotations
import json
import argparse
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

DTYPE = np.float32


def load_rna_mtx(genes_txt: str, cells_csv: str, mtx_path: str):
    cells_df = pd.read_csv(cells_csv)
    cell_names = cells_df["cell_name"].astype(str).values

    genes = pd.read_csv(genes_txt, header=None)[0].astype(str).str.replace('"', "", regex=False).tolist()

    Xgxc = mmread(mtx_path).tocsr()  # genes x cells
    G, N = Xgxc.shape
    if G != len(genes):
        raise ValueError(f"Genes mismatch: mtx rows={G} but Genes.txt={len(genes)}")
    if N != len(cell_names):
        raise ValueError(f"Cells mismatch: mtx cols={N} but Cells.csv rows={len(cell_names)}")

    X = Xgxc.transpose().tocsr()  # cells x genes
    return X, genes, cells_df, cell_names


def cp10k_log1p(X_csr: sparse.csr_matrix, target_sum: float = 1e4):
    X = X_csr.tocsr().copy()
    rs = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    rs[rs == 0] = 1.0
    inv = (target_sum / rs).astype(np.float32)
    X = sparse.diags(inv) @ X
    X.data = np.log1p(X.data).astype(np.float32, copy=False)
    return X


def load_signature_genes(sig_genes_txt: str):
    genes = []
    with open(sig_genes_txt, "r") as f:
        for line in f:
            g = line.strip()
            if g:
                genes.append(g)
    # unique preserving order
    seen = set()
    out = []
    for g in genes:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def select_genes_from_sparse(X: sparse.csr_matrix, all_genes: list[str], wanted_genes: list[str]):
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    idx = []
    used = []
    for g in wanted_genes:
        if g in gene_to_idx:
            idx.append(gene_to_idx[g])
            used.append(g)
    if len(idx) == 0:
        raise ValueError("No signature genes found in Genes.txt")
    Xs = X[:, idx].tocsr()
    return Xs, used


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


def knn_edges_cosine_similarity(X: np.ndarray, k: int, n_jobs: int = 1, power: float = 1.0):
    """
    kNN with cosine distance. Weight = (1 - dist)^power
    dist in [0,2] but for cosine distance with normalized vectors usually [0,2].
    If dist>1, similarity becomes negative; we clip to 0 for stability.
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=n_jobs)
    nbrs.fit(X)
    dists, idxs = nbrs.kneighbors(X, return_distance=True)

    dists = dists[:, 1:]
    idxs = idxs[:, 1:]

    n = X.shape[0]
    src = np.repeat(np.arange(n, dtype=np.int64), k)
    dst = idxs.reshape(-1).astype(np.int64)
    dist = dists.reshape(-1).astype(np.float32)

    sim = 1.0 - dist
    sim = np.clip(sim, 0.0, 1.0).astype(np.float32)
    if power is not None and float(power) != 1.0:
        sim = np.power(sim, float(power)).astype(np.float32)

    edge_index = np.vstack([src, dst])
    edge_weight = sim.astype(DTYPE)
    return edge_index, edge_weight


def symmetrize_and_coalesce(edge_index, edge_weight, n_nodes: int):
    src, dst = edge_index
    rev_edge_index = np.vstack([dst, src])
    rev_edge_weight = edge_weight.copy()

    ei = np.hstack([edge_index, rev_edge_index])
    ew = np.hstack([edge_weight, rev_edge_weight])

    key = ei[0].astype(np.int64) * n_nodes + ei[1].astype(np.int64)
    order = np.argsort(key)
    key = key[order]
    ei = ei[:, order]
    ew = ew[order]

    _, idx_start = np.unique(key, return_index=True)
    ew_max = np.maximum.reduceat(ew, idx_start)
    ei_uniq = ei[:, idx_start]
    return ei_uniq.astype(np.int64), ew_max.astype(DTYPE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cna_csv", required=True)
    ap.add_argument("--cells_csv", required=True)
    ap.add_argument("--genes_txt", required=True)
    ap.add_argument("--mtx", required=True)

    ap.add_argument("--sig_genes_txt", required=True, help="Signature genes list (one per line).")

    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--n_jobs", type=int, default=1)

    # keep sigma arg out; not used here
    ap.add_argument("--power", type=float, default=1.0, help="edge_weight = (1 - cosine_dist)^power. 1.0 = raw sim.")

    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # RNA
    X_rna, genes, cells_df, cell_names = load_rna_mtx(args.genes_txt, args.cells_csv, args.mtx)
    print(f"[RNA] raw counts shape: {X_rna.shape}")

    X_rna = cp10k_log1p(X_rna, target_sum=1e4)
    print(f"[RNA] CP10000 normalized + log1p done.")

    sig_genes = load_signature_genes(args.sig_genes_txt)
    X_sig, used_genes = select_genes_from_sparse(X_rna, genes, sig_genes)
    print(f"[RNA] Using signatures: found {len(used_genes)}/{len(sig_genes)}. X_used={X_sig.shape}")

    # use dense (N x G) as node features
    rna_feat = X_sig.toarray().astype(DTYPE, copy=False)
    print(f"[RNA] rna_feat (N x G): {rna_feat.shape} mode=signatures_cp10k_log1p")
    print("[RNA] feat min/median/max:",
          float(rna_feat.min()), float(np.median(rna_feat)), float(rna_feat.max()))

    # CNA
    X_cna = load_cna_csv(args.cna_csv, cell_order=cell_names)
    print(f"[CNA] shape: {X_cna.shape}")

    # edges from CNA, weights = raw cosine similarity
    edge_index, edge_weight = knn_edges_cosine_similarity(X_cna, k=args.k, n_jobs=args.n_jobs, power=args.power)
    edge_index, edge_weight = symmetrize_and_coalesce(edge_index, edge_weight, n_nodes=X_cna.shape[0])

    print(f"[CNA] edges: {edge_index.shape[1]} weight min/max={float(edge_weight.min()):.4f}/{float(edge_weight.max()):.4f}")

    params = {
        "edges": {
            "k": args.k,
            "metric": "cosine",
            "weight": "raw_cosine_similarity",
            "power": args.power,
            "symmetrize": True,
            "coalesce": "max",
        },
        "rna_features": {
            "type": "signatures",
            "transform": "cp10k_log1p",
            "n_genes": len(used_genes),
        },
        "canonical_order": "Cells.csv order (RNA), CNA reordered to match",
    }

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
