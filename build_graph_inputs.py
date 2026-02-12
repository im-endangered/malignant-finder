#!/usr/bin/env python3
"""
build_graph_inputs.py  (UPDATED VERSION)

KEY CHANGES vs your old version:
1) CNA:
   - Use ALL genes (as before)
   - ADD: per-gene z-score normalization before kNN
   - Keep weighted kNN (cosine + heat kernel)

2) RNA:
   - REPLACE HVGs with your 1756 cancer signature genes
   - Keep log1p transform
   - Compute cell-cell cosine similarity as features
   - ADD: normalize similarity by its max absolute value

OUTPUT:
  gao2021_breast_cna_edges_rna_features.npz
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


# ===================== CNA GRAPH =====================

def cosine_knn_heat_kernel(X, k: int, sigma: float, n_jobs: int = 1):
    """
    Build a directed kNN graph using cosine distance, then convert distances
    to weights with heat kernel.
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=n_jobs)
    nbrs.fit(X)
    dists, idxs = nbrs.kneighbors(X, return_distance=True)

    # Drop self neighbor at position 0
    dists = dists[:, 1:]
    idxs = idxs[:, 1:]

    n = X.shape[0]
    src = np.repeat(np.arange(n, dtype=np.int64), k)
    dst = idxs.reshape(-1).astype(np.int64)
    dist = dists.reshape(-1).astype(np.float32)

    # Convert distance -> similarity in [0,1]
    sim = 1.0 - dist
    sim = np.clip(sim, 0.0, 1.0)

    # Heat kernel weights based on similarity
    if sigma <= 0:
        raise ValueError("sigma must be > 0 for heat kernel.")
    w = np.exp(-(1.0 - sim) / (sigma ** 2)).astype(DTYPE)

    edge_index = np.vstack([src, dst])
    edge_weight = w
    return edge_index, edge_weight


def symmetrize_and_coalesce(edge_index, edge_weight, n_nodes: int):
    """Make graph undirected and coalesce duplicate edges by max weight."""
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

    uniq, idx_start = np.unique(key, return_index=True)
    ew_max = np.maximum.reduceat(ew, idx_start)
    ei_uniq = ei[:, idx_start]

    return ei_uniq.astype(np.int64), ew_max.astype(DTYPE)


def load_cna_csv(cna_csv_path: str, cell_order: np.ndarray):
    """Load CNA CSV (cells x genes) and reorder to match cell_order."""
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

    # === NEW: per-gene z-score normalization ===
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)

    return X


# ===================== RNA LOADING =====================

def load_rna_mtx(genes_txt: str, cells_csv: str, mtx_path: str):
    """Load RNA from MatrixMarket: genes x cells -> return cells x genes."""
    cells_df = pd.read_csv(cells_csv)
    cell_names = cells_df["cell_name"].astype(str).values

    genes = pd.read_csv(genes_txt, header=None)[0] \
        .astype(str).str.replace('"', "", regex=False).tolist()

    Xgxc = mmread(mtx_path).tocsr()  # genes x cells
    G, N = Xgxc.shape
    if G != len(genes):
        raise ValueError(f"Genes mismatch: mtx rows={G} but Genes.txt={len(genes)}")
    if N != len(cell_names):
        raise ValueError(f"Cells mismatch: mtx cols={N} but Cells.csv rows={len(cell_names)}")

    X = Xgxc.transpose().tocsr()  # cells x genes
    return X, genes, cells_df, cell_names


def log1p_sparse(X_csr: sparse.csr_matrix):
    X = X_csr.copy()
    X.data = np.log1p(X.data).astype(DTYPE, copy=False)
    return X


# ===================== RNA FEATURES =====================

def cosine_similarity_matrix_dense(X: sparse.csr_matrix, eps: float = 1e-12):
    """Compute dense cell-cell cosine similarity matrix and normalize it."""
    sq = X.multiply(X).sum(axis=1)
    norms = np.sqrt(np.asarray(sq).ravel() + eps).astype(np.float32)

    dots = (X @ X.T).toarray().astype(np.float32, copy=False)
    denom = norms[:, None] * norms[None, :]
    S = dots / denom

    S = np.clip(S, -1.0, 1.0)

    # === NEW: normalize by max absolute value ===
    S = S / (np.max(np.abs(S)) + 1e-8)

    return S.astype(DTYPE, copy=False)


# ===================== MAIN =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cna_csv", required=True)
    ap.add_argument("--cells_csv", required=True)
    ap.add_argument("--genes_txt", required=True)
    ap.add_argument("--mtx", required=True)
    ap.add_argument("--out", default="gao2021_breast_cna_edges_rna_features.npz")

    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--n_jobs", type=int, default=1)

    args = ap.parse_args()

    # ---------- Load RNA ----------
    X_rna, genes, cells_df, cell_names = load_rna_mtx(
        args.genes_txt, args.cells_csv, args.mtx
    )
    n_cells = X_rna.shape[0]
    print(f"[RNA] X shape (cells x genes): {X_rna.shape}")

    # log1p
    X_log = log1p_sparse(X_rna)
    print("[RNA] log1p done.")

    # === NEW: use 1756 signature genes instead of HVGs ===
    sig_genes = pd.read_csv(
        "signatures/master_signature_genes_unique.txt",
        header=None
    )[0].astype(str).tolist()

    sig_set = set(sig_genes)
    sig_idx = [i for i, g in enumerate(genes) if g in sig_set]

    X_sig = X_log[:, sig_idx].tocsr()
    used_genes = [genes[i] for i in sig_idx]

    print(f"[RNA] Using {len(used_genes)} signature genes, shape={X_sig.shape}")

    # Cosine similarity features
    print("[RNA] Computing dense cosine similarity...")
    rna_feat = cosine_similarity_matrix_dense(X_sig)
    print("[RNA] rna_feat shape:", rna_feat.shape)

    # ---------- Load CNA ----------
    X_cna = load_cna_csv(args.cna_csv, cell_order=cell_names)
    print("[CNA] X shape (cells x genes):", X_cna.shape)

    # ---------- Build CNA graph ----------
    print(f"[CNA] Building weighted kNN graph: k={args.k}, sigma={args.sigma}")
    edge_index, edge_weight = cosine_knn_heat_kernel(
        X_cna, k=args.k, sigma=args.sigma, n_jobs=args.n_jobs
    )

    edge_index, edge_weight = symmetrize_and_coalesce(
        edge_index, edge_weight, n_nodes=n_cells
    )
    print("[CNA] Final edges:", edge_index.shape[1])

    # ---------- Save ----------
    params = {
        "k": args.k,
        "sigma": args.sigma,
        "metric_edges": "cosine",
        "kernel": "heat",
        "rna_transform": "log1p",
        "rna_features": "cosine_similarity_from_1756_signature_genes",
        "cna_normalization": "per-gene zscore",
        "canonical_order": "Cells.csv order",
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
    print("[OUT] N =", n_cells, "E =", edge_index.shape[1], "rna_feat =", rna_feat.shape)


if __name__ == "__main__":
    main()
