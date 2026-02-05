#!/usr/bin/env python3
"""
build_graph_inputs.py

Build graph inputs in an SCGclust-like way:
- Edges: kNN graph from CNA (cosine distance) -> heat-kernel weights
- Node features: RNA cell-cell cosine similarity vectors after log1p + HVGs
- Consistent cell ordering: uses Cells.csv order as canonical node order

Outputs:
  gao2021_breast_cna_edges_rna_features.npz
    - cell_names (N,)
    - edge_index (2, E) int64  (src,dst)
    - edge_weight (E,) float32
    - rna_feat (N, N) float32  (cosine similarity vectors)
    - hvg_genes (G_hvg,) object strings
    - params (json string)
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


def cosine_knn_heat_kernel(X, k: int, sigma: float, n_jobs: int = 1):
    """
    Build a directed kNN graph using cosine distance, then convert distances
    to weights with heat kernel: w = exp( - d^2 / (2*sigma^2) ).
    Returns:
      edge_index: (2, E) int64
      edge_weight: (E,) float32
      distances: (E,) float32  (cosine distances)
    """
    # NearestNeighbors with cosine metric returns cosine distances in [0, 2]
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=n_jobs)
    nbrs.fit(X)
    dists, idxs = nbrs.kneighbors(X, return_distance=True)

    # Drop self neighbor at position 0 (distance 0)
    dists = dists[:, 1:]
    idxs = idxs[:, 1:]

    n = X.shape[0]
    src = np.repeat(np.arange(n, dtype=np.int64), k)
    dst = idxs.reshape(-1).astype(np.int64)
    dist = dists.reshape(-1).astype(np.float32)

    # heat kernel weights
    if sigma <= 0:
        raise ValueError("sigma must be > 0 for heat kernel.")
    w = np.exp(-(dist ** 2) / (2.0 * (sigma ** 2))).astype(DTYPE)

    edge_index = np.vstack([src, dst])
    edge_weight = w
    return edge_index, edge_weight, dist


def symmetrize_and_coalesce(edge_index, edge_weight, n_nodes: int):
    """
    Make graph undirected by adding reverse edges,
    then coalesce duplicates (i,j) using max weight.
    """
    src, dst = edge_index
    rev_edge_index = np.vstack([dst, src])
    rev_edge_weight = edge_weight.copy()

    ei = np.hstack([edge_index, rev_edge_index])
    ew = np.hstack([edge_weight, rev_edge_weight])

    # Coalesce by (src,dst) key
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
    """
    Load CNA CSV (cells x genes) and return matrix ordered by cell_order.
    CNA CSV contains a 'cell_name' column.
    """
    cna = pd.read_csv(cna_csv_path)
    if "cell_name" not in cna.columns:
        raise ValueError("CNA CSV must contain 'cell_name' column")

    # map row index by cell_name
    cna_index = pd.Index(cna["cell_name"].astype(str).values)
    wanted = pd.Index(cell_order.astype(str))
    if not wanted.isin(cna_index).all():
        missing = wanted[~wanted.isin(cna_index)].tolist()[:10]
        raise ValueError(f"Some RNA cells missing in CNA: example {missing}")

    # reorder
    row_pos = cna_index.get_indexer(wanted)
    X = cna.drop(columns=["cell_name"]).to_numpy(dtype=DTYPE, copy=False)
    X = X[row_pos, :]
    return X


def load_rna_mtx(genes_txt: str, cells_csv: str, mtx_path: str):
    """
    Load RNA from MatrixMarket:
      Exp_data_UMIcounts.mtx is genes x cells (per your header)
    Return:
      X (cells x genes) sparse CSR
      genes (list[str]) length G
      cells_df (DataFrame) length N
      cell_names (np.ndarray) length N in Cells.csv order
    """
    cells_df = pd.read_csv(cells_csv)
    cell_names = cells_df["cell_name"].astype(str).values

    # genes file has quotes
    genes = pd.read_csv(genes_txt, header=None)[0].astype(str).str.replace('"', "", regex=False).tolist()

    Xgxc = mmread(mtx_path).tocsr()  # genes x cells
    G, N = Xgxc.shape
    if G != len(genes):
        raise ValueError(f"Genes mismatch: mtx rows={G} but Genes.txt={len(genes)}")
    if N != len(cell_names):
        raise ValueError(f"Cells mismatch: mtx cols={N} but Cells.csv rows={len(cell_names)}")

    # convert to cells x genes
    X = Xgxc.transpose().tocsr()  # cells x genes
    return X, genes, cells_df, cell_names


def log1p_sparse(X_csr: sparse.csr_matrix):
    """
    log1p on sparse matrix in-place on data.
    """
    X = X_csr.copy()
    X.data = np.log1p(X.data).astype(DTYPE, copy=False)
    return X


def select_hvgs_variance(X_csr: sparse.csr_matrix, genes: list, n_hvgs: int):
    """
    Simple HVG selection by variance on log1p expression.
    Returns:
      X_hvg (cells x n_hvgs) CSR
      hvg_genes (list[str])
    """
    # Compute mean and mean of squares per gene on sparse data
    # Var = E[x^2] - (E[x])^2
    n = X_csr.shape[0]
    mean = np.asarray(X_csr.mean(axis=0)).ravel()
    # E[x^2]
    X_sq = X_csr.copy()
    X_sq.data = X_sq.data ** 2
    mean_sq = np.asarray(X_sq.mean(axis=0)).ravel()
    var = mean_sq - mean ** 2

    # choose top genes by variance
    if n_hvgs >= X_csr.shape[1]:
        idx = np.arange(X_csr.shape[1])
    else:
        idx = np.argpartition(-var, n_hvgs)[:n_hvgs]
        idx = idx[np.argsort(-var[idx])]

    hvg_genes = [genes[i] for i in idx.tolist()]
    X_hvg = X_csr[:, idx].tocsr()
    return X_hvg, hvg_genes


def cosine_similarity_matrix_dense(X: sparse.csr_matrix, eps: float = 1e-12):
    """
    Compute dense cosine similarity S = (X X^T) / (||x_i|| ||x_j||)
    X is cells x features (sparse).
    Returns dense float32 matrix (N x N).
    """
    # norms
    sq = X.multiply(X).sum(axis=1)
    norms = np.sqrt(np.asarray(sq).ravel() + eps).astype(np.float32)

    # dot products (sparse) -> dense
    dots = (X @ X.T).toarray().astype(np.float32, copy=False)

    # normalize
    denom = norms[:, None] * norms[None, :]
    S = dots / denom
    # numerical safety
    S = np.clip(S, -1.0, 1.0).astype(DTYPE, copy=False)
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cna_csv", required=True, help="Path to CNAs_Breast.csv")
    ap.add_argument("--cells_csv", required=True, help="Path to Breast/Cells.csv")
    ap.add_argument("--genes_txt", required=True, help="Path to Breast/Genes.txt")
    ap.add_argument("--mtx", required=True, help="Path to Breast/Exp_data_UMIcounts.mtx")
    ap.add_argument("--out", default="gao2021_breast_cna_edges_rna_features.npz")

    ap.add_argument("--k", type=int, default=20, help="k for kNN graph")
    ap.add_argument("--sigma", type=float, default=0.5, help="heat kernel sigma (cosine distance scale)")
    ap.add_argument("--hvgs", type=int, default=2000, help="number of HVGs")
    ap.add_argument("--n_jobs", type=int, default=1, help="n_jobs for kNN")

    args = ap.parse_args()

    # ---------- Load RNA ----------
    X_rna, genes, cells_df, cell_names = load_rna_mtx(args.genes_txt, args.cells_csv, args.mtx)
    n_cells = X_rna.shape[0]
    print(f"[RNA] X shape (cells x genes): {X_rna.shape}")
    print(f"[RNA] Cells: {len(cell_names)}  Genes: {len(genes)}")

    # Ordering sanity check: UMI sums vs Cells.csv complexity
    cell_sums = np.asarray(X_rna.sum(axis=1)).ravel().astype(np.float64)
    comp = cells_df["complexity"].to_numpy(dtype=np.float64, copy=False)
    corr = np.corrcoef(cell_sums, comp)[0, 1]
    print("[RNA] UMI per cell min/median/max:", float(cell_sums.min()), float(np.median(cell_sums)), float(cell_sums.max()))
    print("[RNA] corr(UMI_sums, Cells.csv complexity):", float(corr))

    # log1p
    X_log = log1p_sparse(X_rna)
    print("[RNA] log1p done. nnz:", X_log.nnz)

    # HVGs
    X_hvg, hvg_genes = select_hvgs_variance(X_log, genes, args.hvgs)
    print(f"[RNA] HVGs selected: {len(hvg_genes)}  X_hvg shape: {X_hvg.shape}")

    # Cosine similarity vectors as features (SCGclust-like)
    print("[RNA] Computing dense cosine similarity (this may take a bit)...")
    rna_feat = cosine_similarity_matrix_dense(X_hvg)
    print("[RNA] rna_feat shape:", rna_feat.shape,
          "min/max:", float(rna_feat.min()), float(rna_feat.max()),
          "diag mean:", float(np.mean(np.diag(rna_feat))))

    # ---------- Load CNA (aligned to RNA cell order) ----------
    X_cna = load_cna_csv(args.cna_csv, cell_order=cell_names)
    print("[CNA] X shape (cells x genes):", X_cna.shape,
          "min/max:", float(X_cna.min()), float(X_cna.max()),
          "mean/std:", float(X_cna.mean()), float(X_cna.std()))

    # ---------- Build CNA graph ----------
    print(f"[CNA] Building kNN graph: k={args.k}, metric=cosine, sigma={args.sigma}")
    edge_index, edge_weight, dist = cosine_knn_heat_kernel(X_cna, k=args.k, sigma=args.sigma, n_jobs=args.n_jobs)
    print("[CNA] Directed edges:", edge_index.shape[1],
          "weight min/max:", float(edge_weight.min()), float(edge_weight.max()))

    # Symmetrize + coalesce
    edge_index, edge_weight = symmetrize_and_coalesce(edge_index, edge_weight, n_nodes=n_cells)
    print("[CNA] Edges after sym+coalesce:", edge_index.shape[1],
          "weight min/max:", float(edge_weight.min()), float(edge_weight.max()))

    # ---------- Save ----------
    params = {
        "k": args.k,
        "sigma": args.sigma,
        "hvgs": args.hvgs,
        "metric_edges": "cosine",
        "kernel": "heat",
        "rna_transform": "log1p",
        "rna_features": "cosine_similarity_vectors",
        "canonical_order": "Cells.csv order (RNA), CNA reordered to match",
    }

    np.savez_compressed(
        args.out,
        cell_names=cell_names.astype(object),
        edge_index=edge_index.astype(np.int64),
        edge_weight=edge_weight.astype(DTYPE),
        rna_feat=rna_feat.astype(DTYPE),
        hvg_genes=np.array(hvg_genes, dtype=object),
        params=json.dumps(params),
    )
    print("[DONE] wrote:", args.out)
    print("[OUT] N =", n_cells, "E =", edge_index.shape[1], "rna_feat =", rna_feat.shape)


if __name__ == "__main__":
    main()
