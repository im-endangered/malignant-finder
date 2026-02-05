from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from typing import Literal
from sklearn.metrics.pairwise import cosine_similarity


Metric = Literal["cosine", "pearson", "dot", "euclidean"]


def edges_to_csr(edge_index: np.ndarray, edge_weight: np.ndarray, n_nodes: int) -> sp.csr_matrix:
    """
    Convert (2,E) edge_index + (E,) edge_weight into csr adjacency.
    """
    if edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must be shape (2,E), got {edge_index.shape}")
    row = edge_index[0].astype(np.int64)
    col = edge_index[1].astype(np.int64)
    val = edge_weight.astype(np.float32)
    return sp.csr_matrix((val, (row, col)), shape=(n_nodes, n_nodes), dtype=np.float32)


def build_cell_graph(X: np.ndarray, metric: Metric = "cosine",
                     knn: int | None = 20, symmetrize: bool = True) -> sp.csr_matrix:
    """
    Build cell-cell adjacency from expression features (cells x features).

    Steps:
      1) compute dense similarity/distance
      2) optional kNN sparsification
      3) return sparse adjacency (csr)

    NOTE: for euclidean we convert distance -> similarity by exp(-d)
          (simple and stable).
    """
    if metric == "dot":
        S = X @ X.T
    elif metric == "cosine":
        S = cosine_similarity(X)
    elif metric == "pearson":
        Xc = X - X.mean(axis=1, keepdims=True)
        denom = (np.linalg.norm(Xc, axis=1, keepdims=True) @ np.linalg.norm(Xc, axis=1, keepdims=True).T) + 1e-10
        S = (Xc @ Xc.T) / denom
    elif metric == "euclidean":
        sq = np.sum(X * X, axis=1, keepdims=True)
        D = np.sqrt(np.maximum(sq + sq.T - 2.0 * (X @ X.T), 0.0))
        S = np.exp(-D)  # distance -> similarity
    else:
        raise ValueError(f"Unknown metric: {metric}")

    np.fill_diagonal(S, 0.0)

    if knn is None or knn <= 0 or knn >= S.shape[0]:
        A = sp.csr_matrix(S)
        if symmetrize:
            A = A.maximum(A.T)
        return A

    # kNN: keep top-k per row
    n = S.shape[0]
    rows = []
    cols = []
    vals = []
    for i in range(n):
        idx = np.argpartition(-S[i], kth=min(knn, n - 1))[:knn]
        for j in idx:
            if S[i, j] > 0:
                rows.append(i); cols.append(j); vals.append(S[i, j])

    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    if symmetrize:
        A = A.maximum(A.T)
    return A
