#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmread
from sklearn.neighbors import NearestNeighbors


def read_lines(path: str) -> list[str]:
    with open(path, "r") as f:
        return [x.strip() for x in f if x.strip()]


def strip_quotes(x: str) -> str:
    x = str(x).strip()
    # handle "GENE" or 'GENE'
    if len(x) >= 2 and ((x[0] == '"' and x[-1] == '"') or (x[0] == "'" and x[-1] == "'")):
        x = x[1:-1]
    return x.strip()


def norm_gene(x: str) -> str:
    """
    Normalize gene identifiers for robust matching.
    - remove surrounding quotes (Genes.txt has them in your dataset)
    - drop Ensembl version suffix (ENSG... .1)
    - uppercase for case-insensitive match
    """
    x = strip_quotes(x)
    x = x.split(".")[0]
    x = x.upper()
    return x


def cp10k_log1p(X: sp.spmatrix) -> np.ndarray:
    """
    X: sparse counts (cells x genes)
    Return dense float32 (cells x genes): CP10k normalized + log1p
    """
    if not sp.isspmatrix(X):
        X = sp.csr_matrix(X)

    X = X.tocsr()
    lib = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)  # (cells,)
    scale = 1e4 / (lib + 1e-12)

    # scale rows
    # (diag(scale) @ X) is expensive; do row-wise multiply in CSR:
    X_scaled = X.multiply(scale[:, None])

    # log1p -> dense (OK at your sizes)
    X_scaled = X_scaled.astype(np.float32)
    X_dense = X_scaled.toarray()
    np.log1p(X_dense, out=X_dense)
    return X_dense.astype(np.float32)


def build_knn_cna_edges(
    CNA: np.ndarray,
    k: int,
    power: float,
    symmetrize: bool = True,
    coalesce: str = "max",
    seed: int = 0,
):
    """
    CNA: (N, D) float32
    Build kNN graph with cosine similarity weights (raw cosine sim).
    Return edge_index (2,E) int64, edge_weight (E,) float32.
    """
    rng = np.random.RandomState(seed)
    N = CNA.shape[0]

    # NearestNeighbors w/ cosine distance
    # n_neighbors = k+1 so self appears (distance 0), we drop it
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="auto")
    nn.fit(CNA)
    dist, idx = nn.kneighbors(CNA, return_distance=True)  # (N,k+1)

    # drop self neighbor (usually idx[:,0]==arange(N))
    idx = idx[:, 1:]
    dist = dist[:, 1:]

    sim = 1.0 - dist  # cosine similarity
    sim = np.clip(sim, 0.0, 1.0).astype(np.float32)

    if power is not None and float(power) != 1.0:
        sim = np.power(sim, float(power), dtype=np.float32)

    # directed edges i -> idx[i,j]
    src = np.repeat(np.arange(N, dtype=np.int64), k)
    dst = idx.reshape(-1).astype(np.int64)
    w = sim.reshape(-1).astype(np.float32)

    # remove any accidental self edges
    m = src != dst
    src, dst, w = src[m], dst[m], w[m]

    if not symmetrize:
        edge_index = np.vstack([src, dst])
        edge_weight = w
        return edge_index, edge_weight

    # Symmetrize by coalescing i->j and j->i
    # We'll create keys for pairs and aggregate
    # For max coalesce: keep max weight among duplicates.
    # (This is efficient enough for your edge counts.)
    pairs = np.vstack([src, dst]).T  # (E,2)

    # add reversed edges too so graph is explicitly undirected-ish
    pairs_rev = np.vstack([dst, src]).T
    w_rev = w.copy()

    pairs2 = np.concatenate([pairs, pairs_rev], axis=0)
    w2 = np.concatenate([w, w_rev], axis=0)

    # coalesce duplicates (stable + fast, no structured view)
    # unique key for (src,dst): src * N + dst
    keys = pairs2[:, 0] * np.int64(N) + pairs2[:, 1]

    order = np.argsort(keys, kind="mergesort")
    keys = keys[order]
    pairs2 = pairs2[order]
    w2 = w2[order]

    uniq_keys, start_idx = np.unique(keys, return_index=True)
    end_idx = np.concatenate([start_idx[1:], [len(keys)]])

    out_src = np.empty(len(uniq_keys), dtype=np.int64)
    out_dst = np.empty(len(uniq_keys), dtype=np.int64)
    out_w = np.empty(len(uniq_keys), dtype=np.float32)

    for i, (a, b) in enumerate(zip(start_idx, end_idx)):
        out_src[i] = pairs2[a, 0]
        out_dst[i] = pairs2[a, 1]
        if coalesce == "max":
            out_w[i] = float(np.max(w2[a:b]))
        elif coalesce == "mean":
            out_w[i] = float(np.mean(w2[a:b]))
        else:
            raise ValueError(f"Unknown coalesce={coalesce}")

    edge_index = np.vstack([out_src, out_dst])
    edge_weight = out_w.astype(np.float32)
    return edge_index, edge_weight



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cna_csv", required=True, help="CNA matrix CSV with cell_name as first column/index.")
    ap.add_argument("--cells_csv", required=True, help="Cells.csv containing 'cell_name' and 'cell_type'.")
    ap.add_argument("--genes_txt", required=True, help="Genes.txt list for the MTX gene axis.")
    ap.add_argument("--mtx", required=True, help="UMI counts MTX.")
    ap.add_argument("--sig_genes_txt", required=True, help="Signature genes list (symbols).")

    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--power", type=float, default=4.0)
    ap.add_argument("--symmetrize", action="store_true", help="Symmetrize edges (default True).")
    ap.add_argument("--no_symmetrize", dest="symmetrize", action="store_false")
    ap.set_defaults(symmetrize=True)

    ap.add_argument("--coalesce", default="max", choices=["max", "mean"])
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--out", required=True, help="Output NPZ path.")
    args = ap.parse_args()

    # --- canonical cell order from Cells.csv ---
    cells = pd.read_csv(args.cells_csv)
    if "cell_name" not in cells.columns:
        raise ValueError("Cells.csv must contain column 'cell_name'")
    cell_names = cells["cell_name"].astype(str).tolist()
    N = len(cell_names)

    # --- RNA: load MTX + genes ---
    genes_raw = read_lines(args.genes_txt)
    genes = [strip_quotes(g) for g in genes_raw]  # remove quotes so downstream is consistent
    G = len(genes)

    X_mtx = mmread(args.mtx)
    if not sp.isspmatrix(X_mtx):
        X_mtx = sp.csr_matrix(X_mtx)
    X_mtx = X_mtx.tocsr()

    # Determine orientation: we want (cells x genes)
    # Your earlier logs show (4143, 33694) after processing, so cells=4143, genes=33694.
    if X_mtx.shape == (G, N):
        X_counts = X_mtx.T.tocsr()
    elif X_mtx.shape == (N, G):
        X_counts = X_mtx.tocsr()
    else:
        raise ValueError(
            f"MTX shape {X_mtx.shape} does not match Genes.txt ({G}) and Cells.csv ({N}). "
            f"Expected (G,N) or (N,G)."
        )

    print(f"[RNA] raw counts shape: {X_counts.shape}")

    X_norm = cp10k_log1p(X_counts)
    print("[RNA] CP10000 normalized + log1p done.")

    # signature selection with robust matching
    sig_raw = read_lines(args.sig_genes_txt)
    sig_norm = set(norm_gene(s) for s in sig_raw)

    genes_norm = [norm_gene(g) for g in genes]
    idx = [i for i, gn in enumerate(genes_norm) if gn in sig_norm]

    found = len(idx)
    print(f"[RNA] Using signatures: found {found}/{len(sig_norm)}. X_used=({N}, {found})")
    if found == 0:
        print("[RNA][ERROR] 0 signature matches.")
        print("  Genes(norm) sample:", genes_norm[:10])
        print("  Sig(norm) sample:", list(sig_norm)[:10])
        raise RuntimeError("No signature genes matched Genes.txt. Likely symbol/ID mismatch.")

    X_feat = X_norm[:, idx].astype(np.float32)
    print(f"[RNA] rna_feat (N x G): {X_feat.shape} mode=signatures_cp10k_log1p")
    print(f"[RNA] feat min/median/max: {float(X_feat.min())} {float(np.median(X_feat))} {float(X_feat.max())}")

    # --- CNA ---
    cna = pd.read_csv(args.cna_csv, index_col=0)
    # ensure index is str
    cna.index = cna.index.astype(str)

    # reorder to Cells.csv order
    missing = set(cell_names) - set(cna.index)
    if missing:
        raise ValueError(f"CNA CSV missing {len(missing)} cell_names from Cells.csv (example: {list(missing)[:5]})")

    cna = cna.loc[cell_names]
    CNA = cna.to_numpy(dtype=np.float32, copy=True)
    print(f"[CNA] shape: {CNA.shape}")

    # --- edges from CNA cosine similarity kNN ---
    edge_index, edge_weight = build_knn_cna_edges(
        CNA=CNA,
        k=args.k,
        power=args.power,
        symmetrize=args.symmetrize,
        coalesce=args.coalesce,
        seed=args.seed,
    )
    print(f"[CNA] edges: {edge_index.shape[1]} weight min/max={edge_weight.min():.4f}/{edge_weight.max():.4f}")

    params = {
        "edges": {
            "k": int(args.k),
            "metric": "cosine",
            "weight": "raw_cosine_similarity",
            "power": float(args.power),
            "symmetrize": bool(args.symmetrize),
            "coalesce": args.coalesce,
            "seed": int(args.seed),
        },
        "rna_features": {
            "type": "signatures",
            "transform": "cp10k_log1p",
            "n_genes": int(X_feat.shape[1]),
        },
        "canonical_order": "Cells.csv order (RNA), CNA reordered to match",
    }

    np.savez_compressed(
        args.out,
        rna_feat=X_feat,
        edge_index=edge_index.astype(np.int64),
        edge_weight=edge_weight.astype(np.float32),
        cell_names=np.array(cell_names, dtype=object),
        params=json.dumps(params),
    )
    print(f"[DONE] wrote: {args.out}")


if __name__ == "__main__":
    main()
