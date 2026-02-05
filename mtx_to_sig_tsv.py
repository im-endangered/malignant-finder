#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp


def read_genes_txt(path: str) -> list[str]:
    """
    Genes.txt contains one gene per line, often quoted.
    Returns a list in MTX row order.
    """
    genes = []
    with open(path, "r") as f:
        for line in f:
            g = line.strip()
            if not g:
                continue
            # remove quotes if present
            g = g.strip().strip('"').strip("'")
            genes.append(g)
    return genes


def read_signature_genes(path: str) -> set[str]:
    """
    One gene symbol per line (first token per line).
    """
    sig = set()
    with open(path, "r") as f:
        for line in f:
            g = line.strip()
            if not g:
                continue
            g = re.split(r"[,\t ]+", g)[0].strip()
            if g:
                sig.add(g)
    return sig


def main():
    ap = argparse.ArgumentParser(description="Convert MTX + Genes + Cells to signature-only TSV (genes x cells).")
    ap.add_argument("--mtx", required=True, help="Exp_data_UMIcounts.mtx (MatrixMarket)")
    ap.add_argument("--genes", required=True, help="Genes.txt (one per line, MTX row order)")
    ap.add_argument("--cells_csv", required=True, help="Cells.csv containing column 'cell_name' (MTX col order)")
    ap.add_argument("--sig_genes", required=True, help="Signature genes file (one per line)")
    ap.add_argument("--out_tsv", required=True, help="Output TSV: genes x cells with header row and gene column")
    ap.add_argument("--use_processed", action="store_true",
                    help="If set, outputs float32 (still counts). Default is int32.")
    args = ap.parse_args()

    # Load MTX (genes x cells) sparse
    X = scipy.io.mmread(args.mtx)
    if not sp.issparse(X):
        X = sp.coo_matrix(X)
    X = X.tocsr()

    n_genes, n_cells = X.shape
    print(f"[INFO] Loaded MTX shape: genes={n_genes}, cells={n_cells}, nnz={X.nnz}")

    # Load genes (row names)
    genes = read_genes_txt(args.genes)
    if len(genes) != n_genes:
        raise ValueError(f"Genes.txt length {len(genes)} != MTX rows {n_genes}")

    # Load cell names (col names)
    cells_df = pd.read_csv(args.cells_csv)
    if "cell_name" not in cells_df.columns:
        raise ValueError("Cells.csv must contain a column named 'cell_name'")
    cell_names = cells_df["cell_name"].astype(str).tolist()
    if len(cell_names) != n_cells:
        raise ValueError(f"Cells.csv cell_name count {len(cell_names)} != MTX cols {n_cells}")

    # Load signature genes
    sig = read_signature_genes(args.sig_genes)

    # Determine which rows to keep
    keep_idx = [i for i, g in enumerate(genes) if g in sig]
    keep_genes = [genes[i] for i in keep_idx]

    print(f"[INFO] Signature genes requested: {len(sig)}")
    print(f"[INFO] Signature genes matched in dataset: {len(keep_idx)}")

    if len(keep_idx) == 0:
        raise ValueError("No signature genes matched Genes.txt. Check gene identifiers.")

    # Subset sparse matrix to signature genes (rows)
    Xs = X[keep_idx, :]  # (sig_genes x cells)

    # Convert to dense (this is small: ~1743 x 4143)
    if args.use_processed:
        dense = Xs.toarray().astype(np.float32)
    else:
        dense = Xs.toarray().astype(np.int32)

    # Write TSV: first column = gene, header row = cell IDs
    out_df = pd.DataFrame(dense, index=keep_genes, columns=cell_names)
    out_df.index.name = "gene"

    out_df.to_csv(args.out_tsv, sep="\t")
    print(f"[OK] Wrote signature TSV: {args.out_tsv} shape={out_df.shape} (genes x cells)")


if __name__ == "__main__":
    main()
