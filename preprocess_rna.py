from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from io_rna import load_expression_tsv, maybe_transpose_to_cells_by_genes


def preprocess_counts(X: pd.DataFrame,
                      min_genes_per_cell: int = 200,
                      min_cells_per_gene: int = 3,
                      target_sum: float = 1e4,
                      log1p: bool = True) -> pd.DataFrame:
    """
    Input: cells x genes (counts or normalized)
    Output: cells x genes (normalized + log1p by default)

    If the matrix is already log-normalized, you can set log1p=False
    and target_sum=None by editing this later.
    """
    # Filter genes by detection
    detected_cells_per_gene = (X > 0).sum(axis=0)
    X = X.loc[:, detected_cells_per_gene >= min_cells_per_gene]

    # Filter cells by detected genes
    detected_genes_per_cell = (X > 0).sum(axis=1)
    X = X.loc[detected_genes_per_cell >= min_genes_per_cell, :]

    # Normalize per cell
    libsize = X.sum(axis=1).replace(0, np.nan)
    X = X.div(libsize, axis=0) * float(target_sum)
    X = X.fillna(0.0)

    if log1p:
        X = np.log1p(X)

    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expr_path", required=True, help="Expression matrix TSV (genes x cells or cells x genes).")
    ap.add_argument("--out_path", required=True, help="Output TSV (cells x genes, processed).")
    ap.add_argument("--min_genes_per_cell", type=int, default=200)
    ap.add_argument("--min_cells_per_gene", type=int, default=3)
    ap.add_argument("--target_sum", type=float, default=1e4)
    ap.add_argument("--no_log1p", action="store_true")
    args = ap.parse_args()

    df, _, _ = load_expression_tsv(args.expr_path)
    X = maybe_transpose_to_cells_by_genes(df)

    Xp = preprocess_counts(
        X,
        min_genes_per_cell=args.min_genes_per_cell,
        min_cells_per_gene=args.min_cells_per_gene,
        target_sum=args.target_sum,
        log1p=(not args.no_log1p),
    )

    Xp.to_csv(args.out_path, sep="\t")
    print(f"[OK] Saved processed matrix: {args.out_path}  shape={Xp.shape}")


if __name__ == "__main__":
    main()
