from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple


def load_expression_tsv(path: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Load expression matrix from TSV/CSV-like file.

    Assumes:
      - rows = genes, columns = cells OR rows = cells, columns = genes
      - first column may be gene names (common)

    Returns:
      df (DataFrame numeric),
      row_names,
      col_names
    """
    # try with header and index
    df = pd.read_csv(path, sep="\t", header=0, index_col=0)
    # If everything became object, fallback
    if df.shape[1] == 0:
        df = pd.read_csv(path, sep="\t", header=None)

    # Ensure numeric
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return df, list(df.index.astype(str)), list(df.columns.astype(str))


def maybe_transpose_to_cells_by_genes(df: pd.DataFrame,
                                     genes_expected: Optional[int] = None,
                                     cells_expected: Optional[int] = None) -> pd.DataFrame:
    """
    Heuristic to ensure output shape is (cells x genes).
    - If genes_expected/cells_expected are given, uses them.
    - Otherwise assumes genes usually >> cells in scRNA.
    """
    r, c = df.shape
    # If user provided expectations
    if cells_expected is not None and r == cells_expected:
        return df
    if cells_expected is not None and c == cells_expected:
        return df.T

    if genes_expected is not None and c == genes_expected:
        return df
    if genes_expected is not None and r == genes_expected:
        return df.T

    # Heuristic: genes usually larger than cells
    # If rows > cols, likely genes x cells -> transpose
    if c > r:
        return df.T
    return df


def load_gene_list(path: str) -> List[str]:
    """
    One gene per line (or CSV); returns cleaned unique gene list.
    """
    genes = []
    with open(path, "r") as f:
        for line in f:
            g = line.strip().split(",")[0].strip()
            if g:
                genes.append(g)
    # preserve order unique
    seen = set()
    out = []
    for g in genes:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def subset_genes(X_cells_by_genes: pd.DataFrame, gene_list: List[str]) -> pd.DataFrame:
    """
    Keep only genes present in X. Drops missing genes silently.
    """
    keep = [g for g in gene_list if g in X_cells_by_genes.columns]
    return X_cells_by_genes.loc[:, keep]
