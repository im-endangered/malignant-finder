#!/usr/bin/env python3
from __future__ import annotations
import argparse
import pandas as pd
import numpy as np


def read_names(path: str) -> pd.Series:
    names = pd.read_csv(path, header=None)[0].astype(str)
    names = names[names.str.len() > 0].reset_index(drop=True)
    return names


def read_labels(path: str) -> pd.Series:
    lab = pd.read_csv(path, header=None)[0]
    # allow strings like "1" or "1.0"
    lab = lab.astype(str).str.strip()
    lab = lab[lab.str.len() > 0].reset_index(drop=True)
    # cast to int safely
    lab = lab.astype(float).astype(int)
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells_csv", required=True)
    ap.add_argument("--names_txt", required=True)
    ap.add_argument("--labels_tsv", required=True)
    ap.add_argument("--out_labels_tsv", required=True)
    ap.add_argument("--tumor_frac_thr", type=float, default=0.5)
    ap.add_argument("--malignant_key", default="malignant")
    ap.add_argument("--cell_type_col", default="cell_type")
    ap.add_argument("--force_two_classes", action="store_true",
                    help="If mapping collapses to a single class, force the lowest-malignant cluster to be normal.")
    args = ap.parse_args()

    cells = pd.read_csv(args.cells_csv)
    cells["cell_name"] = cells["cell_name"].astype(str)

    names = read_names(args.names_txt)
    lab = read_labels(args.labels_tsv)

    if len(lab) != len(names):
        raise ValueError(f"labels_tsv has {len(lab)} rows but names_txt has {len(names)}")

    df = pd.DataFrame({"cell_name": names, "cluster": lab})
    df = df.merge(cells[["cell_name", args.cell_type_col]], on="cell_name", how="left")

    ct = df[args.cell_type_col].astype(str).str.lower()
    is_malig = (ct == args.malignant_key.lower())

    # malignant fraction per cluster (ignore missing cell_type -> counts as non-malignant for fraction)
    frac = df.groupby("cluster").apply(
        lambda g: float(((g[args.cell_type_col].astype(str).str.lower() == args.malignant_key.lower())).mean())
        if len(g) > 0 else 0.0
    )
    sizes = df["cluster"].value_counts().sort_index()

    print("=== malignant fraction by cluster ===")
    for c in sorted(frac.index.tolist()):
        print(f"cluster {c:>3}: frac_malignant={frac[c]:.4f}  size={int(sizes.get(c, 0))}")
    print()

    tumor_clusters = [int(c) for c in frac.index if frac[c] >= args.tumor_frac_thr]
    print("Tumor clusters (mapped to 1):", tumor_clusters)
    print()

    # map
    binlab = df["cluster"].apply(lambda c: 1 if int(c) in set(tumor_clusters) else 0).astype(int)

    # safety: avoid all-0 or all-1 if requested
    if args.force_two_classes and binlab.nunique() < 2:
        # force lowest-malignant cluster to be normal (0)
        lowest = int(frac.sort_values().index[0])
        print(f"[WARN] Degenerate mapping (only one class). Forcing lowest-malignant cluster {lowest} to normal (0).")
        binlab = df["cluster"].apply(lambda c: 0 if int(c) == lowest else 1).astype(int)

    # write
    pd.Series(binlab).to_csv(args.out_labels_tsv, sep="\t", header=False, index=False)
    vc = pd.Series(binlab).value_counts().to_dict()
    print("[OK] wrote:", args.out_labels_tsv)
    print("binary counts:", vc)


if __name__ == "__main__":
    main()
