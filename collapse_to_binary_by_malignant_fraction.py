#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pandas as pd
import numpy as np

# headless-safe plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_names(path: str) -> pd.Series:
    names = pd.read_csv(path, header=None)[0].astype(str)
    names = names[names.str.len() > 0].reset_index(drop=True)
    return names


def read_labels(path: str) -> pd.Series:
    lab = pd.read_csv(path, header=None)[0]
    lab = lab.astype(str).str.strip()
    lab = lab[lab.str.len() > 0].reset_index(drop=True)
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
    ap.add_argument(
        "--force_two_classes",
        action="store_true",
        help="If mapping collapses to a single class, force the lowest-malignant cluster to be normal.",
    )

    # plotting options
    ap.add_argument(
        "--out_plot",
        default=None,
        help="Path to save stacked composition plot (e.g., .png). "
             "Default: <out_labels_tsv>.composition.png",
    )
    ap.add_argument(
        "--plot_pdf",
        action="store_true",
        help="Also save a PDF next to the PNG.",
    )
    ap.add_argument(
        "--label_frac_decimals",
        type=int,
        default=2,
        help="Decimals for malignant fraction labels on bars (default 2).",
    )
    args = ap.parse_args()

    # ---- Load inputs ----
    cells = pd.read_csv(args.cells_csv)
    cells["cell_name"] = cells["cell_name"].astype(str)

    names = read_names(args.names_txt)
    lab = read_labels(args.labels_tsv)

    if len(lab) != len(names):
        raise ValueError(f"labels_tsv has {len(lab)} rows but names_txt has {len(names)}")

    df = pd.DataFrame({"cell_name": names, "cluster": lab})
    df = df.merge(cells[["cell_name", args.cell_type_col]], on="cell_name", how="left")

    # malignant indicator (missing cell_type treated as non-malignant)
    ct = df[args.cell_type_col].astype(str).str.lower()
    df["is_malignant"] = (ct == args.malignant_key.lower())

    # ---- Fractions and sizes (no groupby.apply warning) ----
    frac = df.groupby("cluster")["is_malignant"].mean().sort_index()
    sizes = df["cluster"].value_counts().sort_index()

    print("=== malignant fraction by cluster ===")
    for c in frac.index.tolist():
        print(f"cluster {int(c):>3}: frac_malignant={float(frac.loc[c]):.4f}  size={int(sizes.get(c, 0))}")
    print()

    tumor_clusters = [int(c) for c in frac.index if float(frac.loc[c]) >= args.tumor_frac_thr]
    print("Tumor clusters (mapped to 1):", tumor_clusters)
    print()

    # ---- Map to binary labels ----
    tumor_set = set(tumor_clusters)
    binlab = df["cluster"].apply(lambda c: 1 if int(c) in tumor_set else 0).astype(int)

    # safety: avoid all-0 or all-1 if requested
    if args.force_two_classes and binlab.nunique() < 2:
        lowest = int(frac.sort_values().index[0])
        print(f"[WARN] Degenerate mapping (only one class). Forcing lowest-malignant cluster {lowest} to normal (0).")
        binlab = df["cluster"].apply(lambda c: 0 if int(c) == lowest else 1).astype(int)

    # write labels
    pd.Series(binlab).to_csv(args.out_labels_tsv, sep="\t", header=False, index=False)
    vc = pd.Series(binlab).value_counts().to_dict()
    print("[OK] wrote:", args.out_labels_tsv)
    print("binary counts:", vc)

    # ---- Stacked composition plot (counts) with frac labels ----
    out_plot = args.out_plot
    if out_plot is None or str(out_plot).strip() == "":
        out_plot = args.out_labels_tsv + ".composition.png"

    # counts per cluster split by malignant/normal
    counts = (
        df.groupby(["cluster", "is_malignant"])
          .size()
          .unstack(fill_value=0)
          .sort_index()
    )

    # ensure both columns exist
    if True not in counts.columns:
        counts[True] = 0
    if False not in counts.columns:
        counts[False] = 0

    tumor_counts = counts[True].to_numpy(dtype=int)   # bottom
    normal_counts = counts[False].to_numpy(dtype=int) # top
    total_counts = tumor_counts + normal_counts

    clusters = counts.index.astype(int).to_numpy()
    x = np.arange(len(clusters))

    # dynamic width so labels don't collide too badly if many clusters
    fig_w = max(10, 0.75 * len(clusters))
    plt.figure(figsize=(fig_w, 6))

    plt.bar(x, tumor_counts, label="Tumor (malignant)")
    plt.bar(x, normal_counts, bottom=tumor_counts, label="Normal")

    plt.xticks(x, [str(c) for c in clusters], rotation=0)
    plt.xlabel("Cluster")
    plt.ylabel("# of cells")
    plt.title("Cell composition by cluster")
    plt.legend()

    # annotate fraction malignant on top of each bar
    # use frac series aligned to clusters
    frac_aligned = frac.reindex(clusters).to_numpy(dtype=float)

    # small offset so text is above bar
    y_offset = max(1, int(0.01 * (total_counts.max() if len(total_counts) else 1)))

    for i, (tot, f) in enumerate(zip(total_counts, frac_aligned)):
        if tot <= 0:
            continue
        txt = f"{f:.{args.label_frac_decimals}f}"
        plt.text(i, tot + y_offset, txt, ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_plot, dpi=200)
    print("[OK] wrote plot:", out_plot)

    if args.plot_pdf:
        pdf_path = out_plot.rsplit(".", 1)[0] + ".pdf"
        plt.savefig(pdf_path)
        print("[OK] wrote plot:", pdf_path)

    plt.close()


if __name__ == "__main__":
    main()
