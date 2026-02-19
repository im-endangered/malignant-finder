#!/usr/bin/env python3
# plot_tsne_cluster0_labels.py
#
# Make ONE t-SNE figure for ALL cells colored by *cluster id*,
# but additionally overlay cluster 0 cells colored by *ground-truth tumor/normal*.
#
# Example:
#   PFX="results/.../gao2021_RNAfeat_CNAedge_p4_k20"
#   python plot_tsne_cluster0_labels.py --pfx "$PFX" --mode latent \
#     --cells_csv datasets/gao_et_al_2021/Data_Gao2021_Breast/Breast/Cells.csv
#
# Output:
#   ${PFX}.tsne_latent.all_with_cluster0_truth.png

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def read_lines(p):
    with open(p) as f:
        return [x.strip() for x in f if x.strip()]


def infer_truth_is_tumor(cell_type_series: pd.Series, tumor_tokens):
    """
    Returns boolean series: True=tumor, False=normal
    tumor_tokens: list[str] matched as substring (case-insensitive)
    """
    s = cell_type_series.astype(str).str.lower()
    is_tumor = pd.Series(False, index=s.index)
    for tok in tumor_tokens:
        tok = tok.lower()
        is_tumor = is_tumor | s.str.contains(tok, na=False)
    return is_tumor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pfx", required=True, help="Prefix (without extension)")
    ap.add_argument("--mode", choices=["latent", "rna"], default="latent",
                    help="latent: use PFX.latent.tsv ; rna: use NPZ rna_feat")
    ap.add_argument("--npz", default=None, help="Required if --mode rna")

    # t-SNE params
    ap.add_argument("--perplexity", type=float, default=30)
    ap.add_argument("--seed", type=int, default=42)

    # cluster-0 overlay controls
    ap.add_argument("--cluster_id", type=int, default=0,
                    help="Which cluster id to treat as the mixed cluster to overlay (default: 0)")
    ap.add_argument("--cells_csv", required=True,
                    help="Cells.csv that contains cell_name and cell_type columns")
    ap.add_argument("--cell_type_col", default="cell_type",
                    help="Column name in Cells.csv for ground truth type (default: cell_type)")
    ap.add_argument("--tumor_tokens", default="malignant,tumor",
                    help="Comma-separated substrings treated as 'tumor' in cell_type (default: malignant,tumor)")

    # plotting
    ap.add_argument("--out", default=None)
    ap.add_argument("--point_size", type=float, default=8.0,
                    help="Base point size for all cells")
    ap.add_argument("--cluster0_point_size", type=float, default=18.0,
                    help="Point size for overlayed cluster0 cells")
    ap.add_argument("--alpha", type=float, default=0.75,
                    help="Alpha for non-cluster0 points")
    args = ap.parse_args()

    # --- load names + cluster labels ---
    cells_txt = args.pfx + ".cells.txt"
    labels_tsv = args.pfx + ".labels.tsv"
    names = read_lines(cells_txt)
    labels = np.loadtxt(labels_tsv, dtype=int)

    if len(names) != len(labels):
        raise ValueError(f"names ({len(names)}) != labels ({len(labels)})")

    # --- load embedding/features ---
    if args.mode == "latent":
        X = np.loadtxt(args.pfx + ".latent.tsv")
    else:
        if args.npz is None:
            raise ValueError("--npz is required when --mode rna")
        d = np.load(args.npz, allow_pickle=True)
        if "rna_feat" in d.files:
            X = d["rna_feat"]
        elif "x" in d.files:
            X = d["x"]
        else:
            raise KeyError(f"NPZ missing rna_feat/x. keys={d.files}")
        X = np.asarray(X, dtype=np.float32)

    if X.shape[0] != len(names):
        raise ValueError(f"X rows ({X.shape[0]}) != #cells ({len(names)})")

    # --- load truth and align to names ---
    cells = pd.read_csv(args.cells_csv)
    if "cell_name" not in cells.columns:
        raise KeyError(f"Cells.csv missing 'cell_name' column. columns={list(cells.columns)}")
    if args.cell_type_col not in cells.columns:
        raise KeyError(f"Cells.csv missing '{args.cell_type_col}' column. columns={list(cells.columns)}")

    truth = cells[["cell_name", args.cell_type_col]].copy()
    truth["cell_name"] = truth["cell_name"].astype(str)
    name_df = pd.DataFrame({"cell_name": pd.Series(names, dtype=str)})
    merged = name_df.merge(truth, on="cell_name", how="left")

    tumor_tokens = [t.strip() for t in args.tumor_tokens.split(",") if t.strip()]
    is_tumor = infer_truth_is_tumor(merged[args.cell_type_col], tumor_tokens=tumor_tokens).to_numpy()

    # --- t-SNE on ALL cells ---
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        learning_rate="auto",
        init="pca",
        random_state=args.seed,
    )
    Y = tsne.fit_transform(X)

    # --- output name ---
    if args.out is None:
        args.out = f"{args.pfx}.tsne_{args.mode}.all_with_{args.cluster_id}_truth.png"

    # --- split indices ---
    idx0 = np.where(labels == args.cluster_id)[0]
    idx_other = np.where(labels != args.cluster_id)[0]
    if idx0.size == 0:
        raise ValueError(f"No cells found for cluster_id={args.cluster_id}")

    # overlay cluster0 by truth
    idx0_tumor = idx0[is_tumor[idx0]]
    idx0_normal = idx0[~is_tumor[idx0]]

    # --- plot ---
    plt.figure(figsize=(10, 8))

    # base: all non-cluster0, colored by cluster id
    sc = plt.scatter(
        Y[idx_other, 0], Y[idx_other, 1],
        c=labels[idx_other],
        s=args.point_size,
        alpha=args.alpha
    )

    # overlay: cluster0 colored by truth (only for cluster0)
    # (we draw these on top so it stands out)
    if idx0_normal.size > 0:
        plt.scatter(
            Y[idx0_normal, 0], Y[idx0_normal, 1],
            c="blue",
            s=args.cluster0_point_size,
            alpha=0.85,
            edgecolors="k",
            linewidths=0.3,
            label=f"Cluster {args.cluster_id}: Normal (truth)"
        )
    if idx0_tumor.size > 0:
        plt.scatter(
            Y[idx0_tumor, 0], Y[idx0_tumor, 1],
            c="red",
            s=args.cluster0_point_size,
            alpha=0.85,
            edgecolors="k",
            linewidths=0.3,
            label=f"Cluster {args.cluster_id} : Tumor (truth)"
        )

    title = f"t-SNE of {args.mode} — all cells (colored by cluster id)\n"
    title += f"Overlay: cluster {args.cluster_id} colored by truth tumor/normal"
    plt.title(title)
    plt.xlabel("tSNE-1")
    plt.ylabel("tSNE-2")

    cbar = plt.colorbar(sc)
    cbar.set_label(f"cluster id (excluding {args.cluster_id} overlay colors)")

    plt.legend(loc="best", frameon=True)
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print("[OK] wrote:", args.out)
    print(f"[INFO] cluster_id={args.cluster_id} n={idx0.size} "
          f"tumor={idx0_tumor.size} normal={idx0_normal.size}")


if __name__ == "__main__":
    main()
