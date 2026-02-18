#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def read_lines(p):
    with open(p) as f:
        return [x.strip() for x in f if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pfx", required=True)
    ap.add_argument("--cells_csv", required=True)
    ap.add_argument("--cluster", type=int, required=True)
    ap.add_argument("--perplexity", type=float, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # load model outputs
    latent = np.loadtxt(args.pfx + ".latent.tsv")
    labels = np.loadtxt(args.pfx + ".labels.tsv", dtype=int)
    names = read_lines(args.pfx + ".cells.txt")

    # load ground truth
    gt = pd.read_csv(args.cells_csv)
    gt = gt.set_index("cell_name")

    # align truth labels
    truth = []
    for n in names:
        ct = str(gt.loc[n, "cell_type"]).lower()
        truth.append(1 if "malignant" in ct else 0)
    truth = np.array(truth)

    # select cluster
    idx = np.where(labels == args.cluster)[0]

    X = latent[idx]
    truth_sub = truth[idx]

    print(f"cluster {args.cluster}: n={len(idx)} tumor_frac={truth_sub.mean():.4f}")

    # t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        learning_rate="auto",
        init="pca",
        random_state=args.seed,
    )
    Y = tsne.fit_transform(X)

    # plot
    plt.figure(figsize=(9,7))

    tumor = truth_sub == 1
    normal = truth_sub == 0

    plt.scatter(Y[normal,0], Y[normal,1],
                c="blue", s=10, alpha=0.6, label="Normal")

    plt.scatter(Y[tumor,0], Y[tumor,1],
                c="red", s=10, alpha=0.6, label="Tumor")

    plt.legend()
    plt.title(f"t-SNE of latent embedding — cluster {args.cluster}")
    plt.xlabel("tSNE-1")
    plt.ylabel("tSNE-2")

    out = args.pfx + f".tsne_cluster{args.cluster}_truth.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")

    print("[OK] wrote:", out)


if __name__ == "__main__":
    main()
