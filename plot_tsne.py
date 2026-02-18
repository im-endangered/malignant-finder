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
    ap.add_argument("--pfx", required=True, help="Prefix (without extension)")
    ap.add_argument("--mode", choices=["latent", "rna"], default="latent",
                    help="latent: use PFX.latent.tsv ; rna: use NPZ rna_feat")
    ap.add_argument("--npz", default=None, help="Required if --mode rna")
    ap.add_argument("--cluster", type=int, default=None,
                    help="If set, only plot cells in this cluster id")
    ap.add_argument("--perplexity", type=float, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # --- load cluster labels & names ---
    cells_txt = args.pfx + ".cells.txt"
    labels_tsv = args.pfx + ".labels.tsv"

    names = read_lines(cells_txt)
    labels = np.loadtxt(labels_tsv, dtype=int)

    if len(names) != len(labels):
        raise ValueError(f"names ({len(names)}) != labels ({len(labels)})")

    # --- load features / embeddings ---
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

    # --- subset cluster if requested ---
    idx = np.arange(len(names))
    if args.cluster is not None:
        idx = idx[labels == args.cluster]
        if idx.size == 0:
            raise ValueError(f"No cells found for cluster {args.cluster}")

    Xs = X[idx]
    labs = labels[idx]

    # --- t-SNE ---
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        learning_rate="auto",
        init="pca",
        random_state=args.seed,
    )
    Y = tsne.fit_transform(Xs)

    # --- output name ---
    if args.out is None:
        tag = f"cluster{args.cluster}" if args.cluster is not None else "all"
        args.out = f"{args.pfx}.tsne_{args.mode}.{tag}.png"

    # --- plot ---
    plt.figure(figsize=(9, 7))

    # color by cluster id (even if single cluster, this is consistent)
    # For multi-cluster plotting, this makes cluster structure visible.
    plt.scatter(Y[:, 0], Y[:, 1], c=labs, s=8, alpha=0.75)

    title = f"t-SNE of {args.mode} — "
    title += f"cluster {args.cluster}" if args.cluster is not None else "all cells"
    plt.title(title)
    plt.xlabel("tSNE-1")
    plt.ylabel("tSNE-2")

    cbar = plt.colorbar()
    cbar.set_label("cluster id")

    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print("[OK] wrote:", args.out)


if __name__ == "__main__":
    main()
