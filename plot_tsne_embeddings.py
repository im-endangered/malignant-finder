#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", required=True)
    ap.add_argument("--cells_txt", required=True)
    ap.add_argument("--cells_csv", required=True)
    ap.add_argument("--labels_tsv", required=True)
    ap.add_argument("--out", default="tsne_embedding.png")
    args = ap.parse_args()

    print("[1] Loading latent...")
    Z = np.loadtxt(args.latent)

    print("[2] Loading names...")
    names = [x.strip() for x in open(args.cells_txt)]

    print("[3] Loading ground truth...")
    gt = pd.read_csv(args.cells_csv)
    gt = gt.set_index("cell_name")

    print("[4] Loading cluster labels...")
    pred = np.loadtxt(args.labels_tsv).astype(int)

    df = pd.DataFrame({
        "cell": names,
        "cluster": pred,
    })

    df["gt"] = df["cell"].map(gt["cell_type"])

    df["is_tumor"] = df["gt"].astype(str).str.lower() == "malignant"

    print("[5] Running t-SNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42,
    )

    Z2 = tsne.fit_transform(Z)

    print("[6] Plotting...")

    plt.figure(figsize=(8,6))

    colors = df["is_tumor"].map({True: "red", False: "blue"})

    plt.scatter(
        Z2[:,0],
        Z2[:,1],
        c=colors,
        s=8,
        alpha=0.7,
    )

    plt.title("t-SNE of learned embeddings")
    plt.xlabel("tSNE-1")
    plt.ylabel("tSNE-2")

    plt.savefig(args.out, dpi=200, bbox_inches="tight")

    print("[OK] wrote:", args.out)


if __name__ == "__main__":
    main()
