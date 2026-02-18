import argparse
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--npz", required=True)
parser.add_argument("--pfx", required=True)
parser.add_argument("--cells_csv", required=True)
parser.add_argument("--cluster_id", type=int, default=0)
args = parser.parse_args()

# Load RNA features
d = np.load(args.npz, allow_pickle=True)
X = d["rna_feat"]
cell_names = d["cell_names"].astype(str)

# Load cluster labels
labels = pd.read_csv(args.pfx + ".labels.tsv", header=None)[0].values

# Load truth
cells = pd.read_csv(args.cells_csv)
cells["cell_name"] = cells["cell_name"].astype(str)

df = pd.DataFrame({
    "cell_name": cell_names,
    "cluster": labels
})

df = df.merge(cells[["cell_name", "cell_type"]], on="cell_name", how="left")

# Select cluster
df0 = df[df["cluster"] == args.cluster_id]

idx = df0.index.values
X0 = X[idx]

print("Cluster size:", len(idx))
print(df0["cell_type"].value_counts())

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
Z = tsne.fit_transform(X0)

# Plot
colors = df0["cell_type"].apply(
    lambda x: "red" if str(x).lower()=="malignant" else "blue"
)

plt.figure(figsize=(7,6))
plt.scatter(Z[:,0], Z[:,1], c=colors, alpha=0.6)

plt.title(f"t-SNE of RNA features — cluster {args.cluster_id}")
plt.xlabel("tSNE-1")
plt.ylabel("tSNE-2")

plt.savefig(args.pfx + f".cluster{args.cluster_id}.tsne.png", dpi=150)
print("Wrote : "+args.pfx + f".cluster{args.cluster_id}.tsne.png")
plt.show()


#plot the embeddings

#check how far is farthest neighbor in knn . specificically in cluster 0 
#check for random tumor cells in cluster 0 , check if its neighbor are tumor too