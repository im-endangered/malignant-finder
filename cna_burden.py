import pandas as pd
import numpy as np

# Paths
cells_csv = "/gpfs/research/fangroup/pb25e/malignant/datasets/gao_et_al_2021/Data_Gao2021_Breast/Breast/Cells.csv"
names_path = "results/gao2021_breast/runs/11599014_20260202_153936/nclust10/gao2021_breast_scgclustlike.cells.txt"
labels_path = "results/gao2021_breast/runs/11599014_20260202_153936/nclust10/gao2021_breast_scgclustlike.labels.tsv"
cna_csv = "/gpfs/research/fangroup/pb25e/malignant/datasets/gao_et_al_2021/CNA_matrix_Gao2021_Breast/CNAs_Breast.csv"

# Load
cells = pd.read_csv(cells_csv)
names = pd.read_csv(names_path, header=None, names=["cell_name"])
labels = pd.read_csv(labels_path, header=None, names=["cluster"])
cna = pd.read_csv(cna_csv)

# Merge everything
df = names.join(labels)
df = df.merge(cells[["cell_name","cell_type"]], on="cell_name", how="left")

# Compute CNA burden per cell
X = cna.drop(columns=["cell_name"]).to_numpy(dtype=np.float32)
burden = np.sqrt(np.mean(X*X, axis=1))  # RMS burden

cna_df = pd.DataFrame({
    "cell_name": cna["cell_name"],
    "cna_burden_RMS": burden
})

df = df.merge(cna_df, on="cell_name", how="left")

# ---- Hybrid labeling ----
# Step 1: cluster-based tumor
tumor_clusters = {1,2,5,7,9}
df["pred_tumor"] = df["cluster"].isin(tumor_clusters).astype(int)

# Step 2: refine cluster 5 using CNA burden
# threshold = median burden of malignant cells (reasonable default)
thr = df.loc[df["cell_type"]=="Malignant", "cna_burden_RMS"].median()

mask = (df["cluster"] == 5) & (df["cna_burden_RMS"] < thr)
df.loc[mask, "pred_tumor"] = 0   # relabel low-burden cells as normal

# Evaluate cell-by-cell
df["true_malignant"] = (df["cell_type"]=="Malignant").astype(int)

tp = ((df.pred_tumor==1) & (df.true_malignant==1)).sum()
fp = ((df.pred_tumor==1) & (df.true_malignant==0)).sum()
fn = ((df.pred_tumor==0) & (df.true_malignant==1)).sum()
tn = ((df.pred_tumor==0) & (df.true_malignant==0)).sum()

print("TP FP FN TN =", tp, fp, fn, tn)

# Save final labels
out = df[["cell_name","pred_tumor"]]
out.to_csv(
 "results/gao2021_breast/runs/11599014_20260202_153936/nclust10/tumor_normal_hybrid.tsv",
 sep="\t", index=False, header=False
)
print("Wrote tumor_normal_hybrid.tsv")
