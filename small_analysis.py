import pandas as pd
import numpy as np

cells = pd.read_csv("/gpfs/research/fangroup/pb25e/malignant/datasets/gao_et_al_2021/Data_Gao2021_Breast/Breast/Cells.csv")
names = pd.read_csv("results/gao2021_breast/runs/11599014_20260202_153936/nclust10/gao2021_breast_scgclustlike.cells.txt",
                    header=None, names=["cell_name"])
labels = pd.read_csv("results/gao2021_breast/runs/11599014_20260202_153936/nclust10/gao2021_breast_scgclustlike.labels.tsv",
                     header=None, names=["cluster"])

df = names.join(labels)
df = df.merge(cells[["cell_name","cell_type"]], on="cell_name", how="left")

ct = pd.crosstab(df["cluster"], df["cell_type"])
print(ct)
print("\nMalignant fraction per cluster:")
print((ct["Malignant"] / ct.sum(axis=1)).sort_values(ascending=False))
