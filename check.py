import pandas as pd
import numpy as np

# paths from your latest run
cells_csv = "/gpfs/research/fangroup/pb25e/malignant/datasets/gao_et_al_2021/Data_Gao2021_Breast/Breast/Cells.csv"
labels_path = "results/gao2021_breast/runs/11599014_20260202_153936/nclust2/gao2021_breast_scgclustlike.labels.tsv"
names_path  = "results/gao2021_breast/runs/11599014_20260202_153936/nclust2/gao2021_breast_scgclustlike.cells.txt"

cells = pd.read_csv(cells_csv)
names = pd.read_csv(names_path, header=None, names=["cell_name"])
labels = pd.read_csv(labels_path, header=None, names=["cluster"])

df = names.join(labels)
df = df.merge(cells[["cell_name","cell_type"]], on="cell_name", how="left")

ct = pd.crosstab(df["cluster"], df["cell_type"])
print(ct)
print("\nRow-normalized:")
print(ct.div(ct.sum(axis=1), axis=0).round(3))
