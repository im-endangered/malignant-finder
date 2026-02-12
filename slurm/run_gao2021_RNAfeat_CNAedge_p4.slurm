#!/bin/bash
#SBATCH --job-name=gao_RNAfeat_CNAedge_p4
#SBATCH --output=/gpfs/research/fangroup/pb25e/malignant/logs/%x_%j.out
#SBATCH --error=/gpfs/research/fangroup/pb25e/malignant/logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=genacc_q

set -euo pipefail

# ---- run from repo root ----
cd ..

PYBIN=/gpfs/home/pb25e/.conda/envs/scgclust_rna/bin/python

NPZ=/gpfs/research/fangroup/pb25e/malignant/results/gao2021_breast/inputs/v4_RNAfeat_CNAedge_var0.02_k20_rawcos_p4.npz
RUN=/gpfs/research/fangroup/pb25e/malignant/results/gao2021_breast/runs/RNAfeat_CNAedge_p4_k20
PFX=$RUN/gao2021_RNAfeat_CNAedge_p4_k20

mkdir -p "$RUN" logs

echo "[RUN] $PFX"
date

echo "[1/4] Train..."
$PYBIN /gpfs/research/fangroup/pb25e/malignant/train_rna.py \
  --npz "$NPZ" \
  --n_clusters 10 \
  --hidden1 64 --hidden2 16 \
  --dropout 0.0 \
  --epochs 500 \
  --lr 1e-3 \
  --clusterer gmm \
  --lambda_contrast 0.25 \
  --n_neg 256 \
  --temperature 0.1 \
  --zscore_features \
  --out_prefix "$PFX"

echo "[2/4] Collapse clusters -> binary..."
$PYBIN /gpfs/research/fangroup/pb25e/malignant/collapse_to_binary_by_malignant_fraction.py \
  --cells_csv /gpfs/research/fangroup/pb25e/malignant/datasets/gao_et_al_2021/Data_Gao2021_Breast/Breast/Cells.csv \
  --names_txt "${PFX}.cells.txt" \
  --labels_tsv "${PFX}.labels.tsv" \
  --out_labels_tsv "${PFX}.FINAL.binary.labels.tsv" \
  --tumor_frac_thr 0.50 \
  --force_two_classes

echo "[3/4] Evaluate binary..."
$PYBIN /gpfs/research/fangroup/pb25e/malignant/eval_tumor_normal_binary.py \
  --cells_csv /gpfs/research/fangroup/pb25e/malignant/datasets/gao_et_al_2021/Data_Gao2021_Breast/Breast/Cells.csv \
  --names_txt "${PFX}.cells.txt" \
  --labels_tsv "${PFX}.FINAL.binary.labels.tsv" \
  --out_prefix "${PFX}.FINAL.eval"

echo "[4/4] Quick latent diagnostics..."
$PYBIN - <<PY
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

Z = np.loadtxt("${PFX}.latent.tsv")
print("Z shape:", Z.shape)
print("Z var per-dim min/median/max:",
      float(Z.var(axis=0).min()), float(np.median(Z.var(axis=0))), float(Z.var(axis=0).max()))
idx = np.random.RandomState(0).choice(Z.shape[0], size=min(500, Z.shape[0]), replace=False)
S = cosine_similarity(Z[idx])
off = S[np.triu_indices_from(S, k=1)]
print("pairwise cos(Z) offdiag mean/std/min/max:",
      float(off.mean()), float(off.std()), float(off.min()), float(off.max()))
PY

echo "=== Done ==="
date
