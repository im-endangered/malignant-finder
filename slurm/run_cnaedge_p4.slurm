#!/bin/bash
#SBATCH --job-name=cnaedge_p4
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=genacc_q

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYBIN=/gpfs/home/pb25e/.conda/envs/scgclust_rna/bin/python

NPZ="results/gao2021_breast/inputs/v4_RNAfeat_CNAedge_var0.02_k20_cos01_p4.npz"
RUN="results/gao2021_breast/runs/repro_CNAedge_p4"
PFX="$RUN/gao2021_CNAedge_p4"

mkdir -p "$RUN" logs

echo "[RUN] $PFX"
date

echo "[1/4] Train..."
$PYBIN train_rna.py \
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
$PYBIN collapse_to_binary_by_malignant_fraction.py \
  --cells_csv datasets/gao_et_al_2021/Data_Gao2021_Breast/Breast/Cells.csv \
  --names_txt "${PFX}.cells.txt" \
  --labels_tsv "${PFX}.labels.tsv" \
  --out_labels_tsv "${PFX}.FINAL.binary.labels.tsv" \
  --tumor_frac_thr 0.50 \
  --force_two_classes

echo "[3/4] Evaluate binary..."
$PYBIN eval_tumor_normal_binary.py \
  --cells_csv datasets/gao_et_al_2021/Data_Gao2021_Breast/Breast/Cells.csv \
  --names_txt "${PFX}.cells.txt" \
  --labels_tsv "${PFX}.FINAL.binary.labels.tsv" \
  --out_prefix "${PFX}.FINAL.eval"

echo "[4/4] Quick diagnostic: latent variance + pairwise cosine stats..."
$PYBIN - <<'PY'
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

pfx="results/gao2021_breast/runs/repro_CNAedge_p4/gao2021_CNAedge_p4"
Z=np.loadtxt(pfx+".latent.tsv")
print("Z shape:", Z.shape)
print("Z var per-dim min/median/max:",
      float(Z.var(axis=0).min()), float(np.median(Z.var(axis=0))), float(Z.var(axis=0).max()))
idx=np.random.RandomState(0).choice(Z.shape[0], size=500, replace=False)
S=cosine_similarity(Z[idx])
off=S[np.triu_indices_from(S,k=1)]
print("pairwise cos(Z) offdiag mean/std/min/max:",
      float(off.mean()), float(off.std()), float(off.min()), float(off.max()))
PY

echo "=== Done ==="
date
