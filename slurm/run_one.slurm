#!/bin/bash
#SBATCH --job-name=scg_one
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=genacc_q

set -euo pipefail

PYBIN=/gpfs/home/pb25e/.conda/envs/scgclust_rna/bin/python

# --- fixed inputs (edit if needed) ---
NPZ=results/gao2021_breast/inputs/v4_sig_CNAvar0.02_k20_rawcos_p4.npz
CELLS=datasets/gao_et_al_2021/Data_Gao2021_Breast/Breast/Cells.csv

# --- run config (override by sbatch --export=...) ---
RUN_DIR=${RUN_DIR:-results/gao2021_breast/runs/repro_one}
PREFIX_NAME=${PREFIX_NAME:-gao2021_repro_one}
K=${K:-10}
H1=${H1:-64}
H2=${H2:-16}
DROPOUT=${DROPOUT:-0.0}
EPOCHS=${EPOCHS:-500}
LR=${LR:-1e-3}
CLUSTERER=${CLUSTERER:-gmm}
Z1=${Z1:-1}                 # 1 => --zscore_features
LAM=${LAM:-0.25}            # lambda_contrast
TEMP=${TEMP:-0.1}           # temperature
NEG=${NEG:-256}             # n_neg
THR=${THR:-0.50}            # malignant-fraction threshold for mapping clusters->binary

mkdir -p "$RUN_DIR"
PFX="$RUN_DIR/$PREFIX_NAME"

echo "[RUN] $PFX"
date

echo "[1/4] Train..."
ZS_ARG=""
if [ "$Z1" = "1" ]; then ZS_ARG="--zscore_features"; fi

$PYBIN train_rna.py \
  --npz "$NPZ" \
  --n_clusters "$K" \
  --hidden1 "$H1" --hidden2 "$H2" \
  --dropout "$DROPOUT" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --clusterer "$CLUSTERER" \
  --lambda_contrast "$LAM" \
  --n_neg "$NEG" \
  --temperature "$TEMP" \
  $ZS_ARG \
  --out_prefix "$PFX"

echo "[2/4] Collapse clusters -> binary..."
$PYBIN collapse_to_binary_by_malignant_fraction.py \
  --cells_csv "$CELLS" \
  --names_txt "${PFX}.cells.txt" \
  --labels_tsv "${PFX}.labels.tsv" \
  --out_labels_tsv "${PFX}.FINAL.binary.labels.tsv" \
  --tumor_frac_thr "$THR" \
  --force_two_classes

echo "[3/4] Evaluate binary..."
$PYBIN eval_tumor_normal_binary.py \
  --cells_csv "$CELLS" \
  --names_txt "${PFX}.cells.txt" \
  --labels_tsv "${PFX}.FINAL.binary.labels.tsv" \
  --out_prefix "${PFX}.FINAL.eval"

echo "[4/4] Quick diagnostic: latent variance + pairwise cosine stats..."
$PYBIN - <<PY
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

Z = np.loadtxt("${PFX}.latent.tsv")
print("Z shape:", Z.shape)
print("Z var per-dim min/median/max:",
      float(Z.var(0).min()), float(np.median(Z.var(0))), float(Z.var(0).max()))
idx = np.random.RandomState(0).choice(Z.shape[0], size=min(500, Z.shape[0]), replace=False)
S = cosine_similarity(Z[idx])
off = S[np.triu_indices_from(S, k=1)]
print("pairwise cos(Z) offdiag mean/std/min/max:",
      float(off.mean()), float(off.std()), float(off.min()), float(off.max()))
PY

echo "=== Done ==="
date
