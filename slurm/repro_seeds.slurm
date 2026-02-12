#!/bin/bash
#SBATCH --job-name=scg_seed
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=genacc_q
#SBATCH --array=0-9

set -euo pipefail

PYBIN=/gpfs/home/pb25e/.conda/envs/scgclust_rna/bin/python

NPZ=results/gao2021_breast/inputs/v4_sig_CNAvar0.02_k20_rawcos_p4.npz
CELLS=datasets/gao_et_al_2021/Data_Gao2021_Breast/Breast/Cells.csv

SEEDS=(0 1 2 3 4 5 6 7 8 9)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

RUN_DIR=results/gao2021_breast/runs/repro_seeds_p4
mkdir -p "$RUN_DIR"
PFX="$RUN_DIR/gao2021_seed${SEED}"

$PYBIN train_rna_seed.py \
  --npz "$NPZ" \
  --seed "$SEED" \
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

$PYBIN collapse_to_binary_by_malignant_fraction.py \
  --cells_csv "$CELLS" \
  --names_txt "${PFX}.cells.txt" \
  --labels_tsv "${PFX}.labels.tsv" \
  --out_labels_tsv "${PFX}.FINAL.binary.labels.tsv" \
  --tumor_frac_thr 0.50 \
  --force_two_classes

$PYBIN eval_tumor_normal_binary.py \
  --cells_csv "$CELLS" \
  --names_txt "${PFX}.cells.txt" \
  --labels_tsv "${PFX}.FINAL.binary.labels.tsv" \
  --out_prefix "${PFX}.FINAL.eval"

echo "DONE seed=$SEED"
