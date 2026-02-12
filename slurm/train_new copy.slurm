#!/bin/bash
#SBATCH --job-name=scgclust_v4_k2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=genacc_q

RUN=results/gao2021_breast/runs/v4_sig_CNAvar0.02_rawcos_p4_k10_contrast
PFX=$RUN/gao2021_breast_v4_sig_CNAvar0.02_rawcos_p4_k10_contrast

PYBIN=/gpfs/home/pb25e/.conda/envs/scgclust_rna/bin/python


echo "[RUN] Training..."
$PYBIN train_rna.py \
  --npz results/gao2021_breast/inputs/v4_sig_CNAvar0.02_k20_rawcos_p4.npz \
  --n_clusters 10 \
  --hidden1 64 --hidden2 16 \
  --dropout 0.0 \
  --epochs 500 \
  --lr 1e-3 \
  --clusterer gmm \
  --lambda_contrast 1.0 \
  --n_neg 256 \
  --temperature 0.2 \
  --zscore_features \
  --out_prefix $PFX
  
echo "=== Done ==="
date
