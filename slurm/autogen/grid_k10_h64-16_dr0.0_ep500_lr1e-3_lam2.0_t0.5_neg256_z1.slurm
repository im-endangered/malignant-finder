#!/bin/bash
#SBATCH --job-name=grid_k10_h64-16_dr0.0_ep500_lr1e-3_lam2.0_t0.5_neg256_z1
#SBATCH --output=/gpfs/research/fangroup/pb25e/malignant/logs/%x_%j.out
#SBATCH --error=/gpfs/research/fangroup/pb25e/malignant/logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=genacc_q

set -euo pipefail

# always run from project root so relative paths work
cd "/gpfs/research/fangroup/pb25e/malignant"

mkdir -p "results/gao2021_breast/runs/grid_v4_rawcos_p4/k10_h64-16_dr0.0_ep500_lr1e-3_lam2.0_t0.5_neg256_z1"

echo "[RUN] $(date)"
echo "[INFO] NPZ=results/gao2021_breast/inputs/v4_sig_CNAvar0.02_k20_rawcos_p4.npz"
echo "[INFO] OUT=results/gao2021_breast/runs/grid_v4_rawcos_p4/k10_h64-16_dr0.0_ep500_lr1e-3_lam2.0_t0.5_neg256_z1/gao2021_k10_h64-16_dr0.0_ep500_lr1e-3_lam2.0_t0.5_neg256_z1"

CMD="/gpfs/home/pb25e/.conda/envs/scgclust_rna/bin/python train_rna.py \
  --npz results/gao2021_breast/inputs/v4_sig_CNAvar0.02_k20_rawcos_p4.npz \
  --n_clusters 10 \
  --hidden1 64 --hidden2 16 \
  --dropout 0.0 \
  --epochs 500 \
  --lr 1e-3 \
  --clusterer gmm \
  --lambda_contrast 2.0 \
  --n_neg 256 \
  --temperature 0.5 \
  --out_prefix results/gao2021_breast/runs/grid_v4_rawcos_p4/k10_h64-16_dr0.0_ep500_lr1e-3_lam2.0_t0.5_neg256_z1/gao2021_k10_h64-16_dr0.0_ep500_lr1e-3_lam2.0_t0.5_neg256_z1"

CMD="${CMD} --zscore_features"

echo "[CMD] $CMD"
eval "$CMD"

echo "=== Done ==="
date
