#!/bin/bash
#SBATCH --job-name=sigma02
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=genacc_q

set -euo pipefail

echo "=== Job info ==="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"
echo "================"

cd /gpfs/research/fangroup/pb25e/malignant

PYBIN=/gpfs/home/pb25e/.conda/envs/scgclust_rna/bin/python
mkdir -p results/gao2021_breast/runs/v4_sig_CP10K_zscore_k15

$PYBIN train_rna.py \
  --npz results/gao2021_breast/inputs/gao2021_breast_v4_sig_CP10K_zscore_k20_sigma020.npz \
  --n_clusters 15 \
  --hidden1 64 --hidden2 16 \
  --dropout 0.0 \
  --epochs 500 \
  --lr 1e-3 \
  --clusterer gmm \
  --out_prefix results/gao2021_breast/runs/v4_sig_CP10K_zscore_k15/gao2021_breast_v4_sig_CP10K_zscore_k15

echo "=== Done ==="
date
