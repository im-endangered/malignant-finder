#!/bin/bash
#SBATCH --job-name=scgclust_v4_k2
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


echo "[RUN] Training..."
$PYBIN train_rna.py \
  --npz results/gao2021_breast/inputs/v4_sig_CNAvar0.02_k10_rawcos_p2.npz \
  --n_clusters 10 \
  --hidden1 64 --hidden2 16 \
  --dropout 0.0 \
  --epochs 500 \
  --lr 1e-3 \
  --clusterer gmm \
  --out_prefix results/gao2021_breast/runs/v4_sig_CNAvar0.02_rawcos_p2_k10graph/gao2021_breast_v4_sig_CNAvar0.02_rawcos_p2_k10graph
  
echo "=== Done ==="
date
