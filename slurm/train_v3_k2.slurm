#!/bin/bash
#SBATCH --job-name=scgclust_rna_breast_v3_k2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=genacc_q

set -euo pipefail

cd /gpfs/research/fangroup/pb25e/malignant
PYBIN=/gpfs/home/pb25e/.conda/envs/scgclust_rna/bin/python

mkdir -p results/gao2021_breast/runs/v3_k2

$PYBIN train_rna.py \
  --npz results/gao2021_breast/inputs/gao2021_breast_cna_edges_rna_features_v3.npz \
  --n_clusters 2 \
  --hidden1 64 --hidden2 16 \
  --dropout 0.0 \
  --epochs 500 \
  --lr 1e-3 \
  --clusterer gmm \
  --out_prefix results/gao2021_breast/runs/v3_k2/gao2021_breast_scgclustlike_v3_k2

echo "=== Done ==="
date



