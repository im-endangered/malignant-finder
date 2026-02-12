#!/bin/bash
#SBATCH --job-name=scgclust_breast_2runs
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
NPZ=results/gao2021_breast/inputs/gao2021_breast_cna_edges_rna_features.npz

# Make a unique run directory for THIS job
RUNROOT=results/gao2021_breast/runs/${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)
mkdir -p "${RUNROOT}"

echo "[INFO] Run root: ${RUNROOT}"
echo "[INFO] NPZ: ${NPZ}"

# -------------------------
# Run A: n_clusters = 10
# -------------------------
OUTA=${RUNROOT}/nclust10
mkdir -p "${OUTA}"

echo "=== Run A (n_clusters=10) ==="
$PYBIN train_rna.py \
  --npz "${NPZ}" \
  --n_clusters 10 \
  --hidden1 64 --hidden2 16 \
  --dropout 0.0 \
  --epochs 500 \
  --lr 1e-3 \
  --clusterer gmm \
  --out_prefix "${OUTA}/gao2021_breast_scgclustlike"

# -------------------------
# Run B: n_clusters = 2
# -------------------------
OUTB=${RUNROOT}/nclust2
mkdir -p "${OUTB}"

echo "=== Run B (n_clusters=2) ==="
$PYBIN train_rna.py \
  --npz "${NPZ}" \
  --n_clusters 2 \
  --hidden1 64 --hidden2 16 \
  --dropout 0.0 \
  --epochs 500 \
  --lr 1e-3 \
  --clusterer gmm \
  --out_prefix "${OUTB}/gao2021_breast_scgclustlike"

echo "=== Done ==="
date
echo "[OK] Results written under: ${RUNROOT}"
