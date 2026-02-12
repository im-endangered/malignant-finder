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

NPZ="results/gao2021_breast/inputs/gao2021_breast_rnaEdge_cnaWeight_rnaFeat_v4.npz"
OUTDIR="results/gao2021_breast/runs/v4_k2"
OUTPREFIX="${OUTDIR}/gao2021_breast_scgclustlike_v4_k2"

mkdir -p logs "${OUTDIR}"

echo "[RUN] Training..."
$PYBIN train_rna.py \
  --npz "${NPZ}" \
  --n_clusters 2 \
  --hidden1 64 --hidden2 16 \
  --dropout 0.0 \
  --epochs 500 \
  --lr 1e-3 \
  --clusterer gmm \
  --out_prefix "${OUTPREFIX}"

echo "[RUN] Eval tumor/normal (binary)..."
$PYBIN eval_tumor_normal_binary.py \
  --cells_csv datasets/gao_et_al_2021/Data_Gao2021_Breast/Breast/Cells.csv \
  --names_txt "${OUTPREFIX}.cells.txt" \
  --labels_tsv "${OUTPREFIX}.labels.tsv" \
  --out_prefix "${OUTPREFIX}_eval"

echo "=== Done ==="
date
