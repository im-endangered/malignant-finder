#!/bin/bash
#SBATCH --job-name=qian2020_leiden
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=genacc_q
set -euo pipefail

# --- Environment Setup ---
PYBIN=/gpfs/home/pb25e/.conda/envs/scgclust_rna/bin/python

# Paths (Qian 2020)
DATA_DIR=datasets/qian_et_al_2020
CELLS_CSV=${DATA_DIR}/Data_Qian2020_Breast/Cells.csv
GENES_TXT=${DATA_DIR}/Data_Qian2020_Breast/Genes.txt
MTX=${DATA_DIR}/Data_Qian2020_Breast/Exp_data_UMIcounts.mtx

CNA_RAW=${DATA_DIR}/CNA_matrix/CNA_matrix_Qian2020_Breast.csv
CNA_FILT=${DATA_DIR}/CNA_matrix/CNA_matrix_Qian2020_Breast.var_ge_0.02.csv

SIG_GENES=signatures/master_signature_genes_unique.txt

# Outputs
RUN_TAG=qian2020_RNAfeat_CNAedge_leiden
INPUT_NPZ=results/qian2020_breast/inputs/${RUN_TAG}.npz
OUT_DIR=results/qian2020_breast/runs/${RUN_TAG}
PFX=${OUT_DIR}/${RUN_TAG}

mkdir -p results/qian2020_breast/inputs
mkdir -p ${OUT_DIR}
mkdir -p logs

echo "Started at: $(date)"
echo "[INFO] RUN_TAG=${RUN_TAG}"


# --- Step 1: Preprocess Data (build graph inputs) ---
echo "[STEP 1] Building graph inputs..."
$PYBIN build_graph_inputs_v4_cnaedge_rawcos.py \
  --cna_csv "${CNA_FILT}" \
  --cells_csv "${CELLS_CSV}" \
  --genes_txt "${GENES_TXT}" \
  --mtx "${MTX}" \
  --sig_genes_txt "${SIG_GENES}" \
  --k 20 \
  --power 4.0 \
  --allow_missing_cna \
  --out "${INPUT_NPZ}"


# --- Step 2: Train GAT + Leiden clustering ---
echo "[STEP 2] Training GAT model + Leiden clustering..."
$PYBIN train_rna.py \
  --npz "${INPUT_NPZ}" \
  --hidden1 64 \
  --hidden2 16 \
  --epochs 1000 \
  --lr 1e-3 \
  --clusterer leiden \
  --latent_knn 30 \
  --latent_metric euclidean \
  --latent_weight_mode connectivity \
  --resolution 2.0 \
  --lambda_contrast 0.25 \
  --temperature 0.1 \
  --n_neg 256 \
  --zscore_features \
  --out_prefix "${PFX}"

# --- Step 3: Collapse to Binary (tumor/normal) ---
echo "[STEP 3] Collapsing to binary labels..."
$PYBIN collapse_to_binary_by_malignant_fraction.py \
  --cells_csv "${CELLS_CSV}" \
  --names_txt "${PFX}.cells.txt" \
  --labels_tsv "${PFX}.labels.tsv" \
  --out_labels_tsv "${PFX}.FINAL.binary.labels.tsv" \
  --tumor_frac_thr 0.50 \
  --force_two_classes

# --- Step 4: Evaluate ---
echo "[STEP 4] Evaluating results..."
$PYBIN eval_tumor_normal_binary.py \
  --cells_csv "${CELLS_CSV}" \
  --names_txt "${PFX}.cells.txt" \
  --labels_tsv "${PFX}.FINAL.binary.labels.tsv" \
  --out_prefix "${PFX}.FINAL.eval"

echo "=== Job Finished at $(date) ==="
