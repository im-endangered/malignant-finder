#!/bin/bash
#SBATCH --job-name=reproducibility
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=genacc_q

# --- Environment Setup ---
PYBIN=/gpfs/home/pb25e/.conda/envs/scgclust_rna/bin/python

# Define common paths
INPUT_NPZ=results/gao2021_breast/inputs/epoch1000n20.npz
OUT_DIR=results/gao2021_breast/runs/epoch1000n20
PFX=${OUT_DIR}/epoch1000n20
CELLS_CSV=datasets/gao_et_al_2021/Data_Gao2021_Breast/Breast/Cells.csv

# Create output directories if they don't exist
mkdir -p results/gao2021_breast/inputs
mkdir -p $OUT_DIR
mkdir -p logs

echo "Started at: $(date)"

# --- Step 1: Preprocess Data ---
echo "[STEP 1] Building graph inputs..."
$PYBIN build_graph_inputs_v4_cnaedge_rawcos.py \
  --cna_csv datasets/gao_et_al_2021/CNA_matrix_Gao2021_Breast/CNAs_Breast.var_ge_0.02.csv \
  --cells_csv $CELLS_CSV \
  --genes_txt datasets/gao_et_al_2021/Data_Gao2021_Breast/Breast/Genes.txt \
  --mtx datasets/gao_et_al_2021/Data_Gao2021_Breast/Breast/Exp_data_UMIcounts.mtx \
  --sig_genes_txt signatures/master_signature_genes_unique.txt \
  --k 20 \
  --power 4.0 \
  --out $INPUT_NPZ

# --- Step 2: Train GAT ---
echo "[STEP 2] Training GAT model..."
$PYBIN train_rna.py \
  --npz $INPUT_NPZ \
  --n_clusters 20 \
  --hidden1 64 \
  --hidden2 16 \
  --epochs 1000 \
  --lr 1e-3 \
  --clusterer gmm \
  --lambda_contrast 0.25 \
  --temperature 0.1 \
  --n_neg 256 \
  --zscore_features \
  --out_prefix $PFX

# --- Step 3: Collapse to Binary ---
echo "[STEP 3] Collapsing to binary labels..."
$PYBIN collapse_to_binary_by_malignant_fraction.py \
  --cells_csv $CELLS_CSV \
  --names_txt ${PFX}.cells.txt \
  --labels_tsv ${PFX}.labels.tsv \
  --out_labels_tsv ${PFX}.FINAL.binary.labels.tsv \
  --tumor_frac_thr 0.50 \
  --force_two_classes

# --- Step 4: Evaluate ---
echo "[STEP 4] Evaluating results..."
$PYBIN eval_tumor_normal_binary.py \
  --cells_csv $CELLS_CSV \
  --names_txt ${PFX}.cells.txt \
  --labels_tsv ${PFX}.FINAL.binary.labels.tsv \
  --out_prefix ${PFX}.FINAL.eval

echo "=== Job Finished at $(date) ==="