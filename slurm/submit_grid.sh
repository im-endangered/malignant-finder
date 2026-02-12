#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM_DIR="$ROOT/slurm/autogen"
LOG_DIR="$ROOT/logs"

mkdir -p "$SLURM_DIR" "$LOG_DIR"

PYBIN=/gpfs/home/pb25e/.conda/envs/scgclust_rna/bin/python

# ---- fixed inputs (edit these if needed) ----
NPZ="results/gao2021_breast/inputs/v4_sig_CNAvar0.02_k20_rawcos_p4.npz"
BASE_RUN="results/gao2021_breast/runs/grid_v4_rawcos_p4"

# ---- grid (edit freely) ----
# model/training
CLUSTERS=(10)
H1=(64)
H2=(16)
DROPOUT=(0.0)
EPOCHS=(500)
LR=(1e-3)

# contrastive
LAM=(0.25 0.5 1.0 2.0)
TEMP=(0.1 0.2 0.5)
NNEG=(128 256 512)

# feature normalization toggle
ZSCORE=(1)   # 1 => --zscore_features, 0 => (no flag)

# resources
PARTITION="genacc_q"
TIME="08:00:00"
CPUS=8
MEM="64G"

# optional: throttle submissions
MAX_SUBMIT_PER_RUN=999999

submitted=0

for K in "${CLUSTERS[@]}"; do
for hidden1 in "${H1[@]}"; do
for hidden2 in "${H2[@]}"; do
for dr in "${DROPOUT[@]}"; do
for ep in "${EPOCHS[@]}"; do
for lr in "${LR[@]}"; do
for lam in "${LAM[@]}"; do
for t in "${TEMP[@]}"; do
for nn in "${NNEG[@]}"; do
for z in "${ZSCORE[@]}"; do

  TAG="k${K}_h${hidden1}-${hidden2}_dr${dr}_ep${ep}_lr${lr}_lam${lam}_t${t}_neg${nn}_z${z}"
  RUN="${BASE_RUN}/${TAG}"
  PFX="${RUN}/gao2021_${TAG}"

  JOB="grid_${TAG}"
  SLURM_FILE="${SLURM_DIR}/${JOB}.slurm"

  cat > "$SLURM_FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB}
#SBATCH --output=${LOG_DIR}/%x_%j.out
#SBATCH --error=${LOG_DIR}/%x_%j.err
#SBATCH --time=${TIME}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --partition=${PARTITION}

set -euo pipefail

# always run from project root so relative paths work
cd "${ROOT}"

mkdir -p "${RUN}"

echo "[RUN] \$(date)"
echo "[INFO] NPZ=${NPZ}"
echo "[INFO] OUT=${PFX}"

CMD="${PYBIN} train_rna.py \\
  --npz ${NPZ} \\
  --n_clusters ${K} \\
  --hidden1 ${hidden1} --hidden2 ${hidden2} \\
  --dropout ${dr} \\
  --epochs ${ep} \\
  --lr ${lr} \\
  --clusterer gmm \\
  --lambda_contrast ${lam} \\
  --n_neg ${nn} \\
  --temperature ${t} \\
  --out_prefix ${PFX}"

EOF

  if [[ "$z" == "1" ]]; then
    echo 'CMD="${CMD} --zscore_features"' >> "$SLURM_FILE"
  fi

  cat >> "$SLURM_FILE" <<'EOF'

echo "[CMD] $CMD"
eval "$CMD"

echo "=== Done ==="
date
EOF

  chmod +x "$SLURM_FILE"

  # submit
  sbatch "$SLURM_FILE"
  submitted=$((submitted+1))

  if [[ "$submitted" -ge "$MAX_SUBMIT_PER_RUN" ]]; then
    echo "[INFO] Reached MAX_SUBMIT_PER_RUN=$MAX_SUBMIT_PER_RUN; stopping."
    exit 0
  fi

done; done; done; done; done; done; done; done; done; done

echo "[OK] Submitted ${submitted} jobs."
