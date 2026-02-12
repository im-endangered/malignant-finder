#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYBIN=/gpfs/home/pb25e/.conda/envs/scgclust_rna/bin/python

CELLS_CSV="datasets/gao_et_al_2021/Data_Gao2021_Breast/Breast/Cells.csv"

# must match what you used in submit_grid.sh
BASE_RUN="results/gao2021_breast/runs/grid_v4_rawcos_p4"

# threshold used to map clusters -> tumor/normal
TUMOR_THR="${1:-0.50}"   # allow: ./slurm/eval_grid.sh 0.55

OUT_SUMMARY="${BASE_RUN}/_grid_summary_thr${TUMOR_THR}.tsv"

mkdir -p "$BASE_RUN"

echo -e "run_dir\tprefix\tthr\tacc\tprecision\trecall\tf1\tTP\tFP\tFN\tTN\tchosen_tumor_label\tcounts" > "$OUT_SUMMARY"

shopt -s nullglob
runs=( "${BASE_RUN}"/* )
if [[ ${#runs[@]} -eq 0 ]]; then
  echo "[ERR] No runs found under: $BASE_RUN"
  exit 1
fi

echo "[INFO] Found ${#runs[@]} run folders under $BASE_RUN"
echo "[INFO] Using tumor_frac_thr=$TUMOR_THR"
echo "[INFO] Writing summary to: $OUT_SUMMARY"
echo

n_done=0
n_skip=0

for R in "${runs[@]}"; do
  # find a prefix inside this run folder (we wrote gao2021_<TAG>.*)
  # We'll look for exactly one labels file; if multiple, take the newest.
  labels_files=( "$R"/gao2021_*.labels.tsv )
  if [[ ${#labels_files[@]} -eq 0 ]]; then
    n_skip=$((n_skip+1))
    continue
  fi

  # newest labels file
  LABELS="$(ls -t "${labels_files[@]}" | head -n 1)"
  PFX="${LABELS%.labels.tsv}"

  NAMES="${PFX}.cells.txt"
  if [[ ! -s "$NAMES" ]]; then
    echo "[WARN] Missing names_txt: $NAMES (skip)"
    n_skip=$((n_skip+1))
    continue
  fi

  if [[ ! -s "$LABELS" ]]; then
    echo "[WARN] Missing labels_tsv: $LABELS (skip)"
    n_skip=$((n_skip+1))
    continue
  fi

  # (1) collapse clusters -> binary labels
  BIN="${PFX}.FINAL.binary.labels.tsv"
  $PYBIN collapse_to_binary_by_malignant_fraction.py \
    --cells_csv "$CELLS_CSV" \
    --names_txt "$NAMES" \
    --labels_tsv "$LABELS" \
    --out_labels_tsv "$BIN" \
    --tumor_frac_thr "$TUMOR_THR" \
    --force_two_classes \
    > "${PFX}.collapse.log" 2>&1

  # (2) evaluate
  EPFX="${PFX}.FINAL.binary_eval"
  $PYBIN eval_tumor_normal_binary.py \
    --cells_csv "$CELLS_CSV" \
    --names_txt "$NAMES" \
    --labels_tsv "$BIN" \
    --out_prefix "$EPFX" \
    > "${EPFX}.log" 2>&1

  # (3) parse summary.txt into one line
  SUMTXT="${EPFX}.summary.txt"
  if [[ ! -s "$SUMTXT" ]]; then
    echo "[WARN] Missing eval summary: $SUMTXT (skip)"
    n_skip=$((n_skip+1))
    continue
  fi

  # Expected lines:
  # chosen_tumor_label = X
  # tumor_fraction_by_label = {...}
  # TP FP FN TN = ...
  # acc=... precision=... recall=... f1=...
  chosen="$(grep -E 'chosen_tumor_label' "$SUMTXT" | awk -F'=' '{gsub(/ /,"",$2); print $2}' | tail -n 1)"
  counts="$(grep -E '^TP FP FN TN' "$SUMTXT" | sed -E 's/^TP FP FN TN = *//' | tail -n 1)"
  metrics="$(grep -E '^acc=' "$SUMTXT" | tail -n 1)"

  # pull numbers safely
  TP="$(echo "$counts" | awk '{print $1}')"
  FP="$(echo "$counts" | awk '{print $2}')"
  FN="$(echo "$counts" | awk '{print $3}')"
  TN="$(echo "$counts" | awk '{print $4}')"

  acc="$(echo "$metrics" | sed -n 's/.*acc=\([0-9.]*\).*/\1/p')"
  prec="$(echo "$metrics" | sed -n 's/.*precision=\([0-9.]*\).*/\1/p')"
  rec="$(echo "$metrics" | sed -n 's/.*recall=\([0-9.]*\).*/\1/p')"
  f1="$(echo "$metrics" | sed -n 's/.*f1=\([0-9.]*\).*/\1/p')"

  # also record predicted label distribution
  pred_counts="$($PYBIN - <<PY
import pandas as pd
p="$BIN"
s=pd.read_csv(p,header=None)[0]
vc=s.value_counts().to_dict()
print(vc)
PY
)"

  echo -e "${R}\t${PFX}\t${TUMOR_THR}\t${acc}\t${prec}\t${rec}\t${f1}\t${TP}\t${FP}\t${FN}\t${TN}\t${chosen}\t${pred_counts}" >> "$OUT_SUMMARY"

  n_done=$((n_done+1))
  if (( n_done % 20 == 0 )); then
    echo "[INFO] Evaluated $n_done runs..."
  fi
done

echo
echo "[OK] Done. evaluated=$n_done skipped=$n_skip"
echo "[OUT] $OUT_SUMMARY"
echo
echo "Top 15 by f1:"
# print header + top 15 rows by f1 (column 8)
{ head -n 1 "$OUT_SUMMARY"; tail -n +2 "$OUT_SUMMARY" | sort -t $'\t' -k8,8gr | head -n 15; } | column -t -s $'\t'
