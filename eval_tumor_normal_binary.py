#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_tumor_normal_binary.py

Binary tumor/normal evaluation for clustering outputs.

Inputs:
  --cells_csv   : Gao2021 Cells.csv containing columns: cell_name, cell_type (or similar)
  --names_txt   : one cell_name per line, in the order used for training outputs
  --labels_tsv  : either:
                   (A) 1-column file: one integer label per line (no header), OR
                   (B) 2-column TSV: cell_name <tab> label   (header optional)
  --out_prefix  : output prefix

Behavior:
  - Cleans empty lines
  - Aligns labels to names:
      * if labels file includes cell_name column -> join by name
      * else -> assumes same order; if off-by-one, truncates to min length with warning
  - Maps ground truth tumor vs normal from Cells.csv cell_type values:
      tumor if 'tumor' substring in cell_type (case-insensitive), else normal.
      (You can tweak this logic easily below.)
  - Chooses which predicted cluster is "tumor" by higher tumor fraction.
  - Reports TP/FP/FN/TN, accuracy, precision, recall, F1.

Outputs:
  <out_prefix>.summary.txt
  <out_prefix>.per_cell.tsv
"""

from __future__ import annotations
import argparse
import sys
import numpy as np
import pandas as pd


def read_nonempty_lines(path: str) -> list[str]:
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip() != ""]
    return lines


def read_names_txt(names_txt: str) -> pd.Series:
    names = read_nonempty_lines(names_txt)
    if len(names) == 0:
        raise ValueError(f"names_txt is empty after removing blank lines: {names_txt}")
    return pd.Series(names, dtype=str, name="cell_name")


def _read_labels_as_dataframe(labels_tsv: str) -> pd.DataFrame:
    """
    Try multiple parses because some of your files are headerless and 1-column.
    Returns a dataframe with either:
      - columns: ["label"]  (1-col)
      - columns: ["cell_name","label"] (2-col)
    """
    # First try: tab-separated with header inference
    try:
        df = pd.read_csv(labels_tsv, sep="\t", header=0, dtype=str)
        # If it read a single column with weird header (e.g., '1'), treat as no-header
        if df.shape[1] == 1 and df.columns[0] not in ("label", "cluster", "pred", "y", "labels", "cell_name"):
            # fallback to no-header
            raise ValueError("Likely no-header 1-col file; retrying.")
        return df
    except Exception:
        pass

    # Second try: no header, tab-separated
    df = pd.read_csv(labels_tsv, sep="\t", header=None, dtype=str)

    if df.shape[1] == 1:
        df.columns = ["label"]
        return df

    if df.shape[1] >= 2:
        # Assume first two columns are cell_name and label, ignore extras
        df = df.iloc[:, :2].copy()
        df.columns = ["cell_name", "label"]
        return df

    raise ValueError("labels_tsv format not recognized.")


def read_labels(labels_tsv: str, names: pd.Series) -> pd.Series:
    """
    Returns integer labels aligned to `names` order.
    """
    df = _read_labels_as_dataframe(labels_tsv)

    # Clean whitespace and drop empty rows
    for c in df.columns:
        df[c] = df[c].astype(str).map(lambda x: x.strip())
    df = df.replace("", np.nan).dropna(how="any").reset_index(drop=True)

    if df.shape[1] == 1 and "label" in df.columns:
        # Order-based labels
        lab = df["label"].astype(int).reset_index(drop=True)

        n_names = len(names)
        n_lab = len(lab)

        if n_lab == n_names:
            return lab

        # Off-by-one (your case): allow truncate-to-min with warning
        m = min(n_lab, n_names)
        print(
            f"[WARN] labels count ({n_lab}) != names count ({n_names}). "
            f"Proceeding by truncating both to {m} (order-based alignment).",
            file=sys.stderr,
        )
        return lab.iloc[:m].reset_index(drop=True)

    # Name-based labels
    if "cell_name" in df.columns and "label" in df.columns:
        df["label"] = df["label"].astype(int)
        # Merge to names order
        merged = pd.DataFrame({"cell_name": names.values}).merge(df, on="cell_name", how="left")

        missing = merged["label"].isna().sum()
        if missing > 0:
            ex = merged.loc[merged["label"].isna(), "cell_name"].head(5).tolist()
            raise ValueError(f"Missing labels for {missing} cells after name-join. Example missing: {ex}")

        return merged["label"].astype(int)

    raise ValueError("labels_tsv format not recognized after cleaning.")


def load_truth(cells_csv: str) -> pd.DataFrame:
    """
    Loads Cells.csv and returns df with columns:
      cell_name, cell_type, is_tumor (bool)
    """
    cells = pd.read_csv(cells_csv)
    if "cell_name" not in cells.columns:
        raise ValueError("Cells.csv must contain column 'cell_name'.")

    # Accept common variants
    if "cell_type" not in cells.columns:
        # try some fallbacks
        for alt in ["CellType", "type", "annotation", "label"]:
            if alt in cells.columns:
                cells = cells.rename(columns={alt: "cell_type"})
                break

    if "cell_type" not in cells.columns:
        raise ValueError(
            "Cells.csv must contain 'cell_type' (or a recognizable alternative). "
            f"Columns found: {list(cells.columns)[:20]}"
        )

    # tumor if substring 'tumor' present; else normal
    ct = cells["cell_type"].astype(str).str.lower()
    cells["is_tumor"] = ct.str.contains("malignant", regex=False)

    return cells[["cell_name", "cell_type", "is_tumor"]].copy()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    assert y_true.shape == y_pred.shape
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    acc = (tp + tn) / max(1, (tp + fp + fn + tn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = 0.0 if (prec + rec) == 0 else (2 * prec * rec) / (prec + rec)

    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "acc": acc, "precision": prec, "recall": rec, "f1": f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells_csv", required=True)
    ap.add_argument("--names_txt", required=True)
    ap.add_argument("--labels_tsv", required=True)
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    names = read_names_txt(args.names_txt)
    labs = read_labels(args.labels_tsv, names)

    # If order-based truncation happened, names must be truncated to match too
    if len(labs) != len(names):
        m = min(len(labs), len(names))
        names = names.iloc[:m].reset_index(drop=True)
        labs = labs.iloc[:m].reset_index(drop=True)

    truth = load_truth(args.cells_csv)

    # Align truth to our names
    merged = pd.DataFrame({"cell_name": names.values}).merge(truth, on="cell_name", how="left")
    missing_truth = merged["is_tumor"].isna().sum()
    if missing_truth > 0:
        # It's okay if Cells.csv is missing some annotations, but we should not evaluate those.
        ex = merged.loc[merged["is_tumor"].isna(), "cell_name"].head(5).tolist()
        print(f"[WARN] Missing cell_type for {missing_truth} cells. Example: {ex}", file=sys.stderr)

    # Keep only rows with truth available
    keep = ~merged["is_tumor"].isna()
    merged = merged.loc[keep].reset_index(drop=True)
    labs = labs.loc[keep].reset_index(drop=True)

    # Convert to binary prediction: choose tumor_label as the cluster with higher tumor fraction
    y_true = merged["is_tumor"].astype(int).to_numpy()
    y_lab = labs.to_numpy()

    uniq = np.unique(y_lab)
    if len(uniq) < 2:
        print(f"[WARN] Only {len(uniq)} unique predicted label(s): {uniq}. Metrics may be degenerate.", file=sys.stderr)

    tumor_fracs = {}
    for u in uniq:
        mask = (y_lab == u)
        tumor_fracs[int(u)] = float(np.mean(y_true[mask])) if np.any(mask) else 0.0

    tumor_label = max(tumor_fracs, key=tumor_fracs.get)
    y_pred = (y_lab == tumor_label).astype(int)

    metrics = compute_metrics(y_true, y_pred)

    # Write outputs
    out_summary = f"{args.out_prefix}.summary.txt"
    out_per_cell = f"{args.out_prefix}.per_cell.tsv"

    with open(out_summary, "w") as f:
        f.write("=== Tumor/Normal evaluation (binary) ===\n")
        f.write(f"cells_total_with_truth = {len(y_true)}\n")
        f.write(f"unique_pred_labels = {uniq.tolist()}\n")
        f.write(f"tumor_fraction_by_label = {tumor_fracs}\n")
        f.write(f"chosen_tumor_label = {tumor_label}\n")
        f.write(f"TP FP FN TN = {metrics['TP']} {metrics['FP']} {metrics['FN']} {metrics['TN']}\n")
        f.write(f"acc={metrics['acc']:.4f} precision={metrics['precision']:.4f} "
                f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}\n")

    per_cell = merged.copy()
    per_cell["pred_label"] = y_lab
    per_cell["pred_is_tumor"] = y_pred
    per_cell["true_is_tumor"] = y_true
    per_cell.to_csv(out_per_cell, sep="\t", index=False)

    print("=== Tumor/Normal evaluation (binary) ===")
    print("chosen_tumor_label =", tumor_label)
    print("tumor_fraction_by_label =", tumor_fracs)
    print("TP FP FN TN =", metrics["TP"], metrics["FP"], metrics["FN"], metrics["TN"])
    print(f"acc={metrics['acc']:.4f} precision={metrics['precision']:.4f} "
          f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}")
    print("Wrote:")
    print(" ", out_summary)
    print(" ", out_per_cell)


if __name__ == "__main__":
    main()
