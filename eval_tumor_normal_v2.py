#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_tumor_normal_v2.py

Improved tumor/normal evaluation for clustering outputs where tumor cells
may be distributed across multiple clusters.

Inputs:
  --cells_csv   : Gao Breast Cells.csv containing columns: cell_name, cell_type (or similar)
  --names_txt   : model output list of cell names (one per line), in the same order as labels
  --labels_tsv  : model output labels TSV with either:
                  - a column named 'label' OR
                  - 2 columns: cell_name <tab> label
  --out_prefix  : prefix for output files

Outputs:
  <out_prefix>.tumor_cluster_report.tsv  (cluster composition table)
  <out_prefix>.tumor_normal.tsv          (per-cell true/pred for best-k rule)
  <out_prefix>.summary.txt               (metrics)
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def read_names_txt(path: str) -> pd.Series:
    names = pd.read_csv(path, header=None)[0].astype(str)
    return names


def read_labels_tsv(path: str, n: int, names: pd.Series) -> pd.Series:
    df = pd.read_csv(path, sep="\t")
    cols = [c.lower() for c in df.columns]

    # Case 1: column named label
    if "label" in cols:
        lab = df[df.columns[cols.index("label")]].astype(int)
        if len(lab) != n:
            raise ValueError(f"labels_tsv has {len(lab)} rows but names_txt has {n}")
        return lab.reset_index(drop=True)

    # Case 2: two columns (cell_name, label)
    if df.shape[1] >= 2:
        c0 = df.columns[0]
        c1 = df.columns[1]
        tmp = df[[c0, c1]].copy()
        tmp[c0] = tmp[c0].astype(str)
        tmp[c1] = tmp[c1].astype(int)

        # align by name order
        m = pd.DataFrame({"cell_name": names})
        out = m.merge(tmp, how="left", left_on="cell_name", right_on=c0)
        if out[c1].isna().any():
            missing = out.loc[out[c1].isna(), "cell_name"].head(10).tolist()
            raise ValueError(f"Some names in names_txt missing from labels_tsv. Examples: {missing}")
        return out[c1].astype(int)

    raise ValueError("labels_tsv format not recognized. Need a 'label' column or (cell_name, label).")


def normalize_celltype_to_binary(s: pd.Series) -> pd.Series:
    """
    Map various cell_type strings to binary tumor(1)/normal(0).
    You can expand these rules if needed.
    """
    x = s.astype("string")

    # treat NA / empty as missing
    missing_mask = x.isna() | (x.str.strip() == "")
    x = x.fillna("").str.strip().str.lower()

    tumor_keywords = ["tumor", "malignant", "cancer"]
    normal_keywords = ["normal", "immune", "stromal", "endothelial", "tcell", "bcell", "myeloid", "fibro", "epithelial"]

    y = pd.Series(pd.NA, index=x.index, dtype="Int64")
    for kw in tumor_keywords:
        y = y.mask(x.str.contains(kw, regex=False), 1)
    for kw in normal_keywords:
        y = y.mask(x.str.contains(kw, regex=False), 0)

    # if original was missing, keep missing
    y = y.mask(missing_mask, pd.NA)
    return y


def metrics_from_preds(y_true: np.ndarray, y_pred: np.ndarray):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    acc = (tp + tn) / max(1, (tp + fp + fn + tn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = 0.0 if (prec + rec) == 0 else (2 * prec * rec / (prec + rec))
    return tp, fp, fn, tn, acc, prec, rec, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells_csv", required=True)
    ap.add_argument("--names_txt", required=True)
    ap.add_argument("--labels_tsv", required=True)
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    cells = pd.read_csv(args.cells_csv)
    if "cell_name" not in cells.columns:
        raise ValueError("Cells.csv must contain 'cell_name' column")

    # find a likely cell type column
    ct_col = None
    for cand in ["cell_type", "celltype", "type", "annotation", "label"]:
        if cand in cells.columns:
            ct_col = cand
            break
    if ct_col is None:
        # fall back to any column containing "type"
        for c in cells.columns:
            if "type" in c.lower():
                ct_col = c
                break
    if ct_col is None:
        raise ValueError("Could not find a cell type column in Cells.csv")

    names = read_names_txt(args.names_txt)
    labels = read_labels_tsv(args.labels_tsv, n=len(names), names=names)

    # Align cells metadata to model output order
    meta = pd.DataFrame({"cell_name": names})
    meta = meta.merge(cells[["cell_name", ct_col]], how="left", on="cell_name")

    # true binary
    y_true = normalize_celltype_to_binary(meta[ct_col])
    missing = int(y_true.isna().sum())
    total = len(y_true)
    valid_mask = ~y_true.isna()

    print("=== Tumor/Normal evaluation (improved) ===")
    print(f"Rows total: {total}")
    print(f"Valid tumor/normal labels: {int(valid_mask.sum())}")
    print(f"Missing/unknown cell_type: {missing}")
    print(f"Cell type column used: {ct_col}")

    # restrict to valid
    y_true_v = y_true[valid_mask].astype(int).to_numpy()
    labels_v = labels[valid_mask].to_numpy()
    names_v = names[valid_mask].to_numpy()

    # cluster composition table
    dfc = pd.DataFrame({"cell_name": names_v, "cluster": labels_v, "tumor": y_true_v})
    grp = dfc.groupby("cluster").agg(
        n=("tumor", "size"),
        tumor_n=("tumor", "sum"),
    ).reset_index()
    grp["tumor_frac"] = grp["tumor_n"] / grp["n"]
    grp = grp.sort_values(["tumor_frac", "tumor_n", "n"], ascending=[False, False, False])

    report_path = str(out_prefix) + ".tumor_cluster_report.tsv"
    grp.to_csv(report_path, sep="\t", index=False)

    # ---- best single cluster as tumor ----
    best_single = None
    for cl in grp["cluster"].tolist():
        y_pred = (labels_v == cl).astype(int)
        m = metrics_from_preds(y_true_v, y_pred)
        if (best_single is None) or (m[-1] > best_single[-1]):
            best_single = (cl, *m)

    # ---- best multi-cluster set (top-k by tumor enrichment) ----
    clusters_sorted = grp["cluster"].tolist()
    best_multi = None
    best_k = None

    # Try k=1..K
    for k in range(1, len(clusters_sorted) + 1):
        chosen = set(clusters_sorted[:k])
        y_pred = np.isin(labels_v, list(chosen)).astype(int)
        m = metrics_from_preds(y_true_v, y_pred)
        if (best_multi is None) or (m[-1] > best_multi[-1]):
            best_multi = m
            best_k = k

    # build per-cell output for best-k
    chosen_best = set(clusters_sorted[:best_k])
    y_pred_best = np.isin(labels_v, list(chosen_best)).astype(int)

    per_cell = pd.DataFrame({
        "cell_name": names_v,
        "cluster": labels_v,
        "true_tumor": y_true_v,
        "pred_tumor_bestk": y_pred_best,
    })
    tumor_normal_path = str(out_prefix) + ".tumor_normal.tsv"
    per_cell.to_csv(tumor_normal_path, sep="\t", index=False)

    # summary
    summary_path = str(out_prefix) + ".summary.txt"
    with open(summary_path, "w") as f:
        f.write("=== Tumor/Normal evaluation summary ===\n")
        f.write(f"Total rows: {total}\n")
        f.write(f"Valid rows: {int(valid_mask.sum())}\n")
        f.write(f"Missing/unknown: {missing}\n")
        f.write(f"Cell type column: {ct_col}\n\n")

        cl, tp, fp, fn, tn, acc, prec, rec, f1 = best_single
        f.write("== Best SINGLE cluster as tumor ==\n")
        f.write(f"tumor_cluster={cl}\n")
        f.write(f"TP FP FN TN = {tp} {fp} {fn} {tn}\n")
        f.write(f"acc={acc:.4f} precision={prec:.4f} recall={rec:.4f} f1={f1:.4f}\n\n")

        tp, fp, fn, tn, acc, prec, rec, f1 = best_multi
        f.write("== Best MULTI-cluster (top-k by tumor_frac) ==\n")
        f.write(f"k={best_k} clusters={sorted(list(chosen_best))}\n")
        f.write(f"TP FP FN TN = {tp} {fp} {fn} {tn}\n")
        f.write(f"acc={acc:.4f} precision={prec:.4f} recall={rec:.4f} f1={f1:.4f}\n\n")

        f.write("Wrote:\n")
        f.write(f"  {report_path}\n")
        f.write(f"  {tumor_normal_path}\n")
        f.write(f"  {summary_path}\n")

    print(f"[OK] wrote {report_path}")
    print(f"[OK] wrote {tumor_normal_path}")
    print(f"[OK] wrote {summary_path}")

    # print key result to stdout
    cl, tp, fp, fn, tn, acc, prec, rec, f1 = best_single
    print("\n== Best SINGLE cluster as tumor ==")
    print(f"tumor_cluster={cl}")
    print(f"TP FP FN TN = {tp} {fp} {fn} {tn}")
    print(f"acc={acc:.4f} precision={prec:.4f} recall={rec:.4f} f1={f1:.4f}")

    tp, fp, fn, tn, acc, prec, rec, f1 = best_multi
    print("\n== Best MULTI-cluster (top-k by tumor_frac) ==")
    print(f"k={best_k} clusters={sorted(list(chosen_best))}")
    print(f"TP FP FN TN = {tp} {fp} {fn} {tn}")
    print(f"acc={acc:.4f} precision={prec:.4f} recall={rec:.4f} f1={f1:.4f}")


if __name__ == "__main__":
    main()
