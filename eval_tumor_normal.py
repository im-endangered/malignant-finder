#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells_csv", required=True, help="Path to Cells.csv (has cell_type).")
    ap.add_argument("--names_txt", required=True, help="cells.txt (one cell per line, model order).")
    ap.add_argument("--labels_tsv", required=True, help="labels.tsv (one int per line, same order).")
    ap.add_argument("--out_prefix", default="eval", help="Output prefix.")
    ap.add_argument("--tumor_cluster", type=int, default=None,
                    help="Optional: force tumor cluster id. If not set, inferred by malignant fraction.")
    args = ap.parse_args()

    # Load
    cells = pd.read_csv(args.cells_csv, dtype=str)
    names = pd.read_csv(args.names_txt, header=None, names=["cell_name"], dtype=str)
    labels = pd.read_csv(args.labels_tsv, header=None, names=["cluster"])

    # Strip whitespace
    cells["cell_name"] = cells["cell_name"].str.strip()
    names["cell_name"] = names["cell_name"].str.strip()

    # Join in model order
    df = names.join(labels)

    # Add cell_type (ground truth proxy)
    df = df.merge(cells[["cell_name", "cell_type", "sample"]], on="cell_name", how="left")

    # Define "ground truth" malignant flag (proxy)
    df["true_malignant"] = (df["cell_type"] == "Malignant").astype("Int64")

    # Infer which cluster is tumor (highest malignant fraction among annotated cells)
    if args.tumor_cluster is None:
        tmp = df.dropna(subset=["true_malignant"]).copy()
        frac = tmp.groupby("cluster")["true_malignant"].mean().sort_values(ascending=False)
        tumor_cluster = int(frac.index[0])
    else:
        tumor_cluster = int(args.tumor_cluster)

    df["pred_tumor"] = (df["cluster"] == tumor_cluster).astype(int)

    # Compute confusion only on rows where we have true labels (cell_type not missing)
    eval_df = df.dropna(subset=["true_malignant"]).copy()
    y_true = eval_df["true_malignant"].astype(int).to_numpy()
    y_pred = eval_df["pred_tumor"].astype(int).to_numpy()

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    prec = tp / (tp + fp + 1e-12)
    rec  = tp / (tp + fn + 1e-12)
    f1   = 2 * prec * rec / (prec + rec + 1e-12)
    acc  = (tp + tn) / (tp + tn + fp + fn + 1e-12)

    # Save per-cell mapping (like your earlier tumor_normal_by_cna_gmm.tsv)
    # format: cell_name \t pred_tumor
    out_map = f"{args.out_prefix}.tumor_normal.tsv"
    df[["cell_name", "pred_tumor"]].to_csv(out_map, sep="\t", index=False, header=False)

    # Save a richer CSV for inspection
    out_csv = f"{args.out_prefix}.per_cell.csv"
    df.to_csv(out_csv, index=False)

    # Print summary
    print("=== Tumor/Normal evaluation (cell-by-cell) ===")
    print("tumor_cluster =", tumor_cluster)
    print("Rows total:", len(df), "Rows with cell_type:", len(eval_df), "Missing cell_type:", df["cell_type"].isna().sum())
    print("TP FP FN TN =", tp, fp, fn, tn)
    print(f"acc={acc:.4f} precision={prec:.4f} recall={rec:.4f} f1={f1:.4f}")
    print("Wrote:")
    print(" ", out_map)
    print(" ", out_csv)

if __name__ == "__main__":
    main()
