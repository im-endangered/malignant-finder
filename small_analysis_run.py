#!/usr/bin/env python3
import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells_csv", required=True)
    ap.add_argument("--pfx", required=True, help="Prefix without extension (e.g. results/.../gao2021_xxx)")
    ap.add_argument("--pred_tsv", default=None,
                    help="Optional binary labels file. If omitted, will use <pfx>.FINAL.binary.labels.tsv if exists.")
    args = ap.parse_args()

    cells = pd.read_csv(args.cells_csv)
    cells["cell_name"] = cells["cell_name"].astype(str)
    cells["cell_type"] = cells["cell_type"].astype(str).str.lower()

    names = pd.read_csv(args.pfx + ".cells.txt", header=None)[0].astype(str)
    clus  = pd.read_csv(args.pfx + ".labels.tsv", header=None)[0].astype(int)

    df = pd.DataFrame({"cell_name": names, "cluster": clus}).merge(
        cells[["cell_name","cell_type"]], on="cell_name", how="left"
    )

    print("\n=== Cluster composition ===")
    for c in sorted(df["cluster"].unique()):
        sub = df[df.cluster == c]
        frac = (sub["cell_type"] == "malignant").mean()
        top = sub["cell_type"].value_counts().head(5).to_dict()
        print(f"cluster {c:2d} n={len(sub):4d} frac_malignant={frac:.4f} top_types={top}")

    pred_path = args.pred_tsv
    if pred_path is None:
        pred_path = args.pfx + ".FINAL.binary.labels.tsv"

    try:
        pred = pd.read_csv(pred_path, header=None)[0].astype(int)
        df2 = pd.DataFrame({"cell_name": names, "pred_tumor": pred}).merge(
            cells[["cell_name","cell_type"]], on="cell_name", how="left"
        )
        print("\n=== Prediction leakage by true cell_type ===")
        rows = []
        for ct, sub in df2.groupby("cell_type", dropna=False):
            rows.append((ct, len(sub), float(sub["pred_tumor"].mean())))
        out = pd.DataFrame(rows, columns=["cell_type","n","frac_pred_tumor"]).sort_values("n", ascending=False)
        print(out.to_string(index=False))
    except FileNotFoundError:
        print(f"\n[WARN] No pred file found at: {pred_path} (skipping leakage table)")

if __name__ == "__main__":
    main()
