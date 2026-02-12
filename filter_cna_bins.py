#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cna_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--thr", type=float, default=0.95, help="drop bins whose mode fraction >= thr")
    ap.add_argument("--min_unique", type=int, default=2, help="drop bins with < min_unique unique values")
    args = ap.parse_args()

    df = pd.read_csv(args.cna_csv)
    assert "cell_name" in df.columns, "Expected 'cell_name' column in CNA CSV"

    cells = df["cell_name"].astype(str)
    X = df.drop(columns=["cell_name"])

    n = len(X)
    keep = []
    stats = []

    for col in X.columns:
        s = X[col]
        # unique values check (fast path)
        nunq = s.nunique(dropna=False)
        if nunq < args.min_unique:
            stats.append((col, 1.0, int(nunq), "drop_low_unique"))
            continue

        vc = s.value_counts(dropna=False)
        mode_frac = float(vc.iloc[0]) / float(n)
        if mode_frac >= args.thr:
            stats.append((col, mode_frac, int(nunq), "drop_mode_frac"))
            continue

        keep.append(col)
        stats.append((col, mode_frac, int(nunq), "keep"))

    out = pd.concat([cells, X[keep]], axis=1)
    out.to_csv(args.out_csv, index=False)

    st = pd.DataFrame(stats, columns=["bin", "mode_frac", "n_unique", "decision"])
    print("[IN ] cells:", n, "bins:", X.shape[1])
    print("[OUT] bins kept:", len(keep), "bins dropped:", X.shape[1]-len(keep))
    print(st["decision"].value_counts())
    print("mode_frac kept (p50/p90/p99):",
          np.percentile(st.loc[st.decision=="keep","mode_frac"], [50,90,99]).tolist()
          if (st.decision=="keep").any() else "NA")

if __name__ == "__main__":
    main()
