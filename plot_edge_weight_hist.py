import argparse
import numpy as np
import matplotlib.pyplot as plt

def skew_kurt(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    m = x.mean()
    s = x.std() + 1e-12
    z = (x - m) / s
    skew = (z**3).mean()
    kurt = (z**4).mean() - 3.0
    return float(skew), float(kurt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--power", type=float, default=None,
                    help="If your builder used power transform w <- w^power, supply that power here to also plot approx unpowered weights.")
    ap.add_argument("--out", default="edge_weight_hist.png")
    ap.add_argument("--bins", type=int, default=60)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    if "edge_weight" in d.files:
        w = d["edge_weight"].astype(float)
    elif "edge_w" in d.files:
        w = d["edge_w"].astype(float)
    else:
        raise KeyError(f"No edge_weight/edge_w in {args.npz}. Keys={d.files}")

    w = w[np.isfinite(w)]
    print(f"[INFO] {args.npz}")
    print(f"[INFO] edge_weight n={w.size}")
    print(f"[INFO] min/median/max = {w.min():.6g} / {np.median(w):.6g} / {w.max():.6g}")
    sk, ku = skew_kurt(w)
    print(f"[INFO] skew={sk:.4f} kurtosis(excess)={ku:.4f}")

    plt.figure()
    plt.hist(w, bins=args.bins)
    plt.yscale("log")
    plt.xlabel("edge_weight (stored in NPZ)")
    plt.ylabel("count")
    plt.title("Edge weight distribution (KNN edges only)")

    # optional "unpower" view
    if args.power is not None and args.power > 0:
        # if weights were w = raw^power, then raw ≈ w^(1/power)
        w_un = np.clip(w, 0, None) ** (1.0 / args.power)
        sk2, ku2 = skew_kurt(w_un)
        print(f"[INFO] approx unpowered: min/median/max = {w_un.min():.6g} / {np.median(w_un):.6g} / {w_un.max():.6g}")
        print(f"[INFO] approx unpowered: skew={sk2:.4f} kurtosis(excess)={ku2:.4f}")

        plt.figure()
        plt.hist(w_un, bins=args.bins)
        plt.yscale("log")
        plt.xlabel(f"edge weight")
        plt.ylabel("count (log scale)")
        # plt.title("Approx unpowered edge weight distribution")

    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print("[OK] wrote", args.out)

if __name__ == "__main__":
    main()
