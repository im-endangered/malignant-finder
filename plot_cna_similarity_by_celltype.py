#!/usr/bin/env python3
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt

def load_ground_truth_cells_csv(cells_csv):
    """
    Returns dict: barcode -> is_tumor (bool)
    Default rule: cell_type == 'Malignant' => tumor, else normal.
    """
    gt = {}
    with open(cells_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        if "cell_name" not in reader.fieldnames or "cell_type" not in reader.fieldnames:
            raise KeyError(f"Cells.csv must have columns cell_name and cell_type. Found: {reader.fieldnames}")
        for row in reader:
            bc = row["cell_name"]
            ct = row["cell_type"]
            gt[bc] = (ct.strip().lower() == "malignant")
    return gt

def skew_kurt(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    m = x.mean()
    s = x.std() + 1e-12
    z = (x - m) / s
    skew = (z**3).mean()
    kurt = (z**4).mean() - 3.0
    return float(skew), float(kurt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="NPZ with edge_index, edge_weight, cell_names")
    ap.add_argument("--cells_csv", required=True, help="Gao Cells.csv with cell_name + cell_type")
    ap.add_argument("--power", type=float, default=None,
                    help="If stored weight = raw_cos^power, pass power (e.g. 4). We will also compute approx raw_cos.")
    ap.add_argument("--out_prefix", default="cna_sim", help="Prefix for output PNG files")
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_examples", type=int, default=5, help="Total example cells to draw (mixed tumor+normal).")
    ap.add_argument("--example_k_per_type", type=int, default=None,
                    help="If set, overrides n_examples by selecting exactly K tumor and K normal examples.")
    ap.add_argument("--target_peak", type=float, default=0.3,
                    help="Peak value to search for (in UNPOWERED space if --power is given, else in stored weight space).")
    ap.add_argument("--peak_window", type=float, default=0.03,
                    help="+/- window around target_peak for 'peak-heavy' cell selection.")
    ap.add_argument("--force_peak_tumor_cell", action="store_true",
                    help="If set, pick 1 tumor query cell that has many neighbors near target_peak.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    d = np.load(args.npz, allow_pickle=True)
    for k in ["edge_index", "edge_weight", "cell_names"]:
        if k not in d.files:
            raise KeyError(f"Missing {k} in {args.npz}. Keys={d.files}")

    edge_index = d["edge_index"].astype(np.int64)   # (2, E)
    edge_weight = d["edge_weight"].astype(np.float64)  # (E,)
    cell_names = d["cell_names"]  # (N,), object
    cell_names = np.array([str(x) for x in cell_names], dtype=object)
    N = cell_names.size
    E = edge_weight.size

    src = edge_index[0]
    dst = edge_index[1]
    if src.size != E or dst.size != E:
        raise ValueError(f"edge_index shape {edge_index.shape} not consistent with edge_weight shape {edge_weight.shape}")

    # raw/unpowered similarity (optional)
    if args.power is not None and args.power > 0:
        raw_sim = np.clip(edge_weight, 0, None) ** (1.0 / args.power)
    else:
        raw_sim = None

    # Ground truth mapping
    gt = load_ground_truth_cells_csv(args.cells_csv)
    is_tumor = np.zeros(N, dtype=bool)
    missing = 0
    for i, bc in enumerate(cell_names):
        if bc in gt:
            is_tumor[i] = gt[bc]
        else:
            missing += 1
            is_tumor[i] = False  # default to normal if missing, but report loudly

    print(f"[INFO] NPZ: {args.npz}")
    print(f"[INFO] N cells={N}, E edges={E}, k~{E/N:.2f} outgoing per node (directed view)")
    print(f"[INFO] Ground truth: matched={N-missing}, missing={missing}")
    print(f"[INFO] Tumor cells={is_tumor.sum()}  Normal cells={(~is_tumor).sum()}")

    # Build adjacency lists by source: indices of edges for each src node
    edges_by_src = [[] for _ in range(N)]
    for ei in range(E):
        s = src[ei]
        if 0 <= s < N:
            edges_by_src[s].append(ei)

    # Helper to get per-cell neighbor weights split by neighbor type
    def cell_split_weights(i, use_raw=False):
        eidx = edges_by_src[i]
        if len(eidx) == 0:
            return np.array([]), np.array([])
        eidx = np.array(eidx, dtype=np.int64)
        neigh = dst[eidx]
        if use_raw and raw_sim is not None:
            w = raw_sim[eidx]
        else:
            w = edge_weight[eidx]
        neigh_is_tumor = is_tumor[neigh]
        w_tum = w[neigh_is_tumor]
        w_norm = w[~neigh_is_tumor]
        return w_tum, w_norm

    # Aggregate histograms: pool edges where query (source) is tumor or normal
    # Note: this matches professor statement "cell that you select to draw should be only tumor or only normal"
    tumor_edge_mask = is_tumor[src]
    normal_edge_mask = ~is_tumor[src]

    def save_aggregate_fig(mask, title, out_png):
        w = edge_weight[mask]
        sk, ku = skew_kurt(w)
        plt.figure(figsize=(7.2, 4.8))
        plt.hist(w, bins=args.bins)
        plt.yscale("log")
        plt.xlabel("edge_weight (stored)")
        plt.ylabel("count (log scale)")
        plt.title(f"{title}")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        print("[OK] wrote", out_png)

        if raw_sim is not None:
            w2 = raw_sim[mask]
            sk2, ku2 = skew_kurt(w2)
            out2 = out_png.replace(".png", ".unpowered.png")
            plt.figure(figsize=(7.2, 4.8))
            plt.hist(w2, bins=args.bins)
            plt.yscale("log")
            plt.xlabel(f"edge_weight ")
            plt.ylabel("count")
            plt.title(f"{title}")
            plt.tight_layout()
            plt.savefig(out2, dpi=200)
            plt.close()
            print("[OK] wrote", out2)

    save_aggregate_fig(normal_edge_mask, "CNA similarity : NORMAL", f"{args.out_prefix}.agg_query_normal.png")
    save_aggregate_fig(tumor_edge_mask, "CNA similarity: TUMOR", f"{args.out_prefix}.agg_query_tumor.png")

    # Choose example cells
    tumor_ids = np.where(is_tumor)[0]
    normal_ids = np.where(~is_tumor)[0]

    if args.example_k_per_type is not None:
        k = int(args.example_k_per_type)
        pick_t = rng.choice(tumor_ids, size=min(k, tumor_ids.size), replace=False) if tumor_ids.size else np.array([], dtype=int)
        pick_n = rng.choice(normal_ids, size=min(k, normal_ids.size), replace=False) if normal_ids.size else np.array([], dtype=int)
        example_ids = np.concatenate([pick_t, pick_n])
    else:
        # roughly half/half
        nt = args.n_examples // 2
        nn = args.n_examples - nt
        pick_t = rng.choice(tumor_ids, size=min(nt, tumor_ids.size), replace=False) if tumor_ids.size else np.array([], dtype=int)
        pick_n = rng.choice(normal_ids, size=min(nn, normal_ids.size), replace=False) if normal_ids.size else np.array([], dtype=int)
        example_ids = np.concatenate([pick_t, pick_n])

    # Optionally force one "peak-heavy" tumor cell
    if args.force_peak_tumor_cell and tumor_ids.size:
        # define similarity space to search peak in
        use_raw_for_peak = (raw_sim is not None)
        target = args.target_peak
        win = args.peak_window

        best_i = None
        best_cnt = -1
        for i in tumor_ids:
            w_t, w_n = cell_split_weights(i, use_raw=use_raw_for_peak)
            w_all = np.concatenate([w_t, w_n])
            if w_all.size == 0:
                continue
            cnt = np.sum((w_all >= target - win) & (w_all <= target + win))
            if cnt > best_cnt:
                best_cnt = cnt
                best_i = i

        if best_i is not None:
            # ensure it's included
            if best_i not in set(example_ids.tolist()):
                example_ids = np.concatenate([[best_i], example_ids])[:max(1, example_ids.size)]
            print(f"[INFO] Peak-heavy tumor cell chosen: {cell_names[best_i]}  count_in_window={best_cnt}  "
                  f"space={'raw' if use_raw_for_peak else 'stored'} target={target}±{win}")

    # Plot example panels (stored and, if possible, raw)
    def plot_examples(use_raw, out_png):
        M = example_ids.size
        if M == 0:
            print("[WARN] No example cells available to plot.")
            return

        ncols = 3
        nrows = int(np.ceil(M / ncols))
        plt.figure(figsize=(ncols * 5.2, nrows * 4.2))

        for j, i in enumerate(example_ids):
            ax = plt.subplot(nrows, ncols, j + 1)
            w_t, w_n = cell_split_weights(i, use_raw=use_raw)
            w_all = np.concatenate([w_t, w_n])
            if w_all.size == 0:
                ax.set_title(f"{cell_names[i]} ({'T' if is_tumor[i] else 'N'})\nno outgoing edges")
                ax.axis("off")
                continue

            # Two overlaid histograms
            ax.hist(w_t, bins=args.bins, alpha=0.65, label=f"to tumor peers (n={w_t.size})")
            ax.hist(w_n, bins=args.bins, alpha=0.65, label=f"to normal peers (n={w_n.size})")
            ax.set_yscale("log")

            label_space = "approx raw cosine" if (use_raw and raw_sim is not None) else "stored weight"
            ax.set_xlabel(label_space)
            ax.set_ylabel("count (log)")

            frac_t = (w_t.size / w_all.size) if w_all.size else 0.0
            sk, ku = skew_kurt(w_all)
            ax.set_title(f"{cell_names[i]}  [{'TUMOR' if is_tumor[i] else 'NORMAL'}]\n"
                         f"k={w_all.size}  frac_tumor_neighbors={frac_t:.2f}\n"
                         f"median={np.median(w_all):.3f}  skew={sk:.2f}")

            # mark target peak
            ax.axvline(args.target_peak, linestyle="--", linewidth=1)

            ax.legend(fontsize=8, frameon=False)

        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        print("[OK] wrote", out_png)

        print("[INFO] Example cells plotted:")
        for i in example_ids:
            print(f"  - {cell_names[i]}  type={'TUMOR' if is_tumor[i] else 'NORMAL'}")

    plot_examples(use_raw=False, out_png=f"{args.out_prefix}.examples.stored.png")
    if raw_sim is not None:
        plot_examples(use_raw=True, out_png=f"{args.out_prefix}.examples.unpowered.png")

if __name__ == "__main__":
    main()
