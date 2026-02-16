#!/usr/bin/env python3
"""
plot_selected_cells_cna_heatmap.py

Example:
python plot_selected_cells_cna_heatmap.py \
  --csv /gpfs/research/fangroup/pb25e/malignant/datasets/gao_et_al_2021/CNA_matrix_Gao2021_Breast/CNAs_Breast.csv \
  --out selected_5cells_cna_heatmap.png \
  --cells_tumor TAGAGCTAGGCGACAT GGATGTTCACATAACC GTTCATTAGAGACTAT \
  --cells_normal CAAGATCTCTGTACGA AGCCTAATCCCTTGCA \
  --cache gene_coords_cache.tsv \
  --clip 1.0
"""

import argparse
import os
import sys
import time
from io import StringIO
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from urllib.parse import quote

# Try multiple Ensembl mirrors
BIOMART_URLS = [
    "https://www.ensembl.org/biomart/martservice",
    "https://useast.ensembl.org/biomart/martservice",
    "https://asia.ensembl.org/biomart/martservice",
]

# BioMart TSV human headers -> internal names
BM_HEADER_MAP = {
    "Gene name": "external_gene_name",
    "Chromosome/scaffold name": "chromosome_name",
    "Gene start (bp)": "start_position",
    "Gene end (bp)": "end_position",
}

# Colors for label styling
TUMOR_COLOR = "red"
NORMAL_COLOR = "blue"


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def normalize_chr(c) -> str:
    if c is None or (isinstance(c, float) and np.isnan(c)):
        return "NA"
    c = str(c).strip()
    if c.lower().startswith("chr"):
        c = c[3:]
    c = c.replace(" ", "")
    if c in ("MT", "M", "Mt", "m"):
        return "MT"
    if c in ("X", "Y"):
        return c
    if c.isdigit():
        return c
    return "NA"


def chr_sort_key(c: str) -> Tuple[int, str]:
    c = normalize_chr(c)
    if c.isdigit():
        return (int(c), "")
    if c == "X":
        return (23, "")
    if c == "Y":
        return (24, "")
    if c == "MT":
        return (25, "")
    return (99, c)


def build_biomart_xml(gene_names: List[str]) -> str:
    value = ",".join(gene_names)
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" count="" datasetConfigVersion="0.6">
  <Dataset name="hsapiens_gene_ensembl" interface="default">
    <Filter name="external_gene_name" value="{value}"/>
    <Attribute name="external_gene_name"/>
    <Attribute name="chromosome_name"/>
    <Attribute name="start_position"/>
    <Attribute name="end_position"/>
  </Dataset>
</Query>"""
    return xml


def fetch_gene_coords_biomart(genes: List[str], chunk_size: int = 250, sleep_s: float = 0.2) -> pd.DataFrame:
    genes = [g for g in genes if isinstance(g, str) and g and g != "nan"]
    genes = list(dict.fromkeys(genes))  # unique preserve order

    headers = {
        "User-Agent": "Mozilla/5.0 (plot_selected_cells_cna_heatmap; +https://ensembl.org)",
        "Accept": "text/plain,text/tab-separated-values,text/*,*/*",
    }

    all_rows = []

    for i in range(0, len(genes), chunk_size):
        chunk = genes[i : i + chunk_size]
        xml = build_biomart_xml(chunk)
        q = quote(xml, safe="")  # fully encode

        last_err = None
        got = None

        for base in BIOMART_URLS:
            url = f"{base}?query={q}"
            try:
                r = requests.get(url, headers=headers, timeout=90)
            except Exception as ex:
                last_err = f"{base} request failed: {ex}"
                continue

            if r.status_code != 200:
                last_err = f"{base} HTTP {r.status_code}: {r.text[:120]}"
                continue

            text = r.text.strip()
            if text.lower().startswith("<html") or "<head" in text[:200].lower():
                last_err = f"{base} returned HTML landing page"
                continue

            got = text
            break

        if got is None:
            raise RuntimeError(f"BioMart failed for chunk {i}-{i+len(chunk)}. Last error: {last_err}")

        df = pd.read_csv(StringIO(got), sep="\t")
        df = df.rename(columns={c: BM_HEADER_MAP.get(c, c) for c in df.columns})

        expected = {"external_gene_name", "chromosome_name", "start_position", "end_position"}
        if not expected.issubset(df.columns):
            raise RuntimeError(f"BioMart TSV missing expected columns. Got: {list(df.columns)}")

        all_rows.append(df)
        time.sleep(sleep_s)

    out = (
        pd.concat(all_rows, ignore_index=True)
        if all_rows
        else pd.DataFrame(columns=["external_gene_name", "chromosome_name", "start_position", "end_position"])
    )

    out["chromosome_name"] = out["chromosome_name"].map(normalize_chr)
    out = out[out["chromosome_name"].isin([str(i) for i in range(1, 23)] + ["X", "Y", "MT"])].copy()
    out["start_position"] = pd.to_numeric(out["start_position"], errors="coerce")
    out["end_position"] = pd.to_numeric(out["end_position"], errors="coerce")
    out = out.dropna(subset=["start_position"])
    out = out.sort_values(["external_gene_name", "start_position"]).drop_duplicates("external_gene_name", keep="first")
    return out


def load_cache(cache_path: str) -> pd.DataFrame:
    if cache_path and os.path.exists(cache_path):
        df = pd.read_csv(cache_path, sep="\t")
        df = df.rename(columns={c: BM_HEADER_MAP.get(c, c) for c in df.columns})
        for col in ["external_gene_name", "chromosome_name", "start_position", "end_position"]:
            if col not in df.columns:
                df[col] = np.nan
        df["chromosome_name"] = df["chromosome_name"].map(normalize_chr)
        return df[["external_gene_name", "chromosome_name", "start_position", "end_position"]].copy()
    return pd.DataFrame(columns=["external_gene_name", "chromosome_name", "start_position", "end_position"])


def save_cache(df: pd.DataFrame, cache_path: str):
    if not cache_path:
        return
    df = df[["external_gene_name", "chromosome_name", "start_position", "end_position"]].copy()
    df.to_csv(cache_path, sep="\t", index=False)


def get_gene_coords(genes: List[str], cache_path: str) -> pd.DataFrame:
    cache = load_cache(cache_path)
    cached_genes = set(cache["external_gene_name"].astype(str).tolist())

    need = [g for g in genes if isinstance(g, str) and g not in cached_genes]
    if need:
        eprint(f"[INFO] Cache has {len(cached_genes)} genes; fetching {len(need)} missing from Ensembl BioMart...")
        fetched = fetch_gene_coords_biomart(need)
        merged = pd.concat([cache, fetched], ignore_index=True)
        merged = merged.sort_values(["external_gene_name", "start_position"]).drop_duplicates(
            "external_gene_name", keep="first"
        )
        save_cache(merged, cache_path)
        return merged
    else:
        eprint(f"[INFO] All {len(genes)} genes found in cache.")
        return cache


def read_cna_csv_auto(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    first_col = df.columns[0]
    df = df.set_index(first_col)
    return df


def ensure_genes_x_cells(df: pd.DataFrame, selected_cells: List[str]) -> pd.DataFrame:
    cols = set(df.columns.astype(str))
    idx = set(df.index.astype(str))

    in_cols = sum([c in cols for c in selected_cells])
    in_idx = sum([c in idx for c in selected_cells])

    if in_cols >= in_idx and in_cols > 0:
        return df
    if in_idx > 0:
        return df.T
    return df


def compute_chr_blocks(chr_vec: List[str]) -> Tuple[List[float], List[str], List[int]]:
    boundaries = []
    centers = []
    labels = []

    start = 0
    for i in range(1, len(chr_vec) + 1):
        if i == len(chr_vec) or chr_vec[i] != chr_vec[i - 1]:
            end = i
            boundaries.append(end)
            centers.append((start + end) / 2.0)
            labels.append(chr_vec[i - 1])
            start = i
    return centers, labels, boundaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CNA matrix CSV (genes x cells or cells x genes)")
    ap.add_argument("--out", required=True, help="Output image path (png/pdf)")
    ap.add_argument("--cells_tumor", nargs="+", default=[], help="Tumor cell IDs (space-separated)")
    ap.add_argument("--cells_normal", nargs="+", default=[], help="Normal cell IDs (space-separated)")
    ap.add_argument("--cache", default="gene_coords_cache.tsv", help="TSV cache for gene coordinates")
    ap.add_argument("--clip", type=float, default=1.0, help="Clip CNA values to [-clip, clip] (CopyKAT-style)")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--title", default="Selected cells CNA heatmap")
    args = ap.parse_args()

    tumor_cells = [str(x) for x in args.cells_tumor]
    normal_cells = [str(x) for x in args.cells_normal]
    selected_cells = tumor_cells + normal_cells
    if not selected_cells:
        raise SystemExit("ERROR: Provide at least one cell via --cells_tumor and/or --cells_normal.")

    # Load CNA matrix and ensure genes x cells
    df0 = read_cna_csv_auto(args.csv)
    df = ensure_genes_x_cells(df0, selected_cells)

    # Validate selected cells exist as columns
    colnames = df.columns.astype(str).tolist()
    missing = [c for c in selected_cells if c not in colnames]
    if missing:
        eprint("[ERROR] Some selected cells were not found in CNA matrix columns.")
        eprint("Missing:", missing[:20], ("..." if len(missing) > 20 else ""))
        eprint("Tip: Your CNA matrix may be transposed; check the CSV orientation.")
        raise SystemExit(2)

    # Subset and numeric
    sub = df[selected_cells].copy()
    sub = sub.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    genes = sub.index.astype(str).tolist()

    # Fetch coords + order genes
    chr_vec = ["NA"] * len(genes)
    try:
        chr_info = get_gene_coords(genes, args.cache)
        g2 = chr_info.set_index("external_gene_name")

        keep = [g for g in genes if g in g2.index]
        if len(keep) < max(50, 0.1 * len(genes)):
            eprint(f"[WARN] Only {len(keep)}/{len(genes)} genes had coordinates. Plot may lack chr structure.")
        else:
            tmp = g2.loc[keep, ["chromosome_name", "start_position"]].copy()
            tmp["chr_key"] = tmp["chromosome_name"].map(lambda x: chr_sort_key(x)[0])
            tmp = tmp.sort_values(["chr_key", "start_position"])
            ordered_genes = tmp.index.astype(str).tolist()
            chr_vec = tmp["chromosome_name"].astype(str).tolist()
            sub = sub.loc[ordered_genes]
    except Exception as ex:
        eprint(f"[WARN] Could not fetch/order gene coordinates via BioMart: {ex}")
        eprint("[WARN] Proceeding without chromosome labels (gene order as in CSV).")
        chr_vec = ["NA"] * len(genes)

    # Build heatmap matrix: rows=cells, cols=genes
    mat = sub.T.values.astype(np.float32)
    mat = np.clip(mat, -args.clip, args.clip)
    n_cells, n_genes = mat.shape

    # Figure size
    fig_w = min(22, max(10, n_genes / 350))
    fig_h = min(10, max(3.2, n_cells * 0.9))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # IMPORTANT: reserve space for: top chromosome axis + right colorbar + bottom legend
    fig.subplots_adjust(left=0.14, right=0.86, bottom=0.22, top=0.82)

    # Heatmap
    norm = TwoSlopeNorm(vmin=-args.clip, vcenter=0.0, vmax=args.clip)
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="RdBu_r", norm=norm)

    # Y tick labels: ONLY cell names, colored by class
    ylabels = tumor_cells + normal_cells
    ax.set_yticks(np.arange(n_cells))
    ax.set_yticklabels(ylabels, fontsize=11)

    # Color each ytick label
    for i, tick in enumerate(ax.get_yticklabels()):
        if i < len(tumor_cells):
            tick.set_color(TUMOR_COLOR)
            tick.set_fontweight("bold")
        else:
            tick.set_color(NORMAL_COLOR)
            tick.set_fontweight("bold")

    # Separator line between tumor and normal
    if tumor_cells and normal_cells:
        ax.axhline(len(tumor_cells) - 0.5, color="black", linewidth=1.2)

    # X axis chromosome blocks (top) + boundaries
    ax.set_xticks([])
    if any(c != "NA" for c in chr_vec):
        centers, labels, boundaries = compute_chr_blocks(chr_vec)

        # vertical chromosome boundary lines
        for b in boundaries[:-1]:
            ax.axvline(b - 0.5, color="black", linewidth=0.9, alpha=0.9)

        # top axis labels
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks(centers)
        ax_top.set_xticklabels(labels, fontsize=11)
        ax_top.tick_params(axis="x", length=0, pad=2)
        ax_top.set_xlabel("Chromosome", fontsize=12, labelpad=8)

    ax.set_title(args.title, fontsize=15, pad=18)

    # ---- Colorbar OUTSIDE (dedicated axis) so it never overlaps X/Y ----
    # [left, bottom, width, height] in figure coordinates
    cax = fig.add_axes([0.88, 0.24, 0.02, 0.54])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Inferred CNA (log2 ratio)", fontsize=12)

    # ---- Tumor/Normal legend (like your reference) ----
    handles = [
        Patch(facecolor=TUMOR_COLOR, edgecolor="none", label="Tumor"),
        Patch(facecolor=NORMAL_COLOR, edgecolor="none", label="Normal"),
    ]
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.16, 0.06),
        ncol=2,
        frameon=False,
        fontsize=12,
        handlelength=1.0,
        handleheight=1.0,
    )

    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    eprint(f"[DONE] Wrote: {args.out}")


if __name__ == "__main__":
    main()
