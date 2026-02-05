from __future__ import annotations
import argparse
import json
import numpy as np
import tensorflow as tf
import scipy.sparse as sp

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import utils
from models import GAT_AE


def scipy_to_tf_sparse(A: sp.csr_matrix) -> tf.sparse.SparseTensor:
    A = A.tocoo()
    idx = np.vstack([A.row, A.col]).T.astype(np.int64)
    vals = A.data.astype(np.float32)
    shape = np.array(A.shape, dtype=np.int64)
    return tf.sparse.SparseTensor(indices=idx, values=vals, dense_shape=shape)


def load_graph_npz(npz_path: str):
    """
    Loads the .npz produced by build_graph_inputs.py:
      - rna_feat: (N, N) float32
      - edge_index: (2, E) int64
      - edge_weight: (E,) float32
      - cell_names: (N,) str (optional but expected)
      - params: json string or dict-like (optional)
    """
    d = np.load(npz_path, allow_pickle=True)
    keys = set(d.files)

    # --- features ---
    if "rna_feat" in keys:
        X = d["rna_feat"]
    elif "x" in keys:
        X = d["x"]
    else:
        raise KeyError(f"NPZ missing RNA features. Expected 'rna_feat' or 'x'. Found keys={sorted(keys)}")

    # --- edges ---
    if "edge_index" not in keys:
        raise KeyError(f"NPZ missing 'edge_index'. Found keys={sorted(keys)}")
    edge_index = d["edge_index"]

    if "edge_weight" in keys:
        edge_weight = d["edge_weight"]
    elif "edge_w" in keys:
        edge_weight = d["edge_w"]
    else:
        raise KeyError(f"NPZ missing edge weights. Expected 'edge_weight' (or 'edge_w'). Found keys={sorted(keys)}")

    # --- names / params ---
    if "cell_names" in keys:
        cell_names = d["cell_names"].astype(str)
    elif "cells" in keys:
        cell_names = d["cells"].astype(str)
    else:
        cell_names = None

    params = None
    if "params" in keys:
        p = d["params"]
        # could be a 0-d object array, dict, or json string
        try:
            if isinstance(p, np.ndarray) and p.shape == ():
                p = p.item()
            if isinstance(p, (bytes, str)):
                params = json.loads(p.decode() if isinstance(p, bytes) else p)
            elif isinstance(p, dict):
                params = p
            else:
                # last resort: string conversion
                params = json.loads(str(p))
        except Exception:
            params = {"raw_params": str(p)}

    return X, edge_index, edge_weight, cell_names, params


def edges_to_csr(edge_index: np.ndarray, edge_weight: np.ndarray, n_nodes: int) -> sp.csr_matrix:
    """
    edge_index: (2, E)
    edge_weight: (E,)
    Returns: csr adjacency (n_nodes x n_nodes)
    """
    if edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must be shape (2,E), got {edge_index.shape}")
    row = edge_index[0].astype(np.int64)
    col = edge_index[1].astype(np.int64)
    val = edge_weight.astype(np.float32)

    A = sp.csr_matrix((val, (row, col)), shape=(n_nodes, n_nodes), dtype=np.float32)
    return A


def main():
    ap = argparse.ArgumentParser()

    # NEW: graph inputs
    ap.add_argument("--npz", required=True, help="NPZ from build_graph_inputs.py (CNA edges + RNA features).")

    # clustering/training options (kept close to your original)
    ap.add_argument("--n_clusters", type=int, default=4)
    ap.add_argument("--hidden1", type=int, default=64)
    ap.add_argument("--hidden2", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clusterer", default="gmm", choices=["gmm", "kmeans"])
    ap.add_argument("--out_prefix", default="out_rna")

    args = ap.parse_args()

    # Reproducibility
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Load NPZ (RNA features + CNA graph)
    X_np, edge_index, edge_weight, cell_names, params = load_graph_npz(args.npz)

    X_np = np.asarray(X_np, dtype=np.float32)
    n_cells, n_feats = X_np.shape
    print(f"[INFO] Loaded NPZ: {args.npz}")
    print(f"[INFO] X (cells x features): {X_np.shape} dtype={X_np.dtype} min/max={X_np.min():.4f}/{X_np.max():.4f}")
    print(f"[INFO] edge_index: {tuple(edge_index.shape)} edge_weight: {tuple(edge_weight.shape)}")
    if cell_names is not None:
        print(f"[INFO] cell_names: {len(cell_names)} (first={cell_names[0]})")
    if params is not None:
        print(f"[INFO] params: {params}")

    # Build adjacency CSR from CNA edges (already kNN + heat kernel)
    A = edges_to_csr(edge_index, edge_weight, n_nodes=n_cells)

    # Normalize (same function you already use)
    A_norm = utils.normalize_graph(A.copy(), normalized=True, add_self_loops=True)

    # TF inputs
    features = tf.convert_to_tensor(X_np, dtype=tf.float32)
    graph = scipy_to_tf_sparse(A)
    graph_norm = scipy_to_tf_sparse(A_norm)

    # Model (same)
    model = GAT_AE(in_dim=n_feats, hidden1=args.hidden1, hidden2=args.hidden2, dropout_rate=args.dropout)
    opt = tf.keras.optimizers.Adam(args.lr)

    # Training step: reconstruct RNA feature vectors, message passing uses CNA graph
    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            latent, _, recon = model([features, graph_norm, graph], training=True)
            mse = tf.reduce_mean(tf.square(features - recon))
            loss = mse
        grads = tape.gradient(loss, model.trainable_variables)
        grads = [tf.clip_by_norm(g, 5.0) if g is not None else None for g in grads]
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss, latent

    best_sil = -1.0
    best_labels = None
    best_latent = None

    for ep in range(args.epochs):
        loss, latent = train_step()

        if ep % 10 == 0 or ep == args.epochs - 1:
            Z = latent.numpy()

            if args.clusterer == "gmm":
                cl = GaussianMixture(
                    n_components=args.n_clusters,
                    covariance_type="full",
                    random_state=seed
                ).fit(Z)
                labels = cl.predict(Z)
            else:
                cl = KMeans(
                    n_clusters=args.n_clusters,
                    random_state=seed,
                    n_init=10
                ).fit(Z)
                labels = cl.labels_

            sil = silhouette_score(Z, labels) if len(set(labels)) > 1 else -1.0
            if sil > best_sil:
                best_sil = sil
                best_labels = labels.copy()
                best_latent = Z.copy()

            print(f"epoch={ep:4d} loss={float(loss):.6f} silhouette={sil:.4f} best={best_sil:.4f}")

    # Save outputs (keep your old formats, plus a cell_name file for safety)
    np.savetxt(f"{args.out_prefix}.latent.tsv", best_latent, delimiter="\t")
    np.savetxt(f"{args.out_prefix}.labels.tsv", best_labels.astype(int), fmt="%d")

    if cell_names is not None:
        with open(f"{args.out_prefix}.cells.txt", "w") as f:
            for c in cell_names:
                f.write(str(c) + "\n")

    # Save a small run metadata json
    meta = {
        "npz": args.npz,
        "n_cells": int(n_cells),
        "n_features": int(n_feats),
        "n_edges": int(edge_index.shape[1]),
        "best_silhouette": float(best_sil),
        "args": vars(args),
        "npz_params": params,
    }
    with open(f"{args.out_prefix}.meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Saved:")
    print(f"  {args.out_prefix}.latent.tsv")
    print(f"  {args.out_prefix}.labels.tsv")
    if cell_names is not None:
        print(f"  {args.out_prefix}.cells.txt")
    print(f"  {args.out_prefix}.meta.json")


if __name__ == "__main__":
    main()
