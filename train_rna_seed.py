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
        try:
            if isinstance(p, np.ndarray) and p.shape == ():
                p = p.item()
            if isinstance(p, (bytes, str)):
                params = json.loads(p.decode() if isinstance(p, bytes) else p)
            elif isinstance(p, dict):
                params = p
            else:
                params = json.loads(str(p))
        except Exception:
            params = {"raw_params": str(p)}

    return X, edge_index, edge_weight, cell_names, params


def edges_to_csr(edge_index: np.ndarray, edge_weight: np.ndarray, n_nodes: int) -> sp.csr_matrix:
    if edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must be shape (2,E), got {edge_index.shape}")
    row = edge_index[0].astype(np.int64)
    col = edge_index[1].astype(np.int64)
    val = edge_weight.astype(np.float32)
    A = sp.csr_matrix((val, (row, col)), shape=(n_nodes, n_nodes), dtype=np.float32)
    return A


def build_one_pos_neighbor(edge_index: np.ndarray, n_nodes: int, seed: int = 42) -> np.ndarray:
    """
    For each node i, pick ONE positive neighbor pos[i] from outgoing edges.
    If a node has no neighbor (shouldn't happen with kNN), fall back to itself.
    """
    rng = np.random.RandomState(seed)
    src = edge_index[0].astype(np.int64)
    dst = edge_index[1].astype(np.int64)

    neigh = [[] for _ in range(n_nodes)]
    for s, t in zip(src, dst):
        if s != t:
            neigh[s].append(t)

    pos = np.arange(n_nodes, dtype=np.int64)
    for i in range(n_nodes):
        if len(neigh[i]) > 0:
            pos[i] = neigh[i][rng.randint(len(neigh[i]))]
        else:
            pos[i] = i
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)

    ap.add_argument("--n_clusters", type=int, default=10)
    ap.add_argument("--hidden1", type=int, default=64)
    ap.add_argument("--hidden2", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clusterer", default="gmm", choices=["gmm", "kmeans"])
    ap.add_argument("--out_prefix", default="out_rna")

    # NEW: anti-collapse / contrastive
    ap.add_argument("--lambda_contrast", type=float, default=1.0,
                    help="Weight for graph contrastive (InfoNCE) loss. Set 0 to disable.")
    ap.add_argument("--n_neg", type=int, default=256,
                    help="Number of negatives per node for InfoNCE.")
    ap.add_argument("--temperature", type=float, default=0.2,
                    help="InfoNCE temperature.")
    ap.add_argument("--zscore_features", action="store_true",
                    help="Z-score features per gene/feature (after loading). Helps if features are sparse/heavy-tailed.")
    ap.add_argument("--lambda_l2", type=float, default=0.0,
                    help="Optional L2 weight decay on embeddings (small value like 1e-4).")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    seed = args.seed

    np.random.seed(seed)
    tf.random.set_seed(seed)

    X_np, edge_index, edge_weight, cell_names, params = load_graph_npz(args.npz)
    X_np = np.asarray(X_np, dtype=np.float32)
    n_cells, n_feats = X_np.shape

    if args.zscore_features:
        mu = X_np.mean(axis=0, keepdims=True)
        sd = X_np.std(axis=0, keepdims=True) + 1e-6
        X_np = (X_np - mu) / sd

    print(f"[INFO] Loaded NPZ: {args.npz}")
    print(f"[INFO] X: {X_np.shape} min/max={X_np.min():.4f}/{X_np.max():.4f}")
    print(f"[INFO] edges: {edge_index.shape[1]}  weight min/max={edge_weight.min():.4f}/{edge_weight.max():.4f}")
    if params is not None:
        print(f"[INFO] npz params: {params}")

    # adjacency
    A = edges_to_csr(edge_index, edge_weight, n_nodes=n_cells)
    A_norm = utils.normalize_graph(A.copy(), normalized=True, add_self_loops=True)

    features = tf.convert_to_tensor(X_np, dtype=tf.float32)
    graph = scipy_to_tf_sparse(A)
    graph_norm = scipy_to_tf_sparse(A_norm)

    # fixed positive neighbor per node
    pos_idx_np = build_one_pos_neighbor(edge_index, n_nodes=n_cells, seed=seed)
    pos_idx = tf.constant(pos_idx_np, dtype=tf.int32)

    model = GAT_AE(in_dim=n_feats, hidden1=args.hidden1, hidden2=args.hidden2, dropout_rate=args.dropout)
    opt = tf.keras.optimizers.Adam(args.lr)

    @tf.function
    def info_nce_loss(Z: tf.Tensor) -> tf.Tensor:
        # normalize embeddings
        Z = tf.math.l2_normalize(Z, axis=1)

        Z_pos = tf.gather(Z, pos_idx)  # (N,D)
        logit_pos = tf.reduce_sum(Z * Z_pos, axis=1, keepdims=True) / args.temperature  # (N,1)

        # negatives: sample indices uniformly
        neg_idx = tf.random.uniform(shape=(n_cells, args.n_neg), minval=0, maxval=n_cells, dtype=tf.int32)
        Z_neg = tf.gather(Z, neg_idx)  # (N,K,D)
        logit_neg = tf.einsum("nd,nkd->nk", Z, Z_neg) / args.temperature  # (N,K)

        logits = tf.concat([logit_pos, logit_neg], axis=1)  # (N, 1+K)
        labels = tf.zeros((n_cells,), dtype=tf.int32)       # correct class is 0 (the positive)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        return loss

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            latent, _, recon = model([features, graph_norm, graph], training=True)

            # reconstruction
            mse = tf.reduce_mean(tf.square(features - recon))

            # contrastive (anti-collapse)
            c_loss = tf.constant(0.0, tf.float32)
            if args.lambda_contrast > 0.0:
                c_loss = info_nce_loss(latent)

            # small L2 on embeddings (optional)
            l2 = tf.constant(0.0, tf.float32)
            if args.lambda_l2 > 0.0:
                l2 = tf.reduce_mean(tf.reduce_sum(tf.square(latent), axis=1))

            loss = mse + args.lambda_contrast * c_loss + args.lambda_l2 * l2

        grads = tape.gradient(loss, model.trainable_variables)
        grads = [tf.clip_by_norm(g, 5.0) if g is not None else None for g in grads]
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss, mse, c_loss, latent

    best_sil = -1.0
    best_labels = None
    best_latent = None

    for ep in range(args.epochs):
        loss, mse, c_loss, latent = train_step()

        if ep % 10 == 0 or ep == args.epochs - 1:
            Z = latent.numpy()
            # quick collapse diagnostic
            zvar = float(np.median(Z.var(axis=0)))

            if args.clusterer == "gmm":
                cl = GaussianMixture(n_components=args.n_clusters, covariance_type="full", random_state=seed).fit(Z)
                labels = cl.predict(Z)
            else:
                cl = KMeans(n_clusters=args.n_clusters, random_state=seed, n_init=10).fit(Z)
                labels = cl.labels_

            sil = silhouette_score(Z, labels) if len(set(labels)) > 1 else -1.0
            if sil > best_sil:
                best_sil = sil
                best_labels = labels.copy()
                best_latent = Z.copy()

            print(
                f"epoch={ep:4d} loss={float(loss):.6f} mse={float(mse):.6f} "
                f"contrast={float(c_loss):.6f} zvar_med={zvar:.3e} "
                f"sil={sil:.4f} best={best_sil:.4f}"
            )

    np.savetxt(f"{args.out_prefix}.latent.tsv", best_latent, delimiter="\t")
    np.savetxt(f"{args.out_prefix}.labels.tsv", best_labels.astype(int), fmt="%d")

    if cell_names is not None:
        with open(f"{args.out_prefix}.cells.txt", "w") as f:
            for c in cell_names:
                f.write(str(c) + "\n")

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

    print("[OK] Saved:")
    print(f"  {args.out_prefix}.latent.tsv")
    print(f"  {args.out_prefix}.labels.tsv")
    if cell_names is not None:
        print(f"  {args.out_prefix}.cells.txt")
    print(f"  {args.out_prefix}.meta.json")


if __name__ == "__main__":
    main()
