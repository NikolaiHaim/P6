import json
import time
import argparse
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# Utility Functions
# ============================================================

def load_embeddings(path):
    """
    Supported formats:
    - .npy
    - .npz
    - .json

    Returns:
        np.ndarray [num_nodes, embedding_dim]
    """

    if path.endswith(".npy"):

        emb = np.load(path)

    elif path.endswith(".npz"):

        data = np.load(path)

        key = list(data.keys())[0]

        emb = data[key]

    elif path.endswith(".json"):

        with open(path, "r") as f:
            data = json.load(f)

        max_node = max(int(k) for k in data.keys())

        dim = len(next(iter(data.values())))

        emb = np.zeros((max_node + 1, dim), dtype=np.float32)

        for k, v in data.items():
            emb[int(k)] = np.asarray(v, dtype=np.float32)

    else:
        raise ValueError(f"Unsupported embedding format: {path}")

    print(f"Loaded embeddings: {emb.shape}")

    return emb


def load_edges(path, year="2019"):

    with open(path, "r") as f:
        data = json.load(f)

    edges = np.asarray(data[year], dtype=np.int64)

    return edges


def build_edge_set(edges):

    edge_set = set()

    for u, v in edges:

        edge_set.add((u, v))
        edge_set.add((v, u))

    return edge_set


# ============================================================
# Negative Sampling
# ============================================================

def sample_negative_targets(
    u,
    num_nodes,
    edge_set,
    num_negatives,
    rng,
):
    """
    Samples negative target nodes for source node u.
    """

    negatives = []

    while len(negatives) < num_negatives:

        v = rng.integers(0, num_nodes)

        if v == u:
            continue

        if (u, v) in edge_set:
            continue

        negatives.append(v)

    return negatives


# ============================================================
# Scoring Functions
# ============================================================

def compute_scores(
    embeddings,
    edges,
    score_function="dot",
):
    """
    Computes edge scores.
    """

    src = embeddings[edges[:, 0]]
    dst = embeddings[edges[:, 1]]

    if score_function == "dot":

        scores = np.sum(src * dst, axis=1)

    elif score_function == "cosine":

        numerator = np.sum(src * dst, axis=1)

        denominator = (
            np.linalg.norm(src, axis=1)
            * np.linalg.norm(dst, axis=1)
            + 1e-12
        )

        scores = numerator / denominator

    else:
        raise ValueError("score_function must be 'dot' or 'cosine'")

    return scores


# ============================================================
# Main Evaluation
# ============================================================

def evaluate_embeddings(
    embeddings,
    positive_edges,
    num_nodes,
    negatives_per_positive=100,
    k_values=[10, 50, 100],
    score_function="dot",
    seed=42,
):
    """
    Evaluation metrics:

    Classification:
    - AUC
    - Average Precision

    Ranking:
    - MRR
    - Mean Rank
    - Median Rank

    Retrieval:
    - Hit@k
    - Precision@k
    - Recall@k
    - NDCG@k
    """

    rng = np.random.default_rng(seed)

    edge_set = build_edge_set(positive_edges)

    metrics = defaultdict(list)

    all_positive_scores = []
    all_negative_scores = []

    inference_start = time.perf_counter()

    # ========================================================
    # Main Evaluation Loop
    # ========================================================

    for u, v_pos in tqdm(positive_edges):

        # ----------------------------------------------------
        # Sample negatives
        # ----------------------------------------------------

        negatives = sample_negative_targets(
            u=u,
            num_nodes=num_nodes,
            edge_set=edge_set,
            num_negatives=negatives_per_positive,
            rng=rng,
        )

        # ----------------------------------------------------
        # Build candidate edges
        # ----------------------------------------------------

        candidates = [v_pos] + negatives

        edges = np.asarray(
            [[u, v] for v in candidates],
            dtype=np.int64,
        )

        # ----------------------------------------------------
        # Score candidates
        # ----------------------------------------------------

        scores = compute_scores(
            embeddings,
            edges,
            score_function=score_function,
        )

        positive_score = scores[0]

        negative_scores = scores[1:]

        # ----------------------------------------------------
        # Store for AUC/AP
        # ----------------------------------------------------

        all_positive_scores.append(positive_score)

        all_negative_scores.extend(negative_scores.tolist())

        # ----------------------------------------------------
        # Rank computation
        # ----------------------------------------------------

        greater = np.sum(negative_scores > positive_score)

        equal = np.sum(negative_scores == positive_score)

        rank = 1 + greater + (equal / 2)

        metrics["MRR"].append(1.0 / rank)

        metrics["Rank"].append(rank)

        # ----------------------------------------------------
        # Sort candidates
        # ----------------------------------------------------

        sorted_idx = np.argsort(scores)[::-1]

        ranked_candidates = np.asarray(candidates)[sorted_idx]

        relevance = np.asarray(
            [1 if v == v_pos else 0 for v in ranked_candidates],
            dtype=np.int32,
        )

        # ----------------------------------------------------
        # Metrics@k
        # ----------------------------------------------------

        for k in k_values:

            top_k = relevance[:k]

            hits = np.sum(top_k)

            # --------------------------------------------
            # Hit@k
            # --------------------------------------------

            hit = 1.0 if hits > 0 else 0.0

            # --------------------------------------------
            # Precision@k
            # --------------------------------------------

            precision = hits / k

            # --------------------------------------------
            # Recall@k
            # --------------------------------------------

            # Single-positive retrieval setting
            recall = hits / 1.0

            # --------------------------------------------
            # NDCG@k
            # --------------------------------------------

            dcg = 0.0

            for i, rel in enumerate(top_k):

                dcg += rel / np.log2(i + 2)

            idcg = 1.0

            ndcg = dcg / idcg

            metrics[f"Hit@{k}"].append(hit)

            metrics[f"Precision@{k}"].append(precision)

            metrics[f"Recall@{k}"].append(recall)

            metrics[f"NDCG@{k}"].append(ndcg)

    inference_time = time.perf_counter() - inference_start

    # ========================================================
    # Binary Classification Metrics
    # ========================================================

    y_true = np.concatenate([
        np.ones(len(all_positive_scores)),
        np.zeros(len(all_negative_scores)),
    ])

    y_scores = np.concatenate([
        np.asarray(all_positive_scores),
        np.asarray(all_negative_scores),
    ])

    auc = roc_auc_score(y_true, y_scores)

    ap = average_precision_score(y_true, y_scores)

    # ========================================================
    # Aggregate Final Metrics
    # ========================================================

    final_metrics = {}

    # --------------------------------------------
    # Classification
    # --------------------------------------------

    final_metrics["AUC"] = float(auc)

    final_metrics["Average_Precision"] = float(ap)

    # --------------------------------------------
    # Ranking
    # --------------------------------------------

    ranks = np.asarray(metrics["Rank"])

    final_metrics["MRR"] = float(np.mean(metrics["MRR"]))

    final_metrics["Mean_Rank"] = float(np.mean(ranks))

    final_metrics["Median_Rank"] = float(np.median(ranks))

    # --------------------------------------------
    # Retrieval Metrics
    # --------------------------------------------

    for key, values in metrics.items():

        if key in ["MRR", "Rank"]:
            continue

        final_metrics[key] = float(np.mean(values))

    # --------------------------------------------
    # Computational
    # --------------------------------------------

    final_metrics["Inference_Time_Sec"] = float(inference_time)

    return final_metrics


# ============================================================
# Main
# ============================================================

def main(args):

    # --------------------------------------------------------
    # Load Data
    # --------------------------------------------------------

    embeddings = load_embeddings(args.embedding_path)

    test_edges = load_edges(
        args.test_edge_path,
        year=args.year,
    )

    num_nodes = embeddings.shape[0]

    print(f"\nNodes: {num_nodes}")

    print(f"Test edges: {len(test_edges)}")

    print(f"Embedding dimension: {embeddings.shape[1]}")

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    results = evaluate_embeddings(
        embeddings=embeddings,
        positive_edges=test_edges,
        num_nodes=num_nodes,
        negatives_per_positive=args.neg_samples,
        k_values=args.k_values,
        score_function=args.score_function,
        seed=args.seed,
    )

    # --------------------------------------------------------
    # Print Results
    # --------------------------------------------------------

    print("\n================ RESULTS ================\n")

    for key, value in results.items():

        print(f"{key}: {value:.6f}")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    with open(args.output_path, "w") as f:

        json.dump(results, f, indent=4)

    print(f"\nSaved results to: {args.output_path}")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--embedding_path", type=str, required=True)
    parser.add_argument("--test_edge_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--year", type=str, default="2019")
    parser.add_argument("--score_function", type=str, default="dot", choices=["dot", "cosine"])
    parser.add_argument("--neg_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k_values", type=int, nargs="+", default=[10, 50, 100])

    args = parser.parse_args()

    main(args)

# Recommended Usage

"""
## GCN

python evaluate.py \
  --embedding_path gcn_embeddings.npy \
  --test_edge_path edges.json \
  --output_path gcn_results.json \
  --score_function dot \
  --neg_samples 100


## node2vec

python evaluate.py \
  --embedding_path node2vec_embeddings.npy \
  --test_edge_path edges.json \
  --output_path node2vec_results.json \
  --score_function cosine \
  --neg_samples 100


## Spectral Embedding / SVD

python evaluate.py \
  --embedding_path spectral_embeddings.npy \
  --test_edge_path edges.json \
  --output_path spectral_results.json \
  --score_function cosine \
  --neg_samples 100
"""