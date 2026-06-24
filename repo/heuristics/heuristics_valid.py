import json
import math
import random
import pickle
import os
import time
from collections import defaultdict
import argparse
import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

# ============================================================
# ARGPARSE
# ============================================================


parser = argparse.ArgumentParser()

parser.add_argument("--data_path", type=str, default="./dataset/ogbl_collab/processed/combined.json")
parser.add_argument("--results_dir", type=str, default="./results")
parser.add_argument("--train_end_year", type=int, default=2017)
parser.add_argument("--valid_year", type=int, default=2018)
parser.add_argument("--test_year", type=int, default=2019)
parser.add_argument("--negative_samples", type=int, default=10000)
parser.add_argument("--hit_k", type=int, default=50)
parser.add_argument("--k_hop", type=int, default=2)
parser.add_argument("--use_k_hop_restriction", action="store_true")
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--debug_edges", type=int, default=None, help="Limit number of positive edges for debugging")

args = parser.parse_args()


# ============================================================
# GLOBAL CONFIG
# ============================================================

DATA_PATH = args.data_path
RESULTS_DIR = args.results_dir

TRAIN_END_YEAR = args.train_end_year
VALID_YEAR = args.valid_year
TEST_YEAR = args.test_year

NEGATIVE_SAMPLES = args.negative_samples
HIT_K = args.hit_k

K_HOP = args.k_hop
USE_K_HOP_RESTRICTION = False#args.use_k_hop_restriction

RANDOM_SEED = args.random_seed
DEBUG_EDGES = args.debug_edges

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# DATA LOADING
# ============================================================


def load_dataset(path):
    with open(path, "r") as f:
        return json.load(f)

    return results

# ============================================================
# TEMPORAL SPLIT
# ============================================================


def temporal_split(data):
    train_edges = []
    valid_edges = []
    test_edges = []

    for year_str, edges in data.items():
        year = int(year_str)

        if year <= TRAIN_END_YEAR:
            train_edges.extend(edges)

        elif year == VALID_YEAR:
            valid_edges.extend(edges)

        elif year == TEST_YEAR:
            test_edges.extend(edges)

    return train_edges, valid_edges, test_edges


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================


def build_graph(edges):
    G = nx.DiGraph()
    G.add_edges_from(edges)
    return G


# ============================================================
# NEIGHBOR UTILITIES
# ============================================================


def neighbors(G, u):
    if not G.has_node(u):
        return set()

    return set(G.successors(u)) | set(G.predecessors(u))


# ============================================================
# K-HOP CACHING
# ============================================================

def precompute_k_hop_cache(G, k):
    cache = {}

    for node in G.nodes():
        visited = {node}
        frontier = {node}

        for _ in range(k):
            next_frontier = set()

            for current in frontier:
                next_frontier |= neighbors(G, current)

            next_frontier -= visited
            visited |= next_frontier
            frontier = next_frontier

        visited.discard(node)
        cache[node] = visited

    return cache


# ============================================================
# FILTER INVALID EDGES
# ============================================================


def filter_valid_edges(G, edges):
    return [
        (u, v)
        for (u, v) in edges
        if G.has_node(u) and G.has_node(v)
    ]


# ============================================================
# NEGATIVE SAMPLING
# ============================================================


def sample_negatives_for_source(
    G,
    source_node,
    positive_target,
    num_negatives,
    k_hop_cache=None,
):
    nodes = list(G.nodes())

    negatives = set()

    while len(negatives) < num_negatives:
        v = random.choice(nodes)

        if v == source_node:
            continue

        if G.has_edge(source_node, v):
            continue

        if v == positive_target:
            continue

        if USE_K_HOP_RESTRICTION and k_hop_cache is not None:
            if source_node not in k_hop_cache:
                continue

            if v not in k_hop_cache[source_node]:
                continue

        negatives.add((source_node, v))

    return list(negatives)


# ============================================================
# HEURISTIC METHODS
# ============================================================


def score_common_neighbors(neighbor_cache, u, v):
    return len(set(neighbor_cache[u]) & set(neighbor_cache[v]))


def score_adamic_adar(neighbor_cache, u, v):

    common = (set(neighbor_cache[u]) & set(neighbor_cache[v]))

    score = 0

    for w in common:

        deg = len(set(neighbor_cache[w]))

        if deg > 1:
            score += 1 / math.log(deg)

    return score


def score_jaccard(neighbor_cache, u, v):

    nu = set(neighbor_cache[u])
    nv = set(neighbor_cache[v])

    union = len(nu | nv)

    if union == 0:
        return 0

    return len(nu & nv) / union


def score_preferential_attachment(neighbor_cache, u, v):
    return (len(set(neighbor_cache[u])) * len(set(neighbor_cache[v])))


METHODS = {
    "common_neighbors": score_common_neighbors,
    "adamic_adar": score_adamic_adar,
    "jaccard": score_jaccard,
    "preferential_attachment": score_preferential_attachment
}


# ============================================================
# METRICS
# ============================================================


def compute_hits_at_k(ranks, k):
    hits = sum(rank <= k for rank in ranks)
    return hits / len(ranks)


def compute_mrr(ranks):
    reciprocal_ranks = [1 / rank for rank in ranks]
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def compute_recall_at_k(ranks, k):

    recall_values = [
        1 if rank <= k else 0
        for rank in ranks
    ]

    return sum(recall_values) / len(recall_values)


def compute_ndcg_at_k(ranks, k):

    ndcg_values = []

    for rank in ranks:

        if rank <= k:

            ndcg = 1 / np.log2(rank + 1)

        else:

            ndcg = 0

        ndcg_values.append(ndcg)

    return sum(ndcg_values) / len(ndcg_values)


# ============================================================
# EVALUATION
# ============================================================


def evaluate_method(
    method_name,
    score_function,
    G,
    positive_edges,
    k_hop_cache,
    neighbor_cache,
    neighbor_time
):
    print(f"\nEvaluating: {method_name}")

    all_scores = []
    all_labels = []

    ranks = []

    start_inference = time.time()

    for idx, (u, v_pos) in enumerate(positive_edges):
        if idx % 1000 == 0:
            print(f"Processed {idx}/{len(positive_edges)}")
            
        # ----------------------------------------------------
        # Positive edge score
        # ----------------------------------------------------

        positive_score = score_function(neighbor_cache, u, v_pos)

        # ----------------------------------------------------
        # Negative sampling
        # ----------------------------------------------------

        negative_edges = sample_negatives_for_source(
            G,
            u,
            v_pos,
            NEGATIVE_SAMPLES,
            k_hop_cache,
        )

        negative_scores = [
            score_function(G, u, v_neg)
            for (_, v_neg) in negative_edges
        ]

        # ----------------------------------------------------
        # Rank computation
        # ----------------------------------------------------

        # Higher score = better rank
        greater = sum(score > positive_score for score in negative_scores)

        equal = sum(score == positive_score for score in negative_scores)

        rank = 1 + greater + (equal / 2)

        ranks.append(rank)

        # ----------------------------------------------------
        # Global metrics storage
        # ----------------------------------------------------

        all_scores.append(positive_score)
        all_labels.append(1)

        all_scores.extend(negative_scores)
        all_labels.extend([0] * len(negative_scores))

    inference_time = time.time() - start_inference

    # --------------------------------------------------------
    # Compute metrics
    # --------------------------------------------------------

    auc = roc_auc_score(all_labels, all_scores)
    ap = average_precision_score(all_labels, all_scores)

    mrr = compute_mrr(ranks)
    hits_at_50 = compute_hits_at_k(ranks, HIT_K)

    recall_at_k = compute_recall_at_k(
        ranks,
        HIT_K,
    )

    ndcg_at_k = compute_ndcg_at_k(
        ranks,
        HIT_K,
    )

    results = {
        "method": method_name,
        "AUC": auc,
        "AP": ap,
        "MRR": mrr,
        "Hits@50": hits_at_50,
        f"Recall@{HIT_K}": recall_at_k,
        f"NDCG@{HIT_K}": ndcg_at_k,
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "inference_time_seconds": inference_time,
        "num_positive_edges": len(positive_edges),
        "negative_samples_per_edge": NEGATIVE_SAMPLES,
        "neighbor_cache_seconds": neighbor_time,
        "k_hop": K_HOP
    }

    return results

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("Loading dataset...")

    data = load_dataset(DATA_PATH)

    # --------------------------------------------------------
    # Temporal split
    # --------------------------------------------------------

    train_edges, valid_edges, test_edges = temporal_split(data)

    print(f"Train edges: {len(train_edges)}")
    print(f"Validation edges: {len(valid_edges)}")
    print(f"Test edges: {len(test_edges)}")

    # --------------------------------------------------------
    # Build graph
    # --------------------------------------------------------

    print("\nBuilding graph...")

    start_train = time.time()

    G = build_graph(train_edges)

    train_time = time.time() - start_train

    print(
        f"Graph construction time: "
        f"{train_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # Filter unseen-node edges
    # --------------------------------------------------------

    original_test_size = len(test_edges)

    test_edges = filter_valid_edges(
        G,
        valid_edges,
    )

    if len(test_edges) == 0:
        raise ValueError("No valid test edges remain after filtering. "
                         "Try including validation edges in the graph.")

    print(f"Filtered valid test edges: "
          f"{len(test_edges)}/{original_test_size}")

    # --------------------------------------------------------
    # Optional debug limit
    # --------------------------------------------------------

    if DEBUG_EDGES is not None:

        debug_size = min(
            DEBUG_EDGES,
            len(test_edges),
        )

        test_edges = random.sample(
            test_edges,
            debug_size,
        )

        print(
            f"Debug mode enabled: "
            f"{len(test_edges)} edges"
        )

    # --------------------------------------------------------
    # Precompute k-hop neighborhoods
    # --------------------------------------------------------

    k_hop_cache = None

    if USE_K_HOP_RESTRICTION:

        print("\nPrecomputing k-hop neighborhoods...")

        start_khop = time.time()

        k_hop_cache = precompute_k_hop_cache(
            G,
            K_HOP,
        )

        khop_time = time.time() - start_khop

        print(
            f"k-hop precomputation time: "
            f"{khop_time:.2f} seconds"
        )

    else:

        khop_time = 0

    # --------------------------------------------------------
    # Evaluate methods
    # --------------------------------------------------------

    all_results = {}

    for method_name, score_function in METHODS.items():

        cache_path = os.path.join(
            RESULTS_DIR,
            f"{method_name}_valid_results2.pkl",
        )

        # ----------------------------------------------------
        # Load cached results
        # ----------------------------------------------------

        if os.path.exists(cache_path):

            print(
                f"\nLoading cached results "
                f"for {method_name}..."
            )

            with open(cache_path, "rb") as f:
                results = pickle.load(f)

        # ----------------------------------------------------
        # Compute results
        # ----------------------------------------------------

        else:

            print("Precomputing neighbor cache...")

            start_neighbors = time.time()

            neighbor_cache = {
                node: set(G.successors(node)) | set(G.predecessors(node))
                for node in G.nodes()
            }

            neighbor_time = time.time() - start_neighbors

            print(
                f"Neighbor cache time: "
                f"{neighbor_time:.2f} seconds"
            )

            results = evaluate_method(
                method_name,
                score_function,
                G,
                test_edges,
                k_hop_cache,
                neighbor_cache,
                neighbor_time
            )

            results["train_time_seconds"] = train_time

            results["khop_precompute_seconds"] = khop_time

            with open(cache_path, "wb") as f:
                pickle.dump(results, f)

        all_results[method_name] = results

    # --------------------------------------------------------
    # Print final results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    for method_name, results in all_results.items():

        print(f"\nMethod: {method_name}")

        for key, value in results.items():

            if isinstance(value, float):

                print(f"{key}: {value:.6f}")

            else:

                print(f"{key}: {value}")

    # --------------------------------------------------------
    # Save combined JSON
    # --------------------------------------------------------

    json_results_path = os.path.join(
        RESULTS_DIR,
        "heus_valid_ablation.json",
    )

    with open(json_results_path, "w") as f:
        json.dump(
            all_results,
            f,
            indent=4,
        )

    print(
        f"\nSaved combined results to: "
        f"{json_results_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
