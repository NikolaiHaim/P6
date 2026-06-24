import argparse
import csv
import json
import random
import time
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh

# ============================================================
# RANDOM SEED
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# LOAD DATA
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# TEMPORAL SPLIT
# ============================================================

def temporal_split(data, val_year, test_year):

    train_edges = []
    val_edges = []
    test_edges = []

    years = sorted(data.keys(), key=int)

    for year in years:

        edges = data[year]

        if int(year) < val_year:
            train_edges.extend(edges)

        elif int(year) == val_year:
            val_edges.extend(edges)

        elif int(year) == test_year:
            test_edges.extend(edges)

    return train_edges, val_edges, test_edges


# ============================================================
# NODE MAPPING
# ============================================================

def build_node_mapping(all_edges):

    nodes = set()

    for u, v in all_edges:
        nodes.add(u)
        nodes.add(v)

    nodes = sorted(nodes)

    node_to_idx = {n: i for i, n in enumerate(nodes)}
    idx_to_node = {i: n for n, i in node_to_idx.items()}

    return node_to_idx, idx_to_node


# ============================================================
# BUILD GRAPH
# ============================================================

def build_graph(edges, node_to_idx):

    G = nx.Graph()

    G.add_nodes_from(range(len(node_to_idx)))

    for u, v in edges:
        G.add_edge(
            node_to_idx[u],
            node_to_idx[v]
        )

    return G


# ============================================================
# NEGATIVE SAMPLING
# ============================================================

def sample_hard_negatives(
    u,
    amount,
    candidate_dict,
    forbidden
):

    candidates = candidate_dict[u]

    valid = []

    for v in candidates:

        a = min(u, v)
        b = max(u, v)

        if (a, b) not in forbidden:
            valid.append(v)

    if len(valid) == 0:
        return []

    if len(valid) <= amount:
        return valid

    return random.sample(valid, amount)


def build_two_hop_candidates(G):

    candidates = {}

    for u in G.nodes():

        neighbors = set(G.neighbors(u))

        two_hop = set()

        for n in neighbors:
            two_hop.update(G.neighbors(n))

        # remove:
        # self
        # existing neighbors

        two_hop.discard(u)
        two_hop -= neighbors

        candidates[u] = list(two_hop)

    return candidates


# ============================================================
# SPECTRAL EMBEDDING
# ============================================================

def spectral_embedding(G, dim):

    A = nx.to_scipy_sparse_array(
        G,
        format="csr",
        dtype=np.float32
    )

    n = A.shape[0]

    k = min(dim, n - 1)

    start = time.time()

    vals, vecs = eigsh(
        A,
        k=k,
        which="LA"
    )

    vals = np.maximum(vals, 0)

    Z = vecs @ np.diag(np.sqrt(vals))

    train_time = time.time() - start

    memory_mb = (
        A.data.nbytes +
        A.indices.nbytes +
        A.indptr.nbytes +
        Z.nbytes
    ) / (1024 ** 2)

    return Z, train_time, memory_mb


# ============================================================
# SCORE FUNCTION
# ============================================================

def score_pair(Z, u, v):
    return float(np.dot(Z[u], Z[v]))


# ============================================================
# EVALUATION
# ============================================================


def evaluate_ranking(
    split_name,    
    edges,
    Z,
    candidate_dict,
    forbidden,
    negatives_per_positive=100,
    k=10
):

    all_labels = []
    all_scores = []

    reciprocal_ranks = []

    ranks = []

    hits = 0

    recalls = []

    ndcgs = []
    
    inf_time = float(0)

    for u, v_pos in edges:
        new = time.time()
        # ------------------------------------------------
        # positive score
        # ------------------------------------------------

        pos_score = score_pair(Z, u, v_pos)

        # ------------------------------------------------
        # hard negatives
        # ------------------------------------------------

        negatives = sample_hard_negatives(
            u,
            negatives_per_positive,
            candidate_dict,
            forbidden
        )

        if len(negatives) == 0:
            continue

        candidate_scores = []

        # positive
        candidate_scores.append((pos_score, 1))

        all_scores.append(pos_score)
        all_labels.append(1)

        # negatives
        for v_neg in negatives:

            neg_score = score_pair(Z, u, v_neg)

            candidate_scores.append((neg_score, 0))

            all_scores.append(neg_score)
            all_labels.append(0)
        
        inf_time += time.time() - new
        # ------------------------------------------------
        # ranking
        # ------------------------------------------------

        candidate_scores.sort(
            reverse=True,
            key=lambda x: x[0]
        )

        rank = None

        for idx, (_, label) in enumerate(candidate_scores, start=1):

            if label == 1:
                rank = idx
                break

        # ------------------------------------------------
        # MRR
        # ------------------------------------------------

        reciprocal_ranks.append(1 / rank)

        ranks.append(rank)

        # ------------------------------------------------
        # Hit@K
        # ------------------------------------------------

        if rank <= k:
            hits += 1

        # ------------------------------------------------
        # Recall@K
        # ------------------------------------------------

        recalls.append(1 if rank <= k else 0)

        # ------------------------------------------------
        # NDCG@K
        # ------------------------------------------------

        if rank <= k:
            ndcg = 1 / np.log2(rank + 1)
        else:
            ndcg = 0

        ndcgs.append(ndcg)

    # ----------------------------------------------------
    # classification metrics
    # ----------------------------------------------------

    auc = roc_auc_score(
        all_labels,
        all_scores
    )

    ap = average_precision_score(
        all_labels,
        all_scores
    )

    # ----------------------------------------------------
    # final metrics
    # ----------------------------------------------------

    return {
        "split": split_name,
        "edges": len(edges),

        # classification
        "auc": auc,
        "ap": ap,

        # ranking
        "mrr": np.mean(reciprocal_ranks),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),

        f"hit@{k}": hits / len(reciprocal_ranks),
        f"recall@{k}": np.mean(recalls),
        f"ndcg@{k}": np.mean(ndcgs),

        # timing
        "inference_time_sec": inf_time
    }

# ============================================================
# SAVE RESULTS
# ============================================================

def save_results_json(results, output_dir):

    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"results_{args.embed_dim}.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved JSON results -> {path}")


def save_results_csv(results, output_dir):

    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "results.csv"

    file_exists = path.exists()

    row = {
        "model": "spectral_embedding",
        **results["config"],
        **results["computational"],
        **results["validation"],
        **results["test"]
    }

    with open(path, "a", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=row.keys()
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    print(f"Saved CSV results -> {path}")


# ============================================================
# MAIN
# ============================================================

def main(args):

    print("Loading graph...")
    data = load_json(args.json_path)

    print("Creating temporal split...")

    train_edges_raw, val_edges_raw, test_edges_raw = temporal_split(
        data,
        args.val_year,
        args.test_year
    )

    all_edges = (
        train_edges_raw +
        val_edges_raw +
        test_edges_raw
    )

    node_to_idx, idx_to_node = build_node_mapping(
        all_edges
    )

    # --------------------------------------------------------
    # convert edges to integer node ids
    # --------------------------------------------------------

    def convert_edges(edge_list):

        converted = []

        for u, v in edge_list:
            converted.append((
                node_to_idx[u],
                node_to_idx[v]
            ))

        return converted

    train_edges = convert_edges(train_edges_raw)
    val_edges = convert_edges(val_edges_raw)
    test_edges = convert_edges(test_edges_raw)

    # --------------------------------------------------------
    # graph
    # --------------------------------------------------------

    G_train = build_graph(
        train_edges_raw,
        node_to_idx
    )

    print("Building 2-hop candidate negatives...")
    candidate_dict = build_two_hop_candidates(G_train)

    forbidden = set()

    for u, v in all_edges:

        a = min(
            node_to_idx[u],
            node_to_idx[v]
        )

        b = max(
            node_to_idx[u],
            node_to_idx[v]
        )

        forbidden.add((a, b))

    # --------------------------------------------------------
    # train
    # --------------------------------------------------------

    print("Training spectral embedding...")

    Z, train_time, memory_mb = spectral_embedding(
        G_train,
        args.embed_dim
    )

    # --------------------------------------------------------
    # validation
    # --------------------------------------------------------

    print("Evaluating validation...")

    val_results = evaluate_ranking(
        "validation",
        val_edges,
        Z,
        candidate_dict,
        forbidden,
        negatives_per_positive=args.neg_sample,
        k=args.top_k
    )


    # --------------------------------------------------------
    # test
    # --------------------------------------------------------

    print("Evaluating test...")

    test_results = evaluate_ranking(
        "test",
        test_edges,
        Z,
        candidate_dict,
        forbidden,
        negatives_per_positive=args.neg_sample,
        k=args.top_k
    )

    # --------------------------------------------------------
    # collect results
    # --------------------------------------------------------

    results = {

        "config": {
            "val_year": args.val_year,
            "test_year": args.test_year,
            "embedding_dim": args.embed_dim,
            "top_k": args.top_k,
            "nodes": len(node_to_idx),
            "train_edges": len(train_edges),
            "validation_edges": len(val_edges),
            "test_edges": len(test_edges)
        },

        "computational": {
            "training_time_sec": train_time,
            "memory_usage_mb": memory_mb
        },

        "validation": val_results,

        "test": test_results
    }

    # --------------------------------------------------------
    # print results
    # --------------------------------------------------------

    print("\n==============================")
    print("RESULTS")
    print("==============================")

    print("\n--- VALIDATION ---")

    for k, v in val_results.items():

        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    print("\n--- TEST ---")

    for k, v in test_results.items():

        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    print("\n--- COMPUTATIONAL ---")
    print(f"training_time_sec: {train_time:.4f}")
    print(f"memory_usage_mb: {memory_mb:.2f}")

    # --------------------------------------------------------
    # save
    # --------------------------------------------------------

    output_dir = Path(args.output_dir)

    save_results_json(results, output_dir)

    #save_results_csv(results, output_dir)


# ============================================================
# ARGPARSE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Spectral Embedding Link Prediction")
    parser.add_argument( "--json_path", type=str, default="./dataset/ogbl_collab/processed/combined.json", help="Path to graph JSON file")
    parser.add_argument("--val_year", type=int, default=2018, help="Validation year")
    parser.add_argument("--test_year", type=int, default=2019, help="Test year")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K for ranking metrics")
    parser.add_argument("--output_dir", type=str, default="./results/spec_emb2", help="Directory to save results")
    parser.add_argument("--neg_sample", type=int, default=10000)

    args = parser.parse_args()

    main(args)
