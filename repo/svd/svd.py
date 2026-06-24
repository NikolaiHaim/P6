import json
import time
import random
import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from scipy.sparse.linalg import svds
from sklearn.metrics import roc_auc_score


# =========================================================
# ARGPARSE
# =========================================================


parser = argparse.ArgumentParser(description="SVD Link Prediction")

parser.add_argument("--input_path", type=str, required=True, help="Path to edges_by_year.json")
parser.add_argument("--output_dir", type=str, default="results/svd", help="Directory to save outputs")
parser.add_argument("--train_year", type=int, default=2017)
parser.add_argument("--valid_year", type=int, default=2018)
parser.add_argument("--test_year", type=int, default=2019)
parser.add_argument("--embedding_dim", type=int, default=128)
parser.add_argument("--hit_k", type=int, default=50)
parser.add_argument("--negative_ratio", type=int, default=1000)
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()


# =========================================================
# CONFIG
# =========================================================
INPUT_PATH = args.input_path
OUTPUT_DIR = Path(args.output_dir)

TRAIN_YEAR = args.train_year
VALID_YEAR = args.valid_year
TEST_YEAR = args.test_year

EMBEDDING_DIM = args.embedding_dim
HIT_K = args.hit_k
NEGATIVE_RATIO = args.negative_ratio
SEED = args.seed

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)


# =========================================================
# LOAD JSON
# =========================================================
print("\nLoading graph...")

with open(INPUT_PATH, "r") as f:
    yearly_edges = json.load(f)


# =========================================================
# SPLIT DATA
# =========================================================
train_edges = []
valid_edges = []
test_edges = []

for year, edges in yearly_edges.items():

    year = int(year)

    if year <= TRAIN_YEAR:
        train_edges.extend(edges)

    elif year == VALID_YEAR:
        valid_edges.extend(edges)

    elif year == TEST_YEAR:
        test_edges.extend(edges)

print(f"Train edges: {len(train_edges)}")
print(f"Validation edges: {len(valid_edges)}")
print(f"Test edges: {len(test_edges)}")


# =========================================================
# NODE INDEXING
# =========================================================
nodes = set()

for u, v in train_edges + valid_edges + test_edges:

    nodes.add(str(u))
    nodes.add(str(v))

nodes = sorted(list(nodes))

node_to_idx = {
    node: i
    for i, node in enumerate(nodes)
}

idx_to_node = {
    i: node
    for node, i in node_to_idx.items()
}

n_nodes = len(nodes)

print(f"Nodes: {n_nodes}")


# =========================================================
# BUILD TRAIN ADJ MATRIX
# =========================================================
print("\nBuilding adjacency matrix...")

rows = []
cols = []

for u, v in train_edges:

    i = node_to_idx[str(u)]
    j = node_to_idx[str(v)]

    rows.extend([i, j])
    cols.extend([j, i])

data = np.ones(len(rows), dtype=np.float32)

A = sp.csr_matrix(
    (data, (rows, cols)),
    shape=(n_nodes, n_nodes)
)

print("Adjacency matrix complete")


# =========================================================
# EDGE SET
# =========================================================
all_positive_edges = set()

for edge_group in [train_edges, valid_edges, test_edges]:

    for u, v in edge_group:

        i = node_to_idx[str(u)]
        j = node_to_idx[str(v)]

        all_positive_edges.add((i, j))
        all_positive_edges.add((j, i))


# =========================================================
# NEGATIVE SAMPLING
# =========================================================
def sample_negative_edges(num_samples):

    negatives = set()

    while len(negatives) < num_samples:

        i = random.randint(0, n_nodes - 1)
        j = random.randint(0, n_nodes - 1)

        if i == j:
            continue

        if (i, j) in all_positive_edges:
            continue

        negatives.add((i, j))

    return list(negatives)


# =========================================================
# TRAIN SVD
# =========================================================
print("\nTraining SVD...")

train_start = time.perf_counter()

U, S, VT = svds(
    A,
    k=EMBEDDING_DIM
)

# descending singular values
idx = np.argsort(-S)

S = S[idx]
U = U[:, idx]

embeddings = U @ np.diag(np.sqrt(S))

train_end = time.perf_counter()

training_time = train_end - train_start

print(f"SVD training time: {training_time:.4f} sec")


# =========================================================
# SAVE EMBEDDINGS
# =========================================================
print("\nSaving embeddings...")

embedding_path = OUTPUT_DIR / f"embeddings_{args.embedding_dim}.npy"

np.save(
    embedding_path,
    embeddings
)

node_mapping_path = OUTPUT_DIR / f"node_mapping_{args.embedding_dim}.json"

with open(node_mapping_path, "w") as f:

    json.dump(
        node_to_idx,
        f,
        indent=2
    )


# =========================================================
# SCORING
# =========================================================
def predict_score(i, j):

    return embeddings[i] @ embeddings[j]


# =========================================================
# EVALUATION
# =========================================================
def evaluate(edge_list, split_name):

    print(f"\nEvaluating {split_name}...")

    positive_edges = []

    for u, v in edge_list:

        i = node_to_idx[str(u)]
        j = node_to_idx[str(v)]

        positive_edges.append((i, j))

    negative_edges = sample_negative_edges(
        len(positive_edges) * NEGATIVE_RATIO
    )

    y_true = []
    y_scores = []

    inference_start = time.perf_counter()

    # positive scores
    for i, j in positive_edges:

        y_true.append(1)
        y_scores.append(predict_score(i, j))

    # negative scores
    for i, j in negative_edges:

        y_true.append(0)
        y_scores.append(predict_score(i, j))

    inference_end = time.perf_counter()

    inference_time = (
        inference_end - inference_start
    )

    # =====================================================
    # AUC
    # =====================================================

    auc = roc_auc_score(y_true, y_scores)

    # =====================================================
    # HIT@K
    # =====================================================

    hit_at_k = compute_hit_at_k(positive_edges, k=HIT_K, num_negative_samples=NEGATIVE_RATIO)

def compute_hit_at_k(positive_edges, k, num_negative_samples=1000):

    hits = 0

    for pos_i, pos_j in positive_edges:

        positive_score = predict_score(pos_i, pos_j)

        candidate_scores = [positive_score]

        negatives_added = 0

        while negatives_added < num_negative_samples:

            neg_j = random.randint(0, n_nodes - 1)

            if neg_j == pos_i:
                continue

            if (pos_i, neg_j) in all_positive_edges:
                continue

            neg_score = predict_score(pos_i, neg_j)

            candidate_scores.append(neg_score)

            negatives_added += 1

        candidate_scores.sort(reverse=True)

        rank = 1 + sum(
            score > positive_score
            for score in candidate_scores
            )

        if rank <= k:
            hits += 1

    return hits / len(positive_edges)


# =========================================================
# RUN EVALUATION
# =========================================================
validation_metrics = evaluate(
    valid_edges,
    "validation"
)

test_metrics = evaluate(
    test_edges,
    "test"
)


# =========================================================
# FINAL RESULTS
# =========================================================
results = {
    "model": "svd",

    "config": {
        "input": INPUT_PATH,
        "train_year": TRAIN_YEAR,
        "valid_year": VALID_YEAR,
        "test_year": TEST_YEAR,
        "embedding_dim": EMBEDDING_DIM,
        "hit_k": HIT_K,
        "negative_ratio": NEGATIVE_RATIO,
        "seed": SEED
    },

    "graph_stats": {
        "num_nodes": n_nodes,
        "num_train_edges": len(train_edges),
        "num_validation_edges": len(valid_edges),
        "num_test_edges": len(test_edges)
    },

    "training": {
        "training_time_sec": float(training_time)
    },

    "validation": validation_metrics,

    "test": test_metrics
}


# =========================================================
# SAVE RESULTS JSON
# =========================================================
results_path = OUTPUT_DIR / f"results_{args.embedding_dim}.json"

with open(results_path, "w") as f:
    json.dump(results, f, indent=4)

print("\nResults saved:")
print(results_path)

print("\nEmbeddings saved:")
print(embedding_path)

print("\nNode mapping saved:")
print(node_mapping_path)
