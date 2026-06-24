import json
import random
import time
from collections import defaultdict
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# DATA LOADING
# ============================================================


def load_yearly_edges(path):
    with open(path, "r") as f:
        data = json.load(f)

    edges_by_year = {}

    for year, edges in data.items():
        year = int(year)
        edges_by_year[year] = [(str(src), str(dst)) for src, dst in edges]

    return edges_by_year


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================


def temporal_split(edges_by_year, split_year):
    train_edges = []
    val_edges = []
    test_edges = []

    for year, edges in edges_by_year.items():

        if year <= split_year - 1:
            train_edges.extend(edges)

        elif year == split_year:
            val_edges.extend(edges)

        elif year == split_year + 1:
            test_edges.extend(edges)

    return train_edges, val_edges, test_edges


# ============================================================
# NODE INDEXING
# ============================================================


def build_node_mapping(edges):
    nodes = set()

    for src, dst in edges:
        nodes.add(src)
        nodes.add(dst)

    node2idx = {node: idx for idx, node in enumerate(sorted(nodes))}
    idx2node = {idx: node for node, idx in node2idx.items()}

    return node2idx, idx2node


# ============================================================
# ADJACENCY MATRIX
# ============================================================


def build_adjacency(edges, node2idx):
    num_nodes = len(node2idx)

    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float16)

    for src, dst in edges:
        i = node2idx[src]
        j = node2idx[dst]

        A[i, j] = 1.0
        A[j, i] = 1.0

    return A


def build_sparse_adjacency(edges, node2idx):
    row = []
    col = []

    for src, dst in edges:
        i = node2idx[src]
        j = node2idx[dst]

        row.extend([i, j])
        col.extend([j, i])

    indices = torch.tensor([row, col], dtype=torch.long)

    values = torch.ones(indices.shape[1])

    num_nodes = len(node2idx)

    A = torch.sparse_coo_tensor(
        indices,
        values,
        (num_nodes, num_nodes)
    )

    return A.coalesce()


# ============================================================
# NORMALIZATION
# ============================================================


def normalize_adjacency(A):
    num_nodes = A.size(0)

    # ------------------------------------------------
    # Add self-loops
    # ------------------------------------------------

    self_loop_indices = torch.arange(num_nodes)

    self_loops = torch.stack([
        self_loop_indices,
        self_loop_indices
    ])

    self_loop_values = torch.ones(num_nodes)

    I = torch.sparse_coo_tensor(
        self_loops,
        self_loop_values,
        (num_nodes, num_nodes)
    )

    A_hat = (A + I).coalesce()

    # ------------------------------------------------
    # Compute degree vector
    # ------------------------------------------------

    indices = A_hat.indices()
    values = A_hat.values()

    row = indices[0]
    col = indices[1]

    degree = torch.zeros(num_nodes)

    degree.index_add_(0, row, values)

    # ------------------------------------------------
    # Compute D^(-1/2)
    # ------------------------------------------------

    deg_inv_sqrt = degree.pow(-0.5)

    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0

    # ------------------------------------------------
    # Normalize edge values directly
    # ------------------------------------------------

    normalized_values = (
        deg_inv_sqrt[row]
        * values
        * deg_inv_sqrt[col]
    )

    return torch.sparse_coo_tensor(
        indices,
        normalized_values,
        A_hat.shape
    ).coalesce()


# ============================================================
# NEGATIVE SAMPLING
# ============================================================


def build_edge_set(edges, node2idx):
    edge_set = set()

    for src, dst in edges:
        i = node2idx[src]
        j = node2idx[dst]

        edge_set.add((i, j))
        edge_set.add((j, i))

    return edge_set


def sample_negative_edges(num_nodes, num_samples, edge_set):
    negatives = []

    while len(negatives) < num_samples:
        i = random.randint(0, num_nodes - 1)
        j = random.randint(0, num_nodes - 1)

        if i == j:
            continue

        if (i, j) in edge_set:
            continue

        negatives.append((i, j))

    return negatives


# ============================================================
# GCN LAYER
# ============================================================


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, A_hat, X):
        X = self.linear(X)
        X = torch.sparse.mm(A_hat, X)

        return X


# ============================================================
# GCN MODEL
# ============================================================

class GCN(nn.Module):
    def __init__(
        self,
        num_nodes,
        hidden_dim=128,
        embedding_dim=128,
        layers=2,
        dropout=0.3
    ):
        super().__init__()

        self.node_embedding = nn.Embedding(num_nodes, hidden_dim)

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        # Hidden layers
        for _ in range(layers - 1):
            self.layers.append(
                GCNLayer(hidden_dim, hidden_dim)
            )

        # Final output layer
        self.output_layer = GCNLayer(
            hidden_dim,
            embedding_dim
        )

    def forward(self, A_hat):
        X = self.node_embedding.weight

        for layer in self.layers:
            X = layer(A_hat, X)
            X = F.relu(X)
            X = self.dropout(X)

        X = self.output_layer(A_hat, X)

        return X


# ============================================================
# LINK PREDICTION OBJECTIVE
# ============================================================


def edge_scores(Z, edges):
    src = Z[edges[:, 0]]
    dst = Z[edges[:, 1]]

    scores = torch.sum(src * dst, dim=1)

    return torch.sigmoid(scores)



def compute_loss(Z, positive_edges, negative_edges):
    positive_scores = edge_scores(Z, positive_edges)
    negative_scores = edge_scores(Z, negative_edges)

    positive_loss = -torch.log(positive_scores + 1e-15).mean()
    negative_loss = -torch.log(1 - negative_scores + 1e-15).mean()

    return positive_loss + negative_loss


# ============================================================
# TRAINING
# ============================================================


def train_gcn(
    model,
    A_hat,
    train_edges,
    node2idx,
    epochs=100,
    learning_rate=0.001,
    decay=1e-5
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)

    num_nodes = len(node2idx)

    edge_set = build_edge_set(train_edges, node2idx)

    positive_edges = torch.tensor(
        [[node2idx[src], node2idx[dst]] for src, dst in train_edges],
        dtype=torch.long,
    )

    start_time = time.time()

    for epoch in range(epochs):
        model.train()

        negative_edges = sample_negative_edges(
            num_nodes=num_nodes,
            num_samples=len(train_edges),
            edge_set=edge_set,
        )
        
        negative_edges = torch.tensor(negative_edges, dtype=torch.long)

        Z = model(A_hat)

        loss = compute_loss(Z, positive_edges, negative_edges)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    training_time = time.time() - start_time

    print(f"\nTraining complete in {training_time:.2f} seconds")

    return model(A_hat).detach().cpu().numpy(), training_time


# ============================================================
# SAVE EMBEDDINGS
# ============================================================


def save_embeddings(path, embeddings, idx2node):
    output = {}

    for idx, vector in enumerate(embeddings):
        node_id = idx2node[idx]
        output[node_id] = vector.tolist()

    with open(path, "w") as f:
        json.dump(output, f)

    print(f"Saved embeddings to: {path}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main(args):
    DATA_PATH = args.data_path

    SPLIT_YEAR = args.split_year

    HIDDEN_DIM = args.hidden_dim
    EMBEDDING_DIM = args.embedding_dim
    LAYERS = args.layers

    EPOCHS = args.epochs

    LEARNING_RATE = args.learn_rate

    DROPOUT = args.dropout

    WEIGHT_DECAY = args.decay

    OUTPUT_PATH = args.results_path

    print("Loading graph...")

    yearly_edges = load_yearly_edges(DATA_PATH)

    print("Creating temporal split...")

    train_edges, val_edges, test_edges = temporal_split(yearly_edges, SPLIT_YEAR,)

    print(f"Train edges: {len(train_edges):,}")
    print(f"Validation edges: {len(val_edges):,}")
    print(f"Test edges:  {len(test_edges):,}")

    print("Building node mappings...")

    node2idx, idx2node = build_node_mapping(train_edges)

    print(f"Nodes: {len(node2idx):,}")

    print("Building adjacency matrix...")

    A = build_sparse_adjacency(train_edges, node2idx)

    print("Normalizing adjacency...")

    A_hat = normalize_adjacency(A)

    print("Initializing GCN...")

    model = GCN(
        num_nodes=len(node2idx),
        hidden_dim=HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM,
	layers=LAYERS,
        dropout=DROPOUT
    )

    print("Training GCN...")

    embeddings, training_time = train_gcn(
        model=model,
        A_hat=A_hat,
        train_edges=train_edges,
        node2idx=node2idx,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        decay=WEIGHT_DECAY
    )

    print(f"Embeddings shape: {embeddings.shape}")

    save_embeddings(
        OUTPUT_PATH,
        embeddings,
        idx2node,
    )

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="./dataset/ogbl_collab/processed/combined.json")
    parser.add_argument("--results_path", type=str, default="./gcn/gcn_embeddings.json")
    parser.add_argument("--split_year", type=int, default=2018)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learn_rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--decay", type=float, default=1e-5)
    parser.add_argument("--layers", type=int, default=2)

    args = parser.parse_args()
    
    main(args)
