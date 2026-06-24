import os
import json
import time
import argparse
from pathlib import Path

import networkx as nx
from node2vec import Node2Vec
from gensim.models import Word2Vec

def load_weighted_graph(json_path):
    """
    Expected format:
    {
        "150989": [
            ["224881", 3],
            ["29838", 1]
        ]
    }
    """

    print(f"[INFO] Loading graph from: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    G = nx.Graph()

    edge_count = 0

    for src, neighbors in data.items():
        for dst, weight in neighbors:

            weight = float(weight)

            # Avoid duplicate undirected edges
            if not G.has_edge(src, dst):
                G.add_edge(src, dst, weight=weight)
                edge_count += 1

    print(f"[INFO] Nodes: {G.number_of_nodes():,}")
    print(f"[INFO] Edges: {edge_count:,}")

    return G


def save_embeddings(model, out_path):
    """
    Save embeddings in word2vec text format:
    node_id val1 val2 ...
    """

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    model.wv.save_word2vec_format(out_path)

    print(f"[INFO] Embeddings saved to: {out_path}")


def main():

    parser = argparse.ArgumentParser()

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    parser.add_argument(
        "--graph_path",
        type=str,
        default="./dataset/edges_existed_sumweight.json"
    )

    parser.add_argument(
        "--output_path",
        type=str
    )

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--window_size", type=int, default=3)
    parser.add_argument("--neg_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)

    parser.add_argument("--walks", type=int, default=20)
    parser.add_argument("--length", type=int, default=100)

    parser.add_argument("--workers", type=int, default=8)

    # Optional node2vec params
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=1.0)

    args = parser.parse_args()

    print("=" * 60)
    print("Node2Vec Embedding Training")
    print("=" * 60)

    for k, v in vars(args).items():
        print(f"{k}: {v}")

    print("=" * 60)

    # --------------------------------------------------------
    # Load graph
    # --------------------------------------------------------

    graph_start = time.time()

    G = load_weighted_graph(args.graph_path)

    graph_end = time.time()

    print(f"[INFO] Graph loading time: {graph_end - graph_start:.2f}s")

    # --------------------------------------------------------
    # Create node2vec walker
    # --------------------------------------------------------

    print("\n[INFO] Initializing Node2Vec...")

    n2v_start = time.time()

    node2vec = Node2Vec(
        G,
        dimensions=args.emb_dim,
        walk_length=args.length,
        num_walks=args.walks,
        workers=args.workers,
        p=args.p,
        q=args.q,
        weight_key="weight"
    )

    n2v_end = time.time()

    print(f"[INFO] Random walk generation time: {n2v_end - n2v_start:.2f}s")

    # --------------------------------------------------------
    # Train skipgram
    # --------------------------------------------------------

    print("\n[INFO] Training skip-gram model...")

    train_start = time.time()

    model = node2vec.fit(
        window=args.window_size,
        min_count=1,
        batch_words=args.batch_size,
        epochs=args.epochs,
        negative=args.neg_samples,
        sg=1,                  # skipgram
        workers=args.workers,
        alpha=args.lr
    )

    train_end = time.time()

    print(f"[INFO] Training time: {train_end - train_start:.2f}s")

    # --------------------------------------------------------
    # Save embeddings
    # --------------------------------------------------------

    save_embeddings(model, args.output_path)

    print("\n[INFO] Finished successfully")


if __name__ == "__main__":
    main()
