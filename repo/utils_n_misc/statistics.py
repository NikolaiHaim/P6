import json
import numpy as np
import pandas as pd
import networkx as nx


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "./dataset/ogbl_collab/processed/combined.json"

# Example split setup
SPLITS = {
    "train": lambda year: year <= 2017,
    "validation": lambda year: year == 2018,
    "test": lambda year: year == 2019,
}


# ============================================================
# HELPERS
# ============================================================

def build_graph_from_years(data, condition):
    """
    Build graph using edges from years matching condition.
    """

    G = nx.Graph()

    total_edges = 0

    for year_str, edges in data.items():

        year = int(year_str)

        if not condition(year):
            continue

        G.add_edges_from(edges)

        total_edges += len(edges)

    return G, total_edges


def compute_graph_statistics(G, split_name):

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    degrees = np.array([d for _, d in G.degree()])

    # Safety check
    if len(degrees) == 0:
        print(f"{split_name}: Empty graph")
        return None

    avg_degree = degrees.mean()
    median_degree = np.median(degrees)

    min_degree = degrees.min()
    max_degree = degrees.max()

    degree_variance = degrees.var()
    degree_std = degrees.std()

    density = nx.density(G)

    num_components = nx.number_connected_components(G)

    largest_component = max(nx.connected_components(G), key=len)

    largest_component_size = len(largest_component)

    largest_component_ratio = (
        largest_component_size / num_nodes
    )

    avg_clustering = nx.average_clustering(G)

    # Degree percentiles
    p50 = np.percentile(degrees, 50)
    p90 = np.percentile(degrees, 90)
    p95 = np.percentile(degrees, 95)
    p99 = np.percentile(degrees, 99)

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print("=" * 60)
    print(f"{split_name.upper()} GRAPH")
    print("=" * 60)

    print(f"Nodes:                    {num_nodes:,}")
    print(f"Edges:                    {num_edges:,}")

    print()

    print(f"Average degree:           {avg_degree:.4f}")
    print(f"Median degree:            {median_degree:.4f}")
    print(f"Min degree:               {min_degree}")
    print(f"Max degree:               {max_degree}")

    print()

    print(f"Degree variance:          {degree_variance:.4f}")
    print(f"Degree std:               {degree_std:.4f}")

    print()

    print(f"Density:                  {density:.10f}")

    print()

    print(f"Connected components:     {num_components:,}")
    print(f"Largest component size:   {largest_component_size:,}")
    print(f"Largest component ratio:  {largest_component_ratio:.4f}")

    print()

    print(f"Average clustering:       {avg_clustering:.6f}")

    print()

    print("Degree percentiles")
    print(f"  P50:                    {p50:.2f}")
    print(f"  P90:                    {p90:.2f}")
    print(f"  P95:                    {p95:.2f}")
    print(f"  P99:                    {p99:.2f}")

    print()

    return {
        "split": split_name,
        "nodes": num_nodes,
        "edges": num_edges,
        "avg_degree": avg_degree,
        "degree_variance": degree_variance,
        "density": density,
        "components": num_components,
        "largest_component_ratio": largest_component_ratio,
        "avg_clustering": avg_clustering,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading JSON dataset...")

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    all_results = []

    for split_name, condition in SPLITS.items():

        G, edge_count = build_graph_from_years(
            data,
            condition
        )

        print()
        print(f"{split_name}: {edge_count:,} raw edges")

        stats = compute_graph_statistics(
            G,
            split_name
        )

        if stats is not None:
            all_results.append(stats)

    # --------------------------------------------------------
    # Summary table
    # --------------------------------------------------------

    summary_df = pd.DataFrame(all_results)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(summary_df)


if __name__ == "__main__":
    main()