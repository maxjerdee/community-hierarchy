# Convert all directed nomination networks in nomination_networks/ to dominance networks
# in dominance_networks/.
#
# Rules:
# - If A nominates B and B does not nominate A, then B dominates A:
#       B -> A, type="dominant"
# - If A nominates B and B nominates A, the interaction is neutral:
#       A -> B, type="neutral" (stored once per mutual pair)
#
# Extra attributes written to output graphs:
# - Node attribute:    modularity_group
# - Node attribute:    win_percentage
# - Graph attribute:   modularity

from pathlib import Path

import networkx as nx


INPUT_DIR = Path("nomination_networks")
OUTPUT_DIR = Path("dominance_networks")


def convert_graph(G: nx.DiGraph) -> nx.DiGraph:
    if not G.is_directed():
        raise ValueError("Input graph must be directed")

    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))

    for u, v in G.edges():
        if G.has_edge(v, u):
            # Store each neutral pair once
            if not H.has_edge(v, u):
                H.add_edge(u, v, type="neutral")
        else:
            H.add_edge(v, u, type="dominant")

    return H


def add_modularity_attributes(H: nx.DiGraph) -> None:
    U = nx.Graph()
    U.add_nodes_from(H.nodes())
    for u, v, data in H.edges(data=True):
        if not U.has_edge(u, v):
            U.add_edge(u, v)

    communities = list(nx.community.greedy_modularity_communities(U))
    modularity_value = nx.community.modularity(U, communities)

    node_to_group: dict = {}
    for group_id, community in enumerate(communities):
        for node in community:
            node_to_group[node] = group_id

    for node in H.nodes():
        H.nodes[node]["modularity_group"] = int(node_to_group[node])

    H.graph["modularity"] = float(modularity_value)


def add_win_percentage(H: nx.DiGraph) -> None:
    dominant_out = {node: 0.0 for node in H.nodes()}
    dominant_in = {node: 0.0 for node in H.nodes()}
    neutral_degree = {node: 0.0 for node in H.nodes()}

    for u, v, data in H.edges(data=True):
        edge_type = data.get("type", "dominant")
        if edge_type == "neutral":
            neutral_degree[u] += 1.0
            neutral_degree[v] += 1.0
        else:
            dominant_out[u] += 1.0
            dominant_in[v] += 1.0

    for node in H.nodes():
        wins = dominant_out[node]
        losses = dominant_in[node]
        ties = neutral_degree[node]
        H.nodes[node]["win_percentage"] = float((wins - losses) / (wins + losses + ties + 1e-6))


def output_name(input_path: Path) -> Path:
    stem = input_path.stem
    if stem.endswith("_raw"):
        stem = stem[: -len("_raw")]
    return OUTPUT_DIR / f"{stem}.gml"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_files = sorted(INPUT_DIR.glob("*.gml"))
    if not input_files:
        raise FileNotFoundError(f"No .gml files found in {INPUT_DIR}")

    for input_path in input_files:
        print(f"Processing: {input_path}")
        G = nx.read_gml(input_path)
        H = convert_graph(G)
        add_modularity_attributes(H)
        add_win_percentage(H)

        out_path = output_name(input_path)
        nx.write_gml(H, out_path)
        print(f"  Wrote: {out_path} (modularity={H.graph['modularity']:.4f})")


if __name__ == "__main__":
    main()
