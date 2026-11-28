"""Graphs and settings shared by the tests."""

from __future__ import annotations

import math
from pathlib import Path

import networkx as nx

_DATA_IN = Path(__file__).parent.parent / "data" / "inputs"


_DATA_OUT = Path(__file__).parent.parent / "data" / "outputs"


def _c2v(concentration: float) -> float:
    return math.exp(
        math.lgamma(concentration + 0.5) - math.lgamma(0.5) - math.lgamma(concentration + 1)
    )


# Helpers: small directed graph fixture
def _small_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(range(6))
    # Two clique-like groups with directed edges
    edges = [
        (0, 1),
        (1, 0),
        (0, 2),
        (2, 0),
        (1, 2),
        (2, 1),
        (3, 4),
        (4, 3),
        (3, 5),
        (5, 3),
        (4, 5),
        (5, 4),
        (0, 3),
        (3, 0),
    ]
    G.add_edges_from(edges)
    return G


def _graph() -> nx.DiGraph:
    return nx.gnp_random_graph(40, 0.1, seed=1, directed=True)


# 2026-11 audit: argument validation, silent overrides, node order, two-layer
# input handling and GML output.
_QUICK = dict(num_samples=2, num_warmup=1, sweeps_per_sample=1, show_progress=False, seed=1)


def _typed_graph() -> nx.DiGraph:
    """Two cliques joined by one edge; within each clique half the edges are draws."""
    G = nx.DiGraph()
    for block in (range(0, 6), range(6, 12)):
        for i in block:
            for j in block:
                if i < j:
                    G.add_edge(i, j, type="neutral" if (i + j) % 2 else "dominant")
    G.add_edge(0, 6, type="dominant")
    return G


# Coincidence fixture tests (same graph as asbm test cases)
def _triangle_graph() -> nx.DiGraph:
    """3-node directed graph matching asbm's test3: edges 1→2 w=1, 2→3 w=2, 3→1 w=1."""
    G = nx.DiGraph()
    G.add_nodes_from([1, 2, 3])
    G.add_edge(1, 2, weight=1.0)
    G.add_edge(2, 3, weight=2.0)
    G.add_edge(3, 1, weight=1.0)
    return G


# direction_probabilities
def _ranked_graph() -> nx.DiGraph:
    """String-labelled graph with a clear dominant node for hierarchy tests."""
    G = nx.DiGraph()
    for _ in range(10):
        for loser in ("B", "C", "D"):
            w = G.get_edge_data("A", loser, {}).get("weight", 0)
            G.add_edge("A", loser, weight=w + 1)
    for _ in range(5):
        for loser in ("C", "D"):
            w = G.get_edge_data("B", loser, {}).get("weight", 0)
            G.add_edge("B", loser, weight=w + 1)
    return G
