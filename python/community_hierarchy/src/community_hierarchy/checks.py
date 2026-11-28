"""Checks of the arguments of fit() and the reading of the input graph.

The wrappers check every input and raise the errors; the C++ core assumes valid
input (src/interactions.h, src/model_presets.h).
"""

from __future__ import annotations

import math
import numbers
from typing import Any, Iterable, Optional

import networkx as nx
import numpy as np

# Sentinel for unset parameters
_UNSET = object()


def _is_int(x: Any) -> bool:
    """An integer, including numpy integers, but not a bool."""
    return isinstance(x, numbers.Integral) and not isinstance(x, (bool, np.bool_))


def _is_real(x: Any) -> bool:
    """A real number, including numpy scalars, but not a bool."""
    return isinstance(x, numbers.Real) and not isinstance(x, (bool, np.bool_))


def _edge_weight(u: Any, v: Any, data: dict, *, integer: bool, graph: str = "G") -> float:
    """An edge's ``weight`` (1 if absent): a finite positive number, and a whole
    number (a count of interactions) unless ``integer`` is False."""
    w = data.get("weight", 1.0)
    if not _is_real(w):
        raise TypeError(f"Edge ({u!r}, {v!r}) of {graph} has non-numeric weight {w!r}.")
    w = float(w)
    if not math.isfinite(w) or w <= 0.0:
        raise ValueError(
            f"Edge ({u!r}, {v!r}) of {graph} has weight {w!r}; weights must be finite and positive."
        )
    if integer and w != round(w):
        raise ValueError(
            f"Edge ({u!r}, {v!r}) of {graph} has weight {w!r}; without weighted_SBM a weight "
            "is a number of interactions and must be a positive integer. Use "
            "Config(weighted_SBM=True) for real-valued weights."
        )
    return w


def _edges(G: nx.Graph, index: dict, *, integer_weights: bool) -> tuple[list, list, list, list]:
    """The edges of G, checked, as node indices, draw flags and weights.

    An edge with ``type="neutral"`` is a draw and one with ``type="dominant"`` (or no
    type on any edge) a decided interaction; if any edge has a type, every edge
    must.  Self-loops are an error."""
    source, target, draw, weight = [], [], [], []
    typed = untyped = False
    for u, v, data in G.edges(data=True):
        if u == v:
            raise ValueError(
                f"G has a self-loop at node {u!r}; self-loops are not supported (an "
                "interaction needs two nodes). Remove them, e.g. with "
                "G.remove_edges_from(nx.selfloop_edges(G))."
            )
        if "type" in data:
            typed = True
            if data["type"] not in ("dominant", "neutral"):
                raise ValueError(
                    f"Edge ({u!r}, {v!r}) has invalid type {data['type']!r}. "
                    "Must be 'dominant' or 'neutral'."
                )
        else:
            untyped = True
        if typed and untyped:
            raise ValueError(
                "Some edges have a 'type' and some do not; if any edge has a type "
                "('dominant' or 'neutral'), every edge must."
            )
        source.append(index[u])
        target.append(index[v])
        draw.append(data.get("type") == "neutral")
        weight.append(_edge_weight(u, v, data, integer=integer_weights))
    return source, target, draw, weight


def _check_fit_arguments(
    G: nx.Graph,
    *,
    num_groups: Optional[int],
    initial_partition: Optional[Iterable[int]],
    initial_scores: Optional[Iterable[float]],
    initial_individual_depth: Optional[float],
    initial_group_depth: Optional[float],
    seed: Optional[int],
    num_chains: int,
    num_samples: int,
    num_warmup: int,
    num_thin: int,
    sweeps_per_sample: int,
    num_param_warmup: int,
    beta: float,
    num_tempering_chains: int,
    timeout: float,
    show_progress: bool,
) -> None:
    """Check the arguments of fit() other than the model (checked by Config and
    core._check_model_for_graph) and the edges (checked by _edges)."""
    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a networkx.Graph")
    if not G.is_directed():
        raise TypeError(
            "G must be a directed NetworkX graph (e.g., nx.DiGraph or nx.MultiDiGraph)."
        )
    if G.number_of_nodes() == 0:
        raise ValueError("Graph G must have at least one node.")

    for name, val, positive in (
        ("initial_individual_depth", initial_individual_depth, True),
        ("initial_group_depth", initial_group_depth, False),
    ):
        if val is None:
            continue
        if not _is_real(val) or not math.isfinite(float(val)):
            raise TypeError(f"{name} must be a finite number (or None), got {val!r}.")
        if positive and float(val) <= 0:
            raise ValueError(f"{name} must be positive, got {val!r}.")
        if not positive and float(val) < 0:
            raise ValueError(f"{name} must be non-negative, got {val!r}.")

    n = G.number_of_nodes()
    if initial_partition is not None:
        try:
            labels = list(initial_partition)
        except TypeError:
            raise TypeError("initial_partition must be an iterable of integers or None.")
        if len(labels) != n:
            raise ValueError("initial_partition length must match number of nodes in G.")
        if not all(_is_int(v) and v >= 0 for v in labels):
            raise ValueError("initial_partition must contain non-negative integer labels.")
        if len(set(labels)) < max(labels) + 1:
            raise ValueError("initial_partition does not contain members for every group.")
        if num_groups is not None and max(labels) + 1 != num_groups:
            raise ValueError("initial_partition has a different number of groups than num_groups.")

    if initial_scores is not None:
        try:
            scores = list(initial_scores)
        except TypeError:
            raise TypeError("initial_scores must be an iterable of floats or None.")
        if len(scores) != n:
            raise ValueError("initial_scores length must match number of nodes in G.")
        if not all(_is_real(s) and math.isfinite(float(s)) for s in scores):
            raise ValueError("initial_scores must contain finite numeric values.")

    # Chain c is seeded with seed + c in a C int.
    if seed is not None:
        if not _is_int(seed):
            raise TypeError("seed must be an integer or None.")
        if not 0 <= int(seed) < 2**30:
            raise ValueError(f"seed must be in [0, 2**30), got {seed}.")

    for name, val, minimum in (
        ("num_chains", num_chains, 1),
        ("num_samples", num_samples, 1),
        # 0 is allowed: every sample is then the starting state, which is how the
        # description length of a given state is evaluated (tests/test_wasm_parity.py).
        ("sweeps_per_sample", sweeps_per_sample, 0),
        ("num_thin", num_thin, 1),
        ("num_tempering_chains", num_tempering_chains, 1),
        ("num_warmup", num_warmup, 0),
        ("num_param_warmup", num_param_warmup, 0),
    ):
        if not _is_int(val) or val < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}, got {val!r}.")

    if not _is_real(beta) or not math.isfinite(float(beta)) or float(beta) < 0.0:
        raise ValueError("beta must be a non-negative number.")
    if not _is_real(timeout) or float(timeout) <= 0.0:
        raise ValueError("timeout must be a positive number (seconds).")
    if not isinstance(show_progress, bool):
        raise TypeError("show_progress must be a boolean.")


def _group_graph_edges(G: nx.Graph, group_graph: nx.Graph, index: dict) -> tuple[list, list, list]:
    """The edges of group_graph, checked, as node indices of G and weights."""
    if not isinstance(group_graph, nx.Graph):
        raise TypeError("group_graph must be a NetworkX Graph instance.")
    if group_graph.is_directed():
        raise TypeError("group_graph must be undirected (nx.Graph), not directed.")
    G_nodes = set(G.nodes())
    S_nodes = set(group_graph.nodes())
    if G_nodes != S_nodes:
        only_G = G_nodes - S_nodes
        only_S = S_nodes - G_nodes
        msg = "group_graph node set does not match the primary graph."
        if only_G:
            msg += f" Nodes in G but not group_graph: {sorted(only_G, key=repr)[:5]}{'...' if len(only_G) > 5 else ''}."
        if only_S:
            msg += f" Nodes in group_graph but not G: {sorted(only_S, key=repr)[:5]}{'...' if len(only_S) > 5 else ''}."
        raise ValueError(msg)
    if group_graph.number_of_edges() == 0:
        raise ValueError("group_graph has no edges.")
    a, b, w = [], [], []
    for u, v, data in group_graph.edges(data=True):
        if u == v:
            raise ValueError(
                f"group_graph has a self-loop at node {u!r}; self-loops are not "
                "supported in two-layer mode."
            )
        a.append(index[u])
        b.append(index[v])
        w.append(_edge_weight(u, v, data, integer=False, graph="group_graph"))
    return a, b, w


def _get_node_data(
    G: nx.Graph,
    *,
    group_attribute: str | None = None,
    hierarchy_attribute: str | None = None,
) -> tuple[list[int] | None, list[float] | None]:
    """
    Extract node data from the graph and return as lists.
    Attributes are read only when their corresponding attribute-name argument is provided.
    If an attribute name is None, that property is treated as unspecified and should be inferred.
    Raises error if requested attributes are missing for some nodes but not others, or if they are not in the correct format.

    Parameters
    ----------
    G : nx.Graph
        NetworkX graph to extract node data from.
    group_attribute : str or None
        Node attribute name to read fixed groups from. If None, groups are not read from node attributes.
    hierarchy_attribute : str or None
        Node attribute name to read fixed hierarchy scores from. If None, scores are not read from node attributes.

    Returns
    -------
    node_group_list : list[int] or None
        List of group labels for each node, or None if no group data is present.
    node_score_list : list[float] or None
        List of hierarchy scores for each node, or None if no hierarchy data is present.
    """

    node_to_index_dict = {node: i for i, node in enumerate(G.nodes)}

    # Check whether any of the nodes have requested group/score data.
    # If any one does, they must all have that data and it should be of the correct format.
    group_data_found = False
    score_data_found = False
    node_group_list = [
        0
    ] * G.number_of_nodes()  # Default group assignment if no group data is found
    node_group_name_list = []
    node_score_list = [0.0] * G.number_of_nodes()  # Default scores if no score data is found

    for i, data in G.nodes(data=True):
        if group_attribute is not None and group_attribute in data:
            group_data_found = True
            group_val = data[group_attribute]
            # If this is a new group value we haven't seen before, add it to the list of group names
            if group_val not in node_group_name_list:
                node_group_name_list.append(group_val)

            # Add the group label (as an integer index corresponding to the group name) to the node_group_list
            node_group_list[node_to_index_dict[i]] = node_group_name_list.index(group_val)

        if hierarchy_attribute is not None and hierarchy_attribute in data:
            score_data_found = True
            score_val = data[hierarchy_attribute]
            if not isinstance(score_val, (int, float)):
                raise ValueError(
                    f"Node {i} has invalid score value '{score_val}'. Must be a numeric value."
                )
            node_score_list[node_to_index_dict[i]] = float(score_val)

    for i, data in G.nodes(data=True):
        if group_attribute is not None and group_data_found and group_attribute not in data:
            raise ValueError(
                f"Node {i} is missing '{group_attribute}' attribute while other nodes have it."
            )
        if hierarchy_attribute is not None and score_data_found and hierarchy_attribute not in data:
            raise ValueError(
                f"Node {i} is missing '{hierarchy_attribute}' attribute while other nodes have it."
            )

    # If an attribute name was explicitly provided but not found on any nodes, raise an error
    if group_attribute is not None and not group_data_found:
        raise ValueError(
            f"group_attribute '{group_attribute}' was specified but not found on any nodes."
        )
    if hierarchy_attribute is not None and not score_data_found:
        raise ValueError(
            f"hierarchy_attribute '{hierarchy_attribute}' was specified but not found on any nodes."
        )

    if not group_data_found:
        node_group_list = None
    if not score_data_found:
        node_score_list = None

    return node_group_list, node_score_list
