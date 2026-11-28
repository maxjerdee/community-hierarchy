# Handling input and output of data

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Iterable, Any
import math
import numbers
import warnings

import numpy as np

# Sentinel for unset parameters
_UNSET = object()


def _to_gml_safe_value(value: Any) -> Any:
    """Recursively convert values to NetworkX GML-compatible Python types."""
    if isinstance(value, dict):
        return {str(k): _to_gml_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_gml_safe_value(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _to_gml_safe_value(item_method())
        except (TypeError, ValueError):
            pass

    return str(value)


def _sanitize_graph_attributes_for_gml(G: nx.Graph) -> None:
    """Normalize graph, node, and edge attributes in-place for GML serialization."""
    graph_attrs = {str(k): _to_gml_safe_value(v) for k, v in G.graph.items()}
    G.graph.clear()
    G.graph.update(graph_attrs)

    for node in G.nodes:
        attrs = G.nodes[node]
        sanitized = {str(k): _to_gml_safe_value(v) for k, v in attrs.items()}
        attrs.clear()
        attrs.update(sanitized)

    for _, _, attrs in G.edges(data=True):
        sanitized = {str(k): _to_gml_safe_value(v) for k, v in attrs.items()}
        attrs.clear()
        attrs.update(sanitized)


def _is_int(x: Any) -> bool:
    """An integer, including numpy integers, but not a bool."""
    return isinstance(x, numbers.Integral) and not isinstance(x, (bool, np.bool_))


def _is_real(x: Any) -> bool:
    """A real number, including numpy scalars, but not a bool."""
    return isinstance(x, numbers.Real) and not isinstance(x, (bool, np.bool_))


def _check_forced_values(hierarchy_model: Any, interaction: str,
                         group_depth: Any, ties_parameter: Any) -> None:
    """Raise if an explicit value contradicts one the presets force.

    ``hierarchy_model="bradley_terry"`` fixes ``ties_parameter`` at 0 and
    ``interaction="independent"`` fixes ``group_depth`` at 0 (see
    :func:`_set_model_parameters`).  An explicit, different value (including
    None, "infer") would otherwise be overwritten silently.
    """
    if hierarchy_model == "bradley_terry" and ties_parameter is not _UNSET \
            and not (_is_real(ties_parameter) and float(ties_parameter) == 0.0):
        raise ValueError(
            f"hierarchy_model='bradley_terry' fixes ties_parameter at 0, but "
            f"ties_parameter={ties_parameter!r} was given. Use "
            "hierarchy_model='bradley_terry_ties' to model draws."
        )
    if interaction == "independent" and group_depth is not _UNSET \
            and not (_is_real(group_depth) and float(group_depth) == 0.0):
        raise ValueError(
            f"interaction='independent' fixes group_depth at 0, but "
            f"group_depth={group_depth!r} was given. Use interaction='coupled' "
            "to fit group_depth."
        )


def _check_weight(u: Any, v: Any, data: dict, *, integer: bool, what: str = "Edge") -> None:
    """A ``weight`` attribute must be a finite positive number (an integer
    multiplicity unless ``integer`` is False)."""
    if "weight" not in data:
        return
    w = data["weight"]
    if not _is_real(w):
        raise TypeError(f"{what} ({u!r}, {v!r}) has non-numeric weight {w!r}.")
    w = float(w)
    if not math.isfinite(w) or w <= 0.0:
        raise ValueError(f"{what} ({u!r}, {v!r}) has weight {w!r}; weights must be finite and positive.")
    if integer and w != round(w):
        raise ValueError(
            f"{what} ({u!r}, {v!r}) has weight {w!r}; without weighted_SBM a weight is a "
            "multiplicity and must be a positive integer. Use Config(weighted_SBM=True) "
            "for real-valued weights."
        )


# Check that the arguments passed to the function are valid, if not raise an error
def _check_input_validity(
    G: nx.Graph,
    node_groups: Optional[pd.DataFrame] = None,
    *,
    group_attribute: Optional[str] = None,
    hierarchy_attribute: Optional[str] = None,
    groups_model: str = "general_ASBM",
    hierarchy_model: str = _UNSET,
    interaction: str = "coupled",
    assortative: bool = _UNSET,
    mixing_variation: str = _UNSET,
    variation_in: Optional[float] = _UNSET,
    variation_out: Optional[float] = _UNSET,
    degree_correction: bool = _UNSET,
    individual_depth: Optional[float] = _UNSET,
    group_depth: Optional[float] = _UNSET,
    ties_parameter: Optional[float] = _UNSET,
    num_groups: Optional[int] = _UNSET,
    weighted_SBM: bool = False,
    initial_partition: Optional[Iterable[int]] = None,
    initial_scores: Optional[Iterable[float]] = None,
    initial_individual_depth: Optional[float] = None,
    initial_group_depth: Optional[float] = None,
    seed: Optional[int] = None,
    num_chains: int = 1,
    samples_per_chain: int = 1000,
    sweeps_per_sample: int = 10,
    burn_sweeps: int = 500,
    param_warmup_sweeps: int = 0,
    thinning: int = 1,
    beta: float = 1.0,
    num_tempering_chains: int = 1,
    no_cache: bool = False,
    timeout: float = 60.0,
    verbose: bool = True,
) -> None:
    """Validate public API inputs and raise informative errors for invalid values.

    This function checks types, ranges and basic consistency for the parameters
    documented in the public API. It intentionally performs lightweight checks
    only (no heavy computation or graph processing).
    """

    # Graph checks
    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a NetworkX graph object.")
    if not G.is_directed():
        raise TypeError(
            "G must be a directed NetworkX graph (e.g., nx.DiGraph or nx.MultiDiGraph)."
        )
    # Check that the graph is not empty
    if G.number_of_nodes() == 0:
        raise ValueError("Graph G must have at least one node.")

    # Node attribute name checks
    for attr_name, attr_value in (
        ("group_attribute", group_attribute),
        ("hierarchy_attribute", hierarchy_attribute),
    ):
        if attr_value is not None:
            if not isinstance(attr_value, str):
                raise TypeError(f"{attr_name} must be a string or None.")
            if not attr_value:
                raise ValueError(f"{attr_name} must be a non-empty string or None.")

    # Warn if group_attribute is provided alongside a non-default groups_model
    if group_attribute is not None and groups_model != "general_ASBM":
        warnings.warn(
            f"group_attribute '{group_attribute}' was provided alongside groups_model '{groups_model}'. "
            "The group_attribute will be used for model initialization, but the specified model type will still be applied. "
            "To avoid confusion, consider using the default groups_model when providing node attributes.",
            UserWarning,
            stacklevel=3,
        )

    # Edges: self-loops, weights and types.  If any edge has a type, all must,
    # and it must be "dominant" or "neutral".
    type_found = False
    neutral_type_found = False
    for u, v, data in G.edges(data=True):
        if u == v:
            raise ValueError(
                f"G has a self-loop at node {u!r}; self-loops are not supported "
                "(an interaction needs two nodes). Remove them, e.g. with "
                "G.remove_edges_from(nx.selfloop_edges(G))."
            )
        _check_weight(u, v, data, integer=not weighted_SBM)
        if "type" in data:
            type_found = True
            edge_type = data["type"]
            if edge_type == "neutral":
                neutral_type_found = True
            if edge_type not in ["dominant", "neutral"]:
                raise ValueError(
                    f"Edge ({u}, {v}) has invalid type '{edge_type}'. Must be 'dominant' or 'neutral'."
                )

    if type_found:
        for u, v, data in G.edges(data=True):
            if "type" not in data:
                raise ValueError(
                    f"Edge ({u}, {v}) is missing 'type' attribute while other edges have it."
                )

    # groups_model
    valid_models = [
        "traditional_SBM",
        "simple_ASBM",
        "hybrid_ASBM",
        "planted_partition",
        "general_ASBM",
    ]
    if groups_model not in valid_models:
        raise ValueError(
            f"Unknown groups_model '{groups_model}'. Valid options: {valid_models}"
        )

    # hierarchy_model
    original_hierarchy_model = hierarchy_model  # Store original value before defaulting
    if hierarchy_model is _UNSET:
        if neutral_type_found:
            hierarchy_model = "bradley_terry_ties"  # Default when ties are present
        else:
            hierarchy_model = "bradley_terry"  # Default when no ties

    # Warn if hierarchy_attribute is provided alongside an explicitly-provided hierarchy_model
    if hierarchy_attribute is not None and original_hierarchy_model is not _UNSET:
        warnings.warn(
            f"hierarchy_attribute '{hierarchy_attribute}' was provided alongside hierarchy_model '{original_hierarchy_model}'. "
            "The hierarchy_attribute will be used for model initialization, but the specified model type will still be applied. "
            "To avoid confusion, consider using the default hierarchy_model when providing node attributes.",
            UserWarning,
            stacklevel=3,
        )

    # Plain Bradley-Terry gives a draw probability 0: a graph with draws in it
    # has likelihood 0 under that model, so it is an error, not a fit.
    if hierarchy_model == "bradley_terry" and neutral_type_found:
        raise ValueError(
            "hierarchy_model='bradley_terry' has no draws, but G has neutral edges. "
            "Use hierarchy_model='bradley_terry_ties' (the default when G has "
            "neutral edges), or remove the neutral edges."
        )

    valid_hierarchy_models = ["bradley_terry", "bradley_terry_ties"]
    if hierarchy_model not in valid_hierarchy_models:
        raise ValueError(
            f"Unknown hierarchy_model '{hierarchy_model}'. Valid options: {valid_hierarchy_models}"
        )

    # interaction
    valid_interactions = ["coupled", "independent"]
    if interaction not in valid_interactions:
        raise ValueError(
            f"Unknown interaction '{interaction}'. Valid options: {valid_interactions}"
        )

    # Fix the unset parameters to their defaults (to pass the type checks below)
    if assortative is _UNSET:
        assortative = True
    if mixing_variation is _UNSET:
        mixing_variation = "general"
    if variation_in is _UNSET:
        variation_in = None
    if variation_out is _UNSET:
        variation_out = None
    if individual_depth is _UNSET:
        individual_depth = None
    if group_depth is _UNSET:
        group_depth = None
    if ties_parameter is _UNSET:
        ties_parameter = None
    if num_groups is _UNSET:
        num_groups = None

    # assortative
    if not isinstance(assortative, bool):
        raise TypeError("assortative must be a boolean.")

    # mixing_variation
    valid_mixing = ["simple", "none", "internal", "external", "general"]
    if mixing_variation not in valid_mixing:
        raise ValueError(f"mixing_variation must be one of {valid_mixing}")

    # variation_in/out: None or 0 <= x <= 1
    for name, val in (("variation_in", variation_in), ("variation_out", variation_out)):
        if val is not None:
            if not _is_real(val):
                raise TypeError(f"{name} must be a float in [0, 1] or None.")
            if not (0.0 <= float(val) <= 1.0):
                raise ValueError(
                    f"{name} must be between 0 and 1 (inclusive). Got {val}."
                )

    # degree_correction: a binary switch (the Config field is a bool)
    if degree_correction is not _UNSET and not isinstance(degree_correction, bool):
        raise TypeError("degree_correction must be True or False.")

    # Depths and the ties parameter: None or a finite number in range
    for name, val, positive in (
        ("individual_depth", individual_depth, True),
        ("group_depth", group_depth, False),
        ("ties_parameter", ties_parameter, False),
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

    # num_groups must be positive integer if provided and not greater than number of nodes
    if num_groups is not None:
        if not _is_int(num_groups) or num_groups <= 0:
            raise ValueError("num_groups must be a positive integer or None.")
        if G is not None and num_groups > G.number_of_nodes():
            raise ValueError(
                "num_groups cannot be greater than the number of nodes in G."
            )

    # initial_partition: if provided, must be iterable of ints of length equal to number of nodes
    if initial_partition is not None:
        try:
            ip_list = list(initial_partition)
        except Exception:
            raise TypeError(
                "initial_partition must be an iterable of integers or None."
            )
        if G is not None and len(ip_list) != G.number_of_nodes():
            raise ValueError(
                "initial_partition length must match number of nodes in G."
            )
        for v in ip_list:
            if not _is_int(v) or v < 0:
                raise ValueError(
                    "initial_partition must contain non-negative integer labels."
                )
        if num_groups is not None:
            if max(ip_list) >= num_groups:
                raise ValueError("initial_partition contains labels >= num_groups.")
        # Make sure that each group has at least one member
        unique_groups = set(ip_list)
        if len(unique_groups) < max(ip_list) + 1:
            raise ValueError(
                "initial_partition does not contain members for every group."
            )
        if num_groups is not None and len(unique_groups) < num_groups:
            raise ValueError(
                "initial_partition has a different number of groups than num_groups."
            )

    # initial_scores: if provided, must be iterable of floats of length equal to number of nodes
    if initial_scores is not None:
        try:
            is_list = list(initial_scores)
        except Exception:
            raise TypeError("initial_scores must be an iterable of floats or None.")
        if G is not None and len(is_list) != G.number_of_nodes():
            raise ValueError("initial_scores length must match number of nodes in G.")
        for s in is_list:
            if not _is_real(s) or not math.isfinite(float(s)):
                raise ValueError("initial_scores must contain finite numeric values.")

    # seed: chain c is seeded with seed + c in a C int
    if seed is not None:
        if not _is_int(seed):
            raise TypeError("seed must be an integer or None.")
        if not 0 <= int(seed) < 2**30:
            raise ValueError(f"seed must be in [0, 2**30), got {seed}.")

    # Run lengths
    for name, val, minimum in (
        ("num_chains", num_chains, 1),
        ("samples_per_chain", samples_per_chain, 1),
        # 0 is allowed: every sample is then the starting state, which is how the
        # description length of a given state is evaluated (tests/test_wasm_parity.py).
        ("sweeps_per_sample", sweeps_per_sample, 0),
        ("thinning", thinning, 1),
        ("num_tempering_chains", num_tempering_chains, 1),
        ("burn_sweeps", burn_sweeps, 0),
        ("param_warmup_sweeps", param_warmup_sweeps, 0),
    ):
        if not _is_int(val) or val < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}, got {val!r}.")

    # beta (non-negative float)
    if not _is_real(beta) or not math.isfinite(float(beta)) or float(beta) < 0.0:
        raise ValueError("beta must be a non-negative number.")

    # no_cache (boolean; ignored)
    if not isinstance(no_cache, bool):
        raise TypeError("no_cache must be a boolean.")

    # timeout
    if not _is_real(timeout) or float(timeout) <= 0.0:
        raise ValueError("timeout must be a positive number (seconds).")

    # verbose
    if not isinstance(verbose, bool):
        raise TypeError("show_progress must be a boolean.")

    # Passes all checks
    return None


def _check_group_graph(G: nx.Graph, group_graph: nx.Graph) -> None:
    """Validate that group_graph is compatible with the primary graph G."""
    if not isinstance(group_graph, nx.Graph):
        raise TypeError("group_graph must be a NetworkX Graph instance.")
    if group_graph.is_directed():
        raise TypeError("group_graph must be undirected (nx.Graph), not directed.")
    if group_graph.number_of_edges() == 0:
        raise ValueError("group_graph has no edges.")
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
    for u, v, data in group_graph.edges(data=True):
        if u == v:
            raise ValueError(
                f"group_graph has a self-loop at node {u!r}; self-loops are not supported "
                "in two-layer mode."
            )
        _check_weight(u, v, data, integer=False, what="group_graph edge")


def _get_node_data(
    G: nx.Graph,
    *,
    group_attribute: str | None = None,
    hierarchy_attribute: str | None = None,
) -> tuple[list[int] | None, list[str] | None, list[float] | None]:
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
    node_group_name_list : list[str] or None
        List of group names that correspond to each integer, or None if no group data is present.
    node_score_list : list[float] or None
        List of hierarchy scores for each node, or None if no hierarchy data is present.
    """

    node_to_index_dict = {node: i for i, node in enumerate(G.nodes)}

    # Check whether any of the nodes have requested group/score data.
    # If any one does, they must all have that data and it should be of the correct format.
    group_data_found = False
    score_data_found = False
    node_group_list = [0] * G.number_of_nodes()  # Default group assignment if no group data is found
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
        if (
            hierarchy_attribute is not None
            and score_data_found
            and hierarchy_attribute not in data
        ):
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
        node_group_name_list = None
    if not score_data_found:
        node_score_list = None

    return node_group_list, node_group_name_list, node_score_list



def _set_model_parameters(
    groups_model: str,
    hierarchy_model: str,
    interaction: str,
    assortative: bool = _UNSET,
    mixing_variation: str = _UNSET,
    variation_in: Optional[float] = _UNSET,
    variation_out: Optional[float] = _UNSET,
    degree_correction: bool = _UNSET,
    individual_depth: Optional[float] = _UNSET,
    group_depth: Optional[float] = _UNSET,
    ties_parameter: Optional[float] = _UNSET,
    num_groups: Optional[int] = _UNSET,
) -> tuple:
    """
    Convert the model_name to corresponding non-default parameter values (if not explicitly provided)
    """
    # groups_model
    if groups_model == "traditional_SBM":
        if mixing_variation is _UNSET:
            mixing_variation = "simple"
        if assortative is _UNSET:
            assortative = False
    elif groups_model == "simple_ASBM":
        if mixing_variation is _UNSET:
            mixing_variation = "simple"
    elif groups_model == "hybrid_ASBM":
        if mixing_variation is _UNSET:
            mixing_variation = "internal"
    elif groups_model == "planted_partition":
        if mixing_variation is _UNSET:
            mixing_variation = "none"
    elif groups_model == "general_ASBM":
        if mixing_variation is _UNSET:
            mixing_variation = "general"
    # hierarchy_model
    if hierarchy_model == "bradley_terry":
        ties_parameter = 0.0
    elif hierarchy_model == "bradley_terry_ties":
        pass  # Take default or user-provided value

    # interaction
    if interaction == "coupled":
        pass  # all parameters take default values
    elif interaction == "independent":
        group_depth = 0.0

    # If mixing_variation is still _UNSET, set it to 'general' by default
    if mixing_variation is _UNSET:
        mixing_variation = "general"

    # Implement the mixing_variation presets
    if mixing_variation == "general":
        pass  # both inferred
    elif mixing_variation == "simple":
        if variation_in is _UNSET:
            variation_in = 0.5
        if variation_out is _UNSET:
            variation_out = 0.5
    elif mixing_variation == "none":
        if variation_in is _UNSET:
            variation_in = 0.0
        if variation_out is _UNSET:
            variation_out = 0.0
    elif mixing_variation == "internal":
        if variation_in is _UNSET:
            variation_in = 0.5
        if variation_out is _UNSET:
            variation_out = 0.0
    elif mixing_variation == "external":
        if variation_in is _UNSET:
            variation_in = 0.0
        if variation_out is _UNSET:
            variation_out = 0.5

    # Set remaining _UNSET parameters to defaults
    if assortative is _UNSET:
        assortative = True
    # Degree correction is a binary switch, independent of the model preset:
    # True -> 0.5 (Dirichlet alpha = 1, uniform over degree sequences), False -> 0.
    if degree_correction is _UNSET:
        degree_correction = True
    degree_correction = 0.5 if degree_correction else 0.0
    if num_groups is _UNSET:
        num_groups = None
    if variation_in is _UNSET:
        variation_in = None
    if variation_out is _UNSET:
        variation_out = None
    if individual_depth is _UNSET:
        individual_depth = None
    if group_depth is _UNSET:
        group_depth = None
    if ties_parameter is _UNSET:
        ties_parameter = None

    return (
        assortative,
        mixing_variation,
        variation_in,
        variation_out,
        degree_correction,
        individual_depth,
        group_depth,
        ties_parameter,
        num_groups,
    )


def write_gml_with_inferences(
    G: nx.Graph,
    output_filename: str,
    result: "Result",
    *,
    partition: Optional[Any] = None,
    prob: float = 0.90,
) -> None:
    """Write the graph to GML with inferred node and network attributes attached.

    Parameters
    ----------
    G : nx.Graph
        Input network (a copy is written; ``G`` is not modified).  Nodes of
        ``G`` that were not in the fitted graph get no inferred attributes.
    output_filename : str
        Filename to save the GML.
    result : Result
        Inference result from :func:`community_hierarchy.fit`.
    partition : array-like, optional
        Community labels in the fitted graph's node order, for the
        ``inferred_group`` node attribute.  Defaults to
        ``result.mdl_partition``. Pass ``result.consensus_partition()`` to
        use the consensus partition.
    prob : float
        Probability mass of the HDIs written to the graph.  Default 0.90.

    Notes
    -----
    Node attributes: ``inferred_group``, ``inferred_score`` (posterior mean)
    and ``inferred_score_std``.  Graph attributes, for each row of
    :meth:`Result.network_properties_df` (including ``ICC`` and
    ``sigma_T``): ``inferred_<parameter>_mean``, ``_std``, ``_hdi_lo`` and
    ``_hdi_hi``.
    """
    from .core import Result  # local import to avoid circular

    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a NetworkX graph.")
    if not isinstance(output_filename, str):
        raise TypeError("output_filename must be a string path.")
    if not isinstance(result, Result):
        raise TypeError("result must be a community_hierarchy.Result instance.")

    if partition is None:
        partition = result.mdl_partition
    scores = result.score_means
    score_stds = result.score_stds
    network_properties_df = result.network_properties_df(prob=prob)

    G_copy = G.copy()
    node_order = list(result._node_order)
    for idx, node in enumerate(node_order):
        if node in G_copy.nodes:
            G_copy.nodes[node]["inferred_group"] = int(partition[idx])
            G_copy.nodes[node]["inferred_score"] = float(scores[idx])
            G_copy.nodes[node]["inferred_score_std"] = float(score_stds[idx])

    for _, row in network_properties_df.iterrows():
        for col in ("mean", "std", "hdi_lo", "hdi_hi"):
            G_copy.graph[f"inferred_{row['parameter']}_{col}"] = float(row[col])

    _sanitize_graph_attributes_for_gml(G_copy)
    nx.write_gml(G_copy, output_filename)
    


def get_neighbors(G: nx.DiGraph | nx.Graph, node_order=None) -> Iterable[dict]:
    """
    Return the out-neighbors in a directed graph or neighbors in an undirected graph in formats that can be used by the MCMC algorithm
    neighbors_list[i] is a dictionary where the keys are the indices of the neighbors of node i
    and the values are the edge weights.

    Edge weights are stored in the 'weight' attribute; if not present, default to 1.0.
    Parallel edges of a multigraph are summed.

    Parameters
    ----------
    G : nx.DiGraph | nx.Graph
        NetworkX graph to get neighbors for.
    node_order : list, optional
        Explicit node ordering to use for index assignment. When provided,
        node_order[i] maps to integer index i. Required when G is a secondary
        graph (e.g. group_graph) that must share the same node-to-index mapping
        as the primary directed graph.
    """
    if node_order is not None:
        node_to_index_dict = {node: i for i, node in enumerate(node_order)}
    else:
        node_to_index_dict = {node: i for i, node in enumerate(G.nodes)}
    
    neighbors_list = []
    for i in range(len(node_to_index_dict)):
        neighbors_list.append({})
    
    # Iterate over all edges and extract weights
    for u, v, data in G.edges(data=True):
        u_idx = node_to_index_dict[u]
        v_idx = node_to_index_dict[v]
        
        # Get weight from edge data, default to 1.0
        weight = float(data.get('weight', 1.0))
        
        # Parallel edges (multigraphs) are summed.
        neighbors_list[u_idx][v_idx] = neighbors_list[u_idx].get(v_idx, 0.0) + weight

        # For undirected graphs, also add the reverse direction
        if not G.is_directed() and u_idx != v_idx:
            neighbors_list[v_idx][u_idx] = neighbors_list[v_idx].get(u_idx, 0.0) + weight

    return neighbors_list


def plot_node_properties(
    result,
    *,
    partition: str = "auto",
    consensus_method: str = "moc",
    ax=None,
    palette=None,
    xlabel: str = "Community (inferred)",
    ylabel: str = r"$s_i$ (status score)",
    group_labels=None,
    labels: str = "ticks",
    flip: bool = False,
    annotate=None,
    filled=None,
    group_gap: float = 5.0,
    ladder_step=None,
    annotation_fontsize: float = 6.0,
    output_filename: Optional[str] = None,
):
    """Plot posterior node scores ranked within each inferred community.

    Each node is a dot at its rank within its community, with a vertical error bar
    of +-1 posterior standard deviation; a horizontal line marks each community's
    mean score.  Communities run left to right in ascending order of mean score,
    and each takes as much width as it has nodes.

    Parameters
    ----------
    result : Result or pandas.DataFrame
        A fit from :func:`community_hierarchy.fit`, or a table in the format of
        :meth:`Result.node_properties_df` (columns ``node``, ``score_mean``,
        ``score_std``, ``group``), e.g. one saved to CSV.
    partition, consensus_method : str
        For a ``Result``: passed to :meth:`Result.node_properties_df`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. If *None* a new figure is created.
    palette : list, dict, or colormap, optional
        Community colours: a list in ascending mean-score order, a dict
        ``{group: colour}``, or a colormap.  Defaults to ``tab10`` (``tab20`` for
        more than 10 communities).
    xlabel, ylabel : str
        Axis labels (``xlabel`` is not shown with ``labels="ladder"``).
    group_labels : list or dict, optional
        Community names: a list in ascending mean-score order or a dict
        ``{group: name}``.  Defaults to "G1", "G2", ...
    labels : {"ticks", "ladder", "none"}
        Where community names go: as x-axis ticks (default), as coloured labels
        below the data, rising one rung per community from left to right so that
        neighbouring names never collide, or nowhere (communities are told apart
        by colour only).
    flip : bool
        Plot minus the scores, so that low rank is at the top.  Not needed
        to correct edge direction: pass ``source_high_rank=False`` to
        :func:`community_hierarchy.fit` instead, after which higher scores
        always mean higher rank.  Default False.
    annotate : dict, optional
        ``{node: text}``: rotated labels above these nodes' error bars.
    filled : collection, optional
        Nodes drawn as filled markers; the rest are hollow.  *None*: all filled.
    group_gap : float
        Gap between communities, in units of the spacing between nodes.
    ladder_step : float, optional
        Rise of the label ladder per community, in score units.  Defaults to
        3.5% of the score range.
    annotation_fontsize : float
        Font size of the ``annotate`` labels.
    output_filename : str, optional
        If given, save the figure to this path.

    Returns
    -------
    fig, ax : tuple
    """
    import numpy as np
    import pandas as pd
    import matplotlib.patheffects as pe
    from .core import Result  # local import to avoid circular

    if isinstance(result, Result):
        node_df = result.node_properties_df(partition=partition, consensus_method=consensus_method)
    elif isinstance(result, pd.DataFrame):
        node_df = result
    else:
        raise TypeError("result must be a community_hierarchy.Result or a node_properties_df() table.")
    if labels not in ("ticks", "ladder", "none"):
        raise ValueError(f"labels must be 'ticks', 'ladder' or 'none', got {labels!r}")
    df = node_df[["node", "score_mean", "score_std", "group"]].copy()
    if flip:
        df["score_mean"] = -df["score_mean"]

    grp_order = df.groupby("group")["score_mean"].mean().sort_values().index.tolist()
    K = len(grp_order)

    if palette is None:   # tab10, or tab20 for more than 10 communities
        cmap_tab, n_col = (plt.get_cmap("tab10"), 10) if K <= 10 else (plt.get_cmap("tab20"), 20)
        color_of = {g: cmap_tab(i % n_col) for i, g in enumerate(grp_order)}
    elif isinstance(palette, dict):
        color_of = {g: palette.get(g, "#9e9e9e") for g in grp_order}
    elif callable(palette):
        color_of = {g: palette(i / max(K - 1, 1)) for i, g in enumerate(grp_order)}
    else:
        color_of = {g: list(palette)[i] for i, g in enumerate(grp_order)}

    if group_labels is None:
        label_of = {g: f"G{i + 1}" for i, g in enumerate(grp_order)}
    elif isinstance(group_labels, dict):
        label_of = {g: group_labels.get(g, "") for g in grp_order}
    else:
        label_of = {g: str(group_labels[i]) for i, g in enumerate(grp_order)}

    if ax is None:
        w = min(14.0, max(8.0, len(df) * 0.12))
        fig, ax = plt.subplots(figsize=(w, w / 2))
    else:
        fig = ax.get_figure()
    filled = None if filled is None else set(filled)
    annotate = annotate or {}

    pos, centres, floor = 0.0, {}, np.inf
    for g in grp_order:
        grp = df[df["group"] == g].sort_values("score_mean")
        n_g = len(grp)
        xs = pos + np.arange(n_g)
        c = color_of[g]
        alpha, lw = max(0.25, 0.6 - 0.005 * n_g), max(0.3, 0.6 - 0.004 * n_g)
        ax.errorbar(xs, grp["score_mean"], yerr=grp["score_std"], fmt="none",
                    color=c, elinewidth=lw, capsize=0, alpha=alpha, zorder=2)
        full = np.array([filled is None or nd in filled for nd in grp["node"]])
        for sel, face in ((full, c), (~full, "none")):
            ax.scatter(xs[sel], grp["score_mean"].values[sel], s=10, facecolors=face,
                       edgecolors=c, linewidths=0.6, alpha=min(alpha + 0.3, 1.0), zorder=3)
        ax.hlines(grp["score_mean"].mean(), xs[0] - 0.4, xs[-1] + 0.4,
                  colors=c, linewidth=1.6, zorder=4)
        for x, (_, row) in zip(xs, grp.iterrows()):
            if row["node"] in annotate:
                ax.annotate(str(annotate[row["node"]]), xy=(x, row["score_mean"] + row["score_std"]),
                            xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                            rotation=90, fontsize=annotation_fontsize, color="#222222", zorder=5,
                            path_effects=[pe.withStroke(linewidth=1.0, foreground="white")])
        centres[g] = float(xs.mean())
        floor = min(floor, float((grp["score_mean"] - grp["score_std"]).min()))
        pos += n_g - 1 + group_gap

    ax.axhline(0, color="#aaaaaa", lw=0.7, alpha=0.6, zorder=1)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-group_gap / 2, pos - group_gap / 2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if labels == "ticks":
        ax.set_xticks([centres[g] for g in grp_order])
        ax.set_xticklabels([label_of[g] for g in grp_order])
        ax.set_xlabel(xlabel)
    elif labels == "none":
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
    else:
        ylo, yhi = ax.get_ylim()
        step = ladder_step if ladder_step is not None else 0.035 * (yhi - ylo)
        y0 = floor - step
        for i, g in enumerate(grp_order):
            if label_of[g]:
                ax.text(centres[g], y0 + i * step, label_of[g], ha="center", va="top",
                        fontsize=7, fontweight="bold", color=color_of[g], clip_on=False)
        ax.set_ylim(min(ylo, y0 - 2 * step), yhi)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)

    if output_filename is not None:
        fig.savefig(output_filename)

    return fig, ax


def plot_adjacency_matrix(
    result: "Result",
    G: nx.Graph,
    *,
    partition: str = "auto",
    consensus_method: str = "moc",
    cmap=None,
    ax=None,
    title: str = "",
    strip_width: float = 0.06,
    palette=None,
    vmax_percentile: float = 98.0,
    output_filename: Optional[str] = None,
):
    """Plot the adjacency matrix of *G* reordered by inferred community.

    Nodes are sorted first by ascending community mean score, then by
    ascending individual score within each community. Colored strips on
    the top and left edges of the matrix indicate community membership.
    Vertical and horizontal lines mark community boundaries.

    Parameters
    ----------
    result : Result
        Inference result from :func:`community_hierarchy.fit`. Provides the
        node ordering and community labels; does not need to be the result
        of fitting *G* specifically (e.g. you can pass a grooming graph
        while ``result`` was fitted on an aggression graph).
    G : nx.Graph
        Graph whose adjacency to display. Edge weights are taken from the
        ``weight`` attribute; missing weights default to 1.0.
    partition : str
        Which partition to use; passed to :meth:`Result.node_properties_df`.
    consensus_method : {"moc", "moc_modal_k", "binder"}
        Consensus method; passed to :meth:`Result.node_properties_df`.
    cmap : str or colormap, optional
        Colormap for the matrix cells. Defaults to ``"Greens"``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. If *None* a new figure is created.
    title : str
        Title text shown above the matrix (on the top color strip).
    strip_width : float
        Width of the community color strip as a fraction of the matrix
        axis size. Default 0.06.
    palette : list of colors or colormap, optional
        Colors for each community in ascending mean-score order.
        Defaults to ``tab10``.
    vmax_percentile : float
        Percentile of positive edge weights used as the color scale maximum.
        Default 98.
    output_filename : str, optional
        If given, save the figure to this path.

    Returns
    -------
    fig, ax : tuple
    """
    import numpy as np
    import matplotlib.patches as mpatches
    from .core import Result  # local import to avoid circular

    if not isinstance(result, Result):
        raise TypeError("result must be a community_hierarchy.Result instance.")
    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a NetworkX graph.")

    node_df = result.node_properties_df(partition=partition, consensus_method=consensus_method)

    # Sort communities by ascending group mean score, nodes within by score
    grp_mean = node_df.groupby("group")["group_score_mean"].first().sort_values()
    grp_order = grp_mean.index.tolist()
    K = len(grp_order)

    sorted_df = (
        node_df.assign(_grp_rank=node_df["group"].map(
            {g: i for i, g in enumerate(grp_order)}
        ))
        .sort_values(["_grp_rank", "score_mean"])
        .reset_index(drop=True)
    )
    node_order = sorted_df["node"].tolist()
    grp_sizes = sorted_df.groupby("_grp_rank", sort=True).size().values
    boundaries = list(np.concatenate([[0], np.cumsum(grp_sizes)]))

    # Build palette
    if palette is None:
        cmap_tab = plt.get_cmap("tab10")
        colors = [cmap_tab(i % 10) for i in range(K)]
    elif hasattr(palette, "__call__"):
        colors = [palette(i / max(K - 1, 1)) for i in range(K)]
    else:
        colors = list(palette)

    # Build adjacency matrix
    n = len(node_order)
    idx = {nd: i for i, nd in enumerate(node_order)}
    A = np.zeros((n, n))
    for u, v, d in G.edges(data=True):
        if u in idx and v in idx:
            w = d.get("weight", 1.0)
            A[idx[u], idx[v]] += w
            if not G.is_directed():
                A[idx[v], idx[u]] += w

    # Colormap
    if cmap is None:
        cmap = "Greens"

    # Axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.get_figure()

    vmax = float(np.percentile(A[A > 0], vmax_percentile)) if (A > 0).any() else 1.0
    ax.imshow(A, aspect="equal", cmap=cmap, vmin=0, vmax=vmax,
              interpolation="nearest", origin="upper")

    # Community grid lines
    for b in boundaries[1:-1]:
        ax.axvline(b - 0.5, color="black", lw=0.5, zorder=3)
        ax.axhline(b - 0.5, color="black", lw=0.5, zorder=3)

    # Diagonal
    ax.plot([-0.5, n - 0.5], [-0.5, n - 0.5], color="black", lw=0.4, zorder=4)
    ax.set_xticks([])
    ax.set_yticks([])

    # Color strips — finalize layout first so get_position() is accurate
    fig.canvas.draw()
    pos = ax.get_position()
    sh = strip_width * pos.height
    sw = strip_width * pos.width

    ax_top = fig.add_axes([pos.x0, pos.y1, pos.width, sh])
    ax_left = fig.add_axes([pos.x0 - sw, pos.y0, sw, pos.height])

    for k, (s, e) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        c = colors[k]
        ax_top.add_patch(mpatches.Rectangle((s, 0), e - s, 1, color=c, ec="none"))
        ax_left.add_patch(mpatches.Rectangle((0, s), 1, e - s, color=c, ec="none"))

    ax_top.set_xlim(0, n)
    ax_top.set_ylim(0, 1)
    ax_top.axis("off")
    ax_left.set_xlim(0, 1)
    ax_left.set_ylim(n, 0)
    ax_left.axis("off")

    if title:
        ax_top.text(0.5, 1.1, title, transform=ax_top.transAxes,
                    fontsize=plt.rcParams.get("font.size", 9) - 1,
                    va="bottom", ha="center")

    if output_filename is not None:
        fig.savefig(output_filename, bbox_inches="tight")

    return fig, ax


def plot_MCMC_entropy(
    result: "Result",
    *,
    ax=None,
    output_filename: Optional[str] = None,
):
    """Trace plot of the description length over the samples, one line per chain.

    The description length of a sample is ``H_A_b + H_Adir_G_A_s_nu +
    H_s_G_b_depths`` in nats (see :meth:`Result.samples_df`); a trace that
    is still drifting, or chains that sit at different levels, mean the fit
    has not converged.

    Parameters
    ----------
    result : Result
        Inference result from :func:`community_hierarchy.fit`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. If *None* a new figure is created.
    output_filename : str, optional
        If given, save the figure to this path.

    Returns
    -------
    fig, ax : tuple
    """
    from .core import Result  # local import to avoid circular

    if not isinstance(result, Result):
        raise TypeError("result must be a community_hierarchy.Result instance.")

    samples_df = result.samples_df()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()
    dl = samples_df["H_A_b"] + samples_df["H_Adir_G_A_s_nu"] + samples_df["H_s_G_b_depths"]
    chains = samples_df["chain_id"].unique() if len(samples_df) else []
    for c in chains:
        sel = (samples_df["chain_id"] == c).to_numpy()
        ax.plot(np.arange(sel.sum()), dl[sel].to_numpy(),
                lw=1.0, label=f"chain {c}" if len(chains) > 1 else None)
    if len(chains) > 1:
        ax.legend(frameon=False)
    ax.set_xlabel("Sample index within chain")
    ax.set_ylabel("Description length (nats)")
    ax.set_title("Convergence of MCMC")

    if output_filename is not None:
        fig.savefig(output_filename)

    return fig, ax
