# Handling input and output of data

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Iterable, Any
import warnings

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


# Check that the arguments passed to the function are valid, if not raise an error
def _check_input_validity(
    G: nx.Graph,
    node_groups: Optional[pd.DataFrame] = None,
    *,
    group_attribute: Optional[str] = None,
    hierarchy_attribute: Optional[str] = None,
    groups_model: str = "general_canonical",
    hierarchy_model: str = _UNSET,
    interaction: str = "coupled",
    assortative: bool = _UNSET,
    mixing_variation: str = _UNSET,
    variation_in: Optional[float] = _UNSET,
    variation_out: Optional[float] = _UNSET,
    degree_correction: Optional[float] = _UNSET,
    mean_degree_scaling: Optional[float] = _UNSET,
    individual_depth: Optional[float] = _UNSET,
    group_depth: Optional[float] = _UNSET,
    ties_parameter: Optional[float] = _UNSET,
    num_groups: Optional[int] = _UNSET,
    initial_partition: Optional[Iterable[int]] = None,
    initial_scores: Optional[Iterable[float]] = None,
    seed: Optional[int] = None,
    num_samples: int = 1000,
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
    if group_attribute is not None and groups_model != "general_canonical":
        warnings.warn(
            f"group_attribute '{group_attribute}' was provided alongside groups_model '{groups_model}'. "
            "The group_attribute will be used for model initialization, but the specified model type will still be applied. "
            "To avoid confusion, consider using the default groups_model when providing node attributes.",
            UserWarning,
        )
    

    # Check whether any of the edges have type data
    # If any one does, they must all have type data and it should be either "dominant" or "neutral"
    type_found = False
    neutral_type_found = False
    for u, v, data in G.edges(data=True):
        if "type" in data:
            type_found = True
            if data["type"] == "neutral":
                neutral_type_found = True
            edge_type = data["type"]
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
        
    # Check that the graph is not empty
    if G.number_of_nodes() == 0:
        raise ValueError("Graph G must have at least one node.")

    # groups_model
    valid_models = [
        "general_canonical",
        "traditional_SBM",
        "traditional_DCSBM",
        "traditional_GDCSBM",
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
    if hierarchy_attribute is not None and original_hierarchy_model != _UNSET:
        warnings.warn(
            f"hierarchy_attribute '{hierarchy_attribute}' was provided alongside hierarchy_model '{original_hierarchy_model}'. "
            "The hierarchy_attribute will be used for model initialization, but the specified model type will still be applied. "
            "To avoid confusion, consider using the default hierarchy_model when providing node attributes.",
            UserWarning,
        )

    # Warning, neutral edges will be ignored by bradley_terry model
    if hierarchy_model == "bradley_terry" and neutral_type_found:
        warnings.warn(
            "Hierarchy model 'bradley_terry' does not account for neutral edges. These edges will be ignored in hierarchy inference. 'bradley_terry_ties' model will account for them.",
            UserWarning,
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
    if degree_correction is _UNSET:
        degree_correction = None
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
            if not isinstance(val, (int, float)):
                raise TypeError(f"{name} must be a float in [0, 1] or None.")
            if not (0.0 <= float(val) <= 1.0):
                raise ValueError(
                    f"{name} must be between 0 and 1 (inclusive). Got {val}."
                )

    # degree_correction: None or 0 <= x <= 1
    if degree_correction is not None:
        if not isinstance(degree_correction, (int, float)):
            raise TypeError("degree_correction must be a float in [0, 1] or None.")
        if not (0.0 <= float(degree_correction) <= 1.0):
            raise ValueError("degree_correction must be between 0 and 1 (inclusive).")

    # individual_depth: None or float (positive, but must be numeric)
    if individual_depth is not None:
        if not isinstance(individual_depth, (int, float)):
            raise TypeError("individual_depth must be a float (or None).")
        if individual_depth <= 0:
            raise ValueError("individual_depth must be positive.")

    # group_depth: None or float (non-negative, but must be numeric)
    if group_depth is not None:
        if not isinstance(group_depth, (int, float)):
            raise TypeError("group_depth must be a float (or None).")
        if group_depth < 0:
            raise ValueError("group_depth must be non-negative.")

    # num_groups must be positive integer if provided and not greater than number of nodes
    if num_groups is not None:
        if not isinstance(num_groups, int) or num_groups <= 0:
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
            if not isinstance(v, int) or v < 0:
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
            if not isinstance(s, (int, float)):
                raise ValueError("initial_scores must contain numeric values.")

    # seed
    if seed is not None and not isinstance(seed, int):
        raise TypeError("seed must be an integer or None.")

    # num_samples
    if not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError("num_samples must be a positive integer.")

    # beta (non-negative float)
    if not isinstance(beta, (int, float)) or float(beta) < 0.0:
        raise ValueError("beta must be a non-negative number.")

    # num_tempering_chains
    if not isinstance(num_tempering_chains, int) or num_tempering_chains <= 0:
        raise ValueError("num_tempering_chains must be a positive integer.")

    # no_cache (boolean)
    if not isinstance(no_cache, bool):
        raise TypeError("no_cache must be a boolean.")

    # timeout
    if not isinstance(timeout, (int, float)) or float(timeout) <= 0.0:
        raise ValueError("timeout must be a positive number (seconds).")

    # verbose
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a boolean.")

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
            msg += f" Nodes in G but not group_graph: {sorted(only_G)[:5]}{'...' if len(only_G) > 5 else ''}."
        if only_S:
            msg += f" Nodes in group_graph but not G: {sorted(only_S)[:5]}{'...' if len(only_S) > 5 else ''}."
        raise ValueError(msg)


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
    degree_correction: Optional[float] = _UNSET,
    individual_depth: Optional[float] = _UNSET,
    group_depth: Optional[float] = _UNSET,
    ties_parameter: Optional[float] = _UNSET,
    num_groups: Optional[int] = _UNSET,
) -> tuple:
    """
    Convert the model_name to corresponding non-default parameter values (if not explicitly provided)
    """
    # groups_model
    if groups_model == "general_canonical":
        pass  # all parameters take default values
    elif groups_model == "traditional_SBM":
        if mixing_variation is _UNSET:
            mixing_variation = "simple"
        if assortative is _UNSET:
            assortative = False
        if degree_correction is _UNSET:
            degree_correction = 0.0
    elif groups_model == "traditional_DCSBM":
        if mixing_variation is _UNSET:
            mixing_variation = "simple"
        if assortative is _UNSET:
            assortative = False
        if degree_correction is _UNSET:
            degree_correction = 0.5
    elif groups_model == "traditional_GDCSBM":
        if mixing_variation is _UNSET:
            mixing_variation = "simple"
        if assortative is _UNSET:
            assortative = False
    elif groups_model == "simple_ASBM":
        if mixing_variation is _UNSET:
            mixing_variation = "simple"
        if degree_correction is _UNSET:
            degree_correction = 0.0
    elif groups_model == "hybrid_ASBM":
        if mixing_variation is _UNSET:
            mixing_variation = "internal"
        if degree_correction is _UNSET:
            degree_correction = 0.0
    elif groups_model == "planted_partition":
        if mixing_variation is _UNSET:
            mixing_variation = "none"
        if degree_correction is _UNSET:
            degree_correction = 0.0
    elif groups_model == "general_ASBM":
        if mixing_variation is _UNSET:
            mixing_variation = "general"
        if degree_correction is _UNSET:
            degree_correction = 0.0
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
    if degree_correction is _UNSET:
        degree_correction = None
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
) -> None:
    """Write the graph to GML with inferred node and network attributes attached.

    Parameters
    ----------
    G : nx.Graph
        Input network.
    output_filename : str
        Filename to save the GML.
    result : Result
        Inference result from :func:`directedstructure.fit`.
    partition : array-like, optional
        Community partition for the ``inferred_group`` node attribute.
        Defaults to ``result.mdl_partition``. Pass
        ``result.consensus_partition()`` to use the consensus partition.
    """
    from .core import Result  # local import to avoid circular

    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a NetworkX graph.")
    if not isinstance(output_filename, str):
        raise TypeError("output_filename must be a string path.")
    if not isinstance(result, Result):
        raise TypeError("result must be a directedstructure.Result instance.")

    if partition is None:
        partition = result.mdl_partition
    scores = result.score_means
    network_properties_df = result.parameter_df

    G_copy = G.copy()
    node_order = list(result._node_order)
    for idx, node in enumerate(node_order):
        if node in G_copy.nodes:
            G_copy.nodes[node]["inferred_group"] = int(partition[idx])
            G_copy.nodes[node]["inferred_score"] = float(scores[idx])

    for _, row in network_properties_df.iterrows():
        G_copy.graph[f"inferred_{row['parameter']}_mean"] = row["mean"]
        G_copy.graph[f"inferred_{row['parameter']}_std"] = row["std"]

    _sanitize_graph_attributes_for_gml(G_copy)
    nx.write_gml(G_copy, output_filename)
    


def get_neighbors(G: nx.DiGraph | nx.Graph, node_order=None) -> Iterable[dict]:
    """
    Return the out-neighbors in a directed graph or neighbors in an undirected graph in formats that can be used by the MCMC algorithm
    neighbors_list[i] is a dictionary where the keys are the indices of the neighbors of node i
    and the values are the edge weights.

    Edge weights are stored in the 'weight' attribute; if not present, default to 1.0.

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
        weight = data.get('weight', 1.0)
        
        # Add to the neighbors dictionary for u -> v
        neighbors_list[u_idx][v_idx] = weight
        
        # For undirected graphs, also add the reverse direction
        if not G.is_directed():
            neighbors_list[v_idx][u_idx] = weight

    return neighbors_list


def plot_node_properties(
    result: "Result",
    *,
    partition: str = "auto",
    ax=None,
    palette=None,
    xlabel: str = "Community (inferred)",
    ylabel: str = r"$s_i$ (status score)",
    group_labels=None,
    output_filename: Optional[str] = None,
):
    """Plot posterior node scores ranked within each inferred community.

    Each node is plotted as a dot at its rank within its community along the
    x-axis, with a vertical error bar of ±1 posterior standard deviation. A
    horizontal line marks the group mean score. Communities are labelled G1,
    G2, … in ascending order of their mean score.

    Parameters
    ----------
    result : Result
        Inference result from :func:`directedstructure.fit`.
    partition : str
        Which partition to use; passed to :meth:`Result.node_properties_df`.
        ``"auto"`` (default) uses consensus for n ≤ 3000, MDL otherwise.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. If *None* a new figure is created.
    palette : list of colors or colormap, optional
        Colors for each community in ascending mean-score order.
        Defaults to the ``tab10`` colormap.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    group_labels : list or dict, optional
        Override the automatic "G1", "G2", … tick labels.
        Pass a list (one entry per community, in ascending mean-score order)
        or a dict mapping community integer ids to label strings.
    output_filename : str, optional
        If given, save the figure to this path.

    Returns
    -------
    fig, ax : tuple
    """
    import numpy as np
    from .core import Result  # local import to avoid circular

    if not isinstance(result, Result):
        raise TypeError("result must be a directedstructure.Result instance.")

    node_df = result.node_properties_df(partition=partition)

    # Sort communities by ascending group mean score
    grp_mean = node_df.groupby("group")["group_score_mean"].first().sort_values()
    grp_order = grp_mean.index.tolist()
    K = len(grp_order)

    # Build palette
    if palette is None:
        cmap_tab = plt.get_cmap("tab10")
        colors = [cmap_tab(i % 10) for i in range(K)]
    elif hasattr(palette, "__call__"):
        colors = [palette(i / max(K - 1, 1)) for i in range(K)]
    else:
        colors = list(palette)

    color_of = {g: colors[i] for i, g in enumerate(grp_order)}

    # Auto group label mapping
    if group_labels is None:
        label_of = {g: f"G{i + 1}" for i, g in enumerate(grp_order)}
    elif isinstance(group_labels, dict):
        label_of = group_labels
    else:
        label_of = {g: str(group_labels[i]) for i, g in enumerate(grp_order)}

    SCALE = 8.0
    GAP = SCALE * 5.0

    # Assign x-positions: nodes spread within group, gap between groups
    starts = {}
    pos = 0.0
    for g in grp_order:
        starts[g] = pos
        n_g = int((node_df["group"] == g).sum())
        pos += (n_g - 1) * SCALE + GAP

    if ax is None:
        n_nodes = len(node_df)
        w = min(14.0, max(8.0, n_nodes * 0.12))
        fig, ax = plt.subplots(figsize=(w, w / 2))
    else:
        fig = ax.get_figure()

    xtick_pos, xtick_labels = [], []
    for g in grp_order:
        grp = (
            node_df[node_df["group"] == g]
            .sort_values("score_mean")
            .reset_index(drop=True)
        )
        n_g = len(grp)
        xs = starts[g] + np.arange(n_g) * SCALE
        c = color_of[g]
        for k, row in grp.iterrows():
            x = xs[k]
            ax.plot(
                [x, x],
                [row["score_mean"] - row["score_std"],
                 row["score_mean"] + row["score_std"]],
                color=c, lw=0.5, alpha=0.55, zorder=2,
            )
            ax.plot(
                x, row["score_mean"], "o",
                color=c, ms=3.0, alpha=0.85, zorder=3, markeredgewidth=0,
            )
        ax.hlines(
            grp["score_mean"].mean(),
            xs[0] - 0.4 * SCALE, xs[-1] + 0.4 * SCALE,
            colors=c, linewidth=1.2, alpha=0.6, zorder=4,
        )
        xtick_pos.append(float(xs.mean()))
        xtick_labels.append(label_of[g])

    ax.axhline(0, color="#aaaaaa", lw=0.7, ls="-", alpha=0.6, zorder=1)
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-GAP / 2, pos - GAP * 1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if output_filename is not None:
        fig.savefig(output_filename)

    return fig, ax


def plot_adjacency_matrix(
    result: "Result",
    G: nx.Graph,
    *,
    partition: str = "auto",
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
        Inference result from :func:`directedstructure.fit`. Provides the
        node ordering and community labels; does not need to be the result
        of fitting *G* specifically (e.g. you can pass a grooming graph
        while ``result`` was fitted on an aggression graph).
    G : nx.Graph
        Graph whose adjacency to display. Edge weights are taken from the
        ``weight`` attribute; missing weights default to 1.0.
    partition : str
        Which partition to use; passed to :meth:`Result.node_properties_df`.
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
        raise TypeError("result must be a directedstructure.Result instance.")
    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a NetworkX graph.")

    node_df = result.node_properties_df(partition=partition)

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
    output_filename: Optional[str] = None,
):
    """Plot total entropy over MCMC samples to inspect convergence.

    Parameters
    ----------
    result : Result
        Inference result from :func:`directedstructure.fit`.
    output_filename : str, optional
        If given, save the figure to this path.
    """
    from .core import Result  # local import to avoid circular

    if not isinstance(result, Result):
        raise TypeError("result must be a directedstructure.Result instance.")

    samples_df = result.samples_df()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        samples_df["H_A_b"]
        + samples_df["H_Adir_G_A_s_nu"]
        + samples_df["H_s_G_b_depths"]
    )
    ax.set_xlabel("Sample index")
    ax.set_ylabel("H_A_b + H_Adir_G_A_s_nu + H_s_G_b_depths")
    ax.set_title("Convergence of MCMC")

    if output_filename is not None:
        fig.savefig(output_filename)

    return fig, ax

