# Handling input and output of data

import networkx as nx
import pandas as pd
from typing import Optional, Iterable
import warnings

# Sentinel for unset parameters
_UNSET = object()


# Check that the arguments passed to the function are valid, if not raise an error
def _check_input_validity(
    G: nx.Graph,
    node_groups: Optional[pd.DataFrame] = None,
    *,
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
    merge_split_enabled: bool = True,
    beta: float = 1.0,
    num_tempering_chains: int = 1,
    no_cache: bool = False,
    timeout: float = 60.0,
    verbose: bool = False,
) -> None:
    """Validate public API inputs and raise informative errors for invalid values.

    This function checks types, ranges and basic consistency for the parameters
    documented in the public API. It intentionally performs lightweight checks
    only (no heavy computation or graph processing).
    """

    # Graph checks
    if not isinstance(G, nx.DiGraph):
        raise TypeError("G must be a NetworkX DiGraph object.")
    
    # Check whether any of the nodes have type data
    # If any one does, they must all have type data and is should be either "dominant" or "neutral"
    type_found = False
    neutral_type_found = False
    for u, v, data in G.edges(data=True):
        if 'type' in data:
            type_found = True
            if data['type'] == 'neutral':
                neutral_type_found = True
            edge_type = data['type']
            if edge_type not in ['dominant', 'neutral']:
                raise ValueError(f"Edge ({u}, {v}) has invalid type '{edge_type}'. Must be 'dominant' or 'neutral'.")

    if type_found:
        for u, v, data in G.edges(data=True):
            if 'type' not in data:
                raise ValueError(f"Edge ({u}, {v}) is missing 'type' attribute while other edges have it.")

    # groups_model
    valid_models = [
        "general_canonical",
        "general_unified",
        "traditional_SBM",
        "traditional_DCSBM",
        "traditional_GDCSBM",
        "simple_ASBM",
        "hybrid_ASBM",
        "planted_partition",
    ]
    if groups_model not in valid_models:
        raise ValueError(f"Unknown groups_model '{groups_model}'. Valid options: {valid_models}")

    # hierarchy_model
    if hierarchy_model is _UNSET:
        if neutral_type_found:
            hierarchy_model = "bradley_terry_ties" # Default when ties are present
        else:
            hierarchy_model = "bradley_terry" # Default when no ties

    # Warning, neutral edges will be ignored by bradley_terry model
    if hierarchy_model == "bradley_terry" and neutral_type_found:
        warnings.warn("Hierarchy model 'bradley_terry' does not account for neutral edges. These edges will be ignored in hierarchy inference. 'bradley_terry_ties' model will account for them.", UserWarning)

    valid_hierarchy_models = ["bradley_terry", "bradley_terry_ties"]
    if hierarchy_model not in valid_hierarchy_models:
        raise ValueError(f"Unknown hierarchy_model '{hierarchy_model}'. Valid options: {valid_hierarchy_models}")

    # interaction
    valid_interactions = ["coupled", "independent"]
    if interaction not in valid_interactions:
        raise ValueError(f"Unknown interaction '{interaction}'. Valid options: {valid_interactions}")

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
    if mean_degree_scaling is _UNSET:
        mean_degree_scaling = None
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
                raise ValueError(f"{name} must be between 0 and 1 (inclusive). Got {val}.")

    # degree_correction: None or 0 <= x <= 1
    if degree_correction is not None:
        if not isinstance(degree_correction, (int, float)):
            raise TypeError("degree_correction must be a float in [0, 1] or None.")
        if not (0.0 <= float(degree_correction) <= 1.0):
            raise ValueError("degree_correction must be between 0 and 1 (inclusive).")

    # mean_degree_scaling: None or float (no strict bounds here, but must be numeric)
    if mean_degree_scaling is not None and not isinstance(mean_degree_scaling, (int, float)):
        raise TypeError("mean_degree_scaling must be a float (or None).")

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
            raise ValueError("num_groups cannot be greater than the number of nodes in G.")

    # initial_partition: if provided, must be iterable of ints of length equal to number of nodes
    if initial_partition is not None:
        try:
            ip_list = list(initial_partition)
        except Exception:
            raise TypeError("initial_partition must be an iterable of integers or None.")
        if G is not None and len(ip_list) != G.number_of_nodes():
            raise ValueError("initial_partition length must match number of nodes in G.")
        for v in ip_list:
            if not isinstance(v, int) or v < 0:
                raise ValueError("initial_partition must contain non-negative integer labels.")
        if num_groups is not None:
            if max(ip_list) >= num_groups:
                raise ValueError("initial_partition contains labels >= num_groups.")
        # Make sure that each group has at least one member
        unique_groups = set(ip_list)
        if len(unique_groups) < max(ip_list) + 1:
            raise ValueError("initial_partition does not contain members for every group.")
        if num_groups is not None and len(unique_groups) < num_groups:
            raise ValueError("initial_partition has a different number of groups than num_groups.")

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

    # merge_split_enabled
    if not isinstance(merge_split_enabled, bool):
        raise TypeError("merge_split_enabled must be a boolean.")

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

# assortative, mixing_variation, variation_in, variation_out, degree_correction, mean_degree_scaling, individual_depth, group_depth, ties_parameter, num_groups = _set_model_parameters(groups_model, hierarchy_model, interaction, assortative, mixing_variation, variation_in, variation_out, degree_correction, mean_degree_scaling, individual_depth, group_depth, ties_parameter, num_groups)
def _set_model_parameters(groups_model: str,
                    hierarchy_model: str,
                    interaction: str,
                    assortative: bool = _UNSET,
                    mixing_variation: str = _UNSET,
                    variation_in: Optional[float] = _UNSET,
                    variation_out: Optional[float] = _UNSET,
                    degree_correction: Optional[float] = _UNSET,
                    mean_degree_scaling: Optional[float] = _UNSET,
                    individual_depth: Optional[float] = _UNSET,
                    group_depth: Optional[float] = _UNSET,
                    ties_parameter: Optional[float] = _UNSET,
                    num_groups: Optional[int] = _UNSET) -> tuple:
    """
    Convert the model_name to corresponding non-default parameter values (if not explicitly provided)
    """
    # groups_model
    if groups_model == 'general_canonical':
        pass # all parameters take default values
    elif groups_model == 'traditional_SBM':
        if mixing_variation is _UNSET:
            mixing_variation = 'simple'
        if assortative is _UNSET:
            assortative = False
        if degree_correction is _UNSET:
            degree_correction = 0.0
    elif groups_model == 'traditional_DCSBM':
        if mixing_variation is _UNSET:
            mixing_variation = 'simple'
        if assortative is _UNSET:
            assortative = False
        if degree_correction is _UNSET:
            degree_correction = 0.5
    elif groups_model == 'traditional_GDCSBM':
        if mixing_variation is _UNSET:
            mixing_variation = 'simple'
        if assortative is _UNSET:
            assortative = False
    elif groups_model == 'simple_ASBM':
        if mixing_variation is _UNSET:
            mixing_variation = 'simple'
        if degree_correction is _UNSET:
            degree_correction = 0.0
    elif groups_model == 'hybrid_ASBM':
        if mixing_variation is _UNSET:
            mixing_variation = 'internal'
        if degree_correction is _UNSET:
            degree_correction = 0.0
    elif groups_model == 'planted_partition':
        if mixing_variation is _UNSET:
            mixing_variation = 'none'
        if degree_correction is _UNSET:
            degree_correction = 0.0
    elif groups_model == 'general_ASBM':
        if mixing_variation is _UNSET:
            mixing_variation = 'general'
        if degree_correction is _UNSET:
            degree_correction = 0.0
    elif groups_model == 'microcanonical_SBM':
        if assortative is _UNSET:
            assortative = False
        if degree_correction is _UNSET:
            degree_correction = 0.0
        if mean_degree_scaling is _UNSET:
            mean_degree_scaling = -1
    elif groups_model == 'microcanonical_DCSBM':
        if assortative is _UNSET:
            assortative = False
        if degree_correction is _UNSET:
            degree_correction = 0.5
        if mean_degree_scaling is _UNSET:
            mean_degree_scaling = -1
    elif groups_model == 'general_unified':
        if mean_degree_scaling is _UNSET:
            mean_degree_scaling = None # infer
    
    # hierarchy_model
    if hierarchy_model == 'bradley_terry':
        ties_parameter = 0.0
    elif hierarchy_model == 'bradley_terry_ties':
        pass # Take default or user-provided value

    # interaction
    if interaction == 'coupled':
        pass # all parameters take default values
    elif interaction == 'independent':
        group_depth = 0.0

    # If mixing_variation is still _UNSET, set it to 'general' by default
    if mixing_variation is _UNSET:
        mixing_variation = 'general'

    # Implement the mixing_variation presets
    if mixing_variation == 'general':
        pass # both inferred
    elif mixing_variation == 'simple':
        if variation_in is _UNSET:
            variation_in = 0.5
        if variation_out is _UNSET:
            variation_out = 0.5
    elif mixing_variation == 'none':
        if variation_in is _UNSET:
            variation_in = 0.0
        if variation_out is _UNSET:
            variation_out = 0.0
    elif mixing_variation == 'internal':
        if variation_in is _UNSET:
            variation_in = 0.5
        if variation_out is _UNSET:
            variation_out = 0.0
    elif mixing_variation == 'external':
        if variation_in is _UNSET:
            variation_in = 0.0
        if variation_out is _UNSET:
            variation_out = 0.5

    # Set remaining _UNSET parameters to defaults
    if assortative is _UNSET:
        assortative = True
    if degree_correction is _UNSET:
        degree_correction = None
    if mean_degree_scaling is _UNSET:
        mean_degree_scaling = 0 # Defaults to canonical model scaling, but can be set to None to infer the best scaling from the data
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

    return (assortative, mixing_variation, variation_in, variation_out, degree_correction, mean_degree_scaling, individual_depth, group_depth, ties_parameter, num_groups)
            
# Plot the groups on the network
def plot_groups(G: nx.Graph, node_groups: pd.DataFrame, **kwargs) -> None:
    pass

def add_node_group_attribute(G: nx.Graph, node_groups: pd.DataFrame) -> None:
    """
    
    
    :param G: Description
    :type G: nx.Graph
    :param node_groups: Description
    :type node_groups: pd.DataFrame
    :return: Description
    :rtype: Graph
    """
    pass

# Write the GML with a new "group" attribute so external tools (e.g., Gephi) can color by group
def write_gml(G: nx.Graph, node_groups: pd.DataFrame, path: str) -> None:
    pass


def get_neighbors(G: nx.DiGraph | nx.Graph) -> Iterable[dict]:
    """
    Return the out-neighbors in a directed graph or neighbors in an undirected graph in formats that can be used by the MCMC algorithm
    neighbors_list[i] is a dictionary where the keys are the indices of the neighbors of node i
    and the values are the edge weights.

    Parameters
    ----------
    G : nx.DiGraph
        NetworkX graph to get neighbors for.
    """
    node_to_index_dict = {node: i for i, node in enumerate(G.nodes)}

    neighbors_labels = nx.to_dict_of_lists(G)
    neighbors_list = []
    for i in range(len(neighbors_labels)):
        neighbors_list.append({})
    
    # TODO: Allow this to handle weighted graphs and multigraphs

    for node, neighbors in neighbors_labels.items():
        for neighbor in neighbors:
            neighbors_list[node_to_index_dict[node]][node_to_index_dict[neighbor]] = 1.0
            if not G.is_directed():
                neighbors_list[node_to_index_dict[neighbor]][node_to_index_dict[node]] = 1.0

    return neighbors_list

"""
[{}, {}, {}, {8: 1.0}, {}, {}, {263: 1.0}, {}, {3: 1.0}, {11: 1.0, 230: 1.0}, {12: 1.0, 13: 1.0, 16: 1.0}, {9: 1.0, 13: 1.0, 176: 1.0}, {10: 1.0}, {10: 1.0, 11: 1.0}, {}, {}, {10: 1.0, 58: 1.0, 30: 1.0, 19: 1.0}, {321: 1.0}, {}, {16: 1.0}, {21: 1.0}, {20: 1.0}, {}, {}, {26: 1.0, 27: 1.0}, {27: 1.0}, {24: 1.0, 179: 1.0}, {24: 1.0, 25: 1.0, 141: 1.0}, {}, {}, {16: 1.0, 42: 1.0, 57: 1.0, 58: 1.0, 45: 1.0, 296: 1.0, 269: 1.0}, {32: 1.0, 34: 1.0, 35: 1.0}, {31: 1.0, 35: 1.0}, {}, {31: 1.0}, {31: 1.0, 32: 1.0}, {37: 1.0, 38: 1.0}, {36: 1.0, 79: 1.0, 80: 1.0, 82: 1.0, 84: 1.0}, {36: 1.0, 226: 1.0, 239: 1.0, 227: 1.0, 228: 1.0}, {40: 1.0}, {39: 1.0}, {42: 1.0, 43: 1.0, 47: 1.0}, {30: 1.0, 41: 1.0, 101: 1.0}, {41: 1.0, 47: 1.0}, {45: 1.0}, {30: 1.0, 44: 1.0, 86: 1.0, 269: 1.0}, {}, {41: 1.0, 43: 1.0}, {}, {51: 1.0, 52: 1.0, 54: 1.0}, {85: 1.0, 57: 1.0}, {49: 1.0, 76: 1.0, 86: 1.0, 54: 1.0}, {49: 1.0, 222: 1.0, 212: 1.0, 215: 1.0}, {}, {49: 1.0, 51: 1.0, 86: 1.0}, {}, {}, {30: 1.0, 50: 1.0, 73: 1.0, 85: 1.0}, {16: 1.0, 30: 1.0, 112: 1.0, 133: 1.0}, {}, {180: 1.0, 147: 1.0, 181: 1.0}, {}, {}, {}, {}, {}, {67: 1.0}, {66: 1.0}, {}, {70: 1.0, 72: 1.0}, {69: 1.0, 180: 1.0, 181: 1.0}, {}, {69: 1.0, 73: 1.0}, {57: 1.0, 72: 1.0}, {}, {291: 1.0}, {51: 1.0, 78: 1.0}, {}, {76: 1.0, 139: 1.0, 309: 1.0}, {37: 1.0, 80: 1.0, 82: 1.0, 96: 1.0}, {37: 1.0, 79: 1.0, 81: 1.0, 82: 1.0, 225: 1.0, 84: 1.0}, {80: 1.0}, {37: 1.0, 79: 1.0, 80: 1.0, 96: 1.0}, {}, {37: 1.0, 80: 1.0, 225: 1.0}, {50: 1.0, 57: 1.0}, {45: 1.0, 51: 1.0, 54: 1.0, 269: 1.0}, {88: 1.0}, {87: 1.0, 204: 1.0}, {92: 1.0}, {}, {}, {89: 1.0}, {95: 1.0}, {}, {93: 1.0, 136: 1.0}, {79: 1.0, 82: 1.0}, {}, {}, {}, {}, {42: 1.0}, {}, {}, {}, {}, {}, {}, {109: 1.0, 110: 1.0}, {108: 1.0}, {108: 1.0}, {133: 1.0, 292: 1.0}, {58: 1.0, 113: 1.0, 115: 1.0}, {112: 1.0}, {}, {112: 1.0, 197: 1.0}, {}, {118: 1.0}, {117: 1.0, 194: 1.0}, {}, {122: 1.0}, {122: 1.0}, {120: 1.0, 121: 1.0}, {124: 1.0}, {123: 1.0, 240: 1.0}, {241: 1.0}, {}, {}, {180: 1.0}, {130: 1.0}, {129: 1.0}, {}, {147: 1.0}, {58: 1.0, 111: 1.0, 184: 1.0, 186: 1.0}, {187: 1.0, 254: 1.0}, {184: 1.0}, {95: 1.0}, {}, {}, {78: 1.0, 266: 1.0, 267: 1.0}, {304: 1.0}, {27: 1.0}, {}, {145: 1.0}, {237: 1.0, 265: 1.0, 252: 1.0}, {143: 1.0}, {}, {60: 1.0, 132: 1.0, 203: 1.0}, {}, {}, {}, {}, {}, {}, {199: 1.0}, {}, {}, {}, {}, {}, {}, {186: 1.0}, {}, {}, {}, {}, {167: 1.0}, {166: 1.0, 176: 1.0}, {}, {229: 1.0}, {}, {}, {208: 1.0}, {}, {}, {}, {11: 1.0, 167: 1.0}, {}, {}, {26: 1.0}, {60: 1.0, 70: 1.0, 128: 1.0}, {60: 1.0, 70: 1.0}, {}, {}, {133: 1.0, 135: 1.0, 185: 1.0}, {184: 1.0}, {133: 1.0, 161: 1.0}, {134: 1.0}, {}, {}, {}, {}, {}, {276: 1.0, 225: 1.0}, {118: 1.0, 197: 1.0, 198: 1.0}, {}, {}, {115: 1.0, 194: 1.0, 198: 1.0}, {194: 1.0, 197: 1.0}, {154: 1.0}, {}, {}, {}, {147: 1.0}, {88: 1.0, 205: 1.0}, {204: 1.0}, {}, {208: 1.0}, {172: 1.0, 207: 1.0}, {}, {}, {}, {52: 1.0, 215: 1.0}, {}, {}, {52: 1.0, 212: 1.0, 222: 1.0}, {}, {}, {}, {}, {}, {}, {52: 1.0, 215: 1.0}, {}, {}, {80: 1.0, 84: 1.0, 193: 1.0, 276: 1.0}, {38: 1.0, 228: 1.0}, {38: 1.0}, {38: 1.0, 226: 1.0}, {169: 1.0}, {9: 1.0, 234: 1.0}, {}, {}, {}, {230: 1.0, 341: 1.0}, {}, {}, {144: 1.0, 238: 1.0}, {237: 1.0}, {38: 1.0, 274: 1.0}, {124: 1.0}, {125: 1.0}, {}, {}, {}, {301: 1.0}, {}, {}, {249: 1.0}, {248: 1.0}, {}, {}, {144: 1.0}, {295: 1.0, 346: 1.0}, {134: 1.0}, {}, {}, {292: 1.0}, {}, {}, {}, {}, {263: 1.0}, {6: 1.0, 262: 1.0}, {}, {144: 1.0}, {139: 1.0}, {139: 1.0}, {}, {30: 1.0, 45: 1.0, 86: 1.0}, {}, {}, {}, {}, {239: 1.0, 275: 1.0}, {274: 1.0}, {193: 1.0, 225: 1.0}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {291: 1.0}, {}, {75: 1.0, 289: 1.0}, {111: 1.0, 257: 1.0}, {}, {}, {253: 1.0}, {30: 1.0}, {}, {}, {}, {}, {245: 1.0}, {}, {}, {140: 1.0}, {}, {}, {}, {}, {78: 1.0}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {17: 1.0}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {234: 1.0}, {}, {}, {}, {}, {253: 1.0, 349: 1.0}, {}, {}, {346: 1.0}, {}, {}]

Win: 348 -> 60 (weight 1)
Tie: 349 <-> 346 (weight 1)
Win: 351 -> 269 (weight 1)

"""
