# Wrappers for the core functionality

from __future__ import annotations
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
import networkx as nx
import pandas as pd

from .input_output import _check_input_validity, _UNSET, _set_model_parameters, _get_node_data
from ._graph_cache import (
    _store_wins_graph,
    _get_wins_graph,
    _store_ties_graph,
    _get_ties_graph,
)
from .sample import _samples_cached
from .clusterings import consensus_clustering


def samples(
    G: nx.Graph,
    *,
    group_attribute: str | None = None,
    hierarchy_attribute: str | None = None,
    groups_model: str = "general_canonical",
    hierarchy_model: str = _UNSET,
    interaction: str = "coupled",
    assortative: bool = _UNSET,
    mixing_variation: str = _UNSET,
    variation_in: float | None = _UNSET,
    variation_out: float | None = _UNSET,
    degree_correction: float | None = _UNSET,
    mean_degree_scaling: float | None = _UNSET,
    individual_depth: float | None = _UNSET,
    group_depth: float | None = _UNSET,
    ties_parameter: float | None = _UNSET,
    num_groups: int | None = _UNSET,
    initial_partition: list[int] | None = None,
    initial_scores: list[float] | None = None,
    seed: int | None = None,
    num_samples: int = 1000,
    sweeps_per_sample: int = 10,
    merge_split_enabled: bool = True,
    beta: float = 1.0,
    num_tempering_chains: int = 1,
    no_cache: bool = False,
    timeout: float = 60.0,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Get pandas DataFrame of samples from the posterior distribution given the provided network.
    Returns cached samples if the function has been called before with the same parameters. Provide a new seed to generate independent samples.

    Parameters
    ----------

    G : nx.Graph
        NetworkX graph to find group and hierarchy structure of.
        If `group_attribute` is provided distinct values fix node groups.
        If `hierarchy_attribute` is provided numeric values fix hierarchy scores.
        Any unspecified attributes are inferred by the model.
    group_attribute : str, optional
        Node attribute name containing fixed group labels. Defaults to None, in which case
        groups are inferred rather than read from node attributes.
    hierarchy_attribute : str, optional
        Node attribute name containing fixed hierarchy scores. Defaults to None, in which case
        scores are inferred rather than read from node attributes.
    groups_model : str, optional
        Type of model to fit. Choices fix parameters to special cases of interest, but can be overridden by providing specific parameter values.
        Options and corrresponding parameter values:
            - 'general_canonical': All parameters inferred except mean_degree_scaling, which is set to 0 (canonical model) [Default]
            - 'traditional_SBM': assortative = False, mixing_variation = 'simple', degree_correction = 0, mean_degree_scaling = 0
            - 'traditional_DCSBM': assortative = False, mixing_variation = 'simple', degree_correction = 0.5, mean_degree_scaling = 0 (traditional degree-corrected stochastic block model)
            - 'traditional_GDCSBM': assortative = False, mixing_variation = 'simple', degree_correction = None, mean_degree_scaling = 0 (infer degree correction, generalized)
            - 'simple_ASBM': assortative = True, mixing_variation = 'simple', degree_correction = 0, mean_degree_scaling = 0 (assortative with simple mixing variation)
            - 'hybrid_ASBM': assortative = True, mixing_variation = 'internal', degree_correction = 0, mean_degree_scaling = 0 (assortative with internal mixing variation, Zhang and Peixoto 2020)
            - 'planted_partition': assortative = True, mixing_variation = 'none', degree_correction = 0, mean_degree_scaling = 0
            - 'general_ASBM': assortative = True, mixing_variation = 'none', degree_correction = 0, mean_degree_scaling = 0
            - 'microcanonical_SBM': assortative = False, mixing_variation = 'simple', degree_correction = 0, mean_degree_scaling = -1 (microcanonical SBM)
            - 'microcanonical_DCSBM': assortative = False, mixing_variation = 'simple', degree_correction = 0.5, mean_degree_scaling = -1 (microcanonical degree-corrected SBM)
            - 'general_unified': All parameters inferred including mean_degree_scaling (generalizes all models considered)
        Defaults to 'general_canonical'. If a different model_name is provided, the corresponding parameters are set unless explicitly overridden.
    hierarchy_model : str, optional
        Type of hierarchy model to fit. Options:
            - 'bradley_terry': Bradley-Terry model for pairwise comparisons
            - 'bradley_terry_ties': Bradley-Terry model with ties
        Defaults to 'bradley_terry_ties' with neutral interactions are present and 'bradley_terry' otherwise.
        If neutral interactions are present and 'bradley_terry' is chosen only dominant interactions are used.
    interaction : str, optional
        Type of coupling between groups and hierarchy to use. Options:
            - 'coupled': groups and hierarchy are coupled (default)
            - 'independent': groups and hierarchy are inferred independently
    assortative : bool, optional
        Whether to allow the model to consider generically assortative group structure.
        If assortative = False, fix the overall in and out group densities to be equal: rho_in = rho_out.
        Defaults to True.
    mixing_variation : str, optional
        Type of variation in mixing weights to use. Controls variation_in, variation_out parameters.
        Options:
            - 'general': infers variation_in and variation_out
            - 'simple': fixes variation_in = variation_out = 0.5
            - 'none': fixes variation_in = variation_out = 0
            - 'internal': fixes variation_in = 0.5, variation_out = 0
            - 'external': fixes variation_in = 0, variation_out = 0.5
        Defaults to 'general'.
    variation_in : float, optional
        Value of the variation_in (intra-group mixing variation) parameter. Must be from 0 to 1.
        variation_in = 0 yields no variation in internal mixing weights, as found in the planted partition model, while
        variation_in = 0.5 corresponds to typical variation as found in the traditional SBM.
        A value of None allows the parameter to vary freely. Defaults to None.
    variation_out : float, optional
        Value of the variation_out (inter-group mixing variation) parameter. Must be from 0 to 1.
        variation_out = 0 yields no variation in external mixing weights, as found in the planted partition model, while
        variation_out = 0.5 corresponds to typical variation as found in the traditional SBM.
        A value of None allows the parameter to vary freely. Defaults to None.
    degree_correction : float, optional
        Value of the degree_correction (in-group degree inequality) parameter. Must be from 0 to 1.
        A value of 0 corresponds to no degree correction and a value of 0.5 corresponds to typical degree correction.
        A value of None allows the parameter to vary freely. Defaults to None.
    mean_degree_scaling : float, optional
        Value of the mean_degree_scaling (gamma) parameter. Mean degree of nodes in a group of size n scales as n^mean_degree_scaling.
        Traditional canonical models have gamma = 0, while microcanonical models have gamma = -1.
        A value of None allows the parameter to vary freely. Defaults to 0.
    individual_depth : float, optional
        Value of the individual_depth parameter controlling the spread of individual scores within groups. Must be positive.
        A value of None allows the parameter to vary freely. Defaults to 1.0.
    group_depth : float, optional
        Value of the group_depth parameter controlling the spread of group scores. Must be non-negative.
        A value of None allows the parameter to vary freely. Defaults to 0.0, which corresponds to independent groups and hierarchy.
    ties_parameter : float, optional
        Value of the ties_parameter controlling the frequency of ties in the hierarchy model.
        A value of None allows the parameter to vary freely. Defaults to 0.0 (no ties, Bradley-Terry model).
    num_groups : int, optional
        Number of communities (q) in the graph. If None, the number of communities is inferred from the data. Defaults to None.
    initial_partition : List[int], optional
        Initial partition of the graph. If None, a modularity maximized partition is used. Defaults to None.
    initial_scores : List[float], optional
        Initial scores of the nodes for the hierarchy model. If None, win percentages are used. Defaults to None.
    seed : int, optional
        Random seed for reproducibility. Defaults to None.
    num_samples : int, optional
        Number of samples to take. Defaults to 1000.
    sweeps_per_sample : int, optional
        Number of MCMC sweeps to perform between each sample. Defaults to 10.
    merge_split_enabled: bool, optional
        Whether to use merge-split moves in the MCMC sampling. Defaults to True.
    beta : float, optional
        Inverse temperature. Defaults to 1.0.
    num_tempering_chains : int, optional
        Number of parallel tempering chains to use. Defaults to 1, no parallel tempering.
        If greater than 1, samples will include their value of beta ranging from 0 to the provided beta.
    no_cache : bool, optional
        If True, do not use cached samples. Defaults to False.
    timeout : float, optional
        Maximum time in seconds to take samples for before timing out. Defaults to 60.
    verbose : bool, optional
        Whether to print progress and debug information. Defaults to False.
    Returns
    ---------
    pd.DataFrame
        DataFrame of posterior samples. Each row is a sample, and each column is a variable.
    """

    # Check input validity
    _check_input_validity(
        G=G,
        group_attribute=group_attribute,
        hierarchy_attribute=hierarchy_attribute,
        groups_model=groups_model,
        hierarchy_model=hierarchy_model,
        interaction=interaction,
        assortative=assortative,
        mixing_variation=mixing_variation,
        variation_in=variation_in,
        variation_out=variation_out,
        degree_correction=degree_correction,
        mean_degree_scaling=mean_degree_scaling,
        individual_depth=individual_depth,
        group_depth=group_depth,
        ties_parameter=ties_parameter,
        num_groups=num_groups,
        initial_partition=initial_partition,
        initial_scores=initial_scores,
        seed=seed,
        num_samples=num_samples,
        merge_split_enabled=merge_split_enabled,
        beta=beta,
        num_tempering_chains=num_tempering_chains,
        no_cache=no_cache,
        timeout=timeout,
        verbose=verbose,
    )

    # Get node data from graph to potentially fix the group or hierarchy structure
    node_group_list, node_group_name_list, node_score_list = _get_node_data(
        G,
        group_attribute=group_attribute,
        hierarchy_attribute=hierarchy_attribute,
    )

    # Overwrite the initial_partition and initial_scores if they are provided in the node data
    # Also set whether these are fixed
    partition_fixed = False
    if node_group_list is not None:
        if initial_partition is not None:
            print(
                "Warning: initial_partition provided as an argument will be overwritten by node data."
            )
        initial_partition = node_group_list
        partition_fixed = True
    
    scores_fixed = False
    if node_score_list is not None:
        if initial_scores is not None:
            print(
                "Warning: initial_scores provided as an argument will be overwritten by node data."
            )
        initial_scores = node_score_list
        scores_fixed = True 

    # Split the graph into a directed graph of wins and an undirected graph of ties
    G_wins = nx.DiGraph()
    G_ties = nx.Graph()
    G_wins.add_nodes_from(G.nodes(data=True))
    G_ties.add_nodes_from(G.nodes(data=True))
    for u, v, d in G.edges(data=True):
        if d.get("type") == "neutral":
            if hierarchy_model != "bradley_terry":  # Bradley-Terry model drops ties
                # Add or accumulate weight for ties
                if G_ties.has_edge(u, v):
                    G_ties[u][v]["weight"] += d.get("weight", 1.0)
                else:
                    G_ties.add_edge(u, v, weight=d.get("weight", 1.0))
        else:
            # Add or accumulate weight for wins
            if G_wins.has_edge(u, v):
                G_wins[u][v]["weight"] += d.get("weight", 1.0)
            else:
                G_wins.add_edge(u, v, weight=d.get("weight", 1.0))

    if verbose:
        print(
            f"Graph has {G.number_of_nodes()} nodes, {G_wins.number_of_edges()} dominant, and {G_ties.number_of_edges()} neutral edges."
        )

    # Hash and cache the graphs
    G_wins_hash = _store_wins_graph(G_wins)
    G_ties_hash = _store_ties_graph(G_ties)

    # Normalize possibly-unhashable args
    initial_partition_tuple = (
        tuple(initial_partition) if initial_partition is not None else None
    )
    initial_scores_tuple = tuple(initial_scores) if initial_scores is not None else None

    # Set model parameters according to model_name if needed
    (
        assortative,
        mixing_variation,
        variation_in,
        variation_out,
        degree_correction,
        mean_degree_scaling,
        individual_depth,
        group_depth,
        ties_parameter,
        num_groups,
    ) = _set_model_parameters(
        groups_model,
        hierarchy_model,
        interaction,
        assortative,
        mixing_variation,
        variation_in,
        variation_out,
        degree_correction,
        mean_degree_scaling,
        individual_depth,
        group_depth,
        ties_parameter,
        num_groups,
    )

    return _samples_cached(
        G_wins_hash,
        G_ties_hash,
        assortative,
        variation_in,
        variation_out,
        degree_correction,
        mean_degree_scaling,
        individual_depth,
        group_depth,
        ties_parameter,
        num_groups,
        initial_partition_tuple,
        partition_fixed,
        initial_scores_tuple,
        scores_fixed,
        seed,
        num_samples,
        sweeps_per_sample,
        merge_split_enabled,
        beta,
        num_tempering_chains,
        no_cache,
        timeout,
        verbose,
    )

def node_properties(
    G: nx.Graph,
    consensus_clustering_metric="L1",
    samples_df: Optional[pd.DataFrame] = None,
    group_attribute: str | None = None,
    hierarchy_attribute: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Infer properties of individual nodes in the network.
    Returns a pandas DataFrame with the consensus group of each node, its score within the hierarchy, and the average status of their group.

    Parameters
    ----------
    G : nx.Graph
        NetworkX graph to find group and hierarchy structure of.
        If `group_attribute` is provided distinct values fix node groups.
        If `hierarchy_attribute` is provided numeric values fix hierarchy scores.
        Any unspecified attributes are inferred by the model.
    consensus_clustering_metric : str, optional
        Metric to use for finding consensus clustering. Options: {'L1', 'L2'}. Defaults to 'L1'.
    samples_df : pd.DataFrame, optional
        DataFrame of samples from the posterior distribution. If not provided, samples will be generated using the provided graph and kwargs.
    group_attribute : str, optional
        Node attribute name containing fixed group labels. Defaults to None.
    hierarchy_attribute : str, optional
        Node attribute name containing fixed hierarchy scores. Defaults to None.
    kwargs : optional
        Additional keyword arguments to specify the model configuration or sampling. See :func:`samples` for full options.

    """
    # Extract groups and hiearchies that have been provided
    node_group_list, node_group_name_list, node_hierarchy_list = _get_node_data(
        G,
        group_attribute=group_attribute,
        hierarchy_attribute=hierarchy_attribute,
    )

    if samples_df is None:
        samples_df = samples(
            G,
            group_attribute=group_attribute,
            hierarchy_attribute=hierarchy_attribute,
            **kwargs,
        )

    # Extract names of nodes from the graph
    node_names = list(G.nodes())

    # Extract list of clusterings from samples dataframe
    # Find all columns ending with "_group"
    group_columns = [f"{node}_group" for node in node_names]
    clusterings = []
    for col in group_columns:
        clusterings.append(samples_df[col].values)

    clusterings = np.array(clusterings).T

    # Find consensus clustering with L1 metric
    consensus_groups = consensus_clustering(clusterings, metric="L1")

    # If the node_group_name_list is not None, replace the consensus group labels with the provided names
    if node_group_name_list is not None:
        consensus_groups_named = np.empty_like(consensus_groups, dtype=object)
        for i, group in enumerate(consensus_groups):
            consensus_groups_named[i] = node_group_name_list[group]
        consensus_groups = consensus_groups_named

    # Compute the mean and standard deviation of the scores for each node across samples
    score_columns = [f"{node}_score" for node in node_names]
    scores_mean = samples_df[score_columns].mean().to_numpy()
    scores_std = samples_df[score_columns].std().to_numpy()

    # Add column that gives the average score of each group for each sample, then compute the means and std across the samples

    group_members = {group: [] for group in np.unique(consensus_groups)}
    for node, group in zip(node_names, consensus_groups):
        group_members[group].append(node)

    group_score_sample_means = {}
    group_score_means = np.zeros(len(consensus_groups))
    group_score_stds = np.zeros(len(consensus_groups))
    for group, members in group_members.items():
        member_score_columns = [f"{member}_score" for member in members]
        group_score_sample_means[group] = samples_df[member_score_columns].mean(axis=1)
        group_score_mean = group_score_sample_means[group].mean()
        group_score_std = group_score_sample_means[group].std()
        for node in members:
            group_score_means[node_names.index(node)] = group_score_mean
            group_score_stds[node_names.index(node)] = group_score_std

    node_properties_df = pd.DataFrame(
        {
            "node": node_names,
            "consensus_group": consensus_groups,
            "score_mean": scores_mean,
            "score_std": scores_std,
            "group_score_mean": group_score_means,
            "group_score_std": group_score_stds,
        }
    )

    return node_properties_df


def network_properties(
    G: nx.Graph,
    samples_df: Optional[pd.DataFrame] = None,
    group_attribute: str | None = None,
    hierarchy_attribute: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Infer properties of the overall network structure.
    Returns a pandas DataFrame with the inferred parameters and their uncertainties, including those fixed by the model specification.

    These are:
        - num_groups: number of groups in the network
        - density_in: average density of edges within groups
        - density_out: average density of edges between groups
        - variation_in: variation in the internal mixing weights of nodes within groups
        - variation_out: variation in the external mixing weights of nodes within groups
        - degree_correction: level of degree correction, or in-group degree inequality
        - mean_degree_scaling: scaling of mean degree with group size, where mean degree of nodes in a group of size n scales as n^mean_degree_scaling
        - individual_depth: spread of individual scores within groups
        - group_depth: spread of group scores
        - ties_parameter: frequency of ties in the hierarchy model

    Parameters
    ----------

    G : nx.Graph
        NetworkX graph to find group structure of.
    samples_df : pd.DataFrame, optional
        DataFrame of samples from the posterior distribution. If not provided, samples will be generated using the provided graph and kwargs.
    group_attribute : str, optional
        Node attribute name containing fixed group labels. Defaults to None.
    hierarchy_attribute : str, optional
        Node attribute name containing fixed hierarchy scores. Defaults to None.
    kwargs : optional
        Additional keyword arguments to specify the model configuration or sampling. See :func:`samples` for full options.
    """
    if samples_df is None:
        samples_df = samples(
            G,
            group_attribute=group_attribute,
            hierarchy_attribute=hierarchy_attribute,
            **kwargs,
        )

    parameter_list = [
        "num_groups",
        "density_in",
        "density_out",
        "variation_in",
        "variation_out",
        "degree_correction",
        "mean_degree_scaling",
        "individual_depth",
        "group_depth",
        "ties_parameter",
    ]

    # Calculate means and standard deviations for each parameter
    parameters_df = pd.DataFrame(
        {
            "parameter": parameter_list,
            "mean": [samples_df[param].mean() for param in parameter_list],
            "std": [samples_df[param].std() for param in parameter_list],
        }
    )

    return parameters_df

def mixing_matrix(
    G: nx.Graph,
    node_groups: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """
    Estimate the probability of connection between each pair of inferred groups in the network.
    Uses globally inferred parameters to adjust observed edge densities.
    TODO: We could fix this by actually sampling the hyperparameters with these fixed group assignments.

    Parameters
    ----------

    G : nx.Graph
        NetworkX graph to find group structure of.
    node_groups : pd.DataFrame
        DataFrame containing the node groups along which to compute the mixing matrix.
    kwargs : optional
        Additional keyword arguments to specify the model configuration or sampling. See `node_groups` and `sampling` for details.
    """
    # Check input validity
    _check_input_validity(G=G, **kwargs)

    # Warning if num_groups is provided (ignored since this is determined by node_groups)
    if "num_groups" in kwargs:
        import warnings

        warnings.warn(
            "num_groups parameter is ignored in mixing_matrix since groups are determined by node_groups.",
            UserWarning,
        )
    
    

    # Count the number of edges between each pair of groups
    # This should be implemented in C++ since it will be needed by the predictive portion anyway.

    pass
