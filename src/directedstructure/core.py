# Wrappers for the core functionality

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
import networkx as nx
import pandas as pd

from .input_output import _check_input_validity, _UNSET, _set_model_parameters
from ._graph_cache import _store_wins_graph, _get_wins_graph, _store_ties_graph, _get_ties_graph
from .sample import _samples_cached
from .clusterings import consensus_clustering

def samples(
    G: nx.Graph, 
    *,
    groups_model: str = 'general_canonical', 
    hierarchy_model: str = _UNSET,
    interaction: str = 'coupled',
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
        NetworkX graph to find group structure of.
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
    _check_input_validity(G=G, groups_model=groups_model, hierarchy_model=hierarchy_model, interaction=interaction, assortative=assortative, mixing_variation=mixing_variation, variation_in=variation_in, variation_out=variation_out, degree_correction=degree_correction, mean_degree_scaling=mean_degree_scaling,
        individual_depth=individual_depth, group_depth=group_depth, ties_parameter=ties_parameter,
        num_groups=num_groups, initial_partition=initial_partition, initial_scores=initial_scores, seed=seed, num_samples=num_samples, merge_split_enabled=merge_split_enabled, beta=beta, num_tempering_chains=num_tempering_chains, no_cache=no_cache, timeout=timeout, verbose=verbose)

    # Split the graph into a directed graph of wins and an undirected graph of ties
    G_wins = nx.DiGraph()
    G_ties = nx.Graph()
    G_wins.add_nodes_from(G.nodes(data=True))
    G_ties.add_nodes_from(G.nodes(data=True))
    for u, v, d in G.edges(data=True):
        if d.get('type') == 'dominant':
            G_wins.add_edge(u, v)
        elif d.get('type') == 'neutral':
            if hierarchy_model != "bradley_terry": # Bradley-Terry model drops ties
                G_ties.add_edge(u, v)

    if verbose:
        print(f"Graph has {G.number_of_nodes()} nodes, {G_wins.number_of_edges()} dominant, and {G_ties.number_of_edges()} neutral edges.")

    # Hash and cache the graphs
    G_wins_hash = _store_wins_graph(G_wins)
    G_ties_hash = _store_ties_graph(G_ties)

    # Normalize possibly-unhashable args
    initial_partition_tuple = tuple(initial_partition) if initial_partition is not None else None
    initial_scores_tuple = tuple(initial_scores) if initial_scores is not None else None

    # Set model parameters according to model_name if needed
    assortative, mixing_variation, variation_in, variation_out, degree_correction, mean_degree_scaling, individual_depth, group_depth, ties_parameter, num_groups = _set_model_parameters(groups_model, hierarchy_model, interaction, assortative, mixing_variation, variation_in, variation_out, degree_correction, mean_degree_scaling, individual_depth, group_depth, ties_parameter, num_groups)

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
        initial_scores_tuple,
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

def node_groups(
    G: nx.Graph,
    *,
    model_name: str = 'general_canonical',
    assortative: bool = _UNSET,
    mixing_variation: str = _UNSET,
    variation_in: float | None = _UNSET,
    variation_out: float | None = _UNSET,
    degree_correction: float | None = _UNSET,
    mean_degree_scaling: float | None = _UNSET,
    num_groups: int | None = _UNSET,
    verbose: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Find groups of nodes in the network using a stochastic block model with various possible configurations.

    Parameters
    ----------

    G : nx.Graph
        NetworkX graph to find group structure of.
    model_name : str, optional
        Type of model to fit. Choices fix parameters to special cases of interest, but can be overridden by providing specific parameter values.
        Options and corrresponding parameter values:
            - 'general_canonical': All parameters inferred except mean_degree_scaling, which is set to 0 (canonical model)
            - 'general_unified': All parameters inferred including mean_degree_scaling (generalizes all models considered)
            - 'traditional_SBM': assortative = False, mixing_variation = 'simple', degree_correction = 0, mean_degree_scaling = 0 
            - 'traditional_DCSBM': assortative = False, mixing_variation = 'simple', degree_correction = 0.5, mean_degree_scaling = 0 (traditional degree-corrected stochastic block model)
            - 'traditional_GDCSBM': assortative = False, mixing_variation = 'simple', degree_correction = None, mean_degree_scaling = 0 (infer degree correction, generalized)
            - 'simple_ASBM': assortative = True, mixing_variation = 'simple', degree_correction = 0, mean_degree_scaling = 0 (assortative with simple mixing variation)
            - 'hybrid_ASBM': assortative = True, mixing_variation = 'internal', degree_correction = 0, mean_degree_scaling = 0 (assortative with internal mixing variation, Zhang and Peixoto 2020)
            - 'planted_partition': assortative = True, mixing_variation = 'none', degree_correction = 0, mean_degree_scaling = 0
        Defaults to 'general_canonical'. If a different model_name is provided, the corresponding parameters are set unless explicitly overridden.
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
    num_groups : int, optional
        Number of communities (q) in the graph. If None, the number of communities is inferred from the data. Defaults to None.
    verbose : bool, optional
        Whether to print progress and debug information. Defaults to False.
    kwargs : optional
        Additional keyword arguments to specify the sampling procedure. See `sampling` for details.
    """

    # Check input validity
    _check_input_validity(G=G, model_name=model_name, assortative=assortative, mixing_variation=mixing_variation, variation_in=variation_in, variation_out=variation_out, degree_correction=degree_correction, mean_degree_scaling=mean_degree_scaling,
        num_groups=num_groups, verbose=verbose, **kwargs)

    # Get the samples for this model configuration (likely cached)
    samples_df = samples(
        G,
        model_name=model_name,
        assortative=assortative,
        mixing_variation=mixing_variation,
        variation_in=variation_in,
        variation_out=variation_out,
        degree_correction=degree_correction,
        mean_degree_scaling=mean_degree_scaling,
        num_groups=num_groups,
        **kwargs,
    )

    print(samples_df)

    # Get the clusterings returned by the samples (as an array)
    clusterings = samples_df.filter(like='group_').to_numpy()

    print(clusterings)

    # Find the consensus out of the returned clusterings
    clustering = consensus_clustering(clusterings)
    pass

def parameters(
    G: nx.Graph,
    **kwargs,
) -> pd.DataFrame:
    """
    Infer global parameters of the stochastic block model fitted to the network.
    Returns a pandas DataFrame with the inferred parameters and their values, including those fixed by the chosen model.

    Parameters
    ----------

    G : nx.Graph
        NetworkX graph to find group structure of.
    kwargs : optional
        Additional keyword arguments to specify the model configuration or sampling. See `node_groups` and `sampling` for details.
    """
    # Check input validity
    _check_input_validity(G=G, **kwargs)

    # Get the samples for this model configuration (likely cached)
    samples_df = samples(
        G,
        **kwargs,
    )

    # Get the means and standard deviations of each of the inferred global parameters

    pass

def mixing_matrix(
    G: nx.Graph,
    node_groups: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """
    Estimate the probability of connection between each pair of inferred groups in the network. 
    Uses globally inferred parameters to adjust observed edge densities.
    TODO: We could fix this to actually just sample the hyperparameters with these fixed group assignments.

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
    if 'num_groups' in kwargs:
        import warnings
        warnings.warn("num_groups parameter is ignored in mixing_matrix since groups are determined by node_groups.", UserWarning)
    
    # Get the parameters for this model configuration (likely cached)
    parameters_df = parameters(
        G,
        **kwargs,
    )

    # Count the number of edges between each pair of groups
    # This should be implemented in C++ since it will be needed by the predictive portion anyway. 

    pass
