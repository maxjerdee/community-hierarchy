# Python function to handle sampling of models. This includes implementation of parallel tempering

import networkx as nx
import numpy as np
import pandas as pd
from functools import lru_cache
from typing import List, Tuple, Dict, Any
from numpy.typing import ArrayLike
import tqdm
import time
import logging

from ._graph_cache import _get_wins_graph, _get_ties_graph
from .input_output import get_neighbors
from .MCMC_core import MCMC

logger = logging.getLogger(__name__)

def _gen_initial_partition(G_wins: nx.DiGraph, G_ties: nx.Graph, num_groups: int | None) -> List[int]:
    # Generate an initial partition for the MCMC sampler using greedy modularity maximization on the undirected version of the graph.

    # Generate a graph with edges for both wins and ties to use for the initial partition
    G_undirected = nx.Graph()
    G_undirected.add_nodes_from(G_wins.nodes())
    for u, v in G_wins.edges():
        G_undirected.add_edge(u, v)
    for u, v in G_ties.edges():
        G_undirected.add_edge(u, v)

    if num_groups is not None:
        communities = nx.community.greedy_modularity_communities(G_undirected, cutoff = num_groups, best_n = num_groups) # Forces num_groups
    else:
        communities = nx.community.greedy_modularity_communities(G_undirected)

    node_to_index_dict = {node: i for i, node in enumerate(G_undirected.nodes())}
    neighbors_labels = nx.to_dict_of_lists(G_undirected)
    neighbors_list = []
    for i in range(len(neighbors_labels)):
        neighbors_list.append({})

    # Convert the returned communities into a partition list
    partition = [0] * G_undirected.number_of_nodes()
    for i, community in enumerate(communities):
        for node in community:
            partition[node_to_index_dict[node]] = i
    
    return partition

def _gen_initial_scores(G_wins: nx.DiGraph, G_ties: nx.Graph) -> List[float]:
    # Generate initial scores for the MCMC sampler using win percentages. This is a simple heuristic that can help the sampler converge faster.

    node_to_index_dict = {node: i for i, node in enumerate(G_wins.nodes())}
    neighbors_labels = nx.to_dict_of_lists(G_wins)
    neighbors_list = []
    for i in range(len(neighbors_labels)):
        neighbors_list.append({})

    scores = [0.0] * G_wins.number_of_nodes()
    for node in G_wins.nodes():
        wins = G_wins.out_degree(node)
        losses = G_wins.in_degree(node)
        ties = G_ties.degree(node)
        scores[node_to_index_dict[node]] = (wins - losses) / (wins + losses + ties + 1e-6) # Add small constant to avoid division by zero
    
    return scores

@lru_cache(maxsize=16)
def _samples_cached(
    G_wins_key: str,
    G_ties_key: str,
    assortative: bool,
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
    num_samples: int,
    merge_split_enabled: bool,
    beta: float,
    num_tempering_chains: int,
    no_cache: bool,
    timeout: float,
    verbose: bool,
) -> pd.DataFrame:
    """Cached implementation of samples. All arguments are chosen to be hashable.

    The function looks up the graph from the module cache and calls the c++ MCMC sampler.
    """

    G_wins = _get_wins_graph(G_wins_key)
    if G_wins is None:
        raise RuntimeError(f"Graph with key {G_wins_key} not found in internal cache.")
    G_ties = _get_ties_graph(G_ties_key)
    if G_ties is None:
        raise RuntimeError(f"Graph with key {G_ties_key} not found in internal cache.")

    # Size warning
    num_nodes = G_wins.number_of_nodes()
    if num_nodes > 1e5:
        print("Warning: Sampling for large graphs may be very slow and memory intensive.")

    # Convert graph to wins and ties neighbor lists to pass to C++
    wins = get_neighbors(G_wins)
    ties = get_neighbors(G_ties)

    # Print information on the model configuration
    if verbose:
        print(f"Model configuration: assortative={assortative}, variation_in={variation_in}, variation_out={variation_out}, degree_correction={degree_correction}, mean_degree_scaling={mean_degree_scaling}, individual_depth = {individual_depth}, group_depth = {group_depth}, ties_parameter = {ties_parameter}, num_groups={num_groups}")
        print(f"Sampling configuration: seed={seed}, num_samples={num_samples}, merge_split_enabled={merge_split_enabled}, beta={beta}, num_tempering_chains={num_tempering_chains}, no_cache={no_cache}, timeout={timeout}")

    # Convert the initial partition tuple back to a list (undo the hashable conversion)
    initial_partition = list(initial_partition_tuple) if initial_partition_tuple is not None else None

    # Convert the initial scores tuple back to a list (undo the hashable conversion)
    initial_scores = list(initial_scores_tuple) if initial_scores_tuple is not None else None

    # Define the beta values which will be used. We choose these to form a geometric series. 
    # If only one beta is used, then it is set to beta_max
    if num_tempering_chains == 1:
        betas = [beta]
    else:
        # Otherwise we use a geometric distribution and include beta = 0. This will also allow us to compute the Bayesian evidence
        betas = [0]
        beta_min = beta/(num_tempering_chains - 1) # Set the smallest beta value linearly
        log_betas = np.linspace(np.log(beta_min), np.log(beta), num_tempering_chains)
        betas += np.exp(log_betas).tolist()[1:]

    # List the variables that will be written to the samples
    variable_names = ["beta"] # First we give the beta value for this chain
    for node in G_wins.nodes():
        variable_names.append(f"{node}_group") # The group assignment for each node
    for node in G_wins.nodes():
        variable_names.append(f"{node}_score") # The inferred hierarchy score for each node
    variable_names += ["num_groups", "density_in", "density_out", "variation_in", "variation_out", "degree_correction", "mean_degree_scaling", "individual_depth", "group_depth", "ties_parameter"] # Global parameters
    variable_names += ["H_A_b", "H_b", "H_M", "H_k", "H_A_G_k_M_b", "H_Adir_G_A_s_nu", "H_s_G_b_depths"] # Decomposition of the description length
    # variable_names += ["time_H_b", "time_H_M_G_b", "time_H_k_G_M_b", "time_H_A_G_k_M_b"] # Optional extra diagonstic variables

    # Variable names excluding group assignments and beta
    variable_names_no_groups_no_beta = [name for name in variable_names if not name.endswith("_group") and not name.endswith("_score") and name != "beta"]
    group_column_names = [name for name in variable_names if name.endswith("_group")]
    score_column_names = [name for name in variable_names if name.endswith("_score")]

    # Initialize the seed and RNG
    if seed is None:
        seed = np.random.randint(0, 1000000)
    np.random.seed(seed) # Set the random seed for reproducibility

    # For each beta, initialize a MCMC state
    MCMC_states: Dict[float, MCMC] = {}
    for chain_beta in betas:
        # Generate initial partition and parameter values if they are not provided (This is occurring inside the beta loop in case we want to randomize the initial state for each chain, which can help with mixing)
        if initial_partition is None:
            initial_partition = _gen_initial_partition(G_wins, G_ties, num_groups)
        
        initial_num_groups = len(set(initial_partition))

        # Generate initial scores if not provided
        if initial_scores is None:
            initial_scores = _gen_initial_scores(G_wins, G_ties)

        # Record which parameters are fixed and which are to be sampled
        variation_in_fixed = variation_in is not None
        if not variation_in_fixed:
            variation_in = np.random.uniform(0, 1)
        variation_out_fixed = variation_out is not None
        if not variation_out_fixed:
            variation_out = np.random.uniform(0, 1)
        degree_correction_fixed = degree_correction is not None
        if not degree_correction_fixed:
            degree_correction = np.random.uniform(0, 1)
        mean_degree_scaling_fixed = mean_degree_scaling is not None
        if not mean_degree_scaling_fixed:
            mean_degree_scaling = np.random.uniform(0, 1) # TODO: Replace this with a draw from the Cauchy prior
        individual_depth_fixed = individual_depth is not None
        if not individual_depth_fixed:
            individual_depth = np.random.uniform(0, 1)
        group_depth_fixed = group_depth is not None
        if not group_depth_fixed:
            group_depth = np.random.uniform(0, 1)
        ties_parameter_fixed = ties_parameter is not None
        if not ties_parameter_fixed:
            ties_parameter = np.random.uniform(0, 1) # TODO: replace all of these with Cauchy prior draws
        num_groups_fixed = num_groups is not None
        
        # Call the MCMC constructor with initial values and fixed/free parameters
        # Constructor signature: (wins, ties, partition, scores, num_groups, num_groups_fixed, assortative,
        #                         variation_in, variation_in_fixed, variation_out, variation_out_fixed,
        #                         degree_correction, degree_correction_fixed, mean_degree_scaling, mean_degree_scaling_fixed, 
        #                         individual_depth, individual_depth_fixed, group_depth, group_depth_fixed,
        #                         ties_parameter, ties_parameter_fixed, seed)
        MCMC_states[chain_beta] = MCMC(
            wins,
            ties, 
            initial_partition, 
            initial_scores,
            int(initial_num_groups), 
            bool(num_groups_fixed), 
            bool(assortative), 
            float(variation_in), 
            bool(variation_in_fixed), 
            float(variation_out), 
            bool(variation_out_fixed), 
            float(degree_correction), 
            bool(degree_correction_fixed), 
            float(mean_degree_scaling), 
            bool(mean_degree_scaling_fixed),
            float(individual_depth),
            bool(individual_depth_fixed),
            float(group_depth),
            bool(group_depth_fixed),
            float(ties_parameter),
            bool(ties_parameter_fixed),
            int(seed if seed is not None else 0)
        )
    
    # Initialze the dataframe to hold all samples across all betas
    samples_df = pd.DataFrame(columns=variable_names)

    # Initialize the progress bar
    progress = tqdm.tqdm(range(num_samples))
    start_time = time.time()  # Start time for timeout
    # TODO: include a burn-in period for the MCMC sampling
    for sample_num in progress:  # type: ignore[attr-defined]
        # Check for timeout
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            if verbose:
                logger.info("Timeout reached.")
            break
        for chain_beta in betas: # Note that we could parallelize this
            # Run a sweep through the MCMC for this beta.
            # This proposes one single site swap for each node, hyperparameter proposals, 
            # and one global merge/split if enabled.
            for t in range(10):
                MCMC_states[chain_beta].MCMC_sweep(chain_beta, merge_split_enabled)

            # Add a new row to the samples dataframe
            samples_df.at[len(samples_df), "beta"] = chain_beta

            # Record the state variables for this beta
            for variable_name in variable_names_no_groups_no_beta:
                samples_df.at[len(samples_df) - 1, variable_name] = getattr(MCMC_states[beta], variable_name)
            
            # Record the group assignments for each node, this is done all at once for efficiency
            samples_df.loc[len(samples_df)-1, group_column_names] = getattr(MCMC_states[beta], "partition")
            # TODO: Check if splitting the group assignments into separate columns is efficient, or if storing as a list in one column is better

            # Record the inferred scores for each node, this is done all at once for efficiency
            samples_df.loc[len(samples_df)-1, score_column_names] = getattr(MCMC_states[beta], "scores")

        # Propose thermal swaps between neighboring betas if more than one temperature is being used (parallel tempering)
        for i in range(len(betas) - 1):
            beta1 = betas[i]
            beta2 = betas[i + 1]
            delta_beta = beta2 - beta1
            delta_H_A_G_b = MCMC_states[beta2].H_A_G_b - MCMC_states[beta1].H_A_G_b
            if np.random.rand() < np.exp(delta_beta*delta_H_A_G_b):
                state1 = MCMC_states[beta1]
                MCMC_states[beta1] = MCMC_states[beta2]
                MCMC_states[beta2] = state1

        # Show progress and state of the highest beta chain
        progress.set_postfix_str(  # type: ignore[attr-defined]
            f"num_groups = {MCMC_states[beta].num_groups}, individual_depth = {MCMC_states[beta].individual_depth:.3f}, group_depth = {MCMC_states[beta].group_depth:.3f}, ties_parameter = {MCMC_states[beta].ties_parameter:.3f}"
        )

    # TODO: Check if it is more efficient to add samples to a dict or large array as they are taken then convert it to a dataframe at the end. There is currently a big difference between the c++ efficiency and the wrapped version.

    # Only report the second half of the samples to allow for burn-in, this is a common practice but could be changed to something more sophisticated in the future
    samples_df = samples_df.iloc[len(samples_df) // 2 : ].reset_index(drop=True)

    return samples_df