# Using the Wang-Landau algorithm to compute the Bayes factor between
# particular models.

import networkx as nx
from .input_output import _check_input_validity, _set_model_parameters


def _one_group_bayesian_evidence(G: nx.Graph, model_name: str):
    """
    Compute the bayesian evidence for a model restricted to a single group.
    """
    # This will involve integrating the general configuration model over possible concentrations.
    # It may also involve integrating over the mean_degree_scaling parameter if that is not fixed.

    # We note that the one-group model has the form P(A|k) P(k|m,alpha) P(m|rho_in, lambda_in, scaling)
    # It's unclear to me if the scaling is needed here.
    # Since we have this factorization, we can integrate over alpha separately from rho_in, lmabda_in, scaling.
    pass


def bayesian_evidence(G: nx.Graph, model_name: str, **kwargs) -> float:
    """
    Compute the Bayesian model evidence for a specified model using the Wang-Landau algorithm.

    Parameters
    ----------
    G : nx.Graph
        NetworkX graph to compute the model evidence for.
    model_name : str
        Name of the model to compute the evidence for.
    **kwargs :
        Additional keyword arguments to pass to specify model and sampling configuration (see `samples` function for full list).
    """

    # Check input validity
    _check_input_validity(G=G, model_name=model_name, **kwargs)

    # Convert to an undirected graph
    if G.is_directed():
        print("Warning: Converting directed graph to undirected graph.")
        G = G.to_undirected()

    # Set model parameters according to model_name if needed
    (
        assortative,
        mixing_variation,
        variation_in,
        variation_out,
        degree_correction,
        mean_degree_scaling,
        num_groups,
    ) = _set_model_parameters(
        model_name,
        assortative,
        mixing_variation,
        variation_in,
        variation_out,
        degree_correction,
        mean_degree_scaling,
        num_groups,
    )

    # Run a few sweeps of the model to warm up the chain and get a rough estimate of the distribution of the number of groups

    # Set the num_groups_cutoff at the largest number of groups observed during the warm-up phase plus a buffer

    # Run the Wang-Landau algorithm to compute the H(q|A) for this model

    # Instatiate the MCMC object with num_groups_fixed = True and initial_partition = all nodes in group 0

    # Start loop with step size decreasing as 1/t

    # Run a MCMC sweep

    # Propose a move to change the number of groups, and supplement the accept/reject with the current density of state estimates


def bayes_factor(G: nx.Graph, model_1: str, model_2: str, **kwargs) -> float:
    """
    Compute the Bayes factor between two specified models using the Wang-Landau algorithm.
    Numerically converges to the ratio of the model evidence of the first model to the second.

    Parameters
    ----------
    G : nx.Graph
        NetworkX graph to compute the Bayes factor for.
    model_1 : str
        Name of the first model.
    model_2 : str
        Name of the second model.
    **kwargs :
        Additional keyword arguments to pass to specify model and sampling configuration (see `samples` function for full list).
    """

    # Check input validity
    _check_input_validity(G, **kwargs)

    # Implement the Wang-Landau algorithm by performing MCMC sweeps
    # List the allowable pairs (base case), otherwise we will compute ratios of model evidences between models in order to compare them to each other.

    pass
