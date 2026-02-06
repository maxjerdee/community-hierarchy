# Module for computing the posterior probability of edges (observed and unobserved) in the network
# These functions can be intensive and so wrap c++ implementations.

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
import networkx as nx
import pandas as pd

from .input_output import _check_input_validity
from .core import samples


def expected_edges(
    G: nx.Graph,
    **kwargs,
) -> pd.DataFrame:
    """
    Compute the expected number of edges between each pair of nodes in the network according to the posterior predictive.

    Parameters
    ----------
    G : nx.Graph
        NetworkX graph to compute expected edge counts for.
    **kwargs :
        Additional keyword arguments to pass to specify model and sampling configuration (see `samples` function for full list).

    Returns
    -------
    pd.DataFrame
        DataFrame containing the expected edge counts between each pair of nodes.
    """

    # Check input validity
    _check_input_validity(G, **kwargs)

    # Size warning: This function may be very slow for large graphs since it needs to process all pairs of nodes.
    num_nodes = G.number_of_nodes()
    if num_nodes > 1e4:
        print(
            "Warning: Computing expected edges for large graphs may be very slow and memory intensive."
        )

    # Get the samples for this model configuration (likely cached)
    samples_df = samples(G, **kwargs)

    # Convert the samples into a large numpy array

    # Pass these samples to the C++ function to compute expected edges

    pass


def anomalies(G: nx.Graph, **kwargs) -> pd.DataFrame:
    """
    Rank the edges of the network according to how unusual they are as assessed by the posterior predictive (anomaly detection).

    Parameters
    ----------
    G : nx.Graph
        NetworkX graph to compute anomalies for.
    **kwargs :
        Additional keyword arguments to pass to specify model and sampling configuration (see `samples` function for full list).

    Returns
    -------
    pd.DataFrame
        DataFrame containing the anomaly scores for each edge.
    """
    # Check input validity
    _check_input_validity(G, **kwargs)

    # Size warning
    num_nodes = G.number_of_nodes()
    if (
        num_nodes > 1e5
    ):  # Set the limit higher than the other predictive functions since this will only be computed on the present edges
        print(
            "Warning: Computing anomalies for large graphs may be very slow and memory intensive."
        )

    pass


def link_predictions(G: nx.Graph, **kwargs) -> pd.DataFrame:
    """
    Predict which absences of edges are most surprising according to the posterior predictive (link prediction).

    Parameters
    ----------
    G : nx.Graph
        NetworkX graph to compute link predictions for.
    **kwargs :
        Additional keyword arguments to pass to specify model and sampling configuration (see `samples` function for full list).

    Returns
    -------
    pd.DataFrame
        DataFrame containing the link prediction scores for each non-edge.
    """
    # Check input validity
    _check_input_validity(G, **kwargs)

    pass


def posterior_predictive(G_train: nx.Graph, G_test: nx.Graph, **kwargs) -> pd.DataFrame:
    """
    Compute the posterior predictive probabilities of edges in the test graph given the training graph.

    Parameters
    ----------
    G_train : nx.Graph
        Training NetworkX graph.
    G_test : nx.Graph
        Test NetworkX graph.
    **kwargs :
        Additional keyword arguments to pass to specify model and sampling configuration (see `samples` function for full list).

    Returns
    -------
    pd.DataFrame
        DataFrame containing the posterior predictive probabilities for each edge in the test graph.
    """
    pass
