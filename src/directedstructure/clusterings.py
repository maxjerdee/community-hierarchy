# Find the consensus clustering given a set of clusterings

from __future__ import annotations
from typing import List, Tuple

import networkx as nx
import pandas as pd

import numpy as np
from numpy.typing import ArrayLike

try:
    from . import MCMC_core
    HAS_CPP_BINDING = True
except ImportError:
    HAS_CPP_BINDING = False


def consensus_clustering(
    clusterings: ArrayLike, *, metric: str = "L1"
) -> ArrayLike:
    """
    From a set of clusterings, find the consensus clustering. Will return the clustering
    whose coincidence matrix is the closest to the average coincidence matrix under the provided metric.

    Parameters
    ----------
    clusterings : ArrayLike
        A 2D array of clusterings. Each row is a clustering.
    metric : str, optional
        The metric to use to compare clusterings. Defaults to 'L1'. Options: {'L1', 'L2'}.
    
    Returns
    -------
    ArrayLike
        The consensus clustering.
    """

    clusterings = np.array(clusterings, dtype=np.int32)

    # Use C++ implementation if available
    if HAS_CPP_BINDING:
        # Convert clusterings to list of lists for C++
        clusterings_list = clusterings.tolist()
        result = MCMC_core.ConsensusClustering.consensus_clustering(
            clusterings_list, metric
        )
        return np.array(result)
    
    # Fall back to Python implementation if C++ binding not available
    return _consensus_clustering_python(clusterings, metric)


def _consensus_clustering_python(
    clusterings: np.ndarray, metric: str = "L1"
) -> np.ndarray:
    """
    Python fallback implementation for consensus clustering.
    """
    # Get the number of clusterings and the number of nodes
    num_clusterings, num_nodes = clusterings.shape

    # Initialize the coincidence matrix
    aggregate_coincidence_matrix = np.zeros((num_nodes, num_nodes))

    # For each clustering, update the coincidence matrix
    for clustering in clusterings:
        aggregate_coincidence_matrix += np.subtract.outer(clustering, clustering) == 0

    # Normalize to obtain the average coincidence matrix
    aggregate_coincidence_matrix /= num_clusterings

    best_distance = np.inf
    best_clustering = None

    # For each clustering, update the best clustering if the distance is smaller
    for clustering in clusterings:
        coincidence_matrix = np.subtract.outer(clustering, clustering) == 0
        if metric == "L1":
            distance = np.sum(
                np.abs(aggregate_coincidence_matrix - coincidence_matrix)
            )
        elif metric == "L2":
            distance = np.sum(
                (aggregate_coincidence_matrix - coincidence_matrix) ** 2
            )
        if distance < best_distance:
            best_clustering = clustering
            best_distance = distance

    return best_clustering


def correlation_time(clusterings: ArrayLike):
    """
    Compute the correlation time of the clusterings, defined as the number of samples between which the average shared groups is less than twice what chance would give.
    This is useful for understanding whether the Monte Carlo has sufficiently mixed.

    Parameters
    ----------
    clusterings : ArrayLike
        A 2D array of clusterings. Each row is a clustering.

    Returns
    -------
    autocorr : List[float]
        The autocorrelation of the clusterings.
    """

    # Get the number of clusterings and the number of nodes
    num_clusterings, num_nodes = clusterings.shape

    # Initialize the autocorrelation array
    autocorr = np.zeros(num_clusterings)

    # Get the total number of nodes
    num_nodes = clusterings.shape[1]

    # For each clustering, compute the autocorrelation
    all_autocorrelations = []
    for t in range(10):
        autocorrelations = []
        for i in range(num_clusterings - t):
            group_tab = pd.crosstab(clusterings[t], clusterings[0])
            # Get the sum of the diagonal of the group table
            diag_sum = np.sum(np.diag(group_tab.values))
            # Get the expected value of the diagonal sum
            expected_value = num_nodes / max(
                len(group_tab.columns), len(group_tab.index)
            )
            autocorrelations.append(diag_sum / expected_value)
        # Get the average autocorrelation
        all_autocorrelations.append(np.mean(autocorrelations))

    return all_autocorrelations
