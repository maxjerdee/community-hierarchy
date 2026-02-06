# Find the consensus clustering given a set of clusterings
# TODO: Implement this in C++ for speed.

from __future__ import annotations
from typing import List, Tuple

import networkx as nx
import pandas as pd

import numpy as np
from numpy.typing import ArrayLike


def consensus_clustering(
    clusterings: ArrayLike, *, metric: str = "L1", graph_mask: nx.Graph | None = None
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
    graph_mask : nx.Graph, optional
        If a graph is given, only the entries of the coincidence matrix corresponding to the edges of the graph will be considered. Defaults to None.
    """

    clusterings = np.array(clusterings)

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

    # If a graph is given, only consider the entries corresponding to the edges of the graph
    if graph_mask is not None:
        adj_matrix = nx.to_numpy_array(graph_mask)

    # For each clustering, update the best clustering if the distance is smaller
    for clustering in clusterings:
        coincidence_matrix = np.subtract.outer(clustering, clustering) == 0
        if metric == "L1":
            if graph_mask is not None:
                distance = np.sum(
                    np.abs(aggregate_coincidence_matrix - coincidence_matrix)
                    * adj_matrix
                )
            else:
                distance = np.sum(
                    np.abs(aggregate_coincidence_matrix - coincidence_matrix)
                )
        elif metric == "L2":
            if graph_mask is not None:
                distance = np.sum(
                    (aggregate_coincidence_matrix - coincidence_matrix) ** 2
                    * adj_matrix
                )
            else:
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
