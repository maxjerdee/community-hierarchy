"""Writing a graph with the inferred groups and scores to GML."""

from __future__ import annotations

from typing import Any, Optional

import networkx as nx

from .result import Result


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
