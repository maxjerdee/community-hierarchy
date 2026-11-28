"""The input of fit(): the graph, its edges and attributes, and the run arguments."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from community_hierarchy import Config, fit
from helpers import (
    _QUICK,
    _graph,
    _small_graph,
)


def test_fit_bad_config_type():
    G = _small_graph()
    with pytest.raises(TypeError, match="config must be an instance of Config"):
        fit("not_a_config", G)  # type: ignore[arg-type]


def test_fit_bad_graph_type():
    config = Config()
    with pytest.raises(TypeError, match="G must be a networkx"):
        fit(config, "not_a_graph")  # type: ignore[arg-type]


def test_initial_partition_wrong_length_raises():
    G = _small_graph()
    with pytest.raises(ValueError, match="initial_partition length"):
        fit(Config(), G, initial_partition=[0, 1], timeout=5.0)


# Outside weighted_SBM, weights are edge multiplicities.  Non-integer weights
# used to be truncated in the group edge counts but rounded in the degrees.
@pytest.mark.parametrize("bad", [0.5, 2.7])
def test_non_integer_weights_are_rejected(bad):
    G = _graph()
    u, v = next(iter(G.edges()))
    G[u][v]["weight"] = bad
    with pytest.raises(ValueError, match="positive integer"):
        fit(Config(), G, num_samples=2, num_warmup=1, show_progress=False)


def test_weighted_sbm_still_accepts_real_weights():
    G = _graph()
    for i, (u, v) in enumerate(G.edges()):
        G[u][v]["weight"] = 0.5 + 0.25 * (i % 3)
    result = fit(Config(weighted_SBM=True), G, num_samples=2, num_warmup=1, show_progress=False)
    assert np.isfinite(result.mdl_value)


@pytest.mark.parametrize("bad", ["2", float("nan"), float("inf"), 0, -1.0])
def test_bad_weights_are_rejected(bad):
    G = _graph()
    u, v = next(iter(G.edges()))
    G[u][v]["weight"] = bad
    with pytest.raises((TypeError, ValueError), match="weight"):
        fit(Config(weighted_SBM=True), G, **_QUICK)


# Self-loops in binary-count modes (weighted_SBM / separate group graph) would add
# one stub to e_rr instead of two; they are rejected up front.
# (The 2026-09 tempering-convention fix is verified by bit-identical output at
# beta = 1 and by the argument in src/MCMC_core.h; a statistical test of it on the
# depth parameters is not reliable because they mix too slowly.)
def test_self_loops_rejected_in_binary_count_modes():
    G = _graph()
    u = next(iter(G.nodes()))
    G.add_edge(u, u)  # untyped, like the rest of _graph()
    with pytest.raises(ValueError, match="self-loop"):
        fit(Config(weighted_SBM=True), G, num_samples=2, num_warmup=1, show_progress=False)


def test_self_loops_rejected_in_multiplicity_mode():
    G = _graph()
    G.add_edge(0, 0)
    with pytest.raises(ValueError, match="self-loop"):
        fit(Config(), G, **_QUICK)


def test_numpy_partition_labels_and_seed_are_accepted():
    G = _graph()
    part = np.zeros(G.number_of_nodes(), dtype=np.int64)
    part[::2] = 1
    r = fit(Config(), G, initial_partition=part, **{**_QUICK, "seed": np.int64(3)})
    # and a result's partition can be fed back in
    fit(Config(), G, initial_partition=r.consensus_partition(), **_QUICK)


@pytest.mark.parametrize(
    ("kwargs", "name"),
    [
        (dict(num_chains=0), "num_chains"),
        (dict(num_samples=0), "num_samples"),
        (dict(sweeps_per_sample=-1), "sweeps_per_sample"),
        (dict(num_thin=0), "num_thin"),
        (dict(num_warmup=-1), "num_warmup"),
        (dict(num_param_warmup=-1), "num_param_warmup"),
        (dict(seed=-1), "seed"),
    ],
)
def test_run_lengths_are_validated(kwargs, name):
    args = dict(_QUICK)
    args.update(kwargs)
    if args.get("num_samples") is None:
        args.pop("num_samples")
    with pytest.raises(ValueError, match=name):
        fit(Config(), _graph(), **args)


def test_source_high_rank_false_reads_edges_as_target_wins():
    """Reversing every decided edge and setting source_high_rank=False gives the
    same fit, draw for draw; draws (neutral edges) are unaffected."""
    rng = np.random.default_rng(3)
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(12))
    for _ in range(60):
        u, v = (int(x) for x in rng.choice(12, size=2, replace=False))
        if u > v:
            u, v = v, u  # lower label beats higher label, mostly
        if rng.random() < 0.2:
            u, v = v, u
        G.add_edge(u, v, type="neutral" if rng.random() < 0.15 else "dominant")
    R = nx.MultiDiGraph()
    R.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        R.add_edge(*((u, v) if d["type"] == "neutral" else (v, u)), **d)

    kw = dict(num_samples=50, num_warmup=20, seed=7, show_progress=False)
    a = fit(Config(), G, **kw).node_properties_df()
    b = fit(Config(), R, source_high_rank=False, **kw).node_properties_df()
    np.testing.assert_allclose(a["score_mean"], b["score_mean"])
    # node 0 wins most of its matches, so it should sit near the top
    assert a.set_index("node")["score_mean"].loc[0] > a["score_mean"].median()


def test_initial_partition_in_two_layer_mode_follows_G_node_order():
    from community_hierarchy.checks import _group_graph_edges
    from community_hierarchy.MCMC_core import build_group_layer, greedy_modularity_partition

    group_graph = nx.Graph()
    group_graph.add_nodes_from([0, 4, 1, 5, 2, 6, 3, 7])  # a different order
    group_graph.add_edges_from([(i, j) for i in range(4) for j in range(i + 1, 4)])
    group_graph.add_edges_from([(i, j) for i in range(4, 8) for j in range(i + 1, 8)])
    group_graph.add_edge(3, 4)
    G = nx.DiGraph()
    G.add_nodes_from(range(8))
    group_adj = build_group_layer(8, *_group_graph_edges(G, group_graph, {i: i for i in range(8)}))
    p = greedy_modularity_partition([{}] * 8, [{}] * 8, group_adj)
    assert len(set(p[:4])) == 1 and len(set(p[4:])) == 1 and p[0] != p[4]


def test_multigraph_group_graph_sums_parallel_edges():
    from community_hierarchy.checks import _group_graph_edges
    from community_hierarchy.MCMC_core import build_group_layer

    gg = nx.MultiGraph()
    gg.add_edge(0, 1, weight=2.0)
    gg.add_edge(0, 1, weight=3.0)
    gg.add_edge(1, 2)
    G = nx.DiGraph()
    G.add_nodes_from(range(3))
    adj = build_group_layer(3, *_group_graph_edges(G, gg, {0: 0, 1: 1, 2: 2}))
    assert adj == [{1: 5.0}, {0: 5.0, 2: 1.0}, {1: 1.0}]
