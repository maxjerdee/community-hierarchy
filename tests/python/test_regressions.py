"""Regression tests for bugs found in the 2026-09 code review."""

from __future__ import annotations

import subprocess
import sys
import warnings

import networkx as nx
import numpy as np
import pytest

from community_hierarchy import Config, fit


def _graph() -> nx.DiGraph:
    return nx.gnp_random_graph(40, 0.1, seed=1, directed=True)


def _dl_trace(seed: int = 7, **kwargs) -> np.ndarray:
    args = dict(
        num_chains=4,
        num_samples=20,
        sweeps_per_sample=2,
        num_warmup=10,
        seed=seed,
        show_progress=False,
    )
    args.update(kwargs)
    return np.asarray(fit(Config(), _graph(), **args).to_frame()["H_A_b"], dtype=float)


# ---------------------------------------------------------------------------
# All chains used to share ONE global mt19937, drawn from concurrently by the
# std::async chain threads (each MCMC constructor also reseeded it).  Results
# with num_chains > 1 could not be reproduced from the seed.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_tempering_chains", [1, 3])
def test_multichain_fit_is_reproducible(num_tempering_chains):
    a = _dl_trace(num_tempering_chains=num_tempering_chains)
    b = _dl_trace(num_tempering_chains=num_tempering_chains)
    assert np.array_equal(a, b)


def test_different_seeds_give_different_chains():
    assert not np.array_equal(_dl_trace(seed=7), _dl_trace(seed=8))


# ---------------------------------------------------------------------------
# Outside weighted_SBM, weights are edge multiplicities.  Non-integer weights
# used to be truncated in the group edge counts but rounded in the degrees.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Model names do not set degree correction; Config.degree_correction does.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["traditional_DCSBM", "traditional_GDCSBM"])
def test_degree_corrected_model_names_are_rejected(name):
    with pytest.raises(ValueError, match="degree_correction"):
        Config(groups_model=name)


def test_degree_correction_defaults_to_true():
    assert Config().degree_correction is True


@pytest.mark.parametrize(("flag", "expected"), [(True, 0.5), (False, 0.0)])
def test_degree_correction_is_a_binary_switch(flag, expected):
    """The flag alone sets degree correction, for every model preset."""
    for model in ("traditional_SBM", "general_ASBM"):
        result = fit(
            Config(groups_model=model, degree_correction=flag),
            _graph(),
            num_samples=3,
            num_warmup=1,
            show_progress=False,
        )
        assert np.all(result.to_frame()["degree_correction"] == expected), model


@pytest.mark.parametrize("value", [0.5, None, "inferred"])
def test_degree_correction_must_be_binary(value):
    with pytest.raises(ValueError, match="degree_correction"):
        Config(degree_correction=value)


# ---------------------------------------------------------------------------
# Self-loops in binary-count modes (weighted_SBM / separate group graph) would add
# one stub to e_rr instead of two; they are rejected up front.
# (The 2026-09 tempering-convention fix is verified by bit-identical output at
# beta = 1 and by the argument in src/MCMC_core.h; a statistical test of it on the
# depth parameters is not reliable because they mix too slowly.)
# ---------------------------------------------------------------------------


def test_self_loops_rejected_in_binary_count_modes():
    G = _graph()
    u = next(iter(G.nodes()))
    G.add_edge(u, u)  # untyped, like the rest of _graph()
    with pytest.raises(ValueError, match="self-loop"):
        fit(Config(weighted_SBM=True), G, num_samples=2, num_warmup=1, show_progress=False)


# ---------------------------------------------------------------------------
# Tempering replicas were each built with their own seed, but every MCMC
# constructor reseeded the one thread_local generator, so only the last
# replica's seed counted.  fit() now seeds the generator once per chain and
# builds replicas with seed -1, which must leave the stream untouched.
# ---------------------------------------------------------------------------
def test_negative_seed_does_not_reseed_the_generator():
    import community_hierarchy.MCMC_core as native

    G = nx.gnp_random_graph(12, 0.3, seed=5, directed=True)
    wins = [dict() for _ in range(12)]
    for u, v in G.edges():
        wins[u][v] = 1.0
    ties = [dict() for _ in range(12)]

    def make(seed):
        return native.MCMC(
            wins=wins,
            ties=ties,
            partition=[i % 3 for i in range(12)],
            scores=[0.0] * 12,
            num_groups=3,
            seed=seed,
            degree_correction=0.5,
            degree_correction_fixed=True,
        )

    def trace(m):
        out = []
        for _ in range(20):
            m.MCMC_sweep(1.0, 1, True)
            out.append(m.H_A_b)
        return out

    reference = trace(make(11))
    m = make(11)
    make(-1)  # must not disturb the stream m is about to use
    assert trace(m) == reference
    m = make(11)
    make(12)  # a non-negative seed does reseed it
    assert trace(m) != reference


def test_raw_mcmc_degree_correction_default_and_range():
    # The raw class used to default to degree_correction=1.0 (Dirichlet
    # concentration 0): H_k was NaN and the first sweep threw.
    import community_hierarchy.MCMC_core as native

    kw = dict(
        wins=[{1: 1.0}, {}, {}],
        ties=[{}, {}, {}],
        partition=[0, 0, 1],
        scores=[0.0, 0.0, 0.0],
        num_groups=2,
    )
    m = native.MCMC(**kw)
    assert np.isfinite(m.H_k)
    m.MCMC_sweep(1.0, 1, True)
    with pytest.raises(ValueError, match="degree_correction"):
        native.MCMC(degree_correction=1.0, **kw)


# ---------------------------------------------------------------------------
# asbm and community_hierarchy each bind a bare (un-namespaced) MCMC and
# FitResultNative.  Their RTTI names collide, and pybind11's type registry is
# process-wide, so importing both used to raise
#   ImportError: generic_type: type "MCMC" is already registered!
# py::module_local() on both classes keeps each registration inside its own
# extension module.  Either package can be imported first, so test both orders.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("first", ["asbm", "community_hierarchy"])
def test_imports_alongside_community_hierarchy(first):
    pytest.importorskip("asbm")
    second = "community_hierarchy" if first == "asbm" else "asbm"
    code = (
        f"import {first}, {second}; "
        "import asbm.core as ac, community_hierarchy.core as dc; "
        "assert ac.fit_backend is not None, 'asbm native backend failed to load'; "
        "assert dc._fit_backend is not None, 'community_hierarchy native backend failed to load'; "
        "import asbm._asbm, community_hierarchy.MCMC_core as dm; "
        "assert asbm._asbm.MCMC is not dm.MCMC"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr


# ---------------------------------------------------------------------------
# The consensus is Dahl's method at the modal number of groups, followed by a
# greedy single-node descent on the same Binder loss with that count HELD.  The
# descent closes a gap measured at up to 19% of the loss on real networks
# (package-testing/asbm/consensus_refinement) without changing the group count.
# ---------------------------------------------------------------------------
def _binder_loss(p, C):
    p = np.asarray(p)
    n = len(p)
    iu = np.triu_indices(n, 1)
    same = p[:, None] == p[None, :]
    return float(np.where(same, 1 - C, C)[iu].sum())


def test_consensus_refinement_lowers_loss_and_keeps_group_count():
    G = nx.gnp_random_graph(40, 0.12, seed=4, directed=True)
    r = fit(
        Config(),
        G,
        num_chains=2,
        num_samples=30,
        sweeps_per_sample=2,
        num_warmup=50,
        seed=3,
        show_progress=False,
    )
    C = np.asarray(r.coincidence_matrix())
    raw = r.consensus_partition("binder", refine=False)
    ref = r.consensus_partition("binder")

    assert len(set(ref)) == len(set(raw))  # group count held
    assert _binder_loss(ref, C) <= _binder_loss(raw, C) + 1e-9  # never worse
    assert list(r.consensus_partition("binder")) == list(ref)  # deterministic
    # every group is still populated
    assert min(np.bincount(np.asarray(ref))[np.bincount(np.asarray(ref)) > 0]) >= 1


# ---------------------------------------------------------------------------
# An omitted hierarchy_model was resolved from the edge types only inside
# _check_input_validity, and the unset value was what reached
# _set_model_parameters -- so ties_parameter was always fitted, even on a graph
# with no neutral edges, where ties are not an outcome the data could record.
# That inflated the scores of small tie-free networks against plain
# Bradley-Terry (chickens: ~1.6x) in independent mode.
# ---------------------------------------------------------------------------
def _ties(G: nx.DiGraph, **cfg) -> float:
    r = fit(
        Config(interaction="independent", **cfg),
        G,
        num_samples=50,
        sweeps_per_sample=2,
        num_warmup=20,
        seed=3,
        show_progress=False,
    )
    p = r.network_properties_df().set_index("parameter")["mean"]
    return float(p["ties_parameter"])


def test_no_neutral_edges_means_no_tie_parameter():
    assert _ties(_graph()) == 0.0


def test_neutral_edges_fit_the_tie_parameter():
    G = nx.DiGraph()
    for u, v in _graph().edges():
        G.add_edge(u, v, type="dominant")
    G.add_edge(0, 1, type="neutral")
    G.add_edge(2, 3, type="neutral")
    assert _ties(G) > 0.0


def test_explicit_hierarchy_model_is_respected():
    assert _ties(_graph(), hierarchy_model="bradley_terry_ties") > 0.0


# ---------------------------------------------------------------------------
# Default consensus is the maximum-overlap consensus (Peixoto 2021) with the
# number of groups free, as graph-tool's partition_overlap_center;
# "moc_modal_k" and "binder" hold the number of groups at the posterior mode.
# The thinning keeps max(min_samples, ESS of H_A_b) evenly spaced samples.
# ---------------------------------------------------------------------------
def _overlap(c, b):
    from scipy.optimize import linear_sum_assignment

    c, b = np.asarray(c), np.asarray(b)
    _, c = np.unique(c, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    M = np.zeros((c.max() + 1, b.max() + 1))
    np.add.at(M, (c, b), 1)
    r, s = linear_sum_assignment(-M)
    return M[r, s].sum()


def _small_fit():
    G = nx.gnp_random_graph(40, 0.12, seed=4, directed=True)
    return fit(
        Config(),
        G,
        num_chains=2,
        num_samples=40,
        sweeps_per_sample=2,
        num_warmup=50,
        seed=3,
        show_progress=False,
    )


def test_consensus_methods_group_counts():
    r = _small_fit()
    K = int(r.to_frame()["num_groups"].mode().min())
    moc = r.consensus_partition()  # default is unrestricted MOC
    assert list(moc) == list(r.consensus_partition("MOC"))  # case-insensitive, deterministic
    assert len(set(r.consensus_partition("moc_modal_k"))) == K
    assert len(set(r.consensus_partition("binder"))) == K
    with pytest.raises(ValueError):
        r.consensus_partition("vi")


def test_moc_overlap_no_worse_than_its_starts():
    """Each MOC ascends from its starts, so its total overlap with the samples is
    no worse than theirs: for "moc_modal_k" the first modal-K sample, for "moc"
    the one-group partition, the first sample and the "moc_modal_k" consensus."""
    r = _small_fit()
    df = r.to_frame()
    P = df[[f"{v}_group" for v in r._node_order]].values
    K = int(df["num_groups"].mode().min())
    total = lambda c: sum(_overlap(c, p) for p in P)
    fixed, free = total(r.consensus_partition("moc_modal_k")), total(r.consensus_partition())
    assert fixed >= total([p for p in P if len(set(p)) == K][0])
    assert free >= total(P[0]) and free >= total(np.zeros(P.shape[1], int))
    assert free >= fixed


def test_unrestricted_moc_recovers_planted_partition():
    """Peixoto's (and graph-tool's documentation) example: noisy copies of one
    partition, each with a node given a random label among 5.  Run through a
    fit result whose samples are those copies."""
    rng = np.random.default_rng(0)
    x = np.array([5, 5, 2, 0, 1, 0, 1, 0, 0, 0, 0])
    nat = _small_fit()._native
    _, lst = nat.__getstate__()
    template = dict(lst[0])
    samples = []
    for _ in range(200):
        y = x.copy()
        y[rng.integers(len(y))] = rng.integers(5)
        d = dict(template)
        d.update(
            partition=[int(v) for v in y],
            scores=[0.0] * len(x),
            num_groups=len(set(y)),
            H_A_b=float(rng.normal()),
        )
        samples.append(d)
    res = type(nat).__new__(type(nat))
    res.__setstate__((len(x), samples))
    c = np.asarray(res.consensus_partition("moc", True, 1000))
    assert _overlap(c, x) == len(x) and len(set(c)) == len(set(x))


def test_consensus_thinning_floor_and_ess():
    G = nx.gnp_random_graph(40, 0.12, seed=4, directed=True)
    r = fit(
        Config(),
        G,
        num_chains=2,
        num_samples=200,
        sweeps_per_sample=1,
        num_warmup=20,
        seed=5,
        show_progress=False,
    )
    T = r._native.num_samples
    assert len(r._native.consensus_sample_indices(1000)) == T  # fewer than floor: all
    assert len(r._native.consensus_sample_indices(0)) == T  # 0 disables thinning
    ess = r.partition_ess()
    m = len(r._native.consensus_sample_indices(50))
    assert m == min(T, max(50, int(np.ceil(ess))))
    idx = r._native.consensus_sample_indices(50)
    assert idx == sorted(set(idx)) and idx[0] == 0
    try:
        with warnings.catch_warnings():  # ArviZ announces its refactor on import
            warnings.simplefilter("ignore", FutureWarning)
            import arviz as az
    except ImportError:
        return
    df = r.to_frame()
    L = df.groupby("chain_id").size().min()
    x = np.stack([g["H_A_b"].values[:L] for _, g in df.groupby("chain_id")])
    assert abs(ess - float(az.ess(x, method="bulk"))) <= 0.02 * float(az.ess(x, method="bulk")) + 1


# ---------------------------------------------------------------------------
# 2026-11 audit: argument validation, silent overrides, node order, two-layer
# input handling and GML output.
# ---------------------------------------------------------------------------

_QUICK = dict(num_samples=2, num_warmup=1, sweeps_per_sample=1, show_progress=False, seed=1)


def _typed_graph() -> nx.DiGraph:
    """Two cliques joined by one edge; within each clique half the edges are draws."""
    G = nx.DiGraph()
    for block in (range(0, 6), range(6, 12)):
        for i in block:
            for j in block:
                if i < j:
                    G.add_edge(i, j, type="neutral" if (i + j) % 2 else "dominant")
    G.add_edge(0, 6, type="dominant")
    return G


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(hierarchy_model="bradley_terry", ties_parameter=0.5),
        dict(hierarchy_model="bradley_terry", ties_parameter=None),
        dict(interaction="independent", group_depth=0.7),
        dict(interaction="independent", group_depth=None),
    ],
)
def test_values_forced_by_a_preset_are_not_overwritten_silently(kwargs):
    with pytest.raises(ValueError, match="fixes"):
        Config(**kwargs)


def test_forced_values_that_agree_are_accepted():
    Config(hierarchy_model="bradley_terry", ties_parameter=0.0)
    Config(interaction="independent", group_depth=0)


def test_ties_parameter_with_auto_bradley_terry_is_rejected():
    # No neutral edges: an omitted hierarchy_model resolves to bradley_terry.
    with pytest.raises(ValueError, match="no neutral edges"):
        fit(Config(ties_parameter=0.5), _graph(), **_QUICK)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(mixing_variation="bogus"),
        dict(assortative="yes"),
        dict(weighted_SBM=1),
        dict(group_attribute=""),
    ],
)
def test_config_validates_its_fields(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_config_repr_omits_unset_fields():
    assert "object at" not in repr(Config())


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


def test_negative_ties_parameter_is_rejected():
    with pytest.raises(ValueError, match="ties_parameter"):
        fit(Config(ties_parameter=-1.0), _typed_graph(), **_QUICK)


@pytest.mark.parametrize("bad", ["2", float("nan"), float("inf"), 0, -1.0])
def test_bad_weights_are_rejected(bad):
    G = _graph()
    u, v = next(iter(G.edges()))
    G[u][v]["weight"] = bad
    with pytest.raises((TypeError, ValueError), match="weight"):
        fit(Config(weighted_SBM=True), G, **_QUICK)


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


def test_initial_partition_in_two_layer_mode_follows_G_node_order():
    from community_hierarchy.MCMC_core import build_group_layer, greedy_modularity_partition
    from community_hierarchy.input_output import _group_graph_edges

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
    from community_hierarchy.MCMC_core import build_group_layer
    from community_hierarchy.input_output import _group_graph_edges

    gg = nx.MultiGraph()
    gg.add_edge(0, 1, weight=2.0)
    gg.add_edge(0, 1, weight=3.0)
    gg.add_edge(1, 2)
    G = nx.DiGraph()
    G.add_nodes_from(range(3))
    adj = build_group_layer(3, *_group_graph_edges(G, gg, {0: 0, 1: 1, 2: 2}))
    assert adj == [{1: 5.0}, {0: 5.0, 2: 1.0}, {1: 1.0}]


def test_write_gml_uses_network_properties(tmp_path):
    r = fit(Config(), _graph(), **{**_QUICK, "num_samples": 20})
    out = tmp_path / "out.gml"
    import community_hierarchy as ch

    ch.write_gml_with_inferences(_graph(), str(out), r)  # filterwarnings=error: no deprecation
    H = nx.read_gml(out)
    for key in ("inferred_ICC_mean", "inferred_sigma_T_hdi_lo", "inferred_group_depth_hdi_hi"):
        assert key in H.graph


def test_network_properties_df_validates_prob():
    r = fit(Config(), _graph(), **_QUICK)
    for bad in (0.0, 1.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="prob"):
            r.network_properties_df(prob=bad)


def test_plot_MCMC_entropy_draws_one_line_per_chain():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import community_hierarchy as ch

    r = fit(Config(), _graph(), **{**_QUICK, "num_chains": 3, "num_samples": 5})
    fig, ax = ch.plot_MCMC_entropy(r)
    assert len(ax.get_lines()) == 3
    plt.close(fig)


def test_bradley_terry_on_neutral_edges_is_rejected():
    # Plain Bradley-Terry gives a draw probability 0, so a graph with draws has
    # likelihood 0 under it: an error, not a fit that quietly drops them.
    with pytest.raises(ValueError, match="neutral edges"):
        fit(Config(hierarchy_model="bradley_terry"), _typed_graph(), **_QUICK)
