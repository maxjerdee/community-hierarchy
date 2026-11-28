"""Result: the draws, the summaries, the partition estimates, plots and GML output."""

from __future__ import annotations

import warnings

import community_hierarchy as ch
import networkx as nx
import numpy as np
import pytest
from community_hierarchy import Config, fit
from helpers import (
    _QUICK,
    _graph,
    _small_graph,
)


def test_result_to_frame_shape(small_result):
    result = small_result
    G = _small_graph()
    df = result.to_frame()
    assert len(df) > 0
    for node in G.nodes():
        assert f"{node}_group" in df.columns
        assert f"{node}_score" in df.columns
    for param in (
        "num_groups",
        "density_in",
        "density_out",
        "H_A_b",
        "H_Adir_G_A_s_nu",
        "H_s_G_b_depths",
    ):
        assert param in df.columns


def test_result_getitem_shapes(small_result):
    result = small_result
    n, d = _small_graph().number_of_nodes(), len(result.to_frame())
    assert result["scores"].shape == (n, d)
    assert result["groups"].shape == (n, d)
    assert result["individual_depth"].shape == (d,)
    with pytest.raises(KeyError):
        result["nope"]


def test_result_mdl_value(small_result):
    result = small_result
    assert isinstance(result.mdl_value, float)
    df = result.to_frame()
    total_dl = df["H_A_b"] + df["H_Adir_G_A_s_nu"] + df["H_s_G_b_depths"]
    assert result.mdl_value == pytest.approx(float(total_dl.min()))


def test_result_mdl_partition_shape(small_result):
    result = small_result
    G = _small_graph()
    p = result.mdl_partition
    assert isinstance(p, np.ndarray)
    assert len(p) == G.number_of_nodes()


def test_result_score_means_shape(small_result):
    result = small_result
    G = _small_graph()
    s = result.score_means
    assert isinstance(s, np.ndarray)
    assert len(s) == G.number_of_nodes()
    assert s.dtype == float


def test_result_score_stds_shape(small_result):
    result = small_result
    G = _small_graph()
    s = result.score_stds
    assert isinstance(s, np.ndarray)
    assert len(s) == G.number_of_nodes()
    assert s.dtype == float
    assert np.all(s >= 0)


# New API: network_properties_df and node_properties_df
def test_result_network_properties_df_columns(small_result):
    result = small_result
    npdf = result.network_properties_df()
    assert list(npdf.columns) == ["parameter", "mean", "std", "hdi_lo", "hdi_hi"]
    params = set(npdf["parameter"])
    assert "ICC" in params
    assert "sigma_T" in params
    assert "group_depth" in params
    assert "num_groups" in params
    assert len(npdf) == 11  # 9 base + sigma_T + ICC


def test_result_network_properties_df_prob():
    G = _small_graph()
    result = fit(Config(), G, num_samples=50, timeout=10.0)
    npdf90 = result.network_properties_df(prob=0.90)
    npdf50 = result.network_properties_df(prob=0.50)
    # Narrower interval for prob=0.50
    icc90 = npdf90[npdf90["parameter"] == "ICC"].iloc[0]
    icc50 = npdf50[npdf50["parameter"] == "ICC"].iloc[0]
    assert icc90["hdi_hi"] - icc90["hdi_lo"] >= icc50["hdi_hi"] - icc50["hdi_lo"]


def test_result_network_properties_df_agrees_with_draws():
    """network_properties_df means are the means of the draws."""
    G = _small_graph()
    result = fit(Config(), G, num_samples=20, timeout=10.0)
    npdf = result.network_properties_df().set_index("parameter")
    for p in (
        "num_groups",
        "density_in",
        "density_out",
        "variation_in",
        "variation_out",
        "degree_correction",
        "individual_depth",
        "group_depth",
        "ties_parameter",
        "sigma_T",
        "ICC",
    ):
        np.testing.assert_allclose(npdf.loc[p, "mean"], result[p].mean(), rtol=1e-9)


def test_network_properties_df_validates_prob():
    r = fit(Config(), _graph(), **_QUICK)
    for bad in (0.0, 1.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="prob"):
            r.network_properties_df(prob=bad)


def test_result_node_properties_df_columns(small_result):
    result = small_result
    G = _small_graph()
    ndf = result.node_properties_df()
    assert list(ndf.columns) == ["node", "score_mean", "score_std", "group", "group_score_mean"]
    assert len(ndf) == G.number_of_nodes()
    assert set(ndf["node"]) == set(G.nodes())
    assert ndf["group"].dtype in (np.int32, np.int64, int)


def test_result_node_properties_df_partition_modes(small_result):
    result = small_result
    df_auto = result.node_properties_df(partition="auto")
    df_mdl = result.node_properties_df(partition="mdl")
    df_cons = result.node_properties_df(partition="consensus")
    # score_mean must be identical across all modes (it's always the posterior mean)
    np.testing.assert_array_equal(df_auto["score_mean"].values, df_mdl["score_mean"].values)
    np.testing.assert_array_equal(df_auto["score_mean"].values, df_cons["score_mean"].values)


def test_result_node_properties_df_auto_large_warns(monkeypatch):
    """Auto mode must warn and fall back to mdl for n > 3000."""
    G = _small_graph()
    result = fit(Config(), G, num_samples=20, timeout=10.0)
    # Simulate a large network by extending _node_order to 3001 elements.
    # The warning fires in Python before the C++ call; the subsequent
    # DataFrame construction raises ValueError (C++ arrays are 6 elements
    # while _node_order is 3001), which we swallow — only the warning matters.
    big_order = tuple(f"node_{i}" for i in range(3001))
    monkeypatch.setattr(result, "_node_order", big_order)
    with pytest.warns(UserWarning, match="mdl_partition"):
        try:
            result.node_properties_df(partition="auto", consensus_method="binder")
        except (ValueError, Exception):
            pass  # expected: column length mismatch after monkeypatching


def test_result_node_properties_df_invalid_partition(small_result):
    result = small_result
    with pytest.raises(Exception):  # C++ raises for unknown mode string
        result.node_properties_df(partition="bogus")


def test_result_consensus_partition_shape(small_result):
    result = small_result
    G = _small_graph()
    cp = result.consensus_partition()
    assert isinstance(cp, np.ndarray)
    assert len(cp) == G.number_of_nodes()
    assert cp.dtype in (np.int32, np.int64, int)


def test_result_coincidence_matrix_shape(small_result):
    result = small_result
    G = _small_graph()
    cm = result.coincidence_matrix()
    n = G.number_of_nodes()
    assert cm.shape == (n, n)
    np.testing.assert_allclose(np.diag(cm), np.ones(n))


# The consensus is Dahl's method at the modal number of groups, followed by a
# greedy single-node descent on the same Binder loss with that count HELD.  The
# descent closes a gap measured at up to 19% of the loss on real networks
# (package-testing/asbm/consensus_refinement) without changing the group count.
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


# Default consensus is the maximum-overlap consensus (Peixoto 2021) with the
# number of groups free, as graph-tool's partition_overlap_center;
# "moc_modal_k" and "binder" hold the number of groups at the posterior mode.
# The thinning keeps max(min_samples, ESS of H_A_b) evenly spaced samples.
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


def test_direction_probabilities_ties_allowed(ranked_result):
    p = ranked_result.direction_probabilities("A", "D", ties_allowed=True)
    assert set(p.keys()) == {"a_to_b", "tie", "b_to_a"}
    assert sum(p.values()) == pytest.approx(1.0)
    assert p["a_to_b"] > 0.5


def test_direction_probabilities_no_ties(ranked_result):
    p = ranked_result.direction_probabilities("A", "D", ties_allowed=False)
    assert set(p.keys()) == {"a_to_b", "b_to_a"}
    assert "tie" not in p
    assert sum(p.values()) == pytest.approx(1.0)
    assert p["a_to_b"] > 0.5


def test_direction_probabilities_agrees_with_samples(ranked_result):
    """C++ result must match the same formula applied to samples_df().

    Uses the same numerically stable shift as the C++ implementation to avoid
    exp overflow on samples with large score magnitudes.
    """
    samp = ranked_result.to_frame()
    for ties_allowed in (True, False):
        p_cpp = ranked_result.direction_probabilities("A", "D", ties_allowed=ties_allowed)
        sa = samp["A_score"].values
        sd = samp["D_score"].values
        # Subtract per-sample max to match C++ overflow-safe computation
        if ties_allowed:
            nu = samp["ties_parameter"].values
            shift = np.maximum(np.maximum(sa, sd), (sa + sd) / 2.0)
            tie = nu * np.exp((sa + sd) / 2.0 - shift)
        else:
            shift = np.maximum(sa, sd)
            tie = np.zeros(len(sa))
        ea = np.exp(sa - shift)
        ed = np.exp(sd - shift)
        Z = ea + ed + tie
        p_py = {"a_to_b": float((ea / Z).mean()), "b_to_a": float((ed / Z).mean())}
        if ties_allowed:
            p_py["tie"] = float((tie / Z).mean())
        for key in p_cpp:
            assert p_cpp[key] == pytest.approx(p_py[key], abs=1e-12)


def test_direction_probabilities_unknown_node(ranked_result):
    with pytest.raises(ValueError, match="node_a"):
        ranked_result.direction_probabilities("X", "D")
    with pytest.raises(ValueError, match="node_b"):
        ranked_result.direction_probabilities("A", "X")


def test_plot_node_properties_options(small_result):
    """plot_node_properties draws from a Result or a node_properties_df table, with
    every label mode and the per-node options."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from community_hierarchy import plot_node_properties

    result = small_result
    df = result.node_properties_df()
    groups = sorted(df.group.unique())
    first = df.node.iloc[0]
    for src in (result, df):
        for labels in ("ticks", "ladder", "none"):
            fig, ax = plot_node_properties(
                src,
                labels=labels,
                flip=True,
                annotate={first: "x"},
                filled={first},
                palette={g: "C0" for g in groups},
                group_labels={g: f"g{g}" for g in groups},
            )
            assert ax.get_ylabel()
            plt.close(fig)
    with pytest.raises(ValueError):
        plot_node_properties(df, labels="sideways")
    with pytest.raises(TypeError):
        plot_node_properties([1, 2, 3])


def test_plot_MCMC_entropy_draws_one_line_per_chain():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = fit(Config(), _graph(), **{**_QUICK, "num_chains": 3, "num_samples": 5})
    fig, ax = ch.plot_MCMC_entropy(r)
    assert len(ax.get_lines()) == 3
    plt.close(fig)


def test_write_gml_uses_network_properties(tmp_path):
    r = fit(Config(), _graph(), **{**_QUICK, "num_samples": 20})
    out = tmp_path / "out.gml"

    ch.write_gml_with_inferences(_graph(), str(out), r)  # filterwarnings=error: no deprecation
    H = nx.read_gml(out)
    for key in ("inferred_ICC_mean", "inferred_sigma_T_hdi_lo", "inferred_group_depth_hdi_hi"):
        assert key in H.graph
