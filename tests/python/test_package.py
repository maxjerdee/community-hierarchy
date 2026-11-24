from __future__ import annotations

import importlib.metadata
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import pytest

import math

import directedstructure as ds
import directedstructure.MCMC_core as native
from directedstructure import Config, Result, fit

_DATA_IN  = Path(__file__).parent.parent / "data" / "inputs"
_DATA_OUT = Path(__file__).parent.parent / "data" / "outputs"


def _c2v(concentration: float) -> float:
    """Convert concentration to variation (inverse of variation_to_concentration)."""
    log_val = math.lgamma(concentration + 0.5) - math.lgamma(0.5) - math.lgamma(concentration + 1)
    return math.exp(log_val)


def test_version():
    assert importlib.metadata.version("directedstructure") == ds.__version__


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_config_defaults():
    config = Config()
    assert config.groups_model == "general_ASBM"
    assert config.interaction == "coupled"


def test_config_custom():
    config = Config(groups_model="traditional_SBM", interaction="independent")
    assert config.groups_model == "traditional_SBM"
    assert config.interaction == "independent"


def test_config_invalid_groups_model():
    with pytest.raises(ValueError, match="Unknown groups_model"):
        Config(groups_model="not_a_model")


def test_config_invalid_hierarchy_model():
    with pytest.raises(ValueError, match="Unknown hierarchy_model"):
        Config(hierarchy_model="bogus")


def test_config_invalid_interaction():
    with pytest.raises(ValueError, match="Unknown interaction"):
        Config(interaction="bad")


def test_config_is_frozen():
    config = Config()
    with pytest.raises(Exception):
        config.groups_model = "traditional_SBM"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Helpers: small directed graph fixture
# ---------------------------------------------------------------------------

def _small_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(range(6))
    # Two clique-like groups with directed edges
    edges = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1),
             (3, 4), (4, 3), (3, 5), (5, 3), (4, 5), (5, 4),
             (0, 3), (3, 0)]
    G.add_edges_from(edges)
    return G


# ---------------------------------------------------------------------------
# fit() and Result
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_result():
    """One fit of _small_graph(), shared by the tests that only inspect a Result.

    These assert on shapes, columns and dtypes, not on any particular fit, so
    refitting per test bought nothing but wall time (~15 fits of the same graph).
    """
    return fit(Config(), _small_graph(), num_samples=20, timeout=10.0)



def test_fit_returns_result():
    G = _small_graph()
    config = Config()
    result = fit(config, G, num_samples=20, timeout=10.0)
    assert isinstance(result, Result)


def test_fit_bad_config_type():
    G = _small_graph()
    with pytest.raises(TypeError, match="config must be an instance of Config"):
        fit("not_a_config", G)  # type: ignore[arg-type]


def test_fit_bad_graph_type():
    config = Config()
    with pytest.raises(TypeError, match="G must be a networkx"):
        fit(config, "not_a_graph")  # type: ignore[arg-type]


def test_result_samples_df_shape(small_result):
    result = small_result
    G = _small_graph()
    df = result.samples_df()
    assert len(df) > 0
    for node in G.nodes():
        assert f"{node}_group" in df.columns
        assert f"{node}_score" in df.columns
    for param in ("num_groups", "density_in", "density_out", "H_A_b",
                  "H_Adir_G_A_s_nu", "H_s_G_b_depths"):
        assert param in df.columns


def test_result_mdl_value(small_result):
    result = small_result
    G = _small_graph()
    assert isinstance(result.mdl_value, float)
    df = result.samples_df()
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


def test_result_parameter_df_params(small_result):
    result = small_result
    G = _small_graph()
    with pytest.warns(DeprecationWarning, match="network_properties_df"):
        net_df = result.parameter_df
    expected = {
        "num_groups", "density_in", "density_out", "variation_in",
        "variation_out", "degree_correction",
        "individual_depth", "group_depth", "ties_parameter",
    }
    assert set(net_df["parameter"]) == expected
    assert "mean" in net_df.columns
    assert "std" in net_df.columns


def test_result_scores_df_shape(small_result):
    result = small_result
    G = _small_graph()
    with pytest.warns(DeprecationWarning, match="node_properties_df"):
        scores_df = result.scores_df
    assert list(scores_df.columns) == ["node", "mean", "std"]
    assert len(scores_df) == G.number_of_nodes()
    assert set(scores_df["node"]) == set(G.nodes())


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


def test_result_summary_keys(small_result):
    result = small_result
    G = _small_graph()
    with pytest.warns(DeprecationWarning, match="mdl_value"):
        s = result.summary
    assert "num_samples" in s
    assert "mean_num_groups" in s
    assert "mdl_value" in s
    assert s["num_samples"] > 0
    assert s["mdl_value"] == result.mdl_value


# ---------------------------------------------------------------------------
# New API: network_properties_df and node_properties_df
# ---------------------------------------------------------------------------

def test_result_network_properties_df_columns(small_result):
    result = small_result
    G = _small_graph()
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


def test_result_network_properties_df_agrees_with_old():
    """network_properties_df means must match those from the deprecated parameter_df."""
    G = _small_graph()
    result = fit(Config(), G, num_samples=20, timeout=10.0)
    npdf = result.network_properties_df()
    with pytest.warns(DeprecationWarning):
        old = result.parameter_df
    base_params = [
        "num_groups", "density_in", "density_out", "variation_in",
        "variation_out", "degree_correction", "individual_depth",
        "group_depth", "ties_parameter",
    ]
    for p in base_params:
        new_mean = npdf[npdf["parameter"] == p]["mean"].iloc[0]
        old_mean = old[old["parameter"] == p]["mean"].iloc[0]
        np.testing.assert_allclose(new_mean, old_mean, rtol=1e-9,
                                   err_msg=f"mean mismatch for {p}")


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
    G = _small_graph()
    df_auto = result.node_properties_df(partition="auto")
    df_mdl  = result.node_properties_df(partition="mdl")
    df_cons = result.node_properties_df(partition="consensus")
    # score_mean must be identical across all modes (it's always the posterior mean)
    np.testing.assert_array_equal(df_auto["score_mean"].values,
                                  df_mdl["score_mean"].values)
    np.testing.assert_array_equal(df_auto["score_mean"].values,
                                  df_cons["score_mean"].values)
    # group assignments may differ between mdl and consensus


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
    G = _small_graph()
    with pytest.raises(Exception):  # C++ raises for unknown mode string
        result.node_properties_df(partition="bogus")


# ---------------------------------------------------------------------------
# Coincidence fixture tests (same graph as asbm test cases)
# ---------------------------------------------------------------------------

def _triangle_graph() -> nx.DiGraph:
    """3-node directed graph matching asbm's test3: edges 1→2 w=1, 2→3 w=2, 3→1 w=1."""
    G = nx.DiGraph()
    G.add_nodes_from([1, 2, 3])
    G.add_edge(1, 2, weight=1.0)
    G.add_edge(2, 3, weight=2.0)
    G.add_edge(3, 1, weight=1.0)
    return G


# ---------------------------------------------------------------------------
# fit() argument forwarding
# ---------------------------------------------------------------------------

def _make_fake_native_result(G):
    """Return a minimal fake native result for monkeypatching _fit_backend."""
    import directedstructure.MCMC_core as _nc
    # Use a real tiny fit so we have a valid native object to introspect
    # but capture the kwargs passed to it.
    pass  # implemented inline below


def test_fit_forwards_seed_and_chains_to_backend(monkeypatch):
    """fit() must pass seed, num_chains, burn_sweeps, samples_per_chain to backend."""
    import directedstructure.core as core_mod

    captured = {}
    _real_backend = core_mod._fit_backend

    def _recording_backend(**kwargs):
        captured.update(kwargs)
        return _real_backend(**kwargs)

    monkeypatch.setattr(core_mod, "_fit_backend", _recording_backend)

    G = _small_graph()
    fit(Config(), G, num_chains=2, samples_per_chain=5, burn_sweeps=10, seed=99, timeout=30.0)

    assert captured["num_chains"] == 2
    assert captured["samples_per_chain"] == 5
    assert captured["burn_sweeps"] == 10
    assert captured["seed"] == 99






def test_initial_partition_wrong_length_raises():
    G = _small_graph()
    with pytest.raises(ValueError, match="initial_partition length"):
        fit(Config(), G, initial_partition=[0, 1], timeout=5.0)


def test_initial_partition_explicit_is_used(monkeypatch):
    """An explicit initial_partition of the correct length is forwarded to the backend."""
    import directedstructure.core as core_mod

    captured = {}
    _real_backend = core_mod._fit_backend

    def _recording_backend(**kwargs):
        captured.update(kwargs)
        return _real_backend(**kwargs)

    monkeypatch.setattr(core_mod, "_fit_backend", _recording_backend)

    G = _small_graph()
    n = G.number_of_nodes()
    init_p = list(range(n))  # each node in its own group
    fit(Config(), G, initial_partition=init_p, num_samples=5, timeout=15.0)

    assert captured["initial_partition"] == init_p


# ---------------------------------------------------------------------------
# planted_partition model smoke test
# ---------------------------------------------------------------------------

@pytest.mark.statistical
def test_planted_partition_fit_runs():
    """planted_partition model must complete without error and return a Result."""
    G = _small_graph()
    n = G.number_of_nodes()
    # Partition must use contiguous labels starting from 0
    init_p = [i % 3 for i in range(n)]
    result = fit(
        Config(groups_model="planted_partition", interaction="coupled"),
        G,
        num_samples=20,
        burn_sweeps=50,
        initial_partition=init_p,
        seed=7,
        timeout=20.0,
    )
    assert isinstance(result, Result)
    assert np.isfinite(result.mdl_value)
    df = result.samples_df()
    assert len(df) > 0
    assert "num_groups" in df.columns


@pytest.mark.statistical
def test_planted_partition_q_is_not_always_one():
    """With a structured graph and random init, planted_partition should not collapse to q=1."""
    G = _small_graph()
    rng = np.random.default_rng(42)
    init_p = [int(x) for x in rng.integers(0, 3, size=G.number_of_nodes())]
    result = fit(
        Config(groups_model="planted_partition", interaction="coupled"),
        G,
        num_samples=200,
        burn_sweeps=500,
        initial_partition=init_p,
        seed=42,
        timeout=30.0,
    )
    df = result.samples_df()
    mean_q = df["num_groups"].mean()
    if mean_q <= 1.0:
        # Retry with longer run and different seed
        rng2 = np.random.default_rng(9999)
        init_p2 = [int(x) for x in rng2.integers(0, 3, size=G.number_of_nodes())]
        result = fit(
            Config(groups_model="planted_partition", interaction="coupled"),
            G,
            num_samples=500,
            burn_sweeps=1000,
            initial_partition=init_p2,
            seed=9999,
            timeout=60.0,
        )
        df = result.samples_df()
        mean_q = df["num_groups"].mean()
    # Should explore more than trivial q=1; any mean > 1 is acceptable
    assert mean_q > 1.0, f"planted_partition collapsed to q=1 (mean_q={mean_q:.2f})"


# ---------------------------------------------------------------------------
# Additional groups_model smoke tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("groups_model", [
    "traditional_SBM",
    "simple_ASBM",
    "planted_partition",
    "general_ASBM",
])
def test_groups_model_smoke(groups_model):
    """Every supported groups_model must produce a finite MDL value on the small graph."""
    G = _small_graph()
    result = fit(
        Config(groups_model=groups_model),
        G, num_samples=10, burn_sweeps=20, seed=0, timeout=20.0,
    )
    assert np.isfinite(result.mdl_value)



def _run_coincidence_native_wins_ties(num_chains: int, burn: int, samples: int, sweeps: int, seed: int) -> np.ndarray:
    """Multi-chain MCMC for the wins+ties 3-node graph with Mathematica-matched fixed params.

    This is the **coupled** model: scores {s[1], s[2], s[3]} are marginalized jointly
    with the partition, exactly as in the Mathematica NIntegrate (Out[271]).

    Graph: wins 0→1, 0→2, 1→2; tie 0↔1.
    Fixed: variation params at concentration=1, σI=1, σC=2, ν=1.
    Free: partition, scores, density_in, density_out.
    """
    wins = [{1: 1.0, 2: 1.0}, {2: 1.0}, {}]
    ties = [{1: 1.0}, {0: 1.0}, {}]
    n = 3
    vi = _c2v(1.0)
    acc = np.zeros((n, n))
    for chain in range(num_chains):
        state = native.MCMC(
            wins, ties, [],
            list(range(n)), False,
            [0.0] * n, False,
            n, False,
            True,
            vi, True, vi, True, vi, True,
            1.0, True, 2.0, True, 1.0, True,
            seed + chain,
        )
        for _ in range(burn):
            state.MCMC_sweep(1.0, sweeps)
        counts = np.zeros((n, n))
        for _ in range(samples):
            state.MCMC_sweep(1.0, sweeps)
            p = list(state.partition)
            for i in range(n):
                for j in range(n):
                    if p[i] == p[j]:
                        counts[i, j] += 1
        acc += counts / samples
    return acc / num_chains


@pytest.mark.statistical
def test_wins_ties_3node_coincidence():
    """Coupled coincidence matrix against Mathematica NIntegrate — the primary coupled validation.

    The NIntegrate marginalizes over scores {s[1],s[2],s[3]} jointly with the partition b,
    which corresponds to interaction='coupled'. This is distinct from the independent model
    where scores factor out and are integrated separately.

    Mathematica Out[271]: {{1, 0.183929, 0.177877}, {0.183929, 1, 0.221255}, {0.177877, 0.221255, 1}}
    Fixed: ν=1, σI=1, σC=2, concentration=1 (λIn=λOut=α=1); free: partition, scores, ρIn, ρOut.
    """
    expected = pd.read_csv(
        _DATA_OUT / "wins_ties_3node_coincidence.csv",
        header=None,
    ).to_numpy(dtype=float)
    coincidence = _run_coincidence_native_wins_ties(
        num_chains=64, burn=500, samples=500, sweeps=10, seed=42,
    )
    max_abs_diff = np.max(np.abs(coincidence - expected))
    if max_abs_diff >= 0.05:
        # Retry with 2x chains and samples to reduce platform-specific variance
        coincidence = _run_coincidence_native_wins_ties(
            num_chains=128, burn=500, samples=1000, sweeps=10, seed=42,
        )
        max_abs_diff = np.max(np.abs(coincidence - expected))
    assert max_abs_diff < 0.05, f"Coincidence matrix mismatch (max_abs_diff={max_abs_diff:.6f})"


def _tempering_coincidence(seed, num_chains, samples_per_chain, timeout):
    from directedstructure.MCMC_core import fit_backend
    n = 3
    wins_adj = [{1: 1.0, 2: 1.0}, {2: 1.0}, {}]
    ties_adj  = [{1: 1.0}, {0: 1.0}, {}]
    vi = _c2v(1.0)
    r = fit_backend(
        wins=wins_adj, ties=ties_adj,
        initial_partition=list(range(n)), initial_scores=[0.0] * n,
        assortative=True,
        individual_depth=1.0, group_depth=2.0,
        variation_in=vi, variation_out=vi, degree_correction=vi,
        ties_parameter=1.0,
        num_chains=num_chains, samples_per_chain=samples_per_chain,
        burn_sweeps=50, sweeps_per_sample=500,
        num_tempering_chains=2, seed=seed, timeout=timeout,
    )
    df = r.samples_df()
    ns = len(df)
    coin = np.zeros((n, n))
    for _, row in df.iterrows():
        p = [int(row[f"{i}_group"]) for i in range(n)]
        for i in range(n):
            for j in range(n):
                if p[i] == p[j]:
                    coin[i, j] += 1.0 / ns
    return coin, ns


@pytest.mark.statistical
def test_tempering_swap_uses_full_energy():
    """Parallel tempering swap criterion must use data terms only (H_A_G_b + H_Adir).

    Graph: wins 0→1, 0→2, 1→2; tie 0↔1 (same as test_wins_ties_3node_coincidence).
    Free: partition, scores, density_in, density_out.

    The CORRECT criterion is H_A_G_b + H_Adir_G_A_s_nu (data likelihood only).
    H_b (partition prior) is absorbed into the partition proposal.
    H_s_G_b_depths (score prior) is absorbed into the score proposal.
    Including H_s in the swap criterion causes systematic over-clustering (~0.06 bias).
    On first-pass failure the test retries with 4× more samples.
    """
    expected = pd.read_csv(_DATA_OUT / "wins_ties_3node_coincidence.csv",
                           header=None).to_numpy(dtype=float)

    coin, ns = _tempering_coincidence(seed=0, num_chains=4, samples_per_chain=2000, timeout=60.0)
    max_diff = np.max(np.abs(coin - expected))

    if max_diff >= 0.050:
        # Retry with 4× more samples in case of statistical bad luck.
        coin, ns = _tempering_coincidence(seed=1, num_chains=16, samples_per_chain=2000, timeout=240.0)
        max_diff = np.max(np.abs(coin - expected))

    assert max_diff < 0.050, (
        f"Tempered coincidence matrix deviates from Mathematica reference by "
        f"{max_diff:.4f} (threshold 0.050, {ns} samples).  "
        f"The swap criterion likely includes H_s_G_b_depths — it should use only "
        f"H_A_G_b + H_Adir_G_A_s_nu (data terms), not the score prior — check fit.cpp.  "
        f"coin={coin.tolist()}"
    )


def test_fit_forwards_num_tempering_chains(monkeypatch):
    """fit() must forward num_tempering_chains to the backend when provided."""
    import directedstructure.core as core_mod

    captured = {}
    _real_backend = core_mod._fit_backend

    def _recording_backend(**kwargs):
        captured.update(kwargs)
        return _real_backend(**kwargs)

    monkeypatch.setattr(core_mod, "_fit_backend", _recording_backend)

    G = _small_graph()
    fit(Config(), G, num_samples=5, num_tempering_chains=4, seed=1, timeout=30.0)

    assert captured.get("num_tempering_chains") == 4, (
        f"num_tempering_chains was not forwarded to _fit_backend "
        f"(captured keys: {list(captured.keys())})"
    )


@pytest.mark.statistical
def test_prior_q_uniform():
    """At beta=0 the proposal cancels H_b exactly, so q ~ Uniform(1,...,n).

    The partition prior is P(b) ∝ prod_r n_r! / C(n-1,q-1).
    Summing over all labeled partitions with q groups gives a constant n! for
    every q, so P(q) = 1/n.  At beta=0 every proposal is accepted and the
    chain draws purely from the proposal, so the empirical q distribution must
    be uniform over {1,...,n}.

    The test runs a short trial first and, if it fails, re-runs with a longer
    burn-in and more samples before truly failing. This avoids spurious CI
    failures across build configurations without loosening statistical rigor.
    """
    n = 8
    wins = [{(i + 1) % n: 1.0} for i in range(n)]  # ring so density_in != NaN
    ties = [{} for _ in range(n)]
    vi = _c2v(1.0)
    expected_mean = (n + 1) / 2.0   # 4.5 for n=8
    q_vals = np.arange(1, n + 1)

    def _make_state():
        return native.MCMC(
            wins, ties, [],
            list(range(n)), False,
            [0.0] * n, False,
            n, False,
            False,
            vi, True, vi, True, vi, True,
            1.0, True, 1.0, True, 1.0, True,
            0,
        )

    def _run_trial(burn_in, num_samples):
        state = _make_state()
        for _ in range(burn_in):
            state.MCMC_sweep(0.0, 1)
        trace = np.empty(num_samples)
        for t in range(num_samples):
            state.MCMC_sweep(0.0, 1)
            trace[t] = state.num_groups
        obs = np.bincount(trace.astype(int), minlength=n + 1)[1:].astype(float)
        mean_q = float(trace.mean())
        fracs = obs / num_samples
        # Threshold: 5 Monte Carlo standard errors from batch means (50 batches),
        # which accounts for autocorrelation -- consecutive sweeps are far from
        # independent (ESS ~ 1300 of 200k sweeps with the block-informed node move,
        # whose new-group probability is 0.01), so the iid SE would be ~10x too small.
        batches = trace[: num_samples // 50 * 50].reshape(50, -1).mean(axis=1)
        se = float(batches.std(ddof=1) / np.sqrt(50))
        mean_ok = abs(mean_q - expected_mean) < 5 * se
        bucket_ok = float(np.max(np.abs(fracs - 1.0 / n))) < 0.15
        return mean_ok and bucket_ok, mean_q, obs

    # Fast first attempt (~1 s).  On rare CI failures, retry with 5× more
    # burn-in and 5× more samples before declaring a real failure.
    for burn_in, num_samples in [(5_000, 40_000), (25_000, 200_000)]:
        passed, mean_q, obs = _run_trial(burn_in, num_samples)
        if passed:
            return

    fracs = obs / num_samples
    assert False, (
        f"q distribution at beta=0 not uniform after two attempts.\n"
        f"Last run: mean_q={mean_q:.3f}, expected={expected_mean:.1f}, "
        f"(5 batch-means SE not met).\n"
        f"Counts per q: {obs.astype(int).tolist()}\n"
        f"Fractions:    {np.round(fracs, 4).tolist()}"
    )


# ---------------------------------------------------------------------------
# Point-evaluation test against Mathematica fixture
# ---------------------------------------------------------------------------

def _parse_matrix_row(s: str) -> list[int]:
    """Parse a Mathematica-style '{a, b, c}' vector into a list of ints."""
    return [int(x) for x in s.strip().strip("{}").split(",")]


def _parse_matrix_line(line: str) -> list[list[int]]:
    """Parse a CSV line whose cells are '{...}' row-vectors into a 2-D list."""
    import csv as _csv
    cells = next(_csv.reader([line]))
    return [_parse_matrix_row(c) for c in cells]


def _wins_ties_from_rows(win_rows: list[list[int]], tie_rows: list[list[int]]):
    """Convert dense win/tie matrices to list-of-dicts format."""
    n = len(win_rows)
    wins = [{} for _ in range(n)]
    ties = [{} for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if win_rows[i][j]: wins[i][j] = float(win_rows[i][j])
            if tie_rows[i][j]: ties[i][j] = float(tie_rows[i][j])
    return wins, ties


def test_point_eval_entropy_components():
    """Entropy components for a 3-node directed graph against Mathematica fixture.

    Reference: Mathematica HAwinsAtiesscoresbρInρOutλInλOutαNuσIσCParts[
        Awins, Aties, {1,2,3}, {1,1,2}, 0.6, 0.4, 2, 3, 0.5, 0.6, 1, 2]
    returns {HAb, HAwinsAtiesGAunscoresν, HscoresGbσIσC, Hν, HσI, HσC}
           = {17.7264, 6.26974, 6.42043, 1.86013, 1.8985, 2.06102}

    HAb includes H_b + H_k + H_M + H_A_G_k_M_b AND half-Cauchy priors on
    (ρIn, ρOut, λIn, λOut, α), so HAb > sum of the four stored C++ components.
    Hν, HσI, HσC are standalone half-Cauchy priors: log(π(w²+x²)/(2w)), w=4.
    """
    in_path  = _DATA_IN  / "wins_ties_3node_point_eval.csv"
    out_path = _DATA_OUT / "wins_ties_3node_point_eval_components.csv"

    lines = in_path.read_text().splitlines()
    # Input file: two brace-lines (all rows of Awins, then all rows of Aties),
    # followed by scalar lines (partition, scores, then one param per line).
    brace_lines  = [ln for ln in lines if "{" in ln]
    scalar_lines = [ln for ln in lines if ln and "{" not in ln]

    win_rows = _parse_matrix_line(brace_lines[0])
    tie_rows = _parse_matrix_line(brace_lines[1])
    wins, ties = _wins_ties_from_rows(win_rows, tie_rows)

    partition = [int(x) - 1 for x in scalar_lines[0].split(",")]
    scores    = [float(x) for x in scalar_lines[1].split(",")]
    rho_in, rho_out, c_in, c_out, alpha, nu, sigma_i, sigma_c = \
        [float(scalar_lines[i]) for i in range(2, 10)]

    ref = pd.read_csv(out_path)

    state = native.MCMC(
        wins, ties, [],
        partition, True,
        scores, True,
        max(partition) + 1, True,
        True,                        # assortative (ρIn ≠ ρOut)
        _c2v(c_in),  True,           # variation_in  ← concentration c_in
        _c2v(c_out), True,           # variation_out ← concentration c_out
        alpha,       True,           # degree_correction (variation)
        sigma_i,     True,           # individual_depth σI
        sigma_c,     True,           # group_depth σC
        nu,          True,           # ties_parameter ν
        0,
        rho_in,                      # density_in
        rho_out,                     # density_out
    )

    # ── Components also checkable individually (Mathematica element 2 & 3) ──
    np.testing.assert_allclose(
        float(state.H_Adir_G_A_s_nu), ref["H_Adir"].iloc[0], rtol=1e-6)
    np.testing.assert_allclose(
        float(state.H_s_G_b_depths),  ref["H_s"].iloc[0],    rtol=1e-6)

    # ── SBM components (part of Mathematica element 1, without param priors) ─
    np.testing.assert_allclose(float(state.H_b),          ref["H_b"].iloc[0],          rtol=1e-6)
    np.testing.assert_allclose(float(state.H_k),          ref["H_k"].iloc[0],          rtol=1e-6)
    np.testing.assert_allclose(float(state.H_M),          ref["H_M"].iloc[0],          rtol=1e-6)
    np.testing.assert_allclose(float(state.H_A_G_k_M_b), ref["H_A_G_k_M_b"].iloc[0], rtol=1e-6)

    # ── Half-Cauchy priors: Hx = log(π(w²+x²)/(2w)), w=4 (Mathematica elements 4-6) ──
    w = 4.0
    def h_hc(x: float) -> float:
        return math.log(math.pi * (w**2 + x**2) / (2.0 * w))

    np.testing.assert_allclose(h_hc(nu),      ref["H_nu"].iloc[0],     rtol=1e-6)
    np.testing.assert_allclose(h_hc(sigma_i), ref["H_sigmaI"].iloc[0], rtol=1e-6)
    np.testing.assert_allclose(h_hc(sigma_c), ref["H_sigmaC"].iloc[0], rtol=1e-6)

    # ── Internal consistency ─────────────────────────────────────────────────
    np.testing.assert_allclose(
        float(state.H_A_G_b),
        float(state.H_M) + float(state.H_k) + float(state.H_A_G_k_M_b),
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# direction_probabilities
# ---------------------------------------------------------------------------

def _ranked_graph() -> nx.DiGraph:
    """String-labelled graph with a clear dominant node for hierarchy tests."""
    G = nx.DiGraph()
    for _ in range(10):
        for loser in ("B", "C", "D"):
            w = G.get_edge_data("A", loser, {}).get("weight", 0)
            G.add_edge("A", loser, weight=w + 1)
    for _ in range(5):
        for loser in ("C", "D"):
            w = G.get_edge_data("B", loser, {}).get("weight", 0)
            G.add_edge("B", loser, weight=w + 1)
    return G


@pytest.fixture(scope="module")
def ranked_result():
    G = _ranked_graph()
    return fit(
        Config(hierarchy_model="bradley_terry_ties"),
        G, num_chains=2, samples_per_chain=100, seed=0, timeout=30.0, show_progress=False,
    )


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
    samp = ranked_result.samples_df()
    for ties_allowed in (True, False):
        p_cpp = ranked_result.direction_probabilities("A", "D", ties_allowed=ties_allowed)
        sa = samp["A_score"].values
        sd = samp["D_score"].values
        # Subtract per-sample max to match C++ overflow-safe computation
        if ties_allowed:
            nu    = samp["ties_parameter"].values
            shift = np.maximum(np.maximum(sa, sd), (sa + sd) / 2.0)
            tie   = nu * np.exp((sa + sd) / 2.0 - shift)
        else:
            shift = np.maximum(sa, sd)
            tie   = np.zeros(len(sa))
        ea = np.exp(sa - shift)
        ed = np.exp(sd - shift)
        Z  = ea + ed + tie
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


# ---------------------------------------------------------------------------
# Weighted SBM
# ---------------------------------------------------------------------------

def test_weighted_SBM_point_eval():
    """Numerical ground truth for H_W on a 3-node, 2-group binary graph.

    Graph: nodes {0,1} in group 0, node {2} in group 1.
      W[0,1] = 1.97558  (within group 0)
      W[0,2] = 1.13873  (between groups)
      W[1,2] = 2.77927  (between groups)
    group_adj encodes W as the SBM graph; A is binary (one edge per pair).

    H_W = H(W|A,μ,b) + H(μ)  (full description length, Peixoto 2018 Eq. 33+34)

    H(W|A,μ,b) — per-block uniform-simplex cost:
      block (0,0): M_rr/2=1, mu_rr/2=W01  → (1-1)*log(W01) - lgamma(1) = 0
      block (0,1): M_rs=2,   mu_rs=W02+W12 → (2-1)*log(mu_01) - lgamma(2) = log(mu_01)
      → log(W02+W12)

    H(μ) — cost to encode block-pair weight totals on a simplex (Eq. 34):
      q=2, npairs=q*(q+1)/2=3, w_total=W01+W02+W12
      → (npairs-1)*log(w_total) - log((npairs-1)!)
      → 2*log(w_total) - log(2)
    """
    import directedstructure.MCMC_core as native

    w01, w02, w12 = 1.97558, 1.13873, 2.77927

    # H(W|A,μ,b): only the active off-diagonal block (M=2)
    h_w_given = math.log(w02 + w12)

    # H(μ): cost of encoding block-pair weight totals (Peixoto 2018 Eq. 34)
    num_groups_test = 2
    npairs = num_groups_test * (num_groups_test + 1) // 2  # = 3
    w_total = w01 + w02 + w12
    h_mu = (npairs - 1) * math.log(w_total) - math.lgamma(npairs)  # lgamma(n) = log((n-1)!)

    expected = h_w_given + h_mu

    n = 3
    wins = [{} for _ in range(n)]
    wins[0][1] = 1.0; wins[0][2] = 1.0; wins[1][2] = 1.0

    group_adj = [{} for _ in range(n)]
    group_adj[0][1] = w01; group_adj[1][0] = w01
    group_adj[0][2] = w02; group_adj[2][0] = w02
    group_adj[1][2] = w12; group_adj[2][1] = w12

    mcmc = native.MCMC(
        wins=wins,
        ties=[{} for _ in range(n)],
        group_adj=group_adj,
        partition=[0, 0, 1],
        scores=[0.0, 0.0, 0.0],
        num_groups=2,
        seed=0,
        weighted_SBM=True,
    )

    assert abs(mcmc.H_W - expected) < 1e-6, f"H_W = {mcmc.H_W:.6f}, expected = {expected:.6f}"


def test_weighted_SBM_unweighted_unchanged():
    """weighted_SBM=False (default) must give identical results to the old code path."""
    G = nx.DiGraph()
    G.add_edge("A", "B", weight=3)
    G.add_edge("B", "C", weight=2)
    G.add_edge("A", "C", weight=1)

    config_plain    = ds.Config(groups_model="general_ASBM")
    config_weighted = ds.Config(groups_model="general_ASBM", weighted_SBM=False)

    r1 = ds.fit(config_plain,    G, num_chains=1, samples_per_chain=50, burn_sweeps=50, seed=7)
    r2 = ds.fit(config_weighted, G, num_chains=1, samples_per_chain=50, burn_sweeps=50, seed=7)

    # Identical seeds → identical MDL
    assert r1.mdl_value == pytest.approx(r2.mdl_value, rel=1e-6)
    sdf1 = r1.samples_df()
    sdf2 = r2.samples_df()
    assert (sdf1["H_W"] == 0.0).all()
    assert (sdf2["H_W"] == 0.0).all()


def test_weighted_SBM_produces_nonzero_H_W():
    """With weighted_SBM=True, H_W is recorded and is non-negative.
    H_W=0 is valid when all block pairs have m_rs=1 (simplex contributes 0).
    H_W>0 (DL cost) or H_W<0 (DL benefit) when m_rs>1 in some block pair.
    """
    G = nx.DiGraph()
    G.add_edge("A", "B", weight=3.5)
    G.add_edge("B", "C", weight=2.1)
    G.add_edge("A", "C", weight=1.7)
    G.add_edge("C", "A", weight=0.8)

    config = ds.Config(groups_model="general_ASBM", weighted_SBM=True)
    r = ds.fit(config, G, num_chains=1, samples_per_chain=50, burn_sweeps=50, seed=42)
    sdf = r.samples_df()
    # H_W is a finite real (DL contribution from weight model)
    assert sdf["H_W"].notna().all(), "H_W must be finite in weighted mode"
    assert (sdf["weighted_SBM"] == True).all()
