"""Regression tests for bugs found in the 2026-09 code review."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from directedstructure import Config, fit


def _graph() -> nx.DiGraph:
    return nx.gnp_random_graph(40, 0.1, seed=1, directed=True)


def _dl_trace(seed: int = 7, **kwargs) -> np.ndarray:
    args = dict(num_chains=4, samples_per_chain=20, sweeps_per_sample=2, burn_sweeps=10,
                seed=seed, show_progress=False)
    args.update(kwargs)
    return np.asarray(fit(Config(), _graph(), **args).samples_df()["H_A_b"], dtype=float)


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
    with pytest.raises(ValueError, match="non-negative integers"):
        fit(Config(), G, num_samples=2, burn_sweeps=1, show_progress=False)


def test_weighted_sbm_still_accepts_real_weights():
    G = _graph()
    for i, (u, v) in enumerate(G.edges()):
        G[u][v]["weight"] = 0.5 + 0.25 * (i % 3)
    result = fit(Config(weighted_SBM=True), G, num_samples=2, burn_sweeps=1, show_progress=False)
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
    for model in ("general_canonical", "traditional_SBM", "general_ASBM"):
        result = fit(Config(groups_model=model, degree_correction=flag), _graph(),
                     num_samples=3, burn_sweeps=1, show_progress=False)
        assert np.all(result.samples_df()["degree_correction"] == expected), model


@pytest.mark.parametrize("value", [0.5, None, "inferred"])
def test_degree_correction_must_be_binary(value):
    with pytest.raises(ValueError, match="degree_correction"):
        Config(degree_correction=value)


# ---------------------------------------------------------------------------
# Self-loops in binary-count modes (weighted_SBM / separate group graph) would add
# one stub to e_rr instead of two; they are rejected up front.
# (The 2026-09 tempering-convention fix is verified by bit-identical output at
# beta = 1 and by the argument in src/MCMC_core.h; a statistical test of it on the
# depth parameters is not reliable because they mix too slowly -- see the review
# notes in package-testing/directedstructure/review_2026_09.)
# ---------------------------------------------------------------------------

def test_self_loops_rejected_in_binary_count_modes():
    G = _graph()
    u = next(iter(G.nodes()))
    G.add_edge(u, u)          # untyped, like the rest of _graph()
    with pytest.raises(ValueError, match="self-loop"):
        fit(Config(weighted_SBM=True), G, num_samples=2, burn_sweeps=1, show_progress=False)


# ---------------------------------------------------------------------------
# Tempering replicas were each built with their own seed, but every MCMC
# constructor reseeded the one thread_local generator, so only the last
# replica's seed counted.  fit() now seeds the generator once per chain and
# builds replicas with seed -1, which must leave the stream untouched.
# ---------------------------------------------------------------------------
def test_negative_seed_does_not_reseed_the_generator():
    import directedstructure.MCMC_core as native

    G = nx.gnp_random_graph(12, 0.3, seed=5, directed=True)
    wins = [dict() for _ in range(12)]
    for u, v in G.edges():
        wins[u][v] = 1.0
    ties = [dict() for _ in range(12)]

    def make(seed):
        return native.MCMC(wins=wins, ties=ties, partition=[i % 3 for i in range(12)],
                           scores=[0.0] * 12, num_groups=3, seed=seed,
                           degree_correction=0.5, degree_correction_fixed=True)

    def trace(m):
        out = []
        for _ in range(20):
            m.MCMC_sweep(1.0, 1, True)
            out.append(m.H_A_b)
        return out

    reference = trace(make(11))
    m = make(11)
    make(-1)            # must not disturb the stream m is about to use
    assert trace(m) == reference
    m = make(11)
    make(12)            # a non-negative seed does reseed it
    assert trace(m) != reference


def test_raw_mcmc_degree_correction_default_and_range():
    # The raw class used to default to degree_correction=1.0 (Dirichlet
    # concentration 0): H_k was NaN and the first sweep threw.
    import directedstructure.MCMC_core as native
    kw = dict(wins=[{1: 1.0}, {}, {}], ties=[{}, {}, {}], partition=[0, 0, 1],
              scores=[0.0, 0.0, 0.0], num_groups=2)
    m = native.MCMC(**kw)
    assert np.isfinite(m.H_k)
    m.MCMC_sweep(1.0, 1, True)
    with pytest.raises(ValueError, match="degree_correction"):
        native.MCMC(degree_correction=1.0, **kw)
