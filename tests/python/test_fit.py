"""Running fit(): seeds, chains, tempering, thinning and what reaches the C++ core."""

from __future__ import annotations

import community_hierarchy.MCMC_core as native
import networkx as nx
import numpy as np
import pytest
from community_hierarchy import Config, Result, fit
from helpers import (
    _graph,
    _small_graph,
)


# fit() argument forwarding
def _make_fake_native_result(G):
    """Return a minimal fake native result for monkeypatching _fit_backend."""
    # Use a real tiny fit so we have a valid native object to introspect
    # but capture the kwargs passed to it.
    pass  # implemented inline below


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


def test_fit_returns_result():
    G = _small_graph()
    config = Config()
    result = fit(config, G, num_samples=20, timeout=10.0)
    assert isinstance(result, Result)


def test_fit_forwards_seed_and_chains_to_backend(monkeypatch):
    """fit() must pass seed, num_chains, num_warmup, num_samples to backend."""
    import community_hierarchy.fitting as core_mod

    captured = {}
    _real_backend = core_mod._fit_backend

    def _recording_backend(**kwargs):
        captured.update(kwargs)
        return _real_backend(**kwargs)

    monkeypatch.setattr(core_mod, "_fit_backend", _recording_backend)

    G = _small_graph()
    fit(Config(), G, num_chains=2, num_samples=5, num_warmup=10, seed=99, timeout=30.0)

    assert captured["num_chains"] == 2
    assert captured["num_samples"] == 5
    assert captured["num_warmup"] == 10
    assert captured["seed"] == 99


def test_fit_forwards_num_tempering_chains(monkeypatch):
    """fit() must forward num_tempering_chains to the backend when provided."""
    import community_hierarchy.fitting as core_mod

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


def test_initial_partition_explicit_is_used(monkeypatch):
    """An explicit initial_partition of the correct length is forwarded to the backend."""
    import community_hierarchy.fitting as core_mod

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


# All chains used to share ONE global mt19937, drawn from concurrently by the
# std::async chain threads (each MCMC constructor also reseeded it).  Results
# with num_chains > 1 could not be reproduced from the seed.
@pytest.mark.parametrize("num_tempering_chains", [1, 3])
def test_multichain_fit_is_reproducible(num_tempering_chains):
    a = _dl_trace(num_tempering_chains=num_tempering_chains)
    b = _dl_trace(num_tempering_chains=num_tempering_chains)
    assert np.array_equal(a, b)


def test_different_seeds_give_different_chains():
    assert not np.array_equal(_dl_trace(seed=7), _dl_trace(seed=8))


# Tempering replicas were each built with their own seed, but every MCMC
# constructor reseeded the one thread_local generator, so only the last
# replica's seed counted.  fit() now seeds the generator once per chain and
# builds replicas with seed -1, which must leave the stream untouched.
def test_negative_seed_does_not_reseed_the_generator():

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


def test_num_thin_keeps_every_kth_iteration_as_in_stan():
    # As in PyStan: num_samples iterations, ceil(num_samples / num_thin) draws per chain.
    G = _small_graph()
    r = fit(
        Config(),
        G,
        num_chains=2,
        num_samples=9,
        num_thin=4,
        num_warmup=2,
        seed=1,
        show_progress=False,
    )
    assert len(r.to_frame()) == 2 * 3
