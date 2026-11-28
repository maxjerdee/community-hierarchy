"""Fits shared by the tests that only inspect a Result."""

from __future__ import annotations

import pytest
from community_hierarchy import Config, fit
from helpers import _ranked_graph, _small_graph


# fit() and Result
@pytest.fixture(scope="module")
def small_result():
    """One fit of _small_graph(), shared by the tests that only inspect a Result.

    These assert on shapes, columns and dtypes, not on any particular fit, so
    refitting per test bought nothing but wall time (~15 fits of the same graph).
    """
    return fit(Config(), _small_graph(), num_samples=20, timeout=10.0)


@pytest.fixture(scope="module")
def ranked_result():
    G = _ranked_graph()
    return fit(
        Config(hierarchy_model="bradley_terry_ties"),
        G,
        num_chains=2,
        num_samples=100,
        seed=0,
        timeout=30.0,
        show_progress=False,
    )
