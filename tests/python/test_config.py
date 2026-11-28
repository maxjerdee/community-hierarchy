"""Config: the model choices, their checks, and how they resolve for a graph."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from community_hierarchy import Config, fit
from helpers import (
    _QUICK,
    _graph,
    _small_graph,
    _typed_graph,
)


# Config
def test_config_defaults():
    config = Config()
    assert config.groups_model == "traditional_SBM"
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


# Model names do not set degree correction; Config.degree_correction does.
@pytest.mark.parametrize("name", ["traditional_DCSBM", "traditional_GDCSBM"])
def test_degree_corrected_model_names_are_rejected(name):
    with pytest.raises(ValueError, match="degree_correction"):
        Config(groups_model=name)


def test_degree_correction_defaults_to_true():
    assert Config().degree_correction is True


@pytest.mark.parametrize("value", [0.5, None, "inferred"])
def test_degree_correction_must_be_binary(value):
    with pytest.raises(ValueError, match="degree_correction"):
        Config(degree_correction=value)


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


def test_negative_ties_parameter_is_rejected():
    with pytest.raises(ValueError, match="ties_parameter"):
        fit(Config(ties_parameter=-1.0), _typed_graph(), **_QUICK)


# An omitted hierarchy_model was resolved from the edge types only inside
# _check_input_validity, and the unset value was what reached
# _set_model_parameters -- so ties_parameter was always fitted, even on a graph
# with no neutral edges, where ties are not an outcome the data could record.
# That inflated the scores of small tie-free networks against plain
# Bradley-Terry (chickens: ~1.6x) in independent mode.
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


def test_ties_parameter_with_auto_bradley_terry_is_rejected():
    # No neutral edges: an omitted hierarchy_model resolves to bradley_terry.
    with pytest.raises(ValueError, match="no neutral edges"):
        fit(Config(ties_parameter=0.5), _graph(), **_QUICK)


def test_bradley_terry_on_neutral_edges_is_rejected():
    # Plain Bradley-Terry gives a draw probability 0, so a graph with draws has
    # likelihood 0 under it: an error, not a fit that quietly drops them.
    with pytest.raises(ValueError, match="neutral edges"):
        fit(Config(hierarchy_model="bradley_terry"), _typed_graph(), **_QUICK)


# Additional groups_model smoke tests
@pytest.mark.parametrize(
    "groups_model",
    [
        "traditional_SBM",
        "simple_ASBM",
        "planted_partition",
        "general_ASBM",
    ],
)
def test_groups_model_smoke(groups_model):
    """Every supported groups_model must produce a finite MDL value on the small graph."""
    G = _small_graph()
    result = fit(
        Config(groups_model=groups_model),
        G,
        num_samples=10,
        num_warmup=20,
        seed=0,
        timeout=20.0,
    )
    assert np.isfinite(result.mdl_value)
