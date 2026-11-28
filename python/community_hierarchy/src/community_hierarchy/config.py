"""The model configuration: Config and its checks.

The presets a Config selects are applied in the C++ core (src/model_presets.h),
shared with the other bindings.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Optional

from .checks import _UNSET, _is_real

# The choices of each model setting.  The presets they select are applied in the
# C++ core (src/model_presets.h), shared with the other bindings.
_GROUPS_MODELS = (
    "traditional_SBM",
    "general_ASBM",
    "simple_ASBM",
    "hybrid_ASBM",
    "planted_partition",
)


_HIERARCHY_MODELS = ("bradley_terry", "bradley_terry_ties")


_INTERACTIONS = ("coupled", "independent")


_MIXING_VARIATIONS = ("general", "simple", "none", "internal", "external")


_MODEL_SETTINGS = (
    "variation_in",
    "variation_out",
    "individual_depth",
    "group_depth",
    "ties_parameter",
    "num_groups",
)


def _one_of(value, valid: tuple, name: str, hint: str = "") -> None:
    if value not in valid:
        raise ValueError(f"Unknown {name} {value!r}. Valid options: {list(valid)}.{hint}")


def _check_model(config: "Config") -> None:
    """The checks of a Config that need no graph: names, ranges, and explicit
    values that a preset would override."""
    if config.groups_model == "general_canonical":
        raise ValueError(
            "groups_model 'general_canonical' was renamed 'general_ASBM' (the same model)."
        )
    _one_of(
        config.groups_model,
        _GROUPS_MODELS,
        "groups_model",
        " Degree correction is not part of the model name; set it with "
        "Config(degree_correction=True | False).",
    )
    if config.hierarchy_model is not _UNSET:
        _one_of(config.hierarchy_model, _HIERARCHY_MODELS, "hierarchy_model")
    _one_of(config.interaction, _INTERACTIONS, "interaction")
    if config.mixing_variation is not _UNSET:
        _one_of(config.mixing_variation, _MIXING_VARIATIONS, "mixing_variation")

    for name in _MODEL_SETTINGS:
        x = getattr(config, name)
        if x is _UNSET or x is None:
            continue
        x = float(x)
        if not math.isfinite(x):
            raise ValueError(f"{name} must be a finite number or None (infer), got {x!r}.")
        if name in ("variation_in", "variation_out") and not 0.0 <= x <= 1.0:
            raise ValueError(f"{name} must be in [0, 1] or None (infer), got {x!r}.")
        if name == "individual_depth" and not x > 0.0:
            raise ValueError(f"individual_depth must be positive, got {x!r}.")
        if name in ("group_depth", "ties_parameter") and not x >= 0.0:
            raise ValueError(f"{name} must be non-negative, got {x!r}.")
        if name == "num_groups" and not (x >= 1 and x == int(x)):
            raise ValueError(f"num_groups must be a positive integer or None (infer), got {x!r}.")

    def nonzero(x):  # given, and not 0 (None, "infer", counts as nonzero)
        return x is not _UNSET and not (x is not None and float(x) == 0.0)

    if config.hierarchy_model == "bradley_terry" and nonzero(config.ties_parameter):
        raise ValueError(
            f"hierarchy_model='bradley_terry' fixes ties_parameter at 0, but "
            f"ties_parameter={config.ties_parameter!r} was given. Use "
            "hierarchy_model='bradley_terry_ties' to model draws."
        )
    if config.interaction == "independent" and nonzero(config.group_depth):
        raise ValueError(
            f"interaction='independent' fixes group_depth at 0, but "
            f"group_depth={config.group_depth!r} was given. Use interaction='coupled' "
            "to fit group_depth."
        )


def _check_model_for_graph(config: "Config", has_neutral: bool, num_nodes: int) -> None:
    """The checks of a Config that need the graph."""
    if (
        config.hierarchy_model is _UNSET
        and not has_neutral
        and config.ties_parameter is not _UNSET
        and not (config.ties_parameter is not None and float(config.ties_parameter) == 0.0)
    ):
        raise ValueError(
            f"ties_parameter={config.ties_parameter!r} was given, but G has no neutral "
            "edges, so hierarchy_model resolves to 'bradley_terry', which fixes "
            "ties_parameter at 0. Pass hierarchy_model='bradley_terry_ties' to fit a draw "
            "rate anyway."
        )
    if config.hierarchy_model == "bradley_terry" and has_neutral:
        raise ValueError(
            "hierarchy_model='bradley_terry' has no draws, but G has neutral edges. Use "
            "hierarchy_model='bradley_terry_ties' (the default when G has neutral edges), "
            "or remove the neutral edges."
        )
    k = config.num_groups
    if k is not _UNSET and k is not None and int(k) > num_nodes:
        raise ValueError(f"num_groups={int(k)} is larger than the number of nodes ({num_nodes}).")


@dataclass(frozen=True)
class Config:
    """Model configuration for community and hierarchy inference.

    Every field has a default, so ``Config()`` is the full coupled model with
    everything inferred.  Fields left unset take the value of the
    ``groups_model`` / ``hierarchy_model`` / ``interaction`` presets.

    Parameters
    ----------
    groups_model : {"traditional_SBM", "general_ASBM", "simple_ASBM", "hybrid_ASBM", "planted_partition"}
        Community structure preset; sets ``mixing_variation`` (and, for
        ``traditional_SBM``, ``assortative=False``) unless those are given:

        ========================  ====================  ===============
        preset                    mixing_variation      assortative
        ========================  ====================  ===============
        ``"traditional_SBM"``     ``"simple"``          False
        ``"general_ASBM"``        ``"general"``         True
        ``"simple_ASBM"``         ``"simple"``          True
        ``"hybrid_ASBM"``         ``"internal"``        True
        ``"planted_partition"``   ``"none"``            True
        ========================  ====================  ===============

        Default ``"traditional_SBM"``: the degree-corrected SBM, one density
        for all group pairs.  The assortative presets (``*_ASBM``,
        ``planted_partition``) give edges within and between groups separate
        densities.
    hierarchy_model : {"bradley_terry", "bradley_terry_ties"}, optional
        ``"bradley_terry"`` fixes ``ties_parameter`` at 0 (no draws; neutral
        edges are not used for the hierarchy, but still count in the community
        structure).  ``"bradley_terry_ties"``
        models draws.  Unset: ``"bradley_terry_ties"`` if the graph has a
        ``type="neutral"`` edge, else ``"bradley_terry"``.
    interaction : {"coupled", "independent"}
        ``"coupled"`` (default): scores are group mean plus individual
        deviation, with ``group_depth`` inferred.  ``"independent"``:
        ``group_depth`` is fixed at 0, so scores do not depend on the groups.
    assortative : bool, optional
        True: one density for edges within groups and one for edges between
        groups (``density_in``, ``density_out``).  False: a single density.
        Unset: from ``groups_model``.
    mixing_variation : {"general", "simple", "none", "internal", "external"}, optional
        Preset for ``variation_in`` / ``variation_out``: ``"general"`` infers
        both; ``"simple"`` fixes both at 0.5; ``"none"`` both at 0;
        ``"internal"`` 0.5 / 0; ``"external"`` 0 / 0.5.  Explicit
        ``variation_in`` / ``variation_out`` take precedence.  Unset: from
        ``groups_model``.
    variation_in, variation_out : float in [0, 1] or None, optional
        Fixed variation of the within-group / between-group block edge
        counts.  None: inferred.  Unset: from ``mixing_variation``.
    degree_correction : bool
        True (default): degree correction with a uniform prior over each
        group's degree sequence.  False: none.  Model names do not imply it.
    individual_depth : float > 0 or None, optional
        Spread of individual scores about their group mean (sigma_I).
        None or unset: inferred.
    group_depth : float >= 0 or None, optional
        Spread of the group mean scores (sigma_G).  None or unset: inferred
        (fixed at 0 by ``interaction="independent"``; giving any other value
        together with ``"independent"`` raises ``ValueError``).
    ties_parameter : float >= 0 or None, optional
        Draw propensity nu: a draw between i and j has weight
        ``nu * exp((s_i + s_j) / 2)`` against ``exp(s_i)`` and ``exp(s_j)``
        for the two wins.  None or unset: inferred (fixed at 0 by
        ``hierarchy_model="bradley_terry"``; giving any other value then
        raises ``ValueError``).
    num_groups : int or None, optional
        Fix the number of communities.  None or unset: inferred.
    group_attribute : str or None
        Name of a node attribute holding fixed community labels (any
        hashable values).  The partition is then not sampled.
    hierarchy_attribute : str or None
        Name of a numeric node attribute holding fixed scores.  The scores
        are then not sampled.
    weighted_SBM : bool
        Model real-valued edge weights instead of treating them as
        multiplicities: the binary network follows the SBM, the weights between
        two groups are uniform on the simplex of their total (Peixoto 2018), and
        the totals of the group pairs joined by an edge are uniform on the
        simplex of the overall weight.  The inferred groups do not depend on
        the unit of weight.  Weights must be finite and positive; self-loops
        are not supported.  Default False.
    weighted_proposals : bool
        weighted_SBM only.  False (default): the block-informed partition
        proposals pick neighbours and groups by binary edge counts, as
        graph-tool does; True: by edge weight (experimental; no mixing gain on
        the journal network).  Either leaves the posterior unchanged.

    Raises
    ------
    ValueError
        On an unknown preset name, an invalid value, or an explicit value
        that the chosen preset would override (see ``group_depth`` and
        ``ties_parameter``).
    """

    groups_model: str = "traditional_SBM"
    hierarchy_model: object = _UNSET  # str | _UNSET
    interaction: str = "coupled"
    assortative: object = _UNSET  # bool | _UNSET
    mixing_variation: object = _UNSET  # str | _UNSET
    variation_in: object = _UNSET  # float | None | _UNSET
    variation_out: object = _UNSET  # float | None | _UNSET
    degree_correction: bool = True
    individual_depth: object = _UNSET  # float | None | _UNSET
    group_depth: object = _UNSET  # float | None | _UNSET
    ties_parameter: object = _UNSET  # float | None | _UNSET
    num_groups: object = _UNSET  # int | None | _UNSET
    group_attribute: Optional[str] = None
    hierarchy_attribute: Optional[str] = None
    weighted_SBM: bool = False
    weighted_proposals: bool = False

    def __repr__(self) -> str:
        fields = ", ".join(
            f"{f.name}={getattr(self, f.name)!r}"
            for f in dataclasses.fields(self)
            if getattr(self, f.name) is not _UNSET
        )
        return f"Config({fields})"

    def __post_init__(self) -> None:
        for name in ("groups_model", "interaction"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"Unknown {name} {getattr(self, name)!r}")
        for name in ("hierarchy_model", "mixing_variation"):
            val = getattr(self, name)
            if val is not _UNSET and not isinstance(val, str):
                raise ValueError(f"Unknown {name} {val!r}")
        for name in ("degree_correction", "weighted_SBM", "weighted_proposals"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be True or False, got {getattr(self, name)!r}")
        if self.assortative is not _UNSET and not isinstance(self.assortative, bool):
            raise ValueError(f"assortative must be True or False, got {self.assortative!r}")
        for name in _MODEL_SETTINGS:
            val = getattr(self, name)
            if val is not _UNSET and val is not None and not _is_real(val):
                raise TypeError(f"{name} must be a number or None (infer), got {val!r}")
        for name in ("group_attribute", "hierarchy_attribute"):
            val = getattr(self, name)
            if val is not None and (not isinstance(val, str) or not val):
                raise ValueError(f"{name} must be a non-empty string or None, got {val!r}")
        _check_model(self)

    def _model_choice(self) -> dict:
        """The configuration as the C++ core's ModelChoice (src/model_presets.h)."""
        unset = lambda v: "" if v is _UNSET else v
        given = {
            name: (None if getattr(self, name) is None else float(getattr(self, name)))
            for name in _MODEL_SETTINGS
            if getattr(self, name) is not _UNSET
        }
        return dict(
            groups_model=self.groups_model,
            hierarchy_model=unset(self.hierarchy_model),
            interaction=self.interaction,
            mixing_variation=unset(self.mixing_variation),
            assortative=-1 if self.assortative is _UNSET else int(self.assortative),
            degree_correction=self.degree_correction,
            given=given,
        )
