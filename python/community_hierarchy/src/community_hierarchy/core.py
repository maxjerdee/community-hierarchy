from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional
import logging
import math
import warnings

import numpy as np
import networkx as nx
import pandas as pd

from .input_output import (
    _UNSET,
    _check_fit_arguments,
    _edges,
    _get_node_data,
    _group_graph_edges,
    _is_real,
)
from .MCMC_core import build_group_layer as _build_group_layer
from .MCMC_core import build_interactions as _build_interactions
from .MCMC_core import fit_backend as _fit_backend
from .MCMC_core import resolve_model as _resolve_model
from .MCMC_core import sort_groups_by_score as _sort_groups_by_score

logger = logging.getLogger(__name__)


# The choices of each model setting.  The presets they select are applied in the
# C++ core (src/model_presets.h), shared with the other bindings.
_GROUPS_MODELS = ("traditional_SBM", "general_ASBM", "simple_ASBM", "hybrid_ASBM",
                  "planted_partition")
_HIERARCHY_MODELS = ("bradley_terry", "bradley_terry_ties")
_INTERACTIONS = ("coupled", "independent")
_MIXING_VARIATIONS = ("general", "simple", "none", "internal", "external")
_MODEL_SETTINGS = ("variation_in", "variation_out", "individual_depth", "group_depth",
                   "ties_parameter", "num_groups")


def _one_of(value, valid: tuple, name: str, hint: str = "") -> None:
    if value not in valid:
        raise ValueError(f"Unknown {name} {value!r}. Valid options: {list(valid)}.{hint}")


def _check_model(config: "Config") -> None:
    """The checks of a Config that need no graph: names, ranges, and explicit
    values that a preset would override."""
    if config.groups_model == "general_canonical":
        raise ValueError("groups_model 'general_canonical' was renamed 'general_ASBM' "
                         "(the same model).")
    _one_of(config.groups_model, _GROUPS_MODELS, "groups_model",
            " Degree correction is not part of the model name; set it with "
            "Config(degree_correction=True | False).")
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
            "hierarchy_model='bradley_terry_ties' to model draws.")
    if config.interaction == "independent" and nonzero(config.group_depth):
        raise ValueError(
            f"interaction='independent' fixes group_depth at 0, but "
            f"group_depth={config.group_depth!r} was given. Use interaction='coupled' "
            "to fit group_depth.")


def _check_model_for_graph(config: "Config", has_neutral: bool, num_nodes: int) -> None:
    """The checks of a Config that need the graph."""
    if config.hierarchy_model is _UNSET and not has_neutral and \
            config.ties_parameter is not _UNSET and \
            not (config.ties_parameter is not None and float(config.ties_parameter) == 0.0):
        raise ValueError(
            f"ties_parameter={config.ties_parameter!r} was given, but G has no neutral "
            "edges, so hierarchy_model resolves to 'bradley_terry', which fixes "
            "ties_parameter at 0. Pass hierarchy_model='bradley_terry_ties' to fit a draw "
            "rate anyway.")
    if config.hierarchy_model == "bradley_terry" and has_neutral:
        raise ValueError(
            "hierarchy_model='bradley_terry' has no draws, but G has neutral edges. Use "
            "hierarchy_model='bradley_terry_ties' (the default when G has neutral edges), "
            "or remove the neutral edges.")
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
        given = {name: (None if getattr(self, name) is None else float(getattr(self, name)))
                 for name in _MODEL_SETTINGS if getattr(self, name) is not _UNSET}
        return dict(groups_model=self.groups_model,
                    hierarchy_model=unset(self.hierarchy_model),
                    interaction=self.interaction,
                    mixing_variation=unset(self.mixing_variation),
                    assortative=-1 if self.assortative is _UNSET else int(self.assortative),
                    degree_correction=self.degree_correction,
                    given=given)


class Result:
    """Posterior samples of a fit, with summaries.  Obtain via :func:`fit`.

    Every per-node array (``score_means``, ``score_stds``, partitions, rows and
    columns of ``coincidence_matrix``) is in the order of ``G.nodes()`` of the
    graph passed to :func:`fit`; the DataFrame methods carry the node labels
    in a ``node`` column or in the column names.  Scores are on the
    Bradley-Terry log-odds scale: higher means higher rank, whatever
    ``source_high_rank`` was.  Description lengths are in nats.

    A Result can be pickled.
    """

    def __init__(self, native_result, node_order: tuple, config: Config,
                 interactions: dict | None = None) -> None:
        self._native = native_result
        self._node_order = node_order
        self._config = config
        # The edges of G, for direction_loo_df(): node indices a (the winner, or
        # the first end of a draw) and b, draw flags and weights.
        self._interactions = interactions

    # ------------------------------------------------------------------
    # Point estimates — delegated to C++
    # ------------------------------------------------------------------

    @property
    def mdl_value(self) -> float:
        """Smallest description length among the recorded samples, in nats.

        The description length of a sample is ``H_A_b + H_Adir_G_A_s_nu +
        H_s_G_b_depths`` (see :meth:`samples_df`): minus the log joint
        probability of the network, the partition and the scores given the
        continuous parameters.  NaN if no samples were recorded."""
        return float(self._native.mdl_value)

    @property
    def mdl_partition(self) -> np.ndarray:
        """Partition of the sample with the smallest description length,
        shape (n_nodes,), in ``G.nodes()`` order.  Groups are relabelled so
        group 0 has the lowest mean posterior score.  Not a posterior summary;
        see :meth:`consensus_partition`."""
        part = np.asarray(self._native.mdl_partition(), dtype=int)
        return np.asarray(_sort_groups_by_score(part, self.score_means), dtype=int)

    def consensus_partition(self, method: str = "moc", refine: bool = True,
                            min_samples: int = 1000) -> np.ndarray:
        """Point estimate of the community partition.

        Parameters
        ----------
        method : {"moc", "moc_modal_k", "binder"}
            ``"moc"`` (default)
                Maximum-overlap consensus (Peixoto 2021, Phys. Rev. X 11,
                021003), as graph-tool's ``partition_overlap_center``: the
                partition -- any partition, not only a recorded one, with any
                number of groups -- that agrees with the most node labels
                across the samples, after optimally relabelling each sample.
                Coordinate ascent (align, then majority vote), started from the
                one-group partition, from up to three samples and from the
                ``"moc_modal_k"`` consensus, so it is never worse than that.  The number
                of groups is an output, and can differ from the posterior mode
                of K: nodes that move among small, unstable groups join the
                stable group they agree with most.  No n x n matrix.
            ``"moc_modal_k"``
                The same, with the number of groups held at the posterior
                modal K and restarts from up to three modal-K samples.
            ``"binder"``
                Binder loss (Binder 1978; Dahl 2006): the recorded sample with K
                groups whose co-membership is closest in L1 to the posterior
                co-membership matrix, then (``refine=True``) a greedy
                single-node descent on the same loss with K held.
        refine : bool
            Binder only; ``False`` returns the unrefined argmin over samples.
        min_samples : int
            Samples are thinned, evenly spaced, to max(min_samples, ESS), where
            ESS is the rank-normalised bulk ESS of H_A_b (see
            :meth:`partition_ess`).  All samples are used when there are fewer
            than ``min_samples``; ``min_samples=0`` disables thinning.

        Returns
        -------
        numpy.ndarray of int, shape (n_nodes,)
            Group label of each node, in ``G.nodes()`` order, relabelled so
            group 0 has the lowest mean posterior score."""
        m = str(method).lower()
        if m not in ("moc", "moc_modal_k", "binder"):
            raise ValueError(f"method must be 'moc', 'moc_modal_k' or 'binder', got {method!r}")
        part = np.asarray(self._native.consensus_partition(m, refine, int(min_samples)), dtype=int)
        return np.asarray(_sort_groups_by_score(part, self.score_means), dtype=int)

    def partition_ess(self) -> float:
        """Bulk ESS (Vehtari et al. 2021, rank-normalised, split chains, pooled)
        of H_A_b, the description length of the network given the partition.
        Used to thin the samples for :meth:`consensus_partition`."""
        return float(self._native.partition_ess())

    @property
    def score_means(self) -> np.ndarray:
        """Posterior mean score of each node, shape (n_nodes,), in ``G.nodes()`` order."""
        return np.asarray(self._native.score_means(), dtype=float)

    @property
    def score_stds(self) -> np.ndarray:
        """Posterior standard deviation of each node's score, shape (n_nodes,),
        in ``G.nodes()`` order."""
        return np.asarray(self._native.score_stds(), dtype=float)

    def network_properties_df(self, prob: float = 0.90) -> pd.DataFrame:
        """Posterior summary of network-level parameters with HDI credible intervals.

        Parameters
        ----------
        prob : float
            Probability mass of the highest-density interval, in (0, 1).
            Default 0.90.

        Returns
        -------
        pandas.DataFrame
            Columns ``parameter``, ``mean``, ``std``, ``hdi_lo``, ``hdi_hi``;
            one row for each of ``num_groups``, ``density_in``,
            ``density_out``, ``variation_in``, ``variation_out``,
            ``degree_correction``, ``individual_depth`` (sigma_I),
            ``group_depth`` (sigma_G), ``ties_parameter``, ``sigma_T`` =
            sqrt(sigma_G^2 + sigma_I^2) and ``ICC`` = sigma_G^2 / sigma_T^2
            (the share of score variance explained by the groups).  ``std`` is
            the sample standard deviation; the HDI is the shortest interval
            holding ``floor(prob * n_samples)`` samples.  Empty if no samples
            were recorded.

        Raises
        ------
        ValueError
            If ``prob`` is not in (0, 1).

        Notes
        -----
        Computed entirely in C++ without materialising the full
        ``to_frame()``; efficient even for large networks.

        Examples
        --------
        >>> npdf = result.network_properties_df()
        >>> icc_row = npdf[npdf["parameter"] == "ICC"].iloc[0]
        >>> print(f"ICC = {icc_row['mean']:.3f}  [{icc_row['hdi_lo']:.3f}, {icc_row['hdi_hi']:.3f}]")
        """
        if not isinstance(prob, (int, float, np.floating)) or not 0.0 < float(prob) < 1.0:
            raise ValueError(f"prob must be in (0, 1), got {prob!r}")
        d = dict(self._native.network_properties(float(prob)))
        return pd.DataFrame(d, columns=["parameter", "mean", "std", "hdi_lo", "hdi_hi"])

    def node_properties_df(self, partition: str = "auto",
                           consensus_method: str = "moc") -> pd.DataFrame:
        """Posterior summary of node-level properties with community labels.

        Parameters
        ----------
        partition : str
            Which partition to use for community labels:

            ``"auto"`` (default)
                The consensus partition; except with
                ``consensus_method="binder"`` on networks with n > 3000, where
                the MDL partition is used with a ``UserWarning``.
            ``"consensus"``
                Always the consensus partition (see :meth:`consensus_partition`).
            ``"mdl"``
                MDL-optimal single sample; free but not a posterior summary.
        consensus_method : {"moc", "moc_modal_k", "binder"}
            Passed to :meth:`consensus_partition`.  The MOC methods need no
            n x n matrix; ``"binder"`` is O(n^2) in time and memory.

        Returns
        -------
        pandas.DataFrame
            One row per node, in ``G.nodes()`` order, with columns ``node``
            (the label in ``G``), ``score_mean``, ``score_std``, ``group``
            (relabelled so group 0 has the lowest mean score) and
            ``group_score_mean`` (mean of ``score_mean`` over the node's
            group).

        Notes
        -----
        ``"moc_modal_k"`` and ``"binder"`` return a partition with the
        posterior modal number of groups; ``"moc"`` chooses its own.  ``"mdl"``
        is a free byproduct of sampling (not an
        optimisation result).

        Examples
        --------
        >>> npdf = result.node_properties_df()
        >>> print(npdf.sort_values("score_mean", ascending=False).head())
        """
        cm = str(consensus_method).lower()
        if cm not in ("moc", "moc_modal_k", "binder"):
            raise ValueError(f"consensus_method must be 'moc', 'moc_modal_k' or 'binder', "
                             f"got {consensus_method!r}")
        # Resolve "auto" in Python so the threshold check is testable and the
        # warning fires before calling C++ (language-specific concern).
        n = len(self._node_order)
        if partition == "auto":
            mode = "mdl" if (cm == "binder" and n > 3000) else "consensus"
            if mode == "mdl":
                warnings.warn(
                    f"Network is large (n={n} > 3000); using mdl_partition "
                    "instead of the Binder consensus, which is O(n^2) in time "
                    "and memory. Use consensus_method='moc' (no n x n matrix), "
                    "pass partition='consensus' to force Binder, or "
                    "partition='mdl' to suppress this warning.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            _valid = {"consensus", "mdl"}
            if partition not in _valid:
                raise ValueError(
                    f"partition must be one of {sorted(_valid)!r} or 'auto', "
                    f"got {partition!r}"
                )
            mode = partition
        d = dict(self._native.node_properties(mode, cm))
        d.pop("mode_used", None)
        d["node"] = list(self._node_order)
        # Reorder columns: node first, then the rest
        return pd.DataFrame(d, columns=["node", "score_mean", "score_std", "group", "group_score_mean"])

    # ------------------------------------------------------------------
    # Distribution summaries
    # ------------------------------------------------------------------

    def coincidence_matrix(self) -> np.ndarray:
        """Co-assignment frequency matrix, shape (n_nodes, n_nodes).

        Entry [i, j] is the fraction of samples in which the i-th and j-th
        nodes of ``G.nodes()`` are in the same community.  The diagonal is 1.
        Memory is O(n^2).
        """
        return np.asarray(self._native.coincidence_matrix(), dtype=float)

    def to_frame(self) -> pd.DataFrame:
        """All recorded draws, one row per draw (chains concatenated), as
        PyStan's ``fit.to_frame()``.

        Returns
        -------
        pandas.DataFrame
            Columns:

            - ``{node}_group``, ``{node}_score`` for each node label of ``G``
              (group labels are raw sampler labels, not relabelled);
            - parameters ``num_groups``, ``density_in``, ``density_out``,
              ``variation_in``, ``variation_out``, ``degree_correction``,
              ``individual_depth``, ``group_depth``, ``ties_parameter``;
            - derived ``sigma_T`` and ``ICC`` (see :meth:`network_properties_df`);
            - description-length terms in nats: ``H_b`` (partition prior),
              ``H_M``, ``H_k``, ``H_A_G_k_M_b``, ``H_W`` (weighted_SBM only,
              else 0), ``H_A_G_b`` (undirected network given the partition),
              ``H_A_b`` = ``H_A_G_b + H_b``, ``H_Adir_G_A_s_nu`` (edge
              directions given the scores), ``H_s_G_b_depths`` (scores given
              the partition);
            - ``chain_id``, ``beta`` (inverse temperature of the recorded
              replica), ``two_layer_mode``, ``weighted_SBM``.
        """
        df = self._native.samples_df()
        nodes = list(self._node_order)
        rename = {}
        for i, node in enumerate(nodes):
            rename[f"{i}_group"] = f"{node}_group"
            rename[f"{i}_score"] = f"{node}_score"
        return df.rename(columns=rename)

    def __getitem__(self, name: str) -> np.ndarray:
        """The draws of one quantity, as PyStan's ``fit[name]``: the last axis
        runs over the draws (chains concatenated, as in :meth:`to_frame`).

        ``"scores"`` and ``"groups"`` give arrays of shape (n_nodes, n_draws),
        nodes in ``G.nodes()`` order; the groups are the sampler's own labels,
        not aligned across draws (see :meth:`consensus_partition`).  Any other
        column of :meth:`to_frame` (a network parameter such as
        ``"individual_depth"``, or a description-length term) gives shape
        (n_draws,).
        """
        df = self._native.samples_df()
        n = len(self._node_order)
        if name in ("scores", "groups"):
            suffix = "_score" if name == "scores" else "_group"
            return df[[f"{i}{suffix}" for i in range(n)]].to_numpy().T
        if name not in df.columns or name.endswith(("_score", "_group")):
            raise KeyError(f"{name!r} is not a quantity of the fit; use 'scores', 'groups' or a "
                           "column of to_frame().")
        return df[name].to_numpy()

    def diagnostics_df(self) -> pd.DataFrame:
        """MCMC convergence diagnostics (Vehtari et al. 2021), as ArviZ's
        ``ess(method="bulk")``, ``ess(method="tail")`` and ``rhat()``, computed
        in C++ (``src/diagnostics.h``) and shared with the R package.

        Returns
        -------
        pandas.DataFrame
            One row per network-level parameter (the rows of
            :meth:`network_properties_df`), then ``H_A_b`` (the description
            length of the partition, see :meth:`partition_ess`), then
            ``scores_median`` and ``relative_scores_median``: the median over
            nodes of the diagnostics of each node's score, and of each score
            minus the mean score of its sample (the overall level of the scores
            is pinned only by their prior and wanders slowly; the differences,
            which every prediction uses, mix faster).  Columns ``quantity``,
            ``ess_bulk``, ``ess_tail`` and ``rhat``.  Chains are truncated to
            the shortest; R-hat needs two chains (NaN otherwise), and a fixed
            parameter has NaN R-hat.  Per-node values:
            :meth:`score_diagnostics_df`.
        """
        d = pd.DataFrame(self._native.diagnostics())
        rows = [d]
        for label, relative in (("scores_median", False), ("relative_scores_median", True)):
            s = pd.DataFrame(self._native.score_diagnostics(relative))
            rows.append(pd.DataFrame({"quantity": [label], **{c: [s[c].median()] for c in s.columns}}))
        return pd.concat(rows, ignore_index=True)

    def score_diagnostics_df(self, relative: bool = False) -> pd.DataFrame:
        """Convergence diagnostics of each node's score (see
        :meth:`diagnostics_df`): one row per node, columns ``node``,
        ``ess_bulk``, ``ess_tail``, ``rhat``.  ``relative=True``: of the score
        minus the mean score of its sample."""
        d = pd.DataFrame(self._native.score_diagnostics(bool(relative)))
        d.insert(0, "node", list(self._node_order))
        return d

    def direction_loo_df(self, r_eff: float = 1.0) -> pd.DataFrame:
        """Leave-one-out cross-validation of the edge directions (PSIS-LOO).

        Each edge of ``G`` in turn has its outcome left out (who won, or that it
        was a draw); the edge itself stays in the network.  The scores and the
        draw parameter make the outcomes independent, so the leave-one-out
        posterior is the posterior reweighted by 1 / p(outcome), with the weights
        Pareto-smoothed (Vehtari, Gelman & Gabry 2017) as in ArviZ's ``loo``,
        whose numbers it reproduces.  Computed in C++ (``src/loo.h``), shared
        with the R package.

        An edge of weight w counts w times and is left out whole: pass one edge
        per interaction (a MultiDiGraph) for leave-one-interaction-out.

        Parameters
        ----------
        r_eff : float
            Relative efficiency of the samples, which sets the length of the
            smoothed tail.  1 (default) treats the samples as independent, as
            ArviZ does when it has no parameters to estimate it from.

        Returns
        -------
        pandas.DataFrame
            One row per edge of ``G``, in the order of ``G.edges()``:

            - ``winner``, ``loser``: the edge's ends, the winner first (for a
              draw, as in ``G``);
            - ``draw``, ``weight``;
            - ``elpd``: leave-one-out log predictive density of the outcome
              (``elpd.sum()`` is the model's total);
            - ``pareto_k``: above 0.7 that edge's estimate is unreliable;
            - ``p_win``: decided edges, the leave-one-out probability that the
              winner beats the loser given that the interaction is decided
              (plain Bradley-Terry); NaN for draws.

            Two models fitted to the same graph compare edge by edge: the
            difference of ``elpd`` sums, with standard error
            ``sqrt(n) * std(elpd_1 - elpd_2)``.
        """
        if self._interactions is None:
            raise ValueError("This result has no edges recorded (pickled by an older version); refit "
                             "to cross-validate.")
        if self._native.num_samples == 0:
            raise ValueError("The fit has no samples.")
        e = self._interactions
        r = self._native.direction_loo(e["a"], e["b"], e["draw"], e["weight"], float(r_eff))
        nodes = list(self._node_order)
        return pd.DataFrame(dict(
            winner=[nodes[i] for i in e["a"]], loser=[nodes[i] for i in e["b"]],
            draw=np.asarray(e["draw"], dtype=bool), weight=np.asarray(e["weight"], dtype=float),
            elpd=r["elpd"], pareto_k=r["pareto_k"], p_win=r["p_win"]))

    def direction_probabilities(self, node_a, node_b, ties_allowed: bool = True) -> dict[str, float]:
        """Posterior-predictive probabilities of the outcome of a new
        interaction between ``node_a`` and ``node_b``.

        Parameters
        ----------
        node_a, node_b : hashable
            Node labels exactly as in the graph passed to :func:`fit`.
        ties_allowed : bool
            True (default): include a draw, with the sampled
            ``ties_parameter`` (always 0 under ``hierarchy_model=
            "bradley_terry"``).  False: plain Bradley-Terry, conditioned on
            the interaction being decided; the result has no ``"tie"`` key.

        Returns
        -------
        dict
            ``"a_to_b"``: probability that ``node_a`` beats (ranks above)
            ``node_b``; ``"tie"``: probability of a draw (only with
            ``ties_allowed=True``); ``"b_to_a"``: probability that ``node_b``
            beats ``node_a``.  The values are averaged over the samples and
            sum to 1.  "Beats" is the win of the model, so with
            ``source_high_rank=False`` (e.g. citations) ``"a_to_b"`` is the
            probability of an edge from ``node_b`` to ``node_a``.

        Raises
        ------
        ValueError
            If either node is not in the fitted graph.
        """
        nodes = list(self._node_order)
        for name, node in (("node_a", node_a), ("node_b", node_b)):
            if node not in nodes:
                raise ValueError(f"{name}={node!r} not found in fitted graph")
        probs = self._native.direction_probabilities(
            nodes.index(node_a), nodes.index(node_b), ties_allowed
        )
        if ties_allowed:
            return {"a_to_b": probs[0], "tie": probs[1], "b_to_a": probs[2]}
        return {"a_to_b": probs[0], "b_to_a": probs[1]}

    def __getstate__(self) -> dict:
        return {"native": self._native, "node_order": self._node_order, "config": self._config,
                "interactions": self._interactions}

    def __setstate__(self, state: dict) -> None:
        self._native = state["native"]
        self._node_order = state["node_order"]
        self._config = state["config"]
        self._interactions = state.get("interactions")   # absent in results pickled by 0.5

    def __repr__(self) -> str:
        s = dict(self._native.summary())
        n = len(self._node_order)
        return (
            f"community_hierarchy.Result("
            f"n={n}, "
            f"samples={s['num_samples']}, "
            f"mean_groups={s['mean_num_groups']:.1f}, "
            f"mdl={s['mdl_value']:.3f}"
            f")"
        )


def fit(
    config: Config,
    G: nx.DiGraph,
    *,
    group_graph: Optional[nx.Graph] = None,
    source_high_rank: bool = True,
    num_chains: int = 1,
    num_samples: int = 1000,
    num_warmup: int = 500,
    num_thin: int = 1,
    sweeps_per_sample: int = 10,
    num_param_warmup: int = 0,
    beta: float = 1.0,
    num_tempering_chains: int = 1,
    merge_split_enabled: bool = True,
    initial_partition: Optional[list] = None,
    initial_scores: Optional[list] = None,
    initial_individual_depth: Optional[float] = None,
    initial_group_depth: Optional[float] = None,
    num_groups_fixed: bool = False,
    seed: Optional[int] = None,
    timeout: float = 60.0,
    show_progress: bool = True,
) -> Result:
    """Sample the posterior of the community and hierarchy model for a directed graph.

    Parameters
    ----------
    config : Config
        Model specification.
    G : networkx.DiGraph or networkx.MultiDiGraph
        The interactions.  Each edge is one interaction, or ``weight``
        interactions (parallel edges are summed).  An edge with
        ``type="neutral"`` is a draw (a tie, a reciprocated nomination); its
        direction is ignored.  Any other edge (``type="dominant"``, or no
        ``type`` on any edge) is a decided interaction, read according to
        ``source_high_rank``.  If any edge has a ``type``, every edge must.
        Draws count towards the community structure as well as the draw
        rate.  ``weight`` must be a positive integer (a multiplicity), or a
        finite positive number with ``Config(weighted_SBM=True)``.
        Self-loops are not allowed.  The order of ``G.nodes()`` is the node
        order of every array in the result.
    group_graph : networkx.Graph, optional
        Two-layer mode: an undirected graph on the same node set whose edges
        (counted as binary, weights finite and positive) drive the
        communities, while ``G`` drives only the scores.  May be a
        MultiGraph (parallel edges are merged, their weights summed).
    source_high_rank : bool
        How to read a decided edge u -> v.  True (default): the source wins,
        u ranks above v (an edge from winner to loser, as in a sports result
        or a dominance interaction).  False: the target wins, v ranks above u
        (a citation of v by u, an unreciprocated nomination of v by u).
        Either way higher scores mean higher rank.
    num_chains, num_samples, num_warmup, num_thin : int
        The run, named as in PyStan.  One iteration is ``sweeps_per_sample``
        sweeps (a sweep is n proposals).  Each of ``num_chains`` independent
        chains (>= 1, run in parallel threads; default 1) runs ``num_warmup``
        warm-up iterations (>= 0, default 500), during which the proposal step
        sizes adapt, then ``num_samples`` sampling iterations (>= 1, default
        1000), recording the state after every ``num_thin``-th (>= 1, default
        1): ``ceil(num_samples / num_thin)`` draws per chain.
    sweeps_per_sample : int
        Sweeps per iteration (>= 0).  Default 10.  0 records the starting state
        without moving it (with ``num_warmup=0``: evaluates the description
        length of a given ``initial_partition`` / ``initial_scores``).
    num_param_warmup : int
        Iterations before warm-up with the partition and scores frozen, so
        the continuous parameters settle first (>= 0).  Default 0.
    beta : float
        Inverse temperature of the data likelihood (0 = prior, 1 =
        posterior; >= 0).  Default 1.0.
    num_tempering_chains : int
        Parallel tempering replicas per chain (>= 1) on a geometric ladder
        up to ``beta``; only the ``beta`` replica is recorded.  Default 1.
    merge_split_enabled : bool
        Collective merge/split moves (Peixoto arXiv:2003.07070).  Default True.

        Without them, single-site MCMC can converge silently to the WRONG
        answer: on the N=115 football network four of six chains get stuck
        (Gelman-Rubin R-hat 3.37, B_e 8.97 against the true 10.25) while each
        individual chain looks perfectly well-mixed from the inside. Enabling
        them costs roughly 5x on networks single-site already handles.
    initial_partition : sequence of int, optional
        Starting community labels in ``G.nodes()`` order, using every label
        0..K-1.  Default: greedy modularity on the undirected network (or on
        ``group_graph``), with groups smaller than 5% of the nodes merged
        into their best-connected neighbour.
    initial_scores : sequence of float, optional
        Starting scores in ``G.nodes()`` order, used as given; the initial
        ``individual_depth`` is then their RMS unless set.  Default:
        ``(wins - losses) / (wins + losses + draws)`` over distinct opponents.
    initial_individual_depth, initial_group_depth : float, optional
        Starting values (> 0 and >= 0) of the inferred sigma_I and sigma_G.
        Default: drawn from the seed.
    num_groups_fixed : bool
        Hold the number of groups at that of the initial partition.
        (``Config(num_groups=k)`` fixes it at k.)  Default False.
    seed : int, optional
        Random seed in [0, 2**30); chain c uses ``seed + c``.  Default: drawn
        from numpy's global generator.
    timeout : float
        Wall-clock limit in seconds (> 0).  When it expires the fit stops and
        returns the samples recorded so far, with a ``RuntimeWarning``.
        Default 60.0.
    show_progress : bool
        Print progress (warm-up and sampling) to stderr.  Default True.

    Returns
    -------
    Result

    Raises
    ------
    TypeError, ValueError
        On invalid arguments (see each parameter).
    KeyboardInterrupt
        If interrupted with Ctrl+C.
    """
    if not isinstance(config, Config):
        raise TypeError("config must be an instance of Config")
    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a networkx.Graph")

    _check_fit_arguments(
        G,
        num_groups=None if config.num_groups is _UNSET else config.num_groups,
        initial_partition=initial_partition,
        initial_scores=initial_scores,
        initial_individual_depth=initial_individual_depth,
        initial_group_depth=initial_group_depth,
        seed=seed,
        num_chains=num_chains,
        num_samples=num_samples,
        num_warmup=num_warmup,
        num_thin=num_thin,
        sweeps_per_sample=sweeps_per_sample,
        num_param_warmup=num_param_warmup,
        beta=beta,
        num_tempering_chains=num_tempering_chains,
        timeout=timeout,
        show_progress=show_progress,
    )
    for name, val in (("source_high_rank", source_high_rank),
                      ("merge_split_enabled", merge_split_enabled),
                      ("num_groups_fixed", num_groups_fixed)):
        if not isinstance(val, bool):
            raise TypeError(f"{name} must be True or False, got {val!r}")
    if initial_partition is not None:
        initial_partition = [int(v) for v in initial_partition]
    if initial_scores is not None:
        initial_scores = [float(v) for v in initial_scores]

    node_group_list, node_score_list = _get_node_data(
        G,
        group_attribute=config.group_attribute,
        hierarchy_attribute=config.hierarchy_attribute,
    )

    partition_fixed = False
    if node_group_list is not None:
        if initial_partition is not None:
            logger.warning("initial_partition will be overwritten by node attribute data.")
        initial_partition = node_group_list
        partition_fixed = True

    scores_fixed = False
    if node_score_list is not None:
        if initial_scores is not None:
            logger.warning("initial_scores will be overwritten by node attribute data.")
        initial_scores = node_score_list
        scores_fixed = True

    # The interactions, summed into wins and draws in G's node order (C++ core,
    # src/interactions.h).
    node_order = tuple(G.nodes())
    index = {node: i for i, node in enumerate(node_order)}
    interactions = _build_interactions(
        len(node_order), *_edges(G, index, integer_weights=not config.weighted_SBM),
        source_high_rank=source_high_rank)

    # The model's parameters from the presets and the explicit values (C++ core).
    # An unset hierarchy_model models draws only when the graph has neutral edges:
    # without one, ties are not an outcome the data could record.
    _check_model_for_graph(config, interactions.has_neutral, len(node_order))
    model = _resolve_model(**config._model_choice(), has_neutral=interactions.has_neutral)

    assortative = model["assortative"]
    variation_in, variation_out = model["variation_in"], model["variation_out"]
    degree_correction = model["degree_correction"]
    individual_depth, group_depth = model["individual_depth"], model["group_depth"]
    ties_parameter, num_groups = model["ties_parameter"], model["num_groups"]

    if seed is None:
        seed = int(np.random.randint(0, 1_000_000))
    seed = int(seed)

    # Two-layer mode: the group layer as an adjacency list in G's node order.
    if group_graph is not None:
        group_adj = _build_group_layer(len(node_order), *_group_graph_edges(G, group_graph, index))
    else:
        group_adj = []  # empty -> single-layer mode in C++

    # An empty initial_partition / initial_scores asks the C++ core for the default
    # starting state (greedy modularity; win fractions), the same in every binding.
    # Given scores that are not fixed start sigma_I at their RMS (also in C++).
    if initial_partition is None:
        initial_partition = []
    if initial_scores is None:
        initial_scores = []

    native_result = _fit_backend(
        wins=interactions.wins,
        ties=interactions.ties,
        group_adj=group_adj,
        initial_partition=list(initial_partition),
        partition_fixed=bool(partition_fixed),
        initial_scores=list(initial_scores),
        scores_fixed=bool(scores_fixed),
        assortative=bool(assortative),
        variation_in=None if variation_in is None else float(variation_in),
        variation_out=None if variation_out is None else float(variation_out),
        degree_correction=None if degree_correction is None else float(degree_correction),
        individual_depth=None if individual_depth is None else float(individual_depth),
        initial_individual_depth=None if initial_individual_depth is None else float(initial_individual_depth),
        group_depth=None if group_depth is None else float(group_depth),
        initial_group_depth=None if initial_group_depth is None else float(initial_group_depth),
        ties_parameter=None if ties_parameter is None else float(ties_parameter),
        num_groups=None if num_groups is None else int(num_groups),
        num_groups_fixed=bool(num_groups_fixed),
        num_chains=int(num_chains),
        num_samples=int(num_samples),
        num_thin=int(num_thin),
        num_warmup=int(num_warmup),
        num_param_warmup=int(num_param_warmup),
        sweeps_per_sample=int(sweeps_per_sample),
        beta=float(beta),
        num_tempering_chains=int(num_tempering_chains),
        merge_split_enabled=bool(merge_split_enabled),
        seed=seed,
        timeout=float(timeout),
        show_progress=bool(show_progress),
        weighted_SBM=bool(config.weighted_SBM),
        weighted_proposals=bool(config.weighted_proposals),
    )

    # A timeout is enforced by simply breaking out of the sweep loops (see
    # fit.cpp), so an expired run returns normally, with fewer samples than
    # asked for -- or with NONE, if the clock ran out during burn-in. A
    # zero-sample result is built from nothing and comes back looking like a
    # converged fit: one group, every score exactly 0.0. That has to be said
    # out loud rather than left for the caller to notice.
    expected_samples = int(num_chains) * -(-int(num_samples) // int(num_thin))
    recorded_samples = int(getattr(native_result, "num_samples", expected_samples))
    if recorded_samples < expected_samples:
        if recorded_samples == 0:
            warnings.warn(
                f"community_hierarchy: the {timeout:g}s timeout expired during burn-in, "
                f"before a single sample was recorded. The returned result is NOT a fit: "
                f"it will show one group and all-zero scores. Pass a larger timeout= "
                f"(it defaults to 60s) or reduce num_warmup/num_samples.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"community_hierarchy: the {timeout:g}s timeout expired after "
                f"{recorded_samples} of {expected_samples} samples. The result is based "
                f"only on those, so it is noisier than requested and may not have "
                f"converged. Pass a larger timeout= to collect the full run.",
                RuntimeWarning,
                stacklevel=2,
            )

    edges = {"a": interactions.a, "b": interactions.b, "draw": interactions.draw,
             "weight": interactions.weight}
    return Result(native_result, node_order, config, edges)
