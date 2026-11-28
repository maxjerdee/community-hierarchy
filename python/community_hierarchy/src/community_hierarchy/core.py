from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional
import logging
import warnings

import numpy as np
import networkx as nx
import pandas as pd

from .input_output import (
    _UNSET,
    _check_forced_values,
    _check_group_graph,
    _check_input_validity,
    _get_node_data,
    _set_model_parameters,
    get_neighbors,
)
from .MCMC_core import fit_backend as _fit_backend

logger = logging.getLogger(__name__)


def _hdi(arr: np.ndarray, prob: float = 0.90) -> tuple[float, float]:
    """Highest Density Interval for a 1-D array of posterior samples."""
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    interval_idx = int(np.floor(prob * n))
    n_intervals = n - interval_idx
    widths = sorted_arr[interval_idx:] - sorted_arr[:n_intervals]
    min_idx = int(np.argmin(widths))
    return float(sorted_arr[min_idx]), float(sorted_arr[min_idx + interval_idx])

_VALID_GROUPS_MODELS = frozenset({
    "traditional_SBM",
    "simple_ASBM",
    "hybrid_ASBM",
    "planted_partition",
    "general_ASBM",
})

_VALID_HIERARCHY_MODELS = frozenset({"bradley_terry", "bradley_terry_ties"})
_VALID_INTERACTIONS = frozenset({"coupled", "independent"})
_VALID_MIXING_VARIATIONS = frozenset({"general", "simple", "none", "internal", "external"})


@dataclass(frozen=True)
class Config:
    """Model configuration for community and hierarchy inference.

    Every field has a default, so ``Config()`` is the full coupled model with
    everything inferred.  Fields left unset take the value of the
    ``groups_model`` / ``hierarchy_model`` / ``interaction`` presets.

    Parameters
    ----------
    groups_model : {"general_ASBM", "simple_ASBM", "hybrid_ASBM", "planted_partition", "traditional_SBM"}
        Community structure preset; sets ``mixing_variation`` (and, for
        ``traditional_SBM``, ``assortative=False``) unless those are given:

        ========================  ====================  ===============
        preset                    mixing_variation      assortative
        ========================  ====================  ===============
        ``"general_ASBM"``        ``"general"``         True
        ``"simple_ASBM"``         ``"simple"``          True
        ``"hybrid_ASBM"``         ``"internal"``        True
        ``"planted_partition"``   ``"none"``            True
        ``"traditional_SBM"``     ``"simple"``          False
        ========================  ====================  ===============

        Default ``"general_ASBM"``.
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

    groups_model: str = "general_ASBM"
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
        if self.groups_model == "general_canonical":
            # Removed: it was general_ASBM under another name, while asbm's reference
            # notebook uses "general_canonical" for a different model (one that also
            # integrates the degree-correction strength).
            raise ValueError(
                "groups_model 'general_canonical' was renamed 'general_ASBM' "
                "(the same model; it is also the default)."
            )
        if self.groups_model not in _VALID_GROUPS_MODELS:
            raise ValueError(
                f"Unknown groups_model '{self.groups_model}'. "
                f"Valid options: {sorted(_VALID_GROUPS_MODELS)}. "
                "Degree correction is not part of the model name; set it with "
                "Config(degree_correction=True | False)."
            )
        for name in ("degree_correction", "weighted_SBM", "weighted_proposals"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be True or False, got {getattr(self, name)!r}")
        if self.assortative is not _UNSET and not isinstance(self.assortative, bool):
            raise ValueError(f"assortative must be True or False, got {self.assortative!r}")
        if self.mixing_variation is not _UNSET and self.mixing_variation not in _VALID_MIXING_VARIATIONS:
            raise ValueError(
                f"Unknown mixing_variation {self.mixing_variation!r}. "
                f"Valid options: {sorted(_VALID_MIXING_VARIATIONS)}"
            )
        if self.hierarchy_model is not _UNSET and self.hierarchy_model not in _VALID_HIERARCHY_MODELS:
            raise ValueError(
                f"Unknown hierarchy_model '{self.hierarchy_model}'. "
                f"Valid options: {sorted(_VALID_HIERARCHY_MODELS)}"
            )
        if self.interaction not in _VALID_INTERACTIONS:
            raise ValueError(
                f"Unknown interaction '{self.interaction}'. "
                f"Valid options: {sorted(_VALID_INTERACTIONS)}"
            )
        for name in ("group_attribute", "hierarchy_attribute"):
            val = getattr(self, name)
            if val is not None and (not isinstance(val, str) or not val):
                raise ValueError(f"{name} must be a non-empty string or None, got {val!r}")
        # Values the presets force (fixed later in _set_model_parameters).  An
        # explicit, different value used to be overwritten without a word.
        _check_forced_values(self.hierarchy_model, self.interaction,
                             self.group_depth, self.ties_parameter)


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

    def __init__(self, native_result, node_order: tuple, config: Config) -> None:
        self._native = native_result
        self._node_order = node_order
        self._config = config

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

    @staticmethod
    def _sort_groups_by_score(partition: np.ndarray, means: np.ndarray) -> np.ndarray:
        """Relabel group ids so group 0 has the lowest mean score."""
        group_means = {}
        for node_idx, g in enumerate(partition):
            group_means.setdefault(g, []).append(means[node_idx])
        ranked = sorted(group_means, key=lambda g: np.mean(group_means[g]))
        remap = {old: new for new, old in enumerate(ranked)}
        return np.array([remap[g] for g in partition], dtype=int)

    @property
    def mdl_partition(self) -> np.ndarray:
        """Partition of the sample with the smallest description length,
        shape (n_nodes,), in ``G.nodes()`` order.  Groups are relabelled so
        group 0 has the lowest mean posterior score.  Not a posterior summary;
        see :meth:`consensus_partition`."""
        part = np.asarray(self._native.mdl_partition(), dtype=int)
        return self._sort_groups_by_score(part, self.score_means)

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
        return self._sort_groups_by_score(part, self.score_means)

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

    @property
    def scores_df(self) -> pd.DataFrame:
        """Posterior hierarchy scores per node: columns ``node``, ``mean``, ``std``.

        .. deprecated::
            Use :meth:`node_properties_df` instead, which also includes the
            community partition and group mean score.
        """
        warnings.warn(
            "scores_df is deprecated. Use node_properties_df() instead, "
            "which includes the partition and group_score_mean.",
            DeprecationWarning,
            stacklevel=2,
        )
        return pd.DataFrame({
            "node": list(self._node_order),
            "mean": np.asarray(self._native.score_means(), dtype=float),
            "std": np.asarray(self._native.score_stds(), dtype=float),
        })

    @property
    def parameter_df(self) -> pd.DataFrame:
        """Posterior summary of network-level parameters (mean + std only).

        .. deprecated::
            Use :meth:`network_properties_df` instead, which also includes
            HDI credible intervals and ICC / sigma_T.
        """
        warnings.warn(
            "parameter_df is deprecated. Use network_properties_df() instead, "
            "which adds hdi_lo/hdi_hi and includes ICC and sigma_T.",
            DeprecationWarning,
            stacklevel=2,
        )
        means = self._native.parameter_means()
        stds = self._native.parameter_stds()
        parameters = [
            "num_groups",
            "density_in",
            "density_out",
            "variation_in",
            "variation_out",
            "degree_correction",
            "individual_depth",
            "group_depth",
            "ties_parameter",
        ]
        return pd.DataFrame({
            "parameter": parameters,
            "mean": [means[p] for p in parameters],
            "std": [stds[p] for p in parameters],
        })

    # ------------------------------------------------------------------
    # New posterior-summary methods
    # ------------------------------------------------------------------

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
        ``samples_df()``; efficient even for large networks.

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

    def samples_df(self) -> pd.DataFrame:
        """All recorded samples, one row per sample (chains concatenated).

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
        df = df.rename(columns=rename)
        sigma_C = df["group_depth"]
        sigma_I = df["individual_depth"]
        sigma_T = np.sqrt(sigma_C ** 2 + sigma_I ** 2)
        df["sigma_T"] = sigma_T
        df["ICC"] = np.where(sigma_T > 0, sigma_C ** 2 / sigma_T ** 2, 0.0)
        return df

    def summary_df(self, prob: float = 0.90) -> pd.DataFrame:
        """Posterior summary of network-level parameters with HDI credible intervals.

        .. deprecated::
            Use :meth:`network_properties_df` instead. It returns the same
            information computed in C++ (faster), adds ``std``, and drops
            the ``median`` column.
        """
        warnings.warn(
            "summary_df() is deprecated. Use network_properties_df() instead — "
            "same content, computed in C++ without materialising samples_df(), "
            "with std added and median dropped.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.network_properties_df(prob=prob)


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

    @property
    def summary(self) -> dict:
        """High-level summary dict.

        .. deprecated::
            Use ``result.mdl_value``, ``result.network_properties_df()``, or
            ``result._native.num_samples`` directly.
        """
        warnings.warn(
            "summary is deprecated. Use result.mdl_value and "
            "network_properties_df() to access this information.",
            DeprecationWarning,
            stacklevel=2,
        )
        return dict(self._native.summary())

    def __getstate__(self) -> dict:
        return {"native": self._native, "node_order": self._node_order, "config": self._config}

    def __setstate__(self, state: dict) -> None:
        self._native = state["native"]
        self._node_order = state["node_order"]
        self._config = state["config"]

    def __repr__(self) -> str:
        s = dict(self._native.summary())  # bypass DeprecationWarning
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
    samples_per_chain: int = 1000,
    sweeps_per_sample: int = 10,
    burn_sweeps: int = 500,
    param_warmup_sweeps: int = 0,
    thinning: int = 1,
    beta: float = 1.0,
    num_tempering_chains: int = 1,
    merge_split_enabled: bool = True,
    initial_partition: Optional[list] = None,
    initial_scores: Optional[list] = None,
    initial_individual_depth: Optional[float] = None,
    initial_group_depth: Optional[float] = None,
    num_groups_fixed: bool = False,
    seed: Optional[int] = None,
    no_cache: bool = False,
    timeout: float = 60.0,
    show_progress: bool = True,
    # Legacy aliases
    num_samples: Optional[int] = None,
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
    num_chains : int
        Independent chains, run in parallel threads (>= 1).  Default 1.
    samples_per_chain : int
        Samples recorded per chain (>= 1).  Default 1000.
    sweeps_per_sample : int
        MCMC sweeps (n proposals each) between recorded samples.  Default 10.
        0 records the starting state without moving it (with
        ``burn_sweeps=0``: evaluates the description length of a given
        ``initial_partition`` / ``initial_scores``).
    burn_sweeps : int
        Burn-in length per chain (>= 0), in units of ``sweeps_per_sample``
        sweeps: the burn-in is ``burn_sweeps * sweeps_per_sample`` sweeps
        (5000 with the defaults).  Proposal step sizes adapt during burn-in
        and are then frozen.  Default 500.
    param_warmup_sweeps : int
        Before burn-in, this many blocks of ``sweeps_per_sample`` sweeps with
        the partition and scores frozen, so the continuous parameters settle
        first (>= 0).  Default 0.
    thinning : int
        Record every ``thinning``-th block: ``sweeps_per_sample * thinning``
        sweeps between samples (>= 1).  Default 1.
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
    no_cache : bool
        Ignored; kept for backward compatibility.
    timeout : float
        Wall-clock limit in seconds (> 0).  When it expires the fit stops and
        returns the samples recorded so far, with a ``RuntimeWarning``.
        Default 60.0.
    show_progress : bool
        Print progress (warm-up and sampling) to stderr.  Default True.
    num_samples : int, optional
        Legacy alias for ``samples_per_chain``.

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
    if num_samples is not None:
        samples_per_chain = num_samples
    if not isinstance(config, Config):
        raise TypeError("config must be an instance of Config")
    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a networkx.Graph")

    _check_input_validity(
        G=G,
        group_attribute=config.group_attribute,
        hierarchy_attribute=config.hierarchy_attribute,
        groups_model=config.groups_model,
        hierarchy_model=config.hierarchy_model,
        interaction=config.interaction,
        assortative=config.assortative,
        mixing_variation=config.mixing_variation,
        variation_in=config.variation_in,
        variation_out=config.variation_out,
        degree_correction=config.degree_correction,
        individual_depth=config.individual_depth,
        group_depth=config.group_depth,
        ties_parameter=config.ties_parameter,
        num_groups=config.num_groups,
        weighted_SBM=config.weighted_SBM,
        initial_partition=initial_partition,
        initial_scores=initial_scores,
        initial_individual_depth=initial_individual_depth,
        initial_group_depth=initial_group_depth,
        seed=seed,
        num_chains=num_chains,
        samples_per_chain=samples_per_chain,
        sweeps_per_sample=sweeps_per_sample,
        burn_sweeps=burn_sweeps,
        param_warmup_sweeps=param_warmup_sweeps,
        thinning=thinning,
        beta=beta,
        num_tempering_chains=num_tempering_chains,
        no_cache=no_cache,
        timeout=timeout,
        verbose=show_progress,
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

    node_group_list, _, node_score_list = _get_node_data(
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

    # Resolve an omitted hierarchy model from the data: ties are modelled only
    # when the graph has neutral edges.  Without one, ties are not an outcome
    # the data could record, so plain Bradley-Terry (ties_parameter fixed at 0)
    # is the right model, not a tie rate fitted to "no ties seen".
    hierarchy_model = config.hierarchy_model
    if hierarchy_model is _UNSET:
        has_neutral = any(d.get("type") == "neutral" for _, _, d in G.edges(data=True))
        hierarchy_model = "bradley_terry_ties" if has_neutral else "bradley_terry"
        if config.ties_parameter is not _UNSET and config.ties_parameter not in (0, 0.0):
            raise ValueError(
                f"ties_parameter={config.ties_parameter!r} was given, but G has no "
                "neutral edges, so hierarchy_model resolves to 'bradley_terry', which "
                "fixes ties_parameter at 0. Pass hierarchy_model='bradley_terry_ties' "
                "to fit a draw rate anyway."
            )

    # Split edges into wins / ties graphs, in G's node order.
    G_wins = nx.DiGraph()
    G_ties = nx.Graph()
    G_wins.add_nodes_from(G.nodes(data=True))
    G_ties.add_nodes_from(G.nodes(data=True))
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1.0)
        if d.get("type") == "neutral":
            if G_ties.has_edge(u, v):
                G_ties[u][v]["weight"] += w
            else:
                G_ties.add_edge(u, v, weight=w)
        else:
            winner, loser = (u, v) if source_high_rank else (v, u)
            if G_wins.has_edge(winner, loser):
                G_wins[winner][loser]["weight"] += w
            else:
                G_wins.add_edge(winner, loser, weight=w)

    # Resolve model parameters
    (
        assortative,
        mixing_variation,
        variation_in,
        variation_out,
        degree_correction,
        individual_depth,
        group_depth,
        ties_parameter,
        num_groups,
    ) = _set_model_parameters(
        config.groups_model,
        hierarchy_model,
        config.interaction,
        config.assortative,
        config.mixing_variation,
        config.variation_in,
        config.variation_out,
        config.degree_correction,
        config.individual_depth,
        config.group_depth,
        config.ties_parameter,
        config.num_groups,
    )

    if seed is None:
        seed = int(np.random.randint(0, 1_000_000))
    seed = int(seed)

    node_order = tuple(G.nodes())

    # Two-layer mode: validate and convert group_graph to an adjacency list
    # indexed with the same node ordering as G.
    if group_graph is not None:
        _check_group_graph(G, group_graph)
        group_adj = get_neighbors(group_graph, node_order=list(node_order))
    else:
        group_adj = []  # empty -> single-layer mode in C++

    # Generate initial state if not provided
    if initial_partition is None:
        initial_partition = _gen_initial_partition(
            G_wins, G_ties, num_groups, group_graph=group_graph, node_order=node_order
        )
    if initial_scores is None:
        initial_scores = _gen_initial_scores(G_wins, G_ties)
    elif not scores_fixed:
        # Start sigma_I at the RMS of the user-provided scores, so it is on the scale
        # of the scores, which are used as given.  individual_depth remains free.
        scores_arr = np.asarray(initial_scores, dtype=float)
        src_rms = float(np.sqrt(np.mean(scores_arr ** 2)))
        if src_rms > 1e-6 and initial_individual_depth is None and individual_depth is None:
            initial_individual_depth = src_rms

    wins_adj = get_neighbors(G_wins)
    ties_adj = get_neighbors(G_ties)

    native_result = _fit_backend(
        wins=wins_adj,
        ties=ties_adj,
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
        samples_per_chain=int(samples_per_chain),
        sweeps_per_sample=int(sweeps_per_sample),
        burn_sweeps=int(burn_sweeps),
        param_warmup_sweeps=int(param_warmup_sweeps),
        thinning=int(thinning),
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
    expected_samples = int(num_chains) * int(samples_per_chain)
    recorded_samples = int(getattr(native_result, "num_samples", expected_samples))
    if recorded_samples < expected_samples:
        if recorded_samples == 0:
            warnings.warn(
                f"community_hierarchy: the {timeout:g}s timeout expired during burn-in, "
                f"before a single sample was recorded. The returned result is NOT a fit: "
                f"it will show one group and all-zero scores. Pass a larger timeout= "
                f"(it defaults to 60s) or reduce burn_sweeps/samples_per_chain.",
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

    return Result(native_result, node_order, config)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gen_initial_partition(
    G_wins: nx.DiGraph, G_ties: nx.Graph, num_groups, seed: int = 0,
    group_graph: Optional[nx.Graph] = None, node_order=None,
) -> list:
    """Greedy-modularity starting partition, as a list in ``node_order``
    (default: the node order of ``G_wins``, which is that of ``G``)."""
    if node_order is None:
        node_order = list(G_wins.nodes())
    G_undirected = nx.Graph()
    G_undirected.add_nodes_from(node_order)
    if group_graph is not None:
        G_undirected.add_edges_from((u, v) for u, v in group_graph.edges())
    else:
        G_undirected.add_edges_from(G_wins.edges())
        G_undirected.add_edges_from(G_ties.edges())

    if num_groups is not None:
        communities = list(nx.community.greedy_modularity_communities(
            G_undirected, cutoff=num_groups, best_n=num_groups
        ))
    else:
        communities = list(nx.community.greedy_modularity_communities(G_undirected))
        # Merge communities smaller than 5% of N into their best-connected neighbour.
        # Greedy modularity tends to over-partition (resolution limit), producing many
        # small satellites around the dominant communities.  Absorbing them gives the
        # MCMC a much smaller number of groups to work with during burn-in.
        N = G_undirected.number_of_nodes()
        min_size = max(2, int(N * 0.05))
        node_to_comm = {n: ci for ci, comm in enumerate(communities) for n in comm}
        changed = True
        while changed:
            changed = False
            new_comms = list(communities)
            for ci, comm in enumerate(new_comms):
                if len(comm) < min_size:
                    edge_counts: dict = {}
                    for n in comm:
                        for nbr in G_undirected.neighbors(n):
                            cj = node_to_comm[nbr]
                            if cj != ci:
                                edge_counts[cj] = edge_counts.get(cj, 0) + 1
                    if not edge_counts:
                        continue
                    best = max(edge_counts, key=lambda k: edge_counts[k])
                    new_comms[best] = frozenset(new_comms[best] | frozenset(comm))
                    new_comms[ci] = frozenset()
                    node_to_comm.update({n: best for n in comm})
                    changed = True
            communities = [c for c in new_comms if len(c) > 0]
            node_to_comm = {n: ci for ci, comm in enumerate(communities) for n in comm}

    node_to_index = {node: i for i, node in enumerate(node_order)}
    partition = [0] * len(node_order)
    for i, community in enumerate(communities):
        for node in community:
            partition[node_to_index[node]] = i
    return partition


def _gen_initial_scores(G_wins: nx.DiGraph, G_ties: nx.Graph) -> list:
    node_to_index = {node: i for i, node in enumerate(G_wins.nodes())}
    scores = [0.0] * G_wins.number_of_nodes()
    for node in G_wins.nodes():
        wins = G_wins.out_degree(node)
        losses = G_wins.in_degree(node)
        ties = G_ties.degree(node)
        scores[node_to_index[node]] = (wins - losses) / (wins + losses + ties + 1e-6)
    return scores
