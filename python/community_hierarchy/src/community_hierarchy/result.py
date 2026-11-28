"""The result of a fit: the samples, held by the C++ FitResult, and their summaries."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .MCMC_core import sort_groups_by_score as _sort_groups_by_score

if TYPE_CHECKING:
    from .config import Config


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

    def __init__(
        self, native_result, node_order: tuple, config: Config, interactions: dict | None = None
    ) -> None:
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

    def consensus_partition(
        self, method: str = "moc", refine: bool = True, min_samples: int = 1000
    ) -> np.ndarray:
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

    def node_properties_df(
        self, partition: str = "auto", consensus_method: str = "moc"
    ) -> pd.DataFrame:
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
            raise ValueError(
                f"consensus_method must be 'moc', 'moc_modal_k' or 'binder', "
                f"got {consensus_method!r}"
            )
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
                    f"partition must be one of {sorted(_valid)!r} or 'auto', got {partition!r}"
                )
            mode = partition
        d = dict(self._native.node_properties(mode, cm))
        d.pop("mode_used", None)
        d["node"] = list(self._node_order)
        # Reorder columns: node first, then the rest
        return pd.DataFrame(
            d, columns=["node", "score_mean", "score_std", "group", "group_score_mean"]
        )

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
            raise KeyError(
                f"{name!r} is not a quantity of the fit; use 'scores', 'groups' or a "
                "column of to_frame()."
            )
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
            rows.append(
                pd.DataFrame({"quantity": [label], **{c: [s[c].median()] for c in s.columns}})
            )
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
            raise ValueError(
                "This result has no edges recorded (pickled by an older version); refit "
                "to cross-validate."
            )
        if self._native.num_samples == 0:
            raise ValueError("The fit has no samples.")
        e = self._interactions
        r = self._native.direction_loo(e["a"], e["b"], e["draw"], e["weight"], float(r_eff))
        nodes = list(self._node_order)
        return pd.DataFrame(
            dict(
                winner=[nodes[i] for i in e["a"]],
                loser=[nodes[i] for i in e["b"]],
                draw=np.asarray(e["draw"], dtype=bool),
                weight=np.asarray(e["weight"], dtype=float),
                elpd=r["elpd"],
                pareto_k=r["pareto_k"],
                p_win=r["p_win"],
            )
        )

    def direction_probabilities(
        self, node_a, node_b, ties_allowed: bool = True
    ) -> dict[str, float]:
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
        return {
            "native": self._native,
            "node_order": self._node_order,
            "config": self._config,
            "interactions": self._interactions,
        }

    def __setstate__(self, state: dict) -> None:
        self._native = state["native"]
        self._node_order = state["node_order"]
        self._config = state["config"]
        self._interactions = state.get("interactions")  # absent in results pickled by 0.5

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
