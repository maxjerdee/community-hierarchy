from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import logging
import warnings

import numpy as np
import networkx as nx
import pandas as pd

from .input_output import (
    _UNSET,
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
    "general_canonical",
    "traditional_SBM",
    "simple_ASBM",
    "hybrid_ASBM",
    "planted_partition",
    "general_ASBM",
})

_VALID_HIERARCHY_MODELS = frozenset({"bradley_terry", "bradley_terry_ties"})
_VALID_INTERACTIONS = frozenset({"coupled", "independent"})


@dataclass(frozen=True)
class Config:
    """Model configuration for directed structure inference.

    Parameters
    ----------
    groups_model : str
        Community structure model preset. Controls which parameters are fixed vs inferred.
    hierarchy_model : str
        Hierarchy model. Auto-selected from edge types if not given.
    interaction : str
        Coupling between communities and hierarchy: 'coupled' or 'independent'.
    assortative : bool
        Whether to allow assortative community structure. Default: use model preset.
    mixing_variation : str
        Variation in mixing weights ('general', 'simple', 'none', 'internal', 'external').
        Default: use model preset.
    variation_in, variation_out : float or None
        Fixed intra/inter-group mixing variation. None = infer freely. Default: model preset.
    degree_correction : bool
        True (default): degree correction with a uniform prior over each group's
        degree sequence.  False: none.  This is the only place degree correction
        is set; model names do not imply it.
    individual_depth : float or None
        Spread of individual scores within groups. None = infer. Default: model preset.
    group_depth : float or None
        Spread of group scores. None = infer. Default: model preset.
    ties_parameter : float or None
        Frequency of ties in the hierarchy. None = infer. Default: model preset.
    num_groups : int or None
        Fixed number of communities. None = infer. Default: model preset.
    group_attribute : str or None
        Node attribute name with fixed group labels.
    hierarchy_attribute : str or None
        Node attribute name with fixed hierarchy scores.
    """

    groups_model: str = "general_canonical"
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

    def __post_init__(self) -> None:
        if self.groups_model not in _VALID_GROUPS_MODELS:
            raise ValueError(
                f"Unknown groups_model '{self.groups_model}'. "
                f"Valid options: {sorted(_VALID_GROUPS_MODELS)}. "
                "Degree correction is not part of the model name; set it with "
                "Config(degree_correction=True | False)."
            )
        if not isinstance(self.degree_correction, bool):
            raise ValueError(
                f"degree_correction must be True or False, got {self.degree_correction!r}"
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


class Result:
    """Posterior inference result for a directed graph.

    Wraps the C++ FitResult and provides convenient summary methods.
    Obtain via :func:`fit`.
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
        """Minimum description length across all posterior samples."""
        return float(self._native.mdl_value)

    @property
    def mdl_partition(self) -> np.ndarray:
        """Community partition from the sample achieving the minimum description length."""
        return np.asarray(self._native.mdl_partition(), dtype=int)

    def consensus_partition(self, refine: bool = True) -> np.ndarray:
        """Representative partition: the recorded sample at the modal number of
        groups whose coincidence matrix is closest in L1 to the aggregate, then a
        greedy single-node descent on the same loss with that group count held
        fixed.  refine=False returns the unrefined argmin over samples."""
        return np.asarray(self._native.consensus_partition(refine), dtype=int)

    @property
    def score_means(self) -> np.ndarray:
        """Posterior mean hierarchy score for each node, shape (n_nodes,)."""
        return np.asarray(self._native.score_means(), dtype=float)

    @property
    def score_stds(self) -> np.ndarray:
        """Posterior std of hierarchy score for each node, shape (n_nodes,)."""
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
            HDI probability mass (default 0.90).

        Returns
        -------
        DataFrame with columns ``parameter``, ``mean``, ``std``,
        ``hdi_lo``, ``hdi_hi``. One row per parameter. Includes ICC
        and sigma_T in addition to the 9 base C++ parameters.

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
        d = dict(self._native.network_properties(prob))
        return pd.DataFrame(d, columns=["parameter", "mean", "std", "hdi_lo", "hdi_hi"])

    def node_properties_df(self, partition: str = "auto") -> pd.DataFrame:
        """Posterior summary of node-level properties with community labels.

        Parameters
        ----------
        partition : str
            Which partition to use for community labels:

            ``"auto"`` (default)
                Consensus partition for n ≤ 3000 (the recommended posterior
                summary); MDL partition for larger networks, with a
                ``UserWarning``.
            ``"consensus"``
                Always compute the consensus partition — O(n² × samples)
                time and O(n²) memory; may be slow for n > ~3000.
            ``"mdl"``
                MDL-optimal single sample; free but not a posterior summary.

        Returns
        -------
        DataFrame with columns ``node``, ``score_mean``, ``score_std``,
        ``group``, ``group_score_mean``. One row per node.

        Notes
        -----
        The ``"consensus"`` partition is the natural posterior summary for a
        Bayesian method — it is the sample whose coincidence matrix is closest
        to the aggregate. ``"mdl"`` is a free byproduct of sampling (not an
        optimisation result) and is provided as a fast fallback for large n.

        Examples
        --------
        >>> npdf = result.node_properties_df()
        >>> print(npdf.sort_values("score_mean", ascending=False).head())
        """
        # Resolve "auto" in Python so the threshold check is testable and the
        # warning fires before calling C++ (language-specific concern).
        n = len(self._node_order)
        if partition == "auto":
            mode = "consensus" if n <= 3000 else "mdl"
            if mode == "mdl":
                warnings.warn(
                    f"Network is large (n={n} > 3000); using mdl_partition "
                    "instead of consensus_partition. consensus_partition is "
                    "O(n² × samples) and may be slow for networks this size. "
                    "Pass partition='consensus' to force it, or "
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
        d = dict(self._native.node_properties(mode))
        d.pop("mode_used", None)
        d["node"] = list(self._node_order)
        # Reorder columns: node first, then the rest
        return pd.DataFrame(d, columns=["node", "score_mean", "score_std", "group", "group_score_mean"])

    # ------------------------------------------------------------------
    # Distribution summaries
    # ------------------------------------------------------------------

    def coincidence_matrix(self) -> np.ndarray:
        """Co-assignment frequency matrix, shape (n_nodes, n_nodes).

        Entry [i, j] is the fraction of samples in which nodes i and j
        are assigned to the same community. Diagonal entries are always 1.0.
        """
        return np.asarray(self._native.coincidence_matrix(), dtype=float)

    def samples_df(self) -> pd.DataFrame:
        """Raw posterior samples DataFrame.

        Columns: ``{node}_group``, ``{node}_score`` for each node;
        global parameters (``num_groups``, ``density_in``, etc.);
        derived parameters (``sigma_T``, ``ICC``);
        entropy decomposition (``H_A_b``, ``H_Adir_G_A_s_nu``,
        ``H_s_G_b_depths``, etc.).
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
        """Posterior-predictive probability of each directed outcome between node_a and node_b.

        Parameters
        ----------
        node_a, node_b : node labels exactly as they appear in the graph passed to fit.
        ties_allowed : if True (default), include tie probability using the inferred
            ties_parameter. If False, use standard Bradley-Terry (no tie term);
            the returned dict will not contain a ``"tie"`` key.

        Returns
        -------
        dict with keys ``"a_to_b"``, ``"tie"`` (only when ties_allowed=True), ``"b_to_a"``.
        Values are averaged over all posterior samples and sum to 1.0.

        Raises
        ------
        ValueError if either node is not found in the fitted graph.
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
            f"directedstructure.Result("
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
    no_cache: bool = False,  # retained for API compatibility; no longer used
    timeout: float = 60.0,
    show_progress: bool = True,
    # Legacy aliases
    num_samples: Optional[int] = None,
) -> Result:
    """Run posterior inference on a directed graph.

    Parameters
    ----------
    config : Config
        Model specification.
    G : nx.DiGraph
        Directed NetworkX graph. Edges may have a ``type`` attribute
        ('dominant' or 'neutral') and an optional ``weight``.
    num_chains : int
        Number of independent parallel chains. Default 1.
    samples_per_chain : int
        Posterior samples to collect per chain. Default 1000.
    sweeps_per_sample : int
        MCMC sweeps between recorded samples. Default 10.
    burn_sweeps : int
        Burn-in sweeps before sampling begins (per chain). Default 500.
    thinning : int
        Additional sweeps between samples (total = sweeps_per_sample * thinning). Default 1.
    beta : float
        Inverse temperature (0 = prior, 1 = posterior). Default 1.0.
    merge_split_enabled : bool
        Collective merge/split moves (Peixoto arXiv:2003.07070). Default True.

        Without them, single-site MCMC can converge silently to the WRONG
        answer: on the N=115 football network four of six chains get stuck
        (Gelman-Rubin R-hat 3.37, B_e 8.97 against the true 10.25) while each
        individual chain looks perfectly well-mixed from the inside. Enabling
        them costs roughly 5x on networks single-site already handles, and buys
        correctness that is otherwise undetectable without running multi-start
        R-hat on every fit.

        Set False to reproduce pre-merge/split results exactly.
    num_tempering_chains : int
        Parallel tempering chains within each independent chain. Default 1.
    initial_partition : list[int], optional
        Starting community assignment. Inferred via greedy modularity if None.
    initial_scores : list[float], optional
        Starting hierarchy scores. Inferred from win percentages if None.
    seed : int, optional
        Random seed for reproducibility.
    timeout : float
        Maximum wall-clock seconds to run. Default 60.0.
    show_progress : bool
        Print a running sample counter to stderr. Default False.
    num_samples : int, optional
        Legacy alias for ``samples_per_chain`` (used when ``num_chains=1``).

    Returns
    -------
    Result
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
        initial_partition=initial_partition,
        initial_scores=initial_scores,
        seed=seed,
        num_samples=samples_per_chain,
        beta=beta,
        num_tempering_chains=num_tempering_chains,
        no_cache=no_cache,
        timeout=timeout,
        verbose=show_progress,
    )

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

    # Split edges into wins / ties graphs
    G_wins = nx.DiGraph()
    G_ties = nx.Graph()
    G_wins.add_nodes_from(G.nodes(data=True))
    G_ties.add_nodes_from(G.nodes(data=True))
    hierarchy_model = config.hierarchy_model
    for u, v, d in G.edges(data=True):
        if d.get("type") == "neutral":
            if hierarchy_model != "bradley_terry":
                if G_ties.has_edge(u, v):
                    G_ties[u][v]["weight"] += d.get("weight", 1.0)
                else:
                    G_ties.add_edge(u, v, weight=d.get("weight", 1.0))
        else:
            if G_wins.has_edge(u, v):
                G_wins[u][v]["weight"] += d.get("weight", 1.0)
            else:
                G_wins.add_edge(u, v, weight=d.get("weight", 1.0))

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
        config.hierarchy_model,
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

    # Generate initial state if not provided
    if initial_partition is None:
        initial_partition = _gen_initial_partition(G_wins, G_ties, num_groups, group_graph=group_graph)
    if initial_scores is None:
        initial_scores = _gen_initial_scores(G_wins, G_ties)
    elif not scores_fixed:
        # Normalise user-provided scores to unit RMS and set initial_individual_depth
        # to that RMS so scaled_scores start near the prior scale while the actual
        # scores are preserved.  individual_depth remains free (unfixed).
        scores_arr = np.asarray(initial_scores, dtype=float)
        src_rms = float(np.sqrt(np.mean(scores_arr ** 2)))
        if src_rms > 1e-6:
            initial_scores = (scores_arr / src_rms).tolist()
            if initial_individual_depth is None and individual_depth is None:
                initial_individual_depth = src_rms

    wins_adj = get_neighbors(G_wins)
    ties_adj = get_neighbors(G_ties)

    node_order = tuple(G.nodes())

    # Two-layer mode: validate and convert group_graph to an adjacency list
    # indexed with the same node ordering as G.
    if group_graph is not None:
        _check_group_graph(G, group_graph)
        group_adj = get_neighbors(group_graph, node_order=list(node_order))
    else:
        group_adj = []  # empty → single-layer mode in C++

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
        seed=int(seed),
        timeout=float(timeout),
        show_progress=bool(show_progress),
        weighted_SBM=bool(config.weighted_SBM),
    )

    return Result(native_result, node_order, config)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gen_initial_partition(
    G_wins: nx.DiGraph, G_ties: nx.Graph, num_groups, seed: int = 0,
    group_graph: Optional[nx.Graph] = None,
) -> list:
    if group_graph is not None:
        G_undirected = group_graph
    else:
        G_undirected = nx.Graph()
        G_undirected.add_nodes_from(G_wins.nodes())
        for u, v in G_wins.edges():
            G_undirected.add_edge(u, v)
        for u, v in G_ties.edges():
            G_undirected.add_edge(u, v)

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

    node_to_index = {node: i for i, node in enumerate(G_undirected.nodes())}
    partition = [0] * G_undirected.number_of_nodes()
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
