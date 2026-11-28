"""fit(): from a graph and a Config to a Result."""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import networkx as nx
import numpy as np

from .checks import _UNSET, _check_fit_arguments, _edges, _get_node_data, _group_graph_edges
from .config import Config, _check_model_for_graph
from .MCMC_core import build_group_layer as _build_group_layer
from .MCMC_core import build_interactions as _build_interactions
from .MCMC_core import fit_backend as _fit_backend
from .MCMC_core import resolve_model as _resolve_model
from .result import Result

logger = logging.getLogger(__name__)


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
    for name, val in (
        ("source_high_rank", source_high_rank),
        ("merge_split_enabled", merge_split_enabled),
        ("num_groups_fixed", num_groups_fixed),
    ):
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
        len(node_order),
        *_edges(G, index, integer_weights=not config.weighted_SBM),
        source_high_rank=source_high_rank,
    )

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
        initial_individual_depth=None
        if initial_individual_depth is None
        else float(initial_individual_depth),
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

    edges = {
        "a": interactions.a,
        "b": interactions.b,
        "draw": interactions.draw,
        "weight": interactions.weight,
    }
    return Result(native_result, node_order, config, edges)
