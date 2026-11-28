"""Type stubs for the pybind11 extension (bindings/python/bindings.cpp).

Generated with pybind11-stubgen 3.0.0, then edited: parameters after the first
default are keyword-only here (pybind11 accepts them positionally, but a
required parameter cannot follow a defaulted one in Python syntax), and the
Optional / numpy / pandas types are spelled out.  Regenerate after changing
the bindings.
"""
from __future__ import annotations

import numpy
import pandas
__all__: list[str] = ['FitResult', 'Interactions', 'MCMC', 'build_group_layer', 'build_interactions', 'fit_backend', 'get_equal_spend_c', 'get_individual_depth_centred_prob', 'get_merge_split_weight', 'get_prestage_mode', 'get_staging_sweeps', 'greedy_modularity_partition', 'psis', 'resolve_model', 'set_equal_spend_c', 'set_individual_depth_centred_prob', 'set_individual_depth_scale_move', 'set_merge_split_weight', 'set_prestage_mode', 'set_staging_sweeps', 'sort_groups_by_score', 'win_fraction_scores']
class Interactions:
    @property
    def a(self) -> list[int]:
        ...
    @property
    def b(self) -> list[int]:
        ...
    @property
    def draw(self) -> list[bool]:
        ...
    @property
    def has_neutral(self) -> bool:
        ...
    @property
    def ties(self) -> list[dict[int, float]]:
        ...
    @property
    def weight(self) -> list[float]:
        ...
    @property
    def wins(self) -> list[dict[int, float]]:
        ...
class MCMC:
    adapting: bool
    weighted_proposals: bool
    def MCMC_sweep(self, beta: float = 1.0, num_sweeps: int = 1, merge_split_enabled: bool = True) -> None:
        ...
    def __init__(self, wins: list[dict[int, float]], ties: list[dict[int, float]], *, group_adj: list[dict[int, float]] = [], partition: list[int], partition_fixed: bool = False, scores: list[float], scores_fixed: bool = False, num_groups: int, num_groups_fixed: bool = False, assortative: bool = True, variation_in: float = 1.0, variation_in_fixed: bool = False, variation_out: float = 1.0, variation_out_fixed: bool = False, degree_correction: float = 0.5, degree_correction_fixed: bool = False, individual_depth: float = 1.0, individual_depth_fixed: bool = False, group_depth: float = 1.0, group_depth_fixed: bool = False, ties_parameter: float = 1.0, ties_parameter_fixed: bool = False, seed: int = 0, density_in: float = -1.0, density_out: float = -1.0, weighted_SBM: bool = False) -> None:
        ...
    def calculate_H_M(self) -> float:
        ...
    def change_parameter(self, param_name: str, new_value: float) -> None:
        """One Metropolis-Hastings step of a continuous parameter towards new_value,
        which must be drawn from a CDF-space proposal of that parameter's prior (the
        prior is not in the acceptance). Not a setter: the value may be rejected.
        individual_depth rescales the scores with it unless scores are fixed."""
    def check_state_bookkeeping(self, throw_error: bool = True) -> None:
        ...
    def propose_group_score_move(self, r: int = -1) -> None:
        ...
    def propose_merge_move(self) -> None:
        ...
    def propose_merge_split_move(self) -> None:
        ...
    def propose_parameter_move(self) -> None:
        ...
    def propose_score_move(self) -> None:
        ...
    def propose_single_site_move(self) -> None:
        ...
    def propose_split_move(self) -> None:
        ...
    def set_beta(self, arg0: float) -> None:
        ...
    @property
    def H_A_G_b(self) -> float:
        ...
    @property
    def H_A_G_k_M_b(self) -> float:
        ...
    @property
    def H_A_b(self) -> float:
        ...
    @property
    def H_Adir_G_A_s_nu(self) -> float:
        ...
    @property
    def H_M(self) -> float:
        ...
    @property
    def H_W(self) -> float:
        ...
    @property
    def H_b(self) -> float:
        ...
    @property
    def H_k(self) -> float:
        ...
    @property
    def H_s_G_b_depths(self) -> float:
        ...
    @property
    def accepted_group_depth_proposals(self) -> int:
        ...
    @property
    def accepted_group_score_proposals(self) -> int:
        ...
    @property
    def accepted_individual_depth_proposals(self) -> int:
        ...
    @property
    def accepted_merge_proposals(self) -> int:
        ...
    @property
    def accepted_merge_split_proposals(self) -> int:
        ...
    @property
    def accepted_new_group_proposals(self) -> int:
        ...
    @property
    def accepted_score_proposals(self) -> int:
        ...
    @property
    def accepted_single_site_proposals(self) -> int:
        ...
    @property
    def accepted_split_proposals(self) -> int:
        ...
    @property
    def accepted_vacate_proposals(self) -> int:
        ...
    @property
    def degree_correction(self) -> float:
        ...
    @property
    def density_in(self) -> float:
        ...
    @property
    def density_out(self) -> float:
        ...
    @property
    def group_depth(self) -> float:
        ...
    @property
    def group_depth_proposals(self) -> int:
        ...
    @property
    def group_score_proposals(self) -> int:
        ...
    @property
    def individual_depth(self) -> float:
        ...
    @property
    def individual_depth_proposals(self) -> int:
        ...
    @property
    def merge_proposals(self) -> int:
        ...
    @property
    def merge_split_proposals(self) -> int:
        ...
    @property
    def new_group_proposals(self) -> int:
        ...
    @property
    def num_groups(self) -> int:
        ...
    @property
    def partition(self) -> list[int]:
        ...
    @property
    def score_proposals(self) -> int:
        ...
    @property
    def scores(self) -> list[float]:
        ...
    @property
    def single_site_proposals(self) -> int:
        ...
    @property
    def split_proposals(self) -> int:
        ...
    @property
    def step_group_depth(self) -> float:
        ...
    @property
    def step_group_score(self) -> float:
        ...
    @property
    def step_individual_depth(self) -> float:
        ...
    @property
    def step_score(self) -> float:
        ...
    @property
    def step_ties(self) -> float:
        ...
    @property
    def ties_parameter(self) -> float:
        ...
    @property
    def vacate_proposals(self) -> int:
        ...
    @property
    def variation_in(self) -> float:
        ...
    @property
    def variation_out(self) -> float:
        ...
    @property
    def weighted_SBM(self) -> bool:
        ...
class FitResult:
    def __getstate__(self) -> tuple:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def coincidence_matrix(self) -> numpy.ndarray:
        ...
    def consensus_partition(self, method: str = 'moc', refine: bool = True, min_samples: int = 1000) -> list[int]:
        ...
    def consensus_sample_indices(self, min_samples: int = 1000) -> list[int]:
        ...
    def diagnostics(self) -> dict:
        """
        ESS and R-hat of the network-level parameters and H_A_b (src/diagnostics.h).
        """
    def direction_loo(self, a: list[int], b: list[int], draw: list[bool], weight: list[float], r_eff: float = 1.0) -> dict:
        """
        Leave-one-out cross-validation of the edge directions (src/loo.h).
        """
    def score_diagnostics(self, relative: bool) -> dict:
        """
        ESS and R-hat of each node's score (src/diagnostics.h).
        """
    def direction_probabilities(self, node_a: int, node_b: int, ties_allowed: bool = True) -> list[float]:
        ...
    def mdl_partition(self) -> list[int]:
        ...
    def modal_num_groups(self) -> int:
        ...
    def network_properties(self, prob: float = 0.9) -> dict:
        ...
    def node_properties(self, mode: str = 'auto', consensus_method: str = 'moc') -> dict:
        ...
    def partition_ess(self) -> float:
        ...
    def samples_df(self) -> pandas.DataFrame:
        ...
    def score_means(self) -> list[float]:
        ...
    def score_stds(self) -> list[float]:
        ...
    def summary(self) -> dict:
        ...
    @property
    def mdl_value(self) -> float:
        ...
    @property
    def num_samples(self) -> int:
        ...
def build_group_layer(num_nodes: int, a: list[int], b: list[int], weight: list[float]) -> list[dict[int, float]]:
    ...
def build_interactions(num_nodes: int, source: list[int], target: list[int], draw: list[bool], weight: list[float], source_high_rank: bool) -> Interactions:
    ...
def fit_backend(wins: list[dict[int, float]], ties: list[dict[int, float]], *, group_adj: list[dict[int, float]] = [], initial_partition: list[int], partition_fixed: bool = False, initial_scores: list[float], scores_fixed: bool = False, assortative: bool = True, variation_in: float | None = None, variation_out: float | None = None, degree_correction: float | None = None, individual_depth: float | None = None, initial_individual_depth: float | None = None, group_depth: float | None = None, initial_group_depth: float | None = None, ties_parameter: float | None = None, num_groups: int | None = None, num_groups_fixed: bool = False, num_chains: int = 1, num_samples: int = 1000, num_thin: int = 1, num_warmup: int = 500, num_param_warmup: int = 0, sweeps_per_sample: int = 10, beta: float = 1.0, num_tempering_chains: int = 1, merge_split_enabled: bool = True, seed: int = 0, timeout: float = 60.0, show_progress: bool = False, weighted_SBM: bool = False, weighted_proposals: bool = False) -> FitResult:
    ...
def get_equal_spend_c() -> float:
    ...
def get_individual_depth_centred_prob() -> float:
    ...
def get_merge_split_weight() -> float:
    ...
def get_prestage_mode() -> int:
    ...
def get_staging_sweeps() -> int:
    ...
def greedy_modularity_partition(wins: list[dict[int, float]], ties: list[dict[int, float]], group_adj: list[dict[int, float]], num_groups: int | None = None) -> list[int]:
    """
    Greedy modularity starting partition (Clauset-Newman-Moore), labels by size.
    """
def psis(log_ratios: list[float], r_eff: float = 1.0) -> tuple:
    """
    Pareto-smoothed importance sampling (src/loo.h): (log_weights, pareto_k).
    """
def resolve_model(groups_model: str, hierarchy_model: str, interaction: str, mixing_variation: str, assortative: int, degree_correction: bool, given: dict[str, float | None], has_neutral: bool) -> dict:
    ...
def set_equal_spend_c(c: float) -> None:
    """
    divisor of fit()'s equal-spend rule w = q/(C*M); larger = fewer collective moves
    """
def set_individual_depth_centred_prob(p: float) -> None:
    """
    probability that a sigma_I proposal is the centred move (scores held fixed)
    """
def set_individual_depth_scale_move(k: int) -> None:
    """
    0: non-centred sigma_I move (all scores scale); 1: residual rescale (group means fixed)
    """
def set_merge_split_weight(w: float) -> None:
    """
    relative weight of each collective move; <0 = linear-time default
    """
def set_prestage_mode(mode: int) -> None:
    """
    0=all three (default), 1=random split, 2=sequential spreading, 3=sequential coalescence
    """
def set_staging_sweeps(M: int) -> None:
    """
    Gibbs sweeps used to refine a staged split (Peixoto's M)
    """
def sort_groups_by_score(partition: list[int], score_means: list[float]) -> list[int]:
    """
    Relabel groups 0..K-1 in increasing order of their mean score.
    """
def win_fraction_scores(wins: list[dict[int, float]], ties: list[dict[int, float]]) -> list[float]:
    """
    (wins - losses) / (wins + losses + draws + 1e-6) over distinct opponents.
    """
