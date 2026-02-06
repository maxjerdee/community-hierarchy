from __future__ import annotations
from typing import Tuple, List, Dict

from numpy.typing import ArrayLike

class MCMC:
    def __init__(
        self, neighbors: List[Dict[int, float]], partition: ArrayLike
    ) -> None: ...
    def MCMC_sweep(self, beta: float, seed: int) -> None: ...

    beta: float
    partition: ArrayLike
    H_b: float
    H_M: float
    H_k: float
    H_A_G_k_M_b: float
    H_A_G_b: float
    H_A_b: float
    q: int
    alpha_k: float
    gamma: float
    p_in: float
    p_out: float
    time_H_k: float
    time_H_A_G_k_M_b: float
    time_H_M: float
