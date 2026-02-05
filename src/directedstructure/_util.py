# Small helper functions
from __future__ import annotations

from math import lgamma

import numpy as np
from numpy.typing import ArrayLike


def log_factorial(n: float) -> float:
    """Logarithm of factorial of n

    :param n: float
    :type n: float
    :return: log(n!)
    :rtype: float
    """
    return lgamma(n + 1)


def log_factorial2(n: float) -> float:
    """Logarithm of double factorial of n, for integer k, (2k)!! = k!2^k, (2k-1)!! = (2 k)!/(2^k k!)

    :param n: float
    :type n: float
    :return: log(n!!)
    :rtype: float
    """
    if n % 2 == 0:
        k = n / 2
        return log_factorial(k) + k * float(np.log(2))
    k = (n + 1) / 2
    return log_factorial(2 * k) - log_factorial(k) - k * float(np.log(2))


def log_binom(n: float, m: float) -> float:
    """Logarithm of binomial coefficient binomial(n,m)

    :param n: float
    :type n: float
    :param m: float
    :type m: float
    :return: log(binomial(n,m))
    :rtype: float
    """
    return log_factorial(n) - log_factorial(m) - log_factorial(n - m)


def log_sum_exp(x: ArrayLike) -> float | np.complex64:
    """Overflow protected log(sum(exp(x))) of an array x.

    :param x: Array to be summed
    :type x: np.ndarray[np.float_, np.dtype[np.float_]]
    :return: log(sum(exp(x)))
    :rtype: float | np.complex64
    """
    x = np.array(x)
    a: np.float64 | np.complex64 = np.max(x)
    return a + np.log(np.sum(np.exp(x - a)))

def log_mean_and_se(x: ArrayLike) -> tuple[float, float]:
    """Overflow protected log(mean(exp(x))) and standard error of that log(mean) given an array x.

    :param x: Array to be summed
    :type x: np.ndarray[np.float_, np.dtype[np.float_]]
    :return: log(mean(x)), standard error
    :rtype: tuple[float, float]
    """
    log_E_Z = log_sum_exp(x) - np.log(len(x))
    if len(x) > 1:
        log_E_Z_2 = log_sum_exp(2 * np.array(x)) - np.log(len(x))
        # Estimate the error by the standard deviation of the counts
        variance_shifted = np.exp(0) - np.exp(2 * log_E_Z - log_E_Z_2)
        if variance_shifted <= 0.0:
            return log_E_Z, 0.0
        log_std = 0.5 * (
            np.log(variance_shifted)
            + log_E_Z_2
        )
        log_Z_err_est = np.exp(
            log_std - 0.5 * np.log(len(x)) - log_E_Z
        )
    else:
        log_Z_err_est = np.nan

    return log_E_Z, log_Z_err_est


def concentration_to_variation(concentration: float) -> float:
    """Convert concentration parameter (symmetric Dirichlet) to variation parameter (gamma Gini coefficient).

    Parameters
    ----------
    concentration: float
        Concentration parameter in a symmetric Dirichlet distribution. Must be greater than or equal to 0.
    Returns
    -------
    float
        Variation parameter (Gini coefficient of corresponding gamma distribution).
    """
    if concentration < 0.0:
        raise ValueError("Concentration parameter cannot be negative.")
    
    # Hardcoded cases
    if concentration == 0.0:
        return 1.0
    if concentration == np.inf:
        return 0.0
    
    return np.exp(log_binom(concentration - 0.5, - 0.5)) # binom(alpha - 1/2, alpha)

def variation_to_concentration(variation: float, tol: float = 1e-8, max_iter: int = 100) -> float:
    """Convert variation parameter (gamma Gini coefficient) to concentration parameter (symmetric Dirichlet).
    This is not an analytic inversion, but rather numerically solves through iteration.

    Parameters
    ----------
    variation: float
        Variation parameter (Gini coefficient of corresponding gamma distribution). Must be in [0, 1].
    tol: float
        Tolerance for the root-finding procedure. Defaults to 1e-8.
    max_iter: int
        Maximum number of iterations for the root-finding procedure. Defaults to 100.
    Returns
    -------
    float
        Concentration parameter in a symmetric Dirichlet distribution.
    """

    # Hardcoded cases
    if variation == 0.0:
        return np.inf
    if variation == 1.0:
        return 0.0
    if variation < 0.0 or variation > 1.0:
        raise ValueError("Variation parameter must be in [0, 1].")

    # f(c) = concentration_to_variation(c) - variation (find root)
    def f(c: float) -> float:
        return concentration_to_variation(c) - variation

    # lower bound: small positive (avoid zero)
    lo = 1e-12
    f_lo = f(lo)
    # find an upper bound where f(hi) < 0 (variation decreases with concentration)
    hi = 1.0
    f_hi = f(hi)
    it = 0
    while f_hi > 0 and it < 200:
        hi *= 2.0
        f_hi = f(hi)
        it += 1
        if hi > 1e12:
            break

    if f_lo * f_hi > 0:
        # fallback: return the fixed-point seed
        return variation_to_concentration(variation)

    # bisection
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) <= tol:
            return mid
        # shrink interval
        if f_lo * f_mid <= 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid

    return 0.5 * (lo + hi)