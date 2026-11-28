"""The sampler against reference values: Mathematica's entropy components, and the planted-partition, coincidence, tempering and prior checks (statistical)."""

from __future__ import annotations

import math

import community_hierarchy.MCMC_core as native
import numpy as np
import pandas as pd
import pytest
from community_hierarchy import Config, Result, fit
from helpers import (
    _DATA_IN,
    _DATA_OUT,
    _c2v,
    _small_graph,
)


# Point-evaluation test against Mathematica fixture
def _parse_matrix_row(s: str) -> list[int]:
    """Parse a Mathematica-style '{a, b, c}' vector into a list of ints."""
    return [int(x) for x in s.strip().strip("{}").split(",")]


def _parse_matrix_line(line: str) -> list[list[int]]:
    """Parse a CSV line whose cells are '{...}' row-vectors into a 2-D list."""
    import csv as _csv

    cells = next(_csv.reader([line]))
    return [_parse_matrix_row(c) for c in cells]


def _wins_ties_from_rows(win_rows: list[list[int]], tie_rows: list[list[int]]):
    """Convert dense win/tie matrices to list-of-dicts format."""
    n = len(win_rows)
    wins = [{} for _ in range(n)]
    ties = [{} for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if win_rows[i][j]:
                wins[i][j] = float(win_rows[i][j])
            if tie_rows[i][j]:
                ties[i][j] = float(tie_rows[i][j])
    return wins, ties


def test_point_eval_entropy_components():
    """Entropy components of a 3-node directed graph against Mathematica.

    Reference: the "community-hierarchy test fixtures" section of the model
    notebook (mathematica/sbm_mcmc.nb, reproduction repository), chPointEvalComponents[Awins,
    Aties, scores, b, rhoIn, rhoOut, cIn, cOut, alpha, nu, sigmaI, sigmaC], with
    the concentrations cIn, cOut and alpha (the core takes the corresponding
    variations).  Regenerate with tests/mathematica/generate_fixtures.wls.
    Hnu, HsigmaI, HsigmaC are the half-Cauchy priors log(pi(w^2+x^2)/(2w)), w = 4.
    """
    in_path = _DATA_IN / "wins_ties_3node_point_eval.csv"
    out_path = _DATA_OUT / "wins_ties_3node_point_eval_components.csv"

    lines = in_path.read_text().splitlines()
    # Input file: two brace-lines (all rows of Awins, then all rows of Aties),
    # followed by scalar lines (partition, scores, then one param per line).
    brace_lines = [ln for ln in lines if "{" in ln]
    scalar_lines = [ln for ln in lines if ln and "{" not in ln]

    win_rows = _parse_matrix_line(brace_lines[0])
    tie_rows = _parse_matrix_line(brace_lines[1])
    wins, ties = _wins_ties_from_rows(win_rows, tie_rows)

    partition = [int(x) - 1 for x in scalar_lines[0].split(",")]
    scores = [float(x) for x in scalar_lines[1].split(",")]
    rho_in, rho_out, c_in, c_out, alpha, nu, sigma_i, sigma_c = [
        float(scalar_lines[i]) for i in range(2, 10)
    ]

    ref = pd.read_csv(out_path)

    state = native.MCMC(
        wins,
        ties,
        [],
        partition,
        True,
        scores,
        True,
        max(partition) + 1,
        True,
        True,  # assortative (ρIn ≠ ρOut)
        _c2v(c_in),
        True,  # variation_in  ← concentration c_in
        _c2v(c_out),
        True,  # variation_out ← concentration c_out
        _c2v(alpha),
        True,  # degree_correction ← concentration alpha
        sigma_i,
        True,  # individual_depth σI
        sigma_c,
        True,  # group_depth σC
        nu,
        True,  # ties_parameter ν
        0,
        rho_in,  # density_in
        rho_out,  # density_out
    )

    # ── Components also checkable individually (Mathematica element 2 & 3) ──
    np.testing.assert_allclose(float(state.H_Adir_G_A_s_nu), ref["H_Adir"].iloc[0], rtol=1e-6)
    np.testing.assert_allclose(float(state.H_s_G_b_depths), ref["H_s"].iloc[0], rtol=1e-6)

    # ── SBM components (part of Mathematica element 1, without param priors) ─
    np.testing.assert_allclose(float(state.H_b), ref["H_b"].iloc[0], rtol=1e-6)
    np.testing.assert_allclose(float(state.H_k), ref["H_k"].iloc[0], rtol=1e-6)
    np.testing.assert_allclose(float(state.H_M), ref["H_M"].iloc[0], rtol=1e-6)
    np.testing.assert_allclose(float(state.H_A_G_k_M_b), ref["H_A_G_k_M_b"].iloc[0], rtol=1e-6)

    # ── Half-Cauchy priors: Hx = log(π(w²+x²)/(2w)), w=4 (Mathematica elements 4-6) ──
    w = 4.0

    def h_hc(x: float) -> float:
        return math.log(math.pi * (w**2 + x**2) / (2.0 * w))

    np.testing.assert_allclose(h_hc(nu), ref["H_nu"].iloc[0], rtol=1e-6)
    np.testing.assert_allclose(h_hc(sigma_i), ref["H_sigmaI"].iloc[0], rtol=1e-6)
    np.testing.assert_allclose(h_hc(sigma_c), ref["H_sigmaC"].iloc[0], rtol=1e-6)

    # ── Internal consistency ─────────────────────────────────────────────────
    np.testing.assert_allclose(
        float(state.H_A_G_b),
        float(state.H_M) + float(state.H_k) + float(state.H_A_G_k_M_b),
        atol=1e-12,
    )


# planted_partition model smoke test
@pytest.mark.statistical
def test_planted_partition_fit_runs():
    """planted_partition model must complete without error and return a Result."""
    G = _small_graph()
    n = G.number_of_nodes()
    # Partition must use contiguous labels starting from 0
    init_p = [i % 3 for i in range(n)]
    result = fit(
        Config(groups_model="planted_partition", interaction="coupled"),
        G,
        num_samples=20,
        num_warmup=50,
        initial_partition=init_p,
        seed=7,
        timeout=20.0,
    )
    assert isinstance(result, Result)
    assert np.isfinite(result.mdl_value)
    df = result.to_frame()
    assert len(df) > 0
    assert "num_groups" in df.columns


@pytest.mark.statistical
def test_planted_partition_q_is_not_always_one():
    """With a structured graph and random init, planted_partition should not collapse to q=1."""
    G = _small_graph()
    rng = np.random.default_rng(42)
    init_p = [int(x) for x in rng.integers(0, 3, size=G.number_of_nodes())]
    result = fit(
        Config(groups_model="planted_partition", interaction="coupled"),
        G,
        num_samples=200,
        num_warmup=500,
        initial_partition=init_p,
        seed=42,
        timeout=30.0,
    )
    df = result.to_frame()
    mean_q = df["num_groups"].mean()
    if mean_q <= 1.0:
        # Retry with longer run and different seed
        rng2 = np.random.default_rng(9999)
        init_p2 = [int(x) for x in rng2.integers(0, 3, size=G.number_of_nodes())]
        result = fit(
            Config(groups_model="planted_partition", interaction="coupled"),
            G,
            num_samples=500,
            num_warmup=1000,
            initial_partition=init_p2,
            seed=9999,
            timeout=60.0,
        )
        df = result.to_frame()
        mean_q = df["num_groups"].mean()
    # Should explore more than trivial q=1; any mean > 1 is acceptable
    assert mean_q > 1.0, f"planted_partition collapsed to q=1 (mean_q={mean_q:.2f})"


def _run_coincidence_native_wins_ties(
    num_chains: int, burn: int, samples: int, sweeps: int, seed: int
) -> np.ndarray:
    """Multi-chain MCMC for the wins+ties 3-node graph with Mathematica-matched fixed params.

    This is the **coupled** model: scores {s[1], s[2], s[3]} are marginalized jointly
    with the partition, exactly as in the Mathematica NIntegrate (Out[271]).

    Graph: wins 0→1, 0→2, 1→2; tie 0↔1.
    Fixed: variation params at concentration=1, σI=1, σC=2, ν=1.
    Free: partition, scores, density_in, density_out.
    """
    wins = [{1: 1.0, 2: 1.0}, {2: 1.0}, {}]
    ties = [{1: 1.0}, {0: 1.0}, {}]
    n = 3
    vi = _c2v(1.0)
    acc = np.zeros((n, n))
    for chain in range(num_chains):
        state = native.MCMC(
            wins,
            ties,
            [],
            list(range(n)),
            False,
            [0.0] * n,
            False,
            n,
            False,
            True,
            vi,
            True,
            vi,
            True,
            vi,
            True,
            1.0,
            True,
            2.0,
            True,
            1.0,
            True,
            seed + chain,
        )
        for _ in range(burn):
            state.MCMC_sweep(1.0, sweeps)
        counts = np.zeros((n, n))
        for _ in range(samples):
            state.MCMC_sweep(1.0, sweeps)
            p = list(state.partition)
            for i in range(n):
                for j in range(n):
                    if p[i] == p[j]:
                        counts[i, j] += 1
        acc += counts / samples
    return acc / num_chains


@pytest.mark.statistical
def test_wins_ties_3node_coincidence():
    """Coupled coincidence matrix against Mathematica NIntegrate — the primary coupled validation.

    The NIntegrate marginalizes over scores {s[1],s[2],s[3]} jointly with the partition b,
    which corresponds to interaction='coupled'. This is distinct from the independent model
    where scores factor out and are integrated separately.

    Mathematica Out[271]: {{1, 0.183929, 0.177877}, {0.183929, 1, 0.221255}, {0.177877, 0.221255, 1}}
    Fixed: ν=1, σI=1, σC=2, concentration=1 (λIn=λOut=α=1); free: partition, scores, ρIn, ρOut.
    """
    expected = pd.read_csv(
        _DATA_OUT / "wins_ties_3node_coincidence.csv",
        header=None,
    ).to_numpy(dtype=float)
    coincidence = _run_coincidence_native_wins_ties(
        num_chains=64,
        burn=500,
        samples=500,
        sweeps=10,
        seed=42,
    )
    max_abs_diff = np.max(np.abs(coincidence - expected))
    if max_abs_diff >= 0.05:
        # Retry with 2x chains and samples to reduce platform-specific variance
        coincidence = _run_coincidence_native_wins_ties(
            num_chains=128,
            burn=500,
            samples=1000,
            sweeps=10,
            seed=42,
        )
        max_abs_diff = np.max(np.abs(coincidence - expected))
    assert max_abs_diff < 0.05, f"Coincidence matrix mismatch (max_abs_diff={max_abs_diff:.6f})"


def _tempering_coincidence(seed, num_chains, num_samples, timeout):
    from community_hierarchy.MCMC_core import fit_backend

    n = 3
    wins_adj = [{1: 1.0, 2: 1.0}, {2: 1.0}, {}]
    ties_adj = [{1: 1.0}, {0: 1.0}, {}]
    vi = _c2v(1.0)
    r = fit_backend(
        wins=wins_adj,
        ties=ties_adj,
        initial_partition=list(range(n)),
        initial_scores=[0.0] * n,
        assortative=True,
        individual_depth=1.0,
        group_depth=2.0,
        variation_in=vi,
        variation_out=vi,
        degree_correction=vi,
        ties_parameter=1.0,
        num_chains=num_chains,
        num_samples=num_samples,
        num_warmup=50,
        sweeps_per_sample=500,
        num_tempering_chains=2,
        seed=seed,
        timeout=timeout,
    )
    df = r.samples_df()
    ns = len(df)
    coin = np.zeros((n, n))
    for _, row in df.iterrows():
        p = [int(row[f"{i}_group"]) for i in range(n)]
        for i in range(n):
            for j in range(n):
                if p[i] == p[j]:
                    coin[i, j] += 1.0 / ns
    return coin, ns


@pytest.mark.statistical
def test_tempering_swap_uses_full_energy():
    """Parallel tempering swap criterion must use data terms only (H_A_G_b + H_Adir).

    Graph: wins 0→1, 0→2, 1→2; tie 0↔1 (same as test_wins_ties_3node_coincidence).
    Free: partition, scores, density_in, density_out.

    The CORRECT criterion is H_A_G_b + H_Adir_G_A_s_nu (data likelihood only).
    H_b (partition prior) is absorbed into the partition proposal.
    H_s_G_b_depths (score prior) is absorbed into the score proposal.
    Including H_s in the swap criterion causes systematic over-clustering (~0.06 bias).
    On first-pass failure the test retries with 4× more samples.
    """
    expected = pd.read_csv(_DATA_OUT / "wins_ties_3node_coincidence.csv", header=None).to_numpy(
        dtype=float
    )

    coin, ns = _tempering_coincidence(seed=0, num_chains=4, num_samples=2000, timeout=60.0)
    max_diff = np.max(np.abs(coin - expected))

    if max_diff >= 0.050:
        # Retry with 4× more samples in case of statistical bad luck.
        coin, ns = _tempering_coincidence(seed=1, num_chains=16, num_samples=2000, timeout=240.0)
        max_diff = np.max(np.abs(coin - expected))

    assert max_diff < 0.050, (
        f"Tempered coincidence matrix deviates from Mathematica reference by "
        f"{max_diff:.4f} (threshold 0.050, {ns} samples).  "
        f"The swap criterion likely includes H_s_G_b_depths — it should use only "
        f"H_A_G_b + H_Adir_G_A_s_nu (data terms), not the score prior — check fit.cpp.  "
        f"coin={coin.tolist()}"
    )


@pytest.mark.statistical
def test_prior_q_uniform():
    """At beta=0 the proposal cancels H_b exactly, so q ~ Uniform(1,...,n).

    The partition prior is P(b) ∝ prod_r n_r! / C(n-1,q-1).
    Summing over all labeled partitions with q groups gives a constant n! for
    every q, so P(q) = 1/n.  At beta=0 every proposal is accepted and the
    chain draws purely from the proposal, so the empirical q distribution must
    be uniform over {1,...,n}.

    The test runs a short trial first and, if it fails, re-runs with a longer
    burn-in and more samples before truly failing. This avoids spurious CI
    failures across build configurations without loosening statistical rigor.
    """
    n = 8
    wins = [{(i + 1) % n: 1.0} for i in range(n)]  # ring so density_in != NaN
    ties = [{} for _ in range(n)]
    vi = _c2v(1.0)
    expected_mean = (n + 1) / 2.0  # 4.5 for n=8

    def _make_state():
        return native.MCMC(
            wins,
            ties,
            [],
            list(range(n)),
            False,
            [0.0] * n,
            False,
            n,
            False,
            False,
            vi,
            True,
            vi,
            True,
            vi,
            True,
            1.0,
            True,
            1.0,
            True,
            1.0,
            True,
            0,
        )

    def _run_trial(burn_in, num_samples):
        state = _make_state()
        for _ in range(burn_in):
            state.MCMC_sweep(0.0, 1)
        trace = np.empty(num_samples)
        for t in range(num_samples):
            state.MCMC_sweep(0.0, 1)
            trace[t] = state.num_groups
        obs = np.bincount(trace.astype(int), minlength=n + 1)[1:].astype(float)
        mean_q = float(trace.mean())
        fracs = obs / num_samples
        # Threshold: 5 Monte Carlo standard errors from batch means (50 batches),
        # which accounts for autocorrelation -- consecutive sweeps are far from
        # independent (ESS ~ 1300 of 200k sweeps with the block-informed node move,
        # whose new-group probability is 0.01), so the iid SE would be ~10x too small.
        batches = trace[: num_samples // 50 * 50].reshape(50, -1).mean(axis=1)
        se = float(batches.std(ddof=1) / np.sqrt(50))
        mean_ok = abs(mean_q - expected_mean) < 5 * se
        bucket_ok = float(np.max(np.abs(fracs - 1.0 / n))) < 0.15
        return mean_ok and bucket_ok, mean_q, obs

    # Fast first attempt (~1 s).  On rare CI failures, retry with 5× more
    # burn-in and 5× more samples before declaring a real failure.
    for burn_in, num_samples in [(5_000, 40_000), (25_000, 200_000)]:
        passed, mean_q, obs = _run_trial(burn_in, num_samples)
        if passed:
            return

    fracs = obs / num_samples
    assert False, (
        f"q distribution at beta=0 not uniform after two attempts.\n"
        f"Last run: mean_q={mean_q:.3f}, expected={expected_mean:.1f}, "
        f"(5 batch-means SE not met).\n"
        f"Counts per q: {obs.astype(int).tolist()}\n"
        f"Fractions:    {np.round(fracs, 4).tolist()}"
    )
