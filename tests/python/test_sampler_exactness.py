"""Exactness of the individual MCMC kernels against targets computed independently.

  * Partition kernels (block-informed single-node move, merge / split /
    merge-split with the Eq. 19 pair choice) against the EXACT partition
    posterior of a 6-node network, enumerated over all 203 set partitions.
  * The full sweep (scores, group-score block, sigma_G rescale, adapted steps)
    on an edgeless network, where the posterior of the depths is their prior.
  * The fast group-score shift against a full recomputation of the entropies.

Monte Carlo error is estimated by batch means (autocorrelation-aware); a test
fails when the estimate is more than 5 standard errors from the exact value.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import directedstructure.MCMC_core as native


def _c2v(concentration: float) -> float:
    return math.exp(math.lgamma(concentration + 0.5) - math.lgamma(0.5) - math.lgamma(concentration + 1))


def _set_partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for p in _set_partitions(rest):
        for i in range(len(p)):
            yield p[:i] + [[first] + p[i]] + p[i + 1:]
        yield [[first]] + p


def _batch_se(trace, batches=50):
    trace = np.asarray(trace, dtype=float)
    b = trace[: len(trace) // batches * batches].reshape(batches, -1).mean(axis=1)
    return float(b.std(ddof=1) / np.sqrt(batches))


# A small directed network with wins and a tie, scores fixed at distinct values,
# densities / variations / depths fixed: the only random variable is the
# partition, whose posterior is q! exp(-(H_b + H_s + H_A_G_b)) over partitions.
N6 = 6
WINS6 = [{1: 1.0, 2: 1.0}, {2: 1.0}, {}, {4: 1.0, 5: 1.0}, {5: 1.0}, {0: 1.0}]
TIES6 = [{}, {}, {3: 1.0}, {2: 1.0}, {}, {}]      # ties are listed in both directions
SCORES6 = [0.9, 0.3, -0.4, 0.7, 0.1, -0.8]


def _state6(partition, seed=0):
    q = len(set(partition))
    vi = _c2v(1.0)
    return native.MCMC(
        WINS6, TIES6, [],
        list(partition), False,
        list(SCORES6), True,
        q, False,
        True,
        vi, True, vi, True, 0.5, True,
        1.0, True, 1.0, True, 1.0, True,
        seed,
        0.4, 0.15,
    )


def _exact6():
    """Exact P(q) and P(node 0 with node 1), P(node 0 with node 3)."""
    w, Pq, p01, p03 = [], np.zeros(N6 + 1), 0.0, 0.0
    parts = list(_set_partitions(list(range(N6))))
    logs = []
    for p in parts:
        lab = [0] * N6
        for g, members in enumerate(p):
            for i in members:
                lab[i] = g
        st = _state6(lab)
        q = len(p)
        logs.append(math.lgamma(q + 1) - (st.H_b + st.H_s_G_b_depths + st.H_A_G_b))
    logs = np.array(logs)
    w = np.exp(logs - logs.max())
    w /= w.sum()
    for p, wi in zip(parts, w):
        Pq[len(p)] += wi
        g = {i: k for k, members in enumerate(p) for i in members}
        p01 += wi * (g[0] == g[1])
        p03 += wi * (g[0] == g[3])
    return Pq, p01, p03


@pytest.fixture(scope="module")
def exact6():
    return _exact6()


def _run6(moves, steps, seed):
    st = _state6(list(range(N6)), seed=seed)
    st.set_beta(1.0)            # the constructor leaves beta = 0 until the first MCMC_sweep
    rng = np.random.default_rng(seed)
    f = {
        "site": st.propose_single_site_move,
        "merge": st.propose_merge_move,
        "split": st.propose_split_move,
        "ms": st.propose_merge_split_move,
    }
    choice = rng.integers(len(moves), size=steps + 5000)
    q, s01, s03 = [], [], []
    for t, c in enumerate(choice):
        f[moves[c]]()
        if t >= 5000 and t % 2 == 0:
            p = st.partition
            q.append(st.num_groups)
            s01.append(p[0] == p[1])
            s03.append(p[0] == p[3])
    st.check_state_bookkeeping(True)
    return np.array(q, float), np.array(s01, float), np.array(s03, float)


@pytest.mark.statistical
@pytest.mark.parametrize("moves", [
    ("site",),                          # block-informed node move alone (opens / vacates groups)
    ("merge", "split"),                 # q-changing collective moves alone
    ("site", "merge", "split", "ms"),   # the production mix of partition kernels
])
def test_partition_kernels_match_exact_posterior(exact6, moves):
    Pq, p01, p03 = exact6
    q, s01, s03 = _run6(list(moves), 300_000, seed=11)
    Eq = float(np.dot(np.arange(N6 + 1), Pq))
    for name, trace, exact in [("E[q]", q, Eq), ("P(0~1)", s01, p01), ("P(0~3)", s03, p03)]:
        se = _batch_se(trace)
        assert abs(trace.mean() - exact) < 5 * se + 1e-3, (
            f"{moves}: {name} = {trace.mean():.4f}, exact {exact:.4f}, 5 SE = {5 * se:.4f}")


@pytest.mark.statistical
def test_full_sweep_recovers_depth_prior_on_edgeless_network():
    """No edges: the posterior of (sigma_I, sigma_G) is their half-Cauchy(4)
    prior, for which E[log sigma] = log 4 and, by symmetry, E[ICC] = 1/2.  This
    exercises the score moves, the group-score block, the sigma_G move that
    rescales the group means (its Jacobian lambda^q), and adaptation during
    burn-in followed by frozen steps."""
    n = 10
    vi = _c2v(1.0)
    st = native.MCMC(
        [{} for _ in range(n)], [{} for _ in range(n)], [],
        [i % 3 for i in range(n)], False,
        [0.0] * n, False,
        3, False,
        False,
        vi, True, vi, True, 0.5, True,
        1.0, False, 1.0, False, 1.0, True,
        5,
    )
    st.adapting = True
    for _ in range(3000):
        st.MCMC_sweep(1.0, 1)
    st.adapting = False
    steps = (st.step_score, st.step_group_depth, st.step_individual_depth, st.step_group_score)
    lg, li, icc = [], [], []
    for _ in range(60_000):
        st.MCMC_sweep(1.0, 1)
        g, i = st.group_depth, st.individual_depth
        lg.append(math.log(g)); li.append(math.log(i)); icc.append(g * g / (g * g + i * i))
    assert steps == (st.step_score, st.step_group_depth, st.step_individual_depth, st.step_group_score), \
        "step sizes changed after adaptation was switched off"
    for name, trace, exact in [("E[log sigma_G]", lg, math.log(4.0)),
                               ("E[log sigma_I]", li, math.log(4.0)),
                               ("E[ICC]", icc, 0.5)]:
        trace = np.asarray(trace)
        se = _batch_se(trace)
        assert abs(trace.mean() - exact) < 5 * se, (
            f"{name} = {trace.mean():.4f}, exact {exact:.4f}, 5 SE = {5 * se:.4f}")


def test_fast_group_score_shift_matches_full_recompute():
    """propose_group_score_move updates H_s and H_Adir incrementally (O(1) prior,
    one pass over cross-group pairs).  After many accepted shifts the cached
    entropies must equal a full recomputation (check_state_bookkeeping throws
    on a mismatch)."""
    rng = np.random.default_rng(3)
    n = 30
    wins = [dict() for _ in range(n)]
    ties = [dict() for _ in range(n)]
    for _ in range(90):
        i, j = rng.integers(n, size=2)
        if i == j:
            continue
        i, j = int(i), int(j)
        if rng.random() < 0.2:                     # ties are listed in both directions
            ties[i][j] = ties[i].get(j, 0.0) + 1.0
            ties[j][i] = ties[j].get(i, 0.0) + 1.0
        else:
            wins[i][j] = wins[i].get(j, 0.0) + 1.0
    st = native.MCMC(
        wins, ties, [],
        [i % 4 for i in range(n)], False,
        list(rng.normal(size=n)), False,
        4, False,
        True,
        0.5, False, 0.5, False, 0.5, True,
        1.3, False, 0.8, False, 0.7, False,
        1,
    )
    for _ in range(5000):
        st.propose_group_score_move()
    assert st.accepted_group_score_proposals > 500
    st.check_state_bookkeeping(True)
    for _ in range(200):          # and inside full sweeps, mixed with every other move
        st.MCMC_sweep(1.0, 1)
    st.check_state_bookkeeping(True)
