"""
Exact fixture tests for the weighted SBM description length.

The full H_W formula (Peixoto 2018, flat model) is:

  H_W = H(W|A,μ,b) + H(μ)

where
  H(W|A,μ,b) = Σ_{rs: M_rs>0} [(M_rs−1)·log(μ_rs) − log((M_rs−1)!)]

is -log P(W|A,μ,b) for the per-block-pair uniform simplex model, and

  H(μ) = (Q − 1)·log(w) − log((Q − 1)!),   Q = #{r ≤ s : M_rs > 0}

is -log P(μ), the totals of the Q block pairs joined by an edge drawn
uniformly from the simplex of fixed total w.  Block pairs with no edges
carry no weight and are not counted, which keeps H_W differences between
partitions independent of the unit of weight.

H_W is ADDED to the total description length (H_A_G_b += H_W), so
more-negative H_W = better fit = lower description length.

These tests check:
  (a) H_W equals the analytically derived value for a concrete graph
  (b) H_A_G_b is correctly composed (H_A_G_b = H_k + H_M + H_A + H_W)

All expected values are computed analytically from the formula above to avoid
circular testing against the same implementation.
"""
import math
import community_hierarchy.MCMC_core as native


# ---------------------------------------------------------------------------
# Fixture: 3-node symmetric simple graph, partition [0,0,1]
#
#   w01 = 1.9755790294340843  (within group 0)
#   w02 = 1.1387325708524414  (cross-block)
#   w12 = 1.3896369140947227  (cross-block)
#
#   Partition [0,0,1]: group 0={0,1}, group 1={2}
#   Block pairs (q=2 → K=3): (0,0), (0,1), (1,1)
#     M_00=1, μ_00=w01=1.9756       contribution: 0
#     M_01=2, μ_01=w02+w12=2.5284   contribution: log(μ_01)
#     M_11=0                         skipped
#   w_total = w01+w02+w12 = 4.5039
#
#   H_W = 2·log(4.5039) − log(2) + log(2.5284) = 3.24434
# ---------------------------------------------------------------------------

W01 = 1.9755790294340843
W02 = 1.1387325708524414
W12 = 1.3896369140947227


def _make_mcmc(partition, degree_correction=0.5):
    n = 3
    group_adj = [{} for _ in range(n)]
    group_adj[0][1] = W01;  group_adj[1][0] = W01
    group_adj[0][2] = W02;  group_adj[2][0] = W02
    group_adj[1][2] = W12;  group_adj[2][1] = W12
    return native.MCMC(
        wins=[{} for _ in range(n)],
        ties=[{} for _ in range(n)],
        group_adj=group_adj,
        partition=list(partition),
        scores=[0.0] * n,
        num_groups=len(set(partition)),
        seed=0,
        weighted_SBM=True,
        degree_correction=degree_correction,
    )


def _hw_analytic(M_diag, mu_diag, M_off, mu_off, w_total=None):
    """Full Peixoto 2018 H_W = H(W|A,μ,b) + H(μ) for q=2 groups."""
    h = 0.0
    for m, mu in zip(M_diag, mu_diag):
        if m > 0:
            h += (m - 1) * math.log(mu) - math.lgamma(m)
    for m, mu in zip(M_off, mu_off):
        if m > 0:
            h += (m - 1) * math.log(mu) - math.lgamma(m)
    # H(μ): -log P(μ) on the simplex of total weight w_total, over the block
    # pairs joined by at least one edge
    if w_total is not None and w_total > 0:
        npairs = sum(m > 0 for m in list(M_diag) + list(M_off))
        if npairs > 1:
            h += (npairs - 1) * math.log(w_total) - math.lgamma(npairs)
    return h


def test_HW_partition_001():
    """H_W = H(W|A,μ,b) + H(μ) for partition [0,0,1] matches analytic formula."""
    w_total = W01 + W02 + W12
    # (0,0): M=1, μ=W01; (0,1): M=2, μ=W02+W12; (1,1): M=0
    expected = _hw_analytic(
        M_diag=[1, 0], mu_diag=[W01, 0.0],
        M_off=[2],     mu_off=[W02 + W12],
        w_total=w_total,
    )
    mcmc = _make_mcmc([0, 0, 1])
    assert abs(mcmc.H_W - expected) < 1e-9, (
        f"H_W={mcmc.H_W:.10f} expected={expected:.10f}"
    )


def test_HW_partition_011():
    """H_W = H(W|A,μ,b) + H(μ) for partition [0,1,1] matches analytic formula."""
    w_total = W01 + W02 + W12
    # group 0={0}, group 1={1,2}
    # (0,0): M=0; (0,1): M=2, μ=W01+W02; (1,1): M=1, μ=W12
    expected = _hw_analytic(
        M_diag=[0, 1], mu_diag=[0.0, W12],
        M_off=[2],     mu_off=[W01 + W02],
        w_total=w_total,
    )
    mcmc = _make_mcmc([0, 1, 1])
    assert abs(mcmc.H_W - expected) < 1e-9, (
        f"H_W={mcmc.H_W:.10f} expected={expected:.10f}"
    )


def test_HW_changes_between_partitions():
    """H_W must differ between [0,0,1] and [0,1,1]."""
    hw_001 = _make_mcmc([0, 0, 1]).H_W
    hw_011 = _make_mcmc([0, 1, 1]).H_W
    assert abs(hw_001 - hw_011) > 0.01, (
        f"H_W unchanged: {hw_001:.6f} == {hw_011:.6f}"
    )


def test_HW_sign_in_H_A_G_b():
    """
    H_W is ADDED to H_A_G_b (it is -log P, a DL cost, negative when model fits well).
    H_A_G_b = H_k + H_M + H_A_G_k_M_b + H_W.

    Verify the composition formula holds for both partitions.
    """
    for partition in ([0, 0, 1], [0, 1, 1]):
        m = _make_mcmc(partition)
        expected = m.H_k + m.H_M + m.H_A_G_k_M_b + m.H_W
        assert abs(m.H_A_G_b - expected) < 1e-9, (
            f"Partition {partition}: H_A_G_b={m.H_A_G_b:.6f} but "
            f"H_k+H_M+H_A+H_W={expected:.6f} (diff={m.H_A_G_b-expected:.2e})"
        )


def _hw_mathematica_formula(SW, AB, groups):
    """
    Direct implementation of the Mathematica HWGAb formula (full, including H(μ)):

      HWGAb = HWGAMub + HMu

    where HWGAMub is the per-block-pair uniform simplex (Eq. 33) and
    HMu = (Q - 1)*log(w) - log((Q - 1)!), Q the number of block pairs with edges.

    Diagonal (r=r): M[r][r]/2 edges (halved for undirected), μ[r][r]/2 weight.
    Off-diagonal (r<s): M[r][s] edges, μ[r][s] total weight.
    """
    q = len(groups)
    mu = [[sum(SW[i][j] for i in gr for j in gs) for gs in groups]
          for gr in groups]
    M  = [[sum(AB[i][j] for i in gr for j in gs) for gs in groups]
          for gr in groups]
    hwga = 0.0
    for r in range(q):
        if M[r][r] != 0:
            m_d, mu_d = M[r][r] / 2, mu[r][r] / 2
            hwga += (m_d - 1) * math.log(mu_d) - math.lgamma(m_d)
        for s in range(r + 1, q):
            if M[r][s] != 0:
                hwga += (M[r][s] - 1) * math.log(mu[r][s]) - math.lgamma(M[r][s])
    # H(μ): total weight = sum of all μ_rs / 2 (symmetric matrix)
    w_total = sum(mu[r][s] for r in range(q) for s in range(q)) / 2.0
    npairs = sum(M[r][s] != 0 for r in range(q) for s in range(r, q))
    if npairs > 1 and w_total > 0:
        hwga += (npairs - 1) * math.log(w_total) - math.lgamma(npairs)
    return hwga


def test_HW_mathematica_fixture_binary():
    """
    Full HWGAb formula (H(W|A,μ,b) + H(μ)):
      adjMatricesBinary = {{0,1,1},{1,0,1},{0,1,0}}  groups=[[0,1],[2]]
      Active block pair (0,1): M=2, μ=W02+W12.
      w_total = W01+W02+W12 (from the weight matrix SW).
      H_W = log(W02+W12) + 2*log(w_total) - log(2!)
    """
    SW = [[0,   W01,                  W02                 ],
          [W01, 0,                    W12                 ],
          [0,   W12, 0.16869873063752025]]
    AB = [[0, 1, 1],
          [1, 0, 1],
          [0, 1, 0]]
    groups = [[0, 1], [2]]
    result = _hw_mathematica_formula(SW, AB, groups)
    # H(W|A,mu,b) + H(mu) must exceed H(W|A,mu,b) alone, since H(mu) > 0.
    HWAb_only = math.log(W02 + W12)
    assert result > HWAb_only, f"HWGAb={result:.5f} should exceed H(W|A,μ,b)={HWAb_only:.5f}"


def test_HW_mathematica_formula_symmetric_matches_cpp():
    """
    For a symmetric binary graph (no self-loops), the full Mathematica HWGAb
    formula (including H(μ)) matches the C++ H_W.
    """
    SW = [[0,   W01, W02],
          [W01, 0,   W12],
          [W02, W12, 0  ]]
    AB = [[0, 1, 1],
          [1, 0, 1],
          [1, 1, 0]]
    groups = [[0, 1], [2]]
    result = _hw_mathematica_formula(SW, AB, groups)
    mcmc = _make_mcmc([0, 0, 1])
    assert abs(result - mcmc.H_W) < 1e-9, (
        f"Formula={result:.10f} C++={mcmc.H_W:.10f}"
    )


def test_weighted_sbm_prefers_correct_partition():
    """
    Point-evaluation test: the correct K=2 partition has strictly lower H_W
    than K=1 on a clear 2-block weighted graph.

    This is a decisive diagnostic for sign errors in H_W (or its sign in H_A_G_b):
    - If H_W uses the wrong sign, K=1 may appear better even when K=2 is the truth.
    - On a complete binary graph the binary MDL alone barely distinguishes K=1 from K=2;
      almost all the preference for K=2 comes from H_W (the weighted block structure).

    We test directly with the native MCMC constructor at fixed partitions so the
    result is deterministic and independent of MCMC mixing time.
    """
    n_per_block, w_in, w_out = 8, 0.5, 0.02
    n = 2 * n_per_block

    group_adj = [{} for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = w_in if (i // n_per_block == j // n_per_block) else w_out
            group_adj[i][j] = w

    import community_hierarchy.MCMC_core as native

    # K=1: all nodes in group 0
    mcmc_k1 = native.MCMC(
        wins=[{}] * n, ties=[{}] * n, group_adj=group_adj,
        partition=[0] * n, scores=[0.0] * n, num_groups=1,
        seed=0, weighted_SBM=True, degree_correction=0.0,
    )

    # K=2: correct block partition
    part_k2 = [i // n_per_block for i in range(n)]
    mcmc_k2 = native.MCMC(
        wins=[{}] * n, ties=[{}] * n, group_adj=group_adj,
        partition=part_k2, scores=[0.0] * n, num_groups=2,
        seed=0, weighted_SBM=True, degree_correction=0.0,
    )

    # H_W at K=2 (correct partition) must be substantially lower than K=1
    delta_hw = mcmc_k2.H_W - mcmc_k1.H_W
    assert delta_hw < -50, (
        f"H_W(K=2)={mcmc_k2.H_W:.2f}, H_W(K=1)={mcmc_k1.H_W:.2f}, "
        f"ΔH_W={delta_hw:.2f} — correct partition should be favoured by >>50 nats"
    )


def test_HW_differences_do_not_depend_on_the_unit_of_weight():
    """Multiplying every weight by c shifts H_W by the same amount, (m - 1) log c,
    for every partition, so differences between partitions (what the sampler
    uses) do not change.  With the totals' simplex over all q(q+1)/2 block pairs
    instead of the pairs with edges this fails: partitions with empty block pairs
    gain or lose log c per empty pair."""
    import random
    rng = random.Random(1)
    n = 8
    edges = {(i, j): rng.uniform(0.05, 3.0) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.6}
    def hw(partition, c):
        adj = [{} for _ in range(n)]
        for (i, j), w in edges.items():
            adj[i][j] = adj[j][i] = c * w
        return native.MCMC(wins=[{} for _ in range(n)], ties=[{} for _ in range(n)], group_adj=adj,
                           partition=list(partition), scores=[0.0] * n,
                           num_groups=len(set(partition)), seed=0, weighted_SBM=True).H_W
    partitions = ([0] * n, [0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 3, 0, 1, 2, 3], list(range(n)))
    for c in (0.01, 7.0, 1000.0):
        shifts = [hw(p, c) - hw(p, 1.0) for p in partitions]
        expected = (len(edges) - 1) * math.log(c)
        for p, d in zip(partitions, shifts):
            assert abs(d - expected) < 1e-8, (p, c, d, expected)
