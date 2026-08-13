"""
Exact fixture tests for the weighted SBM description length.

The H_W formula (Peixoto 2018 Eq. 33, flat model) is:

  H_W = Σ_{rs: M_rs>0} [(M_rs−1)·log(μ_rs) − log((M_rs−1)!)]

This is -log P(x|A,b) for the per-block-pair uniform simplex model.
It is ADDED to the total description length (H_A_G_b += H_W), so
more-negative H_W = better fit = lower description length.

There is NO outer simplex coupling block pairs: the (K-1)·log(w_total)
term is NOT part of the formula.

These tests check:
  (a) H_W equals the analytically derived value for a concrete graph
  (b) H_A_G_b is correctly composed (H_A_G_b = H_k + H_M + H_A + H_W)

All expected values are computed analytically from the formula above to avoid
circular testing against the same implementation.
"""
import math
import networkx as nx
import directedstructure.MCMC_core as native
import directedstructure as ds


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
    """Peixoto 2018 Eq. 33 H_W for q=2 groups (no outer simplex)."""
    h = 0.0
    for m, mu in zip(M_diag, mu_diag):
        if m > 0:
            h += (m - 1) * math.log(mu) - math.lgamma(m)
    for m, mu in zip(M_off, mu_off):
        if m > 0:
            h += (m - 1) * math.log(mu) - math.lgamma(m)
    return h


def test_HW_partition_001():
    """H_W for partition [0,0,1] matches analytic formula exactly."""
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
    """H_W for partition [0,1,1] matches analytic formula exactly."""
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
    Direct implementation of the Mathematica HWGAb formula.

    Per-block-pair uniform simplex (Peixoto 2018 Eq. 33, no outer simplex):
      H_W = Σ_{rs: M_rs>0} [(M_rs-1)*log(μ_rs) - lgamma(M_rs)]

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
    return hwga


def test_HW_mathematica_fixture_binary():
    """
    Per-block-pair formula (no outer simplex):
      adjMatricesBinary = {{0,1,1},{1,0,1},{0,1,0}}  groups=[[0,1],[2]]
      Only active block pair (0,1): M=2, μ=W02+W12.
      H_W = (2-1)*log(W02+W12) - lgamma(2) = log(W02+W12).
    """
    SW = [[0,   W01,                  W02                 ],
          [W01, 0,                    W12                 ],
          [0,   W12, 0.16869873063752025]]
    AB = [[0, 1, 1],
          [1, 0, 1],
          [0, 1, 0]]
    groups = [[0, 1], [2]]
    expected = math.log(W02 + W12)
    result = _hw_mathematica_formula(SW, AB, groups)
    assert abs(result - expected) < 1e-9, f"HWGAb={result:.5f}, expected {expected:.5f}"


def test_HW_mathematica_formula_symmetric_matches_cpp():
    """
    For a symmetric binary graph (no self-loops), the Mathematica formula reduces to
    our C++ implementation's value of 3.244336.
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


def test_weighted_sbm_finds_structure():
    """
    Weighted SBM on a graph with SRI-scale weights and clear 2-block structure
    must find K>1. With a sign error in H_W the model collapses to K=N.

    Weights are SRI-scale (< 1) so that the prior term (K-1)*log(w_total)
    is strongly negative for large K, making K=N very attractive under the
    wrong sign — so this is a decisive behavioral diagnostic.
    """
    n_per_block, w_in, w_out = 8, 0.5, 0.02
    nodes = list(range(2 * n_per_block))

    G_ds = nx.DiGraph()
    G_ds.add_nodes_from(nodes)
    G_affil = nx.Graph()
    G_affil.add_nodes_from(nodes)
    for i in nodes:
        for j in nodes:
            if i >= j:
                continue
            w = w_in if (i // n_per_block == j // n_per_block) else w_out
            G_ds.add_edge(i, j, weight=w, type="dominant")
            G_affil.add_edge(i, j, weight=w)

    result = ds.fit(
        ds.Config(weighted_SBM=True, degree_correction=0.0, interaction="independent"),
        G_ds, group_graph=G_affil,
        num_samples=500, burn_sweeps=200, seed=42,
    )
    modal_K = int(result.samples_df()["num_groups"].mode().iloc[0])
    assert modal_K > 1, (
        f"Weighted SBM found K={modal_K} (singletons) on a clear 2-block graph — "
        f"likely H_W sign error in H_A_G_b"
    )
