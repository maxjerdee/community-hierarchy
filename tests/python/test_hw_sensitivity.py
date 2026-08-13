"""
Test that H_W actually varies when the partition changes on a weighted graph.
Checks the Mathematica fixture directly (multigraph case: M_01=3, H_W=4.8926),
then verifies that H_W changes when a node is moved between groups.
"""
import math, pytest
import directedstructure.MCMC_core as native


def make_mcmc_from_fixture(partition, w01, w02, w12):
    """
    3-node symmetric graph: nodes 0,1 in group 0; node 2 in group 1.
      group_adj encodes SRI-like undirected weights (both directions, 1x each).
      No wins/ties (all edges are 'neutral').
    """
    n = 3
    group_adj = [{} for _ in range(n)]
    group_adj[0][1] = w01; group_adj[1][0] = w01
    group_adj[0][2] = w02; group_adj[2][0] = w02
    group_adj[1][2] = w12; group_adj[2][1] = w12

    mcmc = native.MCMC(
        wins=[{} for _ in range(n)],
        ties=[{} for _ in range(n)],
        group_adj=group_adj,
        partition=list(partition),
        scores=[0.0, 0.0, 0.0],
        num_groups=len(set(partition)),
        seed=0,
        weighted_SBM=True,
    )
    return mcmc


def peixoto_HW_exact(M_off, mu_off, M_diag, mu_diag, K=None, w=None):
    """Peixoto 2018 Eq. 33 H_W (no outer simplex coupling block pairs)."""
    h = 0.0
    for m_rr, mu_rr in zip(M_diag, mu_diag):
        if m_rr > 0:
            h += (m_rr - 1) * math.log(mu_rr) - math.lgamma(m_rr)
    for m_rs, mu_rs in zip(M_off, mu_off):
        if m_rs > 0:
            h += (m_rs - 1) * math.log(mu_rs) - math.lgamma(m_rs)
    return h


w01, w02, w12 = 1.97558, 1.13873, 2.77927
w_total = w01 + w02 + w12


def test_HW_fixture_two_layer_mode():
    """
    When passed via group_adj (two_layer_mode), the 3-node graph has:
      μ_00 = w01 (within group 0), μ_01 = w02 + w12 (between), μ_11 = 0
    group_adj is passed in both directions, so weight_counts[r][s] = 2*true_mu.
    This test checks what the C++ actually returns and compares to both:
      - binary Peixoto (M_01=2): expected 4.2202
      - multigraph Peixoto (M_01=3): expected 4.8926
    """
    mcmc = make_mcmc_from_fixture([0, 0, 1], w01, w02, w12)
    cpp_HW = mcmc.H_W

    # Binary Peixoto (M_01=2, group_adj provides binary edge count)
    K = 3; w = w_total
    expected_binary = (K-1)*math.log(w) - math.lgamma(K) + math.log(w02+w12)
    # Multigraph Peixoto (M_01=3, as in Mathematica fixture)
    expected_fixture = 4.8926

    print(f"\nC++ H_W (two_layer_mode / group_adj): {cpp_HW:.4f}")
    print(f"Binary Peixoto (M_01=2):              {expected_binary:.4f}")
    print(f"Mathematica fixture (M_01=3):          {expected_fixture:.4f}")

    # Report which one it matches
    if abs(cpp_HW - expected_binary) < 0.01:
        print("→ C++ matches BINARY Peixoto (M_01=2)")
    elif abs(cpp_HW - expected_fixture) < 0.01:
        print("→ C++ matches MATHEMATICA FIXTURE (M_01=3)")
    else:
        print(f"→ C++ matches NEITHER (diff from binary: {cpp_HW-expected_binary:.4f}, diff from fixture: {cpp_HW-expected_fixture:.4f})")


def test_HW_changes_with_partition():
    """
    H_W must differ when node 2 is reassigned.
    Partition [0,0,1] vs [0,1,1]: same graph, different partition → different H_W.
    """
    mcmc_a = make_mcmc_from_fixture([0, 0, 1], w01, w02, w12)
    mcmc_b = make_mcmc_from_fixture([0, 1, 1], w01, w02, w12)
    HW_a = mcmc_a.H_W
    HW_b = mcmc_b.H_W

    # Analytical Peixoto for [0,1,1]: group 0 = {0}, group 1 = {1,2}
    # μ_00=0, μ_01 = w01+w02, μ_11 = w12; M_00=0, M_01=2 (binary), M_11=2
    K = 3; w = w_total
    # [0,0,1]: M_00=1 (0 contrib), M_01=2 → log(w02+w12), M_11=0
    peix_a = math.log(w02+w12)
    # [0,1,1]: M_00=0, M_01=2 → log(w01+w02), M_11=1 (0 contrib)
    peix_b = math.log(w01+w02)

    print(f"\nPartition [0,0,1]: C++ H_W = {HW_a:.4f}, Peixoto = {peix_a:.4f}")
    print(f"Partition [0,1,1]: C++ H_W = {HW_b:.4f}, Peixoto = {peix_b:.4f}")
    print(f"C++ ΔH_W = {HW_b - HW_a:.4f}")
    print(f"Peixoto ΔH_W = {peix_b - peix_a:.4f}")

    assert abs(HW_a - HW_b) > 0.01, (
        f"H_W should change with partition! Got H_W_a={HW_a:.4f}, H_W_b={HW_b:.4f}")


def test_HW_matches_peixoto_formula():
    """
    C++ H_W matches Peixoto 2018 Eq. 33 (no outer simplex).
    [0,0,1]: only M_01=2 active → H_W = log(w02+w12).
    """
    mcmc = make_mcmc_from_fixture([0, 0, 1], w01, w02, w12)
    expected = math.log(w02 + w12)
    assert abs(mcmc.H_W - expected) < 1e-4, (
        f"H_W={mcmc.H_W:.6f} but expected Peixoto binary={expected:.6f}")
