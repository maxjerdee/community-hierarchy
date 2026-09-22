"""
Diagnose whether H_W is correctly sensitive to partition when neutral edges
go through the Python API (ties path) vs group_adj path.
Uses the same 3-node graph to isolate the routing difference.
"""
import math
import networkx as nx
import directedstructure as ds
import directedstructure.MCMC_core as native

w01, w02, w12 = 1.97558, 1.13873, 2.77927

def expected_HW(partition):
    """Binary Peixoto H_W for the 3-node graph with given partition."""
    # Partition {0,0,1}: μ_00=w01, μ_01=w02+w12; M_00=1 (halved), M_01=2
    # Partition {0,1,1}: μ_00=0, μ_01=w01+w02, μ_11=w12; M_01=2, M_11=1 (halved)
    K = 3; w = w01 + w02 + w12
    h = (K-1)*math.log(w) - math.lgamma(K)
    groups = set(partition)
    # Build weight sums per block pair from undirected perspective
    nodes = list(range(3))
    edges = [(0,1,w01),(0,2,w02),(1,2,w12)]
    mu = {}; M = {}
    for u,v,wt in edges:
        r,s = sorted([partition[u],partition[v]])
        k = (r,s)
        mu[k] = mu.get(k,0) + wt
        M[k]  = M.get(k,0)  + 1
    for (r,s),(m,mu_rs) in {k:(M[k],mu[k]) for k in M}.items():
        if r == s:  # diagonal: m is already binary count of pairs
            if m > 0:
                h += (m-1)*math.log(mu_rs) - math.lgamma(m)
        else:
            if m > 0:
                h += (m-1)*math.log(mu_rs) - math.lgamma(m)
    return h


def test_neutral_edge_routing():
    """
    Fit the 3-node graph via the full Python API with neutral edges.
    Compare H_W from samples_df against MCMC constructor (group_adj path)
    and Peixoto formula.
    """
    # Build directed graph with neutral edges (simulating SRI DiGraph)
    G = nx.DiGraph()
    G.add_node(0); G.add_node(1); G.add_node(2)
    for u,v,w in [(0,1,w01),(1,0,w01),(0,2,w02),(2,0,w02),(1,2,w12),(2,1,w12)]:
        G.add_edge(u, v, weight=w, type="neutral")

    # Fix partition [0,0,1]
    for nd in G.nodes(): G.nodes[nd]["group"] = [0,0,1][nd]

    cfg = ds.Config(group_attribute="group", weighted_SBM=True, degree_correction=True)
    r = ds.fit(cfg, G, num_samples=5, seed=42, show_progress=False)
    sdf = r.samples_df()
    HW_api = sdf["H_W"].mean()

    # Via MCMC constructor (group_adj path, binary weights)
    group_adj = [{} for _ in range(3)]
    group_adj[0][1] = w01; group_adj[1][0] = w01
    group_adj[0][2] = w02; group_adj[2][0] = w02
    group_adj[1][2] = w12; group_adj[2][1] = w12
    mcmc = native.MCMC(wins=[{} for _ in range(3)], ties=[{} for _ in range(3)],
                       group_adj=group_adj, partition=[0,0,1],
                       scores=[0.,0.,0.], num_groups=2, seed=0, weighted_SBM=True)
    HW_constructor = mcmc.H_W

    HW_peixoto = expected_HW([0,0,1])

    print(f"\nPartition [0,0,1]:")
    print(f"  Peixoto formula:    {HW_peixoto:.4f}")
    print(f"  MCMC constructor:   {HW_constructor:.4f}")
    print(f"  Python API (fit):   {HW_api:.4f}")

    # Now fix [0,1,1] and compare
    G2 = G.copy()
    for nd in G2.nodes(): G2.nodes[nd]["group"] = [0,1,1][nd]
    r2 = ds.fit(ds.Config(group_attribute="group", weighted_SBM=True, degree_correction=True),
                G2, num_samples=5, seed=42, show_progress=False)
    sdf2 = r2.samples_df()
    HW_api2 = sdf2["H_W"].mean()

    mcmc2 = native.MCMC(wins=[{} for _ in range(3)], ties=[{} for _ in range(3)],
                        group_adj=group_adj, partition=[0,1,1],
                        scores=[0.,0.,0.], num_groups=2, seed=0, weighted_SBM=True)
    HW_constructor2 = mcmc2.H_W
    HW_peixoto2 = expected_HW([0,1,1])

    print(f"\nPartition [0,1,1]:")
    print(f"  Peixoto formula:    {HW_peixoto2:.4f}")
    print(f"  MCMC constructor:   {HW_constructor2:.4f}")
    print(f"  Python API (fit):   {HW_api2:.4f}")

    print(f"\nΔH_W (partition change):")
    print(f"  Peixoto:      {HW_peixoto2 - HW_peixoto:.4f}")
    print(f"  Constructor:  {HW_constructor2 - HW_constructor:.4f}")
    print(f"  Python API:   {HW_api2 - HW_api:.4f}")

    assert abs((HW_api2 - HW_api) - (HW_peixoto2 - HW_peixoto)) < 0.5, (
        f"API ΔH_W={HW_api2-HW_api:.4f} differs from Peixoto ΔH_W={HW_peixoto2-HW_peixoto:.4f}")
