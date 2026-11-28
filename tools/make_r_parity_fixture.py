"""Seeded fit of a 12-node network with the Python package: the fixture of the R
package's parity test (R-package/tests/testthat/test-parity.R).

    python tools/make_r_parity_fixture.py
"""

import sys, random, networkx as nx, pandas as pd, community_hierarchy as ch

out = sys.argv[1] if len(sys.argv) > 1 else "R-package/tests/testthat/fixtures"
rng = random.Random(3)
nodes = [f"n{i}" for i in range(12)]
G = nx.MultiDiGraph()
G.add_nodes_from(nodes)
for _ in range(40):
    u, v = rng.sample(nodes, 2)
    G.add_edge(
        u, v, type="neutral" if rng.random() < 0.2 else "dominant", weight=rng.choice([1, 1, 2])
    )
rows = [(u, v, d["type"], d["weight"]) for u, v, d in G.edges(data=True)]
pd.DataFrame(rows, columns=["from", "to", "type", "weight"]).to_csv(
    f"{out}/parity_edges.csv", index=False
)
p0 = [0] * 6 + [1] * 6
s0 = [round(0.1 * i - 0.5, 2) for i in range(12)]
pd.DataFrame({"node": nodes, "p0": p0, "s0": s0}).to_csv(f"{out}/parity_init.csv", index=False)
res = ch.fit(
    ch.Config(),
    G,
    num_chains=2,
    num_samples=40,
    num_warmup=10,
    seed=11,
    initial_partition=p0,
    initial_scores=s0,
    show_progress=False,
    timeout=600,
)
cols = [
    "chain_id",
    "num_groups",
    "individual_depth",
    "group_depth",
    "ties_parameter",
    "H_A_b",
    "H_Adir_G_A_s_nu",
    "H_s_G_b_depths",
]
res.to_frame()[cols].to_csv(f"{out}/parity_samples.csv", index=False, float_format="%.17g")
# The default start (greedy modularity and win fractions, in the C++ core since 0.6.0).
res = ch.fit(
    ch.Config(),
    G,
    num_chains=2,
    num_samples=40,
    num_warmup=10,
    seed=13,
    show_progress=False,
    timeout=600,
)
res.to_frame()[cols].to_csv(f"{out}/parity_default_samples.csv", index=False, float_format="%.17g")
# Leave-one-out cross-validation of the directions of that fit (C++ core, src/loo.h).
res.direction_loo_df().to_csv(f"{out}/parity_loo.csv", index=False, float_format="%.17g")
print("community_hierarchy", ch.__version__, len(res.to_frame()), "samples")
