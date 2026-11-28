"""Quick check that an installed wheel imports and fits (run by cibuildwheel)."""

import networkx as nx

import community_hierarchy as ch

G = nx.DiGraph()
for i in range(12):
    G.add_edge(i, (i + 1) % 12)
    G.add_edge(i, (i + 3) % 12)
result = ch.fit(
    ch.Config(), G, num_chains=2, num_samples=50, num_warmup=20, seed=1, show_progress=False
)
props = result.node_properties_df()
assert len(props) == 12 and props["score_mean"].notna().all()
assert len(result.network_properties_df()) > 0
print("community_hierarchy", ch.__version__, "wheel smoke test passed")
