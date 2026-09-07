from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt

import directedstructure as ds

here = Path(__file__).resolve().parents[1]
G = nx.read_gml(here / "data" / "networks" / "english-football-2025.gml", label="id")

result = ds.fit(
    ds.Config(),
    G,
    num_chains=1,
    samples_per_chain=500,
    burn_sweeps=500,
    seed=1,
    # show_progress=False,
    show_progress=True,
)

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print(result)

# Network-level posterior summary: mean, std, and 90% HDI for all parameters.
# ICC and sigma_T are included. Computed in C++ — no samples_df() overhead.
print("\nNetwork properties (90% HDI):")
npdf = result.network_properties_df()
print(npdf.to_string(index=False))

# Per-node posterior summary: scores, community labels, group mean score.
# Uses consensus partition for n ≤ 3000 (the Bayesian posterior summary).
print("\nTop nodes by score:")
node_df = result.node_properties_df()
print(node_df.sort_values("score_mean", ascending=False).head().to_string(index=False))

# Minimum description length (useful for model comparison)
print(f"\nMDL value: {result.mdl_value:.3f}")

# Full per-sample DataFrame — only materialise when raw samples are needed
# (e.g. convergence diagnostics, joint distributions).
samples_df = result.samples_df()

# Node properties plot — scores ranked within each inferred community
fig_a, _ = ds.plot_node_properties(result)
fig_a.tight_layout()
plot_a = Path(__file__).resolve().parent / "node_properties.png"
fig_a.savefig(plot_a, dpi=150)
plt.close(fig_a)
print("Saved node properties plot to:", plot_a)

# Adjacency matrix reordered by inferred community
fig_b, _ = ds.plot_adjacency_matrix(result, G, title="english-football-2025")
plot_b = Path(__file__).resolve().parent / "adjacency.png"
fig_b.savefig(plot_b, dpi=150, bbox_inches="tight")
plt.close(fig_b)
print("Saved adjacency matrix to:", plot_b)

# q trajectory convergence diagnostic
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(samples_df.index, samples_df["num_groups"], linewidth=1.5)
ax.set_xlabel("Sample")
ax.set_ylabel("q = num_groups")
ax.set_title("q trajectory")
fig.tight_layout()
plot_path = Path(__file__).resolve().parent / "q_trajectory.png"
fig.savefig(plot_path, dpi=150)
plt.close(fig)
print("Saved q trajectory to:", plot_path)
