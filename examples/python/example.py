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
    show_progress=True,
)

samples_df = result.samples_df()

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print("Samples:", len(samples_df))
print(result)
print("Consensus partition:", result.consensus_partition().tolist())
print("Mean num_groups:", samples_df["num_groups"].mean())

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
