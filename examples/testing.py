import directedstructure as ds
from directedstructure import Config, fit
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import subprocess, sys, json, tempfile, os

# Read GML file
# G = nx.read_gml("examples/data/networks/snr025_sigC2_rep07.gml", label='id')
# G = nx.read_gml("examples/data/networks/snr100_sigC0_rep05.gml", label='id')
# G = nx.read_gml("examples/data/networks/snr100_sigC5_rep00.gml", label='id')
# G = nx.read_gml("examples/data/networks/snr150_sigC0_rep09.gml", label='id')
G = nx.read_gml("examples/data/networks/snr150_sigC5_rep09.gml", label='id')
# G = nx.read_gml("examples/data/networks/highschool_84.gml")
# G = nx.read_gml("examples/data/networks/highschool_51.gml")
# G = nx.read_gml("examples/data/networks/friends.gml")
# G = nx.read_gml("examples/data/networks/english-football.gml")
# G = nx.read_gml("examples/data/networks/english-football-restricted.gml")
# G = nx.read_gml("examples/data/networks/championship_2021.gml")
# G = nx.read_gml("examples/data/networks/business_depts.gml")

print(G)

# Draw graph with colors based on ground-truth group attribute
groups = [G.nodes[n]["group"] for n in G.nodes()]
unique_groups = sorted(set(groups))
cmap = plt.cm.get_cmap("tab10", len(unique_groups))
group_to_color = {g: cmap(i) for i, g in enumerate(unique_groups)}
node_colors = [group_to_color[G.nodes[n]["group"]] for n in G.nodes()]

rng = np.random.default_rng(42)
group_x = {g: i for i, g in enumerate(unique_groups)}
statuses = np.array([G.nodes[n]["status"] for n in G.nodes()])
status_min, status_max = statuses.min(), statuses.max()
jitter_scale = 0.3
pos = {
    n: (
        group_x[G.nodes[n]["group"]] + rng.uniform(-jitter_scale, jitter_scale),
        (G.nodes[n]["status"] - status_min) / (status_max - status_min),
    )
    for n in G.nodes()
}

plt.figure(figsize=(8, 6))
nx.draw(G, pos=pos, node_color=node_colors, with_labels=True, node_size=300,
        font_size=7, arrows=True, edge_color=[(0, 0, 0, 0.3)])
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=group_to_color[g],
                      markersize=10, label=f"Group {g}") for g in unique_groups]
plt.legend(handles=handles, title="Group", loc="best")
plt.savefig("examples/testing_graph.png")
plt.close()

# Print ground-truth edge count matrix between groups
group_index = {g: i for i, g in enumerate(unique_groups)}
n_groups = len(unique_groups)
edge_matrix = np.zeros((n_groups, n_groups), dtype=int)
for u, v in G.edges():
    gi = group_index[G.nodes[u]["group"]]
    gj = group_index[G.nodes[v]["group"]]
    edge_matrix[gi, gj] += 1

print("\nEdge count matrix (rows=source group, cols=target group):")
header = "       " + "  ".join(f"G{g:>2}" for g in unique_groups)
print(header)
for i, g in enumerate(unique_groups):
    row = "  ".join(f"{edge_matrix[i, j]:>4}" for j in range(n_groups))
    print(f"G{g:>2}  {row}")

# Run inference
config = Config(
    groups_model="traditional_SBM",
    hierarchy_model="bradley_terry",
    interaction="independent",
)

result = fit(
    config,
    G,
    num_chains=4,
    samples_per_chain=250,
    sweeps_per_sample=10,
    burn_sweeps=500,
    num_tempering_chains=2,
    seed=0,
    timeout=200,
    show_progress=True,
)

samples_df = result.samples_df()
samples_df.to_csv("examples/testing_samples.csv", index=False)

# --- asbm run in a subprocess (pybind11 MCMC type clash prevents co-import) ---
# Use traditional_SBM dc=False to match ds groups_model="traditional_SBM".
# Pass the same directed GML — asbm symmetrises edges identically to ds.
_asbm_script = """
import asbm, networkx as nx, numpy as np, json, sys
gml_path, out_path = sys.argv[1], sys.argv[2]
G = nx.read_gml(gml_path, label='id')
r = asbm.fit(asbm.Config(model="traditional_SBM", degree_correction=False),
             G, num_chains=4, samples_per_chain=250,
             burn_sweeps=500, sweeps=10, seed=0, show_progress=True)
df = r.samples_df()
coinc = r.coincidence_matrix()
result = {
    "mean_q":       float(df["num_groups"].mean()),
    "mean_H_A_G_b": float(df["H_A_G_b"].mean()),
    "min_H_A_G_b":  float(df["H_A_G_b"].min()),
    "mean_H_k":     float(df["H_k"].mean()),
    "mean_H_M":     float(df["H_M"].mean()),
    "mean_H_A_G_k_M_b": float(df["H_A_G_k_M_b"].mean()),
    "coincidence":  coinc.tolist(),
}
with open(out_path, "w") as f:
    json.dump(result, f)
"""

with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
    asbm_out = tmp.name

subprocess.run(
    [sys.executable, "-c", _asbm_script,
     "examples/data/networks/snr150_sigC5_rep09.gml", asbm_out],
    check=True,
)
with open(asbm_out) as f:
    asbm_data = json.load(f)
os.unlink(asbm_out)

asbm_coinc = np.array(asbm_data["coincidence"])

# --- Compare H_A_G_b and components ---
print(f"\n{'':=<65}")
print(f"  {'Metric':<30}  {'ds (independent)':>16}  {'asbm':>10}")
print(f"  {'-'*30}  {'-'*16}  {'-'*10}")
for label, ds_val, ab_key in [
    ("mean q",          samples_df["num_groups"].mean(),    "mean_q"),
    ("mean H_k",        samples_df["H_k"].mean(),           "mean_H_k"),
    ("mean H_M",        samples_df["H_M"].mean(),           "mean_H_M"),
    ("mean H_A_G_k_M_b",samples_df["H_A_G_k_M_b"].mean(),  "mean_H_A_G_k_M_b"),
    ("mean H_A_G_b",    samples_df["H_A_G_b"].mean(),       "mean_H_A_G_b"),
    ("min H_A_G_b",     samples_df["H_A_G_b"].min(),        "min_H_A_G_b"),
]:
    print(f"  {label:<30}  {ds_val:>16.2f}  {asbm_data[ab_key]:>10.2f}")
print(f"{'':=<65}\n")

# --- Coincidence matrices ---
ds_coinc = result.coincidence_matrix()   # same node order: sorted(G.nodes())

diff = ds_coinc - asbm_coinc
print(f"  Coincidence max |diff|:  {np.abs(diff).max():.4f}")
print(f"  Coincidence mean |diff|: {np.abs(diff).mean():.4f}")
print(f"{'':=<65}\n")

# --- Plot coincidence comparison ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
im0 = axes[0].imshow(ds_coinc,   vmin=0, vmax=1, cmap="viridis")
axes[0].set_title("ds (traditional_SBM, independent)")
plt.colorbar(im0, ax=axes[0])
im1 = axes[1].imshow(asbm_coinc, vmin=0, vmax=1, cmap="viridis")
axes[1].set_title("asbm (traditional_SBM)")
plt.colorbar(im1, ax=axes[1])
im2 = axes[2].imshow(diff, cmap="RdBu_r",
                     vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
axes[2].set_title("ds − asbm")
plt.colorbar(im2, ax=axes[2])
fig.tight_layout()
fig.savefig("examples/testing_coincidence_comparison.png")
plt.close(fig)

# Total description length per sample
samples_df["dl"] = samples_df["H_A_b"] + samples_df["H_Adir_G_A_s_nu"] + samples_df["H_s_G_b_depths"]

# --- Planted-partition baseline DL ---
# Build planted partition: map node → group index 0..K-1
nodes_ordered = list(G.nodes())
planted_groups = sorted(set(G.nodes[n]["group"] for n in nodes_ordered))
group_to_idx = {g: i for i, g in enumerate(planted_groups)}
planted_partition = [group_to_idx[G.nodes[n]["group"]] for n in nodes_ordered]

planted_result = fit(
    config,
    G,
    num_chains=2,
    samples_per_chain=200,
    sweeps_per_sample=10,
    burn_sweeps=200,
    seed=1,
    timeout=60,
    initial_partition=planted_partition,
)
planted_df = planted_result.samples_df()
planted_df["dl"] = planted_df["H_A_b"] + planted_df["H_Adir_G_A_s_nu"] + planted_df["H_s_G_b_depths"]

min_dl_sampled  = samples_df["dl"].min()
min_dl_planted  = planted_df["dl"].min()
mean_q_sampled  = samples_df["num_groups"].mean()
mean_q_planted  = planted_df["num_groups"].mean()

print(f"\n{'':=<55}")
print(f"  Min DL (sampled):  {min_dl_sampled:.2f}   (mean q = {mean_q_sampled:.1f})")
print(f"  Min DL (planted):  {min_dl_planted:.2f}   (mean q = {mean_q_planted:.1f})")
print(f"  DL gap:            {min_dl_sampled - min_dl_planted:+.2f}")
print(f"{'':=<55}\n")

# --- Plot q and H_b trajectories ---
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)

for chain_id, group_df in samples_df.groupby("chain_id"):
    axes[0].plot(group_df.index, group_df["num_groups"], alpha=0.7, label=f"chain {chain_id}")
axes[0].axhline(len(planted_groups), color="k", linestyle="--", linewidth=1, label="planted q")
axes[0].set_ylabel("q (num_groups)")
axes[0].set_title("Number of groups across sampling trajectory")
axes[0].legend(fontsize=7)

for chain_id, group_df in samples_df.groupby("chain_id"):
    axes[1].plot(group_df.index, group_df["H_b"], alpha=0.7, label=f"chain {chain_id}")
axes[1].axhline(planted_df["H_b"].mean(), color="k", linestyle="--", linewidth=1, label="planted H_b")
axes[1].set_ylabel("H_b")
axes[1].set_title("Partition prior H_b across sampling trajectory")
axes[1].legend(fontsize=7)

for chain_id, group_df in samples_df.groupby("chain_id"):
    axes[2].plot(group_df.index, group_df["dl"], alpha=0.7, label=f"chain {chain_id}")
axes[2].axhline(min_dl_planted, color="k", linestyle="--", linewidth=1, label=f"planted min DL ({min_dl_planted:.1f})")
axes[2].set_xlabel("Sample index")
axes[2].set_ylabel("Description length")
axes[2].set_title("Total DL across sampling trajectory")
axes[2].legend(fontsize=7)

fig.tight_layout()
fig.savefig("examples/testing_num_groups.png")
plt.close(fig)

ds.plot_MCMC_entropy(result, output_filename="examples/testing.png")

ds.plot_node_properties(result, output_filename="examples/testing_node_properties.png")
print("Done plotting.")

ds.write_gml_with_inferences(G, "examples/testing_inferences.gml", result)

ds.plot_node_properties(result, partition=result.consensus_partition(),
                        output_filename="examples/testing_node_properties_from_samples.png")
