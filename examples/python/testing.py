import directedstructure as ds
from directedstructure.clusterings import consensus_clustering
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Read gml file
# G = nx.read_gml("examples/data/networks/highschool_51.gml") # Reasonable number of groups
G = nx.read_gml("examples/data/networks/snr025_sigC2_rep07.gml",label='id') # All these synthetic examples have way too many groups
# G = nx.read_gml("examples/data/networks/snr100_sigC0_rep05.gml",label='id')
# G = nx.read_gml("examples/data/networks/snr100_sigC5_rep00.gml",label='id') # Too many groups
# G = nx.read_gml("examples/data/networks/snr150_sigC0_rep09.gml",label='id')
# G = nx.read_gml("examples/data/networks/snr150_sigC5_rep09.gml",label='id') # This is currently giving way too many groups for some reason
# # Need to revisit the SBM code and make sure that it is giving correct results (small graph tests)
# # Otherwise debug why we are finding some many graphs in this case. 

# G = nx.read_gml("examples/data/networks/highschool_84.gml")
# G = nx.read_gml("examples/data/networks/friends.gml")
# G = nx.read_gml("examples/data/networks/english-football.gml")
# G = nx.read_gml("examples/data/networks/english-football-restricted.gml")
# G = nx.read_gml("examples/data/networks/championship_2021.gml")
# G = nx.read_gml("examples/data/networks/business_depts.gml")

# Try converting to an undirected network
G = G.to_undirected()

print(G)

# Draw graph with colors based on group attribute
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
nx.draw(G, pos=pos, node_color=node_colors, with_labels=True, node_size=300, font_size=7, arrows=True, edge_color=[(0, 0, 0, 0.3)])
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=group_to_color[g], markersize=10, label=f"Group {g}") for g in unique_groups]
plt.legend(handles=handles, title="Group", loc="best")
plt.savefig("examples/testing_graph.png")
plt.close()

# Print matrix of edge counts between groups
group_index = {g: i for i, g in enumerate(unique_groups)}
n = len(unique_groups)
edge_matrix = np.zeros((n, n), dtype=int)
for u, v in G.edges():
    gi = group_index[G.nodes[u]["group"]]
    gj = group_index[G.nodes[v]["group"]]
    edge_matrix[gi, gj] += 1

print("\nEdge count matrix (rows=source group, cols=target group):")
header = "       " + "  ".join(f"G{g:>2}" for g in unique_groups)
print(header)
for i, g in enumerate(unique_groups):
    row = "  ".join(f"{edge_matrix[i, j]:>4}" for j in range(n))
    print(f"G{g:>2}  {row}")

model_settings_dict = {
    "groups_model": "traditional_DCSBM",
    # "group_attribute": "group",
    # "hierarchy_model": "bradley_terry_ties",
    "hierarchy_model": "bradley_terry",
    # "interaction": "coupled",
    "interaction": "independent",
    "merge_split_enabled": False,
    "verbose": True,
    "num_samples": 10000,
    "timeout": 200,
    "seed": 0,
    "sweeps_per_sample": 10,
}

samples_df = ds.samples(G, **model_settings_dict)

print(samples_df)

# Save the samples dataframe to a csv file
samples_df.to_csv("examples/testing_samples.csv", index=False)



ds.plot_MCMC_entropy(G, output_filename="examples/testing.png", **model_settings_dict)

ds.plot_node_properties(G, output_filename="examples/testing_node_properties.png", **model_settings_dict)
print("Done plotting.")

# Write a new gml with the inferences
ds.write_gml_with_inferences(G, "examples/testing_inferences.gml", **model_settings_dict)



# Plot node properties using the stored samples
ds.plot_node_properties(G, samples_df, output_filename="examples/testing_node_properties_from_samples.png", **model_settings_dict)



