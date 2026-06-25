# Compare the communities of a high school inferred when we consider
# all of the reported interactions vs. when we only consider the
# neutral interactions. 

import networkx as nx
import numpy as np

import directedstructure as ds

# Read gml file
# network_name = "highschool_51"
network_name = "highschool_30"
G_original = nx.read_gml(f"examples/data/networks/{network_name}.gml")

# Create version of the graph where all nonneutral edges are removed
G_neutral_only = G_original.copy()
edges_to_remove = [(u, v) for u, v, data in G_neutral_only.edges(data=True) if data.get("type") != "neutral"]
G_neutral_only.remove_edges_from(edges_to_remove)

# Print the number of dominant vs. neutral edges in the original graph
num_dominant = sum(1 for _, _, data in G_original.edges(data=True) if data.get("type") == "dominant")
num_neutral = sum(1 for _, _, data in G_original.edges(data=True) if data.get("type") == "neutral")
print(f"Original graph: {G_original.number_of_edges()} edges ({num_dominant} dominant, {num_neutral} neutral)")


# Settings
model_settings_dict = {
    # "groups_model": "traditional_DCSBM",
    "groups_model": "general_canonical",
    # "group_attribute": "group",
    "hierarchy_model": "bradley_terry_ties",
    # "hierarchy_model": "bradley_terry",
    "interaction": "independent",
    "verbose": True,
    "num_samples": 10000,
    "timeout": 60,
    "seed": 1,
    "sweeps_per_sample": 10,
}

# Write version of the graph with the inferences
ds.write_gml_with_inferences(G_original, f"examples/{network_name}_original_inference.gml", **model_settings_dict)

ds.write_gml_with_inferences(G_neutral_only, f"examples/{network_name}_neutral_inference.gml", **model_settings_dict)

# Print contingency table of group assignments for the original vs. neutral graphs
node_properties_original = ds.node_properties(G_original, **model_settings_dict)
node_properties_neutral = ds.node_properties(G_neutral_only, **model_settings_dict)
print(node_properties_neutral)

group_assignments_original = node_properties_original["consensus_group"].to_numpy()
group_assignments_neutral = node_properties_neutral["consensus_group"].to_numpy()
contingency_table = np.zeros((max(group_assignments_original) + 1, max(group_assignments_neutral) + 1), dtype=int)
for g_orig, g_neutral in zip(group_assignments_original, group_assignments_neutral):
    contingency_table[g_orig, g_neutral] += 1
print("Contingency table of group assignments (original vs. neutral):")
print(contingency_table)

# Compute the mutual information between the two group assignments (evaluate without package)
H1 = -np.sum(np.bincount(group_assignments_original) / len(group_assignments_original) * np.log2(np.bincount(group_assignments_original) / len(group_assignments_original) + 1e-10))
H2 = -np.sum(np.bincount(group_assignments_neutral) / len(group_assignments_neutral) * np.log2(np.bincount(group_assignments_neutral) / len(group_assignments_neutral) + 1e-10))
H12 = -np.sum(contingency_table / len(group_assignments_original) * np.log2(contingency_table / len(group_assignments_original) + 1e-10))
mutual_information = H1 + H2 - H12
NMI = mutual_information / max(H1, H2) if max(H1, H2) > 0 else 0
print(f"Mutual information between group assignments: {mutual_information:.4f}")
print(f"Normalized mutual information (NMI): {NMI:.4f}")

# Plot the graph with the original vs. neutral group assignments, also highlight the reciprocated vs. non-reciprocated edges
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
pos = nx.spring_layout(G_original, seed=0)
nx.draw(G_original, pos, ax=axes[0], with_labels=True, node_color=group_assignments_original, cmap="tab10", edge_color="gray", alpha=0.5)
axes[0].set_title("Original graph group assignments")
nx.draw(G_neutral_only, pos, ax=axes[1], with_labels=True, node_color=group_assignments_neutral, cmap="tab10", edge_color="gray", alpha=0.5)
axes[1].set_title("Neutral-only graph group assignments")
plt.savefig(f"examples/{network_name}_group_assignments_comparison.png")