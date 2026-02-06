
import directedstructure as ds
import networkx as nx

# Read gml file
G = nx.read_gml("examples/data/highschool_51.gml")
# G = nx.read_gml("examples/data/highschool_84.gml")
# G = nx.read_gml("examples/data/friends.gml")

samples_df = ds.samples(G, groups_model="traditional_DCSBM", hierarchy_model="bradley_terry_ties", interaction="coupled", verbose=True, num_samples=10000, timeout=300)

print(samples_df["individual_depth"].mean(), samples_df["group_depth"].mean(), samples_df["ties_parameter"].mean())

# node_properties_df = ds.node_properties(G)

# print(node_properties_df)

# network_properties_df = ds.network_properties(G)

# print(network_properties_df)