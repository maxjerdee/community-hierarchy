import directedstructure as ds
import networkx as nx

# Read gml file
G = nx.read_gml("examples/data/highschool_51.gml")
# G = nx.read_gml("examples/data/highschool_84.gml")
# G = nx.read_gml("examples/data/friends.gml")

model_settings_dict = {
    "groups_model": "traditional_DCSBM",
    "hierarchy_model": "bradley_terry_ties",
    "interaction": "coupled",
    "verbose": True,
    "num_samples": 10000,
    "timeout": 10,
    "seed": 0,
    "sweeps_per_sample": 10,
}

samples_df = ds.samples(G, **model_settings_dict)

print(samples_df)

network_properties_df = ds.network_properties(G, **model_settings_dict)

print(network_properties_df)

# node_properties_df = ds.node_properties(G)

# print(node_properties_df)

# network_properties_df = ds.network_properties(G)

# print(network_properties_df)
