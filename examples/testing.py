import directedstructure as ds
from directedstructure.clusterings import consensus_clustering
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Read gml file
# G = nx.read_gml("examples/data/networks/highschool_51.gml")
# G = nx.read_gml("examples/data/networks/highschool_84.gml")
# G = nx.read_gml("examples/data/networks/friends.gml")
# G = nx.read_gml("examples/data/networks/english-football.gml")
G = nx.read_gml("examples/data/networks/english-football-restricted.gml")
# G = nx.read_gml("examples/data/networks/championship_2021.gml")
# G = nx.read_gml("examples/data/networks/business_depts.gml")

model_settings_dict = {
    "groups_model": "traditional_DCSBM",
    # "group_attribute": "group",
    "hierarchy_model": "bradley_terry_ties",
    "interaction": "coupled",
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



