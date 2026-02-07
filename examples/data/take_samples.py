# Script to take and save posterior samples for testing purposes

import networkx as nx
import directedstructure as ds

# input_filename = "examples/data/networks/friends.gml"
# output_filename = "examples/data/samples/friends.csv"

input_filename = "examples/data/networks/highschool_51.gml"
output_filename = "examples/data/samples/highschool_51.csv"

# Read gml file
G = nx.read_gml(input_filename)

# Take samples
samples_df = ds.samples(G, num_samples=10000)

# Save samples to csv
samples_df.to_csv(output_filename, index=False)