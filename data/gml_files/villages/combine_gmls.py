# Script to combine the gml files of the villages to include the specific data on individuals

import networkx as nx
import os

full_data_folder = "full_data" # Contains all node attributed
short_data_folder = "shortened_data" # Contained the community node attribute from graph-tool that we use to initialize the partition sampling

full_gml_folder = "combined_data" # Where to write the generated gml files

for filename in os.listdir(full_data_folder):
    print(f"Processing {filename}")
    G = nx.read_gml(f"{full_data_folder}/{filename}")
    G_short = nx.read_gml(f"{short_data_folder}/{filename}")
    
    node_list = list(G.nodes())
    for node in node_list:
        # If the node is not found in G_short, remove it
        if node not in G_short.nodes():
            G.remove_node(node)
        else:
            # If it is, update the community attribute
            G.nodes[node]['community'] = G_short.nodes[node]['community']
    
    nx.write_gml(G, f"{full_gml_folder}/{filename}")