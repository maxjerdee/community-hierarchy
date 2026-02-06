# Script to process a simple, directed gml file of friendship nominations and transform it into a gml file with interactions annotated as either dominant or neutral by the following rule:
# If A nominates B as a friend, and B does not nominate A, then B "dominates" A (B->A, type = "dominant")
# If A nominates B as a friend, and B also nominates A, then the interaction is "neutral" (A--B, type = "neutral")

import networkx as nx

# input_filename = "friends_raw.gml"
# output_filename = "friends.gml"

# input_filename = "highschool_51_raw.gml"
# output_filename = "highschool_51.gml"

# input_filename = "highschool_30_raw.gml"
# output_filename = "highschool_30.gml"

input_filename = "highschool_84_raw.gml"
output_filename = "highschool_84.gml"

# Read gml file
G = nx.read_gml(input_filename)

# Check that the graph is directed
if not G.is_directed():
    raise ValueError("Input graph must be directed")

# Create a new directed graph to hold the processed interactions out of the same nodes
H = nx.DiGraph()
H.add_nodes_from(G.nodes(data=True))
# Process each edge in the original graph
for u, v in G.edges():
    if G.has_edge(v, u):
        print(u, v)
        # Mutual nomination - add edge with type "neutral" (arbitrary who is the source/target)
        # Avoid over-reporting so check the reverse edge does not already exist
        if not H.has_edge(v, u):
            H.add_edge(u, v, type="neutral")
    else:
        # One-way nomination - add directed edge with type "dominant"
        H.add_edge(v, u, type="dominant")

# Write the processed graph to a new gml file
nx.write_gml(H, output_filename)
