import directedstructure as ds
from directedstructure.clusterings import consensus_clustering
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Read gml file
# G = nx.read_gml("examples/data/networks/highschool_51.gml")
G = nx.read_gml("examples/data/networks/highschool_84.gml")
# G = nx.read_gml("examples/data/networks/friends.gml")

model_settings_dict = {
    "groups_model": "traditional_DCSBM",
    "hierarchy_model": "bradley_terry_ties",
    "interaction": "coupled",
    "verbose": True,
    "num_samples": 10000,
    "timeout": 300,
    "seed": 0,
    "sweeps_per_sample": 10,
}

"""
             parameter       mean       std
1           density_in   0.010075  0.001668
2          density_out   0.010075  0.001668
3         variation_in   0.500000  0.000000
4        variation_out   0.500000  0.000000
5    degree_correction   0.500000  0.000000
6  mean_degree_scaling   0.000000  0.000000
7     individual_depth   2.171837  0.120786
8          group_depth   0.270895  0.196933
9       ties_parameter   0.473869  0.036551
"""

# model_settings_dict = {
#     "groups_model": "general_canonical",
#     "hierarchy_model": "bradley_terry_ties",
#     "interaction": "coupled",
#     "verbose": True,
#     "num_samples": 10000,
#     "timeout": 300,
#     "seed": 0,
#     "sweeps_per_sample": 10,
# }

"""
             parameter       mean       std
0           num_groups  34.664962  4.286266
1           density_in   0.124243  0.017348
2          density_out   0.002105  0.000252
3         variation_in   0.306687  0.088977
4        variation_out   0.737075  0.036637
5    degree_correction   0.175557  0.015112
6  mean_degree_scaling   0.000000  0.000000
7     individual_depth   2.005144  0.106711
8          group_depth   0.573589  0.197378
9       ties_parameter   0.405352  0.028898

             parameter       mean       std
0           num_groups  31.703341  3.403865
1           density_in   0.117098  0.016276
2          density_out   0.002003  0.000310
3         variation_in   0.287372  0.054289
4        variation_out   0.739895  0.037513
5    degree_correction   0.184687  0.015905
6  mean_degree_scaling   0.000000  0.000000
7     individual_depth   2.163164  0.130263
8          group_depth   0.228230  0.150552
9       ties_parameter   0.476474  0.037000
"""

samples_df = ds.samples(G, **model_settings_dict)

print(samples_df)

# Plot the sum of H_A_b,H_Adir_G_A_s_nu,H_s_G_b_depths over the samples to check for convergence
plt.plot(
    samples_df["H_A_b"] + samples_df["H_Adir_G_A_s_nu"] + samples_df["H_s_G_b_depths"]
)
plt.xlabel("Sample index")
plt.ylabel("H_A_b + H_Adir_G_A_s_nu + H_s_G_b_depths")
plt.title("Convergence of MCMC")
plt.savefig("examples/testing.png")

# network_properties_df = ds.network_properties(G, **model_settings_dict)

# print(network_properties_df)

node_properties_df = ds.node_properties(G, **model_settings_dict)

print(node_properties_df)

network_properties_df = ds.network_properties(G, **model_settings_dict)

print(network_properties_df)
