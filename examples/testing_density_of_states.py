# Testing script
# Hash a sample graph using Weisfeiler-Lehman graph hashing

import networkx as nx
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
import directedstructure as ds
import matplotlib.pyplot as plt
import numpy as np


# Read gml file
G = nx.read_gml("examples/data/friends.gml")

neighbors, in_neighbors, out_neighbors = ds.get_neighbors(G)

print(neighbors, in_neighbors, out_neighbors)

n = G.number_of_nodes()

# random initial partition
partition = np.random.randint(0, 2, size=n).tolist()

MCMC_sampler = ds.MCMC(
    neighbors,
    in_neighbors,
    out_neighbors,
        partition,
        2,              
        False,         
        False,           
        0.5,            
        True,         
        0.5,           
        True,        
        0.0,           
        True,          
        0.0,            
        True,          
        42);           

# Running some normal MCMC sweeps
num_samples = 1000

num_groups = []

for i in range(num_samples):
    if (i + 1) % 10000 == 0:
        print(f"MCMC Sweep {i + 1}")
    MCMC_sampler.MCMC_sweep(1.0, False)
    num_groups.append(MCMC_sampler.num_groups)

# # Plot histogram of number of groups
# plt.hist(num_groups, bins=range(1, max(num_groups) + 2), align='left', rwidth=0.8)
# plt.xlabel('Number of Groups')
# plt.ylabel('Frequency')
# plt.title('Histogram of Number of Groups in Samples')
# plt.savefig('examples/testing_num_groups.png')


# Perform Wang-Landau to estimate the density of states
for i in range(100000):
    if (i + 1) % 10000 == 0:
        print(f"Wang-Landau MCMC Sweep {i + 1}")
        # Save the current estimate of the density of states
        log_density_num_groups = getattr(MCMC_sampler, 'log_density_num_groups')[1:15] # Skip zero groups
        plt.plot(range(1, len(log_density_num_groups) + 1), np.exp(log_density_num_groups-np.max(log_density_num_groups))) # Normalize for numerical stability
        plt.xlabel('Number of Groups')
        plt.ylabel('Estimated Density of States (normalized)')
        plt.title('Estimated Density of States from Wang-Landau')
        plt.savefig('examples/testing_density_of_states.png')
        print(log_density_num_groups)
            
            
    wang_landau_modification_factor = 0.001/(1 + i/1000.0) # Decrease modification factor over time
    MCMC_sampler.MCMC_sweep(1.0, False, wang_landau_modification_factor, 15)




# # Print the type of G
# print(type(G))


# if not isinstance(G, nx.Graph):
#     raise TypeError("G must be a NetworkX Graph object.")

# samples_df = ds.samples(G, model_name = "traditional_SBM", seed = 0, num_samples = 10000, merge_split_enabled = False)

# # Restrict to second half of samples
# samples_df = samples_df.iloc[len(samples_df)//2 : ]

# # Plot histogram of number of groups
# plt.hist(samples_df['num_groups'], bins=range(1, samples_df['num_groups'].max() + 2), align='left', rwidth=0.8)
# plt.xlabel('Number of Groups')
# plt.ylabel('Frequency')
# plt.title('Histogram of Number of Groups in Samples')
# plt.savefig('examples/testids.png')

# # node_groups_df = ds.node_groups(G, seed = 0, merge_split_enabled = False, num_samples = 1000)

# # print(node_groups_df)

# # Use Wang-Landau to numerically compute the Bayes factor between two models. Note that note all combinations of models are valid. 

# bayes_factor_value = ds.bayes_factor(G, model_1 = "traditional_SBM", model_2 = "traditional_DCSBM", seed = 0, num_samples = 10000, merge_split_enabled = False)

# print(f"Bayes factor (traditional_SBM vs traditional_DCSBM): {bayes_factor_value}")