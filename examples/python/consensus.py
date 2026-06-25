# Finding the consensus clustering out of a sampled set

import directedstructure as ds
import pandas as pd
import numpy as np

# Read saved samples, could find from scratch using
# samples_df = ds.samples(G)
# samples_df = pd.read_csv("examples/data/samples/friends.csv")
samples_df = pd.read_csv("examples/data/samples/highschool_51.csv")

# Extract list of clusterings from samples dataframe
# Find all columns ending with "_group"
group_columns = [col for col in samples_df.columns if col.endswith("_group")]
clusterings = []
for col in group_columns:
    clusterings.append(samples_df[col].values)

clusterings = np.array(clusterings).T

# Print shape of clusterings array
print("Shape of clusterings array:", clusterings.shape)

# Find consensus clustering with L1 metric
consensus_L1 = ds.consensus_clustering(clusterings, metric="L1")

print("Consensus clustering (L1):", consensus_L1)
