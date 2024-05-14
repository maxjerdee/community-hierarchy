# Analyze the Monte Carlo samples to generate a csv file containing the estimated
# means and errors of the inferred parameters, as well as the information-theoretic representative 
# partition of the posterior distribution. 

# General imports
from os import listdir
import ast
import pandas as pd
import numpy as np

# Local imports
from RMI import compute_RMI # Compute the reduced mutual information

# Custom converter function to parse string representation of lists into lists 
def parse_list(s): 
    if s == "":
        return []
    return ast.literal_eval(s)  # Parse the string to a list using ast.literal_eval

def get_representative_partition(partitions, number_of_subsamples=30,reduction='DM'):
    """Find a representative partition assessed by minimizing the total conditional information cost of the posterior samples given the representative.

    :param partitions: All sampled partitions
    :type partitions: np.array
    :param number_of_subsamples: Number of subsamples of the partitions to take, otherwise too costly to compute RMI between all pairs, defaults to 30
    :type number_of_subsamples: int, optional
    :param reduction: type of reduction for the RMI, defaults to 'DM'
    :type reduction: str, optional
    :return: Representative partition
    :rtype: np.array
    """
    # Subsample
    num_samples = len(partitions)

    if num_samples > number_of_subsamples:
        # Get indices for subsampling by rounding the linspace
        subsample_indices = np.linspace(0,num_samples-1,number_of_subsamples)
        subsample_indices = np.round(subsample_indices).astype(int)

        # Restrict to the subsampled indices
        partitions = partitions[subsample_indices]
    
    
    # Compute the RMIs in an unnormalized way
    RMIs = np.zeros((len(partitions),len(partitions)))
    for i in range(len(partitions)):
        for j in range(len(partitions)):
            RMIs[i,j] = compute_RMI(partitions[i],partitions[j],reduction=reduction,normalization='none')
    
    # Compute which of the partitions is the most representative, should be the one which gives the smallest sum of 
    # the conditional entropies H(g|c) where g runs over all other partitions and we fix c. This may equivalently be found by 
    # choosing the c which maximizes \sum_g RMI(g;c) = \sum_g (H(g) - H(g|c)) since the sum \sum_g H(g) is constant for all c.
    # RMI(g;c) = H(g) - H(g|c)

    RMI_sums = np.sum(RMIs,axis=0)
    representative_partition = partitions[np.argmax(RMI_sums)]

    return representative_partition

number_of_subsamples = 30 # Number of subsamples to take from the posterior samples for the purposes of finding the representative partition

samples_folders = ['../data/samples/AddHealth', '../data/samples/misc', '../data/samples/villages']

gml_files_folders = ['../data/gml_files/AddHealth', '../data/gml_files/misc', '../data/gml_files/villages']

model_types = ['BT','SBM','SBM-BT']

SBM_samples_folder = '../data/samples/' 

##########
# Process the SBM-BT fits
model_type = 'SBM-BT'

summary_filename = f"samples_summary_{model_type}.csv"

# school_num = 81
school_num = 52


df = pd.read_csv(f"{samples_folders[0]}/{model_type}/highschool_{school_num}.csv", converters={"partition": parse_list, "community_status": parse_list, "total_status": parse_list}, on_bad_lines = 'warn') # Some of the written sample lines appear to get messed up,
print(f"{samples_folders[0]}/{model_type}/highschool_{school_num}.csv")
print(df)
                    
##########
# Process the BT fits



