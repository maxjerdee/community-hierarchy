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

summary_df = pd.DataFrame(columns=['basename','sigma_C_mean','sigma_C_std','sigma_I_mean','sigma_I_std','sigma_T_mean','sigma_T_std',
'sigma_T_mean','sigma_T_std','nu_mean','nu_std','rep_partition','q_counts','total_status_mean','total_status_std'])

reset = True

if reset:
    basenames = []

    sigma_C_means = []
    sigma_C_stds = []
    sigma_I_means = []
    sigma_I_stds = []

    sigma_T_means = []
    sigma_T_stds = []
    comm_var_fraction_means = []
    comm_var_fraction_stds = []

    nu_means = []
    nu_stds = []

    rep_partitions = []
    q_counts_all = []
    total_status_means = []
    total_status_stds = []
else: # Load previously computed bits
    df_in_dict = pd.read_csv(summary_filename)

    print(df_in_dict)
    basenames = list(df_in_dict['basename'])

    sigma_C_means = list(df_in_dict['sigma_C_mean'])
    sigma_C_stds = list(df_in_dict['sigma_C_std'])
    sigma_I_means = list(df_in_dict['sigma_I_mean'])
    sigma_I_stds = list(df_in_dict['sigma_I_std'])

    sigma_T_means = list(df_in_dict['sigma_T_mean'])
    sigma_T_stds = list(df_in_dict['sigma_T_std'])
    comm_var_fraction_means = list(df_in_dict['comm_var_fraction_mean'])
    comm_var_fraction_stds = list(df_in_dict['comm_var_fraction_std'])

    nu_means = list(df_in_dict['nu_mean'])
    nu_stds = list(df_in_dict['nu_std'])

    rep_partitions = list(df_in_dict['rep_partition'])
    q_counts_all = list(df_in_dict['q_counts'])
    total_status_means = list(df_in_dict['total_status_mean'])
    total_status_stds = list(df_in_dict['total_status_std'])


for samples_folder_i, samples_folder in enumerate(samples_folders):
    print(f"Processing fits in {samples_folder}/{model_type}")

    filenames = listdir(f"{samples_folder}/{model_type}")

    filenames.sort()

    for filename in filenames:
        if filename[-4:] == ".csv": # Check for the correct extension
            basename = filename[:-4]
            if basename not in basenames: # Check that these samples have not already been processed
                
                print(f"Processing {filename}")

                try:
                # if True:

                    df = pd.read_csv(f"{samples_folder}/{model_type}/{filename}", converters={"partition": parse_list, "community_status": parse_list, "total_status": parse_list}, on_bad_lines = 'warn') # Some of the written sample lines appear to get messed up,
                    print(f"Read {len(df)} samples")

                    basenames.append(basename)

                    # Extract the means and standard deviations of the inferred parameters. 

                    sigma_C_means.append(np.mean(df['sigma_C']))
                    sigma_C_stds.append(np.std(df['sigma_C']))
                    sigma_I_means.append(np.mean(df['sigma_I']))
                    sigma_I_stds.append(np.std(df['sigma_I']))

                    # # Transformed parameters
                    sigma_Ts = np.sqrt(df['sigma_C']**2 + df['sigma_I']**2)
                    sigma_T_means.append(np.mean(sigma_Ts))
                    sigma_T_stds.append(np.std(sigma_Ts))
                    comm_var_fractions = df['sigma_C']**2/(df['sigma_C']**2 + df['sigma_I']**2)
                    comm_var_fraction_means.append(np.mean(comm_var_fractions))
                    comm_var_fraction_stds.append(np.std(comm_var_fractions))

                    nu_means.append(np.mean(df['nu']))
                    nu_stds.append(np.std(df['nu']))

                    
                    # Randomly subsample down to 1000 samples to speed up processing time
                    if len(df) > 1000:
                        df = df.sample(n=1000,random_state=42)

                    partitions = list(df['partition'])
                    partitions = np.array(partitions)


                    total_statuses = list(df['total_status'])
                    total_statuses = np.array(total_statuses)

                    # Extract the numbers of groups by taking the maximum in each partition
                    qs = np.max(partitions, axis=1) + 1

                    # Count the number of times each group number appears (for plotting the posterior distribution)
                    q_max = np.max(qs)
                    q_counts = np.zeros(q_max)
                    for q in qs:
                        q_counts[q-1] += 1
                    
                    # Write as a string
                    q_counts = q_counts.astype(int)
                    q_counts_str = str(q_counts.tolist())
                    q_counts_all.append(q_counts_str)

                    total_status_mean = np.mean(total_statuses,axis=0)
                    total_status_std = np.std(total_statuses,axis=0)

                    total_status_means.append(str(total_status_mean.tolist()))
                    total_status_stds.append(str(total_status_std.tolist()))

                    # Get the information-theoretic representative partition  
                    rep_partition = get_representative_partition(partitions,number_of_subsamples=number_of_subsamples,reduction='flat')

                    rep_partitions.append(str(rep_partition.tolist()))

                    df_out_dict = {'basename':basenames,'sigma_C_mean':sigma_C_means,'sigma_C_std':sigma_C_stds,'sigma_I_mean':sigma_I_means,'sigma_I_std':sigma_I_stds,'sigma_T_mean':sigma_T_means,'sigma_T_std':sigma_T_stds,\
                    'comm_var_fraction_mean':comm_var_fraction_means,'comm_var_fraction_std':comm_var_fraction_stds,'nu_mean':nu_means,'nu_std':nu_stds,'rep_partition':rep_partitions,'q_counts':q_counts_all,'total_status_mean':total_status_means,'total_status_std':total_status_stds}

                    # print(df_out_dict)
                    # for k,v in df_out_dict.items():
                    #     print(k,len(v))

                    summary_df = pd.DataFrame(df_out_dict)
                    summary_df.to_csv(summary_filename)

                except Exception as e:
                    print("Error" ,e, filename)


##########
# Process the BT fits



