# Analyze the Monte Carlo samples from the BT model to generate a csv file containing the estimated
# means and errors of the inferred parameters

# General imports
from os import listdir
import ast
import pandas as pd
import numpy as np
import networkx as nx


# Custom converter function to parse string representation of lists into lists 
def parse_list(s): 
    if s == "":
        return []
    return ast.literal_eval(s)  # Parse the string to a list using ast.literal_eval

samples_folders = ['../data/samples/AddHealth', '../data/samples/misc', '../data/samples/villages']

gml_files_folders = ['../data/gml_files/AddHealth', '../data/gml_files/misc', '../data/gml_files/villages']

model_types = ['BT','SBM','SBM-BT']

SBM_samples_folder = '../data/samples/' 

##########
# Process the SBM-BT fits
model_type = 'BT'

summary_filename = f"samples_summary_{model_type}.csv"

summary_df = pd.DataFrame(columns=['basename','sigma_C_mean','sigma_C_std','sigma_I_mean','sigma_I_std','sigma_T_mean','sigma_T_std',
'sigma_T_mean','sigma_T_std','nu_mean','nu_std','rep_partition','q_counts','total_status_mean','total_status_std'])

reset = True
# reset = False

if reset:
    basenames = []

    sigma_T_means = []
    sigma_T_stds = []

    nu_means = []
    nu_stds = []
    
    total_status_means = []
    total_status_stds = []
else: # Load previously computed bits
    df_in_dict = pd.read_csv(summary_filename)

    print(df_in_dict)
    basenames = list(df_in_dict['basename'])

    sigma_T_means = list(df_in_dict['sigma_T_mean'])
    sigma_T_stds = list(df_in_dict['sigma_T_std'])

    nu_means = list(df_in_dict['nu_mean'])
    nu_stds = list(df_in_dict['nu_std'])

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

                    df = pd.read_csv(f"{samples_folder}/{model_type}/{filename}", converters={"total_status": parse_list}, on_bad_lines = 'warn') # Some of the written sample lines appear to get messed up,
                    print(f"Read {len(df)} samples")

                    basenames.append(basename)

                    # Extract the means and standard deviations of the inferred parameters. 

                    sigma_T_means.append(np.mean(df['sigma_T']))
                    sigma_T_stds.append(np.std(df['sigma_T']))

                    nu_means.append(np.mean(df['nu']))
                    nu_stds.append(np.std(df['nu']))
                    
                    # Randomly subsample down to 1000 samples to speed up processing time
                    if len(df) > 1000:
                        df = df.sample(n=1000,random_state=42)

                    total_statuses = list(df['total_status'])
                    total_statuses = np.array(total_statuses)

                    total_status_mean = np.mean(total_statuses,axis=0)
                    total_status_std = np.std(total_statuses,axis=0)

                    total_status_means.append(str(total_status_mean.tolist()))
                    total_status_stds.append(str(total_status_std.tolist()))

                    df_out_dict = {'basename':basenames,'sigma_T_mean':sigma_T_means,'sigma_T_std':sigma_T_stds,'nu_mean':nu_means,'nu_std':nu_stds,\
                    'total_status_mean':total_status_means,'total_status_std':total_status_stds}

                    # print(df_out_dict)
                    # for k,v in df_out_dict.items():
                    #     print(k,len(v))

                    summary_df = pd.DataFrame(df_out_dict)
                    summary_df.to_csv(summary_filename)

                except Exception as e:
                    print("Error" ,e, filename)
