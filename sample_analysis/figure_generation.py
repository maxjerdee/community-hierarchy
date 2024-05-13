# Plot the summary information about the schools as figures

# From Elizabeth, we would like to generate the following figures:

"""
Can we make the "original one" with the contours:
1) as is
2) with points colored by school size
3) with the "pull out" graphs to show communities and hierarchies of specific schools. 
Please do only one school pull out per graph, as I will cycle through them. Maybe make these pull outs a bit bigger than what you had last time? 
I would do them for 10 of the most extreme points so that we can pick the most illustrative. Let's also do a few for the points in the middle of the distribution. So, say, 15 in total?

Then can we have a set of plots showing, all with fitted line: 
- number of communities by size
- average community size by size
- size versus sigma_t
- side by side: size versus sigma_c and sigma_i in fractions that sum to 1
- racial diversity versus sigma_t
- side by side: racial diversity versus sigma_c and sigma_i in fractions that sum to 1
- percent hispanic or black versus sigma_t
- side by side: percent hispanic or black versus sigma_c and sigma_i in fractions that sum to 1
- each race separately versus sigma_t
- side by side: each race separately versus sigma_c and sigma_i in fractions that sum to 1

Can you give me one with the original figure and points overlaid for both villages and high schools? Ideally, we can do this in two versions. One with all points. Color the same, and the other one with points colored by size. 

it would also be helpful to replicate the statistics and graphs I requested below with the village data, guess so we know what the pattern look like.

size is reflected in different sizes of the dots rather than different colors as well.
"""

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

model_type = 'SBM-BT'

data_set_name = 'highschool'

summary_filename = f"samples_summary_{model_type}.csv"

figures_folder = f'../data/figures/AddHealth/{model_type}'

# Load the summary information
summary_df = pd.read_csv(summary_filename)
print(summary_df)

gml_files_folder = f"../data/gml_files/AddHealth"

rep_partitions = []
ns = []
qs = []
sigma_C_means = []
sigma_I_means = []
sigma_T_means = []
sigma_C_stds = []
sigma_I_stds = []
sigma_T_stds = []
percent_whites = []
percent_blacks = []
percent_hispanics = []
H_races = []

school_nums = []

# Grades to show
grades_all = [9,10,11,12]
races_all = ["white","hispanic","asian","mixed","black"]
sexes_all = ["male","female"]

# Compute the important information
for filename in os.listdir(gml_files_folder):
    # print(filename)
    basename = filename[:-4]
    if len(re.findall(data_set_name,filename)):
        print(basename)
        # Load the graph
        G = nx.read_gml(f"{gml_files_folder}/{basename}.gml")

        # Get all node attributes
        node_attributes = G.nodes(data=True)

        races = []
        sexes = []
        grades = []
        # Print node attributes
        for node, attributes in node_attributes:
            # Write the original independent score here using labels and data read
            # print(f"Node {node}:")
            for key, value in attributes.items():
                # print(f"  {key}: {value}")
                if key == 'race':
                    races.append(value)
                if key == 'sex':
                    sexes.append(value)
                if key == 'grade':
                    grades.append(value)

        row = summary_df[summary_df['basename']==basename]
        if len(row) > 0:

            # Read numpy array from string
            rep_partition = row['rep_partition'].values[0]
            rep_partition = np.fromstring(rep_partition[1:-1], dtype=int, sep=',')
            rep_partitions.append(rep_partition)

            n = len(rep_partition)
            q = len(np.unique(rep_partition))

            
            school_num = int(re.findall('[0-9]+',filename)[0])

            school_nums.append(school_num)

            ns.append(n)
            qs.append(q)
            sigma_C_means.append(row['sigma_C_mean'].values[0])
            sigma_I_means.append(row['sigma_I_mean'].values[0])
            sigma_T_means.append(row['sigma_T_mean'].values[0])
            sigma_C_stds.append(row['sigma_C_std'].values[0])
            sigma_I_stds.append(row['sigma_I_std'].values[0])
            sigma_T_stds.append(row['sigma_T_std'].values[0])

            total_status_mean = row['total_status_mean'].values[0]
            rep_partition = np.fromstring(rep_partition[1:-1], dtype=float, sep=',')

            # # Get the average statuses by group
            # group_statuses = np.zeros(q)
            # group_counts = np.zeros(q)
            # for i in range(n):
            #     group_statuses[rep_partition[i]] += total_status_mean[i]
            #     group_counts[rep_partition[i]] += 1
            # for r in range(q):
            #     group_statuses[i] /= group_counts[i]

            # print(group_statuses)

            # Get the breakdown by race
            total_race_identified = 0
            race_count_dict = {}
            for race in races_all:
                race_count_dict[race] = 0
            for race in races:
                if race in races_all:
                    race_count_dict[race] += 1
                    total_race_identified += 1

            percent_white = race_count_dict['white']/total_race_identified
            percent_black = race_count_dict['black']/total_race_identified
            percent_hispanic = race_count_dict['hispanic']/total_race_identified

            H_race = 0
            for k,v in race_count_dict.items():
                if v > 0:
                    H_race -= v/total_race_identified*np.log(v/total_race_identified)

            H_races.append(H_race)
            percent_whites.append(percent_white)
            percent_blacks.append(percent_black)
            percent_hispanics.append(percent_hispanic)

            # Compute the 

            # Plot individual "pop-out" figure for this school

plt.clf()
plt.scatter(ns,sigma_C_means)
plt.savefig(f"{figures_folder}/n_vs_sigma_C.png")

plt.clf()
plt.scatter(ns,sigma_I_means)
plt.savefig(f"{figures_folder}/n_vs_sigma_I.png")

plt.clf()
plt.scatter(ns,sigma_T_means)
plt.savefig(f"{figures_folder}/n_vs_sigma_T.png")

# Same thing with n on a log scale

plt.clf()
plt.xscale('log')
plt.scatter(ns,sigma_C_means)
plt.savefig(f"{figures_folder}/n_vs_sigma_C_log.png")

plt.clf()
plt.xscale('log')
plt.scatter(ns,sigma_I_means)
plt.savefig(f"{figures_folder}/n_vs_sigma_I_log.png")

plt.clf()
plt.xscale('log')
plt.scatter(ns,sigma_T_means)
plt.savefig(f"{figures_folder}/n_vs_sigma_T_log.png")


# Array of values
max_sigma_I = 6
max_sigma_C = 4
all_sigma_Is = np.arange(0,max_sigma_I,0.1)
all_sigma_Cs = np.arange(0,max_sigma_C,0.1)
probs = np.zeros((len(all_sigma_Is),len(all_sigma_Cs)))
for sigma_I_i in range(len(all_sigma_Is)):
    for sigma_C_i in range(len(all_sigma_Cs)):
        probs[sigma_I_i,sigma_C_i] = np.sqrt(all_sigma_Is[sigma_I_i]**2+all_sigma_Cs[sigma_C_i]**2)

# Key to find the individual plots:
plt.clf()
fig = plt.figure(figsize=(6,5))
# fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
ax.set_xlim((0,max_sigma_I))
ax.set_ylim((0,max_sigma_C))

contour_levels = np.arange(1,10,1)
contour = ax.contour(all_sigma_Is, all_sigma_Cs, probs.T, levels=contour_levels, colors='gray',linestyles='dashed')

# Get the default matplotlib colors
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

ax.plot(all_sigma_Cs,all_sigma_Cs,color='gray') # x = y
ax.errorbar(sigma_I_means,sigma_C_means, xerr=sigma_I_stds, yerr=sigma_C_stds, color=colors[0], ecolor='lightgray', fmt='', ls='none', zorder=1, label='High schools')

# ax.errorbar(village_params_df['sigma_I_avg'],village_params_df['sigma_C_avg'], xerr=village_params_df['sigma_I_std'], yerr=village_params_df['sigma_C_std'], fmt='bo', ecolor='lightblue', zorder=1, label='Indian villages')
for i, txt in enumerate(range(len(sigma_I_means))):
    ax.annotate(school_nums[txt], (sigma_I_means[i], sigma_C_means[i]))

# Set labels and title
plt.legend()
ax.set_xlabel(r"$\sigma_I$ (spread of individuals)")
ax.set_ylabel(r"$\sigma_C$ (spread of communities)")
ax.set_title(f"Spreads of individual and community statuses")
plt.tight_layout(pad = 2)
plt.savefig(f"{figures_folder}/sigma_I_vs_sigma_C_key.png")   


# Plotting the schools by size:
plt.clf()
fig = plt.figure(figsize=(6,5))
# fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
ax.set_xlim((0,max_sigma_I))
ax.set_ylim((0,max_sigma_C))

contour_levels = np.arange(1,10,1)
contour = ax.contour(all_sigma_Is, all_sigma_Cs, probs.T, levels=contour_levels, colors='gray',linestyles='dashed')

# Get the default matplotlib colors
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

sizes = np.sqrt(np.array(ns))

ax.plot(all_sigma_Cs,all_sigma_Cs,color='gray') # x = y
ax.errorbar(sigma_I_means,sigma_C_means, xerr=sigma_I_stds, yerr=sigma_C_stds, ecolor='lightgray', fmt='', ls='none', zorder=1, label='High schools')
# Color the points by the size
ax.scatter(sigma_I_means,sigma_C_means,s=sizes, color=colors[0])

# ax.errorbar(village_params_df['sigma_I_avg'],village_params_df['sigma_C_avg'], xerr=village_params_df['sigma_I_std'], yerr=village_params_df['sigma_C_std'], fmt='bo', ecolor='lightblue', zorder=1, label='Indian villages')


for i, txt in enumerate(range(len(sigma_I_means))):
    ax.annotate(school_nums[txt], (sigma_I_means[i], sigma_C_means[i]))

# Set labels and title
plt.legend()
ax.set_xlabel(r"$\sigma_I$ (spread of individuals)")
ax.set_ylabel(r"$\sigma_C$ (spread of communities)")
ax.set_title(f"Spreads of individual and community statuses")
plt.tight_layout(pad = 2)
plt.savefig(f"{figures_folder}/sigma_I_vs_sigma_C.png")    


    



