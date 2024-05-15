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
import statsmodels.api as sm

model_type = 'SBM-BT'

data_set_names = ['highschool','anyinteractions']

# Names for data sets to use in figures
data_set_labels = {'anyinteractions':'Villages', 'highschool':'High schools'}

summary_filename = f"samples_summary_{model_type}.csv"


sigma_I_label = r'Individual status depth $\sigma_I$'
sigma_C_label = r'Community status depth $\sigma_C$'
sigma_T_label = r'Total status depth $\sigma_T$'
comm_var_fraction_label = r'Community depth fraction $F_C = \frac{\sigma_C^2}{\sigma_T^2}$'

# Plotting functions

# Get the default matplotlib colors
default_colors = plt.cm.tab20.colors

def scatter_plot_with_y_errors(xs,ys,yerrs,data_label,plot_title,x_label,y_label,basename):
    plt.clf()
    plt.scatter(xs,ys, zorder=2, label=data_label)
    plt.errorbar(xs,ys, yerr=yerrs, ecolor='lightgray', fmt='', ls='none', zorder=1)
    # Perform weighted least squares
    weights = 1/np.array(yerrs)
    Xs = sm.add_constant(xs)  # Add constant term for the intercept
    model = sm.WLS(ys, Xs, weights=weights)  # Create WLS model instance
    results = model.fit()  # Fit the model
    plt.plot(xs, results.fittedvalues, color='red', label='Fitted Line',zorder=0)
    # Label figure with the p value
    p_value = results.pvalues[1]
    plt.title(f"{plot_title}, p = {p_value:.3f}")
    # For the q plot, set the y lim to 0
    if y_label == r'Number of communities $q$':
        plt.ylim((0,plt.ylim()[1]))

    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(f"{figures_folder}/{basename}.png")
    plt.savefig(f"{figures_folder}/{basename}.pdf")

def scatter_plot_with_y_errors_size(xs,ys,yerrs,data_label,plot_title,x_label,y_label,sizes,filename):
    plt.clf()
    s_plot = ax.scatter(sigma_I_means,sigma_C_means, s=sizes,c=ns, cmap='viridis', zorder=2, label=data_label)
    # Add colorbar
    cbar = plt.colorbar(s_plot)
    cbar.set_label(r'Size $n$')
    plt.errorbar(xs,ys, yerr=yerrs, ecolor='lightgray', fmt='', ls='none', zorder=1)
    # Perform weighted least squares
    weights = 1/np.array(yerrs)
    Xs = sm.add_constant(xs)  # Add constant term for the intercept
    model = sm.WLS(ys, Xs, weights=weights)  # Create WLS model instance
    results = model.fit()  # Fit the model
    plt.plot(xs, results.fittedvalues, color='red', label='Fitted Line',zorder=0)
    # Label figure with the p value
    p_value = results.pvalues[1]
    plt.title(f"{plot_title}, p = {p_value:.3f}")

    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(f"{figures_folder}/{basename}.png")
    plt.savefig(f"{figures_folder}/{basename}.pdf")

def scatter_plot_log_x_scale_with_y_errors(xs,ys,yerrs,data_label,plot_title,x_label,y_label,filename):
    plt.clf()
    plt.scatter(xs,ys, zorder=2, label=data_label)
    plt.errorbar(xs,ys, yerr=yerrs, ecolor='lightgray', fmt='', ls='none', zorder=1)
    # Perform weighted least squares
    log_xs = np.log(xs)
    weights = 1/np.array(yerrs)
    Xs = sm.add_constant(log_xs)  # Add constant term for the intercept
    model = sm.WLS(ys, Xs, weights=weights)  # Create WLS model instance
    results = model.fit()  # Fit the model
    plt.plot(np.exp(log_xs), results.fittedvalues, color='red', label='Fitted Line',zorder=0)
    plt.xscale('log')
    # plt.xlim((0.8*min(xs),1.3*max(xs)))
    # Label figure with the p value
    p_value = results.pvalues[1]
    plt.title(f"{plot_title}, p = {p_value:.3f}")

    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(f"{figures_folder}/{basename}.png")
    plt.savefig(f"{figures_folder}/{basename}.pdf")

def scatter_plot(xs,ys,data_label,plot_title,x_label,y_label,filename):
    plt.clf()
    plt.scatter(xs,ys, zorder=2, label=data_label)
    # Perform unweighted least squares
    weights = np.ones(len(xs))
    Xs = sm.add_constant(xs)  # Add constant term for the intercept
    model = sm.WLS(ys, Xs, weights=weights)  # Create WLS model instance
    results = model.fit()  # Fit the model
    plt.plot(xs, results.fittedvalues, color='red', label='Fitted Line',zorder=0)
    # Label figure with the p value
    p_value = results.pvalues[1]
    plt.title(f"{plot_title}, p = {p_value:.3f}")

    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(f"{figures_folder}/{basename}.png")
    plt.savefig(f"{figures_folder}/{basename}.pdf")

def draw_contour_background(max_sigma_I,max_sigma_C,fig_width,fig_height):
    # Grid on which to evaluate the sigma_T values
    all_sigma_Is = np.arange(0,max_sigma_I,0.1)
    all_sigma_Cs = np.arange(0,max_sigma_C,0.1)
    probs = np.zeros((len(all_sigma_Is),len(all_sigma_Cs)))
    for sigma_I_i in range(len(all_sigma_Is)):
        for sigma_C_i in range(len(all_sigma_Cs)):
            probs[sigma_I_i,sigma_C_i] = np.sqrt(all_sigma_Is[sigma_I_i]**2+all_sigma_Cs[sigma_C_i]**2)

    # Key to find the individual plots:
    fig = plt.figure(figsize=(fig_width,fig_height))
    ax = fig.add_subplot(111)
    ax.set_xlim((0,max_sigma_I))
    ax.set_ylim((0,max_sigma_C))
    # Set aspect ratio 1
    ax.set_aspect('equal', adjustable='box')

    contour_levels = np.arange(1,10,1)
    contour = ax.contour(all_sigma_Is, all_sigma_Cs, probs.T, levels=contour_levels, colors='gray',linestyles='dashed')
    
    ax.plot(all_sigma_Cs,all_sigma_Cs,color='gray') # x = y

    return ax

# Load the summary information
summary_df = pd.read_csv(summary_filename)

# Information to be tracked across both data sets
sigma_C_means_all = {}
sigma_I_means_all = {}
ns_all = {}

# Where to place the summary plot across both data sets
summary_figures_folder = f'../data/figures'

# Produce plots for the individual data set collections
for data_set_name in data_set_names:
    if data_set_name == 'highschool':
        data_label = data_set_labels[data_set_name]
        figures_folder = f'../data/figures/AddHealth/{model_type}'
        gml_files_folder = f"../data/gml_files/AddHealth"

    if data_set_name == 'anyinteractions':
        data_label = data_set_labels[data_set_name]
        figures_folder = f'../data/figures/villages/{model_type}'
        gml_files_folder = f"../data/gml_files/villages"

    rep_partitions = []
    ns = []
    qs = []
    q_stds = []
    sigma_C_means = []
    sigma_I_means = []
    sigma_T_means = []
    comm_var_fraction_means = []
    sigma_C_stds = []
    sigma_I_stds = []
    sigma_T_stds = []
    comm_var_fraction_stds = []
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

            if data_set_name == 'highschool':
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

                # Read numpy array from string
                q_counts = row['q_counts'].values[0]
                q_counts = np.fromstring(q_counts[1:-1], dtype=int, sep=',')

                # Compute the variance of the sampled q values (with respect to the summary partition q)
                total_q_variance = 0
                total_samples = 0
                for q_minus_1, q_val in enumerate(q_counts):
                    q_sampled = q_minus_1 + 1
                    total_q_variance += q_val*(q - q_sampled)**2
                    total_samples += q_val
                
                q_std = np.sqrt(total_q_variance/n)
                q_std = max(1,q_std) # At a minimum, the resolution on q is 1. 
                # TODO: (Note that I don't think many of these chains have adequately converged w.r.t. q, should use merge-split moves)
                q_stds.append(q_std) 
                
                school_num = int(re.findall('[0-9]+',filename)[0])

                school_nums.append(school_num)

                ns.append(n)
                qs.append(q)
                sigma_C_means.append(row['sigma_C_mean'].values[0])
                sigma_I_means.append(row['sigma_I_mean'].values[0])
                sigma_T_means.append(row['sigma_T_mean'].values[0])
                comm_var_fraction_means.append(row['comm_var_fraction_mean'].values[0])
                sigma_C_stds.append(row['sigma_C_std'].values[0])
                sigma_I_stds.append(row['sigma_I_std'].values[0])
                sigma_T_stds.append(row['sigma_T_std'].values[0])
                comm_var_fraction_stds.append(row['comm_var_fraction_std'].values[0])

                if data_set_name == 'highschool':
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

                #############################
                # Plot individual "pop-out" figure for this school.

                # Get the average statuses by community in the representative partition
                # Read numpy array from string
                total_status_mean = row['total_status_mean'].values[0]
                total_status_mean = np.fromstring(total_status_mean[1:-1], dtype=float, sep=',')
                community_statuses = np.zeros(q)
                community_counts = np.zeros(q)
                for i in range(n):
                    community_statuses[rep_partition[i]] += total_status_mean[i]
                    community_counts[rep_partition[i]] += 1
                for r in range(q):
                    community_statuses[r] /= community_counts[r]

                # Range to set across the individual figures:
                if data_set_name == 'highschool':
                    min_community_status = -4
                    max_community_status = 4
                    min_total_status = -10
                    max_total_status = 10
                if data_set_name == 'anyinteractions':
                    min_community_status = -2
                    max_community_status = 2
                    min_total_status = -5
                    max_total_status = 5
                fig_width = 6
                fig_height = 4.5

                # Create plot with each individual
                plt.clf()
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                ax.set_xlim(min_community_status,max_community_status)
                ax.set_ylim(min_total_status,max_total_status)
                
                colors = []
                xs = []
                ys = []
                for i in range(n):
                    xs.append(community_statuses[rep_partition[i]])
                    ys.append(total_status_mean[i])
                    colors.append(default_colors[rep_partition[i] % len(default_colors)]) # Colors are just different groups

                # Draw gray vertical lines
                for line_position in community_statuses:
                    ax.axvline(x=line_position, color='gray', linestyle='--', linewidth=0.8)

                ax.scatter(xs,ys, c=colors,zorder=2, alpha = 0.5)
                ax.set_xlabel("Community status")
                ax.set_ylabel("Total status")
                ax.set_title(f"Example statuses of communities and individuals")

                plt.tight_layout(pad = 2)
                plt.savefig(f"{figures_folder}/individuals/{basename}.png")

                # Create plot with box and whisket plots to represent each community
                plt.clf()
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                ax.set_xlim(min_community_status,max_community_status)
                ax.set_ylim(min_total_status,max_total_status)
                
                # colors = []
                # xs = []
                # ys = []
                # for i in range(n):
                #     xs.append(community_statuses[rep_partition[i]])
                #     ys.append(total_status_mean[i])
                #     colors.append(default_colors[rep_partition[i] % len(default_colors)]) # Colors are just different groups

                # Draw a box and whisker for each group of different colors
                for r in range(q):
                    xs = []
                    ys = []
                    for i in range(n):
                        if rep_partition[i] == r:
                            xs.append(community_statuses[rep_partition[i]])
                            ys.append(total_status_mean[i])
                    boxprops = dict(facecolor=default_colors[r % len(default_colors)], alpha=0.5)
                    flierprops = dict(marker='o', markerfacecolor=default_colors[r % len(default_colors)], markersize=5, linestyle='none', alpha=0.5)
                    box_plot = ax.boxplot(ys, positions=[community_statuses[r]], widths=0.3, patch_artist=True, boxprops=boxprops, flierprops=flierprops)
                    for median in box_plot['medians']: # Set the median color to black
                        median.set_color('grey')
                # for line_position in community_statuses:
                #     ax.axvline(x=line_position, color='gray', linestyle='--', linewidth=0.8)
                # Define custom x tick positions and labels
                ax.set_xticks([-4,-3,-2,-1,0,1,2,3,4])
                ax.set_xticklabels([-4,-3,-2,-1,0,1,2,3,4])

                # ax.scatter(xs,ys, c=colors,zorder=2, alpha = 0.5)
                ax.set_xlabel("Community status")
                ax.set_ylabel("Total status")
                ax.set_title(f"Example statuses of communities and individuals")

                plt.tight_layout(pad = 2)
                plt.savefig(f"{figures_folder}/individuals/{basename}_box_whisker.png")
                plt.savefig(f"{figures_folder}/individuals/{basename}_box_whisker.pdf")


    # Record the relevant data set values for the summary across data sets
    sigma_C_means_all[data_set_name] = sigma_C_means
    sigma_I_means_all[data_set_name] = sigma_I_means
    ns_all[data_set_name] = ns

    # Errors, no log scale
    scatter_plot_with_y_errors(ns,sigma_C_means,sigma_C_stds,data_label,f"Size vs. community status depth in {data_label.lower()}",r'Size $n$',sigma_C_label,'n_vs_sigma_C')
    scatter_plot_with_y_errors(ns,sigma_I_means,sigma_I_stds,data_label,f"Size vs. individual status depth in {data_label.lower()}",r'Size $n$',sigma_I_label,'n_vs_sigma_I')
    scatter_plot_with_y_errors(ns,sigma_T_means,sigma_T_stds,data_label,f"Size vs. total status depth in {data_label.lower()}",r'Size $n$',sigma_T_label,'n_vs_sigma_T')
    scatter_plot_with_y_errors(ns,comm_var_fraction_means,comm_var_fraction_stds,data_label,f"Size vs. community depth fraction in {data_label.lower()}",r'Size $n$',comm_var_fraction_label,'n_vs_comm_var_fraction')

    scatter_plot_with_y_errors(ns,qs,q_stds,data_label,f"Size vs. number of communities in {data_label.lower()}",r'Size $n$',r'Number of communities $q$','n_vs_q_errors')

    if data_set_name == 'highschool': # Only the high schools have the racial data we are familiar with
        scatter_plot_with_y_errors(H_races,sigma_T_means,sigma_T_stds,data_label,f"Racial diversity vs. total status depth in {data_label.lower()}",r'Diversity $H_{race}$',sigma_T_label,'sigma_T_vs_H_race')
        scatter_plot_with_y_errors(percent_whites,sigma_T_means,sigma_T_stds,data_label,f"Percent white students vs. total status depth in {data_label.lower()}",r'Percent white students',sigma_T_label,'sigma_T_vs_percent_white')

    # Errors, log scale
    scatter_plot_log_x_scale_with_y_errors(ns,sigma_C_means,sigma_C_stds,data_label,f"Size vs. community status depth in {data_label.lower()}",r'Size $n$',sigma_C_label,'n_vs_sigma_C_log')
    scatter_plot_log_x_scale_with_y_errors(ns,sigma_I_means,sigma_I_stds,data_label,f"Size vs. individual status depth in {data_label.lower()}",r'Size $n$',sigma_I_label,'n_vs_sigma_I_log')
    scatter_plot_log_x_scale_with_y_errors(ns,sigma_T_means,sigma_T_stds,data_label,f"Size vs. total status depth in {data_label.lower()}",r'Size $n$',sigma_T_label,'n_vs_sigma_T_log')
    scatter_plot_log_x_scale_with_y_errors(ns,comm_var_fraction_means,comm_var_fraction_stds,data_label,f"Size vs. community depth fraction in {data_label.lower()}",r'Size $n$',comm_var_fraction_label,'n_vs_comm_var_fraction_log')

    # No errors, no log scale
    scatter_plot(ns,qs,data_label,f"Size vs. number of communities in {data_label.lower()}",r'Size $n$',r'Number of communities $q$','n_vs_q')

    # Contour plots of all of the schools

    # Array of values
    if data_set_name == 'highschool':
        fig_width = 6
        fig_height = 4.5
        max_sigma_I = 5.5
        max_sigma_C = 4.7
    if data_set_name == 'anyinteractions':
        fig_width = 6
        fig_height = 4.5
        max_sigma_I = 2.5
        max_sigma_C = 2.5
        
    plt.clf()
    ax = draw_contour_background(max_sigma_I,max_sigma_C,fig_width,fig_height)

    ax.errorbar(sigma_I_means,sigma_C_means, xerr=sigma_I_stds, yerr=sigma_C_stds, ecolor='lightgray', fmt='', ls='none', zorder=1, label='High schools')

    # ax.errorbar(village_params_df['sigma_I_avg'],village_params_df['sigma_C_avg'], xerr=village_params_df['sigma_I_std'], yerr=village_params_df['sigma_C_std'], fmt='bo', ecolor='lightblue', zorder=1, label='Indian villages')
    for i, txt in enumerate(range(len(sigma_I_means))):
        ax.annotate(school_nums[txt], (sigma_I_means[i], sigma_C_means[i]))

    # Set labels and title
    # plt.legend()
    ax.set_xlabel(sigma_I_label)
    ax.set_ylabel(sigma_C_label)
    ax.set_title(f"Individual and community status depths in {data_label.lower()}")
    plt.tight_layout(pad = 2)
    plt.savefig(f"{figures_folder}/sigma_I_vs_sigma_C_key.png")   
    plt.savefig(f"{figures_folder}/sigma_I_vs_sigma_C_key.pdf")   


    # Plotting the schools by size:
    plt.clf()
    ax = draw_contour_background(max_sigma_I,max_sigma_C,fig_width,fig_height)

    sizes = np.sqrt(np.array(ns))

    # Color the points by the size
    s_plot = ax.scatter(sigma_I_means,sigma_C_means, s=sizes,c=ns, cmap='viridis', zorder=2)
    # Add colorbar
    cbar = plt.colorbar(s_plot)
    cbar.set_label(r'Size $n$')


    # Set labels and title
    # plt.legend()
    ax.set_xlabel(sigma_I_label)
    ax.set_ylabel(sigma_C_label)
    ax.set_title(f"Individual and community status depths in {data_label.lower()}")
    plt.tight_layout(pad = 2)
    plt.savefig(f"{figures_folder}/sigma_I_vs_sigma_C.png")    
    plt.savefig(f"{figures_folder}/sigma_I_vs_sigma_C.pdf")    


    
# Summary plot of the sigma distributions for all data sets, size present
plt.clf()
fig_width = 5
fig_height = 4.5
max_sigma_I = 5.5
max_sigma_C = 4.7

alpha = 0.5 # Point alpha values

ax = draw_contour_background(max_sigma_I,max_sigma_C,fig_width,fig_height)

for data_set_i, data_set_name in enumerate(data_set_names):
    sizes = np.sqrt(np.array(ns_all[data_set_name]))
    # Scale the point areas by the population size n
    ax.scatter(sigma_I_means_all[data_set_name],sigma_C_means_all[data_set_name], s=sizes, c=default_colors[data_set_i], zorder=2, label=data_set_labels[data_set_name], alpha=alpha)

# Set labels and title
plt.legend()
ax.set_xlabel(sigma_I_label)
ax.set_ylabel(sigma_C_label)
ax.set_title(f"Individual and community status depths")
plt.tight_layout(pad = 2)
plt.savefig(f"{summary_figures_folder}/sigma_I_vs_sigma_C_size.png")   
plt.savefig(f"{summary_figures_folder}/sigma_I_vs_sigma_C_size.pdf")   

    
# Summary plot of the sigma distributions for all data sets, no size present
plt.clf()
ax = draw_contour_background(max_sigma_I,max_sigma_C,fig_width,fig_height)

for data_set_i, data_set_name in enumerate(data_set_names):
    ax.scatter(sigma_I_means_all[data_set_name],sigma_C_means_all[data_set_name], c=default_colors[data_set_i], zorder=2, label=data_set_labels[data_set_name], alpha=alpha)

# Set labels and title
plt.legend()
ax.set_xlabel(sigma_I_label)
ax.set_ylabel(sigma_C_label)
ax.set_title(f"Individual and community status depths")
plt.tight_layout(pad = 2)
plt.savefig(f"{summary_figures_folder}/sigma_I_vs_sigma_C.png")  
plt.savefig(f"{summary_figures_folder}/sigma_I_vs_sigma_C.pdf")  


