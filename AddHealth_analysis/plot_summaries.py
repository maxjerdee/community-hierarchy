# Plot summary information

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy import stats

input_folder = "summaries"
output_folder = "figures"
data_folder = "data"

model_names = [
    "bradley_terry_general_canonical",
    "bradley_terry_ties_general_canonical",
    "bradley_terry_traditional_DCSBM",
    "bradley_terry_ties_traditional_DCSBM",
    "bradley_terry_independent",
    "bradley_terry_ties_independent",
]

# Labels for plots
individual_depth_label = r'Individual depth $\sigma_I$'
group_depth_label = r'Group depth $\sigma_C$'
total_depth_label = r'Total depth $\sigma_T$'


def draw_contour_background(max_individual_depth, max_group_depth, fig_width=10, fig_height=6):
    """
    Draw a contour plot background for depth comparison plots.
    Contours represent constant total depth (sqrt(individual_depth^2 + group_depth^2))
    """
    # Grid on which to evaluate the total depth values
    all_individual_depths = np.arange(0, max_individual_depth, 0.1)
    all_group_depths = np.arange(0, max_group_depth, 0.1)
    probs = np.zeros((len(all_individual_depths), len(all_group_depths)))
    
    for ind_i in range(len(all_individual_depths)):
        for grp_i in range(len(all_group_depths)):
            probs[ind_i, grp_i] = np.sqrt(
                all_individual_depths[ind_i]**2 + all_group_depths[grp_i]**2
            )

    # Create figure and axes
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(111)
    ax.set_xlim((0, max_individual_depth))
    ax.set_ylim((0, max_group_depth))
    # Set aspect ratio to equal
    ax.set_aspect('equal', adjustable='box')

    # Draw contour lines for constant total depth
    contour_levels = np.arange(1, 10, 1)
    contour = ax.contour(
        all_individual_depths, all_group_depths, probs.T,
        levels=contour_levels, colors='gray', linestyles='dashed'
    )
    
    # Draw the line where individual_depth = group_depth
    ax.plot(all_group_depths, all_group_depths, color='gray')

    return ax


def scatter_plot_with_y_errors(xs, ys, yerrs, model_name, x_label, y_label, basename):
    """Create a scatter plot with error bars and fitted line"""
    plt.clf()
    plt.scatter(xs, ys, zorder=2)
    plt.errorbar(xs, ys, yerr=yerrs, ecolor='lightgray', fmt='', ls='none', zorder=1)
    
    # Handle zero or very small error bars
    yerrs = np.array(yerrs)
    xs = np.array(xs)
    ys = np.array(ys)
    
    # Replace zero or very small errors with small value to avoid division by zero
    yerrs = np.where(yerrs < 1e-10, 1.0, yerrs)
    
    # Perform weighted least squares
    weights = 1 / yerrs**2  # Weights are inverse variance
    
    # Check for invalid weights (NaN or inf)
    if not np.all(np.isfinite(weights)):
        weights = np.ones_like(weights)  # Fall back to unweighted
    
    # Weighted least squares: fit y = slope * x + intercept
    W = np.diag(weights)
    X = np.column_stack([np.ones_like(xs), xs])
    
    try:
        # Solve weighted normal equations: (X^T W X)^-1 X^T W y
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ ys
        params = np.linalg.solve(XtWX, XtWy)
        intercept, slope = params
        
        # Calculate residuals and chi-squared
        y_fit = slope * xs + intercept
        residuals = ys - y_fit
        chi2 = np.sum(weights * residuals**2)
        dof = len(xs) - 2  # degrees of freedom
        
        # Calculate standard error of slope
        cov_matrix = np.linalg.inv(XtWX)
        slope_stderr = np.sqrt(cov_matrix[1, 1])
        
        # Calculate t-statistic and p-value
        if slope_stderr > 0:
            t_stat = slope / slope_stderr
            from scipy.stats import t as t_dist
            p_value = 2 * (1 - t_dist.cdf(abs(t_stat), dof))
        else:
            p_value = np.nan
        
        # Plot fitted line
        x_fit = np.linspace(xs.min(), xs.max(), 100)
        y_fit_plot = slope * x_fit + intercept
        plt.plot(x_fit, y_fit_plot, color='red', label='Fitted Line', zorder=0)
        
    except (np.linalg.LinAlgError, ValueError):
        # If weighted fitting fails, fall back to unweighted
        slope, intercept, r_value, p_value, std_err = stats.linregress(xs, ys)
        x_fit = np.linspace(xs.min(), xs.max(), 100)
        y_fit_plot = slope * x_fit + intercept
        plt.plot(x_fit, y_fit_plot, color='red', label='Fitted Line', zorder=0)
    
    plt.title(f"{model_name}: slope = {slope:.3f}, p = {p_value:.3f}")
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    # Set y-axis limits: cap at 10, but use smaller limit if data allows
    max_y_data = np.max(ys + yerrs)
    y_upper = min(max_y_data * 1.1, 10)  # Add 10% margin, cap at 10
    plt.ylim(0, y_upper)
    
    plt.tight_layout(pad=2)
    plt.savefig(os.path.join(output_folder, f"{basename}.png"))
    plt.close()


for model_name in model_names:
    model_folder = os.path.join(input_folder, model_name)
    is_independent_model = "_independent" in model_name

    individual_depth_means = []
    individual_depth_stds = []
    group_depth_means = []
    group_depth_stds = []
    
    # Network metrics
    num_nodes = []
    num_communities = []
    racial_diversity = []  # Entropy of racial distribution
    percent_white = []
    percent_black = []
    percent_hispanic = []
    
    # Track filenames for reference
    basenames = []

    for filename in os.listdir(model_folder):
        if filename.endswith(".csv") and "network_properties" in filename:
            input_filename = os.path.join(model_folder, filename)
            print(f"Processing {input_filename}")
            network_properties_df = pd.read_csv(input_filename)
            
            individual_depth_means.append(network_properties_df.loc[network_properties_df['parameter'] == 'individual_depth', 'mean'].values[0])
            individual_depth_stds.append(network_properties_df.loc[network_properties_df['parameter'] == 'individual_depth', 'std'].values[0])
            group_depth_means.append(network_properties_df.loc[network_properties_df['parameter'] == 'group_depth', 'mean'].values[0])
            group_depth_stds.append(network_properties_df.loc[network_properties_df['parameter'] == 'group_depth', 'std'].values[0])
            
            # Extract basename to find corresponding GML file
            basename = filename.replace("_network_properties.csv", "")
            basenames.append(basename)
            
            # Load the corresponding GML file to extract network metrics
            gml_path = os.path.join(data_folder, f"{basename}.gml")
            if os.path.exists(gml_path):
                G = nx.read_gml(gml_path)
                n = len(G.nodes())
                num_nodes.append(n)
                
                # Extract node attributes
                races = []
                for node, attributes in G.nodes(data=True):
                    if 'race' in attributes:
                        races.append(attributes['race'])
                
                # Count communities (from node attribute or compute)
                communities = set()
                for node, attributes in G.nodes(data=True):
                    if 'community' in attributes:
                        communities.add(attributes['community'])
                num_communities.append(len(communities))
                
                # Compute racial diversity
                if len(races) > 0:
                    race_counts = {}
                    for race in races:
                        race_counts[race] = race_counts.get(race, 0) + 1
                    
                    # Compute entropy
                    H_race = 0
                    for count in race_counts.values():
                        p = count / len(races)
                        if p > 0:
                            H_race -= p * np.log(p)
                    racial_diversity.append(H_race)
                    
                    # Compute percentages
                    percent_white.append(race_counts.get('white', 0) / len(races))
                    percent_black.append(race_counts.get('black', 0) / len(races))
                    percent_hispanic.append(race_counts.get('hispanic', 0) / len(races))
                else:
                    racial_diversity.append(0)
                    percent_white.append(0)
                    percent_black.append(0)
                    percent_hispanic.append(0)

    # Convert to numpy arrays for easier manipulation
    individual_depth_means = np.array(individual_depth_means)
    individual_depth_stds = np.array(individual_depth_stds)
    group_depth_means = np.array(group_depth_means)
    group_depth_stds = np.array(group_depth_stds)
    
    num_nodes = np.array(num_nodes)
    num_communities = np.array(num_communities)
    racial_diversity = np.array(racial_diversity)
    percent_white = np.array(percent_white)
    percent_black = np.array(percent_black)
    percent_hispanic = np.array(percent_hispanic)
    
    # Compute total depth
    total_depth_means = np.sqrt(individual_depth_means**2 + group_depth_means**2)

    # Determine plot limits based on data, capped at 10
    max_individual_depth = min(individual_depth_means.max() + individual_depth_stds.max() + 0.5, 10)
    max_group_depth = min(group_depth_means.max() + group_depth_stds.max() + 0.5, 10)

    # ====================================
    # Contour plot with error bars
    # ====================================
    ax = draw_contour_background(max_individual_depth, max_group_depth, fig_width=10, fig_height=6)
    
    ax.errorbar(
        individual_depth_means, group_depth_means,
        xerr=individual_depth_stds, yerr=group_depth_stds,
        fmt='o', ecolor='r', capthick=2, label='Data', zorder=2
    )
    
    ax.set_title(f"Depth Decomposition for {model_name}")
    ax.set_xlabel(individual_depth_label)
    ax.set_ylabel(group_depth_label)
    ax.legend()
    plt.tight_layout(pad=2)
    plt.savefig(os.path.join(output_folder, f"{model_name}_depth_comparison.png"))
    plt.close()

    # ====================================
    # Comparison plots with network metrics
    # ====================================
    
    # Create output subfolder for this model
    model_output_folder = os.path.join(output_folder, model_name)
    os.makedirs(model_output_folder, exist_ok=True)
    
    # Plot 1: Individual depth vs number of nodes
    if len(num_nodes) > 0 and np.any(num_nodes > 0):
        scatter_plot_with_y_errors(
            num_nodes, individual_depth_means, individual_depth_stds,
            model_name,
            r'Number of nodes $n$', individual_depth_label,
            os.path.join(model_name, 'individual_depth_vs_num_nodes')
        )
        
        if not is_independent_model:
            scatter_plot_with_y_errors(
                num_nodes, group_depth_means, group_depth_stds,
                model_name,
                r'Number of nodes $n$', group_depth_label,
                os.path.join(model_name, 'group_depth_vs_num_nodes')
            )
    
    # Plot 2: Depth vs number of communities
    if len(num_communities) > 0 and np.any(num_communities > 0):
        scatter_plot_with_y_errors(
            num_communities, individual_depth_means, individual_depth_stds,
            model_name,
            r'Number of communities $q$', individual_depth_label,
            os.path.join(model_name, 'individual_depth_vs_num_communities')
        )
        
        if not is_independent_model:
            scatter_plot_with_y_errors(
                num_communities, group_depth_means, group_depth_stds,
                model_name,
                r'Number of communities $q$', group_depth_label,
                os.path.join(model_name, 'group_depth_vs_num_communities')
            )
    
    # Plot 3: Depth vs racial diversity
    if len(racial_diversity) > 0 and np.any(racial_diversity > 0):
        scatter_plot_with_y_errors(
            racial_diversity, individual_depth_means, individual_depth_stds,
            model_name,
            r'Racial Diversity $H_{race}$', individual_depth_label,
            os.path.join(model_name, 'individual_depth_vs_racial_diversity')
        )
        
        if not is_independent_model:
            scatter_plot_with_y_errors(
                racial_diversity, group_depth_means, group_depth_stds,
                model_name,
                r'Racial Diversity $H_{race}$', group_depth_label,
                os.path.join(model_name, 'group_depth_vs_racial_diversity')
            )
    
    # Plot 4: Depth vs percent white
    if len(percent_white) > 0:
        scatter_plot_with_y_errors(
            percent_white, individual_depth_means, individual_depth_stds,
            model_name,
            r'Percent white students', individual_depth_label,
            os.path.join(model_name, 'individual_depth_vs_percent_white')
        )
        
        if not is_independent_model:
            scatter_plot_with_y_errors(
                percent_white, group_depth_means, group_depth_stds,
                model_name,
                r'Percent white students', group_depth_label,
                os.path.join(model_name, 'group_depth_vs_percent_white')
            )
    
    # Plot 5: Depth vs percent black
    if len(percent_black) > 0:
        scatter_plot_with_y_errors(
            percent_black, individual_depth_means, individual_depth_stds,
            model_name,
            r'Percent black students', individual_depth_label,
            os.path.join(model_name, 'individual_depth_vs_percent_black')
        )
        
        if not is_independent_model:
            scatter_plot_with_y_errors(
                percent_black, group_depth_means, group_depth_stds,
                model_name,
                r'Percent black students', group_depth_label,
                os.path.join(model_name, 'group_depth_vs_percent_black')
            )
    
    # Plot 6: Depth vs percent hispanic
    if len(percent_hispanic) > 0:
        scatter_plot_with_y_errors(
            percent_hispanic, individual_depth_means, individual_depth_stds,
            model_name,
            r'Percent hispanic students', individual_depth_label,
            os.path.join(model_name, 'individual_depth_vs_percent_hispanic')
        )
        
        if not is_independent_model:
            scatter_plot_with_y_errors(
                percent_hispanic, group_depth_means, group_depth_stds,
                model_name,
                r'Percent hispanic students', group_depth_label,
                os.path.join(model_name, 'group_depth_vs_percent_hispanic')
            )