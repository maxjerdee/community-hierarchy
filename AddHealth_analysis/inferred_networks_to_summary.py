# Compile the global results of all network inferences into a single summary CSV
# and generate figures summarising the distributions of inferred parameters.
#
# Summary CSV columns:
#   network_name, num_individuals, num_groups, max_modularity, consensus_modularity,
#   sigma_I_mean, sigma_I_std, sigma_C_mean, sigma_C_std,
#   sigma_T_mean, sigma_T_std, comm_var_fraction_mean, comm_var_fraction_std
#   (plus {param}_mean/std for all other PARAMETER_COLS)

# Further plot the sigma_I and sigma_C of each school (with error bars)

import os
import glob
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm

INFERRED_DIR = "inferred_networks"
SUMMARY_CSV = "summary.csv"
FIGURES_DIR = "school_figures"
SUMMARY_FIGURES_DIR = "summary_figures"

# Scalar graph attributes to pull into the summary table (mean column only)
SUMMARY_SCALAR_ATTRS = [
    "num_groups",
    "max_modularity",
    "consensus_modularity",
    "sigma_I_mean",
    "sigma_I_std",
    "sigma_C_mean",
    "sigma_C_std",
    "sigma_T_mean",
    "sigma_T_std",
    "comm_var_fraction_mean",
    "comm_var_fraction_std",
    "density_in_mean",
    "density_in_std",
    "density_out_mean",
    "density_out_std",
    "variation_in_mean",
    "variation_in_std",
    "variation_out_mean",
    "variation_out_std",
    "degree_correction_mean",
    "degree_correction_std",
    "mean_degree_scaling_mean",
    "mean_degree_scaling_std",
    "ties_parameter_mean",
    "ties_parameter_std",
]


def load_inferred_network(gml_path: str) -> dict:
    """Extract one row of summary data from an inferred network GML."""
    G = nx.read_gml(gml_path)
    name = os.path.splitext(os.path.basename(gml_path))[0]
    row: dict = {
        "network_name": name,
        "num_individuals": G.number_of_nodes(),
    }
    for attr in SUMMARY_SCALAR_ATTRS:
        row[attr] = G.graph.get(attr, float("nan"))
    return row


def plot_parameter_distributions(summary_df: pd.DataFrame, figures_dir: str) -> None:
    pass


def plot_sigma_I_sigma_C(summary_df: pd.DataFrame, figures_dir: str) -> None:
    """
    Scatter plot of sigma_I (individual_depth) vs sigma_C (group_depth) for each
    network, with error bars and contour lines showing constant sigma_T.
    """
    sigma_I = summary_df["sigma_I_mean"].to_numpy(dtype=float)
    sigma_C = summary_df["sigma_C_mean"].to_numpy(dtype=float)
    sigma_I_err = summary_df["sigma_I_std"].to_numpy(dtype=float)
    sigma_C_err = summary_df["sigma_C_std"].to_numpy(dtype=float)
    names = summary_df["network_name"].tolist()

    valid = np.isfinite(sigma_I) & np.isfinite(sigma_C)
    if not valid.any():
        print("  No valid sigma_I/sigma_C data — skipping sigma plot.")
        return

    max_I = np.nanmax(sigma_I + sigma_I_err) * 1.2
    max_C = np.nanmax(sigma_C + sigma_C_err) * 1.2
    axis_max = max(max_I, max_C)

    # -- Shared grid for contour background -----------------------------------
    grid = np.arange(0, axis_max, axis_max / 200)
    XI, XC = np.meshgrid(grid, grid)
    sigma_T_grid = np.sqrt(XI ** 2 + XC ** 2)
    contour_levels = np.arange(0.5, axis_max, 0.5)

    # -- Shared helpers -------------------------------------------------------
    n_valid = int(valid.sum())
    cmap = cm.get_cmap("tab10")

    import re as _re
    def _name_label(name: str) -> str:
        """Extract trailing number from network name, e.g. 'highschool_30' -> '30'."""
        m = _re.search(r"(\d+)\D*$", name)
        return m.group(1) if m else name

    def _add_contour_background(ax):
        cs = ax.contour(
            grid, grid, sigma_T_grid,
            levels=contour_levels, colors="gray", linestyles="dashed",
            linewidths=0.8, alpha=0.6,
        )
        ax.clabel(cs, fmt=r"$\sigma_T=%.1f$", fontsize=7, inline=True)
        ax.plot([0, axis_max], [0, axis_max], color="gray", linewidth=0.8, alpha=0.5)
        ax.set_xlim(0, axis_max)
        ax.set_ylim(0, axis_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"Individual depth $\sigma_I$")
        ax.set_ylabel(r"Community depth $\sigma_C$")

    # -- Figure 1: clean dots, no legend, no markings -------------------------
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    _add_contour_background(ax1)
    for i in range(len(names)):
        if not valid[i]:
            continue
        _, caps, bars = ax1.errorbar(
            sigma_I[i], sigma_C[i],
            xerr=sigma_I_err[i], yerr=sigma_C_err[i],
            fmt="o", color=cmap(i % 10),
            capsize=0, markersize=6,
        )
        for b in bars:
            b.set_alpha(0.5)
    ax1.set_title(r"$\sigma_I$ vs $\sigma_C$ across networks")
    plt.tight_layout()
    out_path = os.path.join(figures_dir, "sigma_I_sigma_C.png")
    fig1.savefig(out_path, dpi=150)
    plt.close(fig1)
    print(f"  Saved: {out_path}")

    # -- Figure 2: name-number dots, no legend --------------------------------
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    _add_contour_background(ax2)
    for i, name in enumerate(names):
        if not valid[i]:
            continue
        label = _name_label(name)
        _, caps, bars = ax2.errorbar(
            sigma_I[i], sigma_C[i],
            xerr=sigma_I_err[i], yerr=sigma_C_err[i],
            fmt="o", color=cmap(i % 10),
            capsize=0, markersize=14,
        )
        for b in bars:
            b.set_alpha(0.5)
        ax2.text(
            sigma_I[i], sigma_C[i], label,
            ha="center", va="center", fontsize=6, fontweight="bold", color="white",
        )
    ax2.set_title(r"$\sigma_I$ vs $\sigma_C$ across networks")
    plt.tight_layout()
    numbered_path = os.path.join(figures_dir, "sigma_I_sigma_C_numbered.png")
    fig2.savefig(numbered_path, dpi=150)
    plt.close(fig2)
    print(f"  Saved: {numbered_path}")


def plot_node_scores(gml_path: str, figures_dir: str) -> None:
    """Scatter plot of individual score_mean vs group_score_mean, coloured by consensus group."""
    G = nx.read_gml(gml_path)
    name = os.path.splitext(os.path.basename(gml_path))[0]

    node_names = list(G.nodes())
    groups = np.array([G.nodes[n].get("consensus_group", 0) for n in node_names])
    score_means = np.array([G.nodes[n].get("score_mean", float("nan")) for n in node_names])
    group_score_means = np.array([G.nodes[n].get("group_score_mean", float("nan")) for n in node_names])

    unique_groups = np.unique(groups)
    cmap = cm.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, g in enumerate(sorted(unique_groups, key=lambda g: -np.nanmean(group_score_means[groups == g]))):
        mask = groups == g
        ax.scatter(
            group_score_means[mask],
            score_means[mask],
            color=cmap(idx % 10),
            label=f"Group {g}",
            alpha=0.7,
        )

    # x = y reference line
    all_vals = np.concatenate([group_score_means, score_means])
    lim = (np.nanmin(all_vals), np.nanmax(all_vals))
    ax.plot(lim, lim, "k--", linewidth=1, alpha=0.4, label="x = y")
    ax.set_xlabel("Group score mean")
    ax.set_ylabel("Individual score mean")
    ax.set_title(f"Node scores — {name}")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(figures_dir, f"{name}_node_scores.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    inferred_dir = os.path.join(script_dir, INFERRED_DIR)
    figures_dir = os.path.join(script_dir, FIGURES_DIR)
    os.makedirs(figures_dir, exist_ok=True)
    summary_figures_dir = os.path.join(script_dir, SUMMARY_FIGURES_DIR)
    os.makedirs(summary_figures_dir, exist_ok=True)

    gml_files = sorted(glob.glob(os.path.join(inferred_dir, "*.gml")))
    if not gml_files:
        print(f"No .gml files found in {inferred_dir}")
        return

    print(f"Found {len(gml_files)} inferred network(s).")

    rows = []
    for gml_path in gml_files:
        print(f"  Loading: {os.path.basename(gml_path)}")
        rows.append(load_inferred_network(gml_path))
        plot_node_scores(gml_path, figures_dir)

    summary_df = pd.DataFrame(rows)
    summary_csv_path = os.path.join(script_dir, SUMMARY_CSV)
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSummary written: {summary_csv_path}")
    print(summary_df.to_string(index=False))

    if len(summary_df) > 0:
        plot_sigma_I_sigma_C(summary_df, summary_figures_dir)


if __name__ == "__main__":
    main()
