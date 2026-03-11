# For each samples CSV in samples/, compute the consensus partition, mean/std of scores,
# mean/std of network parameters, and mean score within each consensus group.
# Writes the results as node/graph attributes to an output GML in inferred_networks/.
#
# Graph-level summary attributes written:
#   num_groups             : number of groups in the consensus partition
#   max_modularity         : modularity of the greedy partition from the input GML
#   consensus_modularity   : modularity of the consensus partition
#   sigma_I_mean/std       : posterior mean/std of individual_depth
#   sigma_C_mean/std       : posterior mean/std of group_depth
#   sigma_T_mean/std       : posterior mean/std of sqrt(sigma_I^2 + sigma_C^2)
#   comm_var_fraction_mean/std : posterior mean/std of sigma_C^2 / sigma_T^2
#   (plus {param}_mean/std for all other PARAMETER_COLS)

import os
import glob
import numpy as np
import pandas as pd
import networkx as nx

SAMPLES_DIR = "samples"
DOMINANCE_DIR = "dominance_networks"
INFERRED_DIR = "inferred_networks"

PARAMETER_COLS = [
    "num_groups",
    "density_in",
    "density_out",
    "variation_in",
    "variation_out",
    "degree_correction",
    "mean_degree_scaling",
    "individual_depth",
    "group_depth",
    "ties_parameter",
]


def consensus_clustering(clusterings: np.ndarray, metric: str = "L1") -> np.ndarray:
    """
    Find the clustering from the input samples that minimizes the L1 (or L2) distance
    to the aggregate coincidence matrix.

    The aggregate coincidence matrix agg[i, j] records the fraction of samples in which
    nodes i and j were placed in the same group. The best clustering is the one whose
    indicator matrix (1 if same group, 0 otherwise) is closest to this aggregate.

    Parameters
    ----------
    clusterings : np.ndarray, shape (num_samples, num_nodes)
        Array of integer cluster assignments for each sample.
    metric : str
        'L1' or 'L2'.

    Returns
    -------
    np.ndarray of shape (num_nodes,) -- the consensus clustering.
    """
    num_samples, num_nodes = clusterings.shape

    # Build aggregate coincidence matrix
    agg = np.zeros((num_nodes, num_nodes))
    for clustering in clusterings:
        for group_id in np.unique(clustering):
            members = np.where(clustering == group_id)[0]
            agg[np.ix_(members, members)] += 1.0
    agg /= num_samples

    # Base distance: assume all pairs are in different groups (target = 0 everywhere)
    if metric == "L1":
        base_distance = float(np.sum(np.abs(agg)))
    else:
        base_distance = float(np.sum(agg ** 2))

    best_distance = np.inf
    best_clustering = clusterings[0].copy()

    for clustering in clusterings:
        distance = base_distance
        for group_id in np.unique(clustering):
            members = np.where(clustering == group_id)[0]
            block = agg[np.ix_(members, members)]
            # Adjust within-group pairs: swap target 0 -> 1
            if metric == "L1":
                distance -= float(np.sum(np.abs(block)))
                distance += float(np.sum(np.abs(block - 1.0)))
            else:
                distance -= float(np.sum(block ** 2))
                distance += float(np.sum((block - 1.0) ** 2))
        if distance < best_distance:
            best_distance = distance
            best_clustering = clustering.copy()

    return best_clustering


def process_samples(samples_csv: str, gml_path: str, output_gml: str) -> None:
    """
    Summarise one samples CSV and write an annotated GML.

    Node attributes added:
        consensus_group     : int   -- consensus partition label
        score_mean          : float -- posterior mean of hierarchy score
        score_std           : float -- posterior std of hierarchy score
        group_score_mean    : float -- mean of per-sample group-average score
        group_score_std     : float -- std  of per-sample group-average score

    Graph attributes added:
        num_groups             : int   -- number of groups in the consensus partition
        max_modularity         : float -- modularity of the greedy partition (from input GML)
        consensus_modularity   : float -- modularity of the consensus partition
        sigma_I_mean/std       : float -- individual_depth posterior mean/std
        sigma_C_mean/std       : float -- group_depth posterior mean/std
        sigma_T_mean/std       : float -- sqrt(sigma_I^2+sigma_C^2) posterior mean/std
        comm_var_fraction_mean/std : float -- sigma_C^2/sigma_T^2 posterior mean/std
        {param}_mean, {param}_std  : float -- for all PARAMETER_COLS
    """
    samples_df = pd.read_csv(samples_csv)
    G = nx.read_gml(gml_path)
    node_names = list(G.nodes())

    # -- Consensus partition ---------------------------------------------------
    group_columns = [f"{node}_group" for node in node_names]
    clusterings = samples_df[group_columns].to_numpy().astype(int)  # (num_samples, num_nodes)
    consensus_groups = consensus_clustering(clusterings, metric="L1")

    # -- Score stats -----------------------------------------------------------
    score_columns = [f"{node}_score" for node in node_names]
    scores_mean = samples_df[score_columns].mean().to_numpy()
    scores_std = samples_df[score_columns].std().to_numpy()

    # -- Group score stats (distribution of per-sample group-mean scores) ------
    unique_groups = np.unique(consensus_groups)
    group_members: dict = {int(g): [] for g in unique_groups}
    for i, group in enumerate(consensus_groups):
        group_members[int(group)].append(i)

    group_score_means = np.zeros(len(node_names))
    group_score_stds = np.zeros(len(node_names))
    for group, member_indices in group_members.items():
        member_score_cols = [score_columns[i] for i in member_indices]
        per_sample_group_mean = samples_df[member_score_cols].mean(axis=1)
        gm = float(per_sample_group_mean.mean())
        gs = float(per_sample_group_mean.std())
        for i in member_indices:
            group_score_means[i] = gm
            group_score_stds[i] = gs

    # -- Write node attributes -------------------------------------------------
    for i, node in enumerate(node_names):
        G.nodes[node]["consensus_group"] = int(consensus_groups[i])
        G.nodes[node]["score_mean"] = float(scores_mean[i])
        G.nodes[node]["score_std"] = float(scores_std[i])
        G.nodes[node]["group_score_mean"] = float(group_score_means[i])
        G.nodes[node]["group_score_std"] = float(group_score_stds[i])

    # -- Consensus partition summary ------------------------------------------
    G.graph["num_groups"] = int(len(unique_groups))

    # max_modularity: greedy partition modularity stored in the input GML
    if "modularity" in G.graph:
        G.graph["max_modularity"] = float(G.graph["modularity"])

    # consensus_modularity: modularity of the consensus partition on the graph
    U = G.to_undirected()
    consensus_communities = [
        {node_names[i] for i in indices}
        for indices in group_members.values()
    ]
    try:
        G.graph["consensus_modularity"] = float(
            nx.community.modularity(U, consensus_communities)
        )
    except Exception:
        G.graph["consensus_modularity"] = float("nan")

    # -- Network parameter stats as graph attributes ---------------------------
    for param in PARAMETER_COLS:
        if param in samples_df.columns:
            G.graph[f"{param}_mean"] = float(samples_df[param].mean())
            G.graph[f"{param}_std"] = float(samples_df[param].std())

    # -- Derived depth parameters ---------------------------------------------
    # sigma_I = individual_depth, sigma_C = group_depth
    # sigma_T           = sqrt(sigma_I^2 + sigma_C^2)
    # comm_var_fraction = sigma_C^2 / sigma_T^2
    if "individual_depth" in samples_df.columns and "group_depth" in samples_df.columns:
        sigma_I = samples_df["individual_depth"]
        sigma_C = samples_df["group_depth"]
        sigma_T = np.sqrt(sigma_I ** 2 + sigma_C ** 2)
        comm_var_fraction = sigma_C ** 2 / (sigma_I ** 2 + sigma_C ** 2)
        G.graph["sigma_I_mean"] = float(sigma_I.mean())
        G.graph["sigma_I_std"] = float(sigma_I.std())
        G.graph["sigma_C_mean"] = float(sigma_C.mean())
        G.graph["sigma_C_std"] = float(sigma_C.std())
        G.graph["sigma_T_mean"] = float(sigma_T.mean())
        G.graph["sigma_T_std"] = float(sigma_T.std())
        G.graph["comm_var_fraction_mean"] = float(comm_var_fraction.mean())
        G.graph["comm_var_fraction_std"] = float(comm_var_fraction.std())

    os.makedirs(os.path.dirname(output_gml), exist_ok=True)
    nx.write_gml(G, output_gml)
    print(f"  Written: {output_gml}")


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    samples_dir = os.path.join(script_dir, SAMPLES_DIR)
    dominance_dir = os.path.join(script_dir, DOMINANCE_DIR)
    inferred_dir = os.path.join(script_dir, INFERRED_DIR)

    csv_files = sorted(glob.glob(os.path.join(samples_dir, "*_samples.csv")))
    if not csv_files:
        print(f"No *_samples.csv files found in {samples_dir}")
        return

    for csv_path in csv_files:
        stem = os.path.basename(csv_path).replace("_samples.csv", "")
        gml_path = os.path.join(dominance_dir, f"{stem}.gml")
        if not os.path.exists(gml_path):
            print(f"Skipping {stem}: no matching GML in {dominance_dir}/")
            continue
        print(f"Processing {stem} ...")
        process_samples(csv_path, gml_path, os.path.join(inferred_dir, f"{stem}.gml"))


if __name__ == "__main__":
    main()
