/* Helper function declarations */

#ifndef MCMC_CORE_H
#define MCMC_CORE_H

#include <string>
#include <vector>
#include <unordered_map>
#include <set>

class MCMC {
public:
    // New constructor signature (matches the implementation order):
    // MCMC(wins, ties, initial_partition, scores, num_groups, num_groups_fixed, assortative,
    //      variation_in, variation_in_fixed, variation_out, variation_out_fixed,
    //      degree_correction, degree_correction_fixed, mean_degree_scaling, mean_degree_scaling_fixed,
    //      individual_depth, individual_depth_fixed, group_depth, group_depth_fixed,
    //      ties_parameter, ties_parameter_fixed, seed)
    MCMC(const std::vector<std::unordered_map<int, double>>& wins,
        const std::vector<std::unordered_map<int, double>>& ties,
        const std::vector<int>& partition,
        const bool& partition_fixed,
        const std::vector<double>& scores,
        const bool& scores_fixed,
        int num_groups,
        const bool& num_groups_fixed,
        const bool& assortative,
        const double& variation_in,
        const bool& variation_in_fixed,
        const double& variation_out,
        const bool& variation_out_fixed,
        const double& degree_correction,
        const bool& degree_correction_fixed,
        const double& mean_degree_scaling,
        const bool& mean_degree_scaling_fixed,
        const double& individual_depth,
        const bool& individual_depth_fixed,
        const double& group_depth,
        const bool& group_depth_fixed,
        const double& ties_parameter,
        const bool& ties_parameter_fixed,
        int seed = 0);

    /* Fixed information */
    // Graph variables
    int n; // Number of nodes
    int m; // Number of edges
    std::vector<std::unordered_map<int, double>> wins; // wins[i] is the list of wins for node i (dominant edges where i dominates j)
    std::vector<std::unordered_map<int, double>> losses; // losses[i] is the list of losses for node i (dominant edges where j dominates i)
    std::vector<std::unordered_map<int, double>> ties; // ties[i] is the list of ties for node i (neutral/reciprocated edges)
    std::vector<std::unordered_map<int, double>> neighbors; // neighbors[i] is the list of neighbors of node i (undirected)

    // Model & sampling parameters
    std::vector<std::string> free_parameters; // Names of free (tunable) parameters
    double beta; // Inverse temperature for MCMC
    bool merge_split_enabled; // Controls whether merge/split proposals are attempted
    // Map of the types of proposals to their weights (for weighted random selection of proposal types)
    std::unordered_map<std::string, double> proposal_weights;

    /* Monte Carlo state variables */
    // Latent variables (changes are proposed on these variables)
    std::vector<int> partition; // Group assignments of each node

    std::vector<double> scores; // Hierarchical scores of each node
    std::vector<double> scaled_scores; // Scores divided by the individual depth. MCMC moves are proposed according to these for better ergodicity.

    // Model parameters
    double variation_in; // Variation in in-group edge weights
    double variation_out; // Variation in out-group edge weights
    double degree_correction; // Degree correction parameter (0 = non-degree corrected, 0.5 = traditional degree correction)
    double mean_degree_scaling; // Mean degree scaling parameter (0 = canonical, -1 = microcanonical)
    // Typical weights, equal to the density when mean_degree_scaling = 0, proposals are made on these so they do not need to be updated each partition move.
    double weight_in; // Typical in-group weight, weight_in = density_in*node_counts_norm_2/node_counts_norm_2scaling_plus_2
    double weight_out; // Typical out-group weight, weight_out = density_out*(n^2 - node_counts_norm_2)/(node_counts_norm_scaling_plus_1^2 - node_counts_norm_2scaling_plus_2)
    double individual_depth; // Depth of individuals within each group
    double group_depth; // Depth of groups within the hierarchy
    double ties_parameter; // Parameter controlling the frequency of ties 

    // Parameter controls (whether proposals change these parameters)
    bool partition_fixed; // Fix the partition (no single-site, merge, or split moves)
    bool scores_fixed; // Fix the scores (no score moves)
    bool variation_in_fixed;
    bool variation_out_fixed;
    bool degree_correction_fixed;
    bool mean_degree_scaling_fixed;
    bool num_groups_fixed; // Fix the number of groups
    bool individual_depth_fixed; // Fix the individual depth parameter
    bool group_depth_fixed; // Fix the group depth parameter
    bool ties_parameter_fixed; // Fix the ties parameter
    bool assortative; // false indicates density_in = density_out, true allows them to vary independently

    /* Bookkeeping information (updated according to changes in the state variables) */
    // Entropies (log-likelihoods/description lengths)
    // Parameter prior, H(density_in,density_out), other parameters are sampled in a way that does not require explicit priors
    double H_parameters;
 
    // Partition prior, H(b)
    double H_b; // Total entropy of the partition prior, H(b) = H(q) + H(nb|q) + H(b|nb)

    // Undirected likelihood, H(A|b)
    double H_M; // Entropy of the edge count matrix given the partition, H(M|b,density_in,density_out,variation_in,variation_out) negative-binomial
    double H_k; // Entropy of the degrees given the edge counts and partition, H(k|M,b,degree_correction) Dirichlet-multinomial
    double H_A_G_k_M_b; // Entropy of the network given the degrees, edge counts, and partition, H(A|k,M,b) Poisson
    double H_A_G_b; // Total entropy of the network given the partition, H(A|b) = H(k) + H(M|b,k) + H(A|k,M,b)

    // Undirected description length, H(A,b)
    double H_A_b; // Total description length of the network and partition, H(A,b) = H(b) + H(A|b)
    
    // Direction likelihood, H(Adir|A,s)
    double H_Adir_G_A_s_nu; // Entropy of the directed edges given the undirected edges and scores, Bradley-Terry

    // Score-community coupling
    double H_s_G_b_depths; // Entropy of the scores given the partition and depth parameters

    // Further bookkeeping information
    int num_groups; // Number of groups (communities)
    std::vector<int> node_counts; // Number of nodes in each community, nb
    std::vector<std::vector<int>> group_members; // List of nodes in each community, for efficient random access
    std::vector<std::unordered_map<int, size_t>> node_indices_in_groups; // Maps node ID to its index in group_members for efficient addition/removal
    std::vector<std::vector<int>> edge_counts; // The edge count matrix M (the number of edges between each pair of groups)
    std::vector<int> edge_counts_sum; // The margin of the edge count matrix M (the number of edge stubs in each group)
    std::vector<double> group_scaled_score_sums; // Sum of scores in each group

    // Concentration versions of variation parameters
    double concentration_in; // Concentration parameter corresponding to variation_in
    double concentration_out; // Concentration parameter corresponding to variation_out
    double degree_correction_concentration; // Concentration parameter corresponding to degree_correction

    // Normalizations of node_count used to adjust the probabilities of connections
    double node_counts_norm_2; // Sum of squares of node counts
    double node_counts_norm_scaling_plus_1; // Sum of node counts to the power mean_degree_scaling + 1
    double node_counts_norm_2scaling_plus_2; // Sum of node counts to the power 2*mean_degree_scaling + 2
    // Typical densities
    double density_in; // Typical density of in-group edges
    double density_out; // Typical density of out-group edges
    
    /* Extra variables */
    // Wang-Landau algorithm for density of states in num_groups
    double wang_landau_modification_factor = -1; // Modification factor f for Wang-Landau. Default -1 indicates WL is not being used.
    std::vector<double> log_density_num_groups; // Estimate of the log density of states for each num_groups
    int max_num_groups_cutoff = -1; // Maximum number of groups to track in Wang-Landau to avoid computationally expensive inference at large group numbers. Default -1 indicates no cutoff.

    // Total execution time trackers (in order to optimize the code)
    double time_H_k = 0.0;
    double time_H_M = 0.0;
    double time_H_A_G_k_M_b = 0.0;
    double time_params = 0.0;
    
    // Proposal timing trackers
    double time_single_site_proposals = 0.0; // Cumulative time for single-site proposals
    double time_merge_proposals = 0.0; // Cumulative time for merge proposals
    double time_split_proposals = 0.0; // Cumulative time for split proposals
    double time_parameter_proposals = 0.0; // Cumulative time for parameter proposals

    // Proposal tracking
    int total_proposals = 0; // Total number of proposals attempted
    int accepted_proposals = 0; // Total number of proposals accepted
    int merge_proposals = 0; // Total number of merge/split proposals attempted
    int accepted_merge_proposals = 0; // Total number of merge/split proposals accepted
    int split_proposals = 0; // Total number of split proposals attempted
    int accepted_split_proposals = 0; // Total number of split proposals accepted
    int single_site_proposals = 0; // Total number of single-site proposals attempted
    int accepted_single_site_proposals = 0; // Total number of single-site proposals accepted
    int score_proposals = 0; // Total number of score proposals attempted
    int accepted_score_proposals = 0; // Total number of score proposals accepted
    int parameter_proposals = 0; // Total number of parameter proposals attempted
    int accepted_parameter_proposals = 0; // Total number of parameter proposals accepted
    int degree_correction_proposals = 0; // Total number of degree correction proposals attempted
    int accepted_degree_correction_proposals = 0; // Total number of degree correction proposals
    int variation_in_proposals = 0; // Total number of variation_in proposals attempted
    int accepted_variation_in_proposals = 0; // Total number of variation_in proposals
    int variation_out_proposals = 0; // Total number of variation_out proposals attempted
    int accepted_variation_out_proposals = 0; // Total number of variation_out proposals
    int mean_degree_scaling_proposals = 0; // Total number of mean_degree_scaling proposals attempted
    int accepted_mean_degree_scaling_proposals = 0; // Total number of mean_degree_scaling proposals
    int individual_depth_proposals = 0; // Total number of individual_depth proposals attempted
    int accepted_individual_depth_proposals = 0; // Total number of individual_depth proposals
    int group_depth_proposals = 0; // Total number of group_depth proposals attempted
    int accepted_group_depth_proposals = 0; // Total number of group_depth proposals accepted
    int ties_parameter_proposals = 0; // Total number of ties_parameter proposals attempted
    int accepted_ties_parameter_proposals = 0; // Total number of ties_parameter proposals

    /* API */
    void MCMC_sweep(double beta = 1.0, bool merge_split_enabled = true, double wang_landau_modification_factor = -1.0, int max_num_groups_cutoff = -1, int num_sweeps = 1); // Perform MCMC sweeps; merge_split toggles merge/split proposals
    void change_parameter(const std::string& param_name, double new_value); // Change a continuous parameter. Accessible for Python entropic sampling. 

    // Functions that include the logic to generate a proposal
    void propose_move(); // Generate and accept/reject a random proposal
    void propose_merge_move(); // Generate and accept/reject a group merge move
    void propose_split_move(); // Generate and accept/reject a group split move
    void propose_single_site_move(); // Generate and accept/reject a single site move
    void propose_parameter_move(); // Generate and accept/reject a change of a continuous parameter
    void propose_score_move(); // Generate and accept/reject a score change move

    // Functions that directly modify the Monte Carlo state (including the bookkeeping information like the entropy)
    void flip_nodes(const std::vector<int>& nodes, int new_group); // Flip multiple nodes of the same old_group to a single new_group
    void swap_groups(int group1, int group2); // Swap the assignments of two groups
    // void change_parameter(const std::string& param_name, double new_value); [Public API above]

    // Merge-split functions
    // Generate a hueristic split of a group into two groups
    void hueristic_split_group(std::vector<int> groups_to_split); // Give the groups of nodes to split (can be one or two groups)
    void hueristic_split_group_random(std::vector<int> nodes_to_split, std::vector<int> groups_after_split); // Random assignment hueristic
    void hueristic_split_group_sequential_new_group(std::vector<int> nodes_to_split, std::vector<int> groups_after_split); // Sequential assignment from a single new group
    void hueristic_split_group_sequential_new_groups(std::vector<int> nodes_to_split, std::vector<int> groups_after_split); // Sequential assignment from individual new groups
    double gibbs_sweep(std::vector<int> nodes_to_sweep, std::vector<int> group_choices); // Refine a split along the given groups with a Gibbs sweep, returns probability of the split given the staging configuration
    double H_reverse_merge(int old_group, int new_group); // Entropy of proposing a merge from new_group to old_group
    double H_reverse_split(int split_group, std::vector<int> split_nodes); // Entropy of proposing a split of split_group into a new group with the given nodes
    
    void check_state_bookkeeping(bool throw_error = true); // Recompute all the bookkeeping information and check that it is consistent with the current state.
    // Optionally, do not throw an error if the bookkeeping is inconsistent, just update the bookkeeping information.

    double calculate_H_k(); // Calculate the entropy of the degrees given the current state (H(k|M,b,alpha))
    double calculate_H_M(); // Calculate the entropy of the edge count matrix given the degrees in the current state (H(M|b,rho_in,rho_out) or H(M|m), etc.)
    double calculate_H_A_G_k_M_b(); // Calculate the entropy of the network given the edge count matrix in the current state (H(A|k,M,b))
    double calculate_H_Adir_G_A_s_nu(); // Calculate the entropy of the directed edges given the undirected edges and scores (H(Adir|A,s,nu))
    double calculate_H_s_G_b_depths(); // Calculate the entropy of the scores given the partition and depth parameters (H(s|b,depths))

    // Printing/debugging functions
    std::string to_string_verbose() const; // Convert the state to a string to be logged
    std::string to_string_csv() const; // Convert the state to a CSV string to be written to a file
};

#endif // MCMC_CORE_H