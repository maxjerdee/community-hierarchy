/* The MCMC state and moves of the directedstructure model (see docs/ARCHITECTURE.md). */

#ifndef MCMC_CORE_H
#define MCMC_CORE_H

#include <string>
#include <vector>
#include <unordered_map>
#include <set>

class MCMC {
public:
    // wins/ties: directed and neutral edges (adjacency maps).  group_adj: optional separate
    // undirected graph that drives the community structure (two-layer mode; empty = use
    // wins + ties).  *_fixed flags exclude a variable from the moves.  density_*_initial < 0
    // means "estimate from the initial partition".
    MCMC(const std::vector<std::unordered_map<int, double>>& wins,
        const std::vector<std::unordered_map<int, double>>& ties,
        const std::vector<std::unordered_map<int, double>>& group_adj,
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
        const double& individual_depth,
        const bool& individual_depth_fixed,
        const double& group_depth,
        const bool& group_depth_fixed,
        const double& ties_parameter,
        const bool& ties_parameter_fixed,
        int seed = 0,
        double density_in_initial = -1.0,
        double density_out_initial = -1.0,
        bool weighted_SBM = false);

    /* Fixed information */
    // Graph variables
    int n; // Number of nodes
    int m; // Number of edges
    std::vector<std::unordered_map<int, double>> wins; // wins[i] is the list of wins for node i (dominant edges where i dominates j)
    std::vector<std::unordered_map<int, double>> losses; // losses[i] is the list of losses for node i (dominant edges where j dominates i)
    std::vector<std::unordered_map<int, double>> ties; // ties[i] is the list of ties for node i (neutral/reciprocated edges)
    std::vector<std::unordered_map<int, double>> neighbors; // neighbors[i] is the list of neighbors of node i (undirected, from A — used for H_Adir normalization)
    std::vector<std::unordered_map<int, double>> neighbors_com; // undirected adjacency used for group (SBM) structure (= neighbors in single-layer mode, = group_adj in two-layer mode)
    bool two_layer_mode; // true when a separate group_adj graph drives community structure
    bool weighted_SBM = false; // true → model real-valued edge weights via uniform simplex prior

    // Model & sampling parameters
    std::vector<std::string> free_parameters; // Names of free (tunable) parameters
    // Inverse temperature.  TEMPERING CONVENTION (every move, the merge/split target and
    // the parallel-tempering swaps in fit.cpp must agree on it):
    //     pi_beta  propto  P(b) P(s | b, depths) P(parameters)
    //                      * exp(-beta * (H_A_G_b + H_Adir_G_A_s_nu))
    // i.e. only the DATA likelihood -- the undirected network H_A_G_b (incl. H_W) and the
    // edge directions H_Adir_G_A_s_nu -- is tempered.  The partition prior H_b, the score
    // prior H_s_G_b_depths and the parameter priors are not.  (The score move's proposal
    // is reversible w.r.t. exp(-H_s_G_b_depths), which is what forces H_s to be untempered.)
    double beta;
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
    double individual_depth; // Depth of individuals within each group
    double group_depth; // Depth of groups within the hierarchy
    double ties_parameter; // Parameter controlling the frequency of ties 

    // Parameter controls (whether proposals change these parameters)
    bool partition_fixed; // Fix the partition (no single-site, merge, or split moves)
    bool scores_fixed; // Fix the scores (no score moves)
    bool variation_in_fixed;
    bool variation_out_fixed;
    bool degree_correction_fixed;
    bool num_groups_fixed; // Fix the number of groups
    bool individual_depth_fixed; // Fix the individual depth parameter
    bool group_depth_fixed; // Fix the group depth parameter
    bool ties_parameter_fixed; // Fix the ties parameter
    bool assortative; // false indicates density_in = density_out, true allows them to vary independently

    /* Bookkeeping information (updated according to changes in the state variables) */
    // Entropies (log-likelihoods/description lengths)
    // Partition prior, H(b)
    double H_b; // Total entropy of the partition prior, H(b) = H(q) + H(nb|q) + H(b|nb)

    // Undirected likelihood, H(A|b)
    double H_M; // Entropy of the edge count matrix given the partition, H(M|b,density_in,density_out,variation_in,variation_out) negative-binomial
    double H_k; // Entropy of the degrees given the edge counts and partition, H(k|M,b,degree_correction) Dirichlet-multinomial
    double H_A_G_k_M_b; // Entropy of the network given the degrees, edge counts, and partition, H(A|k,M,b) Poisson
    double H_W = 0.0; // Weighted SBM entropy H(W|A,μ̂,b) + H(μ̂) (zero in non-weighted mode)
    double H_A_G_b; // Total entropy of the network given the partition, H(A|b) = H(k) + H(M|b,k) + H(A|k,M,b) [+ H_W in weighted mode]

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
    std::vector<std::vector<int>> edge_counts; // The edge count matrix M (binary counts in weighted mode, total weight otherwise)
    std::vector<int> edge_counts_sum; // The margin of the edge count matrix M (binary degree stubs in weighted mode)
    std::vector<std::vector<double>> weight_counts; // μ_rs: total real-valued weight between groups (only populated in weighted_SBM mode)
    std::vector<double> weight_counts_sum; // total weight of each group (only in weighted_SBM mode)
    double w_total = 0.0; // total undirected edge weight (constant; only in weighted_SBM mode)
    std::vector<double> group_scaled_score_sums; // Sum of scores in each group

    // Concentration versions of variation parameters
    double concentration_in; // Concentration parameter corresponding to variation_in
    double concentration_out; // Concentration parameter corresponding to variation_out
    double degree_correction_concentration; // Concentration parameter corresponding to degree_correction

    std::vector<int> node_degrees; // Precomputed weighted degree of each node (sum of neighbor weights)
    double node_counts_norm_2; // Sum of squares of node counts
    double density_in; // Typical density of in-group edges
    double density_out; // Typical density of out-group edges
    
    /* Extra variables */
    // Total execution time trackers (in order to optimize the code)
    double time_H_k = 0.0;
    double time_H_M = 0.0;
    double time_H_A_G_k_M_b = 0.0;
    double time_params = 0.0;
    
    // Proposal timing trackers
    double time_single_site_proposals = 0.0; // Cumulative time for single-site proposals
    double time_parameter_proposals = 0.0; // Cumulative time for parameter proposals

    // Proposal tracking
    int total_proposals = 0; // Total number of proposals attempted
    int accepted_proposals = 0; // Total number of proposals accepted
    int single_site_proposals = 0; // Total number of single-site proposals attempted
    int accepted_single_site_proposals = 0; // Total number of single-site proposals accepted
    int merge_proposals = 0;                // Merge proposals attempted
    int accepted_merge_proposals = 0;       // Merge proposals accepted
    int split_proposals = 0;                // Split proposals attempted
    int accepted_split_proposals = 0;       // Split proposals accepted
    int merge_split_proposals = 0;          // Joint merge-split attempted
    int accepted_merge_split_proposals = 0; // Joint merge-split accepted
    int score_proposals = 0; // Total number of score proposals attempted
    int accepted_score_proposals = 0; // Total number of score proposals accepted
    int group_score_proposals = 0; // Total number of group-score-shift proposals attempted
    int accepted_group_score_proposals = 0; // Total number of group-score-shift proposals accepted
    int parameter_proposals = 0; // Total number of parameter proposals attempted
    int accepted_parameter_proposals = 0; // Total number of parameter proposals accepted
    int degree_correction_proposals = 0; // Total number of degree correction proposals attempted
    int accepted_degree_correction_proposals = 0; // Total number of degree correction proposals
    int variation_in_proposals = 0; // Total number of variation_in proposals attempted
    int accepted_variation_in_proposals = 0; // Total number of variation_in proposals
    int variation_out_proposals = 0; // Total number of variation_out proposals attempted
    int accepted_variation_out_proposals = 0; // Total number of variation_out proposals
    int individual_depth_proposals = 0; // Total number of individual_depth proposals attempted
    int accepted_individual_depth_proposals = 0; // Total number of individual_depth proposals
    int group_depth_proposals = 0; // Total number of group_depth proposals attempted
    int accepted_group_depth_proposals = 0; // Total number of group_depth proposals accepted
    int ties_parameter_proposals = 0; // Total number of ties_parameter proposals attempted
    int accepted_ties_parameter_proposals = 0; // Total number of ties_parameter proposals

    /* API */
    void MCMC_sweep(double beta = 1.0, int num_sweeps = 1, bool merge_split_enabled = true); // Perform MCMC sweeps
    void change_parameter(const std::string& param_name, double new_value); // Change a continuous parameter. Accessible for Python entropic sampling.

    // Functions that include the logic to generate a proposal
    void propose_move(); // Generate and accept/reject a random proposal
    void propose_single_site_move(); // Generate and accept/reject a single site move
    void propose_parameter_move(); // Generate and accept/reject a change of a continuous parameter
    void propose_score_move(); // Generate and accept/reject a score change move
    void propose_group_score_move(); // Shift all scores in a random group by a common offset (moves along the sigma_C ridge)

    // Functions that directly modify the Monte Carlo state (including the bookkeeping information like the entropy)
    void propose_merge_move();       // Merge two groups (Peixoto Eq. 17/34)
    void propose_split_move();       // Split one group in two (Eq. 32/33)
    void propose_merge_split_move(); // Joint merge-then-split, q unchanged (Eq. 35/36)

    /* Merge/split internals -- see src/MCMC_merge_split.cpp */
    double merge_split_log_pi() const;
    static double split_label_factor(int K);  //  log(K+1): partition/labelled multiplicity
    static double merge_label_factor(int K);  // -log(K):   partition/labelled multiplicity
    double merge_logprob(int K) const;        // log P of proposing a given merged partition
    double split_gibbs_sweep(const std::vector<int>& nodes, int ga, int gb,
                             bool accumulate, const std::vector<char>* target,
                             std::vector<char>* out_assignment);
    void set_split_assignment(const std::vector<int>& nodes, int ga, int gb,
                              const std::vector<char>& assignment);
    int stage_split(const std::vector<int>& nodes, int ga, int sweeps);
    int prestage_random(const std::vector<int>& nodes, int ga);
    int prestage_sequential(const std::vector<int>& nodes, int ga, bool coalescence);
    double split_partition_logprob(const std::vector<int>& nodes, int ga, int gb,
                                   const std::vector<char>& staged,
                                   const std::vector<char>& assignment);

    void flip_nodes(const std::vector<int>& nodes, int new_group); // Flip multiple nodes of the same old_group to a single new_group
    void swap_groups(int group1, int group2); // Swap the assignments of two groups
    
    void check_state_bookkeeping(bool throw_error = true); // Recompute all the bookkeeping information and check that it is consistent with the current state.
    // Optionally, do not throw an error if the bookkeeping is inconsistent, just update the bookkeeping information.

    double calculate_H_k(); // Calculate the entropy of the degrees given the current state (H(k|M,b,alpha))
    double calculate_H_M(); // Calculate the entropy of the edge count matrix given the degrees in the current state (H(M|b,rho_in,rho_out) or H(M|m), etc.)
    double calculate_H_A_G_k_M_b(); // Calculate the entropy of the network given the edge count matrix in the current state (H(A|k,M,b))
    double calculate_H_W(); // Calculate weighted SBM entropy H(W|A,μ̂,b) + H(μ̂) (only valid in weighted_SBM mode)
    double calculate_H_Adir_G_A_s_nu(); // Calculate the entropy of the directed edges given the undirected edges and scores (H(Adir|A,s,nu))
    double calculate_H_s_G_b_depths(); // Calculate the entropy of the scores given the partition and depth parameters (H(s|b,depths))

    // Printing/debugging functions
    std::string to_string_verbose() const; // Convert the state to a string to be logged
    std::string to_string_csv() const; // Convert the state to a CSV string to be written to a file
};

#endif // MCMC_CORE_H