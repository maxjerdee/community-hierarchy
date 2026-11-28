/* The MCMC state and moves of the community_hierarchy model (see docs/ARCHITECTURE.md). */

#ifndef MCMC_CORE_H
#define MCMC_CORE_H

#include <string>
#include <algorithm>
#include <vector>
#include <utility>
#include <unordered_map>
#include <set>

#include "globals.h"

class MCMC {
public:
    // Adjacency rows are kept SORTED by node id: they are summed over, and an
    // unordered_map's iteration order differs between standard libraries, which
    // makes a seeded run diverge between the native and wasm builds (see rng.h).
    // wins/ties: directed and neutral edges (adjacency maps).  group_adj: optional separate
    // undirected graph that drives the community structure (two-layer mode; empty = use
    // wins + ties).  *_fixed flags exclude a variable from the moves.  density_*_initial < 0
    // means "estimate from the initial partition".  Input rules (std::invalid_argument
    // otherwise): no self-loops, ties given in both directions, finite weights --
    // non-negative integers (multiplicities) in the plain mode, > 0 with weighted_SBM
    // or a group_adj.  q is taken from the partition; num_groups must match it when
    // num_groups_fixed.  ties_parameter fixed at 0 = plain Bradley-Terry: ties then
    // count in the SBM but not in the direction likelihood.  See MCMC_core.cpp.
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
    std::vector<std::vector<std::pair<int, double>>> wins;  // wins[i] = dominant edges i -> j, SORTED by node id
    std::vector<std::vector<std::pair<int, double>>> losses;  // losses[i] = dominant edges j -> i, SORTED by node id
    std::vector<std::vector<std::pair<int, double>>> ties;  // ties[i] = neutral edges at i, SORTED by node id
    std::vector<std::vector<std::pair<int, double>>> neighbors;  // neighbors[i] = undirected adjacency wins + ties, SORTED by node id
    std::vector<std::vector<std::pair<int, double>>> neighbors_com;  // undirected adjacency driving the groups (= neighbors, or group_adj in two-layer mode), SORTED by node id
    bool two_layer_mode; // true when a separate group_adj graph drives community structure
    bool weighted_SBM = false; // true → model real-valued edge weights via uniform simplex prior
    bool weighted_proposals = false; // block-informed proposals use edge weights (weighted_SBM only; off by default, as in FitParams)
    // false when ties_parameter is fixed at 0 (plain Bradley-Terry): a tie then has
    // probability 0, so neutral edges are left out of the direction likelihood
    // (ties is emptied) while still counting in the SBM through neighbors.
    bool ties_in_direction_model = true;
    // The partition- and score-independent part of H_Adir_G_A_s_nu:
    // sum over outcome counts of log w_ij! (+ log t_ij!) - sum_{i<j} log A^dir_ij!.
    double H_Adir_counts_term = 0.0;

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
    // Fixed order, not an unordered_map: propose_move walks this to CHOOSE the
    // move, so bucket order (libstdc++ vs libc++) would pick different moves for
    // the same draw.  See rng.h.
    std::vector<std::pair<std::string, double>> proposal_weights;

    /* Monte Carlo state variables */
    // Latent variables (changes are proposed on these variables)
    std::vector<int> partition; // Group assignments of each node

    std::vector<double> scores; // Hierarchical scores of each node

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
    // Per-object override of MERGE_SPLIT_WEIGHT, set by fit()'s equal-spend
    // retuning.  Negative means "use the global".  Kept per object so adapting
    // one chain cannot leak into another, into the caller, or into a later fit.
    double collective_weight_override = -1.0;
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
    double H_A_G_k_M_b; // Entropy of the network given the degrees, edge counts, and partition, H(A|k,M,b): microcanonical configuration model (Peixoto 2017)
    double H_W = 0.0; // Weighted SBM entropy -log P(W|A,b) (zero in non-weighted mode)
    double H_A_G_b; // Total entropy of the network given the partition, H(A|b) = H(k) + H(M|b,k) + H(A|k,M,b) [+ H_W in weighted mode]

    // Undirected description length, H(A,b)
    double H_A_b; // Total description length of the network and partition, H(A,b) = H(b) + H(A|b)
    
    // Direction likelihood, H(Adir|A,s)
    double H_Adir_G_A_s_nu; // Entropy of the edge outcomes given the undirected edges and scores: Bradley-Terry with Davidson (1970) ties

    // Score-community coupling
    double H_s_G_b_depths; // Entropy of the scores given the partition and depth parameters

    // Further bookkeeping information
    int num_groups; // Number of groups (communities)
    std::vector<int> node_counts; // Number of nodes in each community, nb
    std::vector<std::vector<int>> group_members; // List of nodes in each community, for efficient random access
    std::vector<std::unordered_map<int, size_t>> node_indices_in_groups; // Maps node ID to its index in group_members for efficient addition/removal
    std::vector<std::vector<int>> edge_counts; // The edge count matrix M, diagonal doubled (binary counts in weighted_SBM and two-layer modes, multiplicities otherwise)
    std::vector<int> edge_counts_sum; // The margin of the edge count matrix M, e_r (binary stubs in weighted_SBM and two-layer modes)
    std::vector<std::vector<double>> weight_counts; // μ_rs: total real-valued weight between groups (only populated in weighted_SBM mode)
    std::vector<double> weight_counts_sum; // total weight of each group (only in weighted_SBM mode)
    int nonzero_pairs = 0;  // block pairs r<=s joined by at least one edge (weighted_SBM bookkeeping)
    double w_total = 0.0; // total undirected edge weight (constant; only in weighted_SBM mode)
    std::vector<double> group_score_sums; // Sum of the scores in each group

    // Concentration versions of variation parameters
    double concentration_in; // Concentration parameter corresponding to variation_in
    double concentration_out; // Concentration parameter corresponding to variation_out
    double degree_correction_concentration; // Concentration parameter corresponding to degree_correction

    std::vector<int> node_degrees; // Degree of each node in neighbors_com: sum of multiplicities, or the number of neighbours (binary) in weighted_SBM and two-layer modes
    double density_in; // Typical density of in-group edges
    double density_out; // Typical density of out-group edges
    
    // Proposal counts (attempted / accepted)
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
    int individual_depth_proposals = 0; // Total number of individual_depth proposals attempted
    int accepted_individual_depth_proposals = 0; // Total number of individual_depth proposals
    int group_depth_proposals = 0; // Total number of group_depth proposals attempted
    int accepted_group_depth_proposals = 0; // Total number of group_depth proposals accepted
    int new_group_proposals = 0;          // single-node moves to a new group (q -> q+1)
    int accepted_new_group_proposals = 0;
    int vacate_proposals = 0;             // single-node moves of a singleton (q -> q-1)
    int accepted_vacate_proposals = 0;

    // Step sizes: initial values below, adapted during burn-in (adapting = true,
    // set by fit()) and frozen for sampling.  See ADAPT_* in globals.h.
    bool adapting = false;
    double step_score = SCORE_STEP_SIZE;            // CDF space, single-node score
    double step_individual_depth = INDIVIDUAL_DEPTH_STEP_SIZE;   // CDF space
    double step_group_depth = GROUP_DEPTH_STEP_SIZE;             // CDF space
    double step_ties = TIES_PARAMETER_STEP_SIZE;                 // CDF space
    double step_individual_depth_centred = INDIVIDUAL_DEPTH_STEP_SIZE;  // CDF space, centred sigma_I move
    long adapt_t_individual_depth_centred = 0;
    double step_individual_depth_residual = INDIVIDUAL_DEPTH_STEP_SIZE;  // CDF space, residual-rescale sigma_I move
    long adapt_t_individual_depth_residual = 0;
    double step_group_depth_rescale = GROUP_DEPTH_STEP_SIZE;   // CDF space, sigma_G rescale move
    long adapt_t_group_depth_rescale = 0;
    // CDF-space steps of the SBM parameters (densities: exponential prior; variations
    // and degree correction: uniform on [0, 1]), tuned like the others.
    double step_density_in = PARAMETER_PROPOSAL_STEP_SIZE, step_density_out = PARAMETER_PROPOSAL_STEP_SIZE;
    double step_variation_in = VARIATION_IN_STEP_SIZE, step_variation_out = VARIATION_OUT_STEP_SIZE;
    double step_degree_correction = DEGREE_CORRECTION_STEP_SIZE;
    long adapt_t_density_in = 0, adapt_t_density_out = 0, adapt_t_variation_in = 0,
         adapt_t_variation_out = 0, adapt_t_degree_correction = 0;
    double step_group_score = GROUP_SCORE_STEP_SIZE;             // scaled-score units
    long adapt_t_score = 0, adapt_t_individual_depth = 0, adapt_t_group_depth = 0,
         adapt_t_ties = 0, adapt_t_group_score = 0;
    void adapt_step(double& step, long& t, bool accepted, double max_step);

    /* API */
    // Equal-spend weighting of the collective moves: sets
    // collective_weight_override from the CURRENT q.  Call it during burn-in and
    // stop once sampling starts -- q changes as the chain runs, and a
    // state-dependent weight would bias the q-marginal.  No-op when the caller
    // has fixed MERGE_SPLIT_WEIGHT.  See globals.h for the derivation.
    void retune_collective_weight();

    void MCMC_sweep(double beta = 1.0, int num_sweeps = 1, bool merge_split_enabled = true); // Perform MCMC sweeps
    // One Metropolis-Hastings step for a continuous parameter towards new_value,
    // which the caller must have drawn from a CDF-space proposal of that
    // parameter's prior (the prior is not in the acceptance).  Not a setter: the
    // value may be rejected.  For individual_depth the scores are rescaled with
    // it, unless scores_fixed (then the scores are held fixed).
    void change_parameter(const std::string& param_name, double new_value);

    // Functions that include the logic to generate a proposal
    void propose_move(); // Generate and accept/reject a random proposal
    void propose_single_site_move(); // Block-informed single-node move (Peixoto 2020 Eq. 15), may open or vacate a group
    void propose_parameter_move(); // Generate and accept/reject a change of a continuous parameter
    void propose_score_move(); // One node's score: CDF-space walk on its exact Gaussian conditional prior (see MCMC_proposals.cpp)
    // Shift all scores in group r (random if r < 0) by a common offset (moves along
    // the sigma_C ridge).  MCMC_sweep calls it once per group at the end of a sweep.
    void propose_group_score_move(int r = -1);
    void propose_group_depth_rescale(double new_group_depth); // sigma_G move carrying the group means (Jacobian lambda^q)
    void propose_individual_depth_centred(); // sigma_I move with the scores held fixed (no Jacobian)
    void propose_individual_depth_residual(); // sigma_I move scaling within-group deviations, group means fixed (Jacobian lambda^(n-q))
    // Group r's term of H_s_G_b_depths: -[log f_r + f_r^2 sigma_G^2 S_r^2 / sigma_I^4],
    // f_r = (1 + n_r sigma_G^2 / sigma_I^2)^(-1/2), S_r = group_score_sums[r] (calculate_H_s_G_b_depths).
    double score_prior_group_term(int r) const;
    double group_cross_dA(int r, double d_raw) const; // exact change of H_Adir when group r's raw scores shift by d_raw

    // Peixoto (2020, PRE 102:012305) block-informed proposals; neighbours from neighbors_com
    double proposal_block_edges(int t, int g) const; // e_tg (or mu_tg with weighted_proposals)
    double proposal_block_degree(int t) const;       // e_t  (or mu_t  with weighted_proposals)
    double block_move_prob(int node, int g) const; // P_e(g | node, b), Eq. 14
    int sample_block_move(int node);                // draw from P_e(. | node, b)
    double partner_logprob(int r, int s) const;     // log P(s | r, b), Eq. 19
    int sample_partner(int r);                       // draw s ~ P(s | r, b), s != r
    double pair_logprob(int r, int s) const;        // log P of choosing the pair {r, s}: (P(s|r) + P(r|s)) / K

    // Functions that directly modify the Monte Carlo state (including the bookkeeping information like the entropy)
    void propose_merge_move();       // Merge two groups (Peixoto Eq. 17/34)
    void propose_split_move();       // Split one group in two (Eq. 32/33)
    void propose_merge_split_move(); // Joint merge-then-split, q unchanged (Eq. 35/36)

    /* Merge/split internals -- see src/MCMC_merge_split.cpp */
    double merge_split_log_pi() const;
    static double split_label_factor(int K);  //  log(K+1): partition/labelled multiplicity
    static double merge_label_factor(int K);  // -log(K):   partition/labelled multiplicity
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

    // Move `nodes` -- all currently in ONE group, old_group -- to new_group
    // (new_group == num_groups opens a group), updating every entropy term
    // incrementally.  If old_group is emptied it is removed and the last group is
    // relabelled into its slot, so labels stay 0..num_groups-1; a group opened in
    // the same call then ends up labelled old_group.
    void flip_nodes(const std::vector<int>& nodes, int new_group);
    // Exchange the labels of groups group1 and group2 (all bookkeeping follows).
    void swap_groups(int group1, int group2);
    
    // Recompute all bookkeeping from the state; with throw_error, throw on any
    // mismatch with the cached values (value_check tolerance), else just repair.
    void check_state_bookkeeping(bool throw_error = true);

    double calculate_H_k(); // Calculate the entropy of the degrees given the current state (H(k|M,b,alpha))
    double calculate_H_M(); // Calculate the entropy of the edge count matrix given the degrees in the current state (H(M|b,rho_in,rho_out) or H(M|m), etc.)
    double calculate_H_A_G_k_M_b(); // Calculate the entropy of the network given the edge count matrix in the current state (H(A|k,M,b))
    double calculate_H_W(); // Calculate weighted SBM entropy -log P(W|A,b) (only valid in weighted_SBM mode)
    double H_W_pair(int r, int s) const;       // block pair (r, s)'s term of calculate_H_W
    double H_W_totals(int nonzero) const;      // the block-pair-totals term of calculate_H_W
    double calculate_H_Adir_G_A_s_nu(); // Calculate the entropy of the directed edges given the undirected edges and scores (H(Adir|A,s,nu))
    double calculate_H_s_G_b_depths(); // Calculate the entropy of the scores given the partition and depth parameters (H(s|b,depths))
};

#endif // MCMC_CORE_H