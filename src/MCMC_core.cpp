// Implementation stripped; this file now only wires the header to pybind.

#include "MCMC_core.h"
#include "globals.h"
#include "helpers.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>


/**
 * MCMC Constructor
 * 
 * Initializes MCMC sampler with graph structure, partition, and model parameters.
 * Sets up all bookkeeping data structures for efficient entropy calculation updates.
 * 
 * @param wins: Adjacency list for dominant edges (wins[i][j] = weight of dominant edge from i to j)
 * @param ties: Adjacency list for neutral edges (ties[i][j] = weight of neutral edge from i to j)
 * @param partition: Initial group assignment (partition[i] = group of node i)
 * @param num_groups: Initial number of groups
 * @param variation_in/out: Overdispersion for within/between group edges
 * @param degree_correction: Degree heterogeneity correction parameter
 * @param mean_degree_scaling: Power-law scaling exponent for expected edge counts
 * @param *_fixed: Flags for fixed (not inferred) parameters
 * @param seed: Random seed for reproducibility
 */
MCMC::MCMC(const std::vector<std::unordered_map<int, double>>& wins,
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
           int seed,
           double density_in_initial,
           double density_out_initial,
           bool weighted_SBM)
    : n(0),
      m(0),
      wins(wins),  // Copy wins adjacency list from const reference
      losses(),
      ties(ties),  // Copy ties adjacency list from const reference
      neighbors(),
      neighbors_com(),
      two_layer_mode(false),
      weighted_SBM(weighted_SBM),
      free_parameters(),
      beta(0.0),  // Inverse temperature (set during MCMC_sweep)
      proposal_weights(),
      partition(partition),  // Copy initial partition from const reference
      partition_fixed(partition_fixed),
      scores(scores), // Copy initial scores from const reference
      scores_fixed(scores_fixed),
    scaled_scores(), // Scores divided by the individual depth. MCMC moves are proposed according to these for better ergodicity.
    variation_in(variation_in),  // Copy variation parameters (stored as values)
    variation_out(variation_out),
    degree_correction(degree_correction),
    variation_in_fixed(variation_in_fixed),
    variation_out_fixed(variation_out_fixed),
      degree_correction_fixed(degree_correction_fixed),
      individual_depth(individual_depth),
      individual_depth_fixed(individual_depth_fixed),
      group_depth(group_depth),
      group_depth_fixed(group_depth_fixed),
      ties_parameter(ties_parameter),
      ties_parameter_fixed(ties_parameter_fixed),
      num_groups_fixed(num_groups_fixed),
      assortative(assortative),
      H_b(0.0),
      H_M(0.0),
      H_k(0.0),
      H_A_G_k_M_b(0.0),
      H_W(0.0),
      H_A_G_b(0.0),
      H_A_b(0.0),
      w_total(0.0),
      num_groups(num_groups),
      node_counts(),  // Will be resized and computed
      group_members(),  // group_members[r] = vector of node IDs in group r
      node_indices_in_groups(),  // node_indices_in_groups[r][node] = position in group_members[r]
      edge_counts(),  // edge_counts[r][s] = total edge weight from group r to s
      edge_counts_sum(),  // edge_counts_sum[r] = total degree of group r
      group_scaled_score_sums(), // Sum of squares of scores in each group
      concentration_in(0.0),  // Negative binomial concentration parameters
      concentration_out(0.0),
      degree_correction_concentration(0.0),
      node_counts_norm_2(0.0),
      density_in(0.0),
      density_out(0.0),
      time_H_k(0.0),  // Performance profiling timers
      time_H_M(0.0),
      time_H_A_G_k_M_b(0.0),
      time_params(0.0),
      time_single_site_proposals(0.0),
      time_parameter_proposals(0.0),
      total_proposals(0),
      accepted_proposals(0),
      single_site_proposals(0),
      accepted_single_site_proposals(0),
      parameter_proposals(0),
      accepted_parameter_proposals(0),
      degree_correction_proposals(0),
      accepted_degree_correction_proposals(0),
      variation_in_proposals(0),
      accepted_variation_in_proposals(0),
      variation_out_proposals(0),
      accepted_variation_out_proposals(0) {

    
    // Seed random number generator for reproducibility
    rng.seed(seed);

    // === Initialize graph information ===
    n = this->partition.size(); // Number of nodes in the graph

    // Compute losses from wins and populate neighbors from all three
    losses.resize(n);
    neighbors.resize(n);
    
    // Process wins: wins[i][j] means i dominates j
    // This creates a loss for j: losses[j][i]
    for(int i = 0; i < n; i++){
        for(const auto& edge : this->wins[i]){
            int j = edge.first;
            double weight = edge.second;
            neighbors[i][j] += weight; // Add to undirected neighbors
            neighbors[j][i] += weight; // Add to undirected neighbors
            losses[j][i] += weight; // j loses to i
        }
    }
    
    // Process ties: symmetric edges
    for(int i = 0; i < n; i++){
        for(const auto& edge : this->ties[i]){
            int j = edge.first;
            double weight = edge.second;
            neighbors[i][j] += weight; // Add to undirected neighbors (other direction will be present)
            // Ties don't contribute to wins/losses asymmetry
        }
    }

    // Build neighbors_com: uses group_adj if provided (two-layer mode),
    // otherwise copies neighbors (single-layer — identical behaviour to before).
    if (!group_adj.empty()) {
        two_layer_mode = true;
        neighbors_com = group_adj;
    } else {
        two_layer_mode = false;
        neighbors_com = neighbors;
    }

    if(verbose){
        for(int i = 0; i < n; i++){
            // Updating edge counts for neighbor 0 of node 96 with A_ij = 0.000000
            print("Neighbors of node " + std::to_string(i) + ":");
            for(const auto& edge : neighbors[i]){
                int j = edge.first;
                double weight = edge.second;
                print("  Node " + std::to_string(j) + " with weight " + std::to_string(weight));
            }
            print("Wins of node " + std::to_string(i) + ":");
            for(const auto& edge : wins[i]){
                int j = edge.first;
                double weight = edge.second;
                print("  Wins against node " + std::to_string(j) + " with weight " + std::to_string(weight));
            }
            print("Ties of node " + std::to_string(i) + ":");
            for(const auto& edge : ties[i]){
                int j = edge.first;
                double weight = edge.second;
                print("  Ties with node " + std::to_string(j) + " with weight " + std::to_string(weight));
            }
            print("Losses of node " + std::to_string(i) + ":");
            for(const auto& edge : losses[i]){
                int j = edge.first;
                double weight = edge.second;
                print("  Loses to node " + std::to_string(j) + " with weight " + std::to_string(weight));
            }
            if(i == 96){
                print(static_cast<int>(neighbors[i].size()));
                // exit(1);
            }
        }
    }

    /*
    Node 108 with weight 2.000000
    Node 60 with weight 1.000000
    Node 58 with weight 1.000000
    Node 56 with weight 3.000000
    Node 24 with weight 2.000000
    Node 53 with weight 1.000000
    Node 23 with weight 2.000000
    Node 52 with weight 1.000000
    Node 22 with weight 1.000000
    Node 51 with weight 8.000000
    Node 80 with weight 5.000000
    Node 21 with weight 3.000000
    Node 76 with weight 1.000000
    Node 17 with weight 6.000000
    Node 12 with weight 1.000000
    Node 41 with weight 2.000000
    Node 20 with weight 5.000000
    Node 7 with weight 1.000000
    Node 19 with weight 1.000000
    Node 48 with weight 2.000000
    Node 6 with weight 3.000000
    Node 16 with weight 1.000000
    Node 3 with weight 1.000000
    Node 61 with weight 1.000000
    Node 2 with weight 2.000000
    Node 25 with weight 1.000000
    Node 54 with weight 1.000000
    Node 26 with weight 2.000000
    Node 86 with weight 1.000000
    Node 27 with weight 2.000000
    Node 28 with weight 1.000000
    Node 30 with weight 5.000000
    Node 34 with weight 1.000000
    Node 38 with weight 1.000000
    Node 39 with weight 1.000000
    Node 103 with weight 2.000000
    Node 44 with weight 3.000000
    Node 106 with weight 1.000000
    Node 47 with weight 1.000000

    Node 108 with weight 2.000000
  Node 60 with weight 1.000000
  Node 58 with weight 1.000000
  Node 56 with weight 3.000000
  Node 24 with weight 2.000000
  Node 53 with weight 1.000000
  Node 23 with weight 2.000000
  Node 52 with weight 1.000000
  Node 22 with weight 1.000000
  Node 51 with weight 8.000000
  Node 80 with weight 5.000000
  Node 21 with weight 3.000000
  Node 76 with weight 1.000000
  Node 17 with weight 6.000000
  Node 0 with weight 0.000000
    */

    double m_sum = 0.0;
    node_degrees.resize(n);
    for (int i = 0; i < n; ++i) {
        int deg = 0;
        for (const auto& edge : this->neighbors_com[i]) {
            deg += static_cast<int>(std::llround(edge.second));
            m_sum += edge.second;
        }
        node_degrees[i] = deg;
    }
    m = static_cast<int>(std::llround(m_sum / 2.0));

    if (this->weighted_SBM || this->two_layer_mode) {
        // In weighted mode or two-layer mode: edge counts are binary (one per neighbour,
        // ignoring actual weights).  weighted_SBM additionally tracks weight sums.
        w_total = m_sum / 2.0;
        double m_binary = 0.0;
        for (int i = 0; i < n; ++i) {
            node_degrees[i] = static_cast<int>(this->neighbors_com[i].size());
            m_binary += node_degrees[i];
        }
        m = static_cast<int>(std::llround(m_binary / 2.0));
        if (!this->weighted_SBM) {
            w_total = 0.0;  // not used in unweighted two-layer mode
        }
    }

    // Reset execution timers for performance profiling
    time_H_k = 0.0;
    time_H_M = 0.0;
    time_H_A_G_k_M_b = 0.0;
    time_params = 0.0;

    // === Initialize bookkeeping data structures ===
    // Determine actual number of groups from partition (may differ from input)
    num_groups = *std::max_element(this->partition.begin(), this->partition.end()) + 1;
    node_counts.resize(num_groups, 0);  // node_counts[r] = number of nodes in group r
    group_members.resize(num_groups);  // group_members[r] = vector of node IDs in group r
    node_indices_in_groups.resize(num_groups);  // Fast lookup: node -> position in group_members
    edge_counts.resize(num_groups, std::vector<int>(num_groups, 0));  // edge_counts[r][s] = edges r->s
    edge_counts_sum.resize(num_groups);  // edge_counts_sum[r] = total degree of group r


    // Collect names of free (inferred) parameters
    free_parameters.clear();
    if (!variation_in_fixed) {
        free_parameters.push_back("variation_in");
    }
    if (!variation_out_fixed) {
        free_parameters.push_back("variation_out");
    }
    if (!degree_correction_fixed) {
        free_parameters.push_back("degree_correction");
    }
    if (!individual_depth_fixed) {
        free_parameters.push_back("individual_depth");
    }
    if (!group_depth_fixed) {
        free_parameters.push_back("group_depth");
    }
    if (!ties_parameter_fixed) {
        free_parameters.push_back("ties_parameter");
    }
    if (assortative) {
        free_parameters.push_back("density_in");
        free_parameters.push_back("density_out");
    } else {
        free_parameters.push_back("density_in"); // density_out == density_in when non-assortative
    }

    // Sentinel: -1 means "initialize from data"; use explicit values if provided.
    density_in = density_in_initial;
    density_out = (density_out_initial >= 0.0) ? density_out_initial : density_in_initial;

    // Run the initial bookkeeping without throwing errors to set initial values
    check_state_bookkeeping(false);
}

/**
 * MCMC_sweep: Perform one full MCMC sweep
 *
 * A sweep consists of n proposal attempts (one per node on average).
 * Each proposal randomly selects a move type weighted by proposal_weights,
 * then accepts/rejects based on Metropolis-Hastings criterion.
 *
 * @param beta: Inverse temperature (higher = more selective), defaults to 1.0
 * @param num_sweeps: Number of sweeps to run, defaults to 1
 */
void MCMC::MCMC_sweep(double beta, int num_sweeps) {
    auto sweep_start_time = std::chrono::high_resolution_clock::now();

    this->beta = beta;

    // Set proposal weights.
    // NB: a sqrt(n) boost here was tried (to propose depths more often) and removed:
    // it broke synthetic recoverability (steals proposals from free-partition moves)
    // and does NOT help group_depth mixing -- the bottleneck is ridge traversal, not
    // proposal frequency (single-variable depth moves can't climb the sigma_I/sigma_C
    // ridge). The real fix is a joint (sigma_I, sigma_C) block proposal.
    proposal_weights = {
        {"parameter_move", static_cast<double>(free_parameters.size())}
    };
    if (!partition_fixed) {
        proposal_weights["single_site_move"] = static_cast<double>(n);
    }
    if (!scores_fixed) {
        proposal_weights["score_move"] = static_cast<double>(n);
        // Collective group-score-shift moves traverse the sigma_C ridge that
        // per-node and single-depth moves cannot. Weight is state-INDEPENDENT
        // (constant n, not num_groups) so the partition-move fraction does not
        // depend on the current q -- otherwise the q-marginal is biased. The move
        // is a no-op (early return) when num_groups <= 1.
        proposal_weights["group_score_move"] = static_cast<double>(n);
    }

    for (int sweep = 0; sweep < num_sweeps; sweep++) {
        for (int i = 0; i < n; i++) {
            if (verbose) {
                print("MCMC sweep " + std::to_string(sweep + 1) + ", proposal " + std::to_string(i + 1) + " of " + std::to_string(n));
            }
            propose_move();
        }
    }
    check_state_bookkeeping(true);

    auto sweep_end_time = std::chrono::high_resolution_clock::now();
    auto sweep_duration = std::chrono::duration_cast<std::chrono::milliseconds>(sweep_end_time - sweep_start_time).count();
}

/**
 * change_parameter: Update a model parameter and recompute affected entropies
 * 
 * Implements Metropolis-Hastings acceptance:
 * 1. Store old values
 * 2. Update parameter to new_value
 * 3. Recalculate affected entropy components
 * 4. Accept with probability exp(-beta * delta_H)
 * 5. If rejected, restore old values
 * 
 * Different parameters affect different entropies:
 * - degree_correction, mean_degree_scaling -> H_k (degree sequence)
 * - variation_in, variation_out -> H_M (edge count distribution)
 * - density_in, density_out -> H_M (exponential prior cancels in acceptance)
 * 
 * @param param_name: Name of parameter to change
 * @param new_value: Proposed new value
 */
void MCMC::change_parameter(const std::string& param_name, double new_value) {
    if(verbose){
        print("Changing parameter " + param_name + " to " + std::to_string(new_value) + " =================================");
    }
    if(param_name == "degree_correction"){
        degree_correction_proposals++;
        double degree_correction_old = degree_correction;  // Store for potential rollback
        double H_k_old = H_k;  // Store old entropy component
        
        // Apply proposed change
        degree_correction = new_value;
        degree_correction_concentration = variation_to_concentration(degree_correction);
        H_k = calculate_H_k();  // Recalculate affected entropy
        
        // Metropolis-Hastings acceptance
        double delta_H = H_k - H_k_old;
        double acceptance_prob = std::exp(-beta * delta_H);
        if(uniform_random(rng) < acceptance_prob){
            // Accept: update total entropies
            accepted_degree_correction_proposals++;
            H_A_G_b += (H_k - H_k_old);
            H_A_b += (H_k - H_k_old);
        } else {
            // Reject: restore old values
            degree_correction = degree_correction_old;
            degree_correction_concentration = variation_to_concentration(degree_correction);
            H_k = H_k_old;
        }
    } else if (param_name == "variation_in"){
        variation_in_proposals++;
        double variation_in_old = variation_in;
        double H_M_old = H_M;
        
        // Apply proposed change
        variation_in = new_value;
        concentration_in = variation_to_concentration(variation_in);
        H_M = calculate_H_M();  // Recalculate edge count entropy
        
        // Metropolis-Hastings acceptance
        double delta_H = H_M - H_M_old;
        double acceptance_prob = std::exp(-beta * delta_H);
        if(uniform_random(rng) < acceptance_prob){
            accepted_variation_in_proposals++;
            H_A_G_b += (H_M - H_M_old);
            H_A_b += (H_M - H_M_old);
        } else {
            // Reject: restore old values
            variation_in = variation_in_old;
            concentration_in = variation_to_concentration(variation_in);
            H_M = H_M_old;
        }
    } else if (param_name == "variation_out"){
        variation_out_proposals++;
        double variation_out_old = variation_out;
        double H_M_old = H_M;
        
        // Apply proposed change
        variation_out = new_value;
        concentration_out = variation_to_concentration(variation_out);
        H_M = calculate_H_M();  // Recalculate edge count entropy
        
        // Metropolis-Hastings acceptance
        double delta_H = H_M - H_M_old;
        double acceptance_prob = std::exp(-beta * delta_H);
        if(uniform_random(rng) < acceptance_prob){
            accepted_variation_out_proposals++;
            H_A_G_b += (H_M - H_M_old);
            H_A_b += (H_M - H_M_old);
        } else {
            // Reject: restore old values
            variation_out = variation_out_old;
            concentration_out = variation_to_concentration(variation_out);
            H_M = H_M_old;
        }
    } else if (param_name == "density_in"){
        double density_in_old = density_in;
        double density_out_old = density_out;
        double H_M_old = H_M;

        density_in = new_value;
        if(!assortative) density_out = density_in;
        H_M = calculate_H_M();

        double delta_H = H_M - H_M_old;
        if(uniform_random(rng) < std::exp(-beta * delta_H)){
            H_A_G_b += delta_H;
            H_A_b += delta_H;
        } else {
            density_in = density_in_old;
            density_out = density_out_old;
            H_M = H_M_old;
        }
    } else if (param_name == "density_out"){
        double density_out_old = density_out;
        double H_M_old = H_M;

        density_out = new_value;
        H_M = calculate_H_M();

        double delta_H = H_M - H_M_old;
        if(uniform_random(rng) < std::exp(-beta * delta_H)){
            H_A_G_b += delta_H;
            H_A_b += delta_H;
        } else {
            density_out = density_out_old;
            H_M = H_M_old;
        }
    } else if (param_name == "individual_depth"){
        individual_depth_proposals++;
        double individual_depth_old = individual_depth;
        double H_s_G_b_depths_old = H_s_G_b_depths;
        double H_Adir_G_A_s_nu_old = H_Adir_G_A_s_nu;

        // Apply proposed change
        individual_depth = new_value;
        // Keep the scaled_scores fixed, but update scores accordingly
        for(int i = 0; i < n; i++){
            scores[i] = scaled_scores[i] * individual_depth;
        }
        H_s_G_b_depths = calculate_H_s_G_b_depths();  // Recalculate affected entropy
        H_Adir_G_A_s_nu = calculate_H_Adir_G_A_s_nu(); // Recalculate affected entropy

        // Metropolis-Hastings acceptance
        double delta_H = H_s_G_b_depths - H_s_G_b_depths_old + H_Adir_G_A_s_nu - H_Adir_G_A_s_nu_old;
        double acceptance_prob = std::exp(-beta * delta_H);

        if(uniform_random(rng) < acceptance_prob){
            // Accept: update total entropies
            accepted_individual_depth_proposals++;
        } else {
            // Reject: restore old values
            individual_depth = individual_depth_old;
            H_s_G_b_depths = H_s_G_b_depths_old;
            H_Adir_G_A_s_nu = H_Adir_G_A_s_nu_old;
            // Restore scores
            for(int i = 0; i < n; i++){
                scores[i] = scaled_scores[i] * individual_depth;
            }
        }
    } else if (param_name == "group_depth"){
        group_depth_proposals++;
        double group_depth_old = group_depth;
        double H_s_G_b_depths_old = H_s_G_b_depths;

        // Apply proposed change
        group_depth = new_value;
        H_s_G_b_depths = calculate_H_s_G_b_depths();  // Recalculate affected entropy

        // Metropolis-Hastings acceptance
        double delta_H = H_s_G_b_depths - H_s_G_b_depths_old;
        double acceptance_prob = std::exp(-beta * delta_H);
        if(uniform_random(rng) < acceptance_prob){
            // Accept: update total entropies
            accepted_group_depth_proposals++;
        } else {
            // Reject: restore old values
            group_depth = group_depth_old;
            H_s_G_b_depths = H_s_G_b_depths_old;
        }
    } else if (param_name == "ties_parameter"){
        ties_parameter_proposals++;
        double ties_parameter_old = ties_parameter;
        double H_Adir_G_A_s_nu_old = H_Adir_G_A_s_nu;

        // Apply proposed change
        ties_parameter = new_value;
        H_Adir_G_A_s_nu = calculate_H_Adir_G_A_s_nu(); // Recalculate affected entropy

        // Metropolis-Hastings acceptance
        double delta_H = H_Adir_G_A_s_nu - H_Adir_G_A_s_nu_old;
        double acceptance_prob = std::exp(-beta * delta_H);
        if(uniform_random(rng) < acceptance_prob){
            // Accept: update total entropies
            accepted_ties_parameter_proposals++;
        } else {
            // Reject: restore old values
            ties_parameter = ties_parameter_old;
            H_Adir_G_A_s_nu = H_Adir_G_A_s_nu_old;
        }
    } else {
        if(verbose){
            print("Unknown parameter name: " + param_name);
        }
    }
}
