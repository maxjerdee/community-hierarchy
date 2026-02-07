// Implementation stripped; this file now only wires the header to pybind.

#include "MCMC_core.h"
#include "consensus.h"
#include "globals.h"
#include "helpers.h"
#include <algorithm>
#include <iostream>
#include <chrono>

#ifdef DIRECTEDSTRUCTURE_PYBIND
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
#endif

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
MCMC::MCMC(std::vector<std::unordered_map<int, double>> wins,
           std::vector<std::unordered_map<int, double>> ties,
           std::vector<int> partition,
           std::vector<double> scores,
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
           int seed)
    : n(0),
      m(0),
      wins(std::move(wins)),  // Move wins adjacency list (avoid copy)
      losses(),
      ties(std::move(ties)),  // Move ties adjacency list (avoid copy)
      neighbors(),
      free_parameters(),
      beta(0.0),  // Inverse temperature (set during MCMC_sweep)
      merge_split_enabled(true),
      proposal_weights(),
      partition(std::move(partition)),  // Move initial partition (avoid copy)
      scores(std::move(scores)), // Move initial scores (avoid copy)
      scaled_scores(), // Scores divided by the individual depth. MCMC moves are proposed according to these for better ergodicity.
      variation_in(variation_in),  // Copy variation parameters (stored as values)
      variation_out(variation_out),
      degree_correction(degree_correction),
      mean_degree_scaling(mean_degree_scaling),
      weight_in(0.0),  // Computed during initial bookkeeping
      weight_out(0.0),
      variation_in_fixed(variation_in_fixed),
      variation_out_fixed(variation_out_fixed),
      degree_correction_fixed(degree_correction_fixed),
      mean_degree_scaling_fixed(mean_degree_scaling_fixed),
      individual_depth(individual_depth),
      individual_depth_fixed(individual_depth_fixed),
      group_depth(group_depth),
      group_depth_fixed(group_depth_fixed),
      ties_parameter(ties_parameter),
      ties_parameter_fixed(ties_parameter_fixed),
      num_groups_fixed(num_groups_fixed),
      assortative(assortative),
      H_parameters(0.0),  // Entropy components (computed during bookkeeping)
      H_b(0.0),
      H_M(0.0),
      H_k(0.0),
      H_A_G_k_M_b(0.0),
      H_A_G_b(0.0),
      H_A_b(0.0),
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
      node_counts_norm_2(0.0),  // Cached normalizations for efficiency
      node_counts_norm_scaling_plus_1(0.0),
      node_counts_norm_2scaling_plus_2(0.0),
      density_in(0.0),
      density_out(0.0),
      wang_landau_modification_factor(-1),
      log_density_num_groups(),
      max_num_groups_cutoff(-1),
      time_H_k(0.0),  // Performance profiling timers
      time_H_M(0.0),
      time_H_A_G_k_M_b(0.0),
      time_params(0.0),
      time_single_site_proposals(0.0),
      time_merge_proposals(0.0),
      time_split_proposals(0.0),
      time_parameter_proposals(0.0),
      total_proposals(0),
      accepted_proposals(0),
      merge_proposals(0),
      accepted_merge_proposals(0),
      split_proposals(0),
      accepted_split_proposals(0),
      single_site_proposals(0),
      accepted_single_site_proposals(0),
      parameter_proposals(0),
      accepted_parameter_proposals(0),
      degree_correction_proposals(0),
      accepted_degree_correction_proposals(0),
      variation_in_proposals(0),
      accepted_variation_in_proposals(0),
      variation_out_proposals(0),
      accepted_variation_out_proposals(0),
      mean_degree_scaling_proposals(0),
      accepted_mean_degree_scaling_proposals(0) {
    
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
        for(const auto& [j, weight] : this->wins[i]){
            neighbors[i][j] += weight; // Add to undirected neighbors
            neighbors[j][i] += weight; // Add to undirected neighbors
            losses[j][i] += weight; // j loses to i
        }
    }
    
    // Process ties: symmetric edges
    for(int i = 0; i < n; i++){
        for(const auto& [j, weight] : this->ties[i]){
            neighbors[i][j] += weight; // Add to undirected neighbors (other direction will be present)
            // Ties don't contribute to wins/losses asymmetry
        }
    }

    // // Print neighbors for debugging
    // for(int i = 0; i < n; i++){
    //     print("Neighbors of node " + std::to_string(i) + ":");
    //     for(const auto& [j, weight] : neighbors[i]){
    //         print("  Node " + std::to_string(j) + " with weight " + std::to_string(weight));
    //     }
    //     print("Wins of node " + std::to_string(i) + ":");
    //     for(const auto& [j, weight] : wins[i]){
    //         print("  Wins against node " + std::to_string(j) + " with weight " + std::to_string(weight));
    //     }
    // }

    m = 0; // Number of edges (will be computed from adjacency list)
    for (const auto& node_neighbors : this->neighbors) {
        m += node_neighbors.size();  // Sum all edge occurrences
    }
    m /= 2; // Each undirected edge is counted twice in adjacency list

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
    if (!mean_degree_scaling_fixed) {
        free_parameters.push_back("mean_degree_scaling");
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
        free_parameters.push_back("weight_in");
        free_parameters.push_back("weight_out");
    } else {
        free_parameters.push_back("weight_in"); // weight_out determined by weight_in and the non-assortative density_in = density_out constraint
    }

    // Set the density and weight parameters to -1 to indicate the initial bookkeeping has not occurred.
    weight_in = -1.0;
    weight_out = -1.0;
    density_in = -1.0;
    density_out = -1.0;

    // Run the initial bookkeeping without throwing errors to set initial values
    check_state_bookkeeping(false);

    // Initialize the density of state estimate for Wang-Landau
    log_density_num_groups.resize(n + 1, 0.0); // Track from 0 to n groups
}

/**
 * MCMC_sweep: Perform one full MCMC sweep
 * 
 * A sweep consists of n proposal attempts (one per node on average).
 * Each proposal randomly selects a move type weighted by proposal_weights,
 * then accepts/rejects based on Metropolis-Hastings criterion.
 * 
 * @param beta: Inverse temperature (higher = more selective), defaults to 1.0
 * @param merge_split_enabled: Enable/disable merge and split moves, defaults to true
 * @param wang_landau_modification_factor: If > 0, use Wang-Landau with this modification factor, defaults to -1 (disabled)
 * @param max_num_groups_cutoff: Maximum number of groups to track in Wang-Landau, defaults to -1 (no cutoff)
 * @param num_sweeps: Number of sweeps to run, defaults to 1
 * 
 */
void MCMC::MCMC_sweep(double beta, bool merge_split_enabled, double wang_landau_modification_factor, int max_num_groups_cutoff, int num_sweeps) {
    auto sweep_start_time = std::chrono::high_resolution_clock::now();
    
    this->beta = beta;  // Set inverse temperature for this sweep
    this->merge_split_enabled = merge_split_enabled;  // Toggle merge/split proposals
    this->wang_landau_modification_factor = wang_landau_modification_factor; // Set Wang-Landau modification factor
    this->max_num_groups_cutoff = max_num_groups_cutoff; // Set maximum number of groups to track in Wang-Landau

    // Set proposal weights based on enabled moves
    proposal_weights = {
        {"single_site_move", static_cast<double>(n)},
        {"score_move", static_cast<double>(n)},
        {"parameter_move", static_cast<double>(free_parameters.size())}
    };
    if(merge_split_enabled){
        proposal_weights["merge_move"] = 1.0;
        proposal_weights["split_move"] = 1.0;
    };

    // // Testing just the score distribution move
    // proposal_weights = {{"score_move", 1.0}};

    // // Temporarily only test the merge/split moves
    // proposal_weights = {
    //     {"merge_move", 1.0},
    //     {"split_move", 1.0}
    // };
    
    // Perform num_sweeps sweeps; each sweep has n proposal attempts (expected: one per node)
    int total_proposals_before = total_proposals;
    for(int sweep = 0; sweep < num_sweeps; sweep++){
        for(int i = 0; i < n; i++){
            if(verbose){
                print("MCMC sweep " + std::to_string(sweep + 1) + ", proposal " + std::to_string(i + 1) + " of " + std::to_string(n));
            }
            propose_move();  // Generate and accept/reject a random proposal

            // If Wang-Landau is enabled, update log density of states
            if(wang_landau_modification_factor > 0){
                log_density_num_groups[num_groups] += wang_landau_modification_factor;
            }
            // // Print the entropies
            // print("H_A_b: " + std::to_string(H_A_b) + ", H_A_G_b: " + std::to_string(H_A_G_b) + ", H_k: " + std::to_string(H_k) + ", H_M: " + std::to_string(H_M) + ", H_parameters: " + std::to_string(H_parameters) + ", H_b: " + std::to_string(H_b));
        }
    }
    // Final bookkeeping check at end of all sweeps
    check_state_bookkeeping(true);
    
    auto sweep_end_time = std::chrono::high_resolution_clock::now();
    auto sweep_duration = std::chrono::duration_cast<std::chrono::milliseconds>(sweep_end_time - sweep_start_time).count();
    // Print time taken
    // std::cout << "MCMC_sweep: " << sweep_duration << "ms, " << (total_proposals - total_proposals_before) << " proposals, n=" << n << std::endl;
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
 * - weight_in, weight_out -> H_parameters (parameter prior)
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
    } else if (param_name == "mean_degree_scaling"){
        mean_degree_scaling_proposals++;
        double mean_degree_scaling_old = mean_degree_scaling;
        double density_in_old = density_in;
        double density_out_old = density_out;
        double node_counts_norm_scaling_plus_1_old = node_counts_norm_scaling_plus_1;
        double node_counts_norm_2scaling_plus_2_old = node_counts_norm_2scaling_plus_2;
        double H_k_old = H_k;
        double H_M_old = H_M;
        double H_parameters_old = H_parameters;
        
        // Apply proposed change
        mean_degree_scaling = new_value;
        H_k = calculate_H_k();  // Recalculate degree sequence entropy

        if(verbose){
            print("Normalizations and densities before recalculation:");
            print("  node_counts_norm_scaling_plus_1: " + std::to_string(node_counts_norm_scaling_plus_1_old));
            print("  node_counts_norm_2scaling_plus_2: " + std::to_string(node_counts_norm_2scaling_plus_2_old));
            print("  density_in: " + std::to_string(density_in_old));
            print("  density_out: " + std::to_string(density_out_old));
            print("  weight_in: " + std::to_string(weight_in));
            print("  weight_out: " + std::to_string(weight_out));
        }

        // Recalculate cached normalizations
        node_counts_norm_scaling_plus_1 = 0.0;
        node_counts_norm_2scaling_plus_2 = 0.0;
        for(int r = 0; r < num_groups; r++){
            node_counts_norm_scaling_plus_1 += std::pow(node_counts[r], mean_degree_scaling + 1);
            node_counts_norm_2scaling_plus_2 += std::pow(node_counts[r], 2 * (mean_degree_scaling + 1));
        }

        // Recalculate densities
        density_in = weight_in * node_counts_norm_2scaling_plus_2 / node_counts_norm_2;
        if(num_groups > 1){
            density_out = weight_out * 
                         (node_counts_norm_scaling_plus_1 * node_counts_norm_scaling_plus_1 - node_counts_norm_2scaling_plus_2) / (n * n - node_counts_norm_2);
        }

        // Recalculate H(M)
        H_M = calculate_H_M();

        // Update H_parameters
        if(assortative){
            H_parameters = density_in + density_out;
        }else{
            H_parameters = density_in;
        }
        
        // Metropolis-Hastings acceptance
        double delta_H = H_k - H_k_old + H_M - H_M_old + H_parameters - H_parameters_old;

        double acceptance_prob = std::exp(-beta * delta_H);
        if(uniform_random(rng) < acceptance_prob){
            accepted_mean_degree_scaling_proposals++;
            if(verbose){
                print("Accepted change to mean_degree_scaling ============================");
            }
            H_A_G_b += H_k - H_k_old + H_M - H_M_old;
            H_A_b += H_k - H_k_old + H_M - H_M_old;
        } else {
            if(verbose){
                print("Rejected change to mean_degree_scaling ============================");
            }
            // Reject: restore old value
            mean_degree_scaling = mean_degree_scaling_old;
            H_k = H_k_old;
            H_M = H_M_old;
            H_parameters = H_parameters_old;

            // Reset cached normalizations
            node_counts_norm_scaling_plus_1 = node_counts_norm_scaling_plus_1_old;
            node_counts_norm_2scaling_plus_2 = node_counts_norm_2scaling_plus_2_old;

            // Reset densities
            density_in = density_in_old;
            density_out = density_out_old;
        }
    } else if (param_name == "weight_in"){
        double weight_in_old = weight_in;
        double weight_out_old = weight_out;
        double density_in_old = density_in;
        double density_out_old = density_out;
        double H_M_old = H_M;
        double H_parameters_old = H_parameters;
        
        // Apply proposed change
        weight_in = new_value;
        density_in = weight_in * node_counts_norm_2scaling_plus_2 / node_counts_norm_2;
        H_parameters += density_in - density_in_old;  // Update parameter prior entropy
        
        // Non-assortative constraint: density_in = density_out
        if(!assortative){
            density_out = density_in;
            if(num_groups > 1){
                weight_out = density_out * (n * n - node_counts_norm_2) / 
                             (node_counts_norm_scaling_plus_1 * node_counts_norm_scaling_plus_1 - node_counts_norm_2scaling_plus_2);
            }
        }
        
        H_M = calculate_H_M();  // Recalculate edge count entropy
        
        // Metropolis-Hastings acceptance (includes both H_M and H_parameters changes)
        double delta_H = H_M - H_M_old + H_parameters - H_parameters_old;
        double acceptance_prob = std::exp(-beta * delta_H);
        if(uniform_random(rng) < acceptance_prob){
            H_A_G_b += (H_M - H_M_old);
            H_A_b += (H_M - H_M_old);
        } else {
            // Reject: restore all affected values
            H_M = H_M_old;
            H_parameters = H_parameters_old;
            density_in = density_in_old;
            density_out = density_out_old;
            weight_in = weight_in_old;
            weight_out = weight_out_old;
        }
    } else if (param_name == "weight_out"){
        double weight_out_old = weight_out;
        double H_M_old = H_M;
        double density_out_old = density_out;
        double H_parameters_old = H_parameters;
        
        // Apply proposed change
        weight_out = new_value;
        density_out = weight_out * (node_counts_norm_scaling_plus_1 * node_counts_norm_scaling_plus_1 - node_counts_norm_2scaling_plus_2) / (n * n - node_counts_norm_2);
        H_parameters += density_out - density_out_old;  // Update parameter prior entropy
        H_M = calculate_H_M();  // Recalculate edge count entropy
        
        // Metropolis-Hastings acceptance (includes both H_M and H_parameters changes)
        double delta_H = H_M - H_M_old + H_parameters - H_parameters_old;
        double acceptance_prob = std::exp(-beta * delta_H);
        if(uniform_random(rng) < acceptance_prob){
            if(verbose){
                print("Accepted change to weight_out ============================");
            }
            H_A_G_b += (H_M - H_M_old);
            H_A_b += (H_M - H_M_old);
        } else {
            if(verbose){
                print("Rejected change to weight_out ============================");
            }
            // Reject: restore old values
            weight_out = weight_out_old;
            density_out = density_out_old;
            H_M = H_M_old;
            H_parameters = H_parameters_old;
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

#ifdef DIRECTEDSTRUCTURE_PYBIND
PYBIND11_MODULE(MCMC_core, m) {
    py::class_<MCMC>(m, "MCMC")
        .def(py::init<
                 std::vector<std::unordered_map<int, double>>,
                 std::vector<std::unordered_map<int, double>>,
                 std::vector<int>,
                 std::vector<double>,
                 int,
                 const bool&,
                 const bool&,
                 const double&,
                 const bool&,
                 const double&,
                 const bool&,
                 const double&,
                 const bool&,
                 const double&,
                 const bool&,
                 const double&,
                 const bool&,
                 const double&,
                 const bool&,
                 const double&,
                 const bool&,
                 int> (),
             py::arg("wins"),
             py::arg("ties"),
             py::arg("partition"),
             py::arg("scores"),
             py::arg("num_groups"),
             py::arg("num_groups_fixed") = false,
             py::arg("assortative") = true,
             py::arg("variation_in") = 1.0,
             py::arg("variation_in_fixed") = false,
             py::arg("variation_out") = 1.0,
             py::arg("variation_out_fixed") = false,
             py::arg("degree_correction") = 1.0,
             py::arg("degree_correction_fixed") = false,
             py::arg("mean_degree_scaling") = 0.0,
             py::arg("mean_degree_scaling_fixed") = false,
             py::arg("individual_depth") = 1.0,
             py::arg("individual_depth_fixed") = false,
             py::arg("group_depth") = 1.0,
             py::arg("group_depth_fixed") = false,
             py::arg("ties_parameter") = 1.0,
             py::arg("ties_parameter_fixed") = false,
             py::arg("seed") = 0)
           .def_readonly("num_groups", &MCMC::num_groups)
           .def_readonly("partition", &MCMC::partition)
           .def_readonly("scores", &MCMC::scores)
           .def_readonly("density_in", &MCMC::density_in)
           .def_readonly("density_out", &MCMC::density_out)
           .def_readonly("variation_in", &MCMC::variation_in)
           .def_readonly("variation_out", &MCMC::variation_out)
           .def_readonly("degree_correction", &MCMC::degree_correction)
           .def_readonly("mean_degree_scaling", &MCMC::mean_degree_scaling)
           .def_readonly("individual_depth", &MCMC::individual_depth)
           .def_readonly("group_depth", &MCMC::group_depth)
           .def_readonly("ties_parameter", &MCMC::ties_parameter)
           .def_readonly("H_A_b", &MCMC::H_A_b)
           .def_readonly("H_b", &MCMC::H_b)
           .def_readonly("H_M", &MCMC::H_M)
           .def_readonly("H_k", &MCMC::H_k)
           .def_readonly("H_A_G_k_M_b", &MCMC::H_A_G_k_M_b)
           .def_readonly("H_A_G_b", &MCMC::H_A_G_b)
           .def_readonly("H_Adir_G_A_s_nu", &MCMC::H_Adir_G_A_s_nu)
           .def_readonly("H_s_G_b_depths", &MCMC::H_s_G_b_depths)
           .def_readonly("time_H_k", &MCMC::time_H_k)
           .def_readonly("time_H_M", &MCMC::time_H_M)
           .def_readonly("time_H_A_G_k_M_b", &MCMC::time_H_A_G_k_M_b)
           .def_readonly("time_single_site_proposals", &MCMC::time_single_site_proposals)
           .def_readonly("time_merge_proposals", &MCMC::time_merge_proposals)
           .def_readonly("time_split_proposals", &MCMC::time_split_proposals)
           .def_readonly("time_parameter_proposals", &MCMC::time_parameter_proposals)
           .def_readonly("wang_landau_modification_factor", &MCMC::wang_landau_modification_factor)
           .def_readonly("max_num_groups_cutoff", &MCMC::max_num_groups_cutoff)
           .def_readonly("log_density_num_groups", &MCMC::log_density_num_groups)
           .def("MCMC_sweep", &MCMC::MCMC_sweep, py::arg("beta") = 1.0, py::arg("merge_split_enabled") = true, py::arg("wang_landau_modification_factor") = -1.0, py::arg("max_num_groups_cutoff") = -1, py::arg("num_sweeps") = 1)
           .def("change_parameter", &MCMC::change_parameter, py::arg("param_name"), py::arg("new_value"));

    // Bind ConsensusClustering class
    py::class_<ConsensusClustering>(m, "ConsensusClustering")
        .def_static("consensus_clustering", &ConsensusClustering::consensus_clustering,
                    py::arg("clusterings"),
                    py::arg("metric") = "L1",
                    "Compute consensus clustering from a set of clusterings.\n\n"
                    "Finds the clustering whose coincidence matrix is closest to the average\n"
                    "coincidence matrix under the specified metric.\n\n"
                    "Parameters:\n"
                    "  clusterings: List of clusterings (each is a list of node assignments)\n"
                    "  metric: 'L1' or 'L2' distance metric (default: 'L1')\n\n"
                    "Returns:\n"
                    "  The consensus clustering as a list of node group assignments");

}
#endif
