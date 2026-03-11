#include "MCMC_core.h"
#include "globals.h"
#include "helpers.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <chrono>

/**
 * propose_move: Select and execute a random MCMC proposal
 * 
 * Randomly selects a proposal type (single-site, merge, split, or parameter)
 * weighted by proposal_weights, then dispatches to the appropriate handler.
 * 
 * Proposal types:
 * - single_site_move: Move one node to a different group (weight: n)
 * - merge_move: Merge two groups into one (weight: 1)
 * - split_move: Split one group into two (weight: 1)
 * - parameter_move: Update a model parameter (weight: # free parameters)
 */
void MCMC::propose_move() {
    start_time = std::chrono::high_resolution_clock::now();
    
    double p = uniform_random(rng);
    double cumulative_weight = 0.0;
    double total_weight = 0.0;
    
    // Calculate total weight
    for (const auto& [proposal_type, weight] : proposal_weights) {
        total_weight += weight;
    }
    
    // Select proposal type by weighted sampling
    for (const auto& [proposal_type, weight] : proposal_weights) {
        cumulative_weight += weight;
        if (p < cumulative_weight / total_weight) {
            if (proposal_type == "single_site_move") {
                propose_single_site_move();
            } else if (proposal_type == "merge_move") {
                propose_merge_move();
            } else if (proposal_type == "split_move") {
                propose_split_move();
            } else if (proposal_type == "parameter_move") {
                propose_parameter_move();
            } else if (proposal_type == "score_move") {
                propose_score_move();
            }
            
            // Track timing for this proposal
            end_time = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration<double>(end_time - start_time).count();
            
            if (proposal_type == "single_site_move") {
                time_single_site_proposals += duration;
            } else if (proposal_type == "merge_move") {
                time_merge_proposals += duration;
            } else if (proposal_type == "split_move") {
                time_split_proposals += duration;
            } else if (proposal_type == "parameter_move") {
                time_parameter_proposals += duration;
            }
            break;
        }
    }
}

/**
 * propose_single_site_move: Move a single node to a different group
 * 
 * Two types of moves:
 * - Type II (probability q/(n+1)): Move node to NEW group (increases num_groups)
 * - Type I (probability n/(n+1)): Move node to EXISTING group
 * 
 * Accept/reject based on Metropolis-Hastings criterion using entropy change.
 * If rejected, flip the node back to its original group.
 */
void MCMC::propose_single_site_move() {
    single_site_proposals++;
    if (verbose) {
        print("Proposing single site move =================================");
    }
    double p = uniform_random(rng);
    int old_group;
    int new_group;
    int node;
    
    // Type II move: Create new group with probability q/(n+1)
    if (p < static_cast<double>(num_groups) / (n + 1)) {
        if(verbose){
            print("Proposing type II move to new group ==================================");
        }
        // Simple implementation (breaks permutation symmetry)
        // Select random group and node from it
        old_group = std::uniform_int_distribution<int>(0, num_groups - 1)(rng);
        if (node_counts[old_group] > 1) {
            node = group_members[old_group][std::uniform_int_distribution<int>(0, node_counts[old_group] - 1)(rng)];
        } else {
            return;  // Can't move if it would empty the only node in the group
        }
        new_group = num_groups;  // New group index

        // Original implementation (preserves permutation symmetry)
       // Come up with a version of this that also preserves the s
    } else {
        // Type I move: Move to existing group
        if(num_groups <= 1){  // Need at least 2 groups to move between
            return;
        }
        old_group = std::uniform_int_distribution<int>(0, num_groups - 1)(rng);
        // Select different group (avoid selecting same group)
        new_group = std::uniform_int_distribution<int>(0, num_groups - 2)(rng);
        if (new_group >= old_group) {
            new_group += 1;  // Skip old_group in the selection
        }
        node = group_members[old_group][std::uniform_int_distribution<int>(0, node_counts[old_group] - 1)(rng)];
    }
    
    // Store old entropy values for acceptance calculation
    double H_A_G_b_old = H_A_G_b;
    double H_parameters_old = H_parameters;
    double H_s_G_b_depths_old = H_s_G_b_depths;
    int num_groups_old = num_groups;
    
    // Execute the proposed move
    flip_nodes({node}, new_group);

    // // Check that the bookkeeping is correct after the move (before acceptance check)
    // check_state_bookkeeping(true); // throw_error

    // If Wang-Landau is enabled and num_groups has changed, adjust acceptance probability (inverse of density of states)
    double wang_landau_adjustment = 0.0;
    if(wang_landau_modification_factor > 0 && num_groups != num_groups_old){
        wang_landau_adjustment = log_density_num_groups[num_groups] - log_density_num_groups[num_groups_old];
    }
    // Also if the proposed number of groups exceeds the cutoff, reject the move if it increases num_groups
    if(max_num_groups_cutoff > 0 && num_groups > max_num_groups_cutoff && num_groups > num_groups_old){
        wang_landau_adjustment = INFINITY;
    }

    // Metropolis-Hastings acceptance
    double delta_H = H_A_G_b - H_A_G_b_old + H_parameters - H_parameters_old + H_s_G_b_depths - H_s_G_b_depths_old;
    double acceptance_prob = std::exp(-beta * delta_H - wang_landau_adjustment);
    if (uniform_random(rng) >= acceptance_prob) {
        // print("Rejected single site move: node " + std::to_string(node) + " from group " + std::to_string(old_group) + " to group " + std::to_string(new_group));
        // if(node == 0){  // Debugging: track moves of node 0
        //     print("Rejected single site move for node 0: from group " + std::to_string(old_group) + " to group " + std::to_string(new_group) + " with delta_H " + std::to_string(delta_H));
        //     // Print the old and new values of the entropies
        //     print("H_A_G_b_old: " + std::to_string(H_A_G_b_old) + ", H_A_G_b: " + std::to_string(H_A_G_b));
        //     print("H_parameters_old: " + std::to_string(H_parameters_old) + ", H_parameters: " + std::to_string(H_parameters));
        //     print("H_s_G_b_depths_old: " + std::to_string(H_s_G_b_depths_old) + ", H_s_G_b_depths: " + std::to_string(H_s_G_b_depths));
        //     print("num_groups_old: " + std::to_string(num_groups_old) + ", num_groups: " + std::to_string(num_groups));
        // }
        // Reject: flip node back to original group
        flip_nodes({node}, old_group);
    } else {
        // if(node == 0){  // Debugging: track moves of node 0
        //     print("Accepted single site move for node 0: from group " + std::to_string(old_group) + " to group " + std::to_string(new_group) + " with delta_H " + std::to_string(delta_H));
        // }
        // print("Accepted single site move: node " + std::to_string(node) + " from group " + std::to_string(old_group) + " to group " + std::to_string(new_group));
        accepted_single_site_proposals++;
    }
}

/**
 * propose_parameter_move: Propose updating a model parameter
 * 
 * Randomly selects one free (non-fixed) parameter and proposes a new value
 * using a uniform or positive-uniform proposal distribution.
 * 
 * The change_parameter function handles the actual update and acceptance logic.
 * If the proposal is rejected inside change_parameter, we call it again with
 * the old value to restore the state.
 * 
 * Parameters can include: variation_in, variation_out, degree_correction,
 * mean_degree_scaling, weight_in, weight_out
 */
void MCMC::propose_parameter_move() {
    parameter_proposals++;
    if (verbose) {
        print("Proposing parameter move");
    }
    if (free_parameters.empty()) {
        return;  // No free parameters to update
    }
    
    // Randomly select a free parameter
    std::uniform_int_distribution<std::size_t> dist(0, free_parameters.size() - 1);
    const auto& parameter = free_parameters[dist(rng)];

    double old_value = 0.0;
    double new_value = 0.0;
    
    // Generate proposal based on parameter type
    if (parameter == "variation_in") {
        old_value = variation_in;
        new_value = uniform_proposal(variation_in, VARIATION_IN_STEP_SIZE);
    } else if (parameter == "variation_out") {
        old_value = variation_out;
        new_value = uniform_proposal(variation_out, VARIATION_OUT_STEP_SIZE);
    } else if (parameter == "degree_correction") {
        old_value = degree_correction;
        new_value = uniform_proposal(degree_correction, DEGREE_CORRECTION_STEP_SIZE);
    } else if (parameter == "mean_degree_scaling") {
        old_value = mean_degree_scaling;
        new_value = -uniform_proposal(-mean_degree_scaling, MEAN_DEGREE_SCALING_STEP_SIZE);  // Flip sign for proposal
    } else if (parameter == "weight_in") {
        old_value = weight_in;
        new_value = positive_uniform_proposal(weight_in, WEIGHT_IN_STEP_SIZE);  // Must remain positive
    } else if (parameter == "weight_out") {
        old_value = weight_out;
        new_value = positive_uniform_proposal(weight_out, WEIGHT_OUT_STEP_SIZE);  // Must remain positive
    } else if (parameter == "individual_depth") {
        old_value = individual_depth;
        new_value = half_cauchy_proposal(individual_depth, DEPTH_PRIOR_WIDTH, INDIVIDUAL_DEPTH_STEP_SIZE);
    } else if (parameter == "group_depth") {
        old_value = group_depth;
        new_value = half_cauchy_proposal(group_depth, DEPTH_PRIOR_WIDTH, GROUP_DEPTH_STEP_SIZE);
    } else if (parameter == "ties_parameter") {
        old_value = ties_parameter;
        new_value = half_cauchy_proposal(ties_parameter, DEPTH_PRIOR_WIDTH, TIES_PARAMETER_STEP_SIZE);  // Must remain positive    
    } else {
        return;  // Unknown parameter
    }

    // Attempt parameter change (includes acceptance logic)
    change_parameter(parameter, new_value);
}

void MCMC::propose_score_move() {
    score_proposals++;
    if (verbose) {
        print("Proposing score move");
    }
    // Select a random node
    int node = std::uniform_int_distribution<int>(0, n - 1)(rng);
    double old_scaled_score = scaled_scores[node];
    // Propose new score according to the Gaussian distribution
    // Mean and standard deviation of the s_i dependence of P(s|b,depths):
    double f_normalization = 1/std::sqrt(1 + node_counts[partition[node]]*group_depth*group_depth/(individual_depth*individual_depth));
    double f_combination = f_normalization*f_normalization*group_depth*group_depth/(individual_depth*individual_depth);
    double proposal_mean = (group_scaled_score_sums[partition[node]] - old_scaled_score)*f_combination/(1 - f_combination);
    double proposal_variance = 0.5/(1 - f_combination);

    if(verbose){
        print("Node " + std::to_string(node) + " in group " + std::to_string(partition[node]) + " with old scaled score " + std::to_string(old_scaled_score));
        print("Proposal mean: " + std::to_string(proposal_mean));
        print("Proposal variance: " + std::to_string(proposal_variance));
    }

    double new_scaled_score = gaussian_proposal(old_scaled_score, proposal_mean, std::sqrt(proposal_variance), PARAMETER_PROPOSAL_STEP_SIZE);
    // // Add a Gaussian random variable with the specified variance to the current scaled score
    // double new_scaled_score = scaled_scores[node] + std::normal_distribution<double>(0, 0.001)(rng);

    if(verbose){
        print("Proposed new scaled score: " + std::to_string(new_scaled_score));
    }

    // Store old entropy values for acceptance calculation
    double H_s_G_b_depths_old = H_s_G_b_depths;
    double H_Adir_G_A_s_nu_old = H_Adir_G_A_s_nu;
    
    // H_s_G_b_depths, subtract terms
    H_s_G_b_depths -= scaled_scores[node]*scaled_scores[node];
    f_normalization = 1/std::sqrt(1 + node_counts[partition[node]]*group_depth*group_depth/(individual_depth*individual_depth));
    H_s_G_b_depths += std::log(f_normalization) + f_normalization*f_normalization*group_depth*group_depth*group_scaled_score_sums[partition[node]]*group_scaled_score_sums[partition[node]]/(individual_depth*individual_depth);
    
    // H_Adir_G_A_s_nu, subtract terms
    if(verbose){
        print("Old H_Adir_G_A_s_nu: " + std::to_string(H_Adir_G_A_s_nu));
    }
    for(const auto& [j, Adir_ij] : wins[node]){
        double log_p_win_ij = scores[node] - log_sum_exp(std::vector<double>{scores[node], scores[j], std::log(ties_parameter) + scores[node]/2.0 + scores[j]/2.0}); // P(node -> j|node - j)
        H_Adir_G_A_s_nu -= 0.0 - Adir_ij * log_p_win_ij;
    }
    for(const auto& [j, Adir_ji] : losses[node]){
        double log_p_win_ji = scores[j] - log_sum_exp(std::vector<double>{scores[j], scores[node], std::log(ties_parameter) + scores[j]/2.0 + scores[node]/2.0}); // P(j -> node|j - node)
        H_Adir_G_A_s_nu -= 0.0 - Adir_ji * log_p_win_ji;
    }
    for(const auto& [j, Aties_ij] : ties[node]){
        double log_p_tie_ij = std::log(ties_parameter) + scores[node]/2.0 + scores[j]/2.0 - log_sum_exp(std::vector<double>{scores[node], scores[j], std::log(ties_parameter) + scores[node]/2.0 + scores[j]/2.0}); // P(i <-> j|i - j)
        H_Adir_G_A_s_nu -= 0.0 - Aties_ij * log_p_tie_ij;
    }
    if(verbose){
        print("Updated H_Adir_G_A_s_nu after subtracting old terms: " + std::to_string(H_Adir_G_A_s_nu));
    }

    // Apply proposed change
    scores[node] = new_scaled_score * individual_depth;
    scaled_scores[node] = new_scaled_score;
    // Adjust the group score sums
    group_scaled_score_sums[partition[node]] += (new_scaled_score - old_scaled_score);

    // H_s_G_b_depths, add terms
    H_s_G_b_depths += scaled_scores[node]*scaled_scores[node];
    f_normalization = 1/std::sqrt(1 + node_counts[partition[node]]*group_depth*group_depth/(individual_depth*individual_depth));
    H_s_G_b_depths -= std::log(f_normalization) + f_normalization*f_normalization*group_depth*group_depth*group_scaled_score_sums[partition[node]]*group_scaled_score_sums[partition[node]]/(individual_depth*individual_depth);
    
    // H_Adir_G_A_s_nu, add terms
    for(const auto& [j, Adir_ij] : wins[node]){
        double log_p_win_ij = scores[node] - log_sum_exp(std::vector<double>{scores[node], scores[j], std::log(ties_parameter) + scores[node]/2.0 + scores[j]/2.0}); // P(node -> j|node - j)
        H_Adir_G_A_s_nu += 0.0 - Adir_ij * log_p_win_ij;
    }
    for(const auto& [j, Adir_ji] : losses[node]){
        double log_p_win_ji = scores[j] - log_sum_exp(std::vector<double>{scores[j], scores[node], std::log(ties_parameter) + scores[j]/2.0 + scores[node]/2.0}); // P(j -> node|j - node)
        H_Adir_G_A_s_nu += 0.0 - Adir_ji * log_p_win_ji;
    }
    for(const auto& [j, Aties_ij] : ties[node]){
        double log_p_tie_ij = std::log(ties_parameter) + scores[node]/2.0 + scores[j]/2.0 - log_sum_exp(std::vector<double>{scores[node], scores[j], std::log(ties_parameter) + scores[node]/2.0 + scores[j]/2.0}); // P(i <-> j|i - j)
        H_Adir_G_A_s_nu += 0.0 - Aties_ij * log_p_tie_ij;
    }
    if(verbose){
        print("Updated H_Adir_G_A_s_nu after adding new terms: " + std::to_string(H_Adir_G_A_s_nu));
    }

    // Metropolis-Hastings acceptance
    double delta_H = H_Adir_G_A_s_nu - H_Adir_G_A_s_nu_old; // Only accept/reject on the change in the Adir term, the change to H_s_G_b_depths is built into the proposal distribution.
    // double delta_H = H_s_G_b_depths - H_s_G_b_depths_old + H_Adir_G_A_s_nu - H_Adir_G_A_s_nu_old;
    double acceptance_prob = std::exp(-beta * delta_H);
    if (uniform_random(rng) < acceptance_prob) {
        // Accept
        accepted_score_proposals++;
    } else {
        // Reject: restore old score
        scaled_scores[node] = old_scaled_score;
        scores[node] = old_scaled_score * individual_depth;
        group_scaled_score_sums[partition[node]] -= (new_scaled_score - old_scaled_score);

        H_s_G_b_depths = H_s_G_b_depths_old;
        H_Adir_G_A_s_nu = H_Adir_G_A_s_nu_old;
    }
}

/**
 * flip_nodes: Move a set of nodes from their current group to a new group
 * 
 * This is the core function for partition updates. It efficiently updates all
 * bookkeeping and entropy components by:
 * 1. Identifying impacted groups (before and after the move)
 * 2. Subtracting entropy contributions from old configuration
 * 3. Updating partition, group_members, node_indices, edge_counts
 * 4. Adding entropy contributions from new configuration
 * 
 * Key features:
 * - Handles group creation (when new_group == num_groups)
 * - Handles group deletion (when old group becomes empty)
 * - Uses swap-and-pop for O(1) node removal from groups
 * - Incrementally updates entropies (only recomputes affected terms)
 * 
 * 
 * @param nodes: Vector of node IDs to move
 * @param new_group: Target group index (may be == num_groups for new group)
 */
void MCMC::flip_nodes(const std::vector<int>& nodes, int new_group) {
    int old_group = partition[nodes[0]];
    if (verbose) {
        std::string node_list;
        for (int node : nodes) {
            node_list += std::to_string(node) + " ";
        }
        print("Flipping nodes " + node_list + " from group " + std::to_string(old_group) + " to group " + std::to_string(new_group) + " =================================");
        print("Current partition:");
        for (int i = 0; i < partition.size(); i++) {
            printf("%d ", partition[i]);
        }
        print("");
        print("Current M: ");
        print(edge_counts);
        print("Current H(M): " + std::to_string(H_M));
        double H_M_check = calculate_H_M();
        print("Computed H(M): " + std::to_string(H_M_check));
        print("density_in: " + std::to_string(density_in));
        print("density_out: " + std::to_string(density_out));
        print("weight_in: " + std::to_string(weight_in));
        print("weight_out: " + std::to_string(weight_out));
    }

    // Remove nodes that are already in the target group
    std::vector<int> filtered_nodes;
    for(int node: nodes){
        if(partition[node] != new_group){
            filtered_nodes.push_back(node);
        }
    }

    if(filtered_nodes.empty()){
        if(verbose){
            print("No nodes to flip after filtering.");
        }
        return;  // No nodes to move
    }
    
    // Check if this move will empty the old group
    bool empty_old_group = (node_counts[old_group] == static_cast<int>(filtered_nodes.size()));

    // === Identify impacted groups ===
    // Initial: groups affected in the current partition
    // Final: groups affected in the proposed partition
    std::unordered_set<int> initial_impacted_groups;
    std::unordered_set<int> final_impacted_groups;
    
    initial_impacted_groups.insert(old_group);
    if(new_group != num_groups){  // If not creating a new group
        initial_impacted_groups.insert(new_group);
    }
    final_impacted_groups.insert(old_group);
    final_impacted_groups.insert(new_group);
    
    // If emptying old_group, the last group will be moved to fill the gap
    if(new_group != num_groups && empty_old_group){
        if(verbose){
            print("Merging groups " + std::to_string(old_group) + " and " + std::to_string(new_group));
        } 
        initial_impacted_groups.insert(num_groups - 1);  // Last group moves
        if(old_group == num_groups - 1){
            final_impacted_groups.erase(old_group);
        }
        if(new_group == num_groups - 1){
            final_impacted_groups.erase(new_group);
        }
    }
    if(verbose){
        print("Initial impacted groups:");
        for(int r: initial_impacted_groups){
            print(std::to_string(r));
        }
        print("Final impacted groups:");
        for(int r: final_impacted_groups){
            print(std::to_string(r));
        }
    }

    // Store old entropy values
    double H_b_old = H_b;
    double H_k_old = H_k;
    double H_M_old = H_M;
    double H_A_G_k_M_b_old = H_A_G_k_M_b;
    double H_s_G_b_depths_old = H_s_G_b_depths;

    if(verbose){
        // Print the entropies
        print("Computed Entropies:");
        print("H_b: " + std::to_string(H_b));
        print("H_k: " + std::to_string(H_k));
        print("H_M: " + std::to_string(H_M));
        print("H_A_G_k_M_b: " + std::to_string(H_A_G_k_M_b));
        print("H_A_G_b: " + std::to_string(H_A_G_b));
    }

    if(verbose){
        // Check if neighbors[96] has an entry for 0, if so print it
        print("Neighbors of node 96:");
        for(const auto& [j, weight] : neighbors[96]){
            print("  Node " + std::to_string(j) + " with weight " + std::to_string(weight));
        }
    }

    /* === PHASE 1: Remove entropy terms from initial configuration === */
    
    // H(nb|q): Partition count prior
    H_b -= log_binom(n - 1, num_groups - 1);
    
    // H(b|nb): Group size distribution
    for(int r: initial_impacted_groups){
        H_b += log_factorial(node_counts[r]);  // Will be subtracted back after update
    }
    
    // H(k): Degree sequence entropy
    if(timing){
        start_time = std::chrono::high_resolution_clock::now();
    }
    for(int r: initial_impacted_groups){
        if(degree_correction_concentration < MAX_CONCENTRATION){
            H_k -= log_binom(edge_counts_sum[r] + node_counts[r]*degree_correction_concentration - 1, 
                        node_counts[r]*degree_correction_concentration - 1);
        }else{
            // Multinomial approximation for large concentration
            H_k += log_factorial(edge_counts_sum[r]);
            H_k -= edge_counts_sum[r]*std::log(node_counts[r]);
        }
    }
    if(timing){
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration<double>(end_time - start_time).count();
        time_H_k += duration;
    }

    // H(M): Edge count distribution entropy
    if(verbose){
        print("Initial H(M): " + std::to_string(H_M) + " before removing terms");
    }
    if(timing){
        start_time = std::chrono::high_resolution_clock::now();
    }
    for(int r: initial_impacted_groups){
        for(int s = 0; s < num_groups; s++){
            if(r == s){
                // Within-group edges (undirected, so divide by 2)
                H_M += log_neg_binom(edge_counts[r][r]/2.0, 
                                    weight_in*std::pow(node_counts[r]*node_counts[r], mean_degree_scaling + 1)/2.0, 
                                    concentration_in);
            }else{
                // Between-group edges
                // Use symmetry_factor to avoid double-counting when both r and s are impacted
                double symmetry_factor = 1;
                if(initial_impacted_groups.find(s) != initial_impacted_groups.end()){
                    symmetry_factor = 0.5;
                }
                H_M += symmetry_factor*log_neg_binom(edge_counts[r][s], 
                                                     weight_out*std::pow(node_counts[r]*node_counts[s], mean_degree_scaling + 1), 
                                                     concentration_out);
            }
        }
    }
    if(verbose){
        print("H(M) after removing initial terms: " + std::to_string(H_M));
    }
    if(timing){
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration<double>(end_time - start_time).count();
        time_H_M += duration;
    }
    
    // H(A|k,M,b): Adjacency matrix entropy given edge counts
    if(timing){
        start_time = std::chrono::high_resolution_clock::now();
    }
    if(verbose){
        print("Initial H(A|k,M,b): " + std::to_string(H_A_G_k_M_b) + " before removing terms");
    }
    for(int r: initial_impacted_groups){
        for(int s = 0; s < num_groups; s++){
            if(r == s){
                H_A_G_k_M_b += log_factorial2(edge_counts[r][r]);
            }else{
                // If s is also in initial_impacted_groups, only add half the term to avoid double counting
                double symmetry_factor = 1;
                if(initial_impacted_groups.find(s) != initial_impacted_groups.end()){
                    symmetry_factor = 0.5;
                }
                H_A_G_k_M_b += symmetry_factor*log_factorial(edge_counts[r][s]);
            }
        }
        H_A_G_k_M_b -= log_factorial(edge_counts_sum[r]);
    }
    if(verbose){
        print("H(A|k,M,b) after removing initial terms: " + std::to_string(H_A_G_k_M_b));
    }
    if(timing){
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration<double>(end_time - start_time).count();
        time_H_A_G_k_M_b += duration;
    }

    // H(s|b,depths): hierarchy-community entropy
    for(int r: initial_impacted_groups){
        double f_normalization = 1/std::sqrt(1 + node_counts[r]*group_depth*group_depth/(individual_depth*individual_depth));
        H_s_G_b_depths += std::log(f_normalization) + f_normalization*f_normalization*group_depth*group_depth*group_scaled_score_sums[r]*group_scaled_score_sums[r]/(individual_depth*individual_depth);
    }

    // Subtract cached normalization terms (will be recalculated after update)
    for(int r: initial_impacted_groups){
        node_counts_norm_2 -= std::pow(node_counts[r],2);
        node_counts_norm_scaling_plus_1 -= std::pow(node_counts[r],mean_degree_scaling + 1);
        node_counts_norm_2scaling_plus_2 -= std::pow(node_counts[r],2*mean_degree_scaling + 2);
    }

    if(verbose){
        print("Entropies after removing initial configuration terms:");
        print("H_b: " + std::to_string(H_b));
        print("H_k: " + std::to_string(H_k));
        print("H_M: " + std::to_string(H_M));
        print("H_A_G_k_M_b: " + std::to_string(H_A_G_k_M_b));
        print("H_s_G_b_depths: " + std::to_string(H_s_G_b_depths));
    }

    /* === PHASE 2: Update partition and bookkeeping === */
    for(int node: filtered_nodes){
        partition[node] = new_group;
        
        // Remove node from old_group using swap-and-pop (O(1) removal)
        size_t node_index = node_indices_in_groups[old_group][node];
        int last_node = group_members[old_group].back();
        group_members[old_group][node_index] = last_node;  // Move last node to removed position
        node_indices_in_groups[old_group][last_node] = node_index;  // Update moved node's index
        group_members[old_group].pop_back();  // Remove duplicate at end
        node_indices_in_groups[old_group].erase(node);  // Remove node from index map
        node_counts[old_group]--;

        if(verbose){
            // Check if neighbors[96] has an entry for 0, if so print it
            print("Neighbors of node 96:");
            for(const auto& [j, weight] : neighbors[96]){
                print("  Node " + std::to_string(j) + " with weight " + std::to_string(weight));
            }
        }

        // Update edge counts: subtract edges from old_group
        for(const auto& [j, A_ij] : neighbors[node]){
            if(verbose){
                print("Updating edge counts for neighbor " + std::to_string(j) + " of node " + std::to_string(node) + " with A_ij = " + std::to_string(A_ij));
                print(static_cast<int>(neighbors[node].size()));
            }
            int neighbor_group = partition[j];
            edge_counts[old_group][neighbor_group] -= A_ij;
            edge_counts[neighbor_group][old_group] -= A_ij;
            edge_counts_sum[old_group] -= A_ij;
            edge_counts_sum[neighbor_group] -= A_ij;
        }

        // Update group_scaled_score_sums: subtract scaled score from old_group
        group_scaled_score_sums[old_group] -= scaled_scores[node];

        // If creating a new group, resize all data structures
        if(new_group == num_groups){
            group_members.push_back(std::vector<int>());
            node_indices_in_groups.push_back(std::unordered_map<int, size_t>());
            node_counts.push_back(0);
            edge_counts_sum.push_back(0);
            edge_counts.push_back(std::vector<int>(num_groups, 0));  // Add new row
            for(auto& row: edge_counts){
                row.push_back(0);  // Add new column to all rows
            }
            group_scaled_score_sums.push_back(0.0);
            num_groups += 1;
        }

        // Add node to new_group
        group_members[new_group].push_back(node);
        node_indices_in_groups[new_group][node] = group_members[new_group].size() - 1;
        node_counts[new_group]++;
        
        // Update edge counts: add edges to new_group
        for(const auto& [j, A_ij] : neighbors[node]){
            int neighbor_group = partition[j];
            edge_counts[new_group][neighbor_group] += A_ij;
            edge_counts[neighbor_group][new_group] += A_ij;
            edge_counts_sum[new_group] += A_ij;
            edge_counts_sum[neighbor_group] += A_ij;
        }

        // Update group_scaled_score_sums: add scaled score to new_group
        group_scaled_score_sums[new_group] += scaled_scores[node];
    }
    
    // If old_group is now empty, compact by moving last group to fill the gap
    if(empty_old_group){
        int last_group = num_groups - 1;
        if(old_group != last_group){
            swap_groups(old_group, last_group);
        }
        
        // Remove last group's data structures
        group_members.pop_back();
        node_counts.pop_back();
        edge_counts_sum.pop_back();
        edge_counts.pop_back();  // Remove last row
        for(auto& row: edge_counts){
            row.pop_back();  // Remove last column from all rows
        }
        group_scaled_score_sums.pop_back();
        num_groups -= 1;
    }

    if(verbose){
        print("Updated partition:");
        for (int i = 0; i < partition.size(); i++) {
            printf("%d ", partition[i]);
        }
        print("");
        print("Updated M: ");
        print(edge_counts);
    }

    /* === PHASE 3: Add entropy terms from final configuration === */
    
    // Normalizations
    for(int r: final_impacted_groups){
        node_counts_norm_2 += std::pow(node_counts[r],2);
        node_counts_norm_scaling_plus_1 += std::pow(node_counts[r],mean_degree_scaling + 1);
        node_counts_norm_2scaling_plus_2 += std::pow(node_counts[r],2*mean_degree_scaling + 2);
    }
    
    // In case the weight_out has never been computed, and we have more than one group, compute it now
    if(weight_out == -1.0 && num_groups > 1){
        density_out = density_in;
        weight_out = density_out * (n * n - node_counts_norm_2) / 
                           (node_counts_norm_scaling_plus_1 * node_counts_norm_scaling_plus_1 - node_counts_norm_2scaling_plus_2);
        if(verbose){
            print("Initialized weight_out to: " + std::to_string(weight_out));
            print("Initialized density_out to: " + std::to_string(density_out));
        }
    }

    // Recompute density and parameter prior entropy
    density_in = weight_in * node_counts_norm_2scaling_plus_2 / node_counts_norm_2;
    if(num_groups > 1){
        if(assortative){
            density_out = weight_out * (node_counts_norm_scaling_plus_1 * node_counts_norm_scaling_plus_1 - node_counts_norm_2scaling_plus_2) / (n * n - node_counts_norm_2);
        }else{ // Enforce non-assortative condition
            density_out = density_in;
            weight_out = density_out * (n * n - node_counts_norm_2) / 
                               (node_counts_norm_scaling_plus_1 * node_counts_norm_scaling_plus_1 - node_counts_norm_2scaling_plus_2);
        }
    }

    if(assortative){
        H_parameters = density_in + density_out;
    }else{
        H_parameters = density_in;
    }

    // H(nb|q): Partition count with new num_groups
    H_b += log_binom(n - 1, num_groups - 1);
    
    // H(b|nb): Group size distribution with new partition
    for(int r: final_impacted_groups){
        H_b -= log_factorial(node_counts[r]);
    }
    
    // H(k): Degree sequence entropy with new configuration
    if(timing){
        start_time = std::chrono::high_resolution_clock::now();
    }
    for(int r: final_impacted_groups){
        if(degree_correction_concentration < MAX_CONCENTRATION){
            H_k += log_binom(edge_counts_sum[r] + node_counts[r]*degree_correction_concentration - 1, 
                node_counts[r]*degree_correction_concentration - 1);
        }else{
            // Multinomial approximation for large concentration
            H_k -= log_factorial(edge_counts_sum[r]);
            H_k += edge_counts_sum[r]*std::log(node_counts[r]);
        }
    }
    if(timing){
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration<double>(end_time - start_time).count();
        time_H_k += duration;
    }
    
    // H(M): Edge count distribution with new configuration
    if(timing){
        start_time = std::chrono::high_resolution_clock::now();
    }
    for(int r: final_impacted_groups){
        for(int s = 0; s < num_groups; s++){
            if(r == s){
                H_M -= log_neg_binom(edge_counts[r][r]/2.0, 
                                    weight_in*std::pow(node_counts[r]*node_counts[r], mean_degree_scaling + 1)/2.0, 
                                    concentration_in);
            }else{
                // Use symmetry_factor to avoid double-counting
                double symmetry_factor = 1;
                if(final_impacted_groups.find(s) != final_impacted_groups.end()){
                    symmetry_factor = 0.5;
                }
                H_M -= symmetry_factor*log_neg_binom(edge_counts[r][s], 
                                                     weight_out*std::pow(node_counts[r]*node_counts[s], mean_degree_scaling + 1), 
                                                     concentration_out);
            }
        }
    }
    if(timing){
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration<double>(end_time - start_time).count();
        time_H_M += duration;
    }
    
    // H(A|k,M,b): Adjacency matrix entropy with new configuration
    if(timing){
        start_time = std::chrono::high_resolution_clock::now();
    }
    for(int r: final_impacted_groups){
        for(int s = 0; s < num_groups; s++){
            if(r == s){
                H_A_G_k_M_b -= log_factorial2(edge_counts[r][r]);
            }else{
                // If s is also in initial_impacted_groups, only add half the term to avoid double counting
                double symmetry_factor = 1;
                if(final_impacted_groups.find(s) != final_impacted_groups.end()){
                    symmetry_factor = 0.5;
                }
                H_A_G_k_M_b -= symmetry_factor*log_factorial(edge_counts[r][s]);
            }
        }
        H_A_G_k_M_b += log_factorial(edge_counts_sum[r]);
    }
    if(verbose){
        print("H(A|k,M,b) after adding final terms: " + std::to_string(H_A_G_k_M_b));
    }
    if(timing){
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration<double>(end_time - start_time).count();
        time_H_A_G_k_M_b += duration;
    }

    // H(s|b,depths): hierarchy-community entropy with new configuration
    for(int r: final_impacted_groups){
        double f_normalization = 1/std::sqrt(1 + node_counts[r]*group_depth*group_depth/(individual_depth*individual_depth));
        H_s_G_b_depths -= std::log(f_normalization) + f_normalization*f_normalization*group_depth*group_depth*group_scaled_score_sums[r]*group_scaled_score_sums[r]/(individual_depth*individual_depth);
    }
    
    if(verbose){
        // Print the entropies
        print("Old Entropies:");
        print("H_b: " + std::to_string(H_b_old));
        print("H_k: " + std::to_string(H_k_old));
        print("H_M: " + std::to_string(H_M_old));
        print("H_A_G_k_M_b: " + std::to_string(H_A_G_k_M_b_old));
        print("H_A_G_b: " + std::to_string(H_A_G_b));
        print("Computed Entropies:");
        print("H_b: " + std::to_string(H_b));
        print("H_k: " + std::to_string(H_k));
        print("H_M: " + std::to_string(H_M));
        print("H_A_G_k_M_b: " + std::to_string(H_A_G_k_M_b));
        print("H_b_delta: " + std::to_string(H_b - H_b_old));
        print("H_k_delta: " + std::to_string(H_k - H_k_old));
        print("H_M_delta: " + std::to_string(H_M - H_M_old));
        print("H_A_G_k_M_b_delta: " + std::to_string(H_A_G_k_M_b - H_A_G_k_M_b_old));
        print("M:");
        print(edge_counts);

    }

    // Update total entropies: add delta from component changes
    H_A_G_b += (H_k - H_k_old) + (H_M - H_M_old) + (H_A_G_k_M_b - H_A_G_k_M_b_old);
    H_A_b += (H_b - H_b_old) + (H_k - H_k_old) + (H_M - H_M_old) + (H_A_G_k_M_b - H_A_G_k_M_b_old);


    if(verbose){
        print("Updated partition:");
        for (int i = 0; i < partition.size(); i++) {
            printf("%d ", partition[i]);
        }
        print("");
    }

    if(verbose){
        // Check if neighbors[96] has an entry for 0, if so print it
        print("Neighbors of node 96:");
        for(const auto& [j, weight] : neighbors[96]){
            print("  Node " + std::to_string(j) + " with weight " + std::to_string(weight));
        }
    }
}

void MCMC::swap_groups(int group1, int group2) {
    // Swap entire group structures (O(1) pointer swap)
    std::swap(group_members[group1], group_members[group2]);
    std::swap(node_indices_in_groups[group1], node_indices_in_groups[group2]);
    std::swap(node_counts[group1], node_counts[group2]);
    
    // Swap edge_counts matrix rows and columns
    for(int s = 0; s < num_groups; s++){
        std::swap(edge_counts[group1][s], edge_counts[group2][s]);
    }
    for(int r = 0; r < num_groups; r++){
        std::swap(edge_counts[r][group1], edge_counts[r][group2]);
    }
    std::swap(edge_counts_sum[group1], edge_counts_sum[group2]);

    // Swap group score sums
    std::swap(group_scaled_score_sums[group1], group_scaled_score_sums[group2]);
    
    // Update partition references for moved nodes
    for(int node: group_members[group1]){
        partition[node] = group1;
    }
    for(int node: group_members[group2]){
        partition[node] = group2;
    }
}