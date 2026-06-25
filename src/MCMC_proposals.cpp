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
 * Randomly selects a proposal type (single-site, score, or parameter)
 * weighted by proposal_weights, then dispatches to the appropriate handler.
 *
 * Proposal types:
 * - single_site_move: Move one node to a different group (weight: n)
 * - score_move: Update a node's hierarchy score (weight: n)
 * - parameter_move: Update a model parameter (weight: # free parameters)
 */
void MCMC::propose_move() {
    start_time = std::chrono::high_resolution_clock::now();

    double p = uniform_random(rng);
    double cumulative_weight = 0.0;
    double total_weight = 0.0;

    for (const auto& [proposal_type, weight] : proposal_weights) {
        total_weight += weight;
    }

    for (const auto& [proposal_type, weight] : proposal_weights) {
        cumulative_weight += weight;
        if (p < cumulative_weight / total_weight) {
            if (proposal_type == "single_site_move") {
                propose_single_site_move();
            } else if (proposal_type == "parameter_move") {
                propose_parameter_move();
            } else if (proposal_type == "score_move") {
                propose_score_move();
            }

            end_time = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration<double>(end_time - start_time).count();

            if (proposal_type == "single_site_move") {
                time_single_site_proposals += duration;
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
 * Proposal samples from the partition prior so the Hastings ratio exactly
 * cancels H_b; acceptance is based only on the full likelihood change
 * (H_A_G_b + H_s_G_b_depths).  See arXiv:2501.07668.
 *
 * Node selection: pick a group uniformly, then a node uniformly from that group.
 * - Singleton node (empties group): with prob (K-1)/n do nothing; otherwise merge
 *   into one of the K-1 remaining groups chosen uniformly.
 * - Non-singleton node: with prob K/n split into a new group; otherwise move to
 *   one of the K existing groups uniformly (same group = no-op).
 *
 * Rejection undo:
 * - merge (is_merge=true): compaction renamed groups, so undo by splitting the
 *   node back out into a brand-new singleton (flip to num_groups).
 * - split / standard move: old_group label is still valid, flip back directly.
 */
void MCMC::propose_single_site_move() {
    single_site_proposals++;
    if (verbose) {
        print("Proposing single site move =================================");
    }

    double p = uniform_random(rng);
    int old_group, new_group, node;
    bool is_merge = false;

    // Sample a random group, then a random node from it
    old_group = std::uniform_int_distribution<int>(0, num_groups - 1)(rng);
    node = group_members[old_group][std::uniform_int_distribution<int>(0, node_counts[old_group] - 1)(rng)];
    bool empties_group = node_counts[old_group] == 1;

    if (empties_group) {
        // Singleton: merge or no-op
        if (p <= static_cast<double>(num_groups - 1) / n) {
            return; // no-op: splitting a singleton and immediately reinstating it
        }
        new_group = std::uniform_int_distribution<int>(0, num_groups - 2)(rng);
        is_merge = true;
    } else {
        // Non-singleton: split or move (possibly same group = no-op)
        if (p <= static_cast<double>(num_groups) / n) {
            new_group = num_groups; // split into a new singleton group
        } else {
            new_group = std::uniform_int_distribution<int>(0, num_groups - 1)(rng);
            if (new_group == old_group) return; // no-op
        }
    }

    // Store old entropy values for acceptance calculation
    double H_A_G_b_old = H_A_G_b;
    double H_s_G_b_depths_old = H_s_G_b_depths;

    flip_nodes({node}, new_group);

    // Proposal cancels H_b exactly; parameter proposals use CDF-transformed distributions
    // (exponential prior on density_in/density_out cancels in acceptance)
    double delta_H = H_A_G_b - H_A_G_b_old + H_s_G_b_depths - H_s_G_b_depths_old;
    if (uniform_random(rng) >= std::exp(-beta * delta_H)) {
        // Reject: undo the move
        if (is_merge) {
            // Compaction occurred: old_group was removed, so undo by splitting
            // the node back out into a new singleton group.
            flip_nodes({node}, num_groups);
        } else {
            flip_nodes({node}, old_group);
        }
    } else {
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
 * density_in, density_out
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
    } else if (parameter == "density_in") {
        old_value = density_in;
        new_value = exponential_proposal(density_in, 1.0, PARAMETER_PROPOSAL_STEP_SIZE);
    } else if (parameter == "density_out") {
        old_value = density_out;
        new_value = exponential_proposal(density_out, 1.0, PARAMETER_PROPOSAL_STEP_SIZE);
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
    for(const auto& edge : wins[node]){
        int j = edge.first;
        double Adir_ij = edge.second;
        double log_p_win_ij = scores[node] - log_sum_exp(std::vector<double>{scores[node], scores[j], std::log(ties_parameter) + scores[node]/2.0 + scores[j]/2.0}); // P(node -> j|node - j)
        H_Adir_G_A_s_nu -= 0.0 - Adir_ij * log_p_win_ij;
    }
    for(const auto& edge : losses[node]){
        int j = edge.first;
        double Adir_ji = edge.second;
        double log_p_win_ji = scores[j] - log_sum_exp(std::vector<double>{scores[j], scores[node], std::log(ties_parameter) + scores[j]/2.0 + scores[node]/2.0}); // P(j -> node|j - node)
        H_Adir_G_A_s_nu -= 0.0 - Adir_ji * log_p_win_ji;
    }
    for(const auto& edge : ties[node]){
        int j = edge.first;
        double Aties_ij = edge.second;
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
    for(const auto& edge : wins[node]){
        int j = edge.first;
        double Adir_ij = edge.second;
        double log_p_win_ij = scores[node] - log_sum_exp(std::vector<double>{scores[node], scores[j], std::log(ties_parameter) + scores[node]/2.0 + scores[j]/2.0}); // P(node -> j|node - j)
        H_Adir_G_A_s_nu += 0.0 - Adir_ij * log_p_win_ij;
    }
    for(const auto& edge : losses[node]){
        int j = edge.first;
        double Adir_ji = edge.second;
        double log_p_win_ji = scores[j] - log_sum_exp(std::vector<double>{scores[j], scores[node], std::log(ties_parameter) + scores[j]/2.0 + scores[node]/2.0}); // P(j -> node|j - node)
        H_Adir_G_A_s_nu += 0.0 - Adir_ji * log_p_win_ji;
    }
    for(const auto& edge : ties[node]){
        int j = edge.first;
        double Aties_ij = edge.second;
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

    // Case 4: self-merge — caller passed new_group == old_group while the group would be
    // emptied. Redirect to the last group so compaction renames it into old_group's slot.
    bool empty_old_group_check = (node_counts[old_group] == static_cast<int>(nodes.size()));
    if (empty_old_group_check && new_group == old_group) {
        new_group = num_groups - 1;
    }

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
        for(const auto& edge : neighbors[96]){
            int j = edge.first;
            double weight = edge.second;
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
                                    density_in*node_counts[r]*node_counts[r]/2.0,
                                    concentration_in);
            }else{
                // Between-group edges
                // Use symmetry_factor to avoid double-counting when both r and s are impacted
                double symmetry_factor = 1;
                if(initial_impacted_groups.find(s) != initial_impacted_groups.end()){
                    symmetry_factor = 0.5;
                }
                H_M += symmetry_factor*log_neg_binom(edge_counts[r][s],
                                                     density_out*node_counts[r]*node_counts[s],
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

    for(int r: initial_impacted_groups){
        node_counts_norm_2 -= std::pow(node_counts[r],2);
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
        // Remove node from old_group using swap-and-pop (O(1) removal)
        size_t node_index = node_indices_in_groups[old_group][node];
        int last_node = group_members[old_group].back();
        group_members[old_group][node_index] = last_node;
        node_indices_in_groups[old_group][last_node] = node_index;
        group_members[old_group].pop_back();
        node_indices_in_groups[old_group].erase(node);
        node_counts[old_group]--;

        // Update edge counts: subtract edges from old_group.
        // partition[node] still holds old_group here, so self-loops (j==node) correctly
        // read old_group as the neighbor group.  Skip the reverse direction for self-loops:
        // neighbors[node][node] already encodes 2×weight, so treating both directions
        // separately would subtract 4× instead of 2×.
        for(const auto& [j, A_ij] : neighbors[node]){
            int neighbor_group = partition[j];
            edge_counts[old_group][neighbor_group] -= A_ij;
            edge_counts_sum[old_group] -= A_ij;
            if (j != node) {
                edge_counts[neighbor_group][old_group] -= A_ij;
                edge_counts_sum[neighbor_group] -= A_ij;
            }
        }

        // Update partition AFTER edge subtraction so self-loop reads above are correct.
        partition[node] = new_group;

        // Update group_scaled_score_sums: subtract scaled score from old_group
        group_scaled_score_sums[old_group] -= scaled_scores[node];

        // If creating a new group, resize all data structures
        if(new_group == num_groups){
            group_members.push_back(std::vector<int>());
            node_indices_in_groups.push_back(std::unordered_map<int, size_t>());
            node_counts.push_back(0);
            edge_counts_sum.push_back(0);
            edge_counts.push_back(std::vector<int>(num_groups, 0));
            for(auto& row: edge_counts){
                row.push_back(0);
            }
            group_scaled_score_sums.push_back(0.0);
            num_groups += 1;
        }

        // Add node to new_group
        group_members[new_group].push_back(node);
        node_indices_in_groups[new_group][node] = group_members[new_group].size() - 1;
        node_counts[new_group]++;

        // Update edge counts: add edges to new_group.
        // Same self-loop guard: skip reverse direction when j==node.
        for(const auto& [j, A_ij] : neighbors[node]){
            int neighbor_group = partition[j];
            edge_counts[new_group][neighbor_group] += A_ij;
            edge_counts_sum[new_group] += A_ij;
            if (j != node) {
                edge_counts[neighbor_group][new_group] += A_ij;
                edge_counts_sum[neighbor_group] += A_ij;
            }
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
    }

    // Enforce non-assortative constraint; initialize density_out on first split
    if(!assortative || (density_out == -1.0 && num_groups > 1)){
        density_out = density_in;
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
                                    density_in*node_counts[r]*node_counts[r]/2.0,
                                    concentration_in);
            }else{
                // Use symmetry_factor to avoid double-counting
                double symmetry_factor = 1;
                if(final_impacted_groups.find(s) != final_impacted_groups.end()){
                    symmetry_factor = 0.5;
                }
                H_M -= symmetry_factor*log_neg_binom(edge_counts[r][s],
                                                     density_out*node_counts[r]*node_counts[s],
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
        for(const auto& edge : neighbors[96]){
            int j = edge.first;
            double weight = edge.second;
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