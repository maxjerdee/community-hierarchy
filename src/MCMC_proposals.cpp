#include "MCMC_core.h"
#include "globals.h"
#include "helpers.h"

#include <algorithm>
#include <set>
#include <cmath>
#include <random>
#include <string>
#include <chrono>

/**
 * propose_move: Select and execute one random MCMC proposal.
 *
 * The move type is drawn with the (state-independent) weights set in MCMC_sweep:
 * - single_site_move:   move one node to another group (weight n)
 * - merge/split/merge_split_move: collective partition moves (MCMC_merge_split.cpp;
 *                       weight MERGE_SPLIT_WEIGHT each, linear-time default)
 * - score_move:         update one node's hierarchy score (weight n)
 * - group_score_move:   shift every score in one group by a common offset (weight n)
 * - parameter_move:     update one continuous parameter (weight = # free parameters)
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
            } else if (proposal_type == "group_score_move") {
                propose_group_score_move();
            } else if (proposal_type == "merge_move") {
                propose_merge_move();
            } else if (proposal_type == "split_move") {
                propose_split_move();
            } else if (proposal_type == "merge_split_move") {
                propose_merge_split_move();
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
 * The Hastings ratio of this proposal exactly cancels the prior of the
 * PARTITION, q! exp(-H_b) -- not the labelled exp(-H_b): for a split into a new
 * group, forward (1/q)(1/n_r)(q/n) and reverse (1/(q+1))(1-q/n)(1/q) against
 * exp(-dH_b) = q/((n-q) n_r) leave 1/(q+1) = q!/(q+1)!.  So the chain samples
 * partitions with weight q! exp(-H_b) x ..., the same target the merge/split
 * label factors produce.  The acceptance involves only
 * beta*dH_A_G_b + dH_s_G_b_depths.  See arXiv:2501.07668.
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
    old_group = next_int(rng, 0, num_groups - 1);
    node = group_members[old_group][next_int(rng, 0, node_counts[old_group] - 1)];
    bool empties_group = node_counts[old_group] == 1;

    if (empties_group) {
        // Singleton: merge or no-op
        if (p <= static_cast<double>(num_groups - 1) / n) {
            return; // no-op: splitting a singleton and immediately reinstating it
        }
        if (num_groups_fixed) return;
        new_group = next_int(rng, 0, num_groups - 2);
        is_merge = true;
    } else {
        // Non-singleton: split or move (possibly same group = no-op)
        if (p <= static_cast<double>(num_groups) / n) {
            if (num_groups_fixed) return;
            new_group = num_groups; // split into a new singleton group
        } else {
            new_group = next_int(rng, 0, num_groups - 1);
            if (new_group == old_group) return; // no-op
        }
    }

    // Store old entropy values for acceptance calculation
    double H_A_G_b_old = H_A_G_b;
    double H_s_G_b_depths_old = H_s_G_b_depths;

    flip_nodes({node}, new_group);

    // The proposal cancels the partition prior.  Only the data likelihood is tempered: the
    // score prior H_s_G_b_depths enters at full weight (convention of MCMC_core.h).
    double delta_H = beta * (H_A_G_b - H_A_G_b_old) + (H_s_G_b_depths - H_s_G_b_depths_old);
    if (uniform_random(rng) >= std::exp(-delta_H)) {
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
 * propose_parameter_move: update one free continuous parameter.
 *
 * Proposals are random walks in the CDF space of each parameter's prior (reflected
 * at 0 and 1), so the prior cancels from the Hastings ratio:
 *   variation_in/out, degree_correction ~ Uniform(0,1)
 *   density_in/out                     ~ Exponential(mean 1)
 *   individual_depth, group_depth, ties_parameter ~ half-Cauchy(DEPTH_PRIOR_WIDTH)
 * change_parameter() recomputes the affected entropy and accepts or restores.
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
    // next_int, not uniform_int_distribution: portable across standard libraries (rng.h)
    const auto& parameter = free_parameters[
        static_cast<std::size_t>(next_int(rng, 0, static_cast<int>(free_parameters.size()) - 1))];

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
    int node = next_int(rng, 0, n - 1);
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
    // The proposal is reversible w.r.t. the score prior exp(-H_s_G_b_depths) (it is
    // a CDF-space random walk on the exact conditional of this node's score), so the
    // prior cancels and only the tempered direction likelihood remains.  This is why
    // H_s_G_b_depths must be UNtempered in every other move too (MCMC_core.h).
    double delta_H = H_Adir_G_A_s_nu - H_Adir_G_A_s_nu_old;
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
 * propose_group_score_move: shift every node's score in one random group by a
 * common offset delta (random-walk in scaled-score units).
 *
 * This translates a whole group along the between-group (sigma_C) direction.
 * Within-group score differences are invariant under the shift, so the
 * orientation likelihood only changes for edges crossing the group boundary;
 * the score prior changes via the per-node terms and the group term. Accept on
 * the full delta_H (the proposal is a symmetric random walk on delta).
 */
void MCMC::propose_group_score_move() {
    group_score_proposals++;
    if (num_groups <= 1) return;  // no between-group structure to move
    int r = next_int(rng, 0, num_groups - 1);
    if (group_members[r].empty()) return;
    double delta = next_normal(rng, GROUP_SCORE_STEP_SIZE);
    if (delta == 0.0) return;

    double H_s_old = H_s_G_b_depths;
    double H_A_old = H_Adir_G_A_s_nu;

    // --- H_Adir_G_A_s_nu: remove old contributions of cross-group edges ---
    for (int i : group_members[r]) {
        for (const auto& edge : wins[i]) {
            int j = edge.first; if (partition[j] == r) continue;
            double lp = scores[i] - log_sum_exp(std::vector<double>{scores[i], scores[j], std::log(ties_parameter) + scores[i]/2.0 + scores[j]/2.0});
            H_Adir_G_A_s_nu -= 0.0 - edge.second * lp;
        }
        for (const auto& edge : losses[i]) {
            int j = edge.first; if (partition[j] == r) continue;
            double lp = scores[j] - log_sum_exp(std::vector<double>{scores[j], scores[i], std::log(ties_parameter) + scores[j]/2.0 + scores[i]/2.0});
            H_Adir_G_A_s_nu -= 0.0 - edge.second * lp;
        }
        for (const auto& edge : ties[i]) {
            int j = edge.first; if (partition[j] == r) continue;
            double lp = std::log(ties_parameter) + scores[i]/2.0 + scores[j]/2.0 - log_sum_exp(std::vector<double>{scores[i], scores[j], std::log(ties_parameter) + scores[i]/2.0 + scores[j]/2.0});
            H_Adir_G_A_s_nu -= 0.0 - edge.second * lp;
        }
    }

    // --- H_s_G_b_depths: net contribution of group r (sigma-dependent part) ---
    auto group_H_s_contrib = [&]() {
        double pernode = 0.0;
        for (int i : group_members[r]) pernode += scaled_scores[i] * scaled_scores[i];
        double S = group_scaled_score_sums[r];
        double f = 1.0 / std::sqrt(1 + node_counts[r]*group_depth*group_depth/(individual_depth*individual_depth));
        double group_term = std::log(f) + f*f*group_depth*group_depth*S*S/(individual_depth*individual_depth);
        return pernode - group_term;  // matches calculate_H_s_G_b_depths (per-node added, group subtracted)
    };
    double old_contrib = group_H_s_contrib();

    // Apply the shift to all members.
    for (int i : group_members[r]) {
        scaled_scores[i] += delta;
        scores[i] = scaled_scores[i] * individual_depth;
    }
    group_scaled_score_sums[r] += node_counts[r] * delta;

    H_s_G_b_depths += group_H_s_contrib() - old_contrib;

    // --- H_Adir_G_A_s_nu: add new contributions of cross-group edges ---
    for (int i : group_members[r]) {
        for (const auto& edge : wins[i]) {
            int j = edge.first; if (partition[j] == r) continue;
            double lp = scores[i] - log_sum_exp(std::vector<double>{scores[i], scores[j], std::log(ties_parameter) + scores[i]/2.0 + scores[j]/2.0});
            H_Adir_G_A_s_nu += 0.0 - edge.second * lp;
        }
        for (const auto& edge : losses[i]) {
            int j = edge.first; if (partition[j] == r) continue;
            double lp = scores[j] - log_sum_exp(std::vector<double>{scores[j], scores[i], std::log(ties_parameter) + scores[j]/2.0 + scores[i]/2.0});
            H_Adir_G_A_s_nu += 0.0 - edge.second * lp;
        }
        for (const auto& edge : ties[i]) {
            int j = edge.first; if (partition[j] == r) continue;
            double lp = std::log(ties_parameter) + scores[i]/2.0 + scores[j]/2.0 - log_sum_exp(std::vector<double>{scores[i], scores[j], std::log(ties_parameter) + scores[i]/2.0 + scores[j]/2.0});
            H_Adir_G_A_s_nu += 0.0 - edge.second * lp;
        }
    }

    // Metropolis acceptance on the full change (symmetric RW proposal on delta).
    // Only the likelihood is tempered; the score prior enters at full weight.
    double delta_H = beta * (H_Adir_G_A_s_nu - H_A_old) + (H_s_G_b_depths - H_s_old);
    if (uniform_random(rng) < std::exp(-delta_H)) {
        accepted_group_score_proposals++;
    } else {
        for (int i : group_members[r]) {
            scaled_scores[i] -= delta;
            scores[i] = scaled_scores[i] * individual_depth;
        }
        group_scaled_score_sums[r] -= node_counts[r] * delta;
        H_s_G_b_depths = H_s_old;
        H_Adir_G_A_s_nu = H_A_old;
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
    // std::set, not unordered_set: these are walked to accumulate the energy
    // terms, and a different iteration order changes the floating-point rounding,
    // which eventually flips a Metropolis decision (see rng.h).
    std::set<int> initial_impacted_groups;
    std::set<int> final_impacted_groups;
    
    initial_impacted_groups.insert(old_group);
    if(new_group != num_groups){  // If not creating a new group
        initial_impacted_groups.insert(new_group);
    }
    final_impacted_groups.insert(old_group);
    final_impacted_groups.insert(new_group);
    
    // Case 5 - empty_old_group with new_group == num_groups (whole group moves
    // to a BRAND-NEW group).  The freshly created group is immediately compacted
    // back into old_group's slot, so this is a pure relabelling no-op: num_groups
    // is unchanged and the nodes end up labelled old_group.  The transient index
    // `num_groups` must therefore NOT stay in final_impacted_groups -- after the
    // compaction it is out of range, and the Phase 3 re-add would read past the end
    // of node_counts/edge_counts, giving H_M=inf, H_k=nan, H_A_G_k_M_b=-inf.
    // The partition itself stays valid, so this corrupts SILENTLY.
    // Unreachable from propose_single_site_move (its singleton branch only merges
    // into existing groups; its split branch only fires for non-singletons), which
    // is why it has never shown up -- but merge/split needs the batch path.
    if(new_group == num_groups && empty_old_group){
        final_impacted_groups.erase(new_group);
    }

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
    double H_W_old = H_W;
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

    // H_W: remove per-block-pair contributions of affected groups (weighted SBM only)
    if (weighted_SBM) {
        // H(μ): remove old P(μ) term — depends on q = num_groups, which is
        // still the pre-proposal value here (Phase 2 hasn't run yet).
        if (w_total > 0.0) {
            long long q_old   = static_cast<long long>(num_groups);
            long long np_old  = q_old * (q_old + 1) / 2;
            if (np_old > 1)
                H_W -= static_cast<double>(np_old - 1) * std::log(w_total)
                       - log_factorial(static_cast<int>(np_old - 1));
        }
        for (int r : initial_impacted_groups) {
            // Diagonal (r,r): stored as 2× — halve
            {
                int    m  = edge_counts[r][r]   / 2;
                double mu = weight_counts[r][r] / 2.0;
                if (m > 0) H_W -= (m - 1) * std::log(mu) - log_factorial(m - 1);
            }
            // Off-diagonal: use symmetry factor to avoid double-counting
            for (int s = 0; s < num_groups; s++) {
                if (s == r) continue;
                double sym = initial_impacted_groups.count(s) ? 0.5 : 1.0;
                int    ms  = edge_counts[r][s];
                double mus = weight_counts[r][s];
                if (ms > 0) H_W -= sym * ((ms - 1) * std::log(mus) - log_factorial(ms - 1));
            }
        }
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
        // neighbors_com[node][node] already encodes 2×weight, so treating both directions
        // separately would subtract 4× instead of 2×.
        for(const auto& [j, A_ij] : neighbors_com[node]){
            int neighbor_group = partition[j];
            int count_delta = (weighted_SBM || two_layer_mode) ? 1 : static_cast<int>(A_ij);
            edge_counts[old_group][neighbor_group] -= count_delta;
            edge_counts_sum[old_group] -= count_delta;
            if (j != node) {
                edge_counts[neighbor_group][old_group] -= count_delta;
                edge_counts_sum[neighbor_group] -= count_delta;
            }
            if (weighted_SBM) {
                weight_counts[old_group][neighbor_group] -= A_ij;
                weight_counts_sum[old_group] -= A_ij;
                if (j != node) {
                    weight_counts[neighbor_group][old_group] -= A_ij;
                    weight_counts_sum[neighbor_group] -= A_ij;
                }
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
            if (weighted_SBM) {
                weight_counts_sum.push_back(0.0);
                weight_counts.push_back(std::vector<double>(num_groups, 0.0));
                for(auto& row: weight_counts){
                    row.push_back(0.0);
                }
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
        for(const auto& [j, A_ij] : neighbors_com[node]){
            int neighbor_group = partition[j];
            int count_delta = (weighted_SBM || two_layer_mode) ? 1 : static_cast<int>(A_ij);
            edge_counts[new_group][neighbor_group] += count_delta;
            edge_counts_sum[new_group] += count_delta;
            if (j != node) {
                edge_counts[neighbor_group][new_group] += count_delta;
                edge_counts_sum[neighbor_group] += count_delta;
            }
            if (weighted_SBM) {
                weight_counts[new_group][neighbor_group] += A_ij;
                weight_counts_sum[new_group] += A_ij;
                if (j != node) {
                    weight_counts[neighbor_group][new_group] += A_ij;
                    weight_counts_sum[neighbor_group] += A_ij;
                }
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
        if (weighted_SBM) {
            weight_counts.pop_back();
            for(auto& row: weight_counts){
                row.pop_back();
            }
            weight_counts_sum.pop_back();
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

    // H_W: add per-block-pair contributions of affected groups in new configuration (weighted SBM only)
    if (weighted_SBM) {
        for (int r : final_impacted_groups) {
            // Diagonal (r,r): stored as 2× — halve
            {
                int    m  = edge_counts[r][r]   / 2;
                double mu = weight_counts[r][r] / 2.0;
                if (m > 0) H_W += (m - 1) * std::log(mu) - log_factorial(m - 1);
            }
            // Off-diagonal: use symmetry factor to avoid double-counting
            for (int s = 0; s < num_groups; s++) {
                if (s == r) continue;
                double sym = final_impacted_groups.count(s) ? 0.5 : 1.0;
                int    ms  = edge_counts[r][s];
                double mus = weight_counts[r][s];
                if (ms > 0) H_W += sym * ((ms - 1) * std::log(mus) - log_factorial(ms - 1));
            }
        }
        // H(μ): add new P(μ) term — num_groups has already been updated in Phase 2
        // if a group was created or merged.
        if (w_total > 0.0) {
            long long q_new  = static_cast<long long>(num_groups);
            long long np_new = q_new * (q_new + 1) / 2;
            if (np_new > 1)
                H_W += static_cast<double>(np_new - 1) * std::log(w_total)
                       - log_factorial(static_cast<int>(np_new - 1));
        }
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

    // Update total entropies: add delta from component changes.
    // H_W = -log P (description-length contribution, negative when model fits well), so add its delta.
    H_A_G_b += (H_k - H_k_old) + (H_M - H_M_old) + (H_A_G_k_M_b - H_A_G_k_M_b_old) + (H_W - H_W_old);
    H_A_b += (H_b - H_b_old) + (H_k - H_k_old) + (H_M - H_M_old) + (H_A_G_k_M_b - H_A_G_k_M_b_old) + (H_W - H_W_old);


    if(verbose){
        print("Updated partition:");
        for (int i = 0; i < partition.size(); i++) {
            printf("%d ", partition[i]);
        }
        print("");
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

    // Swap weight_counts matrix rows and columns (weighted mode only)
    if (weighted_SBM) {
        for(int s = 0; s < num_groups; s++){
            std::swap(weight_counts[group1][s], weight_counts[group2][s]);
        }
        for(int r = 0; r < num_groups; r++){
            std::swap(weight_counts[r][group1], weight_counts[r][group2]);
        }
        std::swap(weight_counts_sum[group1], weight_counts_sum[group2]);
    }

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