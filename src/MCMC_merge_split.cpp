#include "MCMC_core.h"
#include "globals.h"
#include "helpers.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <chrono>


/**
 * propose_merge_move: Propose merging two groups (stub)
 * 
 * TODO: Implementation pending
 */
void MCMC::propose_merge_move() {
    merge_proposals++;
    if (verbose || debug) {
        print("Proposing merge move =================================");
    }
    // Check that there are at least two groups to merge
    if (num_groups < 2) {
        return;
    }

    // Store initial state values
    double H_A_G_b_old = H_A_G_b;
    double H_b_old = H_b;
    double H_parameters_old = H_parameters;
    int num_groups_old = num_groups;

    if(debug){
        printf("Original partition: ");
        for(int i = 0; i < n; i++){
            printf("%d ", partition[i]);
        }
        print("");
    }

    // Select random group to merge into another group
    int group_to_merge = std::uniform_int_distribution<int>(0, num_groups - 1)(rng);
    
    double H_merge = std::log(num_groups); // Entropy of choosing this group to merge

    // Relabel the group to be the last group in order to avoid issues with empty group re-labeling during the split
    swap_groups(group_to_merge, num_groups - 1);
    group_to_merge = num_groups - 1;

    if(debug){
        printf("Partition after relabeling: ");
        for(int i = 0; i < n; i++){
            printf("%d ", partition[i]);
        }
        print("");
    }

    std::vector<int> nodes_to_merge; // Store the nodes in this group to merge into all groups
    for(int node: group_members[group_to_merge]){
        nodes_to_merge.push_back(node);
    }

    std::unordered_map<int, double> group_log_weights; // Store entropies of merging into each group
    for(int new_group = 0; new_group < num_groups; new_group++){
        if(new_group != group_to_merge){
            // Temporarily merge into group new_group
            flip_nodes(nodes_to_merge, new_group);
            group_log_weights[new_group] = -(beta*(H_A_G_b + H_parameters) + H_b);
        }
        if(debug){
            print("Partition after merging to group: " + std::to_string(new_group) + " ");
            for(int i = 0; i < n; i++){
                printf("%d ", partition[i]);
            }
            print("");
        }
    }
    // Sample the new_group based on these weights
    std::pair<int, double> sampled_merge = sample_log_weights_map(group_log_weights);
    int new_group = sampled_merge.first;
    double entropy = sampled_merge.second;
    H_merge += entropy; // Entropy of choosing this group to merge into

    // Flip nodes into the selected new_group
    flip_nodes(nodes_to_merge, new_group);

    // Change the new_group to the last group to avoid re-labeling issues during the split
    swap_groups(new_group, num_groups - 1);
    new_group = num_groups - 1;

    if(debug){
        printf("Partition after merge proposal: ");
        for(int i = 0; i < n; i++){
            printf("%d ", partition[i]);
        }
        print("");
    }

    if(verbose || debug){
        print("Merged group " + std::to_string(group_to_merge) + " into group " + std::to_string(new_group));
    }

    // Compute the entropy of the reverse split
    double H_split = H_reverse_split(new_group, nodes_to_merge);

    if(debug){
        printf("Reverse split entropy: %f\n", H_split);
        printf("Partition after computing reverse split entropy: ");
        for(int i = 0; i < n; i++){
            printf("%d ", partition[i]);
        }
        print("");
    }

    if(verbose || debug){
        printf("H_merge: %f, H_split: %f\n", H_merge, H_split);
        printf("H_b: %f, H_b_old: %f\n", H_b, H_b_old);
    }

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
    double delta_H = beta*(H_A_G_b - H_A_G_b_old + H_parameters - H_parameters_old) + H_b - H_b_old + H_split - H_merge;
    double acceptance_prob = std::exp(-delta_H - wang_landau_adjustment);
    if (uniform_random(rng) >= acceptance_prob) {
        // Reject: split back the groups
        flip_nodes(nodes_to_merge, group_to_merge);
    } else {
        accepted_merge_proposals++;
    }

    // print("Check merge");
    // exit(1);
}

/**
 * propose_split_move: Propose splitting one group into two (stub)
 * 
 * TODO: Implementation pending
 */
void MCMC::propose_split_move() {
    split_proposals++;
    if (verbose || debug) {
        print("Proposing split move =================================");
    }
    // Store initial state values
    double H_A_G_b_old = H_A_G_b;
    double H_b_old = H_b;
    double H_parameters_old = H_parameters;
    int num_groups_old = num_groups;
    std::vector<int> partition_original;
    partition_original.resize(partition.size());
    std::copy(partition.begin(), partition.end(), partition_original.begin());


    if(verbose || debug){
        printf("Original partition: ");
        for(int i = 0; i < n; i++){
            printf("%d ", partition[i]);
        }
        print("");
    }

    // Select a random group to split
    int group_to_split = std::uniform_int_distribution<int>(0, num_groups - 1)(rng);

    // Relabel the group to be the last group in order to avoid issues with empty group re-labeling during the split
    swap_groups(group_to_split, num_groups - 1);
    group_to_split = num_groups - 1;

    double H_split_given_staging = std::log(num_groups); // Entropy of choosing this group to split

    // Store the nodes that will be split
    std::vector<int> nodes_to_split;
    for(int node: group_members[group_to_split]){
        nodes_to_split.push_back(node);
    }

    if(nodes_to_split.size() < 2){
        // Can't meaningfully split a group with less than 2 nodes
        return;
    }

    int new_group_index = num_groups; // New group will be at the end after relabeling

    if(verbose || debug){
        print("Splitting group: " + std::to_string(group_to_split));
        print("Nodes to split:");
        for(int node: nodes_to_split){
            print("Node: " + std::to_string(node));
        }
    }

    // Make an initial split using a hueristic, creates new group at last index
    hueristic_split_group({group_to_split});

    if(verbose || debug){
        printf("Initial hueristic split: ");
        for(int i = 0; i < n; i++){
            printf("%d ", partition[i]);
        }
        print("");
    }

    for(int sweep = 0; sweep < GIBBS_REFINEMENT_SWEEPS; sweep++){
        double _ = gibbs_sweep(nodes_to_split, {group_to_split, new_group_index}); // Don't need split probability here
    }

    if(verbose || debug){
        printf("Staged split: ");
        for(int i = 0; i < n; i++){
            printf("%d ", partition[i]);
        }
        print("");
        printf("Current H_split_given_staging: %f\n", H_split_given_staging);
    }

    
    std::vector<int> partition_staged;
    partition_staged.resize(partition.size());
    std::copy(partition.begin(), partition.end(), partition_staged.begin());

    // Sample the final split and get the split probability
    H_split_given_staging += gibbs_sweep(nodes_to_split, {group_to_split, new_group_index});

    if(verbose || debug){
        printf("Final split: ");
        for(int i = 0; i < n; i++){
            printf("%d ", partition[i]);
        }
        print("");
        printf("H_split_given_staging: %f\n", H_split_given_staging);
    }

    std::vector<int> partition_proposed;
    partition_proposed.resize(partition.size());
    std::copy(partition.begin(), partition.end(), partition_proposed.begin());

    // Compute the probability of the reverse merge
    double H_merge_old_new = H_reverse_merge(group_to_split, new_group_index);
    swap_groups(group_to_split, new_group_index); // Swap indices
    double H_merge_new_old = H_reverse_merge(group_to_split, new_group_index);
    swap_groups(group_to_split, new_group_index); // Swap back
    double H_merge = -std::log(std::exp(-H_merge_old_new) + std::exp(-H_merge_new_old)); // Combine both merge directions

    if(debug || verbose){
        printf("H_merge_old_new: %f, H_merge_new_old: %f, H_merge: %f\n", H_merge_old_new, H_merge_new_old, H_merge);
    }

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
    double delta_H = beta*(H_A_G_b - H_A_G_b_old + H_parameters - H_parameters_old) + H_b - H_b_old + H_merge - H_split_given_staging;
    if(debug || verbose){
        printf("H_b: %f, H_b_old: %f\n", H_b, H_b_old);
        printf("H_merge: %f, H_split_given_staging: %f\n", H_merge, H_split_given_staging);
        printf("Partition original: ");
        for(int i = 0; i < n; i++){
            printf("%d ", partition_original[i]);
        }
        print("");
        printf("Partition staged: ");
        for(int i = 0; i < n; i++){
            printf("%d ", partition_staged[i]);
        }
        print("");
        printf("Partition proposed: ");
        for(int i = 0; i < n; i++){
            printf("%d ", partition_proposed[i]);
        }
        print("");
    }
    double acceptance_prob = std::exp(-delta_H - wang_landau_adjustment);
    if (uniform_random(rng) >= acceptance_prob) {
        // Reject: merge back the groups
        flip_nodes(group_members[new_group_index], group_to_split);
    } else {
        accepted_split_proposals++;
    }

    // print("Check split");
    // exit(1);
}

// Heuristic split/merge helper functions (stubs)

/**
 * hueristic_split_group: Heuristic for splitting groups, randomly selects among the three methods
 * 
 * @param groups_to_split: Indices of the (one or two) groups to split
 */
void MCMC::hueristic_split_group(std::vector<int> groups_to_split) {
    // If one group is provided, split some of its members off into the last index
    // If two groups are provided, combine their members and split into the same groups
    if(verbose || debug){
        print("Performing heuristic split of groups");
        for(int group: groups_to_split){
            print("Group to split: " + std::to_string(group));
        }
    }

    std::vector<int> groups_after_split;
    if(groups_to_split.size() == 1){
        groups_after_split = {groups_to_split[0], num_groups}; // New group at the end
    }else if(groups_to_split.size() == 2){
        groups_after_split = {groups_to_split[0], groups_to_split[1]};  // Same groups
    }else{
        print("Error: Unsupported number of groups to split in heuristic_split_group");
        exit(1);
    }

    if(verbose || debug){
        print("Groups after split will be:");
        for(int group: groups_after_split){
            print("Group: " + std::to_string(group));
        }
    }

    // Make vector containing the indices of nodes in the group(s) to split
    std::vector<int> nodes_to_split;
    for(int group: groups_to_split){
        for(int node: group_members[group]){
            nodes_to_split.push_back(node);
        }
    }

    // If less than 2 nodes are present, no meaningful split can be made
    if(nodes_to_split.size() < 2){
        return;
    }

    // Randomly permute these nodes
    std::shuffle(nodes_to_split.begin(), nodes_to_split.end(), rng);

    if(verbose || debug){
        print("Nodes to split:");
        for(int node: nodes_to_split){
            print("Node: " + std::to_string(node));
        }
    }

    double p = uniform_random(rng);
    if(p < 1.0/3.0){
        hueristic_split_group_random(nodes_to_split, groups_after_split);
    }else if(p < 2.0/3.0){
        hueristic_split_group_sequential_new_group(nodes_to_split, groups_after_split);
    }else{
        hueristic_split_group_sequential_new_groups(nodes_to_split, groups_after_split);
    }
}

/**
 * hueristic_split_group_random: Random heuristic split
 * 
 * @param nodes_to_split: Nodes to be split
 * @param groups_after_split: Groups after the split (old and new)
 */
void MCMC::hueristic_split_group_random(std::vector<int> nodes_to_split, std::vector<int> groups_after_split) {
    if(verbose || debug){
        print("Performing random heuristic split");
    }
    // Choose a random split point excluding the trivial splits
    int split_index = std::uniform_int_distribution<int>(1, nodes_to_split.size() - 1)(rng);
    std::vector<int> nodes_group1(nodes_to_split.begin(), nodes_to_split.begin() + split_index);
    std::vector<int> nodes_group2(nodes_to_split.begin() + split_index, nodes_to_split.end());
    flip_nodes(nodes_group1, groups_after_split[0]);
    flip_nodes(nodes_group2, groups_after_split[1]);
}

/**
 * hueristic_split_group_sequential_new_group: Sequential split heuristic (stub)
 * 
 * TODO: Implementation pending
 * @param nodes_to_split: Nodes to be split
 * @param groups_after_split: Groups after the split (old and new)
 */
void MCMC::hueristic_split_group_sequential_new_group(std::vector<int> nodes_to_split, std::vector<int> groups_after_split) {
    if(verbose || debug){
        print("Performing sequential heuristic split with new group");
    }
    // Assign the first node to group 1, the second to group 2, then the rest to a new group
    flip_nodes({nodes_to_split[0]}, groups_after_split[0]);
    flip_nodes({nodes_to_split[1]}, groups_after_split[1]);
    std::vector<int> remaining_nodes_to_split(nodes_to_split.begin() + 2, nodes_to_split.end());
    if(remaining_nodes_to_split.empty()){
        return; // No remaining nodes to assign
    }
    if(verbose || debug){
        print("Assigned first two nodes to existing groups");
        print("Remaining nodes to split:");
        for(int node: remaining_nodes_to_split){
            print("Node: " + std::to_string(node));
        }
        print("Partition:");
        for(int i = 0; i < n; i++){
            printf("%d ", partition[i]);
        }
    }
    flip_nodes(remaining_nodes_to_split, num_groups); // New group
    // Sequentially assign the remaining nodes to either of the groups in a Gibbs-like manner
    gibbs_sweep(remaining_nodes_to_split, groups_after_split);
}

/**
 * hueristic_split_group_sequential_new_groups: Multi-group sequential split (stub)
 * 
 * TODO: Implementation pending
 * @param nodes_to_split: Nodes to be split
 * @param groups_after_split: Groups after the split (old and new)
 */
void MCMC::hueristic_split_group_sequential_new_groups(std::vector<int> nodes_to_split, std::vector<int> groups_after_split) {
    if(verbose || debug){
        print("Performing sequential heuristic split with individual new groups");
    }
    // Assign the first node to group 1, the second to group 2
    flip_nodes({nodes_to_split[0]}, groups_after_split[0]);
    flip_nodes({nodes_to_split[1]}, groups_after_split[1]);
    std::vector<int> remaining_nodes_to_split(nodes_to_split.begin() + 2, nodes_to_split.end());
    if(remaining_nodes_to_split.empty()){
        return; // No remaining nodes to assign
    }
    if(debug){
        print("Assigned first two nodes to existing groups");
        print("Remaining nodes to split:");
        for(int node: remaining_nodes_to_split){
            print("Node: " + std::to_string(node));
        }
        print("Partition:");
        for(int i = 0; i < n; i++){
            printf("%d ", partition[i]);
        }
    }
    // Assign the remaining nodes each to its own completely new group
    for(size_t i = 2; i < nodes_to_split.size(); i++){
        flip_nodes({nodes_to_split[i]}, num_groups); // New group
    }
    // Sequentially assign the remaining nodes to either of the groups in a Gibbs-like manner
    gibbs_sweep(remaining_nodes_to_split, groups_after_split);
}


/**
 * gibbs_split_refinement: Refine a split using local Gibbs sampling
 * 
 * Takes a proposed split (where nodes have been divided into initial and new groups)
 * and refines the partition by performing Gibbs sampling sweeps on the split nodes.
 * Each node is considered for reassignment to either the original group or the new group,
 * with assignment probabilities proportional to the likelihood of the network given the partition.
 * This helps avoid local optima in the split proposal and can improve the quality of the partition.
 * Returns probability of the split given the staging configuration.
 * 
 * @param nodes_to_sweep: Nodes to sweep over for reassignment
 * @param group_choices: Groups to consider for reassignment
 * TODO: Implement Gibbs sampling refinement for split moves
 */
double MCMC::gibbs_sweep(std::vector<int> nodes_to_sweep, std::vector<int> group_choices) {
    // TODO: Implement Gibbs sampling refinement
    // For each group in groups_to_split, iterate over all nodes in that group and:
    // 1. Temporarily remove the node from its current group
    // 2. Calculate the entropy delta for assigning it to each of the candidate groups
    // 3. Sample the new group assignment with probability proportional to exp(-beta * H_delta)
    // 4. Accept the assignment
    // Note that we should make sure that the group_choices are the last groups in the partition to avoid issues with relabeling during the sweep.
    // We also do not allow the Gibbs sweep to empty a group

    // TODO: come up with a more elegant way to deal with re-labeling during the sweep


    if(verbose || debug){
        print("Performing Gibbs sweep over nodes:");
        for(int node: nodes_to_sweep){
            print("Node to sweep: " + std::to_string(node));
        }
        print("Considering group choices:");
        for(int group: group_choices){
            print("Group choice: " + std::to_string(group));
        }
        print("Initial partition:");
        for(int i = 0; i < n; i++){
            printf("%d ", partition[i]);
        }
        print("");
    }

    double gibbs_H = 0.0; // Cumulative entropy of the assignments made during the sweep
    for(int node : nodes_to_sweep){
        // Need to be careful when this node is the last member of its group, as removing it will cause relabeling

        // Map between group choice and entropy delta
        std::unordered_map<int, double> group_log_weights;
        // Save the current group's entropy if it is among the choices
        if(std::find(group_choices.begin(), group_choices.end(), partition[node]) != group_choices.end()){
            group_log_weights[partition[node]] = -(beta*(H_A_G_b + H_parameters) + H_b);
        }
        // Flip this node to the other potential choices and record the resulting entropies
        for(int group_choice : group_choices){
            if(verbose || debug){
                print("Gibbs considering node " + std::to_string(node) + " for group " + std::to_string(group_choice) + " from current group " + std::to_string(partition[node]) + " with " + std::to_string(node_counts[partition[node]]) + " members");
                printf("Partition before flip attempt: ");
                for(int i = 0; i < n; i++){
                    printf("%d ", partition[i]);
                }
                print("");
            }
            if((node_counts[partition[node]] > 1 || std::find(group_choices.begin(), group_choices.end(), partition[node]) == group_choices.end()) && partition[node] != group_choice){ // Make sure that we cannot empty a group among the provided choices
                if(verbose || debug){
                    print("Partition before flip:");
                    for(int i = 0; i < n; i++){
                        printf("%d ", partition[i]);
                    }
                    print("");
                }
                flip_nodes({node}, group_choice);
                if(verbose || debug){
                    print("Partition after flip:");
                    for(int i = 0; i < n; i++){
                        printf("%d ", partition[i]);
                    }
                    print("");
                }
                group_log_weights[group_choice] = -(beta*(H_A_G_b + H_parameters) + H_b);
                // If this has led to the node being in a different group than group_choice, due to a group emptying relabeling, swap the group_entropy assignements
                if(partition[node] != group_choice){
                    // Throw error if partition[node] is not in group_log_weights
                    if(group_log_weights.find(partition[node]) == group_log_weights.end()){
                        print("Error: Group relabeling occurred during Gibbs sweep, not yet handled.");
                        exit(1);
                    }
                    std::swap(group_log_weights[partition[node]], group_log_weights[group_choice]);
                }
            }
        }
        if(verbose || debug){
            print("Gibbs log weights for node " + std::to_string(node) + ":");
            for(std::unordered_map<int, double>::const_iterator it = group_log_weights.begin(); it != group_log_weights.end(); ++it){
                int group = it->first;
                double log_weight = it->second;
                print("Group " + std::to_string(group) + ": log weight " + std::to_string(log_weight));
            }
        }
        // Unpack sample and entropy from log_weights map
        std::pair<int, double> sampled_gibbs = sample_log_weights_map(group_log_weights);
        int new_group = sampled_gibbs.first;
        double entropy = sampled_gibbs.second;
        gibbs_H += entropy;
        if(verbose || debug){
            print("Gibbs sampled node " + std::to_string(node) + " to group " + std::to_string(new_group) + " with entropy " + std::to_string(entropy) + " new cumulative gibbs_H: " + std::to_string(gibbs_H));
        }
        // Flip node to sampled group if not already there
        if(partition[node] != new_group){
            flip_nodes({node}, new_group);
        }
    }
    return gibbs_H;
}

/**
 * H_reverse_merge: Calculate entropy change from reverse merge new_group -> old_group
 * 
 * TODO: Implementation pending
 * @param old_group: Source group
 * @param new_group: Target group
 * @return Entropy change from merge
 */
double MCMC::H_reverse_merge(int old_group, int new_group) {

    // TODO: Add the probability of merging along either path, new_group -> old_group or old_group -> new_group (should probably also change the names to be more symemtric in this case too)

    if(verbose || debug){
        print("Calculating reverse merge entropy for merging group " + std::to_string(new_group) + " into group " + std::to_string(old_group));
        print("Partition before reverse merge:");
        for(int i = 0; i < n; i++){
            printf("%d ", partition[i]);
        }
        print("");
    }

    if(new_group != num_groups - 1){
        print("Error: H_reverse_merge currently only supports merging from the last group.");
        exit(1);
    }

    std::vector<int> nodes_new_group; // Store the nodes in the original new_group to restore at the end
    for(int node: group_members[new_group]){
        nodes_new_group.push_back(node);
    }

    double H_merge = std::log(num_groups); // Entropy of choosing new_group to merge into old_group

    std::unordered_map<int, double> group_log_weights;
    for(int r = 0; r < num_groups; r++){
        if(r != new_group){
            // Temporarily merge new_group into group r
            flip_nodes(nodes_new_group, r);
            group_log_weights[r] = -(beta*(H_A_G_b + H_parameters) + H_b);
        }
    }
    // Among these options compute the entropy of having chosen new_group to merge into
    std::vector<double> log_weights;
    for(std::unordered_map<int, double>::const_iterator it = group_log_weights.begin(); it != group_log_weights.end(); ++it){
        log_weights.push_back(it->second);
    }
    double log_total_weight = log_sum_exp(log_weights);
    H_merge += log_total_weight - group_log_weights[old_group];
    // Restore original partition
    flip_nodes(nodes_new_group, new_group);

    return H_merge;
}

/**
 * H_reverse_split: Calculate entropy change from reverse split moving old_group_indices from new_group to old_group
 * 
 * TODO: Implementation pending
 * @param split_group: Group being split in reverse
 * @param split_indices: Nodes that were split off
 * @return Entropy change from split
 */
double MCMC::H_reverse_split(int split_group, std::vector<int> split_nodes) {
    if(verbose || debug){
        print("Calculating reverse split entropy for group " + std::to_string(split_group));
        print("Partition before reverse split:");
        for(int i = 0; i < n; i++){
            printf("%d ", partition[i]);
        }
        print("");
        print("Nodes being moved from group " + std::to_string(split_group) + " to new group:");
        for(int node: split_nodes){
            print("Node: " + std::to_string(node));
        }
    }
    
    // Ensure that new_group is the last group
    if(split_group != num_groups - 1){
        print("Error: H_reverse_split currently only supports splitting from the last group.");
        exit(1);
    }

    double H_split_given_staging = std::log(num_groups); // Entropy of choosing this group to split

    int group_to_split = split_group;
    // Store the nodes that will be split
    std::vector<int> nodes_to_split;
    for(int node: group_members[group_to_split]){
        nodes_to_split.push_back(node);
    }

    // Make an initial split using a hueristic, creates new group at last index
    hueristic_split_group({group_to_split});

    if(verbose || debug){
        print("Partition after heuristic split for reverse split entropy:");
        for(int i = 0; i < n; i++){
            printf("%d ", partition[i]);
        }
        print("");
    }

    std::vector<int> group_choices = {group_to_split, num_groups - 1};

    for(int sweep = 0; sweep < GIBBS_REFINEMENT_SWEEPS; sweep++){
        if(verbose || debug){
            print("Gibbs sweep " + std::to_string(sweep) + " for reverse split");
            print("Group choices:");
            for(int group: group_choices){
                print("Group choice: " + std::to_string(group));
            }
        }
        double _ = gibbs_sweep(nodes_to_split, group_choices); // Don't need split probability here
    }

    // Get the entropy of making the particular specified split
    double gibbs_H = 0.0; // Cumulative entropy of the assignments made during the sweep
    for(int node : nodes_to_split){
        int specified_group;
        if(std::find(split_nodes.begin(), split_nodes.end(), node) != split_nodes.end()){
            specified_group = num_groups - 1;
        }else{
            specified_group = group_to_split;
        }

        // Need to be careful when this node is the last member of its group, as removing it will cause relabeling

        // Map between group choice and entropy delta
        std::unordered_map<int, double> group_log_weights;
        // Save the current group's entropy
        group_log_weights[partition[node]] = -(beta*(H_A_G_b + H_parameters) + H_b);
        // Flip this node to the other potential choices and record the resulting entropies
        for(int group_choice : group_choices){
            if(partition[node] != group_choice){
                flip_nodes({node}, group_choice);
                group_log_weights[group_choice] = -(beta*(H_A_G_b + H_parameters) + H_b);
                // If this has led to the node being in a different group than group_choice, due to a group emptying relabeling, swap the group_entropy assignements
                if(partition[node] != group_choice){
                    // Throw error if partition[node] is not in group_log_weights
                    if(group_log_weights.find(partition[node]) == group_log_weights.end()){
                        print("Error: Group relabeling occurred during Gibbs sweep, not yet handled.");
                        exit(1);
                    }
                    std::swap(group_log_weights[partition[node]], group_log_weights[group_choice]);
                }
            }
        }
        // Find probability of the actual split choice
        std::vector<double> log_weights;
        for(int group_choice : group_choices){
            log_weights.push_back(group_log_weights[group_choice]);
        }
        double log_total_weight = log_sum_exp(log_weights);
        double entropy = log_total_weight - group_log_weights[specified_group];

        gibbs_H += entropy;
        // Flip node to specified group if not already there
        if(partition[node] != specified_group){
            flip_nodes({node}, specified_group);
        }
    }
    H_split_given_staging += gibbs_H;

    // Move all the nodes back to new_group. Function only supports flipping nodes of a single group, so select the nodes in the last group
    std::vector<int> nodes_in_new_group;
    for(int node: group_members[num_groups - 1]){
        nodes_in_new_group.push_back(node);
    }
    flip_nodes(nodes_in_new_group, split_group);

    return H_split_given_staging;
}