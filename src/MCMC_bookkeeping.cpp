#define _USE_MATH_DEFINES
#include "MCMC_core.h"
#include "globals.h"
#include "helpers.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <vector>


/**
 * check_state_bookkeeping: Validate and recompute all bookkeeping variables
 * 
 * This function serves two purposes:
 * 1. Validation: Check that cached bookkeeping matches actual partition state
 * 2. Initialization/Repair: Recompute bookkeeping from scratch if needed
 * 
 * It verifies and updates:
 * - num_groups: Number of groups in partition
 * - node_counts[r]: Number of nodes in each group
 * - group_members[r]: Vector of node IDs in each group
 * - node_indices_in_groups[r][node]: Position of node in group_members[r]
 * - edge_counts[r][s]: Total edge weight between groups r and s
 * - edge_counts_sum[r]: Total degree of group r
 * - Normalization terms: node_counts_norm_* for efficient entropy calculation
 * - All entropy components: H_b, H_k, H_M, H_A_G_k_M_b, H_A_G_b, H_A_b
 * - Density parameters: density_in, density_out
 * 
 * @param throw_error: If true, exit on inconsistency; if false, repair silently
 */
void MCMC::check_state_bookkeeping(bool throw_error) {
    if(verbose){
        print("Checking state bookkeeping consistency, current partition:");
        for (int i = 0; i < partition.size(); i++) {
            printf("%d ", partition[i]);
        }
        print("");
    }

    // === Validate num_groups ===
    int num_groups_actual = *std::max_element(partition.begin(), partition.end()) + 1;
    num_groups = value_check(num_groups_actual, num_groups, NUMERICAL_CHECK_EPSILON, throw_error, "num_groups");

    // === Recompute node_counts, group_members, node_indices_in_groups from partition ===
    std::vector<int> node_counts_actual(num_groups, 0);
    std::vector<std::vector<int>> group_members_actual(num_groups);
    std::vector<std::unordered_map<int, size_t>> node_indices_in_groups_actual(num_groups);
    
    // Initialize structures
    for (int r = 0; r < num_groups; r++) {
        node_counts_actual[r] = 0;
        group_members_actual[r] = std::vector<int>();
        node_indices_in_groups_actual[r] = std::unordered_map<int, size_t>();
    }
    
    // Populate from partition
    for (int i = 0; i < n; i++) {
        int group = partition[i];
        node_counts_actual[group]++;
        group_members_actual[group].push_back(i);
        // Store index position for O(1) removal later
        node_indices_in_groups_actual[group][i] = group_members_actual[group].size() - 1;
    }
    
    // === Validate or update node_counts ===
    for (std::size_t r = 0; r < static_cast<std::size_t>(num_groups); r++) {
        if (r < node_counts.size()) {
            if (node_counts_actual[r] != node_counts[r] && throw_error) {
                std::cerr << "Error: node_counts does not match the actual value.\n";
                std::cerr << "node_counts[" << r << "]: " << node_counts[r] << ", node_counts_actual[" << r << "]: " << node_counts_actual[r] << "\n";
                exit(1);
            }
            node_counts[r] = node_counts_actual[r];
        } else {
            if (throw_error) {
                std::cerr << "Error: node_counts is not of correct length.\n";
                exit(1);
            }
            node_counts.push_back(node_counts_actual[r]);
        }
        
        // === Validate or update group_members ===
        if (r < group_members_actual.size()) {
            for (int i_actual : group_members_actual[r]) {
                if (throw_error && std::find(group_members[r].begin(), group_members[r].end(), i_actual) == group_members[r].end()) {
                    std::cout << "Error: student " << i_actual << " from actual group " << r << " not found in bookkeeping group_members[" << r << "]\n";
                    exit(1);
                }
            }
            for (int i : group_members[r]) {
                if (throw_error && std::find(group_members_actual[r].begin(), group_members_actual[r].end(), i) == group_members_actual[r].end()) {
                    std::cout << "Error: student " << i << " from bookkeeping group " << r << " not found in the actual group_members[" << r << "]\n";
                    exit(1);
                }
            }
            // Update cached bookkeeping with actual values
            group_members[r] = group_members_actual[r];
            node_indices_in_groups[r] = node_indices_in_groups_actual[r];
        } else {
            if (throw_error) {
                std::cerr << "Error: group_members is not of correct length.\n";
                exit(1);
            }
            group_members.push_back(group_members_actual[r]);
            node_indices_in_groups.push_back(node_indices_in_groups_actual[r]);
        }
    }

    // === Recompute normalization terms ===
    double node_counts_norm_2_actual = 0;
    for (int r = 0; r < num_groups; r++) {
        node_counts_norm_2_actual += std::pow(node_counts[r], 2);
    }
    node_counts_norm_2 = value_check(node_counts_norm_2_actual, node_counts_norm_2, NUMERICAL_CHECK_EPSILON, throw_error, "node_counts_norm_2");

    // === Recompute edge_counts matrix ===
    // In two-layer or weighted mode: edge_counts = binary (1 per neighbour, weights ignored for counts).
    // In single-layer unweighted mode: edge_counts = sum of rounded integer weights.
    // weighted_SBM additionally fills weight_counts with real-valued weight sums.
    std::vector<std::vector<int>> edge_counts_actual(num_groups, std::vector<int>(num_groups, 0));
    std::vector<int> edge_counts_sum_actual(num_groups, 0);
    std::vector<std::vector<double>> weight_counts_actual(num_groups, std::vector<double>(num_groups, 0.0));
    std::vector<double> weight_counts_sum_actual(num_groups, 0.0);
    for (int i = 0; i < n; i++) {
        int group_i = partition[i];
        for (const auto& edge : neighbors_com[i]) {
            int j = edge.first;
            double weight = edge.second;
            int group_j = partition[j];
            if (weighted_SBM || two_layer_mode) {
                edge_counts_actual[group_i][group_j] += 1;
            } else {
                edge_counts_actual[group_i][group_j] += static_cast<int>(weight);
            }
            if (weighted_SBM) {
                weight_counts_actual[group_i][group_j] += weight;
            }
        }
    }
    for (int r = 0; r < num_groups; r++) {
        for (int s = 0; s < num_groups; s++) {
            edge_counts_sum_actual[r] += edge_counts_actual[r][s];
        }
    }
    for (std::size_t r = 0; r < static_cast<std::size_t>(num_groups); r++) {
        for (std::size_t s = 0; s < static_cast<std::size_t>(num_groups); s++) {
            if (r < edge_counts.size() && s < edge_counts[r].size()) {
                edge_counts[r][s] = value_check(edge_counts_actual[r][s], edge_counts[r][s], NUMERICAL_CHECK_EPSILON, throw_error, "edge_counts[" + std::to_string(r) + "][" + std::to_string(s) + "]");
            } else {
                if (throw_error) {
                    std::cerr << "Error: edge_counts is not of correct length.\n";
                    exit(1);
                }
                edge_counts[r].push_back(edge_counts_actual[r][s]);
            }
        }
        if (r < edge_counts_sum.size()) {
            edge_counts_sum[r] = value_check(edge_counts_sum_actual[r], edge_counts_sum[r], NUMERICAL_CHECK_EPSILON, throw_error, "edge_counts_sum[" + std::to_string(r) + "]");
        } else {
            if (throw_error) {
                std::cerr << "Error: edge_counts_sum is not of correct length.\n";
                exit(1);
            }
            edge_counts_sum.push_back(edge_counts_sum_actual[r]);
        }
    }
    // === Validate edge_counts dimensions ===
    if (edge_counts.size() != static_cast<std::size_t>(num_groups) || edge_counts[0].size() != static_cast<std::size_t>(num_groups)) {
        if (throw_error) {
            std::cerr << "Error: edge_counts dimensions do not match num_groups.\n";
            exit(1);
        } else {
            // Repair: resize to match num_groups
            edge_counts.resize(num_groups);
            for (auto& row : edge_counts) {
                row.resize(num_groups, 0);
            }
        }
    }

    // === Rebuild weight_counts (weighted mode only) ===
    if (weighted_SBM) {
        weight_counts = weight_counts_actual;
        weight_counts_sum = weight_counts_sum_actual;
        for (int r = 0; r < num_groups; r++) {
            for (int s = 0; s < num_groups; s++) {
                weight_counts_sum[r] += weight_counts[r][s];
            }
        }
    }

    // === Initialize or validate density/weight parameters ===
    // On first call, density_in < 0 indicates uninitialized state
    if (density_in < 0.0) {
        // Initialize to overall graph density
        density_in = static_cast<double>(m) / (static_cast<double>(n * n) / 2.0);
        density_out = density_in;
    }
    
    // If not assortative, check density_out = density_in
    if (!assortative) {
        if(num_groups > 1){
            density_out = value_check(density_in, density_out, NUMERICAL_CHECK_EPSILON, throw_error, "density_out");
        }
    }


    // === Validate concentration parameters ===
    // Convert variation parameters to concentration (for negative binomial)
    double concentration_in_actual = variation_to_concentration(variation_in);
    double concentration_out_actual = variation_to_concentration(variation_out);
    double degree_correction_concentration_actual = variation_to_concentration(degree_correction);
    
    concentration_in = value_check(concentration_in_actual, concentration_in, NUMERICAL_CHECK_EPSILON, throw_error, "concentration_in");
    concentration_out = value_check(concentration_out_actual, concentration_out, NUMERICAL_CHECK_EPSILON, throw_error, "concentration_out");
    degree_correction_concentration = value_check(degree_correction_concentration_actual, degree_correction_concentration, NUMERICAL_CHECK_EPSILON, throw_error, "degree_correction_concentration");
    
    // === Validate scaled_scores and group_scaled_score_sums ===
    std::vector<double> scaled_scores_actual(n, 0.0);
    for (int i = 0; i < n; i++) {
        scaled_scores_actual[i] = scores[i] / individual_depth;
    }
    // Initialize scaled_scores if empty
    if (scaled_scores.size() != static_cast<std::size_t>(n)) {
        scaled_scores = scaled_scores_actual;
    }
    for (int i = 0; i < n; i++) {
        scaled_scores[i] = value_check(scaled_scores_actual[i], scaled_scores[i], NUMERICAL_CHECK_EPSILON, throw_error, "scaled_scores[" + std::to_string(i) + "]");
    }

    std::vector<double> group_scaled_score_sums_actual(num_groups, 0.0);
    for (int i = 0; i < n; i++) {
        int group = partition[i];
        group_scaled_score_sums_actual[group] += scaled_scores[i];
    }
    for (std::size_t r = 0; r < static_cast<std::size_t>(num_groups); r++) {
        if (r < group_scaled_score_sums.size()) {
            if(throw_error && (std::isnan(group_scaled_score_sums_actual[r]) ||  std::isnan(group_scaled_score_sums[r]) || std::abs(group_scaled_score_sums_actual[r] - group_scaled_score_sums[r]) > NUMERICAL_CHECK_EPSILON)){
                std::cerr << "Error: group_scaled_score_sums does not match the actual value for group " << r << ".\n";
                std::cerr << "group_scaled_score_sums[" << r << "]: " << group_scaled_score_sums[r] << ", group_scaled_score_sums_actual[" << r << "]: " << group_scaled_score_sums_actual[r] << "\n";
                // Print out the node_counts and group members for debugging
                std::cerr << "node_counts[" << r << "]: " << node_counts[r] << "\n";
                std::cerr << "group_members[" << r << "]: "; 
                for (int node : group_members[r]) {
                    std::cerr << node << " ";
                }                
                std::cerr << "\n";
                // Print the scores of the nodes in the group for further debugging
                std::cerr << "scores of nodes in group " << r << ": ";
                for (int node : group_members[r]) {
                    std::cerr << "node " << node << ": score " << scores[node] << ", scaled_score " << scaled_scores[node] << "\n";
                }
                // Print the individual_depth
                std::cerr << "individual_depth: " << individual_depth << "\n";
                exit(1);
            }
            group_scaled_score_sums[r] = value_check(group_scaled_score_sums_actual[r], group_scaled_score_sums[r], NUMERICAL_CHECK_EPSILON, throw_error, "group_scaled_score_sums[" + std::to_string(r) + "]");
        } else {
            if (throw_error) {
                std::cerr << "Error: group_scaled_score_sums is not of correct length.\n";
                exit(1);
            }
            group_scaled_score_sums.push_back(group_scaled_score_sums_actual[r]);
        }
    }


    // === Compute H_b: Partition entropy ===
    // H_b = H(b|n,q) = log(n) + log(C(n-1, q-1)) + log(n!) - sum_r log(n[r]!)
    // - log(C(n-1, q-1)): Number of ways to partition n nodes into q groups
    // - log(n!) - sum_r log(n[r]!): Multinomial coefficient for group sizes
    double H_b_actual = std::log(n);
    H_b_actual += log_binom(n - 1, num_groups - 1);  // H(nb|q): Number of partitions
    H_b_actual += log_factorial(n);  // H(b|nb): Group assignment entropy
    for (int r = 0; r < num_groups; r++) {
        H_b_actual -= log_factorial(node_counts[r]);  // Normalize by group size permutations
    }
    H_b = value_check(H_b_actual, H_b, NUMERICAL_CHECK_EPSILON, throw_error, "H_b");

    // === Compute H_M: Edge count distribution entropy ===
    // Sum of negative binomial entropies for within/between-group edge counts
    double H_M_actual = calculate_H_M();
    H_M = value_check(H_M_actual, H_M, NUMERICAL_CHECK_EPSILON, throw_error, "H_M");

    // === Compute H_k: Degree sequence entropy ===
    // Entropy of degree distribution given group structure
    double H_k_actual = calculate_H_k();
    H_k = value_check(H_k_actual, H_k, NUMERICAL_CHECK_EPSILON, throw_error, "H_k");

    // === Compute H_A_G_k_M_b: Adjacency matrix entropy ===
    // Entropy of specific edge placements given edge counts
    double H_A_G_k_M_b_actual = calculate_H_A_G_k_M_b();
    H_A_G_k_M_b = value_check(H_A_G_k_M_b_actual, H_A_G_k_M_b, NUMERICAL_CHECK_EPSILON, throw_error, "H_A_G_k_M_b");

    // === Compute total entropies ===
    // In weighted mode: H_W = log P(W|A,μ̂,b) is the log marginal likelihood
    // (positive = better fit). It REDUCES description length, so subtract it.
    if (weighted_SBM) {
        double H_W_actual = calculate_H_W();
        H_W = value_check(H_W_actual, H_W, NUMERICAL_CHECK_EPSILON, throw_error, "H_W");
    }
    double H_A_G_b_actual = H_A_G_k_M_b + H_k + H_M - H_W;
    H_A_G_b = value_check(H_A_G_b_actual, H_A_G_b, NUMERICAL_CHECK_EPSILON, throw_error, "H_A_G_b");

    // H(A,b) = H(A,G,b) + H(b)
    double H_A_b_actual = H_A_G_b + H_b;
    H_A_b = value_check(H_A_b_actual, H_A_b, NUMERICAL_CHECK_EPSILON, throw_error, "H_A_b");

    // === Compute hierarchy entropies ===
    // Entropy of the directed edges given the undirected edges and scores, Bradley-Terry
    double H_Adir_G_A_s_nu_actual = calculate_H_Adir_G_A_s_nu();
    H_Adir_G_A_s_nu = value_check(H_Adir_G_A_s_nu_actual, H_Adir_G_A_s_nu, NUMERICAL_CHECK_EPSILON, throw_error, "H_Adir_G_A_s_nu");
    // Entropy of the scores given the partition and depth parameters
    double H_s_G_b_depths_actual = calculate_H_s_G_b_depths();
    if(verbose){
        print("H_s_G_b_depths: " + std::to_string(H_s_G_b_depths));
    }
    H_s_G_b_depths = value_check(H_s_G_b_depths_actual, H_s_G_b_depths, NUMERICAL_CHECK_EPSILON, throw_error, "H_s_G_b_depths");
    
    if (verbose) {
        if(throw_error){
            print("Bookkeeping check passed successfully.");
        }else{
            print("Bookkeeping initialized.");
        }
    }
}

/**
 * calculate_H_k: Compute degree sequence entropy H(k|b)
 * 
 * Models the degree distribution within each group as a Dirichlet-multinomial.
 * The total degree edge_counts_sum[r] is distributed among node_counts[r] nodes
 * in group r with concentration parameter degree_correction_concentration.
 * 
 * Higher degree_correction allows more degree heterogeneity within groups.
 * 
 * @return H_k entropy value
 */
double MCMC::calculate_H_k() {
    if(verbose){
        print("Calculating H_k with current edge counts and parameters:");
        print("degree_correction_concentration: " + std::to_string(degree_correction_concentration));
    }
    double H_k_actual = 0.0;

    if(degree_correction_concentration < MAX_CONCENTRATION){
        // For each group: multinomial coefficient for degree distribution
        for(int r = 0; r < num_groups; r++) {
            H_k_actual += log_binom(edge_counts_sum[r] + node_counts[r]*degree_correction_concentration - 1, 
                                node_counts[r]*degree_correction_concentration - 1);
        }
        
        // Subtract normalization for each node's degree
        for(int i = 0; i < n; i++) {
            H_k_actual -= log_binom(node_degrees[i] + degree_correction_concentration - 1, 
                                degree_correction_concentration - 1);
        } 
    }else{
        // For very large concentration, approximate with multinomial
        for(int r = 0; r < num_groups; r++) {
            H_k_actual -= log_factorial(edge_counts_sum[r]);
            H_k_actual += edge_counts_sum[r]*std::log(node_counts[r]);
        }
        for(int i = 0; i < n; i++) {
            H_k_actual += log_factorial(node_degrees[i]);
        }
    }
    if(verbose){
        printf("Degrees k_i:\n");
        for(int i = 0; i < n; i++){
            printf("%d ", node_degrees[i]);
        }
        print("");
        printf("edge_counts_sum[r]:\n");
        for(int r = 0; r < num_groups; r++){
            printf("%d ", edge_counts_sum[r]);
        }
        print("");
    }
    
    return H_k_actual;
}

/**
 * calculate_H_M: Compute edge count distribution entropy H(M|b)
 * 
 * Models edge counts M_{rs} between groups as negative binomial distributions.
 * 
 * Expected edge counts scale with group sizes:
 * - Within-group: E[M_{rr}/2] = density_in * n[r]^2 / 2
 * - Between-group: E[M_{rs}] = density_out * n[r] * n[s]
 * 
 * Concentration parameters control overdispersion:
 * - concentration=1: Geometric distribution (high variance)
 * - concentration→∞: Poisson distribution (variance=mean)
 * 
 * @return H_M entropy value
 */
double MCMC::calculate_H_M() {
    if(verbose){
        print("Calculating H_M with current edge counts and parameters:");
        print("density_in: " + std::to_string(density_in) + ", density_out: " + std::to_string(density_out));
        print("concentration_in: " + std::to_string(concentration_in) + ", concentration_out: " + std::to_string(concentration_out));
    }
    
    double H_M_actual = 0.0; 
    
    // Sum over unique group pairs (r <= s to avoid double counting)
    for(int r = 0; r < num_groups; r++){
        for(int s = r; s < num_groups; s++){
            if(r == s){
                // Within-group edges (undirected, so divide by 2)
                H_M_actual -= log_neg_binom(edge_counts[r][r]/2.0,
                                           density_in*node_counts[r]*node_counts[r]/2.0,
                                           concentration_in);
            }else{
                // Between-group edges
                H_M_actual -= log_neg_binom(edge_counts[r][s],
                                           density_out*node_counts[r]*node_counts[s],
                                           concentration_out);
            }
        }
    }
    return H_M_actual;
}

/**
 * calculate_H_A_G_k_M_b: Compute adjacency matrix entropy H(A|G,k,M,b)
 * 
 * Given edge counts M_{rs} and degree sequence k, this computes the entropy
 * of the specific edge placements in the adjacency matrix A.
 * 
 * Uses multinomial coefficients to count valid adjacency matrices:
 * - For each node i: distribute k_i edges among possible targets
 * - For each group pair (r,s): ensure total edges equals M_{rs}
 * 
 * Self-edges use double factorial (!!), regular edges use factorial (!).
 * 
 * @return H_A_G_k_M_b entropy value
 */
double MCMC::calculate_H_A_G_k_M_b() {
    double H_A_G_k_M_b_actual = 0.0;
    
    // Sum over all edges (weighted by multiplicity).
    // In weighted mode, each edge is binary (A_ij=1), so log(A_ij!)=0 for all edges;
    // the per-edge loop is skipped entirely.
    if (!weighted_SBM) {
        for(int i = 0; i < n; i++){
            for(const auto& edge : neighbors_com[i]){
                int j = edge.first;
                double A_ij = edge.second;
                if(i == j){
                    H_A_G_k_M_b_actual += log_factorial2(A_ij);
                }else{
                    H_A_G_k_M_b_actual += log_factorial(A_ij);
                }
            }
        }
    }
    for(int i = 0; i < n; i++){
        // Normalize by degree factorial
        H_A_G_k_M_b_actual -= log_factorial(node_degrees[i]);
    }
    
    // Subtract group-level edge count factorials (unique pairs only)
    for(int r = 0; r < num_groups; r++){
        for(int s = r; s < num_groups; s++){
            if(r == s){
                // Within-group: double factorial for undirected edges
                H_A_G_k_M_b_actual -= log_factorial2(edge_counts[r][r]);
            }else{
                // Between-group: regular factorial
                H_A_G_k_M_b_actual -= log_factorial(edge_counts[r][s]);
            }
        }
        // Add normalization for total group degree
        H_A_G_k_M_b_actual += log_factorial(edge_counts_sum[r]);
    }
    return H_A_G_k_M_b_actual;
}

/**
 * calculate_H_W: Weighted SBM entropy H(W|A,μ̂,b) + H(μ̂)
 *
 * Implements the uniform-simplex prior on per-block-pair weight distributions.
 * Only valid (and called) when weighted_SBM == true.
 *
 * Formula:
 *   H_W = (K-1)·log(w) − log((K-1)!)
 *       + Σ_{r<s, M_rs>0} [(M_rs−1)·log(μ_rs) − log((M_rs−1)!)]
 *       + Σ_{r,  M_rr>0} [(M_rr/2−1)·log(μ_rr/2) − log((M_rr/2−1)!)]
 *
 * where K = q(q+1)/2, w = w_total (constant), M_rs = edge_counts[r][s] (binary),
 * μ_rs = weight_counts[r][s].  Diagonal entries are stored doubled (2×M_rr, 2×μ_rr).
 */
double MCMC::calculate_H_W() {
    int K = num_groups * (num_groups + 1) / 2;
    double h = (K - 1) * std::log(w_total) - log_factorial(K - 1);

    for (int r = 0; r < num_groups; r++) {
        // Diagonal: stored as 2× (double-counted by symmetric neighbor loop) → halve
        int    m_rr  = edge_counts[r][r]   / 2;
        double mu_rr = weight_counts[r][r] / 2.0;
        if (m_rr > 0)
            h += (m_rr - 1) * std::log(mu_rr) - log_factorial(m_rr - 1);

        for (int s = r + 1; s < num_groups; s++) {
            int    m_rs  = edge_counts[r][s];
            double mu_rs = weight_counts[r][s];
            if (m_rs > 0)
                h += (m_rs - 1) * std::log(mu_rs) - log_factorial(m_rs - 1);
        }
    }
    return h;
}

/**
 * calculate_H_Adir_G_A_s_nu: Compute directed edge entropy H(Adir|A,s,nu) using the Bradley-Terry model
 *
 * P(Adir|A,s) = \sum_{(i,j)} A_ij!/(Adir_ij!Adir_ji!) p_ij^Adir_ij p_ji^Adir_ji
 */
double MCMC::calculate_H_Adir_G_A_s_nu() {
    double H_Adir_G_A_s_nu_actual = 0.0;
    // Iterate over all undirected edges
    for(int i = 0; i < n; i++){
        for(const auto& edge : wins[i]){
            int j = edge.first;
            double Awins_ij = edge.second;
            double log_p_win_ij = scores[i] - log_sum_exp(std::vector<double>{scores[i], scores[j], std::log(ties_parameter) + scores[i]/2.0 + scores[j]/2.0}); // P(i -> j|i - j)
            H_Adir_G_A_s_nu_actual += log_factorial(Awins_ij) - Awins_ij * log_p_win_ij;
            if(verbose){
                print("Edge (" + std::to_string(i) + " -> " + std::to_string(j) + "): Awins_ij = " + std::to_string(Awins_ij) + ", log_p_win_ij = " + std::to_string(log_p_win_ij) + ", contribution = " + std::to_string(log_factorial(Awins_ij) - Awins_ij * log_p_win_ij));
            }
        }
        // for(const auto& [j, Alosses_ij] : losses[i]){
        //     double log_p_loss_ij = scores[j] - log_sum_exp(std::vector<double>{scores[i], scores[j], std::log(ties_parameter)}); // P(i <- j|i - j)
        //     H_Adir_G_A_s_nu_actual += log_factorial(Alosses_ij) - Alosses_ij * log_p_loss_ij;
        //     if(verbose){
        //         print("Edge (" + std::to_string(i) + " <- " + std::to_string(j) + "): Alosses_ij = " + std::to_string(Alosses_ij) + ", log_p_loss_ij = " + std::to_string(log_p_loss_ij) + ", contribution = " + std::to_string(log_factorial(Alosses_ij) - Alosses_ij * log_p_loss_ij));
        //     }
        // }
        for(const auto& edge : ties[i]){
            int j = edge.first;
            double Aties_ij = edge.second;
            if(i < j){
                double log_p_tie_ij = std::log(ties_parameter) + scores[i]/2.0 + scores[j]/2.0 - log_sum_exp(std::vector<double>{scores[i], scores[j], std::log(ties_parameter) + scores[i]/2.0 + scores[j]/2.0}); // P(i <-> j|i - j): ν·exp((si+sj)/2) / Z
                H_Adir_G_A_s_nu_actual += log_factorial(Aties_ij) - Aties_ij * log_p_tie_ij;
                if(verbose){
                    print("Edge (" + std::to_string(i) + " <-> " + std::to_string(j) + "): Aties_ij = " + std::to_string(Aties_ij) + ", log_p_tie_ij = " + std::to_string(log_p_tie_ij) + ", contribution = " + std::to_string(log_factorial(Aties_ij) - Aties_ij * log_p_tie_ij));
                }
            }
        }
        for(const auto& edge : neighbors[i]){
            int j = edge.first;
            double A_ij = edge.second;
            if(i < j){ // Ensure each undirected edge is counted once
                H_Adir_G_A_s_nu_actual -= log_factorial(A_ij);
                if(verbose){
                    print("Edge (" + std::to_string(i) + " -- " + std::to_string(j) + "): A_ij = " + std::to_string(A_ij) + ", contribution = " + std::to_string(-log_factorial(A_ij)));
                }
            }
        }
    }
    if(verbose){
        print("scores:");
        for(int i = 0; i < n; i++){
            print("score[" + std::to_string(i) + "]: " + std::to_string(scores[i]));
        }
        print("ties_parameter: " + std::to_string(ties_parameter));
        print("H_Adir_G_A_s_nu_actual: " + std::to_string(H_Adir_G_A_s_nu_actual));
    }
    return H_Adir_G_A_s_nu_actual;
}

// 123456789101112

/*
* calculate_H_s_G_b_depths: Compute score entropy H(s|b,depths)
* 
* Models scores within each group as Gaussian with variance depending on depth parameters.
* 
* @return H_s_G_b_depths entropy value
*/
double MCMC::calculate_H_s_G_b_depths() {
    if(verbose){
        print("scaled_scores:");
        for(int i = 0; i < n; i++){
            print("scaled_score[" + std::to_string(i) + "]: " + std::to_string(scaled_scores[i]));
        }
        print("individual_depth: " + std::to_string(individual_depth));
        print("group_depth: " + std::to_string(group_depth));
    }
    double H_s_G_b_depths_actual = 0.0;
    for(int i = 0; i < n; i++){
        H_s_G_b_depths_actual += std::log(std::sqrt(M_PI)) + scaled_scores[i]*scaled_scores[i];
        if(verbose){
            print("scaled_score[" + std::to_string(i) + "]: " + std::to_string(scaled_scores[i]));
            print("Contribution to H_s_G_b_depths from node " + std::to_string(i) + ": " + std::to_string(std::log(std::sqrt(M_PI)) + scaled_scores[i]*scaled_scores[i]));
        }
    }
    for(int r = 0; r < num_groups; r++){
        double f_normalization = 1/std::sqrt(1 + node_counts[r]*group_depth*group_depth/(individual_depth*individual_depth));
        H_s_G_b_depths_actual -= std::log(f_normalization) + f_normalization*f_normalization*group_depth*group_depth*group_scaled_score_sums[r]*group_scaled_score_sums[r]/(individual_depth*individual_depth);
        if(verbose){
            print("group " + std::to_string(r) + " normalization factor: " + std::to_string(f_normalization));
            print("Contribution to H_s_G_b_depths from group " + std::to_string(r) + ": " + std::to_string(std::log(f_normalization) + f_normalization*f_normalization*group_depth*group_depth*group_scaled_score_sums[r]*group_scaled_score_sums[r]/(individual_depth*individual_depth)));
        }
    }
    if(verbose){
        print("individual_depth: " + std::to_string(individual_depth));
        print("group_depth: " + std::to_string(group_depth));
        print("score of node 0: " + std::to_string(scores[0]));
        print("score of node 1: " + std::to_string(scores[1]));
        print("H_s_G_b_depths_actual: " + std::to_string(H_s_G_b_depths_actual));
    }
    return H_s_G_b_depths_actual;
}
   

/**
 * to_string_verbose: Generate detailed human-readable state summary (stub)
 * 
 * TODO: Implementation pending
 * @return Verbose string representation of MCMC state
 */
std::string MCMC::to_string_verbose() const {
    return {};
}

/**
 * to_string_csv: Generate CSV row with state variables (stub)
 * 
 * TODO: Implementation pending
 * @return CSV-formatted string of MCMC state
 */
std::string MCMC::to_string_csv() const {
    return {};
}
