#define _USE_MATH_DEFINES
#include "MCMC_core.h"
#include "globals.h"
#include "helpers.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
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
 * - edge_counts[r][s]: edge count between groups r and s (diagonal doubled)
 * - edge_counts_sum[r]: Total degree of group r
 * - weighted_SBM: weight_counts, weight_counts_sum and nonzero_pairs
 * - group_score_sums
 * - All entropy components: H_b, H_k, H_M, H_A_G_k_M_b, H_W, H_A_G_b, H_A_b,
 *   H_Adir_G_A_s_nu, H_s_G_b_depths
 * - Density parameters: density_in, density_out
 *
 * Tolerance: value_check, |cached - actual| <= tol + NUMERICAL_CHECK_REL_TOL |actual|.
 *
 * @param throw_error: If true, throw std::runtime_error on inconsistency; if
 *                     false, repair silently
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
                throw std::runtime_error("node_counts[" + std::to_string(r) + "] mismatch: cached=" + std::to_string(node_counts[r]) + ", actual=" + std::to_string(node_counts_actual[r]));
            }
            node_counts[r] = node_counts_actual[r];
        } else {
            if (throw_error) {
                throw std::runtime_error("node_counts is too short: size=" + std::to_string(node_counts.size()) + ", needed index " + std::to_string(r));
            }
            node_counts.push_back(node_counts_actual[r]);
        }
        
        // === Validate or update group_members ===
        if (r < group_members_actual.size()) {
            for (int i_actual : group_members_actual[r]) {
                if (throw_error && std::find(group_members[r].begin(), group_members[r].end(), i_actual) == group_members[r].end()) {
                    throw std::runtime_error("node " + std::to_string(i_actual) + " from actual group " + std::to_string(r) + " not found in cached group_members[" + std::to_string(r) + "]");
                }
            }
            for (int i : group_members[r]) {
                if (throw_error && std::find(group_members_actual[r].begin(), group_members_actual[r].end(), i) == group_members_actual[r].end()) {
                    throw std::runtime_error("node " + std::to_string(i) + " from cached group " + std::to_string(r) + " not found in actual group_members[" + std::to_string(r) + "]");
                }
            }
            // Update cached bookkeeping with actual values
            group_members[r] = group_members_actual[r];
            node_indices_in_groups[r] = node_indices_in_groups_actual[r];
        } else {
            if (throw_error) {
                throw std::runtime_error("group_members is too short: size=" + std::to_string(group_members.size()) + ", needed index " + std::to_string(r));
            }
            group_members.push_back(group_members_actual[r]);
            node_indices_in_groups.push_back(node_indices_in_groups_actual[r]);
        }
    }

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
                    throw std::runtime_error("edge_counts[" + std::to_string(r) + "] too short for column " + std::to_string(s));
                }
                edge_counts[r].push_back(edge_counts_actual[r][s]);
            }
        }
        if (r < edge_counts_sum.size()) {
            edge_counts_sum[r] = value_check(edge_counts_sum_actual[r], edge_counts_sum[r], NUMERICAL_CHECK_EPSILON, throw_error, "edge_counts_sum[" + std::to_string(r) + "]");
        } else {
            if (throw_error) {
                throw std::runtime_error("edge_counts_sum is too short: size=" + std::to_string(edge_counts_sum.size()) + ", needed index " + std::to_string(r));
            }
            edge_counts_sum.push_back(edge_counts_sum_actual[r]);
        }
    }
    // === Validate edge_counts dimensions ===
    if (edge_counts.size() != static_cast<std::size_t>(num_groups) || edge_counts[0].size() != static_cast<std::size_t>(num_groups)) {
        if (throw_error) {
            throw std::runtime_error("edge_counts dimensions " + std::to_string(edge_counts.size()) + "x" + std::to_string(edge_counts.empty() ? 0 : edge_counts[0].size()) + " do not match num_groups=" + std::to_string(num_groups));
        } else {
            // Repair: resize to match num_groups
            edge_counts.resize(num_groups);
            for (auto& row : edge_counts) {
                row.resize(num_groups, 0);
            }
        }
    }

    // === Validate and rebuild weight_counts (weighted mode only) ===
    // weight_counts are real sums updated by += / -= in flip_nodes, so they are
    // compared with a tolerance relative to the total weight; nonzero_pairs is an
    // integer and must match exactly.  Then the cache is replaced by the recount.
    if (weighted_SBM) {
        int nonzero_pairs_actual = 0;
        for (int r = 0; r < num_groups; r++)
            for (int s = r; s < num_groups; s++)
                if (edge_counts_actual[r][s] > 0) nonzero_pairs_actual++;
        for (int r = 0; r < num_groups; r++)
            for (int s = 0; s < num_groups; s++)
                weight_counts_sum_actual[r] += weight_counts_actual[r][s];
        if (throw_error) {
            const double wtol = NUMERICAL_CHECK_EPSILON + NUMERICAL_CHECK_REL_TOL * 2.0 * w_total;
            if (weight_counts.size() != static_cast<std::size_t>(num_groups) ||
                weight_counts_sum.size() != static_cast<std::size_t>(num_groups))
                throw std::runtime_error("weight_counts dimensions do not match num_groups=" + std::to_string(num_groups));
            for (int r = 0; r < num_groups; r++) {
                if (weight_counts[r].size() != static_cast<std::size_t>(num_groups))
                    throw std::runtime_error("weight_counts[" + std::to_string(r) + "] has the wrong length");
                for (int s = 0; s < num_groups; s++)
                    value_check(weight_counts_actual[r][s], weight_counts[r][s], wtol, true,
                                "weight_counts[" + std::to_string(r) + "][" + std::to_string(s) + "]");
                value_check(weight_counts_sum_actual[r], weight_counts_sum[r], wtol, true,
                            "weight_counts_sum[" + std::to_string(r) + "]");
            }
            if (nonzero_pairs != nonzero_pairs_actual)
                throw std::runtime_error("Bookkeeping mismatch for nonzero_pairs: cached=" + std::to_string(nonzero_pairs) +
                                         ", actual=" + std::to_string(nonzero_pairs_actual));
        }
        weight_counts = weight_counts_actual;
        weight_counts_sum = weight_counts_sum_actual;
        nonzero_pairs = nonzero_pairs_actual;
    }

    // === Initialize or validate density/weight parameters ===
    // On first call, density_in < 0 indicates uninitialized state
    if (density_in < 0.0) {
        // Initialize to overall graph density
        density_in = static_cast<double>(m) / (static_cast<double>(n) * static_cast<double>(n) / 2.0);
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
    
    // === Validate group_score_sums ===
    std::vector<double> group_score_sums_actual(num_groups, 0.0);
    for (int i = 0; i < n; i++) group_score_sums_actual[partition[i]] += scores[i];
    for (std::size_t r = 0; r < static_cast<std::size_t>(num_groups); r++) {
        if (r < group_score_sums.size()) {
            if(throw_error && (std::isnan(group_score_sums_actual[r]) ||  std::isnan(group_score_sums[r]) || std::abs(group_score_sums_actual[r] - group_score_sums[r]) > NUMERICAL_CHECK_EPSILON * std::max(1.0, std::abs(group_score_sums_actual[r])))){
                throw std::runtime_error("group_score_sums[" + std::to_string(r) + "] mismatch: cached=" + std::to_string(group_score_sums[r]) + ", actual=" + std::to_string(group_score_sums_actual[r]));
            }
            group_score_sums[r] = group_score_sums_actual[r];
        } else {
            if (throw_error) {
                throw std::runtime_error("group_score_sums is too short: size=" + std::to_string(group_score_sums.size()) + ", needed index " + std::to_string(r));
            }
            group_score_sums.push_back(group_score_sums_actual[r]);
        }
    }


    // === Compute H_b: Partition entropy ===
    // H_b = -log P(b) with Peixoto's nonparametric partition prior (Peixoto 2017,
    // Phys. Rev. E 95, 012317), P(b) = P(b | n_r) P(n_r | q) P(q):
    //   log(n)                         P(q) uniform on 1..n
    //   log C(n-1, q-1)                P(n_r | q): uniform over the C(n-1, q-1)
    //                                  compositions of n into q non-empty group sizes
    //   log(n!) - sum_r log(n_r!)      P(b | n_r): uniform over labellings with those sizes
    double H_b_actual = std::log(n);
    H_b_actual += log_binom(n - 1, num_groups - 1);  // H(nb|q): Number of partitions
    H_b_actual += log_factorial(n);  // H(b|nb): Group assignment entropy
    for (int r = 0; r < num_groups; r++) {
        H_b_actual -= log_factorial(node_counts[r]);  // Normalize by group size permutations
    }
    // Tolerance of the SBM entropies.  Their incremental updates add and subtract
    // log-factorial-sized terms, up to ~log((2m)!) + log(n!), and the round-off
    // scales with those terms, not with the total (H_M can be ~10 while its terms
    // are ~1e5).  So the relative part is taken of that term scale.
    const double sbm_tol = NUMERICAL_CHECK_EPSILON +
        NUMERICAL_CHECK_REL_TOL * (std::lgamma(2.0 * m + 1.0) + std::lgamma(n + 1.0));
    H_b = value_check(H_b_actual, H_b, sbm_tol, throw_error, "H_b");

    // === Compute H_M: Edge count distribution entropy ===
    // Sum of negative binomial entropies for within/between-group edge counts
    double H_M_actual = calculate_H_M();
    H_M = value_check(H_M_actual, H_M, sbm_tol, throw_error, "H_M");

    // === Compute H_k: Degree sequence entropy ===
    // Entropy of degree distribution given group structure
    double H_k_actual = calculate_H_k();
    H_k = value_check(H_k_actual, H_k, sbm_tol, throw_error, "H_k");

    // === Compute H_A_G_k_M_b: Adjacency matrix entropy ===
    // Entropy of specific edge placements given edge counts
    double H_A_G_k_M_b_actual = calculate_H_A_G_k_M_b();
    H_A_G_k_M_b = value_check(H_A_G_k_M_b_actual, H_A_G_k_M_b, sbm_tol, throw_error, "H_A_G_k_M_b");

    // === Compute total entropies ===
    // calculate_H_W returns -log P(W|A,μ̂,b): the description-length contribution
    // of the weight model, which is negative when the simplex fits well (density > 1).
    // Add it (not subtract) so that a good fit reduces total DL.
    if (weighted_SBM) {
        double H_W_actual = calculate_H_W();
        H_W = value_check(H_W_actual, H_W, sbm_tol, throw_error, "H_W");
    }
    double H_A_G_b_actual = H_A_G_k_M_b + H_k + H_M + H_W;
    H_A_G_b = value_check(H_A_G_b_actual, H_A_G_b, sbm_tol, throw_error, "H_A_G_b");

    // H(A,b) = H(A,G,b) + H(b)
    double H_A_b_actual = H_A_G_b + H_b;
    H_A_b = value_check(H_A_b_actual, H_A_b, sbm_tol, throw_error, "H_A_b");

    // === Compute hierarchy entropies ===
    // Entropy of the directed edges given the undirected edges and scores, Bradley-Terry
    double H_Adir_G_A_s_nu_actual = calculate_H_Adir_G_A_s_nu();
    H_Adir_G_A_s_nu = value_check(H_Adir_G_A_s_nu_actual, H_Adir_G_A_s_nu, NUMERICAL_CHECK_EPSILON, throw_error, "H_Adir_G_A_s_nu");
    // Entropy of the scores given the partition and depth parameters
    double H_s_G_b_depths_actual = calculate_H_s_G_b_depths();
    if(verbose){
        print("H_s_G_b_depths: " + std::to_string(H_s_G_b_depths));
    }
    // H_s is a difference of sums of s_hat^2-sized terms; far in the depth priors'
    // tails (e.g. sigma_G ~ 1e5 on an edgeless graph, s_hat^2 ~ 1e10) rounding
    // alone exceeds an absolute 1e-6, so scale the tolerance with those terms.
    double s_hat_sq = 0.0;
    for (int i = 0; i < n; i++) s_hat_sq += scores[i] * scores[i] / (individual_depth * individual_depth);
    H_s_G_b_depths = value_check(H_s_G_b_depths_actual, H_s_G_b_depths,
                                 std::max(NUMERICAL_CHECK_EPSILON, 1e-12 * s_hat_sq), throw_error, "H_s_G_b_depths");
    
    if (verbose) {
        if(throw_error){
            print("Bookkeeping check passed successfully.");
        }else{
            print("Bookkeeping initialized.");
        }
    }
}

/**
 * calculate_H_k: Compute degree sequence entropy H(k|M,b)
 *
 * The e_r = edge_counts_sum[r] stubs of group r are spread over its n_r nodes by a
 * Dirichlet-multinomial with symmetric concentration kappa
 * (= degree_correction_concentration):
 *     P(k_r | e_r) = prod_{i in r} C(k_i + kappa - 1, k_i) / C(e_r + n_r kappa - 1, e_r).
 * kappa = 1 (degree_correction = 0.5) is the uniform distribution over the degree
 * sequences of each group, the uniform degree prior of the microcanonical DC-SBM
 * (Peixoto 2017, Phys. Rev. E 95, 012317).  kappa -> infinity (degree_correction
 * = 0) is the multinomial with equal node probabilities, i.e. no degree
 * correction; it is used beyond MAX_CONCENTRATION.  Higher degree_correction
 * allows more degree heterogeneity within groups (degree_correction is the Gini
 * coefficient of Gamma(kappa), helpers.h).
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
 * Each block-pair edge count is negative binomial, parameterised by its mean and a
 * concentration (shape) c -- the gamma-Poisson mixture: a Poisson count whose rate
 * is Gamma(shape c) with that mean, variance mean + mean^2 / c (log_neg_binom).
 * The means scale with the number of node pairs:
 * - Within-group: E[M_{rr}/2] = density_in * n[r]^2 / 2   (M_rr is stored doubled)
 * - Between-group: E[M_{rs}] = density_out * n[r] * n[s]
 * c is concentration_in / concentration_out, obtained from variation_in/out (the
 * Gini coefficient of Gamma(c), helpers.h):
 * - concentration=1 (variation 0.5): Geometric distribution
 * - concentration→∞ (variation 0): Poisson distribution (variance=mean)
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
 * calculate_H_A_G_k_M_b: Compute adjacency matrix entropy H(A|k,M,b)
 *
 * The microcanonical degree-corrected SBM (Peixoto 2017, Phys. Rev. E 95, 012317):
 * the stubs are paired uniformly at random subject to the block edge counts e_rs,
 * giving
 *     P(A | k, e, b) = prod_{r<s} e_rs! prod_r e_rr!! prod_i k_i!
 *                      / ( prod_r e_r! prod_{i<j} A_ij! prod_i A_ii!! ),
 * so -log P = sum_{i<j} log A_ij! + sum_i log A_ii!! - sum_i log k_i!
 *             - sum_{r<s} log e_rs! - sum_r log e_rr!! + sum_r log e_r!.
 * Self-loop convention: diagonal entries are stored DOUBLED, A_ii = 2 x (number of
 * self-loops at i) and e_rr = 2 x (edges inside r), hence the double factorials.
 * The constructor rejects self-loops, so the A_ii!! term is always 0 here.
 *
 * The per-edge term sum_{i<j} log A_ij! is counted once per unordered pair (it
 * used to be counted for (i,j) and (j,i)).  It does not depend on the partition,
 * so it never affects the sampling, only H_A_G_k_M_b and the description length.
 * In weighted_SBM and two-layer mode the SBM graph is binary (A_ij in {0, 1}),
 * so the term is 0 and the loop is skipped (two-layer mode used to read the
 * group_adj weights here although the edge counts are binary).
 *
 * The incremental updates in flip_nodes carry only the partition-dependent terms.
 *
 * @return H_A_G_k_M_b entropy value
 */
double MCMC::calculate_H_A_G_k_M_b() {
    double H_A_G_k_M_b_actual = 0.0;
    
    // sum_{i<j} log A_ij! (+ sum_i log A_ii!!): multigraph mode only.
    if (!weighted_SBM && !two_layer_mode) {
        for(int i = 0; i < n; i++){
            for(const auto& edge : neighbors_com[i]){
                const int j = edge.first;
                const int A_ij = static_cast<int>(std::llround(edge.second));
                if(i == j){
                    H_A_G_k_M_b_actual += log_factorial2(A_ij);
                }else if(i < j){
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
 * Weighted SBM entropy H_W = -log P(W|A,b).
 *
 * The weights between groups r and s are uniform on the simplex of their total
 * mu_rs, in the setting of Peixoto's nonparametric weighted SBM (Peixoto,
 * "Nonparametric weighted stochastic block models", Phys. Rev. E 97, 012306
 * (2018); the exact form below was not checked against an equation there),
 *     P(W_rs | mu_rs) = (m_rs - 1)! / mu_rs^(m_rs - 1),
 * and the totals of the Q block pairs joined by at least one edge are uniform on
 * the simplex of the total weight w,
 *     P(mu | w) = (Q - 1)! / w^(Q - 1).
 * A pair with no edges carries no weight (mu_rs = 0) and contributes no factor:
 * counting only the Q pairs with edges gives every partition a density over the
 * same m - 1 free weights, so the inferred partition does not depend on the unit
 * of weight.  (Counting all q(q+1)/2 pairs, as an earlier version did, makes it
 * depend on the unit: with weights below 1 it favours all-singleton partitions.)
 * Diagonal entries are stored doubled (2 m_rr, 2 mu_rr) and are halved on read.
 */
double MCMC::H_W_pair(int r, int s) const {
    const int m_rs = (r == s) ? edge_counts[r][r] / 2 : edge_counts[r][s];
    if (m_rs <= 0) return 0.0;
    const double mu_rs = (r == s) ? weight_counts[r][r] / 2.0 : weight_counts[r][s];
    return (m_rs - 1) * std::log(mu_rs) - log_factorial(m_rs - 1);
}

double MCMC::H_W_totals(int nonzero) const {
    if (w_total <= 0.0 || nonzero <= 1) return 0.0;
    return (nonzero - 1) * std::log(w_total) - log_factorial(nonzero - 1);
}

double MCMC::calculate_H_W() {
    double h = 0.0;
    int nonzero = 0;
    for (int r = 0; r < num_groups; r++) {
        for (int s = r; s < num_groups; s++) {
            h += H_W_pair(r, s);
            if (edge_counts[r][s] > 0) nonzero++;
        }
    }
    return h + H_W_totals(nonzero);
}

/**
 * calculate_H_Adir_G_A_s_nu: Compute the outcome entropy H(Adir|A,s,nu)
 *
 * Each unordered pair {i, j} with A^dir_ij = w_ij + w_ji + t_ij interactions has
 * multinomial outcome counts (w_ij wins of i, w_ji wins of j, t_ij ties):
 *     P(Adir|A,s) = prod_{i<j} A^dir_ij! / (w_ij! w_ji! t_ij!) p_ij^w_ij p_ji^w_ji p_tie^t_ij,
 * with Davidson's (1970, J. Am. Stat. Assoc. 65:317) extension of the
 * Bradley-Terry model to ties, pi_i = exp(s_i):
 *     p_ij  = pi_i / Z,   p_tie = nu sqrt(pi_i pi_j) / Z,
 *     Z     = pi_i + pi_j + nu sqrt(pi_i pi_j).
 * nu = 0 is plain Bradley-Terry; with ties_parameter fixed at 0 the ties are not
 * outcomes (ties_in_direction_model = false) and t_ij is dropped from A^dir.
 * The multinomial coefficients are the constant H_Adir_counts_term (constructor).
 */
double MCMC::calculate_H_Adir_G_A_s_nu() {
    double H_Adir_G_A_s_nu_actual = 0.0;
    // Iterate over all undirected edges
    for(int i = 0; i < n; i++){
        for(const auto& edge : wins[i]){
            int j = edge.first;
            double Awins_ij = edge.second;
            double log_p_win_ij = scores[i] - log_sum_exp(std::vector<double>{scores[i], scores[j], std::log(ties_parameter) + scores[i]/2.0 + scores[j]/2.0}); // P(i -> j|i - j)
            H_Adir_G_A_s_nu_actual -= Awins_ij * log_p_win_ij;
            if(verbose){
                print("Edge (" + std::to_string(i) + " -> " + std::to_string(j) + "): Awins_ij = " + std::to_string(Awins_ij) + ", log_p_win_ij = " + std::to_string(log_p_win_ij));
            }
        }
        for(const auto& edge : ties[i]){
            int j = edge.first;
            double Aties_ij = edge.second;
            if(i < j){
                double log_p_tie_ij = std::log(ties_parameter) + scores[i]/2.0 + scores[j]/2.0 - log_sum_exp(std::vector<double>{scores[i], scores[j], std::log(ties_parameter) + scores[i]/2.0 + scores[j]/2.0}); // P(i <-> j|i - j): ν·exp((si+sj)/2) / Z
                H_Adir_G_A_s_nu_actual -= Aties_ij * log_p_tie_ij;
                if(verbose){
                    print("Edge (" + std::to_string(i) + " <-> " + std::to_string(j) + "): Aties_ij = " + std::to_string(Aties_ij) + ", log_p_tie_ij = " + std::to_string(log_p_tie_ij));
                }
            }
        }
    }
    H_Adir_G_A_s_nu_actual += H_Adir_counts_term;
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

/*
* calculate_H_s_G_b_depths: Compute score entropy H(s|b,depths)
*
* s_i = m_{b_i} + e_i with independent group means m_r ~ N(0, sigma_G^2/2) and
* deviations e_i ~ N(0, sigma_I^2/2); the group means are integrated out
* analytically, leaving a Gaussian density over the scores themselves (below).
*
* @return H_s_G_b_depths entropy value
*/
double MCMC::calculate_H_s_G_b_depths() {
    // -log P(s | b, sigma_I, sigma_G), a density over the scores themselves:
    // s ~ N(0, sigma_I^2/2 I + sigma_G^2/2 B), B the same-group indicator, with each
    // group's mean integrated out.  Per node log(sqrt(pi) sigma_I) + s_i^2 / sigma_I^2,
    // per group score_prior_group_term(r).
    double H = n * (std::log(std::sqrt(M_PI)) + std::log(individual_depth));
    const double inv_var = 1.0 / (individual_depth * individual_depth);
    for (int i = 0; i < n; i++) H += scores[i] * scores[i] * inv_var;
    for (int r = 0; r < num_groups; r++) H += score_prior_group_term(r);
    return H;
}

double MCMC::score_prior_group_term(int r) const {
    const double rho = group_depth * group_depth / (individual_depth * individual_depth);
    const double f2 = 1.0 / (1.0 + node_counts[r] * rho);
    const double S = group_score_sums[r];
    return -(0.5 * std::log(f2) + f2 * rho * S * S / (individual_depth * individual_depth));
}
   