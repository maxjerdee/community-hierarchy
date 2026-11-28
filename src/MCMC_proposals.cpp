#include "MCMC_core.h"
#include "globals.h"
#include "helpers.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <set>
#include <string>

/**
 * propose_move: Select and execute one random MCMC proposal.
 *
 * The move type is drawn with the (state-independent) weights set in MCMC_sweep:
 * - single_site_move:   block-informed single-node move, Peixoto 2020 Eq. 15 (weight n)
 * - merge/split/merge_split_move: collective partition moves (MCMC_merge_split.cpp;
 *                       weight MERGE_SPLIT_WEIGHT each, see globals.h)
 * - score_move:         update one node's hierarchy score (weight n)
 * - parameter_move:     update one continuous parameter
 *                       (weight PARAMETER_MOVE_WEIGHT x # free parameters)
 * The group-score shift is not drawn here: MCMC_sweep ends every sweep with one
 * shift per group.
 */
void MCMC::propose_move() {
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
            } else if (proposal_type == "merge_move") {
                propose_merge_move();
            } else if (proposal_type == "split_move") {
                propose_split_move();
            } else if (proposal_type == "merge_split_move") {
                propose_merge_split_move();
            }

            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Block-informed partition proposals (Peixoto 2020, Phys. Rev. E 102, 012305;
// the graph-tool defaults).  Neighbours are neighbors_com (the graph that drives
// the groups), sorted by node id, so the draws are the same on every platform.
// ---------------------------------------------------------------------------

// Eq. 14: P_e(g | i, b) = sum_t (k_i^t / k_i) (e_tg + c) / (e_t + c B), with k_i^t
// the number of i's (non-self) neighbours in group t.  Isolated nodes: uniform.
//
// With weighted_proposals (weighted_SBM only) the same formula is used with edge
// WEIGHTS in place of edge counts: k_i^t is the weight from i into group t and
// e_tg the block-pair weight total mu_tg (weight_counts).  This is still a valid
// MH proposal (forward and reverse use the same function); it matters on dense
// weighted networks, where the binary counts are nearly uniform.
double MCMC::proposal_block_edges(int t, int g) const {
    return weighted_proposals ? weight_counts[t][g] : static_cast<double>(edge_counts[t][g]);
}

double MCMC::proposal_block_degree(int t) const {
    return weighted_proposals ? weight_counts_sum[t] : static_cast<double>(edge_counts_sum[t]);
}

double MCMC::block_move_prob(int node, int g) const {
    const auto& nb = neighbors_com[static_cast<std::size_t>(node)];
    double k = 0.0;
    std::vector<double> kt(static_cast<std::size_t>(num_groups), 0.0);
    for (const auto& e : nb) {
        const double a = weighted_proposals ? e.second : 1.0;
        k += a;
        kt[static_cast<std::size_t>(partition[e.first])] += a;
    }
    if (k <= 0.0) return 1.0 / num_groups;
    const double c = BLOCK_SITE_EPS;
    double p = 0.0;
    for (int t = 0; t < num_groups; ++t)
        if (kt[static_cast<std::size_t>(t)] > 0.0)
            p += kt[static_cast<std::size_t>(t)] / k * (proposal_block_edges(t, g) + c) /
                 (proposal_block_degree(t) + c * num_groups);
    return p;
}

// Draw from Eq. 14: a neighbour j (uniform, or by weight), t = b_j, then g
// with probability (e_tg + c) / (e_t + c B).
int MCMC::sample_block_move(int node) {
    const auto& nb = neighbors_com[static_cast<std::size_t>(node)];
    double k = 0.0;
    for (const auto& e : nb) k += weighted_proposals ? e.second : 1.0;
    if (k <= 0.0) return next_int(rng, 0, num_groups - 1);
    int t = -1;
    if (weighted_proposals) {
        const double u = uniform_random(rng) * k;
        double cum = 0.0;
        for (const auto& e : nb) {
            t = partition[e.first];
            cum += e.second;
            if (u < cum) break;
        }
    } else {
        int idx = next_int(rng, 0, static_cast<int>(k) - 1);
        for (const auto& e : nb) {
            if (idx == 0) {
                t = partition[e.first];
                break;
            }
            --idx;
        }
    }
    const double c = BLOCK_SITE_EPS;
    const double tot = proposal_block_degree(t) + c * num_groups;
    const double u = uniform_random(rng) * tot;
    double cum = 0.0;
    for (int g = 0; g < num_groups; ++g) {
        cum += proposal_block_edges(t, g) + c;
        if (u < cum) return g;
    }
    return num_groups - 1;
}

// Eq. 19: P(s | r, b) = (1/n_r) sum_{i in r} P_e(s|i,b) / (1 - P_e(r|i,b)).
double MCMC::partner_logprob(int r, int s) const {
    double acc = 0.0;
    for (int i : group_members[static_cast<std::size_t>(r)])
        acc += block_move_prob(i, s) / (1.0 - block_move_prob(i, r));
    return std::log(acc / static_cast<double>(node_counts[static_cast<std::size_t>(r)]));
}

// Draw s ~ P(s | r, b): a uniform member i of r, then P_e(.|i,b) conditioned on s != r.
int MCMC::sample_partner(int r) {
    const int i = group_members[static_cast<std::size_t>(r)][static_cast<std::size_t>(
        next_int(rng, 0, node_counts[static_cast<std::size_t>(r)] - 1))];
    for (;;) {
        const int s = sample_block_move(i);
        if (s != r) return s;
    }
}

// Probability of choosing the unordered pair {r, s} for a merge: r uniform, then
// s ~ P(s|r,b), or the other way round.
double MCMC::pair_logprob(int r, int s) const {
    const double a = partner_logprob(r, s), b = partner_logprob(s, r), mx = std::max(a, b);
    return mx + std::log(std::exp(a - mx) + std::exp(b - mx)) -
           std::log(static_cast<double>(num_groups));
}

/**
 * propose_single_site_move: Peixoto (2020) Eq. 15 single-node move.
 *
 * A uniform node i moves to a NEW group with probability d = BLOCK_NEW_GROUP_PROB,
 * otherwise to s ~ P_e(s | i, b) (Eq. 14; s = b_i is a no-op).  A singleton that
 * moves vacates its group.  The target is over PARTITIONS, q! exp(-H_b) x ...,
 * the same as the merge/split moves, so a move that changes q carries their
 * label factor: split_label_factor(K) when a new group opens (K -> K+1),
 * merge_label_factor(K) when one is vacated (K -> K-1).
 *
 * Hastings terms: forward d or (1-d) P_e(s|i,b) in the current state; reverse
 * evaluated after the move -- d for undoing a vacate (recreate the singleton),
 * (1-d) P_e(r|i,b') otherwise.  The partition prior H_b does NOT cancel here and
 * enters the acceptance; only the data likelihood is tempered.
 *
 * Rejection undo: after a vacate the labels were compacted, so recreate the
 * singleton as group num_groups; otherwise r is still valid.
 */
void MCMC::propose_single_site_move() {
    single_site_proposals++;
    if (partition_fixed) return;
    const int node = next_int(rng, 0, n - 1);
    const int r = partition[static_cast<std::size_t>(node)], K = num_groups;
    const bool singleton = node_counts[static_cast<std::size_t>(r)] == 1;
    const bool to_new = uniform_random(rng) < BLOCK_NEW_GROUP_PROB;
    int s;
    double log_fwd;
    if (to_new) {
        if (singleton || num_groups_fixed) return;  // same partition / not allowed
        s = K;
        log_fwd = std::log(BLOCK_NEW_GROUP_PROB);
    } else {
        s = sample_block_move(node);
        if (s == r) return;
        if (singleton && num_groups_fixed) return;
        log_fwd = std::log(1.0 - BLOCK_NEW_GROUP_PROB) + std::log(block_move_prob(node, s));
    }
    if (to_new) ++new_group_proposals;
    if (singleton) ++vacate_proposals;

    const double H_b_old = H_b, H_A_G_b_old = H_A_G_b, H_s_old = H_s_G_b_depths;
    flip_nodes({node}, s);  // vacating r compacts the labels
    double log_rev, label = 0.0;
    if (singleton) {
        log_rev = std::log(BLOCK_NEW_GROUP_PROB);
        label = merge_label_factor(K);  // K -> K-1
    } else {
        log_rev = std::log(1.0 - BLOCK_NEW_GROUP_PROB) + std::log(block_move_prob(node, r));
        if (to_new) label = split_label_factor(K);  // K -> K+1
    }
    const double delta_H = (H_b - H_b_old) + beta * (H_A_G_b - H_A_G_b_old) +
                           (H_s_G_b_depths - H_s_old) + log_fwd - log_rev - label;
    if (uniform_random(rng) >= std::exp(-delta_H)) {
        if (singleton)
            flip_nodes({node}, num_groups);  // recreate the singleton
        else
            flip_nodes({node}, r);
    } else {
        accepted_single_site_proposals++;
        if (to_new) ++accepted_new_group_proposals;
        if (singleton) ++accepted_vacate_proposals;
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
 *
 * sigma_I and sigma_G each have several kernels, chosen by state-independent
 * coins (globals.h).  With scores_fixed only the kernels that leave the scores
 * unchanged are used -- the centred sigma_I move and the plain sigma_G move --
 * each a valid MH kernel for p(sigma | s, b) on its own, so the posterior of
 * the depths given the fixed scores is still targeted exactly.
 */
void MCMC::propose_parameter_move() {
    if (free_parameters.empty()) {
        return;  // No free parameters to update
    }

    // Randomly select a free parameter
    // next_int, not uniform_int_distribution: portable across standard libraries (rng.h)
    const auto& parameter = free_parameters[static_cast<std::size_t>(
        next_int(rng, 0, static_cast<int>(free_parameters.size()) - 1))];

    double new_value = 0.0;

    // Generate proposal based on parameter type
    if (parameter == "variation_in") {
        new_value = uniform_proposal(variation_in, step_variation_in);
    } else if (parameter == "variation_out") {
        new_value = uniform_proposal(variation_out, step_variation_out);
    } else if (parameter == "degree_correction") {
        new_value = uniform_proposal(degree_correction, step_degree_correction);
    } else if (parameter == "density_in") {
        new_value = exponential_proposal(density_in, 1.0, step_density_in);
    } else if (parameter == "density_out") {
        new_value = exponential_proposal(density_out, 1.0, step_density_out);
    } else if (parameter == "individual_depth") {
        // scores_fixed: the centred move only (the others rescale the scores).
        if (scores_fixed ||
            uniform_random(rng) < INDIVIDUAL_DEPTH_CENTRED_PROB) {  // state-independent coin
            propose_individual_depth_centred();
            return;
        }
        if (INDIVIDUAL_DEPTH_SCALE_MOVE == 1) {
            propose_individual_depth_residual();
            return;
        }
        new_value =
            half_cauchy_proposal(individual_depth, DEPTH_PRIOR_WIDTH, step_individual_depth);
    } else if (parameter == "group_depth") {
        // State-independent coin first, so each of the two sigma_G kernels draws
        // from, and adapts, its own step size: the plain move touches only the
        // score prior and can step far; the rescale move moves the group means,
        // which the data constrain.  scores_fixed: the plain move only (the
        // rescale move moves the group means, i.e. the scores).
        if (!scores_fixed && uniform_random(rng) < GROUP_DEPTH_RESCALE_PROB) {
            const double v =
                half_cauchy_proposal(group_depth, DEPTH_PRIOR_WIDTH, step_group_depth_rescale);
            propose_group_depth_rescale(v);
            adapt_step(step_group_depth_rescale, adapt_t_group_depth_rescale, group_depth == v,
                       ADAPT_MAX_STEP_CDF);
            return;
        }
        new_value = half_cauchy_proposal(group_depth, DEPTH_PRIOR_WIDTH, step_group_depth);
    } else if (parameter == "ties_parameter") {
        new_value = half_cauchy_proposal(ties_parameter, DEPTH_PRIOR_WIDTH,
                                         step_ties);  // Must remain positive
    } else {
        return;  // Unknown parameter
    }

    // Attempt parameter change (includes acceptance logic).  (The sigma_G rescale
    // move, taken with probability GROUP_DEPTH_RESCALE_PROB, returned above.)
    change_parameter(parameter, new_value);

    if (parameter == "individual_depth")
        adapt_step(step_individual_depth, adapt_t_individual_depth, individual_depth == new_value,
                   ADAPT_MAX_STEP_CDF);
    else if (parameter == "group_depth")
        adapt_step(step_group_depth, adapt_t_group_depth, group_depth == new_value,
                   ADAPT_MAX_STEP_CDF);
    else if (parameter == "ties_parameter")
        adapt_step(step_ties, adapt_t_ties, ties_parameter == new_value, ADAPT_MAX_STEP_CDF);
    else if (parameter == "density_in")
        adapt_step(step_density_in, adapt_t_density_in, density_in == new_value,
                   ADAPT_MAX_STEP_CDF);
    else if (parameter == "density_out")
        adapt_step(step_density_out, adapt_t_density_out, density_out == new_value,
                   ADAPT_MAX_STEP_CDF);
    else if (parameter == "variation_in")
        adapt_step(step_variation_in, adapt_t_variation_in, variation_in == new_value,
                   ADAPT_MAX_STEP_CDF);
    else if (parameter == "variation_out")
        adapt_step(step_variation_out, adapt_t_variation_out, variation_out == new_value,
                   ADAPT_MAX_STEP_CDF);
    else if (parameter == "degree_correction")
        adapt_step(step_degree_correction, adapt_t_degree_correction,
                   degree_correction == new_value, ADAPT_MAX_STEP_CDF);
}

/**
 * propose_group_depth_rescale: sigma_G -> sigma_G' (proposed in the CDF space of
 * its half-Cauchy prior, so the prior cancels) with every group's mean score m_r
 * scaled by lambda = sigma_G' / sigma_G about 0; deviations from the group means
 * are unchanged.  The plain sigma_G move holds the group means fixed, so sigma_G
 * can only move as far as the current means allow.  The deterministic map on the
 * q means has Jacobian lambda^q.  Not used with scores_fixed.  O(n + m).
 */
void MCMC::propose_group_depth_rescale(double new_value) {
    group_depth_proposals++;
    const double gd_old = group_depth, lam = new_value / gd_old;
    const double H_s_old = H_s_G_b_depths, H_A_old = H_Adir_G_A_s_nu;
    const std::vector<double> sc_old = scores, S_old = group_score_sums;
    for (int i = 0; i < n; i++) {
        const int r = partition[i];
        scores[i] += (lam - 1.0) * S_old[r] / node_counts[r];
    }
    std::fill(group_score_sums.begin(), group_score_sums.end(), 0.0);  // exact, no drift
    for (int i = 0; i < n; i++) group_score_sums[partition[i]] += scores[i];
    group_depth = new_value;
    H_s_G_b_depths = calculate_H_s_G_b_depths();
    H_Adir_G_A_s_nu = calculate_H_Adir_G_A_s_nu();
    // Only the direction likelihood is tempered (MCMC_core.h).
    const double log_acc = -((H_s_G_b_depths - H_s_old) + beta * (H_Adir_G_A_s_nu - H_A_old)) +
                           num_groups * std::log(lam);
    if (std::log(uniform_random(rng)) < log_acc) {
        accepted_group_depth_proposals++;
    } else {
        scores = sc_old;
        group_score_sums = S_old;
        group_depth = gd_old;
        H_s_G_b_depths = H_s_old;
        H_Adir_G_A_s_nu = H_A_old;
    }
}

/**
 * propose_individual_depth_centred: sigma_I -> sigma_I' (CDF space of its
 * half-Cauchy prior, so the prior cancels) with the scores held fixed.  The
 * likelihood depends on the scores only, so it is unchanged; only the score prior
 * decides, and there is no Jacobian.  The sigma_I counterpart of the plain sigma_G
 * move (see INDIVIDUAL_DEPTH_CENTRED_PROB).  O(n).
 */
void MCMC::propose_individual_depth_centred() {
    individual_depth_proposals++;
    const double id_old = individual_depth;
    const double id_new =
        half_cauchy_proposal(individual_depth, DEPTH_PRIOR_WIDTH, step_individual_depth_centred);
    const double H_s_old = H_s_G_b_depths;
    individual_depth = id_new;
    H_s_G_b_depths = calculate_H_s_G_b_depths();
    const bool accepted = std::log(uniform_random(rng)) < -(H_s_G_b_depths - H_s_old);
    if (accepted) {
        accepted_individual_depth_proposals++;
    } else {
        individual_depth = id_old;
        H_s_G_b_depths = H_s_old;
    }
    adapt_step(step_individual_depth_centred, adapt_t_individual_depth_centred, accepted,
               ADAPT_MAX_STEP_CDF);
}

/**
 * propose_individual_depth_residual: sigma_I -> sigma_I' (CDF space of its
 * half-Cauchy prior, so the prior cancels) with each score's deviation from its
 * group's mean scaled by lambda = sigma_I' / sigma_I and the group means (hence the
 * group score sums) held fixed.  The map acts on the n - q deviations: Jacobian
 * lambda^(n-q).  The counterpart of propose_group_depth_rescale (group means scaled,
 * deviations kept).  The scores change, so the direction likelihood is recomputed.
 * Not used with scores_fixed.  O(n + m).
 */
void MCMC::propose_individual_depth_residual() {
    individual_depth_proposals++;
    const double id_old = individual_depth;
    const double id_new =
        half_cauchy_proposal(individual_depth, DEPTH_PRIOR_WIDTH, step_individual_depth_residual);
    const double lam = id_new / id_old;
    const double H_s_old = H_s_G_b_depths, H_A_old = H_Adir_G_A_s_nu;
    const std::vector<double> sc_old = scores;
    for (int i = 0; i < n; i++) {
        const int r = partition[i];
        const double m = group_score_sums[r] / node_counts[r];
        scores[i] = m + lam * (scores[i] - m);
    }
    individual_depth = id_new;
    H_s_G_b_depths = calculate_H_s_G_b_depths();
    H_Adir_G_A_s_nu = calculate_H_Adir_G_A_s_nu();
    // Only the direction likelihood is tempered (MCMC_core.h).
    const double log_acc = -((H_s_G_b_depths - H_s_old) + beta * (H_Adir_G_A_s_nu - H_A_old)) +
                           (n - num_groups) * std::log(lam);
    const bool accepted = std::log(uniform_random(rng)) < log_acc;
    if (accepted) {
        accepted_individual_depth_proposals++;
    } else {
        individual_depth = id_old;
        scores = sc_old;
        H_s_G_b_depths = H_s_old;
        H_Adir_G_A_s_nu = H_A_old;
    }
    adapt_step(step_individual_depth_residual, adapt_t_individual_depth_residual, accepted,
               ADAPT_MAX_STEP_CDF);
}

// Robbins-Monro step adaptation (burn-in only): log s += gamma_t (accepted - target).
void MCMC::adapt_step(double& step, long& t, bool accepted, double max_step) {
    if (!adapting) return;
    ++t;
    const double gamma =
        ADAPT_GAIN / std::pow(1.0 + static_cast<double>(t) / ADAPT_T0, ADAPT_KAPPA);
    step *= std::exp(gamma * ((accepted ? 1.0 : 0.0) - ADAPT_TARGET));
    step = std::min(std::max(step, ADAPT_MIN_STEP), max_step);
}

/**
 * propose_score_move: update one uniformly chosen node's score s_i.
 *
 * Given the other scores, the score prior P(s | b, depths) makes s_i Gaussian with
 * mean f S_{-i} / (1 - f) and variance sigma_I^2 / (2 (1 - f)), where S_{-i} is the
 * sum of the other scores in i's group and f = f_r^2 sigma_G^2 / sigma_I^2.  The
 * proposal is a reflected random walk in the CDF space of that conditional
 * (gaussian_proposal), which is reversible with respect to it, so the prior
 * cancels and only the tempered direction likelihood decides.  This needs the CDF
 * and quantile to be exact inverses (helpers.cpp).  H_Adir is updated over the
 * node's own edges only; H_s by its node and group terms.  No-op if scores_fixed.
 */
void MCMC::propose_score_move() {
    if (scores_fixed) return;
    score_proposals++;
    // Select a random node
    int node = next_int(rng, 0, n - 1);
    const double old_score = scores[node];
    // Propose new score according to the Gaussian distribution
    // Mean and standard deviation of the s_i dependence of P(s|b,depths):
    double f_normalization =
        1 / std::sqrt(1 + node_counts[partition[node]] * group_depth * group_depth /
                              (individual_depth * individual_depth));
    double f_combination = f_normalization * f_normalization * group_depth * group_depth /
                           (individual_depth * individual_depth);
    double proposal_mean =
        (group_score_sums[partition[node]] - old_score) * f_combination / (1 - f_combination);
    double proposal_variance = 0.5 * individual_depth * individual_depth / (1 - f_combination);

    const double new_score =
        gaussian_proposal(old_score, proposal_mean, std::sqrt(proposal_variance), step_score);

    // Store old entropy values for acceptance calculation
    double H_s_G_b_depths_old = H_s_G_b_depths;
    double H_Adir_G_A_s_nu_old = H_Adir_G_A_s_nu;

    // H_s_G_b_depths, subtract terms
    H_s_G_b_depths -= scores[node] * scores[node] / (individual_depth * individual_depth) +
                      score_prior_group_term(partition[node]);

    // H_Adir_G_A_s_nu, subtract terms
    const double log_nu = std::log(ties_parameter);
    for (const auto& edge : wins[node]) {
        int j = edge.first;
        double Adir_ij = edge.second;
        double log_p_win_ij = scores[node] - davidson_log_z(scores[node], scores[j], log_nu);
        H_Adir_G_A_s_nu -= 0.0 - Adir_ij * log_p_win_ij;
    }
    for (const auto& edge : losses[node]) {
        int j = edge.first;
        double Adir_ji = edge.second;
        double log_p_win_ji = scores[j] - davidson_log_z(scores[j], scores[node], log_nu);
        H_Adir_G_A_s_nu -= 0.0 - Adir_ji * log_p_win_ji;
    }
    for (const auto& edge : ties[node]) {
        int j = edge.first;
        double Aties_ij = edge.second;
        double log_p_tie_ij = davidson_log_tie(scores[node], scores[j], log_nu) -
                              davidson_log_z(scores[node], scores[j], log_nu);
        H_Adir_G_A_s_nu -= 0.0 - Aties_ij * log_p_tie_ij;
    }

    // Apply proposed change
    scores[node] = new_score;
    group_score_sums[partition[node]] += (new_score - old_score);

    // H_s_G_b_depths, add terms
    H_s_G_b_depths += scores[node] * scores[node] / (individual_depth * individual_depth) +
                      score_prior_group_term(partition[node]);

    // H_Adir_G_A_s_nu, add terms
    for (const auto& edge : wins[node]) {
        int j = edge.first;
        double Adir_ij = edge.second;
        double log_p_win_ij = scores[node] - davidson_log_z(scores[node], scores[j], log_nu);
        H_Adir_G_A_s_nu += 0.0 - Adir_ij * log_p_win_ij;
    }
    for (const auto& edge : losses[node]) {
        int j = edge.first;
        double Adir_ji = edge.second;
        double log_p_win_ji = scores[j] - davidson_log_z(scores[j], scores[node], log_nu);
        H_Adir_G_A_s_nu += 0.0 - Adir_ji * log_p_win_ji;
    }
    for (const auto& edge : ties[node]) {
        int j = edge.first;
        double Aties_ij = edge.second;
        double log_p_tie_ij = davidson_log_tie(scores[node], scores[j], log_nu) -
                              davidson_log_z(scores[node], scores[j], log_nu);
        H_Adir_G_A_s_nu += 0.0 - Aties_ij * log_p_tie_ij;
    }

    // Metropolis-Hastings acceptance
    // The proposal is reversible w.r.t. the score prior exp(-H_s_G_b_depths) (it is
    // a CDF-space random walk on the exact conditional of this node's score), so the
    // prior cancels and only the tempered direction likelihood remains.  This is why
    // H_s_G_b_depths must be UNtempered in every other move too (MCMC_core.h).
    double delta_H = H_Adir_G_A_s_nu - H_Adir_G_A_s_nu_old;
    double acceptance_prob = std::exp(-beta * delta_H);
    const bool accepted = uniform_random(rng) < acceptance_prob;
    adapt_step(step_score, adapt_t_score, accepted, ADAPT_MAX_STEP_CDF);
    if (accepted) {
        // Accept
        accepted_score_proposals++;
    } else {
        // Reject: restore old score
        scores[node] = old_score;
        group_score_sums[partition[node]] -= (new_score - old_score);

        H_s_G_b_depths = H_s_G_b_depths_old;
        H_Adir_G_A_s_nu = H_Adir_G_A_s_nu_old;
    }
}

/**
 * propose_group_score_move: shift every node's score in group r (a random group
 * if r < 0) by a common offset delta ~ sigma_I N(0, step_group_score).  MCMC_sweep calls it once per group at the end of every sweep.
 *
 * This translates a whole group along the between-group (sigma_C) direction.
 * Within-group score differences are invariant under the shift, so the
 * orientation likelihood only changes for edges crossing the group boundary.
 * Both changes are computed BEFORE touching the state, and the state is changed
 * only on acceptance:
 *   score prior, O(1): sum_i (s_i + d)^2 = sum s^2 + 2 d S + n d^2 (over sigma_I^2), the group
 *     normaliser log f_r is unchanged, and S -> S + n d;
 *   direction likelihood: group_cross_dA, one pass over r's cross-group pairs.
 * Accept on the full delta_H (symmetric random walk on delta).
 */
void MCMC::propose_group_score_move(int r) {
    if (scores_fixed) return;
    group_score_proposals++;
    if (num_groups <= 1) return;  // no between-group structure to move
    if (r < 0) r = next_int(rng, 0, num_groups - 1);
    if (group_members[r].empty()) return;
    const double delta =
        individual_depth *
        next_normal(rng, step_group_score);  // symmetric; sigma_I is not moved here
    if (delta == 0.0 || !std::isfinite(delta)) return;

    const double nr = static_cast<double>(node_counts[r]);
    const double rho = group_depth * group_depth / (individual_depth * individual_depth);
    const double S0 = group_score_sums[r], S1 = S0 + nr * delta;
    const double f2 = 1.0 / (1.0 + nr * rho);
    const double dHs = ((2.0 * delta * S0 + nr * delta * delta) - f2 * rho * (S1 * S1 - S0 * S0)) /
                       (individual_depth * individual_depth);
    const double dA = group_cross_dA(r, delta);

    // Only the likelihood is tempered; the score prior enters at full weight.
    const double delta_H = beta * dA + dHs;
    const bool accepted = uniform_random(rng) < std::exp(-delta_H);
    adapt_step(step_group_score, adapt_t_group_score, accepted, ADAPT_MAX_STEP_GROUP_SCORE);
    if (accepted) {
        accepted_group_score_proposals++;
        for (int i : group_members[r]) scores[i] += delta;
        group_score_sums[r] += nr * delta;
        H_s_G_b_depths += dHs;
        H_Adir_G_A_s_nu += dA;
    }
}

// Exact change of H_Adir when every raw score in group r shifts by d.  Within-group
// games are invariant (they depend on score differences only).  For each
// cross-group opponent pair (i in r, j not in r) the win, loss and tie counts share
// one denominator L = logsumexp(s_i, s_j, log nu + (s_i + s_j)/2), so
//   H_pair = -[w s_i + l s_j + t (log nu + (s_i + s_j)/2)] + (w + l + t) L,
// whose change is -(w + t/2) d + (w + l + t)(L1 - L0).  wins/losses/ties are sorted
// by node id, so a 3-way merge visits each opponent once (the result is exact in
// any order: the terms are linear in the counts).
double MCMC::group_cross_dA(int r, double d) const {
    const double lnu = std::log(ties_parameter);
    double dA = 0.0;
    for (int i : group_members[r]) {
        const auto &W = wins[i], &Lo = losses[i], &T = ties[i];
        std::size_t a = 0, b = 0, c = 0;
        const double si = scores[i], si1 = si + d;
        while (a < W.size() || b < Lo.size() || c < T.size()) {
            int j = std::numeric_limits<int>::max();
            if (a < W.size()) j = std::min(j, W[a].first);
            if (b < Lo.size()) j = std::min(j, Lo[b].first);
            if (c < T.size()) j = std::min(j, T[c].first);
            double w = 0, l = 0, t = 0;
            if (a < W.size() && W[a].first == j) w = W[a++].second;
            if (b < Lo.size() && Lo[b].first == j) l = Lo[b++].second;
            if (c < T.size() && T[c].first == j) t = T[c++].second;
            if (partition[j] == r) continue;
            const double sj = scores[j];
            const double L0 = davidson_log_z(si, sj, lnu);
            const double L1 = davidson_log_z(si1, sj, lnu);
            dA += -(w + 0.5 * t) * d + (w + l + t) * (L1 - L0);
        }
    }
    return dA;
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
 * PRECONDITION: every node in `nodes` is in the same group, old_group =
 * partition[nodes[0]] (not checked).
 *
 * Labels after the move: when old_group is emptied, the last group is relabelled
 * into old_group's slot (swap_groups) and num_groups drops by one.  So a whole
 * group moved to a NEW group (new_group == num_groups) keeps the label
 * old_group, and a group merged into the last group ends up labelled old_group.
 * Callers that track groups across a flip must re-read partition[] afterwards.
 *
 * @param nodes: Vector of node IDs to move
 * @param new_group: Target group index (may be == num_groups for new group)
 */
void MCMC::flip_nodes(const std::vector<int>& nodes, int new_group) {
    int old_group = partition[nodes[0]];

    // Self-merge: the caller passed new_group == old_group while the group would be
    // emptied. Redirect to the last group so compaction renames it into old_group's slot.
    bool empty_old_group_check = (node_counts[old_group] == static_cast<int>(nodes.size()));
    if (empty_old_group_check && new_group == old_group) {
        new_group = num_groups - 1;
    }

    // Remove nodes that are already in the target group
    std::vector<int> filtered_nodes;
    for (int node : nodes) {
        if (partition[node] != new_group) {
            filtered_nodes.push_back(node);
        }
    }

    if (filtered_nodes.empty()) {
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
    if (new_group != num_groups) {  // If not creating a new group
        initial_impacted_groups.insert(new_group);
    }
    final_impacted_groups.insert(old_group);
    final_impacted_groups.insert(new_group);

    // Relabel: empty_old_group with new_group == num_groups (whole group moves
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
    if (new_group == num_groups && empty_old_group) {
        final_impacted_groups.erase(new_group);
    }

    // If emptying old_group, the last group will be moved to fill the gap
    if (new_group != num_groups && empty_old_group) {
        initial_impacted_groups.insert(num_groups - 1);  // Last group moves
        if (old_group == num_groups - 1) {
            final_impacted_groups.erase(old_group);
        }
        if (new_group == num_groups - 1) {
            final_impacted_groups.erase(new_group);
        }
    }

    // Store old entropy values
    double H_b_old = H_b;
    double H_k_old = H_k;
    double H_M_old = H_M;
    double H_A_G_k_M_b_old = H_A_G_k_M_b;
    double H_W_old = H_W;

    /* === PHASE 1: Remove entropy terms from initial configuration === */

    // H(nb|q): Partition count prior
    H_b -= log_binom(n - 1, num_groups - 1);

    // H(b|nb): Group size distribution
    for (int r : initial_impacted_groups) {
        H_b += log_factorial(node_counts[r]);  // Will be subtracted back after update
    }

    // H(k): Degree sequence entropy
    for (int r : initial_impacted_groups) {
        if (degree_correction_concentration < MAX_CONCENTRATION) {
            H_k -=
                log_binom(edge_counts_sum[r] + node_counts[r] * degree_correction_concentration - 1,
                          node_counts[r] * degree_correction_concentration - 1);
        } else {
            // Multinomial approximation for large concentration
            H_k += log_factorial(edge_counts_sum[r]);
            H_k -= edge_counts_sum[r] * std::log(node_counts[r]);
        }
    }

    // H(M): Edge count distribution entropy
    for (int r : initial_impacted_groups) {
        for (int s = 0; s < num_groups; s++) {
            if (r == s) {
                // Within-group edges (undirected, so divide by 2)
                H_M += log_neg_binom(edge_counts[r][r] / 2.0,
                                     density_in * node_counts[r] * node_counts[r] / 2.0,
                                     concentration_in);
            } else {
                // Between-group edges
                // Use symmetry_factor to avoid double-counting when both r and s are impacted
                double symmetry_factor = 1;
                if (initial_impacted_groups.find(s) != initial_impacted_groups.end()) {
                    symmetry_factor = 0.5;
                }
                H_M +=
                    symmetry_factor * log_neg_binom(edge_counts[r][s],
                                                    density_out * node_counts[r] * node_counts[s],
                                                    concentration_out);
            }
        }
    }

    // H(A|k,M,b): Adjacency matrix entropy given edge counts
    for (int r : initial_impacted_groups) {
        for (int s = 0; s < num_groups; s++) {
            if (r == s) {
                H_A_G_k_M_b += log_factorial2(edge_counts[r][r]);
            } else {
                // If s is also in initial_impacted_groups, only add half the term to avoid double counting
                double symmetry_factor = 1;
                if (initial_impacted_groups.find(s) != initial_impacted_groups.end()) {
                    symmetry_factor = 0.5;
                }
                H_A_G_k_M_b += symmetry_factor * log_factorial(edge_counts[r][s]);
            }
        }
        H_A_G_k_M_b -= log_factorial(edge_counts_sum[r]);
    }

    // H_W: remove per-block-pair contributions of affected groups (weighted SBM only)
    double nonzero_removed = 0.0;  // impacted block pairs with edges, before the move
    if (weighted_SBM) {
        H_W -= H_W_totals(nonzero_pairs);
        for (int r : initial_impacted_groups) {
            H_W -= H_W_pair(r, r);
            if (edge_counts[r][r] > 0) nonzero_removed += 1.0;
            for (int s = 0; s < num_groups; s++) {
                if (s == r) continue;
                double sym = initial_impacted_groups.count(s)
                                 ? 0.5
                                 : 1.0;  // pairs of two impacted groups are visited twice
                H_W -= sym * H_W_pair(r, s);
                if (edge_counts[r][s] > 0) nonzero_removed += sym;
            }
        }
    }

    // H(s|b,depths): hierarchy-community entropy
    for (int r : initial_impacted_groups) {
        H_s_G_b_depths -= score_prior_group_term(r);
    }

    /* === PHASE 2: Update partition and bookkeeping === */
    for (int node : filtered_nodes) {
        // Remove node from old_group using swap-and-pop (O(1) removal)
        size_t node_index = node_indices_in_groups[old_group][node];
        int last_node = group_members[old_group].back();
        group_members[old_group][node_index] = last_node;
        node_indices_in_groups[old_group][last_node] = node_index;
        group_members[old_group].pop_back();
        node_indices_in_groups[old_group].erase(node);
        node_counts[old_group]--;

        // Update edge counts: subtract edges from old_group (both directions: the
        // constructor rejects self-loops, so j != node).
        for (const auto& [j, A_ij] : neighbors_com[node]) {
            int neighbor_group = partition[j];
            int count_delta = (weighted_SBM || two_layer_mode) ? 1 : static_cast<int>(A_ij);
            edge_counts[old_group][neighbor_group] -= count_delta;
            edge_counts_sum[old_group] -= count_delta;
            edge_counts[neighbor_group][old_group] -= count_delta;
            edge_counts_sum[neighbor_group] -= count_delta;
            if (weighted_SBM) {
                weight_counts[old_group][neighbor_group] -= A_ij;
                weight_counts_sum[old_group] -= A_ij;
                weight_counts[neighbor_group][old_group] -= A_ij;
                weight_counts_sum[neighbor_group] -= A_ij;
            }
        }

        partition[node] = new_group;

        group_score_sums[old_group] -= scores[node];

        // If creating a new group, resize all data structures
        if (new_group == num_groups) {
            group_members.push_back(std::vector<int>());
            node_indices_in_groups.push_back(std::unordered_map<int, size_t>());
            node_counts.push_back(0);
            edge_counts_sum.push_back(0);
            edge_counts.push_back(std::vector<int>(num_groups, 0));
            for (auto& row : edge_counts) {
                row.push_back(0);
            }
            if (weighted_SBM) {
                weight_counts_sum.push_back(0.0);
                weight_counts.push_back(std::vector<double>(num_groups, 0.0));
                for (auto& row : weight_counts) {
                    row.push_back(0.0);
                }
            }
            group_score_sums.push_back(0.0);
            num_groups += 1;
        }

        // Add node to new_group
        group_members[new_group].push_back(node);
        node_indices_in_groups[new_group][node] = group_members[new_group].size() - 1;
        node_counts[new_group]++;

        // Update edge counts: add edges to new_group.
        for (const auto& [j, A_ij] : neighbors_com[node]) {
            int neighbor_group = partition[j];
            int count_delta = (weighted_SBM || two_layer_mode) ? 1 : static_cast<int>(A_ij);
            edge_counts[new_group][neighbor_group] += count_delta;
            edge_counts_sum[new_group] += count_delta;
            edge_counts[neighbor_group][new_group] += count_delta;
            edge_counts_sum[neighbor_group] += count_delta;
            if (weighted_SBM) {
                weight_counts[new_group][neighbor_group] += A_ij;
                weight_counts_sum[new_group] += A_ij;
                weight_counts[neighbor_group][new_group] += A_ij;
                weight_counts_sum[neighbor_group] += A_ij;
            }
        }

        group_score_sums[new_group] += scores[node];
    }

    // If old_group is now empty, compact by moving last group to fill the gap
    if (empty_old_group) {
        int last_group = num_groups - 1;
        if (old_group != last_group) {
            swap_groups(old_group, last_group);
        }

        // Remove last group's data structures
        group_members.pop_back();
        node_indices_in_groups.pop_back();
        node_counts.pop_back();
        edge_counts_sum.pop_back();
        edge_counts.pop_back();  // Remove last row
        for (auto& row : edge_counts) {
            row.pop_back();  // Remove last column from all rows
        }
        if (weighted_SBM) {
            weight_counts.pop_back();
            for (auto& row : weight_counts) {
                row.pop_back();
            }
            weight_counts_sum.pop_back();
        }
        group_score_sums.pop_back();
        num_groups -= 1;
    }

    /* === PHASE 3: Add entropy terms from final configuration === */

    // Enforce the non-assortative constraint
    if (!assortative) {
        density_out = density_in;
    }

    // H(nb|q): Partition count with new num_groups
    H_b += log_binom(n - 1, num_groups - 1);

    // H(b|nb): Group size distribution with new partition
    for (int r : final_impacted_groups) {
        H_b -= log_factorial(node_counts[r]);
    }

    // H(k): Degree sequence entropy with new configuration
    for (int r : final_impacted_groups) {
        if (degree_correction_concentration < MAX_CONCENTRATION) {
            H_k +=
                log_binom(edge_counts_sum[r] + node_counts[r] * degree_correction_concentration - 1,
                          node_counts[r] * degree_correction_concentration - 1);
        } else {
            // Multinomial approximation for large concentration
            H_k -= log_factorial(edge_counts_sum[r]);
            H_k += edge_counts_sum[r] * std::log(node_counts[r]);
        }
    }

    // H(M): Edge count distribution with new configuration
    for (int r : final_impacted_groups) {
        for (int s = 0; s < num_groups; s++) {
            if (r == s) {
                H_M -= log_neg_binom(edge_counts[r][r] / 2.0,
                                     density_in * node_counts[r] * node_counts[r] / 2.0,
                                     concentration_in);
            } else {
                // Use symmetry_factor to avoid double-counting
                double symmetry_factor = 1;
                if (final_impacted_groups.find(s) != final_impacted_groups.end()) {
                    symmetry_factor = 0.5;
                }
                H_M -=
                    symmetry_factor * log_neg_binom(edge_counts[r][s],
                                                    density_out * node_counts[r] * node_counts[s],
                                                    concentration_out);
            }
        }
    }

    // H(A|k,M,b): Adjacency matrix entropy with new configuration
    for (int r : final_impacted_groups) {
        for (int s = 0; s < num_groups; s++) {
            if (r == s) {
                H_A_G_k_M_b -= log_factorial2(edge_counts[r][r]);
            } else {
                // If s is also in initial_impacted_groups, only add half the term to avoid double counting
                double symmetry_factor = 1;
                if (final_impacted_groups.find(s) != final_impacted_groups.end()) {
                    symmetry_factor = 0.5;
                }
                H_A_G_k_M_b -= symmetry_factor * log_factorial(edge_counts[r][s]);
            }
        }
        H_A_G_k_M_b += log_factorial(edge_counts_sum[r]);
    }

    // H_W: add per-block-pair contributions of affected groups in new configuration (weighted SBM only)
    if (weighted_SBM) {
        double nonzero_added = 0.0;
        for (int r : final_impacted_groups) {
            H_W += H_W_pair(r, r);
            if (edge_counts[r][r] > 0) nonzero_added += 1.0;
            for (int s = 0; s < num_groups; s++) {
                if (s == r) continue;
                double sym = final_impacted_groups.count(s) ? 0.5 : 1.0;
                H_W += sym * H_W_pair(r, s);
                if (edge_counts[r][s] > 0) nonzero_added += sym;
            }
        }
        nonzero_pairs += static_cast<int>(std::lround(nonzero_added - nonzero_removed));
        H_W += H_W_totals(nonzero_pairs);
    }

    // H(s|b,depths): hierarchy-community entropy with new configuration
    for (int r : final_impacted_groups) {
        H_s_G_b_depths += score_prior_group_term(r);
    }

    // Update total entropies: add delta from component changes.
    // H_W = -log P (description-length contribution, negative when model fits well), so add its delta.
    H_A_G_b +=
        (H_k - H_k_old) + (H_M - H_M_old) + (H_A_G_k_M_b - H_A_G_k_M_b_old) + (H_W - H_W_old);
    H_A_b += (H_b - H_b_old) + (H_k - H_k_old) + (H_M - H_M_old) + (H_A_G_k_M_b - H_A_G_k_M_b_old) +
             (H_W - H_W_old);
}

// Exchange the labels of groups group1 and group2: members, counts, the rows
// and columns of edge_counts (and weight_counts), score sums and partition[].
// No entropy changes -- every term is symmetric under relabelling.
void MCMC::swap_groups(int group1, int group2) {
    // Swap entire group structures (O(1) pointer swap)
    std::swap(group_members[group1], group_members[group2]);
    std::swap(node_indices_in_groups[group1], node_indices_in_groups[group2]);
    std::swap(node_counts[group1], node_counts[group2]);

    // Swap edge_counts matrix rows and columns
    for (int s = 0; s < num_groups; s++) {
        std::swap(edge_counts[group1][s], edge_counts[group2][s]);
    }
    for (int r = 0; r < num_groups; r++) {
        std::swap(edge_counts[r][group1], edge_counts[r][group2]);
    }
    std::swap(edge_counts_sum[group1], edge_counts_sum[group2]);

    // Swap weight_counts matrix rows and columns (weighted mode only)
    if (weighted_SBM) {
        for (int s = 0; s < num_groups; s++) {
            std::swap(weight_counts[group1][s], weight_counts[group2][s]);
        }
        for (int r = 0; r < num_groups; r++) {
            std::swap(weight_counts[r][group1], weight_counts[r][group2]);
        }
        std::swap(weight_counts_sum[group1], weight_counts_sum[group2]);
    }

    // Swap group score sums
    std::swap(group_score_sums[group1], group_score_sums[group2]);

    // Update partition references for moved nodes
    for (int node : group_members[group1]) {
        partition[node] = group1;
    }
    for (int node : group_members[group2]) {
        partition[node] = group2;
    }
}