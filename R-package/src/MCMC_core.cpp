// MCMC constructor, MCMC_sweep and change_parameter.  Moves live in MCMC_proposals.cpp and
// MCMC_merge_split.cpp; entropy bookkeeping in MCMC_bookkeeping.cpp; the multi-chain driver
// (incl. parallel tempering) in fit.cpp.  See docs/ARCHITECTURE.md.

#include "MCMC_core.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include "globals.h"
#include "helpers.h"

/**
 * MCMC Constructor
 * 
 * Initializes MCMC sampler with graph structure, partition, and model parameters.
 * Sets up all bookkeeping data structures for efficient entropy calculation updates.
 * 
 * @param wins: Adjacency list for dominant edges (wins[i][j] = weight of dominant edge from i to j)
 * @param ties: Adjacency list for neutral edges.  An undirected tie must be given in
 *              BOTH directions with the same weight (ties[i][j] == ties[j][i]).
 * @param group_adj: Optional separate undirected graph driving the communities (two-layer
 *                   mode); empty = the communities are fitted to wins + ties
 * @param partition, partition_fixed: Initial group assignment, labels 0..q-1; fixed = no
 *                   partition moves
 * @param scores, scores_fixed: Initial hierarchy scores; fixed = no move changes them
 *                   (score, group-shift and the score-rescaling depth moves are off)
 * @param num_groups, num_groups_fixed: q is ALWAYS taken from the partition
 *                   (max label + 1).  num_groups is only checked: with
 *                   num_groups_fixed it must equal the partition's q (else
 *                   std::invalid_argument); otherwise it is ignored.  fixed = q never
 *                   changes.
 * @param variation_in/out: Overdispersion of within/between-group edge counts
 * @param degree_correction: Degree correction (0 = none, 0.5 = uniform over degree sequences)
 * @param individual_depth, group_depth: spread of scores within groups / of group means
 * @param ties_parameter: frequency of ties in the direction model.  Fixed at 0 means
 *        plain Bradley-Terry: neutral edges still count in the SBM (H_A_G_b), but are
 *        left out of the direction likelihood H_Adir_G_A_s_nu, where a tie would have
 *        probability 0.
 * @param *_fixed: Flags for fixed (not inferred) parameters
 * @param seed: >= 0 reseeds this thread's generator (see globals.h); < 0 leaves
 *        it as is -- fit() seeds it once per chain and builds every tempering
 *        replica with -1, so the replicas share one stream seeded by seed + chain.
 * @param density_in/out_initial: < 0 = estimate from the initial partition
 * @param weighted_SBM: model real-valued edge weights (uniform simplex prior)
 *
 * What the weights mean:
 *   - plain mode: edge MULTIPLICITIES, non-negative integers.  The SBM sees the
 *     multigraph wins + ties; the direction model counts w_ij wins of i over j.
 *   - weighted_SBM: the SBM sees the binary graph (one edge per neighbour pair) and
 *     H_W models the real weights, which must be finite and > 0.
 *   - two-layer (group_adj non-empty): the SBM sees group_adj as a binary graph;
 *     wins/ties only drive the scores.  All weights must be finite and > 0.
 * Self-loops are rejected in every mode (std::invalid_argument).  Peixoto's
 * convention would store a self-loop as A_ii = 2 x its multiplicity, with a
 * log A_ii!! term in H_A_G_k_M_b; the direction model has no meaning for i vs i.
 */
MCMC::MCMC(const std::vector<std::unordered_map<int, double>>& wins_in,
           const std::vector<std::unordered_map<int, double>>& ties_in,
           const std::vector<std::unordered_map<int, double>>& group_adj,
           const std::vector<int>& partition, const bool& partition_fixed,
           const std::vector<double>& scores, const bool& scores_fixed, int num_groups,
           const bool& num_groups_fixed, const bool& assortative, const double& variation_in,
           const bool& variation_in_fixed, const double& variation_out,
           const bool& variation_out_fixed, const double& degree_correction,
           const bool& degree_correction_fixed, const double& individual_depth,
           const bool& individual_depth_fixed, const double& group_depth,
           const bool& group_depth_fixed, const double& ties_parameter,
           const bool& ties_parameter_fixed, int seed, double density_in_initial,
           double density_out_initial, bool weighted_SBM)
    : n(0),
      m(0),
      wins(),
      losses(),
      ties(),
      neighbors(),
      neighbors_com(),
      two_layer_mode(false),
      weighted_SBM(weighted_SBM),
      free_parameters(),
      beta(0.0),  // Inverse temperature (set during MCMC_sweep)
      proposal_weights(),
      partition(partition),        // Copy initial partition from const reference
      scores(scores),              // Copy initial scores from const reference
      variation_in(variation_in),  // Copy variation parameters (stored as values)
      variation_out(variation_out),
      degree_correction(degree_correction),
      individual_depth(individual_depth),
      group_depth(group_depth),
      ties_parameter(ties_parameter),
      partition_fixed(partition_fixed),
      scores_fixed(scores_fixed),
      variation_in_fixed(variation_in_fixed),
      variation_out_fixed(variation_out_fixed),
      degree_correction_fixed(degree_correction_fixed),
      num_groups_fixed(num_groups_fixed),
      individual_depth_fixed(individual_depth_fixed),
      group_depth_fixed(group_depth_fixed),
      ties_parameter_fixed(ties_parameter_fixed),
      assortative(assortative),
      H_b(0.0),
      H_M(0.0),
      H_k(0.0),
      H_A_G_k_M_b(0.0),
      H_W(0.0),
      H_A_G_b(0.0),
      H_A_b(0.0),
      num_groups(num_groups),
      node_counts(),             // Will be resized and computed
      group_members(),           // group_members[r] = vector of node IDs in group r
      node_indices_in_groups(),  // node_indices_in_groups[r][node] = position in group_members[r]
      edge_counts(),             // edge_counts[r][s] = total edge weight from group r to s
      edge_counts_sum(),         // edge_counts_sum[r] = total degree of group r
      w_total(0.0),
      group_score_sums(),     // Sum of the scores in each group
      concentration_in(0.0),  // Negative binomial concentration parameters
      concentration_out(0.0),
      degree_correction_concentration(0.0),
      density_in(0.0),
      density_out(0.0),
      single_site_proposals(0),
      accepted_single_site_proposals(0) {
    // Seed random number generator for reproducibility (negative: keep the stream)
    if (seed >= 0) rng.seed(static_cast<std::mt19937::result_type>(seed));

    // degree_correction is a variation in [0, 1): 1 maps to Dirichlet concentration
    // 0, where H_k is undefined (NaN).  The API only ever uses 0 or 0.5.
    if (!(degree_correction >= 0.0 && degree_correction < 1.0))
        throw std::invalid_argument("degree_correction must lie in [0, 1), got " +
                                    std::to_string(degree_correction));

    // === Initialize graph information ===
    n = static_cast<int>(this->partition.size());  // Number of nodes in the graph
    if (n == 0)
        throw std::invalid_argument("the partition is empty: the network needs at least one node");
    if (static_cast<int>(wins_in.size()) != n || static_cast<int>(ties_in.size()) != n ||
        static_cast<int>(scores.size()) != n ||
        (!group_adj.empty() && static_cast<int>(group_adj.size()) != n))
        throw std::invalid_argument(
            "wins, ties, scores (and group_adj, if given) must have one entry per node of the "
            "partition");
    for (int i = 0; i < n; ++i)
        if (this->partition[static_cast<std::size_t>(i)] < 0)
            throw std::invalid_argument("partition labels must be >= 0");

    // Input validation, before anything is built from the edges.
    //   * node ids in range;
    //   * no self-loops in any mode: wins and ties count a self-loop differently
    //     (twice vs once), the binary modes would make e_rr odd, and the direction
    //     model has no meaning for a node against itself;
    //   * weights finite; > 0 in weighted_SBM and two-layer mode (a zero weight
    //     there still counts as one binary edge, and H_W needs log mu > -inf);
    //     the plain mode's integer check is below, on the combined graph;
    //   * ties symmetric (each undirected tie given in both directions).
    two_layer_mode = !group_adj.empty();
    const bool positive_weights = this->weighted_SBM || two_layer_mode;
    auto validate = [&](const std::vector<std::unordered_map<int, double>>& adj, const char* name) {
        for (int i = 0; i < n; ++i) {
            for (const auto& [j, w] : adj[static_cast<std::size_t>(i)]) {
                const std::string where = std::string(name) + " edge (" + std::to_string(i) + ", " +
                                          std::to_string(j) + ")";
                if (j < 0 || j >= n)
                    throw std::invalid_argument(where + ": node id out of range [0, " +
                                                std::to_string(n) + ")");
                if (j == i) throw std::invalid_argument(where + ": self-loops are not supported");
                if (!std::isfinite(w))
                    throw std::invalid_argument(where + " has weight " + std::to_string(w) +
                                                "; edge weights must be finite");
                if (positive_weights && !(w > 0.0))
                    throw std::invalid_argument(where + " has weight " + std::to_string(w) +
                                                "; with weighted_SBM or a separate group graph "
                                                "every edge weight must be finite and > 0");
            }
        }
    };
    validate(wins_in, "wins");
    validate(ties_in, "ties");
    if (two_layer_mode) validate(group_adj, "group_adj");
    for (int i = 0; i < n; ++i) {
        for (const auto& [j, w] : ties_in[static_cast<std::size_t>(i)]) {
            const auto& back = ties_in[static_cast<std::size_t>(j)];
            const auto it = back.find(i);
            if (it == back.end() || it->second != w)
                throw std::invalid_argument(
                    "ties edge (" + std::to_string(i) + ", " + std::to_string(j) +
                    "): ties must be given in both directions with the same weight");
        }
    }

    // Built as maps (accumulating duplicate entries), then stored SORTED: the
    // members are summed over, and unordered_map iteration order differs between
    // standard libraries, which would make a seeded run diverge between the
    // native and wasm builds.  See rng.h.
    std::vector<std::unordered_map<int, double>> losses_map(static_cast<std::size_t>(n));
    std::vector<std::unordered_map<int, double>> neighbors_map(static_cast<std::size_t>(n));

    // Process wins: wins[i][j] means i dominates j
    // This creates a loss for j: losses[j][i]
    for (int i = 0; i < n; i++) {
        for (const auto& edge : wins_in[static_cast<std::size_t>(i)]) {
            int j = edge.first;
            double weight = edge.second;
            neighbors_map[static_cast<std::size_t>(i)][j] += weight;  // Add to undirected neighbors
            neighbors_map[static_cast<std::size_t>(j)][i] += weight;  // Add to undirected neighbors
            losses_map[static_cast<std::size_t>(j)][i] += weight;     // j loses to i
        }
    }

    // Process ties: symmetric edges (both directions are in ties_in)
    for (int i = 0; i < n; i++) {
        for (const auto& edge : ties_in[static_cast<std::size_t>(i)]) {
            int j = edge.first;
            double weight = edge.second;
            neighbors_map[static_cast<std::size_t>(i)][j] +=
                weight;  // other direction will be present
            // Ties don't contribute to wins/losses asymmetry
        }
    }

    // Freeze every adjacency list into sorted order.
    auto to_sorted = [](const std::vector<std::unordered_map<int, double>>& src, int rows) {
        std::vector<std::vector<std::pair<int, double>>> out(static_cast<std::size_t>(rows));
        for (int i = 0; i < rows; ++i) {
            auto& row = out[static_cast<std::size_t>(i)];
            row.assign(src[static_cast<std::size_t>(i)].begin(),
                       src[static_cast<std::size_t>(i)].end());
            std::sort(row.begin(), row.end(),
                      [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                          return a.first < b.first;
                      });
        }
        return out;
    };
    wins = to_sorted(wins_in, n);
    ties = to_sorted(ties_in, n);
    losses = to_sorted(losses_map, n);
    neighbors = to_sorted(neighbors_map, n);
    neighbors_com = two_layer_mode ? to_sorted(group_adj, n) : neighbors;

    // Plain Bradley-Terry (ties_parameter fixed at 0): a tie has probability 0, so
    // neutral edges cannot be outcomes of the direction model.  They stay in
    // neighbors (hence in the SBM, unless group_adj replaces it) and are dropped
    // from the direction likelihood only.
    ties_in_direction_model = !(ties_parameter_fixed && ties_parameter == 0.0);
    if (!ties_in_direction_model)
        for (auto& row : ties) row.clear();

    // Constant part of H_Adir_G_A_s_nu: for each unordered pair {i, j} with
    // outcome counts w_ij, w_ji (and t_ij), the multinomial coefficient
    // A^dir_ij! / (w_ij! w_ji! t_ij!), A^dir_ij = w_ij + w_ji (+ t_ij).
    // log_factorial_real: real weights (weighted_SBM) are not truncated.
    H_Adir_counts_term = 0.0;
    for (int i = 0; i < n; ++i) {
        for (const auto& [j, w] : wins[static_cast<std::size_t>(i)])
            H_Adir_counts_term += log_factorial_real(w);
        for (const auto& [j, t] : ties[static_cast<std::size_t>(i)])
            if (i < j) H_Adir_counts_term += log_factorial_real(t);
        // A^dir_ij over the pairs i < j: merge i's sorted wins, losses and ties rows.
        const auto &W = wins[static_cast<std::size_t>(i)],
                   &Lo = losses[static_cast<std::size_t>(i)],
                   &T = ties[static_cast<std::size_t>(i)];
        std::size_t a = 0, b = 0, c = 0;
        while (a < W.size() || b < Lo.size() || c < T.size()) {
            int j = std::numeric_limits<int>::max();
            if (a < W.size()) j = std::min(j, W[a].first);
            if (b < Lo.size()) j = std::min(j, Lo[b].first);
            if (c < T.size()) j = std::min(j, T[c].first);
            double A = 0.0;
            if (a < W.size() && W[a].first == j) A += W[a++].second;
            if (b < Lo.size() && Lo[b].first == j) A += Lo[b++].second;
            if (c < T.size() && T[c].first == j) A += T[c++].second;
            if (i < j) H_Adir_counts_term -= log_factorial_real(A);
        }
    }

    // Block-informed proposals use binary edge counts unless the caller switches
    // weighted_proposals on afterwards (weighted_SBM only; see block_move_prob).
    // Off by default, the same default as FitParams::weighted_proposals.
    this->weighted_proposals = false;

    // Outside weighted_SBM / two-layer mode the weights are edge MULTIPLICITIES:
    // edge_counts truncates them (int +=) while node_degrees rounds them, so a
    // non-integer weight would make the two describe different graphs.  Reject
    // it.  (weighted_SBM and two-layer mode count edges as binary and are fine.)
    if (!this->weighted_SBM && !this->two_layer_mode) {
        for (int i = 0; i < n; ++i) {
            for (const auto& edge : this->neighbors_com[i]) {
                const double w = edge.second;
                if (!std::isfinite(w) || w < 0.0 || std::abs(w - std::round(w)) > 1e-9) {
                    throw std::invalid_argument(
                        "edge (" + std::to_string(i) + ", " + std::to_string(edge.first) +
                        ") has weight " + std::to_string(w) +
                        "; edge weights must be non-negative integers unless weighted_SBM is used");
                }
            }
        }
    }

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

    // === Initialize bookkeeping data structures ===
    // q comes from the partition (max label + 1).  The num_groups argument is
    // only checked: a fixed q that disagrees with the partition used to be
    // silently replaced by the partition's q.
    const int q_partition = *std::max_element(this->partition.begin(), this->partition.end()) + 1;
    if (num_groups_fixed && num_groups > 0 && num_groups != q_partition)
        throw std::invalid_argument(
            "num_groups_fixed with num_groups = " + std::to_string(num_groups) +
            ", but the initial partition has " + std::to_string(q_partition) + " groups");
    num_groups = q_partition;
    node_counts.resize(num_groups, 0);          // node_counts[r] = number of nodes in group r
    group_members.resize(num_groups);           // group_members[r] = vector of node IDs in group r
    node_indices_in_groups.resize(num_groups);  // Fast lookup: node -> position in group_members
    edge_counts.resize(num_groups,
                       std::vector<int>(num_groups, 0));  // edge_counts[r][s] = edges r->s
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
        free_parameters.push_back("density_in");  // density_out == density_in when non-assortative
    }

    // Sentinel: -1 means "initialize from data"; use explicit values if provided.
    density_in = density_in_initial;
    density_out = (density_out_initial >= 0.0) ? density_out_initial : density_in_initial;

    // Run the initial bookkeeping without throwing errors to set initial values
    check_state_bookkeeping(false);
}

/**
 * MCMC_sweep: Perform num_sweeps MCMC sweeps, then check the bookkeeping
 *
 * A sweep consists of n proposal attempts (one per node on average).
 * Each proposal randomly selects a move type weighted by proposal_weights,
 * then accepts/rejects based on Metropolis-Hastings criterion.  Each sweep then
 * ends with one group-score shift per group (unless scores_fixed), a
 * deterministic composition of reversible kernels.  check_state_bookkeeping(true)
 * runs once at the end.
 *
 * @param beta: Inverse temperature (higher = more selective), defaults to 1.0
 * @param num_sweeps: Number of sweeps to run, defaults to 1
 * @param merge_split_enabled: include the collective merge/split moves
 */
SampleState MCMC::snapshot(int chain_id) const {
    SampleState s;
    s.chain_id = chain_id;
    s.beta = beta;
    s.partition = partition;
    s.scores = scores;
    s.num_groups = num_groups;
    s.density_in = density_in;
    s.density_out = density_out;
    s.variation_in = variation_in;
    s.variation_out = variation_out;
    s.degree_correction = degree_correction;
    s.individual_depth = individual_depth;
    s.group_depth = group_depth;
    s.ties_parameter = ties_parameter;
    s.H_A_b = H_A_b;
    s.H_b = H_b;
    s.H_M = H_M;
    s.H_k = H_k;
    s.H_A_G_k_M_b = H_A_G_k_M_b;
    s.H_W = H_W;
    s.H_A_G_b = H_A_G_b;
    s.H_Adir_G_A_s_nu = H_Adir_G_A_s_nu;
    s.H_s_G_b_depths = H_s_G_b_depths;
    s.two_layer_mode = two_layer_mode;
    s.weighted_SBM = weighted_SBM;
    return s;
}

void MCMC::MCMC_sweep(double beta, int num_sweeps, bool merge_split_enabled) {
    this->beta = beta;

    // Set proposal weights.
    // NB: a sqrt(n) boost here was tried (to propose depths more often) and removed:
    // it broke synthetic recoverability (steals proposals from free-partition moves)
    // and does NOT help group_depth mixing -- the bottleneck is ridge traversal, not
    // proposal frequency (single-variable depth moves can't climb the sigma_I/sigma_C
    // ridge). The real fix is a joint (sigma_I, sigma_C) block proposal.
    proposal_weights.clear();
    // Each free parameter at weight PARAMETER_MOVE_WEIGHT (globals.h): about
    // 10 proposals per parameter per sweep, cheap next to the n node moves.
    proposal_weights.emplace_back(
        "parameter_move", PARAMETER_MOVE_WEIGHT * static_cast<double>(free_parameters.size()));
    if (!partition_fixed) {
        proposal_weights.emplace_back("single_site_move", static_cast<double>(n));
    }
    // Collective merge/split moves (Peixoto Eq. 37).  Weights are CONSTANT --
    // deliberately NOT proportional to num_groups, for the same reason the
    // group-score move uses a constant: a state-dependent weight makes the
    // partition-move fraction depend on q and biases the q-marginal.
    // MERGE_SPLIT_MOVES selects which of the three to include (testing dial).
    if (!partition_fixed && merge_split_enabled) {
        // See MERGE_SPLIT_WEIGHT in globals.h for why this is not simply 1.0.
        double w =
            collective_weight_override >= 0.0 ? collective_weight_override : MERGE_SPLIT_WEIGHT;
        if (w < 0.0) {
            const double denom =
                static_cast<double>(MERGE_SPLIT_GIBBS_SWEEPS > 0 ? MERGE_SPLIT_GIBBS_SWEEPS : 1) *
                static_cast<double>(n + m);
            w = denom > 0.0 ? static_cast<double>(n) / denom : 1.0;
        }
        if (MERGE_SPLIT_MOVES & 1) proposal_weights.emplace_back("merge_move", w);
        if (MERGE_SPLIT_MOVES & 2) proposal_weights.emplace_back("split_move", w);
        if (MERGE_SPLIT_MOVES & 4) proposal_weights.emplace_back("merge_split_move", w);
    }
    if (!scores_fixed) {
        proposal_weights.emplace_back("score_move", static_cast<double>(n));
        // The group-score shift is NOT in the random scan: each sweep ends with
        // one shift per group (below).
    }

    for (int sweep = 0; sweep < num_sweeps; sweep++) {
        for (int i = 0; i < n; i++) {
            propose_move();
        }
        // Group-score block: one shift per group.  Collective group-score shifts
        // traverse the sigma_C ridge that per-node and single-depth moves cannot.
        // This is a COMPOSITION of kernels, not a mixture: each shift is
        // reversible with respect to the target and leaves the partition -- hence
        // q -- unchanged, so sweeping over the current q groups keeps the target
        // invariant with no selection-probability correction (a random-scan weight
        // proportional to q would bias the q-marginal).  Cost O(sum of group
        // degrees) = O(m) per sweep with the fast path in propose_group_score_move.
        if (!scores_fixed && num_groups > 1)
            for (int r = 0; r < num_groups; r++) propose_group_score_move(r);
    }
    check_state_bookkeeping(true);
}

/**
 * change_parameter: one Metropolis-Hastings step of a continuous parameter
 *
 * new_value is a PROPOSAL, drawn by the caller in the CDF space of the
 * parameter's prior (propose_parameter_move), so the prior cancels and is not in
 * the acceptance.  The value may be rejected: this is not a setter.
 *
 * Implements Metropolis-Hastings acceptance:
 * 1. Store old values
 * 2. Update parameter to new_value
 * 3. Recalculate affected entropy components
 * 4. Accept with probability exp(-beta * delta_H)
 * 5. If rejected, restore old values
 * 
 * Different parameters affect different entropies:
 * - degree_correction -> H_k (degree sequence)
 * - variation_in, variation_out, density_in, density_out -> H_M (edge counts)
 * - individual_depth -> H_s_G_b_depths and H_Adir_G_A_s_nu (every score scales with
 *   sigma_I); with scores_fixed, H_s_G_b_depths only (the scores are held fixed)
 * - group_depth -> H_s_G_b_depths
 * - ties_parameter -> H_Adir_G_A_s_nu
 * Tempering: only the likelihood terms are multiplied by beta; H_s_G_b_depths is
 * the score prior and is not (see MCMC_core.h).
 * 
 * @param param_name: Name of parameter to change
 * @param new_value: Proposed new value
 */
void MCMC::change_parameter(const std::string& param_name, double new_value) {
    if (param_name == "degree_correction") {
        double degree_correction_old = degree_correction;  // Store for potential rollback
        double H_k_old = H_k;                              // Store old entropy component

        // Apply proposed change
        degree_correction = new_value;
        degree_correction_concentration = variation_to_concentration(degree_correction);
        H_k = calculate_H_k();  // Recalculate affected entropy

        // Metropolis-Hastings acceptance
        double delta_H = H_k - H_k_old;
        double acceptance_prob = std::exp(-beta * delta_H);
        if (uniform_random(rng) < acceptance_prob) {
            // Accept: update total entropies
            H_A_G_b += (H_k - H_k_old);
            H_A_b += (H_k - H_k_old);
        } else {
            // Reject: restore old values
            degree_correction = degree_correction_old;
            degree_correction_concentration = variation_to_concentration(degree_correction);
            H_k = H_k_old;
        }
    } else if (param_name == "variation_in") {
        double variation_in_old = variation_in;
        double H_M_old = H_M;

        // Apply proposed change
        variation_in = new_value;
        concentration_in = variation_to_concentration(variation_in);
        H_M = calculate_H_M();  // Recalculate edge count entropy

        // Metropolis-Hastings acceptance
        double delta_H = H_M - H_M_old;
        double acceptance_prob = std::exp(-beta * delta_H);
        if (uniform_random(rng) < acceptance_prob) {
            H_A_G_b += (H_M - H_M_old);
            H_A_b += (H_M - H_M_old);
        } else {
            // Reject: restore old values
            variation_in = variation_in_old;
            concentration_in = variation_to_concentration(variation_in);
            H_M = H_M_old;
        }
    } else if (param_name == "variation_out") {
        double variation_out_old = variation_out;
        double H_M_old = H_M;

        // Apply proposed change
        variation_out = new_value;
        concentration_out = variation_to_concentration(variation_out);
        H_M = calculate_H_M();  // Recalculate edge count entropy

        // Metropolis-Hastings acceptance
        double delta_H = H_M - H_M_old;
        double acceptance_prob = std::exp(-beta * delta_H);
        if (uniform_random(rng) < acceptance_prob) {
            H_A_G_b += (H_M - H_M_old);
            H_A_b += (H_M - H_M_old);
        } else {
            // Reject: restore old values
            variation_out = variation_out_old;
            concentration_out = variation_to_concentration(variation_out);
            H_M = H_M_old;
        }
    } else if (param_name == "density_in") {
        double density_in_old = density_in;
        double density_out_old = density_out;
        double H_M_old = H_M;

        density_in = new_value;
        if (!assortative) density_out = density_in;
        H_M = calculate_H_M();

        double delta_H = H_M - H_M_old;
        if (uniform_random(rng) < std::exp(-beta * delta_H)) {
            H_A_G_b += delta_H;
            H_A_b += delta_H;
        } else {
            density_in = density_in_old;
            density_out = density_out_old;
            H_M = H_M_old;
        }
    } else if (param_name == "density_out") {
        // Non-assortative: density_out is tied to density_in and moves with it.
        if (!assortative) return;
        double density_out_old = density_out;
        double H_M_old = H_M;

        density_out = new_value;
        H_M = calculate_H_M();

        double delta_H = H_M - H_M_old;
        if (uniform_random(rng) < std::exp(-beta * delta_H)) {
            H_A_G_b += delta_H;
            H_A_b += delta_H;
        } else {
            density_out = density_out_old;
            H_M = H_M_old;
        }
    } else if (param_name == "individual_depth" && scores_fixed) {
        // Scores fixed: the centred kernel (scores held, no Jacobian).  The
        // likelihood depends on the scores only, so only the untempered score prior
        // decides.  Same kernel as propose_individual_depth_centred.
        individual_depth_proposals++;
        const double individual_depth_old = individual_depth;
        const double H_s_G_b_depths_old = H_s_G_b_depths;
        individual_depth = new_value;
        H_s_G_b_depths = calculate_H_s_G_b_depths();
        if (std::log(uniform_random(rng)) < -(H_s_G_b_depths - H_s_G_b_depths_old)) {
            accepted_individual_depth_proposals++;
        } else {
            individual_depth = individual_depth_old;
            H_s_G_b_depths = H_s_G_b_depths_old;
        }
    } else if (param_name == "individual_depth") {
        // Every score scales with sigma_I: s -> lambda s, lambda = sigma_I' / sigma_I,
        // a deterministic map on the n scores with Jacobian lambda^n.  (Formerly
        // "scaled scores s / sigma_I held fixed"; the same kernel.)
        individual_depth_proposals++;
        const double individual_depth_old = individual_depth,
                     lam = new_value / individual_depth_old;
        const double H_s_G_b_depths_old = H_s_G_b_depths;
        const double H_Adir_G_A_s_nu_old = H_Adir_G_A_s_nu;
        const std::vector<double> scores_old = scores, sums_old = group_score_sums;

        individual_depth = new_value;
        for (int i = 0; i < n; i++) scores[i] *= lam;
        for (auto& S : group_score_sums) S *= lam;
        H_s_G_b_depths = calculate_H_s_G_b_depths();
        H_Adir_G_A_s_nu = calculate_H_Adir_G_A_s_nu();

        // Only the direction likelihood is tempered; the score prior H_s_G_b_depths
        // enters at full weight (MCMC_core.h).  The prior on sigma_I cancels (CDF-space
        // proposal).
        const double log_acc = -((H_s_G_b_depths - H_s_G_b_depths_old) +
                                 beta * (H_Adir_G_A_s_nu - H_Adir_G_A_s_nu_old)) +
                               n * std::log(lam);
        if (std::log(uniform_random(rng)) < log_acc) {
            accepted_individual_depth_proposals++;
        } else {
            individual_depth = individual_depth_old;
            scores = scores_old;
            group_score_sums = sums_old;
            H_s_G_b_depths = H_s_G_b_depths_old;
            H_Adir_G_A_s_nu = H_Adir_G_A_s_nu_old;
        }
    } else if (param_name == "group_depth") {
        group_depth_proposals++;
        double group_depth_old = group_depth;
        double H_s_G_b_depths_old = H_s_G_b_depths;

        // Apply proposed change
        group_depth = new_value;
        H_s_G_b_depths = calculate_H_s_G_b_depths();  // Recalculate affected entropy

        // Metropolis-Hastings acceptance.  group_depth only enters the score prior
        // H_s_G_b_depths, which is not tempered (MCMC_core.h).
        double delta_H = H_s_G_b_depths - H_s_G_b_depths_old;
        double acceptance_prob = std::exp(-delta_H);
        if (uniform_random(rng) < acceptance_prob) {
            // Accept: update total entropies
            accepted_group_depth_proposals++;
        } else {
            // Reject: restore old values
            group_depth = group_depth_old;
            H_s_G_b_depths = H_s_G_b_depths_old;
        }
    } else if (param_name == "ties_parameter") {
        double ties_parameter_old = ties_parameter;
        double H_Adir_G_A_s_nu_old = H_Adir_G_A_s_nu;

        // Apply proposed change
        ties_parameter = new_value;
        H_Adir_G_A_s_nu = calculate_H_Adir_G_A_s_nu();  // Recalculate affected entropy

        // Metropolis-Hastings acceptance
        double delta_H = H_Adir_G_A_s_nu - H_Adir_G_A_s_nu_old;
        double acceptance_prob = std::exp(-beta * delta_H);
        if (uniform_random(rng) < acceptance_prob) {
            // Accept: update total entropies
        } else {
            // Reject: restore old values
            ties_parameter = ties_parameter_old;
            H_Adir_G_A_s_nu = H_Adir_G_A_s_nu_old;
        }
    } else {
    }
}

void MCMC::retune_collective_weight() {
    if (MERGE_SPLIT_WEIGHT >= 0.0) return;  // caller fixed it
    const double M =
        static_cast<double>(MERGE_SPLIT_GIBBS_SWEEPS > 0 ? MERGE_SPLIT_GIBBS_SWEEPS : 1);
    const double equal_spend = static_cast<double>(num_groups) / (MERGE_SPLIT_EQUAL_SPEND_C * M);
    const double linear_time = (n + m) > 0 ? static_cast<double>(n) / (M * (n + m)) : 1.0;
    collective_weight_override = std::max(equal_spend, linear_time);
}
