#ifndef FIT_H
#define FIT_H

#include <atomic>
#include <functional>
#include <optional>
#include <unordered_map>
#include <vector>
#include "MCMC_core.h"
#include "fit_result.h"

// All parameters controlling a single call to fit().
struct FitParams {
    std::vector<std::unordered_map<int, double>> wins;
    std::vector<std::unordered_map<int, double>> ties;
    std::vector<std::unordered_map<int, double>> group_adj;  // empty = single-layer mode
    bool weighted_SBM = false;
    bool weighted_proposals =
        false;  // weighted_SBM only: true = block-informed proposals follow the weights (experimental)

    // Empty = the default starting state (initial_state.h): greedy modularity, and
    // win fractions.  Given scores that are not fixed also set the starting sigma_I
    // to their RMS, unless individual_depth(_init) is set.
    std::vector<int> initial_partition;
    bool partition_fixed = false;
    std::vector<double> initial_scores;
    bool scores_fixed = false;

    bool assortative = true;

    std::optional<double> variation_in;
    std::optional<double> variation_out;
    std::optional<double> degree_correction;
    std::optional<double> individual_depth;
    std::optional<double> individual_depth_init;  // starting value only, not fixed
    std::optional<double> group_depth;
    std::optional<double> group_depth_init;  // starting value only, not fixed
    std::optional<double> ties_parameter;
    std::optional<int> num_groups;
    bool num_groups_fixed = false;

    // Run lengths, named as in PyStan.  One iteration is sweeps_per_sample sweeps.
    int num_chains = 1;
    int num_samples = 1000;  // sampling iterations per chain
    int num_thin = 1;        // record every num_thin-th: ceil(num_samples / num_thin) draws
    int num_warmup = 500;    // warm-up iterations; step sizes adapt here only
    int sweeps_per_sample = 10;
    // Iterations before warm-up with the partition and scores fixed, so that the
    // continuous parameters (variation_in/out, degree_correction,
    // individual_depth, group_depth, ties_parameter) settle on the initial
    // partition before it is free to move.
    int num_param_warmup = 0;
    double beta = 1.0;
    // Collective merge/split moves (Peixoto arXiv:2003.07070).
    // ON by default: without them single-site MCMC can converge silently to the
    // WRONG answer -- on the N=115 football network four of six chains get stuck
    // (R-hat 3.37, B_e 8.97 vs the true 10.25) while each chain looks perfectly
    // healthy from the inside.  Costs roughly 5x on networks that single-site
    // already handles, which buys correctness you otherwise cannot detect
    // without multi-start R-hat on every fit.
    // Set false to reproduce pre-merge/split results exactly.
    bool merge_split_enabled = true;

    int num_tempering_chains = 1;
    int seed = 0;
    double timeout = 60.0;

    // Optional callbacks for progress reporting (all thread-safe).
    // Each receives (completed, total) where total = num_chains × phase_length.
    std::function<void(int, int)> on_warmup;   // once per iteration per chain (parameter warm-up)
    std::function<void(int, int)> on_burn_in;  // once per iteration per chain (warm-up)
    std::function<void(int, int)> on_sample;   // once per recorded draw per chain

    // Optional pointer to an external abort flag. When the flag is set, fit()
    // exits cleanly at the next loop boundary (warmup / burn-in / sampling).
    // The caller is responsible for setting this from a signal handler or a
    // monitoring thread, and for checking the flag after fit() returns.
    std::atomic<bool>* abort_flag = nullptr;
};

// Draws recorded per chain: ceil(num_samples / num_thin).
inline int draws_per_chain(const FitParams& p) {
    return (p.num_samples + p.num_thin - 1) / p.num_thin;
}

// Run parallel-tempered MCMC and return the collected samples.
FitResult fit(FitParams params);

#endif  // FIT_H
