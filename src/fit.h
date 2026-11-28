#ifndef FIT_H
#define FIT_H

#include "MCMC_core.h"
#include "fit_result.h"
#include <atomic>
#include <functional>
#include <optional>
#include <unordered_map>
#include <vector>

// All parameters controlling a single call to fit().
struct FitParams {
    std::vector<std::unordered_map<int, double>> wins;
    std::vector<std::unordered_map<int, double>> ties;
    std::vector<std::unordered_map<int, double>> group_adj; // empty = single-layer mode
    bool weighted_SBM = false;
    bool weighted_proposals = false;  // weighted_SBM only: true = block-informed proposals follow the weights (experimental)

    // Empty = the default starting state (initial_state.h): greedy modularity, and
    // win fractions.  Given scores that are not fixed also set the starting sigma_I
    // to their RMS, unless individual_depth(_init) is set.
    std::vector<int>    initial_partition;
    bool                partition_fixed   = false;
    std::vector<double> initial_scores;
    bool                scores_fixed      = false;

    bool assortative = true;

    std::optional<double> variation_in;
    std::optional<double> variation_out;
    std::optional<double> degree_correction;
    std::optional<double> individual_depth;
    std::optional<double> individual_depth_init; // starting value only, not fixed
    std::optional<double> group_depth;
    std::optional<double> group_depth_init;      // starting value only, not fixed
    std::optional<double> ties_parameter;
    std::optional<int>    num_groups;
    bool                  num_groups_fixed = false;

    int    num_chains           = 1;
    int    samples_per_chain    = 1000;
    int    sweeps_per_sample    = 10;
    int    burn_sweeps          = 500;
    // Number of additional sweeps at the start of burn-in with partition and
    // scores fixed, so that continuous parameters (variation_in/out,
    // degree_correction, individual_depth, group_depth, ties_parameter)
    // can equilibrate to the initial partition before the partition is free.
    int    param_warmup_sweeps  = 0;
    int    thinning             = 1;
    double beta                 = 1.0;
    // Collective merge/split moves (Peixoto arXiv:2003.07070).
    // ON by default: without them single-site MCMC can converge silently to the
    // WRONG answer -- on the N=115 football network four of six chains get stuck
    // (R-hat 3.37, B_e 8.97 vs the true 10.25) while each chain looks perfectly
    // healthy from the inside.  Costs roughly 5x on networks that single-site
    // already handles, which buys correctness you otherwise cannot detect
    // without multi-start R-hat on every fit.
    // Set false to reproduce pre-merge/split results exactly.
    bool   merge_split_enabled  = true;

    int    num_tempering_chains = 1;
    int    seed                 = 0;
    double timeout              = 60.0;

    // Optional callbacks for progress reporting (all thread-safe).
    // Each receives (completed, total) where total = num_chains × phase_length.
    std::function<void(int, int)> on_warmup;   // called once per sweep per chain (param warmup)
    std::function<void(int, int)> on_burn_in;  // called once per sweep per chain (burn-in)
    std::function<void(int, int)> on_sample;   // called once per recorded sample per chain

    // Optional pointer to an external abort flag. When the flag is set, fit()
    // exits cleanly at the next loop boundary (warmup / burn-in / sampling).
    // The caller is responsible for setting this from a signal handler or a
    // monitoring thread, and for checking the flag after fit() returns.
    std::atomic<bool>* abort_flag = nullptr;
};

// Run parallel-tempered MCMC and return the collected samples.
FitResult fit(FitParams params);

#endif // FIT_H
