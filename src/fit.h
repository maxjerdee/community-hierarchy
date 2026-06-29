#ifndef FIT_H
#define FIT_H

#include "MCMC_core.h"
#include "fit_result.h"
#include <functional>
#include <optional>
#include <unordered_map>
#include <vector>

// All parameters controlling a single call to fit().
struct FitParams {
    std::vector<std::unordered_map<int, double>> wins;
    std::vector<std::unordered_map<int, double>> ties;

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
    int    num_tempering_chains = 1;
    int    seed                 = 0;
    double timeout              = 60.0;

    // Optional callback invoked each time a sample is recorded (thread-safe).
    // Receives (completed_samples, total_samples).
    std::function<void(int, int)> on_sample;
};

// Run parallel-tempered MCMC and return the collected samples.
FitResult fit(FitParams params);

#endif // FIT_H
