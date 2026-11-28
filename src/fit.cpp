#include "fit.h"
#include "globals.h"
#include "initial_state.h"
#include "rng.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <future>
#include <random>
#include <vector>

FitResult fit(FitParams p) {
    int n = static_cast<int>(p.wins.size());
    complete_start(p);  // the default starting state (initial_state.h)
    const int total_samples = p.num_chains * draws_per_chain(p);
    std::atomic<int> completed_warmup{0};
    std::atomic<int> completed_burn{0};
    std::atomic<int> completed_samples{0};

    auto run_chain = [&](int chain_id) -> std::vector<SampleState> {
        // Build tempering stack — all chains in the stack share the same graph
        // but start from the same initial state and evolve independently.
        std::vector<MCMC> chain_stack;
        chain_stack.reserve(static_cast<std::size_t>(p.num_tempering_chains));

        // This thread's move generator: seeded once per chain.  The replicas below
        // are built with seed -1 so they do not reseed it; they share one stream.
        // (Previously each replica passed its own seed, but every constructor
        // reseeded the same thread_local generator, so only the last one counted.)
        // With one replica this is the same seed as before: seed + chain.
        // seed + chain in 64-bit arithmetic: an int sum overflows (undefined
        // behaviour) for seeds near INT_MAX.  For every seed that did not overflow
        // the resulting generator seeds are unchanged.
        const std::uint64_t chain_seed = static_cast<std::uint64_t>(
            static_cast<std::int64_t>(p.seed) + static_cast<std::int64_t>(chain_id));
        rng.seed(static_cast<std::mt19937::result_type>(chain_seed));

        std::mt19937_64 init_rng(chain_seed * 2654435761ULL);

        for (int ci = 0; ci < p.num_tempering_chains; ++ci)
            chain_stack.push_back(new_chain(p, init_rng));

        // Beta schedule: [0, ..., beta] on a log scale; single chain → [beta].
        std::vector<double> betas;
        if (p.num_tempering_chains == 1) {
            betas.push_back(p.beta);
        } else {
            betas.push_back(0.0);
            double log_bmin = std::log(p.beta / (p.num_tempering_chains - 1));
            double log_bmax = std::log(p.beta);
            for (int i = 1; i < p.num_tempering_chains; ++i) {
                double lb = log_bmin + (log_bmax - log_bmin) * i / (p.num_tempering_chains - 1);
                betas.push_back(std::exp(lb));
            }
        }

        std::mt19937_64 swap_rng(chain_seed + 999983ULL);

        // Swap criterion: the tempered energy, i.e. the data likelihood terms only.
        // Every move targets exp(-(H_b + H_s_G_b_depths + priors + beta * H_total)),
        // with the partition prior H_b and the score prior H_s_G_b_depths untempered
        // (the convention of MCMC_core.h), so exchanging configurations between
        // inverse temperatures involves H_total alone.  (Before 2026-09 some moves
        // tempered H_s_G_b_depths and others did not, so the hot chains had no
        // consistent target -- the "~0.06 over-clustering" once seen when H_s was
        // added here was a symptom of that, not of the swap rule.)
        auto H_total = [](const MCMC& c) { return c.H_A_G_b + c.H_Adir_G_A_s_nu; };

        auto do_swaps = [&]() {
            for (int ci = 0; ci < p.num_tempering_chains - 1; ++ci) {
                double db = betas[ci + 1] - betas[ci];
                double dH = H_total(chain_stack[ci + 1]) - H_total(chain_stack[ci]);
                if (next_uniform(swap_rng) < std::exp(db * dH))
                    std::swap(chain_stack[ci], chain_stack[ci + 1]);
            }
        };

        auto t_start = std::chrono::steady_clock::now();
        auto elapsed = [&]() {
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start)
                .count();
        };
        auto aborted = [&]() {
            return p.abort_flag && p.abort_flag->load(std::memory_order_relaxed);
        };

        // Parameter warmup: freeze partition and scores so that continuous
        // parameters (variation_in/out, degree_correction, individual_depth,
        // group_depth, ties_parameter) can reach their conditional posterior
        // before the partition is free to move.
        if (p.num_param_warmup > 0) {
            for (int ci = 0; ci < p.num_tempering_chains; ++ci) {
                chain_stack[ci].partition_fixed = true;
                chain_stack[ci].scores_fixed = true;
            }
            for (int b = 0; b < p.num_param_warmup && elapsed() < p.timeout && !aborted(); ++b) {
                for (int ci = 0; ci < p.num_tempering_chains; ++ci)
                    chain_stack[ci].MCMC_sweep(betas[ci], p.sweeps_per_sample,
                                               p.merge_split_enabled);
                if (p.on_warmup) {
                    int done = completed_warmup.fetch_add(1, std::memory_order_relaxed) + 1;
                    p.on_warmup(done, p.num_chains * p.num_param_warmup);
                }
            }
            for (int ci = 0; ci < p.num_tempering_chains; ++ci) {
                chain_stack[ci].partition_fixed = p.partition_fixed;
                chain_stack[ci].scores_fixed = p.scores_fixed;
            }
        }

        // Equal-spend weighting of the collective moves, while the caller leaves
        // MERGE_SPLIT_WEIGHT negative.  The old auto value, the linear-time
        // condition n/(M(n+m)), budgets against a WORST-CASE cost of M(n+m) per
        // collective move, as if restaging touched the whole graph; it restages
        // only the two groups involved, k ~ 2n/q nodes.  The overestimate grows
        // with density, so on a dense network collective moves stop firing (~5 in
        // a 1250-sweep run on the 456-node English football graph, which then
        // missed the leagues by ~1000 nats).  Equating the true cost with the
        // single-node sweep cost gives w = q/(6 C M); the per-node cost cancels,
        // which is why density does not appear.
        //
        // Updated only DURING BURN-IN and frozen for sampling: q changes as the
        // chain runs, and a state-dependent weight would bias the q-marginal.
        // Never below the old linear-time value, so no run loses collective moves.
        auto retune = [&]() {
            for (int ci = 0; ci < p.num_tempering_chains; ++ci)
                chain_stack[static_cast<std::size_t>(ci)].retune_collective_weight();
        };
        retune();  // from the initial partition

        // Step sizes adapt during burn-in only (Robbins-Monro, globals.h) and are
        // frozen for sampling, so the sampling kernel is fixed and exact.
        for (auto& c : chain_stack) c.adapting = true;

        // Warm-up: num_warmup iterations of sweeps_per_sample sweeps each.
        for (int b = 0; b < p.num_warmup && elapsed() < p.timeout && !aborted(); ++b) {
            for (int ci = 0; ci < p.num_tempering_chains; ++ci)
                chain_stack[ci].MCMC_sweep(betas[ci], p.sweeps_per_sample, p.merge_split_enabled);
            do_swaps();
            retune();  // adapt, then freeze at sampling
            if (p.on_burn_in) {
                int done = completed_burn.fetch_add(1, std::memory_order_relaxed) + 1;
                p.on_burn_in(done, p.num_chains * p.num_warmup);
            }
        }

        for (auto& c : chain_stack) c.adapting = false;

        // Sampling
        std::vector<SampleState> samples;
        samples.reserve(static_cast<std::size_t>(draws_per_chain(p)));

        // Sampling: num_samples iterations; the draw of every num_thin-th is recorded.
        for (int it = 0; it < p.num_samples && elapsed() < p.timeout && !aborted(); ++it) {
            for (int ci = 0; ci < p.num_tempering_chains; ++ci)
                chain_stack[ci].MCMC_sweep(betas[ci], p.sweeps_per_sample, p.merge_split_enabled);
            if (it % p.num_thin != 0) {
                do_swaps();
                continue;
            }

            // Record the replica at the target beta (last in the schedule).
            samples.push_back(chain_stack[p.num_tempering_chains - 1].snapshot(chain_id));

            int done = completed_samples.fetch_add(1, std::memory_order_relaxed) + 1;
            if (p.on_sample) p.on_sample(done, total_samples);

            do_swaps();
        }

        return samples;
    };

    // Launch all independent chains in parallel
    std::vector<std::future<std::vector<SampleState>>> futures;
    futures.reserve(static_cast<std::size_t>(p.num_chains));
    for (int ci = 0; ci < p.num_chains; ++ci)
        futures.emplace_back(std::async(std::launch::async, run_chain, ci));

    std::vector<SampleState> all_samples;
    all_samples.reserve(static_cast<std::size_t>(total_samples));
    for (auto& f : futures) {
        auto chunk = f.get();
        all_samples.insert(all_samples.end(), std::make_move_iterator(chunk.begin()),
                           std::make_move_iterator(chunk.end()));
    }

    return FitResult(n, std::move(all_samples));
}
