#include "fit.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <future>
#include <random>
#include <vector>

FitResult fit(FitParams p) {
    int n = static_cast<int>(p.wins.size());
    int initial_num_groups = *std::max_element(p.initial_partition.begin(),
                                               p.initial_partition.end()) + 1;
    const int total_samples = p.num_chains * p.samples_per_chain;
    std::atomic<int> completed_warmup{0};
    std::atomic<int> completed_burn{0};
    std::atomic<int> completed_samples{0};

    auto run_chain = [&](int chain_id) -> std::vector<SampleState> {
        // Build tempering stack — all chains in the stack share the same graph
        // but start from the same initial state and evolve independently.
        std::vector<MCMC> chain_stack;
        chain_stack.reserve(static_cast<std::size_t>(p.num_tempering_chains));

        std::mt19937_64 init_rng(static_cast<std::uint64_t>(p.seed + chain_id) * 2654435761ULL);
        std::uniform_real_distribution<double> unif01(0.0, 1.0);

        for (int ci = 0; ci < p.num_tempering_chains; ++ci) {
            double vi  = p.variation_in.value_or(unif01(init_rng));
            double vo  = p.variation_out.value_or(unif01(init_rng));
            double dc  = p.degree_correction.value_or(unif01(init_rng));
            double id_ = p.individual_depth.has_value()      ? *p.individual_depth :
                         p.individual_depth_init.has_value() ? *p.individual_depth_init
                                                             : unif01(init_rng);
            double gd  = p.group_depth.has_value()           ? *p.group_depth :
                         p.group_depth_init.has_value()      ? *p.group_depth_init
                                                             : unif01(init_rng);
            double tp  = p.ties_parameter.value_or(unif01(init_rng));
            int    ng  = p.num_groups.value_or(initial_num_groups);

            chain_stack.emplace_back(
                p.wins, p.ties, p.group_adj,
                p.initial_partition, p.partition_fixed,
                p.initial_scores,   p.scores_fixed,
                ng, p.num_groups.has_value() || p.num_groups_fixed,
                p.assortative,
                vi,  p.variation_in.has_value(),
                vo,  p.variation_out.has_value(),
                dc,  p.degree_correction.has_value(),
                id_, p.individual_depth.has_value(),
                gd,  p.group_depth.has_value(),
                tp,  p.ties_parameter.has_value(),
                p.seed + chain_id * p.num_tempering_chains + ci,
                -1.0, -1.0,
                p.weighted_SBM
            );
        }

        // Beta schedule: [0, ..., beta] on a log scale; single chain → [beta].
        std::vector<double> betas;
        if (p.num_tempering_chains == 1) {
            betas.push_back(p.beta);
        } else {
            betas.push_back(0.0);
            double log_bmin = std::log(p.beta / (p.num_tempering_chains - 1));
            double log_bmax = std::log(p.beta);
            for (int i = 1; i < p.num_tempering_chains; ++i) {
                double lb = log_bmin + (log_bmax - log_bmin)
                            * i / (p.num_tempering_chains - 1);
                betas.push_back(std::exp(lb));
            }
        }

        std::mt19937_64 swap_rng(static_cast<std::uint64_t>(p.seed + chain_id) + 999983ULL);
        std::uniform_real_distribution<double> swap_unif(0.0, 1.0);

        // Swap criterion: data likelihood terms only.
        // H_b (partition prior) cancels via the asymmetric partition proposal.
        // H_s_G_b_depths (score prior) cancels via the asymmetric score CDF proposal.
        // Including H_s in the criterion causes systematic over-clustering (~0.06 bias).
        auto H_total = [](const MCMC& c) {
            return c.H_A_G_b + c.H_Adir_G_A_s_nu;
        };

        auto do_swaps = [&]() {
            for (int ci = 0; ci < p.num_tempering_chains - 1; ++ci) {
                double db = betas[ci + 1] - betas[ci];
                double dH = H_total(chain_stack[ci + 1]) - H_total(chain_stack[ci]);
                if (swap_unif(swap_rng) < std::exp(db * dH))
                    std::swap(chain_stack[ci], chain_stack[ci + 1]);
            }
        };

        auto t_start = std::chrono::steady_clock::now();
        auto elapsed = [&]() {
            return std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_start).count();
        };
        auto aborted = [&]() {
            return p.abort_flag && p.abort_flag->load(std::memory_order_relaxed);
        };

        // Parameter warmup: freeze partition and scores so that continuous
        // parameters (variation_in/out, degree_correction, individual_depth,
        // group_depth, ties_parameter) can reach their conditional posterior
        // before the partition is free to move.
        if (p.param_warmup_sweeps > 0) {
            for (int ci = 0; ci < p.num_tempering_chains; ++ci) {
                chain_stack[ci].partition_fixed = true;
                chain_stack[ci].scores_fixed    = true;
            }
            for (int b = 0; b < p.param_warmup_sweeps && elapsed() < p.timeout && !aborted(); ++b) {
                for (int ci = 0; ci < p.num_tempering_chains; ++ci)
                    chain_stack[ci].MCMC_sweep(betas[ci], p.sweeps_per_sample, p.merge_split_enabled);
                if (p.on_warmup) {
                    int done = completed_warmup.fetch_add(1, std::memory_order_relaxed) + 1;
                    p.on_warmup(done, p.num_chains * p.param_warmup_sweeps);
                }
            }
            for (int ci = 0; ci < p.num_tempering_chains; ++ci) {
                chain_stack[ci].partition_fixed = p.partition_fixed;
                chain_stack[ci].scores_fixed    = p.scores_fixed;
            }
        }

        // Burn-in
        for (int b = 0; b < p.burn_sweeps && elapsed() < p.timeout && !aborted(); ++b) {
            for (int ci = 0; ci < p.num_tempering_chains; ++ci)
                chain_stack[ci].MCMC_sweep(betas[ci], p.sweeps_per_sample, p.merge_split_enabled);
            do_swaps();
            if (p.on_burn_in) {
                int done = completed_burn.fetch_add(1, std::memory_order_relaxed) + 1;
                p.on_burn_in(done, p.num_chains * p.burn_sweeps);
            }
        }

        // Sampling
        std::vector<SampleState> samples;
        samples.reserve(static_cast<std::size_t>(p.samples_per_chain));

        for (int s = 0; s < p.samples_per_chain && elapsed() < p.timeout && !aborted(); ++s) {
            for (int ci = 0; ci < p.num_tempering_chains; ++ci)
                chain_stack[ci].MCMC_sweep(betas[ci], p.sweeps_per_sample, p.merge_split_enabled);

            // Record the β=1 chain (last in schedule)
            const MCMC& c = chain_stack[p.num_tempering_chains - 1];
            SampleState ss;
            ss.chain_id          = chain_id;
            ss.beta              = betas[p.num_tempering_chains - 1];
            ss.partition         = c.partition;
            ss.scores            = c.scores;
            ss.num_groups        = c.num_groups;
            ss.density_in        = c.density_in;
            ss.density_out       = c.density_out;
            ss.variation_in      = c.variation_in;
            ss.variation_out     = c.variation_out;
            ss.degree_correction = c.degree_correction;
            ss.individual_depth  = c.individual_depth;
            ss.group_depth       = c.group_depth;
            ss.ties_parameter    = c.ties_parameter;
            ss.H_A_b             = c.H_A_b;
            ss.H_b               = c.H_b;
            ss.H_M               = c.H_M;
            ss.H_k               = c.H_k;
            ss.H_A_G_k_M_b       = c.H_A_G_k_M_b;
            ss.H_A_G_b           = c.H_A_G_b;
            ss.H_Adir_G_A_s_nu   = c.H_Adir_G_A_s_nu;
            ss.H_s_G_b_depths    = c.H_s_G_b_depths;
            ss.two_layer_mode    = c.two_layer_mode;
            ss.weighted_SBM      = c.weighted_SBM;
            ss.H_W               = c.H_W;
            samples.push_back(std::move(ss));

            int done = completed_samples.fetch_add(1, std::memory_order_relaxed) + 1;
            if (p.on_sample) p.on_sample(done, total_samples);

            // Thinning sweeps between recorded samples
            for (int t = 1; t < p.thinning; ++t)
                for (int ci = 0; ci < p.num_tempering_chains; ++ci)
                    chain_stack[ci].MCMC_sweep(betas[ci], p.sweeps_per_sample, p.merge_split_enabled);

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
        all_samples.insert(all_samples.end(),
                           std::make_move_iterator(chunk.begin()),
                           std::make_move_iterator(chunk.end()));
    }

    return FitResult(n, std::move(all_samples));
}
