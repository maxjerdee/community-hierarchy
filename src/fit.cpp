#include "fit.h"
#include "globals.h"
#include "initial_state.h"
#include "rng.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <future>
#include <random>
#include <vector>

namespace {

// Progress counters shared by the chains (atomic: the chains run in parallel).
struct Progress {
    std::atomic<int> param_warmup{0}, warmup{0}, samples{0};
};

// The inverse temperatures of the replicas: beta alone, or with K > 1 replicas
// 0 and then K - 1 values geometrically spaced from beta / (K - 1) to beta.  The
// last replica is at beta, and is the one recorded.
std::vector<double> beta_ladder(double beta, int K) {
    if (K == 1) return {beta};
    std::vector<double> betas = {0.0};
    const double log_bmin = std::log(beta / (K - 1)), log_bmax = std::log(beta);
    for (int i = 1; i < K; ++i)
        betas.push_back(std::exp(log_bmin + (log_bmax - log_bmin) * i / (K - 1)));
    return betas;
}

// One chain of a fit: its tempering replicas, run through parameter warm-up,
// warm-up and sampling.
class ChainRun {
public:
    ChainRun(const FitParams& p, int chain_id, Progress& progress)
        : p_(p),
          chain_id_(chain_id),
          progress_(progress),
          // seed + chain in 64-bit arithmetic (an int sum would overflow near INT_MAX).
          seed_(static_cast<std::uint64_t>(static_cast<std::int64_t>(p.seed) + chain_id)),
          swap_rng_(seed_ + 999983ULL),
          betas_(beta_ladder(p.beta, p.num_tempering_chains)) {
        // This thread's move generator, seeded once per chain; the replicas are
        // built without reseeding it (new_chain), so they share one stream.
        rng.seed(static_cast<std::mt19937::result_type>(seed_));
        std::mt19937_64 init_rng(seed_ * 2654435761ULL);
        for (int ci = 0; ci < p.num_tempering_chains; ++ci)
            replicas_.push_back(new_chain(p, init_rng));
        start_ = std::chrono::steady_clock::now();  // the timeout counts from here
    }

    std::vector<SampleState> run() {
        param_warmup();
        warmup();
        return sample();
    }

private:
    // Before warm-up: the partition and scores frozen, so the continuous parameters
    // reach their conditional posterior before the partition is free to move.
    void param_warmup() {
        if (p_.num_param_warmup == 0) return;
        for (auto& c : replicas_) c.partition_fixed = c.scores_fixed = true;
        for (int it = 0; it < p_.num_param_warmup && running(); ++it) {
            sweep();
            report(p_.on_warmup, progress_.param_warmup, p_.num_chains * p_.num_param_warmup);
        }
        for (auto& c : replicas_) {
            c.partition_fixed = p_.partition_fixed;
            c.scores_fixed = p_.scores_fixed;
        }
    }

    // Warm-up: the step sizes adapt (Robbins-Monro, globals.h) and the weight of the
    // collective moves is retuned; both are frozen for sampling, so the sampling
    // kernel is fixed and exact.
    void warmup() {
        retune();  // from the initial partition
        for (auto& c : replicas_) c.adapting = true;
        for (int it = 0; it < p_.num_warmup && running(); ++it) {
            sweep();
            swap();
            retune();
            report(p_.on_burn_in, progress_.warmup, p_.num_chains * p_.num_warmup);
        }
        for (auto& c : replicas_) c.adapting = false;
    }

    // Sampling: num_samples iterations; the state of every num_thin-th is recorded.
    std::vector<SampleState> sample() {
        std::vector<SampleState> draws;
        draws.reserve(static_cast<std::size_t>(draws_per_chain(p_)));
        for (int it = 0; it < p_.num_samples && running(); ++it) {
            sweep();
            if (it % p_.num_thin == 0) {
                draws.push_back(replicas_.back().snapshot(chain_id_));
                report(p_.on_sample, progress_.samples, p_.num_chains * draws_per_chain(p_));
            }
            swap();
        }
        return draws;
    }

    // One iteration of every replica: sweeps_per_sample sweeps at its beta.
    void sweep() {
        for (std::size_t ci = 0; ci < replicas_.size(); ++ci)
            replicas_[ci].MCMC_sweep(betas_[ci], p_.sweeps_per_sample, p_.merge_split_enabled);
    }

    // Exchange the states of neighbouring replicas (Metropolis).  The criterion is
    // the tempered energy, the data likelihood terms only: every move targets
    // exp(-(H_b + H_s_G_b_depths + priors + beta * H_total)), with the partition
    // prior H_b and the score prior H_s_G_b_depths untempered (MCMC_core.h), so an
    // exchange between inverse temperatures involves H_total alone.
    void swap() {
        auto H_total = [](const MCMC& c) { return c.H_A_G_b + c.H_Adir_G_A_s_nu; };
        for (std::size_t ci = 0; ci + 1 < replicas_.size(); ++ci) {
            const double db = betas_[ci + 1] - betas_[ci];
            const double dH = H_total(replicas_[ci + 1]) - H_total(replicas_[ci]);
            if (next_uniform(swap_rng_) < std::exp(db * dH))
                std::swap(replicas_[ci], replicas_[ci + 1]);
        }
    }

    // The weight of the collective moves, while the caller leaves
    // MERGE_SPLIT_WEIGHT negative: the equal-spend rule w = q / (C M), never below
    // the linear-time value n / (M (n + m)) (globals.h).  The linear-time value
    // alone budgets a worst-case cost of M (n + m) per collective move, as if
    // restaging touched the whole graph, though it restages only the two groups
    // involved; on dense networks collective moves then stop firing (about 5 in a
    // 1250-sweep run on the 456-node English football graph, which missed the
    // leagues by ~1000 nats).  Retuned during warm-up only: q changes as the chain
    // runs, and a state-dependent weight would bias the q-marginal.
    void retune() {
        for (auto& c : replicas_) c.retune_collective_weight();
    }

    bool running() const {
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
        const bool aborted = p_.abort_flag && p_.abort_flag->load(std::memory_order_relaxed);
        return elapsed < p_.timeout && !aborted;
    }

    static void report(const std::function<void(int, int)>& callback, std::atomic<int>& counter,
                       int total) {
        if (!callback) return;
        callback(counter.fetch_add(1, std::memory_order_relaxed) + 1, total);
    }

    const FitParams& p_;
    const int chain_id_;
    Progress& progress_;
    const std::uint64_t seed_;
    std::mt19937_64 swap_rng_;
    const std::vector<double> betas_;
    std::chrono::steady_clock::time_point start_;
    std::vector<MCMC> replicas_;
};

}  // namespace

FitResult fit(FitParams p) {
    complete_start(p);  // the default starting state (initial_state.h)
    Progress progress;

    // The chains run in parallel, each on its own thread (and its own thread_local
    // generator, globals.h).
    std::vector<std::future<std::vector<SampleState>>> futures;
    for (int c = 0; c < p.num_chains; ++c)
        futures.emplace_back(std::async(
            std::launch::async, [&p, &progress, c] { return ChainRun(p, c, progress).run(); }));

    std::vector<SampleState> samples;
    for (auto& f : futures) {
        auto chunk = f.get();
        samples.insert(samples.end(), std::make_move_iterator(chunk.begin()),
                       std::make_move_iterator(chunk.end()));
    }
    return FitResult(static_cast<int>(p.wins.size()), std::move(samples));
}
