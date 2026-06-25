#include "MCMC_core.h"
#include "fit_result.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// FitResultNative — extends FitResult with Python-facing helpers
// ---------------------------------------------------------------------------

class FitResultNative : public FitResult {
public:
    FitResultNative() = default;
    FitResultNative(int num_nodes, std::vector<SampleState> samples)
        : FitResult(num_nodes, std::move(samples)) {}

    py::object samples_df() const {
        py::module_ pd = py::module_::import("pandas");
        int ns = static_cast<int>(samples_.size());
        int n  = num_nodes_;

        // Build column list
        std::vector<std::string> cols;
        for (int i = 0; i < n; ++i) cols.push_back(std::to_string(i) + "_group");
        for (int i = 0; i < n; ++i) cols.push_back(std::to_string(i) + "_score");
        for (const auto& c : {"num_groups", "density_in", "density_out",
                               "variation_in", "variation_out", "degree_correction",
                               "individual_depth", "group_depth",
                               "ties_parameter", "H_A_b", "H_b", "H_M", "H_k",
                               "H_A_G_k_M_b", "H_A_G_b", "H_Adir_G_A_s_nu",
                               "H_s_G_b_depths", "chain_id", "beta"})
            cols.push_back(c);

        // Build data dict
        py::dict data;
        for (int i = 0; i < n; ++i) {
            std::vector<int> col(ns);
            for (int s = 0; s < ns; ++s) col[s] = samples_[s].partition[i];
            data[py::str(std::to_string(i) + "_group")] = py::cast(col);
        }
        for (int i = 0; i < n; ++i) {
            std::vector<double> col(ns);
            for (int s = 0; s < ns; ++s) col[s] = samples_[s].scores[i];
            data[py::str(std::to_string(i) + "_score")] = py::cast(col);
        }

        auto fill = [&](const char* name, auto getter) {
            using T = decltype(getter(samples_[0]));
            std::vector<T> col(ns);
            for (int s = 0; s < ns; ++s) col[s] = getter(samples_[s]);
            data[py::str(name)] = py::cast(col);
        };
        fill("num_groups",        [](const SampleState& s){ return s.num_groups; });
        fill("density_in",        [](const SampleState& s){ return s.density_in; });
        fill("density_out",       [](const SampleState& s){ return s.density_out; });
        fill("variation_in",      [](const SampleState& s){ return s.variation_in; });
        fill("variation_out",     [](const SampleState& s){ return s.variation_out; });
        fill("degree_correction", [](const SampleState& s){ return s.degree_correction; });
        fill("individual_depth",  [](const SampleState& s){ return s.individual_depth; });
        fill("group_depth",       [](const SampleState& s){ return s.group_depth; });
        fill("ties_parameter",    [](const SampleState& s){ return s.ties_parameter; });
        fill("H_A_b",             [](const SampleState& s){ return s.H_A_b; });
        fill("H_b",               [](const SampleState& s){ return s.H_b; });
        fill("H_M",               [](const SampleState& s){ return s.H_M; });
        fill("H_k",               [](const SampleState& s){ return s.H_k; });
        fill("H_A_G_k_M_b",       [](const SampleState& s){ return s.H_A_G_k_M_b; });
        fill("H_A_G_b",           [](const SampleState& s){ return s.H_A_G_b; });
        fill("H_Adir_G_A_s_nu",   [](const SampleState& s){ return s.H_Adir_G_A_s_nu; });
        fill("H_s_G_b_depths",    [](const SampleState& s){ return s.H_s_G_b_depths; });
        fill("chain_id",          [](const SampleState& s){ return s.chain_id; });
        fill("beta",              [](const SampleState& s){ return s.beta; });

        return pd.attr("DataFrame")(data);
    }

    py::dict summary() const {
        double mean_k = 0.0;
        double mean_dl = 0.0;
        for (const auto& s : samples_) {
            mean_k += s.num_groups;
            mean_dl += s.H_A_b + s.H_Adir_G_A_s_nu + s.H_s_G_b_depths;
        }
        int ns = static_cast<int>(samples_.size());
        if (ns > 0) { mean_k /= ns; mean_dl /= ns; }
        py::dict d;
        d["num_samples"]        = ns;
        d["mean_num_groups"]    = mean_k;
        d["mean_description_length"] = mean_dl;
        d["mdl_value"]          = mdl_value();
        return d;
    }

    py::object coincidence_matrix_numpy() const {
        auto flat = coincidence_matrix();
        int n = num_nodes_;
        py::module_ np = py::module_::import("numpy");
        auto arr = py::cast(flat);
        return np.attr("reshape")(arr, py::make_tuple(n, n));
    }
};

// ---------------------------------------------------------------------------
// fit_backend — full sampling loop in C++
// ---------------------------------------------------------------------------

FitResultNative fit_backend(
    std::vector<std::unordered_map<int, double>> wins,
    std::vector<std::unordered_map<int, double>> ties,
    std::vector<int>    initial_partition,
    bool                partition_fixed,
    std::vector<double> initial_scores,
    bool                scores_fixed,
    bool                assortative,
    std::optional<double> variation_in_opt,
    std::optional<double> variation_out_opt,
    std::optional<double> degree_correction_opt,
    std::optional<double> individual_depth_opt,
    std::optional<double> group_depth_opt,
    std::optional<double> ties_parameter_opt,
    std::optional<int>    num_groups_opt,
    int    num_chains,
    int    samples_per_chain,
    int    sweeps_per_sample,
    int    burn_sweeps,
    int    thinning,
    double beta,
    int    num_tempering_chains,
    int    seed,
    double timeout,
    bool   show_progress
) {
    int n = static_cast<int>(wins.size());
    int initial_num_groups = *std::max_element(initial_partition.begin(), initial_partition.end()) + 1;


    // --- Progress counter (shared across all parallel chains) ---
    const int total_samples = num_chains * samples_per_chain;
    std::atomic<int> completed_samples{0};
    std::mutex progress_mutex;
    std::condition_variable progress_cv;
    bool stop_progress = false;
    std::thread progress_thread;

    auto emit_progress = [&](int done) {
        const double pct = 100.0 * static_cast<double>(done) / static_cast<double>(total_samples);
        std::ostringstream ss;
        ss << '\r' << "Sampling progress: " << done << '/' << total_samples
           << " (" << std::fixed << std::setprecision(1) << pct << "%)";
        std::cerr << ss.str() << std::flush;
    };

    if (show_progress) {
        progress_thread = std::thread([&]() {
            int last_reported = -1;
            emit_progress(0);
            while (true) {
                std::unique_lock<std::mutex> lock(progress_mutex);
                progress_cv.wait_for(lock, std::chrono::milliseconds(100), [&]() {
                    return stop_progress ||
                           completed_samples.load(std::memory_order_relaxed) != last_reported;
                });
                const int current = completed_samples.load(std::memory_order_relaxed);
                const bool stopping = stop_progress;
                lock.unlock();
                if (current != last_reported) {
                    last_reported = current;
                    emit_progress(current);
                }
                if (stopping) break;
            }
        });
    }

    auto finish_progress = [&]() {
        if (progress_thread.joinable()) {
            {
                std::lock_guard<std::mutex> lock(progress_mutex);
                stop_progress = true;
            }
            progress_cv.notify_all();
            progress_thread.join();
            if (show_progress) std::cerr << '\n' << std::flush;
        }
    };

    // --- Per-chain sampling lambda ---
    auto run_chain = [&](int chain_id) -> std::vector<SampleState> {
        // Each independent chain gets its own copy of the tempering stack
        std::vector<MCMC> chain_stack;
        chain_stack.reserve(static_cast<std::size_t>(num_tempering_chains));

        std::mt19937_64 chain_init_rng(static_cast<std::uint64_t>(seed + chain_id) * 2654435761ULL);
        std::uniform_real_distribution<double> unif01(0.0, 1.0);

        for (int ci = 0; ci < num_tempering_chains; ++ci) {
            double vi  = variation_in_opt.has_value()      ? *variation_in_opt      : unif01(chain_init_rng);
            double vo  = variation_out_opt.has_value()     ? *variation_out_opt     : unif01(chain_init_rng);
            double dc  = degree_correction_opt.has_value() ? *degree_correction_opt : unif01(chain_init_rng);
            double id_ = individual_depth_opt.has_value()  ? *individual_depth_opt  : unif01(chain_init_rng);
            double gd  = group_depth_opt.has_value()       ? *group_depth_opt       : unif01(chain_init_rng);
            double tp  = ties_parameter_opt.has_value()    ? *ties_parameter_opt    : unif01(chain_init_rng);
            int    ng  = num_groups_opt.has_value()        ? *num_groups_opt        : initial_num_groups;

            chain_stack.emplace_back(
                wins, ties,
                initial_partition, partition_fixed,
                initial_scores,   scores_fixed,
                ng, num_groups_opt.has_value(),
                assortative,
                vi,  variation_in_opt.has_value(),
                vo,  variation_out_opt.has_value(),
                dc,  degree_correction_opt.has_value(),
                id_, individual_depth_opt.has_value(),
                gd,  group_depth_opt.has_value(),
                tp,  ties_parameter_opt.has_value(),
                seed + chain_id * num_tempering_chains + ci
            );
        }

        // Build per-chain beta schedule (same structure as the original)
        std::vector<double> chain_betas;
        if (num_tempering_chains == 1) {
            chain_betas.push_back(beta);
        } else {
            chain_betas.push_back(0.0);
            double log_bmin = std::log(beta / (num_tempering_chains - 1));
            double log_bmax = std::log(beta);
            for (int i = 1; i < num_tempering_chains; ++i) {
                double lb = log_bmin + (log_bmax - log_bmin) * i / (num_tempering_chains - 1);
                chain_betas.push_back(std::exp(lb));
            }
        }

        std::mt19937_64 swap_rng(static_cast<std::uint64_t>(seed + chain_id) + 999983ULL);
        std::uniform_real_distribution<double> swap_unif(0.0, 1.0);

        auto t_start = std::chrono::steady_clock::now();

        // Burn-in
        for (int b = 0; b < burn_sweeps; ++b) {
            auto elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_start).count();
            if (elapsed > timeout) break;
            for (int ci = 0; ci < num_tempering_chains; ++ci)
                chain_stack[ci].MCMC_sweep(chain_betas[ci], sweeps_per_sample);
            for (int ci = 0; ci < num_tempering_chains - 1; ++ci) {
                double db = chain_betas[ci + 1] - chain_betas[ci];
                double dH = chain_stack[ci + 1].H_A_G_b - chain_stack[ci].H_A_G_b;
                if (swap_unif(swap_rng) < std::exp(db * dH))
                    std::swap(chain_stack[ci], chain_stack[ci + 1]);
            }
        }

        // Sampling
        std::vector<SampleState> chain_samples;
        chain_samples.reserve(static_cast<std::size_t>(samples_per_chain));

        for (int s = 0; s < samples_per_chain; ++s) {
            auto elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_start).count();
            if (elapsed > timeout) break;

            for (int ci = 0; ci < num_tempering_chains; ++ci)
                chain_stack[ci].MCMC_sweep(chain_betas[ci], sweeps_per_sample);

            // Collect from the β=1 chain (last in schedule)
            {
                const MCMC& c = chain_stack[num_tempering_chains - 1];
                SampleState ss;
                ss.chain_id          = chain_id;
                ss.beta              = chain_betas[num_tempering_chains - 1];
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
                chain_samples.push_back(std::move(ss));
            }

            completed_samples.fetch_add(1, std::memory_order_relaxed);
            if (show_progress) progress_cv.notify_one();

            // Thinning sweeps (between recorded samples)
            for (int t = 1; t < thinning; ++t) {
                for (int ci = 0; ci < num_tempering_chains; ++ci)
                    chain_stack[ci].MCMC_sweep(chain_betas[ci], sweeps_per_sample);
            }

            // Parallel tempering swap proposals
            for (int ci = 0; ci < num_tempering_chains - 1; ++ci) {
                double db = chain_betas[ci + 1] - chain_betas[ci];
                double dH = chain_stack[ci + 1].H_A_G_b - chain_stack[ci].H_A_G_b;
                if (swap_unif(swap_rng) < std::exp(db * dH))
                    std::swap(chain_stack[ci], chain_stack[ci + 1]);
            }
        }

        return chain_samples;
    };

    // --- Launch chains in parallel ---
    std::vector<std::future<std::vector<SampleState>>> futures;
    futures.reserve(static_cast<std::size_t>(num_chains));
    for (int ci = 0; ci < num_chains; ++ci)
        futures.emplace_back(std::async(std::launch::async, run_chain, ci));

    std::vector<SampleState> all_samples;
    all_samples.reserve(static_cast<std::size_t>(total_samples));
    for (auto& f : futures) {
        auto chain_result = f.get();
        all_samples.insert(all_samples.end(),
                           std::make_move_iterator(chain_result.begin()),
                           std::make_move_iterator(chain_result.end()));
    }

    finish_progress();

    return FitResultNative(n, std::move(all_samples));
}

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(MCMC_core, m) {
    // --- MCMC class (kept for testing / advanced use) ---
    py::class_<MCMC>(m, "MCMC")
        .def(py::init<
                 std::vector<std::unordered_map<int, double>>,
                 std::vector<std::unordered_map<int, double>>,
                 std::vector<int>,
                 const bool&,
                 std::vector<double>,
                 const bool&,
                 int,
                 const bool&,
                 const bool&,
                 const double&,
                 const bool&,
                 const double&,
                 const bool&,
                 const double&,
                 const bool&,
                 const double&,
                 const bool&,
                 const double&,
                 const bool&,
                 const double&,
                 const bool&,
                 int,
                 double,
                 double> (),
             py::arg("wins"),
             py::arg("ties"),
             py::arg("partition"),
             py::arg("partition_fixed") = false,
             py::arg("scores"),
             py::arg("scores_fixed") = false,
             py::arg("num_groups"),
             py::arg("num_groups_fixed") = false,
             py::arg("assortative") = true,
             py::arg("variation_in") = 1.0,
             py::arg("variation_in_fixed") = false,
             py::arg("variation_out") = 1.0,
             py::arg("variation_out_fixed") = false,
             py::arg("degree_correction") = 1.0,
             py::arg("degree_correction_fixed") = false,
             py::arg("individual_depth") = 1.0,
             py::arg("individual_depth_fixed") = false,
             py::arg("group_depth") = 1.0,
             py::arg("group_depth_fixed") = false,
             py::arg("ties_parameter") = 1.0,
             py::arg("ties_parameter_fixed") = false,
             py::arg("seed") = 0,
             py::arg("density_in") = -1.0,
             py::arg("density_out") = -1.0)
           .def_readonly("num_groups", &MCMC::num_groups)
           .def_readonly("partition", &MCMC::partition)
           .def_readonly("scores", &MCMC::scores)
           .def_readonly("density_in", &MCMC::density_in)
           .def_readonly("density_out", &MCMC::density_out)
           .def_readonly("variation_in", &MCMC::variation_in)
           .def_readonly("variation_out", &MCMC::variation_out)
           .def_readonly("degree_correction", &MCMC::degree_correction)
           .def_readonly("individual_depth", &MCMC::individual_depth)
           .def_readonly("group_depth", &MCMC::group_depth)
           .def_readonly("ties_parameter", &MCMC::ties_parameter)
           .def_readonly("H_A_b", &MCMC::H_A_b)
           .def_readonly("H_b", &MCMC::H_b)
           .def_readonly("H_M", &MCMC::H_M)
           .def_readonly("H_k", &MCMC::H_k)
           .def_readonly("H_A_G_k_M_b", &MCMC::H_A_G_k_M_b)
           .def_readonly("H_A_G_b", &MCMC::H_A_G_b)
           .def_readonly("H_Adir_G_A_s_nu", &MCMC::H_Adir_G_A_s_nu)
           .def_readonly("H_s_G_b_depths", &MCMC::H_s_G_b_depths)
           .def_readonly("time_H_k", &MCMC::time_H_k)
           .def_readonly("time_H_M", &MCMC::time_H_M)
           .def_readonly("time_H_A_G_k_M_b", &MCMC::time_H_A_G_k_M_b)
           .def_readonly("time_single_site_proposals", &MCMC::time_single_site_proposals)
           .def_readonly("time_parameter_proposals", &MCMC::time_parameter_proposals)
           .def("MCMC_sweep", &MCMC::MCMC_sweep,
                py::arg("beta") = 1.0,
                py::arg("num_sweeps") = 1)
           .def("change_parameter", &MCMC::change_parameter,
                py::arg("param_name"),
                py::arg("new_value"));

    // --- FitResultNative class ---
    py::class_<FitResultNative>(m, "FitResult")
        .def_property_readonly("mdl_value",  &FitResultNative::mdl_value)
        .def_property_readonly("num_samples", &FitResultNative::num_samples)
        .def("mdl_partition",      &FitResultNative::mdl_partition)
        .def("consensus_partition",&FitResultNative::consensus_partition)
        .def("coincidence_matrix", &FitResultNative::coincidence_matrix_numpy)
        .def("score_means",        &FitResultNative::score_means)
        .def("score_stds",         &FitResultNative::score_stds)
        .def("samples_df",         &FitResultNative::samples_df)
        .def("summary",            &FitResultNative::summary);

    // --- fit_backend function ---
    m.def("fit_backend", [](
        std::vector<std::unordered_map<int, double>> wins,
        std::vector<std::unordered_map<int, double>> ties,
        std::vector<int>    initial_partition,
        bool                partition_fixed,
        std::vector<double> initial_scores,
        bool                scores_fixed,
        bool                assortative,
        py::object          variation_in_py,
        py::object          variation_out_py,
        py::object          degree_correction_py,
        py::object          individual_depth_py,
        py::object          group_depth_py,
        py::object          ties_parameter_py,
        py::object          num_groups_py,
        int    num_chains,
        int    samples_per_chain,
        int    sweeps_per_sample,
        int    burn_sweeps,
        int    thinning,
        double beta,
        int    num_tempering_chains,
        int    seed,
        double timeout,
        bool   show_progress
    ) {
        auto to_opt_double = [](py::object o) -> std::optional<double> {
            if (o.is_none()) return std::nullopt;
            return py::cast<double>(o);
        };
        auto to_opt_int = [](py::object o) -> std::optional<int> {
            if (o.is_none()) return std::nullopt;
            return py::cast<int>(o);
        };

        py::gil_scoped_release release;
        return fit_backend(
            std::move(wins), std::move(ties),
            std::move(initial_partition), partition_fixed,
            std::move(initial_scores), scores_fixed,
            assortative,
            to_opt_double(variation_in_py),
            to_opt_double(variation_out_py),
            to_opt_double(degree_correction_py),
            to_opt_double(individual_depth_py),
            to_opt_double(group_depth_py),
            to_opt_double(ties_parameter_py),
            to_opt_int(num_groups_py),
            num_chains, samples_per_chain,
            sweeps_per_sample, burn_sweeps, thinning,
            beta, num_tempering_chains,
            seed, timeout, show_progress
        );
    },
    py::arg("wins"),
    py::arg("ties"),
    py::arg("initial_partition"),
    py::arg("partition_fixed") = false,
    py::arg("initial_scores"),
    py::arg("scores_fixed") = false,
    py::arg("assortative") = true,
    py::arg("variation_in") = py::none(),
    py::arg("variation_out") = py::none(),
    py::arg("degree_correction") = py::none(),
    py::arg("individual_depth") = py::none(),
    py::arg("group_depth") = py::none(),
    py::arg("ties_parameter") = py::none(),
    py::arg("num_groups") = py::none(),
    py::arg("num_chains") = 1,
    py::arg("samples_per_chain") = 1000,
    py::arg("sweeps_per_sample") = 10,
    py::arg("burn_sweeps") = 500,
    py::arg("thinning") = 1,
    py::arg("beta") = 1.0,
    py::arg("num_tempering_chains") = 1,
    py::arg("seed") = 0,
    py::arg("timeout") = 60.0,
    py::arg("show_progress") = false);
}
