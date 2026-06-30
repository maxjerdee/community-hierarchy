#include "MCMC_core.h"
#include "fit.h"
#include "fit_result.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// FitResultNative — extends FitResult with Python-facing helpers
// ---------------------------------------------------------------------------

class FitResultNative : public FitResult {
public:
    FitResultNative() = default;
    FitResultNative(FitResult r) : FitResult(std::move(r)) {}

    py::object samples_df() const {
        py::module_ pd = py::module_::import("pandas");
        int ns = static_cast<int>(samples_.size());
        int n  = num_nodes_;

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
        double mean_k = 0.0, mean_dl = 0.0;
        for (const auto& s : samples_) {
            mean_k  += s.num_groups;
            mean_dl += s.H_A_b + s.H_Adir_G_A_s_nu + s.H_s_G_b_depths;
        }
        int ns = static_cast<int>(samples_.size());
        if (ns > 0) { mean_k /= ns; mean_dl /= ns; }
        py::dict d;
        d["num_samples"]             = ns;
        d["mean_num_groups"]         = mean_k;
        d["mean_description_length"] = mean_dl;
        d["mdl_value"]               = mdl_value();
        return d;
    }

    py::object coincidence_matrix_numpy() const {
        auto flat = coincidence_matrix();
        int n = num_nodes_;
        py::module_ np = py::module_::import("numpy");
        return np.attr("reshape")(py::cast(flat), py::make_tuple(n, n));
    }
};

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
           .def_readonly("num_groups",        &MCMC::num_groups)
           .def_readonly("partition",          &MCMC::partition)
           .def_readonly("scores",             &MCMC::scores)
           .def_readonly("density_in",         &MCMC::density_in)
           .def_readonly("density_out",        &MCMC::density_out)
           .def_readonly("variation_in",       &MCMC::variation_in)
           .def_readonly("variation_out",      &MCMC::variation_out)
           .def_readonly("degree_correction",  &MCMC::degree_correction)
           .def_readonly("individual_depth",   &MCMC::individual_depth)
           .def_readonly("group_depth",        &MCMC::group_depth)
           .def_readonly("ties_parameter",     &MCMC::ties_parameter)
           .def_readonly("H_A_b",              &MCMC::H_A_b)
           .def_readonly("H_b",                &MCMC::H_b)
           .def_readonly("H_M",                &MCMC::H_M)
           .def_readonly("H_k",                &MCMC::H_k)
           .def_readonly("H_A_G_k_M_b",        &MCMC::H_A_G_k_M_b)
           .def_readonly("H_A_G_b",            &MCMC::H_A_G_b)
           .def_readonly("H_Adir_G_A_s_nu",    &MCMC::H_Adir_G_A_s_nu)
           .def_readonly("H_s_G_b_depths",     &MCMC::H_s_G_b_depths)
           .def_readonly("time_H_k",                    &MCMC::time_H_k)
           .def_readonly("time_H_M",                    &MCMC::time_H_M)
           .def_readonly("time_H_A_G_k_M_b",            &MCMC::time_H_A_G_k_M_b)
           .def_readonly("time_single_site_proposals",   &MCMC::time_single_site_proposals)
           .def_readonly("time_parameter_proposals",     &MCMC::time_parameter_proposals)
           .def("MCMC_sweep", &MCMC::MCMC_sweep,
                py::arg("beta") = 1.0,
                py::arg("num_sweeps") = 1)
           .def("change_parameter", &MCMC::change_parameter,
                py::arg("param_name"),
                py::arg("new_value"));

    // --- FitResultNative class ---
    py::class_<FitResultNative>(m, "FitResult")
        .def_property_readonly("mdl_value",   &FitResultNative::mdl_value)
        .def_property_readonly("num_samples", &FitResultNative::num_samples)
        .def("mdl_partition",       &FitResultNative::mdl_partition)
        .def("consensus_partition", &FitResultNative::consensus_partition)
        .def("coincidence_matrix",  &FitResultNative::coincidence_matrix_numpy)
        .def("score_means",         &FitResultNative::score_means)
        .def("score_stds",          &FitResultNative::score_stds)
        .def("parameter_means",     &FitResultNative::parameter_means)
        .def("parameter_stds",      &FitResultNative::parameter_stds)
        .def("samples_df",          &FitResultNative::samples_df)
        .def("summary",             &FitResultNative::summary);

    // --- fit_backend: thin type-conversion wrapper around the C++ core fit() ---
    m.def("fit_backend", [](
        std::vector<std::unordered_map<int, double>> wins,
        std::vector<std::unordered_map<int, double>> ties,
        std::vector<int>    initial_partition,
        bool                partition_fixed,
        std::vector<double> initial_scores,
        bool                scores_fixed,
        bool                assortative,
        py::object variation_in_py,
        py::object variation_out_py,
        py::object degree_correction_py,
        py::object individual_depth_py,
        py::object individual_depth_init_py,
        py::object group_depth_py,
        py::object group_depth_init_py,
        py::object ties_parameter_py,
        py::object num_groups_py,
        bool   num_groups_fixed,
        int    num_chains,
        int    samples_per_chain,
        int    sweeps_per_sample,
        int    burn_sweeps,
        int    param_warmup_sweeps,
        int    thinning,
        double beta,
        int    num_tempering_chains,
        int    seed,
        double timeout,
        bool   show_progress
    ) {
        auto to_opt_d = [](py::object o) -> std::optional<double> {
            return o.is_none() ? std::nullopt : std::optional<double>(py::cast<double>(o));
        };
        auto to_opt_i = [](py::object o) -> std::optional<int> {
            return o.is_none() ? std::nullopt : std::optional<int>(py::cast<int>(o));
        };

        FitParams params;
        params.wins                 = std::move(wins);
        params.ties                 = std::move(ties);
        params.initial_partition    = std::move(initial_partition);
        params.partition_fixed      = partition_fixed;
        params.initial_scores       = std::move(initial_scores);
        params.scores_fixed         = scores_fixed;
        params.assortative          = assortative;
        params.variation_in         = to_opt_d(variation_in_py);
        params.variation_out        = to_opt_d(variation_out_py);
        params.degree_correction    = to_opt_d(degree_correction_py);
        params.individual_depth     = to_opt_d(individual_depth_py);
        params.individual_depth_init = to_opt_d(individual_depth_init_py);
        params.group_depth          = to_opt_d(group_depth_py);
        params.group_depth_init     = to_opt_d(group_depth_init_py);
        params.ties_parameter       = to_opt_d(ties_parameter_py);
        params.num_groups           = to_opt_i(num_groups_py);
        params.num_groups_fixed     = num_groups_fixed;
        params.num_chains           = num_chains;
        params.samples_per_chain    = samples_per_chain;
        params.sweeps_per_sample    = sweeps_per_sample;
        params.burn_sweeps          = burn_sweeps;
        params.param_warmup_sweeps  = param_warmup_sweeps;
        params.thinning             = thinning;
        params.beta                 = beta;
        params.num_tempering_chains = num_tempering_chains;
        params.seed                 = seed;
        params.timeout              = timeout;

        // Progress reporting: phase-aware stderr display (pystan style).
        std::mutex mtx;
        std::condition_variable cv;
        std::atomic<int> warmup_done{0}, burn_done{0}, sample_done{0};
        bool fit_finished = false;
        std::thread progress_thread;

        if (show_progress) {
            params.on_warmup = [&](int completed, int) {
                warmup_done.store(completed, std::memory_order_relaxed);
                cv.notify_one();
            };
            params.on_burn_in = [&](int completed, int) {
                burn_done.store(completed, std::memory_order_relaxed);
                cv.notify_one();
            };
            params.on_sample = [&](int completed, int) {
                sample_done.store(completed, std::memory_order_relaxed);
                cv.notify_one();
            };

            const int warmup_total = num_chains * param_warmup_sweeps;
            const int burn_total   = num_chains * burn_sweeps;
            const int sample_total = num_chains * samples_per_chain;

            progress_thread = std::thread([&, warmup_total, burn_total, sample_total]() {
                auto emit = [](const char* label, int cur, int total) {
                    int pct = total > 0 ? static_cast<int>(100.0 * cur / total) : 100;
                    std::ostringstream ss;
                    ss << '\r' << std::left  << std::setw(11) << label
                       << std::right << std::setw(6) << cur << " / " << total
                       << " [" << std::setw(3) << pct << "%]";
                    std::cerr << ss.str() << std::flush;
                };

                auto run_phase = [&](std::atomic<int>& counter, int total,
                                     const char* label) {
                    if (total <= 0) return;
                    emit(label, 0, total);
                    int last = -1;
                    while (true) {
                        std::unique_lock<std::mutex> lock(mtx);
                        cv.wait_for(lock, std::chrono::milliseconds(100),
                            [&]{ return fit_finished || counter.load() != last; });
                        int cur = counter.load(std::memory_order_relaxed);
                        bool stopping = fit_finished;
                        lock.unlock();
                        if (cur != last) { emit(label, cur, total); last = cur; }
                        if (cur >= total || stopping) break;
                    }
                    // Only print a final line if timeout ended the phase early.
                    int final_cur = counter.load(std::memory_order_relaxed);
                    if (final_cur < total)
                        emit(label, final_cur, total);
                    std::cerr << '\n' << std::flush;
                };

                // Show both param-warmup and burn-in under a single "Warmup:" line.
                const int total_warmup_combined = warmup_total + burn_total;
                if (total_warmup_combined > 0) {
                    emit("Warmup:   ", 0, total_warmup_combined);
                    int last = -1;
                    while (true) {
                        std::unique_lock<std::mutex> lock(mtx);
                        int cur = warmup_done.load(std::memory_order_relaxed)
                                + burn_done.load(std::memory_order_relaxed);
                        cv.wait_for(lock, std::chrono::milliseconds(100),
                            [&]{ return fit_finished || cur != last; });
                        cur = warmup_done.load(std::memory_order_relaxed)
                            + burn_done.load(std::memory_order_relaxed);
                        bool stopping = fit_finished;
                        lock.unlock();
                        if (cur != last) { emit("Warmup:   ", cur, total_warmup_combined); last = cur; }
                        if (cur >= total_warmup_combined || stopping) break;
                    }
                    int final_cur = warmup_done.load() + burn_done.load();
                    if (final_cur < total_warmup_combined)
                        emit("Warmup:   ", final_cur, total_warmup_combined);
                    std::cerr << '\n' << std::flush;
                }
                run_phase(sample_done, sample_total, "Sampling: ");
            });
        }

        FitResult result = [&]() {
            py::gil_scoped_release release;
            return fit(std::move(params));
        }();

        if (progress_thread.joinable()) {
            { std::lock_guard<std::mutex> lock(mtx); fit_finished = true; }
            cv.notify_all();
            progress_thread.join();
        }

        return FitResultNative(std::move(result));
    },
    py::arg("wins"),
    py::arg("ties"),
    py::arg("initial_partition"),
    py::arg("partition_fixed")           = false,
    py::arg("initial_scores"),
    py::arg("scores_fixed")              = false,
    py::arg("assortative")               = true,
    py::arg("variation_in")              = py::none(),
    py::arg("variation_out")             = py::none(),
    py::arg("degree_correction")         = py::none(),
    py::arg("individual_depth")          = py::none(),
    py::arg("initial_individual_depth")  = py::none(),
    py::arg("group_depth")               = py::none(),
    py::arg("initial_group_depth")       = py::none(),
    py::arg("ties_parameter")            = py::none(),
    py::arg("num_groups")                = py::none(),
    py::arg("num_groups_fixed")          = false,
    py::arg("num_chains")                = 1,
    py::arg("samples_per_chain")         = 1000,
    py::arg("sweeps_per_sample")         = 10,
    py::arg("burn_sweeps")               = 500,
    py::arg("param_warmup_sweeps")       = 0,
    py::arg("thinning")                  = 1,
    py::arg("beta")                      = 1.0,
    py::arg("num_tempering_chains")      = 1,
    py::arg("seed")                      = 0,
    py::arg("timeout")                   = 60.0,
    py::arg("show_progress")             = false);
}
