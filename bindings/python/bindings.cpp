#include "MCMC_core.h"
#include "globals.h"
#include "fit.h"
#include "fit_result.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>

// ---------------------------------------------------------------------------
// SIGINT handler installed around fit() so Ctrl+C works in both
// show_progress=True and show_progress=False modes.
//
// CPython's PyErr_CheckSignals() is a no-op from non-main threads, so the
// progress-thread approach cannot deliver interrupts. Instead, we swap in
// our own SIGINT handler that sets an abort_flag which fit() polls every
// sweep boundary.
// ---------------------------------------------------------------------------
static std::atomic<bool>* s_fit_abort = nullptr;
static void fit_sigint_handler(int /*sig*/) {
    if (s_fit_abort) s_fit_abort->store(true, std::memory_order_release);
}

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
        fill("two_layer_mode",    [](const SampleState& s){ return s.two_layer_mode; });
        fill("weighted_SBM",      [](const SampleState& s){ return s.weighted_SBM; });
        fill("H_W",               [](const SampleState& s){ return s.H_W; });
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

    py::dict network_properties_dict(double prob = 0.90) const {
        auto r = FitResult::network_properties(prob);
        py::dict d;
        d["parameter"] = py::cast(r.parameter);
        d["mean"]      = py::cast(r.mean);
        d["std"]       = py::cast(r.std);
        d["hdi_lo"]    = py::cast(r.hdi_lo);
        d["hdi_hi"]    = py::cast(r.hdi_hi);
        return d;
    }

    py::dict node_properties_dict(const std::string& mode = "auto",
                                  const std::string& consensus_method = "moc") const {
        auto r = FitResult::node_properties(mode, consensus_method);
        py::dict d;
        d["mode_used"]        = py::cast(r.mode_used);
        d["score_mean"]       = py::cast(r.score_mean);
        d["score_std"]        = py::cast(r.score_std);
        d["group"]            = py::cast(r.group);
        d["group_score_mean"] = py::cast(r.group_score_mean);
        return d;
    }

    py::object coincidence_matrix_numpy() const {
        auto flat = coincidence_matrix();
        int n = num_nodes_;
        py::module_ np = py::module_::import("numpy");
        return np.attr("reshape")(py::cast(flat), py::make_tuple(n, n));
    }

    py::tuple getstate() const {
        py::list lst;
        for (const auto& s : samples_) {
            py::dict d;
            d["chain_id"]          = s.chain_id;
            d["beta"]              = s.beta;
            d["partition"]         = s.partition;
            d["scores"]            = s.scores;
            d["num_groups"]        = s.num_groups;
            d["density_in"]        = s.density_in;
            d["density_out"]       = s.density_out;
            d["variation_in"]      = s.variation_in;
            d["variation_out"]     = s.variation_out;
            d["degree_correction"] = s.degree_correction;
            d["individual_depth"]  = s.individual_depth;
            d["group_depth"]       = s.group_depth;
            d["ties_parameter"]    = s.ties_parameter;
            d["H_A_b"]             = s.H_A_b;
            d["H_b"]               = s.H_b;
            d["H_M"]               = s.H_M;
            d["H_k"]               = s.H_k;
            d["H_A_G_k_M_b"]       = s.H_A_G_k_M_b;
            d["H_A_G_b"]           = s.H_A_G_b;
            d["H_Adir_G_A_s_nu"]   = s.H_Adir_G_A_s_nu;
            d["H_s_G_b_depths"]    = s.H_s_G_b_depths;
            d["two_layer_mode"]    = s.two_layer_mode;
            d["weighted_SBM"]      = s.weighted_SBM;
            d["H_W"]               = s.H_W;
            lst.append(d);
        }
        return py::make_tuple(num_nodes_, lst);
    }

    static FitResultNative setstate(py::tuple state) {
        int num_nodes = state[0].cast<int>();
        auto lst = state[1].cast<py::list>();
        std::vector<SampleState> samples;
        samples.reserve(py::len(lst));
        for (auto item : lst) {
            auto d = item.cast<py::dict>();
            SampleState s;
            s.chain_id          = d["chain_id"].cast<int>();
            s.beta              = d["beta"].cast<double>();
            s.partition         = d["partition"].cast<std::vector<int>>();
            s.scores            = d["scores"].cast<std::vector<double>>();
            s.num_groups        = d["num_groups"].cast<int>();
            s.density_in        = d["density_in"].cast<double>();
            s.density_out       = d["density_out"].cast<double>();
            s.variation_in      = d["variation_in"].cast<double>();
            s.variation_out     = d["variation_out"].cast<double>();
            s.degree_correction = d["degree_correction"].cast<double>();
            s.individual_depth  = d["individual_depth"].cast<double>();
            s.group_depth       = d["group_depth"].cast<double>();
            s.ties_parameter    = d["ties_parameter"].cast<double>();
            s.H_A_b             = d["H_A_b"].cast<double>();
            s.H_b               = d["H_b"].cast<double>();
            s.H_M               = d["H_M"].cast<double>();
            s.H_k               = d["H_k"].cast<double>();
            s.H_A_G_k_M_b       = d["H_A_G_k_M_b"].cast<double>();
            s.H_A_G_b           = d["H_A_G_b"].cast<double>();
            s.H_Adir_G_A_s_nu   = d["H_Adir_G_A_s_nu"].cast<double>();
            s.H_s_G_b_depths    = d["H_s_G_b_depths"].cast<double>();
            s.two_layer_mode    = d.contains("two_layer_mode") ? d["two_layer_mode"].cast<bool>() : false;
            s.weighted_SBM      = d.contains("weighted_SBM")  ? d["weighted_SBM"].cast<bool>()   : false;
            s.H_W               = d.contains("H_W")           ? d["H_W"].cast<double>()           : 0.0;
            samples.push_back(std::move(s));
        }
        return FitResultNative(FitResult(num_nodes, std::move(samples)));
    }
};

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(MCMC_core, m) {
    // Merge/split tuning dials.  Auxiliary variables: they change the proposal
    // distribution and hence the acceptance rate, never the target (see
    // src/MCMC_merge_split.cpp).  Exposed for experiments, not part of the API.
    m.def("set_prestage_mode", [](int mode){ MERGE_SPLIT_PRESTAGE_MODE = mode; },
          py::arg("mode"),
          "0=all three (default), 1=random split, 2=sequential spreading, 3=sequential coalescence");
    m.def("get_prestage_mode", [](){ return MERGE_SPLIT_PRESTAGE_MODE; });
    m.def("set_staging_sweeps", [](int M){ MERGE_SPLIT_GIBBS_SWEEPS = M; }, py::arg("M"),
          "Gibbs sweeps used to refine a staged split (Peixoto's M)");
    m.def("get_staging_sweeps", [](){ return MERGE_SPLIT_GIBBS_SWEEPS; });
    m.def("set_merge_split_weight", [](double w){ MERGE_SPLIT_WEIGHT = w; }, py::arg("w"),
          "relative weight of each collective move; <0 = linear-time default");
    m.def("get_merge_split_weight", [](){ return MERGE_SPLIT_WEIGHT; });
    m.def("set_individual_depth_centred_prob", [](double p){ INDIVIDUAL_DEPTH_CENTRED_PROB = p; }, py::arg("p"),
          "probability that a sigma_I proposal is the centred move (scores held fixed)");
    m.def("get_individual_depth_centred_prob", [](){ return INDIVIDUAL_DEPTH_CENTRED_PROB; });
    m.def("set_individual_depth_scale_move", [](int k){ INDIVIDUAL_DEPTH_SCALE_MOVE = k; }, py::arg("k"),
          "0: non-centred sigma_I move (all scores scale); 1: residual rescale (group means fixed)");
    m.def("set_equal_spend_c", [](double c){ MERGE_SPLIT_EQUAL_SPEND_C = c; }, py::arg("c"),
          "divisor of fit()'s equal-spend rule w = q/(C*M); larger = fewer collective moves");
    m.def("get_equal_spend_c", [](){ return MERGE_SPLIT_EQUAL_SPEND_C; });

    // --- MCMC class (kept for testing / advanced use) ---
    // module_local(): asbm and community_hierarchy each define a bare (un-namespaced)
    // MCMC and FitResultNative, so both compile to the same RTTI name.  pybind11's
    // type registry is shared process-wide, so without this the second package
    // imported raises  ImportError: generic_type: type "MCMC" is already registered
    // (or, where the import error is caught, degrades silently).  Keep it on both
    // classes in BOTH packages: either can be imported first.
    py::class_<MCMC>(m, "MCMC", py::module_local())
        .def(py::init<
                 std::vector<std::unordered_map<int, double>>,
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
                 double,
                 bool> (),
             py::arg("wins"),
             py::arg("ties"),
             py::arg("group_adj") = std::vector<std::unordered_map<int, double>>{},
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
             py::arg("degree_correction") = 0.5,
             py::arg("degree_correction_fixed") = false,
             py::arg("individual_depth") = 1.0,
             py::arg("individual_depth_fixed") = false,
             py::arg("group_depth") = 1.0,
             py::arg("group_depth_fixed") = false,
             py::arg("ties_parameter") = 1.0,
             py::arg("ties_parameter_fixed") = false,
             py::arg("seed") = 0,
             py::arg("density_in") = -1.0,
             py::arg("density_out") = -1.0,
             py::arg("weighted_SBM") = false)
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
           .def_readonly("H_W",                &MCMC::H_W)
           .def_readonly("H_Adir_G_A_s_nu",    &MCMC::H_Adir_G_A_s_nu)
           .def_readonly("H_s_G_b_depths",     &MCMC::H_s_G_b_depths)
           .def_readonly("weighted_SBM",        &MCMC::weighted_SBM)
           .def_readwrite("weighted_proposals",  &MCMC::weighted_proposals)
           .def_readwrite("adapting",           &MCMC::adapting)
           .def_readonly("step_score",          &MCMC::step_score)
           .def_readonly("step_individual_depth", &MCMC::step_individual_depth)
           .def_readonly("step_group_depth",    &MCMC::step_group_depth)
           .def_readonly("step_ties",           &MCMC::step_ties)
           .def_readonly("step_group_score",    &MCMC::step_group_score)
           .def_readonly("single_site_proposals",          &MCMC::single_site_proposals)
           .def_readonly("accepted_single_site_proposals", &MCMC::accepted_single_site_proposals)
           .def_readonly("score_proposals",                &MCMC::score_proposals)
           .def_readonly("accepted_score_proposals",       &MCMC::accepted_score_proposals)
           .def_readonly("group_score_proposals",          &MCMC::group_score_proposals)
           .def_readonly("accepted_group_score_proposals", &MCMC::accepted_group_score_proposals)
           .def_readonly("group_depth_proposals",          &MCMC::group_depth_proposals)
           .def_readonly("accepted_group_depth_proposals", &MCMC::accepted_group_depth_proposals)
           .def_readonly("individual_depth_proposals",          &MCMC::individual_depth_proposals)
           .def_readonly("accepted_individual_depth_proposals", &MCMC::accepted_individual_depth_proposals)
           .def_readonly("new_group_proposals",            &MCMC::new_group_proposals)
           .def_readonly("accepted_new_group_proposals",   &MCMC::accepted_new_group_proposals)
           .def_readonly("vacate_proposals",               &MCMC::vacate_proposals)
           .def_readonly("accepted_vacate_proposals",      &MCMC::accepted_vacate_proposals)
           // time_*: profiling counters, filled only when the global `timing` flag is
           // set at compile time (globals.cpp); 0 otherwise.
           .def_readonly("time_H_k",                    &MCMC::time_H_k)
           .def_readonly("time_H_M",                    &MCMC::time_H_M)
           .def_readonly("time_H_A_G_k_M_b",            &MCMC::time_H_A_G_k_M_b)
           .def_readonly("time_single_site_proposals",   &MCMC::time_single_site_proposals)
           .def_readonly("time_parameter_proposals",     &MCMC::time_parameter_proposals)
           .def("MCMC_sweep", &MCMC::MCMC_sweep,
                py::arg("beta") = 1.0,
                py::arg("num_sweeps") = 1,
                py::arg("merge_split_enabled") = true)
           .def_readonly("merge_proposals",                &MCMC::merge_proposals)
           .def_readonly("accepted_merge_proposals",       &MCMC::accepted_merge_proposals)
           .def_readonly("split_proposals",                &MCMC::split_proposals)
           .def_readonly("accepted_split_proposals",       &MCMC::accepted_split_proposals)
           .def_readonly("merge_split_proposals",          &MCMC::merge_split_proposals)
           .def_readonly("accepted_merge_split_proposals", &MCMC::accepted_merge_split_proposals)
           .def("calculate_H_M", &MCMC::calculate_H_M)
           // single moves, for exactness tests of each kernel on its own
           .def("propose_single_site_move", &MCMC::propose_single_site_move)
           .def("propose_score_move",       &MCMC::propose_score_move)
           .def("propose_group_score_move", &MCMC::propose_group_score_move, py::arg("r") = -1)
           .def("propose_parameter_move",   &MCMC::propose_parameter_move)
           .def("propose_merge_move",       &MCMC::propose_merge_move)
           .def("propose_split_move",       &MCMC::propose_split_move)
           .def("propose_merge_split_move", &MCMC::propose_merge_split_move)
           .def("set_beta", [](MCMC& self, double b) { self.beta = b; })
           .def("check_state_bookkeeping", &MCMC::check_state_bookkeeping, py::arg("throw_error") = true)
           .def("change_parameter", &MCMC::change_parameter,
                py::arg("param_name"),
                py::arg("new_value"),
                "One Metropolis-Hastings step of a continuous parameter towards new_value, "
                "which must be drawn from a CDF-space proposal of that parameter's prior "
                "(the prior is not in the acceptance). Not a setter: the value may be "
                "rejected. individual_depth rescales the scores with it unless scores are "
                "fixed.");

    // --- FitResultNative class ---
    py::class_<FitResultNative>(m, "FitResult", py::module_local())
        .def_property_readonly("mdl_value",   &FitResultNative::mdl_value)
        .def_property_readonly("num_samples", &FitResultNative::num_samples)
        .def("mdl_partition",       &FitResultNative::mdl_partition)
        .def("consensus_partition", &FitResultNative::consensus_partition,
             py::arg("method") = "moc", py::arg("refine") = true,
             py::arg("min_samples") = 1000)
        .def("consensus_sample_indices", &FitResultNative::consensus_sample_indices,
             py::arg("min_samples") = 1000)
        .def("partition_ess",       &FitResultNative::partition_ess)
        .def("modal_num_groups",    &FitResultNative::modal_num_groups)
        .def("coincidence_matrix",  &FitResultNative::coincidence_matrix_numpy)
        .def("score_means",         &FitResultNative::score_means)
        .def("score_stds",          &FitResultNative::score_stds)
        .def("parameter_means",     &FitResultNative::parameter_means)
        .def("parameter_stds",      &FitResultNative::parameter_stds)
        .def("direction_probabilities", &FitResultNative::direction_probabilities,
             py::arg("node_a"), py::arg("node_b"), py::arg("ties_allowed") = true)
        .def("network_properties",   &FitResultNative::network_properties_dict,
             py::arg("prob") = 0.90)
        .def("node_properties",     &FitResultNative::node_properties_dict,
             py::arg("mode") = "auto", py::arg("consensus_method") = "moc")
        .def("samples_df",          &FitResultNative::samples_df)
        .def("summary",             &FitResultNative::summary)
        .def("__getstate__",        &FitResultNative::getstate)
        .def("__setstate__",        [](FitResultNative& self, py::tuple state) {
            new (&self) FitResultNative(FitResultNative::setstate(state));
        });

    // --- fit_backend: thin type-conversion wrapper around the C++ core fit() ---
    m.def("fit_backend", [](
        std::vector<std::unordered_map<int, double>> wins,
        std::vector<std::unordered_map<int, double>> ties,
        std::vector<std::unordered_map<int, double>> group_adj,
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
        bool   merge_split_enabled,
        int    seed,
        double timeout,
        bool   show_progress,
        bool   weighted_SBM,
        bool   weighted_proposals
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
        params.group_adj            = std::move(group_adj);
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
        params.merge_split_enabled  = merge_split_enabled;
        params.seed                 = seed;
        params.timeout              = timeout;
        params.weighted_SBM        = weighted_SBM;
        params.weighted_proposals  = weighted_proposals;

        // Abort flag — set by our SIGINT handler so fit() can exit early.
        std::atomic<bool> abort_flag{false};
        params.abort_flag = &abort_flag;

        // Install a custom SIGINT handler for the duration of fit().
        // We swap back the previous handler once fit() returns, then raise
        // KeyboardInterrupt if the flag was set.
        s_fit_abort = &abort_flag;
        auto old_sigint = std::signal(SIGINT, fit_sigint_handler);

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

        // Restore the original SIGINT handler before doing anything Python-side.
        std::signal(SIGINT, old_sigint);
        s_fit_abort = nullptr;

        if (progress_thread.joinable()) {
            { std::lock_guard<std::mutex> lock(mtx); fit_finished = true; }
            cv.notify_all();
            progress_thread.join();
        }

        // If the abort flag was set (Ctrl+C during fit), raise KeyboardInterrupt.
        if (abort_flag.load(std::memory_order_relaxed)) {
            PyErr_SetNone(PyExc_KeyboardInterrupt);
            throw py::error_already_set();
        }

        return FitResultNative(std::move(result));
    },
    py::arg("wins"),
    py::arg("ties"),
    py::arg("group_adj")                  = std::vector<std::unordered_map<int, double>>{},
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
    py::arg("merge_split_enabled")       = true,
    py::arg("seed")                      = 0,
    py::arg("timeout")                   = 60.0,
    py::arg("show_progress")             = false,
    py::arg("weighted_SBM")              = false,
    py::arg("weighted_proposals")        = false);
}
