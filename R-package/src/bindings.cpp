// R bindings of the community-hierarchy C++ core.
//
// The core (every other .cpp / .h in this directory) is a copy of the repository's
// src/; edit it there and run tools/sync_r_core.py.
//
// A fit returns its samples to R as plain vectors and matrices (one row per sample),
// so an R result can be saved with saveRDS(), together with a handle to the C++
// FitResult, which the summaries use.  A handle does not survive saveRDS(); the R
// side then rebuilds it once from the samples (.ch_build_result).

#include <R_ext/Utils.h>
#include <Rcpp.h>

#include <atomic>
#include <chrono>
#include <exception>
#include <future>
#include <string>
#include <unordered_map>
#include <vector>

#include "diagnostics.h"
#include "fit.h"
#include "fit_result.h"
#include "interactions.h"
#include "loo.h"
#include "model_presets.h"

using namespace Rcpp;

namespace {

std::optional<double> opt_double(const List& opts, const char* name) {
    SEXP x = opts[name];
    if (Rf_isNull(x)) return std::nullopt;
    return as<double>(x);
}

std::optional<int> opt_int(const List& opts, const char* name) {
    SEXP x = opts[name];
    if (Rf_isNull(x)) return std::nullopt;
    return as<int>(x);
}

// R_CheckUserInterrupt() longjmps on an interrupt; run it inside R_ToplevelExec so
// the C++ stack is unwound normally instead.
void check_interrupt_fn(void*) {
    R_CheckUserInterrupt();
}
bool interrupt_pending() {
    return R_ToplevelExec(check_interrupt_fn, nullptr) == FALSE;
}

void progress_line(const char* label, int cur, int total) {
    const int pct = total > 0 ? static_cast<int>(100.0 * cur / total) : 100;
    REprintf("\r%-10s %6d / %d [%3d%%]", label, cur, total, pct);
}

// The samples of a fit as R objects: params (a list of columns, the fields of
// sample_fields() then sigma_T and ICC), and the partitions and scores as
// samples x nodes matrices.
List samples_to_r(const FitResult& res) {
    const auto& ss = res.samples();
    const int m = static_cast<int>(ss.size()), n = res.num_nodes();
    IntegerMatrix partitions(m, n);
    NumericMatrix scores(m, n);
    for (int k = 0; k < m; ++k)
        for (int i = 0; i < n; ++i) {
            partitions(k, i) = ss[k].partition[i];
            scores(k, i) = ss[k].scores[i];
        }
    List params;
    for (const SampleField& f : sample_fields()) {
        NumericVector col(m);
        for (int k = 0; k < m; ++k) col[k] = f.get(ss[k]);
        if (f.type == SampleField::Int)
            params[f.name] = as<IntegerVector>(col);
        else if (f.type == SampleField::Bool)
            params[f.name] = as<LogicalVector>(col);
        else
            params[f.name] = col;
    }
    for (const auto& p : network_parameters()) {
        if (params.containsElementNamed(p.name)) continue;  // sigma_T and ICC
        NumericVector col(m);
        for (int k = 0; k < m; ++k) col[k] = p.get(ss[k]);
        params[p.name] = col;
    }
    return List::create(_["params"] = params, _["partitions"] = partitions, _["scores"] = scores);
}

// The inverse of samples_to_r: a FitResult for the summaries.
FitResult build_result(const List& params, const IntegerMatrix& partitions,
                       const NumericMatrix& scores) {
    const int m = partitions.nrow(), n = partitions.ncol();
    std::vector<SampleState> ss(static_cast<std::size_t>(m));
    for (int k = 0; k < m; ++k) {
        ss[k].partition.resize(n);
        ss[k].scores.resize(n);
        for (int i = 0; i < n; ++i) {
            ss[k].partition[i] = partitions(k, i);
            ss[k].scores[i] = scores(k, i);
        }
    }
    for (const SampleField& f : sample_fields()) {
        if (!params.containsElementNamed(f.name)) continue;
        const NumericVector col = as<NumericVector>(params[f.name]);
        for (int k = 0; k < m; ++k) f.set(ss[k], col[k]);
    }
    return FitResult(n, std::move(ss));
}

}  // namespace

// [[Rcpp::export(.ch_fit_native)]]
List ch_fit_native(int n, IntegerVector from, IntegerVector to, LogicalVector draw,
                   NumericVector weight, bool source_high_rank, IntegerVector group_a,
                   IntegerVector group_b, NumericVector group_weight, bool two_layer,
                   IntegerVector initial_partition, bool partition_fixed,
                   NumericVector initial_scores, bool scores_fixed, List opts, bool show_progress) {
    std::vector<bool> is_draw(draw.size());
    for (R_xlen_t k = 0; k < draw.size(); ++k) is_draw[k] = draw[k] == TRUE;
    Interactions edges =
        build_interactions(n, as<std::vector<int>>(from), as<std::vector<int>>(to), is_draw,
                           as<std::vector<double>>(weight), source_high_rank);
    FitParams p;
    p.wins = edges.wins;
    p.ties = edges.ties;
    if (two_layer)
        p.group_adj =
            build_group_layer(n, as<std::vector<int>>(group_a), as<std::vector<int>>(group_b),
                              as<std::vector<double>>(group_weight));
    p.initial_partition = as<std::vector<int>>(initial_partition);
    p.partition_fixed = partition_fixed;
    p.initial_scores = as<std::vector<double>>(initial_scores);
    p.scores_fixed = scores_fixed;

    p.assortative = as<bool>(opts["assortative"]);
    p.variation_in = opt_double(opts, "variation_in");
    p.variation_out = opt_double(opts, "variation_out");
    p.degree_correction = opt_double(opts, "degree_correction");
    p.individual_depth = opt_double(opts, "individual_depth");
    p.individual_depth_init = opt_double(opts, "initial_individual_depth");
    p.group_depth = opt_double(opts, "group_depth");
    p.group_depth_init = opt_double(opts, "initial_group_depth");
    p.ties_parameter = opt_double(opts, "ties_parameter");
    p.num_groups = opt_int(opts, "num_groups");
    p.num_groups_fixed = as<bool>(opts["num_groups_fixed"]);
    p.weighted_SBM = as<bool>(opts["weighted_SBM"]);
    p.weighted_proposals = as<bool>(opts["weighted_proposals"]);

    p.num_chains = as<int>(opts["num_chains"]);
    p.num_samples = as<int>(opts["num_samples"]);
    p.num_warmup = as<int>(opts["num_warmup"]);
    p.num_thin = as<int>(opts["num_thin"]);
    p.sweeps_per_sample = as<int>(opts["sweeps_per_sample"]);
    p.num_param_warmup = as<int>(opts["num_param_warmup"]);
    p.beta = as<double>(opts["beta"]);
    p.num_tempering_chains = as<int>(opts["num_tempering_chains"]);
    p.merge_split_enabled = as<bool>(opts["merge_split_enabled"]);
    p.seed = as<int>(opts["seed"]);
    p.timeout = as<double>(opts["timeout"]);

    // The chains run on worker threads, which must not touch R.  This (R's) thread
    // waits, prints progress and watches for an interrupt, which stops the fit
    // through the abort flag.
    std::atomic<bool> abort_flag{false};
    std::atomic<int> warm_done{0}, burn_done{0}, sample_done{0};
    p.abort_flag = &abort_flag;
    if (show_progress) {
        p.on_warmup = [&](int c, int) { warm_done.store(c, std::memory_order_relaxed); };
        p.on_burn_in = [&](int c, int) { burn_done.store(c, std::memory_order_relaxed); };
        p.on_sample = [&](int c, int) { sample_done.store(c, std::memory_order_relaxed); };
    }
    const int warm_total = p.num_chains * (p.num_param_warmup + p.num_warmup);
    const int sample_total = p.num_chains * draws_per_chain(p);

    std::future<FitResult> job = std::async(
        std::launch::async, [params = std::move(p)]() mutable { return fit(std::move(params)); });
    bool interrupted = false;
    int shown_warm = -1, shown_sample = -1;
    bool warm_line_open = false;
    while (job.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready) {
        if (!interrupted && interrupt_pending()) {
            interrupted = true;
            abort_flag.store(true);
        }
        if (show_progress) {
            const int w = warm_done.load() + burn_done.load(), s = sample_done.load();
            if (s == 0 && w != shown_warm && warm_total > 0) {
                progress_line("Warmup:", w, warm_total);
                shown_warm = w;
                warm_line_open = true;
            } else if (s > 0 && s != shown_sample) {
                if (warm_line_open) {
                    progress_line("Warmup:", w, warm_total);
                    REprintf("\n");
                    warm_line_open = false;
                }
                progress_line("Sampling:", s, sample_total);
                shown_sample = s;
            }
        }
    }
    FitResult res;
    std::exception_ptr err;
    try {
        res = job.get();
    } catch (...) {
        err = std::current_exception();
    }
    if (show_progress && (shown_warm >= 0 || shown_sample >= 0)) {
        if (shown_sample >= 0) progress_line("Sampling:", sample_done.load(), sample_total);
        REprintf("\n");
    }
    if (interrupted) throw Rcpp::internal::InterruptedException();
    if (err) std::rethrow_exception(err);
    List out = samples_to_r(res);
    out["native"] = XPtr<FitResult>(new FitResult(std::move(res)), true);
    // The edges as the core reads them, nodes from 1: a won against b, or they drew.
    IntegerVector a(edges.a.begin(), edges.a.end()), b(edges.b.begin(), edges.b.end());
    out["interactions"] = List::create(_["a"] = a + 1, _["b"] = b + 1, _["draw"] = wrap(edges.draw),
                                       _["weight"] = wrap(edges.weight));
    return out;
}

// [[Rcpp::export(.ch_build_result)]]
SEXP ch_build_result(List params, IntegerMatrix partitions, NumericMatrix scores) {
    return XPtr<FitResult>(new FitResult(build_result(params, partitions, scores)), true);
}

// [[Rcpp::export(.ch_handle_valid)]]
bool ch_handle_valid(SEXP handle) {
    return TYPEOF(handle) == EXTPTRSXP && R_ExternalPtrAddr(handle) != nullptr;
}

// [[Rcpp::export(.ch_network_properties)]]
List ch_network_properties(XPtr<FitResult> res, double prob) {
    const NetworkPropertiesResult net = res->network_properties(prob);
    return List::create(_["parameter"] = net.parameter, _["mean"] = net.mean, _["std"] = net.std,
                        _["hdi_lo"] = net.hdi_lo, _["hdi_hi"] = net.hdi_hi);
}

// [[Rcpp::export(.ch_node_properties)]]
List ch_node_properties(XPtr<FitResult> res, std::string mode, std::string consensus_method) {
    const NodePropertiesResult nodes = res->node_properties(mode, consensus_method);
    return List::create(_["score_mean"] = nodes.score_mean, _["score_std"] = nodes.score_std,
                        _["group"] = nodes.group, _["group_score_mean"] = nodes.group_score_mean,
                        _["mode_used"] = nodes.mode_used);
}

// [[Rcpp::export(.ch_consensus_partition)]]
IntegerVector ch_consensus_partition(XPtr<FitResult> res, std::string method, bool refine,
                                     int min_samples) {
    return wrap(FitResult::sort_groups_by_score(
        res->consensus_partition(method, refine, min_samples), res->score_means()));
}

// [[Rcpp::export(.ch_mdl)]]
List ch_mdl(XPtr<FitResult> res) {
    return List::create(
        _["value"] = res->mdl_value(),
        _["partition"] = FitResult::sort_groups_by_score(res->mdl_partition(), res->score_means()));
}

// [[Rcpp::export(.ch_partition_ess)]]
double ch_partition_ess(XPtr<FitResult> res) {
    return res->partition_ess();
}

// [[Rcpp::export(.ch_coincidence_matrix)]]
NumericMatrix ch_coincidence_matrix(XPtr<FitResult> res) {
    const int n = res->num_nodes();
    const std::vector<double> c = res->coincidence_matrix();
    NumericMatrix out(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) out(i, j) = c[static_cast<size_t>(i) * n + j];
    return out;
}

// [[Rcpp::export(.ch_direction_probabilities)]]
NumericVector ch_direction_probabilities(XPtr<FitResult> res, int a, int b, bool ties_allowed) {
    return wrap(res->direction_probabilities(a, b, ties_allowed));
}

// [[Rcpp::export(.ch_direction_loo)]]
List ch_direction_loo(XPtr<FitResult> res, IntegerVector a, IntegerVector b, LogicalVector draw,
                      NumericVector weight, double r_eff) {
    std::vector<bool> d(draw.size());
    for (R_xlen_t i = 0; i < draw.size(); ++i) d[i] = draw[i] == TRUE;
    DirectionLooResult r = direction_loo(*res, as<std::vector<int>>(a), as<std::vector<int>>(b), d,
                                         as<std::vector<double>>(weight), r_eff);
    return List::create(_["elpd"] = r.elpd, _["pareto_k"] = r.pareto_k, _["p_win"] = r.p_win);
}

namespace {
List diagnostics_to_r(const DiagnosticsResult& r) {
    List out =
        List::create(_["ess_bulk"] = r.ess_bulk, _["ess_tail"] = r.ess_tail, _["rhat"] = r.rhat);
    if (!r.quantity.empty()) out.push_front(wrap(r.quantity), "quantity");
    return out;
}
}  // namespace

// [[Rcpp::export(.ch_diagnostics)]]
List ch_diagnostics(XPtr<FitResult> res) {
    return diagnostics_to_r(network_diagnostics(*res));
}

// [[Rcpp::export(.ch_score_diagnostics)]]
List ch_score_diagnostics(XPtr<FitResult> res, bool relative) {
    return diagnostics_to_r(score_diagnostics(*res, relative));
}

// --- model presets (model_presets.h) ----------------------------------------------

namespace {

// given: a named list of the explicit settings; NULL = infer.
ModelChoice model_choice(std::string groups_model, std::string hierarchy_model,
                         std::string interaction, std::string mixing_variation, int assortative,
                         bool degree_correction, List given) {
    ModelChoice m;
    m.groups_model = groups_model;
    m.hierarchy_model = hierarchy_model;
    m.interaction = interaction;
    m.mixing_variation = mixing_variation;
    m.assortative = assortative;
    m.degree_correction = degree_correction;
    if (given.size() > 0) {
        CharacterVector names = given.names();
        for (R_xlen_t k = 0; k < given.size(); ++k) {
            SEXP x = given[k];
            m.given[as<std::string>(names[k])] =
                Rf_isNull(x) ? std::nullopt : std::optional<double>(as<double>(x));
        }
    }
    return m;
}

SEXP opt(const std::optional<double>& x) {
    return x ? wrap(*x) : R_NilValue;
}

}  // namespace

// [[Rcpp::export(.ch_resolve_model)]]
List ch_resolve_model(std::string groups_model, std::string hierarchy_model,
                      std::string interaction, std::string mixing_variation, int assortative,
                      bool degree_correction, List given, bool has_neutral) {
    const ResolvedModel r =
        resolve_model(model_choice(groups_model, hierarchy_model, interaction, mixing_variation,
                                   assortative, degree_correction, given),
                      has_neutral);
    List out = List::create(
        _["hierarchy_model"] = r.hierarchy_model, _["assortative"] = r.assortative,
        _["variation_in"] = opt(r.variation_in), _["variation_out"] = opt(r.variation_out),
        _["degree_correction"] = r.degree_correction,
        _["individual_depth"] = opt(r.individual_depth), _["group_depth"] = opt(r.group_depth),
        _["ties_parameter"] = opt(r.ties_parameter),
        _["num_groups"] = r.num_groups ? wrap(*r.num_groups) : R_NilValue);
    return out;
}
