// C ABI over the community_hierarchy MCMC engine, for the WebAssembly build.
//
// A fit is a DsSession advanced in chunks by the caller (ds_session_step), so
// a Web Worker can return to its event loop between chunks and see a "cancel"
// message -- no pthreads, SharedArrayBuffer or ASYNCIFY required.  The JS wrapper
// (js/runner.js) drives this; bindings should not need to call it directly.
//
// Strings returned by these functions stay valid until the next call that
// returns a string for the same handle (or, for handle-less calls, forever).
// Every call catches C++ exceptions and reports them via ds_last_error().
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "MCMC_core.h"
#include "fit_result.h"
#include "globals.h"
#include "initial_state.h"

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define DS_EXPORT extern "C" EMSCRIPTEN_KEEPALIVE
#else
#define DS_EXPORT extern "C"
#endif

namespace {

enum class Phase { BURN, SAMPLE, DONE };

struct DsSession {
    std::unique_ptr<MCMC> mcmc;
    Phase phase;
    int burn_done, burn_total;
    int samples_done, samples_total;
    int sweeps_per_sample;
    bool merge_split_enabled;
    std::vector<SampleState> samples_collected;
    std::unique_ptr<FitResult> result;  // built by ds_session_finish
    std::string text;                   // last string handed out for this handle
};

std::unordered_map<int, DsSession>& sessions() {
    static std::unordered_map<int, DsSession> s;
    return s;
}
int next_handle = 1;
std::string last_error;

DsSession* get(int handle) {
    auto it = sessions().find(handle);
    if (it == sessions().end()) {
        last_error = "invalid session handle " + std::to_string(handle);
        return nullptr;
    }
    return &it->second;
}

// JSON number, with null for NaN / infinities.
std::string num(double x) {
    if (!std::isfinite(x)) return "null";
    std::ostringstream s;
    s.precision(17);
    s << x;
    return s.str();
}

void int_array(std::ostringstream& s, const std::vector<int>& v) {
    s << '[';
    for (std::size_t i = 0; i < v.size(); ++i) s << (i ? "," : "") << v[i];
    s << ']';
}

void double_array(std::ostringstream& s, const std::vector<double>& v) {
    s << '[';
    for (std::size_t i = 0; i < v.size(); ++i) s << (i ? "," : "") << num(v[i]);
    s << ']';
}

SampleState capture_sample(const MCMC& c) {
    SampleState ss;
    ss.chain_id = 0;
    ss.beta = 1.0;
    ss.partition = c.partition;
    ss.scores = c.scores;
    ss.num_groups = c.num_groups;
    ss.density_in = c.density_in;
    ss.density_out = c.density_out;
    ss.variation_in = c.variation_in;
    ss.variation_out = c.variation_out;
    ss.degree_correction = c.degree_correction;
    ss.individual_depth = c.individual_depth;
    ss.group_depth = c.group_depth;
    ss.ties_parameter = c.ties_parameter;
    ss.H_A_b = c.H_A_b;
    ss.H_b = c.H_b;
    ss.H_M = c.H_M;
    ss.H_k = c.H_k;
    ss.H_A_G_k_M_b = c.H_A_G_k_M_b;
    ss.H_A_G_b = c.H_A_G_b;
    ss.H_Adir_G_A_s_nu = c.H_Adir_G_A_s_nu;
    ss.H_s_G_b_depths = c.H_s_G_b_depths;
    ss.two_layer_mode = c.two_layer_mode;
    ss.weighted_SBM = c.weighted_SBM;
    ss.H_W = c.H_W;
    return ss;
}

double description_length(const SampleState& s) {
    return s.H_A_b + s.H_Adir_G_A_s_nu + s.H_s_G_b_depths;
}

double description_length(const MCMC& c) {
    return c.H_A_b + c.H_Adir_G_A_s_nu + c.H_s_G_b_depths;
}

std::string current_json(const MCMC& c) {
    std::ostringstream s;
    s << "{\"partition\":";
    int_array(s, c.partition);
    s << ",\"scores\":";
    double_array(s, c.scores);
    s << ",\"num_groups\":" << c.num_groups
      << ",\"description_length\":" << num(description_length(c)) << '}';
    return s.str();
}

std::string progress_json(const DsSession& sess) {
    std::ostringstream s;
    s << "{\"burn_done\":" << sess.burn_done << ",\"burn_total\":" << sess.burn_total
      << ",\"samples_done\":" << sess.samples_done << ",\"samples_total\":" << sess.samples_total
      << '}';
    return s.str();
}

std::string result_json(const FitResult& r, bool cancelled) {
    std::ostringstream s;
    // Compute score_means once; used both for output and to sort group labels.
    auto smeans = r.score_means();
    s << "{\"num_nodes\":" << r.num_nodes() << ",\"cancelled\":" << (cancelled ? "true" : "false")
      << ",\"num_samples\":" << r.num_samples() << ",\"mdl_value\":" << num(r.mdl_value())
      << ",\"mdl_partition\":";
    int_array(s, FitResult::sort_groups_by_score(r.mdl_partition(), smeans));
    s << ",\"consensus_partition\":";
    int_array(s, FitResult::sort_groups_by_score(r.consensus_partition(), smeans));
    s << ",\"score_means\":";
    double_array(s, smeans);
    s << ",\"score_stds\":";
    double_array(s, r.score_stds());
    s << ",\"samples\":[";
    for (std::size_t i = 0; i < r.samples().size(); ++i) {
        const SampleState& x = r.samples()[i];
        double dl = description_length(x);
        s << (i ? "," : "") << "{\"chain\":" << x.chain_id << ",\"num_groups\":" << x.num_groups
          << ",\"description_length\":" << num(dl)
          << ",\"H_Adir_G_A_s_nu\":" << num(x.H_Adir_G_A_s_nu)
          << ",\"H_s_G_b_depths\":" << num(x.H_s_G_b_depths)
          << ",\"individual_depth\":" << num(x.individual_depth)
          << ",\"group_depth\":" << num(x.group_depth)
          << ",\"variation_in\":" << num(x.variation_in)
          << ",\"variation_out\":" << num(x.variation_out)
          << ",\"degree_correction\":" << num(x.degree_correction)
          << ",\"ties_parameter\":" << num(x.ties_parameter) << '}';
    }
    s << "]}";
    return s.str();
}

}  // namespace

DS_EXPORT const char* ds_last_error() {
    return last_error.c_str();
}

// Returns a handle > 0, or 0 on error (see ds_last_error).
// Node ids are 0-based, in [0, num_nodes); an edge wins_from[e] -> wins_to[e]
// means wins_from[e] beats wins_to[e].  Self-loops are rejected.
// wins_weight / ties_weight may be null (1.0 per edge); weights must be finite
// and positive.  initial_partition / initial_scores may be null; if given they
// must have length num_nodes, and the partition must use every label 0..K-1.
// num_groups <= 0 or not fixed: K comes from the initial partition (the greedy
// modularity start of initial_state.h when none is given).
DS_EXPORT int ds_session_create(int num_nodes, const int* wins_from, const int* wins_to,
                                const double* wins_weight, int num_wins, const int* ties_from,
                                const int* ties_to, const double* ties_weight, int num_ties,
                                int assortative, int num_groups, int num_groups_fixed,
                                int merge_split_enabled, int burn_sweeps, int samples,
                                int sweeps_per_sample, int seed, const int* initial_partition,
                                int initial_partition_length, const double* initial_scores,
                                int initial_scores_length, int degree_correction_mode,
                                int weighted_sbm, int independent_coupling,
                                double variation_in_fixed_value, double variation_out_fixed_value) {
    last_error.clear();
    try {
        if (num_nodes < 1) throw std::invalid_argument("num_nodes must be at least 1");
        auto check_edges = [&](const char* name, const int* from, const int* to,
                               const double* weight, int count) {
            if (count < 0) throw std::invalid_argument(std::string(name) + ": negative edge count");
            if (count > 0 && (!from || !to))
                throw std::invalid_argument(std::string(name) + ": missing edge arrays");
            for (int e = 0; e < count; ++e) {
                if (from[e] < 0 || from[e] >= num_nodes || to[e] < 0 || to[e] >= num_nodes)
                    throw std::invalid_argument(
                        std::string(name) + " edge " + std::to_string(e) + " (" +
                        std::to_string(from[e]) + ", " + std::to_string(to[e]) +
                        "): node ids must be in [0, num_nodes = " + std::to_string(num_nodes) +
                        ")");
                if (from[e] == to[e])
                    throw std::invalid_argument(std::string(name) + " edge " + std::to_string(e) +
                                                ": self-loop at node " + std::to_string(from[e]) +
                                                " (not supported)");
                if (weight && !(std::isfinite(weight[e]) && weight[e] > 0.0))
                    throw std::invalid_argument(std::string(name) + " edge " + std::to_string(e) +
                                                ": weights must be finite and positive");
            }
        };
        check_edges("wins", wins_from, wins_to, wins_weight, num_wins);
        check_edges("ties", ties_from, ties_to, ties_weight, num_ties);
        if (initial_partition && initial_partition_length != num_nodes)
            throw std::invalid_argument("initial_partition must have length num_nodes");
        if (initial_scores && initial_scores_length != num_nodes)
            throw std::invalid_argument("initial_scores must have length num_nodes");
        if (initial_partition) {
            std::vector<char> used(static_cast<std::size_t>(num_nodes), 0);
            int max_label = -1;
            for (int i = 0; i < num_nodes; ++i) {
                const int g = initial_partition[i];
                if (g < 0 || g >= num_nodes)
                    throw std::invalid_argument(
                        "initial_partition labels must be in [0, num_nodes)");
                used[static_cast<std::size_t>(g)] = 1;
                max_label = std::max(max_label, g);
            }
            for (int g = 0; g <= max_label; ++g)
                if (!used[static_cast<std::size_t>(g)])
                    throw std::invalid_argument(
                        "initial_partition must use every label 0..K-1 (label " +
                        std::to_string(g) + " is empty)");
            if (num_groups_fixed && num_groups > 0 && max_label + 1 != num_groups)
                throw std::invalid_argument(
                    "initial_partition has " + std::to_string(max_label + 1) +
                    " groups but num_groups_fixed asks for " + std::to_string(num_groups));
        }
        if (initial_scores)
            for (int i = 0; i < num_nodes; ++i)
                if (!std::isfinite(initial_scores[i]))
                    throw std::invalid_argument("initial_scores must be finite");

        // Build adjacency maps
        std::vector<std::unordered_map<int, double>> wins_adj(static_cast<std::size_t>(num_nodes));
        for (int e = 0; e < num_wins; ++e)
            wins_adj[wins_from[e]][wins_to[e]] += wins_weight ? wins_weight[e] : 1.0;

        std::vector<std::unordered_map<int, double>> ties_adj(static_cast<std::size_t>(num_nodes));
        for (int e = 0; e < num_ties; ++e) {
            double w = ties_weight ? ties_weight[e] : 1.0;
            ties_adj[ties_from[e]][ties_to[e]] += w;
            ties_adj[ties_to[e]][ties_from[e]] += w;
        }

        // Empty group_adj = single-layer mode (community structure from wins+ties).
        // Must be genuinely empty (size 0) — MCMC_core checks !group_adj.empty() to
        // detect two-layer mode, so a vector pre-sized to num_nodes (even with all
        // empty inner maps) would incorrectly activate two-layer mode and make
        // neighbors_com empty, collapsing m=0 and zeroing H_M/H_k/H_A_G_k_M_b.
        std::vector<std::unordered_map<int, double>> group_adj;

        // ---------------------------------------------------------------
        // Initial scores — mirrors community_hierarchy.fit() (core.py):
        //   - no scores given: win fractions (win_fraction_scores, initial_state.h)
        //   - scores given: used as given; if their RMS is > 1e-6 it is the
        //     initial individual_depth (id_init), else id is drawn
        // id_init > 0 means we skip the init_rng draw for individual_depth (as
        // fit.cpp does when p.individual_depth_init.has_value() == true).
        // ---------------------------------------------------------------
        std::vector<double> scores;
        double id_init = -1.0;  // -1 = draw id from init_rng below
        if (initial_scores) {
            scores.assign(initial_scores, initial_scores + initial_scores_length);
            double sum_sq = 0.0;
            for (double s : scores) sum_sq += s * s;
            double rms = std::sqrt(sum_sq / static_cast<double>(scores.size()));
            if (rms > 1e-6) id_init = rms;  // initial individual_depth = score RMS
        } else {
            // The default starting scores of every binding (initial_state.h).
            scores = win_fraction_scores(wins_adj, ties_adj);
        }

        // ---------------------------------------------------------------
        // Initial partition: the default of every binding (initial_state.h),
        // greedy modularity, with num_groups groups when they are fixed.
        // ---------------------------------------------------------------
        std::vector<int> partition;
        if (initial_partition) {
            partition.assign(initial_partition, initial_partition + initial_partition_length);
        } else {
            std::optional<int> k;
            if (num_groups_fixed && num_groups > 0) k = num_groups;
            partition = greedy_modularity_partition(wins_adj, ties_adj, {}, k);
        }
        int ng = *std::max_element(partition.begin(), partition.end()) + 1;
        if (num_groups_fixed && num_groups > 0) ng = num_groups;

        // ---------------------------------------------------------------
        // Seed and draw continuous parameters — exactly as fit.cpp does for
        // a single chain (chain_id = 0):
        //   rng.seed(seed)                           — shared move generator
        //   init_rng = mt19937_64(seed * 2654435761) — parameter initializer
        //   vi, vo ~ next_uniform(init_rng)
        //   dc = 0.5 (fixed -- p.degree_correction.has_value() == true, BUT
        //              C++ evaluates value_or's argument eagerly, so fit.cpp
        //              always consumes one draw for dc even when it's fixed.
        //              We must do the same to keep subsequent draws aligned.)
        //   id ~ next_uniform(init_rng) OR id_init (when score normalization set it,
        //              matching p.individual_depth_init.has_value() == true)
        //   gd, tp ~ next_uniform(init_rng)
        // next_uniform (rng.h), not std::uniform_real_distribution, whose draws
        // differ between libstdc++ and libc++.
        // ---------------------------------------------------------------
        rng.seed(static_cast<std::mt19937::result_type>(seed));
        std::mt19937_64 init_rng(static_cast<std::uint64_t>(seed) * 2654435761ULL);
        double vi = next_uniform(init_rng);
        double vo = next_uniform(init_rng);
        // A fixed mixing variation (>= 0) replaces the drawn start and is not
        // sampled -- the model presets (simple 0.5/0.5, hybrid 0.5/0, planted
        // 0/0, traditional 0.5/0.5), as Python's Config sets them. Negative:
        // inferred (general). The draws above happen regardless, as fit.cpp's
        // value_or makes them, so seeded runs stay in step with Python.
        const bool vi_fixed = variation_in_fixed_value >= 0.0;
        const bool vo_fixed = variation_out_fixed_value >= 0.0;
        if (vi_fixed) vi = variation_in_fixed_value;
        if (vo_fixed) vo = variation_out_fixed_value;
        (void)next_uniform(
            init_rng);  // mirrors fit.cpp: degree_correction.value_or(next_uniform(...)) is eager
        // degree_correction_mode: 0 = fixed at 0 (no DC), 1 = fixed at 0.5 (default)
        const double dc = (degree_correction_mode == 0) ? 0.0 : 0.5;
        double id = (id_init > 0.0) ? id_init : next_uniform(init_rng);
        // independent_coupling: when set, group_depth is fixed at 0.0 (hierarchy
        // decoupled from communities — mirrors Python Config(interaction='independent')).
        //
        // fit.cpp uses a ternary for gd: p.group_depth.has_value() ? *p.group_depth : next_uniform(...)
        // Unlike value_or, the ternary is lazy — the draw is skipped when the value is explicit.
        // So independent mode consumes NO draw for gd, and tp becomes draw5 instead of draw6.
        // Coupled mode (no explicit group_depth) consumes draw5 → gd, draw6 → tp.
        const bool gd_fixed = independent_coupling != 0;
        double gd = gd_fixed ? 0.0 : next_uniform(init_rng);
        double tp = next_uniform(init_rng);  // drawn regardless, as fit.cpp's value_or does
        // No neutral edges: ties are not an outcome the data could record, so the
        // model is plain Bradley-Terry with ties_parameter fixed at 0 -- what the
        // Python front end resolves an omitted hierarchy_model to (core.fit). It
        // used to be fitted here unconditionally.
        double tie_weight = 0.0;
        for (int e = 0; e < num_ties; ++e) tie_weight += ties_weight ? ties_weight[e] : 1.0;
        const bool ties_fixed = !(tie_weight > 0.0);
        if (ties_fixed) tp = 0.0;

        const int handle = next_handle++;
        DsSession& sess = sessions()[handle];
        sess.mcmc = std::make_unique<MCMC>(
            wins_adj, ties_adj, group_adj, partition, /*partition_fixed=*/false, scores,
            /*scores_fixed=*/false, ng, /*num_groups_fixed=*/num_groups_fixed != 0,
            /*assortative=*/assortative != 0, vi, /*variation_in_fixed=*/vi_fixed, vo,
            /*variation_out_fixed=*/vo_fixed, dc, /*degree_correction_fixed=*/true, id,
            /*individual_depth_fixed=*/false, gd, /*group_depth_fixed=*/gd_fixed, tp,
            /*ties_parameter_fixed=*/ties_fixed,
            /*seed=*/-1,  // rng already seeded above
            /*density_in_initial=*/-1.0,
            /*density_out_initial=*/-1.0,
            /*weighted_SBM=*/weighted_sbm != 0);

        sess.mcmc->retune_collective_weight();  // from the initial partition
        sess.mcmc->adapting = burn_sweeps > 0;  // step sizes adapt in burn-in only, as in fit()
        sess.phase = Phase::BURN;
        sess.burn_done = 0;
        sess.burn_total = burn_sweeps;
        sess.samples_done = 0;
        sess.samples_total = samples;
        sess.sweeps_per_sample = sweeps_per_sample >= 0 ? sweeps_per_sample : 1;
        sess.merge_split_enabled = merge_split_enabled != 0;
        return handle;
    } catch (const std::exception& e) {
        last_error = e.what();
    } catch (...) {
        last_error = "unknown C++ exception in ds_session_create";
    }
    return 0;
}

// Runs up to max_sweeps iterations.  Returns 1 when finished, 0 if not, -1 on error.
DS_EXPORT int ds_session_step(int handle, int max_sweeps) {
    DsSession* sess = get(handle);
    if (!sess) return -1;
    try {
        for (int i = 0; i < max_sweeps; ++i) {
            if (sess->phase == Phase::BURN) {
                if (sess->burn_total > 0) {
                    // sweeps_per_sample sweeps per burn step, as fit() does: the
                    // burn-in is burn_sweeps * sweeps_per_sample sweeps in both.
                    sess->mcmc->MCMC_sweep(1.0, sess->sweeps_per_sample, sess->merge_split_enabled);
                    sess->burn_done++;
                    // Same equal-spend retuning fit() does, so this session and a
                    // native fit stay identical for the same seed.  This binding
                    // runs its own loop rather than calling fit(), so anything the
                    // driver does during burn-in has to be mirrored here.
                    sess->mcmc->retune_collective_weight();
                }
                if (sess->burn_done >= sess->burn_total) {
                    sess->mcmc->adapting = false;  // freeze the adapted steps, as fit() does
                    sess->phase = Phase::SAMPLE;
                }
            } else if (sess->phase == Phase::SAMPLE) {
                sess->mcmc->MCMC_sweep(1.0, sess->sweeps_per_sample, sess->merge_split_enabled);
                sess->samples_collected.push_back(capture_sample(*sess->mcmc));
                sess->samples_done++;
                if (sess->samples_done >= sess->samples_total) {
                    sess->phase = Phase::DONE;
                    return 1;
                }
            } else {
                return 1;  // already done
            }
        }
        return sess->phase == Phase::DONE ? 1 : 0;
    } catch (const std::exception& e) {
        last_error = e.what();
    } catch (...) {
        last_error = "unknown C++ exception in ds_session_step";
    }
    return -1;
}

// {"burn_done":..,"burn_total":..,"samples_done":..,"samples_total":..}
DS_EXPORT const char* ds_session_progress(int handle) {
    DsSession* sess = get(handle);
    if (!sess) return nullptr;
    sess->text = progress_json(*sess);
    return sess->text.c_str();
}

// Current MCMC state: {"partition":[...],"scores":[...],"num_groups":N,"description_length":X}
DS_EXPORT const char* ds_session_current(int handle) {
    DsSession* sess = get(handle);
    if (!sess) return nullptr;
    try {
        if (sess->phase == Phase::BURN && sess->burn_done == 0) {
            // Nothing recorded yet: return empty arrays
            sess->text =
                "{\"partition\":[],\"scores\":[],\"num_groups\":0,\"description_length\":null}";
        } else {
            sess->text = current_json(*sess->mcmc);
        }
        return sess->text.c_str();
    } catch (const std::exception& e) {
        last_error = e.what();
    } catch (...) {
        last_error = "unknown C++ exception in ds_session_current";
    }
    return nullptr;
}

// The lowest-description-length sample recorded so far, same JSON shape as
// ds_session_current.  Returns empty arrays if no samples collected yet.
DS_EXPORT const char* ds_session_best(int handle) {
    DsSession* sess = get(handle);
    if (!sess) return nullptr;
    try {
        if (sess->samples_collected.empty()) {
            sess->text =
                "{\"partition\":[],\"scores\":[],\"num_groups\":0,\"description_length\":null}";
        } else {
            int best = 0;
            double best_dl = description_length(sess->samples_collected[0]);
            for (std::size_t i = 1; i < sess->samples_collected.size(); ++i) {
                double dl = description_length(sess->samples_collected[i]);
                if (dl < best_dl) {
                    best_dl = dl;
                    best = static_cast<int>(i);
                }
            }
            const SampleState& s = sess->samples_collected[best];
            std::ostringstream os;
            os << "{\"partition\":";
            int_array(os, s.partition);
            os << ",\"scores\":";
            double_array(os, s.scores);
            os << ",\"num_groups\":" << s.num_groups << ",\"description_length\":" << num(best_dl)
               << '}';
            sess->text = os.str();
        }
        return sess->text.c_str();
    } catch (const std::exception& e) {
        last_error = e.what();
    } catch (...) {
        last_error = "unknown C++ exception in ds_session_best";
    }
    return nullptr;
}

// Freezes the result and returns it as JSON.
// Call ds_session_coincidence after this.
DS_EXPORT const char* ds_session_finish(int handle, int cancelled) {
    DsSession* sess = get(handle);
    if (!sess) return nullptr;
    try {
        int n = sess->mcmc->n;
        sess->result = std::make_unique<FitResult>(n, std::move(sess->samples_collected));
        sess->samples_collected.clear();
        sess->text = result_json(*sess->result, cancelled != 0);
        return sess->text.c_str();
    } catch (const std::exception& e) {
        last_error = e.what();
    } catch (...) {
        last_error = "unknown C++ exception in ds_session_finish";
    }
    return nullptr;
}

// Writes the num_nodes x num_nodes coincidence matrix (row-major) into `out`.
// Requires ds_session_finish first.  Returns 0 on success, -1 on error.
DS_EXPORT int ds_session_coincidence(int handle, double* out) {
    DsSession* sess = get(handle);
    if (!sess) return -1;
    if (!sess->result) {
        last_error = "call ds_session_finish before ds_session_coincidence";
        return -1;
    }
    if (!out) {
        last_error = "ds_session_coincidence: null output buffer";
        return -1;
    }
    try {
        const std::vector<double> m = sess->result->coincidence_matrix();
        std::copy(m.begin(), m.end(), out);
        return 0;
    } catch (const std::exception& e) {
        last_error = e.what();
    } catch (...) {
        last_error = "unknown C++ exception in ds_session_coincidence";
    }
    return -1;
}

DS_EXPORT void ds_session_free(int handle) {
    sessions().erase(handle);
}
