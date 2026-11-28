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
#include "interactions.h"

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

        // The fit as the native driver's FitParams, so the starting state and the
        // starting draws are those of fit() (initial_state.h): a seeded session runs
        // the chain of a single-chain native fit.
        std::vector<int> from, to;
        std::vector<bool> draw;
        std::vector<double> weight;
        auto add_edges = [&](const int* u, const int* v, const double* w, int m, bool is_draw) {
            for (int e = 0; e < m; ++e) {
                from.push_back(u[e]);
                to.push_back(v[e]);
                draw.push_back(is_draw);
                weight.push_back(w ? w[e] : 1.0);
            }
        };
        add_edges(wins_from, wins_to, wins_weight, num_wins, false);
        add_edges(ties_from, ties_to, ties_weight, num_ties, true);
        const Interactions edges =
            build_interactions(num_nodes, from, to, draw, weight, /*source_high_rank=*/true);

        FitParams p;
        p.wins = edges.wins;
        p.ties = edges.ties;
        p.weighted_SBM = weighted_sbm != 0;
        if (initial_partition)
            p.initial_partition.assign(initial_partition,
                                       initial_partition + initial_partition_length);
        if (initial_scores)
            p.initial_scores.assign(initial_scores, initial_scores + initial_scores_length);
        p.assortative = assortative != 0;
        // A fixed mixing variation (>= 0): the model presets; negative: inferred.
        if (variation_in_fixed_value >= 0.0) p.variation_in = variation_in_fixed_value;
        if (variation_out_fixed_value >= 0.0) p.variation_out = variation_out_fixed_value;
        p.degree_correction = degree_correction_mode == 0 ? 0.0 : 0.5;
        if (independent_coupling != 0) p.group_depth = 0.0;
        // No neutral edges: plain Bradley-Terry, ties_parameter fixed at 0 -- what
        // the Python and R front ends resolve an omitted hierarchy_model to.
        if (!edges.has_neutral) p.ties_parameter = 0.0;
        if (num_groups_fixed && num_groups > 0) p.num_groups = num_groups;
        p.num_groups_fixed = num_groups_fixed != 0;
        complete_start(p);

        rng.seed(static_cast<std::mt19937::result_type>(seed));
        std::mt19937_64 init_rng(static_cast<std::uint64_t>(seed) * 2654435761ULL);

        const int handle = next_handle++;
        DsSession& sess = sessions()[handle];
        sess.mcmc = std::make_unique<MCMC>(new_chain(p, init_rng));

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
                sess->samples_collected.push_back(sess->mcmc->snapshot(0));
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
