// C ABI over the directedstructure MCMC engine, for the WebAssembly build.
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
#include <iomanip>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "MCMC_core.h"
#include "fit_result.h"
#include "globals.h"

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define DS_EXPORT extern "C" EMSCRIPTEN_KEEPALIVE
#else
#define DS_EXPORT extern "C"
#endif

// ---------------------------------------------------------------------------
// greedy_modularity_partition — Clauset-Newman-Moore algorithm on the
// undirected projection of wins+ties, matching Python's
// nx.community.greedy_modularity_communities + small-community merging.
// The result gives the same starting communities as Python's _gen_initial_partition
// (when num_groups is None), so MCMC starts in the same basin as ds.fit().
// ---------------------------------------------------------------------------
static std::vector<int> greedy_modularity_partition(
    const std::vector<std::unordered_map<int, double>>& wins_adj,
    const std::vector<std::unordered_map<int, double>>& ties_adj,
    int n)
{
    if (n == 0) return {};

    // Build undirected adjacency (unweighted, as Python treats all edges equally)
    std::vector<std::unordered_set<int>> G(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        for (auto& [j, w] : wins_adj[static_cast<std::size_t>(i)]) {
            G[static_cast<std::size_t>(i)].insert(j);
            G[static_cast<std::size_t>(j)].insert(i);
        }
    }
    for (int i = 0; i < n; ++i) {
        for (auto& [j, w] : ties_adj[static_cast<std::size_t>(i)]) {
            if (j > i) {
                G[static_cast<std::size_t>(i)].insert(j);
                G[static_cast<std::size_t>(j)].insert(i);
            }
        }
    }

    double m = 0.0;
    for (int i = 0; i < n; ++i) m += static_cast<double>(G[static_cast<std::size_t>(i)].size());
    m /= 2.0;
    if (m == 0.0) {
        // No edges: each node is its own community
        std::vector<int> p(static_cast<std::size_t>(n));
        std::iota(p.begin(), p.end(), 0);
        return p;
    }

    // e[ci][cj] = (# edges between community ci and cj) / (2m)
    // a[ci]     = degree[ci] / (2m)
    // Initially each node is its own community labelled by its index.
    std::unordered_map<int, std::unordered_map<int, double>> e;
    std::unordered_map<int, double> a;
    e.reserve(static_cast<std::size_t>(n));
    a.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<double>(G[static_cast<std::size_t>(i)].size()) / (2.0 * m);
        for (int j : G[static_cast<std::size_t>(i)])
            e[i][j] = 1.0 / (2.0 * m);
    }

    std::vector<int> comm(static_cast<std::size_t>(n));
    std::iota(comm.begin(), comm.end(), 0);

    // merge_into: fold community cj into ci.
    auto merge_into = [&](int ci, int cj) {
        for (auto& [k, ekj] : e[cj]) {
            if (k == ci || k == cj) continue;
            e[ci][k] += ekj;
            e[k][ci] += ekj;
            e[k].erase(cj);
        }
        e[ci].erase(cj);
        e.erase(cj);
        a[ci] += a[cj];
        a.erase(cj);
        for (int i = 0; i < n; ++i)
            if (comm[static_cast<std::size_t>(i)] == cj)
                comm[static_cast<std::size_t>(i)] = ci;
    };

    // Greedily merge the pair with maximum ΔQ = 2*(e_ij - a_i * a_j) > 0.
    while (e.size() > 1) {
        double best_dq = 0.0;
        int best_ci = -1, best_cj = -1;
        for (auto& [ci, row] : e) {
            double ai = a.at(ci);
            for (auto& [cj, eij] : row) {
                if (cj <= ci) continue;
                double dq = 2.0 * (eij - ai * a.at(cj));
                if (dq > best_dq) { best_dq = dq; best_ci = ci; best_cj = cj; }
            }
        }
        if (best_ci == -1) break;
        merge_into(best_ci, best_cj);
    }

    // Merge communities smaller than 5% of N into their best-connected neighbour,
    // matching Python's _gen_initial_partition small-community merging.
    const int min_size = std::max(2, n / 20);
    bool changed = true;
    while (changed) {
        changed = false;
        std::unordered_map<int, int> sizes;
        for (int c : comm) sizes[c]++;
        for (auto& [ci, sz] : sizes) {
            if (sz < min_size && e.count(ci)) {
                int best_nbr = -1;
                double best_w = -1.0;
                for (auto& [cj, eij] : e.at(ci)) {
                    if (eij > best_w) { best_w = eij; best_nbr = cj; }
                }
                if (best_nbr != -1) {
                    merge_into(best_nbr, ci);
                    changed = true;
                    break;
                }
            }
        }
    }

    // Compact labels to 0..K-1.
    std::unordered_map<int, int> lbl;
    for (int c : comm) if (!lbl.count(c)) lbl[c] = static_cast<int>(lbl.size());
    std::vector<int> partition(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) partition[static_cast<std::size_t>(i)] = lbl.at(comm[static_cast<std::size_t>(i)]);
    return partition;
}

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
    ss.chain_id          = 0;
    ss.beta              = 1.0;
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
    s << "{\"burn_done\":" << sess.burn_done
      << ",\"burn_total\":" << sess.burn_total
      << ",\"samples_done\":" << sess.samples_done
      << ",\"samples_total\":" << sess.samples_total << '}';
    return s.str();
}

std::string result_json(const FitResult& r, bool cancelled) {
    std::ostringstream s;
    // Compute score_means once; used both for output and to sort group labels.
    auto smeans = r.score_means();
    s << "{\"num_nodes\":" << r.num_nodes()
      << ",\"cancelled\":" << (cancelled ? "true" : "false")
      << ",\"num_samples\":" << r.num_samples()
      << ",\"mdl_value\":" << num(r.mdl_value())
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
        s << (i ? "," : "") << "{\"chain\":" << x.chain_id
          << ",\"num_groups\":" << x.num_groups
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

DS_EXPORT const char* ds_last_error() { return last_error.c_str(); }

// Returns a handle > 0, or 0 on error (see ds_last_error).
// Node ids are 0-based.  wins_weight may be null (defaults to 1.0 per edge).
// ties_weight may be null (defaults to 1.0).
// initial_partition / initial_scores may be null.
// If num_groups <= 0, defaults to min(n, 5) balanced groups.
DS_EXPORT int ds_session_create(
    int num_nodes,
    const int* wins_from, const int* wins_to, const double* wins_weight, int num_wins,
    const int* ties_from, const int* ties_to, const double* ties_weight, int num_ties,
    int assortative,
    int num_groups, int num_groups_fixed,
    int merge_split_enabled,
    int burn_sweeps, int samples, int sweeps_per_sample,
    int seed,
    const int* initial_partition, int initial_partition_length,
    const double* initial_scores, int initial_scores_length,
    int degree_correction_mode,
    int weighted_sbm,
    int independent_coupling)
{
    last_error.clear();
    try {
        // Build adjacency maps
        std::vector<std::unordered_map<int, double>> wins_adj(
            static_cast<std::size_t>(num_nodes));
        for (int e = 0; e < num_wins; ++e)
            wins_adj[wins_from[e]][wins_to[e]] += wins_weight ? wins_weight[e] : 1.0;

        std::vector<std::unordered_map<int, double>> ties_adj(
            static_cast<std::size_t>(num_nodes));
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
        // Initial scores — mirrors Python's core.py initialization logic:
        //   - no scores given: generate win-rate scores (_gen_initial_scores)
        //   - scores given + non-zero RMS: normalize to unit RMS, record id_init
        //   - scores given + zero RMS: use as-is
        // id_init > 0 means we skip the init_rng draw for individual_depth (as
        // fit.cpp does when p.individual_depth_init.has_value() == true).
        // ---------------------------------------------------------------
        std::vector<double> scores;
        double id_init = -1.0;  // -1 = draw id from unif01 below
        if (initial_scores && initial_scores_length == num_nodes) {
            scores.assign(initial_scores, initial_scores + initial_scores_length);
            // Normalize to unit RMS, matching Python's score normalization for
            // user-provided scores (core.py lines 619-628).
            double sum_sq = 0.0;
            for (double s : scores) sum_sq += s * s;
            double rms = std::sqrt(sum_sq / static_cast<double>(scores.size()));
            if (rms > 1e-6) {
                for (double& s : scores) s /= rms;
                id_init = rms;  // initial individual_depth = original score RMS
            }
        } else {
            // Generate win-rate scores from the adjacency, matching Python's
            // _gen_initial_scores: score[i] = (wins - losses) / (wins + losses + ties + eps)
            // Match Python's _gen_initial_scores: count distinct opponents
            // (out_degree / in_degree / degree), not sum of edge weights.
            scores.resize(static_cast<std::size_t>(num_nodes), 0.0);
            std::vector<double> losses_deg(static_cast<std::size_t>(num_nodes), 0.0);
            for (int i = 0; i < num_nodes; ++i)
                for (auto& [j, w] : wins_adj[static_cast<std::size_t>(i)])
                    losses_deg[static_cast<std::size_t>(j)] += 1.0;
            for (int i = 0; i < num_nodes; ++i) {
                double w = static_cast<double>(wins_adj[static_cast<std::size_t>(i)].size());
                double t = static_cast<double>(ties_adj[static_cast<std::size_t>(i)].size());
                scores[static_cast<std::size_t>(i)] =
                    (w - losses_deg[static_cast<std::size_t>(i)]) /
                    (w + losses_deg[static_cast<std::size_t>(i)] + t + 1e-6);
            }
        }

        // ---------------------------------------------------------------
        // Initial partition — if not provided, run greedy modularity on the
        // undirected projection of wins+ties, matching Python's
        // _gen_initial_partition (Clauset-Newman-Moore + small-community merging).
        // This places MCMC in the same basin as Python's ds.fit(), giving
        // comparable community counts and description-length trajectories.
        // ---------------------------------------------------------------
        int ng = (num_groups > 0) ? num_groups : std::min(num_nodes, 5);
        std::vector<int> partition;
        if (initial_partition && initial_partition_length == num_nodes) {
            partition.assign(initial_partition, initial_partition + initial_partition_length);
            int max_g = *std::max_element(partition.begin(), partition.end());
            if (!num_groups_fixed) ng = max_g + 1;
        } else {
            partition = greedy_modularity_partition(wins_adj, ties_adj, num_nodes);
            int K_gm = partition.empty() ? 1 :
                       *std::max_element(partition.begin(), partition.end()) + 1;

            if (K_gm > 1 && !num_groups_fixed) {
                // Greedy found real community structure — use its K.
                ng = K_gm;
            } else if (K_gm > ng) {
                // num_groups_fixed: map greedy labels into [0, ng) via modulo.
                for (int& g : partition) g = g % ng;
            } else if (K_gm < ng) {
                // Greedy collapsed (e.g. complete graph) or K_gm < requested ng.
                // Fall back to score-sorted quantile with ng groups so the
                // partition has the right number of labels for MCMC.
                std::vector<int> order(static_cast<std::size_t>(num_nodes));
                std::iota(order.begin(), order.end(), 0);
                // Stable so equal scores preserve node-index order,
                // giving a predictable sequential split when scores are flat.
                std::stable_sort(order.begin(), order.end(),
                                 [&](int a, int b) { return scores[a] < scores[b]; });
                for (int r = 0; r < num_nodes; ++r)
                    partition[static_cast<std::size_t>(order[r])] = r * ng / num_nodes;
            }
            // K_gm == ng && num_groups_fixed: partition is already correct.
        }

        // ---------------------------------------------------------------
        // Seed and draw continuous parameters — exactly as fit.cpp does for
        // a single chain (chain_id = 0):
        //   rng.seed(seed)                           — shared move generator
        //   init_rng = mt19937_64(seed * 2654435761) — parameter initializer
        //   vi, vo ~ unif01(init_rng)
        //   dc = 0.5 (fixed — p.degree_correction.has_value() == true, BUT
        //              C++ evaluates value_or's argument eagerly, so fit.cpp
        //              always consumes one draw for dc even when it's fixed.
        //              We must do the same to keep subsequent draws aligned.)
        //   id ~ unif01(init_rng) OR id_init (when score normalization set it,
        //              matching p.individual_depth_init.has_value() == true)
        //   gd, tp ~ unif01(init_rng)
        //
        // IMPORTANT: do NOT use std::uniform_real_distribution<double> with
        // mt19937_64 here.  mt19937_64::max() - mt19937_64::min() + 1 == 2^64,
        // which overflows uint64_t to 0.  libstdc++ (Python) handles this by
        // doing one generator draw; libc++ (Emscripten / WASM) falls into a
        // different branch and consumes two draws per call.  The mismatch shifts
        // every subsequent draw and makes Python and WASM start from different
        // parameter states even with the same seed.
        //
        // Fix: extract the double directly.  For a 64-bit generator the
        // canonical libstdc++ formula is: result = raw_value * (1 / 2^64).
        // This is exactly one draw and produces the same value as Python.
        // ---------------------------------------------------------------
        rng.seed(static_cast<std::mt19937::result_type>(seed));
        std::mt19937_64 init_rng(static_cast<std::uint64_t>(seed) * 2654435761ULL);
        // Matches libstdc++ generate_canonical with 1 draw from mt19937_64.
        auto unif01 = [](std::mt19937_64& g) -> double {
            return static_cast<double>(g()) * (1.0 / 18446744073709551616.0);
        };
        double vi = unif01(init_rng);
        double vo = unif01(init_rng);
        (void)unif01(init_rng);  // mirrors fit.cpp: degree_correction.value_or(unif01) is eager
        // degree_correction_mode: 0 = fixed at 0 (no DC), 1 = fixed at 0.5 (default)
        const double dc = (degree_correction_mode == 0) ? 0.0 : 0.5;
        double id = (id_init > 0.0) ? id_init : unif01(init_rng);
        // independent_coupling: when set, group_depth is fixed at 0.0 (hierarchy
        // decoupled from communities — mirrors Python Config(interaction='independent')).
        //
        // fit.cpp uses a ternary for gd: p.group_depth.has_value() ? *p.group_depth : next_uniform(...)
        // Unlike value_or, the ternary is lazy — the draw is skipped when the value is explicit.
        // So independent mode consumes NO draw for gd, and tp becomes draw5 instead of draw6.
        // Coupled mode (no explicit group_depth) consumes draw5 → gd, draw6 → tp.
        const bool gd_fixed = independent_coupling != 0;
        double gd = gd_fixed ? 0.0 : unif01(init_rng);
        double tp = unif01(init_rng);   // drawn regardless, as fit.cpp's value_or does
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
            wins_adj, ties_adj, group_adj,
            partition, /*partition_fixed=*/false,
            scores,    /*scores_fixed=*/false,
            ng,        /*num_groups_fixed=*/num_groups_fixed != 0,
            /*assortative=*/assortative != 0,
            vi, /*variation_in_fixed=*/false,
            vo, /*variation_out_fixed=*/false,
            dc, /*degree_correction_fixed=*/true,
            id, /*individual_depth_fixed=*/false,
            gd_fixed ? 0.0 : gd, /*group_depth_fixed=*/gd_fixed,
            tp, /*ties_parameter_fixed=*/ties_fixed,
            /*seed=*/-1,          // rng already seeded above
            /*density_in_initial=*/-1.0,
            /*density_out_initial=*/-1.0,
            /*weighted_SBM=*/weighted_sbm != 0);

        sess.mcmc->retune_collective_weight();   // from the initial partition
        sess.phase              = Phase::BURN;
        sess.burn_done          = 0;
        sess.burn_total         = burn_sweeps;
        sess.samples_done       = 0;
        sess.samples_total      = samples;
        sess.sweeps_per_sample  = sweeps_per_sample >= 0 ? sweeps_per_sample : 1;
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
                    // One sweep per burn step so burn_sweeps == total burn sweeps,
                    // matching Python where burn_sweeps and sweeps_per_sample are independent.
                    sess->mcmc->MCMC_sweep(1.0, 1, sess->merge_split_enabled);
                    sess->burn_done++;
                    // Same equal-spend retuning fit() does, so this session and a
                    // native fit stay identical for the same seed.  This binding
                    // runs its own loop rather than calling fit(), so anything the
                    // driver does during burn-in has to be mirrored here.
                    sess->mcmc->retune_collective_weight();
                }
                if (sess->burn_done >= sess->burn_total)
                    sess->phase = Phase::SAMPLE;
            } else if (sess->phase == Phase::SAMPLE) {
                sess->mcmc->MCMC_sweep(1.0, sess->sweeps_per_sample,
                                       sess->merge_split_enabled);
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
            sess->text = "{\"partition\":[],\"scores\":[],\"num_groups\":0,\"description_length\":null}";
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
            sess->text = "{\"partition\":[],\"scores\":[],\"num_groups\":0,\"description_length\":null}";
        } else {
            int best = 0;
            double best_dl = description_length(sess->samples_collected[0]);
            for (std::size_t i = 1; i < sess->samples_collected.size(); ++i) {
                double dl = description_length(sess->samples_collected[i]);
                if (dl < best_dl) { best_dl = dl; best = static_cast<int>(i); }
            }
            const SampleState& s = sess->samples_collected[best];
            std::ostringstream os;
            os << "{\"partition\":";
            int_array(os, s.partition);
            os << ",\"scores\":";
            double_array(os, s.scores);
            os << ",\"num_groups\":" << s.num_groups
               << ",\"description_length\":" << num(best_dl) << '}';
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
// Call ds_session_scores and ds_session_coincidence after this.
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

// {"score_means":[...],"score_stds":[...]} — requires ds_session_finish first.
DS_EXPORT const char* ds_session_scores(int handle) {
    DsSession* sess = get(handle);
    if (!sess) return nullptr;
    if (!sess->result) {
        last_error = "call ds_session_finish before ds_session_scores";
        return nullptr;
    }
    try {
        std::ostringstream s;
        s << "{\"score_means\":";
        double_array(s, sess->result->score_means());
        s << ",\"score_stds\":";
        double_array(s, sess->result->score_stds());
        s << '}';
        sess->text = s.str();
        return sess->text.c_str();
    } catch (const std::exception& e) {
        last_error = e.what();
    } catch (...) {
        last_error = "unknown C++ exception in ds_session_scores";
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
    const std::vector<double> m = sess->result->coincidence_matrix();
    std::copy(m.begin(), m.end(), out);
    return 0;
}

DS_EXPORT void ds_session_free(int handle) { sessions().erase(handle); }

// Debug: return the first n values from mt19937_64 seeded with seed*2654435761.
// Used to verify that init_rng produces the same sequence as Python's libstdc++.
DS_EXPORT const char* ds_debug_rng64(int seed_int, int count) {
    try {
        std::mt19937_64 g(static_cast<std::uint64_t>(seed_int) * 2654435761ULL);
        std::ostringstream os;
        os << '[';
        for (int i = 0; i < count; ++i) {
            if (i) os << ',';
            // Same formula as the unif01 lambda in ds_session_create
            os << std::setprecision(17) << static_cast<double>(g()) * (1.0 / 18446744073709551616.0);
        }
        os << ']';
        static std::string buf;
        buf = os.str();
        return buf.c_str();
    } catch (...) { return "[]"; }
}
