#include "fit_result.h"

#include "diagnostics.h"
#include "helpers.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <numeric>
#include <cctype>
#include <stdexcept>

FitResult::FitResult(int num_nodes, std::vector<SampleState> samples)
    : num_nodes_(num_nodes), samples_(std::move(samples)) {}

// ---------------------------------------------------------------------------
// sort_groups_by_score
// ---------------------------------------------------------------------------

std::vector<int> FitResult::sort_groups_by_score(
    const std::vector<int>& partition,
    const std::vector<double>& score_means)
{
    // Accumulate sum and count per group using std::map for deterministic order.
    std::map<int, double> group_sum;
    std::map<int, int>    group_cnt;
    int n = static_cast<int>(partition.size());
    for (int i = 0; i < n; ++i) {
        group_sum[partition[i]] += score_means[i];
        group_cnt[partition[i]]++;
    }
    // Build (mean_score, old_label) pairs and sort ascending by mean.
    std::vector<std::pair<double, int>> ranked;
    ranked.reserve(group_sum.size());
    for (auto& kv : group_sum)
        ranked.push_back({kv.second / group_cnt[kv.first], kv.first});
    std::sort(ranked.begin(), ranked.end());

    // Map old label → new label (0 = lowest mean score).
    std::unordered_map<int, int> remap;
    for (int new_label = 0; new_label < static_cast<int>(ranked.size()); ++new_label)
        remap[ranked[new_label].second] = new_label;

    std::vector<int> out(n);
    for (int i = 0; i < n; ++i) out[i] = remap[partition[i]];
    return out;
}

int FitResult::best_index() const {
    if (samples_.empty()) return -1;
    int best = 0;
    double best_dl = std::numeric_limits<double>::infinity();
    for (int i = 0; i < static_cast<int>(samples_.size()); ++i) {
        const auto& s = samples_[i];
        double dl = s.H_A_b + s.H_Adir_G_A_s_nu + s.H_s_G_b_depths;
        if (dl < best_dl) { best_dl = dl; best = i; }
    }
    return best;
}

double FitResult::mdl_value() const {
    int i = best_index();
    if (i < 0) return std::numeric_limits<double>::quiet_NaN();
    const auto& s = samples_[i];
    return s.H_A_b + s.H_Adir_G_A_s_nu + s.H_s_G_b_depths;
}

std::vector<int> FitResult::mdl_partition() const {
    int i = best_index();
    if (i < 0) return std::vector<int>(num_nodes_, 0);
    return samples_[i].partition;
}

// ---------------------------------------------------------------------------
// Consensus partition
// ---------------------------------------------------------------------------
namespace {

// Hungarian algorithm, square S x S cost matrix (row-major), minimisation.
// Returns row_to_col.  O(S^3).
std::vector<int> hungarian_min(const std::vector<double>& a, int S) {
    const double INF = std::numeric_limits<double>::infinity();
    std::vector<double> u(S + 1, 0.0), v(S + 1, 0.0), minv(S + 1);
    std::vector<int> p(S + 1, 0), way(S + 1, 0);
    std::vector<char> used(S + 1);
    for (int i = 1; i <= S; ++i) {
        p[0] = i;
        int j0 = 0;
        std::fill(minv.begin(), minv.end(), INF);
        std::fill(used.begin(), used.end(), 0);
        do {
            used[j0] = 1;
            const int i0 = p[j0];
            int j1 = 0;
            double delta = INF;
            for (int j = 1; j <= S; ++j) {
                if (used[j]) continue;
                const double cur = a[static_cast<std::size_t>(i0 - 1) * S + (j - 1)] - u[i0] - v[j];
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                if (minv[j] < delta) { delta = minv[j]; j1 = j; }
            }
            for (int j = 0; j <= S; ++j) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else minv[j] -= delta;
            }
            j0 = j1;
        } while (p[j0] != 0);
        do { const int j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
    }
    std::vector<int> row_to_col(S, -1);
    for (int j = 1; j <= S; ++j)
        if (p[j]) row_to_col[p[j] - 1] = j - 1;
    return row_to_col;
}

std::vector<int> compact_labels(const std::vector<int>& p, int& K) {
    std::vector<int> labels(p);
    std::sort(labels.begin(), labels.end());
    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
    K = static_cast<int>(labels.size());
    std::vector<int> out(p.size());
    for (std::size_t i = 0; i < p.size(); ++i)
        out[i] = static_cast<int>(std::lower_bound(labels.begin(), labels.end(), p[i]) - labels.begin());
    return out;
}

}  // namespace

int FitResult::modal_num_groups() const {
    if (samples_.empty()) return 1;
    // std::map, not unordered_map: on a TIE the winner would otherwise depend on
    // bucket order, which differs between standard libraries (see rng.h).  Ties
    // go to the smallest q.
    std::map<int, int> group_hist;
    for (const auto& s : samples_) ++group_hist[s.num_groups];
    int modal_k = samples_.front().num_groups, modal_count = -1;
    for (const auto& kv : group_hist)
        if (kv.second > modal_count) { modal_count = kv.second; modal_k = kv.first; }
    return modal_k;
}

double FitResult::partition_ess() const {
    return ess_bulk(chains_of(*this, [](const SampleState& s) { return s.H_A_b; }));
}

// Thin to max(min_samples, ESS) evenly spaced samples.  The ESS is that of
// H_A_b, the part of the description length that only the partition moves:
// samples closer together than its autocorrelation time add little to a
// partition summary.  min_samples is the floor that keeps the summary from
// resting on a handful of draws when the partition mixes slowly.
std::vector<int> FitResult::consensus_sample_indices(int min_samples) const {
    const int T = static_cast<int>(samples_.size());
    std::vector<int> idx;
    if (T == 0) return idx;
    int m = T;
    if (min_samples > 0 && T > min_samples) {
        const double ess = partition_ess();
        const double target = std::isfinite(ess) ? std::max<double>(min_samples, std::ceil(ess)) : min_samples;
        m = static_cast<int>(std::min<double>(T, target));
    }
    idx.reserve(static_cast<std::size_t>(m));
    const double step = static_cast<double>(T) / m;
    for (int k = 0; k < m; ++k) idx.push_back(static_cast<int>(std::floor(k * step)));
    return idx;
}

std::vector<int> FitResult::consensus_partition(const std::string& method, bool refine,
                                                int min_samples) const {
    if (samples_.empty()) return std::vector<int>(num_nodes_, 0);
    std::string m = method;
    std::transform(m.begin(), m.end(), m.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    const auto idx = consensus_sample_indices(min_samples);
    const int modal_k = modal_num_groups();
    if (m == "moc") return moc_unrestricted(idx);
    if (m == "moc_modal_k") return moc_consensus(idx, modal_k);
    if (m == "binder") return binder_consensus(idx, modal_k, refine);
    throw std::invalid_argument("consensus method must be \"moc\", \"moc_modal_k\" or \"binder\", got \"" + method + "\"");
}

// Dahl's method at the modal number of groups.  Dahl's method is an argmin over
// the recorded samples, so with a broad posterior over q it can land on a sample
// with an atypical q; fixing q to the mode first makes the reported group count
// a stable summary of the run.
std::vector<int> FitResult::binder_consensus(const std::vector<int>& idx, int modal_k, bool refine) const {
    const int n = num_nodes_;
    const double inv = 1.0 / static_cast<double>(idx.size());
    // Aggregate coincidence: C[i*n+j] = fraction of samples co-assigning i and j (i < j)
    std::vector<double> C(static_cast<std::size_t>(n) * n, 0.0);
    for (int si : idx) {
        const auto& p = samples_[static_cast<std::size_t>(si)].partition;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                if (p[i] == p[j]) C[static_cast<std::size_t>(i) * n + j] += inv;
    }
    // L1(s) = sum_{i<j, same in s}(1 - C[i,j]) + sum_{i<j, diff in s} C[i,j]
    //       = const + sum_{i<j, same in s}(1 - 2*C[i,j])
    auto pick = [&](bool from_idx) {
        int best_idx = -1;
        double best_score = std::numeric_limits<double>::infinity();
        const int cnt = from_idx ? static_cast<int>(idx.size()) : static_cast<int>(samples_.size());
        for (int k = 0; k < cnt; ++k) {
            const int si = from_idx ? idx[static_cast<std::size_t>(k)] : k;
            const auto& s = samples_[static_cast<std::size_t>(si)];
            if (s.num_groups != modal_k) continue;
            double score = 0.0;
            for (int i = 0; i < n; ++i)
                for (int j = i + 1; j < n; ++j)
                    if (s.partition[i] == s.partition[j]) score += 1.0 - 2.0 * C[static_cast<std::size_t>(i) * n + j];
            if (score < best_score) { best_score = score; best_idx = si; }
        }
        return best_idx;
    };
    int best_idx = pick(true);
    if (best_idx < 0) best_idx = pick(false);   // thinning skipped every modal-K sample
    if (best_idx < 0) best_idx = 0;             // modal_k comes from a sample, so unreachable
    std::vector<int> best = samples_[static_cast<std::size_t>(best_idx)].partition;
    if (refine) refine_consensus(best, C, n);
    return best;
}

// Maximum-overlap consensus (Peixoto 2021, PRX 11:021003, Sec. IV) with the
// number of groups HELD at the modal K.  Maximises
//     sum_t max_{pi_t} sum_i 1[c_i = pi_t(b_t,i)]
// over every partition c with K labels -- not only the recorded ones -- by
// Peixoto's coordinate ascent: (1) align each sample's labels to c by maximum
// bipartite matching of the contingency table (Hungarian); groups of a sample
// with more than K groups stay unmatched and count as disagreements; (2) give
// every node its majority aligned label among the K.  Step (2) is exact given
// the alignments, so the overlap never decreases; a group emptied by the vote is
// re-seeded with the node that loses least by moving to it, keeping exactly K.
// Restarts from up to three modal-K samples; the best overlap wins.  Cost per
// iteration O(T (n + K^3)), no n x n matrix.  Deterministic: ties go to the
// current label, then the smallest.
std::vector<int> FitResult::moc_consensus(const std::vector<int>& idx, int K) const {
    const int n = num_nodes_;
    const int T = static_cast<int>(idx.size());
    std::vector<std::vector<int>> P(static_cast<std::size_t>(T));
    std::vector<int> Kt(static_cast<std::size_t>(T));
    for (int t = 0; t < T; ++t) P[t] = compact_labels(samples_[static_cast<std::size_t>(idx[t])].partition, Kt[t]);
    if (K <= 1) return std::vector<int>(n, 0);

    std::vector<int> inits;
    for (int t = 0; t < T; ++t) if (Kt[t] == K) inits.push_back(t);
    std::vector<std::vector<int>> starts;
    if (inits.empty()) {                              // thinning skipped every modal-K sample
        for (const auto& s : samples_)
            if (s.num_groups == K) { int k; starts.push_back(compact_labels(s.partition, k)); break; }
        if (starts.empty()) return std::vector<int>(n, 0);
    } else {
        const int R = std::min<int>(3, static_cast<int>(inits.size()));
        for (int r = 0; r < R; ++r)
            starts.push_back(P[static_cast<std::size_t>(inits[static_cast<std::size_t>(r * static_cast<int>(inits.size()) / R)])]);
    }

    std::vector<int> best_c;
    long long best_total = -1;
    std::vector<int> votes(static_cast<std::size_t>(n) * K);
    std::vector<double> cost;
    for (auto c : starts) {
        std::vector<int> prev;
        for (int it = 0; it < 100; ++it) {
            std::fill(votes.begin(), votes.end(), 0);
            long long total = 0;
            for (int t = 0; t < T; ++t) {
                const int S = std::max(K, Kt[t]);
                cost.assign(static_cast<std::size_t>(S) * S, 0.0);
                const auto& b = P[static_cast<std::size_t>(t)];
                for (int i = 0; i < n; ++i) cost[static_cast<std::size_t>(c[i]) * S + b[i]] -= 1.0;
                const auto row_to_col = hungarian_min(cost, S);
                std::vector<int> col_to_row(static_cast<std::size_t>(S), -1);
                for (int k = 0; k < K; ++k) col_to_row[static_cast<std::size_t>(row_to_col[k])] = k;
                for (int i = 0; i < n; ++i) {
                    const int k = col_to_row[static_cast<std::size_t>(b[i])];
                    if (k >= 0) { ++votes[static_cast<std::size_t>(i) * K + k]; if (k == c[i]) ++total; }
                }
            }
            if (total > best_total) { best_total = total; best_c = c; }
            prev = c;
            for (int i = 0; i < n; ++i) {
                const int* v = &votes[static_cast<std::size_t>(i) * K];
                int bk = c[i];
                for (int k = 0; k < K; ++k) if (v[k] > v[bk]) bk = k;
                c[i] = bk;
            }
            std::vector<int> counts(static_cast<std::size_t>(K), 0);
            for (int g : c) ++counts[static_cast<std::size_t>(g)];
            for (int e = 0; e < K; ++e) {
                if (counts[static_cast<std::size_t>(e)] > 0) continue;
                int bi = -1, bl = std::numeric_limits<int>::max();
                for (int i = 0; i < n; ++i) {
                    if (counts[static_cast<std::size_t>(c[i])] <= 1) continue;
                    const int loss = votes[static_cast<std::size_t>(i) * K + c[i]] - votes[static_cast<std::size_t>(i) * K + e];
                    if (loss < bl) { bl = loss; bi = i; }
                }
                if (bi < 0) break;
                --counts[static_cast<std::size_t>(c[bi])]; ++counts[static_cast<std::size_t>(e)]; c[bi] = e;
            }
            if (c == prev) break;
        }
    }
    return best_c;
}

// Maximum-overlap consensus (Peixoto 2021, PRX 11:021003, Sec. IV) with the
// number of groups FREE, as graph-tool's partition_overlap_center.  Maximises
//     sum_t max_{pi_t} sum_i 1[c_i = pi_t(b_t,i)]
// over all partitions c by the same coordinate ascent: (1) align each sample to
// c by maximum bipartite matching of the contingency table (Hungarian); a group
// of the sample left without a partner among c's groups gets a fresh label
// beyond c's, so the vote can open new groups; (2) give every node its majority
// aligned label.  Groups that win no node vanish, so K is set by the samples.
// Starts from the one-group partition (graph-tool's default), from up to three
// evenly spaced samples and from the modal-K consensus; the best overlap wins.  Deterministic: ties go
// to the current label, then the smallest.
std::vector<int> FitResult::moc_unrestricted(const std::vector<int>& idx) const {
    const int n = num_nodes_;
    const int T = static_cast<int>(idx.size());
    if (T == 0) return std::vector<int>(n, 0);
    std::vector<std::vector<int>> P(static_cast<std::size_t>(T));
    std::vector<int> Kt(static_cast<std::size_t>(T));
    int Kmax = 1;
    for (int t = 0; t < T; ++t) {
        P[t] = compact_labels(samples_[static_cast<std::size_t>(idx[t])].partition, Kt[t]);
        Kmax = std::max(Kmax, Kt[t]);
    }
    std::vector<std::vector<int>> starts{std::vector<int>(n, 0)};
    const int R = std::min(3, T);
    for (int r = 0; r < R; ++r) starts.push_back(P[static_cast<std::size_t>(r * T / R)]);
    // Also start from the modal-K consensus: the ascent never lowers the overlap of
    // its start, so the unrestricted consensus is never worse than the K-held one.
    starts.push_back(moc_consensus(idx, modal_num_groups()));

    std::vector<int> best_c;
    long long best_total = -1;
    std::vector<int> votes;
    std::vector<double> cost;
    for (auto c : starts) {
        int Kc;
        c = compact_labels(c, Kc);
        for (int it = 0; it < 100; ++it) {
            const int L = std::max(Kc, Kmax);
            votes.assign(static_cast<std::size_t>(n) * L, 0);
            long long total = 0;
            for (int t = 0; t < T; ++t) {
                const int S = std::max(Kc, Kt[t]);
                cost.assign(static_cast<std::size_t>(S) * S, 0.0);
                const auto& b = P[static_cast<std::size_t>(t)];
                for (int i = 0; i < n; ++i) cost[static_cast<std::size_t>(c[i]) * S + b[i]] -= 1.0;
                // rows: c's groups (< Kc), then fresh labels; columns: the sample's groups
                const auto row_to_col = hungarian_min(cost, S);
                std::vector<int> col_to_row(static_cast<std::size_t>(S), -1);
                for (int k = 0; k < S; ++k) col_to_row[static_cast<std::size_t>(row_to_col[k])] = k;
                for (int i = 0; i < n; ++i) {
                    const int k = col_to_row[static_cast<std::size_t>(b[i])];
                    ++votes[static_cast<std::size_t>(i) * L + k];
                    if (k == c[i]) ++total;
                }
            }
            if (total > best_total) { best_total = total; best_c = c; }
            std::vector<int> next(c);
            for (int i = 0; i < n; ++i) {
                const int* v = &votes[static_cast<std::size_t>(i) * L];
                int bk = c[i];
                for (int k = 0; k < L; ++k) if (v[k] > v[bk]) bk = k;
                next[i] = bk;
            }
            next = compact_labels(next, Kc);
            if (next == c) break;
            c = std::move(next);
        }
    }
    return best_c;
}

// Greedy single-node descent on the same Binder loss with the number of groups
// HELD FIXED: a node moves only to another existing group, and a node alone in
// its group cannot move.  Dahl's method picks the best of the partitions the
// chain happened to record; this closes the remaining gap to the best partition
// at that group count, measured at up to 19% of the loss
// (package-testing/asbm/consensus_refinement).  One descent lands within 0.5% of
// what multi-start greedy or simulated annealing reach, so a heavier optimiser
// buys nothing.  O(n^2) per sweep, the same cost as building C.
// Deterministic: labels compacted in ascending order, nodes in index order, ties
// to the smallest label.
void FitResult::refine_consensus(std::vector<int>& p, const std::vector<double>& C, int n) {
    if (n < 3) return;
    const auto at = [&C, n](int i, int j) {
        return (i < j) ? C[i * n + j] : C[j * n + i];      // C holds the upper triangle
    };
    std::vector<int> labels(p);
    std::sort(labels.begin(), labels.end());
    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
    for (int& g : p)
        g = static_cast<int>(std::lower_bound(labels.begin(), labels.end(), g) - labels.begin());
    const int K = static_cast<int>(labels.size());
    if (K < 2) return;

    std::vector<int> counts(static_cast<std::size_t>(K), 0);
    for (int g : p) ++counts[static_cast<std::size_t>(g)];

    std::vector<double> S(static_cast<std::size_t>(K), 0.0);
    for (int sweep = 0; sweep < 100; ++sweep) {
        bool moved = false;
        for (int i = 0; i < n; ++i) {
            const int a = p[static_cast<std::size_t>(i)];
            if (counts[static_cast<std::size_t>(a)] == 1) continue;
            std::fill(S.begin(), S.end(), 0.0);
            for (int j = 0; j < n; ++j)
                if (j != i) S[static_cast<std::size_t>(p[static_cast<std::size_t>(j)])] += 1.0 - 2.0 * at(i, j);
            int b = a;
            double best_delta = 0.0;
            for (int g = 0; g < K; ++g) {
                if (g == a || counts[static_cast<std::size_t>(g)] == 0) continue;
                const double delta = S[static_cast<std::size_t>(g)] - S[static_cast<std::size_t>(a)];
                if (delta < best_delta - 1e-12) { best_delta = delta; b = g; }
            }
            if (b != a) {
                --counts[static_cast<std::size_t>(a)];
                ++counts[static_cast<std::size_t>(b)];
                p[static_cast<std::size_t>(i)] = b;
                moved = true;
            }
        }
        if (!moved) break;
    }
}

std::vector<double> FitResult::coincidence_matrix() const {
    int n = num_nodes_;
    int ns = static_cast<int>(samples_.size());
    std::vector<double> mat(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
    if (ns == 0) return mat;
    double inv = 1.0 / ns;
    for (const auto& s : samples_) {
        for (int i = 0; i < n; ++i) {
            for (int j = i; j < n; ++j) {
                if (s.partition[i] == s.partition[j]) {
                    const std::size_t N = static_cast<std::size_t>(n);
                    mat[static_cast<std::size_t>(i) * N + static_cast<std::size_t>(j)] += inv;
                    if (i != j) mat[static_cast<std::size_t>(j) * N + static_cast<std::size_t>(i)] += inv;
                }
            }
        }
    }
    return mat;
}

std::vector<double> FitResult::score_means() const {
    int n = num_nodes_;
    int ns = static_cast<int>(samples_.size());
    std::vector<double> means(n, 0.0);
    if (ns == 0) return means;
    for (const auto& s : samples_) {
        for (int i = 0; i < n; ++i) means[i] += s.scores[i];
    }
    for (auto& v : means) v /= ns;
    return means;
}

std::vector<double> FitResult::score_stds() const {
    int n = num_nodes_;
    int ns = static_cast<int>(samples_.size());
    std::vector<double> stds(n, 0.0);
    if (ns <= 1) return stds;
    auto means = score_means();
    std::vector<double> vars(n, 0.0);
    for (const auto& s : samples_) {
        for (int i = 0; i < n; ++i) {
            double d = s.scores[i] - means[i];
            vars[i] += d * d;
        }
    }
    for (int i = 0; i < n; ++i) stds[i] = std::sqrt(vars[i] / (ns - 1));
    return stds;
}

// Posterior-predictive outcome probabilities of one new interaction between
// node_a and node_b, averaged over the samples, under the direction model of
// calculate_H_Adir_G_A_s_nu (Davidson's Bradley-Terry with ties, pi = exp(s)):
//   ties_allowed: {P(a beats b), P(tie), P(b beats a)} with each sample's nu;
//   otherwise:    {P(a beats b), P(b beats a)}, plain Bradley-Terry (nu = 0).
std::vector<double> FitResult::direction_probabilities(int node_a, int node_b, bool ties_allowed) const {
    if (node_a < 0 || node_a >= num_nodes_ || node_b < 0 || node_b >= num_nodes_)
        throw std::out_of_range("direction_probabilities: node index out of range [0, " +
                                std::to_string(num_nodes_) + ")");
    int ns = static_cast<int>(samples_.size());
    if (ns == 0) return ties_allowed ? std::vector<double>{0.0, 0.0, 0.0}
                                     : std::vector<double>{0.0, 0.0};
    double sum_a = 0.0, sum_tie = 0.0, sum_b = 0.0;
    for (const auto& s : samples_) {
        const double sa = s.scores[node_a], sb = s.scores[node_b];
        const double log_nu = ties_allowed ? std::log(s.ties_parameter)
                                           : -std::numeric_limits<double>::infinity();
        const double log_z = davidson_log_z(sa, sb, log_nu);
        sum_a += std::exp(sa - log_z);
        sum_tie += std::exp(davidson_log_tie(sa, sb, log_nu) - log_z);
        sum_b += std::exp(sb - log_z);
    }
    if (ties_allowed) return {sum_a / ns, sum_tie / ns, sum_b / ns};
    return {sum_a / ns, sum_b / ns};
}

const std::vector<NetworkParameter>& network_parameters() {
    static const std::vector<NetworkParameter> p = {
        {"num_groups",        [](const SampleState& s) { return static_cast<double>(s.num_groups); }},
        {"density_in",        [](const SampleState& s) { return s.density_in; }},
        {"density_out",       [](const SampleState& s) { return s.density_out; }},
        {"variation_in",      [](const SampleState& s) { return s.variation_in; }},
        {"variation_out",     [](const SampleState& s) { return s.variation_out; }},
        {"degree_correction", [](const SampleState& s) { return s.degree_correction; }},
        {"individual_depth",  [](const SampleState& s) { return s.individual_depth; }},
        {"group_depth",       [](const SampleState& s) { return s.group_depth; }},
        {"ties_parameter",    [](const SampleState& s) { return s.ties_parameter; }},
        {"sigma_T",           [](const SampleState& s) { return sigma_T(s); }},
        {"ICC",               [](const SampleState& s) { return icc(s); }},
    };
    return p;
}

double sigma_T(const SampleState& s) {
    return std::sqrt(s.group_depth * s.group_depth + s.individual_depth * s.individual_depth);
}

double icc(const SampleState& s) {
    const double t = sigma_T(s);
    return t > 0.0 ? s.group_depth * s.group_depth / (t * t) : 0.0;
}

std::map<std::string, double> FitResult::parameter_means() const {
    int ns = static_cast<int>(samples_.size());
    std::map<std::string, double> means;
    for (const auto& [name, get] : network_parameters()) {
        double sum = 0.0;
        if (ns > 0) {
            for (const auto& s : samples_) sum += get(s);
            sum /= ns;
        }
        means[name] = sum;
    }
    return means;
}

std::map<std::string, double> FitResult::parameter_stds() const {
    int ns = static_cast<int>(samples_.size());
    auto means = parameter_means();
    std::map<std::string, double> stds;
    for (const auto& [name, get] : network_parameters()) {
        double var = 0.0;
        if (ns > 1) {
            double m = means[name];
            for (const auto& s : samples_) {
                double d = get(s) - m;
                var += d * d;
            }
            var /= (ns - 1);
        }
        stds[name] = std::sqrt(var);
    }
    return stds;
}

// ---------------------------------------------------------------------------
// HDI helper
// ---------------------------------------------------------------------------

// Highest Density Interval: shortest interval containing floor(prob * n) samples.
std::pair<double, double> FitResult::hdi_bounds(std::vector<double> arr, double prob) {
    if (arr.empty()) return {0.0, 0.0};
    std::sort(arr.begin(), arr.end());
    int n = static_cast<int>(arr.size());
    int interval_idx = static_cast<int>(std::floor(prob * n));
    if (interval_idx >= n) return {arr.front(), arr.back()};
    int n_intervals = n - interval_idx;
    int best = 0;
    double best_width = arr[interval_idx] - arr[0];
    for (int i = 1; i < n_intervals; ++i) {
        double w = arr[i + interval_idx] - arr[i];
        if (w < best_width) { best_width = w; best = i; }
    }
    return {arr[best], arr[best + interval_idx]};
}

// ---------------------------------------------------------------------------
// network_properties
// ---------------------------------------------------------------------------

NetworkPropertiesResult FitResult::network_properties(double prob) const {
    int ns = static_cast<int>(samples_.size());

    NetworkPropertiesResult result;

    auto add_param = [&](const std::string& name, std::vector<double> vals) {
        if (vals.empty()) return;
        int n = static_cast<int>(vals.size());
        double mean = 0.0;
        for (double v : vals) mean += v;
        mean /= n;
        double var = 0.0;
        for (double v : vals) { double d = v - mean; var += d * d; }
        double std_val = (n > 1) ? std::sqrt(var / (n - 1)) : 0.0;
        auto [lo, hi] = hdi_bounds(std::move(vals), prob);
        result.parameter.push_back(name);
        result.mean.push_back(mean);
        result.std.push_back(std_val);
        result.hdi_lo.push_back(lo);
        result.hdi_hi.push_back(hi);
    };

    for (const auto& [name, get] : network_parameters()) {
        std::vector<double> vals(ns);
        for (int s = 0; s < ns; ++s) vals[s] = get(samples_[s]);
        add_param(name, std::move(vals));
    }

    return result;
}

// ---------------------------------------------------------------------------
// node_properties
// ---------------------------------------------------------------------------

NodePropertiesResult FitResult::node_properties(const std::string& mode,
                                                const std::string& consensus_method) const {
    // Resolve "auto": consensus, except Binder on large networks (its n x n
    // coincidence matrix), which falls back to mdl.  MOC needs no n x n matrix.
    std::string cm = consensus_method;
    std::transform(cm.begin(), cm.end(), cm.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    std::string resolved = mode;
    if (mode == "auto")
        resolved = (cm == "binder" && num_nodes_ > 3000) ? "mdl" : "consensus";

    std::vector<int> partition;
    if (resolved == "consensus")
        partition = consensus_partition(cm);
    else
        partition = mdl_partition();

    auto means = score_means();
    auto stds  = score_stds();

    // Sort group labels so group 0 has the lowest mean score.
    partition = sort_groups_by_score(partition, means);

    // group_score_mean: for each node, mean posterior score of its group.
    std::unordered_map<int, double> group_sum;
    std::unordered_map<int, int>    group_cnt;
    for (int i = 0; i < num_nodes_; ++i) {
        group_sum[partition[i]] += means[i];
        group_cnt[partition[i]]++;
    }
    std::vector<double> gsm(num_nodes_);
    for (int i = 0; i < num_nodes_; ++i) {
        int g = partition[i];
        gsm[i] = group_sum[g] / group_cnt[g];
    }

    NodePropertiesResult result;
    result.mode_used        = resolved;
    result.score_mean       = std::move(means);
    result.score_std        = std::move(stds);
    result.group            = std::move(partition);
    result.group_score_mean = std::move(gsm);
    return result;
}
