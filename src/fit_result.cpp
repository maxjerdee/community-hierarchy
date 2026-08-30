#include "fit_result.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <numeric>

FitResult::FitResult(int num_nodes, std::vector<SampleState> samples)
    : num_nodes_(num_nodes), samples_(std::move(samples)) {}

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

std::vector<int> FitResult::consensus_partition() const {
    if (samples_.empty()) return std::vector<int>(num_nodes_, 0);
    int n = num_nodes_;
    int ns = static_cast<int>(samples_.size());

    // Aggregate coincidence: C[i*n+j] = fraction of samples co-assigning i and j (i < j)
    std::vector<double> C(n * n, 0.0);
    const double inv = 1.0 / ns;
    for (const auto& s : samples_)
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                if (s.partition[i] == s.partition[j])
                    C[i * n + j] += inv;

    // L1(s) = sum_{i<j, same in s}(1 - C[i,j]) + sum_{i<j, diff in s} C[i,j]
    //       = const + sum_{i<j, same in s}(1 - 2*C[i,j])
    // Return the sample that minimises this score.
    int best_idx = 0;
    double best_score = std::numeric_limits<double>::infinity();
    for (int si = 0; si < ns; ++si) {
        const auto& s = samples_[si];
        double score = 0.0;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                if (s.partition[i] == s.partition[j])
                    score += 1.0 - 2.0 * C[i * n + j];
        if (score < best_score) { best_score = score; best_idx = si; }
    }
    return samples_[best_idx].partition;
}

std::vector<double> FitResult::coincidence_matrix() const {
    int n = num_nodes_;
    int ns = static_cast<int>(samples_.size());
    std::vector<double> mat(n * n, 0.0);
    if (ns == 0) return mat;
    double inv = 1.0 / ns;
    for (const auto& s : samples_) {
        for (int i = 0; i < n; ++i) {
            for (int j = i; j < n; ++j) {
                if (s.partition[i] == s.partition[j]) {
                    mat[i * n + j] += inv;
                    if (i != j) mat[j * n + i] += inv;
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

std::vector<double> FitResult::direction_probabilities(int node_a, int node_b, bool ties_allowed) const {
    int ns = static_cast<int>(samples_.size());
    if (ns == 0) return ties_allowed ? std::vector<double>{0.0, 0.0, 0.0}
                                     : std::vector<double>{0.0, 0.0};
    double sum_a = 0.0, sum_tie = 0.0, sum_b = 0.0;
    for (const auto& s : samples_) {
        double sa = s.scores[node_a], sb = s.scores[node_b];
        // Subtract max to avoid exp overflow (numerically equivalent, cancels in ratios)
        double shift = std::max(sa, sb);
        if (ties_allowed) shift = std::max(shift, (sa + sb) / 2.0);
        double ea  = std::exp(sa - shift);
        double eb  = std::exp(sb - shift);
        double tie = ties_allowed
            ? s.ties_parameter * std::exp((sa + sb) / 2.0 - shift)
            : 0.0;
        double Z = ea + eb + tie;
        sum_a   += ea  / Z;
        sum_tie += tie / Z;
        sum_b   += eb  / Z;
    }
    if (ties_allowed) return {sum_a / ns, sum_tie / ns, sum_b / ns};
    return {sum_a / ns, sum_b / ns};
}

// The 9 network-level scalar parameters, in display order, paired with an
// accessor into SampleState.  num_groups (int) is accumulated as a double.
namespace {
const std::vector<std::pair<std::string, double (*)(const SampleState&)>>&
parameter_accessors() {
    static const std::vector<std::pair<std::string, double (*)(const SampleState&)>> a = {
        {"num_groups",        [](const SampleState& s) { return static_cast<double>(s.num_groups); }},
        {"density_in",        [](const SampleState& s) { return s.density_in; }},
        {"density_out",       [](const SampleState& s) { return s.density_out; }},
        {"variation_in",      [](const SampleState& s) { return s.variation_in; }},
        {"variation_out",     [](const SampleState& s) { return s.variation_out; }},
        {"degree_correction", [](const SampleState& s) { return s.degree_correction; }},
        {"individual_depth",  [](const SampleState& s) { return s.individual_depth; }},
        {"group_depth",       [](const SampleState& s) { return s.group_depth; }},
        {"ties_parameter",    [](const SampleState& s) { return s.ties_parameter; }},
    };
    return a;
}
}  // namespace

std::map<std::string, double> FitResult::parameter_means() const {
    int ns = static_cast<int>(samples_.size());
    std::map<std::string, double> means;
    for (const auto& [name, get] : parameter_accessors()) {
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
    for (const auto& [name, get] : parameter_accessors()) {
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
// Matches the Python _hdi() algorithm in core.py.
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

    // Compute sigma_T and ICC per sample from group_depth and individual_depth.
    std::vector<double> sigma_T_vals(ns), icc_vals(ns);
    for (int s = 0; s < ns; ++s) {
        double gd = samples_[s].group_depth;
        double id = samples_[s].individual_depth;
        double sT = std::sqrt(gd * gd + id * id);
        sigma_T_vals[s] = sT;
        icc_vals[s] = (sT > 0.0) ? (gd * gd) / (sT * sT) : 0.0;
    }

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

    for (const auto& [name, get] : parameter_accessors()) {
        std::vector<double> vals(ns);
        for (int s = 0; s < ns; ++s) vals[s] = get(samples_[s]);
        add_param(name, std::move(vals));
    }
    add_param("sigma_T", sigma_T_vals);
    add_param("ICC",     icc_vals);

    return result;
}

// ---------------------------------------------------------------------------
// node_properties
// ---------------------------------------------------------------------------

NodePropertiesResult FitResult::node_properties(const std::string& mode) const {
    // Resolve "auto": consensus for small networks, mdl for large ones.
    std::string resolved = mode;
    if (mode == "auto")
        resolved = (num_nodes_ <= 3000) ? "consensus" : "mdl";

    std::vector<int> partition;
    if (resolved == "consensus")
        partition = consensus_partition();
    else
        partition = mdl_partition();

    auto means = score_means();
    auto stds  = score_stds();

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
