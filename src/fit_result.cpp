#include "fit_result.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

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
