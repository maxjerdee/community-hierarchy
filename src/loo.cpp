#include "loo.h"

#include "helpers.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();
constexpr double kInf = std::numeric_limits<double>::infinity();

double mean_log1p(double b, const std::vector<double>& x) {
    double s = 0.0;
    for (double v : x) s += std::log1p(-b * v);
    return s / static_cast<double>(x.size());
}

// Empirical Bayes estimate of the generalised Pareto shape k and scale sigma for
// sorted positive exceedances x (Zhang and Stephens 2009), as arviz.stats._gpdfit.
void gpd_fit(const std::vector<double>& x, double& k, double& sigma) {
    const double prior_bs = 3.0, prior_k = 10.0;
    const int n = static_cast<int>(x.size());
    const int m = 30 + static_cast<int>(std::sqrt(static_cast<double>(n)));
    const double q = x[static_cast<std::size_t>(static_cast<int>(n / 4.0 + 0.5) - 1)];
    std::vector<double> b(static_cast<std::size_t>(m)), len(static_cast<std::size_t>(m));
    for (int j = 0; j < m; ++j) {
        b[j] = (1.0 - std::sqrt(m / (j + 1 - 0.5))) / (prior_bs * q) + 1.0 / x.back();
        const double kj = mean_log1p(b[j], x);
        len[j] = n * (std::log(-(b[j] / kj)) - kj - 1.0);
    }
    std::vector<double> w(static_cast<std::size_t>(m));
    for (int j = 0; j < m; ++j) {
        double s = 0.0;
        for (int l = 0; l < m; ++l) s += std::exp(len[l] - len[j]);
        w[j] = 1.0 / s;
    }
    double wsum = 0.0, bpost = 0.0;
    for (int j = 0; j < m; ++j)
        if (w[j] >= 10 * kEps) wsum += w[j];
    for (int j = 0; j < m; ++j)
        if (w[j] >= 10 * kEps) bpost += b[j] * (w[j] / wsum);
    const double kpost = mean_log1p(bpost, x);
    sigma = -kpost / bpost;
    k = (n * kpost + prior_k * 0.5) / (n + prior_k);
}

// Generalised Pareto quantile at p in (0, 1).
double gpd_quantile(double p, double k, double sigma) {
    if (!(sigma > 0)) return std::numeric_limits<double>::quiet_NaN();
    const double x = std::abs(k) < kEps ? -std::log1p(-p) : std::expm1(-k * std::log1p(-p)) / k;
    return x * sigma;
}

}  // namespace

PsisResult psis(std::vector<double> x, double r_eff) {
    const int S = static_cast<int>(x.size());
    if (S == 0) return {{}, kInf};
    if (!(r_eff > 0)) throw std::invalid_argument("psis: r_eff must be positive");
    const int M = static_cast<int>(std::ceil(std::min(S / 5.0, 3.0 * std::sqrt(S / r_eff))));
    double max_x = -kInf;
    for (double v : x) max_x = std::max(max_x, v);
    for (double& v : x) v -= max_x;

    std::vector<int> order(static_cast<std::size_t>(S));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int i, int j) { return x[i] < x[j]; });
    // arviz: x[argsort(x)[-M - 1]], the (M + 1)-th largest (Python wraps a negative index)
    int ci = S - M - 1;
    if (ci < 0) ci += S;
    const double cutoff = std::max(x[order[std::max(ci, 0)]], std::log(DBL_MIN));
    const double exp_cutoff = std::exp(cutoff);

    std::vector<int> tail;  // indices in increasing order of x
    for (int i : order)
        if (x[i] > cutoff) tail.push_back(i);
    double k = kInf;
    if (tail.size() > 4) {
        std::vector<double> ex(tail.size());
        for (std::size_t t = 0; t < tail.size(); ++t) ex[t] = std::exp(x[tail[t]]) - exp_cutoff;
        double sigma = 0.0;
        gpd_fit(ex, k, sigma);
        if (std::isfinite(k)) {
            const double L = static_cast<double>(tail.size());
            for (std::size_t t = 0; t < tail.size(); ++t)
                x[tail[t]] = std::log(gpd_quantile((t + 0.5) / L, k, sigma) + exp_cutoff);
            for (double& v : x)
                if (v > 0) v = 0;
        }
    }
    const double lse = log_sum_exp(x);
    for (double& v : x) v -= lse;
    return {std::move(x), k};
}

DirectionLooResult direction_loo(const FitResult& result, const std::vector<int>& a,
                                 const std::vector<int>& b, const std::vector<bool>& draw,
                                 const std::vector<double>& weight, double r_eff) {
    const std::size_t m = a.size();
    if (b.size() != m || draw.size() != m || weight.size() != m)
        throw std::invalid_argument(
            "direction_loo: a, b, draw and weight must have the same length");
    const int n = result.num_nodes();
    for (std::size_t i = 0; i < m; ++i)
        if (a[i] < 0 || a[i] >= n || b[i] < 0 || b[i] >= n || a[i] == b[i])
            throw std::out_of_range("direction_loo: interaction " + std::to_string(i) +
                                    " has a node index outside [0, " + std::to_string(n) +
                                    ") or a self-loop");
    const auto& samples = result.samples();
    const std::size_t S = samples.size();
    if (S == 0) throw std::invalid_argument("direction_loo: the fit has no samples");

    DirectionLooResult out;
    out.elpd.resize(m);
    out.pareto_k.resize(m);
    out.p_win.resize(m);
    std::vector<double> ll(S), neg(S);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t s = 0; s < S; ++s) {
            const double sa = samples[s].scores[a[i]], sb = samples[s].scores[b[i]];
            const double log_nu = std::log(samples[s].ties_parameter);
            const double log_outcome = draw[i] ? davidson_log_tie(sa, sb, log_nu) : sa;
            ll[s] = weight[i] * (log_outcome - davidson_log_z(sa, sb, log_nu));
            neg[s] = -ll[s];
        }
        PsisResult r = psis(neg, r_eff);
        std::vector<double> lw(S);
        double p = 0.0;
        for (std::size_t s = 0; s < S; ++s) {
            lw[s] = r.log_weights[s] + ll[s];
            const double d = samples[s].scores[a[i]] - samples[s].scores[b[i]];
            p += std::exp(r.log_weights[s]) / (1.0 + std::exp(-d));
        }
        out.elpd[i] = log_sum_exp(lw);
        out.pareto_k[i] = r.pareto_k;
        out.p_win[i] = draw[i] ? std::numeric_limits<double>::quiet_NaN() : p;
    }
    return out;
}
