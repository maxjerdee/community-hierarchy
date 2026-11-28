#include "diagnostics.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// Inverse standard normal CDF (Acklam's rational approximation, |rel err| < 1.2e-9).
double inv_norm_cdf(double p) {
    static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                               1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                               6.680131188771972e+01, -1.328068155288572e+01};
    static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                               -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
    static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
                               3.754408661907416e+00};
    const double lo = 0.02425, hi = 1.0 - lo;
    if (p < lo) {
        double q = std::sqrt(-2.0 * std::log(p));
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0);
    }
    if (p > hi) {
        double q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0);
    }
    double q = p - 0.5, r = q * q;
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q /
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0);
}

bool too_short(const Chains& x, std::size_t min_chains) {
    if (x.size() < min_chains) return true;
    for (const auto& c : x)
        if (c.size() < 4) return true;
    for (const auto& c : x)
        for (double v : c)
            if (std::isnan(v)) return true;
    return false;
}

// The first and the last half of every chain (the middle draw of an odd chain is
// dropped).
Chains split(const Chains& x) {
    Chains out;
    for (const auto& c : x) {
        const std::size_t h = c.size() / 2;
        out.emplace_back(c.begin(), c.begin() + static_cast<std::ptrdiff_t>(h));
    }
    for (const auto& c : x) {
        const std::size_t h = c.size() / 2;
        out.emplace_back(c.end() - static_cast<std::ptrdiff_t>(h), c.end());
    }
    return out;
}

// Normal scores of the pooled ranks (average rank for ties), (rank - 3/8) / (S + 1/4).
Chains z_scale(Chains x) {
    std::vector<std::pair<double, std::pair<std::size_t, std::size_t>>> all;
    for (std::size_t m = 0; m < x.size(); ++m)
        for (std::size_t t = 0; t < x[m].size(); ++t) all.push_back({x[m][t], {m, t}});
    std::sort(all.begin(), all.end());
    const double S = static_cast<double>(all.size());
    for (std::size_t i = 0; i < all.size();) {
        std::size_t j = i;
        while (j + 1 < all.size() && all[j + 1].first == all[i].first) ++j;
        const double rank = 0.5 * static_cast<double>(i + j) + 1.0;
        const double z = inv_norm_cdf((rank - 0.375) / (S + 0.25));
        for (std::size_t k = i; k <= j; ++k) x[all[k].second.first][all[k].second.second] = z;
        i = j + 1;
    }
    return x;
}

double mean(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}

double var(const std::vector<double>& v) {  // ddof = 1
    const double m = mean(v);
    double s = 0.0;
    for (double x : v) s += (x - m) * (x - m);
    return s / static_cast<double>(v.size() - 1);
}

// Effective sample size of equal-length chains (arviz _ess): Geyer's initial
// positive then initial monotone sequence of the multi-chain autocorrelation.
double ess(const Chains& x) {
    const int M = static_cast<int>(x.size());
    const int N = static_cast<int>(x[0].size());
    double lo = x[0][0], hi = x[0][0];
    for (const auto& c : x)
        for (double v : c) { lo = std::min(lo, v); hi = std::max(hi, v); }
    if (hi - lo < 1e-15) return static_cast<double>(M) * N;   // numpy finfo(float).resolution

    std::vector<double> chain_mean(static_cast<std::size_t>(M));
    for (int m = 0; m < M; ++m) chain_mean[m] = mean(x[m]);
    auto acov = [&](int lag) {  // mean over chains of the biased autocovariance
        double s = 0.0;
        for (int m = 0; m < M; ++m) {
            double a = 0.0;
            for (int t = 0; t + lag < N; ++t) a += (x[m][t] - chain_mean[m]) * (x[m][t + lag] - chain_mean[m]);
            s += a / N;
        }
        return s / M;
    };
    const double mean_var = acov(0) * N / (N - 1.0);
    double var_plus = mean_var * (N - 1.0) / N;
    if (M > 1) var_plus += var(chain_mean);
    auto rho = [&](int lag) { return 1.0 - (mean_var - acov(lag)) / var_plus; };

    std::vector<double> r(static_cast<std::size_t>(N), 0.0);
    double even = 1.0, odd = rho(1);
    r[0] = even;
    r[1] = odd;
    int t = 1;
    while (t < N - 3 && even + odd > 0.0) {
        even = rho(t + 1);
        odd = rho(t + 2);
        if (even + odd >= 0.0) {
            r[t + 1] = even;
            r[t + 2] = odd;
        }
        t += 2;
    }
    const int max_t = t - 2;
    if (even > 0.0) r[max_t + 1] = even;
    for (int k = 1; k <= max_t - 2; k += 2)
        if (r[k + 1] + r[k + 2] > r[k - 1] + r[k]) r[k + 1] = r[k + 2] = 0.5 * (r[k - 1] + r[k]);
    double tau = -1.0;
    for (int k = 0; k <= max_t; ++k) tau += 2.0 * r[k];
    if (max_t + 1 < N) tau += r[max_t + 1];
    const double S = static_cast<double>(M) * N;
    tau = std::max(tau, 1.0 / std::log10(S));
    for (double v : r)
        if (std::isnan(v)) return kNaN;
    return S / tau;
}

// Split R-hat of chains that are already split (arviz _rhat).
double rhat_split(const Chains& x) {
    const double N = static_cast<double>(x[0].size());
    std::vector<double> chain_mean, chain_var;
    for (const auto& c : x) {
        chain_mean.push_back(mean(c));
        chain_var.push_back(var(c));
    }
    const double between = N * var(chain_mean);
    const double within = mean(chain_var);
    return std::sqrt((between / within + N - 1.0) / N);
}

// Linear-interpolation quantile of the pooled draws (numpy's default, type 7).
double quantile(const Chains& x, double p) {
    std::vector<double> all;
    for (const auto& c : x) all.insert(all.end(), c.begin(), c.end());
    std::sort(all.begin(), all.end());
    const double h = p * static_cast<double>(all.size() - 1);
    const std::size_t i = static_cast<std::size_t>(std::floor(h));
    if (i + 1 >= all.size()) return all.back();
    return all[i] + (h - static_cast<double>(i)) * (all[i + 1] - all[i]);
}

double ess_below(const Chains& x, double q) {
    Chains ind = x;
    for (auto& c : ind)
        for (double& v : c) v = v <= q ? 1.0 : 0.0;
    return ess(split(ind));
}

DiagnosticsResult summarise(const std::vector<Chains>& quantities) {
    DiagnosticsResult r;
    for (const auto& c : quantities) {
        r.ess_bulk.push_back(ess_bulk(c));
        r.ess_tail.push_back(ess_tail(c));
        r.rhat.push_back(rhat(c));
    }
    return r;
}

}  // namespace

double ess_bulk(const Chains& x) {
    if (too_short(x, 1)) return kNaN;
    return ess(z_scale(split(x)));
}

double ess_tail(const Chains& x) {
    if (too_short(x, 1)) return kNaN;
    return std::min(ess_below(x, quantile(x, 0.05)), ess_below(x, quantile(x, 0.95)));
}

double rhat(const Chains& x) {
    if (too_short(x, 2)) return kNaN;
    const Chains s = split(x);
    double med = 0.0;
    {
        std::vector<double> all;
        for (const auto& c : s) all.insert(all.end(), c.begin(), c.end());
        std::sort(all.begin(), all.end());
        const std::size_t n = all.size();
        med = n % 2 ? all[n / 2] : 0.5 * (all[n / 2 - 1] + all[n / 2]);
    }
    Chains folded = s;
    for (auto& c : folded)
        for (double& v : c) v = std::abs(v - med);
    return std::max(rhat_split(z_scale(s)), rhat_split(z_scale(folded)));
}

Chains chains_of(const FitResult& result, double (*get)(const SampleState&)) {
    std::map<int, std::vector<double>> by_chain;
    for (const auto& s : result.samples()) by_chain[s.chain_id].push_back(get(s));
    std::size_t len = std::numeric_limits<std::size_t>::max();
    for (const auto& kv : by_chain) len = std::min(len, kv.second.size());
    Chains out;
    for (auto& kv : by_chain) out.emplace_back(kv.second.begin(), kv.second.begin() + static_cast<std::ptrdiff_t>(len));
    return out;
}

DiagnosticsResult network_diagnostics(const FitResult& result) {
    std::vector<Chains> q;
    std::vector<std::string> names;
    for (const auto& p : network_parameters()) {
        names.push_back(p.name);
        q.push_back(chains_of(result, p.get));
    }
    names.push_back("H_A_b");
    q.push_back(chains_of(result, [](const SampleState& s) { return s.H_A_b; }));
    DiagnosticsResult r = summarise(q);
    r.quantity = std::move(names);
    return r;
}

DiagnosticsResult score_diagnostics(const FitResult& result, bool relative) {
    const int n = result.num_nodes();
    std::map<int, std::vector<const SampleState*>> by_chain;
    for (const auto& s : result.samples()) by_chain[s.chain_id].push_back(&s);
    std::size_t len = std::numeric_limits<std::size_t>::max();
    for (const auto& kv : by_chain) len = std::min(len, kv.second.size());
    if (by_chain.empty()) len = 0;
    std::vector<Chains> q(static_cast<std::size_t>(n), Chains(by_chain.size(), std::vector<double>(len)));
    std::size_t m = 0;
    for (const auto& kv : by_chain) {
        for (std::size_t t = 0; t < len; ++t) {
            const auto& scores = kv.second[t]->scores;
            double shift = 0.0;
            if (relative) {
                for (double v : scores) shift += v;
                shift /= n;
            }
            for (int i = 0; i < n; ++i) q[i][m][t] = scores[i] - shift;
        }
        ++m;
    }
    return summarise(q);
}
