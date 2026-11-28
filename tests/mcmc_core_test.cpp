// Core model checks for community_hierarchy (plain main(), exits non-zero on failure).
//
//  1. INPUT VALIDATION -- self-loops (wins and ties), non-finite / non-positive
//     weights in weighted_SBM and two-layer mode, asymmetric ties, and a fixed
//     num_groups that disagrees with the partition all throw std::invalid_argument.
//  2. DESCRIPTION LENGTH, EXACT -- H_A_G_k_M_b = -log P(A | k, e, b) of the
//     microcanonical configuration model (Peixoto 2017), checked against a brute-
//     force enumeration of every stub matching of a tiny multigraph.  Shares no
//     code with calculate_H_A_G_k_M_b.  Two-layer mode ignores the group-graph
//     weights (binary counts).
//  3. SCORES FIXED -- with scores_fixed no move (sigma_I / sigma_G kernels,
//     change_parameter) changes a score, and the depths still sample their exact
//     conditional posterior p(sigma_I, sigma_G | s, b), computed by quadrature.
//  4. GAUSSIAN QUANTILE / SCORE PROPOSAL -- the quantile inverts the CDF to
//     ~1e-15 relative over (1e-300, 1); the CDF-space score proposal, accepted
//     every time, leaves N(mu, sigma^2) invariant (its Hastings cancellation),
//     including from a start far in the upper tail.
//  5. PLAIN BRADLEY-TERRY -- with ties_parameter fixed at 0, neutral edges count in
//     the SBM exactly as with draws modelled, and are absent from H_Adir.
//  6. MISCELLANEOUS -- value_check's tolerance on negative values, the n*n
//     density initialisation beyond n = 46341, change_parameter("density_out")
//     in non-assortative mode.

#include "MCMC_core.h"
#include "globals.h"
#include "helpers.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using Adj = std::vector<std::unordered_map<int, double>>;
int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("  %-74s %s\n", what.c_str(), ok ? "PASS" : "FAIL");
    if (!ok) ++failures;
}

struct Opts {
    bool partition_fixed = true, scores_fixed = true, num_groups_fixed = false, assortative = true;
    double vi = 0.5, vo = 0.5, dc = 0.5, id = 1.0, gd = 1.0, tp = 0.5;
    bool vi_fixed = true, vo_fixed = true, dc_fixed = true, id_fixed = true, gd_fixed = true,
         tp_fixed = true;
    int num_groups = -1;
    double din = 0.3, dout = 0.3;
    bool weighted = false;
};

MCMC build(const Adj& W, const Adj& T, const Adj& G, const std::vector<int>& part,
           const std::vector<double>& sc, const Opts& o, int seed = 1) {
    const int q = o.num_groups > 0 ? o.num_groups : *std::max_element(part.begin(), part.end()) + 1;
    return MCMC(W, T, G, part, o.partition_fixed, sc, o.scores_fixed, q, o.num_groups_fixed,
                o.assortative, o.vi, o.vi_fixed, o.vo, o.vo_fixed, o.dc, o.dc_fixed, o.id,
                o.id_fixed, o.gd, o.gd_fixed, o.tp, o.tp_fixed, seed, o.din, o.dout, o.weighted);
}

// Expect std::invalid_argument whose message contains `needle`.
void expect_invalid(const std::function<void()>& f, const std::string& needle,
                    const std::string& what) {
    bool ok = false;
    std::string msg = "(no exception)";
    try {
        f();
    } catch (const std::invalid_argument& e) {
        msg = e.what();
        ok = msg.find(needle) != std::string::npos;
    } catch (const std::exception& e) {
        msg = std::string("wrong type: ") + e.what();
    }
    check(ok, what);
    if (!ok) std::printf("      got: %s\n", msg.c_str());
}

// ---------------------------------------------------------------------------
// 1. input validation
// ---------------------------------------------------------------------------
void test_validation() {
    std::printf("1. input validation\n");
    const int N = 4;
    const std::vector<int> part{0, 0, 1, 1};
    const std::vector<double> sc(N, 0.0);
    auto ring = [&]() {
        Adj W(N);
        for (int i = 0; i < N; ++i) W[i][(i + 1) % N] = 1.0;
        return W;
    };
    const Opts o;

    {
        Adj W = ring(), T(N), G;
        W[2][2] = 1.0;
        expect_invalid([&] { build(W, T, G, part, sc, o); }, "self-loop", "win self-loop rejected");
    }
    {
        Adj W = ring(), T(N), G;
        T[1][1] = 1.0;  // used to throw log_factorial2 deep inside
        expect_invalid([&] { build(W, T, G, part, sc, o); }, "self-loop", "tie self-loop rejected");
    }
    {
        Adj W = ring(), T(N), G;
        T[1][1] = 2.0;  // used to be silently counted as half
        expect_invalid([&] { build(W, T, G, part, sc, o); }, "self-loop",
                       "even-weight tie self-loop rejected");
    }
    {
        Adj W = ring(), T(N), G;
        T[0][2] = 1.0;  // one direction only
        expect_invalid([&] { build(W, T, G, part, sc, o); }, "both directions",
                       "asymmetric ties rejected");
    }
    {
        Adj W = ring(), T(N), G;
        W[0][7] = 1.0;
        expect_invalid([&] { build(W, T, G, part, sc, o); }, "out of range",
                       "node id out of range rejected");
    }
    Opts w = o;
    w.weighted = true;
    for (double bad : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::infinity()}) {
        Adj W = ring(), T(N), G;
        W[0][1] = bad;
        const char* needle = std::isfinite(bad) ? "> 0" : "finite";
        expect_invalid([&] { build(W, T, G, part, sc, w); }, needle,
                       "weighted_SBM weight " + std::to_string(bad) + " rejected");
    }
    {
        Adj W = ring(), T(N), G;
        T[0][2] = 0.0;
        T[2][0] = 0.0;
        expect_invalid([&] { build(W, T, G, part, sc, w); }, "> 0",
                       "weighted_SBM zero-weight tie rejected");
    }
    {
        Adj W = ring(), T(N), G(N);
        for (int i = 0; i < N; ++i) {
            G[i][(i + 1) % N] = 1.0;
            G[(i + 1) % N][i] = 1.0;
        }
        G[0][1] = 0.0;
        expect_invalid([&] { build(W, T, G, part, sc, o); }, "> 0",
                       "two-layer zero group-graph weight rejected");
    }
    {
        Adj W = ring(), T(N), G;
        W[0][1] = 1.5;
        expect_invalid([&] { build(W, T, G, part, sc, o); }, "integers",
                       "non-integer multiplicity rejected");
    }
    {
        Adj W = ring(), T(N), G;
        Opts f = o;
        f.num_groups = 3;
        f.num_groups_fixed = true;
        expect_invalid([&] { build(W, T, G, part, sc, f); }, "num_groups",
                       "fixed num_groups != partition's q rejected");
    }
    {
        Adj W = ring(), T(N), G;
        Opts f = o;
        f.num_groups = 3;
        f.num_groups_fixed = false;
        bool ok = true;
        try {
            MCMC m = build(W, T, G, part, sc, f);
            ok = m.num_groups == 2;
        } catch (...) {
            ok = false;
        }
        check(ok, "free num_groups: q taken from the partition");
    }
}

// ---------------------------------------------------------------------------
// 2. H_A_G_k_M_b against an exact enumeration of stub matchings
// ---------------------------------------------------------------------------
// -log P(A | k, e, b) = -log (fraction of the perfect matchings of the labelled
// stubs that respect the block edge counts e and give the multigraph A).
// A is the upper triangle of edge multiplicities, A[i][i] = number of self-loops.
double exact_H_A(const std::vector<std::vector<int>>& A, const std::vector<int>& b) {
    const int N = static_cast<int>(A.size());
    std::vector<int> stub;  // stub -> node
    for (int i = 0; i < N; ++i) {
        int k = 2 * A[i][i];
        for (int j = 0; j < N; ++j)
            if (j != i) k += (i < j) ? A[i][j] : A[j][i];
        for (int s = 0; s < k; ++s) stub.push_back(i);
    }
    const int q = *std::max_element(b.begin(), b.end()) + 1;
    auto block_counts = [&](const std::vector<std::vector<int>>& M) {
        std::vector<std::vector<int>> e(q, std::vector<int>(q, 0));
        for (int i = 0; i < N; ++i)
            for (int j = i; j < N; ++j)
                if (M[i][j]) {
                    e[b[i]][b[j]] += M[i][j];
                    if (b[i] != b[j]) e[b[j]][b[i]] += M[i][j];
                }
        return e;
    };
    const auto e_target = block_counts(A);
    const int S = static_cast<int>(stub.size());
    std::vector<char> used(S, 0);
    std::vector<std::vector<int>> cur(N, std::vector<int>(N, 0));
    double n_e = 0.0, n_A = 0.0;
    std::function<void()> rec = [&]() {
        int first = -1;
        for (int s = 0; s < S; ++s)
            if (!used[s]) {
                first = s;
                break;
            }
        if (first < 0) {
            if (block_counts(cur) != e_target) return;
            n_e += 1.0;
            if (cur == A) n_A += 1.0;
            return;
        }
        used[first] = 1;
        for (int s = first + 1; s < S; ++s) {
            if (used[s]) continue;
            used[s] = 1;
            const int i = std::min(stub[first], stub[s]), j = std::max(stub[first], stub[s]);
            ++cur[i][j];
            rec();
            --cur[i][j];
            used[s] = 0;
        }
        used[first] = 0;
    };
    rec();
    return -std::log(n_A / n_e);
}

void test_description_length() {
    std::printf("\n2. H_A_G_k_M_b = -log P(A|k,e,b), exact enumeration of stub matchings\n");
    // multigraph on 4 nodes, 6 edges: A01 = 2, A12 = 1, A23 = 2, A03 = 1
    const int N = 4;
    std::vector<std::vector<int>> A(N, std::vector<int>(N, 0));
    A[0][1] = 2;
    A[1][2] = 1;
    A[2][3] = 2;
    A[0][3] = 1;
    // given as wins (multiplicities) plus one tie, which counts in the SBM like any edge
    Adj W(N), T(N), G;
    W[0][1] = 2.0;
    W[1][2] = 1.0;
    W[3][2] = 2.0;
    T[0][3] = 1.0;
    T[3][0] = 1.0;
    const std::vector<double> sc(N, 0.0);
    for (const auto& part : {std::vector<int>{0, 0, 0, 0}, std::vector<int>{0, 0, 1, 1},
                             std::vector<int>{0, 1, 0, 1}, std::vector<int>{0, 1, 1, 2}}) {
        MCMC m = build(W, T, G, part, sc, Opts{});
        const double ex = exact_H_A(A, part);
        const double got = m.calculate_H_A_G_k_M_b();
        char buf[160];
        std::snprintf(buf, sizeof buf, "b = {%d,%d,%d,%d}: code %.12f, exact %.12f", part[0],
                      part[1], part[2], part[3], got, ex);
        check(std::fabs(got - ex) < 1e-10 && std::fabs(m.H_A_G_k_M_b - ex) < 1e-10, buf);
    }
    {  // 2 nodes, A01 = 2: probability 2/3 (2 of the 3 matchings of 4 stubs)
        Adj W2(2), T2(2), G2;
        W2[0][1] = 2.0;
        MCMC m = build(W2, T2, G2, {0, 0}, {0.0, 0.0}, Opts{});
        check(std::fabs(m.calculate_H_A_G_k_M_b() - std::log(1.5)) < 1e-12,
              "2 nodes, A01 = 2: H = log(3/2)");
    }
    {  // two-layer: the group-graph weights do not enter (binary counts)
        Adj W2(N), T2(N);
        for (int i = 0; i < N; ++i) W2[i][(i + 1) % N] = 1.0;
        auto ga = [&](double wgt) {
            Adj G2(N);
            for (int i = 0; i < N; ++i) {
                G2[i][(i + 1) % N] = wgt;
                G2[(i + 1) % N][i] = wgt;
            }
            return G2;
        };
        MCMC m1 = build(W2, T2, ga(1.0), {0, 0, 1, 1}, sc, Opts{});
        MCMC m3 = build(W2, T2, ga(3.0), {0, 0, 1, 1}, sc, Opts{});
        check(m1.two_layer_mode && std::fabs(m1.H_A_G_k_M_b - m3.H_A_G_k_M_b) < 1e-12,
              "two-layer: H_A_G_k_M_b independent of the group-graph weights");
    }
}

// ---------------------------------------------------------------------------
// 3. scores_fixed
// ---------------------------------------------------------------------------
void test_scores_fixed() {
    std::printf("\n3. scores_fixed: scores never move; depths sample p(sigma_I, sigma_G | s, b)\n");
    const int N = 8;
    Adj W(N), T(N), G;
    for (int i = 0; i < N; ++i) W[i][(i + 1) % N] = 1.0;
    W[0][4] = 2.0;
    W[2][6] = 1.0;
    T[1][5] = 1.0;
    T[5][1] = 1.0;
    const std::vector<int> part{0, 0, 0, 0, 1, 1, 1, 1};
    const std::vector<double> sc{1.0, 0.5, 0.0, -0.5, 0.3, 0.2, -0.1, -1.0};

    // (a) free partition and depths, every depth kernel enabled
    {
        Opts o;
        o.partition_fixed = false;
        o.id_fixed = false;
        o.gd_fixed = false;
        o.tp_fixed = false;
        MCMC m = build(W, T, G, part, sc, o, 3);
        m.adapting = true;
        for (int s = 0; s < 300; ++s) m.MCMC_sweep(1.0, 1, true);
        for (int s = 0; s < 20; ++s) {  // change_parameter called directly, too
            m.change_parameter("individual_depth", m.individual_depth * 1.3);
            m.change_parameter("group_depth", m.group_depth * 0.8);
        }
        double maxd = 0.0;
        for (int i = 0; i < N; ++i) maxd = std::max(maxd, std::fabs(m.scores[i] - sc[i]));
        char buf[160];
        std::snprintf(buf, sizeof buf, "300 sweeps + change_parameter: max |score change| = %g",
                      maxd);
        check(maxd == 0.0 && m.accepted_individual_depth_proposals > 0 &&
                  m.accepted_group_depth_proposals > 0,
              buf);
    }
    // (b) exact conditional posterior of the depths.  In the CDF coordinates
    // u = F(sigma_I), v = F(sigma_G) of the half-Cauchy priors the prior is
    // uniform, so the target density is exp(-H_s(sigma_I(u), sigma_G(v))).
    {
        Opts o;
        o.id_fixed = false;
        o.gd_fixed = false;
        MCMC grid = build(W, T, G, part, sc, o, 4);
        const int K = 400;
        double Z = 0.0, Eu = 0.0, Ev = 0.0, Hmin = std::numeric_limits<double>::infinity();
        std::vector<double> H(static_cast<std::size_t>(K) * K);
        for (int a = 0; a < K; ++a)
            for (int c = 0; c < K; ++c) {
                grid.individual_depth = half_cauchy_inv_cdf((a + 0.5) / K, DEPTH_PRIOR_WIDTH);
                grid.group_depth = half_cauchy_inv_cdf((c + 0.5) / K, DEPTH_PRIOR_WIDTH);
                H[a * K + c] = grid.calculate_H_s_G_b_depths();
                Hmin = std::min(Hmin, H[a * K + c]);
            }
        for (int a = 0; a < K; ++a)
            for (int c = 0; c < K; ++c) {
                const double wgt = std::exp(-(H[a * K + c] - Hmin));
                Z += wgt;
                Eu += wgt * (a + 0.5) / K;
                Ev += wgt * (c + 0.5) / K;
            }
        Eu /= Z;
        Ev /= Z;

        double su = 0.0, sv = 0.0, cnt = 0.0;
        for (int chain = 0; chain < 4; ++chain) {
            MCMC m = build(W, T, G, part, sc, o, 100 + chain);
            m.adapting = true;
            for (int s = 0; s < 2000; ++s) m.MCMC_sweep(1.0, 1, false);
            m.adapting = false;
            for (int s = 0; s < 40000; ++s) {
                m.MCMC_sweep(1.0, 1, false);
                su += half_cauchy_cdf(m.individual_depth, DEPTH_PRIOR_WIDTH);
                sv += half_cauchy_cdf(m.group_depth, DEPTH_PRIOR_WIDTH);
                cnt += 1.0;
            }
        }
        su /= cnt;
        sv /= cnt;
        char buf[160];
        std::snprintf(buf, sizeof buf, "E[F(sigma_I)] sampled %.4f vs quadrature %.4f", su, Eu);
        check(std::fabs(su - Eu) < 0.01, buf);
        std::snprintf(buf, sizeof buf, "E[F(sigma_G)] sampled %.4f vs quadrature %.4f", sv, Ev);
        check(std::fabs(sv - Ev) < 0.01, buf);
    }
}

// ---------------------------------------------------------------------------
// 4. Gaussian quantile and the score proposal
// ---------------------------------------------------------------------------
void test_gaussian_proposal() {
    std::printf("\n4. Gaussian quantile accuracy and the score proposal's Hastings cancellation\n");
    // (a) quantile against a long-double CDF
    double worst_lo = 0.0;
    for (double lt = -300.0; lt <= std::log10(0.5); lt += 0.01) {
        const double t = std::pow(10.0, lt);
        const double x = gaussian_inv_cdf(t, 0.0, 1.0);
        const long double back = 0.5L * std::erfc(-static_cast<long double>(x) / std::sqrt(2.0L));
        worst_lo = std::max(worst_lo, static_cast<double>(std::fabs(back - t) / t));
    }
    double worst_hi = 0.0;  // p in (0.5, 1): relative to the upper tail 1 - p
    for (double lt = -16.0; lt <= std::log10(0.5); lt += 0.01) {
        const double p = 1.0 - std::pow(10.0, lt), ue = 1.0 - p;  // exact tail of the double p
        if (!(ue > 0.0)) continue;
        const double x = gaussian_inv_cdf(p, 0.0, 1.0);
        const long double back = 0.5L * std::erfc(static_cast<long double>(x) / std::sqrt(2.0L));
        worst_hi = std::max(worst_hi, static_cast<double>(std::fabs(back - ue) / ue));
    }
    char buf[200];
    std::snprintf(buf, sizeof buf, "quantile, t in [1e-300, 0.5]: max |Phi(x) - t| / t = %.2e",
                  worst_lo);
    check(worst_lo < 1e-12, buf);
    std::snprintf(buf, sizeof buf,
                  "quantile, p in (0.5, 1 - 1e-16]: max rel. error of 1 - Phi(x) = %.2e", worst_hi);
    check(worst_hi < 1e-12, buf);
    double worst_z = 0.0;
    for (double z = -37.0; z <= 0.0; z += 0.01)
        worst_z =
            std::max(worst_z, std::fabs(gaussian_inv_cdf(gaussian_cdf(z, 0.0, 1.0), 0.0, 1.0) - z) /
                                  std::max(1.0, std::fabs(z)));
    std::snprintf(buf, sizeof buf, "round trip z in [-37, 0]: max |z' - z| / max(1, |z|) = %.2e",
                  worst_z);
    check(worst_z < 1e-12, buf);

    // (b) invariance: the proposal is reversible w.r.t. N(mu, sigma^2), so a chain
    // that accepts every proposal must keep that distribution.
    const double mu = 0.7, sigma = 1.9;
    rng.seed(12345);
    for (double start : {0.0, 30.0}) {  // 30 sigma: a raw CDF value would be exactly 1
        double z = mu + sigma * start;
        const int T = 2000000;
        double s1 = 0.0, s2 = 0.0, lo3 = 0.0, hi3 = 0.0;
        for (int t = 0; t < T; ++t) {
            z = gaussian_proposal(z, mu, sigma, 0.3);
            const double y = (z - mu) / sigma;
            s1 += y;
            s2 += y * y;
            lo3 += (y < -3.0);
            hi3 += (y > 3.0);
        }
        const double m1 = s1 / T, var = s2 / T - m1 * m1, plo = lo3 / T, phi = hi3 / T;
        const double tail = 0.5 * std::erfc(3.0 / std::sqrt(2.0));  // 1.35e-3
        std::snprintf(buf, sizeof buf,
                      "start %+.0f sd: mean %.4f var %.4f P(<-3) %.5f P(>3) %.5f (exact %.5f)",
                      start, m1, var, plo, phi, tail);
        check(std::fabs(m1) < 0.01 && std::fabs(var - 1.0) < 0.02 &&
                  std::fabs(plo - tail) < 0.15 * tail && std::fabs(phi - tail) < 0.15 * tail,
              buf);
    }
}

// ---------------------------------------------------------------------------
// 5. plain Bradley-Terry keeps neutral edges in the SBM
// ---------------------------------------------------------------------------
void test_bradley_terry_neutral() {
    std::printf("\n5. ties_parameter fixed at 0: ties count in the SBM, not in H_Adir\n");
    const int N = 8;
    Adj W(N), T(N), Tnone(N), G;
    for (int i = 0; i < N; ++i) W[i][(i + 1) % N] = 1.0;
    W[0][2] = 1.0;
    W[3][0] = 2.0;
    for (auto [a, b] : {std::pair<int, int>{1, 3}, {4, 6}, {5, 7}, {0, 5}}) {
        T[a][b] = 1.0;
        T[b][a] = 1.0;
    }
    const std::vector<int> part{0, 0, 0, 0, 1, 1, 1, 1};
    const std::vector<double> sc{0.4, 0.1, -0.2, 0.3, -0.5, 0.0, 0.2, -0.1};
    Opts bt;
    bt.tp = 0.0;
    bt.tp_fixed = true;
    Opts btt;
    btt.tp = 0.5;
    btt.tp_fixed = false;
    MCMC m_bt = build(W, T, G, part, sc, bt);
    MCMC m_btt = build(W, T, G, part, sc, btt);
    MCMC m_wins = build(W, Tnone, G, part, sc, bt);
    check(std::isfinite(m_bt.H_Adir_G_A_s_nu), "H_Adir finite with ties and ties_parameter = 0");
    check(std::fabs(m_bt.H_A_G_b - m_btt.H_A_G_b) < 1e-12 &&
              std::fabs(m_bt.H_k - m_btt.H_k) < 1e-12 &&
              std::fabs(m_bt.H_A_G_k_M_b - m_btt.H_A_G_k_M_b) < 1e-12,
          "SBM terms identical with and without draws modelled");
    check(std::fabs(m_bt.H_A_G_b - m_wins.H_A_G_b) > 1e-6,
          "SBM terms differ from the graph without the ties");
    check(std::fabs(m_bt.H_Adir_G_A_s_nu - m_wins.H_Adir_G_A_s_nu) < 1e-12,
          "H_Adir equals that of the wins alone");
    Opts fr = bt;
    fr.partition_fixed = false;
    fr.scores_fixed = false;
    fr.id_fixed = false;
    fr.gd_fixed = false;
    MCMC m = build(W, T, G, part, sc, fr, 9);
    bool ok = true;
    try {
        for (int s = 0; s < 500; ++s) m.MCMC_sweep(1.0, 1, true);
    } catch (const std::exception& e) {
        ok = false;
        std::printf("      %s\n", e.what());
    }
    check(ok && std::isfinite(m.H_Adir_G_A_s_nu), "500 free sweeps: bookkeeping consistent");
}

// ---------------------------------------------------------------------------
// 6. miscellaneous
// ---------------------------------------------------------------------------
void test_misc() {
    std::printf("\n6. miscellaneous\n");
    auto throws = [](double actual, double cached) {
        try {
            value_check(actual, cached, NUMERICAL_CHECK_EPSILON, true, "x");
        } catch (const std::runtime_error&) {
            return true;
        }
        return false;
    };
    check(throws(-1000.0, -1000.01) && throws(1000.0, 1000.01),
          "value_check: 0.01 drift at |H| = 1000 fails for either sign");
    check(!throws(-1e5, -1e5 * (1.0 + 1e-12)) && !throws(1e5, 1e5 * (1.0 + 1e-12)),
          "value_check: 1e-12 relative round-off passes for either sign");
    check(throws(-1e5, -1e5 - 0.1) && throws(1e5, 1e5 + 0.1),
          "value_check: 0.1 drift at |H| = 1e5 fails");
    {
        const int N = 50000;  // n * n overflows int
        Adj W(N), T(N), G;
        for (int i = 0; i + 1 < N; i += 2) W[i][i + 1] = 1.0;
        Opts o;
        o.din = -1.0;
        o.dout = -1.0;
        MCMC m = build(W, T, G, std::vector<int>(N, 0), std::vector<double>(N, 0.0), o);
        const double expect = (N / 2) / (static_cast<double>(N) * static_cast<double>(N) / 2.0);
        check(std::fabs(m.density_in - expect) < 1e-15 && std::isfinite(m.H_M),
              "n = 50000: initial density = m / (n^2 / 2), H_M finite");
    }
    {
        const int N = 6;
        Adj W(N), T(N), G;
        for (int i = 0; i < N; ++i) W[i][(i + 1) % N] = 1.0;
        Opts o;
        o.assortative = false;
        MCMC m = build(W, T, G, {0, 0, 0, 1, 1, 1}, std::vector<double>(N, 0.0), o);
        for (int k = 0; k < 20; ++k) m.change_parameter("density_out", 0.9);
        bool ok = m.density_out == m.density_in;
        try {
            m.check_state_bookkeeping(true);
        } catch (...) {
            ok = false;
        }
        check(ok, "non-assortative: change_parameter(density_out) keeps density_out = density_in");
    }
}

}  // namespace

int main() {
    std::printf("community_hierarchy core model checks\n\n");
    test_validation();
    test_description_length();
    test_scores_fixed();
    test_gaussian_proposal();
    test_bradley_terry_neutral();
    test_misc();
    std::printf("\n%s (failures=%d)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
