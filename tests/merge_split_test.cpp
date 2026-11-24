// Validate directedstructure's merge/split moves.
//
//  1. BOOKKEEPING -- check_state_bookkeeping(true) after every move, so any
//     incremental-update error aborts immediately.
//  2. PRIOR (beta=0) -- with no data and a partition-independent score prior
//     (group_depth = 0) the target is exp(-H_b), whose q-marginal is exactly
//     uniform on 1..n.  Run from BOTH extremes (B=1 and B=N) so a non-mixing
//     config shows up as disagreement between chains.
//  2b. SCORE PRIOR (beta=0, exact) -- with fixed scores and group_depth > 0 the
//     partition target is exp(-H_b - H_s_G_b_depths): the score prior is NOT
//     tempered (MCMC_core.h).  Enumerate all partitions of n=5 and compare.
//  3. AGREEMENT (beta=1) -- on a real tournament, single-site and merge/split
//     must produce the same coincidence matrix.  This is the check that would
//     catch omitting H_s_G_b_depths from the energy: that term VANISHES at
//     beta=0, so test 2 alone cannot see it.

#include "MCMC_core.h"
#include "globals.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <random>
#include <unordered_map>
#include <vector>

namespace {
int failures = 0;

MCMC build(const std::vector<std::unordered_map<int, double>>& wins,
           const std::vector<std::unordered_map<int, double>>& ties,
           int N, const std::vector<int>& partition, int q, int seed,
           double group_depth = 0.5) {
    std::vector<std::unordered_map<int, double>> gadj;
    std::vector<double> scores(N);
    for (int i = 0; i < N; ++i) scores[i] = (i - N / 2) / static_cast<double>(N);
    return MCMC(wins, ties, gadj, partition, false, scores, /*scores_fixed=*/true,
                q, false, false,
                0.5, true, 0.5, true, 0.0, true,
                0.5, true, group_depth, true, 0.5, true,
                seed, 0.4, 0.4, false);
}

// Canonical form of a partition: relabel groups by first occurrence.
std::vector<int> canon(const std::vector<int>& p) {
    std::map<int, int> m; std::vector<int> out;
    for (int x : p) { auto it = m.find(x); if (it == m.end()) it = m.emplace(x, (int)m.size()).first; out.push_back(it->second); }
    return out;
}

// ---- 2b. exact partition target at beta = 0 with a partition-DEPENDENT score prior ----
void score_prior_check(bool enabled, int moves, const char* label, int sweeps = 60000) {
    const int N = 5;
    const double GD = 0.8;                       // group_depth > 0: H_s depends on b
    MERGE_SPLIT_MOVES = moves;
    std::vector<std::unordered_map<int, double>> wins(N), ties(N);   // NO data
    // exact: all set partitions (restricted growth strings), weight q! exp(-H_b - H_s)
    std::map<std::vector<int>, double> exact;
    std::vector<int> rgs(N, 0);
    double wsum = 0.0;
    std::function<void(int, int)> rec = [&](int i, int qmax) {
        if (i == N) {
            MCMC st = build(wins, ties, N, rgs, qmax, 1, GD);
            const double lw = std::lgamma(qmax + 1.0) - st.H_b - st.H_s_G_b_depths;
            exact[rgs] = lw; return;
        }
        for (int g = 0; g <= qmax && g < N; ++g) { rgs[i] = g; rec(i + 1, std::max(qmax, g + 1)); }
    };
    rec(0, 0);
    double mx = -1e300; for (auto& [k, v] : exact) mx = std::max(mx, v);
    for (auto& [k, v] : exact) { v = std::exp(v - mx); wsum += v; }
    for (auto& [k, v] : exact) v /= wsum;
    // sampled
    std::map<std::vector<int>, double> freq; double total = 0.0;
    for (int c = 0; c < 4; ++c) {
        std::vector<int> part(N); int q;
        if (c % 2 == 0) { for (int i = 0; i < N; ++i) part[i] = 0; q = 1; }
        else            { for (int i = 0; i < N; ++i) part[i] = i; q = N; }
        MCMC st = build(wins, ties, N, part, q, 5001 + c, GD);
        for (int i = 0; i < sweeps / 10; ++i) st.MCMC_sweep(0.0, 1, enabled);
        for (int i = 0; i < sweeps; ++i) { st.MCMC_sweep(0.0, 1, enabled); freq[canon(st.partition)] += 1.0; total += 1.0; }
    }
    double tv = 0.0;
    for (auto& [k, p] : exact) tv += std::fabs((freq.count(k) ? freq[k] / total : 0.0) - p) / 2.0;
    const bool ok = tv < 0.02;
    if (!ok) ++failures;
    std::printf("  %-34s TV(partition)=%.5f over %zu partitions  %s\n", label, tv, exact.size(), ok ? "PASS" : "FAIL");
}

// ---- 2. beta=0 prior: P(q) must be uniform ----
void prior_check(int N, bool enabled, int moves, const char* label, int sweeps = 60000) {
    MERGE_SPLIT_MOVES = moves;
    std::vector<std::unordered_map<int, double>> wins(N), ties(N);   // NO data
    const int CH = 4;
    std::vector<std::vector<double>> frac(CH, std::vector<double>(N + 1, 0.0));
    for (int c = 0; c < CH; ++c) {
        std::vector<int> part(N);
        int q;
        if (c % 2 == 0) { for (int i = 0; i < N; ++i) part[i] = 0; q = 1; }
        else            { for (int i = 0; i < N; ++i) part[i] = i; q = N; }
        MCMC st = build(wins, ties, N, part, q, 3001 + c, /*group_depth=*/0.0);
        for (int i = 0; i < sweeps / 10; ++i) st.MCMC_sweep(0.0, 1, enabled);
        std::vector<double> cnt(N + 1, 0.0);
        for (int i = 0; i < sweeps; ++i) {
            st.MCMC_sweep(0.0, 1, enabled);
            cnt[st.num_groups] += 1.0;
        }
        for (int q2 = 1; q2 <= N; ++q2) frac[c][q2] = cnt[q2] / sweeps;
    }
    double tv = 0.0, worst = 0.0;
    for (int q2 = 1; q2 <= N; ++q2) {
        double m = 0.0;
        for (int c = 0; c < CH; ++c) m += frac[c][q2] / CH;
        tv += std::fabs(m - 1.0 / N) / 2.0;
        worst = std::max(worst, std::fabs(m - 1.0 / N));
    }
    const bool ok = tv < 0.02;
    if (!ok) ++failures;
    std::printf("  %-34s TV(q)=%.5f  max|dev|=%.5f  %s\n",
                label, tv, worst, ok ? "PASS" : "FAIL");
}

// ---- 3. beta=1 agreement of the coincidence matrix ----
std::vector<double> coincidence(int N, bool enabled, int moves, int sweeps) {
    MERGE_SPLIT_MOVES = moves;
    std::mt19937 g(99);
    std::vector<std::unordered_map<int, double>> wins(N), ties(N);
    std::uniform_int_distribution<int> pick(0, N - 1);
    for (int i = 0; i < N; ++i) wins[i][(i + 1) % N] += 1.0;
    for (int e = 0; e < 3 * N; ++e) { int a = pick(g), b = pick(g); if (a != b) wins[a][b] += 1.0; }
    std::vector<double> C(N * N, 0.0);
    const int CH = 4;
    for (int c = 0; c < CH; ++c) {
        std::vector<int> part(N);
        int q;
        if (c % 2 == 0) { for (int i = 0; i < N; ++i) part[i] = 0; q = 1; }
        else            { for (int i = 0; i < N; ++i) part[i] = i; q = N; }
        MCMC st = build(wins, ties, N, part, q, 7001 + c);
        for (int i = 0; i < sweeps / 5; ++i) st.MCMC_sweep(1.0, 1, enabled);
        for (int i = 0; i < sweeps; ++i) {
            st.MCMC_sweep(1.0, 1, enabled);
            for (int a = 0; a < N; ++a)
                for (int b = 0; b < N; ++b)
                    if (st.partition[a] == st.partition[b]) C[a * N + b] += 1.0;
        }
    }
    for (auto& v : C) v /= (double)sweeps * CH;
    return C;
}
}  // namespace

int main() {
    std::printf("directedstructure merge/split validation\n\n");

    // 1. bookkeeping integrity
    {
        const int N = 8;
        std::vector<std::unordered_map<int, double>> wins(N), ties(N);
        for (int i = 0; i < N; ++i) wins[i][(i + 1) % N] += 1.0;
        std::vector<int> part(N, 0);
        MCMC st = build(wins, ties, N, part, 1, 12);
        st.beta = 1.0;
        for (int i = 0; i < 4000; ++i) {
            if (i % 3 == 0)      st.propose_split_move();
            else if (i % 3 == 1) st.propose_merge_move();
            else                 st.propose_merge_split_move();
            st.check_state_bookkeeping(true);
        }
        std::printf("1. bookkeeping: 4000 merge/split/joint moves consistent (q=%d)\n",
                    st.num_groups);
        std::printf("   merge %d/%d  split %d/%d  joint %d/%d\n\n",
                    st.accepted_merge_proposals, st.merge_proposals,
                    st.accepted_split_proposals, st.split_proposals,
                    st.accepted_merge_split_proposals, st.merge_split_proposals);
    }

    // 2. beta=0 prior
    std::printf("2. beta=0 prior, P(q) must be uniform = 1/n\n");
    for (int N : {4, 6}) {
        std::printf("   n=%d\n", N);
        // The block-informed node move opens a group with probability 0.01 per
        // proposal (graph-tool's d), so on its own q mixes ~10x slower: run longer.
        prior_check(N, false, 7, "single-site only", 600000);
        prior_check(N, true,  3, "+ merge/split");
        prior_check(N, true,  7, "+ merge/split + joint");
    }

    // 2b. beta=0 with a partition-dependent score prior: exact enumeration
    std::printf("\n2b. beta=0, fixed scores, group_depth > 0: exact target exp(-H_b - H_s)\n");
    score_prior_check(false, 7, "single-site only", 600000);   // slow q mixing alone, see above
    score_prior_check(true,  3, "+ merge/split");
    score_prior_check(true,  7, "+ merge/split + joint");

    // 3. beta=1 agreement
    std::printf("\n3. beta=1 coincidence agreement (N=10 tournament)\n");
    auto c0 = coincidence(10, false, 7, 30000);
    auto c1 = coincidence(10, true,  7, 30000);
    double mx = 0.0, mean = 0.0;
    for (std::size_t i = 0; i < c0.size(); ++i) {
        const double d = std::fabs(c0[i] - c1[i]);
        mx = std::max(mx, d); mean += d / c0.size();
    }
    const bool ok = mx < 0.10;
    if (!ok) ++failures;
    std::printf("   mean|diff|=%.4f  max|diff|=%.4f  %s\n", mean, mx, ok ? "PASS" : "FAIL");

    std::printf("\n%s (failures=%d)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
