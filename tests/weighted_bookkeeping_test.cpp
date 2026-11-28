// Bookkeeping of the weighted SBM (H_W, weight_counts, nonzero_pairs) and of
// two-layer mode, which no other C++ test builds.  Plain main(), exits non-zero
// on failure.
//
//  1. WEIGHTED SBM -- 200 random weighted graphs; on each, 200 flip_nodes calls of
//     every shape (single node, random subset, whole group, to a new group, to an
//     existing group, self-merge), each followed by a from-scratch recount of H_W
//     and nonzero_pairs and check_state_bookkeeping(true), which now also compares
//     weight_counts / weight_counts_sum / nonzero_pairs with a recount.  Then 20
//     full sweeps with merge/split.
//  2. TWO-LAYER -- random group graphs driving the communities; full sweeps with
//     merge/split, checked at the end of every sweep.

#include "MCMC_core.h"
#include "globals.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {
using Adj = std::vector<std::unordered_map<int, double>>;
int failures = 0;

int count_nonzero(const MCMC& m) {
    int c = 0;
    for (int r = 0; r < m.num_groups; r++)
        for (int s = r; s < m.num_groups; s++)
            if (m.edge_counts[r][s] > 0) c++;
    return c;
}

void random_graph(std::mt19937& g, int N, int n_wins, int n_ties, Adj& W, Adj& T) {
    std::uniform_real_distribution<double> U(0.05, 3.0);
    std::uniform_int_distribution<int> pick(0, N - 1);
    W.assign(N, {});
    T.assign(N, {});
    for (int e = 0; e < n_wins; e++) {
        int a = pick(g), b = pick(g);
        if (a != b && !W[b].count(a)) W[a][b] += U(g);
    }
    for (int e = 0; e < n_ties; e++) {
        int a = pick(g), b = pick(g);
        if (a != b) {
            double w = U(g);
            T[a][b] += w;
            T[b][a] += w;
        }
    }
}

std::vector<int> random_partition(std::mt19937& g, int N, int q) {
    std::uniform_int_distribution<int> pick(0, N - 1);
    std::vector<int> part(N);
    for (int i = 0; i < N; i++) part[i] = i < q ? i : pick(g) % q;
    return part;
}

void test_weighted() {
    const int N = 12;
    std::mt19937 g(5);
    std::uniform_int_distribution<int> pick(0, N - 1);
    int flips = 0, sweeps = 0;
    for (int rep = 0; rep < 200; rep++) {
        Adj W, T, G;
        random_graph(g, N, 20, 4, W, T);
        const int q = 1 + rep % 5;
        const std::vector<int> part = random_partition(g, N, q);
        const std::vector<double> sc(N, 0.1);
        MCMC m(W, T, G, part, false, sc, false, q, false, true, 0.5, false, 0.5, false, 0.5, false,
               1.0, false, 1.0, false, 1.0, false, rep, -1, -1, true);
        for (int k = 0; k < 200; k++) {
            const int src = pick(g) % m.num_groups;
            const auto mem = m.group_members[src];
            std::vector<int> sub;
            const int mode = k % 4;
            if (mode == 0)
                sub = {mem[0]};
            else if (mode == 1)
                sub = mem;  // whole group
            else {
                for (int x : mem)
                    if (g() % 2) sub.push_back(x);
                if (sub.empty()) sub = {mem[0]};
            }
            const int tgt = (g() % 3 == 0) ? m.num_groups : static_cast<int>(g() % m.num_groups);
            m.flip_nodes(sub, tgt);
            ++flips;
            const double hw_true = m.calculate_H_W();
            if (m.nonzero_pairs != count_nonzero(m) ||
                std::fabs(m.H_W - hw_true) > 1e-8 * std::max(1.0, std::fabs(hw_true))) {
                if (failures < 5)
                    std::printf("    mismatch rep %d k %d: nonzero %d vs %d, H_W %.12g vs %.12g\n",
                                rep, k, m.nonzero_pairs, count_nonzero(m), m.H_W, hw_true);
                ++failures;
            }
            try {
                m.check_state_bookkeeping(true);
            } catch (const std::exception& e) {
                if (failures < 5)
                    std::printf("    bookkeeping rep %d k %d: %s\n", rep, k, e.what());
                ++failures;
            }
        }
        try {
            for (int s = 0; s < 20; s++, ++sweeps) m.MCMC_sweep(1.0, 1, true);
        } catch (const std::exception& e) {
            ++failures;
            std::printf("    sweep rep %d: %s\n", rep, e.what());
        }
    }
    std::printf("1. weighted SBM: %d flips and %d sweeps checked  %s\n", flips, sweeps,
                failures ? "FAIL" : "PASS");
}

void test_two_layer() {
    const int before = failures;
    const int N = 14;
    std::mt19937 g(11);
    std::uniform_int_distribution<int> pick(0, N - 1);
    int sweeps = 0;
    for (int rep = 0; rep < 50; rep++) {
        Adj W, T, Gw;
        random_graph(g, N, 25, 5, W, T);
        // integer-valued wins/ties (multiplicities for the direction model)
        for (auto& row : W)
            for (auto& e : row) e.second = std::ceil(e.second);
        for (int i = 0; i < N; ++i)
            for (auto& e : T[i])
                if (i < e.first) {
                    e.second = std::ceil(e.second);
                    T[e.first][i] = e.second;
                }
        Adj G(N);
        for (int e = 0; e < 30; e++) {
            const int a = pick(g), b = pick(g);
            if (a != b) {
                const double w = 0.5 + (g() % 4);
                G[a][b] = w;
                G[b][a] = w;
            }
        }
        const int q = 1 + rep % 4;
        const std::vector<int> part = random_partition(g, N, q);
        const std::vector<double> sc(N, 0.0);
        MCMC m(W, T, G, part, false, sc, false, q, false, true, 0.5, false, 0.5, false, 0.5, false,
               1.0, false, 1.0, false, 0.5, false, 100 + rep, -1, -1, rep % 2 == 1);
        try {
            for (int s = 0; s < 40; s++, ++sweeps) m.MCMC_sweep(1.0, 1, true);
        } catch (const std::exception& e) {
            ++failures;
            std::printf("    two-layer rep %d: %s\n", rep, e.what());
        }
    }
    std::printf("2. two-layer (unweighted and weighted_SBM): %d sweeps checked  %s\n", sweeps,
                failures == before ? "PASS" : "FAIL");
}
}  // namespace

int main() {
    std::printf("community_hierarchy weighted / two-layer bookkeeping\n\n");
    test_weighted();
    test_two_layer();
    std::printf("\n%s (failures=%d)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
