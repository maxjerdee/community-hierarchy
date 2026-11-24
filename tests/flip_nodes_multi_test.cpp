// Validate the MULTI-NODE path of MCMC::flip_nodes in community_hierarchy.
//
// Ported from the asbm test of the same name.  Every call site in the package
// passes {node}, so the batch path -- the node_counts == nodes.size() test,
// deferred compaction, and the intra-batch edge bookkeeping -- has never
// executed.  Merge and split moves depend on it entirely.
//
// Two independent oracles (a single oracle sharing code with the thing under
// test proves little):
//   ORACLE 1 (from scratch): after a batch flip, build a fresh MCMC on the
//     resulting partition and compare every entropy.  Shares no code with the
//     incremental update path.
//   ORACLE 2 (differential): perform the same physical move as a SEQUENCE of
//     single-node flips -- the path already exercised in production -- and
//     require identical partition and entropies.
//
// All three shapes are covered:
//   SPLIT   : strict subset -> brand-new group
//   MERGE   : whole group   -> existing different group
//   RELABEL : whole group   -> brand-new group   (Case 5; used to corrupt
//             H_M/H_k/H_A_G_k_M_b silently while leaving the partition valid)

#include "MCMC_core.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

int failures = 0, checks = 0;

void expect_close(double a, double b, const std::string& what, double tol = 1e-8) {
    ++checks;
    if (std::isnan(a) && std::isnan(b)) return;
    const double scale = std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
    if (std::fabs(a - b) > tol * scale) {
        std::printf("    FAIL %-24s %.12g vs %.12g  (diff %.3g)\n",
                    what.c_str(), a, b, a - b);
        ++failures;
    }
}
void expect_eq(int a, int b, const std::string& what) {
    ++checks;
    if (a != b) { std::printf("    FAIL %-24s %d vs %d\n", what.c_str(), a, b); ++failures; }
}

const int N = 14;
std::vector<std::unordered_map<int, double>> WINS, TIES, GADJ;

// A deterministic tournament: ring of wins plus random extra wins and ties.
void make_graph(unsigned seed) {
    WINS.assign(N, {}); TIES.assign(N, {}); GADJ.clear();
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pick(0, N - 1);
    for (int i = 0; i < N; ++i) WINS[i][(i + 1) % N] += 1.0;
    for (int e = 0; e < 2 * N; ++e) {
        int a = pick(rng), b = pick(rng);
        if (a != b) WINS[a][b] += 1.0;
    }
    for (int e = 0; e < N; ++e) {
        int a = pick(rng), b = pick(rng);
        if (a != b) { TIES[a][b] += 1.0; TIES[b][a] += 1.0; }
    }
}

MCMC build(const std::vector<int>& partition, int q, int seed) {
    // Depth/ties parameters default to U(0,1) draws in fit.cpp, so 0.0 is
    // degenerate (it makes scaled_scores NaN).  Use 0.5, and give the nodes
    // distinct scores so the score-dependent terms are actually exercised.
    std::vector<double> scores(N);
    for (int i = 0; i < N; ++i) scores[i] = (i - N / 2) / static_cast<double>(N);
    return MCMC(WINS, TIES, GADJ, partition, /*partition_fixed=*/false,
                scores, /*scores_fixed=*/true,
                q, /*num_groups_fixed=*/false, /*assortative=*/false,
                0.5, true,     // variation_in fixed
                0.5, true,     // variation_out fixed
                0.0, true,     // degree_correction fixed
                0.5, true,     // individual_depth fixed
                0.5, true,     // group_depth fixed
                0.5, true,     // ties_parameter fixed
                seed, 0.4, 0.4, /*weighted_SBM=*/false);
}

void compare(MCMC& got, MCMC& want, const std::string& tag) {
    expect_eq(got.num_groups, want.num_groups, tag + " num_groups");
    expect_close(got.H_b,             want.H_b,             tag + " H_b");
    expect_close(got.H_M,             want.H_M,             tag + " H_M");
    expect_close(got.H_k,             want.H_k,             tag + " H_k");
    expect_close(got.H_A_G_k_M_b,     want.H_A_G_k_M_b,     tag + " H_A_G_k_M_b");
    expect_close(got.H_A_G_b,         want.H_A_G_b,         tag + " H_A_G_b");
    expect_close(got.H_s_G_b_depths,  want.H_s_G_b_depths,  tag + " H_s_G_b_depths");
}

}  // namespace

int main() {
    make_graph(4242u);
    std::mt19937 rng(31337);
    int batches = 0;
    std::printf("community_hierarchy: validating multi-node flip_nodes (N=%d)\n", N);

    for (int trial = 0; trial < 400; ++trial) {
        int q = std::uniform_int_distribution<int>(1, 5)(rng);
        std::vector<int> partition(N);
        for (int i = 0; i < N; ++i)
            partition[i] = std::uniform_int_distribution<int>(0, q - 1)(rng);
        std::unordered_map<int, int> remap;
        for (int i = 0; i < N; ++i) {
            if (!remap.count(partition[i])) {
                int nxt = static_cast<int>(remap.size());
                remap[partition[i]] = nxt;
            }
            partition[i] = remap[partition[i]];
        }
        q = static_cast<int>(remap.size());

        MCMC batch = build(partition, q, 7);
        batch.check_state_bookkeeping(true);

        int src = std::uniform_int_distribution<int>(0, batch.num_groups - 1)(rng);
        std::vector<int> members = batch.group_members[src];
        if (members.size() < 2) continue;
        std::vector<int> subset;
        for (int nd : members)
            if (std::uniform_int_distribution<int>(0, 1)(rng)) subset.push_back(nd);
        if (subset.size() < 2) continue;

        int target;
        if (std::uniform_int_distribution<int>(0, 1)(rng) || batch.num_groups < 2) {
            target = batch.num_groups;                 // brand-new group
        } else {
            target = std::uniform_int_distribution<int>(0, batch.num_groups - 2)(rng);
            if (target >= src) ++target;               // existing, != src
        }

        const int q_before = batch.num_groups;
        batch.flip_nodes(subset, target);
        batch.check_state_bookkeeping(true);

        // ORACLE 1: fresh object on the resulting partition
        std::vector<int> after = batch.partition;
        MCMC fresh = build(after, batch.num_groups, 7);
        compare(batch, fresh, "scratch");

        // ORACLE 2: the same move one node at a time
        MCMC seqm = build(partition, q, 7);
        int tgt = target;
        for (std::size_t i = 0; i < subset.size(); ++i) {
            seqm.flip_nodes({subset[i]}, tgt);
            if (i == 0 && target == q_before) tgt = seqm.partition[subset[0]];
            seqm.check_state_bookkeeping(true);
        }
        compare(batch, seqm, "sequential");
        for (int i = 0; i < N; ++i)
            expect_eq(batch.partition[i], seqm.partition[i],
                      "partition[" + std::to_string(i) + "]");
        ++batches;
    }

    std::printf("\n  trials with genuine batches: %d\n", batches);
    std::printf("  assertions: %d,  failures: %d\n", checks, failures);
    std::printf("  %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
