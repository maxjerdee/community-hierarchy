// =============================================================================
//  Merge and split moves  (Peixoto, arXiv:2003.07070, Eqs. 17, 32, 33, 34, 38)
// =============================================================================
//
//  Ported from the validated reference implementation
//  NetworksProjects/Community/MCMC/merge_split_mcmc.py, which is checked against
//  the exact partition prior (uniform P(q)) and per-move detailed balance.
//
//  THE ONE IDEA THAT MAKES THIS WORK (Eqs. 25-27): the staged split is an
//  AUXILIARY VARIABLE.  Detailed balance in the augmented space is independent
//  of P(xhat|b), so the probability of the prestage and of the M staging sweeps
//  is NEVER computed and never enters the Hastings ratio.  Only the single
//  final Gibbs sweep is scored.  Computing the staging probability would need a
//  sum over every intermediate trajectory and is intractable; not needing it is
//  the whole point.
//
//  PITFALLS -- every one of these was a real bug found by testing, and three of
//  them are live in netgroups/src/MCMC_merge_split.cpp:
//
//   P1. Do NOT multiply the forward split probability by the prestage/staging
//       probability.  (netgroups gets this right.)
//   P2. The reverse split must stage THE MERGED GROUP, not a random one.
//   P3. The reverse staging must draw from the SAME distribution the forward
//       split uses -- i.e. call the same staging routine.
//   P4. The Gibbs sweep is ORDER-DEPENDENT.  Both directions must walk the
//       nodes in the same canonical order.  We sort.  netgroups takes the order
//       from group_members, whose order is history-dependent -- a silent bias.
//   P5. [AGGREGATE] Because this is a LABEL-ARRAY representation, two distinct
//       labellings can describe the same partition, and their probabilities
//       must be SUMMED:
//         (a) a split assignment and its complement (which side is "new");
//         (b) merging r into s and s into r.
//       Omitting (a) is not merely a factor ~2: under the per-node no-empty
//       constraint a large fraction of individual assignments have probability
//       exactly zero while the PARTITION does not, so dropping the aggregation
//       makes perfectly good merges unreversible and costs ergodicity.
//
//  EMPTY GROUPS: a sweep may never empty either side.  The exclusion is applied
//  PER NODE and the two-way choice is renormalised on the spot, so the
//  sequential product is already the exact proposal probability -- no global
//  normalisation constant is needed.  (Conditioning globally instead, i.e.
//  sampling freely and discarding degenerate outcomes, would require P(no-op),
//  a sum over all trajectories, which is intractable.)  It also keeps every
//  intermediate state legal, so compaction never fires mid-sweep and the two
//  group labels stay put.
// =============================================================================

#include "MCMC_core.h"
#include "globals.h"
#include "helpers.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
constexpr double NEG_INF = -std::numeric_limits<double>::infinity();
}

// -----------------------------------------------------------------------------
//  log pi of the current LABELLED state.  Merge/split proposals do not cancel
//  H_b, so it appears explicitly; the label-multiplicity factor below converts
//  to the partition-level target q! exp(-...), which is also what
//  propose_single_site_move samples (its proposal cancels q! exp(-H_b)).
// -----------------------------------------------------------------------------
double MCMC::merge_split_log_pi() const {
    // directedstructure's target over the partition (scores held fixed by these
    // moves, so H_Adir_G_A_s_nu is constant and omitted):
    //     pi propto exp(-(H_b + H_s_G_b_depths + beta * H_A_G_b)).
    // Only the data likelihood H_A_G_b is tempered; H_s_G_b_depths is the PRIOR of
    // the scores given the partition and, like H_b, is not (the tempering
    // convention of MCMC_core.h, shared by every move and by the PT swaps).
    // H_A_G_b already contains H_A_G_k_M_b + H_k + H_M + H_W, so the weighted-SBM
    // term needs no separate handling.  H_s_G_b_depths MUST be here: the
    // partition changes the group score means.
    return -(H_b + H_s_G_b_depths + beta * H_A_G_b);
}

// -----------------------------------------------------------------------------
//  LABEL-MULTIPLICITY FACTOR.
//
//  Nothing here redefines H_b.  H_b stays exactly what it is everywhere else --
//  the description length of a LABELLED assignment, as reported in the samples
//  dataframe, used for the MDL, and cancelled by the single-node proposal.
//
//  The reason a factor is needed at all: our merge/split proposal probabilities
//  are aggregated over labellings (that is what the two [AGGREGATE] steps do),
//  so they are probabilities of a PARTITION.  A partition with q blocks has q!
//  labellings, all sharing the same H_b, so
//        P(partition) = q! * exp(-H_b).
//  Matching partition-space proposals against a labelled target is inconsistent
//  by q!/q'!, and since q changes by exactly one that collapses to a single
//  factor on the acceptance ratio, with K = the group count BEFORE the move:
//
//        split  (K -> K+1):  acceptance ratio x (K+1)
//        merge  (K -> K-1):  acceptance ratio x 1/K
//
//  Omit it and every split is penalised by (K+1); the chain collapses to low q
//  (measured TV(q) 0.17 at n=4 instead of 0.002).
//
//  Returned as a LOG, to be subtracted from delta (since acceptance = exp(-delta)).
// -----------------------------------------------------------------------------
double MCMC::split_label_factor(int K) {
    return std::log(static_cast<double>(K + 1));   // split: K -> K+1
}
double MCMC::merge_label_factor(int K) {
    return -std::log(static_cast<double>(K));      // merge: K -> K-1
}

// -----------------------------------------------------------------------------
//  One Gibbs sweep over `nodes`, each of which sits in group ga or gb.
//
//  A single routine serves all three uses so the forward and reverse paths can
//  never drift apart (that drift is exactly pitfall P4):
//     target == nullptr : sample an outcome
//     target != nullptr : walk to that exact assignment and score it
//     accumulate        : whether to return the log-probability
//
//  Returns NEG_INF if a forced move is blocked by the no-empty rule, which does
//  happen for a sizeable minority of assignments; the [AGGREGATE] sum over the
//  complement is what recovers a non-zero probability for the PARTITION.
//  The state is always left with both sides non-empty, so the caller can always
//  restore it with set_split_assignment().
// -----------------------------------------------------------------------------
double MCMC::split_gibbs_sweep(const std::vector<int>& nodes, int ga, int gb,
                               bool accumulate,
                               const std::vector<char>* target,
                               std::vector<char>* out_assignment) {
    double log_p = 0.0;

    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const int node  = nodes[i];
        const int cur   = partition[node];
        const int other = (cur == ga) ? gb : ga;

        const double logw_stay = merge_split_log_pi();

        // Moving is forbidden when it would empty the node's current side.  We
        // must not even tentatively flip in that case: emptying a group fires
        // compaction, which would renumber ga/gb underneath us.
        const bool can_move = (node_counts[cur] > 1);
        double logw_move = NEG_INF;
        if (can_move) {
            flip_nodes({node}, other);
            logw_move = merge_split_log_pi();
            flip_nodes({node}, cur);   // `other` had >=1 member, so this cannot empty it
        }

        double p_stay = 1.0;
        if (can_move) {
            const double m = std::max(logw_stay, logw_move);
            const double a = std::exp(logw_stay - m);
            const double b = std::exp(logw_move - m);
            p_stay = a / (a + b);
        }

        bool stay;
        if (target != nullptr) {
            const int want = ((*target)[i] == 0) ? ga : gb;
            stay = (want == cur);
            if (!stay && !can_move) return NEG_INF;   // unreachable assignment
        } else {
            stay = (uniform_random(rng) < p_stay);
        }

        if (accumulate) {
            const double p = stay ? p_stay : (1.0 - p_stay);
            if (!(p > 0.0)) return NEG_INF;
            log_p += std::log(p);
        }

        if (!stay) flip_nodes({node}, other);
        if (out_assignment != nullptr) {
            (*out_assignment)[i] = (partition[node] == ga) ? 0 : 1;
        }
    }
    return log_p;
}

// -----------------------------------------------------------------------------
//  Force `nodes` into the given assignment without ever emptying a side.
//
//  A naive two-phase pass can empty a side transiently (if every current member
//  of one side is destined for the other).  Instead we defer blocked moves and
//  retry.  This terminates: if a wrong node were the sole member of its side,
//  the target -- which has both sides non-empty -- would require some other node
//  on that side, and every other node is already correct, so one of them is
//  there.  Contradiction.
// -----------------------------------------------------------------------------
void MCMC::set_split_assignment(const std::vector<int>& nodes, int ga, int gb,
                                const std::vector<char>& assignment) {
    bool progress = true;
    while (progress) {
        progress = false;
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            const int want = (assignment[i] == 0) ? ga : gb;
            const int cur  = partition[nodes[i]];
            if (cur == want) continue;
            if (node_counts[cur] > 1) {
                flip_nodes({nodes[i]}, want);
                progress = true;
            }
        }
    }
}

// -----------------------------------------------------------------------------
//  Prestage + M staging sweeps for `nodes`, currently all in group `ga`.
//  Creates the second side as a brand-new group and returns its index.
//
//  AUXILIARY: no probability is computed or returned (P1).  Peixoto samples the
//  prestage uniformly from three strategies; we implement his "random split"
//  (size m uniform in [1,k-1], then a uniform subset of that size).  Note that
//  this is NOT a uniform random bitmask, which would weight m by C(k,m) and so
//  almost always produce a balanced split.  Adding his sequential spreading and
//  sequential coalescence strategies would improve the acceptance rate; being
//  auxiliary, they cannot affect correctness.
// -----------------------------------------------------------------------------
//  Peixoto samples the prestage uniformly from THREE strategies, and is explicit
//  that the variety is the point: "counterintuitively, what we need is a variety
//  of split proposals, that are both good and bad".  Good splits let good splits
//  be proposed; bad splits are what let a bad MERGE be accepted, since a merge's
//  acceptance needs the reverse split to be plausible.
//
//  All of this is AUXILIARY -- none of these probabilities is ever computed, so
//  the choice of strategy cannot affect correctness, only the acceptance rate.
//  The one hard requirement (P3) is that both directions call this same routine.

// Strategy 1: random split -- size m uniform in [1,k-1], then a uniform subset
// of that size.  NOT a uniform random bitmask, which would weight m by C(k,m)
// and so nearly always produce a balanced split.
int MCMC::prestage_random(const std::vector<int>& nodes, int /*ga*/) {
    const int k = static_cast<int>(nodes.size());
    std::vector<int> shuffled(nodes);
    for (int i = k - 1; i > 0; --i)
        std::swap(shuffled[i], shuffled[next_int(rng, 0, i)]);
    const int m = next_int(rng, 1, k - 1);
    std::vector<int> subset(shuffled.begin(), shuffled.begin() + m);
    flip_nodes(subset, num_groups);           // proper subset, so ga cannot empty
    return partition[subset[0]];
}

// Strategies 2 and 3: place nodes one at a time, in random order, with
// probability proportional to pi of the resulting working partition.  Two seeds
// are forced onto opposite sides so neither can be empty.
//
// They differ ONLY in where the not-yet-placed nodes sit, which is what changes
// the pi weights:
//   2. sequential spreading   -- they stay together in the source group
//   3. sequential coalescence -- each sits in its own singleton group
// Coalescence makes the prior favour absorbing the stragglers, which is what
// gives it its agglomerative character.
//
// Group indices are tracked through the SEED NODES, not by index: in the
// coalescence case each placement empties a singleton group and triggers
// compaction, which can renumber ga/gb underneath us.
int MCMC::prestage_sequential(const std::vector<int>& nodes, int ga, bool coalescence) {
    const int k = static_cast<int>(nodes.size());
    std::vector<int> order(nodes);
    for (int i = k - 1; i > 0; --i)
        std::swap(order[i], order[next_int(rng, 0, i)]);

    const int seed_a = order[0];
    const int seed_b = order[1];

    if (coalescence) {
        // every node except seed_a into its own fresh group
        for (int i = 1; i < k; ++i) flip_nodes({order[i]}, num_groups);
    } else {
        flip_nodes({seed_b}, num_groups);     // just split off the second seed
    }
    (void)ga;                                  // ga is tracked via seed_a from here

    for (int i = 2; i < k; ++i) {
        const int node = order[i];
        int A = partition[seed_a];
        flip_nodes({node}, A);                 // may compact (coalescence case)
        A = partition[seed_a];
        int B = partition[seed_b];
        const double logw_a = merge_split_log_pi();

        flip_nodes({node}, B);                 // A keeps seed_a, so cannot empty
        const double logw_b = merge_split_log_pi();

        const double m = std::max(logw_a, logw_b);
        const double wa = std::exp(logw_a - m), wb = std::exp(logw_b - m);
        if (uniform_random(rng) < wa / (wa + wb)) {
            flip_nodes({node}, partition[seed_a]);
        }
    }
    return partition[seed_b];
}

// -----------------------------------------------------------------------------
//  Prestage (one of the three strategies, uniformly) + M staging sweeps.
//  Creates the second side as a new group and returns its index.
//  AUXILIARY: no probability is computed or returned (P1).
// -----------------------------------------------------------------------------
int MCMC::stage_split(const std::vector<int>& nodes, int ga, int sweeps) {
    const int which = (MERGE_SPLIT_PRESTAGE_MODE == 0)
                    ? next_int(rng, 0, 2)
                    : MERGE_SPLIT_PRESTAGE_MODE - 1;
    int gb;
    if (which == 0)      gb = prestage_random(nodes, ga);
    else                 gb = prestage_sequential(nodes, ga, /*coalescence=*/(which == 2));

    const int ga_now = partition[nodes[0]] == gb ? -1 : partition[nodes[0]];
    // ga may have been renumbered by compaction during a coalescence prestage;
    // recover it as "the side that is not gb".
    int ga_eff = ga_now;
    if (ga_eff < 0) {
        for (int nd : nodes) if (partition[nd] != gb) { ga_eff = partition[nd]; break; }
    }

    for (int s = 0; s < sweeps; ++s) {
        split_gibbs_sweep(nodes, ga_eff, gb, /*accumulate=*/false,
                          /*target=*/nullptr, /*out=*/nullptr);
    }
    return gb;
}

// -----------------------------------------------------------------------------
//  Probability of PROPOSING the partition described by `assignment`, given the
//  staged state.  [AGGREGATE] (a): sums the assignment and its complement,
//  which describe the same partition.  Restores the staged state in between so
//  both are scored from the same starting point.
// -----------------------------------------------------------------------------
double MCMC::split_partition_logprob(const std::vector<int>& nodes, int ga, int gb,
                                     const std::vector<char>& staged,
                                     const std::vector<char>& assignment) {
    set_split_assignment(nodes, ga, gb, staged);
    const double lp1 = split_gibbs_sweep(nodes, ga, gb, true, &assignment, nullptr);

    std::vector<char> flipped(assignment.size());
    for (std::size_t i = 0; i < assignment.size(); ++i) flipped[i] = assignment[i] ? 0 : 1;

    set_split_assignment(nodes, ga, gb, staged);
    const double lp2 = split_gibbs_sweep(nodes, ga, gb, true, &flipped, nullptr);

    if (lp1 == NEG_INF && lp2 == NEG_INF) return NEG_INF;
    const double m = std::max(lp1, lp2);
    return m + std::log((lp1 == NEG_INF ? 0.0 : std::exp(lp1 - m)) +
                        (lp2 == NEG_INF ? 0.0 : std::exp(lp2 - m)));
}

// -----------------------------------------------------------------------------
//  [AGGREGATE] (b): log P of proposing a particular merged partition.  An
//  ordered pair (r,s) is drawn uniformly with probability 1/(K(K-1)), but
//  merging r into s and s into r give the SAME partition, so the outcome has
//  probability 2/(K(K-1)).
// -----------------------------------------------------------------------------
double MCMC::merge_logprob(int K) const {
    if (K < 2) return NEG_INF;
    return std::log(2.0) - std::log(static_cast<double>(K))
                         - std::log(static_cast<double>(K - 1));
}

// =============================================================================
//  SPLIT MOVE  (proposal Eq. 32, acceptance Eq. 33/38)
// =============================================================================
void MCMC::propose_split_move() {
    split_proposals++;
    if (partition_fixed || num_groups_fixed) return;

    const int K = num_groups;
    const double H_A_G_b_old = H_A_G_b;
    const double H_s_G_b_depths_old = H_s_G_b_depths;

    const double H_b_old = H_b;
    const int r = next_int(rng, 0, K - 1);
    std::vector<int> nodes = group_members[r];
    if (nodes.size() < 2) return;                     // singleton: nothing to split
    std::sort(nodes.begin(), nodes.end());            // canonical order (P4)

    const int gb = stage_split(nodes, r, MERGE_SPLIT_GIBBS_SWEEPS);
    const int ga = r;

    std::vector<char> staged(nodes.size());
    for (std::size_t i = 0; i < nodes.size(); ++i)
        staged[i] = (partition[nodes[i]] == ga) ? 0 : 1;

    // final sweep: this one IS scored
    std::vector<char> assignment(nodes.size());
    split_gibbs_sweep(nodes, ga, gb, /*accumulate=*/false, nullptr, &assignment);

    const double log_sweep = split_partition_logprob(nodes, ga, gb, staged, assignment);
    set_split_assignment(nodes, ga, gb, assignment);   // land on the proposal

    if (log_sweep == NEG_INF) {
        const std::vector<int> side_b = group_members[gb];   // copy: flip_nodes mutates it
        flip_nodes(side_b, ga);
        return;
    }

    const double log_fwd = -std::log(static_cast<double>(K)) + log_sweep;  // Eq. 32
    const double log_rev = pair_logprob(ga, gb);   // reverse is a merge of {ga, gb}, Eq. 19, in the K+1-group state

    const double delta = (H_b - H_b_old)
                       + beta * (H_A_G_b - H_A_G_b_old)
                       + (H_s_G_b_depths - H_s_G_b_depths_old)   // score prior: untempered (merge_split_log_pi)
                       + log_fwd - log_rev
                       - split_label_factor(K);      // x (K+1); H_b untouched

    if (uniform_random(rng) >= std::exp(-delta)) {
        const std::vector<int> side_b = group_members[gb];   // copy: flip_nodes mutates it
        flip_nodes(side_b, ga);                       // reject: merge the two sides back
    } else {
        accepted_split_proposals++;
    }
}

// =============================================================================
//  MERGE MOVE  (proposal Eq. 17, acceptance Eq. 34/38)
// =============================================================================
void MCMC::propose_merge_move() {
    merge_proposals++;
    if (partition_fixed || num_groups_fixed) return;

    const int K = num_groups;
    if (K < 2) return;
    const double H_A_G_b_old = H_A_G_b;
    const double H_s_G_b_depths_old = H_s_G_b_depths;

    const double H_b_old = H_b;
    // Pair choice (Peixoto 2020 Eq. 19): r uniform, s ~ P(s | r, b).
    int r = next_int(rng, 0, K - 1);
    int s = sample_partner(r);

    std::vector<int> nodes_r = group_members[r];
    std::vector<int> nodes   = nodes_r;
    nodes.insert(nodes.end(), group_members[s].begin(), group_members[s].end());
    std::sort(nodes.begin(), nodes.end());            // canonical order (P4)

    // target for the reverse split: the original two groups.  Read off the
    // partition BEFORE merging -- O(k), unlike a search of nodes_r per node.
    std::vector<char> target(nodes.size());
    for (std::size_t i = 0; i < nodes.size(); ++i)
        target[i] = (partition[nodes[i]] == r) ? 0 : 1;

    const double log_fwd = pair_logprob(r, s);      // P of choosing the unordered pair {r, s}

    // --- perform the merge ---
    flip_nodes(nodes_r, s);
    const int merged = partition[nodes[0]];

    // --- reverse split probability: stage THE MERGED GROUP (P2), using the
    //     same staging routine the forward split uses (P3) ---
    const int gb = stage_split(nodes, merged, MERGE_SPLIT_GIBBS_SWEEPS);
    const int ga = merged;

    std::vector<char> staged(nodes.size());
    for (std::size_t i = 0; i < nodes.size(); ++i)
        staged[i] = (partition[nodes[i]] == ga) ? 0 : 1;

    const double log_sweep = split_partition_logprob(nodes, ga, gb, staged, target);

    // collapse the scratch split back so the proposed state is the merged one
    const std::vector<int> scratch_side = group_members[gb];  // copy: flip_nodes mutates it
    flip_nodes(scratch_side, ga);

    if (log_sweep == NEG_INF) {                       // reverse impossible -> reject
        flip_nodes(nodes_r, num_groups);
        return;
    }

    const double log_rev = -std::log(static_cast<double>(K - 1)) + log_sweep;  // Eq. 32

    const double delta = (H_b - H_b_old)
                       + beta * (H_A_G_b - H_A_G_b_old)
                       + (H_s_G_b_depths - H_s_G_b_depths_old)   // score prior: untempered (merge_split_log_pi)
                       + log_fwd - log_rev
                       - merge_label_factor(K);      // x 1/K; H_b untouched

    if (uniform_random(rng) >= std::exp(-delta)) {
        // Reject: split the original nodes back out into a fresh group.  Labels
        // differ from the original but the PARTITION is identical, which is all
        // that matters under the label-permutation symmetry (the same argument
        // propose_single_site_move already relies on for its merge undo).
        flip_nodes(nodes_r, num_groups);
    } else {
        accepted_merge_proposals++;
    }
}


// =============================================================================
//  JOINT MERGE-SPLIT MOVE  (Eq. 35, accepted with Eq. 36)
//
//  Merge r into s and immediately split the result again.  q is UNCHANGED, so
//  this redistributes nodes between two groups without passing through a state
//  with a different number of groups.  Peixoto reports it helps when the
//  posterior pins down q but not the assignment -- exactly the regime where a
//  plain merge followed later by a plain split would have to cross a barrier.
//
//  Two simplifications fall out of q being unchanged:
//    * both directions draw an ordered pair from the SAME number of groups, so
//      the 2/(K(K-1)) merge factors cancel;
//    * the label-multiplicity factor is q!/q'! = 1, so there is NO (K+1) or 1/K
//      term here.  That is a real difference from the plain moves, not an
//      oversight.
//  Only the two final-sweep probabilities survive in the ratio.
// =============================================================================
void MCMC::propose_merge_split_move() {
    merge_split_proposals++;
    if (partition_fixed) return;
    const int K = num_groups;
    if (K < 2) return;

    const double H_b_old = H_b, H_A_G_b_old = H_A_G_b;
    const double H_s_G_b_depths_old = H_s_G_b_depths;

    // Pair choice (Peixoto 2020 Eq. 19): r uniform, s ~ P(s | r, b).
    int r = next_int(rng, 0, K - 1);
    int s = sample_partner(r);
    const double log_pair_fwd = pair_logprob(r, s);

    const std::vector<int> nodes_r = group_members[r];
    std::vector<int> nodes = nodes_r;
    nodes.insert(nodes.end(), group_members[s].begin(), group_members[s].end());
    std::sort(nodes.begin(), nodes.end());            // canonical order (P4)

    // the original split, as an assignment over `nodes`
    std::vector<char> original(nodes.size());
    for (std::size_t i = 0; i < nodes.size(); ++i)
        original[i] = (partition[nodes[i]] == r) ? 0 : 1;   // before the merge: O(k)

    // ---- forward: merge, then stage and split again ----
    flip_nodes(nodes_r, s);
    int merged = partition[nodes[0]];
    int gb = stage_split(nodes, merged, MERGE_SPLIT_GIBBS_SWEEPS);
    int ga = -1;
    for (int nd : nodes) if (partition[nd] != gb) { ga = partition[nd]; break; }
    // stage_split always leaves both sides non-empty.  If that ever failed, the
    // chain would be sitting in the MERGED state, and returning would accept it
    // with no Metropolis test -- so fail loudly rather than bias silently.
    if (ga < 0) throw std::logic_error("propose_merge_split_move: staged split has an empty side");

    std::vector<char> staged(nodes.size());
    for (std::size_t i = 0; i < nodes.size(); ++i) staged[i] = (partition[nodes[i]] == ga) ? 0 : 1;

    std::vector<char> proposed(nodes.size());
    split_gibbs_sweep(nodes, ga, gb, false, nullptr, &proposed);
    const double log_fwd = split_partition_logprob(nodes, ga, gb, staged, proposed);

    // ---- reverse: collapse again, re-stage, and score the ORIGINAL split ----
    set_split_assignment(nodes, ga, gb, proposed);
    const std::vector<int> side_b = group_members[gb];
    flip_nodes(side_b, ga);
    int merged2 = partition[nodes[0]];
    int gb2 = stage_split(nodes, merged2, MERGE_SPLIT_GIBBS_SWEEPS);
    int ga2 = -1;
    for (int nd : nodes) if (partition[nd] != gb2) { ga2 = partition[nd]; break; }
    if (ga2 < 0) throw std::logic_error("propose_merge_split_move: reverse staged split has an empty side");

    std::vector<char> staged2(nodes.size());
    for (std::size_t i = 0; i < nodes.size(); ++i) staged2[i] = (partition[nodes[i]] == ga2) ? 0 : 1;

    const double log_rev = split_partition_logprob(nodes, ga2, gb2, staged2, original);

    if (log_fwd == NEG_INF || log_rev == NEG_INF) {
        set_split_assignment(nodes, ga2, gb2, original);   // back to where we started
        return;
    }

    set_split_assignment(nodes, ga2, gb2, proposed);        // land on the proposal
    // The Eq. 19 pair probability depends on the state, so unlike a uniform pair
    // choice it does not cancel: the reverse picks {ga2, gb2} in the proposed state.
    const double log_pair_rev = pair_logprob(ga2, gb2);

    // Eq. 36.  No label factor: q is unchanged.
    const double delta = (H_b - H_b_old)
                       + beta * (H_A_G_b - H_A_G_b_old)
                       + (H_s_G_b_depths - H_s_G_b_depths_old)   // score prior: untempered (merge_split_log_pi)
                       + log_fwd - log_rev + log_pair_fwd - log_pair_rev;

    if (uniform_random(rng) >= std::exp(-delta)) {
        set_split_assignment(nodes, ga2, gb2, original);    // reject
    } else {
        accepted_merge_split_proposals++;
    }
}
