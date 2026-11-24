/* Global variable declarations */

#ifndef GLOBALS_H
#define GLOBALS_H

#include <random>

#include "rng.h"
#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Global information about the problem
extern int n; // Number of nodes
extern int m; // Number of edges
extern std::vector<std::unordered_set<int>> neighbors; // neighbors[i] is the list of neighbors of node i

// inline bool verbose = true; // Whether to print verbose output during MCMC
extern bool verbose; // Whether to print verbose output during MCMC
// inline bool debug = true; // Whether to print debug output during MCMC
extern bool debug; // Whether to print debug output during MCMC

// Timing variables.  thread_local: fit() runs chains concurrently on std::async
// threads, and shared timers would be a data race (undefined behaviour).
extern bool timing; // Whether to track detailed timing information
extern thread_local std::chrono::high_resolution_clock::time_point start_time, end_time;
extern thread_local double duration; // Accumulated or temporary duration (seconds)

// Constants
// Base proposal step size (kept for backward compatibility)
const double NUMERICAL_CHECK_EPSILON = 1e-6; // Amount we will allow the bookkeeping values to differ by
const double CONCENTRATION_EPSILON = 1e-12; // Small value to avoid zero concentration parameters
const double MAX_CONCENTRATION = 1e8; // Maximum concentration parameter before approximating with multinomial
const double PARAMETER_PROPOSAL_STEP_SIZE = 0.01; // Global default step size (legacy)
// Per-parameter proposal step sizes (can be tuned independently)
const double VARIATION_IN_STEP_SIZE = PARAMETER_PROPOSAL_STEP_SIZE;
const double VARIATION_OUT_STEP_SIZE = PARAMETER_PROPOSAL_STEP_SIZE;
const double DEGREE_CORRECTION_STEP_SIZE = PARAMETER_PROPOSAL_STEP_SIZE;
const double INDIVIDUAL_DEPTH_STEP_SIZE = PARAMETER_PROPOSAL_STEP_SIZE;
const double GROUP_DEPTH_STEP_SIZE = PARAMETER_PROPOSAL_STEP_SIZE;
const double TIES_PARAMETER_STEP_SIZE = PARAMETER_PROPOSAL_STEP_SIZE;
const double GROUP_SCORE_STEP_SIZE = 0.1; // INITIAL RW step (scaled-score units) for the group-score shift; adapted in burn-in

// ---- step-size adaptation (burn-in only; frozen for sampling) ----
// Robbins-Monro on log step: log s += gamma_t (accepted - target),
// gamma_t = ADAPT_GAIN / (1 + t / ADAPT_T0)^ADAPT_KAPPA.  Applied to the score,
// sigma_I, sigma_G, ties and group-score-shift steps; the others keep their fixed
// steps.  Adapting only during burn-in leaves the sampling kernel fixed, so the
// chain stays exactly invariant.  Target 0.44 is the optimal one-dimensional
// random-walk rate (Roberts & Rosenthal 2001); the gain schedule is ad hoc.
const double ADAPT_TARGET = 0.44;
const double ADAPT_GAIN = 0.5;
const double ADAPT_T0 = 20.0;
const double ADAPT_KAPPA = 0.6;
const double ADAPT_MIN_STEP = 1e-5;
const double ADAPT_MAX_STEP_CDF = 0.5;   // CDF-space steps (scores, sigma_I, sigma_G, ties)
const double ADAPT_MAX_STEP_GROUP_SCORE = 10.0;
const double SCORE_STEP_SIZE = 0.01;     // INITIAL CDF-space step of single-node score moves

// ---- proposal mix ----
// Each free continuous parameter is proposed PARAMETER_MOVE_WEIGHT times as often
// as under one-per-parameter weighting (weight = this x number of free parameters).
const double PARAMETER_MOVE_WEIGHT = 10.0;
// Probability that a sigma_G proposal also rescales every group's mean scaled
// score about 0 by lambda = sigma_G' / sigma_G (Jacobian lambda^q).  A
// state-independent coin, so the mixture of the two kernels stays valid.
const double GROUP_DEPTH_RESCALE_PROB = 0.5;

// ---- single-node group move (Peixoto 2020, Phys. Rev. E 102, 012305, Eq. 15) ----
// With probability BLOCK_NEW_GROUP_PROB the node moves to a new group; otherwise
// s ~ P_e(s | i, b) (Eq. 14): a uniform neighbour j, t = b_j, then s with
// probability (e_ts + c) / (e_t + c B), c = BLOCK_SITE_EPS.  graph-tool defaults.
const double BLOCK_SITE_EPS = 0.5;
const double BLOCK_NEW_GROUP_PROB = 0.01;


const int GIBBS_REFINEMENT_SWEEPS = 1; // Change back to 10 // Number of Gibbs sweeps to use when refining splits

const double DEPTH_PRIOR_WIDTH = 4.0; // Keeping with the depth paper, scale of half-Cauchy prior on depth parameters

// One generator PER THREAD.  fit() runs each chain on its own std::async thread,
// and MCMC constructors call rng.seed(seed).  With a single shared
// generator the chains reseeded one another and then drew from it concurrently:
// a data race, chains that were not independent, and num_chains > 1 results that
// could not be reproduced from the seed.  Each chain constructs its MCMC objects
// on its own thread (see fit.cpp), so each gets its own stream.  fit() seeds it
// once per chain with seed + chain; the chain's tempering replicas are built with
// seed -1 (no reseed) and share that stream.
extern thread_local std::mt19937 rng;
// Not std::uniform_real_distribution: implementation-defined, so a seeded run
// would differ between libstdc++ and libc++ (wasm).  See rng.h.
inline constexpr UniformRandom uniform_random{};


// ---- merge/split tuning (auxiliary: cannot affect correctness) ----
// Number of Gibbs sweeps used to refine a split during STAGING.  This is the
// auxiliary variable, so ANY value (including 0) gives a valid kernel; it trades
// proposal quality against cost.  Peixoto reports M=10 across several networks;
// on small easy networks M=0-1 is faster, but on celegansneural (N=297) M=0
// FAILS to converge.  Default 3: with weight 1 per collective move (below) and
// the block-informed single-node move, M=3 mixed the international-football and
// high-school networks as well as M=10 per unit time, with merge-split taking
// about 60% of the run time on football.
extern int MERGE_SPLIT_GIBBS_SWEEPS;

// Which prestage strategy to draw from: 0 = all three uniformly (Peixoto),
// 1 = random split, 2 = sequential spreading, 3 = sequential coalescence.
// NOTE strategy 2 ALONE fails at M=0 -- it only ever produces plausible splits,
// so bad merges become unreversible ("we need a variety of split proposals, that
// are both good and bad").  Keep the default at 0.
extern int MERGE_SPLIT_PRESTAGE_MODE;

// Which collective moves to include, as a bitmask: 1=merge, 2=split, 4=joint
// merge-split.  Default 7 (all three).  This exists so tests can isolate a
// single move type; it is NOT part of the public API -- callers turn the whole
// family on or off with the merge_split_enabled flag instead.
// NOTE isolating merge (1) or split (2) alone gives a BIASED chain by
// construction: a merge-only chain can only decrease q and its reverse move is
// never proposed, so detailed balance cannot hold.  Only 3 (merge+split), 4
// (joint alone, which preserves q) and 7 are valid samplers.
extern int MERGE_SPLIT_MOVES;

// Relative weight of EACH collective move (merge, split, joint merge-split).
// Negative (the default) means "compute the linear-time value", see below.
//
// Peixoto gives two choices for the move propensities (Eq. 37 and the
// discussion after it):
//   * omega_single = N, omega_collective = 1  -- what he used experimentally
//   * omega_single = O[M(N+E) * omega_collective] -- the condition for an
//     entire sweep of N proposals to cost O(N+E), i.e. to stay LINEAR
// With the first choice the expected number of collective attempts per sweep is
// N * 3/N = 3, INDEPENDENT of network size, while the cost of each is O(M*k) in
// the group size k.  Measured on an N=1000 synthetic network that puts 0.29% of
// the moves (2.9 per sweep) in charge of ~99% of the runtime -- a 112x slowdown
// against single-site alone.
//
// We therefore default to the linear-time condition, but implement it by
// LOWERING the collective weight rather than raising omega_single:
//     omega_collective = n / (M * (n + m))
// Raising omega_single would also make parameter moves ~M(n+m)/n times rarer
// relative to node moves, which would disturb parameter equilibration; scaling
// the collective weight leaves the single-site : parameter ratio untouched.
//
// The value is STATE-INDEPENDENT (n, m and M are fixed for a given run), which
// matters: a weight that depended on num_groups would make the partition-move
// fraction depend on q and bias the q-marginal.
//
// DEFAULT NOW 1.0 (Peixoto's omega_collective = 1 with omega_single = N): with
// M = 3 Gibbs sweeps and the block-informed single-node move this gave the best
// partition mixing per unit time on the benchmark networks.  Set it negative to
// get the equal-spend retuning described below.
extern double MERGE_SPLIT_WEIGHT;

// Divisor of the equal-spend rule fit() applies while MERGE_SPLIT_WEIGHT is
// negative: w = q / (C * M).  C folds in the staging overhead (each staged node
// is flipped and flipped back, and the split is scored both ways), measured at
// 1-2 per staged node, giving C ~ 9 for "collective moves cost about as much per
// sweep as the single-node moves".  See package-testing/asbm/merge_split_weight.
extern double MERGE_SPLIT_EQUAL_SPEND_C;

#endif // GLOBALS_H
