/* Global variable declarations */

#ifndef GLOBALS_H
#define GLOBALS_H

#include <random>

#include "rng.h"

// Constants
// Bookkeeping check (value_check): a cached value passes when
//     |cached - actual| <= tol + NUMERICAL_CHECK_REL_TOL * |actual|,
// with tol = NUMERICAL_CHECK_EPSILON unless the call site passes a larger one.
const double NUMERICAL_CHECK_EPSILON = 1e-6;  // absolute part
const double NUMERICAL_CHECK_REL_TOL = 1e-9;  // relative part (incremental round-off)
const double CONCENTRATION_EPSILON = 1e-12; // Small value to avoid zero concentration parameters
const double MAX_CONCENTRATION = 1e8; // Maximum concentration parameter before approximating with multinomial
const double PARAMETER_PROPOSAL_STEP_SIZE = 0.01; // initial CDF-space step of the parameter moves (adapted in burn-in)
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
// gamma_t = ADAPT_GAIN / (1 + t / ADAPT_T0)^ADAPT_KAPPA.  Applied to every
// continuous move: scores, group-score shifts, both kernels of each depth, ties,
// densities, variations and degree correction (each with its own step).  Adapting only during burn-in leaves the sampling kernel fixed, so the
// chain stays exactly invariant.  Target 0.44 is the optimal one-dimensional
// random-walk rate (Roberts & Rosenthal 2001); the gain schedule is ad hoc.
const double ADAPT_TARGET = 0.44;
const double ADAPT_GAIN = 0.5;
const double ADAPT_T0 = 20.0;
const double ADAPT_KAPPA = 0.6;
const double ADAPT_MIN_STEP = 1e-5;
const double ADAPT_MAX_STEP_CDF = 0.5;   // CDF-space steps (scores, depths, ties, densities, variations)
const double ADAPT_MAX_STEP_GROUP_SCORE = 10.0;
const double SCORE_STEP_SIZE = 0.01;     // INITIAL CDF-space step of single-node score moves

// ---- proposal mix ----
// Each free continuous parameter is proposed PARAMETER_MOVE_WEIGHT times as often
// as under one-per-parameter weighting (weight = this x number of free parameters).
const double PARAMETER_MOVE_WEIGHT = 10.0;
// Probability that a sigma_G proposal also rescales every group's mean score
// about 0 by lambda = sigma_G' / sigma_G (Jacobian lambda^q).  A
// state-independent coin, so the mixture of the two kernels stays valid.  With
// scores_fixed only the plain move is used (the rescale move changes scores).
const double GROUP_DEPTH_RESCALE_PROB = 0.5;
// Probability that a sigma_I proposal is the CENTRED move: sigma_I changes with the
// scores held fixed (no Jacobian), so the likelihood is untouched and only the
// score prior decides -- the counterpart of the plain sigma_G move.  Otherwise a
// move that rescales the scores with sigma_I (INDIVIDUAL_DEPTH_SCALE_MOVE).
// Mixing the two is the strategy of Papaspiliopoulos, Roberts & Skold (2007,
// Stat. Sci. 22:59) and Yu & Meng (2011, JCGS 20:531); the coin is state-
// independent, so the mixture stays valid.  With scores_fixed only the centred
// move is used (the other two change scores).  Mutable for benchmarking.
extern double INDIVIDUAL_DEPTH_CENTRED_PROB;
// Which move the other (1 - INDIVIDUAL_DEPTH_CENTRED_PROB) sigma_I proposals use:
//   1 (default): residual rescale -- each score's deviation from its group mean
//      scales by sigma_I' / sigma_I and the group means stay put, Jacobian
//      lambda^(n-q): the exact counterpart of the sigma_G rescale move, which scales
//      the group means and keeps the deviations;
//   0: every score scales with sigma_I, Jacobian lambda^n.
// The centred move (scores fixed) alone gets stuck at small sigma_I when the data
// are weak; a scaling move alone is limited to tiny steps when the data pin the
// scores (dense or weighted networks).  Mutable for benchmarking.
extern int INDIVIDUAL_DEPTH_SCALE_MOVE;


// ---- single-node group move (Peixoto 2020, Phys. Rev. E 102, 012305, Eq. 15) ----
// With probability BLOCK_NEW_GROUP_PROB the node moves to a new group; otherwise
// s ~ P_e(s | i, b) (Eq. 14): a uniform neighbour j, t = b_j, then s with
// probability (e_ts + c) / (e_t + c B), c = BLOCK_SITE_EPS.  graph-tool defaults.
const double BLOCK_SITE_EPS = 0.5;
const double BLOCK_NEW_GROUP_PROB = 0.01;


const double DEPTH_PRIOR_WIDTH = 4.0; // scale of the half-Cauchy prior on individual_depth, group_depth and ties_parameter

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
// Default 1.0: Peixoto's omega_single = N, omega_collective = 1 (arXiv:2003.07070,
// Eq. 37 and the discussion after it).  With M = 3 Gibbs sweeps and the
// block-informed single-node move this gave the best partition mixing per unit
// time on the benchmark networks.
//
// A NEGATIVE value selects an automatic, state-independent weight:
//   * MCMC_sweep on its own uses Peixoto's linear-time condition
//         omega_collective = n / (M * (n + m)),
//     which keeps a sweep O(n + m) by lowering the collective weight rather than
//     raising omega_single (raising omega_single would also make the parameter
//     moves rarer relative to the node moves);
//   * fit() instead retunes it during burn-in with the equal-spend rule (see
//     MERGE_SPLIT_EQUAL_SPEND_C and MCMC::retune_collective_weight), never below
//     the linear-time value, and freezes it for sampling.
// With omega_collective = 1 the expected number of collective attempts per sweep
// is 3, independent of network size, while each costs O(M k) in the group size
// k; on an N = 1000 synthetic network that put 0.29% of the moves in charge of
// ~99% of the runtime.  A weight that depended on num_groups would make the
// partition-move fraction depend on q and bias the q-marginal, which is why the
// automatic weight is frozen once sampling starts.
extern double MERGE_SPLIT_WEIGHT;

// Divisor of the equal-spend rule fit() applies while MERGE_SPLIT_WEIGHT is
// negative: w = q / (C * M).  C folds in the staging overhead (each staged node
// is flipped and flipped back, and the split is scored both ways), measured at
// 1-2 per staged node, giving C ~ 9 for "collective moves cost about as much per
// sweep as the single-node moves".  See package-testing/asbm/merge_split_weight.
extern double MERGE_SPLIT_EQUAL_SPEND_C;

#endif // GLOBALS_H
