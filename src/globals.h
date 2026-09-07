/* Global variable declarations */

#ifndef GLOBALS_H
#define GLOBALS_H

#include <random>
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

// Timing variables
extern bool timing; // Whether to track detailed timing information
extern std::chrono::high_resolution_clock::time_point start_time, end_time;
extern double duration; // Accumulated or temporary duration (seconds)

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
const double GROUP_SCORE_STEP_SIZE = 0.1; // RW step (scaled-score units) for the collective group-score-shift move


const int GIBBS_REFINEMENT_SWEEPS = 1; // Change back to 10 // Number of Gibbs sweeps to use when refining splits

const double DEPTH_PRIOR_WIDTH = 4.0; // Keeping with the depth paper, scale of half-Cauchy prior on depth parameters

extern std::random_device rd;
extern std::mt19937 rng; // Initialize Mersenne Twister (global RNG)
extern std::uniform_real_distribution<double> uniform_random; // Uniform distribution on [0,1]


// ---- merge/split tuning (auxiliary: cannot affect correctness) ----
// Number of Gibbs sweeps used to refine a split during STAGING.  This is the
// auxiliary variable, so ANY value (including 0) gives a valid kernel; it trades
// proposal quality against cost.  Peixoto reports M=10 across several networks;
// on small easy networks M=0-1 is faster, but on celegansneural (N=297) M=0
// FAILS to converge, so 10 is the safe default.
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

#endif // GLOBALS_H
