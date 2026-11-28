// The starting state of a chain, shared by every binding (Python, R, WebAssembly),
// so that a seeded fit is the same in every language.

#ifndef INITIAL_STATE_H
#define INITIAL_STATE_H

#include <random>
#include <vector>

#include "MCMC_core.h"
#include "fit.h"         // FitParams
#include "modularity.h"  // AdjacencyList; greedy_modularity_partition (the default partition)

// (wins - losses) / (wins + losses + draws + 1e-6) for each node, each counted over
// distinct opponents (not interactions).
std::vector<double> win_fraction_scores(const AdjacencyList& wins, const AdjacencyList& ties);

// Fill in the default starting state: greedy modularity partition (with num_groups
// groups when it is set) when initial_partition is empty, and win fractions when
// initial_scores is empty.  Given scores that are not fixed are used as given, and
// start sigma_I at their RMS unless individual_depth(_init) is set.
void complete_start(FitParams& p);

// A chain at the starting state of p (complete_start already applied): each
// continuous parameter is its fixed value, else its starting value, else a uniform
// draw from init_rng.  The order of the draws is part of the seeded result:
// variation_in, variation_out and degree_correction always consume a draw (even
// when fixed), individual_depth and group_depth only when they have neither a
// fixed nor a starting value, and ties_parameter always.
MCMC new_chain(const FitParams& p, std::mt19937_64& init_rng);

#endif  // INITIAL_STATE_H
