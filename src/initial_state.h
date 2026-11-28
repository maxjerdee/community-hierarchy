// Default starting state of a fit, shared by every binding (Python, R, WebAssembly).
//
// fit() calls greedy_modularity_partition (modularity.h) when
// FitParams.initial_partition is empty and win_fraction_scores when initial_scores is
// empty, so a seeded fit is the same in every language.  Both are deterministic.

#ifndef INITIAL_STATE_H
#define INITIAL_STATE_H

#include <vector>

#include "modularity.h"  // AdjacencyList; greedy_modularity_partition (the default partition)

// (wins - losses) / (wins + losses + draws + 1e-6) for each node, each counted over
// distinct opponents (not interactions).
std::vector<double> win_fraction_scores(const AdjacencyList& wins, const AdjacencyList& ties);

#endif  // INITIAL_STATE_H
