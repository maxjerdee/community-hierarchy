// The interactions of a fit, built from the edges of the input graph, shared by
// every binding.  The bindings check the edges (no self-loops; weights finite,
// positive, and whole numbers unless weighted_SBM; edge types) and pass them as
// parallel vectors, nodes numbered from 0 in the order of their graph's nodes.

#ifndef INTERACTIONS_H
#define INTERACTIONS_H

#include <vector>

#include "modularity.h"  // AdjacencyList

struct Interactions {
    AdjacencyList wins;  // wins[i][j]: total weight of i's wins over j
    AdjacencyList ties;  // ties[i][j] = ties[j][i]: total weight of their draws
    // One entry per input edge, in input order: a won against b, or (draw) they drew.
    std::vector<int> a, b;
    std::vector<bool> draw;
    std::vector<double> weight;
    bool has_neutral = false;
};

// Edge k runs from[k] -> to[k] with weight[k].  draw[k]: a draw (its direction is
// ignored); otherwise a win of from[k] if source_high_rank, else of to[k].
// Parallel edges add up.  Throws std::invalid_argument if the vectors differ in
// length or an endpoint is outside [0, num_nodes).
Interactions build_interactions(int num_nodes, const std::vector<int>& from,
                                const std::vector<int>& to, const std::vector<bool>& draw,
                                const std::vector<double>& weight, bool source_high_rank);

// The undirected group layer of two-layer mode: symmetric, parallel edges added up.
AdjacencyList build_group_layer(int num_nodes, const std::vector<int>& a, const std::vector<int>& b,
                                const std::vector<double>& weight);

#endif  // INTERACTIONS_H
