// Greedy modularity optimisation, the default starting partition of a fit (see
// initial_state.h), shared by every binding.

#ifndef MODULARITY_H
#define MODULARITY_H

#include <optional>
#include <unordered_map>
#include <vector>

using AdjacencyList = std::vector<std::unordered_map<int, double>>;

// Greedy modularity partition (Clauset, Newman and Moore 2004, PRE 70:066111) of the
// undirected simple network underlying the interactions: the union of the edges of
// `wins` (directed, adj[i][j]) and `ties` (undirected), or of `group_adj` when it
// is non-empty (two-layer mode).  Weights and multiplicities are ignored.
//
// Communities start as single nodes; the pair with the largest modularity gain is
// merged while the gain is >= 0.  The gains are compared exactly (in integers), and
// equal gains go to the pair with the smaller ids.  With num_groups, merging
// continues past a negative gain until num_groups communities remain (when no
// connected pair is left, the two largest communities are merged), as networkx's
// greedy_modularity_communities(cutoff=k, best_n=k).  Without num_groups,
// communities smaller than max(2, floor(n / 20)) nodes are then merged into the
// community they share the most edges with (pass after pass, until none moves),
// because greedy modularity leaves many small satellites around the large groups.
//
// Labels 0..K-1 number the communities by size, largest first (ties: the community
// with the smaller first node); merged satellites keep their target's label.
std::vector<int> greedy_modularity_partition(const AdjacencyList& wins,
                                             const AdjacencyList& ties,
                                             const AdjacencyList& group_adj,
                                             std::optional<int> num_groups);

#endif  // MODULARITY_H
