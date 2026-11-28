#include "modularity.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>

namespace {

// Sorted neighbour lists of the undirected simple network.
std::vector<std::vector<int>> simple_undirected(const AdjacencyList& wins,
                                                const AdjacencyList& ties,
                                                const AdjacencyList& group_adj) {
    const int n = static_cast<int>(wins.size());
    std::vector<std::set<int>> nb(static_cast<std::size_t>(n));
    auto add = [&](const AdjacencyList& adj) {
        for (int i = 0; i < static_cast<int>(adj.size()); ++i)
            for (const auto& kv : adj[static_cast<std::size_t>(i)]) {
                const int j = kv.first;
                if (j == i || j < 0 || j >= n) continue;
                nb[static_cast<std::size_t>(i)].insert(j);
                nb[static_cast<std::size_t>(j)].insert(i);
            }
    };
    if (!group_adj.empty()) {
        add(group_adj);
    } else {
        add(wins);
        add(ties);
    }
    std::vector<std::vector<int>> out(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        out[static_cast<std::size_t>(i)].assign(nb[static_cast<std::size_t>(i)].begin(),
                                                nb[static_cast<std::size_t>(i)].end());
    return out;
}

}  // namespace

std::vector<int> greedy_modularity_partition(const AdjacencyList& wins, const AdjacencyList& ties,
                                             const AdjacencyList& group_adj,
                                             std::optional<int> num_groups) {
    const int n = static_cast<int>(wins.size());
    if (n == 0) return {};
    if (num_groups && (*num_groups < 1 || *num_groups > n))
        throw std::invalid_argument("num_groups must be in [1, number of nodes]");
    const auto G = simple_undirected(wins, ties, group_adj);
    std::int64_t two_m = 0;
    for (const auto& row : G) two_m += static_cast<std::int64_t>(row.size());

    // Community c (id = its smallest node) has degree deg[c], size size[c], and
    // E[c][d] edges to community d.  The modularity gain of merging c and d is
    // 2 (E_cd / 2m - deg_c deg_d / 4m^2), proportional to  gain = 2m E_cd - deg_c deg_d.
    std::vector<std::int64_t> deg(static_cast<std::size_t>(n));
    std::vector<int> size(static_cast<std::size_t>(n), 1), comm(static_cast<std::size_t>(n));
    std::vector<std::map<int, std::int64_t>> E(static_cast<std::size_t>(n));
    std::iota(comm.begin(), comm.end(), 0);
    for (int i = 0; i < n; ++i) {
        deg[static_cast<std::size_t>(i)] =
            static_cast<std::int64_t>(G[static_cast<std::size_t>(i)].size());
        for (int j : G[static_cast<std::size_t>(i)]) E[static_cast<std::size_t>(i)][j] = 1;
    }
    // Candidate merges, best first: (-gain, c, d) with c < d.
    std::set<std::tuple<std::int64_t, int, int>> queue;
    auto gain = [&](int c, int d) {
        return two_m * E[static_cast<std::size_t>(c)].at(d) -
               deg[static_cast<std::size_t>(c)] * deg[static_cast<std::size_t>(d)];
    };
    auto key = [&](int c, int d) {
        return std::make_tuple(-gain(c, d), std::min(c, d), std::max(c, d));
    };
    for (int c = 0; c < n; ++c)
        for (const auto& kv : E[static_cast<std::size_t>(c)])
            if (c < kv.first) queue.insert(key(c, kv.first));

    std::set<int> alive;
    for (int c = 0; c < n; ++c) alive.insert(c);
    std::vector<std::vector<int>> members(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) members[static_cast<std::size_t>(i)] = {i};

    // Merge d into c (or c into d: the smaller id survives).
    auto merge = [&](int c, int d) {
        if (d < c) std::swap(c, d);
        for (const auto& kv : E[static_cast<std::size_t>(c)]) queue.erase(key(c, kv.first));
        for (const auto& kv : E[static_cast<std::size_t>(d)])
            if (kv.first != c) queue.erase(key(d, kv.first));
        for (const auto& kv : E[static_cast<std::size_t>(d)]) {
            const int k = kv.first;
            if (k == c) continue;
            E[static_cast<std::size_t>(c)][k] += kv.second;
            E[static_cast<std::size_t>(k)][c] += kv.second;
            E[static_cast<std::size_t>(k)].erase(d);
        }
        E[static_cast<std::size_t>(c)].erase(d);
        E[static_cast<std::size_t>(d)].clear();
        deg[static_cast<std::size_t>(c)] += deg[static_cast<std::size_t>(d)];
        size[static_cast<std::size_t>(c)] += size[static_cast<std::size_t>(d)];
        for (int v : members[static_cast<std::size_t>(d)]) comm[static_cast<std::size_t>(v)] = c;
        auto& mc = members[static_cast<std::size_t>(c)];
        mc.insert(mc.end(), members[static_cast<std::size_t>(d)].begin(),
                  members[static_cast<std::size_t>(d)].end());
        members[static_cast<std::size_t>(d)].clear();
        alive.erase(d);
        for (const auto& kv : E[static_cast<std::size_t>(c)]) queue.insert(key(c, kv.first));
    };

    const int target = num_groups.value_or(1);
    while (static_cast<int>(alive.size()) > target) {
        if (queue.empty()) {
            if (!num_groups) break;
            // No connected pair left: merge the two largest communities.
            std::vector<int> order(alive.begin(), alive.end());
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
                return size[static_cast<std::size_t>(a)] > size[static_cast<std::size_t>(b)];
            });
            merge(order[0], order[1]);
            continue;
        }
        const auto best = *queue.begin();
        if (-std::get<0>(best) < 0 && !num_groups) break;
        merge(std::get<1>(best), std::get<2>(best));
    }

    // Communities by size, largest first; ties by first node (= id).
    std::vector<int> order(alive.begin(), alive.end());
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return size[static_cast<std::size_t>(a)] > size[static_cast<std::size_t>(b)];
    });

    if (!num_groups) {
        // Merge satellites smaller than max(2, floor(n / 20)) into the community they
        // share the most edges with (ties: the earlier one in the order).
        const int min_size = std::max(2, n / 20);
        std::vector<int> pos(static_cast<std::size_t>(n), -1);
        bool changed = true;
        while (changed) {
            changed = false;
            for (int k = 0; k < static_cast<int>(order.size()); ++k)
                pos[static_cast<std::size_t>(order[static_cast<std::size_t>(k)])] = k;
            std::vector<bool> gone(order.size(), false);
            for (std::size_t k = 0; k < order.size(); ++k) {
                const int c = order[k];
                if (gone[k] || size[static_cast<std::size_t>(c)] >= min_size) continue;
                std::map<int, int> links;  // position of the other community -> edges
                for (int v : members[static_cast<std::size_t>(c)])
                    for (int u : G[static_cast<std::size_t>(v)]) {
                        const int other = comm[static_cast<std::size_t>(u)];
                        if (other != c) ++links[pos[static_cast<std::size_t>(other)]];
                    }
                if (links.empty()) continue;
                int best_pos = links.begin()->first, best_count = links.begin()->second;
                for (const auto& kv : links)
                    if (kv.second > best_count) {
                        best_pos = kv.first;
                        best_count = kv.second;
                    }
                const int target_c = order[static_cast<std::size_t>(best_pos)];
                for (int v : members[static_cast<std::size_t>(c)])
                    comm[static_cast<std::size_t>(v)] = target_c;
                auto& mt = members[static_cast<std::size_t>(target_c)];
                mt.insert(mt.end(), members[static_cast<std::size_t>(c)].begin(),
                          members[static_cast<std::size_t>(c)].end());
                size[static_cast<std::size_t>(target_c)] += size[static_cast<std::size_t>(c)];
                members[static_cast<std::size_t>(c)].clear();
                size[static_cast<std::size_t>(c)] = 0;
                gone[k] = true;
                changed = true;
            }
            std::vector<int> kept;
            for (std::size_t k = 0; k < order.size(); ++k)
                if (!gone[k]) kept.push_back(order[k]);
            order.swap(kept);
        }
    }

    std::vector<int> label(static_cast<std::size_t>(n), -1);
    for (int k = 0; k < static_cast<int>(order.size()); ++k)
        label[static_cast<std::size_t>(order[static_cast<std::size_t>(k)])] = k;
    std::vector<int> partition(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        partition[static_cast<std::size_t>(i)] =
            label[static_cast<std::size_t>(comm[static_cast<std::size_t>(i)])];
    return partition;
}
