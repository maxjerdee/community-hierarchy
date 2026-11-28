#include "interactions.h"

#include <stdexcept>
#include <string>

namespace {

void check_shape(int n, const std::vector<int>& u, const std::vector<int>& v, std::size_t m) {
    if (u.size() != m || v.size() != m)
        throw std::invalid_argument("interactions: edge vectors of different lengths");
    for (std::size_t k = 0; k < m; ++k)
        if (u[k] < 0 || u[k] >= n || v[k] < 0 || v[k] >= n)
            throw std::invalid_argument("interactions: edge " + std::to_string(k) +
                                        " has an endpoint outside [0, " + std::to_string(n) + ")");
}

}  // namespace

Interactions build_interactions(int n, const std::vector<int>& from, const std::vector<int>& to,
                                const std::vector<bool>& draw, const std::vector<double>& weight,
                                bool source_high_rank) {
    const std::size_t m = weight.size();
    check_shape(n, from, to, m);
    if (draw.size() != m) throw std::invalid_argument("interactions: edge vectors of different lengths");
    Interactions r;
    r.wins.assign(static_cast<std::size_t>(n), {});
    r.ties.assign(static_cast<std::size_t>(n), {});
    for (std::size_t k = 0; k < m; ++k) {
        const int a = draw[k] || source_high_rank ? from[k] : to[k];
        const int b = draw[k] || source_high_rank ? to[k] : from[k];
        if (draw[k]) {
            r.ties[a][b] += weight[k];
            r.ties[b][a] += weight[k];
            r.has_neutral = true;
        } else {
            r.wins[a][b] += weight[k];
        }
        r.a.push_back(a);
        r.b.push_back(b);
        r.draw.push_back(draw[k]);
        r.weight.push_back(weight[k]);
    }
    return r;
}

AdjacencyList build_group_layer(int n, const std::vector<int>& a, const std::vector<int>& b,
                                const std::vector<double>& weight) {
    check_shape(n, a, b, weight.size());
    AdjacencyList adj(static_cast<std::size_t>(n));
    for (std::size_t k = 0; k < weight.size(); ++k) {
        adj[a[k]][b[k]] += weight[k];
        adj[b[k]][a[k]] += weight[k];
    }
    return adj;
}
