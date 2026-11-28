#include "initial_state.h"

std::vector<double> win_fraction_scores(const AdjacencyList& wins, const AdjacencyList& ties) {
    const int n = static_cast<int>(wins.size());
    std::vector<int> w(static_cast<std::size_t>(n), 0), l(static_cast<std::size_t>(n), 0),
        t(static_cast<std::size_t>(n), 0);
    for (int i = 0; i < n; ++i)
        for (const auto& kv : wins[static_cast<std::size_t>(i)]) {
            if (kv.first == i) continue;
            ++w[static_cast<std::size_t>(i)];
            ++l[static_cast<std::size_t>(kv.first)];
        }
    for (int i = 0; i < static_cast<int>(ties.size()) && i < n; ++i)
        for (const auto& kv : ties[static_cast<std::size_t>(i)])
            if (kv.first != i) ++t[static_cast<std::size_t>(i)];
    std::vector<double> s(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const double wi = w[static_cast<std::size_t>(i)], li = l[static_cast<std::size_t>(i)];
        s[static_cast<std::size_t>(i)] =
            (wi - li) / (wi + li + t[static_cast<std::size_t>(i)] + 1e-6);
    }
    return s;
}
