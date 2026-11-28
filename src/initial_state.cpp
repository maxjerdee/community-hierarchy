#include "initial_state.h"

#include <algorithm>
#include <cmath>

#include "rng.h"

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

void complete_start(FitParams& p) {
    if (p.initial_partition.empty())
        p.initial_partition =
            greedy_modularity_partition(p.wins, p.ties, p.group_adj, p.num_groups);
    if (p.initial_scores.empty()) {
        p.initial_scores = win_fraction_scores(p.wins, p.ties);
    } else if (!p.scores_fixed && !p.individual_depth && !p.individual_depth_init) {
        double ss = 0.0;
        for (double x : p.initial_scores) ss += x * x;
        const double rms = std::sqrt(ss / static_cast<double>(p.initial_scores.size()));
        if (rms > 1e-6) p.individual_depth_init = rms;
    }
}

MCMC new_chain(const FitParams& p, std::mt19937_64& init_rng) {
    const double vi = p.variation_in.value_or(next_uniform(init_rng));
    const double vo = p.variation_out.value_or(next_uniform(init_rng));
    const double dc = p.degree_correction.value_or(next_uniform(init_rng));
    const double id = p.individual_depth        ? *p.individual_depth
                      : p.individual_depth_init ? *p.individual_depth_init
                                                : next_uniform(init_rng);
    const double gd = p.group_depth        ? *p.group_depth
                      : p.group_depth_init ? *p.group_depth_init
                                           : next_uniform(init_rng);
    const double tp = p.ties_parameter.value_or(next_uniform(init_rng));
    const int q = *std::max_element(p.initial_partition.begin(), p.initial_partition.end()) + 1;

    MCMC chain(
        p.wins, p.ties, p.group_adj, p.initial_partition, p.partition_fixed, p.initial_scores,
        p.scores_fixed, p.num_groups.value_or(q), p.num_groups.has_value() || p.num_groups_fixed,
        p.assortative, vi, p.variation_in.has_value(), vo, p.variation_out.has_value(), dc,
        p.degree_correction.has_value(), id, p.individual_depth.has_value(), gd,
        p.group_depth.has_value(), tp, p.ties_parameter.has_value(),
        /*seed=*/-1, /*density_in_initial=*/-1.0, /*density_out_initial=*/-1.0, p.weighted_SBM);
    chain.weighted_proposals = p.weighted_SBM && p.weighted_proposals;
    return chain;
}
