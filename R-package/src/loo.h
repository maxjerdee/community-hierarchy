// Leave-one-out cross-validation of the edge directions, shared by every binding.
//
// Only the outcome of an interaction is left out (who won, or that it was a draw);
// the interaction itself stays in the network.  Given the scores s and the draw
// parameter nu the direction likelihood is a product over interactions, and the
// network and partition terms do not depend on the outcomes, so the leave-one-out
// posterior is the full posterior reweighted by 1 / p(y_i | s, nu).  Pareto-smoothed
// importance sampling (PSIS; Vehtari, Gelman and Gabry 2017, Stat. Comput. 27:1413)
// stabilises those weights, as in ArviZ's loo / psislw and R's loo package.

#ifndef LOO_H
#define LOO_H

#include <vector>

#include "fit_result.h"

struct PsisResult {
    std::vector<double> log_weights;  // smoothed and normalised: logsumexp = 0
    double pareto_k;                  // shape of the fitted tail; inf if not fitted
};

// PSIS of one vector of log importance ratios (one per sample), ported from ArviZ
// 0.23 (arviz.stats.psislw): the largest ceil(min(S / 5, 3 sqrt(S / r_eff))) ratios
// are replaced by the quantiles of a generalised Pareto distribution fitted to them
// (Zhang and Stephens 2009, Technometrics 51:316, with the weakly informative prior
// on k of Vehtari et al.), and capped at the largest raw ratio.
PsisResult psis(std::vector<double> log_ratios, double r_eff = 1.0);

// One entry per interaction; a and b index nodes from 0.
struct DirectionLooResult {
    std::vector<double> elpd;      // leave-one-out log predictive density of the outcome
    std::vector<double> pareto_k;  // above 0.7 the estimate is unreliable
    std::vector<double> p_win;     // decided interactions: leave-one-out P(a beats b |
                                   // decided), plain Bradley-Terry; NaN for draws
};

// Interaction i is a win of a[i] over b[i] (draw[i] false) or a draw between them,
// counting weight[i] times (the log-likelihood is weight[i] log p, as in the
// sampler's direction term).  Leaving out an interaction of weight w leaves out all
// w of its outcomes together; pass one edge per match for leave-one-match-out.
DirectionLooResult direction_loo(const FitResult& result, const std::vector<int>& a,
                                 const std::vector<int>& b, const std::vector<bool>& draw,
                                 const std::vector<double>& weight, double r_eff = 1.0);

#endif  // LOO_H
