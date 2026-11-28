// MCMC convergence diagnostics, shared by every binding: the rank-normalised
// effective sample sizes and R-hat of Vehtari, Gelman, Simpson, Carpenter and
// Buerkner (2021, Bayesian Analysis 16:667), ported from ArviZ 0.23
// (arviz.stats.diagnostics: ess(method="bulk" / "tail"), rhat(method="rank")).

#ifndef DIAGNOSTICS_H
#define DIAGNOSTICS_H

#include <string>
#include <vector>

#include "fit_result.h"

// The draws of one quantity, one vector per chain, all of the same length.
using Chains = std::vector<std::vector<double>>;

// Each chain is split in half, the pooled halves are rank-normalised, and the
// effective sample size is Geyer's initial-monotone-sequence estimate over all the
// halves.  NaN with fewer than 4 draws per chain; the number of draws if they are
// all equal.
double ess_bulk(const Chains& chains);
// The smaller of the effective sample sizes of the indicators x <= q05 and
// x <= q95, q the pooled 5% and 95% quantiles: how well the tails are estimated.
double ess_tail(const Chains& chains);
// The larger of the split R-hat of the rank-normalised draws and of their distance
// from the median.  NaN with fewer than 2 chains or 4 draws per chain, or when
// the draws are all equal.
double rhat(const Chains& chains);

// One quantity's draws from a fit, by chain (samples are stored chain by chain),
// each chain truncated to the length of the shortest.
Chains chains_of(const FitResult& result, double (*get)(const SampleState&));

struct DiagnosticsResult {
    std::vector<std::string> quantity;
    std::vector<double> ess_bulk, ess_tail, rhat;
};

// The network-level parameters (network_parameters(), fit_result.h) and the
// description length of the partition, H_A_b.
DiagnosticsResult network_diagnostics(const FitResult& result);

// One entry per node (quantity is empty).  relative: of each score minus the mean
// score of its sample.  The mean level of the scores is pinned only by their prior
// and wanders slowly; the differences, which every prediction uses, mix faster.
DiagnosticsResult score_diagnostics(const FitResult& result, bool relative);

#endif  // DIAGNOSTICS_H
