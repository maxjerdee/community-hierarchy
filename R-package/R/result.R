check_fit <- function(fit) {
  if (!inherits(fit, "ch_fit")) stop("fit must come from ch_fit().", call. = FALSE)
  if (nrow(fit$partitions) == 0) stop("The fit has no samples.", call. = FALSE)
}

# The C++ FitResult of a fit: the handle made by ch_fit(), or, after saveRDS() /
# readRDS() (which do not keep handles), one rebuilt once from the samples.
native <- function(fit) {
  env <- fit$native
  if (!is.environment(env)) {
    env <- new.env(parent = emptyenv())
  }
  if (!.ch_handle_valid(env$handle)) {
    env$handle <- .ch_build_result(fit$params, fit$partitions, fit$scores)
  }
  env$handle
}

# 0-based groups from the core -> 1-based groups in R.
groups_r <- function(g) as.integer(g) + 1L

#' Network-level posterior summaries
#'
#' @param fit A fit from [ch_fit()].
#' @param prob Probability mass of the highest-density interval, in (0, 1).
#' @return A data frame with columns `parameter`, `mean`, `std`, `hdi_lo` and
#'   `hdi_hi`, one row for each of `num_groups`, `density_in`, `density_out`,
#'   `variation_in`, `variation_out`, `degree_correction`, `individual_depth`
#'   (sigma_I), `group_depth` (sigma_G), `ties_parameter` (nu), `sigma_T` =
#'   sqrt(sigma_G^2 + sigma_I^2) and `ICC` = sigma_G^2 / sigma_T^2, the share of
#'   the variance in scores that the groups explain.  The interval is the
#'   shortest one holding `floor(prob * n_samples)` samples.
#' @export
network_properties <- function(fit, prob = 0.90) {
  check_fit(fit)
  if (!(is_number(prob) && prob > 0 && prob < 1)) stop("prob must be in (0, 1).", call. = FALSE)
  as.data.frame(.ch_network_properties(native(fit), prob))
}

#' Node-level posterior summaries
#'
#' @inheritParams network_properties
#' @param partition `"consensus"` (default) for the groups of
#'   [consensus_partition()], or `"mdl"` for those of the sample with the
#'   shortest description length.
#' @param consensus_method Passed to [consensus_partition()].
#' @return A data frame with one row per node, in the order of the vertices of
#'   `G`: `node`, `score_mean`, `score_std`, `group` (numbered from 1, the group
#'   with the lowest mean score first) and `group_score_mean` (the mean of
#'   `score_mean` over the node's group).
#' @export
node_properties <- function(fit, partition = c("consensus", "mdl"),
                            consensus_method = c("moc", "moc_modal_k", "binder")) {
  check_fit(fit)
  partition <- match.arg(partition)
  consensus_method <- match.arg(consensus_method)
  d <- .ch_node_properties(native(fit), partition, consensus_method)
  data.frame(node = fit$nodes, score_mean = d$score_mean, score_std = d$score_std,
             group = groups_r(d$group), group_score_mean = d$group_score_mean)
}

#' Point estimate of the partition
#'
#' @inheritParams network_properties
#' @param method
#'   - `"moc"` (default): the maximum-overlap consensus (Peixoto 2021), the
#'     partition, with any number of groups, that agrees with the most node labels
#'     across the samples after each sample is optimally relabelled.
#'   - `"moc_modal_k"`: the same, with the number of groups held at its posterior
#'     mode.
#'   - `"binder"`: the sample, among those with the modal number of groups, whose
#'     co-membership is closest to the posterior co-membership matrix (Binder
#'     loss), then refined node by node.  Its cost grows with the square of the
#'     number of nodes.
#' @param refine `"binder"` only: `FALSE` returns the best sample unrefined.
#' @param min_samples The samples are thinned, evenly spaced, to the larger of
#'   `min_samples` and the effective sample size of the description length
#'   ([partition_ess()]); 0 turns thinning off.
#' @return An integer vector of groups, named by node, numbered from 1 with the
#'   group with the lowest mean score first.
#' @export
consensus_partition <- function(fit, method = c("moc", "moc_modal_k", "binder"),
                                refine = TRUE, min_samples = 1000) {
  check_fit(fit)
  method <- match.arg(method)
  if (!is_flag(refine)) stop("refine must be TRUE or FALSE.", call. = FALSE)
  if (!(is_number(min_samples) && min_samples >= 0)) {
    stop("min_samples must be a number >= 0.", call. = FALSE)
  }
  p <- .ch_consensus_partition(native(fit), method, refine,
                               as.integer(min_samples))
  stats::setNames(groups_r(p), as.character(fit$nodes))
}

#' The sample with the shortest description length
#'
#' The description length of a sample is `H_A_b + H_Adir_G_A_s_nu +
#' H_s_G_b_depths` (see [samples()]): minus the log joint probability of the
#' network, the partition and the scores, given the continuous parameters.  It is
#' a by-product of sampling, not a posterior summary.
#'
#' @inheritParams network_properties
#' @return A list with `value` (in nats) and `partition` (an integer vector named
#'   by node, numbered as in [consensus_partition()]).
#' @export
mdl_partition <- function(fit) {
  check_fit(fit)
  d <- .ch_mdl(native(fit))
  list(value = d$value, partition = stats::setNames(groups_r(d$partition), as.character(fit$nodes)))
}

#' Effective sample size of the partition
#'
#' The rank-normalised split-chain bulk effective sample size (Vehtari et al.
#' 2021) of the description length `H_A_b`, pooled over chains.
#'
#' @inheritParams network_properties
#' @return A number.
#' @export
partition_ess <- function(fit) {
  check_fit(fit)
  .ch_partition_ess(native(fit))
}

#' Co-membership matrix
#'
#' @inheritParams network_properties
#' @return An n x n matrix, with rows and columns named by node: the fraction of
#'   samples in which two nodes are in the same group.
#' @export
coincidence_matrix <- function(fit) {
  check_fit(fit)
  m <- .ch_coincidence_matrix(native(fit))
  dimnames(m) <- list(as.character(fit$nodes), as.character(fit$nodes))
  m
}

#' Predicted outcome of an interaction
#'
#' The posterior predictive probabilities of the outcome of a new interaction
#' between two nodes, averaged over the samples.
#'
#' @inheritParams network_properties
#' @param a,b Two nodes, as named in `G`.
#' @param ties_allowed `TRUE` (default): include a draw, with the sampled
#'   `ties_parameter`.  `FALSE`: plain Bradley-Terry, given that the interaction
#'   is decided.
#' @return A named vector: `a_wins`, `tie` (only with `ties_allowed = TRUE`) and
#'   `b_wins`, summing to 1.  "Wins" is the win of the model, so with
#'   `source_high_rank = FALSE` (for example citations) `a_wins` is the
#'   probability of an edge from `b` to `a`.
#' @export
direction_probabilities <- function(fit, a, b, ties_allowed = TRUE) {
  check_fit(fit)
  idx <- match(as.character(c(a, b)), as.character(fit$nodes))
  if (length(idx) != 2 || anyNA(idx)) stop("a and b must be nodes of the fitted graph.", call. = FALSE)
  if (!is_flag(ties_allowed)) stop("ties_allowed must be TRUE or FALSE.", call. = FALSE)
  p <- .ch_direction_probabilities(native(fit), idx[1] - 1L,
                                   idx[2] - 1L, ties_allowed)
  stats::setNames(p, if (ties_allowed) c("a_wins", "tie", "b_wins") else c("a_wins", "b_wins"))
}

#' Leave-one-out cross-validation of the edge directions
#'
#' Each edge of `G` in turn has its outcome left out (who won, or that it was a
#' draw); the edge itself stays in the network.  The scores and the draw parameter
#' make the outcomes independent, so the leave-one-out posterior is the posterior
#' reweighted by 1 / p(outcome), with the weights Pareto-smoothed (PSIS-LOO,
#' Vehtari, Gelman and Gabry 2017) as in the `loo` package and ArviZ.  The
#' computation is the C++ core's, shared with the Python package.
#'
#' An edge of weight w counts w times and is left out whole: give one edge per
#' interaction (a multigraph) for leave-one-interaction-out.
#'
#' @inheritParams network_properties
#' @param r_eff Relative efficiency of the samples, which sets the length of the
#'   smoothed tail.  1 (default) treats the samples as independent.
#' @return A data frame with one row per edge of `G`, in edge order:
#'   - `winner`, `loser`: the edge's ends, the winner first (for a draw, as in `G`);
#'   - `draw`, `weight`;
#'   - `elpd`: the leave-one-out log predictive density of the outcome
#'     (`sum(elpd)` is the model's total);
#'   - `pareto_k`: above 0.7 that edge's estimate is unreliable;
#'   - `p_win`: decided edges, the leave-one-out probability that the winner beats
#'     the loser given that the interaction is decided (plain Bradley-Terry); `NA`
#'     for draws.
#'
#'   Two models fitted to the same graph compare edge by edge: the difference of
#'   the `elpd` sums, with standard error `sqrt(n) * sd(elpd1 - elpd2)`.
#' @export
direction_loo <- function(fit, r_eff = 1) {
  check_fit(fit)
  if (is.null(fit$interactions)) {
    stop("This fit has no edges recorded (made by an older version); refit to cross-validate.",
         call. = FALSE)
  }
  if (!(is_number(r_eff) && r_eff > 0)) stop("r_eff must be a positive number.", call. = FALSE)
  e <- fit$interactions
  r <- .ch_direction_loo(native(fit), as.integer(e$a - 1L), as.integer(e$b - 1L), e$draw,
                         as.numeric(e$weight), r_eff)
  p_win <- r$p_win
  p_win[e$draw] <- NA_real_
  data.frame(winner = fit$nodes[e$a], loser = fit$nodes[e$b], draw = e$draw, weight = e$weight,
             elpd = r$elpd, pareto_k = r$pareto_k, p_win = p_win)
}

#' The recorded samples
#'
#' @inheritParams network_properties
#' @param what `"parameters"` (default): a data frame with one row per sample and
#'   the columns
#'   - `chain_id` (from 1) and `beta`, the inverse temperature of the replica;
#'   - the parameters `num_groups`, `density_in`, `density_out`, `variation_in`,
#'     `variation_out`, `degree_correction`, `individual_depth`, `group_depth`,
#'     `ties_parameter`, and the derived `sigma_T` and `ICC`;
#'   - the description-length terms in nats: `H_b` (partition prior), `H_M`,
#'     `H_k`, `H_A_G_k_M_b`, `H_W` (`weighted_SBM` only, else 0), `H_A_G_b`
#'     (undirected network given the partition), `H_A_b` = `H_A_G_b + H_b`,
#'     `H_Adir_G_A_s_nu` (edge directions given the scores) and `H_s_G_b_depths`
#'     (scores given the partition).
#'
#'   `"scores"` or `"groups"`: a samples x nodes matrix of scores or of group
#'   labels (the sampler's own labels, from 1, not aligned across samples).
#' @return A data frame or a matrix.
#' @export
samples <- function(fit, what = c("parameters", "scores", "groups")) {
  if (!inherits(fit, "ch_fit")) stop("fit must come from ch_fit().", call. = FALSE)
  what <- match.arg(what)
  if (what == "scores") return(fit$scores)
  if (what == "groups") return(fit$partitions + 1L)
  d <- fit$params
  d$chain_id <- d$chain_id + 1L
  d
}

#' MCMC convergence diagnostics
#'
#' The rank-normalised effective sample sizes and R-hat of Vehtari et al. (2021),
#' as ArviZ's `ess(method = "bulk")`, `ess(method = "tail")` and `rhat()` and the
#' `posterior` package's `ess_bulk()`, `ess_tail()` and `rhat()`, computed in the
#' C++ core and shared with the Python package.
#'
#' @inheritParams network_properties
#' @param scores `FALSE` (default): the network-level parameters.  `"absolute"`
#'   or `"relative"`: each node's score, or each score minus the mean score of its
#'   sample.  The overall level of the scores is pinned only by their prior and
#'   wanders slowly; the differences, which every prediction uses, mix faster.
#' @return A data frame with columns `ess_bulk`, `ess_tail` and `rhat`, and
#'   `quantity` (the rows of [network_properties()], then `H_A_b`, the description
#'   length of the partition, as in [partition_ess()]) or `node`.  Chains are
#'   truncated to the shortest; R-hat needs two chains (`NaN` otherwise), and a
#'   fixed parameter has `NaN` R-hat.
#' @export
diagnostics <- function(fit, scores = FALSE) {
  check_fit(fit)
  if (isFALSE(scores)) return(as.data.frame(.ch_diagnostics(native(fit))))
  scores <- match.arg(scores, c("absolute", "relative"))
  d <- as.data.frame(.ch_score_diagnostics(native(fit), scores == "relative"))
  data.frame(node = fit$nodes, d)
}

#' @export
print.ch_fit <- function(x, ...) {
  n_samples <- nrow(x$partitions)
  cat(sprintf("<ch_fit> %d nodes, %d samples from %d chain(s), seed %d\n",
              length(x$nodes), n_samples, x$settings$num_chains, x$seed))
  if (n_samples > 0) {
    np <- network_properties(x)
    rows <- np[np$parameter %in% c("num_groups", "individual_depth", "group_depth",
                                   "sigma_T", "ICC", "ties_parameter"), ]
    rownames(rows) <- NULL
    print(rows, digits = 3)
  }
  invisible(x)
}

#' @export
summary.ch_fit <- function(object, prob = 0.90, ...) network_properties(object, prob)
