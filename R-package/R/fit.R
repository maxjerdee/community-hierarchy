#' Fit the community and hierarchy model
#'
#' Samples the posterior of the joint model of communities and ranking for a
#' directed network by Markov chain Monte Carlo.
#'
#' @param G The interactions: a directed [igraph][igraph::igraph] graph, or a data
#'   frame of edges whose first two columns are the endpoints (as for
#'   [igraph::graph_from_data_frame()]).
#'   - Each edge is one interaction, or `weight` interactions; parallel edges are
#'     summed.  Without `weighted_SBM`, a weight must be a positive whole number.
#'   - An edge with `type = "neutral"` is a draw (a tie, or a reciprocated
#'     nomination), and its direction is ignored.  Any other edge is a decided
#'     interaction, read according to `source_high_rank`.  If edges have a `type`,
#'     every edge must be `"dominant"` or `"neutral"`.
#'   - Self-loops are not allowed.
#'   - The order of the vertices is the order of every per-node result.
#' @param config The model, from [ch_config()].
#' @param vertices With a data frame `G`: an optional data frame of vertices whose
#'   first column is the vertex name (it carries node attributes such as
#'   `group_attribute`, and adds isolated nodes).
#' @param group_graph Two-layer mode: an undirected graph on the same vertex names
#'   (an igraph graph or an edge data frame) whose edges drive the communities,
#'   while `G` drives only the scores.  Its weights must be finite and positive.
#' @param source_high_rank How to read a decided edge u -> v.  `TRUE` (default):
#'   the source wins (a sports result, a dominance interaction).  `FALSE`: the
#'   target wins (a citation of v by u, a friendship nomination of v by u).
#'   Either way, higher scores mean higher rank.
#' @param num_chains,num_samples,num_warmup,num_thin The run, named as in
#'   PyStan.  One iteration is `sweeps_per_sample` sweeps.  Each of `num_chains`
#'   independent chains (run in parallel threads) runs `num_warmup` warm-up
#'   iterations, during which the proposal step sizes adapt, then `num_samples`
#'   sampling iterations, recording the state after every `num_thin`-th:
#'   `ceiling(num_samples / num_thin)` draws per chain.
#' @param sweeps_per_sample Sweeps (one proposal per node each) per iteration.
#' @param num_param_warmup Iterations before warm-up with the partition and
#'   scores frozen, so that the continuous parameters settle first.
#' @param beta Inverse temperature of the likelihood (0 = prior, 1 = posterior).
#' @param num_tempering_chains Parallel-tempering replicas per chain; only the
#'   `beta` replica is recorded.
#' @param merge_split_enabled Merge-split moves for the partition (Peixoto 2020).
#'   Without them, single-node moves can converge to a wrong answer without any
#'   sign of it within a chain.
#' @param initial_partition Starting groups, one label per node (any labels).
#'   Default: greedy modularity on the undirected network, with groups smaller
#'   than 5\% of the nodes merged into their best-connected neighbour (computed
#'   by the C++ core, as in the 'Python' package).
#' @param initial_scores Starting scores, one per node.  Default: (wins - losses)
#'   / (wins + losses + draws) over distinct opponents.
#' @param initial_individual_depth,initial_group_depth Starting values of the
#'   inferred sigma_I (> 0) and sigma_G (>= 0).
#' @param num_groups_fixed Hold the number of groups at that of
#'   `initial_partition`.
#' @param seed Random seed in \[0, 2^30); chain c uses `seed + c`.  Default: drawn
#'   from R's random number generator, so [set.seed()] makes a fit repeatable.
#'   With the same seed, a fit is identical on every platform and to the
#'   'Python' package's (version 0.6.0 or later; earlier versions made their
#'   default start in 'Python').
#' @param timeout Wall-clock limit in seconds.  When it expires, the fit stops and
#'   returns the samples recorded so far, with a warning.
#' @param show_progress Print progress to the console.
#'
#' @return An object of class `ch_fit`.  See [network_properties()],
#'   [node_properties()], [consensus_partition()], [direction_probabilities()]
#'   and [samples()].
#' @examples
#' # Four players: a beats b and c, b beats c, c and d draw twice.
#' games <- data.frame(winner = c("a", "a", "b", "c"), loser = c("b", "c", "c", "d"),
#'                     type = c("dominant", "dominant", "dominant", "neutral"),
#'                     weight = c(1, 1, 1, 2))
#' fit <- ch_fit(games, num_chains = 1, num_samples = 50, num_warmup = 20,
#'               seed = 1, show_progress = FALSE)
#' network_properties(fit)
#' node_properties(fit)
#' @export
ch_fit <- function(G, config = ch_config(), vertices = NULL, group_graph = NULL,
                   source_high_rank = TRUE, num_chains = 1, num_samples = 1000,
                   num_warmup = 500, num_thin = 1, sweeps_per_sample = 10,
                   num_param_warmup = 0, beta = 1, num_tempering_chains = 1,
                   merge_split_enabled = TRUE, initial_partition = NULL,
                   initial_scores = NULL, initial_individual_depth = NULL,
                   initial_group_depth = NULL, num_groups_fixed = FALSE, seed = NULL,
                   timeout = 60, show_progress = interactive()) {
  if (!inherits(config, "ch_config")) stop("config must come from ch_config().", call. = FALSE)
  g <- as_directed_graph(G, vertices)
  n <- igraph::vcount(g)
  nodes <- node_labels(g)

  check_count <- function(x, name, min) {
    if (!(is_number(x) && x == round(x) && x >= min)) {
      stop(sprintf("%s must be a whole number >= %d.", name, min), call. = FALSE)
    }
    as.integer(x)
  }
  run <- list(
    num_chains = check_count(num_chains, "num_chains", 1),
    num_samples = check_count(num_samples, "num_samples", 1),
    num_warmup = check_count(num_warmup, "num_warmup", 0),
    num_thin = check_count(num_thin, "num_thin", 1),
    sweeps_per_sample = check_count(sweeps_per_sample, "sweeps_per_sample", 0),
    num_param_warmup = check_count(num_param_warmup, "num_param_warmup", 0),
    num_tempering_chains = check_count(num_tempering_chains, "num_tempering_chains", 1)
  )
  if (!(is_number(beta) && beta >= 0)) stop("beta must be a number >= 0.", call. = FALSE)
  if (!(is_number(timeout) && timeout > 0)) stop("timeout must be a positive number.", call. = FALSE)
  for (name in c("source_high_rank", "merge_split_enabled", "num_groups_fixed", "show_progress")) {
    if (!is_flag(get(name))) stop(name, " must be TRUE or FALSE.", call. = FALSE)
  }
  if (is.null(seed)) seed <- sample.int(1000000L, 1L) - 1L
  if (!(is_number(seed) && seed == round(seed) && seed >= 0 && seed < 2^30)) {
    stop("seed must be a whole number in [0, 2^30).", call. = FALSE)
  }
  for (name in c("initial_individual_depth", "initial_group_depth")) {
    x <- get(name)
    if (!is.null(x) && !(is_number(x) && (x > 0 || (x == 0 && name == "initial_group_depth")))) {
      stop(name, if (name == "initial_group_depth") " must be >= 0." else " must be > 0.",
           call. = FALSE)
    }
  }

  edges <- interactions(g, config$weighted_SBM)
  has_neutral <- any(edges$neutral)
  model <- resolve_config(config, has_neutral, n)
  hierarchy_model <- model$hierarchy_model
  model$hierarchy_model <- NULL

  group_edges <- NULL
  if (!is.null(group_graph)) group_edges <- group_layer(group_graph, nodes)

  partition_fixed <- !is.null(config$group_attribute)
  if (partition_fixed) {
    if (!is.null(initial_partition)) {
      warning("initial_partition is replaced by the group_attribute.", call. = FALSE)
    }
    initial_partition <- node_attribute(g, config$group_attribute)
  }
  # NULL: the core's default start (greedy modularity), as in every binding.
  if (!is.null(initial_partition)) {
    initial_partition <- relabel_partition(initial_partition, n, model$num_groups)
  }

  scores_fixed <- !is.null(config$hierarchy_attribute)
  if (scores_fixed) {
    if (!is.null(initial_scores)) {
      warning("initial_scores are replaced by the hierarchy_attribute.", call. = FALSE)
    }
    initial_scores <- node_attribute(g, config$hierarchy_attribute)
    if (!is.numeric(initial_scores) || any(!is.finite(initial_scores))) {
      stop("The hierarchy_attribute must be a finite number on every node.", call. = FALSE)
    }
  }
  # NULL: the core's default (win fractions).  Given scores that are not fixed start
  # sigma_I at their RMS (also in the core).
  if (!is.null(initial_scores) && (!is.numeric(initial_scores) || length(initial_scores) != n ||
                                   any(!is.finite(initial_scores)))) {
    stop("initial_scores must be ", n, " finite numbers, one per node.", call. = FALSE)
  }

  opts <- c(model, run, list(
    initial_individual_depth = initial_individual_depth, initial_group_depth = initial_group_depth,
    num_groups_fixed = num_groups_fixed, weighted_SBM = config$weighted_SBM,
    weighted_proposals = config$weighted_proposals, beta = as.numeric(beta),
    merge_split_enabled = merge_split_enabled, seed = as.integer(seed),
    timeout = as.numeric(timeout)))
  opts <- opts[!vapply(opts, is.null, logical(1))]
  # C++ reads absent optional entries as NULL (inferred).
  for (name in c("variation_in", "variation_out", "individual_depth", "group_depth",
                 "ties_parameter", "num_groups", "initial_individual_depth",
                 "initial_group_depth")) {
    if (!name %in% names(opts)) opts[name] <- list(NULL)
  }

  out <- .ch_fit_native(
    n, as.integer(edges$from - 1L), as.integer(edges$to - 1L), edges$neutral,
    as.numeric(edges$weight), source_high_rank,
    as.integer(if (is.null(group_edges)) integer() else group_edges$a - 1L),
    as.integer(if (is.null(group_edges)) integer() else group_edges$b - 1L),
    as.numeric(if (is.null(group_edges)) numeric() else group_edges$weight),
    !is.null(group_edges), as.integer(initial_partition), partition_fixed,
    as.numeric(initial_scores), scores_fixed, opts, show_progress)
  native <- new.env(parent = emptyenv())
  native$handle <- out$native

  expected <- run$num_chains * ceiling(run$num_samples / run$num_thin)
  recorded <- nrow(out$partitions)
  if (recorded == 0) {
    warning(sprintf(paste0(
      "The %gs timeout expired during burn-in, before a single sample was recorded. ",
      "The result is NOT a fit. Pass a larger timeout or reduce num_warmup / ",
      "num_samples."), timeout), call. = FALSE)
  } else if (recorded < expected) {
    warning(sprintf(paste0(
      "The %gs timeout expired after %d of %d samples. The result is based only on ",
      "those, so it is noisier than requested and may not have converged. Pass a ",
      "larger timeout to collect the full run."), timeout, recorded, expected), call. = FALSE)
  }
  colnames(out$partitions) <- colnames(out$scores) <- as.character(nodes)
  structure(list(nodes = nodes, params = as.data.frame(out$params),
                 partitions = out$partitions, scores = out$scores, config = config,
                 hierarchy_model = hierarchy_model, seed = as.integer(seed),
                 settings = run, version = as.character(utils::packageVersion("communityhierarchy")),
                 # The edges of G as the core read them, for direction_loo(): a won
                 # against b, or they drew; nodes numbered from 1.
                 interactions = as.data.frame(out$interactions),
                 native = native),
            class = "ch_fit")
}

# --- input ----------------------------------------------------------------------

as_directed_graph <- function(G, vertices = NULL) {
  if (inherits(G, "igraph")) {
    if (!is.null(vertices)) stop("vertices is only used with a data frame G.", call. = FALSE)
    if (!igraph::is_directed(G)) {
      stop("G must be a directed graph. Mark draws with type = \"neutral\".", call. = FALSE)
    }
    g <- G
  } else if (is.data.frame(G)) {
    if (ncol(G) < 2) stop("A data frame G needs two columns of endpoints.", call. = FALSE)
    g <- igraph::graph_from_data_frame(G, directed = TRUE, vertices = vertices)
  } else {
    stop("G must be an igraph graph or a data frame of edges.", call. = FALSE)
  }
  if (igraph::vcount(g) < 2) stop("G must have at least two nodes.", call. = FALSE)
  if (igraph::ecount(g) == 0) stop("G has no edges.", call. = FALSE)
  g
}

node_labels <- function(g) {
  nm <- igraph::V(g)$name
  if (is.null(nm)) seq_len(igraph::vcount(g)) else nm
}

check_weights <- function(w, integer, what) {
  if (!is.numeric(w) || any(!is.finite(w)) || any(w <= 0)) {
    stop(what, " weights must be finite and positive.", call. = FALSE)
  }
  if (integer && any(w != round(w))) {
    stop("Without weighted_SBM a weight is a number of interactions and must be a whole ",
         "number. Use ch_config(weighted_SBM = TRUE) for real-valued weights.", call. = FALSE)
  }
}

# Edges of G as 1-based endpoints, a neutral flag and weights.
interactions <- function(g, weighted) {
  el <- igraph::as_edgelist(g, names = FALSE)
  if (any(el[, 1] == el[, 2])) {
    stop("G has a self-loop; a node cannot interact with itself.", call. = FALSE)
  }
  type <- igraph::edge_attr(g, "type")
  if (is.null(type)) {
    neutral <- rep(FALSE, nrow(el))
  } else {
    type <- as.character(type)
    if (anyNA(type) || !all(type %in% c("dominant", "neutral"))) {
      stop("Every edge type must be \"dominant\" or \"neutral\".", call. = FALSE)
    }
    neutral <- type == "neutral"
  }
  w <- igraph::edge_attr(g, "weight")
  if (is.null(w)) {
    w <- rep(1, nrow(el))
  } else {
    check_weights(w, !weighted, "Edge")
  }
  list(from = el[, 1], to = el[, 2], neutral = neutral, weight = as.numeric(w))
}

group_layer <- function(group_graph, nodes) {
  if (is.data.frame(group_graph)) {
    group_graph <- igraph::graph_from_data_frame(group_graph, directed = FALSE)
  }
  if (!inherits(group_graph, "igraph")) {
    stop("group_graph must be an igraph graph or a data frame of edges.", call. = FALSE)
  }
  if (igraph::is_directed(group_graph)) stop("group_graph must be undirected.", call. = FALSE)
  gn <- node_labels(group_graph)
  if (length(gn) != length(nodes) || !setequal(as.character(gn), as.character(nodes))) {
    stop("group_graph must have the same vertices as G.", call. = FALSE)
  }
  el <- igraph::as_edgelist(group_graph, names = FALSE)
  if (any(el[, 1] == el[, 2])) stop("group_graph has a self-loop.", call. = FALSE)
  w <- igraph::edge_attr(group_graph, "weight")
  if (is.null(w)) w <- rep(1, nrow(el)) else check_weights(w, FALSE, "group_graph")
  idx <- match(as.character(gn), as.character(nodes))
  list(a = idx[el[, 1]], b = idx[el[, 2]], weight = as.numeric(w))
}

node_attribute <- function(g, name) {
  x <- igraph::vertex_attr(g, name)
  if (is.null(x)) stop("No node attribute \"", name, "\" in G.", call. = FALSE)
  if (anyNA(x)) stop("The node attribute \"", name, "\" is missing on some nodes.", call. = FALSE)
  x
}

# Any labels -> 0..K-1, in sorted order of the labels.
relabel_partition <- function(p, n, num_groups) {
  if (length(p) != n || anyNA(p)) {
    stop("A partition must give a group to each of the ", n, " nodes.", call. = FALSE)
  }
  out <- match(p, sort(unique(p))) - 1L
  if (!is.null(num_groups) && max(out) + 1L != num_groups) {
    stop("The partition has ", max(out) + 1L, " groups, but num_groups = ", num_groups, ".",
         call. = FALSE)
  }
  out
}
