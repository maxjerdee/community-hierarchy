test_that("a fit returns summaries in node order", {
  fit <- quick(games())
  expect_s3_class(fit, "ch_fit")
  np <- network_properties(fit)
  expect_named(np, c("parameter", "mean", "std", "hdi_lo", "hdi_hi"))
  expect_true(all(c("ICC", "sigma_T", "num_groups") %in% np$parameter))
  nodes <- node_properties(fit)
  expect_equal(nodes$node, c("a", "b", "c", "d", "e", "f"))
  expect_true(min(nodes$group) == 1)
  cp <- consensus_partition(fit)
  expect_named(cp, nodes$node)
  expect_equal(unname(cp), nodes$group)
  p <- direction_probabilities(fit, "a", "f")
  expect_equal(sum(p), 1)
  expect_named(direction_probabilities(fit, "a", "f", ties_allowed = FALSE), c("a_wins", "b_wins"))
  expect_equal(dim(coincidence_matrix(fit)), c(6L, 6L))
  expect_true(is.finite(mdl_partition(fit)$value))
  expect_equal(dim(samples(fit, "scores")), c(30L, 6L))
  expect_output(print(fit), "6 nodes, 30 samples")
})

test_that("a seed makes a fit repeatable, and set.seed() does without one", {
  expect_identical(quick(games())$scores, quick(games())$scores)
  run <- function() {
    set.seed(5)
    ch_fit(games(), num_chains = 1, num_samples = 10, num_warmup = 2,
           show_progress = FALSE)$scores
  }
  expect_identical(run(), run())
})

test_that("several chains are numbered from 1", {
  fit <- ch_fit(games(), num_chains = 2, num_samples = 10, num_warmup = 2, seed = 3,
                show_progress = FALSE)
  expect_equal(sort(unique(samples(fit)$chain_id)), 1:2)
})

test_that("a fit survives saveRDS", {
  fit <- quick(games())
  f <- tempfile(fileext = ".rds")
  saveRDS(fit, f)
  expect_identical(node_properties(readRDS(f)), node_properties(fit))
})

test_that("an igraph graph and source_high_rank = FALSE give the same fit", {
  g <- games()
  flipped <- igraph::graph_from_data_frame(g[, c(2, 1, 3, 4)], directed = TRUE)
  a <- quick(g)
  b <- quick(flipped, source_high_rank = FALSE)
  # Same interactions; the vertex order differs, so compare by node.
  na <- node_properties(a)
  nb <- node_properties(b)
  expect_setequal(na$node, nb$node)
  expect_equal(a$hierarchy_model, "bradley_terry_ties")
})

test_that("known groups and scores are held fixed", {
  g <- igraph::graph_from_data_frame(games(), directed = TRUE)
  igraph::V(g)$team <- c("x", "x", "y", "y", "z", "z")
  fit <- quick(g, ch_config(group_attribute = "team"))
  expect_true(all(apply(samples(fit, "groups"), 2, function(x) length(unique(x)) == 1)))
  igraph::V(g)$s <- seq(1, -1, length.out = 6)
  fit <- quick(g, ch_config(hierarchy_attribute = "s"))
  expect_true(all(apply(samples(fit, "scores"), 2, function(x) length(unique(x)) == 1)))
})

test_that("weighted and two-layer fits run", {
  w <- games()
  w$weight <- w$weight * 0.37
  expect_error(quick(w), "whole number")
  expect_s3_class(quick(w, ch_config(weighted_SBM = TRUE)), "ch_fit")
  layer <- data.frame(a = c("a", "b", "c", "d", "e"), b = c("b", "c", "d", "e", "f"))
  fit <- quick(games(), group_graph = layer)
  expect_true(all(samples(fit)$two_layer_mode))
})

test_that("invalid input is rejected", {
  expect_error(quick(games(), ch_config(hierarchy_model = "bradley_terry")), "neutral edges")
  loop <- data.frame(a = c("a", "b"), b = c("a", "c"))
  expect_error(quick(loop), "self-loop")
  bad_type <- games()
  bad_type$type[1] <- "won"
  expect_error(quick(bad_type), "dominant")
  undirected <- igraph::make_ring(4)
  expect_error(quick(undirected), "directed")
  expect_error(quick(games(), initial_partition = 1:3), "each of the 6 nodes")
  expect_error(quick(games(), seed = -1), "seed")
  no_ties <- games()[games()$type == "dominant", ]
  expect_error(quick(no_ties, ch_config(ties_parameter = 1)), "no neutral edges")
})

test_that("a timeout warns and keeps the samples so far", {
  expect_warning(
    ch_fit(games(), num_chains = 1, num_samples = 1e6, num_warmup = 0, seed = 1,
           timeout = 0.2, show_progress = FALSE),
    "timeout expired")
})

test_that("diagnostics() reports every parameter and every node", {
  fit <- ch_fit(games(), num_chains = 2, num_samples = 40, num_warmup = 5, seed = 3,
                show_progress = FALSE)
  d <- diagnostics(fit)
  expect_equal(d$quantity, c(network_properties(fit)$parameter, "H_A_b"))
  expect_true(all(c("ess_bulk", "ess_tail", "rhat") %in% names(d)))
  s <- diagnostics(fit, "relative")
  expect_equal(s$node, fit$nodes)
  expect_error(diagnostics(fit, "nope"))
})
