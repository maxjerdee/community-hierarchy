test_that("a seeded fit gives the same samples as the Python package", {
  # Fixture: tools/make_r_parity_fixture.py (the same with community-hierarchy 0.5.0).
  e <- read.csv(test_path("fixtures", "parity_edges.csv"),
                colClasses = c("character", "character", "character", "numeric"))
  init <- read.csv(test_path("fixtures", "parity_init.csv"),
                   colClasses = c("character", "integer", "numeric"))
  py <- read.csv(test_path("fixtures", "parity_samples.csv"))
  fit <- ch_fit(e, vertices = data.frame(name = init$node), num_chains = 2,
                samples_per_chain = 40, burn_sweeps = 10, seed = 11,
                initial_partition = init$p0, initial_scores = init$s0,
                show_progress = FALSE, timeout = 600)
  r <- samples(fit)
  expect_equal(r$chain_id, py$chain_id + 1L)
  for (col in setdiff(names(py), "chain_id")) {
    expect_equal(r[[col]], py[[col]], tolerance = 1e-12, info = col)
  }
})

test_that("direction_loo() gives the Python package's leave-one-out", {
  # Fixture: tools/make_r_parity_fixture.py, the default-start fit (seed 13).
  e <- read.csv(test_path("fixtures", "parity_edges.csv"),
                colClasses = c("character", "character", "character", "numeric"))
  init <- read.csv(test_path("fixtures", "parity_init.csv"),
                   colClasses = c("character", "integer", "numeric"))
  py <- read.csv(test_path("fixtures", "parity_loo.csv"),
                 colClasses = c("character", "character", "logical", "numeric", "numeric",
                                "numeric", "numeric"))
  fit <- ch_fit(e, vertices = data.frame(name = init$node), num_chains = 2,
                samples_per_chain = 40, burn_sweeps = 10, seed = 13,
                show_progress = FALSE, timeout = 600)
  r <- direction_loo(fit)
  expect_equal(r$winner, py$winner)
  expect_equal(r$loser, py$loser)
  expect_equal(r$draw, py$draw)
  expect_equal(r$weight, py$weight)
  for (col in c("elpd", "pareto_k", "p_win")) {
    expect_equal(r[[col]], py[[col]], tolerance = 1e-12, info = col)
  }
  expect_true(all(is.na(r$p_win) == r$draw))
  # Survives saveRDS() / readRDS(), through the rebuilt handle.
  f <- tempfile(fileext = ".rds")
  saveRDS(fit, f)
  expect_equal(direction_loo(readRDS(f))$elpd, r$elpd)
})
