test_that("a seeded fit from the default start gives the Python package's samples", {
  # Both start from the C++ core's default (src/modularity.cpp, src/initial_state.cpp) since 0.6.0.
  # Fixture: tools/make_r_parity_fixture.py.
  e <- read.csv(test_path("fixtures", "parity_edges.csv"),
                colClasses = c("character", "character", "character", "numeric"))
  init <- read.csv(test_path("fixtures", "parity_init.csv"),
                   colClasses = c("character", "integer", "numeric"))
  py <- read.csv(test_path("fixtures", "parity_default_samples.csv"))
  fit <- ch_fit(e, vertices = data.frame(name = init$node), num_chains = 2,
                samples_per_chain = 40, burn_sweeps = 10, seed = 13, show_progress = FALSE,
                timeout = 600)
  r <- samples(fit)
  for (col in setdiff(names(py), "chain_id")) {
    expect_equal(r[[col]], py[[col]], tolerance = 1e-12, info = col)
  }
})

test_that("summaries work after saveRDS, from a rebuilt handle", {
  fit <- quick(games())
  f <- tempfile(fileext = ".rds")
  saveRDS(fit, f)
  back <- readRDS(f)
  expect_false(communityhierarchy:::.ch_handle_valid(back$native$handle))
  expect_identical(consensus_partition(back), consensus_partition(fit))
  expect_true(communityhierarchy:::.ch_handle_valid(back$native$handle))
})
