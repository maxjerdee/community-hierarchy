test_that("presets resolve as in the Python package", {
  r <- communityhierarchy:::resolve_config
  m <- r(ch_config(), FALSE, 10)               # default: the degree-corrected SBM
  expect_equal(c(m$variation_in, m$variation_out), c(0.5, 0.5))
  expect_false(m$assortative)
  expect_equal(m$ties_parameter, 0)
  expect_equal(m$degree_correction, 0.5)
  expect_equal(m$hierarchy_model, "bradley_terry")
  m <- r(ch_config(groups_model = "general_ASBM"), TRUE, 10)
  expect_null(m$variation_in)
  expect_true(m$assortative)
  expect_null(m$ties_parameter)
  m <- r(ch_config(groups_model = "hybrid_ASBM", variation_out = NULL), FALSE, 10)
  expect_equal(m$variation_in, 0.5)
  expect_null(m$variation_out)
  expect_equal(r(ch_config(interaction = "independent"), FALSE, 10)$group_depth, 0)
  expect_equal(r(ch_config(degree_correction = FALSE), FALSE, 10)$degree_correction, 0)
})

test_that("invalid configurations are rejected", {
  expect_error(ch_config(groups_model = "nope"), "groups_model")
  expect_error(ch_config(groups_model = "general_canonical"), "renamed")
  expect_error(ch_config(variation_in = 2), "variation_in")
  expect_error(ch_config(hierarchy_model = "bradley_terry", ties_parameter = 1), "fixes ties_parameter")
  expect_error(ch_config(interaction = "independent", group_depth = 1), "fixes group_depth")
  expect_error(ch_config(num_groups = 1.5), "num_groups")
  expect_error(ch_config(variation_in = "a"), "variation_in")
  r <- communityhierarchy:::resolve_config
  expect_error(r(ch_config(num_groups = 11), FALSE, 10), "larger than the number of nodes")
  expect_error(r(ch_config(hierarchy_model = "bradley_terry"), TRUE, 10), "neutral edges")
  expect_output(print(ch_config(variation_in = NULL)), "variation_in = NULL")
})
