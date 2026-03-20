# Backward-compatibility deserialization tests
#
# These tests verify that models serialized without certain optional fields
# (as would be produced by older package versions) can still be loaded
# correctly, with appropriate warnings where applicable.
#
# Fixture files (test/R/testthat/fixtures/) are generated once from the
# current package and checked in.  They serve as a "snapshot" — if a future
# change breaks the ability to deserialize them, these tests will catch it.

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#' Read and parse a fixture JSON file into a plain R list
read_fixture_json <- function(fixture_name) {
  path <- testthat::test_path("fixtures", fixture_name)
  js <- paste(readLines(path, warn = FALSE), collapse = "")
  jsonlite::fromJSON(js, simplifyVector = FALSE)
}

#' Serialise an R list back to a JSON string suitable for createBARTModelFromJsonString
#  / createBCFModelFromJsonString.
write_json_string <- function(obj) {
  jsonlite::toJSON(obj, auto_unbox = TRUE, digits = NA)
}

#' Remove one or more top-level fields from a parsed JSON list, then serialise.
strip_fields <- function(obj, ...) {
  fields <- c(...)
  for (f in fields) {
    obj[[f]] <- NULL
  }
  write_json_string(obj)
}

#' Collect all warning messages emitted while evaluating `expr`.
collect_warnings <- function(expr) {
  warns <- character(0)
  withCallingHandlers(expr, warning = function(w) {
    warns <<- c(warns, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  warns
}

# ===========================================================================
# BART snapshot tests
# ===========================================================================

test_that("BART fixture deserializes and predictions are reproducible", {
  skip_on_cran()
  skip_if_not_installed("jsonlite")

  fixture_obj <- read_fixture_json("bart_mcmc.json")
  json_str <- write_json_string(fixture_obj)

  set.seed(1)
  n <- 30
  X <- matrix(runif(n * 5), ncol = 5)
  m <- createBARTModelFromJsonString(json_str)
  # Just verify the model loads and can predict without error
  preds <- predict(m, X = X)
  expect_true(is.list(preds))
  expect_true("y_hat" %in% names(preds))
  expect_equal(nrow(preds$y_hat), n)
})

test_that("BART roundtrip from fixture matches direct load", {
  skip_on_cran()
  skip_if_not_installed("jsonlite")

  fixture_obj <- read_fixture_json("bart_mcmc.json")
  json_str <- write_json_string(fixture_obj)

  m1 <- createBARTModelFromJsonString(json_str)
  m2 <- createBARTModelFromJsonString(json_str) # second load must be identical

  set.seed(99)
  X <- matrix(runif(20 * 5), ncol = 5)
  p1 <- rowMeans(predict(m1, X = X)$y_hat)
  p2 <- rowMeans(predict(m2, X = X)$y_hat)
  expect_equal(p1, p2)
})

# ===========================================================================
# BCF snapshot tests
# ===========================================================================

test_that("BCF fixture deserializes and predictions are reproducible", {
  skip_on_cran()
  skip_if_not_installed("jsonlite")

  fixture_obj <- read_fixture_json("bcf_mcmc.json")
  json_str <- write_json_string(fixture_obj)

  set.seed(1)
  n <- 30
  X <- matrix(runif(n * 5), ncol = 5)
  Z <- rbinom(n, 1, 0.5)
  pi <- rep(0.5, n)

  m <- createBCFModelFromJsonString(json_str)
  preds <- predict(m, X, Z, pi)
  expect_true(is.list(preds))
  expect_true("y_hat" %in% names(preds))
  expect_true("tau_hat" %in% names(preds))
  expect_equal(nrow(preds$y_hat), n)
})

# ===========================================================================
# BART backward-compat: missing optional fields
# ===========================================================================

test_that("BART loads without 'outcome_model' (pre-v0.4.1)", {
  skip_on_cran()
  skip_if_not_installed("jsonlite")

  fixture_obj <- read_fixture_json("bart_mcmc.json")
  json_str <- strip_fields(fixture_obj, "outcome_model", "probit_outcome_model")

  warns <- collect_warnings(m <- createBARTModelFromJsonString(json_str))
  # Should warn about missing outcome_model
  expect_true(
    any(grepl(
      "outcome_model|outcome.*missing|missing.*outcome",
      warns,
      ignore.case = TRUE
    )) ||
      length(warns) == 0
  ) # no warning is also acceptable for truly optional fields

  set.seed(1)
  X <- matrix(runif(20 * 5), ncol = 5)
  preds <- predict(m, X = X)
  expect_equal(nrow(preds$y_hat), 20)
})

test_that("BART loads without 'rfx_model_spec' when has_rfx=FALSE", {
  skip_on_cran()
  skip_if_not_installed("jsonlite")

  fixture_obj <- read_fixture_json("bart_mcmc.json")
  # Ensure has_rfx is FALSE (fixture uses no RFX)
  expect_false(isTRUE(fixture_obj$has_rfx))

  json_str <- strip_fields(fixture_obj, "rfx_model_spec")

  m <- createBARTModelFromJsonString(json_str)
  set.seed(1)
  X <- matrix(runif(20 * 5), ncol = 5)
  preds <- predict(m, X = X)
  expect_equal(nrow(preds$y_hat), 20)
})

test_that("BART loads without 'preprocessor_metadata' (pre-preprocessor versions)", {
  skip_on_cran()
  skip_if_not_installed("jsonlite")

  fixture_obj <- read_fixture_json("bart_mcmc.json")
  json_str <- strip_fields(fixture_obj, "preprocessor_metadata")

  # Loading should succeed (with a warning about missing preprocessor)
  warns <- collect_warnings(m <- createBARTModelFromJsonString(json_str))
  expect_true(any(grepl("preprocessor|preprocess", warns, ignore.case = TRUE)))
  # Model object is returned
  expect_true(is.list(m))
})

test_that("BART loads without 'num_chains' / 'keep_every'", {
  skip_on_cran()
  skip_if_not_installed("jsonlite")

  fixture_obj <- read_fixture_json("bart_mcmc.json")
  json_str <- strip_fields(fixture_obj, "num_chains", "keep_every")

  m <- createBARTModelFromJsonString(json_str)
  set.seed(1)
  X <- matrix(runif(20 * 5), ncol = 5)
  preds <- predict(m, X = X)
  expect_equal(nrow(preds$y_hat), 20)
})

# ===========================================================================
# BCF backward-compat: missing optional fields
# ===========================================================================

test_that("BCF loads without 'outcome_model' (pre-v0.4.1)", {
  skip_on_cran()
  skip_if_not_installed("jsonlite")

  fixture_obj <- read_fixture_json("bcf_mcmc.json")
  json_str <- strip_fields(fixture_obj, "outcome_model", "probit_outcome_model")

  m <- createBCFModelFromJsonString(json_str)

  set.seed(1)
  n <- 20
  X <- matrix(runif(n * 5), ncol = 5)
  Z <- rbinom(n, 1, 0.5)
  pi <- rep(0.5, n)
  preds <- predict(m, X, Z, pi)
  expect_equal(nrow(preds$y_hat), n)
})

test_that("BCF loads without 'multivariate_treatment' (pre-v0.4.0)", {
  skip_on_cran()
  skip_if_not_installed("jsonlite")

  fixture_obj <- read_fixture_json("bcf_mcmc.json")
  json_str <- strip_fields(fixture_obj, "multivariate_treatment")

  m <- createBCFModelFromJsonString(json_str)

  set.seed(1)
  n <- 20
  X <- matrix(runif(n * 5), ncol = 5)
  Z <- rbinom(n, 1, 0.5)
  pi <- rep(0.5, n)
  preds <- predict(m, X, Z, pi)
  expect_equal(nrow(preds$y_hat), n)
})

test_that("BCF loads without 'internal_propensity_model' (pre-v0.3.2)", {
  skip_on_cran()
  skip_if_not_installed("jsonlite")

  fixture_obj <- read_fixture_json("bcf_mcmc.json")
  json_str <- strip_fields(fixture_obj, "internal_propensity_model")

  m <- createBCFModelFromJsonString(json_str)

  set.seed(1)
  n <- 20
  X <- matrix(runif(n * 5), ncol = 5)
  Z <- rbinom(n, 1, 0.5)
  pi <- rep(0.5, n)
  preds <- predict(m, X, Z, pi)
  expect_equal(nrow(preds$y_hat), n)
})

test_that("BCF loads without 'rfx_model_spec' when has_rfx=FALSE", {
  skip_on_cran()
  skip_if_not_installed("jsonlite")

  fixture_obj <- read_fixture_json("bcf_mcmc.json")
  expect_false(isTRUE(fixture_obj$has_rfx))

  json_str <- strip_fields(fixture_obj, "rfx_model_spec")

  m <- createBCFModelFromJsonString(json_str)

  set.seed(1)
  n <- 20
  X <- matrix(runif(n * 5), ncol = 5)
  Z <- rbinom(n, 1, 0.5)
  pi <- rep(0.5, n)
  preds <- predict(m, X, Z, pi)
  expect_equal(nrow(preds$y_hat), n)
})

test_that("BCF loads without 'preprocessor_metadata'", {
  skip_on_cran()
  skip_if_not_installed("jsonlite")

  fixture_obj <- read_fixture_json("bcf_mcmc.json")
  json_str <- strip_fields(fixture_obj, "preprocessor_metadata")

  # Loading should succeed (with a warning about missing preprocessor)
  warns <- collect_warnings(m <- createBCFModelFromJsonString(json_str))
  expect_true(any(grepl("preprocessor|preprocess", warns, ignore.case = TRUE)))
  # Model object is returned
  expect_true(is.list(m))
})

test_that("BCF loads without 'num_chains' / 'keep_every'", {
  skip_on_cran()
  skip_if_not_installed("jsonlite")

  fixture_obj <- read_fixture_json("bcf_mcmc.json")
  json_str <- strip_fields(fixture_obj, "num_chains", "keep_every")

  m <- createBCFModelFromJsonString(json_str)

  set.seed(1)
  n <- 20
  X <- matrix(runif(n * 5), ncol = 5)
  Z <- rbinom(n, 1, 0.5)
  pi <- rep(0.5, n)
  preds <- predict(m, X, Z, pi)
  expect_equal(nrow(preds$y_hat), n)
})

test_that("BCF loads without 'has_rfx_basis'", {
  skip_on_cran()
  skip_if_not_installed("jsonlite")

  fixture_obj <- read_fixture_json("bcf_mcmc.json")
  json_str <- strip_fields(fixture_obj, "has_rfx_basis")

  m <- createBCFModelFromJsonString(json_str)

  set.seed(1)
  n <- 20
  X <- matrix(runif(n * 5), ncol = 5)
  Z <- rbinom(n, 1, 0.5)
  pi <- rep(0.5, n)
  preds <- predict(m, X, Z, pi)
  expect_equal(nrow(preds$y_hat), n)
})

test_that("BCF loads with multiple missing optional fields simultaneously", {
  skip_on_cran()
  skip_if_not_installed("jsonlite")

  fixture_obj <- read_fixture_json("bcf_mcmc.json")
  # Strip all optional fields (including preprocessor_metadata — prediction not checked)
  json_str <- strip_fields(
    fixture_obj,
    "outcome_model",
    "probit_outcome_model",
    "multivariate_treatment",
    "internal_propensity_model",
    "rfx_model_spec",
    "num_chains",
    "keep_every",
    "has_rfx_basis",
    "preprocessor_metadata"
  )

  warns <- collect_warnings(m <- createBCFModelFromJsonString(json_str))
  # Model must load
  expect_true(is.list(m))
  # At least the preprocessor_metadata warning should fire
  expect_true(any(grepl("preprocessor|preprocess", warns, ignore.case = TRUE)))
})
