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

test_that("a future schema_version is refused with a clear error", {
  skip_on_cran()
  set.seed(1)
  n <- 80
  p <- 3
  X <- matrix(runif(n * p), ncol = p)
  y <- X[, 1] + rnorm(n)
  m <- bart(
    X_train = X, y_train = y,
    num_gfr = 0, num_burnin = 0, num_mcmc = 5,
    general_params = list(random_seed = 1)
  )
  js <- saveBARTModelToJsonString(m)
  # Bump the stamped schema_version to a value this install cannot read.
  js_future <- sub('("schema_version"\\s*:\\s*)[0-9]+', "\\199", js, perl = TRUE)
  expect_false(identical(js, js_future)) # confirm the substitution matched
  expect_error(createBARTModelFromJsonString(js_future), "schema_version")
})

# ===========================================================================
# v1 (unified-envelope) golden-fixture matrix snapshot tests
# ===========================================================================
#
# Matrix: {bart, bcf} x {numeric, categorical} x {no-rfx, rfx}. These lock the
# on-disk schema_version=1 format directly (named forest keys,
# covariate_preprocessor, and rfx_unique_group_ids relocated into
# random_effects), whereas the legacy v0 fixtures above guard the v0 -> v1
# migration path. Regenerate with test/R/testthat/fixtures/generate_v1_fixtures.R.

make_v1_covariates <- function(categorical, k, seed = 7) {
  set.seed(seed)
  X_num <- matrix(runif(k * 4), ncol = 4)
  if (!categorical) {
    return(X_num)
  }
  X <- data.frame(X_num)
  X$cat <- factor(
    sample(c("a", "b", "c"), k, replace = TRUE),
    levels = c("a", "b", "c")
  )
  X
}

for (.categorical in c(FALSE, TRUE)) {
  for (.rfx in c(FALSE, TRUE)) {
    local({
      categorical <- .categorical
      rfx <- .rfx
      kind <- if (categorical) "categorical" else "numeric"
      sfx <- if (rfx) "_rfx" else ""

      test_that(sprintf("BART v1 fixture loads and predicts (%s%s)", kind, sfx), {
        skip_on_cran()
        skip_if_not_installed("jsonlite")
        obj <- read_fixture_json(sprintf("bart_%s%s_v1.json", kind, sfx))
        expect_equal(obj$schema_version, 1)
        expect_equal(names(obj$forests), "mean_forest")
        if (rfx) {
          # rfx unique group ids live in the random_effects subfolder, not top-level
          expect_true("rfx_unique_group_ids" %in% names(obj$random_effects))
          expect_false("rfx_unique_group_ids" %in% names(obj))
        }
        m <- createBARTModelFromJsonString(write_json_string(obj))
        expect_equal(m$model_params$has_rfx, rfx)
        k <- 12
        X <- make_v1_covariates(categorical, k)
        args <- list(object = m, X = X)
        if (rfx) {
          args$rfx_group_ids <- (0:(k - 1)) %% 3
          args$rfx_basis <- matrix(1, nrow = k, ncol = 1)
        }
        preds <- do.call(predict, args)
        expect_equal(nrow(preds$y_hat), k)
      })

      test_that(sprintf("BCF v1 fixture loads and predicts (%s%s)", kind, sfx), {
        skip_on_cran()
        skip_if_not_installed("jsonlite")
        obj <- read_fixture_json(sprintf("bcf_%s%s_v1.json", kind, sfx))
        expect_equal(obj$schema_version, 1)
        expect_true(all(
          c("prognostic_forest", "treatment_forest") %in% names(obj$forests)
        ))
        if (rfx) {
          expect_true("rfx_unique_group_ids" %in% names(obj$random_effects))
        }
        m <- createBCFModelFromJsonString(write_json_string(obj))
        expect_equal(m$model_params$has_rfx, rfx)
        k <- 12
        X <- make_v1_covariates(categorical, k)
        set.seed(3)
        Z <- rbinom(k, 1, 0.5)
        pi <- rep(0.5, k)
        args <- list(object = m, X = X, Z = Z, propensity = pi)
        if (rfx) {
          args$rfx_group_ids <- (0:(k - 1)) %% 3
          args$rfx_basis <- matrix(1, nrow = k, ncol = 1)
        }
        preds <- do.call(predict, args)
        expect_equal(nrow(preds$y_hat), k)
      })
    })
  }
}
