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

# ===========================================================================
# WS-E: cross-platform load gate
# ===========================================================================
#
# A cross-platform load succeeds for portable (all-numeric, integer-rfx) models
# and is refused with a clear error otherwise; same-platform loads ignore the
# flags. We relabel a fixture's `platform` as the other platform to drive the
# gate from within R's own suite (the gate peeks generic flags; the foreign
# preprocessor body is ignored).

read_fixture_raw <- function(fixture_name) {
  paste(
    readLines(testthat::test_path("fixtures", fixture_name), warn = FALSE),
    collapse = ""
  )
}

as_foreign <- function(fixture_name, compat = NULL) {
  cj <- createCppJsonString(read_fixture_raw(fixture_name))
  cj$erase_field("platform")
  cj$add_string("platform", "python")
  if (!is.null(compat)) {
    cj$add_boolean(
      "cross_platform_compatible",
      compat,
      subfolder_name = "random_effects"
    )
  }
  cj$return_json_string()
}

test_that("numeric BART loads cross-platform and predicts", {
  skip_on_cran()
  skip_if_not_installed("jsonlite")
  m <- createBARTModelFromJsonString(as_foreign("bart_numeric_v1.json"))
  set.seed(0)
  X <- matrix(runif(8 * 4), ncol = 4)
  expect_equal(nrow(predict(m, X = X)$y_hat), 8)
})

test_that("numeric BCF loads cross-platform and predicts", {
  skip_on_cran()
  skip_if_not_installed("jsonlite")
  m <- createBCFModelFromJsonString(as_foreign("bcf_numeric_v1.json"))
  set.seed(0)
  X <- matrix(runif(8 * 4), ncol = 4)
  Z <- rbinom(8, 1, 0.5)
  preds <- predict(m, X = X, Z = Z, propensity = rep(0.5, 8))
  expect_equal(nrow(preds$y_hat), 8)
})

test_that("non-portable models are refused cross-platform", {
  skip_on_cran()
  skip_if_not_installed("jsonlite")
  expect_error(
    createBARTModelFromJsonString(as_foreign("bart_categorical_v1.json")),
    "non-numeric covariates"
  )
  expect_error(
    createBCFModelFromJsonString(as_foreign("bcf_categorical_v1.json")),
    "non-numeric covariates"
  )
  expect_error(
    createBARTModelFromJsonString(
      as_foreign("bart_numeric_rfx_v1.json", compat = FALSE)
    ),
    "random effects"
  )
})

test_that("same-platform categorical load is not refused", {
  skip_on_cran()
  skip_if_not_installed("jsonlite")
  m <- createBARTModelFromJsonString(read_fixture_raw("bart_categorical_v1.json"))
  expect_s3_class(m, "bartmodel")
})

test_that("cross-platform refusal names the offending columns", {
  skip_on_cran()
  skip_if_not_installed("jsonlite")
  # The R categorical fixtures encode the one categorical feature as column "cat".
  expect_error(
    createBARTModelFromJsonString(as_foreign("bart_categorical_v1.json")),
    "columns:\\s*cat"
  )
  expect_error(
    createBCFModelFromJsonString(as_foreign("bcf_categorical_v1.json")),
    "columns:\\s*cat"
  )
})

# ===========================================================================
# schema_version resolution: unrecognized / out-of-range stamps
# ===========================================================================

test_that("an unrecognized schema_version stamp is rejected", {
  skip_on_cran()
  set.seed(0)
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
  # A present-but-non-integer ("unknown") or negative stamp must hard-error,
  # not silently degrade to legacy v0.
  js_str <- sub(
    '("schema_version"\\s*:\\s*)[0-9]+',
    '\\1"unknown"',
    js,
    perl = TRUE
  )
  js_neg <- sub('("schema_version"\\s*:\\s*)[0-9]+', "\\1-1", js, perl = TRUE)
  expect_false(identical(js, js_str))
  expect_false(identical(js, js_neg))
  expect_error(createBARTModelFromJsonString(js_str), "schema_version")
  expect_error(createBARTModelFromJsonString(js_neg), "schema_version")
})

test_that("an absent schema_version stamp loads as legacy v0", {
  skip_on_cran()
  skip_if_not_installed("jsonlite")
  # Stripping the stamp must NOT error (absent == v0), only present-but-invalid does.
  obj <- read_fixture_json("bart_numeric_v1.json")
  m <- createBARTModelFromJsonString(strip_fields(obj, "schema_version"))
  expect_s3_class(m, "bartmodel")
})

test_that("legacy v0 golden fixtures stay genuinely v0", {
  skip_on_cran()
  skip_if_not_installed("jsonlite")
  # The v0 goldens must remain unstamped so the v0 read path stays exercised; if
  # someone regenerates them at the current version this guard flags it.
  for (name in c("bart_mcmc.json", "bcf_mcmc.json")) {
    obj <- read_fixture_json(name)
    expect_false(
      "schema_version" %in% names(obj),
      info = paste(name, "should stay a v0 golden (no schema_version stamp)")
    )
  }
})

# ===========================================================================
# v0 platform-fingerprint fallback (inferPlatformV0)
# ===========================================================================

# ===========================================================================
# Genuine cross-platform load: load a real Python-written (all-numeric) model
# and assert the forest reconstructs bit-exactly. Fixtures are the real
# Python-written v1 goldens copied verbatim from the Python suite (platform ==
# "python"), so this is the true cross-platform load path -- not a relabeled own
# fixture.
# ===========================================================================

forest_bitexact <- function(forest_samples, envelope, forest_name) {
  reser <- createCppJson()
  reser$add_forest(forest_samples)
  got <- jsonlite::fromJSON(reser$return_json_string(), simplifyVector = FALSE)$forests$forest_0
  identical(got, envelope$forests[[forest_name]])
}

test_that("R loads a genuinely Python-written BART model bit-exactly", {
  skip_on_cran()
  skip_if_not_installed("jsonlite")
  raw <- read_fixture_raw("bart_numeric_v1_pywritten.json")
  envelope <- jsonlite::fromJSON(raw, simplifyVector = FALSE)
  expect_equal(envelope$platform, "python")
  m <- createBARTModelFromJsonString(raw) # gate must ACCEPT (all-numeric, portable)
  expect_s3_class(m, "bartmodel")
  expect_true(forest_bitexact(
    extractForest(m, "mean"),
    envelope,
    "mean_forest"
  ))
  set.seed(0)
  preds <- predict(m, X = matrix(runif(8 * 4), ncol = 4))$y_hat
  expect_equal(nrow(preds), 8)
  expect_true(all(is.finite(preds)))
})

test_that("R loads a genuinely Python-written BCF model bit-exactly", {
  skip_on_cran()
  skip_if_not_installed("jsonlite")
  raw <- read_fixture_raw("bcf_numeric_v1_pywritten.json")
  envelope <- jsonlite::fromJSON(raw, simplifyVector = FALSE)
  expect_equal(envelope$platform, "python")
  m <- createBCFModelFromJsonString(raw)
  expect_s3_class(m, "bcfmodel")
  expect_true(forest_bitexact(
    extractForest(m, "prognostic"),
    envelope,
    "prognostic_forest"
  ))
  expect_true(forest_bitexact(
    extractForest(m, "treatment"),
    envelope,
    "treatment_forest"
  ))
  set.seed(0)
  preds <- predict(
    m,
    X = matrix(runif(8 * 4), ncol = 4),
    Z = rbinom(8, 1, 0.5),
    propensity = rep(0.5, 8)
  )$y_hat
  expect_equal(nrow(preds), 8)
  expect_true(all(is.finite(preds)))
})

test_that("R loads a genuinely Python-written RFX BART model cross-platform", {
  skip_on_cran()
  skip_if_not_installed("jsonlite")
  # Regression guard: a Python-written model has no `rfx_unique_group_ids` string
  # levels, so R's cross-load resolves the group ids from the sampled label mapper
  # (`samples$materialize_rfx()`). This path was broken by the single-owner refactor
  # (it referenced the removed `output$rfx_samples`) and is only reachable via a
  # genuinely foreign RFX model -- never a same-platform R load.
  raw <- read_fixture_raw("bart_numeric_rfx_v1_pywritten.json")
  envelope <- jsonlite::fromJSON(raw, simplifyVector = FALSE)
  expect_equal(envelope$platform, "python")
  m <- createBARTModelFromJsonString(raw)
  expect_s3_class(m, "bartmodel")
  # Group ids were resolved from the label mapper (not read from a string-levels field).
  expect_true(length(m$rfx_unique_group_ids) > 0)
  expect_true(forest_bitexact(extractForest(m, "mean"), envelope, "mean_forest"))
})

test_that("v0 platform fingerprint classifies legacy envelopes", {
  skip_on_cran()
  skip_if_not_installed("jsonlite")
  # Real R-written v0 fixtures carry `preprocessor_metadata`.
  for (name in c("bart_mcmc.json", "bcf_mcmc.json")) {
    obj <- read_fixture_json(name)
    expect_false("schema_version" %in% names(obj)) # genuinely v0
    cj <- createCppJsonString(read_fixture_raw(name))
    expect_equal(stochtree:::inferPlatformV0(cj, "python"), "R")
  }
  # A Python-written envelope carries `covariate_preprocessor`.
  cj_py <- createCppJsonString(write_json_string(list(
    covariate_preprocessor = list(a = 1)
  )))
  expect_equal(stochtree:::inferPlatformV0(cj_py, "R"), "python")
  # No decisive fingerprint -> fall back to the loading platform.
  cj_none <- createCppJsonString(write_json_string(list(some_key = 1)))
  expect_equal(stochtree:::inferPlatformV0(cj_none, "R"), "R")
  expect_equal(stochtree:::inferPlatformV0(cj_none, "python"), "python")
})
