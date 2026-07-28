test_that("BART Serialization", {
  skip_on_cran()

  # Generate simulated data
  n <- 100
  p <- 5
  X <- matrix(runif(n * p), ncol = p)
  # fmt: skip
  f_XW <- (
        ((0 <= X[,1]) & (0.25 > X[,1])) * (-7.5) + 
        ((0.25 <= X[,1]) & (0.5 > X[,1])) * (-2.5) + 
        ((0.5 <= X[,1]) & (0.75 > X[,1])) * (2.5) + 
        ((0.75 <= X[,1]) & (1 > X[,1])) * (7.5)
    )
  noise_sd <- 1
  y <- f_XW + rnorm(n, 0, noise_sd)
  test_set_pct <- 0.2
  n_test <- round(test_set_pct * n)
  n_train <- n - n_test
  test_inds <- sort(sample(1:n, n_test, replace = FALSE))
  train_inds <- (1:n)[!((1:n) %in% test_inds)]
  X_test <- X[test_inds, ]
  X_train <- X[train_inds, ]
  y_test <- y[test_inds]
  y_train <- y[train_inds]

  # Sample a BART model
  general_param_list <- list(num_chains = 1, keep_every = 1)
  bart_model <- bart(
    X_train = X_train,
    y_train = y_train,
    num_gfr = 0,
    num_burnin = 10,
    num_mcmc = 10,
    general_params = general_param_list
  )
  y_hat_orig <- rowMeans(predict(bart_model, X = X_test)$y_hat)

  # Save to JSON
  bart_json_string <- saveBARTModelToJsonString(bart_model)

  # Reload as a BART model
  bart_model_roundtrip <- createBARTModelFromJsonString(bart_json_string)

  # Predict from the roundtrip BART model
  y_hat_reloaded <- rowMeans(predict(bart_model_roundtrip, X = X_test)$y_hat)

  # Assertion
  expect_equal(y_hat_orig, y_hat_reloaded)
})

test_that("BCF Serialization", {
  skip_on_cran()

  n <- 500
  x1 <- runif(n)
  x2 <- runif(n)
  x3 <- runif(n)
  x4 <- runif(n)
  x5 <- runif(n)
  X <- cbind(x1, x2, x3, x4, x5)
  p <- ncol(X)
  pi_x <- 0.25 + 0.5 * X[, 1]
  mu_x <- pi_x * 5
  tau_x <- X[, 2] * 2
  Z <- rbinom(n, 1, pi_x)
  E_XZ <- mu_x + Z * tau_x
  y <- E_XZ + rnorm(n, 0, 1)
  test_set_pct <- 0.2
  n_test <- round(test_set_pct * n)
  n_train <- n - n_test
  test_inds <- sort(sample(1:n, n_test, replace = FALSE))
  train_inds <- (1:n)[!((1:n) %in% test_inds)]
  X_test <- X[test_inds, ]
  X_train <- X[train_inds, ]
  pi_test <- pi_x[test_inds]
  pi_train <- pi_x[train_inds]
  Z_test <- Z[test_inds]
  Z_train <- Z[train_inds]
  y_test <- y[test_inds]
  y_train <- y[train_inds]
  mu_test <- mu_x[test_inds]
  mu_train <- mu_x[train_inds]
  tau_test <- tau_x[test_inds]
  tau_train <- tau_x[train_inds]

  # Sample a BCF model
  bcf_model <- bcf(
    X_train = X_train,
    Z_train = Z_train,
    y_train = y_train,
    propensity_train = pi_train,
    num_gfr = 100,
    num_burnin = 0,
    num_mcmc = 100
  )
  bcf_preds_orig <- predict(bcf_model, X_test, Z_test, pi_test)
  mu_hat_orig <- rowMeans(bcf_preds_orig[["mu_hat"]])
  tau_hat_orig <- rowMeans(bcf_preds_orig[["tau_hat"]])
  y_hat_orig <- rowMeans(bcf_preds_orig[["y_hat"]])

  # Save to JSON
  bcf_json_string <- saveBCFModelToJsonString(bcf_model)

  # Reload as a BCF model
  bcf_model_roundtrip <- createBCFModelFromJsonString(bcf_json_string)

  # Predict from the roundtrip BCF model
  bcf_preds_reloaded <- predict(bcf_model_roundtrip, X_test, Z_test, pi_test)
  mu_hat_reloaded <- rowMeans(bcf_preds_reloaded[["mu_hat"]])
  tau_hat_reloaded <- rowMeans(bcf_preds_reloaded[["tau_hat"]])
  y_hat_reloaded <- rowMeans(bcf_preds_reloaded[["y_hat"]])

  # Assertion
  expect_equal(y_hat_orig, y_hat_reloaded)
})

test_that("BCF Serialization (no propensity)", {
  skip_on_cran()

  n <- 500
  x1 <- runif(n)
  x2 <- runif(n)
  x3 <- runif(n)
  x4 <- runif(n)
  x5 <- runif(n)
  X <- cbind(x1, x2, x3, x4, x5)
  p <- ncol(X)
  pi_x <- 0.25 + 0.5 * X[, 1]
  mu_x <- pi_x * 5
  tau_x <- X[, 2] * 2
  Z <- rbinom(n, 1, pi_x)
  E_XZ <- mu_x + Z * tau_x
  y <- E_XZ + rnorm(n, 0, 1)
  test_set_pct <- 0.2
  n_test <- round(test_set_pct * n)
  n_train <- n - n_test
  test_inds <- sort(sample(1:n, n_test, replace = FALSE))
  train_inds <- (1:n)[!((1:n) %in% test_inds)]
  X_test <- X[test_inds, ]
  X_train <- X[train_inds, ]
  pi_test <- pi_x[test_inds]
  pi_train <- pi_x[train_inds]
  Z_test <- Z[test_inds]
  Z_train <- Z[train_inds]
  y_test <- y[test_inds]
  y_train <- y[train_inds]
  mu_test <- mu_x[test_inds]
  mu_train <- mu_x[train_inds]
  tau_test <- tau_x[test_inds]
  tau_train <- tau_x[train_inds]

  # Sample a BCF model
  bcf_model <- bcf(
    X_train = X_train,
    Z_train = Z_train,
    y_train = y_train,
    num_gfr = 100,
    num_burnin = 0,
    num_mcmc = 100
  )
  bcf_preds_orig <- predict(bcf_model, X_test, Z_test)
  mu_hat_orig <- rowMeans(bcf_preds_orig[["mu_hat"]])
  tau_hat_orig <- rowMeans(bcf_preds_orig[["tau_hat"]])
  y_hat_orig <- rowMeans(bcf_preds_orig[["y_hat"]])

  # Save to JSON
  bcf_json_string <- saveBCFModelToJsonString(bcf_model)

  # Reload as a BCF model
  bcf_model_roundtrip <- createBCFModelFromJsonString(bcf_json_string)

  # Predict from the roundtrip BCF model
  bcf_preds_reloaded <- predict(bcf_model_roundtrip, X_test, Z_test)
  mu_hat_reloaded <- rowMeans(bcf_preds_reloaded[["mu_hat"]])
  tau_hat_reloaded <- rowMeans(bcf_preds_reloaded[["tau_hat"]])
  y_hat_reloaded <- rowMeans(bcf_preds_reloaded[["y_hat"]])

  # Assertion
  expect_equal(y_hat_orig, y_hat_reloaded)
})

test_that("BCF JSON uses canonical field names (sigma2_init, b1_samples, b0_samples)", {
  skip_on_cran()

  set.seed(1)
  n <- 100
  X <- matrix(runif(n * 5), ncol = 5)
  pi_x <- 0.25 + 0.5 * X[, 1]
  Z <- rbinom(n, 1, pi_x)
  y <- pi_x * 5 + Z * X[, 2] * 2 + rnorm(n)

  bcf_model <- bcf(
    X_train = X, Z_train = Z, y_train = y,
    propensity_train = pi_x,
    num_gfr = 0, num_burnin = 0, num_mcmc = 10,
    general_params = list(adaptive_coding = TRUE)
  )
  json_string <- saveBCFModelToJsonString(bcf_model)

  # New canonical names must be present
  expect_true(grepl('"sigma2_init"', json_string))
  expect_true(grepl('"b1_samples"', json_string))
  expect_true(grepl('"b0_samples"', json_string))

  # Legacy names must not be present
  expect_false(grepl('"initial_sigma2"', json_string))
  expect_false(grepl('"b_1_samples"', json_string))
  expect_false(grepl('"b_0_samples"', json_string))
})

test_that("BCF JSON deserialization handles legacy field names with warnings", {
  skip_on_cran()

  set.seed(2)
  n <- 100
  X <- matrix(runif(n * 5), ncol = 5)
  pi_x <- 0.25 + 0.5 * X[, 1]
  Z <- rbinom(n, 1, pi_x)
  y <- pi_x * 5 + Z * X[, 2] * 2 + rnorm(n)
  X_test <- matrix(runif(20 * 5), ncol = 5)
  pi_test <- rep(0.5, 20)
  Z_test <- rbinom(20, 1, 0.5)

  bcf_model <- bcf(
    X_train = X, Z_train = Z, y_train = y,
    propensity_train = pi_x,
    num_gfr = 0, num_burnin = 0, num_mcmc = 10,
    general_params = list(adaptive_coding = TRUE)
  )
  preds_orig <- predict(bcf_model, X_test, Z_test, pi_test)

  # Simulate a legacy JSON by replacing canonical names with old names
  json_new <- saveBCFModelToJsonString(bcf_model)
  json_legacy <- gsub('"sigma2_init"', '"initial_sigma2"', json_new, fixed = TRUE)
  json_legacy <- gsub('"b1_samples"', '"b_1_samples"', json_legacy, fixed = TRUE)
  json_legacy <- gsub('"b0_samples"', '"b_0_samples"', json_legacy, fixed = TRUE)

  # Loading a legacy JSON should emit deprecation warnings for all renamed fields
  all_warnings <- character(0)
  withCallingHandlers(
    bcf_legacy <- createBCFModelFromJsonString(json_legacy),
    warning = function(w) {
      all_warnings <<- c(all_warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  expect_true(any(grepl("initial_sigma2.*deprecated|deprecated.*initial_sigma2", all_warnings)))
  expect_true(any(grepl("b_1_samples.*deprecated|b_0_samples.*deprecated|deprecated.*b_[01]_samples", all_warnings)))

  # Predictions must still match
  preds_legacy <- predict(bcf_legacy, X_test, Z_test, pi_test)
  expect_equal(rowMeans(preds_legacy[["y_hat"]]), rowMeans(preds_orig[["y_hat"]]))
  expect_equal(rowMeans(preds_legacy[["tau_hat"]]), rowMeans(preds_orig[["tau_hat"]]))
})

# Fields that the BCF JSON contract must preserve across a round-trip. These are
# the metadata/flags consumed by print(), summary(), and predict(); the prior
# hyperparameters (a_global, treatment-specific priors, etc.) are intentionally
# not serialized and fall back to package defaults on reload, so they are
# excluded here on purpose.
.bcf_must_survive_fields <- c(
  "binary_treatment",
  "multivariate_treatment",
  "treatment_dim",
  "adaptive_coding",
  "sample_tau_0",
  "internal_propensity_model",
  "propensity_covariate",
  "include_variance_forest",
  "has_rfx",
  "standardize",
  "num_samples"
)

# Assert every contract field survives a round-trip with an equal value, and
# that the reloaded model can be printed (the GH #393 failure mode).
expect_bcf_roundtrip_ok <- function(original, reloaded, label) {
  for (f in .bcf_must_survive_fields) {
    expect_false(
      is.null(reloaded$model_params[[f]]),
      info = paste0(label, ": field '", f, "' missing after round-trip")
    )
    expect_equal(
      reloaded$model_params[[f]],
      original$model_params[[f]],
      info = paste0(label, ": field '", f, "' changed after round-trip")
    )
  }
  # Behavioral smoke: print must not error (this is what GH #393 hit).
  expect_error(capture.output(print(reloaded)), NA)
}

test_that("BCF round-trip preserves contract fields and supports print (GH #393)", {
  skip_on_cran()

  set.seed(393)
  n <- 300
  X <- matrix(runif(n * 5), ncol = 5)
  pi_x <- 0.25 + 0.5 * X[, 1]
  Z <- rbinom(n, 1, pi_x)
  y <- pi_x * 5 + Z * X[, 2] * 2 + rnorm(n)

  # The exact #393 config: binary treatment + internal propensity model
  # (no propensity supplied) + a print() of the reloaded object.
  bcf_model <- bcf(
    X_train = X, Z_train = Z, y_train = y,
    num_gfr = 10, num_burnin = 0, num_mcmc = 10
  )
  expect_true(bcf_model$model_params$binary_treatment)
  expect_true(bcf_model$model_params$internal_propensity_model)

  json_string <- saveBCFModelToJsonString(bcf_model)
  reloaded_string <- createBCFModelFromJsonString(json_string)
  expect_bcf_roundtrip_ok(bcf_model, reloaded_string, "string")

  # The internal propensity model is serialized as a full BART model, so its
  # preprocessor must reload clean (GH #393 problem 2). X has 5 numeric columns.
  prop_numeric_vars <-
    reloaded_string$bart_propensity_model$train_set_metadata$numeric_vars
  expect_false(anyNA(prop_numeric_vars))
  expect_equal(prop_numeric_vars, paste0("x", 1:5))

  # The combined-JSON loader does not support cached internal propensity
  # models (a multi-chain merge limitation, unrelated to #393), so exercise
  # that path with a supplied-propensity binary model.
  bcf_supplied <- bcf(
    X_train = X, Z_train = Z, y_train = y, propensity_train = pi_x,
    num_gfr = 10, num_burnin = 0, num_mcmc = 10
  )
  expect_true(bcf_supplied$model_params$binary_treatment)
  json_supplied <- saveBCFModelToJsonString(bcf_supplied)
  expect_bcf_roundtrip_ok(
    bcf_supplied,
    createBCFModelFromCombinedJsonString(list(json_supplied)),
    "combinedString"
  )
})

test_that("BCF multivariate round-trip preserves treatment_dim and supports print", {
  skip_on_cran()

  set.seed(395)
  n <- 300
  X <- matrix(runif(n * 5), ncol = 5)
  Z <- matrix(rnorm(n * 2), ncol = 2)
  pi_x <- matrix(0.5, n, 2)
  y <- X[, 1] + Z[, 1] * 0.5 + Z[, 2] * 0.3 + rnorm(n)

  bcf_model <- bcf(
    X_train = X, Z_train = Z, y_train = y, propensity_train = pi_x,
    num_gfr = 10, num_burnin = 0, num_mcmc = 10
  )
  expect_true(bcf_model$model_params$multivariate_treatment)
  expect_equal(bcf_model$model_params$treatment_dim, 2)

  json_string <- saveBCFModelToJsonString(bcf_model)
  reloaded <- createBCFModelFromJsonString(json_string)
  expect_equal(reloaded$model_params$treatment_dim, 2)
  # The multivariate branch of print() dereferences treatment_dim.
  expect_error(capture.output(print(reloaded)), NA)
})

test_that("BCF legacy JSON (no binary_treatment/treatment_dim) infers them (GH #393)", {
  skip_on_cran()

  set.seed(394)
  n <- 200
  X <- matrix(runif(n * 5), ncol = 5)
  pi_x <- 0.25 + 0.5 * X[, 1]
  Z <- rbinom(n, 1, pi_x)
  y <- pi_x * 5 + Z * X[, 2] * 2 + rnorm(n)

  # adaptive_coding = TRUE is only allowed for binary treatment, so a legacy
  # JSON missing binary_treatment must be inferred back to TRUE.
  bcf_model <- bcf(
    X_train = X, Z_train = Z, y_train = y, propensity_train = pi_x,
    num_gfr = 0, num_burnin = 0, num_mcmc = 10,
    general_params = list(adaptive_coding = TRUE)
  )
  json_new <- saveBCFModelToJsonString(bcf_model)

  # Simulate legacy JSON by renaming the fields so has_field() misses them.
  json_legacy <- gsub(
    '"binary_treatment"', '"binary_treatment_legacy"', json_new,
    fixed = TRUE
  )
  json_legacy <- gsub(
    '"treatment_dim"', '"treatment_dim_legacy"', json_legacy,
    fixed = TRUE
  )

  expect_warning(
    bcf_legacy <- createBCFModelFromJsonString(json_legacy),
    "binary_treatment"
  )
  expect_true(bcf_legacy$model_params$binary_treatment)
  expect_equal(bcf_legacy$model_params$treatment_dim, 1)
  expect_error(capture.output(print(bcf_legacy)), NA)
})

# Assert the reconstructed leaf-model fields and that print()/summary() work.
expect_bart_leaf_fields_ok <- function(reloaded, has_basis, leaf_dim, label) {
  mp <- reloaded$model_params
  expect_equal(mp$has_basis, has_basis, info = label)
  expect_equal(mp$leaf_regression, has_basis, info = label)
  expect_equal(mp$is_leaf_constant, !has_basis, info = label)
  expect_equal(mp$leaf_dimension, leaf_dim, info = label)
  expect_error(capture.output(print(reloaded)), NA)
  expect_error(capture.output(summary(reloaded)), NA)
}

test_that("BART round-trip reconstructs leaf-model fields and supports print/summary", {
  skip_on_cran()

  set.seed(396)
  n <- 200
  X <- matrix(runif(n * 5), ncol = 5)
  y <- X[, 1] + rnorm(n)

  # Constant-leaf model: has_basis = FALSE, leaf_dimension = 1
  m_const <- bart(
    X_train = X, y_train = y, num_gfr = 0, num_burnin = 0, num_mcmc = 10
  )
  json_const <- saveBARTModelToJsonString(m_const)
  expect_bart_leaf_fields_ok(
    createBARTModelFromJsonString(json_const), FALSE, 1, "constant/string"
  )
  expect_bart_leaf_fields_ok(
    createBARTModelFromCombinedJsonString(list(json_const)),
    FALSE, 1, "constant/combined"
  )

  # Regression-leaf (basis) model: has_basis = TRUE, leaf_dimension = ncol(W)
  W <- matrix(runif(n * 2), ncol = 2)
  m_reg <- bart(
    X_train = X, leaf_basis_train = W, y_train = y,
    num_gfr = 0, num_burnin = 0, num_mcmc = 10
  )
  json_reg <- saveBARTModelToJsonString(m_reg)
  expect_bart_leaf_fields_ok(
    createBARTModelFromJsonString(json_reg), TRUE, 2, "regression/string"
  )
  expect_bart_leaf_fields_ok(
    createBARTModelFromCombinedJsonString(list(json_reg)),
    TRUE, 2, "regression/combined"
  )
})

test_that("BART multi-chain and combined round-trips reconstruct fields", {
  skip_on_cran()

  set.seed(397)
  n <- 200
  X <- matrix(runif(n * 5), ncol = 5)
  y <- X[, 1] + rnorm(n)

  # Multi-chain single model (num_chains = 2): samples stored consecutively.
  m_mc <- bart(
    X_train = X, y_train = y, num_gfr = 0, num_burnin = 0, num_mcmc = 10,
    general_params = list(num_chains = 2)
  )
  expect_equal(m_mc$model_params$num_chains, 2)
  reloaded_mc <- createBARTModelFromJsonString(saveBARTModelToJsonString(m_mc))
  expect_equal(reloaded_mc$model_params$num_chains, 2)
  expect_bart_leaf_fields_ok(reloaded_mc, FALSE, 1, "bart/multichain")

  # Combine two separately-sampled models via the combined-string loader.
  m_a <- bart(X_train = X, y_train = y, num_gfr = 0, num_burnin = 0, num_mcmc = 10)
  m_b <- bart(X_train = X, y_train = y, num_gfr = 0, num_burnin = 0, num_mcmc = 10)
  combined <- createBARTModelFromCombinedJsonString(
    list(saveBARTModelToJsonString(m_a), saveBARTModelToJsonString(m_b))
  )
  expect_bart_leaf_fields_ok(combined, FALSE, 1, "bart/combined-two")
  expect_equal(combined$model_params$num_samples, 20)
})

test_that("BCF multi-chain and combined round-trips preserve contract fields", {
  skip_on_cran()

  set.seed(398)
  n <- 300
  X <- matrix(runif(n * 5), ncol = 5)
  pi_x <- 0.25 + 0.5 * X[, 1]
  Z <- rbinom(n, 1, pi_x)
  y <- pi_x * 5 + Z * X[, 2] * 2 + rnorm(n)

  # Multi-chain single model (supplied propensity).
  m_mc <- bcf(
    X_train = X, Z_train = Z, y_train = y, propensity_train = pi_x,
    num_gfr = 0, num_burnin = 0, num_mcmc = 10,
    general_params = list(num_chains = 2)
  )
  expect_equal(m_mc$model_params$num_chains, 2)
  expect_bcf_roundtrip_ok(
    m_mc, createBCFModelFromJsonString(saveBCFModelToJsonString(m_mc)),
    "bcf/multichain"
  )

  # Combine two separately-sampled models (num_samples is the sum).
  m_a <- bcf(
    X_train = X, Z_train = Z, y_train = y, propensity_train = pi_x,
    num_gfr = 0, num_burnin = 0, num_mcmc = 10
  )
  m_b <- bcf(
    X_train = X, Z_train = Z, y_train = y, propensity_train = pi_x,
    num_gfr = 0, num_burnin = 0, num_mcmc = 10
  )
  combined <- createBCFModelFromCombinedJsonString(
    list(saveBCFModelToJsonString(m_a), saveBCFModelToJsonString(m_b))
  )
  expect_true(combined$model_params$binary_treatment)
  expect_equal(combined$model_params$treatment_dim, 1)
  expect_equal(combined$model_params$num_samples, 20)
  expect_error(capture.output(print(combined)), NA)
})
