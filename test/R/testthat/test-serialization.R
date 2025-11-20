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
