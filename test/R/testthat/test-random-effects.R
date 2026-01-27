test_that("Random Effects BART with Default Numbering", {
  skip_on_cran()

  # Generate simulated data
  n <- 100
  p <- 5
  p_w <- 2
  X <- matrix(runif(n * p), ncol = p)
  W <- matrix(runif(n * p_w), ncol = p_w)
  # fmt: skip
  f_XW <- (((0 <= X[, 1]) & (0.25 > X[, 1])) * (-7.5 * W[, 1]) +
    ((0.25 <= X[, 1]) & (0.5 > X[, 1])) * (-2.5 * W[, 1]) +
    ((0.5 <= X[, 1]) & (0.75 > X[, 1])) * (2.5 * W[, 1]) +
    ((0.75 <= X[, 1]) & (1 > X[, 1])) * (7.5 * W[, 1]))
  rfx_group_ids <- sample(1:2, size = n, replace = TRUE)
  rfx_basis <- cbind(rep(1, n), runif(n))
  num_rfx_components <- ncol(rfx_basis)
  num_rfx_groups <- length(unique(rfx_group_ids))
  rfx_coefs <- matrix(c(-5, 5, 1, -1), ncol = 2, byrow = T)
  rfx_term <- rowSums(rfx_coefs[rfx_group_ids, ] * rfx_basis)
  noise_sd <- 1
  y <- f_XW + rfx_term + rnorm(n, 0, noise_sd)
  test_set_pct <- 0.2
  n_test <- round(test_set_pct * n)
  n_train <- n - n_test
  test_inds <- sort(sample(1:n, n_test, replace = FALSE))
  train_inds <- (1:n)[!((1:n) %in% test_inds)]
  X_test <- X[test_inds, ]
  X_train <- X[train_inds, ]
  W_test <- W[test_inds, ]
  W_train <- W[train_inds, ]
  rfx_group_ids_test <- rfx_group_ids[test_inds]
  rfx_group_ids_train <- rfx_group_ids[train_inds]
  rfx_basis_test <- rfx_basis[test_inds, ]
  rfx_basis_train <- rfx_basis[train_inds, ]
  y_test <- y[test_inds]
  y_train <- y[train_inds]

  # Specify no rfx parameters
  general_param_list <- list()
  mean_forest_param_list <- list(sample_sigma2_leaf = FALSE)
  expect_no_error(
    bart_model <- bart(
      X_train = X_train,
      y_train = y_train,
      X_test = X_test,
      leaf_basis_train = W_train,
      leaf_basis_test = W_test,
      rfx_group_ids_train = rfx_group_ids_train,
      rfx_group_ids_test = rfx_group_ids_test,
      rfx_basis_train = rfx_basis_train,
      rfx_basis_test = rfx_basis_test,
      num_gfr = 0,
      num_burnin = 10,
      num_mcmc = 10,
      general_params = general_param_list,
      mean_forest_params = mean_forest_param_list
    )
  )

  # Specify all rfx parameters as scalars
  rfx_param_list <- list(
    working_parameter_prior_mean = 1.,
    group_parameter_prior_mean = 1.,
    working_parameter_prior_cov = 1.,
    group_parameter_prior_cov = 1.,
    variance_prior_shape = 1,
    variance_prior_scale = 1
  )
  mean_forest_param_list <- list(sample_sigma2_leaf = FALSE)
  expect_no_error(
    bart_model <- bart(
      X_train = X_train,
      y_train = y_train,
      X_test = X_test,
      leaf_basis_train = W_train,
      leaf_basis_test = W_test,
      rfx_group_ids_train = rfx_group_ids_train,
      rfx_group_ids_test = rfx_group_ids_test,
      rfx_basis_train = rfx_basis_train,
      rfx_basis_test = rfx_basis_test,
      num_gfr = 0,
      num_burnin = 10,
      num_mcmc = 10,
      mean_forest_params = mean_forest_param_list,
      random_effects_params = rfx_param_list
    )
  )

  # Specify all relevant rfx parameters as vectors
  rfx_param_list <- list(
    working_parameter_prior_mean = c(1., 1.),
    group_parameter_prior_mean = c(1., 1.),
    working_parameter_prior_cov = diag(1., 2),
    group_parameter_prior_cov = diag(1., 2),
    variance_prior_shape = 1,
    variance_prior_scale = 1
  )
  mean_forest_param_list <- list(sample_sigma2_leaf = FALSE)
  expect_no_error(
    bart_model <- bart(
      X_train = X_train,
      y_train = y_train,
      X_test = X_test,
      leaf_basis_train = W_train,
      leaf_basis_test = W_test,
      rfx_group_ids_train = rfx_group_ids_train,
      rfx_group_ids_test = rfx_group_ids_test,
      rfx_basis_train = rfx_basis_train,
      rfx_basis_test = rfx_basis_test,
      num_gfr = 0,
      num_burnin = 10,
      num_mcmc = 10,
      mean_forest_params = mean_forest_param_list,
      random_effects_params = rfx_param_list
    )
  )

  # Specify simpler intercept-only RFX model
  rfx_param_list <- list(
    model_spec = "intercept_only"
  )
  mean_forest_param_list <- list(sample_sigma2_leaf = FALSE)
  expect_no_error({
    bart_model <- bart(
      X_train = X_train,
      y_train = y_train,
      X_test = X_test,
      leaf_basis_train = W_train,
      leaf_basis_test = W_test,
      rfx_group_ids_train = rfx_group_ids_train,
      rfx_group_ids_test = rfx_group_ids_test,
      num_gfr = 0,
      num_burnin = 10,
      num_mcmc = 10,
      mean_forest_params = mean_forest_param_list,
      random_effects_params = rfx_param_list
    )
    preds <- predict(
      bart_model,
      X = X_test,
      leaf_basis = W_test,
      rfx_group_ids = rfx_group_ids_test,
      type = "posterior",
      terms = "rfx"
    )
  })
})

test_that("Random Effects BART with Default Numbering", {
  skip_on_cran()

  # Generate simulated data
  n <- 100
  p <- 5
  p_w <- 2
  X <- matrix(runif(n * p), ncol = p)
  W <- matrix(runif(n * p_w), ncol = p_w)
  # fmt: skip
  f_XW <- (((0 <= X[, 1]) & (0.25 > X[, 1])) * (-7.5 * W[, 1]) +
    ((0.25 <= X[, 1]) & (0.5 > X[, 1])) * (-2.5 * W[, 1]) +
    ((0.5 <= X[, 1]) & (0.75 > X[, 1])) * (2.5 * W[, 1]) +
    ((0.75 <= X[, 1]) & (1 > X[, 1])) * (7.5 * W[, 1]))
  rfx_group_ids <- sample(1:2, size = n, replace = TRUE)
  rfx_basis <- cbind(rep(1, n), runif(n))
  num_rfx_components <- ncol(rfx_basis)
  num_rfx_groups <- length(unique(rfx_group_ids))
  rfx_coefs <- matrix(c(-5, 5, 1, -1), ncol = 2, byrow = T)
  rfx_term <- rowSums(rfx_coefs[rfx_group_ids, ] * rfx_basis)
  noise_sd <- 1
  y <- f_XW + rfx_term + rnorm(n, 0, noise_sd)
  test_set_pct <- 0.2
  n_test <- round(test_set_pct * n)
  n_train <- n - n_test
  test_inds <- sort(sample(1:n, n_test, replace = FALSE))
  train_inds <- (1:n)[!((1:n) %in% test_inds)]
  X_test <- X[test_inds, ]
  X_train <- X[train_inds, ]
  W_test <- W[test_inds, ]
  W_train <- W[train_inds, ]
  rfx_group_ids_test <- rfx_group_ids[test_inds]
  rfx_group_ids_train <- rfx_group_ids[train_inds]
  rfx_basis_test <- rfx_basis[test_inds, ]
  rfx_basis_train <- rfx_basis[train_inds, ]
  y_test <- y[test_inds]
  y_train <- y[train_inds]

  # Specify no rfx basis directly
  mean_forest_param_list <- list(sample_sigma2_leaf = FALSE)
  expect_no_error({
    bart_model <- bart(
      X_train = X_train,
      y_train = y_train,
      X_test = X_test,
      leaf_basis_train = W_train,
      leaf_basis_test = W_test,
      rfx_group_ids_train = rfx_group_ids_train,
      rfx_group_ids_test = rfx_group_ids_test,
      rfx_basis_train = rfx_basis_train,
      rfx_basis_test = rfx_basis_test,
      num_gfr = 0,
      num_burnin = 10,
      num_mcmc = 10,
      mean_forest_params = mean_forest_param_list
    )
    preds <- predict(
      bart_model,
      X = X_test,
      leaf_basis = W_test,
      rfx_group_ids = rfx_group_ids_test,
      rfx_basis = rfx_basis_test,
      type = "posterior",
      terms = "rfx"
    )
  })

  # Intercept-only RFX model
  rfx_param_list <- list(
    model_spec = "intercept_only"
  )
  mean_forest_param_list <- list(sample_sigma2_leaf = FALSE)
  expect_no_error({
    bart_model <- bart(
      X_train = X_train,
      y_train = y_train,
      X_test = X_test,
      leaf_basis_train = W_train,
      leaf_basis_test = W_test,
      rfx_group_ids_train = rfx_group_ids_train,
      rfx_group_ids_test = rfx_group_ids_test,
      num_gfr = 0,
      num_burnin = 10,
      num_mcmc = 10,
      mean_forest_params = mean_forest_param_list,
      random_effects_params = rfx_param_list
    )
    preds <- predict(
      bart_model,
      X = X_test,
      leaf_basis = W_test,
      rfx_group_ids = rfx_group_ids_test,
      type = "posterior",
      terms = "rfx"
    )
  })
})

test_that("Random Effects BART with Offset Numbering", {
  skip_on_cran()

  # Generate simulated data
  n <- 100
  p <- 5
  p_w <- 2
  X <- matrix(runif(n * p), ncol = p)
  W <- matrix(runif(n * p_w), ncol = p_w)
  # fmt: skip
  f_XW <- (((0 <= X[, 1]) & (0.25 > X[, 1])) * (-7.5 * W[, 1]) +
    ((0.25 <= X[, 1]) & (0.5 > X[, 1])) * (-2.5 * W[, 1]) +
    ((0.5 <= X[, 1]) & (0.75 > X[, 1])) * (2.5 * W[, 1]) +
    ((0.75 <= X[, 1]) & (1 > X[, 1])) * (7.5 * W[, 1]))
  rfx_group_ids <- sample(1:2, size = n, replace = TRUE) + 2
  rfx_basis <- cbind(rep(1, n), runif(n))
  num_rfx_components <- ncol(rfx_basis)
  num_rfx_groups <- length(unique(rfx_group_ids))
  rfx_coefs <- matrix(c(-5, 5, 1, -1), ncol = 2, byrow = T)
  rfx_term <- rowSums(rfx_coefs[rfx_group_ids - 2, ] * rfx_basis)
  noise_sd <- 1
  y <- f_XW + rfx_term + rnorm(n, 0, noise_sd)
  test_set_pct <- 0.2
  n_test <- round(test_set_pct * n)
  n_train <- n - n_test
  test_inds <- sort(sample(1:n, n_test, replace = FALSE))
  train_inds <- (1:n)[!((1:n) %in% test_inds)]
  X_test <- X[test_inds, ]
  X_train <- X[train_inds, ]
  W_test <- W[test_inds, ]
  W_train <- W[train_inds, ]
  rfx_group_ids_test <- rfx_group_ids[test_inds]
  rfx_group_ids_train <- rfx_group_ids[train_inds]
  rfx_basis_test <- rfx_basis[test_inds, ]
  rfx_basis_train <- rfx_basis[train_inds, ]
  y_test <- y[test_inds]
  y_train <- y[train_inds]

  # Specify no rfx basis directly
  mean_forest_param_list <- list(sample_sigma2_leaf = FALSE)
  expect_no_error({
    bart_model <- bart(
      X_train = X_train,
      y_train = y_train,
      X_test = X_test,
      leaf_basis_train = W_train,
      leaf_basis_test = W_test,
      rfx_group_ids_train = rfx_group_ids_train,
      rfx_group_ids_test = rfx_group_ids_test,
      rfx_basis_train = rfx_basis_train,
      rfx_basis_test = rfx_basis_test,
      num_gfr = 0,
      num_burnin = 10,
      num_mcmc = 10,
      mean_forest_params = mean_forest_param_list
    )
    rfx_preds <- predict(
      bart_model,
      X = X_test,
      leaf_basis = W_test,
      rfx_group_ids = rfx_group_ids_test,
      rfx_basis = rfx_basis_test,
      type = "posterior",
      terms = "rfx"
    )
    yhat_preds <- predict(
      bart_model,
      X = X_test,
      leaf_basis = W_test,
      rfx_group_ids = rfx_group_ids_test,
      rfx_basis = rfx_basis_test,
      type = "posterior",
      terms = "y_hat"
    )
  })

  # Intercept-only RFX model
  rfx_param_list <- list(
    model_spec = "intercept_only"
  )
  mean_forest_param_list <- list(sample_sigma2_leaf = FALSE)
  expect_no_error({
    bart_model <- bart(
      X_train = X_train,
      y_train = y_train,
      X_test = X_test,
      leaf_basis_train = W_train,
      leaf_basis_test = W_test,
      rfx_group_ids_train = rfx_group_ids_train,
      rfx_group_ids_test = rfx_group_ids_test,
      num_gfr = 0,
      num_burnin = 10,
      num_mcmc = 10,
      mean_forest_params = mean_forest_param_list,
      random_effects_params = rfx_param_list
    )
    rfx_preds <- predict(
      bart_model,
      X = X_test,
      leaf_basis = W_test,
      rfx_group_ids = rfx_group_ids_test,
      type = "posterior",
      terms = "rfx"
    )
    yhat_preds <- predict(
      bart_model,
      X = X_test,
      leaf_basis = W_test,
      rfx_group_ids = rfx_group_ids_test,
      type = "posterior",
      terms = "y_hat"
    )
  })
})

test_that("Random Effects BCF with Default Numbering", {
  skip_on_cran()

  # Generate simulated data
  n <- 100
  p <- 5
  X <- matrix(runif(n * p), ncol = p)
  # fmt: skip
  mu_X <- (((0 <= X[, 1]) & (0.25 > X[, 1])) * (-7.5) +
    ((0.25 <= X[, 1]) & (0.5 > X[, 1])) * (-2.5) +
    ((0.5 <= X[, 1]) & (0.75 > X[, 1])) * (2.5) +
    ((0.75 <= X[, 1]) & (1 > X[, 1])) * (7.5))
  # fmt: skip
  pi_X <- (((0 <= X[, 1]) & (0.25 > X[, 1])) * (0.2) +
    ((0.25 <= X[, 1]) & (0.5 > X[, 1])) * (0.4) +
    ((0.5 <= X[, 1]) & (0.75 > X[, 1])) * (0.6) +
    ((0.75 <= X[, 1]) & (1 > X[, 1])) * (0.8))
  # fmt: skip
  tau_X <- (((0 <= X[, 2]) & (0.25 > X[, 2])) * (0.5) +
    ((0.25 <= X[, 2]) & (0.5 > X[, 2])) * (1.0) +
    ((0.5 <= X[, 2]) & (0.75 > X[, 2])) * (1.5) +
    ((0.75 <= X[, 2]) & (1 > X[, 2])) * (2.0))
  Z <- rbinom(n, 1, pi_X)
  rfx_group_ids <- sample(1:2, size = n, replace = TRUE)
  rfx_basis <- cbind(rep(1, n), runif(n))
  num_rfx_components <- ncol(rfx_basis)
  num_rfx_groups <- length(unique(rfx_group_ids))
  rfx_coefs <- matrix(c(-5, 5, 1, -1), ncol = 2, byrow = T)
  rfx_term <- rowSums(rfx_coefs[rfx_group_ids, ] * rfx_basis)
  noise_sd <- 1
  y <- mu_X + tau_X * Z + rfx_term + rnorm(n, 0, noise_sd)

  test_set_pct <- 0.2
  n_test <- round(test_set_pct * n)
  n_train <- n - n_test
  test_inds <- sort(sample(1:n, n_test, replace = FALSE))
  train_inds <- (1:n)[!((1:n) %in% test_inds)]
  X_test <- X[test_inds, ]
  X_train <- X[train_inds, ]
  Z_test <- Z[test_inds]
  Z_train <- Z[train_inds]
  pi_test <- pi_X[test_inds]
  pi_train <- pi_X[train_inds]
  mu_test <- mu_X[test_inds]
  mu_train <- mu_X[train_inds]
  tau_test <- tau_X[test_inds]
  tau_train <- tau_X[train_inds]
  rfx_group_ids_test <- rfx_group_ids[test_inds]
  rfx_group_ids_train <- rfx_group_ids[train_inds]
  rfx_basis_test <- rfx_basis[test_inds, ]
  rfx_basis_train <- rfx_basis[train_inds, ]
  y_test <- y[test_inds]
  y_train <- y[train_inds]

  # Fit a BCF model with random effects by passing the basis directly
  expect_no_error({
    bcf_model <- bcf(
      X_train = X_train,
      y_train = y_train,
      Z_train = Z_train,
      propensity_train = pi_train,
      rfx_group_ids_train = rfx_group_ids_train,
      rfx_basis_train = rfx_basis_train,
      X_test = X_test,
      Z_test = Z_test,
      propensity_test = pi_test,
      rfx_group_ids_test = rfx_group_ids_test,
      rfx_basis_test = rfx_basis_test,
      num_gfr = 10,
      num_burnin = 0,
      num_mcmc = 10
    )
    rfx_preds <- predict(
      bcf_model,
      X = X_test,
      Z = Z_test,
      propensity = pi_test,
      rfx_group_ids = rfx_group_ids_test,
      rfx_basis = rfx_basis_test,
      type = "posterior",
      terms = "rfx"
    )
    yhat_preds <- predict(
      bcf_model,
      X = X_test,
      Z = Z_test,
      propensity = pi_test,
      rfx_group_ids = rfx_group_ids_test,
      rfx_basis = rfx_basis_test,
      type = "posterior",
      terms = "y_hat"
    )
  })

  # Fit a BCF model with random effects by specifying an "intercept only" model
  expect_no_error({
    rfx_param_list <- list(
      model_spec = "intercept_only"
    )
    bcf_model <- bcf(
      X_train = X_train,
      y_train = y_train,
      Z_train = Z_train,
      propensity_train = pi_train,
      rfx_group_ids_train = rfx_group_ids_train,
      X_test = X_test,
      Z_test = Z_test,
      propensity_test = pi_test,
      rfx_group_ids_test = rfx_group_ids_test,
      num_gfr = 10,
      num_burnin = 0,
      num_mcmc = 10,
      random_effects_params = rfx_param_list
    )
    rfx_preds <- predict(
      bcf_model,
      X = X_test,
      Z = Z_test,
      propensity = pi_test,
      rfx_group_ids = rfx_group_ids_test,
      type = "posterior",
      terms = "rfx"
    )
    yhat_preds <- predict(
      bcf_model,
      X = X_test,
      Z = Z_test,
      propensity = pi_test,
      rfx_group_ids = rfx_group_ids_test,
      type = "posterior",
      terms = "y_hat"
    )
  })

  # Fit a BCF model with random effects by specifying an "intercept plus treatment" model
  expect_no_error({
    rfx_param_list <- list(
      model_spec = "intercept_plus_treatment"
    )
    bcf_model <- bcf(
      X_train = X_train,
      y_train = y_train,
      Z_train = Z_train,
      propensity_train = pi_train,
      rfx_group_ids_train = rfx_group_ids_train,
      X_test = X_test,
      Z_test = Z_test,
      propensity_test = pi_test,
      rfx_group_ids_test = rfx_group_ids_test,
      num_gfr = 10,
      num_burnin = 0,
      num_mcmc = 10,
      random_effects_params = rfx_param_list
    )
    rfx_preds <- predict(
      bcf_model,
      X = X_test,
      Z = Z_test,
      propensity = pi_test,
      rfx_group_ids = rfx_group_ids_test,
      type = "posterior",
      terms = "rfx"
    )
    yhat_preds <- predict(
      bcf_model,
      X = X_test,
      Z = Z_test,
      propensity = pi_test,
      rfx_group_ids = rfx_group_ids_test,
      type = "posterior",
      terms = "y_hat"
    )
  })
})

test_that("Random Effects BCF with Offset Numbering", {
  skip_on_cran()

  # Generate simulated data
  n <- 100
  p <- 5
  X <- matrix(runif(n * p), ncol = p)
  # fmt: skip
  mu_X <- (((0 <= X[, 1]) & (0.25 > X[, 1])) * (-7.5) +
    ((0.25 <= X[, 1]) & (0.5 > X[, 1])) * (-2.5) +
    ((0.5 <= X[, 1]) & (0.75 > X[, 1])) * (2.5) +
    ((0.75 <= X[, 1]) & (1 > X[, 1])) * (7.5))
  # fmt: skip
  pi_X <- (((0 <= X[, 1]) & (0.25 > X[, 1])) * (0.2) +
    ((0.25 <= X[, 1]) & (0.5 > X[, 1])) * (0.4) +
    ((0.5 <= X[, 1]) & (0.75 > X[, 1])) * (0.6) +
    ((0.75 <= X[, 1]) & (1 > X[, 1])) * (0.8))
  # fmt: skip
  tau_X <- (((0 <= X[, 2]) & (0.25 > X[, 2])) * (0.5) +
    ((0.25 <= X[, 2]) & (0.5 > X[, 2])) * (1.0) +
    ((0.5 <= X[, 2]) & (0.75 > X[, 2])) * (1.5) +
    ((0.75 <= X[, 2]) & (1 > X[, 2])) * (2.0))
  Z <- rbinom(n, 1, pi_X)
  rfx_group_ids <- sample(1:2, size = n, replace = TRUE) + 2
  rfx_basis <- cbind(rep(1, n), runif(n))
  num_rfx_components <- ncol(rfx_basis)
  num_rfx_groups <- length(unique(rfx_group_ids))
  rfx_coefs <- matrix(c(-5, 5, 1, -1), ncol = 2, byrow = T)
  rfx_term <- rowSums(rfx_coefs[rfx_group_ids - 2, ] * rfx_basis)
  noise_sd <- 1
  y <- mu_X + tau_X * Z + rfx_term + rnorm(n, 0, noise_sd)

  test_set_pct <- 0.2
  n_test <- round(test_set_pct * n)
  n_train <- n - n_test
  test_inds <- sort(sample(1:n, n_test, replace = FALSE))
  train_inds <- (1:n)[!((1:n) %in% test_inds)]
  X_test <- X[test_inds, ]
  X_train <- X[train_inds, ]
  Z_test <- Z[test_inds]
  Z_train <- Z[train_inds]
  pi_test <- pi_X[test_inds]
  pi_train <- pi_X[train_inds]
  mu_test <- mu_X[test_inds]
  mu_train <- mu_X[train_inds]
  tau_test <- tau_X[test_inds]
  tau_train <- tau_X[train_inds]
  rfx_group_ids_test <- rfx_group_ids[test_inds]
  rfx_group_ids_train <- rfx_group_ids[train_inds]
  rfx_basis_test <- rfx_basis[test_inds, ]
  rfx_basis_train <- rfx_basis[train_inds, ]
  y_test <- y[test_inds]
  y_train <- y[train_inds]

  # Fit a BCF model with random effects by passing the basis directly
  expect_no_error({
    bcf_model <- bcf(
      X_train = X_train,
      y_train = y_train,
      Z_train = Z_train,
      propensity_train = pi_train,
      rfx_group_ids_train = rfx_group_ids_train,
      rfx_basis_train = rfx_basis_train,
      X_test = X_test,
      Z_test = Z_test,
      propensity_test = pi_test,
      rfx_group_ids_test = rfx_group_ids_test,
      rfx_basis_test = rfx_basis_test,
      num_gfr = 10,
      num_burnin = 0,
      num_mcmc = 10
    )
    rfx_preds <- predict(
      bcf_model,
      X = X_test,
      Z = Z_test,
      propensity = pi_test,
      rfx_group_ids = rfx_group_ids_test,
      rfx_basis = rfx_basis_test,
      type = "posterior",
      terms = "rfx"
    )
    yhat_preds <- predict(
      bcf_model,
      X = X_test,
      Z = Z_test,
      propensity = pi_test,
      rfx_group_ids = rfx_group_ids_test,
      rfx_basis = rfx_basis_test,
      type = "posterior",
      terms = "y_hat"
    )
  })

  # Fit a BCF model with random effects by specifying an "intercept only" model
  expect_no_error({
    rfx_param_list <- list(
      model_spec = "intercept_only"
    )
    bcf_model <- bcf(
      X_train = X_train,
      y_train = y_train,
      Z_train = Z_train,
      propensity_train = pi_train,
      rfx_group_ids_train = rfx_group_ids_train,
      X_test = X_test,
      Z_test = Z_test,
      propensity_test = pi_test,
      rfx_group_ids_test = rfx_group_ids_test,
      num_gfr = 10,
      num_burnin = 0,
      num_mcmc = 10,
      random_effects_params = rfx_param_list
    )
    rfx_preds <- predict(
      bcf_model,
      X = X_test,
      Z = Z_test,
      propensity = pi_test,
      rfx_group_ids = rfx_group_ids_test,
      type = "posterior",
      terms = "rfx"
    )
    yhat_preds <- predict(
      bcf_model,
      X = X_test,
      Z = Z_test,
      propensity = pi_test,
      rfx_group_ids = rfx_group_ids_test,
      type = "posterior",
      terms = "y_hat"
    )
  })

  # Fit a BCF model with random effects by specifying an "intercept plus treatment" model
  expect_no_error({
    rfx_param_list <- list(
      model_spec = "intercept_plus_treatment"
    )
    bcf_model <- bcf(
      X_train = X_train,
      y_train = y_train,
      Z_train = Z_train,
      propensity_train = pi_train,
      rfx_group_ids_train = rfx_group_ids_train,
      X_test = X_test,
      Z_test = Z_test,
      propensity_test = pi_test,
      rfx_group_ids_test = rfx_group_ids_test,
      num_gfr = 10,
      num_burnin = 0,
      num_mcmc = 10,
      random_effects_params = rfx_param_list
    )
    rfx_preds <- predict(
      bcf_model,
      X = X_test,
      Z = Z_test,
      propensity = pi_test,
      rfx_group_ids = rfx_group_ids_test,
      type = "posterior",
      terms = "rfx"
    )
    yhat_preds <- predict(
      bcf_model,
      X = X_test,
      Z = Z_test,
      propensity = pi_test,
      rfx_group_ids = rfx_group_ids_test,
      type = "posterior",
      terms = "y_hat"
    )
  })
})
