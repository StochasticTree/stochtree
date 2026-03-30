test_that("MCMC BCF", {
  skip_on_cran()

  # Generate simulated data
  n <- 100
  p <- 5
  X <- matrix(runif(n * p), ncol = p)
  mu_X <- (((0 <= X[, 1]) & (0.25 > X[, 1])) *
    (-7.5) +
    ((0.25 <= X[, 1]) & (0.5 > X[, 1])) * (-2.5) +
    ((0.5 <= X[, 1]) & (0.75 > X[, 1])) * (2.5) +
    ((0.75 <= X[, 1]) & (1 > X[, 1])) * (7.5))
  pi_X <- (((0 <= X[, 1]) & (0.25 > X[, 1])) *
    (0.2) +
    ((0.25 <= X[, 1]) & (0.5 > X[, 1])) * (0.4) +
    ((0.5 <= X[, 1]) & (0.75 > X[, 1])) * (0.6) +
    ((0.75 <= X[, 1]) & (1 > X[, 1])) * (0.8))
  tau_X <- (((0 <= X[, 2]) & (0.25 > X[, 2])) *
    (0.5) +
    ((0.25 <= X[, 2]) & (0.5 > X[, 2])) * (1.0) +
    ((0.5 <= X[, 2]) & (0.75 > X[, 2])) * (1.5) +
    ((0.75 <= X[, 2]) & (1 > X[, 2])) * (2.0))
  Z <- rbinom(n, 1, pi_X)
  noise_sd <- 1
  y <- mu_X + tau_X * Z + rnorm(n, 0, noise_sd)
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
  y_test <- y[test_inds]
  y_train <- y[train_inds]

  # 1 chain, no thinning
  general_param_list <- list(num_chains = 1, keep_every = 1)
  expect_no_error(
    bcf_model <- bcf(
      X_train = X_train,
      y_train = y_train,
      Z_train = Z_train,
      propensity_train = pi_train,
      X_test = X_test,
      Z_test = Z_test,
      propensity_test = pi_test,
      num_gfr = 0,
      num_burnin = 10,
      num_mcmc = 10,
      general_params = general_param_list
    )
  )

  # 1 chain, no thinning, matrix leaf scale parameter provided
  general_param_list <- list(num_chains = 1, keep_every = 1)
  mu_forest_param_list <- list(sigma2_leaf_init = as.matrix(0.5))
  tau_forest_param_list <- list(sigma2_leaf_init = as.matrix(0.5))
  expect_no_error(
    bcf_model <- bcf(
      X_train = X_train,
      y_train = y_train,
      Z_train = Z_train,
      propensity_train = pi_train,
      X_test = X_test,
      Z_test = Z_test,
      propensity_test = pi_test,
      num_gfr = 0,
      num_burnin = 10,
      num_mcmc = 10,
      general_params = general_param_list,
      prognostic_forest_params = mu_forest_param_list,
      treatment_effect_forest_params = tau_forest_param_list
    )
  )

  # 1 chain, no thinning, scalar leaf scale parameter provided
  general_param_list <- list(num_chains = 1, keep_every = 1)
  mu_forest_param_list <- list(sigma2_leaf_init = 0.5)
  tau_forest_param_list <- list(sigma2_leaf_init = 0.5)
  expect_no_error(
    bcf_model <- bcf(
      X_train = X_train,
      y_train = y_train,
      Z_train = Z_train,
      propensity_train = pi_train,
      X_test = X_test,
      Z_test = Z_test,
      propensity_test = pi_test,
      num_gfr = 0,
      num_burnin = 10,
      num_mcmc = 10,
      general_params = general_param_list,
      prognostic_forest_params = mu_forest_param_list,
      treatment_effect_forest_params = tau_forest_param_list
    )
  )

  # 3 chains, no thinning
  general_param_list <- list(num_chains = 3, keep_every = 1)
  expect_no_error(
    bcf_model <- bcf(
      X_train = X_train,
      y_train = y_train,
      Z_train = Z_train,
      propensity_train = pi_train,
      X_test = X_test,
      Z_test = Z_test,
      propensity_test = pi_test,
      num_gfr = 0,
      num_burnin = 10,
      num_mcmc = 10,
      general_params = general_param_list
    )
  )

  # 1 chain, thinning
  general_param_list <- list(num_chains = 1, keep_every = 5)
  expect_no_error(
    bcf_model <- bcf(
      X_train = X_train,
      y_train = y_train,
      Z_train = Z_train,
      propensity_train = pi_train,
      X_test = X_test,
      Z_test = Z_test,
      propensity_test = pi_test,
      num_gfr = 0,
      num_burnin = 10,
      num_mcmc = 10,
      general_params = general_param_list
    )
  )

  # 3 chains, thinning
  general_param_list <- list(num_chains = 3, keep_every = 5)
  expect_no_error(
    bcf_model <- bcf(
      X_train = X_train,
      y_train = y_train,
      Z_train = Z_train,
      propensity_train = pi_train,
      X_test = X_test,
      Z_test = Z_test,
      propensity_test = pi_test,
      num_gfr = 0,
      num_burnin = 10,
      num_mcmc = 10,
      general_params = general_param_list
    )
  )
})

test_that("GFR BCF", {
  skip_on_cran()

  # Generate simulated data
  n <- 100
  p <- 5
  X <- matrix(runif(n * p), ncol = p)
  mu_X <- (((0 <= X[, 1]) & (0.25 > X[, 1])) *
    (-7.5) +
    ((0.25 <= X[, 1]) & (0.5 > X[, 1])) * (-2.5) +
    ((0.5 <= X[, 1]) & (0.75 > X[, 1])) * (2.5) +
    ((0.75 <= X[, 1]) & (1 > X[, 1])) * (7.5))
  pi_X <- (((0 <= X[, 1]) & (0.25 > X[, 1])) *
    (0.2) +
    ((0.25 <= X[, 1]) & (0.5 > X[, 1])) * (0.4) +
    ((0.5 <= X[, 1]) & (0.75 > X[, 1])) * (0.6) +
    ((0.75 <= X[, 1]) & (1 > X[, 1])) * (0.8))
  tau_X <- (((0 <= X[, 2]) & (0.25 > X[, 2])) *
    (0.5) +
    ((0.25 <= X[, 2]) & (0.5 > X[, 2])) * (1.0) +
    ((0.5 <= X[, 2]) & (0.75 > X[, 2])) * (1.5) +
    ((0.75 <= X[, 2]) & (1 > X[, 2])) * (2.0))
  Z <- rbinom(n, 1, pi_X)
  noise_sd <- 1
  y <- mu_X + tau_X * Z + rnorm(n, 0, noise_sd)
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
  y_test <- y[test_inds]
  y_train <- y[train_inds]

  # 1 chain, no thinning
  general_param_list <- list(num_chains = 1, keep_every = 1)
  expect_no_error(
    bcf_model <- bcf(
      X_train = X_train,
      y_train = y_train,
      Z_train = Z_train,
      propensity_train = pi_train,
      X_test = X_test,
      Z_test = Z_test,
      propensity_test = pi_test,
      num_gfr = 10,
      num_burnin = 10,
      num_mcmc = 10,
      general_params = general_param_list
    )
  )

  # 3 chains, no thinning
  general_param_list <- list(num_chains = 3, keep_every = 1)
  expect_no_error(
    bcf_model <- bcf(
      X_train = X_train,
      y_train = y_train,
      Z_train = Z_train,
      propensity_train = pi_train,
      X_test = X_test,
      Z_test = Z_test,
      propensity_test = pi_test,
      num_gfr = 10,
      num_burnin = 10,
      num_mcmc = 10,
      general_params = general_param_list
    )
  )

  # 1 chain, thinning
  general_param_list <- list(num_chains = 1, keep_every = 5)
  expect_no_error(
    bcf_model <- bcf(
      X_train = X_train,
      y_train = y_train,
      Z_train = Z_train,
      propensity_train = pi_train,
      X_test = X_test,
      Z_test = Z_test,
      propensity_test = pi_test,
      num_gfr = 10,
      num_burnin = 10,
      num_mcmc = 10,
      general_params = general_param_list
    )
  )

  # 3 chains, thinning
  general_param_list <- list(num_chains = 3, keep_every = 5)
  expect_no_error(
    bcf_model <- bcf(
      X_train = X_train,
      y_train = y_train,
      Z_train = Z_train,
      propensity_train = pi_train,
      X_test = X_test,
      Z_test = Z_test,
      propensity_test = pi_test,
      num_gfr = 10,
      num_burnin = 10,
      num_mcmc = 10,
      general_params = general_param_list
    )
  )

  # Check for error when more chains than GFR forests
  general_param_list <- list(num_chains = 11, keep_every = 1)
  expect_error(
    bcf_model <- bcf(
      X_train = X_train,
      y_train = y_train,
      Z_train = Z_train,
      propensity_train = pi_train,
      X_test = X_test,
      Z_test = Z_test,
      propensity_test = pi_test,
      num_gfr = 10,
      num_burnin = 10,
      num_mcmc = 10,
      general_params = general_param_list
    )
  )

  # Check for error when more chains than GFR forests
  general_param_list <- list(num_chains = 11, keep_every = 5)
  expect_error(
    bcf_model <- bcf(
      X_train = X_train,
      y_train = y_train,
      Z_train = Z_train,
      propensity_train = pi_train,
      X_test = X_test,
      Z_test = Z_test,
      propensity_test = pi_test,
      num_gfr = 10,
      num_burnin = 10,
      num_mcmc = 10,
      general_params = general_param_list
    )
  )
})

test_that("Warmstart BCF", {
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
  noise_sd <- 1
  y <- mu_X + tau_X * Z + rnorm(n, 0, noise_sd)
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
  y_test <- y[test_inds]
  y_train <- y[train_inds]

  # Run a BCF model with only GFR
  general_param_list <- list(num_chains = 1, keep_every = 1)
  bcf_model <- bcf(
    X_train = X_train,
    y_train = y_train,
    Z_train = Z_train,
    propensity_train = pi_train,
    X_test = X_test,
    Z_test = Z_test,
    propensity_test = pi_test,
    num_gfr = 10,
    num_burnin = 0,
    num_mcmc = 0,
    general_params = general_param_list
  )

  # Save to JSON string
  bcf_model_json_string <- saveBCFModelToJsonString(bcf_model)

  # Run a new BCF chain from the existing (X)BCF model
  general_param_list <- list(num_chains = 3, keep_every = 5)
  expect_no_error(
    bcf_model <- bcf(
      X_train = X_train,
      y_train = y_train,
      Z_train = Z_train,
      propensity_train = pi_train,
      X_test = X_test,
      Z_test = Z_test,
      propensity_test = pi_test,
      num_gfr = 0,
      num_burnin = 10,
      num_mcmc = 10,
      previous_model_json = bcf_model_json_string,
      previous_model_warmstart_sample_num = 10,
      general_params = general_param_list
    )
  )
  expect_warning(
    bcf_model <- bcf(
      X_train = X_train,
      y_train = y_train,
      Z_train = Z_train,
      propensity_train = pi_train,
      X_test = X_test,
      Z_test = Z_test,
      propensity_test = pi_test,
      num_gfr = 0,
      num_burnin = 10,
      num_mcmc = 10,
      previous_model_json = bcf_model_json_string,
      previous_model_warmstart_sample_num = 1,
      general_params = general_param_list
    )
  )

  # Generate simulated data with random effects
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
  rfx_basis <- rep(1, n)
  rfx_coefs <- c(-5, 5)
  rfx_term <- rfx_coefs[rfx_group_ids] * rfx_basis
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
  rfx_basis_test <- rfx_basis[test_inds]
  rfx_basis_train <- rfx_basis[train_inds]
  y_test <- y[test_inds]
  y_train <- y[train_inds]

  # Run a BCF model with only GFR
  general_param_list <- list(num_chains = 1, keep_every = 1)
  bcf_model <- bcf(
    X_train = X_train,
    y_train = y_train,
    Z_train = Z_train,
    propensity_train = pi_train,
    X_test = X_test,
    Z_test = Z_test,
    rfx_group_ids_train = rfx_group_ids_train,
    rfx_group_ids_test = rfx_group_ids_test,
    rfx_basis_train = rfx_basis_train,
    rfx_basis_test = rfx_basis_test,
    propensity_test = pi_test,
    num_gfr = 10,
    num_burnin = 0,
    num_mcmc = 0,
    general_params = general_param_list
  )

  # Save to JSON string
  bcf_model_json_string <- saveBCFModelToJsonString(bcf_model)

  # Run a new BCF chain from the existing (X)BCF model
  general_param_list <- list(num_chains = 3, keep_every = 5)
  expect_no_error(
    bcf_model <- bcf(
      X_train = X_train,
      y_train = y_train,
      Z_train = Z_train,
      propensity_train = pi_train,
      X_test = X_test,
      Z_test = Z_test,
      rfx_group_ids_train = rfx_group_ids_train,
      rfx_group_ids_test = rfx_group_ids_test,
      rfx_basis_train = rfx_basis_train,
      rfx_basis_test = rfx_basis_test,
      propensity_test = pi_test,
      num_gfr = 0,
      num_burnin = 10,
      num_mcmc = 10,
      previous_model_json = bcf_model_json_string,
      previous_model_warmstart_sample_num = 10,
      general_params = general_param_list
    )
  )
})

test_that("Multivariate Treatment MCMC BCF", {
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
  pi_X_1 <- (((0 <= X[, 1]) & (0.25 > X[, 1])) * (0.2) +
    ((0.25 <= X[, 1]) & (0.5 > X[, 1])) * (0.4) +
    ((0.5 <= X[, 1]) & (0.75 > X[, 1])) * (0.6) +
    ((0.75 <= X[, 1]) & (1 > X[, 1])) * (0.8))
  # fmt: skip
  pi_X_2 <- (((0 <= X[, 2]) & (0.25 > X[, 2])) * (0.8) +
    ((0.25 <= X[, 2]) & (0.5 > X[, 2])) * (0.4) +
    ((0.5 <= X[, 2]) & (0.75 > X[, 2])) * (0.6) +
    ((0.75 <= X[, 2]) & (1 > X[, 2])) * (0.2))
  pi_X <- cbind(pi_X_1, pi_X_2)
  # fmt: skip
  tau_X_1 <- (((0 <= X[, 2]) & (0.25 > X[, 2])) * (0.5) +
    ((0.25 <= X[, 2]) & (0.5 > X[, 2])) * (1.0) +
    ((0.5 <= X[, 2]) & (0.75 > X[, 2])) * (1.5) +
    ((0.75 <= X[, 2]) & (1 > X[, 2])) * (2.0))
  # fmt: skip
  tau_X_2 <- (((0 <= X[, 3]) & (0.25 > X[, 3])) * (-0.5) +
    ((0.25 <= X[, 3]) & (0.5 > X[, 3])) * (-1.5) +
    ((0.5 <= X[, 3]) & (0.75 > X[, 3])) * (-1.0) +
    ((0.75 <= X[, 3]) & (1 > X[, 3])) * (0.0))
  tau_X <- cbind(tau_X_1, tau_X_2)
  Z_1 <- as.numeric(rbinom(n, 1, pi_X_1))
  Z_2 <- as.numeric(rbinom(n, 1, pi_X_2))
  Z <- cbind(Z_1, Z_2)
  noise_sd <- 1
  y <- mu_X + rowSums(tau_X * Z) + rnorm(n, 0, noise_sd)
  test_set_pct <- 0.2
  n_test <- round(test_set_pct * n)
  n_train <- n - n_test
  test_inds <- sort(sample(1:n, n_test, replace = FALSE))
  train_inds <- (1:n)[!((1:n) %in% test_inds)]
  X_test <- X[test_inds, ]
  X_train <- X[train_inds, ]
  Z_test <- Z[test_inds, ]
  Z_train <- Z[train_inds, ]
  pi_test <- pi_X[test_inds, ]
  pi_train <- pi_X[train_inds, ]
  mu_test <- mu_X[test_inds]
  mu_train <- mu_X[train_inds]
  tau_test <- tau_X[test_inds, ]
  tau_train <- tau_X[train_inds, ]
  y_test <- y[test_inds]
  y_train <- y[train_inds]

  # 1 chain, no thinning
  general_param_list <- list(
    num_chains = 1,
    keep_every = 1,
    adaptive_coding = F
  )
  expect_no_error({
    bcf_model <- bcf(
      X_train = X_train,
      y_train = y_train,
      Z_train = Z_train,
      propensity_train = pi_train,
      X_test = X_test,
      Z_test = Z_test,
      propensity_test = pi_test,
      num_gfr = 0,
      num_burnin = 10,
      num_mcmc = 10,
      general_params = general_param_list
    )
    predict(bcf_model, X = X_test, Z = Z_test, propensity = pi_test)
  })
})

test_that("BCF Predictions", {
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
  noise_sd <- 1
  y <- mu_X + tau_X * Z + rnorm(n, 0, noise_sd)
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
  y_test <- y[test_inds]
  y_train <- y[train_inds]

  # Run a BCF model with only GFR
  general_params <- list(num_chains = 1, keep_every = 1)
  variance_forest_params <- list(num_trees = 50)
  expect_warning(
    bcf_model <- bcf(
      X_train = X_train,
      y_train = y_train,
      Z_train = Z_train,
      propensity_train = pi_train,
      X_test = X_test,
      Z_test = Z_test,
      propensity_test = pi_test,
      num_gfr = 10,
      num_burnin = 0,
      num_mcmc = 10,
      general_params = general_params,
      variance_forest_params = variance_forest_params
    )
  )

  # Check that cached predictions agree with results of predict() function
  train_preds <- predict(
    bcf_model,
    X = X_train,
    Z = Z_train,
    propensity = pi_train
  )
  train_preds_mean_cached <- bcf_model$y_hat_train
  train_preds_mean_recomputed <- train_preds$y_hat
  train_preds_variance_cached <- bcf_model$sigma2_x_hat_train
  train_preds_variance_recomputed <- train_preds$variance_forest_predictions

  # Assertion
  expect_equal(train_preds_mean_cached, train_preds_mean_recomputed)
  expect_equal(train_preds_variance_cached, train_preds_variance_recomputed)
})

test_that("Random Effects BCF", {
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

  # Specify no rfx parameters
  general_param_list <- list()
  expect_no_error(
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
      num_mcmc = 10,
      general_params = general_param_list
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
  expect_no_error(
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
      num_mcmc = 10,
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
  expect_no_error(
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
      num_mcmc = 10,
      random_effects_params = rfx_param_list
    )
  )
})

test_that("BCF internal propensity model works with data frame covariates", {
  skip_on_cran()

  # X_test_raw was incorrectly passed to the internal BART propensity model instead of the preprocessed X_test,
  # causing "undefined columns selected" when X contained factor columns.
  set.seed(42)
  n <- 100
  p <- 4
  X_num <- matrix(runif(n * p), ncol = p)
  # Add a factor column — triggers preprocessTrainData to one-hot encode,
  # changing the column structure and exposing the raw/processed mismatch
  X_cat <- factor(sample(c("a", "b", "c"), n, replace = TRUE))
  X <- data.frame(X_num, cat = X_cat)

  pi_X <- 0.25 + 0.5 * X_num[, 1]
  Z <- rbinom(n, 1, pi_X)
  mu_X <- X_num[, 1] * 2
  tau_X <- X_num[, 2]
  y <- mu_X + tau_X * Z + rnorm(n)

  test_inds <- 1:20
  train_inds <- 21:n
  X_train <- X[train_inds, ]
  X_test <- X[test_inds, ]
  Z_train <- Z[train_inds]
  Z_test <- Z[test_inds]
  y_train <- y[train_inds]

  # No propensity_train provided — triggers internal BART propensity model.
  # Before the fix, this raised "undefined columns selected" because X_test_raw
  # (unprocessed) was passed instead of preprocessed X_test.
  expect_no_error(
    bcf_model <- bcf(
      X_train = X_train,
      y_train = y_train,
      Z_train = Z_train,
      X_test = X_test,
      Z_test = Z_test,
      num_gfr = 5,
      num_burnin = 0,
      num_mcmc = 5
    )
  )
  expect_true(!is.null(bcf_model$tau_hat_train))
  expect_true(!is.null(bcf_model$tau_hat_test))
})

test_that("BCF JSON serialization roundtrip covers all deserialization paths", {
  skip_on_cran()

  # Generate simulated data
  set.seed(42)
  n <- 100
  p <- 5
  X <- matrix(runif(n * p), ncol = p)
  pi_x <- 0.25 + 0.5 * X[, 1]
  mu_x <- pi_x * 5
  tau_x <- X[, 2] * 2
  Z <- rbinom(n, 1, pi_x)
  y <- mu_x + Z * tau_x + rnorm(n, 0, 1)
  test_set_pct <- 0.2
  n_test <- round(test_set_pct * n)
  test_inds <- sort(sample(1:n, n_test, replace = FALSE))
  train_inds <- (1:n)[!((1:n) %in% test_inds)]
  X_test <- X[test_inds, ]
  X_train <- X[train_inds, ]
  pi_test <- pi_x[test_inds]
  pi_train <- pi_x[train_inds]
  Z_test <- Z[test_inds]
  Z_train <- Z[train_inds]
  y_train <- y[train_inds]

  # Fit model
  bcf_model <- bcf(
    X_train = X_train,
    Z_train = Z_train,
    y_train = y_train,
    propensity_train = pi_train,
    num_gfr = 0,
    num_burnin = 0,
    num_mcmc = 10
  )
  preds_orig <- predict(bcf_model, X_test, Z_test, pi_test)
  y_hat_orig <- rowMeans(preds_orig[["y_hat"]])
  tau_hat_orig <- rowMeans(preds_orig[["tau_hat"]])

  # Path 1: in-memory JSON object
  bcf_json <- saveBCFModelToJson(bcf_model)
  bcf_rt <- createBCFModelFromJson(bcf_json)
  preds_rt <- predict(bcf_rt, X_test, Z_test, pi_test)
  expect_equal(rowMeans(preds_rt[["y_hat"]]), y_hat_orig)
  expect_equal(rowMeans(preds_rt[["tau_hat"]]), tau_hat_orig)

  # Path 2: JSON string
  bcf_json_string <- saveBCFModelToJsonString(bcf_model)
  bcf_rt <- createBCFModelFromJsonString(bcf_json_string)
  preds_rt <- predict(bcf_rt, X_test, Z_test, pi_test)
  expect_equal(rowMeans(preds_rt[["y_hat"]]), y_hat_orig)
  expect_equal(rowMeans(preds_rt[["tau_hat"]]), tau_hat_orig)

  # Path 3: JSON file
  tmpjson <- tempfile(fileext = ".json")
  saveBCFModelToJsonFile(bcf_model, tmpjson)
  bcf_rt <- createBCFModelFromJsonFile(tmpjson)
  unlink(tmpjson)
  preds_rt <- predict(bcf_rt, X_test, Z_test, pi_test)
  expect_equal(rowMeans(preds_rt[["y_hat"]]), y_hat_orig)
  expect_equal(rowMeans(preds_rt[["tau_hat"]]), tau_hat_orig)

  # Path 4: list of in-memory JSON objects (combined)
  bcf_rt <- createBCFModelFromCombinedJson(list(bcf_json))
  preds_rt <- predict(bcf_rt, X_test, Z_test, pi_test)
  expect_equal(rowMeans(preds_rt[["y_hat"]]), y_hat_orig)
  expect_equal(rowMeans(preds_rt[["tau_hat"]]), tau_hat_orig)

  # Path 5: list of JSON strings (combined)
  bcf_rt <- createBCFModelFromCombinedJsonString(list(bcf_json_string))
  preds_rt <- predict(bcf_rt, X_test, Z_test, pi_test)
  expect_equal(rowMeans(preds_rt[["y_hat"]]), y_hat_orig)
  expect_equal(rowMeans(preds_rt[["tau_hat"]]), tau_hat_orig)
})

test_that("BCF factor-valued treatment handling", {
  skip_on_cran()

  # Shared data: binary treatment DGP
  n <- 100
  p <- 5
  set.seed(42)
  X <- matrix(runif(n * p), ncol = p)
  pi_X <- 0.4 + 0.2 * X[, 1]
  Z_numeric <- rbinom(n, 1, pi_X)
  tau_X <- 1 + X[, 2]
  mu_X <- 2 * X[, 3]
  y <- mu_X + tau_X * Z_numeric + rnorm(n, 0, 1)

  # Binary factor treatment: levels "0" and "1"
  # Verify the conversion produces 0/1 values identical to the original
  Z_factor_binary <- factor(Z_numeric)
  expect_equal(levels(Z_factor_binary), c("0", "1"))
  expect_equal(as.integer(Z_factor_binary) - 1L, as.integer(Z_numeric))

  # Factor treatment should run without error and emit an informative message
  expect_message(
    suppressWarnings(bcf(
      X_train = X, y_train = y, Z_train = Z_factor_binary,
      propensity_train = pi_X, num_gfr = 0, num_burnin = 5, num_mcmc = 5
    )),
    regexp = "Z_train is a factor"
  )

  # Logical treatment converted to factor: levels "FALSE" and "TRUE"
  # as.factor(logical) sorts alphabetically: "FALSE" = 0, "TRUE" = 1
  Z_logical <- as.logical(Z_numeric)
  Z_factor_logical <- as.factor(Z_logical)
  expect_equal(levels(Z_factor_logical), c("FALSE", "TRUE"))
  expect_equal(as.integer(Z_factor_logical) - 1L, as.integer(Z_numeric))

  expect_message(
    suppressWarnings(bcf(
      X_train = X, y_train = y, Z_train = Z_factor_logical,
      propensity_train = pi_X, num_gfr = 0, num_burnin = 5, num_mcmc = 5
    )),
    regexp = "Z_train is a factor"
  )

  # Factor treatment with more than 2 levels should error immediately
  Z_factor_categorical <- factor(sample(c("A", "B", "C"), n, replace = TRUE))
  expect_error(
    bcf(
      X_train = X, y_train = y, Z_train = Z_factor_categorical,
      propensity_train = pi_X, num_gfr = 0, num_burnin = 5, num_mcmc = 5
    ),
    regexp = "exactly 2 levels"
  )
})
