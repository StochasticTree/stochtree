test_that("MCMC BART", {
  skip_on_cran()

  # Generate simulated data
  n <- 100
  p <- 5
  X <- matrix(runif(n * p), ncol = p)
  # fmt: skip
  f_XW <- (((0 <= X[, 1]) & (0.25 > X[, 1])) * (-7.5) +
    ((0.25 <= X[, 1]) & (0.5 > X[, 1])) * (-2.5) +
    ((0.5 <= X[, 1]) & (0.75 > X[, 1])) * (2.5) +
    ((0.75 <= X[, 1]) & (1 > X[, 1])) * (7.5))
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

  # 1 chain, no thinning
  general_param_list <- list(num_chains = 1, keep_every = 1)
  expect_no_error(
    bart_model <- bart(
      X_train = X_train,
      y_train = y_train,
      X_test = X_test,
      num_gfr = 0,
      num_burnin = 10,
      num_mcmc = 10,
      general_params = general_param_list
    )
  )

  # 3 chains, no thinning
  general_param_list <- list(num_chains = 3, keep_every = 1)
  expect_no_error(
    bart_model <- bart(
      X_train = X_train,
      y_train = y_train,
      X_test = X_test,
      num_gfr = 0,
      num_burnin = 10,
      num_mcmc = 10,
      general_params = general_param_list
    )
  )

  # 1 chain, thinning
  general_param_list <- list(num_chains = 1, keep_every = 5)
  expect_no_error(
    bart_model <- bart(
      X_train = X_train,
      y_train = y_train,
      X_test = X_test,
      num_gfr = 0,
      num_burnin = 10,
      num_mcmc = 10,
      general_params = general_param_list
    )
  )

  # 3 chains, thinning
  general_param_list <- list(num_chains = 3, keep_every = 5)
  expect_no_error(
    bart_model <- bart(
      X_train = X_train,
      y_train = y_train,
      X_test = X_test,
      num_gfr = 0,
      num_burnin = 10,
      num_mcmc = 10,
      general_params = general_param_list
    )
  )

  # Generate simulated data with a leaf basis
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
  noise_sd <- 1
  y <- f_XW + rnorm(n, 0, noise_sd)
  test_set_pct <- 0.2
  n_test <- round(test_set_pct * n)
  n_train <- n - n_test
  test_inds <- sort(sample(1:n, n_test, replace = FALSE))
  train_inds <- (1:n)[!((1:n) %in% test_inds)]
  X_test <- X[test_inds, ]
  X_train <- X[train_inds, ]
  W_test <- W[test_inds, ]
  W_train <- W[train_inds, ]
  y_test <- y[test_inds]
  y_train <- y[train_inds]

  # 3 chains, thinning, leaf regression
  general_param_list <- list(num_chains = 3, keep_every = 5)
  mean_forest_param_list <- list(sample_sigma2_leaf = FALSE)
  expect_no_error(
    bart_model <- bart(
      X_train = X_train,
      y_train = y_train,
      X_test = X_test,
      leaf_basis_train = W_train,
      leaf_basis_test = W_test,
      num_gfr = 0,
      num_burnin = 10,
      num_mcmc = 10,
      general_params = general_param_list,
      mean_forest_params = mean_forest_param_list
    )
  )

  # 3 chains, thinning, leaf regression with a scalar leaf scale
  general_param_list <- list(num_chains = 3, keep_every = 5)
  mean_forest_param_list <- list(
    sample_sigma2_leaf = FALSE,
    sigma2_leaf_init = 0.5
  )
  expect_no_error(
    bart_model <- bart(
      X_train = X_train,
      y_train = y_train,
      X_test = X_test,
      leaf_basis_train = W_train,
      leaf_basis_test = W_test,
      num_gfr = 0,
      num_burnin = 10,
      num_mcmc = 10,
      general_params = general_param_list,
      mean_forest_params = mean_forest_param_list
    )
  )

  # 3 chains, thinning, leaf regression with a scalar leaf scale, random leaf scale
  general_param_list <- list(num_chains = 3, keep_every = 5)
  mean_forest_param_list <- list(sample_sigma2_leaf = T, sigma2_leaf_init = 0.5)
  expect_warning(
    bart_model <- bart(
      X_train = X_train,
      y_train = y_train,
      X_test = X_test,
      leaf_basis_train = W_train,
      leaf_basis_test = W_test,
      num_gfr = 0,
      num_burnin = 10,
      num_mcmc = 10,
      general_params = general_param_list,
      mean_forest_params = mean_forest_param_list
    )
  )
})

test_that("GFR BART", {
  skip_on_cran()

  # Generate simulated data
  n <- 100
  p <- 5
  X <- matrix(runif(n * p), ncol = p)
  # fmt: skip
  f_XW <- (((0 <= X[, 1]) & (0.25 > X[, 1])) * (-7.5) +
    ((0.25 <= X[, 1]) & (0.5 > X[, 1])) * (-2.5) +
    ((0.5 <= X[, 1]) & (0.75 > X[, 1])) * (2.5) +
    ((0.75 <= X[, 1]) & (1 > X[, 1])) * (7.5))
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

  # 1 chain, no thinning
  general_param_list <- list(num_chains = 1, keep_every = 1)
  expect_no_error(
    bart_model <- bart(
      X_train = X_train,
      y_train = y_train,
      X_test = X_test,
      num_gfr = 10,
      num_burnin = 10,
      num_mcmc = 10,
      general_params = general_param_list
    )
  )

  # 3 chains, no thinning
  general_param_list <- list(num_chains = 3, keep_every = 1)
  expect_no_error(
    bart_model <- bart(
      X_train = X_train,
      y_train = y_train,
      X_test = X_test,
      num_gfr = 10,
      num_burnin = 10,
      num_mcmc = 10,
      general_params = general_param_list
    )
  )

  # 1 chain, thinning
  general_param_list <- list(num_chains = 1, keep_every = 5)
  expect_no_error(
    bart_model <- bart(
      X_train = X_train,
      y_train = y_train,
      X_test = X_test,
      num_gfr = 10,
      num_burnin = 10,
      num_mcmc = 10,
      general_params = general_param_list
    )
  )

  # 3 chains, thinning
  general_param_list <- list(num_chains = 3, keep_every = 5)
  expect_no_error(
    bart_model <- bart(
      X_train = X_train,
      y_train = y_train,
      X_test = X_test,
      num_gfr = 10,
      num_burnin = 10,
      num_mcmc = 10,
      general_params = general_param_list
    )
  )

  # Check for error when more chains than GFR forests
  general_param_list <- list(num_chains = 11, keep_every = 1)
  expect_error(
    bart_model <- bart(
      X_train = X_train,
      y_train = y_train,
      X_test = X_test,
      num_gfr = 10,
      num_burnin = 10,
      num_mcmc = 10,
      general_params = general_param_list
    )
  )

  # Check for error when more chains than GFR forests
  general_param_list <- list(num_chains = 11, keep_every = 5)
  expect_error(
    bart_model <- bart(
      X_train = X_train,
      y_train = y_train,
      X_test = X_test,
      num_gfr = 10,
      num_burnin = 10,
      num_mcmc = 10,
      general_params = general_param_list
    )
  )
})

test_that("Warmstart BART", {
  skip_on_cran()

  # Generate simulated data
  n <- 100
  p <- 5
  X <- matrix(runif(n * p), ncol = p)
  # fmt: skip
  f_XW <- (((0 <= X[, 1]) & (0.25 > X[, 1])) * (-7.5) +
    ((0.25 <= X[, 1]) & (0.5 > X[, 1])) * (-2.5) +
    ((0.5 <= X[, 1]) & (0.75 > X[, 1])) * (2.5) +
    ((0.75 <= X[, 1]) & (1 > X[, 1])) * (7.5))
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

  # Run a BART model with only GFR
  general_param_list <- list(num_chains = 1, keep_every = 1)
  bart_model <- bart(
    X_train = X_train,
    y_train = y_train,
    X_test = X_test,
    num_gfr = 10,
    num_burnin = 0,
    num_mcmc = 0,
    general_params = general_param_list
  )

  # Save to JSON string
  bart_model_json_string <- saveBARTModelToJsonString(bart_model)

  # Run a new BART chain from the existing (X)BART model
  general_param_list <- list(num_chains = 3, keep_every = 5)
  expect_no_error(
    bart_model <- bart(
      X_train = X_train,
      y_train = y_train,
      X_test = X_test,
      num_gfr = 0,
      num_burnin = 10,
      num_mcmc = 10,
      previous_model_json = bart_model_json_string,
      previous_model_warmstart_sample_num = 10,
      general_params = general_param_list
    )
  )
  expect_warning(
    bart_model <- bart(
      X_train = X_train,
      y_train = y_train,
      X_test = X_test,
      num_gfr = 0,
      num_burnin = 10,
      num_mcmc = 10,
      previous_model_json = bart_model_json_string,
      previous_model_warmstart_sample_num = 1,
      general_params = general_param_list
    )
  )

  # Generate simulated data with random effects
  n <- 100
  p <- 5
  X <- matrix(runif(n * p), ncol = p)
  # fmt: skip
  f_XW <- (((0 <= X[, 1]) & (0.25 > X[, 1])) * (-7.5) +
    ((0.25 <= X[, 1]) & (0.5 > X[, 1])) * (-2.5) +
    ((0.5 <= X[, 1]) & (0.75 > X[, 1])) * (2.5) +
    ((0.75 <= X[, 1]) & (1 > X[, 1])) * (7.5))
  rfx_group_ids <- sample(1:2, size = n, replace = TRUE)
  rfx_basis <- rep(1, n)
  rfx_coefs <- c(-5, 5)
  rfx_term <- rfx_coefs[rfx_group_ids] * rfx_basis
  noise_sd <- 1
  y <- f_XW + rfx_term + rnorm(n, 0, noise_sd)
  test_set_pct <- 0.2
  n_test <- round(test_set_pct * n)
  n_train <- n - n_test
  test_inds <- sort(sample(1:n, n_test, replace = FALSE))
  train_inds <- (1:n)[!((1:n) %in% test_inds)]
  X_test <- X[test_inds, ]
  X_train <- X[train_inds, ]
  rfx_group_ids_test <- rfx_group_ids[test_inds]
  rfx_group_ids_train <- rfx_group_ids[train_inds]
  rfx_basis_test <- rfx_basis[test_inds]
  rfx_basis_train <- rfx_basis[train_inds]
  y_test <- y[test_inds]
  y_train <- y[train_inds]

  # Run a BART model with only GFR
  general_param_list <- list(num_chains = 1, keep_every = 1)
  bart_model <- bart(
    X_train = X_train,
    y_train = y_train,
    X_test = X_test,
    rfx_group_ids_train = rfx_group_ids_train,
    rfx_group_ids_test = rfx_group_ids_test,
    rfx_basis_train = rfx_basis_train,
    rfx_basis_test = rfx_basis_test,
    num_gfr = 10,
    num_burnin = 0,
    num_mcmc = 0,
    general_params = general_param_list
  )

  # Save to JSON string
  bart_model_json_string <- saveBARTModelToJsonString(bart_model)

  # Run a new BART chain from the existing (X)BART model
  general_param_list <- list(num_chains = 4, keep_every = 5)
  expect_no_error(
    bart_model <- bart(
      X_train = X_train,
      y_train = y_train,
      X_test = X_test,
      rfx_group_ids_train = rfx_group_ids_train,
      rfx_group_ids_test = rfx_group_ids_test,
      rfx_basis_train = rfx_basis_train,
      rfx_basis_test = rfx_basis_test,
      num_gfr = 0,
      num_burnin = 10,
      num_mcmc = 10,
      previous_model_json = bart_model_json_string,
      previous_model_warmstart_sample_num = 10,
      general_params = general_param_list
    )
  )
  expect_warning(
    bart_model <- bart(
      X_train = X_train,
      y_train = y_train,
      X_test = X_test,
      rfx_group_ids_train = rfx_group_ids_train,
      rfx_group_ids_test = rfx_group_ids_test,
      rfx_basis_train = rfx_basis_train,
      rfx_basis_test = rfx_basis_test,
      num_gfr = 0,
      num_burnin = 10,
      num_mcmc = 10,
      previous_model_json = bart_model_json_string,
      previous_model_warmstart_sample_num = 1,
      general_params = general_param_list
    )
  )
})

test_that("BART Predictions", {
  skip_on_cran()

  # Generate simulated data
  n <- 100
  p <- 5
  X <- matrix(runif(n * p), ncol = p)
  # fmt: skip
  f_XW <- (((0 <= X[, 1]) & (0.25 > X[, 1])) * (-7.5) +
    ((0.25 <= X[, 1]) & (0.5 > X[, 1])) * (-2.5) +
    ((0.5 <= X[, 1]) & (0.75 > X[, 1])) * (2.5) +
    ((0.75 <= X[, 1]) & (1 > X[, 1])) * (7.5))
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

  # Run a BART model with only GFR
  general_params <- list(num_chains = 1, sample_sigma2_global = FALSE)
  variance_forest_params <- list(num_trees = 50)
  bart_model <- bart(
    X_train = X_train,
    y_train = y_train,
    X_test = X_test,
    num_gfr = 10,
    num_burnin = 0,
    num_mcmc = 10,
    general_params = general_params,
    variance_forest_params = variance_forest_params
  )

  # Check that cached predictions agree with results of predict() function
  train_preds <- predict(bart_model, X = X_train)
  train_preds_mean_cached <- bart_model$y_hat_train
  train_preds_mean_recomputed <- train_preds$mean_forest_predictions
  train_preds_variance_cached <- bart_model$sigma2_x_hat_train
  train_preds_variance_recomputed <- train_preds$variance_forest_predictions

  # Assertion
  expect_equal(train_preds_mean_cached, train_preds_mean_recomputed)
  expect_equal(train_preds_variance_cached, train_preds_variance_recomputed)
})

test_that("Random Effects BART", {
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

test_that("Probit BART", {
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
  z <- f_XW + rnorm(n, 0, 1)
  y <- as.numeric(z > 0)
  test_set_pct <- 0.2
  n_test <- round(test_set_pct * n)
  n_train <- n - n_test
  test_inds <- sort(sample(1:n, n_test, replace = FALSE))
  train_inds <- (1:n)[!((1:n) %in% test_inds)]
  X_test <- X[test_inds, ]
  X_train <- X[train_inds, ]
  W_test <- W[test_inds, ]
  W_train <- W[train_inds, ]
  y_test <- y[test_inds]
  y_train <- y[train_inds]

  # Fit probit model
  general_param_list <- list(
    outcome_model = outcome_model(outcome = "binary", link = "probit"),
    sample_sigma2_global = F
  )
  mean_forest_param_list <- list(sample_sigma2_leaf = FALSE)
  expect_no_error(
    bart_model <- bart(
      X_train = X_train,
      y_train = y_train,
      X_test = X_test,
      leaf_basis_train = W_train,
      leaf_basis_test = W_test,
      num_gfr = 0,
      num_burnin = 10,
      num_mcmc = 10,
      general_params = general_param_list,
      mean_forest_params = mean_forest_param_list
    )
  )

  # Predict from model on linear scale
  expect_no_error({
    preds <- predict(
      bart_model,
      X = X_test,
      leaf_basis = W_test,
      type = "posterior",
      scale = "linear"
    )
  })

  # Predict from model on probability scale
  expect_no_error({
    preds <- predict(
      bart_model,
      X = X_test,
      leaf_basis = W_test,
      type = "posterior",
      scale = "probability"
    )
  })
})

test_that("Cloglog Binary BART", {
  skip_on_cran()

  # Generate simulated data
  n <- 200
  p <- 5
  X <- matrix(runif(n * p), ncol = p)
  f_X <- 2 * (X[, 1] > 0.5) - 1
  prob <- 1 - exp(-exp(f_X))
  y <- rbinom(n, 1, prob)
  test_set_pct <- 0.2
  n_test <- round(test_set_pct * n)
  n_train <- n - n_test
  test_inds <- sort(sample(1:n, n_test, replace = FALSE))
  train_inds <- (1:n)[!((1:n) %in% test_inds)]
  X_test <- X[test_inds, ]
  X_train <- X[train_inds, ]
  y_test <- y[test_inds]
  y_train <- y[train_inds]

  # Fit cloglog binary model (MCMC only)
  general_param_list <- list(
    outcome_model = outcome_model(outcome = "binary", link = "cloglog"),
    sample_sigma2_global = FALSE,
    num_chains = 1
  )
  expect_no_error(
    bart_model <- bart(
      X_train = X_train,
      y_train = y_train,
      X_test = X_test,
      num_gfr = 0,
      num_burnin = 10,
      num_mcmc = 10,
      general_params = general_param_list
    )
  )

  # Check model outputs
  expect_true(!is.null(bart_model$y_hat_train))
  expect_true(!is.null(bart_model$y_hat_test))
  expect_true(is.null(bart_model$cloglog_cutpoint_samples))
  expect_equal(nrow(bart_model$y_hat_train), n_train)
  expect_equal(ncol(bart_model$y_hat_train), 10)
  expect_equal(nrow(bart_model$y_hat_test), n_test)
  expect_equal(ncol(bart_model$y_hat_test), 10)

  # Predict from model on linear scale (terms = "y_hat" for single matrix return)
  expect_no_error({
    preds_linear <- predict(
      bart_model,
      X = X_test,
      type = "posterior",
      scale = "linear",
      terms = "y_hat"
    )
  })
  expect_equal(nrow(preds_linear), n_test)
  expect_equal(ncol(preds_linear), 10)

  # Predict from model on probability scale
  expect_no_error({
    preds_prob <- predict(
      bart_model,
      X = X_test,
      type = "posterior",
      scale = "probability",
      terms = "y_hat"
    )
  })
  expect_equal(nrow(preds_prob), n_test)
  expect_equal(ncol(preds_prob), 10)

  # Predict posterior mean on linear scale
  expect_no_error({
    preds_mean <- predict(
      bart_model,
      X = X_test,
      type = "mean",
      scale = "linear",
      terms = "y_hat"
    )
  })
  expect_equal(length(preds_mean), n_test)

  # Predict posterior mean on probability scale
  expect_no_error({
    preds_mean_prob <- predict(
      bart_model,
      X = X_test,
      type = "mean",
      scale = "probability",
      terms = "y_hat"
    )
  })
  expect_equal(length(preds_mean_prob), n_test)

  # Predict class labels
  expect_no_error({
    preds_class <- predict(
      bart_model,
      X = X_test,
      type = "posterior",
      scale = "class",
      terms = "y_hat"
    )
  })
  expect_true(all(preds_class >= 0 & preds_class <= 1))
})

test_that("Cloglog Binary BART with GFR", {
  skip_on_cran()

  # Generate simulated data
  n <- 200
  p <- 5
  X <- matrix(runif(n * p), ncol = p)
  f_X <- 2 * (X[, 1] > 0.5) - 1
  prob <- 1 - exp(-exp(f_X))
  y <- rbinom(n, 1, prob)
  test_set_pct <- 0.2
  n_test <- round(test_set_pct * n)
  n_train <- n - n_test
  test_inds <- sort(sample(1:n, n_test, replace = FALSE))
  train_inds <- (1:n)[!((1:n) %in% test_inds)]
  X_test <- X[test_inds, ]
  X_train <- X[train_inds, ]
  y_test <- y[test_inds]
  y_train <- y[train_inds]

  # Fit cloglog binary model with GFR warmstart
  general_param_list <- list(
    outcome_model = outcome_model(outcome = "binary", link = "cloglog"),
    sample_sigma2_global = FALSE,
    num_chains = 1
  )
  expect_no_error(
    bart_model <- bart(
      X_train = X_train,
      y_train = y_train,
      X_test = X_test,
      num_gfr = 10,
      num_burnin = 0,
      num_mcmc = 10,
      general_params = general_param_list
    )
  )

  # Check model outputs
  expect_true(!is.null(bart_model$y_hat_train))
  expect_true(!is.null(bart_model$y_hat_test))
  expect_true(is.null(bart_model$cloglog_cutpoint_samples))
  expect_equal(nrow(bart_model$y_hat_train), n_train)
  expect_equal(ncol(bart_model$y_hat_train), 10)
})

test_that("Cloglog Ordinal BART", {
  skip_on_cran()

  # Generate simulated ordinal data (3 categories)
  n <- 300
  p <- 5
  X <- matrix(runif(n * p), ncol = p)
  f_X <- 2 * (X[, 1] > 0.5) - 1
  gamma_true <- c(-1.0, 0.5)
  n_categories <- 3

  # Compute ordinal class probabilities
  true_probs <- matrix(0, n, n_categories)
  true_probs[, 1] <- 1 - exp(-exp(gamma_true[1] + f_X))
  true_probs[, 2] <- exp(-exp(gamma_true[1] + f_X)) *
    (1 - exp(-exp(gamma_true[2] + f_X)))
  true_probs[, 3] <- 1 - true_probs[, 1] - true_probs[, 2]

  # Generate ordinal outcomes (1-indexed)
  y <- sapply(1:n, function(i) sample(1:n_categories, 1, prob = true_probs[i, ]))

  test_set_pct <- 0.2
  n_test <- round(test_set_pct * n)
  n_train <- n - n_test
  test_inds <- sort(sample(1:n, n_test, replace = FALSE))
  train_inds <- (1:n)[!((1:n) %in% test_inds)]
  X_test <- X[test_inds, ]
  X_train <- X[train_inds, ]
  y_test <- y[test_inds]
  y_train <- y[train_inds]

  # Fit cloglog ordinal model (MCMC only)
  general_param_list <- list(
    outcome_model = outcome_model(outcome = "ordinal", link = "cloglog"),
    sample_sigma2_global = FALSE,
    num_chains = 1
  )
  expect_no_error(
    bart_model <- bart(
      X_train = X_train,
      y_train = y_train,
      X_test = X_test,
      num_gfr = 0,
      num_burnin = 10,
      num_mcmc = 10,
      general_params = general_param_list
    )
  )

  # Check model outputs
  expect_true(!is.null(bart_model$y_hat_train))
  expect_true(!is.null(bart_model$y_hat_test))
  expect_true(!is.null(bart_model$cloglog_cutpoint_samples))
  expect_equal(nrow(bart_model$y_hat_train), n_train)
  expect_equal(ncol(bart_model$y_hat_train), 10)
  expect_equal(nrow(bart_model$y_hat_test), n_test)
  expect_equal(ncol(bart_model$y_hat_test), 10)
  # 3 categories means 2 cutpoint rows
  expect_equal(nrow(bart_model$cloglog_cutpoint_samples), 2)
  expect_equal(ncol(bart_model$cloglog_cutpoint_samples), 10)
  expect_equal(bart_model$model_params$cloglog_num_categories, 3)

  # Predict from model on linear scale
  expect_no_error({
    preds_linear <- predict(
      bart_model,
      X = X_test,
      type = "posterior",
      scale = "linear",
      terms = "y_hat"
    )
  })
  expect_equal(nrow(preds_linear), n_test)
  expect_equal(ncol(preds_linear), 10)

  # Predict posterior mean on linear scale
  expect_no_error({
    preds_mean <- predict(
      bart_model,
      X = X_test,
      type = "mean",
      scale = "linear",
      terms = "y_hat"
    )
  })
  expect_equal(length(preds_mean), n_test)
})

test_that("Cloglog Ordinal BART with GFR", {
  skip_on_cran()

  # Generate simulated ordinal data (3 categories)
  n <- 300
  p <- 5
  X <- matrix(runif(n * p), ncol = p)
  f_X <- 0.5 * (X[, 1] > 0.5) - 0.25
  gamma_true <- c(-0.5, 0.5)
  n_categories <- 3

  # Compute ordinal class probabilities
  true_probs <- matrix(0, n, n_categories)
  true_probs[, 1] <- 1 - exp(-exp(gamma_true[1] + f_X))
  true_probs[, 2] <- exp(-exp(gamma_true[1] + f_X)) *
    (1 - exp(-exp(gamma_true[2] + f_X)))
  true_probs[, 3] <- 1 - true_probs[, 1] - true_probs[, 2]

  # Generate ordinal outcomes (1-indexed)
  y <- sapply(1:n, function(i) sample(1:n_categories, 1, prob = true_probs[i, ]))

  test_set_pct <- 0.2
  n_test <- round(test_set_pct * n)
  n_train <- n - n_test
  test_inds <- sort(sample(1:n, n_test, replace = FALSE))
  train_inds <- (1:n)[!((1:n) %in% test_inds)]
  X_test <- X[test_inds, ]
  X_train <- X[train_inds, ]
  y_test <- y[test_inds]
  y_train <- y[train_inds]

  # Fit cloglog ordinal model with GFR warmstart
  general_param_list <- list(
    outcome_model = outcome_model(outcome = "ordinal", link = "cloglog"),
    sample_sigma2_global = FALSE,
    num_chains = 1
  )
  expect_no_error(
    bart_model <- bart(
      X_train = X_train,
      y_train = y_train,
      X_test = X_test,
      num_gfr = 10,
      num_burnin = 0,
      num_mcmc = 10,
      general_params = general_param_list
    )
  )

  # Check model outputs
  expect_true(!is.null(bart_model$y_hat_train))
  expect_true(!is.null(bart_model$y_hat_test))
  expect_true(!is.null(bart_model$cloglog_cutpoint_samples))
  expect_equal(nrow(bart_model$y_hat_train), n_train)
  expect_equal(ncol(bart_model$y_hat_train), 10)
  # 3 categories means 2 cutpoint rows
  expect_equal(nrow(bart_model$cloglog_cutpoint_samples), 2)
  expect_equal(ncol(bart_model$cloglog_cutpoint_samples), 10)
  expect_equal(bart_model$model_params$cloglog_num_categories, 3)
})

test_that("Cloglog BART multi-chain", {
  skip_on_cran()

  # Generate simulated data
  n <- 200
  p <- 5
  X <- matrix(runif(n * p), ncol = p)
  f_X <- 2 * (X[, 1] > 0.5) - 1
  prob <- 1 - exp(-exp(f_X))
  y <- rbinom(n, 1, prob)
  test_set_pct <- 0.2
  n_test <- round(test_set_pct * n)
  n_train <- n - n_test
  test_inds <- sort(sample(1:n, n_test, replace = FALSE))
  train_inds <- (1:n)[!((1:n) %in% test_inds)]
  X_test <- X[test_inds, ]
  X_train <- X[train_inds, ]
  y_test <- y[test_inds]
  y_train <- y[train_inds]

  # Fit cloglog binary model with 2 chains
  general_param_list <- list(
    outcome_model = outcome_model(outcome = "binary", link = "cloglog"),
    sample_sigma2_global = FALSE,
    num_chains = 2
  )
  expect_no_error(
    bart_model <- bart(
      X_train = X_train,
      y_train = y_train,
      X_test = X_test,
      num_gfr = 10,
      num_burnin = 0,
      num_mcmc = 10,
      general_params = general_param_list
    )
  )

  # Check model outputs: 2 chains x 10 MCMC = 20 total samples
  expect_true(!is.null(bart_model$y_hat_train))
  expect_true(is.null(bart_model$cloglog_cutpoint_samples))
  expect_equal(nrow(bart_model$y_hat_train), n_train)
  expect_equal(ncol(bart_model$y_hat_train), 20)
})

test_that("Cloglog BART JSON round-trip", {
  skip_on_cran()

  # Generate simulated ordinal data (3 categories)
  n <- 200
  p <- 5
  X <- matrix(runif(n * p), ncol = p)
  f_X <- 2 * (X[, 1] > 0.5) - 1
  gamma_true <- c(-1.0, 0.5)
  n_categories <- 3
  true_probs <- matrix(0, n, n_categories)
  true_probs[, 1] <- 1 - exp(-exp(gamma_true[1] + f_X))
  true_probs[, 2] <- exp(-exp(gamma_true[1] + f_X)) *
    (1 - exp(-exp(gamma_true[2] + f_X)))
  true_probs[, 3] <- 1 - true_probs[, 1] - true_probs[, 2]
  y <- sapply(1:n, function(i) sample(1:n_categories, 1, prob = true_probs[i, ]))

  test_set_pct <- 0.2
  n_test <- round(test_set_pct * n)
  n_train <- n - n_test
  test_inds <- sort(sample(1:n, n_test, replace = FALSE))
  train_inds <- (1:n)[!((1:n) %in% test_inds)]
  X_test <- X[test_inds, ]
  X_train <- X[train_inds, ]
  y_test <- y[test_inds]
  y_train <- y[train_inds]

  # Fit cloglog ordinal model
  general_param_list <- list(
    outcome_model = outcome_model(outcome = "ordinal", link = "cloglog"),
    sample_sigma2_global = FALSE,
    num_chains = 1
  )
  expect_no_error(
    bart_model <- bart(
      X_train = X_train,
      y_train = y_train,
      X_test = X_test,
      num_gfr = 0,
      num_burnin = 10,
      num_mcmc = 10,
      general_params = general_param_list
    )
  )

  # Save to JSON
  json_string <- saveBARTModelToJsonString(bart_model)
  expect_true(nchar(json_string) > 0)

  # Load from JSON
  expect_no_error(
    bart_model_loaded <- createBARTModelFromJsonString(json_string)
  )

  # Check that cloglog parameters survive round-trip
  expect_equal(
    bart_model_loaded$model_params$cloglog_num_categories,
    bart_model$model_params$cloglog_num_categories
  )
  expect_equal(
    bart_model_loaded$cloglog_cutpoint_samples,
    bart_model$cloglog_cutpoint_samples
  )

  # Check that forest predictions survive round-trip
  pred_orig <- predict(
    bart_model, X = X_test, type = "posterior",
    scale = "linear", terms = "y_hat"
  )
  pred_loaded <- predict(
    bart_model_loaded, X = X_test, type = "posterior",
    scale = "linear", terms = "y_hat"
  )
  expect_equal(pred_orig, pred_loaded)
})
