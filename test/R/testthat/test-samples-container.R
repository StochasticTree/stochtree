test_that("BARTSamples wraps a fitted model's forests and parameters", {
  skip_on_cran()
  set.seed(1)
  n <- 100
  p <- 4
  X <- matrix(runif(n * p), n, p)
  y <- X[, 1] * 2 + rnorm(n, 0, 0.5)
  m <- bart(X_train = X, y_train = y, num_gfr = 0, num_burnin = 0, num_mcmc = 10)

  sc <- BARTSamples$new(
    mean_forest = m$mean_forests,
    variance_forest = NULL,
    global_var_samples = m$sigma2_global_samples,
    leaf_scale_samples = if (!is.null(m$sigma2_leaf_samples)) {
      m$sigma2_leaf_samples
    } else {
      NULL
    },
    y_bar = m$model_params$outcome_mean,
    y_std = m$model_params$outcome_scale,
    num_samples = m$model_params$num_samples
  )

  # Scalars / counts match the model
  expect_equal(sc$num_samples(), m$model_params$num_samples)
  expect_true(sc$has_mean_forest())
  expect_false(sc$has_variance_forest())
  expect_equal(sc$y_bar(), m$model_params$outcome_mean)

  # Materialized mean forest predicts identically to the model's forest (faithful deep copy)
  fc <- sc$materialize_mean_forest()
  expect_false(is.null(fc))
  expect_equal(fc$num_samples(), m$mean_forests$num_samples())
  ds <- createForestDataset(X)
  expect_equal(m$mean_forests$predict(ds), fc$predict(ds))

  # Parameter traces round-trip
  if (!is.null(m$sigma2_global_samples)) {
    expect_equal(
      as.numeric(sc$global_var_samples()),
      as.numeric(m$sigma2_global_samples)
    )
  }
})

test_that("BARTSamples merge concatenates draws", {
  skip_on_cran()
  set.seed(2)
  n <- 80
  p <- 3
  X <- matrix(runif(n * p), n, p)
  y <- X[, 1] + rnorm(n, 0, 0.5)
  m <- bart(X_train = X, y_train = y, num_gfr = 0, num_burnin = 0, num_mcmc = 8)

  build <- function() {
    BARTSamples$new(
      mean_forest = m$mean_forests,
      global_var_samples = m$sigma2_global_samples,
      y_bar = m$model_params$outcome_mean,
      y_std = m$model_params$outcome_scale,
      num_samples = m$model_params$num_samples
    )
  }
  # Two containers from the same model share standardization (merge guards against mismatch).
  a <- build()
  b <- build()
  na <- a$num_samples()
  nb <- b$num_samples()

  a$merge(b)

  expect_equal(a$num_samples(), na + nb)
  expect_equal(a$materialize_mean_forest()$num_samples(), na + nb)
})

test_that("BCFSamples wraps a fitted model's forests and parameters", {
  skip_on_cran()
  set.seed(3)
  n <- 120
  p <- 4
  X <- matrix(runif(n * p), n, p)
  pi_x <- 0.3 + 0.4 * X[, 2]
  Z <- rbinom(n, 1, pi_x)
  y <- 1 + 2 * X[, 1] + 1.5 * X[, 3] * Z + 0.5 * rnorm(n)
  m <- bcf(
    X_train = X, Z_train = Z, y_train = y, propensity_train = pi_x,
    num_gfr = 0, num_burnin = 0, num_mcmc = 10
  )

  sc <- BCFSamples$new(
    mu_forest = m$forests_mu,
    tau_forest = m$forests_tau,
    variance_forest = NULL,
    global_var_samples = m$sigma2_global_samples,
    y_bar = m$model_params$outcome_mean,
    y_std = m$model_params$outcome_scale,
    num_samples = m$model_params$num_samples,
    treatment_dim = 1L
  )

  expect_equal(sc$num_samples(), m$model_params$num_samples)
  expect_equal(sc$treatment_dim(), 1L)
  expect_true(sc$has_mu_forest())
  expect_true(sc$has_tau_forest())
  expect_false(sc$has_variance_forest())

  # Both forests materialize as deep copies. (Prediction parity is awkward to check directly here:
  # BCF trains its forests on propensity-augmented covariates, and the treatment forest needs a
  # basis -- so verify the copies via sample count.)
  fc_mu <- sc$materialize_mu_forest()
  fc_tau <- sc$materialize_tau_forest()
  expect_false(is.null(fc_mu))
  expect_false(is.null(fc_tau))
  expect_equal(fc_mu$num_samples(), m$forests_mu$num_samples())
  expect_equal(fc_tau$num_samples(), m$forests_tau$num_samples())

  # Global error variance round-trips
  if (!is.null(m$sigma2_global_samples)) {
    expect_equal(
      as.numeric(sc$global_var_samples()),
      as.numeric(m$sigma2_global_samples)
    )
  }
})
