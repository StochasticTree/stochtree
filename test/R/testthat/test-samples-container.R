test_that("BARTSamples round-trips a fitted model's forests and parameters", {
  skip_on_cran()
  set.seed(1)
  n <- 100
  p <- 4
  X <- matrix(runif(n * p), n, p)
  y <- X[, 1] * 2 + rnorm(n, 0, 0.5)
  m <- bart(X_train = X, y_train = y, num_gfr = 0, num_burnin = 0, num_mcmc = 10)

  # Rebuild an independent BARTSamples from the model's serialized form.
  sc <- createBARTSamplesFromJson(
    createCppJsonString(saveBARTModelToJsonString(m))
  )

  # Scalars / counts match the model
  expect_equal(sc$num_samples(), m$model_params$num_samples)
  expect_true(sc$has_mean_forest())
  expect_false(sc$has_variance_forest())
  expect_equal(sc$y_bar(), m$model_params$outcome_mean)

  # Materialized mean forest predicts identically to the model's forest (faithful deep copy)
  model_mean_forest <- m$samples$materialize_mean_forest()
  fc <- sc$materialize_mean_forest()
  expect_false(is.null(fc))
  expect_equal(fc$num_samples(), model_mean_forest$num_samples())
  ds <- createForestDataset(X)
  expect_equal(model_mean_forest$predict(ds), fc$predict(ds))

  # Parameter traces round-trip
  if (length(m$samples$global_var_samples()) > 0) {
    expect_equal(
      as.numeric(sc$global_var_samples()),
      as.numeric(m$samples$global_var_samples())
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
    createBARTSamplesFromJson(createCppJsonString(saveBARTModelToJsonString(m)))
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

  model_mu_forest <- m$samples$materialize_mu_forest()
  model_tau_forest <- m$samples$materialize_tau_forest()
  sc <- createBCFSamplesFromJson(
    createCppJsonString(saveBCFModelToJsonString(m))
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
  expect_equal(fc_mu$num_samples(), model_mu_forest$num_samples())
  expect_equal(fc_tau$num_samples(), model_tau_forest$num_samples())

  # Global error variance round-trips
  if (length(m$samples$global_var_samples()) > 0) {
    expect_equal(
      as.numeric(sc$global_var_samples()),
      as.numeric(m$samples$global_var_samples())
    )
  }
})

test_that("direct forest access is a hard error pointing at the extraction path", {
  skip_on_cran()
  set.seed(7)
  n <- 60
  p <- 3
  X <- matrix(runif(n * p), n, p)
  y <- X[, 1] + rnorm(n, 0, 0.5)
  m <- bart(X_train = X, y_train = y, num_gfr = 0, num_burnin = 0, num_mcmc = 5)

  # Removed fields raise, and the message names the supported extraction call.
  expect_error(m$mean_forests, "extractForest")
  expect_error(m$variance_forests, "extractForest")

  # Non-forest fields still read through normally.
  expect_false(is.null(m$model_params))
  expect_false(is.null(m$samples))

  # The documented extraction path works.
  expect_false(is.null(m$samples$materialize_mean_forest()))
})

test_that("BCF direct forest access is a hard error", {
  skip_on_cran()
  set.seed(8)
  n <- 100
  p <- 3
  X <- matrix(runif(n * p), n, p)
  pi_x <- 0.3 + 0.4 * X[, 2]
  Z <- rbinom(n, 1, pi_x)
  y <- 1 + 2 * X[, 1] + 1.5 * X[, 3] * Z + 0.5 * rnorm(n)
  m <- bcf(
    X_train = X, Z_train = Z, y_train = y, propensity_train = pi_x,
    num_gfr = 0, num_burnin = 0, num_mcmc = 5
  )
  expect_error(m$forests_mu, "extractForest")
  expect_error(m$forests_tau, "extractForest")
  expect_false(is.null(m$samples$materialize_tau_forest()))
})
