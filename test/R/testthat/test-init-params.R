# Regression tests that user-supplied initialization / calibration parameters
# are threaded into the C++ sampler rather than silently dropped. Each test would
# fail if the corresponding parameter were ignored on the C++ path.

test_that("BART honors user-supplied sigma2_global_init", {
  skip_on_cran()
  set.seed(1)
  n <- 200
  p <- 3
  X <- matrix(runif(n * p), ncol = p)
  y <- X[, 1] + rnorm(n)
  fit <- function(s) {
    general_params <- list(standardize = FALSE, random_seed = 1)
    if (!is.null(s)) {
      general_params$sigma2_global_init <- s
    }
    bart(
      X_train = X, y_train = y,
      num_gfr = 0, num_burnin = 0, num_mcmc = 5,
      general_params = general_params
    )
  }
  m_set <- fit(9.0)
  m_default <- fit(NULL)
  # User-supplied global variance init must be honored, not hardcoded to 1.0.
  expect_equal(m_set$model_params$sigma2_init, 9.0, tolerance = 1e-8)
  expect_false(isTRUE(all.equal(m_default$model_params$sigma2_init, 9.0)))
})

test_that("BART honors user-supplied var_forest_leaf_init", {
  skip_on_cran()
  set.seed(2)
  n <- 200
  p <- 3
  X <- matrix(runif(n * p), ncol = p)
  # Heteroskedastic outcome so the variance forest is meaningfully fit.
  y <- X[, 1] + rnorm(n) * exp(0.5 * X[, 2])
  fit <- function(v) {
    variance_forest_params <- list(num_trees = 50)
    if (!is.null(v)) {
      variance_forest_params$var_forest_leaf_init <- v
    }
    m <- bart(
      X_train = X, y_train = y, X_test = X,
      num_gfr = 0, num_burnin = 0, num_mcmc = 3,
      general_params = list(standardize = FALSE, random_seed = 1),
      mean_forest_params = list(num_trees = 50),
      variance_forest_params = variance_forest_params
    )
    m$sigma2_x_hat_test
  }
  # Same seed; differing only by the variance-forest leaf init must change output.
  expect_false(isTRUE(all.equal(fit(0.05), fit(2.0))))
})

test_that("BCF honors delta_max for probit treatment leaf calibration", {
  skip_on_cran()
  set.seed(3)
  n <- 300
  p <- 5
  X <- matrix(runif(n * p), ncol = p)
  Z <- rbinom(n, 1, 0.5)
  propensity <- rep(0.5, n)
  prob <- pnorm(X[, 1] - 0.5 + 0.5 * Z)
  y <- rbinom(n, 1, prob)
  num_trees_tau <- 50
  fit <- function(dm) {
    m <- bcf(
      X_train = X, Z_train = Z, y_train = y, propensity_train = propensity,
      num_gfr = 0, num_burnin = 0, num_mcmc = 5,
      general_params = list(
        random_seed = 1,
        outcome_model = OutcomeModel(outcome = "binary", link = "probit")
      ),
      treatment_effect_forest_params = list(
        num_trees = num_trees_tau,
        delta_max = dm
      )
    )
    m$model_params$sigma2_leaf_tau
  }
  expected <- function(dm) {
    p_coverage <- 0.6827
    q_quantile <- qnorm((p_coverage + 1) / 2)
    ((dm / (q_quantile * dnorm(0)))^2) / num_trees_tau
  }
  v_05 <- fit(0.5)
  v_09 <- fit(0.9)
  # delta_max must drive the treatment-effect leaf scale calibration.
  expect_false(isTRUE(all.equal(v_05, v_09)))
  expect_equal(v_05, expected(0.5), tolerance = 1e-8)
  expect_equal(v_09, expected(0.9), tolerance = 1e-8)
})

test_that("BART honors observation_weights", {
  skip_on_cran()
  set.seed(4)
  n <- 200
  p <- 3
  X <- matrix(runif(n * p), ncol = p)
  y <- X[, 1] + rnorm(n)
  w <- runif(n, 0.1, 2.0)
  fit <- function(weights) {
    bart(
      X_train = X, y_train = y, X_test = X,
      num_gfr = 0, num_burnin = 0, num_mcmc = 5,
      general_params = list(standardize = FALSE, random_seed = 1),
      observation_weights_train = weights
    )$y_hat_test
  }
  # Non-uniform observation weights must change the fit (same seed).
  expect_false(isTRUE(all.equal(fit(NULL), fit(w))))
})

test_that("BCF honors observation_weights", {
  skip_on_cran()
  set.seed(5)
  n <- 200
  p <- 5
  X <- matrix(runif(n * p), ncol = p)
  Z <- rbinom(n, 1, 0.5)
  propensity <- rep(0.5, n)
  y <- X[, 1] + Z * X[, 2] + rnorm(n)
  w <- runif(n, 0.1, 2.0)
  fit <- function(weights) {
    bcf(
      X_train = X, Z_train = Z, y_train = y, propensity_train = propensity,
      X_test = X, Z_test = Z, propensity_test = propensity,
      num_gfr = 0, num_burnin = 0, num_mcmc = 5,
      general_params = list(standardize = FALSE, random_seed = 1),
      observation_weights_train = weights
    )$y_hat_test
  }
  expect_false(isTRUE(all.equal(fit(NULL), fit(w))))
})

test_that("BCF internal propensity model is reproducible with random_seed", {
  skip_on_cran()
  set.seed(7)
  n <- 200
  p <- 5
  X <- matrix(runif(n * p), ncol = p)
  Z <- rbinom(n, 1, 0.5)
  y <- X[, 1] + Z * X[, 2] + rnorm(n)
  fit <- function() {
    # No propensity_train -> BCF estimates it with an internal BART model.
    bcf(
      X_train = X, Z_train = Z, y_train = y,
      num_gfr = 0, num_burnin = 0, num_mcmc = 10,
      general_params = list(random_seed = 99)
    )$y_hat_train
  }
  # With a fixed random_seed, the internally-estimated propensity (and hence the
  # full fit) must be reproducible across runs.
  expect_equal(fit(), fit())
})

test_that("verbose prints sampler progress", {
  skip_on_cran()
  set.seed(8)
  n <- 100
  p <- 3
  X <- matrix(runif(n * p), ncol = p)
  y <- X[, 1] + rnorm(n)
  # verbose = TRUE prints GFR/MCMC progress to the console.
  expect_output(
    bart(
      X_train = X, y_train = y,
      num_gfr = 5, num_burnin = 0, num_mcmc = 20,
      general_params = list(random_seed = 1, verbose = TRUE)
    ),
    "Running GFR sampler"
  )
  # verbose = FALSE is silent. Assign the result inside capture.output so the
  # returned model is not auto-printed (its print method itself mentions GFR/MCMC).
  silent_output <- capture.output(
    fit_silent <- bart(
      X_train = X, y_train = y,
      num_gfr = 5, num_burnin = 0, num_mcmc = 20,
      general_params = list(random_seed = 1, verbose = FALSE)
    )
  )
  expect_false(any(grepl("Running GFR sampler|GFR:|MCMC:", silent_output)))
})
