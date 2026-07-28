make_bart_data <- function(n = 100, p = 5, seed = 42) {
  set.seed(seed)
  X <- matrix(runif(n * p), ncol = p)
  y <- sin(X[, 1] * pi) + rnorm(n, 0, 0.1)
  n_train <- as.integer(0.8 * n)
  list(
    X_train = X[1:n_train, ], y_train = y[1:n_train],
    X_test = X[(n_train + 1):n, ], n_train = n_train, n_test = n - n_train
  )
}

make_bcf_data <- function(n = 100, p = 5, seed = 42) {
  set.seed(seed)
  X <- matrix(runif(n * p), ncol = p)
  pi_X <- 0.25 + 0.5 * X[, 1]
  Z <- rbinom(n, 1, pi_X)
  y <- pi_X * 5 + X[, 2] * 2 * Z + rnorm(n, 0, 1)
  n_train <- as.integer(0.8 * n)
  list(
    X_train = X[1:n_train, ], Z_train = Z[1:n_train], y_train = y[1:n_train],
    pi_train = pi_X[1:n_train], X_test = X[(n_train + 1):n, ],
    Z_test = Z[(n_train + 1):n], pi_test = pi_X[(n_train + 1):n],
    n_train = n_train, n_test = n - n_train
  )
}

test_that("BART: uniform weights produce identical predictions to no weights", {
  skip_on_cran()
  d <- make_bart_data()
  num_mcmc <- 10

  set.seed(1)
  m1 <- bart(
    X_train = d$X_train, y_train = d$y_train, X_test = d$X_test,
    num_gfr = 0, num_burnin = 0, num_mcmc = num_mcmc,
    general_params = list(random_seed = 1L)
  )

  set.seed(1)
  m2 <- bart(
    X_train = d$X_train, y_train = d$y_train, X_test = d$X_test,
    observation_weights = rep(1.0, d$n_train),
    num_gfr = 0, num_burnin = 0, num_mcmc = num_mcmc,
    general_params = list(random_seed = 1L)
  )

  expect_equal(m1$y_hat_train, m2$y_hat_train)
  expect_equal(m1$y_hat_test, m2$y_hat_test)
})

test_that("BART: non-uniform weights run and produce correct output shape", {
  skip_on_cran()
  d <- make_bart_data()
  num_mcmc <- 10
  weights <- runif(d$n_train, 0.5, 2.0)

  expect_no_error(
    m <- bart(
      X_train = d$X_train, y_train = d$y_train, X_test = d$X_test,
      observation_weights = weights,
      num_gfr = 0, num_burnin = 0, num_mcmc = num_mcmc
    )
  )
  expect_equal(dim(m$y_hat_train), c(d$n_train, num_mcmc))
  expect_equal(dim(m$y_hat_test), c(d$n_test, num_mcmc))
})

test_that("BART: all-zero weights (prior mode) run with num_gfr = 0", {
  skip_on_cran()
  d <- make_bart_data()
  num_mcmc <- 10

  expect_no_error(
    m <- bart(
      X_train = d$X_train, y_train = d$y_train,
      observation_weights = rep(0.0, d$n_train),
      num_gfr = 0, num_burnin = 0, num_mcmc = num_mcmc
    )
  )
  expect_equal(dim(m$y_hat_train), c(d$n_train, num_mcmc))
})

test_that("BART: non-numeric observation_weights raises error", {
  skip_on_cran()
  d <- make_bart_data()
  expect_error(
    bart(
      X_train = d$X_train, y_train = d$y_train,
      observation_weights = as.character(rep(1, d$n_train)),
      num_gfr = 0, num_burnin = 0, num_mcmc = 5
    ),
    "numeric"
  )
})

test_that("BART: wrong-length observation_weights raises error", {
  skip_on_cran()
  d <- make_bart_data()
  expect_error(
    bart(
      X_train = d$X_train, y_train = d$y_train,
      observation_weights = rep(1.0, d$n_train + 1),
      num_gfr = 0, num_burnin = 0, num_mcmc = 5
    ),
    "nrow"
  )
})

test_that("BART: negative observation_weights raises error", {
  skip_on_cran()
  d <- make_bart_data()
  weights <- rep(1.0, d$n_train)
  weights[1] <- -1.0
  expect_error(
    bart(
      X_train = d$X_train, y_train = d$y_train,
      observation_weights = weights,
      num_gfr = 0, num_burnin = 0, num_mcmc = 5
    ),
    "negative"
  )
})

test_that("BART: all-zero weights with num_gfr > 0 raises error", {
  skip_on_cran()
  d <- make_bart_data()
  expect_error(
    bart(
      X_train = d$X_train, y_train = d$y_train,
      observation_weights = rep(0.0, d$n_train),
      num_gfr = 5, num_burnin = 0, num_mcmc = 10
    ),
    "num_gfr"
  )
})

test_that("BART: observation_weights with cloglog outcome raises error", {
  skip_on_cran()
  d <- make_bart_data()
  y_ord <- sample(1:3, d$n_train, replace = TRUE)
  expect_error(
    bart(
      X_train = d$X_train, y_train = y_ord,
      observation_weights = rep(1.0, d$n_train),
      num_gfr = 0, num_burnin = 0, num_mcmc = 5,
      general_params = list(outcome_model = OutcomeModel(outcome = "ordinal", link = "cloglog"))
    ),
    "cloglog"
  )
})

test_that("BART: observation_weights with variance forest raises warning", {
  skip_on_cran()
  d <- make_bart_data()
  expect_warning(
    bart(
      X_train = d$X_train, y_train = d$y_train,
      observation_weights = rep(1.0, d$n_train),
      num_gfr = 0, num_burnin = 0, num_mcmc = 5,
      variance_forest_params = list(num_trees = 5)
    ),
    "variance forest"
  )
})

test_that("BCF: uniform weights produce identical predictions to no weights", {
  skip_on_cran()
  d <- make_bcf_data()
  num_mcmc <- 10

  set.seed(1)
  m1 <- bcf(
    X_train = d$X_train, Z_train = d$Z_train, y_train = d$y_train,
    propensity_train = d$pi_train, X_test = d$X_test,
    Z_test = d$Z_test, propensity_test = d$pi_test,
    num_gfr = 0, num_burnin = 0, num_mcmc = num_mcmc,
    general_params = list(random_seed = 1L)
  )

  set.seed(1)
  m2 <- bcf(
    X_train = d$X_train, Z_train = d$Z_train, y_train = d$y_train,
    propensity_train = d$pi_train, X_test = d$X_test,
    Z_test = d$Z_test, propensity_test = d$pi_test,
    observation_weights = rep(1.0, d$n_train),
    num_gfr = 0, num_burnin = 0, num_mcmc = num_mcmc,
    general_params = list(random_seed = 1L)
  )

  expect_equal(m1$y_hat_train, m2$y_hat_train)
  expect_equal(m1$tau_hat_train, m2$tau_hat_train)
})

test_that("BCF: non-uniform weights run and produce correct output shape", {
  skip_on_cran()
  d <- make_bcf_data()
  num_mcmc <- 10
  weights <- runif(d$n_train, 0.5, 2.0)

  expect_no_error(
    m <- bcf(
      X_train = d$X_train, Z_train = d$Z_train, y_train = d$y_train,
      propensity_train = d$pi_train, X_test = d$X_test,
      Z_test = d$Z_test, propensity_test = d$pi_test,
      observation_weights = weights,
      num_gfr = 0, num_burnin = 0, num_mcmc = num_mcmc
    )
  )
  expect_equal(dim(m$y_hat_train), c(d$n_train, num_mcmc))
  expect_equal(dim(m$tau_hat_train), c(d$n_train, num_mcmc))
  expect_equal(dim(m$y_hat_test), c(d$n_test, num_mcmc))
  expect_equal(dim(m$tau_hat_test), c(d$n_test, num_mcmc))
})

test_that("BCF: negative observation_weights raises error", {
  skip_on_cran()
  d <- make_bcf_data()
  weights <- rep(1.0, d$n_train)
  weights[1] <- -1.0
  expect_error(
    bcf(
      X_train = d$X_train, Z_train = d$Z_train, y_train = d$y_train,
      propensity_train = d$pi_train,
      observation_weights = weights,
      num_gfr = 0, num_burnin = 0, num_mcmc = 5
    ),
    "negative"
  )
})

test_that("BCF: all-zero weights with num_gfr > 0 raises error", {
  skip_on_cran()
  d <- make_bcf_data()
  expect_error(
    bcf(
      X_train = d$X_train, Z_train = d$Z_train, y_train = d$y_train,
      propensity_train = d$pi_train,
      observation_weights = rep(0.0, d$n_train),
      num_gfr = 5, num_burnin = 0, num_mcmc = 10
    ),
    "num_gfr"
  )
})

test_that("BCF: observation_weights with cloglog outcome raises error", {
  skip_on_cran()
  d <- make_bcf_data()
  y_bin <- rbinom(d$n_train, 1, 0.5)
  expect_error(
    bcf(
      X_train = d$X_train, Z_train = d$Z_train, y_train = y_bin,
      propensity_train = d$pi_train,
      observation_weights = rep(1.0, d$n_train),
      num_gfr = 0, num_burnin = 0, num_mcmc = 5,
      general_params = list(outcome_model = OutcomeModel(outcome = "binary", link = "cloglog"))
    ),
    "cloglog"
  )
})
