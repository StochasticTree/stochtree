test_that("BART continuation appends and matches one-shot in distribution", {
  skip_on_cran()

  n <- 200
  p <- 4
  X <- matrix(runif(n * p), ncol = p)
  y <- 2 * X[, 1] - 1 + rnorm(n, 0, 0.3)
  N <- 10
  M <- 90
  seed <- 42

  ref <- bart(
    X_train = X, y_train = y, num_gfr = 0, num_burnin = 10, num_mcmc = N + M,
    general_params = list(random_seed = seed)
  )
  cont <- bart(
    X_train = X, y_train = y, num_gfr = 0, num_burnin = 10, num_mcmc = N,
    general_params = list(random_seed = seed)
  )
  cont <- continueSampling(cont, X_train = X, y_train = y, num_mcmc = M)

  expect_equal(cont$model_params$num_samples, N + M)

  pr <- predict(ref, X)$y_hat
  pc <- predict(cont, X)$y_hat
  expect_equal(ncol(pr), ncol(pc))

  # The retained history is carried forward verbatim (bit-identical).
  expect_equal(pr[, 1:N], pc[, 1:N], tolerance = 1e-10)

  # Continuation resumes the RNG stream but not the pre-drawn leaf-normal cache, so the continued
  # draws are only statistically equivalent (not bit-identical): the full-posterior mean agrees to
  # within Monte Carlo noise.
  expect_lt(max(abs(rowMeans(pr) - rowMeans(pc))), 0.25)
})

test_that("BART continuation is deterministic", {
  skip_on_cran()

  n <- 200
  p <- 4
  X <- matrix(runif(n * p), ncol = p)
  y <- 2 * X[, 1] - 1 + rnorm(n, 0, 0.3)

  run <- function() {
    m <- bart(
      X_train = X, y_train = y, num_gfr = 0, num_burnin = 5, num_mcmc = 10,
      general_params = list(random_seed = 1234)
    )
    m <- continueSampling(m, X_train = X, y_train = y, num_mcmc = 8)
    predict(m, X)$y_hat
  }
  expect_equal(run(), run())
})

test_that("BART GFR continuation appends retained warm-start draws", {
  skip_on_cran()

  n <- 200
  p <- 4
  X <- matrix(runif(n * p), ncol = p)
  y <- 2 * X[, 1] - 1 + rnorm(n, 0, 0.3)

  m <- bart(
    X_train = X, y_train = y, num_gfr = 10, num_burnin = 0, num_mcmc = 15,
    general_params = list(random_seed = 3, keep_gfr = TRUE)
  )
  expect_equal(m$model_params$num_samples, 25)
  # Append 10 more retained GFR draws (25 -> 35); no MCMC draws.
  m <- continueSampling(
    m, X_train = X, y_train = y, num_gfr = 10, num_mcmc = 0,
    general_params = list(keep_gfr = TRUE)
  )
  expect_equal(m$model_params$num_samples, 35)
})

test_that("BART variance forest continuation is supported", {
  skip_on_cran()

  n <- 200
  p <- 4
  X <- matrix(runif(n * p), ncol = p)
  y <- 2 * X[, 1] - 1 + rnorm(n, 0, 0.3)

  m <- suppressWarnings(bart(
    X_train = X, y_train = y, num_gfr = 0, num_burnin = 5, num_mcmc = 6,
    variance_forest_params = list(num_trees = 20),
    general_params = list(random_seed = 3)
  ))
  m <- continueSampling(m, X_train = X, y_train = y, num_mcmc = 6)
  expect_equal(m$model_params$num_samples, 12)
  vf <- predict(m, X)$variance_forest_predictions
  expect_equal(ncol(vf), 12)
  expect_true(all(vf > 0))
  expect_true(all(is.finite(vf)))
})

test_that("BART random effects continuation is supported", {
  skip_on_cran()

  n <- 200
  p <- 4
  X <- matrix(runif(n * p), ncol = p)
  g <- sample(0:2, n, replace = TRUE)
  rb <- matrix(1, n, 1)
  y <- 2 * X[, 1] - 1 + g * 0.5 + rnorm(n, 0, 0.3)

  m <- bart(
    X_train = X, y_train = y, rfx_group_ids_train = g, rfx_basis_train = rb,
    num_gfr = 0, num_burnin = 5, num_mcmc = 6, general_params = list(random_seed = 3)
  )
  m <- continueSampling(
    m, X_train = X, y_train = y, rfx_group_ids_train = g, rfx_basis_train = rb, num_mcmc = 6
  )
  expect_equal(m$model_params$num_samples, 12)
  yh <- predict(m, X, rfx_group_ids = g, rfx_basis = rb)$y_hat
  expect_equal(ncol(yh), 12)
  expect_true(all(is.finite(yh)))

  # Group ids must be re-supplied for an rfx model.
  expect_error(continueSampling(m, X_train = X, y_train = y, num_mcmc = 3))
})

test_that("BART continuation re-specifies split-variable selection without warning", {
  skip_on_cran()

  n <- 200
  p <- 5
  X <- matrix(runif(n * p), ncol = p)
  y <- 2 * X[, 1] - 1 + rnorm(n, 0, 0.3)

  # continueSampling appends in place onto the model's shared C++ samples (external pointer), so use a
  # fresh base model for each independent continuation.
  fresh <- function() {
    bart(
      X_train = X, y_train = y, num_gfr = 0, num_burnin = 5, num_mcmc = 6,
      general_params = list(random_seed = 3)
    )
  }

  # keep_vars / drop_vars / variable_weights are changeable: no warning, and continuation proceeds.
  expect_no_warning(
    m1 <- continueSampling(
      fresh(), X_train = X, y_train = y, num_mcmc = 6,
      mean_forest_params = list(drop_vars = c(3, 4, 5))
    )
  )
  expect_equal(m1$model_params$num_samples, 12)

  expect_no_warning(
    m2 <- continueSampling(
      fresh(), X_train = X, y_train = y, num_mcmc = 6,
      general_params = list(variable_weights = c(0.5, 0.5, 0, 0, 0))
    )
  )
  expect_equal(m2$model_params$num_samples, 12)

  # Malformed variable_weights are rejected.
  expect_error(
    continueSampling(
      fresh(), X_train = X, y_train = y, num_mcmc = 6,
      general_params = list(variable_weights = c(0.5, 0.5))
    )
  )

  # keep_vars by name works on a data.frame model.
  Xdf <- as.data.frame(X)
  names(Xdf) <- paste0("v", 1:5)
  md <- bart(
    X_train = Xdf, y_train = y, num_gfr = 0, num_burnin = 5, num_mcmc = 6,
    general_params = list(random_seed = 3)
  )
  expect_no_warning(
    md <- continueSampling(
      md, X_train = Xdf, y_train = y, num_mcmc = 6,
      mean_forest_params = list(keep_vars = c("v1", "v2"))
    )
  )
  expect_equal(md$model_params$num_samples, 12)
})

test_that("BART continuation warns on frozen parameters and applies changeable ones", {
  skip_on_cran()

  n <- 200
  p <- 4
  X <- matrix(runif(n * p), ncol = p)
  y <- 2 * X[, 1] - 1 + rnorm(n, 0, 0.3)

  m <- bart(
    X_train = X, y_train = y, num_gfr = 0, num_burnin = 5, num_mcmc = 6,
    general_params = list(random_seed = 3)
  )
  # num_trees is structural (frozen); alpha is a changeable tree-prior parameter.
  expect_warning(
    m2 <- continueSampling(
      m, X_train = X, y_train = y, num_mcmc = 4,
      mean_forest_params = list(alpha = 0.8, num_trees = 999)
    ),
    "cannot be changed on continuation"
  )
  expect_equal(m2$model_params$num_samples, 10)
})

test_that("BART continuation applies keep_every without a spurious warning", {
  skip_on_cran()

  n <- 200
  p <- 4
  X <- matrix(runif(n * p), ncol = p)
  y <- 2 * X[, 1] - 1 + rnorm(n, 0, 0.3)

  m <- bart(
    X_train = X, y_train = y, num_gfr = 0, num_burnin = 5, num_mcmc = 6,
    general_params = list(random_seed = 3)
  )
  # keep_every is a changeable knob; passing it must not trigger the "cannot be changed" warning.
  expect_no_warning(
    m2 <- continueSampling(
      m, X_train = X, y_train = y, num_mcmc = 8,
      general_params = list(keep_every = 2)
    )
  )
  # num_mcmc is the retained count (keep_every sets the thinning gap, not the count), so 8 draws are
  # retained and appended to the original 6.
  expect_equal(m2$model_params$num_samples, 14)
})
