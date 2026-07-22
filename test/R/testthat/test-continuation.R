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

test_that("BART continuation supports test data", {
  skip_on_cran()

  n <- 200
  p <- 4
  X <- matrix(runif(n * p), ncol = p)
  y <- 2 * X[, 1] - 1 + rnorm(n, 0, 0.3)
  n_test <- 50
  X_test <- matrix(runif(n_test * p), ncol = p)

  # Fit with no test set, then supply one on continuation.
  m <- bart(
    X_train = X, y_train = y, num_gfr = 0, num_burnin = 5, num_mcmc = 10,
    general_params = list(random_seed = 7)
  )
  m <- continueSampling(m, X_train = X, y_train = y, X_test = X_test, num_mcmc = 8)

  expect_equal(m$model_params$num_samples, 18)
  # Test predictions are recomputed from ALL retained forests, so the stored test-pred trace covers
  # every sample and is bit-identical to a fresh predict() on the same X_test.
  stored <- m$samples$y_hat_test()
  expect_equal(dim(stored), c(n_test, 18))
  expect_equal(stored, predict(m, X_test)$y_hat, tolerance = 1e-10)
})

test_that("BART continuation drops stale test predictions with a warning", {
  skip_on_cran()

  n <- 200
  p <- 4
  X <- matrix(runif(n * p), ncol = p)
  y <- 2 * X[, 1] - 1 + rnorm(n, 0, 0.3)
  X_test <- matrix(runif(50 * p), ncol = p)

  # Fit WITH a test set, then continue WITHOUT re-supplying it.
  m <- bart(
    X_train = X, y_train = y, X_test = X_test,
    num_gfr = 0, num_burnin = 5, num_mcmc = 10,
    general_params = list(random_seed = 7)
  )
  expect_true(m$samples$has_mean_forest_predictions_test())

  expect_warning(
    m <- continueSampling(m, X_train = X, y_train = y, num_mcmc = 8),
    "test-set predictions are stale"
  )
  # Stale test predictions were dropped, and the accessor reflects the cleared state.
  expect_false(m$samples$has_mean_forest_predictions_test())
  expect_equal(m$model_params$num_samples, 18)
})

# --- BCF continuation ----------------------------------------------------------------------------

test_that("BCF continuation appends draws (basic + adaptive + variance + rfx)", {
  skip_on_cran()

  set.seed(101)
  n <- 200
  p <- 5
  X <- matrix(runif(n * p), ncol = p)
  pi_x <- 0.25 + 0.5 * X[, 1]
  Z <- rbinom(n, 1, pi_x)
  g <- sample(1:3, n, replace = TRUE)
  rb <- matrix(rep(1, n), ncol = 1)
  y <- X[, 1] * 2 + X[, 2] * (-1) * Z + (-2 * (g == 1) + 2 * (g == 2)) + rnorm(n)

  # Basic
  m <- bcf(X_train = X, Z_train = Z, y_train = y, propensity_train = pi_x,
           num_gfr = 0, num_burnin = 5, num_mcmc = 10)
  m <- continueSampling(m, X_train = X, Z_train = Z, y_train = y, propensity_train = pi_x, num_mcmc = 8)
  expect_equal(m$model_params$num_samples, 18)

  # Adaptive coding
  ma <- bcf(X_train = X, Z_train = Z, y_train = y, propensity_train = pi_x,
            num_gfr = 0, num_burnin = 5, num_mcmc = 10,
            general_params = list(adaptive_coding = TRUE))
  ma <- continueSampling(ma, X_train = X, Z_train = Z, y_train = y, propensity_train = pi_x, num_mcmc = 8)
  expect_equal(ma$model_params$num_samples, 18)
  expect_length(ma$samples$b0_samples(), 18)

  # Variance forest
  mv <- bcf(X_train = X, Z_train = Z, y_train = y, propensity_train = pi_x,
            num_gfr = 0, num_burnin = 5, num_mcmc = 10,
            variance_forest_params = list(num_trees = 20))
  mv <- continueSampling(mv, X_train = X, Z_train = Z, y_train = y, propensity_train = pi_x, num_mcmc = 8)
  expect_equal(mv$model_params$num_samples, 18)

  # Random effects
  mr <- bcf(X_train = X, Z_train = Z, y_train = y, propensity_train = pi_x,
            rfx_group_ids_train = g, rfx_basis_train = rb,
            num_gfr = 0, num_burnin = 5, num_mcmc = 10)
  mr <- continueSampling(mr, X_train = X, Z_train = Z, y_train = y, propensity_train = pi_x,
                         rfx_group_ids_train = g, rfx_basis_train = rb, num_mcmc = 8)
  expect_equal(mr$model_params$num_samples, 18)
  bs <- extractRandomEffectSamples(mr)$beta_samples
  expect_equal(dim(bs)[length(dim(bs))], 18)  # last dim = num_samples

  # random_effects_params is a changeable dict on continuation (no warning, proceeds).
  mr2 <- bcf(X_train = X, Z_train = Z, y_train = y, propensity_train = pi_x,
             rfx_group_ids_train = g, rfx_basis_train = rb,
             num_gfr = 0, num_burnin = 5, num_mcmc = 10)
  expect_no_warning(
    mr2 <- continueSampling(mr2, X_train = X, Z_train = Z, y_train = y, propensity_train = pi_x,
                            rfx_group_ids_train = g, rfx_basis_train = rb, num_mcmc = 8,
                            random_effects_params = list(variance_prior_shape = 2, variance_prior_scale = 2))
  )
  expect_equal(mr2$model_params$num_samples, 18)
})

test_that("BCF continuation is deterministic", {
  skip_on_cran()

  set.seed(202)
  n <- 200
  p <- 5
  X <- matrix(runif(n * p), ncol = p)
  pi_x <- 0.25 + 0.5 * X[, 1]
  Z <- rbinom(n, 1, pi_x)
  y <- X[, 1] * 2 + X[, 2] * (-1) * Z + rnorm(n)

  run <- function() {
    m <- bcf(X_train = X, Z_train = Z, y_train = y, propensity_train = pi_x,
             num_gfr = 0, num_burnin = 5, num_mcmc = 10,
             general_params = list(random_seed = 99))
    m <- continueSampling(m, X_train = X, Z_train = Z, y_train = y, propensity_train = pi_x, num_mcmc = 8)
    rowMeans(predict(m, X = X, Z = Z, propensity = pi_x)$y_hat)
  }
  expect_equal(run(), run())
})

test_that("BCF continuation re-specifies split-variable selection without warning", {
  skip_on_cran()

  set.seed(303)
  n <- 200
  p <- 5
  X <- matrix(runif(n * p), ncol = p)
  pi_x <- 0.25 + 0.5 * X[, 1]
  Z <- rbinom(n, 1, pi_x)
  y <- X[, 1] * 2 + X[, 2] * (-1) * Z + rnorm(n)

  # continueSampling appends in place onto the model's shared C++ samples, so use a fresh base model
  # for each independent continuation.
  fresh <- function() {
    bcf(
      X_train = X, Z_train = Z, y_train = y, propensity_train = pi_x,
      num_gfr = 0, num_burnin = 5, num_mcmc = 6,
      general_params = list(random_seed = 7)
    )
  }

  # keep_vars / drop_vars (per forest) + variable_weights are changeable: no warning, continuation proceeds.
  expect_no_warning(
    m1 <- continueSampling(
      fresh(), X_train = X, Z_train = Z, y_train = y, propensity_train = pi_x, num_mcmc = 6,
      prognostic_forest_params = list(keep_vars = c(1, 2)),
      treatment_effect_forest_params = list(drop_vars = c(3, 4, 5))
    )
  )
  expect_equal(m1$model_params$num_samples, 12)

  expect_no_warning(
    m2 <- continueSampling(
      fresh(), X_train = X, Z_train = Z, y_train = y, propensity_train = pi_x, num_mcmc = 6,
      general_params = list(variable_weights = c(0.5, 0.5, 0, 0, 0))
    )
  )
  expect_equal(m2$model_params$num_samples, 12)

  # Malformed variable_weights are rejected (wrong length).
  expect_error(
    continueSampling(
      fresh(), X_train = X, Z_train = Z, y_train = y, propensity_train = pi_x, num_mcmc = 6,
      general_params = list(variable_weights = c(0.5, 0.5))
    )
  )

  # keep_vars by name works on a data.frame model.
  Xdf <- as.data.frame(X)
  names(Xdf) <- paste0("v", 1:5)
  md <- bcf(
    X_train = Xdf, Z_train = Z, y_train = y, propensity_train = pi_x,
    num_gfr = 0, num_burnin = 5, num_mcmc = 6,
    general_params = list(random_seed = 7)
  )
  expect_no_warning(
    md <- continueSampling(
      md, X_train = Xdf, Z_train = Z, y_train = y, propensity_train = pi_x, num_mcmc = 6,
      treatment_effect_forest_params = list(keep_vars = c("v1", "v2"))
    )
  )
  expect_equal(md$model_params$num_samples, 12)
})

test_that("BCF continuation drops stale test predictions with a warning", {
  skip_on_cran()

  set.seed(303)
  n <- 200
  p <- 5
  X <- matrix(runif(n * p), ncol = p)
  pi_x <- 0.25 + 0.5 * X[, 1]
  Z <- rbinom(n, 1, pi_x)
  y <- X[, 1] * 2 + X[, 2] * (-1) * Z + rnorm(n)
  test_inds <- 1:40
  m <- bcf(X_train = X, Z_train = Z, y_train = y, propensity_train = pi_x,
           X_test = X[test_inds, ], Z_test = Z[test_inds], propensity_test = pi_x[test_inds],
           num_gfr = 0, num_burnin = 5, num_mcmc = 10)
  expect_warning(
    m <- continueSampling(m, X_train = X, Z_train = Z, y_train = y, propensity_train = pi_x, num_mcmc = 8),
    "test-set predictions are stale"
  )
  expect_equal(m$model_params$num_samples, 18)
})

test_that("BCF continuation supports a re-supplied test set", {
  skip_on_cran()

  set.seed(313)
  n <- 200
  p <- 5
  X <- matrix(runif(n * p), ncol = p)
  pi_x <- 0.25 + 0.5 * X[, 1]
  Z <- rbinom(n, 1, pi_x)
  y <- X[, 1] * 2 + X[, 2] * (-1) * Z + rnorm(n)
  test_inds <- 1:40
  Xte <- X[test_inds, ]
  Zte <- Z[test_inds]
  pite <- pi_x[test_inds]

  # Continuation recomputes the FULL test-prediction trace from all retained forests, so the stored
  # test preds match a direct predict() call on the same test set (no warning when X_test is supplied).
  m <- bcf(X_train = X, Z_train = Z, y_train = y, propensity_train = pi_x,
           num_gfr = 0, num_burnin = 5, num_mcmc = 10, general_params = list(random_seed = 5))
  expect_no_warning(
    m <- continueSampling(m, X_train = X, Z_train = Z, y_train = y, propensity_train = pi_x,
                          X_test = Xte, Z_test = Zte, propensity_test = pite, num_mcmc = 8)
  )
  expect_equal(m$model_params$num_samples, 18)
  y_hat_test <- matrix(m$samples$y_hat_test(), nrow = length(test_inds))
  expect_equal(dim(y_hat_test), c(length(test_inds), 18))
  # Stored test preds equal a direct predict() over the full posterior.
  pred <- predict(m, X = Xte, Z = Zte, propensity = pite)
  expect_equal(y_hat_test, pred$y_hat, tolerance = 1e-8)
})

test_that("BCF continuation supports multivariate treatment", {
  skip_on_cran()

  set.seed(404)
  n <- 300
  p <- 5
  X <- matrix(runif(n * p), ncol = p)
  Z <- matrix(runif(n * 2), ncol = 2)  # multivariate treatment
  y <- X[, 1] * 2 + Z[, 1] * X[, 2] - Z[, 2] * X[, 3] + rnorm(n)
  mv <- bcf(X_train = X, Z_train = Z, y_train = y, num_gfr = 0, num_burnin = 5, num_mcmc = 10)
  mv <- continueSampling(mv, X_train = X, Z_train = Z, y_train = y, num_mcmc = 8)
  expect_equal(mv$model_params$num_samples, 18)
})

test_that("BART probit continuation is supported", {
  skip_on_cran()
  set.seed(11)
  n <- 300; p <- 4
  X <- matrix(runif(n * p), ncol = p)
  z <- (2 * X[, 1] - 1) + rnorm(n)
  y <- as.numeric(z > 0)
  m <- bart(X_train = X, y_train = y, num_gfr = 0, num_burnin = 5, num_mcmc = 10,
            general_params = list(random_seed = 1, sample_sigma2_global = FALSE,
                                  outcome_model = OutcomeModel(outcome = "binary", link = "probit")))
  m <- continueSampling(m, X_train = X, y_train = y, num_mcmc = 8)
  expect_equal(m$model_params$num_samples, 18)
})

test_that("BART cloglog binary continuation is supported", {
  skip_on_cran()
  set.seed(21)
  n <- 300; p <- 4
  X <- matrix(runif(n * p), ncol = p)
  f_X <- 2 * (X[, 1] > 0.5) - 1
  y <- rbinom(n, 1, 1 - exp(-exp(f_X)))
  run <- function() {
    m <- bart(X_train = X, y_train = y, num_gfr = 0, num_burnin = 5, num_mcmc = 10,
              general_params = list(random_seed = 1, sample_sigma2_global = FALSE,
                                    outcome_model = OutcomeModel(outcome = "binary", link = "cloglog")))
    m <- continueSampling(m, X_train = X, y_train = y, num_mcmc = 8)
    m
  }
  m <- run()
  expect_equal(m$model_params$num_samples, 18)
  expect_equal(ncol(m$samples$y_hat_train()), 18)
  # Deterministic under a fixed seed (RNG state resumed).
  expect_equal(run()$samples$y_hat_train(), m$samples$y_hat_train(), tolerance = 1e-10)
})

test_that("BART cloglog ordinal continuation is supported", {
  skip_on_cran()
  set.seed(22)
  n <- 300; p <- 4
  X <- matrix(runif(n * p), ncol = p)
  # 3-category ordinal outcome
  lin <- 1.5 * X[, 1] - 0.5
  y <- as.integer(cut(lin + rnorm(n, 0, 0.5), breaks = c(-Inf, -0.3, 0.3, Inf))) - 1L
  run <- function() {
    m <- bart(X_train = X, y_train = y, num_gfr = 0, num_burnin = 5, num_mcmc = 10,
              general_params = list(random_seed = 1, sample_sigma2_global = FALSE,
                                    outcome_model = OutcomeModel(outcome = "ordinal", link = "cloglog")))
    m <- continueSampling(m, X_train = X, y_train = y, num_mcmc = 8)
    m
  }
  m <- run()
  expect_equal(m$model_params$num_samples, 18)
  # Cutpoint samples extended to the full trace (2 cutpoints for 3 categories).
  expect_equal(ncol(m$samples$cloglog_cutpoint_samples()), 18)
  # Deterministic under a fixed seed.
  expect_equal(run()$samples$y_hat_train(), m$samples$y_hat_train(), tolerance = 1e-10)
})

test_that("BCF probit continuation is supported", {
  skip_on_cran()
  set.seed(12)
  n <- 300; p <- 5
  X <- matrix(runif(n * p), ncol = p)
  pi_x <- 0.3 + 0.4 * X[, 2]
  Z <- rbinom(n, 1, pi_x)
  lin <- X[, 1] + 0.8 * X[, 3] * Z
  y <- as.numeric(lin + rnorm(n) > 0.9)
  m <- bcf(X_train = X, Z_train = Z, y_train = y, propensity_train = pi_x,
           num_gfr = 0, num_burnin = 5, num_mcmc = 10,
           general_params = list(random_seed = 1, sample_sigma2_global = FALSE,
                                 outcome_model = OutcomeModel(outcome = "binary", link = "probit")))
  m <- continueSampling(m, X_train = X, Z_train = Z, y_train = y, propensity_train = pi_x, num_mcmc = 8)
  expect_equal(m$model_params$num_samples, 18)
})
