# Tests for multi-chain BART and BCF sampling.
#
# Covers sample-count correctness, GFR warm-start path, chain independence,
# extractParameter dimensions, serialization round-trip, and the
# num_gfr >= num_chains validation.

# ---------------------------------------------------------------------------
# Shared test data helpers
# ---------------------------------------------------------------------------

.make_bart_data <- function() {
  set.seed(42)
  n <- 200; p <- 5
  X <- matrix(runif(n * p), ncol = p)
  y <- 5 * X[, 1] + rnorm(n)
  test_inds  <- sort(sample(1:n, 40))
  train_inds <- setdiff(1:n, test_inds)
  list(
    X_train  = X[train_inds, ],
    X_test   = X[test_inds,  ],
    y_train  = y[train_inds],
    n_train  = length(train_inds),
    n_test   = length(test_inds)
  )
}

.make_bcf_data <- function() {
  set.seed(42)
  n <- 200; p <- 5
  X  <- matrix(runif(n * p), ncol = p)
  pi_X <- 0.25 + 0.5 * X[, 1]
  Z  <- rbinom(n, 1, pi_X)
  y  <- 5 * X[, 1] + 2 * X[, 2] * Z + rnorm(n)
  test_inds  <- sort(sample(1:n, 40))
  train_inds <- setdiff(1:n, test_inds)
  list(
    X_train   = X[train_inds, ],
    X_test    = X[test_inds,  ],
    Z_train   = Z[train_inds],
    Z_test    = Z[test_inds],
    y_train   = y[train_inds],
    pi_train  = pi_X[train_inds],
    pi_test   = pi_X[test_inds],
    n_train   = length(train_inds),
    n_test    = length(test_inds)
  )
}

# ---------------------------------------------------------------------------
# BARTModel multi-chain tests
# ---------------------------------------------------------------------------

test_that("BART multi-chain: sample counts with no GFR", {
  skip_on_cran()
  d <- .make_bart_data()
  n_chains <- 3; n_mcmc <- 10
  m <- bart(
    X_train = d$X_train, y_train = d$y_train, X_test = d$X_test,
    num_gfr = 0, num_burnin = 0, num_mcmc = n_mcmc,
    general_params = list(num_chains = n_chains, num_threads = 1)
  )
  expected <- n_chains * n_mcmc
  expect_length(m$samples$global_var_samples(), expected)
  expect_equal(dim(m$samples$y_hat_train()), c(d$n_train, expected))
  expect_equal(dim(m$samples$y_hat_test()),  c(d$n_test,  expected))
})

test_that("BART multi-chain: sample counts with GFR warm-start", {
  skip_on_cran()
  d <- .make_bart_data()
  n_chains <- 3; n_mcmc <- 10; n_gfr <- 6
  m <- bart(
    X_train = d$X_train, y_train = d$y_train, X_test = d$X_test,
    num_gfr = n_gfr, num_burnin = 5, num_mcmc = n_mcmc,
    general_params = list(num_chains = n_chains, num_threads = 1)
  )
  expected <- n_chains * n_mcmc
  expect_length(m$samples$global_var_samples(), expected)
  expect_equal(dim(m$samples$y_hat_train()), c(d$n_train, expected))
})

test_that("BART multi-chain: leaf-scale sample count", {
  skip_on_cran()
  d <- .make_bart_data()
  n_chains <- 3; n_mcmc <- 10
  m <- bart(
    X_train = d$X_train, y_train = d$y_train, X_test = d$X_test,
    num_gfr = 0, num_burnin = 0, num_mcmc = n_mcmc,
    general_params = list(
      num_chains = n_chains, num_threads = 1,
      sample_sigma2_global = FALSE
    ),
    mean_forest_params = list(sample_sigma2_leaf = TRUE)
  )
  expect_length(m$samples$leaf_scale_samples(), n_chains * n_mcmc)
})

test_that("BART multi-chain: chain independence (no GFR)", {
  skip_on_cran()
  d <- .make_bart_data()
  n_mcmc <- 10
  m <- bart(
    X_train = d$X_train, y_train = d$y_train, X_test = d$X_test,
    num_gfr = 0, num_burnin = 0, num_mcmc = n_mcmc,
    general_params = list(num_chains = 2, num_threads = 1)
  )
  chain1 <- m$samples$global_var_samples()[seq_len(n_mcmc)]
  chain2 <- m$samples$global_var_samples()[seq(n_mcmc + 1, 2 * n_mcmc)]
  expect_false(isTRUE(all.equal(chain1, chain2)),
               label = "Chains should produce distinct sigma2 samples")
})

test_that("BART multi-chain: chain independence (with GFR)", {
  skip_on_cran()
  d <- .make_bart_data()
  n_mcmc <- 10; n_gfr <- 4
  m <- bart(
    X_train = d$X_train, y_train = d$y_train, X_test = d$X_test,
    num_gfr = n_gfr, num_burnin = 5, num_mcmc = n_mcmc,
    general_params = list(num_chains = 2, num_threads = 1)
  )
  chain1 <- m$samples$global_var_samples()[seq_len(n_mcmc)]
  chain2 <- m$samples$global_var_samples()[seq(n_mcmc + 1, 2 * n_mcmc)]
  expect_false(isTRUE(all.equal(chain1, chain2)))
})

test_that("BART multi-chain: extractParameter dimensions", {
  skip_on_cran()
  d <- .make_bart_data()
  n_chains <- 3; n_mcmc <- 10
  m <- bart(
    X_train = d$X_train, y_train = d$y_train, X_test = d$X_test,
    num_gfr = 0, num_burnin = 0, num_mcmc = n_mcmc,
    general_params = list(num_chains = n_chains, num_threads = 1)
  )
  expected <- n_chains * n_mcmc
  s2 <- extractParameter(m, "sigma2_global")
  expect_length(s2, expected)
  yht <- extractParameter(m, "y_hat_train")
  expect_equal(dim(yht), c(d$n_train, expected))
})

test_that("BART multi-chain: predict() shape and finiteness (no GFR)", {
  skip_on_cran()
  d <- .make_bart_data()
  n_chains <- 3; n_mcmc <- 10
  m <- bart(
    X_train = d$X_train, y_train = d$y_train, X_test = d$X_test,
    num_gfr = 0, num_burnin = 0, num_mcmc = n_mcmc,
    general_params = list(num_chains = n_chains, num_threads = 1)
  )
  expected_cols <- n_chains * n_mcmc
  result <- predict(m, X = d$X_test, terms = "y_hat")
  expect_equal(dim(result), c(d$n_test, expected_cols))
  expect_true(all(is.finite(result)))
})

test_that("BART multi-chain: predict() shape and finiteness (GFR path)", {
  skip_on_cran()
  d <- .make_bart_data()
  n_chains <- 3; n_mcmc <- 10
  m <- bart(
    X_train = d$X_train, y_train = d$y_train, X_test = d$X_test,
    num_gfr = 6, num_burnin = 5, num_mcmc = n_mcmc,
    general_params = list(num_chains = n_chains, num_threads = 1)
  )
  result <- predict(m, X = d$X_test, terms = "y_hat")
  expect_equal(dim(result), c(d$n_test, n_chains * n_mcmc))
  expect_true(all(is.finite(result)))
})

test_that("BART multi-chain: num_gfr < num_chains raises an error", {
  skip_on_cran()
  d <- .make_bart_data()
  expect_error(
    bart(
      X_train = d$X_train, y_train = d$y_train, X_test = d$X_test,
      num_gfr = 2, num_burnin = 0, num_mcmc = 5,
      general_params = list(num_chains = 4, num_threads = 1)
    )
  )
})

test_that("BART multi-chain: sigma2 samples are finite and positive with GFR", {
  skip_on_cran()
  d <- .make_bart_data()
  m <- bart(
    X_train = d$X_train, y_train = d$y_train, X_test = d$X_test,
    num_gfr = 6, num_burnin = 10, num_mcmc = 10,
    general_params = list(num_chains = 3, num_threads = 1)
  )
  expect_true(all(is.finite(m$samples$global_var_samples())))
  expect_true(all(m$samples$global_var_samples() > 0))
})

# Regression test: RestoreStateFromGFRSnapshot used to recreate samples.variance_forests
# at the start of every chain after the first, so the variance forest container ended up
# holding only the LAST chain's draws while num_samples counted all of them.  predict()
# then read num_samples columns out of a shorter container (heap over-read -> garbage).
# Only fires on the GFR warm-start path, hence num_gfr > 0 here.
test_that("BART multi-chain: variance forest container holds every chain's draws (GFR path)", {
  skip_on_cran()
  d <- .make_bart_data()
  n_chains <- 3; n_mcmc <- 10
  m <- bart(
    X_train = d$X_train, y_train = d$y_train, X_test = d$X_test,
    num_gfr = 6, num_burnin = 5, num_mcmc = n_mcmc,
    general_params = list(num_chains = n_chains, num_threads = 1),
    variance_forest_params = list(num_trees = 10)
  )
  expect_equal(m$model_params$num_samples, n_chains * n_mcmc)
  expect_equal(extractForest(m, "mean")$num_samples(), n_chains * n_mcmc)
  expect_equal(extractForest(m, "variance")$num_samples(), n_chains * n_mcmc)

  post <- predict(m, X = d$X_test, terms = "variance_forest", type = "posterior")
  expect_equal(dim(post), c(d$n_test, n_chains * n_mcmc))
  expect_true(all(is.finite(post)))
  expect_true(all(post > 0))

  avg <- predict(m, X = d$X_test, terms = "variance_forest", type = "mean")
  expect_length(avg, d$n_test)
  expect_true(all(is.finite(avg)))
  expect_true(all(avg > 0))
})

test_that("BART multi-chain: variance forest recovers a step function (GFR path)", {
  skip_on_cran()
  set.seed(101)
  n <- 500; p <- 5
  X <- matrix(runif(n * p), ncol = p)
  sigma_x <- ifelse(X[, 1] < 0.5, 0.5, 3)
  y <- 2 * X[, 2] + rnorm(n) * sigma_x
  m <- bart(
    X_train = X, y_train = y,
    num_gfr = 10, num_burnin = 0, num_mcmc = 20,
    general_params = list(
      sample_sigma2_global = FALSE, num_chains = 4, num_threads = 1, random_seed = 101
    ),
    mean_forest_params = list(sample_sigma2_leaf = FALSE, num_trees = 20),
    variance_forest_params = list(num_trees = 20)
  )
  sigma2_hat <- predict(m, X = X, terms = "variance_forest", type = "mean")
  # Both groups should land in the right neighborhood (truth: 0.25 and 9) and be
  # clearly separated.  Under the container-wipe bug this read out of bounds and
  # produced values spanning many orders of magnitude, including negatives.
  lo <- mean(sigma2_hat[X[, 1] < 0.5])
  hi <- mean(sigma2_hat[X[, 1] >= 0.5])
  expect_true(all(sigma2_hat > 0))
  expect_true(lo > 0.05 && lo < 1.5)
  expect_true(hi > 4 && hi < 16)
  expect_gt(hi / lo, 4)
})

# ---------------------------------------------------------------------------
# BCFModel multi-chain tests
# ---------------------------------------------------------------------------

test_that("BCF multi-chain: sample counts with no GFR", {
  skip_on_cran()
  d <- .make_bcf_data()
  n_chains <- 3; n_mcmc <- 10
  m <- bcf(
    X_train = d$X_train, Z_train = d$Z_train, y_train = d$y_train,
    propensity_train = d$pi_train,
    X_test = d$X_test, Z_test = d$Z_test, propensity_test = d$pi_test,
    num_gfr = 0, num_burnin = 10, num_mcmc = n_mcmc,
    general_params = list(num_chains = n_chains, num_threads = 1)
  )
  expected <- n_chains * n_mcmc
  expect_length(m$samples$global_var_samples(), expected)
  expect_equal(dim(m$samples$tau_forest_predictions_train()), c(d$n_train, expected))
  expect_equal(dim(m$samples$mu_forest_predictions_train()),  c(d$n_train, expected))
  expect_equal(dim(m$samples$tau_forest_predictions_test()),  c(d$n_test,  expected))
})

test_that("BCF multi-chain: sample counts with GFR warm-start", {
  skip_on_cran()
  d <- .make_bcf_data()
  n_chains <- 3; n_mcmc <- 10; n_gfr <- 6
  m <- bcf(
    X_train = d$X_train, Z_train = d$Z_train, y_train = d$y_train,
    propensity_train = d$pi_train,
    X_test = d$X_test, Z_test = d$Z_test, propensity_test = d$pi_test,
    num_gfr = n_gfr, num_burnin = 5, num_mcmc = n_mcmc,
    general_params = list(num_chains = n_chains, num_threads = 1, adaptive_coding = TRUE)
  )
  expected <- n_chains * n_mcmc
  expect_length(m$samples$global_var_samples(), expected)
  expect_equal(dim(m$samples$tau_forest_predictions_train()), c(d$n_train, expected))
  expect_equal(dim(m$samples$mu_forest_predictions_train()),  c(d$n_train, expected))
  # BCF-specific scalar parameter arrays
  expect_length(m$samples$b0_samples(), expected)
  expect_length(m$samples$b1_samples(), expected)
  expect_length(m$samples$leaf_scale_mu_samples(), expected)
})

test_that("BCF multi-chain: chain independence (no GFR)", {
  skip_on_cran()
  d <- .make_bcf_data()
  n_mcmc <- 10
  m <- bcf(
    X_train = d$X_train, Z_train = d$Z_train, y_train = d$y_train,
    propensity_train = d$pi_train,
    X_test = d$X_test, Z_test = d$Z_test, propensity_test = d$pi_test,
    num_gfr = 0, num_burnin = 10, num_mcmc = n_mcmc,
    general_params = list(num_chains = 2, num_threads = 1)
  )
  chain1 <- m$samples$global_var_samples()[seq_len(n_mcmc)]
  chain2 <- m$samples$global_var_samples()[seq(n_mcmc + 1, 2 * n_mcmc)]
  expect_false(isTRUE(all.equal(chain1, chain2)),
               label = "BCF chains should produce distinct sigma2 samples")
})

test_that("BCF multi-chain: chain independence (with GFR)", {
  skip_on_cran()
  d <- .make_bcf_data()
  n_mcmc <- 10
  m <- bcf(
    X_train = d$X_train, Z_train = d$Z_train, y_train = d$y_train,
    propensity_train = d$pi_train,
    X_test = d$X_test, Z_test = d$Z_test, propensity_test = d$pi_test,
    num_gfr = 4, num_burnin = 5, num_mcmc = n_mcmc,
    general_params = list(num_chains = 2, num_threads = 1)
  )
  chain1 <- m$samples$global_var_samples()[seq_len(n_mcmc)]
  chain2 <- m$samples$global_var_samples()[seq(n_mcmc + 1, 2 * n_mcmc)]
  expect_false(isTRUE(all.equal(chain1, chain2)))
})

test_that("BCF multi-chain: all samples finite with GFR + multiple chains", {
  skip_on_cran()
  # Exercises the tau_0 / adaptive-coding reset logic introduced to prevent
  # residual blowup when transitioning between chains.
  d <- .make_bcf_data()
  m <- bcf(
    X_train = d$X_train, Z_train = d$Z_train, y_train = d$y_train,
    propensity_train = d$pi_train,
    X_test = d$X_test, Z_test = d$Z_test, propensity_test = d$pi_test,
    num_gfr = 6, num_burnin = 20, num_mcmc = 10,
    general_params = list(num_chains = 3, num_threads = 1, adaptive_coding = TRUE)
  )
  expect_true(all(is.finite(m$samples$global_var_samples())),
              label = "sigma2 samples must be finite (no chain-transition blowup)")
  expect_true(all(m$samples$global_var_samples() > 0))
  expect_true(all(is.finite(m$samples$b0_samples())))
  expect_true(all(is.finite(m$samples$b1_samples())))
})

test_that("BCF multi-chain: extractParameter dimensions", {
  skip_on_cran()
  d <- .make_bcf_data()
  n_chains <- 3; n_mcmc <- 10
  m <- bcf(
    X_train = d$X_train, Z_train = d$Z_train, y_train = d$y_train,
    propensity_train = d$pi_train,
    X_test = d$X_test, Z_test = d$Z_test, propensity_test = d$pi_test,
    num_gfr = 0, num_burnin = 10, num_mcmc = n_mcmc,
    general_params = list(num_chains = n_chains, num_threads = 1)
  )
  expected <- n_chains * n_mcmc
  s2 <- extractParameter(m, "sigma2_global")
  expect_length(s2, expected)
  tau_test <- extractParameter(m, "tau_hat_test")
  expect_equal(dim(tau_test), c(d$n_test, expected))
  tau_train <- extractParameter(m, "tau_hat_train")
  expect_equal(dim(tau_train), c(d$n_train, expected))
})

test_that("BCF multi-chain: predict() shape and finiteness for all forest terms (no GFR)", {
  skip_on_cran()
  d <- .make_bcf_data()
  n_chains <- 3; n_mcmc <- 10
  m <- bcf(
    X_train = d$X_train, Z_train = d$Z_train, y_train = d$y_train,
    propensity_train = d$pi_train,
    X_test = d$X_test, Z_test = d$Z_test, propensity_test = d$pi_test,
    num_gfr = 0, num_burnin = 10, num_mcmc = n_mcmc,
    general_params = list(num_chains = n_chains, num_threads = 1)
  )
  expected_cols <- n_chains * n_mcmc
  for (term in c("y_hat", "cate", "prognostic_function", "mu", "tau")) {
    result <- predict(m, X = d$X_test, Z = d$Z_test, propensity = d$pi_test, terms = term)
    expect_equal(dim(result), c(d$n_test, expected_cols),
                 label = paste0("dim for term='", term, "'"))
    expect_true(all(is.finite(result)),
                label = paste0("finiteness for term='", term, "'"))
  }
})

test_that("BCF multi-chain: predict() shape and finiteness for all forest terms (GFR path)", {
  skip_on_cran()
  d <- .make_bcf_data()
  n_chains <- 3; n_mcmc <- 10; n_gfr <- 6
  m <- bcf(
    X_train = d$X_train, Z_train = d$Z_train, y_train = d$y_train,
    propensity_train = d$pi_train,
    X_test = d$X_test, Z_test = d$Z_test, propensity_test = d$pi_test,
    num_gfr = n_gfr, num_burnin = 5, num_mcmc = n_mcmc,
    general_params = list(num_chains = n_chains, num_threads = 1)
  )
  expected_cols <- n_chains * n_mcmc
  for (term in c("y_hat", "cate", "prognostic_function", "mu", "tau")) {
    result <- predict(m, X = d$X_test, Z = d$Z_test, propensity = d$pi_test, terms = term)
    expect_equal(dim(result), c(d$n_test, expected_cols),
                 label = paste0("dim for term='", term, "'"))
    expect_true(all(is.finite(result)),
                label = paste0("finiteness for term='", term, "'"))
  }
})

test_that("BCF multi-chain: predict() shape and positivity for variance forest term", {
  skip_on_cran()
  d <- .make_bcf_data()
  n_chains <- 3; n_mcmc <- 10
  m <- bcf(
    X_train = d$X_train, Z_train = d$Z_train, y_train = d$y_train,
    propensity_train = d$pi_train,
    X_test = d$X_test, Z_test = d$Z_test, propensity_test = d$pi_test,
    num_gfr = 0, num_burnin = 10, num_mcmc = n_mcmc,
    general_params = list(num_chains = n_chains, num_threads = 1),
    variance_forest_params = list(num_trees = 10)
  )
  result <- predict(m, X = d$X_test, Z = d$Z_test, propensity = d$pi_test,
                    terms = "variance_forest")
  expect_equal(dim(result), c(d$n_test, n_chains * n_mcmc))
  expect_true(all(is.finite(result)))
  expect_true(all(result > 0))
})

# Regression test: BCFSampler::RestoreStateFromGFRSnapshot had the same container-wipe
# bug as the BART sampler -- samples.variance_forests was recreated per chain, so it kept
# only the last chain's draws while num_samples counted all of them.  The no-GFR test
# above cannot catch it; the snapshot restore path requires num_gfr > 0.
test_that("BCF multi-chain: variance forest container holds every chain's draws (GFR path)", {
  skip_on_cran()
  d <- .make_bcf_data()
  n_chains <- 3; n_mcmc <- 10
  m <- bcf(
    X_train = d$X_train, Z_train = d$Z_train, y_train = d$y_train,
    propensity_train = d$pi_train,
    X_test = d$X_test, Z_test = d$Z_test, propensity_test = d$pi_test,
    num_gfr = 6, num_burnin = 5, num_mcmc = n_mcmc,
    general_params = list(num_chains = n_chains, num_threads = 1),
    variance_forest_params = list(num_trees = 10)
  )
  expect_equal(m$model_params$num_samples, n_chains * n_mcmc)
  expect_equal(extractForest(m, "variance")$num_samples(), n_chains * n_mcmc)

  post <- predict(m, X = d$X_test, Z = d$Z_test, propensity = d$pi_test,
                  terms = "variance_forest", type = "posterior")
  expect_equal(dim(post), c(d$n_test, n_chains * n_mcmc))
  expect_true(all(is.finite(post)))
  expect_true(all(post > 0))

  avg <- predict(m, X = d$X_test, Z = d$Z_test, propensity = d$pi_test,
                 terms = "variance_forest", type = "mean")
  expect_length(avg, d$n_test)
  expect_true(all(is.finite(avg)))
  expect_true(all(avg > 0))
})

test_that("BCF multi-chain: num_gfr < num_chains raises an error", {
  skip_on_cran()
  d <- .make_bcf_data()
  expect_error(
    bcf(
      X_train = d$X_train, Z_train = d$Z_train, y_train = d$y_train,
      propensity_train = d$pi_train,
      X_test = d$X_test, Z_test = d$Z_test, propensity_test = d$pi_test,
      num_gfr = 2, num_burnin = 0, num_mcmc = 5,
      general_params = list(num_chains = 4, num_threads = 1)
    )
  )
})

test_that("BCF multi-chain: serialization round-trip preserves predictions", {
  skip_on_cran()
  d <- .make_bcf_data()
  n_chains <- 2; n_mcmc <- 10; n_gfr <- 4
  m <- bcf(
    X_train = d$X_train, Z_train = d$Z_train, y_train = d$y_train,
    propensity_train = d$pi_train,
    X_test = d$X_test, Z_test = d$Z_test, propensity_test = d$pi_test,
    num_gfr = n_gfr, num_burnin = 5, num_mcmc = n_mcmc,
    general_params = list(num_chains = n_chains, num_threads = 1)
  )

  json_str <- saveBCFModelToJsonString(m)
  m2 <- createBCFModelFromJsonString(json_str)

  pred_orig <- predict(
    m, X = d$X_test, Z = d$Z_test, propensity = d$pi_test, terms = "cate"
  )
  pred_rt <- predict(
    m2, X = d$X_test, Z = d$Z_test, propensity = d$pi_test, terms = "cate"
  )
  expect_equal(dim(pred_orig), dim(pred_rt))
  expect_equal(pred_orig, pred_rt)
})
