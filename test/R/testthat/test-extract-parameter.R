test_that("extractParameter.bartmodel", {
  skip_on_cran()

  # Generate simulated data
  set.seed(42)
  n <- 100
  p <- 5
  X <- matrix(runif(n * p), ncol = p)
  y <- 5 * X[, 1] + rnorm(n)
  test_inds <- sort(sample(1:n, 20))
  train_inds <- setdiff(1:n, test_inds)
  X_train <- X[train_inds, ]
  X_test <- X[test_inds, ]
  y_train <- y[train_inds]
  n_train <- length(train_inds)
  n_test <- length(test_inds)
  num_mcmc <- 10

  # sigma2 / global_error_scale / sigma2_global all return same samples
  bart_sigma2 <- bart(
    X_train = X_train,
    y_train = y_train,
    X_test = X_test,
    num_gfr = 0,
    num_burnin = 0,
    num_mcmc = num_mcmc,
    general_params = list(sample_sigma2_global = TRUE),
    mean_forest_params = list(sample_sigma2_leaf = FALSE)
  )
  s2 <- extractParameter(bart_sigma2, "sigma2")
  expect_length(s2, num_mcmc)
  expect_equal(s2, extractParameter(bart_sigma2, "global_error_scale"))
  expect_equal(s2, extractParameter(bart_sigma2, "sigma2_global"))

  # sigma2_leaf / leaf_scale both return leaf-scale samples
  bart_leaf <- bart(
    X_train = X_train,
    y_train = y_train,
    X_test = X_test,
    num_gfr = 0,
    num_burnin = 0,
    num_mcmc = num_mcmc,
    general_params = list(sample_sigma2_global = FALSE),
    mean_forest_params = list(sample_sigma2_leaf = TRUE)
  )
  sl <- extractParameter(bart_leaf, "sigma2_leaf")
  expect_length(sl, num_mcmc)
  expect_equal(sl, extractParameter(bart_leaf, "leaf_scale"))

  # y_hat_train returns matrix of shape (n_train, num_mcmc)
  bart_base <- bart(
    X_train = X_train,
    y_train = y_train,
    X_test = X_test,
    num_gfr = 0,
    num_burnin = 0,
    num_mcmc = num_mcmc,
    general_params = list(sample_sigma2_global = FALSE),
    mean_forest_params = list(sample_sigma2_leaf = FALSE)
  )
  yht <- extractParameter(bart_base, "y_hat_train")
  expect_equal(dim(yht), c(n_train, num_mcmc))

  # y_hat_test returns matrix of shape (n_test, num_mcmc)
  yhtest <- extractParameter(bart_base, "y_hat_test")
  expect_equal(dim(yhtest), c(n_test, num_mcmc))

  # sigma2_x_train / var_x_train return variance forest in-sample predictions
  bart_var <- bart(
    X_train = X_train,
    y_train = y_train,
    X_test = X_test,
    num_gfr = 0,
    num_burnin = 0,
    num_mcmc = num_mcmc,
    general_params = list(sample_sigma2_global = FALSE),
    mean_forest_params = list(sample_sigma2_leaf = FALSE),
    variance_forest_params = list(num_trees = 10)
  )
  s2x_train <- extractParameter(bart_var, "sigma2_x_train")
  expect_equal(dim(s2x_train), c(n_train, num_mcmc))
  expect_equal(s2x_train, extractParameter(bart_var, "var_x_train"))

  # sigma2_x_test / var_x_test return variance forest test-set predictions
  s2x_test <- extractParameter(bart_var, "sigma2_x_test")
  expect_equal(dim(s2x_test), c(n_test, num_mcmc))
  expect_equal(s2x_test, extractParameter(bart_var, "var_x_test"))

  # Error: sigma2 when not sampled
  bart_nosigma2 <- bart(
    X_train = X_train,
    y_train = y_train,
    num_gfr = 0,
    num_burnin = 0,
    num_mcmc = num_mcmc,
    general_params = list(sample_sigma2_global = FALSE),
    mean_forest_params = list(sample_sigma2_leaf = FALSE)
  )
  expect_error(
    extractParameter(bart_nosigma2, "sigma2"),
    "global variance"
  )

  # Error: sigma2_leaf when not sampled
  expect_error(
    extractParameter(bart_nosigma2, "sigma2_leaf"),
    "leaf variance"
  )

  # Error: y_hat_test without a test set
  expect_error(
    extractParameter(bart_nosigma2, "y_hat_test"),
    "test set"
  )

  # Error: sigma2_x_train without variance forest
  expect_error(
    extractParameter(bart_nosigma2, "sigma2_x_train"),
    "variance forest"
  )

  # Error: invalid term
  expect_error(
    extractParameter(bart_nosigma2, "not_a_real_term"),
    "not a valid BART model term"
  )
})

test_that("extractParameter.bcfmodel", {
  skip_on_cran()

  # Generate simulated data
  set.seed(42)
  n <- 100
  p <- 5
  X <- matrix(runif(n * p), ncol = p)
  pi_X <- 0.2 + 0.6 * X[, 1]
  Z <- rbinom(n, 1, pi_X)
  mu_X <- 5 * X[, 1]
  tau_X <- 2 * X[, 2]
  y <- mu_X + tau_X * Z + rnorm(n)
  test_inds <- sort(sample(1:n, 20))
  train_inds <- setdiff(1:n, test_inds)
  X_train <- X[train_inds, ]
  X_test <- X[test_inds, ]
  Z_train <- Z[train_inds]
  Z_test <- Z[test_inds]
  pi_train <- pi_X[train_inds]
  pi_test <- pi_X[test_inds]
  y_train <- y[train_inds]
  y_test <- y[test_inds]
  n_train <- length(train_inds)
  n_test <- length(test_inds)
  num_mcmc <- 10

  # sigma2 / global_error_scale / sigma2_global all return same samples
  bcf_sigma2 <- bcf(
    X_train = X_train,
    y_train = y_train,
    Z_train = Z_train,
    propensity_train = pi_train,
    X_test = X_test,
    Z_test = Z_test,
    propensity_test = pi_test,
    num_gfr = 0,
    num_burnin = 0,
    num_mcmc = num_mcmc,
    general_params = list(sample_sigma2_global = TRUE)
  )
  s2 <- extractParameter(bcf_sigma2, "sigma2")
  expect_length(s2, num_mcmc)
  expect_equal(s2, extractParameter(bcf_sigma2, "global_error_scale"))
  expect_equal(s2, extractParameter(bcf_sigma2, "sigma2_global"))

  # sigma2_leaf_mu / leaf_scale_mu / mu_leaf_scale for prognostic forest
  bcf_leaf_mu <- bcf(
    X_train = X_train,
    y_train = y_train,
    Z_train = Z_train,
    propensity_train = pi_train,
    X_test = X_test,
    Z_test = Z_test,
    propensity_test = pi_test,
    num_gfr = 0,
    num_burnin = 0,
    num_mcmc = num_mcmc,
    general_params = list(sample_sigma2_global = FALSE),
    prognostic_forest_params = list(sample_sigma2_leaf = TRUE),
    treatment_effect_forest_params = list(sample_sigma2_leaf = FALSE)
  )
  sl_mu <- extractParameter(bcf_leaf_mu, "sigma2_leaf_mu")
  expect_length(sl_mu, num_mcmc)
  expect_equal(sl_mu, extractParameter(bcf_leaf_mu, "leaf_scale_mu"))
  expect_equal(sl_mu, extractParameter(bcf_leaf_mu, "mu_leaf_scale"))

  # sigma2_leaf_tau / leaf_scale_tau / tau_leaf_scale for treatment effect forest
  bcf_leaf_tau <- bcf(
    X_train = X_train,
    y_train = y_train,
    Z_train = Z_train,
    propensity_train = pi_train,
    X_test = X_test,
    Z_test = Z_test,
    propensity_test = pi_test,
    num_gfr = 0,
    num_burnin = 0,
    num_mcmc = num_mcmc,
    general_params = list(sample_sigma2_global = FALSE),
    prognostic_forest_params = list(sample_sigma2_leaf = FALSE),
    treatment_effect_forest_params = list(sample_sigma2_leaf = TRUE)
  )
  sl_tau <- extractParameter(bcf_leaf_tau, "sigma2_leaf_tau")
  expect_length(sl_tau, num_mcmc)
  expect_equal(sl_tau, extractParameter(bcf_leaf_tau, "leaf_scale_tau"))
  expect_equal(sl_tau, extractParameter(bcf_leaf_tau, "tau_leaf_scale"))

  # adaptive_coding returns matrix with 2 rows (control, treated)
  bcf_ac <- bcf(
    X_train = X_train,
    y_train = y_train,
    Z_train = Z_train,
    propensity_train = pi_train,
    X_test = X_test,
    Z_test = Z_test,
    propensity_test = pi_test,
    num_gfr = 0,
    num_burnin = 0,
    num_mcmc = num_mcmc,
    general_params = list(sample_sigma2_global = FALSE, adaptive_coding = TRUE)
  )
  ac <- extractParameter(bcf_ac, "adaptive_coding")
  expect_equal(dim(ac), c(2L, num_mcmc))

  # y_hat_train and y_hat_test
  bcf_base <- bcf(
    X_train = X_train,
    y_train = y_train,
    Z_train = Z_train,
    propensity_train = pi_train,
    X_test = X_test,
    Z_test = Z_test,
    propensity_test = pi_test,
    num_gfr = 0,
    num_burnin = 0,
    num_mcmc = num_mcmc,
    general_params = list(sample_sigma2_global = FALSE)
  )
  yht <- extractParameter(bcf_base, "y_hat_train")
  expect_equal(dim(yht), c(n_train, num_mcmc))
  yhtest <- extractParameter(bcf_base, "y_hat_test")
  expect_equal(dim(yhtest), c(n_test, num_mcmc))

  # mu_hat_train / prognostic_function_train
  mht <- extractParameter(bcf_base, "mu_hat_train")
  expect_equal(dim(mht), c(n_train, num_mcmc))
  expect_equal(mht, extractParameter(bcf_base, "prognostic_function_train"))

  # mu_hat_test / prognostic_function_test
  mhtest <- extractParameter(bcf_base, "mu_hat_test")
  expect_equal(dim(mhtest), c(n_test, num_mcmc))
  expect_equal(mhtest, extractParameter(bcf_base, "prognostic_function_test"))

  # tau_hat_train / cate_train
  tht <- extractParameter(bcf_base, "tau_hat_train")
  expect_equal(dim(tht), c(n_train, num_mcmc))
  expect_equal(tht, extractParameter(bcf_base, "cate_train"))

  # tau_hat_test / cate_test
  thtest <- extractParameter(bcf_base, "tau_hat_test")
  expect_equal(dim(thtest), c(n_test, num_mcmc))
  expect_equal(thtest, extractParameter(bcf_base, "cate_test"))

  # sigma2_x_train / var_x_train and sigma2_x_test / var_x_test (variance forest)
  bcf_var <- bcf(
    X_train = X_train,
    y_train = y_train,
    Z_train = Z_train,
    propensity_train = pi_train,
    X_test = X_test,
    Z_test = Z_test,
    propensity_test = pi_test,
    num_gfr = 0,
    num_burnin = 0,
    num_mcmc = num_mcmc,
    general_params = list(sample_sigma2_global = FALSE),
    variance_forest_params = list(num_trees = 10)
  )
  s2x_train <- extractParameter(bcf_var, "sigma2_x_train")
  expect_equal(dim(s2x_train), c(n_train, num_mcmc))
  expect_equal(s2x_train, extractParameter(bcf_var, "var_x_train"))

  s2x_test <- extractParameter(bcf_var, "sigma2_x_test")
  expect_equal(dim(s2x_test), c(n_test, num_mcmc))
  expect_equal(s2x_test, extractParameter(bcf_var, "var_x_test"))

  # Error: sigma2 when not sampled
  bcf_noextras <- bcf(
    X_train = X_train,
    y_train = y_train,
    Z_train = Z_train,
    propensity_train = pi_train,
    num_gfr = 0,
    num_burnin = 0,
    num_mcmc = num_mcmc,
    general_params = list(sample_sigma2_global = FALSE),
    prognostic_forest_params = list(sample_sigma2_leaf = FALSE),
    treatment_effect_forest_params = list(sample_sigma2_leaf = FALSE)
  )
  expect_error(
    extractParameter(bcf_noextras, "sigma2"),
    "global variance"
  )

  # Error: sigma2_leaf_mu when not sampled
  expect_error(
    extractParameter(bcf_noextras, "sigma2_leaf_mu"),
    "prognostic forest leaf variance"
  )

  # Error: sigma2_leaf_tau when not sampled
  expect_error(
    extractParameter(bcf_noextras, "sigma2_leaf_tau"),
    "treatment effect forest leaf variance"
  )

  # Error: adaptive_coding when disabled
  bcf_noac <- bcf(
    X_train = X_train,
    y_train = y_train,
    Z_train = Z_train,
    propensity_train = pi_train,
    num_gfr = 0,
    num_burnin = 0,
    num_mcmc = num_mcmc,
    general_params = list(sample_sigma2_global = FALSE, adaptive_coding = FALSE)
  )
  expect_error(
    extractParameter(bcf_noac, "adaptive_coding"),
    "adaptive coding"
  )

  # Error: y_hat_test without a test set
  expect_error(
    extractParameter(bcf_noextras, "y_hat_test"),
    "test set"
  )

  # Error: tau_hat_test without a test set
  expect_error(
    extractParameter(bcf_noextras, "tau_hat_test"),
    "test set"
  )

  # Error: sigma2_x_train without variance forest
  expect_error(
    extractParameter(bcf_noextras, "sigma2_x_train"),
    "variance forest"
  )

  # Error: invalid term
  expect_error(
    extractParameter(bcf_noextras, "not_a_real_term"),
    "not a valid BCF model term"
  )
})

test_that("extractForest.bcfmodel", {
  skip_on_cran()

  set.seed(42)
  n <- 100
  p <- 5
  X <- matrix(runif(n * p), ncol = p)
  pi_X <- 0.2 + 0.6 * X[, 1]
  Z <- rbinom(n, 1, pi_X)
  y <- 5 * X[, 1] + 2 * X[, 2] * Z + rnorm(n)

  bcf_model <- bcf(
    X_train = X, y_train = y, Z_train = Z, propensity_train = pi_X,
    num_gfr = 0, num_burnin = 0, num_mcmc = 10
  )

  # Prognostic / treatment forests are extractable under all aliases; variance forest is absent.
  expect_s3_class(extractForest(bcf_model, "prognostic"), "ForestSamples")
  expect_s3_class(extractForest(bcf_model, "mu"), "ForestSamples")
  expect_s3_class(extractForest(bcf_model, "treatment"), "ForestSamples")
  expect_s3_class(extractForest(bcf_model, "tau"), "ForestSamples")
  expect_error(extractForest(bcf_model, "variance"), "does not have a variance forest")
  expect_error(extractForest(bcf_model, "not_a_forest"), "not a valid BCF forest term")
})

test_that("extractRandomEffectSamples matches legacy getRandomEffectSamples (BART + BCF)", {
  skip_on_cran()

  set.seed(42)
  n <- 200
  p <- 5
  X <- matrix(runif(n * p), ncol = p)
  g <- sample(1:3, n, replace = TRUE)
  rb <- matrix(rep(1, n), ncol = 1)

  # BART
  y_bart <- 2 * X[, 1] + (-2 * (g == 1) + 2 * (g == 2)) + rnorm(n)
  bart_model <- bart(
    X_train = X, y_train = y_bart, rfx_group_ids_train = g, rfx_basis_train = rb,
    num_gfr = 0, num_burnin = 0, num_mcmc = 10
  )
  e_bart <- extractRandomEffectSamples(bart_model)
  expect_setequal(names(e_bart), c("beta_samples", "xi_samples", "alpha_samples", "sigma_samples"))
  expect_identical(e_bart, getRandomEffectSamples(bart_model))

  # BCF
  pi_X <- 0.2 + 0.6 * X[, 1]
  Z <- rbinom(n, 1, pi_X)
  y_bcf <- 5 * X[, 1] + 2 * X[, 2] * Z + (-2 * (g == 1) + 2 * (g == 2)) + rnorm(n)
  bcf_model <- bcf(
    X_train = X, y_train = y_bcf, Z_train = Z, propensity_train = pi_X,
    rfx_group_ids_train = g, rfx_basis_train = rb,
    num_gfr = 0, num_burnin = 0, num_mcmc = 10
  )
  e_bcf <- extractRandomEffectSamples(bcf_model)
  expect_setequal(names(e_bcf), c("beta_samples", "xi_samples", "alpha_samples", "sigma_samples"))
  expect_identical(e_bcf, getRandomEffectSamples(bcf_model))

  # No-rfx model warns and returns an empty list
  bart_norfx <- bart(X_train = X, y_train = y_bart, num_gfr = 0, num_burnin = 0, num_mcmc = 5)
  expect_warning(res <- extractRandomEffectSamples(bart_norfx), "no RFX terms")
  expect_length(res, 0)
})
