test_that("extract_parameter.bartmodel", {
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
  s2 <- extract_parameter(bart_sigma2, "sigma2")
  expect_length(s2, num_mcmc)
  expect_equal(s2, extract_parameter(bart_sigma2, "global_error_scale"))
  expect_equal(s2, extract_parameter(bart_sigma2, "sigma2_global"))

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
  sl <- extract_parameter(bart_leaf, "sigma2_leaf")
  expect_length(sl, num_mcmc)
  expect_equal(sl, extract_parameter(bart_leaf, "leaf_scale"))

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
  yht <- extract_parameter(bart_base, "y_hat_train")
  expect_equal(dim(yht), c(n_train, num_mcmc))

  # y_hat_test returns matrix of shape (n_test, num_mcmc)
  yhtest <- extract_parameter(bart_base, "y_hat_test")
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
  s2x_train <- extract_parameter(bart_var, "sigma2_x_train")
  expect_equal(dim(s2x_train), c(n_train, num_mcmc))
  expect_equal(s2x_train, extract_parameter(bart_var, "var_x_train"))

  # sigma2_x_test / var_x_test return variance forest test-set predictions
  s2x_test <- extract_parameter(bart_var, "sigma2_x_test")
  expect_equal(dim(s2x_test), c(n_test, num_mcmc))
  expect_equal(s2x_test, extract_parameter(bart_var, "var_x_test"))

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
    extract_parameter(bart_nosigma2, "sigma2"),
    "global variance"
  )

  # Error: sigma2_leaf when not sampled
  expect_error(
    extract_parameter(bart_nosigma2, "sigma2_leaf"),
    "leaf variance"
  )

  # Error: y_hat_test without a test set
  expect_error(
    extract_parameter(bart_nosigma2, "y_hat_test"),
    "test set"
  )

  # Error: sigma2_x_train without variance forest
  expect_error(
    extract_parameter(bart_nosigma2, "sigma2_x_train"),
    "variance forest"
  )

  # Error: invalid term
  expect_error(
    extract_parameter(bart_nosigma2, "not_a_real_term"),
    "not a valid BART model term"
  )
})

test_that("extract_parameter.bcfmodel", {
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
  s2 <- extract_parameter(bcf_sigma2, "sigma2")
  expect_length(s2, num_mcmc)
  expect_equal(s2, extract_parameter(bcf_sigma2, "global_error_scale"))
  expect_equal(s2, extract_parameter(bcf_sigma2, "sigma2_global"))

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
  sl_mu <- extract_parameter(bcf_leaf_mu, "sigma2_leaf_mu")
  expect_length(sl_mu, num_mcmc)
  expect_equal(sl_mu, extract_parameter(bcf_leaf_mu, "leaf_scale_mu"))
  expect_equal(sl_mu, extract_parameter(bcf_leaf_mu, "mu_leaf_scale"))

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
  sl_tau <- extract_parameter(bcf_leaf_tau, "sigma2_leaf_tau")
  expect_length(sl_tau, num_mcmc)
  expect_equal(sl_tau, extract_parameter(bcf_leaf_tau, "leaf_scale_tau"))
  expect_equal(sl_tau, extract_parameter(bcf_leaf_tau, "tau_leaf_scale"))

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
  ac <- extract_parameter(bcf_ac, "adaptive_coding")
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
  yht <- extract_parameter(bcf_base, "y_hat_train")
  expect_equal(dim(yht), c(n_train, num_mcmc))
  yhtest <- extract_parameter(bcf_base, "y_hat_test")
  expect_equal(dim(yhtest), c(n_test, num_mcmc))

  # tau_hat_train and tau_hat_test
  tht <- extract_parameter(bcf_base, "tau_hat_train")
  expect_equal(dim(tht), c(n_train, num_mcmc))
  thtest <- extract_parameter(bcf_base, "tau_hat_test")
  expect_equal(dim(thtest), c(n_test, num_mcmc))

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
  s2x_train <- extract_parameter(bcf_var, "sigma2_x_train")
  expect_equal(dim(s2x_train), c(n_train, num_mcmc))
  expect_equal(s2x_train, extract_parameter(bcf_var, "var_x_train"))

  s2x_test <- extract_parameter(bcf_var, "sigma2_x_test")
  expect_equal(dim(s2x_test), c(n_test, num_mcmc))
  expect_equal(s2x_test, extract_parameter(bcf_var, "var_x_test"))

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
    extract_parameter(bcf_noextras, "sigma2"),
    "global variance"
  )

  # Error: sigma2_leaf_mu when not sampled
  expect_error(
    extract_parameter(bcf_noextras, "sigma2_leaf_mu"),
    "prognostic forest leaf variance"
  )

  # Error: sigma2_leaf_tau when not sampled
  expect_error(
    extract_parameter(bcf_noextras, "sigma2_leaf_tau"),
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
    extract_parameter(bcf_noac, "adaptive_coding"),
    "adaptive coding"
  )

  # Error: y_hat_test without a test set
  expect_error(
    extract_parameter(bcf_noextras, "y_hat_test"),
    "test set"
  )

  # Error: tau_hat_test without a test set
  expect_error(
    extract_parameter(bcf_noextras, "tau_hat_test"),
    "test set"
  )

  # Error: sigma2_x_train without variance forest
  expect_error(
    extract_parameter(bcf_noextras, "sigma2_x_train"),
    "variance forest"
  )

  # Error: invalid term
  expect_error(
    extract_parameter(bcf_noextras, "not_a_real_term"),
    "not a valid BCF model term"
  )
})
