# Load libraries
library(stochtree)

# Set seed
random_seed <- 1234
set.seed(random_seed)

# Prepare simulation study
n_sim <- 50
ate_squared_errors_classic_homogeneous <- rep(NA_real_, n_sim)
ate_coverage_classic_homogeneous <- rep(NA_integer_, n_sim)
ate_squared_errors_parametric_homogeneous <- rep(NA_real_, n_sim)
ate_coverage_parametric_homogeneous <- rep(NA_integer_, n_sim)
ate_squared_errors_classic_heterogeneous <- rep(NA_real_, n_sim)
ate_coverage_classic_heterogeneous <- rep(NA_integer_, n_sim)
ate_squared_errors_parametric_heterogeneous <- rep(NA_real_, n_sim)
ate_coverage_parametric_heterogeneous <- rep(NA_integer_, n_sim)

# Below we run two different simulation studies, in which
# we compare "traditional" BCF with a treatment effect forest
# to a modified version that includes a parametric treatment effect term
# and a forest-based offset in the case where the true treatment effect
# is homogeneous and where the true treatment effect is heterogeneous
num_trees_tau_classic <- 100
num_trees_tau_parametric <- 100
leaf_scale_tau_classic <- 1 / num_trees_tau_classic
leaf_scale_tau_parametric <- 0.25 / num_trees_tau_parametric
leaf_scale_tau_0_parametric <- 1
num_gfr <- 3
num_burnin <- 200
num_mcmc <- 500

for (i in 1:n_sim) {
  # Shared aspects of both DGPs
  n <- 500
  p <- 5
  X <- matrix(runif(n * p), n, p)
  pi_X <- X[, 1] * 0.6 + 0.2
  mu_X <- (pi_X - 0.5) * 5
  Z <- rbinom(n, 1, pi_X)

  # Generate data with no treatment effect heterogeneity
  tau_X <- 0.5
  y <- mu_X + tau_X * Z + rnorm(n)
  ATE_true <- tau_X

  # Run traditional BCF
  bcf_model_classic <- bcf(
    X_train = X,
    Z_train = Z,
    y_train = y,
    propensity_train = pi_X,
    num_gfr = num_gfr,
    num_burnin = num_burnin,
    num_mcmc = num_mcmc,
    general_params = list(
      adaptive_coding = FALSE
    ),
    treatment_effect_forest_params = list(
      num_trees = num_trees_tau_classic,
      sample_intercept = FALSE,
      sigma2_leaf_init = leaf_scale_tau_classic
    )
  )
  CATE_posterior_classic <- predict(
    bcf_model_classic,
    X = X,
    Z = Z,
    propensity = pi_X,
    type = "posterior",
    terms = "cate"
  )
  ATE_posterior_classic <- colMeans(CATE_posterior_classic)
  ate_squared_errors_classic_homogeneous[i] <- (mean(ATE_posterior_classic) -
    ATE_true)^2
  ate_coverage_classic_homogeneous[i] <- (quantile(
    ATE_posterior_classic,
    0.025
  ) <=
    ATE_true &
    ATE_true <= quantile(ATE_posterior_classic, 0.975))

  # Run BCF with parametric term
  bcf_model_parametric <- bcf(
    X_train = X,
    Z_train = Z,
    y_train = y,
    propensity_train = pi_X,
    num_gfr = num_gfr,
    num_burnin = num_burnin,
    num_mcmc = num_mcmc,
    general_params = list(
      adaptive_coding = FALSE
    ),
    treatment_effect_forest_params = list(
      num_trees = num_trees_tau_parametric,
      sample_intercept = TRUE,
      tau_0_prior_var = leaf_scale_tau_0_parametric,
      sigma2_leaf_init = leaf_scale_tau_parametric
    )
  )
  CATE_posterior_parametric <- predict(
    bcf_model_parametric,
    X = X,
    Z = Z,
    propensity = pi_X,
    type = "posterior",
    terms = "cate"
  )
  ATE_posterior_parametric <- colMeans(CATE_posterior_parametric)
  ate_squared_errors_parametric_homogeneous[i] <- (mean(
    ATE_posterior_parametric
  ) -
    ATE_true)^2
  ate_coverage_parametric_homogeneous[i] <- (quantile(
    ATE_posterior_parametric,
    0.025
  ) <=
    ATE_true &
    ATE_true <= quantile(ATE_posterior_parametric, 0.975))

  # Generate data with significant treatment effect heterogeneity
  tau_X <- 2 * X[, 2] - 1
  y <- mu_X + tau_X * Z + rnorm(n)
  ATE_true <- mean(tau_X)

  # Run traditional BCF
  bcf_model_classic <- bcf(
    X_train = X,
    Z_train = Z,
    y_train = y,
    propensity_train = pi_X,
    num_gfr = num_gfr,
    num_burnin = num_burnin,
    num_mcmc = num_mcmc,
    general_params = list(
      adaptive_coding = FALSE
    ),
    treatment_effect_forest_params = list(
      num_trees = num_trees_tau_classic,
      sample_intercept = FALSE,
      sigma2_leaf_init = leaf_scale_tau_classic
    )
  )
  CATE_posterior_classic <- predict(
    bcf_model_classic,
    X = X,
    Z = Z,
    propensity = pi_X,
    type = "posterior",
    terms = "cate"
  )
  ATE_posterior_classic <- colMeans(CATE_posterior_classic)
  ate_squared_errors_classic_heterogeneous[i] <- (mean(ATE_posterior_classic) -
    ATE_true)^2
  ate_coverage_classic_heterogeneous[i] <- (quantile(
    ATE_posterior_classic,
    0.025
  ) <=
    ATE_true &
    ATE_true <= quantile(ATE_posterior_classic, 0.975))

  # Run BCF with parametric term
  bcf_model_parametric <- bcf(
    X_train = X,
    Z_train = Z,
    y_train = y,
    propensity_train = pi_X,
    num_gfr = num_gfr,
    num_burnin = num_burnin,
    num_mcmc = num_mcmc,
    general_params = list(
      adaptive_coding = FALSE
    ),
    treatment_effect_forest_params = list(
      num_trees = num_trees_tau_parametric,
      sample_intercept = TRUE,
      tau_0_prior_var = leaf_scale_tau_0_parametric,
      sigma2_leaf_init = leaf_scale_tau_parametric
    )
  )
  CATE_posterior_parametric <- predict(
    bcf_model_parametric,
    X = X,
    Z = Z,
    propensity = pi_X,
    type = "posterior",
    terms = "cate"
  )
  ATE_posterior_parametric <- colMeans(CATE_posterior_parametric)
  ate_squared_errors_parametric_heterogeneous[i] <- (mean(
    ATE_posterior_parametric
  ) -
    ATE_true)^2
  ate_coverage_parametric_heterogeneous[i] <- (quantile(
    ATE_posterior_parametric,
    0.025
  ) <=
    ATE_true &
    ATE_true <= quantile(ATE_posterior_parametric, 0.975))
}

mean(ate_squared_errors_classic_homogeneous)
mean(ate_squared_errors_parametric_homogeneous)
mean(ate_coverage_classic_homogeneous)
mean(ate_coverage_parametric_homogeneous)
mean(ate_squared_errors_classic_heterogeneous)
mean(ate_squared_errors_parametric_heterogeneous)
mean(ate_coverage_classic_heterogeneous)
mean(ate_coverage_parametric_heterogeneous)
