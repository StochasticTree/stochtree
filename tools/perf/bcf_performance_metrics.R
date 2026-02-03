################################################################################
## Simulation study of BCF / warmstart BCF performance
##   (a) CATE test set RMSE
##   (b) CATE interval coverage
##   (c) CATE interval length
##
## The estimators we compare are
##   (1) Warm-start BCF (stochtree)
##   (2) BCF (stochtree)
################################################################################

# Load libraries
library(foreach)
library(doParallel)

# Helper functions
g <- function(x) {
  ifelse(x[, 5] == 1, 2, ifelse(x[, 5] == 2, -1, -4))
}
mu1 <- function(x) {
  1 + g(x) + x[, 1] * x[, 3]
}
mu2 <- function(x) {
  1 + g(x) + 6 * abs(x[, 3] - 1)
}
tau1 <- function(x) {
  rep(3, nrow(x))
}
tau2 <- function(x) {
  1 + 2 * x[, 2] * x[, 4]
}

# Define a function that runs one iteration of the simulation
simulation_function = function(
  n,
  mu = mu1,
  tau = tau2,
  gfr_iter = 10,
  burnin_iter = 1000,
  mcmc_iter = 1000,
  alpha = 0.05,
  snr = 3,
  sample_tau = F
) {
  # Generate data
  x1 <- rnorm(n)
  x2 <- rnorm(n)
  x3 <- rnorm(n)
  x4 <- as.numeric(rbinom(n, 1, 0.5))
  x5 <- as.numeric(sample(1:3, n, replace = TRUE))
  X <- cbind(x1, x2, x3, x4, x5)
  p <- ncol(X)
  mu_x <- mu(X)
  tau_x <- tau(X)
  pi_x <- 0.8 *
    pnorm((3 * mu_x / sd(mu_x)) - 0.5 * X[, 1]) +
    0.05 +
    runif(n) / 10
  Z <- rbinom(n, 1, pi_x)
  E_XZ <- mu_x + Z * tau_x
  y <- E_XZ + rnorm(n, 0, 1) * (sd(E_XZ) / snr)
  X <- as.data.frame(X)
  X$x4 <- factor(X$x4, ordered = TRUE)
  X$x5 <- factor(X$x5, ordered = TRUE)

  # Split data into test and train sets
  test_set_pct <- 0.5
  n_test <- round(test_set_pct * n)
  n_train <- n - n_test
  test_inds <- sort(sample(1:n, n_test, replace = FALSE))
  train_inds <- (1:n)[!((1:n) %in% test_inds)]
  X_test <- X[test_inds, ]
  X_train <- X[train_inds, ]
  pi_test <- pi_x[test_inds]
  pi_train <- pi_x[train_inds]
  Z_test <- Z[test_inds]
  Z_train <- Z[train_inds]
  y_test <- y[test_inds]
  y_train <- y[train_inds]
  mu_test <- mu_x[test_inds]
  mu_train <- mu_x[train_inds]
  tau_test <- tau_x[test_inds]
  tau_train <- tau_x[train_inds]

  # Warmstart BCF
  num_gfr <- gfr_iter
  num_burnin <- 0
  num_mcmc <- mcmc_iter
  num_samples <- num_gfr + num_burnin + num_mcmc
  bcf_model_warmstart <- bcf(
    X_train = X_train,
    Z_train = Z_train,
    y_train = y_train,
    pi_train = pi_train,
    X_test = X_test,
    Z_test = Z_test,
    pi_test = pi_test,
    num_gfr = num_gfr,
    num_burnin = num_burnin,
    num_mcmc = num_mcmc,
    sample_sigma_leaf_mu = sample_tau,
    sample_sigma_leaf_tau = F,
    num_trees_mu = 200,
    num_trees_tau = 50
  )
  sample_inds <- (num_gfr + num_burnin + 1):num_samples
  bcf_warmstart_rmse <- sqrt(mean(
    (rowMeans(bcf_model_warmstart$tau_hat_test[, sample_inds]) - tau_test)^2
  ))
  test_lb <- apply(
    bcf_model_warmstart$tau_hat_test[, sample_inds],
    1,
    quantile,
    alpha / 2
  )
  test_ub <- apply(
    bcf_model_warmstart$tau_hat_test[, sample_inds],
    1,
    quantile,
    1 - (alpha / 2)
  )
  cover <- ((test_lb <= tau_test) & (test_ub >= tau_test))
  bcf_warmstart_cover <- mean(cover)
  bcf_warmstart_interval_len <- mean(test_ub - test_lb)

  # Regular BCF
  num_gfr <- 0
  num_burnin <- burnin_iter
  num_mcmc <- mcmc_iter
  num_samples <- num_gfr + num_burnin + num_mcmc
  bcf_model_root <- bcf(
    X_train = X_train,
    Z_train = Z_train,
    y_train = y_train,
    pi_train = pi_train,
    X_test = X_test,
    Z_test = Z_test,
    pi_test = pi_test,
    num_gfr = num_gfr,
    num_burnin = num_burnin,
    num_mcmc = num_mcmc,
    sample_sigma_leaf_mu = sample_tau,
    sample_sigma_leaf_tau = F,
    num_trees_mu = 200,
    num_trees_tau = 50
  )
  sample_inds <- (num_gfr + num_burnin + 1):num_samples
  bcf_root_rmse <- sqrt(mean(
    (rowMeans(bcf_model_root$tau_hat_test[, sample_inds]) - tau_test)^2
  ))
  test_lb <- apply(
    bcf_model_root$tau_hat_test[, sample_inds],
    1,
    quantile,
    alpha / 2
  )
  test_ub <- apply(
    bcf_model_root$tau_hat_test[, sample_inds],
    1,
    quantile,
    1 - (alpha / 2)
  )
  cover <- ((test_lb <= tau_test) & (test_ub >= tau_test))
  bcf_root_cover <- mean(cover)
  bcf_root_interval_len <- mean(test_ub - test_lb)

  # Aggregating all results
  rmse_results <- c(bcf_warmstart_rmse, bcf_root_rmse)
  cover_results <- c(bcf_warmstart_cover, bcf_root_cover)
  interval_len_results <- c(bcf_warmstart_interval_len, bcf_root_interval_len)

  # Output results as a row vector per simulation
  return(c(rmse_results, cover_results, interval_len_results))
}

# Simulation config
n <- 500
n_sim <- 20

# Setup a parallel cluster
cl <- makeCluster(10)
registerDoParallel(cl)

# Run the simulation for dgp 1
result_dgp1 <- foreach(
  i = 1:n_sim,
  .combine = rbind,
  .packages = c("stochtree")
) %dopar%
  {
    set.seed(i)
    simulation_function(
      n,
      mu = mu1,
      tau = tau2,
      gfr_iter = 20,
      burnin_iter = 1000,
      mcmc_iter = 1000,
      alpha = 0.05,
      snr = 2,
      sample_tau = F
    )
  }

# Run the simulation for dgp 2
result_dgp2 <- foreach(
  i = 1:n_sim,
  .combine = rbind,
  .packages = c("stochtree")
) %dopar%
  {
    set.seed(i)
    simulation_function(
      n,
      mu = mu2,
      tau = tau2,
      gfr_iter = 20,
      burnin_iter = 1000,
      mcmc_iter = 1000,
      alpha = 0.05,
      snr = 2,
      sample_tau = F
    )
  }

# Close the cluster
stopCluster(cl)

# Warmstart and root-initialized BCF columns
metric_cols_ws <- c(1, 3, 5)
metric_cols_bcf <- c(2, 4, 6)

# DGP 1 metrics
colMeans(result_dgp1[, metric_cols_ws])
colMeans(result_dgp1[, metric_cols_bcf])

# DGP 2 metrics
colMeans(result_dgp2[, metric_cols_ws])
colMeans(result_dgp2[, metric_cols_bcf])
