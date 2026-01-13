# Load library
library(stochtree)

# Generate the data
n <- 500
p_X <- 10
p_W <- 1
X <- matrix(runif(n * p_X), ncol = p_X)
W <- matrix(runif(n * p_W), ncol = p_W)
beta_W <- c(5)
f_XW <- (((0 <= X[, 1]) & (0.25 > X[, 1])) *
  (-3) +
  ((0.25 <= X[, 1]) & (0.5 > X[, 1])) * (-1) +
  ((0.5 <= X[, 1]) & (0.75 > X[, 1])) * (1) +
  ((0.75 <= X[, 1]) & (1 > X[, 1])) * (3))
lm_term <- W %*% beta_W
y <- lm_term + f_XW + rnorm(n, 0, 1)

# Standardize outcome
y_bar <- mean(y)
y_std <- sd(y)
resid <- (y - y_bar) / y_std

# Set sampler parameters
alpha_bart <- 0.9
beta_bart <- 1.25
min_samples_leaf <- 1
max_depth <- 10
num_trees <- 100
cutpoint_grid_size = 100
global_variance_init = 1.
tau_init = 0.5
leaf_prior_scale = matrix(c(tau_init), ncol = 1)
nu <- 4
lambda <- 0.5
a_leaf <- 2.
b_leaf <- 0.5
leaf_regression <- F
feature_types <- as.integer(rep(0, p_X)) # 0 = numeric
var_weights <- rep(1 / p_X, p_X)
beta_tau <- 20

# Initialize C++ objects
# Data
if (leaf_regression) {
  forest_dataset <- createForestDataset(X, W)
  outcome_model_type <- 1
} else {
  forest_dataset <- createForestDataset(X)
  outcome_model_type <- 0
}
outcome <- createOutcome(resid)

# Random number generator (boost::random::mt19937)
rng <- createRNG()

# Sampling data structures
forest_model <- createForestModel(
  forest_dataset,
  feature_types,
  num_trees,
  n,
  alpha_bart,
  beta_bart,
  min_samples_leaf,
  max_depth
)

# Container of forest samples
if (leaf_regression) {
  forest_samples <- createForestContainer(num_trees, 1, F)
} else {
  forest_samples <- createForestContainer(num_trees, 1, T)
}

# Sampler preparation
num_warmstart <- 20
num_mcmc <- 100
num_samples <- num_warmstart + num_mcmc
beta_init <- 0
global_var_samples <- c(global_variance_init, rep(0, num_samples))
leaf_scale_samples <- c(tau_init, rep(0, num_samples))
beta_hat_samples <- c(beta_init, rep(0, num_samples))

# GFR loop
for (i in 1:num_warmstart) {
  # Initialize vectors needed for posterior sampling
  if (i == 1) {
    beta_hat <- beta_init
    yhat_forest <- rep(0, n)
    partial_res <- resid - yhat_forest
  } else {
    yhat_forest <- forest_samples$predict_raw_single_forest(
      forest_dataset,
      (i - 1) - 1
    )
    partial_res <- resid - yhat_forest
    outcome$add_vector(W %*% beta_hat)
  }
  # Sample beta from bayesian linear model with gaussian prior
  sigma2 <- global_var_samples[i]
  beta_posterior_mean <- sum(partial_res * W[, 1]) /
    (sigma2 + sum(W[, 1] * W[, 1]))
  beta_posterior_var <- (sigma2 * beta_tau) / (sigma2 + sum(W[, 1] * W[, 1]))
  beta_hat <- rnorm(1, beta_posterior_mean, sqrt(beta_posterior_var))
  beta_hat_samples[i + 1] <- beta_hat
  # Update partial residual before sampling forest
  outcome$subtract_vector(W %*% beta_hat)

  # Sample forest
  forest_model$sample_one_iteration(
    forest_dataset,
    outcome,
    forest_samples,
    rng,
    feature_types,
    outcome_model_type,
    leaf_prior_scale,
    var_weights,
    sigma2,
    cutpoint_grid_size,
    gfr = T
  )

  # Sample global variance parameter
  global_var_samples[i + 1] <- sample_sigma2_one_iteration(
    outcome,
    rng,
    nu,
    lambda
  )
}

# MCMC Loop
for (i in (num_warmstart + 1):num_samples) {
  # Initialize vectors needed for posterior sampling
  if (i == 1) {
    beta_hat <- beta_init
    yhat_forest <- rep(0, n)
    partial_res <- resid - yhat_forest
  } else {
    yhat_forest <- forest_samples$predict_raw_single_forest(
      forest_dataset,
      (i - 1) - 1
    )
    partial_res <- resid - yhat_forest
    outcome$add_vector(W %*% beta_hat)
  }
  # Sample beta from bayesian linear model with gaussian prior
  sigma2 <- global_var_samples[i]
  beta_posterior_mean <- sum(partial_res * W[, 1]) /
    (sigma2 + sum(W[, 1] * W[, 1]))
  beta_posterior_var <- (sigma2 * beta_tau) / (sigma2 + sum(W[, 1] * W[, 1]))
  beta_hat <- rnorm(1, beta_posterior_mean, sqrt(beta_posterior_var))
  beta_hat_samples[i + 1] <- beta_hat
  # Update partial residual before sampling forest
  outcome$subtract_vector(W %*% beta_hat)

  # Sample forest
  forest_model$sample_one_iteration(
    forest_dataset,
    outcome,
    forest_samples,
    rng,
    feature_types,
    outcome_model_type,
    leaf_prior_scale,
    var_weights,
    global_var_samples[i],
    cutpoint_grid_size,
    gfr = F
  )

  # Sample global variance parameter
  global_var_samples[i + 1] <- sample_sigma2_one_iteration(
    outcome,
    rng,
    nu,
    lambda
  )
}

# Extract samples
# Linear model predictions
lm_preds <- (sapply(1:num_samples, function(x) {
  W[, 1] * beta_hat_samples[x + 1]
})) *
  y_std

# Forest predictions
forest_preds <- forest_samples$predict(forest_dataset) * y_std + y_bar

# Overall predictions
preds <- forest_preds + lm_preds

# Global error variance
sigma_samples <- sqrt(global_var_samples) * y_std

# Inspect results
# GFR
plot(sigma_samples[1:num_warmstart], ylab = "sigma")
plot(beta_hat_samples[1:num_warmstart] * y_std, ylab = "beta")
plot(
  rowMeans(preds[, 1:num_warmstart]),
  y,
  pch = 16,
  cex = 0.75,
  xlab = "pred",
  ylab = "actual"
)
abline(0, 1, col = "red", lty = 2, lwd = 2.5)

# MCMC
plot(sigma_samples[(num_warmstart + 1):num_samples], ylab = "sigma")
plot(beta_hat_samples[(num_warmstart + 1):num_samples] * y_std, ylab = "beta")
plot(
  rowMeans(preds[, (num_warmstart + 1):num_samples]),
  y,
  pch = 16,
  cex = 0.75,
  xlab = "pred",
  ylab = "actual"
)
abline(0, 1, col = "red", lty = 2, lwd = 2.5)
