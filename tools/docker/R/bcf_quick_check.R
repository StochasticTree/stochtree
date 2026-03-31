# Minimal BCF divergence check
# Run on two platforms and compare the printed output

# Load libraries
library(stochtree)

# Generate synthetic data
set.seed(1234)
n <- 10000
p <- 5
X <- matrix(runif(n * p), ncol = p)
mu_x <- 1 + 2 * X[, 1] + X[, 2]
pi_x <- pnorm(0.5 * X[, 1] - 0.5 * X[, 3])
Z <- rbinom(n, 1, pi_x)
tau_x <- 1 + 0.5 * X[, 4]
y <- mu_x + tau_x * Z + rnorm(n, 0, 10)

# Run BCF
bcf_model <- stochtree::bcf(
  X_train = X,
  Z_train = Z,
  y_train = y,
  propensity_train = pi_x,
  num_gfr = 5,
  num_burnin = 0,
  num_mcmc = 100,
  general_params = list(random_seed = 1234)
)

# Extract parameters and estimates
sigma2 <- as.numeric(bcf_model$sigma2_global_samples)
y_hat <- as.numeric(predict(
  bcf_model,
  X = X,
  Z = Z,
  propensity = pi_x,
  type = "mean",
  terms = "y_hat"
))
tau_hat <- as.numeric(predict(
  bcf_model,
  X = X,
  Z = Z,
  propensity = pi_x,
  type = "mean",
  terms = "tau"
))

# Print first 10 samples of each
cat("sigma2[1:10]:", round(sigma2[1:10], 8), "\n")
cat("y_hat[1:10]: ", round(y_hat[1:10], 8), "\n")
cat("tau_hat[1:10]:", round(tau_hat[1:10], 8), "\n")
