# Minimal BART divergence check
# Run on two platforms and compare the printed output

# Load libraries
library(stochtree)

# Generate synthetic data
set.seed(1234)
n <- 10000
p <- 5
X <- matrix(runif(n * p), ncol = p)
f_x <- 1 + 2 * X[, 1] + X[, 2]
y <- f_x + rnorm(n, 0, 10)

# Run BART
bart_model <- stochtree::bart(
  X_train = X,
  y_train = y,
  num_gfr = 5,
  num_burnin = 0,
  num_mcmc = 100,
  general_params = list(random_seed = 1234)
)

# Extract parameters and estimates
sigma2 <- as.numeric(bart_model$sigma2_global_samples)
y_hat <- as.numeric(predict(
  bart_model,
  X = X,
  type = "mean",
  terms = "y_hat"
))

# Print first 10 samples of each
cat("sigma2[1:10]:", round(sigma2[1:10], 8), "\n")
cat("y_hat[1:10]: ", round(y_hat[1:10], 8), "\n")
