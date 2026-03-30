# Reproducibility check script:
# Run BCF on a simple dataset and compare results across platforms

# Load libraries
library(stochtree)

# Set seed for reproducibility
random_seed <- 1234
set.seed(random_seed)

# Generate data
n <- 200
p <- 10
X <- matrix(runif(n * p), ncol = p)
mu_x <- sin(2 * pi * X[, 1]) + cos(2 * pi * X[, 2])
pi_x <- 0.8 * pnorm(3 * mu_x / sd(mu_x) - 0.5 * X[, 3]) + 0.05 + runif(n) / 10
Z <- rbinom(n, 1, pi_x)
tau_x <- 1 + 2 * X[, 4] * X[, 5]
f_XZ <- mu_x + tau_x * Z
eps <- rnorm(n, 0, 1)
y <- f_XZ + eps

# Fit BCF with default settings
bcf_model <- stochtree::bcf(
  X_train = X,
  Z_train = Z,
  y_train = y,
  propensity_train = pi_x,
  general_params = list(random_seed = random_seed)
)

# Obtain traceplot of global error scale
global_error_scale_trace <- bcf_model$sigma2_global_samples

# Obtain in-sample posterior mean estimate of E[y | X, Z]
y_hat_in_sample <- predict(
  bcf_model,
  X = X,
  Z = Z,
  propensity = pi_x,
  type = "mean",
  terms = "y_hat"
)

# Compare to stored results
global_error_scale_trace_comparison <- as.numeric(read.csv(
  "tools/reproducibility/R/bcf_global_error_scale_trace.csv"
)[, 1])
y_hat_in_sample_comparison <- as.numeric(read.csv(
  "tools/reproducibility/R/bcf_y_hat_in_sample.csv"
)[, 1])
TOL <- 0.000001
y_hat_mismatch <- !all(abs(y_hat_in_sample_comparison - y_hat_in_sample) < TOL)
y_hat_mismatch_loc <- abs(y_hat_in_sample_comparison - y_hat_in_sample) >= TOL
global_error_scale_mismatch <- !all(
  abs(global_error_scale_trace_comparison - global_error_scale_trace) < TOL
)
global_error_scale_mismatch_loc <- abs(
  global_error_scale_trace_comparison - global_error_scale_trace
) >=
  TOL
if (y_hat_mismatch) {
  cat(
    "Differences in posterior mean: \n",
    paste0(
      (1:length(y_hat_in_sample))[y_hat_mismatch_loc],
      ": ",
      y_hat_in_sample_comparison[y_hat_mismatch_loc],
      " vs ",
      y_hat_in_sample[y_hat_mismatch_loc],
      collapse = "\n"
    )
  )
} else {
  cat("No mismatches found in the posterior mean\n")
}
if (global_error_scale_mismatch) {
  cat(
    "Differences in global error scale trace: \n",
    paste0(
      (1:length(global_error_scale_trace))[global_error_scale_mismatch_loc],
      ": ",
      global_error_scale_trace_comparison[global_error_scale_mismatch_loc],
      " vs ",
      global_error_scale_trace[global_error_scale_mismatch_loc],
      collapse = "\n"
    )
  )
} else {
  cat("No mismatches found in the global error scale trace")
}
