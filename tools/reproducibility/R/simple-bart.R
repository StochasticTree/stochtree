# Reproducibility check script:
# Run BART on a simple dataset and compare results across platforms

# Load libraries
library(stochtree)

# Set seed for reproducibility
random_seed <- 1234
set.seed(random_seed)

# Generate data
n <- 200
p <- 10
X <- matrix(runif(n * p), ncol = p)
f_X <- sin(2 * pi * X[, 1]) + cos(2 * pi * X[, 2])
eps <- rnorm(n, 0, 1)
y <- f_X + eps

# Fit BART with default settings
bart_model <- stochtree::bart(
  X_train = X,
  y_train = y,
  general_params = list(random_seed = random_seed)
)

# Obtain traceplot of global error scale
global_error_scale_trace <- bart_model$sigma2_global_samples

# Obtain in-sample posterior mean estimate of f(X)
y_hat_in_sample <- predict(bart_model, X = X, type = "mean", terms = "y_hat")

# Compare to stored results
global_error_scale_trace_comparison <- as.numeric(read.csv(
  "tools/reproducibility/R/bart_global_error_scale_trace.csv"
)[, 1])
y_hat_in_sample_comparison <- as.numeric(read.csv(
  "tools/reproducibility/R/bart_y_hat_in_sample.csv"
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
