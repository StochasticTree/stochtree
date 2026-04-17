## Benchmark: C++ sampler loop vs. R sampler loop -- heteroskedastic BART.
##
## Compares runtime, mean-forest RMSE (vs. true f(X)) and RMSE of the estimated
## conditional standard deviation (vs. the true s(X)) across run_cpp = TRUE /
## FALSE in bart() with both a mean forest and a variance forest
## (num_trees_variance > 0).
##
## DGP: f(X) is a step function of X[,1]; s(X) varies by quadrant of X[,1]
## and linearly with X[,3], matching the heteroskedastic Python benchmark.
##
## A variance-only model (num_trees_mean = 0, num_trees_variance > 0) is
## supported in the C++ path.  The mean-forest RMSE is reported as NA in
## that case since there is no mean forest to evaluate.
##
## Usage:
##   Rscript debug/benchmark_cpp_vs_r_sampler_heteroskedastic.R
##   or source() from an interactive session after devtools::load_all('.')

library(stochtree)

# ---------------------------------------------------------------------------
# Data-generating process
# ---------------------------------------------------------------------------
set.seed(1234)

n <- 2000
p <- 10
X <- matrix(runif(n * p), ncol = p)

# True conditional mean
f_X <- (((0.00 <= X[, 1]) & (X[, 1] < 0.25)) *
  (-3.0) +
  ((0.25 <= X[, 1]) & (X[, 1] < 0.50)) * (-1.0) +
  ((0.50 <= X[, 1]) & (X[, 1] < 0.75)) * (1.0) +
  ((0.75 <= X[, 1]) & (X[, 1] < 1.00)) * (3.0))

# True conditional standard deviation
s_X <- (((0.00 <= X[, 1]) & (X[, 1] < 0.25)) *
  (0.5 * X[, 3]) +
  ((0.25 <= X[, 1]) & (X[, 1] < 0.50)) * (1.0 * X[, 3]) +
  ((0.50 <= X[, 1]) & (X[, 1] < 0.75)) * (2.0 * X[, 3]) +
  ((0.75 <= X[, 1]) & (X[, 1] < 1.00)) * (3.0 * X[, 3]))

y <- f_X + rnorm(n, 0, 1) * s_X

test_frac <- 0.2
n_test <- round(test_frac * n)
n_train <- n - n_test
test_inds <- sort(sample(seq_len(n), n_test, replace = FALSE))
train_inds <- setdiff(seq_len(n), test_inds)

X_train <- X[train_inds, ]
X_test <- X[test_inds, ]
y_train <- y[train_inds]
y_test <- y[test_inds]
f_test <- f_X[test_inds]
s_test <- s_X[test_inds]

# ---------------------------------------------------------------------------
# Benchmark settings
# ---------------------------------------------------------------------------
num_gfr <- 10
num_burnin <- 0
num_mcmc <- 100
num_trees_mean <- 200
num_trees_variance <- 50
n_reps <- 3

cat(sprintf(
  "n_train=%d  n_test=%d  p=%d  num_trees_mean=%d  num_trees_variance=%d  num_gfr=%d  num_burnin=%d  num_mcmc=%d  reps=%d\n\n",
  n_train,
  n_test,
  p,
  num_trees_mean,
  num_trees_variance,
  num_gfr,
  num_burnin,
  num_mcmc,
  n_reps
))

# ---------------------------------------------------------------------------
# Helper: run one configuration and return timing + metrics
# ---------------------------------------------------------------------------
run_once <- function(run_cpp, seed) {
  t0 <- proc.time()
  m <- bart(
    X_train = X_train,
    y_train = y_train,
    X_test = X_test,
    num_gfr = num_gfr,
    num_burnin = num_burnin,
    num_mcmc = num_mcmc,
    general_params = list(random_seed = seed, sample_sigma2_global = FALSE),
    mean_forest_params = list(num_trees = num_trees_mean),
    variance_forest_params = list(num_trees = num_trees_variance),
    run_cpp = run_cpp
  )
  elapsed <- (proc.time() - t0)[["elapsed"]]

  # Mean-forest RMSE -- only defined when a mean forest was fitted
  if (num_trees_mean > 0) {
    f_hat <- rowMeans(m$y_hat_test)
    rmse_f <- sqrt(mean((f_hat - f_test)^2))
  } else {
    rmse_f <- NA_real_
  }

  # Variance-forest RMSE of estimated conditional std dev vs. true s(X)
  sigma2_x_hat_test <- extractParameter(m, "sigma2_x_test")
  s_hat <- rowMeans(sqrt(sigma2_x_hat_test))
  rmse_s <- sqrt(mean((s_hat - s_test)^2))

  list(elapsed = elapsed, rmse_f = rmse_f, rmse_s = rmse_s)
}

# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------
seeds <- 1000 + seq_len(n_reps)

results_cpp <- vector("list", n_reps)
results_r <- vector("list", n_reps)

cat("Running C++ sampler (run_cpp = TRUE)...\n")
for (i in seq_len(n_reps)) {
  cat(sprintf("  rep %d/%d\n", i, n_reps))
  results_cpp[[i]] <- run_once(run_cpp = TRUE, seed = seeds[i])
}

cat("\nRunning R sampler (run_cpp = FALSE)...\n")
for (i in seq_len(n_reps)) {
  cat(sprintf("  rep %d/%d\n", i, n_reps))
  results_r[[i]] <- run_once(run_cpp = FALSE, seed = seeds[i])
}

# ---------------------------------------------------------------------------
# Summarise
# ---------------------------------------------------------------------------
summarise <- function(results, label) {
  elapsed <- sapply(results, `[[`, "elapsed")
  rmse_f <- sapply(results, `[[`, "rmse_f")
  rmse_s <- sapply(results, `[[`, "rmse_s")
  data.frame(
    sampler = label,
    elapsed_mean = mean(elapsed),
    elapsed_sd = sd(elapsed),
    rmse_f_mean = mean(rmse_f, na.rm = TRUE),
    rmse_s_mean = mean(rmse_s),
    row.names = NULL
  )
}

res <- rbind(
  summarise(results_cpp, "cpp (run_cpp=TRUE)"),
  summarise(results_r, "R   (run_cpp=FALSE)")
)

cat("\n--- Results ---\n")
cat(sprintf(
  "%-22s  %10s  %10s  %12s  %12s\n",
  "Sampler",
  "Time (s)",
  "SD",
  "RMSE f(X)",
  "RMSE s(X)"
))
cat(strrep("-", 74), "\n")
for (i in seq_len(nrow(res))) {
  cat(sprintf(
    "%-22s  %10.3f  %10.3f  %12s  %12.4f\n",
    res$sampler[i],
    res$elapsed_mean[i],
    res$elapsed_sd[i],
    if (is.nan(res$rmse_f_mean[i])) {
      "nan"
    } else {
      sprintf("%.4f", res$rmse_f_mean[i])
    },
    res$rmse_s_mean[i]
  ))
}

speedup <- res$elapsed_mean[2] / res$elapsed_mean[1]
cat(sprintf("\nSpeedup (R / C++): %.2fx\n", speedup))
cat(sprintf(
  "RMSE s(X) delta (cpp - R): %.4f\n",
  res$rmse_s_mean[1] - res$rmse_s_mean[2]
))
