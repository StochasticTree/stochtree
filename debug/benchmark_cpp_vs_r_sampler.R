## Benchmark: C++ sampler loop vs. R sampler loop
## Compares runtime and test-set RMSE across run_cpp = TRUE / FALSE in bart().
##
## Usage: Rscript debug/benchmark_cpp_vs_r_sampler.R
##        or source() from an interactive session after devtools::load_all('.')
library(stochtree)

# ---------------------------------------------------------------------------
# Data-generating process
# ---------------------------------------------------------------------------
set.seed(1234)

n <- 2000
p <- 10
X <- matrix(runif(n * p), ncol = p)
f_X <- (((0.00 <= X[, 1]) & (X[, 1] < 0.25)) *
  (-7.5) +
  ((0.25 <= X[, 1]) & (X[, 1] < 0.50)) * (-2.5) +
  ((0.50 <= X[, 1]) & (X[, 1] < 0.75)) * (2.5) +
  ((0.75 <= X[, 1]) & (X[, 1] < 1.00)) * (7.5))
noise_sd <- 1
y <- f_X + rnorm(n, 0, noise_sd)

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

# ---------------------------------------------------------------------------
# Benchmark settings
# ---------------------------------------------------------------------------
num_gfr <- 10
num_mcmc <- 100
num_trees <- 200
n_reps <- 3 # repeated runs for stable timing

cat(sprintf(
  "n_train=%d  n_test=%d  p=%d  num_trees=%d  num_gfr=%d  num_mcmc=%d  reps=%d\n\n",
  n_train,
  n_test,
  p,
  num_trees,
  num_gfr,
  num_mcmc,
  n_reps
))

# ---------------------------------------------------------------------------
# Helper: run one configuration and return timing + RMSE
# ---------------------------------------------------------------------------
run_once <- function(run_cpp, seed = -1) {
  t0 <- proc.time()
  m <- bart(
    X_train = X_train,
    y_train = y_train,
    X_test = X_test,
    num_gfr = num_gfr,
    num_burnin = 0,
    num_mcmc = num_mcmc,
    mean_forest_params = list(num_trees = num_trees),
    general_params = list(random_seed = seed),
    run_cpp = run_cpp
  )
  elapsed <- (proc.time() - t0)[["elapsed"]]

  yhat <- rowMeans(m$y_hat_test)
  rmse <- sqrt(mean((yhat - y_test)^2))
  rmse_f <- sqrt(mean((yhat - f_test)^2))

  list(elapsed = elapsed, rmse = rmse, rmse_f = rmse_f)
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
  rmse <- sapply(results, `[[`, "rmse")
  rmse_f <- sapply(results, `[[`, "rmse_f")
  data.frame(
    sampler = label,
    elapsed_mean = mean(elapsed),
    elapsed_sd = sd(elapsed),
    rmse_mean = mean(rmse),
    rmse_sd = sd(rmse),
    rmse_f_mean = mean(rmse_f),
    rmse_f_sd = sd(rmse_f),
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
  "  SD",
  "RMSE (obs)",
  "RMSE (f)"
))
cat(strrep("-", 72), "\n")
for (i in seq_len(nrow(res))) {
  cat(sprintf(
    "%-22s  %10.3f  %10.3f  %12.4f  %12.4f\n",
    res$sampler[i],
    res$elapsed_mean[i],
    res$elapsed_sd[i],
    res$rmse_mean[i],
    res$rmse_f_mean[i]
  ))
}

speedup <- res$elapsed_mean[2] / res$elapsed_mean[1]
cat(sprintf("\nSpeedup (R / C++): %.2fx\n", speedup))
cat(sprintf(
  "RMSE delta (cpp - R):  obs=%.4f  f=%.4f\n",
  res$rmse_mean[1] - res$rmse_mean[2],
  res$rmse_f_mean[1] - res$rmse_f_mean[2]
))
