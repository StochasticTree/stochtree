## Benchmark: C++ sampler loop vs. R sampler loop -- univariate leaf regression.
##
## Compares runtime and test-set RMSE across run_cpp = TRUE / FALSE in bart()
## with a univariate leaf regression basis (leaf_basis_train with one column).
##
## DGP: f(X, Z) = tau(X) * Z, where tau(X) is a step function of X[,1] and
## Z is drawn uniform [0, 1].  A constant noise term is added.  The leaf basis
## passed to the sampler is just Z (shape n x 1).
##
## Usage:
##   Rscript debug/benchmark_cpp_vs_r_sampler_leaf_regression.R
##   or source() from an interactive session after devtools::load_all('.')

library(stochtree)

# ---------------------------------------------------------------------------
# Data-generating process
# ---------------------------------------------------------------------------
set.seed(1234)

n <- 2000
p <- 10
X <- matrix(runif(n * p), ncol = p)
Z <- runif(n) # scalar moderating variable / leaf basis

# Heterogeneous slope on Z, partitioned by X[,1]
tau_X <- (((0.00 <= X[, 1]) & (X[, 1] < 0.25)) *
  (-2.0) +
  ((0.25 <= X[, 1]) & (X[, 1] < 0.50)) * (-1.0) +
  ((0.50 <= X[, 1]) & (X[, 1] < 0.75)) * (1.0) +
  ((0.75 <= X[, 1]) & (X[, 1] < 1.00)) * (2.0))
f_XZ <- tau_X * Z
noise_sd <- 1.0
y <- f_XZ + rnorm(n, 0, noise_sd)

test_frac <- 0.2
n_test <- round(test_frac * n)
n_train <- n - n_test
test_inds <- sort(sample(seq_len(n), n_test, replace = FALSE))
train_inds <- setdiff(seq_len(n), test_inds)

X_train <- X[train_inds, ]
X_test <- X[test_inds, ]
Z_train <- Z[train_inds]
Z_test <- Z[test_inds]
y_train <- y[train_inds]
y_test <- y[test_inds]
f_test <- f_XZ[test_inds]

# Leaf basis matrices (n x 1)
basis_train <- matrix(Z_train, ncol = 1)
basis_test <- matrix(Z_test, ncol = 1)

# ---------------------------------------------------------------------------
# Benchmark settings
# ---------------------------------------------------------------------------
num_gfr <- 10
num_burnin <- 0
num_mcmc <- 100
num_trees <- 200
n_reps <- 3

cat(sprintf(
  "n_train=%d  n_test=%d  p=%d  num_trees=%d  num_gfr=%d  num_burnin=%d  num_mcmc=%d  reps=%d\n\n",
  n_train,
  n_test,
  p,
  num_trees,
  num_gfr,
  num_burnin,
  num_mcmc,
  n_reps
))

# ---------------------------------------------------------------------------
# Helper: run one configuration and return timing + RMSE
# ---------------------------------------------------------------------------
run_once <- function(run_cpp, seed) {
  t0 <- proc.time()
  m <- bart(
    X_train = X_train,
    y_train = y_train,
    leaf_basis_train = basis_train,
    X_test = X_test,
    leaf_basis_test = basis_test,
    num_gfr = num_gfr,
    num_burnin = num_burnin,
    num_mcmc = num_mcmc,
    general_params = list(random_seed = seed),
    mean_forest_params = list(num_trees = num_trees),
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
    rmse_f_mean = mean(rmse_f),
    row.names = NULL
  )
}

res <- rbind(
  summarise(results_cpp, "cpp (run_cpp=TRUE)"),
  summarise(results_r, "R   (run_cpp=FALSE)")
)

cat("\n--- Results ---\n")
cat(sprintf(
  "%-22s  %10s  %10s  %12s  %13s\n",
  "Sampler",
  "Time (s)",
  "SD",
  "RMSE (obs)",
  "RMSE f(X,Z)"
))
cat(strrep("-", 74), "\n")
for (i in seq_len(nrow(res))) {
  cat(sprintf(
    "%-22s  %10.3f  %10.3f  %12.4f  %13.4f\n",
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
