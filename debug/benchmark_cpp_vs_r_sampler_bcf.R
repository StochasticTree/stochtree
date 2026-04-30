## Benchmark: C++ sampler loop vs. R sampler loop (BCF)
## Simplest BCF case: univariate binary treatment, no RFX, no adaptive coding,
## no treatment intercept.
##
## Usage: Rscript debug/benchmark_cpp_vs_r_sampler_bcf.R
##        or source() from an interactive session after devtools::load_all('.')
library(stochtree)

args <- commandArgs(trailingOnly = TRUE)
num_chains <- 1L
idx <- grep("^--num-chains=", args)
if (length(idx)) {
  num_chains <- as.integer(sub("^--num-chains=", "", args[idx[1]]))
}

# ---------------------------------------------------------------------------
# Data-generating process
# ---------------------------------------------------------------------------
set.seed(1234)

n <- 2000
p <- 10
X <- matrix(runif(n * p), ncol = p)

# Prognostic function: step function on X[,1]
mu_X <- (((0.00 <= X[, 1]) & (X[, 1] < 0.25)) *
  (-7.5) +
  ((0.25 <= X[, 1]) & (X[, 1] < 0.50)) * (-2.5) +
  ((0.50 <= X[, 1]) & (X[, 1] < 0.75)) * (2.5) +
  ((0.75 <= X[, 1]) & (X[, 1] < 1.00)) * (7.5))

# Treatment effect: linear in X[,2]
tau_X <- 2 + 4 * X[, 2]

# Propensity score: mild confounding via X[,3]
pi_X <- 0.2 + 0.6 * X[, 3]
Z <- rbinom(n, 1, pi_X)

noise_sd <- 1
y <- mu_X + tau_X * Z + rnorm(n, 0, noise_sd)

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
pi_train <- pi_X[train_inds]
pi_test <- pi_X[test_inds]
mu_test <- mu_X[test_inds]
tau_test <- tau_X[test_inds]
f_test <- mu_test + tau_test * Z_test # E[y|X,Z]

# ---------------------------------------------------------------------------
# Benchmark settings
# ---------------------------------------------------------------------------
num_gfr <- 10
num_mcmc <- 100
num_trees_mu <- 200
num_trees_tau <- 50
n_reps <- 3

cat(sprintf(
  "n_train=%d  n_test=%d  p=%d  mu_trees=%d  tau_trees=%d  num_gfr=%d  num_mcmc=%d  num_chains=%d  reps=%d\n\n",
  n_train,
  n_test,
  p,
  num_trees_mu,
  num_trees_tau,
  num_gfr,
  num_mcmc,
  num_chains,
  n_reps
))

# ---------------------------------------------------------------------------
# Helper: run one configuration and return timing + RMSE
# ---------------------------------------------------------------------------
run_once <- function(run_cpp, num_gfr, num_mcmc, seed = -1) {
  t0 <- proc.time()
  m <- bcf(
    X_train = X_train,
    Z_train = Z_train,
    y_train = y_train,
    propensity_train = pi_train,
    X_test = X_test,
    Z_test = Z_test,
    propensity_test = pi_test,
    num_gfr = num_gfr,
    num_burnin = 0,
    num_mcmc = num_mcmc,
    prognostic_forest_params = list(num_trees = num_trees_mu),
    treatment_effect_forest_params = list(
      num_trees = num_trees_tau,
      sample_intercept = FALSE
    ),
    general_params = list(
      random_seed = seed,
      num_chains = num_chains,
      adaptive_coding = FALSE,
      propensity_covariate = "prognostic"
    ),
    run_cpp = run_cpp
  )
  elapsed <- (proc.time() - t0)[["elapsed"]]

  yhat <- rowMeans(m$y_hat_test)
  tauhat <- rowMeans(m$tau_hat_test)

  rmse_y <- sqrt(mean((yhat - y_test)^2))
  rmse_f <- sqrt(mean((yhat - f_test)^2))
  rmse_tau <- sqrt(mean((tauhat - tau_test)^2))

  list(elapsed = elapsed, rmse_y = rmse_y, rmse_f = rmse_f, rmse_tau = rmse_tau)
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
  results_cpp[[i]] <- run_once(
    run_cpp = TRUE,
    num_gfr = num_gfr,
    num_mcmc = num_mcmc,
    seed = seeds[i]
  )
}

cat("\nRunning R sampler (run_cpp = FALSE)...\n")
for (i in seq_len(n_reps)) {
  cat(sprintf("  rep %d/%d\n", i, n_reps))
  results_r[[i]] <- run_once(
    run_cpp = FALSE,
    num_gfr = num_gfr,
    num_mcmc = num_mcmc,
    seed = seeds[i]
  )
}

# ---------------------------------------------------------------------------
# Summarise
# ---------------------------------------------------------------------------
summarise <- function(results, label) {
  elapsed <- sapply(results, `[[`, "elapsed")
  rmse_y <- sapply(results, `[[`, "rmse_y")
  rmse_f <- sapply(results, `[[`, "rmse_f")
  rmse_tau <- sapply(results, `[[`, "rmse_tau")
  data.frame(
    sampler = label,
    elapsed_mean = mean(elapsed),
    elapsed_sd = sd(elapsed),
    rmse_y_mean = mean(rmse_y),
    rmse_f_mean = mean(rmse_f),
    rmse_tau_mean = mean(rmse_tau),
    row.names = NULL
  )
}

res <- rbind(
  summarise(results_cpp, "cpp (run_cpp=TRUE)"),
  summarise(results_r, "R   (run_cpp=FALSE)")
)

cat("\n--- Results ---\n")
cat(sprintf(
  "%-22s  %10s  %10s  %12s  %12s  %12s\n",
  "Sampler",
  "Time (s)",
  "  SD",
  "RMSE (obs)",
  "RMSE (f)",
  "RMSE (tau)"
))
cat(strrep("-", 84), "\n")
for (i in seq_len(nrow(res))) {
  cat(sprintf(
    "%-22s  %10.3f  %10.3f  %12.4f  %12.4f  %12.4f\n",
    res$sampler[i],
    res$elapsed_mean[i],
    res$elapsed_sd[i],
    res$rmse_y_mean[i],
    res$rmse_f_mean[i],
    res$rmse_tau_mean[i]
  ))
}

speedup <- res$elapsed_mean[2] / res$elapsed_mean[1]
cat(sprintf("\nSpeedup (R / C++): %.2fx\n", speedup))
cat(sprintf(
  "RMSE delta (cpp - R):  obs=%.4f  f=%.4f  tau=%.4f\n",
  res$rmse_y_mean[1] - res$rmse_y_mean[2],
  res$rmse_f_mean[1] - res$rmse_f_mean[2],
  res$rmse_tau_mean[1] - res$rmse_tau_mean[2]
))
