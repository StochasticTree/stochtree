## Benchmark: C++ sampler loop vs. R sampler loop – BART with random effects.
## Compares runtime and RMSE (vs. test outcomes and vs. true mean) across
## run_cpp = TRUE / FALSE in bart().
##
## DGP: continuous outcome with an additive intercept-only random effect.
##   y_i = f(X_i) + alpha_{g_i} + eps_i,  eps ~ N(0, 0.5^2)
##   f(X) is a piecewise-constant step function on X[,1].
##   Group intercepts alpha_g ~ N(0, 1), 10 groups.
##
## Usage: Rscript debug/benchmark_cpp_vs_r_sampler_rfx.R
##        or source() from an interactive session after devtools::load_all('.')
library(stochtree)

# ---------------------------------------------------------------------------
# Data-generating process
# ---------------------------------------------------------------------------
set.seed(1234)

n          <- 2000
p          <- 10
num_groups <- 10

X <- matrix(runif(n * p), ncol = p)
f_X <- ifelse(X[, 1] < 0.25, -7.5,
       ifelse(X[, 1] < 0.5,  -2.5,
       ifelse(X[, 1] < 0.75,  2.5, 7.5)))

group_ids <- sample(seq_len(num_groups), n, replace = TRUE)
rfx_coefs <- rnorm(num_groups)
rfx_term  <- rfx_coefs[group_ids]

mu_true <- f_X + rfx_term
y <- mu_true + rnorm(n, 0, 0.5)

test_frac  <- 0.2
n_test     <- round(test_frac * n)
test_inds  <- sort(sample(seq_len(n), n_test, replace = FALSE))
train_inds <- setdiff(seq_len(n), test_inds)

X_train         <- X[train_inds, ]
X_test          <- X[test_inds, ]
y_train         <- y[train_inds]
y_test          <- y[test_inds]
group_ids_train <- group_ids[train_inds]
group_ids_test  <- group_ids[test_inds]
rfx_basis_train <- matrix(1, nrow = length(train_inds), ncol = 1)
rfx_basis_test  <- matrix(1, nrow = n_test, ncol = 1)
mu_test         <- mu_true[test_inds]

# ---------------------------------------------------------------------------
# Benchmark settings
# ---------------------------------------------------------------------------
num_gfr    <- 10
num_burnin <- 100
num_mcmc   <- 100
num_trees  <- 200
n_reps     <- 3

cat(sprintf(
  "n_train=%d  n_test=%d  p=%d  num_groups=%d  num_trees=%d  num_gfr=%d  num_burnin=%d  num_mcmc=%d  reps=%d\n\n",
  length(train_inds), n_test, p, num_groups, num_trees, num_gfr, num_burnin, num_mcmc, n_reps
))

# ---------------------------------------------------------------------------
# Helper: run one configuration and return timing + metrics
# ---------------------------------------------------------------------------
run_once <- function(run_cpp, seed = -1) {
  t0 <- proc.time()
  m <- bart(
    X_train             = X_train,
    y_train             = y_train,
    X_test              = X_test,
    num_gfr             = num_gfr,
    num_burnin          = num_burnin,
    num_mcmc            = num_mcmc,
    rfx_group_ids_train = group_ids_train,
    rfx_group_ids_test  = group_ids_test,
    rfx_basis_train     = rfx_basis_train,
    rfx_basis_test      = rfx_basis_test,
    mean_forest_params  = list(num_trees = num_trees),
    general_params      = list(random_seed = seed),
    run_cpp             = run_cpp
  )
  elapsed <- (proc.time() - t0)[["elapsed"]]

  y_hat <- rowMeans(m$y_hat_test)
  rmse_y  <- sqrt(mean((y_hat - y_test)^2))
  rmse_mu <- sqrt(mean((y_hat - mu_test)^2))

  list(elapsed = elapsed, rmse_y = rmse_y, rmse_mu = rmse_mu)
}

# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------
seeds <- 1000 + seq_len(n_reps)

results_cpp <- vector("list", n_reps)
results_r   <- vector("list", n_reps)

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
  rmse_y  <- sapply(results, `[[`, "rmse_y")
  rmse_mu <- sapply(results, `[[`, "rmse_mu")
  data.frame(
    sampler      = label,
    elapsed_mean = mean(elapsed),
    elapsed_sd   = sd(elapsed),
    rmse_y_mean  = mean(rmse_y),
    rmse_mu_mean = mean(rmse_mu),
    row.names    = NULL
  )
}

res <- rbind(
  summarise(results_cpp, "cpp (run_cpp=TRUE)"),
  summarise(results_r,   "R   (run_cpp=FALSE)")
)

cat("\n--- Results ---\n")
cat(sprintf(
  "%-22s  %10s  %10s  %10s  %10s\n",
  "Sampler", "Time (s)", "SD", "RMSE (y)", "RMSE (mu)"
))
cat(strrep("-", 70), "\n")
for (i in seq_len(nrow(res))) {
  cat(sprintf(
    "%-22s  %10.3f  %10.3f  %10.4f  %10.4f\n",
    res$sampler[i], res$elapsed_mean[i], res$elapsed_sd[i],
    res$rmse_y_mean[i], res$rmse_mu_mean[i]
  ))
}

speedup <- res$elapsed_mean[2] / res$elapsed_mean[1]
cat(sprintf("\nSpeedup (R / C++): %.2fx\n", speedup))
cat(sprintf(
  "RMSE-y  delta (cpp - R): %.4f\nRMSE-mu delta (cpp - R): %.4f\n",
  res$rmse_y_mean[1] - res$rmse_y_mean[2],
  res$rmse_mu_mean[1] - res$rmse_mu_mean[2]
))
