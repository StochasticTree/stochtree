## Benchmark: C++ sampler loop vs. R sampler loop -- BCF with treatment intercept.
##
## Exercises SampleParametricTreatmentEffect() for both univariate and multivariate
## treatment by toggling sample_intercept=TRUE.  Verifies:
##   - The C++ path runs without error.
##   - tau_0_samples is populated and has the right shape.
##   - CATE RMSE (cpp) is close to CATE RMSE (R) -- large differences indicate
##     a residual accounting bug in the new intercept step.
##   - Speedup is reported for reference, though the primary goal is correctness.
##
## DGP (univariate):
##   mu(X) = step function on X[,1]
##   tau_0  = 1.5  (true global intercept)
##   tau(X) = tau_0 + 2*X[,3]  (full CATE = tau_0 + forest component)
##   pi(X)  = 0.2 + 0.6*X[,4]
##   Z ~ Bernoulli(pi(X))
##   y = mu(X) + (tau_0 + tau_forest(X)) * Z + noise
##
## DGP (multivariate, treatment_dim=2):
##   Same mu(X); tau_0 = c(0.5, 1.0); tau_k(X) = tau_0[k] + X[,k+1]
##
## Usage: Rscript debug/benchmark_cpp_vs_r_sampler_bcf_tau_intercept.R
##        or source() from an interactive session after devtools::load_all('.')
library(stochtree)

args <- commandArgs(trailingOnly = TRUE)
num_chains <- 1L
idx <- grep("^--num-chains=", args)
if (length(idx)) {
  num_chains <- as.integer(sub("^--num-chains=", "", args[idx[1]]))
}

# ---------------------------------------------------------------------------
# Shared settings
# ---------------------------------------------------------------------------
set.seed(1234)

n <- 2000
p <- 10
noise_sd <- 1.0
test_frac <- 0.2
n_test <- round(test_frac * n)
n_train <- n - n_test

num_gfr <- 10
num_burnin <- 0
num_mcmc <- 100
num_trees_mu <- 200
num_trees_tau <- 50
n_reps <- 3

# ---------------------------------------------------------------------------
# DGP: univariate binary treatment with global tau_0
# ---------------------------------------------------------------------------
X_all <- matrix(runif(n * p), ncol = p)

mu_X <- (
  ((0.00 <= X_all[, 1]) & (X_all[, 1] < 0.25)) * (-7.5) +
  ((0.25 <= X_all[, 1]) & (X_all[, 1] < 0.50)) * (-2.5) +
  ((0.50 <= X_all[, 1]) & (X_all[, 1] < 0.75)) * ( 2.5) +
  ((0.75 <= X_all[, 1]) & (X_all[, 1] < 1.00)) * ( 7.5)
)
TRUE_TAU0_UNIVARIATE <- 1.5
tau_forest_X <- 2.0 * X_all[, 3]               # forest component only
tau_X <- TRUE_TAU0_UNIVARIATE + tau_forest_X    # full CATE
pi_X <- 0.2 + 0.6 * X_all[, 4]
Z_all <- rbinom(n, 1, pi_X)
y_all <- mu_X + tau_X * Z_all + rnorm(n, 0, noise_sd)

test_inds  <- sort(sample(seq_len(n), n_test, replace = FALSE))
train_inds <- setdiff(seq_len(n), test_inds)

X_train_u  <- X_all[train_inds, ];  X_test_u  <- X_all[test_inds, ]
Z_train_u  <- Z_all[train_inds];    Z_test_u  <- Z_all[test_inds]
pi_train_u <- pi_X[train_inds];     pi_test_u <- pi_X[test_inds]
y_train_u  <- y_all[train_inds];    y_test_u  <- y_all[test_inds]
mu_test_u  <- mu_X[test_inds]
tau_test_u <- tau_X[test_inds]
f_test_u   <- mu_test_u + tau_test_u * Z_test_u

# ---------------------------------------------------------------------------
# DGP: multivariate (2-column) treatment with per-arm global tau_0
# ---------------------------------------------------------------------------
X_all_mv <- matrix(runif(n * p), ncol = p)
TRUE_TAU0_MV <- c(0.5, 1.0)

pi_mv <- cbind(0.25 + 0.5 * X_all_mv[, 1], 0.75 - 0.5 * X_all_mv[, 2])
mu_mv <- pi_mv[, 1] * 5 + pi_mv[, 2] * 2 + 2 * X_all_mv[, 3]
tau_forest_mv <- cbind(X_all_mv[, 2], X_all_mv[, 3])   # forest component only
tau_mv <- sweep(tau_forest_mv, 2, TRUE_TAU0_MV, "+")    # full CATE
Z_mv <- (matrix(runif(n * 2), ncol = 2) < pi_mv) * 1.0
y_mv <- mu_mv + rowSums(Z_mv * tau_mv) + rnorm(n, 0, noise_sd)

test_inds_mv  <- sort(sample(seq_len(n), n_test, replace = FALSE))
train_inds_mv <- setdiff(seq_len(n), test_inds_mv)

X_train_mv  <- X_all_mv[train_inds_mv, ];  X_test_mv  <- X_all_mv[test_inds_mv, ]
Z_train_mv  <- Z_mv[train_inds_mv, ];      Z_test_mv  <- Z_mv[test_inds_mv, ]
pi_train_mv <- pi_mv[train_inds_mv, ];     pi_test_mv <- pi_mv[test_inds_mv, ]
y_train_mv  <- y_mv[train_inds_mv];        y_test_mv  <- y_mv[test_inds_mv]
mu_test_mv  <- mu_mv[test_inds_mv]
tau_test_mv <- tau_mv[test_inds_mv, ]
f_test_mv   <- mu_test_mv + rowSums(Z_test_mv * tau_test_mv)

cat(sprintf(
  "n_train=%d  n_test=%d  p=%d  mu_trees=%d  tau_trees=%d  num_gfr=%d  num_mcmc=%d  num_chains=%d  reps=%d\n\n",
  n_train, n_test, p, num_trees_mu, num_trees_tau, num_gfr, num_mcmc, num_chains, n_reps
))

# ---------------------------------------------------------------------------
# Runner: univariate treatment with sample_intercept=TRUE
# ---------------------------------------------------------------------------
run_once_univariate <- function(run_cpp, seed = -1) {
  t0 <- proc.time()
  m <- bcf(
    X_train          = X_train_u,
    Z_train          = Z_train_u,
    y_train          = y_train_u,
    propensity_train = pi_train_u,
    X_test           = X_test_u,
    Z_test           = Z_test_u,
    propensity_test  = pi_test_u,
    num_gfr          = num_gfr,
    num_burnin       = num_burnin,
    num_mcmc         = num_mcmc,
    prognostic_forest_params = list(num_trees = num_trees_mu),
    treatment_effect_forest_params = list(
      num_trees        = num_trees_tau,
      sample_intercept = TRUE
    ),
    general_params = list(
      random_seed          = seed,
      num_chains           = num_chains,
      adaptive_coding      = FALSE,
      propensity_covariate = "prognostic"
    ),
    run_cpp = run_cpp
  )
  elapsed <- (proc.time() - t0)[["elapsed"]]

  tau_0_shape <- if (!is.null(m$tau_0_samples)) dim(m$tau_0_samples) else NULL
  tau_0_mean  <- if (!is.null(m$tau_0_samples)) mean(m$tau_0_samples) else NA_real_

  yhat   <- rowMeans(m$y_hat_test)
  tauhat <- rowMeans(m$tau_hat_test)

  list(
    elapsed    = elapsed,
    tau_0_mean = tau_0_mean,
    tau_0_shape = tau_0_shape,
    rmse_y     = sqrt(mean((yhat   - y_test_u)   ^ 2)),
    rmse_f     = sqrt(mean((yhat   - f_test_u)   ^ 2)),
    rmse_tau   = sqrt(mean((tauhat - tau_test_u) ^ 2))
  )
}

# ---------------------------------------------------------------------------
# Runner: multivariate (2-column) treatment with sample_intercept=TRUE
# ---------------------------------------------------------------------------
run_once_multivariate <- function(run_cpp, seed = -1) {
  t0 <- proc.time()
  m <- bcf(
    X_train          = X_train_mv,
    Z_train          = Z_train_mv,
    y_train          = y_train_mv,
    propensity_train = pi_train_mv,
    X_test           = X_test_mv,
    Z_test           = Z_test_mv,
    propensity_test  = pi_test_mv,
    num_gfr          = num_gfr,
    num_burnin       = num_burnin,
    num_mcmc         = num_mcmc,
    prognostic_forest_params = list(num_trees = num_trees_mu),
    treatment_effect_forest_params = list(
      num_trees         = num_trees_tau,
      sample_sigma2_leaf = FALSE,
      sample_intercept  = TRUE
    ),
    general_params = list(
      random_seed     = seed,
      num_chains      = num_chains,
      adaptive_coding = FALSE
    ),
    run_cpp = run_cpp
  )
  elapsed <- (proc.time() - t0)[["elapsed"]]

  tau_0_shape  <- if (!is.null(m$tau_0_samples)) dim(m$tau_0_samples) else NULL
  tau_0_mean_0 <- if (!is.null(m$tau_0_samples)) mean(m$tau_0_samples[1, ]) else NA_real_
  tau_0_mean_1 <- if (!is.null(m$tau_0_samples)) mean(m$tau_0_samples[2, ]) else NA_real_

  yhat    <- rowMeans(m$y_hat_test)
  tauhat1 <- rowMeans(m$tau_hat_test[, 1, ])
  tauhat2 <- rowMeans(m$tau_hat_test[, 2, ])

  list(
    elapsed      = elapsed,
    tau_0_mean_0 = tau_0_mean_0,
    tau_0_mean_1 = tau_0_mean_1,
    tau_0_shape  = tau_0_shape,
    rmse_y       = sqrt(mean((yhat    - y_test_mv)          ^ 2)),
    rmse_f       = sqrt(mean((yhat    - f_test_mv)          ^ 2)),
    rmse_tau1    = sqrt(mean((tauhat1 - tau_test_mv[, 1])   ^ 2)),
    rmse_tau2    = sqrt(mean((tauhat2 - tau_test_mv[, 2])   ^ 2))
  )
}

# ---------------------------------------------------------------------------
# Run: univariate
# ---------------------------------------------------------------------------
seeds <- 1000 + seq_len(n_reps)

cat(strrep("=", 60), "\n")
cat("UNIVARIATE TREATMENT  (sample_intercept=TRUE)\n")
cat(sprintf("True tau_0 = %.4f\n", TRUE_TAU0_UNIVARIATE))
cat(strrep("=", 60), "\n")

results_cpp_u <- vector("list", n_reps)
results_r_u   <- vector("list", n_reps)

cat("Running C++ sampler (run_cpp = TRUE)...\n")
for (i in seq_len(n_reps)) {
  cat(sprintf("  rep %d/%d\n", i, n_reps))
  results_cpp_u[[i]] <- run_once_univariate(run_cpp = TRUE, seed = seeds[i])
}

cat("\nRunning R sampler (run_cpp = FALSE)...\n")
for (i in seq_len(n_reps)) {
  cat(sprintf("  rep %d/%d\n", i, n_reps))
  results_r_u[[i]] <- run_once_univariate(run_cpp = FALSE, seed = seeds[i])
}

summarise_u <- function(results, label) {
  data.frame(
    sampler      = label,
    elapsed_mean = mean(sapply(results, `[[`, "elapsed")),
    elapsed_sd   = sd(sapply(results, `[[`, "elapsed")),
    tau_0_mean   = mean(sapply(results, `[[`, "tau_0_mean")),
    rmse_y       = mean(sapply(results, `[[`, "rmse_y")),
    rmse_f       = mean(sapply(results, `[[`, "rmse_f")),
    rmse_tau     = mean(sapply(results, `[[`, "rmse_tau")),
    row.names    = NULL
  )
}

res_u <- rbind(
  summarise_u(results_cpp_u, "cpp (run_cpp=TRUE)"),
  summarise_u(results_r_u,   "R   (run_cpp=FALSE)")
)

cat("\n--- Univariate Results ---\n")
shape_cpp <- results_cpp_u[[1]]$tau_0_shape
shape_r   <- results_r_u[[1]]$tau_0_shape
cat(sprintf("tau_0_samples shape  cpp=%s  R=%s\n",
  if (is.null(shape_cpp)) "NULL" else paste0("[", paste(shape_cpp, collapse=","), "]"),
  if (is.null(shape_r))   "NULL" else paste0("[", paste(shape_r,   collapse=","), "]")))
cat(sprintf(
  "%-22s  %8s  %8s  %10s  %9s  %9s  %10s\n",
  "Sampler", "Time (s)", "SD", "tau_0 mean", "RMSE(y)", "RMSE(f)", "RMSE(tau)"
))
cat(strrep("-", 90), "\n")
for (i in seq_len(nrow(res_u))) {
  cat(sprintf(
    "%-22s  %8.3f  %8.3f  %10.4f  %9.4f  %9.4f  %10.4f\n",
    res_u$sampler[i], res_u$elapsed_mean[i], res_u$elapsed_sd[i],
    res_u$tau_0_mean[i], res_u$rmse_y[i], res_u$rmse_f[i], res_u$rmse_tau[i]
  ))
}
cat(sprintf("True tau_0:  %.4f\n", TRUE_TAU0_UNIVARIATE))
speedup_u <- res_u$elapsed_mean[2] / res_u$elapsed_mean[1]
cat(sprintf("Speedup (R / C++): %.2fx\n", speedup_u))
cat(sprintf(
  "RMSE delta (cpp - R):  y=%.4f  f=%.4f  tau=%.4f\n",
  res_u$rmse_y[1]   - res_u$rmse_y[2],
  res_u$rmse_f[1]   - res_u$rmse_f[2],
  res_u$rmse_tau[1] - res_u$rmse_tau[2]
))

# ---------------------------------------------------------------------------
# Run: multivariate
# ---------------------------------------------------------------------------
cat("\n")
cat(strrep("=", 60), "\n")
cat("MULTIVARIATE TREATMENT  (treatment_dim=2, sample_intercept=TRUE)\n")
cat(sprintf("True tau_0 = [%.4f, %.4f]\n", TRUE_TAU0_MV[1], TRUE_TAU0_MV[2]))
cat(strrep("=", 60), "\n")

results_cpp_mv <- vector("list", n_reps)
results_r_mv   <- vector("list", n_reps)

cat("Running C++ sampler (run_cpp = TRUE)...\n")
for (i in seq_len(n_reps)) {
  cat(sprintf("  rep %d/%d\n", i, n_reps))
  results_cpp_mv[[i]] <- run_once_multivariate(run_cpp = TRUE, seed = seeds[i])
}

cat("\nRunning R sampler (run_cpp = FALSE)...\n")
for (i in seq_len(n_reps)) {
  cat(sprintf("  rep %d/%d\n", i, n_reps))
  results_r_mv[[i]] <- run_once_multivariate(run_cpp = FALSE, seed = seeds[i])
}

summarise_mv <- function(results, label) {
  data.frame(
    sampler      = label,
    elapsed_mean = mean(sapply(results, `[[`, "elapsed")),
    elapsed_sd   = sd(sapply(results, `[[`, "elapsed")),
    tau_0_mean_0 = mean(sapply(results, `[[`, "tau_0_mean_0")),
    tau_0_mean_1 = mean(sapply(results, `[[`, "tau_0_mean_1")),
    rmse_y       = mean(sapply(results, `[[`, "rmse_y")),
    rmse_f       = mean(sapply(results, `[[`, "rmse_f")),
    rmse_tau1    = mean(sapply(results, `[[`, "rmse_tau1")),
    rmse_tau2    = mean(sapply(results, `[[`, "rmse_tau2")),
    row.names    = NULL
  )
}

res_mv <- rbind(
  summarise_mv(results_cpp_mv, "cpp (run_cpp=TRUE)"),
  summarise_mv(results_r_mv,   "R   (run_cpp=FALSE)")
)

cat("\n--- Multivariate Results ---\n")
shape_cpp_mv <- results_cpp_mv[[1]]$tau_0_shape
shape_r_mv   <- results_r_mv[[1]]$tau_0_shape
cat(sprintf("tau_0_samples shape  cpp=%s  R=%s\n",
  if (is.null(shape_cpp_mv)) "NULL" else paste0("[", paste(shape_cpp_mv, collapse=","), "]"),
  if (is.null(shape_r_mv))   "NULL" else paste0("[", paste(shape_r_mv,   collapse=","), "]")))
cat(sprintf(
  "%-22s  %8s  %8s  %9s  %9s  %8s  %8s  %10s  %10s\n",
  "Sampler", "Time (s)", "SD", "tau_0[1]", "tau_0[2]",
  "RMSE(y)", "RMSE(f)", "RMSE(tau1)", "RMSE(tau2)"
))
cat(strrep("-", 105), "\n")
for (i in seq_len(nrow(res_mv))) {
  cat(sprintf(
    "%-22s  %8.3f  %8.3f  %9.4f  %9.4f  %8.4f  %8.4f  %10.4f  %10.4f\n",
    res_mv$sampler[i], res_mv$elapsed_mean[i], res_mv$elapsed_sd[i],
    res_mv$tau_0_mean_0[i], res_mv$tau_0_mean_1[i],
    res_mv$rmse_y[i], res_mv$rmse_f[i], res_mv$rmse_tau1[i], res_mv$rmse_tau2[i]
  ))
}
cat(sprintf("True tau_0:  [%.4f, %.4f]\n", TRUE_TAU0_MV[1], TRUE_TAU0_MV[2]))
speedup_mv <- res_mv$elapsed_mean[2] / res_mv$elapsed_mean[1]
cat(sprintf("Speedup (R / C++): %.2fx\n", speedup_mv))
cat(sprintf(
  "RMSE delta (cpp - R):  y=%.4f  f=%.4f  tau1=%.4f  tau2=%.4f\n",
  res_mv$rmse_y[1]    - res_mv$rmse_y[2],
  res_mv$rmse_f[1]    - res_mv$rmse_f[2],
  res_mv$rmse_tau1[1] - res_mv$rmse_tau1[2],
  res_mv$rmse_tau2[1] - res_mv$rmse_tau2[2]
))
