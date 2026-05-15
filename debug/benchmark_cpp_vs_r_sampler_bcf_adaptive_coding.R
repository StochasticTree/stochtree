## Benchmark: C++ sampler loop vs. R sampler loop -- BCF with adaptive coding.
##
## Exercises SampleAdaptiveCodingParameters() for binary treatment.  Verifies:
##   - The C++ path runs without error.
##   - b_0_samples and b_1_samples are populated with the right shape.
##   - mu_hat + Z * tau_hat == y_hat (internal decomposition check).
##   - CATE RMSE (cpp) is close to CATE RMSE (R) -- large differences indicate
##     a residual accounting bug in SampleAdaptiveCodingParameters or the
##     mu/tau prediction split.
##   - Speedup is reported for reference, though the primary goal is correctness.
##
## DGP:
##   mu(X) = step function on X[,1]
##   tau_forest(X) = 2 * X[,3]         (forest component, scale-free)
##   true_b_0 = -0.5
##   true_b_1 =  1.5
##   CATE(X)  = (true_b_1 - true_b_0) * tau_forest(X)  = 4 * X[,3]
##   pi(X)    = 0.2 + 0.6 * X[,4]
##   Z ~ Bernoulli(pi(X))
##   y = mu(X) + (true_b_0*(1-Z) + true_b_1*Z) * tau_forest(X) + noise
##
## Usage: Rscript debug/benchmark_cpp_vs_r_sampler_bcf_adaptive_coding.R
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
# DGP: binary treatment with adaptive coding
# ---------------------------------------------------------------------------
X_all <- matrix(runif(n * p), ncol = p)

mu_X <- (
  ((0.00 <= X_all[, 1]) & (X_all[, 1] < 0.25)) * (-7.5) +
  ((0.25 <= X_all[, 1]) & (X_all[, 1] < 0.50)) * (-2.5) +
  ((0.50 <= X_all[, 1]) & (X_all[, 1] < 0.75)) * ( 2.5) +
  ((0.75 <= X_all[, 1]) & (X_all[, 1] < 1.00)) * ( 7.5)
)
TRUE_B_0 <- -0.5
TRUE_B_1 <-  1.5
tau_forest_X <- 2.0 * X_all[, 3]                       # forest component (no coding)
tau_X        <- (TRUE_B_1 - TRUE_B_0) * tau_forest_X   # CATE = (b_1 - b_0) * tau_forest
pi_X         <- 0.2 + 0.6 * X_all[, 4]
Z_all        <- rbinom(n, 1, pi_X)
coded_basis  <- TRUE_B_0 * (1 - Z_all) + TRUE_B_1 * Z_all
y_all        <- mu_X + coded_basis * tau_forest_X + rnorm(n, 0, noise_sd)

test_inds  <- sort(sample(seq_len(n), n_test, replace = FALSE))
train_inds <- setdiff(seq_len(n), test_inds)

X_train  <- X_all[train_inds, ];  X_test  <- X_all[test_inds, ]
Z_train  <- Z_all[train_inds];    Z_test  <- Z_all[test_inds]
pi_train <- pi_X[train_inds];     pi_test <- pi_X[test_inds]
y_train  <- y_all[train_inds];    y_test  <- y_all[test_inds]
mu_test  <- mu_X[test_inds]
tau_test <- tau_X[test_inds]                            # true CATE
coded_test <- TRUE_B_0 * (1 - Z_test) + TRUE_B_1 * Z_test
f_test   <- mu_test + coded_test * tau_forest_X[test_inds]

cat(sprintf(
  "n_train=%d  n_test=%d  p=%d  mu_trees=%d  tau_trees=%d  num_gfr=%d  num_mcmc=%d  num_chains=%d  reps=%d\n",
  n_train, n_test, p, num_trees_mu, num_trees_tau, num_gfr, num_mcmc, num_chains, n_reps
))
cat(sprintf("true_b_0=%.1f  true_b_1=%.1f\n\n", TRUE_B_0, TRUE_B_1))

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
run_once <- function(run_cpp, seed = -1) {
  t0 <- proc.time()
  m <- bcf(
    X_train          = X_train,
    Z_train          = Z_train,
    y_train          = y_train,
    propensity_train = pi_train,
    num_gfr          = num_gfr,
    num_burnin       = num_burnin,
    num_mcmc         = num_mcmc,
    prognostic_forest_params = list(num_trees = num_trees_mu),
    treatment_effect_forest_params = list(num_trees = num_trees_tau),
    general_params = list(
      adaptive_coding      = TRUE,
      random_seed          = seed,
      num_chains           = num_chains,
      propensity_covariate = "prognostic"
    ),
    run_cpp = run_cpp
  )
  elapsed_sample <- (proc.time() - t0)[["elapsed"]]

  t1 <- proc.time()
  preds <- predict(m, X = X_test, Z = Z_test, propensity = pi_test, run_cpp = run_cpp)
  elapsed_predict <- (proc.time() - t1)[["elapsed"]]

  # Internal consistency: y_hat == mu_hat + Z * tau_hat
  max_decomp_err_train <- max(abs(
    m$y_hat_train - (m$mu_hat_train + Z_train * m$tau_hat_train)
  ))
  max_decomp_err_test <- max(abs(
    preds$y_hat - (preds$mu_hat + Z_test * preds$tau_hat)
  ))

  yhat   <- rowMeans(preds$y_hat)
  tauhat <- rowMeans(preds$tau_hat)

  list(
    elapsed              = elapsed_sample + elapsed_predict,
    elapsed_sample       = elapsed_sample,
    elapsed_predict      = elapsed_predict,
    b0_mean              = mean(m$b_0_samples),
    b1_mean              = mean(m$b_1_samples),
    b0_length            = length(m$b_0_samples),
    b1_length            = length(m$b_1_samples),
    max_decomp_err_train = max_decomp_err_train,
    max_decomp_err_test  = max_decomp_err_test,
    rmse_y               = sqrt(mean((yhat   - y_test)   ^ 2)),
    rmse_f               = sqrt(mean((yhat   - f_test)   ^ 2)),
    rmse_tau             = sqrt(mean((tauhat - tau_test)  ^ 2))
  )
}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
seeds <- 1000 + seq_len(n_reps)

cat(strrep("=", 60), "\n")
cat("BINARY TREATMENT  (adaptive_coding=TRUE)\n")
cat(sprintf("True b_0=%.1f  b_1=%.1f  CATE=(b1-b0)*tau_forest(X)\n", TRUE_B_0, TRUE_B_1))
cat(strrep("=", 60), "\n")

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

summarise <- function(results, label) {
  data.frame(
    sampler              = label,
    elapsed_mean         = mean(sapply(results, `[[`, "elapsed")),
    elapsed_sd           = sd(sapply(results, `[[`, "elapsed")),
    elapsed_sample_mean  = mean(sapply(results, `[[`, "elapsed_sample")),
    elapsed_predict_mean = mean(sapply(results, `[[`, "elapsed_predict")),
    b0_mean              = mean(sapply(results, `[[`, "b0_mean")),
    b1_mean              = mean(sapply(results, `[[`, "b1_mean")),
    max_decomp_err_train = mean(sapply(results, `[[`, "max_decomp_err_train")),
    max_decomp_err_test  = mean(sapply(results, `[[`, "max_decomp_err_test")),
    rmse_y               = mean(sapply(results, `[[`, "rmse_y")),
    rmse_f               = mean(sapply(results, `[[`, "rmse_f")),
    rmse_tau             = mean(sapply(results, `[[`, "rmse_tau")),
    row.names            = NULL
  )
}

res <- rbind(
  summarise(results_cpp, "cpp (run_cpp=TRUE)"),
  summarise(results_r,   "R   (run_cpp=FALSE)")
)

cat("\n--- Results ---\n")
cat(sprintf("b_0_samples length  cpp=%d  R=%d\n",
  results_cpp[[1]]$b0_length, results_r[[1]]$b0_length))
cat(sprintf("b_1_samples length  cpp=%d  R=%d\n",
  results_cpp[[1]]$b1_length, results_r[[1]]$b1_length))
cat("\n")
cat(sprintf(
  "%-22s  %8s  %8s  %8s  %6s  %8s  %8s  %13s  %13s  %8s  %8s  %10s\n",
  "Sampler", "Total (s)", "Samp (s)", "Pred (s)", "SD",
  "b_0 mean", "b_1 mean",
  "max_decomp_tr", "max_decomp_te",
  "RMSE(y)", "RMSE(f)", "RMSE(tau)"
))
cat(strrep("-", 140), "\n")
for (i in seq_len(nrow(res))) {
  cat(sprintf(
    "%-22s  %8.3f  %8.3f  %8.3f  %6.3f  %8.4f  %8.4f  %13.2e  %13.2e  %8.4f  %8.4f  %10.4f\n",
    res$sampler[i], res$elapsed_mean[i],
    res$elapsed_sample_mean[i], res$elapsed_predict_mean[i],
    res$elapsed_sd[i],
    res$b0_mean[i], res$b1_mean[i],
    res$max_decomp_err_train[i], res$max_decomp_err_test[i],
    res$rmse_y[i], res$rmse_f[i], res$rmse_tau[i]
  ))
}
cat(sprintf("True b_0=%.4f  b_1=%.4f\n", TRUE_B_0, TRUE_B_1))
speedup <- res$elapsed_mean[2] / res$elapsed_mean[1]
cat(sprintf("Speedup (R / C++): %.2fx\n", speedup))
cat(sprintf(
  "RMSE delta (cpp - R):  y=%.4f  f=%.4f  tau=%.4f\n",
  res$rmse_y[1]   - res$rmse_y[2],
  res$rmse_f[1]   - res$rmse_f[2],
  res$rmse_tau[1] - res$rmse_tau[2]
))
