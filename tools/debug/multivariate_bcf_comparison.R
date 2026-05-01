################################################################################
# Multivariate treatment BCF: R slow path vs R C++ path.
#
# Generates data in R, runs both paths, and prints correlations / RMSEs
# against ground truth for mu, tau1, tau2, and y.
#
# Usage (from repo root):
#   NOT_CRAN=true Rscript -e "devtools::load_all('.'); source('tools/debug/multivariate_bcf_comparison.R')"
################################################################################

suppressPackageStartupMessages(devtools::load_all("."))

# ── DGP ───────────────────────────────────────────────────────────────────────
set.seed(1234)
n   <- 500
p   <- 5
snr <- 2.0

X     <- matrix(runif(n * p), ncol = p)
pi_x  <- cbind(0.25 + 0.5 * X[, 1], 0.75 - 0.5 * X[, 2])
mu_x  <- pi_x[, 1] * 5 + pi_x[, 2] * 2 + 2 * X[, 3]
tau_x <- cbind(X[, 2], X[, 3])
Z     <- matrix(NA_integer_, nrow = n, ncol = 2)
for (j in 1:2) Z[, j] <- rbinom(n, 1, pi_x[, j])
E_XZ  <- mu_x + rowSums(Z * tau_x)
y     <- E_XZ + rnorm(n, 0, 1) * (sd(E_XZ) / snr)

test_set_pct <- 0.2
n_test     <- round(test_set_pct * n)
test_inds  <- sort(sample(seq_len(n), n_test, replace = FALSE))
train_inds <- setdiff(seq_len(n), test_inds)

X_train  <- X[train_inds, ];  X_test  <- X[test_inds, ]
Z_train  <- Z[train_inds, ];  Z_test  <- Z[test_inds, ]
pi_train <- pi_x[train_inds, ]; pi_test <- pi_x[test_inds, ]
y_train  <- y[train_inds];     y_test  <- y[test_inds]
mu_test  <- mu_x[test_inds]
tau_test <- tau_x[test_inds, ]

cat(sprintf("n_train=%d  n_test=%d  treatment_dim=2\n\n", length(train_inds), n_test))

num_gfr    <- 10
num_burnin <- 0
num_mcmc   <- 100
general_params <- list(adaptive_coding = FALSE)
prog_params <- list(sample_sigma2_leaf = FALSE)
trt_params  <- list(sample_sigma2_leaf = FALSE)

# Helper: posterior means from a 3D tau array (n x treatment_dim x num_samples)
tau_mean <- function(tau_arr, dim_idx) apply(tau_arr[, dim_idx, ], 1, mean)

report <- function(label, mu_hat, tau_hat1, tau_hat2, y_hat) {
  cat(sprintf("[%s]\n", label))
  cat(sprintf("  cor(mu_hat,   mu_true)    = %.4f   RMSE = %.4f\n",
              cor(mu_hat, mu_test), sqrt(mean((mu_hat - mu_test)^2))))
  cat(sprintf("  cor(tau_hat1, tau_true1)  = %.4f   RMSE = %.4f\n",
              cor(tau_hat1, tau_test[, 1]), sqrt(mean((tau_hat1 - tau_test[, 1])^2))))
  cat(sprintf("  cor(tau_hat2, tau_true2)  = %.4f   RMSE = %.4f\n",
              cor(tau_hat2, tau_test[, 2]), sqrt(mean((tau_hat2 - tau_test[, 2])^2))))
  cat(sprintf("  cor(y_hat,    y_test)     = %.4f   RMSE = %.4f\n",
              cor(y_hat, y_test), sqrt(mean((y_hat - y_test)^2))))
}

# ── R slow path ───────────────────────────────────────────────────────────────
cat("Running R slow path ...\n")
t0 <- proc.time()
bcf_slow <- bcf(
  X_train = X_train, Z_train = Z_train, y_train = y_train,
  propensity_train = pi_train,
  X_test  = X_test,  Z_test  = Z_test,  propensity_test = pi_test,
  num_gfr = num_gfr, num_burnin = num_burnin, num_mcmc = num_mcmc,
  general_params = general_params,
  prognostic_forest_params       = prog_params,
  treatment_effect_forest_params = trt_params
)
slow_time <- (proc.time() - t0)[3]
cat(sprintf("  done in %.1f s\n\n", slow_time))

# tau_hat_test: (n_test, treatment_dim, num_mcmc) for multivariate slow path
report(
  "Slow path",
  rowMeans(bcf_slow$mu_hat_test),
  tau_mean(bcf_slow$tau_hat_test, 1),
  tau_mean(bcf_slow$tau_hat_test, 2),
  rowMeans(bcf_slow$y_hat_test)
)

# ── R C++ path ────────────────────────────────────────────────────────────────
cat("\nRunning R C++ path ...\n")
t0 <- proc.time()
bcf_cpp <- bcf(
  X_train = X_train, Z_train = Z_train, y_train = y_train,
  propensity_train = pi_train,
  X_test  = X_test,  Z_test  = Z_test,  propensity_test = pi_test,
  num_gfr = num_gfr, num_burnin = num_burnin, num_mcmc = num_mcmc,
  general_params = general_params,
  prognostic_forest_params       = prog_params,
  treatment_effect_forest_params = trt_params,
  run_cpp = TRUE
)
cpp_time <- (proc.time() - t0)[3]
cat(sprintf("  done in %.1f s\n\n", cpp_time))

# tau_hat_test: (n_test, treatment_dim, num_mcmc) for multivariate C++ path
report(
  "C++ path",
  rowMeans(bcf_cpp$mu_hat_test),
  tau_mean(bcf_cpp$tau_hat_test, 1),
  tau_mean(bcf_cpp$tau_hat_test, 2),
  rowMeans(bcf_cpp$y_hat_test)
)

# ── Cross-path agreement ──────────────────────────────────────────────────────
cat("\n[Cross-path agreement: cor(C++, slow)]\n")
cat(sprintf("  mu_hat:   %.4f\n", cor(rowMeans(bcf_cpp$mu_hat_test),   rowMeans(bcf_slow$mu_hat_test))))
cat(sprintf("  tau_hat1: %.4f\n", cor(tau_mean(bcf_cpp$tau_hat_test, 1), tau_mean(bcf_slow$tau_hat_test, 1))))
cat(sprintf("  tau_hat2: %.4f\n", cor(tau_mean(bcf_cpp$tau_hat_test, 2), tau_mean(bcf_slow$tau_hat_test, 2))))
cat(sprintf("  y_hat:    %.4f\n", cor(rowMeans(bcf_cpp$y_hat_test),    rowMeans(bcf_slow$y_hat_test))))
cat(sprintf("\nSpeedup: %.2fx\n", slow_time / cpp_time))
