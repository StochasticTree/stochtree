## Benchmark: C++ sampler loop vs. R sampler loop -- multivariate treatment BCF.
##
## Compares runtime and accuracy across run_cpp = TRUE / FALSE in bcf()
## with a 2-column binary treatment (multivariate BCF).
##
## DGP: propensity pi(X) = cbind(0.25 + 0.5*X[,1], 0.75 - 0.5*X[,2]) (2-column).
##      mu(X)  = 5*pi[,1] + 2*pi[,2] + 2*X[,3].
##      tau(X) = cbind(X[,2], X[,3]) (2-column CATE).
##      Z ~ Bernoulli(pi(X)) column-wise (binary, shape n x 2).
##      y = mu(X) + rowSums(Z * tau(X)) + noise.
## Adaptive coding is disabled; sigma2_leaf is not sampled for the tau forest.
##
## Usage:
##   Rscript debug/benchmark_cpp_vs_r_sampler_bcf_multivariate.R
##   or source() from an interactive session after devtools::load_all('.')

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
p <- 5
snr <- 2.0

X    <- matrix(runif(n * p), ncol = p)
pi_x <- cbind(0.25 + 0.5 * X[, 1], 0.75 - 0.5 * X[, 2])
mu_x <- pi_x[, 1] * 5 + pi_x[, 2] * 2 + 2 * X[, 3]
tau_x <- cbind(X[, 2], X[, 3])
Z    <- matrix(
  as.numeric(matrix(runif(n * 2), ncol = 2) < pi_x),
  ncol = 2
)
E_XZ <- mu_x + rowSums(Z * tau_x)
y    <- E_XZ + rnorm(n, sd = sd(E_XZ) / snr)

test_frac  <- 0.2
n_test     <- round(test_frac * n)
n_train    <- n - n_test
test_inds  <- sort(sample(seq_len(n), n_test, replace = FALSE))
train_inds <- setdiff(seq_len(n), test_inds)

X_train  <- X[train_inds, ]
X_test   <- X[test_inds, ]
Z_train  <- Z[train_inds, ]
Z_test   <- Z[test_inds, ]
pi_train <- pi_x[train_inds, ]
pi_test  <- pi_x[test_inds, ]
y_train  <- y[train_inds]
y_test   <- y[test_inds]
mu_test  <- mu_x[test_inds]
tau_test <- tau_x[test_inds, ]
f_test   <- mu_test + rowSums(Z_test * tau_test)

# ---------------------------------------------------------------------------
# Benchmark settings
# ---------------------------------------------------------------------------
num_gfr       <- 10
num_burnin    <- 0
num_mcmc      <- 100
num_trees_mu  <- 200
num_trees_tau <- 50
n_reps        <- 3

cat(sprintf(
  "n_train=%d  n_test=%d  p=%d  treatment_dim=2\nmu_trees=%d  tau_trees=%d  num_gfr=%d  num_burnin=%d  num_mcmc=%d  num_chains=%d  reps=%d\n\n",
  n_train, n_test, p,
  num_trees_mu, num_trees_tau,
  num_gfr, num_burnin, num_mcmc,
  num_chains, n_reps
))

# ---------------------------------------------------------------------------
# Helper: run one configuration and return timing + accuracy metrics
# ---------------------------------------------------------------------------
run_once <- function(run_cpp, seed = -1) {
  t0 <- proc.time()
  m <- bcf(
    X_train           = X_train,
    Z_train           = Z_train,
    y_train           = y_train,
    propensity_train  = pi_train,
    X_test            = X_test,
    Z_test            = Z_test,
    propensity_test   = pi_test,
    num_gfr           = num_gfr,
    num_burnin        = num_burnin,
    num_mcmc          = num_mcmc,
    prognostic_forest_params = list(num_trees = num_trees_mu),
    treatment_effect_forest_params = list(
      num_trees        = num_trees_tau,
      sample_sigma2_leaf = FALSE
    ),
    general_params = list(
      adaptive_coding = FALSE,
      random_seed     = seed,
      num_chains      = num_chains
    ),
    run_cpp = run_cpp
  )
  elapsed <- (proc.time() - t0)[["elapsed"]]

  y_hat    <- rowMeans(m$y_hat_test)
  mu_hat   <- rowMeans(m$mu_hat_test)
  # tau_hat_test: array(n_test, treatment_dim, num_samples)
  tau_hat1 <- apply(m$tau_hat_test[, 1, ], 1, mean)
  tau_hat2 <- apply(m$tau_hat_test[, 2, ], 1, mean)

  list(
    elapsed   = elapsed,
    rmse_y    = sqrt(mean((y_hat    - y_test)         ^ 2)),
    rmse_f    = sqrt(mean((y_hat    - f_test)         ^ 2)),
    rmse_mu   = sqrt(mean((mu_hat   - mu_test)        ^ 2)),
    rmse_tau1 = sqrt(mean((tau_hat1 - tau_test[, 1])  ^ 2)),
    rmse_tau2 = sqrt(mean((tau_hat2 - tau_test[, 2])  ^ 2))
  )
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
  get <- function(key) sapply(results, `[[`, key)
  data.frame(
    sampler       = label,
    elapsed_mean  = mean(get("elapsed")),
    elapsed_sd    = sd(get("elapsed")),
    rmse_y_mean   = mean(get("rmse_y")),
    rmse_f_mean   = mean(get("rmse_f")),
    rmse_mu_mean  = mean(get("rmse_mu")),
    rmse_tau1_mean = mean(get("rmse_tau1")),
    rmse_tau2_mean = mean(get("rmse_tau2")),
    row.names = NULL
  )
}

res <- rbind(
  summarise(results_cpp, "cpp (run_cpp=TRUE)"),
  summarise(results_r,   "R   (run_cpp=FALSE)")
)

cat("\n--- Results ---\n")
cat(sprintf(
  "%-22s  %8s  %8s  %9s  %9s  %9s  %10s  %10s\n",
  "Sampler", "Time (s)", "SD",
  "RMSE(y)", "RMSE(f)", "RMSE(mu)", "RMSE(tau1)", "RMSE(tau2)"
))
cat(strrep("-", 97), "\n")
for (i in seq_len(nrow(res))) {
  cat(sprintf(
    "%-22s  %8.3f  %8.3f  %9.4f  %9.4f  %9.4f  %10.4f  %10.4f\n",
    res$sampler[i],
    res$elapsed_mean[i],
    res$elapsed_sd[i],
    res$rmse_y_mean[i],
    res$rmse_f_mean[i],
    res$rmse_mu_mean[i],
    res$rmse_tau1_mean[i],
    res$rmse_tau2_mean[i]
  ))
}

speedup <- res$elapsed_mean[2] / res$elapsed_mean[1]
cat(sprintf("\nSpeedup (R / C++): %.2fx\n", speedup))
cat(sprintf(
  "RMSE delta (cpp - R):  y=%.4f  f=%.4f  mu=%.4f  tau1=%.4f  tau2=%.4f\n",
  res$rmse_y_mean[1]    - res$rmse_y_mean[2],
  res$rmse_f_mean[1]    - res$rmse_f_mean[2],
  res$rmse_mu_mean[1]   - res$rmse_mu_mean[2],
  res$rmse_tau1_mean[1] - res$rmse_tau1_mean[2],
  res$rmse_tau2_mean[1] - res$rmse_tau2_mean[2]
))
