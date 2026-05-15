## Benchmark: C++ sampler loop vs. R sampler loop -- probit BCF.
##
## Compares runtime, Brier score on outcome, and latent-scale tau RMSE across
## run_cpp = TRUE / FALSE in bcf().
##
## DGP (latent-index model):
##     w = mu(X) + tau(X)*Z + eps,  eps ~ N(0, 1)
##     y = 1(w > 0)
##     mu(X) = 1 + 2*X[,1] + X[,2]           (confounded with propensity)
##     tau(X) = 0.5 + X[,3]                    (latent-scale CATE)
##     pi(X) = 0.4 + 0.2*X[,1]                (mild confounding)
##     Z ~ Bernoulli(pi(X))
##
## Metrics:
##     Brier score: mean((mean_s Phi(mu_hat[i,s] + tau_hat[i,s]*Z[i]) - y[i])^2)
##     RMSE(tau):   sqrt(mean((mean_s tau_hat_test[i,s] - tau_test[i])^2))
##
## Usage:
##   Rscript debug/benchmark_cpp_vs_r_sampler_bcf_probit.R
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

X    <- matrix(runif(n * p), ncol = p)
mu_X  <- 1 + 2 * X[, 1] + X[, 2]
tau_X <- 0.5 + X[, 3]
pi_X  <- 0.4 + 0.2 * X[, 1]
Z     <- rbinom(n, 1, pi_X)

w <- mu_X + tau_X * Z + rnorm(n)
y <- as.numeric(w > 0)

test_frac  <- 0.2
n_test     <- round(test_frac * n)
n_train    <- n - n_test
test_inds  <- sort(sample(seq_len(n), n_test, replace = FALSE))
train_inds <- setdiff(seq_len(n), test_inds)

X_train  <- X[train_inds, ]
X_test   <- X[test_inds, ]
Z_train  <- Z[train_inds]
Z_test   <- Z[test_inds]
pi_train <- pi_X[train_inds]
pi_test  <- pi_X[test_inds]
y_train  <- y[train_inds]
y_test   <- y[test_inds]
tau_test <- tau_X[test_inds]

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
  "n_train=%d  n_test=%d  p=%d\nmu_trees=%d  tau_trees=%d  num_gfr=%d  num_burnin=%d  num_mcmc=%d  num_chains=%d  reps=%d\n\n",
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
    num_gfr           = num_gfr,
    num_burnin        = num_burnin,
    num_mcmc          = num_mcmc,
    prognostic_forest_params = list(
      num_trees          = num_trees_mu,
      sample_sigma2_leaf = FALSE
    ),
    treatment_effect_forest_params = list(
      num_trees          = num_trees_tau,
      sample_sigma2_leaf = FALSE,
      sample_intercept   = FALSE
    ),
    general_params = list(
      random_seed        = seed,
      num_chains         = num_chains,
      outcome_model      = OutcomeModel(outcome = "binary", link = "probit"),
      sample_sigma2_global = FALSE
    ),
    run_cpp = run_cpp
  )
  elapsed_sample <- (proc.time() - t0)[["elapsed"]]

  # Request latent-scale mu and tau (scale = "linear", no probit transform applied)
  t1 <- proc.time()
  preds <- predict(
    m, X = X_test, Z = Z_test, propensity = pi_test,
    terms = c("mu", "tau"), scale = "linear", run_cpp = run_cpp
  )
  elapsed_predict <- (proc.time() - t1)[["elapsed"]]

  # mu_hat, tau_hat: (n_test, num_samples) — latent scale
  mu_hat  <- preds$mu_hat   # n_test x num_samples
  tau_hat <- preds$tau_hat  # n_test x num_samples

  # P(Y=1 | X, Z, sample s) = Phi(mu_hat[i,s] + tau_hat[i,s] * Z_test[i])
  linear_pred    <- mu_hat + tau_hat * Z_test  # broadcasts Z_test over columns
  p_hat_samples  <- pnorm(linear_pred)          # n_test x num_samples
  p_hat_mean     <- rowMeans(p_hat_samples)     # n_test

  tau_hat_mean   <- rowMeans(tau_hat)           # n_test

  brier    <- mean((p_hat_mean - y_test)^2)
  rmse_tau <- sqrt(mean((tau_hat_mean - tau_test)^2))

  list(
    elapsed         = elapsed_sample + elapsed_predict,
    elapsed_sample  = elapsed_sample,
    elapsed_predict = elapsed_predict,
    brier           = brier,
    rmse_tau        = rmse_tau
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
    sampler              = label,
    elapsed_mean         = mean(get("elapsed")),
    elapsed_sd           = sd(get("elapsed")),
    elapsed_sample_mean  = mean(get("elapsed_sample")),
    elapsed_predict_mean = mean(get("elapsed_predict")),
    brier_mean           = mean(get("brier")),
    rmse_tau_mean        = mean(get("rmse_tau")),
    row.names            = NULL
  )
}

res <- rbind(
  summarise(results_cpp, "cpp (run_cpp=TRUE)"),
  summarise(results_r,   "R   (run_cpp=FALSE)")
)

cat("\n--- Results ---\n")
cat(sprintf(
  "%-22s  %10s  %10s  %10s  %10s  %10s  %12s\n",
  "Sampler", "Total (s)", "Samp (s)", "Pred (s)", "SD", "Brier", "RMSE (tau)"
))
cat(strrep("-", 92), "\n")
for (i in seq_len(nrow(res))) {
  cat(sprintf(
    "%-22s  %10.3f  %10.3f  %10.3f  %10.3f  %10.4f  %12.4f\n",
    res$sampler[i],
    res$elapsed_mean[i],
    res$elapsed_sample_mean[i],
    res$elapsed_predict_mean[i],
    res$elapsed_sd[i],
    res$brier_mean[i],
    res$rmse_tau_mean[i]
  ))
}

speedup <- res$elapsed_mean[2] / res$elapsed_mean[1]
cat(sprintf("\nSpeedup (R / C++): %.2fx\n", speedup))
cat(sprintf(
  "Delta (cpp - R):  brier=%.4f  rmse_tau=%.4f\n",
  res$brier_mean[1] - res$brier_mean[2],
  res$rmse_tau_mean[1] - res$rmse_tau_mean[2]
))
