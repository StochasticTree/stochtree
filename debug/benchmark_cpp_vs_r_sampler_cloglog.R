## Benchmark: C++ sampler loop vs. R sampler loop – cloglog BART
## Compares runtime, Brier score, and RMSE-to-truth (vs. true P(Y=1|X)) across
## run_cpp = TRUE / FALSE in bart().
##
## DGP uses the cloglog link: P(Y=1|X) = 1 - exp(-exp(f(X))).
## f(X) is a smooth sinusoidal function of two covariates, keeping probabilities
## in [~0.25, ~0.75] for stable mixing. GFR is disabled (num_gfr = 0).
##
## Usage: Rscript debug/benchmark_cpp_vs_r_sampler_cloglog.R
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

# Latent mean on the cloglog (log-log) scale.
# f_X is centred near -0.5 so that P(Y=1|X) = 1 - exp(-exp(f_X)) stays
# in [~0.25, ~0.75], avoiding extreme probabilities that inflate tree depth.
f_X <- 0.6 * sin(2 * pi * X[, 1]) + 0.4 * cos(2 * pi * X[, 2]) - 0.5
p_X <- 1 - exp(-exp(f_X)) # true P(Y = 1 | X)
y <- rbinom(n, 1L, p_X) # observed binary outcome

test_frac <- 0.2
n_test <- round(test_frac * n)
n_train <- n - n_test
test_inds <- sort(sample(seq_len(n), n_test, replace = FALSE))
train_inds <- setdiff(seq_len(n), test_inds)

X_train <- X[train_inds, ]
X_test <- X[test_inds, ]
y_train <- y[train_inds]
y_test <- y[test_inds]
p_test <- p_X[test_inds]

# ---------------------------------------------------------------------------
# Benchmark settings
# ---------------------------------------------------------------------------
num_gfr <- 0
num_burnin <- 100
num_mcmc <- 100
num_trees <- 200
n_reps <- 3

cat(sprintf(
  "n_train=%d  n_test=%d  p=%d  num_trees=%d  num_gfr=%d  num_burnin=%d  num_mcmc=%d  num_chains=%d  reps=%d\n\n",
  n_train,
  n_test,
  p,
  num_trees,
  num_gfr,
  num_burnin,
  num_mcmc,
  num_chains,
  n_reps
))

# ---------------------------------------------------------------------------
# Helper: run one configuration and return timing + metrics
# ---------------------------------------------------------------------------
run_once <- function(run_cpp, seed = -1) {
  t0 <- proc.time()
  m <- bart(
    X_train = X_train,
    y_train = y_train,
    X_test = X_test,
    num_gfr = num_gfr,
    num_burnin = num_burnin,
    num_mcmc = num_mcmc,
    mean_forest_params = list(
      num_trees = num_trees,
      sample_sigma2_leaf = FALSE
    ),
    general_params = list(
      random_seed = seed,
      outcome_model = OutcomeModel(outcome = "binary", link = "cloglog"),
      sample_sigma2_global = FALSE,
      num_chains = num_chains
    ),
    run_cpp = run_cpp
  )
  elapsed <- (proc.time() - t0)[["elapsed"]]

  # Posterior-mean predicted probability on the test set
  p_hat_mat <- predict(
    m,
    X = X_test,
    type = "posterior",
    terms = "y_hat",
    scale = "probability"
  )
  if (is.null(dim(p_hat_mat))) {
    p_hat_mat <- matrix(p_hat_mat, ncol = 1)
  }
  p_hat <- rowMeans(p_hat_mat)

  brier <- mean((p_hat - y_test)^2) # Brier score (lower is better)
  rmse_p <- sqrt(mean((p_hat - p_test)^2)) # RMSE vs. true cloglog probabilities

  list(elapsed = elapsed, brier = brier, rmse_p = rmse_p)
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
  brier <- sapply(results, `[[`, "brier")
  rmse_p <- sapply(results, `[[`, "rmse_p")
  data.frame(
    sampler = label,
    elapsed_mean = mean(elapsed),
    elapsed_sd = sd(elapsed),
    brier_mean = mean(brier),
    rmse_p_mean = mean(rmse_p),
    row.names = NULL
  )
}

res <- rbind(
  summarise(results_cpp, "cpp (run_cpp=TRUE)"),
  summarise(results_r, "R   (run_cpp=FALSE)")
)

cat("\n--- Results ---\n")
cat(sprintf(
  "%-22s  %10s  %10s  %12s  %15s\n",
  "Sampler",
  "Time (s)",
  "SD",
  "Brier",
  "RMSE (vs truth)"
))
cat(strrep("-", 75), "\n")
for (i in seq_len(nrow(res))) {
  cat(sprintf(
    "%-22s  %10.3f  %10.3f  %12.4f  %15.4f\n",
    res$sampler[i],
    res$elapsed_mean[i],
    res$elapsed_sd[i],
    res$brier_mean[i],
    res$rmse_p_mean[i]
  ))
}

speedup <- res$elapsed_mean[2] / res$elapsed_mean[1]
cat(sprintf("\nSpeedup (R / C++): %.2fx\n", speedup))
cat(sprintf(
  "Brier delta (cpp - R):  %.4f\nRMSE-p delta (cpp - R): %.4f\n",
  res$brier_mean[1] - res$brier_mean[2],
  res$rmse_p_mean[1] - res$rmse_p_mean[2]
))
