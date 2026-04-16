## Benchmark: C++ sampler loop vs. R sampler loop – ordinal cloglog BART
## Compares runtime, mean Brier score, and mean RMSE-to-truth (vs. true class
## probabilities) across run_cpp = TRUE / FALSE in bart().
##
## DGP uses 4 ordinal categories with a cloglog link.
## The latent step function f(X) is on the log-log scale, and each category
## boundary (gamma_k) is fixed at log(k) for k = 1, 2, 3 so the four
## cumulative probabilities are P(Y <= k | X) = 1 - exp(-exp(f(X) - gamma_k)).
##
## Usage: Rscript debug/benchmark_cpp_vs_r_sampler_ordinal_cloglog.R
##        or source() from an interactive session after devtools::load_all('.')
library(stochtree)

# ---------------------------------------------------------------------------
# Data-generating process
# ---------------------------------------------------------------------------
set.seed(1234)

n <- 2000
p <- 10
X <- matrix(runif(n * p), ncol = p)

# Latent step function on the cloglog scale
f_X <- (((0.00 <= X[, 1]) & (X[, 1] < 0.25)) *
  (-2.0) +
  ((0.25 <= X[, 1]) & (X[, 1] < 0.50)) * (-0.5) +
  ((0.50 <= X[, 1]) & (X[, 1] < 0.75)) * (0.5) +
  ((0.75 <= X[, 1]) & (X[, 1] < 1.00)) * (1.0))

# Fixed log-scale cutpoints (gamma_k); K = 4 categories => K-1 = 3 cutpoints
# gamma_0 is fixed at 0 for identifiability; gamma_1 = log(2), gamma_2 = log(3)
K <- 4
gamma_true <- c(0, log(2), log(3))

# True cumulative probabilities: P(Y <= k | X) = 1 - exp(-exp(f_X - gamma_k))
# True class probabilities: P(Y = k | X) = P(Y <= k) - P(Y <= k-1)
cum_prob <- outer(f_X, gamma_true, function(f, g) 1 - exp(-exp(f - g)))
p_X <- cbind(
  cum_prob[, 1],
  cum_prob[, 2] - cum_prob[, 1],
  cum_prob[, 3] - cum_prob[, 2],
  1 - cum_prob[, 3]
) # n x K matrix of true class probs

# Draw ordinal outcomes (1-indexed: 1, 2, 3, 4)
u <- runif(n)
y <- as.integer(u > cum_prob[, 1]) +
  as.integer(u > cum_prob[, 2]) +
  as.integer(u > cum_prob[, 3]) +
  1L

test_frac <- 0.2
n_test <- round(test_frac * n)
n_train <- n - n_test
test_inds <- sort(sample(seq_len(n), n_test, replace = FALSE))
train_inds <- setdiff(seq_len(n), test_inds)

X_train <- X[train_inds, ]
X_test <- X[test_inds, ]
y_train <- y[train_inds]
y_test <- y[test_inds]
p_test <- p_X[test_inds, ] # n_test x K matrix of true class probabilities

# ---------------------------------------------------------------------------
# Benchmark settings
# ---------------------------------------------------------------------------
num_gfr <- 10
num_mcmc <- 100
num_trees <- 200
n_reps <- 3

cat(sprintf(
  "K=%d  n_train=%d  n_test=%d  p=%d  num_trees=%d  num_gfr=%d  num_mcmc=%d  reps=%d\n\n",
  K,
  n_train,
  n_test,
  p,
  num_trees,
  num_gfr,
  num_mcmc,
  n_reps
))

# ---------------------------------------------------------------------------
# Helper: run one configuration and return timing + metrics
# ---------------------------------------------------------------------------
run_once <- function(run_cpp, num_gfr, num_mcmc, seed = -1) {
  t0 <- proc.time()
  m <- bart(
    X_train = X_train,
    y_train = y_train,
    X_test = X_test,
    num_gfr = num_gfr,
    num_burnin = 0,
    num_mcmc = num_mcmc,
    mean_forest_params = list(num_trees = num_trees),
    general_params = list(
      random_seed = seed,
      outcome_model = OutcomeModel(outcome = "ordinal", link = "cloglog"),
      sample_sigma2_global = FALSE
    ),
    run_cpp = run_cpp
  )
  elapsed <- (proc.time() - t0)[["elapsed"]]

  # Posterior-mean predicted class probabilities on the test set
  # predict() returns an n_test x K x num_mcmc array for ordinal outcomes
  p_hat_arr <- predict(
    m,
    X = X_test,
    type = "posterior",
    terms = "y_hat",
    scale = "probability"
  )
  p_hat <- apply(p_hat_arr, c(1, 2), mean) # n_test x K posterior mean

  # Mean Brier score across classes (multi-class generalisation)
  brier <- mean((p_hat - p_test)^2)

  # Per-class RMSE vs. true probabilities, then averaged
  rmse_p <- mean(sqrt(colMeans((p_hat - p_test)^2)))

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

speedup <- res$elapsed_mean[res$sampler == "R   (run_cpp=FALSE)"] /
  res$elapsed_mean[res$sampler == "cpp (run_cpp=TRUE)"]
cat(sprintf(
  "\nSpeedup (R / cpp): %.2fx\n",
  speedup
))
cat(sprintf(
  "Brier delta (cpp - R):  %.4f\nRMSE-p delta (cpp - R): %.4f\n",
  res$brier_mean[1] - res$brier_mean[2],
  res$rmse_p_mean[1] - res$rmse_p_mean[2]
))
