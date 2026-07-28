## Benchmark: C++ sampler loop vs. R sampler loop – probit BART
## Compares runtime, Brier score, and RMSE-to-truth (vs. pnorm(f_X)) across
## run_cpp = TRUE / FALSE in bart().
##
## Usage: Rscript debug/benchmark_cpp_vs_r_sampler_probit.R
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

# Latent mean on the probit (standard-normal) scale – same step function as
# the continuous benchmark, keeping values well within identifiable range.
f_X <- (((0.00 <= X[, 1]) & (X[, 1] < 0.25)) *
  (-7.5) +
  ((0.25 <= X[, 1]) & (X[, 1] < 0.50)) * (-2.5) +
  ((0.50 <= X[, 1]) & (X[, 1] < 0.75)) * (2.5) +
  ((0.75 <= X[, 1]) & (X[, 1] < 1.00)) * (7.5))
p_X <- pnorm(f_X) # true P(Y = 1 | X)
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
num_gfr <- 10
num_mcmc <- 100
num_trees <- 200
n_reps <- 3

cat(sprintf(
  "n_train=%d  n_test=%d  p=%d  num_trees=%d  num_gfr=%d  num_mcmc=%d  num_chains=%d  reps=%d\n\n",
  n_train,
  n_test,
  p,
  num_trees,
  num_gfr,
  num_mcmc,
  num_chains,
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
    num_gfr = num_gfr,
    num_burnin = 0,
    num_mcmc = num_mcmc,
    mean_forest_params = list(num_trees = num_trees),
    general_params = list(
      random_seed = seed,
      outcome_model = OutcomeModel(outcome = "binary", link = "probit"),
      sample_sigma2_global = FALSE,
      num_chains = num_chains
    ),
    run_cpp = run_cpp
  )
  elapsed_sample <- (proc.time() - t0)[["elapsed"]]

  t1 <- proc.time()
  p_hat_mat <- predict(
    m,
    X       = X_test,
    type    = "posterior",
    terms   = "y_hat",
    scale   = "probability",
    run_cpp = run_cpp
  )
  elapsed_predict <- (proc.time() - t1)[["elapsed"]]

  raw <- if (is.list(p_hat_mat)) p_hat_mat$y_hat else p_hat_mat
  if (is.null(dim(raw))) {
    p_hat <- raw
  } else {
    p_hat <- rowMeans(raw)
  }

  brier  <- mean((p_hat - y_test)^2)
  rmse_p <- sqrt(mean((p_hat - p_test)^2))

  list(
    elapsed         = elapsed_sample + elapsed_predict,
    elapsed_sample  = elapsed_sample,
    elapsed_predict = elapsed_predict,
    brier           = brier,
    rmse_p          = rmse_p
  )
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
  elapsed         <- sapply(results, `[[`, "elapsed")
  elapsed_sample  <- sapply(results, `[[`, "elapsed_sample")
  elapsed_predict <- sapply(results, `[[`, "elapsed_predict")
  brier  <- sapply(results, `[[`, "brier")
  rmse_p <- sapply(results, `[[`, "rmse_p")
  data.frame(
    sampler              = label,
    elapsed_mean         = mean(elapsed),
    elapsed_sd           = sd(elapsed),
    elapsed_sample_mean  = mean(elapsed_sample),
    elapsed_predict_mean = mean(elapsed_predict),
    brier_mean           = mean(brier),
    rmse_p_mean          = mean(rmse_p),
    row.names            = NULL
  )
}

res <- rbind(
  summarise(results_cpp, "cpp (run_cpp=TRUE)"),
  summarise(results_r, "R   (run_cpp=FALSE)")
)

cat("\n--- Results ---\n")
cat(sprintf(
  "%-22s  %10s  %10s  %11s  %10s  %12s  %16s\n",
  "Sampler", "Total (s)", "Sample (s)", "Predict (s)", "SD", "Brier", "RMSE (vs pnorm)"
))
cat(strrep("-", 98), "\n")
for (i in seq_len(nrow(res))) {
  cat(sprintf(
    "%-22s  %10.3f  %10.3f  %11.3f  %10.3f  %12.4f  %16.4f\n",
    res$sampler[i], res$elapsed_mean[i], res$elapsed_sample_mean[i],
    res$elapsed_predict_mean[i], res$elapsed_sd[i],
    res$brier_mean[i], res$rmse_p_mean[i]
  ))
}

speedup <- res$elapsed_mean[2] / res$elapsed_mean[1]
cat(sprintf("\nSpeedup (R / C++): %.2fx\n", speedup))
cat(sprintf(
  "Brier delta (cpp - R):  %.4f\nRMSE-p delta (cpp - R): %.4f\n",
  res$brier_mean[1] - res$brier_mean[2],
  res$rmse_p_mean[1] - res$rmse_p_mean[2]
))
