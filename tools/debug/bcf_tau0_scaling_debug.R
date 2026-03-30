# Debug script: verify that colMeans(tau(X)) + tau_0 == colMeans(CATE)
# for the reparameterized BCF model (sample_intercept = TRUE).
# Based on DGP from vignettes/ReparameterizedCausalInference.Rmd

library(stochtree)

set.seed(42)

# --- DGP (from vignette) ---
n <- 400
p <- 20
snr <- 2
X <- matrix(runif(n * p), n, p)
mu_x <- sin(pi * X[, 1] * X[, 2]) + 2 * (X[, 3] - 0.5)^2 + X[, 4]
tau_x <- 5
pi_x <- pnorm(1.5 * X[, 1] - 0.5 * X[, 2])
Z <- rbinom(n, 1, pi_x)
E_XZ <- mu_x + Z * tau_x
sigma_true <- sd(E_XZ) / snr
y <- E_XZ + rnorm(n, 0, 1) * sigma_true

test_set_pct <- 0.2
n_test <- round(test_set_pct * n)
n_train <- n - n_test
test_inds <- sort(sample(1:n, n_test, replace = FALSE))
train_inds <- (1:n)[!((1:n) %in% test_inds)]
X_test <- X[test_inds, ]; X_train <- X[train_inds, ]
pi_test <- pi_x[test_inds]; pi_train <- pi_x[train_inds]
Z_test <- Z[test_inds]; Z_train <- Z[train_inds]
y_test <- y[test_inds]; y_train <- y[train_inds]

# --- Fit reparameterized BCF ---
bcf_model <- bcf(
  X_train = X_train,
  Z_train = Z_train,
  y_train = y_train,
  propensity_train = pi_train,
  X_test = X_test,
  Z_test = Z_test,
  propensity_test = pi_test,
  num_gfr = 0,
  num_burnin = 1000,
  num_mcmc = 500,
  general_params = list(adaptive_coding = FALSE, num_chains = 4, keep_every = 5),
  treatment_effect_forest_params = list(
    num_trees = 50,
    sample_intercept = TRUE,
    sigma2_leaf_init = 0.25 / 50,
    tau_0_prior_var = 1
  )
)

# --- Extract components ---
# tau(X) forest-only predictions: n_test x num_samples
tau_x_posterior <- predict(
  bcf_model, X = X_test, Z = Z_test, propensity = pi_test,
  type = "posterior", terms = "tau"
)

# CATE = tau_0 + tau(X): n_test x num_samples
cate_posterior <- predict(
  bcf_model, X = X_test, Z = Z_test, propensity = pi_test,
  type = "posterior", terms = "cate"
)

# tau_0 samples (stored as p_tau0 x num_samples matrix, p_tau0 = 1 here)
tau_0_samples <- extractParameter(bcf_model, "tau_0")

cat("--- Dimensions ---\n")
cat("tau_x_posterior:", paste(dim(tau_x_posterior), collapse = " x "), "\n")
cat("cate_posterior: ", paste(dim(cate_posterior),  collapse = " x "), "\n")
cat("tau_0_samples:  ", paste(dim(tau_0_samples),   collapse = " x "), "\n")

# --- ATE posteriors ---
# ATE via colMeans of CATE
ate_via_cate <- colMeans(cate_posterior)

# ATE via colMeans(tau(X)) + tau_0
ate_via_parts <- colMeans(tau_x_posterior) + as.numeric(tau_0_samples)

cat("\n--- First 10 sample-level comparison: colMeans(tau) + tau_0  vs  colMeans(cate) ---\n")
comparison <- data.frame(
  tau_x_mean    = colMeans(tau_x_posterior)[1:10],
  tau_0         = as.numeric(tau_0_samples)[1:10],
  sum_parts     = ate_via_parts[1:10],
  cate_mean     = ate_via_cate[1:10],
  diff          = (ate_via_parts - ate_via_cate)[1:10]
)
print(round(comparison, 6))

cat("\n--- Max absolute difference across all samples ---\n")
cat("max|colMeans(tau) + tau_0 - colMeans(cate)|:", max(abs(ate_via_parts - ate_via_cate)), "\n")

# --- Observation-level check: tau_x[i,s] + tau_0[s] vs cate[i,s] ---
# Reconstruct expected CATE from parts
cate_reconstructed <- sweep(tau_x_posterior, 2, as.numeric(tau_0_samples), "+")
cat("\n--- Max absolute difference (observation-level): tau_x + tau_0 vs cate ---\n")
cat("max|tau_x[i,s] + tau_0[s] - cate[i,s]|:", max(abs(cate_reconstructed - cate_posterior)), "\n")

# --- Scale checks ---
cat("\n--- Scale diagnostics ---\n")
cat("outcome_scale (y_std):", bcf_model$model_params$outcome_scale, "\n")
cat("outcome_mean  (y_bar):", bcf_model$model_params$outcome_mean, "\n")
cat("mean(tau_x_posterior):", mean(tau_x_posterior), "\n")
cat("mean(tau_0_samples):  ", mean(tau_0_samples), "\n")
cat("mean(cate_posterior): ", mean(cate_posterior), "\n")
cat("true ATE:             ", tau_x, "\n")

# --- Posterior summaries ---
cat("\n--- Posterior mean of ATE (via CATE) ---\n")
cat("mean:", mean(ate_via_cate), " 95% CI: [",
    quantile(ate_via_cate, 0.025), ",", quantile(ate_via_cate, 0.975), "]\n")

cat("\n--- Posterior of tau_0 alone ---\n")
cat("mean:", mean(tau_0_samples), " 95% CI: [",
    quantile(tau_0_samples, 0.025), ",", quantile(tau_0_samples, 0.975), "]\n")

cat("\n--- Posterior of colMeans(tau(X)) alone ---\n")
tau_x_test_mean <- colMeans(tau_x_posterior)
cat("mean:", mean(tau_x_test_mean), " 95% CI: [",
    quantile(tau_x_test_mean, 0.025), ",", quantile(tau_x_test_mean, 0.975), "]\n")
