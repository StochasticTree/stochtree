# Demo of updated predict method for BCF

# Load library
library(stochtree)

# Generate data
n <- 500
p <- 5
X <- matrix(rnorm(n * p), ncol = p)
mu_x <- X[, 1]
tau_x <- 0.25 * X[, 2]
pi_x <- pnorm(0.5 * X[, 1])
Z <- rbinom(n, 1, pi_x)
E_XZ <- mu_x + Z * tau_x
snr <- 2
y <- E_XZ + rnorm(n, 0, 1) * (sd(E_XZ) / snr)

# Train-test split
test_set_pct <- 0.2
n_test <- round(test_set_pct * n)
n_train <- n - n_test
test_inds <- sort(sample(1:n, n_test, replace = FALSE))
train_inds <- (1:n)[!((1:n) %in% test_inds)]
X_test <- X[test_inds, ]
X_train <- X[train_inds, ]
pi_test <- pi_x[test_inds]
pi_train <- pi_x[train_inds]
Z_test <- Z[test_inds]
Z_train <- Z[train_inds]
y_test <- y[test_inds]
y_train <- y[train_inds]
mu_test <- mu_x[test_inds]
mu_train <- mu_x[train_inds]
tau_test <- tau_x[test_inds]
tau_train <- tau_x[train_inds]

# Fit a simple BCF model
bcf_model <- bcf(
  X_train = X_train,
  Z_train = Z_train,
  y_train = y_train,
  propensity_train = pi_train,
  X_test = X_test,
  Z_test = Z_test,
  propensity_test = pi_test,
  num_gfr = 10,
  num_burnin = 0,
  num_mcmc = 1000
)

# Check several predict approaches
y_hat_posterior_test <- predict(
  bcf_model,
  X = X_test,
  Z = Z_test,
  propensity = pi_test
)$y_hat
pred <- predict(
  bcf_model,
  X = X_test,
  Z = Z_test,
  propensity = pi_test,
  type = "mean",
  terms = c("all")
)
# Check that this throws a warning
y_hat_test <- predict(
  bcf_model,
  X = X_test,
  Z = Z_test,
  propensity = pi_test,
  type = "mean",
  terms = c("rfx", "variance")
)

# Compute intervals around model terms (including E[y | X, Z])
y_hat_intervals <- compute_bcf_posterior_interval(
  model_object = bcf_model,
  scale = "linear",
  terms = c("all"),
  X = X_test,
  Z = Z_test,
  propensity = pi_test,
  level = 0.95
)

# Estimate coverage of intervals on tau(X)
(tau_coverage <- mean(
  (y_hat_intervals$tau_hat$upper >= tau_test) &
    (y_hat_intervals$tau_hat$lower <= tau_test)
))

# Posterior predictive coverage and MSE checks
quantiles <- c(0.05, 0.95)
ppd_samples <- sample_bcf_posterior_predictive(
  model_object = bcf_model,
  X = X_test,
  Z = Z_test,
  propensity = pi_test,
  num_draws = 1
)
yhat_ppd <- apply(ppd_samples, 1, mean)
yhat_interval_ppd <- apply(ppd_samples, 1, quantile, probs = quantiles)
mean((yhat_interval_ppd[1, ] <= y_test) & (yhat_interval_ppd[2, ] >= y_test))
sqrt(mean((yhat_ppd - y_test)^2))

# Generate probit outcome data
n <- 1000
p <- 5
X <- matrix(rnorm(n * p), ncol = p)
mu_x <- X[, 1]
tau_x <- 0.25 * X[, 2]
pi_x <- pnorm(0.5 * X[, 1])
Z <- rbinom(n, 1, pi_x)
E_XZ <- mu_x + Z * tau_x
W <- E_XZ + rnorm(n, 0, 1)
y <- (W > 0) * 1

# Train-test split
test_set_pct <- 0.2
n_test <- round(test_set_pct * n)
n_train <- n - n_test
test_inds <- sort(sample(1:n, n_test, replace = FALSE))
train_inds <- (1:n)[!((1:n) %in% test_inds)]
X_test <- X[test_inds, ]
X_train <- X[train_inds, ]
pi_test <- pi_x[test_inds]
pi_train <- pi_x[train_inds]
Z_test <- Z[test_inds]
Z_train <- Z[train_inds]
W_test <- W[test_inds]
W_train <- W[train_inds]
mu_test <- mu_x[test_inds]
mu_train <- mu_x[train_inds]
tau_test <- tau_x[test_inds]
tau_train <- tau_x[train_inds]
y_test <- y[test_inds]
y_train <- y[train_inds]

# Fit a simple BCF model
bcf_model <- bcf(
  X_train = X_train,
  Z_train = Z_train,
  y_train = y_train,
  propensity_train = pi_train,
  X_test = X_test,
  Z_test = Z_test,
  propensity_test = pi_test,
  num_gfr = 10,
  num_burnin = 0,
  num_mcmc = 1000,
  general_params = list(probit_outcome_model = TRUE)
)

# Predict on latent scale
y_hat_post <- predict(
  object = bcf_model,
  type = "posterior",
  terms = c("y_hat"),
  X = X_test,
  Z = Z_test,
  propensity = pi_test,
  scale = "linear"
)

# Predict on probability scale
y_hat_post_prob <- predict(
  object = bcf_model,
  type = "posterior",
  terms = c("y_hat"),
  X = X_test,
  Z = Z_test,
  propensity = pi_test,
  scale = "probability"
)

# Compute intervals on latent scale
y_hat_intervals <- compute_bcf_posterior_interval(
  model_object = bcf_model,
  scale = "linear",
  terms = c("y_hat"),
  X = X_test,
  Z = Z_test,
  propensity = pi_test,
  level = 0.95
)

# Compute intervals on probability scale
y_hat_prob_intervals <- compute_bcf_posterior_interval(
  model_object = bcf_model,
  scale = "probability",
  terms = c("y_hat"),
  X = X_test,
  Z = Z_test,
  propensity = pi_test,
  level = 0.95
)

# Compute posterior means
y_hat_mean_latent <- rowMeans(y_hat_post)
y_hat_mean_prob <- rowMeans(y_hat_post_prob)

# Plot on latent scale
sort_inds <- order(y_hat_mean_latent)
plot(y_hat_mean_latent[sort_inds], ylim = range(y_hat_intervals))
lines(y_hat_intervals$lower[sort_inds])
lines(y_hat_intervals$upper[sort_inds])

# Plot on probability scale
sort_inds <- order(y_hat_mean_prob)
plot(y_hat_mean_prob[sort_inds], ylim = range(y_hat_prob_intervals))
lines(y_hat_prob_intervals$lower[sort_inds])
lines(y_hat_prob_intervals$upper[sort_inds])

# Draw from posterior predictive for covariates / treatment values in the test set
ppd_samples <- sample_bcf_posterior_predictive(
  model_object = bcf_model,
  X = X_test,
  Z = Z_test,
  propensity = pi_test,
  num_draws = 10
)

# Compute histogram of PPD probabilities for both outcome classes
ppd_samples_prob <- apply(ppd_samples, 1, mean)
ppd_outcome_0 <- ppd_samples_prob[y_test == 0]
ppd_outcome_1 <- ppd_samples_prob[y_test == 1]
hist(ppd_outcome_0, breaks = 50, xlim = c(0, 1))
hist(ppd_outcome_1, breaks = 50, xlim = c(0, 1))

# Generate data with random effects
X <- matrix(rnorm(n * p), ncol = p)
mu_x <- X[, 1]
tau_x <- 0.25 * X[, 2]
pi_x <- pnorm(0.5 * X[, 1])
Z <- rbinom(n, 1, pi_x)
rfx_group_ids <- sample(1:3, n, replace = TRUE)
rfx_basis <- cbind(1, Z)
rfx_coefs <- matrix(c(-2, -0.5, 0, 0, 2, 0.5), byrow = T, nrow = 3)
rfx_term <- rowSums(rfx_basis * rfx_coefs[rfx_group_ids, ])
E_XZ <- mu_x + Z * tau_x + rfx_term
snr <- 2
y <- E_XZ + rnorm(n, 0, 1) * (sd(E_XZ) / snr)

# Train-test split
test_set_pct <- 0.2
n_test <- round(test_set_pct * n)
n_train <- n - n_test
test_inds <- sort(sample(1:n, n_test, replace = FALSE))
train_inds <- (1:n)[!((1:n) %in% test_inds)]
X_test <- X[test_inds, ]
X_train <- X[train_inds, ]
pi_test <- pi_x[test_inds]
pi_train <- pi_x[train_inds]
Z_test <- Z[test_inds]
Z_train <- Z[train_inds]
y_test <- y[test_inds]
y_train <- y[train_inds]
mu_test <- mu_x[test_inds]
mu_train <- mu_x[train_inds]
tau_test <- tau_x[test_inds]
tau_train <- tau_x[train_inds]
rfx_group_ids_test <- rfx_group_ids[test_inds]
rfx_group_ids_train <- rfx_group_ids[train_inds]

# Fit a simple BCF model
rfx_params = list(
  model_spec = "intercept_plus_treatment"
)
bcf_model <- bcf(
  X_train = X_train,
  Z_train = Z_train,
  y_train = y_train,
  propensity_train = pi_train,
  rfx_group_ids_train = rfx_group_ids_train,
  X_test = X_test,
  Z_test = Z_test,
  propensity_test = pi_test,
  rfx_group_ids_test = rfx_group_ids_test,
  num_gfr = 10,
  num_burnin = 0,
  num_mcmc = 1000,
  random_effects_params = rfx_params
)

# Retrieve all predictions
posterior_preds_test <- predict(
  bcf_model,
  X = X_test,
  Z = Z_test,
  propensity = pi_test,
  rfx_group_ids = rfx_group_ids_test
)

# Check that mu + tau + rfx = prognostic + cate
comp_mat <- (abs(
  (posterior_preds_test$mu_hat +
    posterior_preds_test$tau_hat * Z_test +
    posterior_preds_test$rfx_predictions) -
    (posterior_preds_test$prognostic_function +
      posterior_preds_test$cate * Z_test)
) <
  0.0001)
all(comp_mat)

# Retrieve just prognostic predictions
prog_fn_test <- predict(
  bcf_model,
  X = X_test,
  Z = Z_test,
  propensity = pi_test,
  rfx_group_ids = rfx_group_ids_test,
  terms = c("prognostic_function")
)

# Compare to prognostic function returned from the larger prediction
all(abs(prog_fn_test - posterior_preds_test$prognostic_function) < 0.0001)

# Retrieve just mu predictions
mu_hat_test <- predict(
  bcf_model,
  X = X_test,
  Z = Z_test,
  propensity = pi_test,
  rfx_group_ids = rfx_group_ids_test,
  terms = c("mu")
)

# Compare to prognostic function returned from the larger prediction
all(abs(mu_hat_test - posterior_preds_test$mu_hat) < 0.0001)

# Retrieve just CATE predictions
cate_fn_test <- predict(
  bcf_model,
  X = X_test,
  Z = Z_test,
  propensity = pi_test,
  rfx_group_ids = rfx_group_ids_test,
  terms = c("cate")
)

# Compare to prognostic function returned from the larger prediction
all(abs(cate_fn_test - posterior_preds_test$cate) < 0.0001)

# Retrieve just mu predictions
tau_hat_test <- predict(
  bcf_model,
  X = X_test,
  Z = Z_test,
  propensity = pi_test,
  rfx_group_ids = rfx_group_ids_test,
  terms = c("tau")
)

# Compare to prognostic function returned from the larger prediction
all(abs(tau_hat_test - posterior_preds_test$tau_hat) < 0.0001)

# Compute intervals for all of the model terms
posterior_intervals_test <- compute_bcf_posterior_interval(
  model_object = bcf_model,
  scale = "linear",
  terms = "all",
  X = X_test,
  Z = Z_test,
  propensity = pi_test,
  rfx_group_ids = rfx_group_ids_test,
  level = 0.95
)

# Compute intervals for just the prognostic term
prog_intervals_test <- compute_bcf_posterior_interval(
  model_object = bcf_model,
  scale = "linear",
  terms = "prognostic_function",
  X = X_test,
  Z = Z_test,
  propensity = pi_test,
  rfx_group_ids = rfx_group_ids_test,
  level = 0.95
)

# Compute intervals for just the CATE term
cate_intervals_test <- compute_bcf_posterior_interval(
  model_object = bcf_model,
  scale = "linear",
  terms = "cate",
  X = X_test,
  Z = Z_test,
  propensity = pi_test,
  rfx_group_ids = rfx_group_ids_test,
  level = 0.95
)

# Check that they match the corresponding terms from the full interval list
all(
  abs(
    posterior_intervals_test$prognostic_function$lower -
      prog_intervals_test$lower
  ) <
    0.0001
)
all(
  abs(
    posterior_intervals_test$prognostic_function$upper -
      prog_intervals_test$upper
  ) <
    0.0001
)
all(
  abs(
    posterior_intervals_test$cate$lower -
      cate_intervals_test$lower
  ) <
    0.0001
)
all(
  abs(
    posterior_intervals_test$cate$upper -
      cate_intervals_test$upper
  ) <
    0.0001
)

# Check that the prog and CATE intervals are different from the mu and tau intervals
mu_intervals_test <- compute_bcf_posterior_interval(
  model_object = bcf_model,
  scale = "linear",
  terms = "mu",
  X = X_test,
  Z = Z_test,
  propensity = pi_test,
  rfx_group_ids = rfx_group_ids_test,
  level = 0.95
)
tau_intervals_test <- compute_bcf_posterior_interval(
  model_object = bcf_model,
  scale = "linear",
  terms = "tau",
  X = X_test,
  Z = Z_test,
  propensity = pi_test,
  rfx_group_ids = rfx_group_ids_test,
  level = 0.95
)
all(
  abs(
    mu_intervals_test$lower -
      prog_intervals_test$lower
  ) >
    0.0001
)
all(
  abs(
    mu_intervals_test$upper -
      prog_intervals_test$upper
  ) >
    0.0001
)
all(
  abs(
    tau_intervals_test$lower -
      cate_intervals_test$lower
  ) >
    0.0001
)
all(
  abs(
    tau_intervals_test$upper -
      cate_intervals_test$upper
  ) >
    0.0001
)
