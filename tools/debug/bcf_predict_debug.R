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
y_hat_test <- predict(
  bcf_model,
  X = X_test,
  Z = Z_test,
  propensity = pi_test,
  type = "mean",
  terms = c("rfx", "variance")
)

y_hat_intervals <- compute_bcf_posterior_interval(
  model_object = bcf_model,
  scale = "linear",
  terms = c("all"),
  covariates = X_test,
  treatment = Z_test,
  propensity = pi_test,
  level = 0.95
)

(tau_coverage <- mean(
  (y_hat_intervals$tau_hat$upper >= tau_test) &
    (y_hat_intervals$tau_hat$lower <= tau_test)
))

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
  covariates = X_test,
  treatment = Z_test,
  propensity = pi_test,
  level = 0.95
)

# Compute intervals on probability scale
y_hat_prob_intervals <- compute_bcf_posterior_interval(
  model_object = bcf_model,
  scale = "probability",
  terms = c("y_hat"),
  covariates = X_test,
  treatment = Z_test,
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
  covariates = X_test,
  treatment = Z_test,
  propensity = pi_test,
  num_draws = 10
)

# Compute histogram of PPD probabilities for both outcome classes
ppd_samples_prob <- apply(ppd_samples, 1, mean)
ppd_outcome_0 <- ppd_samples_prob[y_test == 0]
ppd_outcome_1 <- ppd_samples_prob[y_test == 1]
hist(ppd_outcome_0, breaks = 50, xlim = c(0, 1))
hist(ppd_outcome_1, breaks = 50, xlim = c(0, 1))
