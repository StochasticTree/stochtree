# Demo of CATE computation function for BCF

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

# Fit BCF model
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

# Compute CATE posterior
tau_hat_posterior_test <- compute_contrast_bcf_model(
  bcf_model,
  X_0 = X_test,
  X_1 = X_test,
  Z_0 = rep(0, n_test),
  Z_1 = rep(1, n_test),
  propensity_0 = pi_test,
  propensity_1 = pi_test,
  type = "posterior",
  scale = "linear"
)

# Compute the same quantity via predict
tau_hat_posterior_test_comparison <- predict(
  bcf_model,
  X = X_test,
  Z = Z_test,
  propensity = pi_test,
  type = "posterior",
  terms = "cate",
  scale = "linear"
)

# Compare results
tau_diff <- tau_hat_posterior_test_comparison - tau_hat_posterior_test
all(
  abs(tau_diff) < 0.001
)

# Generate data for a BCF model with random effects
X <- matrix(rnorm(n * p), ncol = p)
mu_x <- X[, 1]
tau_x <- 0.25 * X[, 2]
pi_x <- pnorm(0.5 * X[, 1])
Z <- rbinom(n, 1, pi_x)
group_ids <- rep(c(1, 2), n %/% 2)
rfx_coefs <- matrix(c(-1, -1, 1, 1), nrow = 2, byrow = TRUE)
rfx_basis <- cbind(1, Z)
rfx_term <- rowSums(rfx_coefs[group_ids, ] * rfx_basis)
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
group_ids_test <- group_ids[test_inds]
group_ids_train <- group_ids[train_inds]
rfx_basis_test <- rfx_basis[test_inds, ]
rfx_basis_train <- rfx_basis[train_inds, ]

# Fit BCF model
bcf_model <- bcf(
  X_train = X_train,
  Z_train = Z_train,
  y_train = y_train,
  propensity_train = pi_train,
  rfx_group_ids_train = group_ids_train,
  rfx_basis_train = rfx_basis_train,
  X_test = X_test,
  Z_test = Z_test,
  propensity_test = pi_test,
  rfx_group_ids_test = group_ids_test,
  rfx_basis_test = rfx_basis_test,
  num_gfr = 10,
  num_burnin = 0,
  num_mcmc = 1000
)

# Compute CATE posterior
tau_hat_posterior_test <- compute_contrast_bcf_model(
  bcf_model,
  X_0 = X_test,
  X_1 = X_test,
  Z_0 = rep(0, n_test),
  Z_1 = rep(1, n_test),
  propensity_0 = pi_test,
  propensity_1 = pi_test,
  rfx_group_ids_0 = group_ids_test,
  rfx_group_ids_1 = group_ids_test,
  rfx_basis_0 = cbind(1, rep(0, n_test)),
  rfx_basis_1 = cbind(1, rep(1, n_test)),
  type = "posterior",
  scale = "linear"
)

# Compute the same quantity via predict
tau_forest_posterior_test <- predict(
  bcf_model,
  X = X_test,
  Z = Z_test,
  propensity = pi_test,
  rfx_group_ids = group_ids_test,
  rfx_basis = rfx_basis_test,
  type = "posterior",
  terms = "cate",
  scale = "linear"
)
rfx_term_posterior_treated <- predict(
  bcf_model,
  X = X_test,
  Z = Z_test,
  propensity = pi_test,
  rfx_group_ids = group_ids_test,
  rfx_basis = cbind(1, rep(1, n_test)),
  type = "posterior",
  terms = "rfx",
  scale = "linear"
)
rfx_term_posterior_control <- predict(
  bcf_model,
  X = X_test,
  Z = Z_test,
  propensity = pi_test,
  rfx_group_ids = group_ids_test,
  rfx_basis = cbind(1, rep(0, n_test)),
  type = "posterior",
  terms = "rfx",
  scale = "linear"
)
tau_hat_posterior_test_comparison <- (tau_forest_posterior_test +
  (rfx_term_posterior_treated - rfx_term_posterior_control))

# Compare results
tau_diff <- tau_hat_posterior_test_comparison - tau_hat_posterior_test
all(
  abs(tau_diff) < 0.001
)

# Generate data for a probit BCF model with random effects
X <- matrix(rnorm(n * p), ncol = p)
mu_x <- X[, 1]
tau_x <- 0.25 * X[, 2]
pi_x <- pnorm(0.5 * X[, 1])
Z <- rbinom(n, 1, pi_x)
group_ids <- rep(c(1, 2), n %/% 2)
rfx_coefs <- matrix(c(-1, -1, 1, 1), nrow = 2, byrow = TRUE)
rfx_basis <- cbind(1, Z)
rfx_term <- rowSums(rfx_coefs[group_ids, ] * rfx_basis)
E_XZ <- mu_x + Z * tau_x + rfx_term
# E_XZ <- mu_x + Z * tau_x + rfx_term
W <- E_XZ + rnorm(n, 0, 1)
y <- as.numeric(W > 0)

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
y_test <- y[test_inds]
y_train <- y[train_inds]
mu_test <- mu_x[test_inds]
mu_train <- mu_x[train_inds]
tau_test <- tau_x[test_inds]
tau_train <- tau_x[train_inds]
group_ids_test <- group_ids[test_inds]
group_ids_train <- group_ids[train_inds]
rfx_basis_test <- rfx_basis[test_inds, ]
rfx_basis_train <- rfx_basis[train_inds, ]

# Fit BCF model
bcf_model <- bcf(
  X_train = X_train,
  Z_train = Z_train,
  y_train = y_train,
  propensity_train = pi_train,
  rfx_group_ids_train = group_ids_train,
  rfx_basis_train = rfx_basis_train,
  X_test = X_test,
  Z_test = Z_test,
  propensity_test = pi_test,
  rfx_group_ids_test = group_ids_test,
  rfx_basis_test = rfx_basis_test,
  num_gfr = 10,
  num_burnin = 0,
  num_mcmc = 1000,
  general_params = list(probit_outcome_model = T)
)

# Compute CATE posterior on probability scale
tau_hat_posterior_test <- compute_contrast_bcf_model(
  bcf_model,
  X_0 = X_test,
  X_1 = X_test,
  Z_0 = rep(0, n_test),
  Z_1 = rep(1, n_test),
  propensity_0 = pi_test,
  propensity_1 = pi_test,
  rfx_group_ids_0 = group_ids_test,
  rfx_group_ids_1 = group_ids_test,
  rfx_basis_0 = cbind(1, rep(0, n_test)),
  rfx_basis_1 = cbind(1, rep(1, n_test)),
  type = "posterior",
  scale = "probability"
)

# Compute the same quantity via predict
mu_forest_posterior_test <- predict(
  bcf_model,
  X = X_test,
  Z = Z_test,
  propensity = pi_test,
  rfx_group_ids = group_ids_test,
  rfx_basis = rfx_basis_test,
  type = "posterior",
  terms = "prognostic_function",
  scale = "linear"
)
tau_forest_posterior_test <- predict(
  bcf_model,
  X = X_test,
  Z = Z_test,
  propensity = pi_test,
  rfx_group_ids = group_ids_test,
  rfx_basis = rfx_basis_test,
  type = "posterior",
  terms = "cate",
  scale = "linear"
)
rfx_term_posterior_treated <- predict(
  bcf_model,
  X = X_test,
  Z = Z_test,
  propensity = pi_test,
  rfx_group_ids = group_ids_test,
  rfx_basis = cbind(1, rep(1, n_test)),
  type = "posterior",
  terms = "rfx",
  scale = "linear"
)
rfx_term_posterior_control <- predict(
  bcf_model,
  X = X_test,
  Z = Z_test,
  propensity = pi_test,
  rfx_group_ids = group_ids_test,
  rfx_basis = cbind(1, rep(0, n_test)),
  type = "posterior",
  terms = "rfx",
  scale = "linear"
)
w_hat_0 <- mu_forest_posterior_test +
  rfx_term_posterior_control
w_hat_1 <- mu_forest_posterior_test +
  tau_forest_posterior_test +
  rfx_term_posterior_treated
tau_hat_posterior_test_comparison <- pnorm(w_hat_1) - pnorm(w_hat_0)

# Compare results
tau_diff <- tau_hat_posterior_test_comparison - tau_hat_posterior_test
all(
  abs(tau_diff) < 0.001
)
