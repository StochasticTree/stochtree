# Reproducing issues in GFR when the covariates have a large number of ties
library(stochtree)

# Generate covariates that are essentially categorical, but real-valued
n <- 1000
p <- 10
X <- matrix(NA_real_, n, p)
for (i in 1:p) {
  coef_vector <- runif(15, -5, 5)
  X[, i] <- sample(coef_vector, n, replace = TRUE)
}
f_X <- 5 * X[, 1] - 2 * X[, 2] + X[, 3]
eps <- rnorm(n, 0, 1)
y <- f_X + eps

# Train test split
test_set_pct <- 0.2
n_test <- round(test_set_pct * n)
n_train <- n - n_test
test_inds <- sort(sample(1:n, n_test, replace = FALSE))
train_inds <- (1:n)[!((1:n) %in% test_inds)]
X_test <- X[test_inds, ]
X_train <- X[train_inds, ]
y_test <- y[test_inds]
y_train <- y[train_inds]
E_y_test <- f_X[test_inds]
E_y_train <- f_X[train_inds]

# Attempt to fit a GFR-only predictive model
xbart_model <- bart(
  X_train = X_train,
  y_train = y_train,
  num_gfr = 10,
  num_burnin = 0,
  num_mcmc = 0
)

# Inspect the model fit
y_hat_test <- predict(
  xbart_model,
  covariates = X_test,
  type = "mean",
  terms = "y_hat"
)
plot(y_hat_test, y_test)
abline(0, 1)

# Not great! Let's add a few MCMC samples and see how it shakes out
bart_model <- bart(
  X_train = X_train,
  y_train = y_train,
  num_gfr = 10,
  num_burnin = 0,
  num_mcmc = 10
)

# Inspect the model fit
y_hat_test <- predict(
  bart_model,
  covariates = X_test,
  type = "mean",
  terms = "y_hat"
)
plot(y_hat_test, y_test)
abline(0, 1)

# Generate covariates with zero-inflation
X <- matrix(NA_real_, n, p)
for (i in 1:p) {
  p_zero <- runif(1, 0.1, 0.9)
  zero_vector <- rbinom(n, 1, p_zero)
  runif_vector <- runif(n)
  X[, i] <- zero_vector * 0 + (1 - zero_vector) * runif_vector
}
f_X <- 5 * X[, 1] - 2 * X[, 2] + X[, 3]
eps <- rnorm(n, 0, 1)
y <- f_X + eps

# Train test split
n_test <- round(test_set_pct * n)
n_train <- n - n_test
test_inds <- sort(sample(1:n, n_test, replace = FALSE))
train_inds <- (1:n)[!((1:n) %in% test_inds)]
X_test <- X[test_inds, ]
X_train <- X[train_inds, ]
y_test <- y[test_inds]
y_train <- y[train_inds]
E_y_test <- f_X[test_inds]
E_y_train <- f_X[train_inds]

# Attempt to fit a GFR-only predictive model
xbart_model <- bart(
  X_train = X_train,
  y_train = y_train,
  num_gfr = 10,
  num_burnin = 0,
  num_mcmc = 0
)

# Inspect the model fit
y_hat_test <- predict(
  xbart_model,
  covariates = X_test,
  type = "mean",
  terms = "y_hat"
)
plot(y_hat_test, y_test)
abline(0, 1)

# Also not great! Let's add a few MCMC samples and see how it shakes out
bart_model <- bart(
  X_train = X_train,
  y_train = y_train,
  num_gfr = 10,
  num_burnin = 0,
  num_mcmc = 10
)

# Inspect the model fit
y_hat_test <- predict(
  bart_model,
  covariates = X_test,
  type = "mean",
  terms = "y_hat"
)
plot(y_hat_test, y_test)
abline(0, 1)

# Perform the same checks as above for BCF

# Generate covariates that are essentially categorical, but real-valued
X <- matrix(NA_real_, n, p)
for (i in 1:p) {
  coef_vector <- runif(15, -5, 5)
  X[, i] <- sample(coef_vector, n, replace = TRUE)
}
mu_x <- 5 * X[, 1] - 2 * X[, 2] + X[, 3]
pi_x <- pnorm(0.1 * mu_x)
tau_x <- X[, 4]
Z <- rbinom(n, 1, pi_x)
E_Y_XZ <- mu_x + tau_x * (Z)
eps <- rnorm(n, 0, 1)
y <- E_Y_XZ + eps

# Train test split
n_test <- round(test_set_pct * n)
n_train <- n - n_test
test_inds <- sort(sample(1:n, n_test, replace = FALSE))
train_inds <- (1:n)[!((1:n) %in% test_inds)]
X_test <- X[test_inds, ]
X_train <- X[train_inds, ]
Z_test <- Z[test_inds]
Z_train <- Z[train_inds]
propensity_test <- pi_x[test_inds]
propensity_train <- pi_x[train_inds]
y_test <- y[test_inds]
y_train <- y[train_inds]

# Attempt to fit a GFR-only BCF model
xbcf_model <- bcf(
  X_train = X_train,
  Z_train = Z_train,
  propensity_train = propensity_train,
  y_train = y_train,
  num_gfr = 10,
  num_burnin = 0,
  num_mcmc = 0
)

# Inspect the model fit
y_hat_test <- predict(
  xbcf_model,
  X = X_test,
  Z = Z_test,
  propensity = propensity_test,
  type = "mean",
  terms = "y_hat"
)
plot(y_hat_test, y_test)
abline(0, 1)

# Not great! Let's add a few MCMC samples and see how it shakes out
bcf_model <- bcf(
  X_train = X_train,
  Z_train = Z_train,
  propensity_train = propensity_train,
  y_train = y_train,
  num_gfr = 10,
  num_burnin = 0,
  num_mcmc = 10
)

# Inspect the model fit
y_hat_test <- predict(
  bcf_model,
  X = X_test,
  Z = Z_test,
  propensity = propensity_test,
  type = "mean",
  terms = "y_hat"
)
plot(y_hat_test, y_test)
abline(0, 1)

# Generate covariates that are zero-inflated
X <- matrix(NA_real_, n, p)
for (i in 1:p) {
  p_zero <- runif(1, 0.1, 0.9)
  zero_vector <- rbinom(n, 1, p_zero)
  runif_vector <- runif(n)
  X[, i] <- zero_vector * 0 + (1 - zero_vector) * runif_vector
}
mu_x <- 5 * X[, 1] - 2 * X[, 2] + X[, 3]
pi_x <- pnorm(0.1 * mu_x)
tau_x <- X[, 4]
Z <- rbinom(n, 1, pi_x)
E_Y_XZ <- mu_x + tau_x * (Z)
eps <- rnorm(n, 0, 1)
y <- E_Y_XZ + eps

# Train test split
n_test <- round(test_set_pct * n)
n_train <- n - n_test
test_inds <- sort(sample(1:n, n_test, replace = FALSE))
train_inds <- (1:n)[!((1:n) %in% test_inds)]
X_test <- X[test_inds, ]
X_train <- X[train_inds, ]
Z_test <- Z[test_inds]
Z_train <- Z[train_inds]
propensity_test <- pi_x[test_inds]
propensity_train <- pi_x[train_inds]
y_test <- y[test_inds]
y_train <- y[train_inds]

# Attempt to fit a GFR-only BCF model
xbcf_model <- bcf(
  X_train = X_train,
  Z_train = Z_train,
  propensity_train = propensity_train,
  y_train = y_train,
  num_gfr = 10,
  num_burnin = 0,
  num_mcmc = 0
)

# Inspect the model fit
y_hat_test <- predict(
  xbcf_model,
  X = X_test,
  Z = Z_test,
  propensity = propensity_test,
  type = "mean",
  terms = "y_hat"
)
plot(y_hat_test, y_test)
abline(0, 1)

# Not great! Let's add a few MCMC samples and see how it shakes out
bcf_model <- bcf(
  X_train = X_train,
  Z_train = Z_train,
  propensity_train = propensity_train,
  y_train = y_train,
  num_gfr = 10,
  num_burnin = 0,
  num_mcmc = 10
)

# Inspect the model fit
y_hat_test <- predict(
  bcf_model,
  X = X_test,
  Z = Z_test,
  propensity = propensity_test,
  type = "mean",
  terms = "y_hat"
)
plot(y_hat_test, y_test)
abline(0, 1)
