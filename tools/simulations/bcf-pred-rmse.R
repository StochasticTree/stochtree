# Load library
library(stochtree)

# Simulation parameters
n <- 500
p <- 5
n_sim <- 100
test_set_pct <- 0.2

# Simulation container
rmses_cached <- rep(NA_real_, n_sim)
rmses_pred <- rep(NA_real_, n_sim)

# Run the simulation
for (i in 1:n_sim) {
  # Generate data
  X <- matrix(rnorm(n * p), ncol = p)
  mu_x <- X[, 1]
  tau_x <- 0.25 * X[, 2]
  pi_x <- pnorm(0.5 * X[, 1])
  Z <- rbinom(n, 1, pi_x)
  E_XZ <- mu_x + Z * tau_x
  snr <- 2
  y <- E_XZ + rnorm(n, 0, 1) * (sd(E_XZ) / snr)

  # Train-test split
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
  E_XZ_test <- E_XZ[test_inds]
  E_XZ_train <- E_XZ[train_inds]

  # Fit a simple BCF model
  bcf_model <- bcf(
    X_train = X_train,
    Z_train = Z_train,
    y_train = y_train,
    propensity_train = pi_train,
    X_test = X_test,
    Z_test = Z_test,
    propensity_test = pi_test
  )

  # Predict out of sample
  y_hat_test <- predict(
    bcf_model,
    X = X_test,
    Z = Z_test,
    propensity = pi_test,
    type = "mean",
    terms = "y_hat"
  )

  # Compute RMSE using both cached predictions and those returned by predict()
  rmses_cached[i] <- sqrt(mean((rowMeans(bcf_model$y_hat_test) - E_XZ_test)^2))
  rmses_pred[i] <- sqrt(mean((y_hat_test - E_XZ_test)^2))
}

# Inspect results
mean(rmses_cached)
mean(rmses_pred)
