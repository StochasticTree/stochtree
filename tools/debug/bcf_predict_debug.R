# Demo of updated predict method for BCF

# Load library
library(stochtree)

# Generate data
n <- 100
g <- function(x) {
  ifelse(x[, 5] == 1, 2, ifelse(x[, 5] == 2, -1, -4))
}
x1 <- rnorm(n)
x2 <- rnorm(n)
x3 <- rnorm(n)
x4 <- as.numeric(rbinom(n, 1, 0.5))
x5 <- as.numeric(sample(1:3, n, replace = TRUE))
X <- cbind(x1, x2, x3, x4, x5)
p <- ncol(X)
mu_x <- 1 + g(X) + X[, 1] * X[, 3]
tau_x <- 1 + 2 * X[, 2] * X[, 4]
pi_x <- 0.8 * pnorm((3 * mu_x / sd(mu_x)) - 0.5 * X[, 1]) + 0.05 + runif(n) / 10
Z <- rbinom(n, 1, pi_x)
E_XZ <- mu_x + Z * tau_x
snr <- 2
y <- E_XZ + rnorm(n, 0, 1) * (sd(E_XZ) / snr)
X <- as.data.frame(X)
X$x4 <- factor(X$x4, ordered = TRUE)
X$x5 <- factor(X$x5, ordered = TRUE)

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
  num_mcmc = 10
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

