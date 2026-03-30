# Load package
library(stochtree)

# Helper functions
g <- function(x) {
  ifelse(x[, 5] == 1, 2, ifelse(x[, 5] == 2, -1, -4))
}
mu1 <- function(x) {
  1 + g(x) + x[, 1] * x[, 3]
}
mu2 <- function(x) {
  1 + g(x) + 6 * abs(x[, 3] - 1)
}
tau1 <- function(x) {
  rep(3, nrow(x))
}
tau2 <- function(x) {
  1 + 2 * x[, 2] * x[, 4]
}

# Generate data
n <- 500
snr <- 3
x1 <- rnorm(n)
x2 <- rnorm(n)
x3 <- rnorm(n)
x4 <- as.numeric(rbinom(n, 1, 0.5))
x5 <- as.numeric(sample(1:3, n, replace = TRUE))
X <- cbind(x1, x2, x3, x4, x5)
p <- ncol(X)
mu_x <- mu1(X)
tau_x <- tau2(X)
pi_x <- 0.8 * pnorm((3 * mu_x / sd(mu_x)) - 0.5 * X[, 1]) + 0.05 + runif(n) / 10
Z <- rbinom(n, 1, pi_x)
E_XZ <- mu_x + Z * tau_x
y <- E_XZ + rnorm(n, 0, 1) * (sd(E_XZ) / snr)
X <- as.data.frame(X)
X$x4 <- factor(X$x4, ordered = TRUE)
X$x5 <- factor(X$x5, ordered = TRUE)

# Split data into test and train sets
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

# Sample the model
general_params <- list(num_threads = 1, num_chains = 4)
bcf_model <- bcf(
  X_train = X_train,
  Z_train = Z_train,
  y_train = y_train,
  propensity_train = pi_train,
  X_test = X_test,
  Z_test = Z_test,
  propensity_test = pi_test,
  num_gfr = 10,
  num_burnin = 1000,
  num_mcmc = 100,
  general_params = general_params
)

# Plot true versus estimated prognostic function
mu_hat_test <- predict(
  bcf_model,
  X = X_test,
  Z = Z_test,
  propensity = pi_test,
  terms = "prognostic_function"
)
plot(
  rowMeans(mu_hat_test),
  mu_test,
  xlab = "predicted",
  ylab = "actual",
  main = "Prognostic function"
)
abline(0, 1, col = "red", lty = 3, lwd = 3)

# Plot true versus estimated CATE function
tau_hat_test <- predict(
  bcf_model,
  X = X_test,
  Z = Z_test,
  propensity = pi_test,
  terms = "cate"
)
plot(
  rowMeans(tau_hat_test),
  tau_test,
  xlab = "predicted",
  ylab = "actual",
  main = "Treatment effect"
)
abline(0, 1, col = "red", lty = 3, lwd = 3)
sqrt(mean((rowMeans(tau_hat_test) - tau_test)^2))
cor(rowMeans(tau_hat_test), tau_test)

# Inspect sigma^2 traceplot
sigma_observed <- var(y - E_XZ)
sigma2_global_samples <- extractParameter(bcf_model, "sigma2_global")
plot_bounds <- c(
  min(c(sigma2_global_samples, sigma_observed)),
  max(c(sigma2_global_samples, sigma_observed))
)
plot(
  sigma2_global_samples,
  ylim = plot_bounds,
  ylab = "sigma^2",
  xlab = "Sample",
  main = "Global variance parameter"
)
abline(h = sigma_observed, lty = 3, lwd = 3, col = "blue")

# Assess CATE function coverage
test_lb <- apply(tau_hat_test, 1, quantile, 0.025)
test_ub <- apply(tau_hat_test, 1, quantile, 0.975)
cover <- ((test_lb <= tau_x[test_inds]) &
  (test_ub >= tau_x[test_inds]))
mean(cover)
