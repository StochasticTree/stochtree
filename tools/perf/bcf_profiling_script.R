# Load libraries
library(stochtree)

# Capture command line arguments
args <- commandArgs(trailingOnly = T)
if (length(args) > 0) {
  n <- as.integer(args[1])
  p <- as.integer(args[2])
  num_gfr <- as.integer(args[3])
  num_mcmc <- as.integer(args[4])
  snr <- as.numeric(args[5])
  num_threads <- as.numeric(args[6])
} else {
  # Default arguments
  n <- 1000
  p <- 5
  num_gfr <- 10
  num_mcmc <- 100
  snr <- 3.0
  num_threads <- -1
}
cat(
  "n = ",
  n,
  "\np = ",
  p,
  "\nnum_gfr = ",
  num_gfr,
  "\nnum_mcmc = ",
  num_mcmc,
  "\nsnr = ",
  snr,
  "\nnum_threads = ",
  num_threads,
  "\n",
  sep = ""
)

# Generate data needed to train BCF model
g <- function(x) {
  ifelse(x[, 5] == 1, 2, ifelse(x[, 5] == 2, -1, -4))
}
mu1 <- function(x) {
  1 + g(x) + x[, 1] * x[, 3]
}
tau1 <- function(x) {
  1 + 2 * x[, 2] * x[, 4]
}
x1 <- rnorm(n)
x2 <- rnorm(n)
x3 <- rnorm(n)
x4 <- as.numeric(rbinom(n, 1, 0.5))
x5 <- as.numeric(sample(1:3, n, replace = TRUE))
X <- cbind(x1, x2, x3, x4, x5)
p_remaining <- p - 5
if (p_remaining > 0) {
  X_remaining <- matrix(rnorm(n * p_remaining), ncol = p_remaining)
  X <- cbind(X, X_remaining)
}
mu_x <- mu1(X)
tau_x <- tau1(X)
pi_x <- 0.8 * pnorm((3 * mu_x / sd(mu_x)) - 0.5 * X[, 1]) + 0.05 + runif(n) / 10
Z <- rbinom(n, 1, pi_x)
E_XZ <- mu_x + Z * tau_x
rfx_group_ids <- rep(c(1, 2), n %/% 2)
rfx_coefs <- matrix(c(-1, -1, 1, 1), nrow = 2, byrow = TRUE)
rfx_basis <- cbind(1, runif(n, -1, 1))
rfx_term <- rowSums(rfx_coefs[rfx_group_ids, ] * rfx_basis)
y <- E_XZ + rfx_term + rnorm(n, 0, 1) * (sd(E_XZ) / snr)
X <- as.data.frame(X)
X$x4 <- factor(X$x4, ordered = TRUE)
X$x5 <- factor(X$x5, ordered = TRUE)

# Split into train and test sets
test_set_pct <- 0.2
n_test <- round(test_set_pct * n)
n_train <- n - n_test
test_inds <- sort(sample(1:n, n_test, replace = FALSE))
train_inds <- (1:n)[!((1:n) %in% test_inds)]
X_test <- X[test_inds, ]
X_train <- X[train_inds, ]
Z_test <- Z[test_inds]
Z_train <- Z[train_inds]
pi_test <- pi_x[test_inds]
pi_train <- pi_x[train_inds]
y_test <- y[test_inds]
y_train <- y[train_inds]

system.time({
  # Sample BCF model
  general_params <- list(num_threads = num_threads)
  bcf_model <- bcf(
    X_train = X_train,
    Z_train = Z_train,
    propensity_train = pi_train,
    y_train = y_train,
    X_test = X_test,
    Z_test = Z_test,
    propensity_test = pi_test,
    num_gfr = num_gfr,
    num_mcmc = num_mcmc,
    general_params = general_params
  )

  # Predict on the test set
  test_preds <- predict(bcf_model, X = X_test, Z = Z_test, propensity = pi_test)
})
