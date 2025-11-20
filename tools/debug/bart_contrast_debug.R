# Demo of CATE computation function for BCF

# Load library
library(stochtree)

# Generate data
n <- 500
p <- 5
X <- matrix(rnorm(n * p), ncol = p)
W <- matrix(rnorm(n * 1), ncol = 1)
# fmt: skip
f_XW <- (
    ((0 <= X[,1]) & (0.25 > X[,1])) * (-7.5 * W[,1]) +
    ((0.25 <= X[,1]) & (0.5 > X[,1])) * (-2.5 * W[,1]) +
    ((0.5 <= X[,1]) & (0.75 > X[,1])) * (2.5 * W[,1]) +
    ((0.75 <= X[,1]) & (1 > X[,1])) * (7.5 * W[,1])
)
E_Y <- f_XW
snr <- 2
y <- E_Y + rnorm(n, 0, 1) * (sd(E_Y) / snr)

# Train-test split
test_set_pct <- 0.2
n_test <- round(test_set_pct * n)
n_train <- n - n_test
test_inds <- sort(sample(1:n, n_test, replace = FALSE))
train_inds <- (1:n)[!((1:n) %in% test_inds)]
X_test <- X[test_inds, ]
X_train <- X[train_inds, ]
W_test <- W[test_inds]
W_train <- W[train_inds]
y_test <- y[test_inds]
y_train <- y[train_inds]

# Fit BART model
bart_model <- bart(
  X_train = X_train,
  leaf_basis_train = W_train,
  y_train = y_train,
  num_gfr = 10,
  num_burnin = 0,
  num_mcmc = 1000
)

# Compute contrast posterior
contrast_posterior_test <- compute_contrast_bart_model(
  bart_model,
  X_0 = X_test,
  X_1 = X_test,
  leaf_basis_0 = matrix(0, nrow = n_test, ncol = 1),
  leaf_basis_1 = matrix(1, nrow = n_test, ncol = 1),
  type = "posterior",
  scale = "linear"
)

# Compute the same quantity via two predict calls
y_hat_posterior_test_0 <- predict(
  bart_model,
  X = X_test,
  leaf_basis = matrix(0, nrow = n_test, ncol = 1),
  type = "posterior",
  term = "y_hat",
  scale = "linear"
)
y_hat_posterior_test_1 <- predict(
  bart_model,
  X = X_test,
  leaf_basis = matrix(1, nrow = n_test, ncol = 1),
  type = "posterior",
  term = "y_hat",
  scale = "linear"
)
contrast_posterior_test_comparison <- (y_hat_posterior_test_1 -
  y_hat_posterior_test_0)

# Compare results
contrast_diff <- contrast_posterior_test_comparison - contrast_posterior_test
all(
  abs(contrast_diff) < 0.001
)

# Generate data for a BCF model with random effects
X <- matrix(rnorm(n * p), ncol = p)
W <- matrix(rnorm(n * 1), ncol = 1)
# fmt: skip
f_XW <- (
    ((0 <= X[,1]) & (0.25 > X[,1])) * (-7.5 * W[,1]) +
    ((0.25 <= X[,1]) & (0.5 > X[,1])) * (-2.5 * W[,1]) +
    ((0.5 <= X[,1]) & (0.75 > X[,1])) * (2.5 * W[,1]) +
    ((0.75 <= X[,1]) & (1 > X[,1])) * (7.5 * W[,1])
)
group_ids <- rep(c(1, 2), n %/% 2)
rfx_coefs <- matrix(c(-1, -1, 1, 1), nrow = 2, byrow = TRUE)
rfx_basis <- cbind(1, runif(n))
rfx_term <- rowSums(rfx_coefs[group_ids, ] * rfx_basis)
E_Y <- f_XW + rfx_term
snr <- 2
y <- E_Y + rnorm(n, 0, 1) * (sd(E_Y) / snr)

# Train-test split
n_test <- round(test_set_pct * n)
n_train <- n - n_test
test_inds <- sort(sample(1:n, n_test, replace = FALSE))
train_inds <- (1:n)[!((1:n) %in% test_inds)]
X_test <- X[test_inds, ]
X_train <- X[train_inds, ]
W_test <- W[test_inds]
W_train <- W[train_inds]
y_test <- y[test_inds]
y_train <- y[train_inds]
group_ids_test <- group_ids[test_inds]
group_ids_train <- group_ids[train_inds]
rfx_basis_test <- rfx_basis[test_inds, ]
rfx_basis_train <- rfx_basis[train_inds, ]

# Fit BART model
bart_model <- bart(
  X_train = X_train,
  leaf_basis_train = W_train,
  y_train = y_train,
  rfx_group_ids_train = group_ids_train,
  rfx_basis_train = rfx_basis_train,
  num_gfr = 10,
  num_burnin = 0,
  num_mcmc = 1000
)

# Compute contrast posterior
contrast_posterior_test <- compute_contrast_bart_model(
  bart_model,
  X_0 = X_test,
  X_1 = X_test,
  leaf_basis_0 = matrix(0, nrow = n_test, ncol = 1),
  leaf_basis_1 = matrix(1, nrow = n_test, ncol = 1),
  rfx_group_ids_0 = group_ids_test,
  rfx_group_ids_1 = group_ids_test,
  rfx_basis_0 = rfx_basis_test,
  rfx_basis_1 = rfx_basis_test,
  type = "posterior",
  scale = "linear"
)

# Compute the same quantity via two predict calls
y_hat_posterior_test_0 <- predict(
  bart_model,
  X = X_test,
  leaf_basis = matrix(0, nrow = n_test, ncol = 1),
  rfx_group_ids = group_ids_test,
  rfx_basis = rfx_basis_test,
  type = "posterior",
  term = "y_hat",
  scale = "linear"
)
y_hat_posterior_test_1 <- predict(
  bart_model,
  X = X_test,
  leaf_basis = matrix(1, nrow = n_test, ncol = 1),
  rfx_group_ids = group_ids_test,
  rfx_basis = rfx_basis_test,
  type = "posterior",
  term = "y_hat",
  scale = "linear"
)
contrast_posterior_test_comparison <- (y_hat_posterior_test_1 -
  y_hat_posterior_test_0)

# Compare results
contrast_diff <- contrast_posterior_test_comparison - contrast_posterior_test
all(
  abs(contrast_diff) < 0.001
)
