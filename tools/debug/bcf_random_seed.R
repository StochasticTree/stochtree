# Load libraries
library(stochtree)

# Generate data
random_seed <- 1234
set.seed(random_seed)
n <- 500
p <- 50
X <- matrix(runif(n * p), ncol = p)
# fmt: skip
f_X <- (
    ((0 <= X[, 1]) & (0.25 > X[, 1])) * (-7.5) +
    ((0.25 <= X[, 1]) & (0.5 > X[, 1])) * (-2.5) +
    ((0.5 <= X[, 1]) & (0.75 > X[, 1])) * (2.5) +
    ((0.75 <= X[, 1]) & (1 > X[, 1])) * (7.5)
)
mu_X <- f_X
pi_X <- pnorm(f_X * 0.25)
tau_X <- 0.5 * X[,2]
Z <- rbinom(n, 1, pi_X)
E_XZ <- mu_X + Z * tau_X
y <- E_XZ + rnorm(n, 0, 1)

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
pi_test <- pi_X[test_inds]
pi_train <- pi_X[train_inds]
y_test <- y[test_inds]
y_train <- y[train_inds]

# Set a different global seed as a test
set.seed(837475)

# Run BCF model
general_params <- list(num_threads = 1, random_seed = random_seed)
bcf_model <- bcf(
    X_train = X_train,
    Z_train = Z_train,
    propensity_train = pi_train,
    y_train = y_train,
    X_test = X_test,
    Z_test = Z_test,
    propensity_test = pi_test,
    num_gfr = 100,
    num_mcmc = 100,
    general_params = general_params
)

# # Save results
# write.csv(
#   bcf_model$y_hat_test,
#   file = "tools/debug/seed_benchmark_bcf_y_hat.csv",
#   row.names = FALSE
# )

# Read results and compare to our estimates
y_hat_test_benchmark <- as.matrix(read.csv(
    "tools/debug/seed_benchmark_bcf_y_hat.csv"
))

# Compare results
sum(abs(y_hat_test_benchmark - bcf_model$y_hat_test) > 1e-6)

# Generate probit data
random_seed <- 1234
set.seed(random_seed)
n <- 500
p <- 50
X <- matrix(runif(n * p), ncol = p)
# fmt: skip
f_X <- (
    ((0 <= X[, 1]) & (0.25 > X[, 1])) * (-7.5) +
    ((0.25 <= X[, 1]) & (0.5 > X[, 1])) * (-2.5) +
    ((0.5 <= X[, 1]) & (0.75 > X[, 1])) * (2.5) +
    ((0.75 <= X[, 1]) & (1 > X[, 1])) * (7.5)
)
mu_X <- f_X
pi_X <- pnorm(f_X * 0.25)
tau_X <- 0.5 * X[,2]
Z <- rbinom(n, 1, pi_X)
E_XZ <- mu_X + Z * tau_X
W <- E_XZ + rnorm(n, 0, 1)
y <- (W > 0) * 1

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
pi_test <- pi_X[test_inds]
pi_train <- pi_X[train_inds]
W_test <- W[test_inds]
W_train <- W[train_inds]
y_test <- y[test_inds]
y_train <- y[train_inds]

# Set a different global seed as a test
set.seed(23446345)

# Run BCF model
general_params <- list(num_threads = 1, random_seed = random_seed, 
                       probit_outcome_model = T)
bcf_model <- bcf(
    X_train = X_train,
    Z_train = Z_train,
    propensity_train = pi_train,
    y_train = y_train,
    X_test = X_test,
    Z_test = Z_test,
    propensity_test = pi_test,
    num_gfr = 100,
    num_mcmc = 100,
    general_params = general_params
)

# # Save results
# write.csv(
#   bcf_model$y_hat_test,
#   file = "tools/debug/seed_benchmark_bcf_probit_y_hat.csv",
#   row.names = FALSE
# )

# Read results and compare to our estimates
y_hat_test_benchmark <- as.matrix(read.csv(
    "tools/debug/seed_benchmark_bcf_probit_y_hat.csv"
))

# Compare results
sum(abs(y_hat_test_benchmark - bcf_model$y_hat_test) > 1e-6)
