# Load libraries
library(stochtree)
library(microbenchmark)

# Generate the data
n <- 1000
p <- 100
snr <- 2
X <- matrix(rnorm(n * p), ncol = p)
mu_x <- 1 +
  2 * X[, 1] -
  4 * (X[, 2] < 0) +
  4 * (X[, 2] >= 0) +
  3 * (abs(X[, 3]) - sqrt(2 / pi))
tau_x <- 1 + 2 * X[, 4]
u <- runif(n)
pi_x <- ((mu_x - 1) / 4) + 4 * (u - 0.5)
Z <- pi_x + rnorm(n, 0, 1)
E_XZ <- mu_x + Z * tau_x
noise_sd <- sd(E_XZ) / snr
y <- E_XZ + rnorm(n, 0, 1) * noise_sd

# Split data into test and train sets
test_set_pct <- 0.2
n_test <- round(test_set_pct * n)
n_train <- n - n_test
test_inds <- sort(sample(1:n, n_test, replace = FALSE))
train_inds <- (1:n)[!((1:n) %in% test_inds)]
X_test <- X[test_inds, ]
X_train <- X[train_inds, ]
Z_test <- Z[test_inds]
Z_train <- Z[train_inds]
pi_x_test <- pi_x[test_inds]
pi_x_train <- pi_x[train_inds]
y_test <- y[test_inds]
y_train <- y[train_inds]

# Sampler settings
num_gfr <- 100
num_burnin <- 0
num_mcmc <- 0
general_params <- list(sample_sigma2_global = T)
prog_params_a <- list(num_trees = 100, num_features_subsample = 5)
trt_params_a <- list(num_trees = 100, num_features_subsample = 5)
prog_params_b <- list(num_trees = 50)
trt_params_b <- list(num_trees = 50)

# Benchmark sampler with and without feature subsampling
microbenchmark::microbenchmark(
  stochtree::bcf(
    X_train = X,
    Z_train = Z,
    propensity_train = pi_x,
    y_train = y,
    num_gfr = num_gfr,
    num_burnin = num_burnin,
    num_mcmc = num_mcmc,
    general_params = general_params,
    prognostic_forest_params = prog_params_a,
    treatment_effect_forest_params = trt_params_a
  ),
  stochtree::bcf(
    X_train = X,
    Z_train = Z,
    propensity_train = pi_x,
    y_train = y,
    num_gfr = num_gfr,
    num_burnin = num_burnin,
    num_mcmc = num_mcmc,
    general_params = general_params,
    prognostic_forest_params = prog_params_b,
    treatment_effect_forest_params = trt_params_b
  ),
  times = 5
)

Rprof()
model_subsampling <- stochtree::bcf(
  X_train = X_train,
  Z_train = Z_train,
  propensity_train = pi_x_train,
  y_train = y_train,
  num_gfr = num_gfr,
  num_burnin = num_burnin,
  num_mcmc = num_mcmc,
  general_params = general_params,
  prognostic_forest_params = prog_params_a,
  treatment_effect_forest_params = trt_params_a
)
Rprof(NULL)
summaryRprof()

Rprof()
model_no_subsampling <- stochtree::bcf(
  X_train = X_train,
  Z_train = Z_train,
  propensity_train = pi_x_train,
  y_train = y_train,
  num_gfr = num_gfr,
  num_burnin = num_burnin,
  num_mcmc = num_mcmc,
  general_params = general_params,
  prognostic_forest_params = prog_params_b,
  treatment_effect_forest_params = trt_params_b
)
Rprof(NULL)
summaryRprof()

# Compare out of sample RMSE of the two models
y_hat_test_subsampling <- rowMeans(
  predict(
    model_subsampling,
    X = X_test,
    Z = Z_test,
    propensity = pi_x_test
  )$y_hat
)
rmse_subsampling <- (sqrt(mean((y_hat_test_subsampling - y_test)^2)))
y_hat_test_no_subsampling <- rowMeans(
  predict(
    model_no_subsampling,
    X = X_test,
    Z = Z_test,
    propensity = pi_x_test
  )$y_hat
)
rmse_no_subsampling <- (sqrt(mean((y_hat_test_no_subsampling - y_test)^2)))
