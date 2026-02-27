# Load libraries
library(stochtree)
library(microbenchmark)

# Generate the data
n <- 1000
p <- 100
snr <- 2
X <- matrix(runif(n * p), ncol = p)
f_XW <- sin(4 * pi * X[, 1]) +
  cos(4 * pi * X[, 2]) +
  sin(4 * pi * X[, 3]) +
  cos(4 * pi * X[, 4])
noise_sd <- sd(f_XW) / snr
y <- f_XW + rnorm(n, 0, 1) * noise_sd

# Split data into test and train sets
test_set_pct <- 0.2
n_test <- round(test_set_pct * n)
n_train <- n - n_test
test_inds <- sort(sample(1:n, n_test, replace = FALSE))
train_inds <- (1:n)[!((1:n) %in% test_inds)]
X_test <- as.data.frame(X[test_inds, ])
X_train <- as.data.frame(X[train_inds, ])
y_test <- y[test_inds]
y_train <- y[train_inds]

# Sampler settings
num_gfr <- 100
num_burnin <- 0
num_mcmc <- 0
general_params <- list(sample_sigma2_global = T)
mean_params_a <- list(num_trees = 100, num_features_subsample = 5)
mean_params_b <- list(num_trees = 100)

# Benchmark sampler with and without feature subsampling
microbenchmark::microbenchmark(
  stochtree::bart(
    X_train = X,
    y_train = y,
    num_gfr = num_gfr,
    num_burnin = num_burnin,
    num_mcmc = num_mcmc,
    general_params = general_params,
    mean_forest_params = mean_params_a
  ),
  stochtree::bart(
    X_train = X,
    y_train = y,
    num_gfr = num_gfr,
    num_burnin = num_burnin,
    num_mcmc = num_mcmc,
    general_params = general_params,
    mean_forest_params = mean_params_b
  ),
  times = 5
)

Rprof()
model_subsampling <- stochtree::bart(
  X_train = X_train,
  y_train = y_train,
  num_gfr = num_gfr,
  num_burnin = num_burnin,
  num_mcmc = num_mcmc,
  general_params = general_params,
  mean_forest_params = mean_params_a
)
Rprof(NULL)
summaryRprof()

Rprof()
model_no_subsampling <- stochtree::bart(
  X_train = X_train,
  y_train = y_train,
  num_gfr = num_gfr,
  num_burnin = num_burnin,
  num_mcmc = num_mcmc,
  general_params = general_params,
  mean_forest_params = mean_params_b
)
Rprof(NULL)
summaryRprof()

# Compare out of sample RMSE of the two models
y_hat_test_subsampling <- rowMeans(predict(model_subsampling, X = X_test)$y_hat)
rmse_subsampling <- (sqrt(mean((y_hat_test_subsampling - y_test)^2)))
y_hat_test_no_subsampling <- rowMeans(
  predict(model_no_subsampling, X = X_test)$y_hat
)
rmse_no_subsampling <- (sqrt(mean((y_hat_test_no_subsampling - y_test)^2)))
