# Load libraries
library(stochtree)

# Sampler settings
num_chains <- 1
num_gfr <- 10
num_burnin <- 0
num_mcmc <- 20
num_trees <- 100

# Generate the data
n <- 500
p <- 5
snr <- 2
X <- matrix(runif(n*p), ncol = p)
mu_x <- (
    ((0 <= X[,1]) & (0.25 > X[,1])) * (-7.5) + 
        ((0.25 <= X[,1]) & (0.5 > X[,1])) * (-2.5) + 
        ((0.5 <= X[,1]) & (0.75 > X[,1])) * (2.5) + 
        ((0.75 <= X[,1]) & (1 > X[,1])) * (7.5)
)
pi_x <- (
    ((0 <= X[,1]) & (0.25 > X[,1])) * (0.2) + 
        ((0.25 <= X[,1]) & (0.5 > X[,1])) * (0.4) + 
        ((0.5 <= X[,1]) & (0.75 > X[,1])) * (0.6) + 
        ((0.75 <= X[,1]) & (1 > X[,1])) * (0.8)
)
tau_x <- (
    ((0 <= X[,2]) & (0.25 > X[,2])) * (0.5) + 
        ((0.25 <= X[,2]) & (0.5 > X[,2])) * (1.0) + 
        ((0.5 <= X[,2]) & (0.75 > X[,2])) * (1.5) + 
        ((0.75 <= X[,2]) & (1 > X[,2])) * (2.0)
)
Z <- rbinom(n, 1, pi_x)
f_XZ <- mu_x + tau_x*Z
noise_sd <- sd(f_XZ) / snr
y <- f_XZ + rnorm(n, 0, 1)*noise_sd

# Split data into test and train sets
test_set_pct <- 0.2
n_test <- round(test_set_pct*n)
n_train <- n - n_test
test_inds <- sort(sample(1:n, n_test, replace = FALSE))
train_inds <- (1:n)[!((1:n) %in% test_inds)]
X_test <- X[test_inds,]
X_train <- X[train_inds,]
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

# Run the GFR algorithm
general_params <- list(sample_sigma2_global = T)
xbcf_model <- bcf(X_train = X_train, Z_train = Z_train, y_train = y_train, 
                  propensity_train = pi_train, X_test = X_test, Z_test = Z_test, 
                  propensity_test = pi_test, num_gfr = num_gfr, num_burnin = 0, 
                  num_mcmc = 0, general_params = general_params)

# Inspect results
plot(rowMeans(xbcf_model$y_hat_test), y_test); abline(0,1)
cat(paste0("RMSE = ", sqrt(mean((rowMeans(xbcf_model$y_hat_test) - y_test)^2)), "\n"))
plot(xbcf_model$sigma2_global_samples)
xbcf_model_string <- stochtree::saveBCFModelToJsonString(xbcf_model)

# Run the BCF MCMC sampler, initialized from the XBART sampler
general_params <- list(sample_sigma2_global = T)
bcf_model <- bcf(X_train = X_train, Z_train = Z_train, y_train = y_train, 
                  propensity_train = pi_train, X_test = X_test, Z_test = Z_test, 
                  propensity_test = pi_test, num_gfr = 0, num_burnin = num_burnin, 
                  num_mcmc = num_mcmc, general_params = general_params, 
                  previous_model_json = xbcf_model_string, 
                  previous_model_warmstart_sample_num = num_gfr)

# Inspect the results
plot(rowMeans(bcf_model$y_hat_test), y_test); abline(0,1)
cat(paste0("RMSE = ", sqrt(mean((rowMeans(bcf_model$y_hat_test) - y_test)^2)), "\n"))
plot(bcf_model$sigma2_global_samples)

# Compare to a single chain of MCMC samples initialized at root
bcf_model_root <- bcf(X_train = X_train, Z_train = Z_train, y_train = y_train, 
                      propensity_train = pi_train, X_test = X_test, Z_test = Z_test, 
                      propensity_test = pi_test, num_gfr = 0, num_burnin = num_burnin, 
                      num_mcmc = num_mcmc, general_params = general_params)
plot(rowMeans(bcf_model_root$y_hat_test), y_test); abline(0,1)
plot(bcf_model_root$sigma2_global_samples)
cat(paste0("RMSE = ", sqrt(mean((rowMeans(bcf_model_root$y_hat_test) - y_test)^2)), "\n"))
