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
p_x <- 10
snr <- 2
X <- matrix(runif(n*p_x), ncol = p_x)
f_XW <- sin(4*pi*X[,1]) + cos(4*pi*X[,2]) + sin(4*pi*X[,3]) +cos(4*pi*X[,4])
noise_sd <- sd(f_XW) / snr
y <- f_XW + rnorm(n, 0, 1)*noise_sd

# Split data into test and train sets
test_set_pct <- 0.2
n_test <- round(test_set_pct*n)
n_train <- n - n_test
test_inds <- sort(sample(1:n, n_test, replace = FALSE))
train_inds <- (1:n)[!((1:n) %in% test_inds)]
X_test <- as.data.frame(X[test_inds,])
X_train <- as.data.frame(X[train_inds,])
y_test <- y[test_inds]
y_train <- y[train_inds]
f_XW_test <- f_XW[test_inds]
f_XW_train <- f_XW[train_inds]

# Run the GFR algorithm
general_params <- list(sample_sigma2_global = T)
mean_forest_params <- list(num_trees = num_trees, alpha = 0.95, 
                           beta = 2.0, max_depth = -1, 
                           min_samples_leaf = 1, 
                           sample_sigma2_leaf = F, 
                           sigma2_leaf_init = 1.0/num_trees)
xbart_model <- stochtree::bart(
    X_train = X_train, y_train = y_train, X_test = X_test, 
    num_gfr = num_gfr, num_burnin = 0, num_mcmc = 0, 
    general_params = general_params, 
    mean_forest_params = mean_forest_params
)

# Inspect results
plot(rowMeans(xbart_model$y_hat_test), y_test); abline(0,1)
cat(paste0("RMSE = ", sqrt(mean((rowMeans(xbart_model$y_hat_test) - y_test)^2)), "\n"))
cat(paste0("Interval coverage = ", mean((apply(xbart_model$y_hat_test, 1, quantile, probs=0.025) <= f_XW_test) & (apply(xbart_model$y_hat_test, 1, quantile, probs=0.975) >= f_XW_test)), "\n"))
plot(xbart_model$sigma2_global_samples)
xbart_model_string <- stochtree::saveBARTModelToJsonString(xbart_model)

# Run the BART MCMC sampler, initialized from the XBART sampler
general_params <- list(sample_sigma2_global = T)
mean_forest_params <- list(num_trees = num_trees, alpha = 0.95, 
                           beta = 2.0, max_depth = -1, 
                           min_samples_leaf = 1, 
                           sample_sigma2_leaf = F, 
                           sigma2_leaf_init = 1.0/num_trees)
bart_model <- stochtree::bart(
    X_train = X_train, y_train = y_train, X_test = X_test, 
    num_gfr = 0, num_burnin = num_burnin, num_mcmc = num_mcmc, 
    general_params = general_params, mean_forest_params = mean_forest_params, 
    previous_model_json = xbart_model_string, 
    previous_model_warmstart_sample_num = num_gfr
)

# Inspect the results
plot(rowMeans(bart_model$y_hat_test), y_test); abline(0,1)
cat(paste0("RMSE = ", sqrt(mean((rowMeans(bart_model$y_hat_test) - y_test)^2)), "\n"))
cat(paste0("Interval coverage = ", mean((apply(bart_model$y_hat_test, 1, quantile, probs=0.025) <= f_XW_test) & (apply(bart_model$y_hat_test, 1, quantile, probs=0.975) >= f_XW_test)), "\n"))
plot(bart_model$sigma2_global_samples)

# Compare to a single chain of MCMC samples initialized at root
bart_model_root <- stochtree::bart(
    X_train = X_train, y_train = y_train, X_test = X_test, 
    num_gfr = 0, num_burnin = 0, num_mcmc = num_mcmc, 
    general_params = general_params, mean_forest_params = mean_forest_params
)
plot(rowMeans(bart_model_root$y_hat_test), y_test); abline(0,1)
cat(paste0("RMSE = ", sqrt(mean((rowMeans(bart_model_root$y_hat_test) - y_test)^2)), "\n"))
cat(paste0("Interval coverage = ", mean((apply(bart_model_root$y_hat_test, 1, quantile, probs=0.025) <= f_XW_test) & (apply(bart_model_root$y_hat_test, 1, quantile, probs=0.975) >= f_XW_test)), "\n"))
plot(bart_model_root$sigma2_global_samples)
