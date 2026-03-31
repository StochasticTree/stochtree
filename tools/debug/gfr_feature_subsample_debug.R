# Load libraries
library(stochtree)

# Generate the data
n <- 500
p <- 20
snr <- 2
X <- matrix(runif(n*p), ncol = p)
f_XW <- sin(4*pi*X[,1]) + cos(4*pi*X[,2]) + sin(4*pi*X[,3]) + cos(4*pi*X[,4])
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

# Sampler settings
num_gfr <- 100
num_burnin <- 0
num_mcmc <- 20

# Run the sampler
general_params <- list(random_seed = 1234, standardize = F, sample_sigma2_global = T)
mean_params <- list(num_trees = 100, num_features_subsample = 5)
bart_model <- stochtree::bart(
    X_train = X, y_train = y, 
    num_gfr = num_gfr, num_burnin = num_burnin, num_mcmc = num_mcmc, 
    general_params = general_params, mean_forest_params = mean_params
)