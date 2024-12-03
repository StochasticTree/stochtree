# Load libraries
library(stochtree)
library(foreach)
library(doParallel)

# Sampler settings
num_chains <- 6
num_gfr <- 10
num_burnin <- 0
num_mcmc <- 20
num_trees_mu <- 100
num_trees_tau <- 20

# Generate the data
n <- 500
x1 <- rnorm(n)
x2 <- rnorm(n)
x3 <- rnorm(n)
x4 <- rnorm(n,x2,1)
X <- cbind(x1,x2,x3,x4)
p <- ncol(X)
mu <- function(x) {-1*(x[,1]>(x[,2])) + 1*(x[,1]<(x[,2])) - 0.1}
tau <- function(x) {1/(1 + exp(-x[,3])) + x[,2]/10}
mu_x <- mu(X)
tau_x <- tau(X)
pi_x <- pnorm(mu_x)
Z <- rbinom(n,1,pi_x)
E_XZ <- mu_x + Z*tau_x
sigma <- diff(range(mu_x + tau_x*pi))/8
y <- E_XZ + sigma*rnorm(n)
X <- as.data.frame(X)

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
xbcf_params <- list(num_trees_mu = num_trees_mu, num_trees_tau = num_trees_tau, 
                    alpha_mu = 0.95, beta_mu = 1, max_depth_mu = -1, 
                    alpha_tau = 0.8, beta_tau = 2, max_depth_tau = 10)
xbcf_model <- stochtree::bcf(
    X_train = X_train, Z_train = Z_train, y_train = y_train, pi_train = pi_train,
    X_test = X_test, Z_test = Z_test, pi_test = pi_test, num_gfr = num_gfr, 
    num_burnin = 0, num_mcmc = 0, params = xbcf_params
)
plot(rowMeans(xbcf_model$y_hat_test), y_test); abline(0,1)
cat(sqrt(mean((rowMeans(xbcf_model$y_hat_test) - y_test)^2)), "\n")
cat(mean((apply(xbcf_model$y_hat_test, 1, quantile, probs=0.05) <= y_test) & (apply(xbcf_model$y_hat_test, 1, quantile, probs=0.95) >= y_test)), "\n")
xbcf_model_string <- stochtree::saveBCFModelToJsonString(xbcf_model)

# Parallel setup
ncores <- parallel::detectCores()
cl <- makeCluster(ncores)
registerDoParallel(cl)

# Run the parallel BART MCMC samplers
bcf_model_outputs <- foreach (i = 1:num_chains) %dopar% {
    random_seed <- i
    bcf_params <- list(num_trees_mu = num_trees_mu, num_trees_tau = num_trees_tau, 
                       random_seed = random_seed)
    bcf_model <- stochtree::bcf(
        X_train = X_train, Z_train = Z_train, y_train = y_train, pi_train = pi_train,
        X_test = X_test, Z_test = Z_test, pi_test = pi_test, 
        num_gfr = 0, num_burnin = num_burnin, num_mcmc = num_mcmc, params = bcf_params, 
        previous_model_json = xbcf_model_string, warmstart_sample_num = num_gfr - i + 1, 
    )
    bcf_model_string <- stochtree::saveBCFModelToJsonString(bcf_model)
    y_hat_test <- bcf_model$y_hat_test
    list(model=bcf_model_string, yhat=y_hat_test)
}

# Close the cluster connection
stopCluster(cl)

# Combine the forests
bcf_model_strings <- list()
bcf_model_yhats <- matrix(NA, nrow = length(y_test), ncol = num_chains)
for (i in 1:length(bcf_model_outputs)) {
    bcf_model_strings[[i]] <- bcf_model_outputs[[i]]$model
    bcf_model_yhats[,i] <- rowMeans(bcf_model_outputs[[i]]$yhat)
}
combined_bcf <- createBCFModelFromCombinedJsonString(bcf_model_strings)

# Inspect the results
yhat_combined <- predict(combined_bcf, X_test)$y_hat
par(mfrow = c(1,2))
for (i in 1:num_chains) {
    offset <- (i-1)*num_mcmc
    inds_start <- offset + 1
    inds_end <- offset + num_mcmc
    plot(rowMeans(yhat_combined[,inds_start:inds_end]), bcf_model_yhats[,i],
         xlab = "deserialized", ylab = "original", 
         main = paste0("Chain ", i, "\nPredictions"))
    abline(0,1,col="red",lty=3,lwd=3)
}
for (i in 1:num_chains) {
    offset <- (i-1)*num_mcmc
    inds_start <- offset + 1
    inds_end <- offset + num_mcmc
    plot(rowMeans(yhat_combined[,inds_start:inds_end]), y_test,
         xlab = "predicted", ylab = "actual", 
         main = paste0("Chain ", i, "\nPredictions"))
    abline(0,1,col="red",lty=3,lwd=3)
    cat(sqrt(mean((rowMeans(yhat_combined[,inds_start:inds_end]) - y_test)^2)), "\n")
    cat(mean((apply(yhat_combined[,inds_start:inds_end], 1, quantile, probs=0.05) <= y_test) & (apply(yhat_combined[,inds_start:inds_end], 1, quantile, probs=0.95) >= y_test)), "\n")
}
par(mfrow = c(1,1))

# Compare to a single chain of MCMC samples initialized at root
bcf_params <- list(sample_sigma_global = T, sample_sigma_leaf = T, 
                    num_trees_mean = num_trees, alpha_mean = 0.95, beta_mean = 2)
bcf_model <- stochtree::bcf(
    X_train = X_train, Z_train = Z_train, y_train = y_train, pi_train = pi_train,
    X_test = X_test, Z_test = Z_test, pi_test = pi_test, 
    num_gfr = 0, num_burnin = 0, num_mcmc = num_mcmc, params = bcf_params
)
plot(rowMeans(bcf_model$y_hat_test), y_test, xlab = "predicted", ylab = "actual"); abline(0,1)
cat(sqrt(mean((rowMeans(bcf_model$y_hat_test) - y_test)^2)), "\n")
cat(mean((apply(bcf_model$y_hat_test, 1, quantile, probs=0.05) <= y_test) & (apply(bcf_model$y_hat_test, 1, quantile, probs=0.95) >= y_test)), "\n")
