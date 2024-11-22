# Load libraries
library(stochtree)
library(foreach)
library(doParallel)

# Sampler settings
num_chains <- 4
num_gfr <- 10
num_burnin <- 0
num_mcmc <- 100
num_trees <- 100

# Generate the data
n <- 500
p_x <- 10
p_w <- 1
snr <- 3
X <- matrix(runif(n*p_x), ncol = p_x)
W <- matrix(runif(n*p_w), ncol = p_w)
f_XW <- (
    ((0 <= X[,1]) & (0.25 > X[,1])) * (-7.5*W[,1]) + 
        ((0.25 <= X[,1]) & (0.5 > X[,1])) * (-2.5*W[,1]) + 
        ((0.5 <= X[,1]) & (0.75 > X[,1])) * (2.5*W[,1]) + 
        ((0.75 <= X[,1]) & (1 > X[,1])) * (7.5*W[,1])
)
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
W_test <- W[test_inds,]
W_train <- W[train_inds,]
y_test <- y[test_inds]
y_train <- y[train_inds]

# Run the GFR algorithm
xbart_params <- list(sample_sigma_global = T, sample_sigma_leaf = T, 
                     num_trees_mean = num_trees)
xbart_model <- stochtree::bart(
    X_train = X_train, W_train = W_train, y_train = y_train, X_test = X_test, W_test = W_test, 
    num_gfr = num_gfr, num_burnin = 0, num_mcmc = 0, params = xbart_params
)
xbart_model_string <- stochtree::saveBARTModelToJsonString(xbart_model)

# Parallel setup
ncores <- parallel::detectCores()
cl <- makeCluster(ncores)
registerDoParallel(cl)

# Run the parallel BART MCMC samplers
bart_model_outputs <- foreach (i = 1:num_chains) %dopar% {
    random_seed <- i
    bart_params <- list(sample_sigma_global = T, sample_sigma_leaf = T, 
                        num_trees_mean = num_trees, random_seed = random_seed)
    bart_model <- stochtree::bart(
        X_train = X_train, W_train = W_train, y_train = y_train, X_test = X_test, W_test = W_test, 
        num_gfr = 0, num_burnin = num_burnin, num_mcmc = num_mcmc, params = bart_params, 
        previous_model_json = xbart_model_string, warmstart_sample_num = num_gfr - i + 1, 
    )
    bart_model_string <- stochtree::saveBARTModelToJsonString(bart_model)
    y_hat_test <- bart_model$y_hat_test
    list(model=bart_model_string, yhat=y_hat_test)
}

# Close the cluster connection
stopCluster(cl)

# Combine the forests
bart_model_strings <- list()
bart_model_yhats <- matrix(NA, nrow = length(y_test), ncol = num_chains)
for (i in 1:length(bart_model_outputs)) {
    bart_model_strings[[i]] <- bart_model_outputs[[i]]$model
    bart_model_yhats[,i] <- rowMeans(bart_model_outputs[[i]]$yhat)
}
combined_bart <- createBARTModelFromCombinedJsonString(bart_model_strings)

# Inspect the results
yhat_combined <- predict(combined_bart, X_test, W_test)$y_hat
par(mfrow = c(1,2))
for (i in 1:num_chains) {
    offset <- (i-1)*num_mcmc
    inds_start <- offset + 1
    inds_end <- offset + num_mcmc
    plot(rowMeans(yhat_combined[,inds_start:inds_end]), bart_model_yhats[,i],
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
}
par(mfrow = c(1,1))
