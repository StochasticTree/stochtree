################################################################################
## Comparison of several stochastic tree ensemble implementations for causal 
## inference in terms of
##   (a) Run time (s)
##   (b) Train set RMSE
##   (c) Test set RMSE
## 
## The implementations we compare are
##   (1) Warm-start BCF (StochTree)
##   (2) BCF (StochTree)
##   (3) BCF (multibart)
################################################################################

# Load libraries
library(here)
library(foreach)
library(doParallel)

# Helper functions
g <- function(x) {ifelse(x[,5]==1,2,ifelse(x[,5]==2,-1,-4))}
mu1 <- function(x) {1+g(x)+x[,1]*x[,3]}
mu2 <- function(x) {1+g(x)+6*abs(x[,3]-1)}
tau1 <- function(x) {rep(3,nrow(x))}
tau2 <- function(x) {1+2*x[,2]*x[,4]}

# Define a function that runs one iteration of the simulation
simulation_function = function(n, mu=mu1, tau=tau2, gfr_iter=10, burnin_iter=1000, 
                               mcmc_iter=1000, alpha=0.05, snr=3, sample_tau=F){
    # Generate data
    x1 <- rnorm(n)
    x2 <- rnorm(n)
    x3 <- rnorm(n)
    x4 <- as.numeric(rbinom(n,1,0.5))
    x5 <- as.numeric(sample(1:3,n,replace=T))
    X <- cbind(x1,x2,x3,x4,x5)
    p <- ncol(X)
    mu_x <- mu(X)
    tau_x <- tau(X)
    pi_x <- 0.8*pnorm((3*mu_x/sd(mu_x)) - 0.5*X[,1]) + 0.05 + runif(n)/10
    Z <- rbinom(n,1,pi_x)
    E_XZ <- mu_x + Z*tau_x
    y <- E_XZ + rnorm(n, 0, 1)*(sd(E_XZ)/snr)
    X <- as.data.frame(X)
    X$x4 <- factor(X$x4, ordered = T)
    X$x5 <- factor(X$x5, ordered = T)
    
    # Split data into test and train sets
    test_set_pct <- 0.5
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
    
    # Warmstart BCF
    start_time <- proc.time()
    num_gfr <- gfr_iter
    num_burnin <- 0
    num_mcmc <- mcmc_iter
    num_samples <- num_gfr + num_burnin + num_mcmc
    bcf_model_warmstart <- bcf(
        X_train = X_train, Z_train = Z_train, y_train = y_train, pi_train = pi_train, 
        X_test = X_test, Z_test = Z_test, pi_test = pi_test, 
        num_gfr = num_gfr, num_burnin = num_burnin, num_mcmc = num_mcmc, 
        sample_sigma_leaf_mu = sample_tau, sample_sigma_leaf_tau = F, 
        num_trees_mu = 200, num_trees_tau = 50
    )
    sample_inds <- (num_gfr + num_burnin + 1):num_samples
    bcf_warmstart_rmse <- sqrt(mean((rowMeans(bcf_model_warmstart$tau_hat_test[,sample_inds]) - tau_test)^2))
    test_lb <- apply(bcf_model_warmstart$tau_hat_test[,sample_inds], 1, quantile, alpha/2)
    test_ub <- apply(bcf_model_warmstart$tau_hat_test[,sample_inds], 1, quantile, 1 - (alpha/2))
    cover <- (
        (test_lb <= tau_test) & (test_ub >= tau_test)
    )
    bcf_warmstart_cover <- mean(cover)
    bcf_warmstart_interval_len <- mean(test_ub - test_lb)
    end_time <- proc.time()
    bcf_warmstart_runtime <- end_time[3] - start_time[3]
    
    # Regular BCF
    start_time <- proc.time()
    num_gfr <- 0
    num_burnin <- burnin_iter
    num_mcmc <- mcmc_iter
    num_samples <- num_gfr + num_burnin + num_mcmc
    bcf_model_root <- bcf(
        X_train = X_train, Z_train = Z_train, y_train = y_train, pi_train = pi_train, 
        X_test = X_test, Z_test = Z_test, pi_test = pi_test, 
        num_gfr = num_gfr, num_burnin = num_burnin, num_mcmc = num_mcmc, 
        sample_sigma_leaf_mu = sample_tau, sample_sigma_leaf_tau = F, 
        num_trees_mu = 200, num_trees_tau = 50
    )
    sample_inds <- (num_gfr + num_burnin + 1):num_samples
    bcf_root_rmse <- sqrt(mean((rowMeans(bcf_model_root$tau_hat_test[,sample_inds]) - tau_test)^2))
    test_lb <- apply(bcf_model_root$tau_hat_test[,sample_inds], 1, quantile, alpha/2)
    test_ub <- apply(bcf_model_root$tau_hat_test[,sample_inds], 1, quantile, 1 - (alpha/2))
    cover <- (
        (test_lb <= tau_test) & (test_ub >= tau_test)
    )
    bcf_root_cover <- mean(cover)
    bcf_root_interval_len <- mean(test_ub - test_lb)
    end_time <- proc.time()
    bcf_root_runtime <- end_time[3] - start_time[3]
    
    # multibart BCF
    start_time <- proc.time()
    num_gfr <- 0
    num_burnin <- burnin_iter
    num_mcmc <- mcmc_iter
    num_samples <- num_gfr + num_burnin + num_mcmc
    bcf_fit <- multibart::bcf_binary(y_train, Z_train, X_train, X_train, pi_train, nburn=num_burnin, nsim=num_mcmc, nthin=1)
    tau_fit <- bcf_fit$moderate_fit
    tau_post <- t(multibart::get_forest_fit(tau_fit, X_test))
    multibart_bcf_rmse <- sqrt(mean((rowMeans(tau_post) - tau_test)^2))
    test_lb <- apply(tau_post, 1, quantile, alpha/2)
    test_ub <- apply(tau_post, 1, quantile, 1 - (alpha/2))
    cover <- (
        (test_lb <= tau_test) & (test_ub >= tau_test)
    )
    multibart_bcf_cover <- mean(cover)
    multibart_bcf_len <- mean(test_ub - test_lb)
    end_time <- proc.time()
    multibart_bcf_runtime <- end_time[3] - start_time[3]
    
    # Aggregating all results
    rmse_results <- c(bcf_warmstart_rmse, bcf_root_rmse, multibart_bcf_rmse)
    cover_results <- c(bcf_warmstart_cover, bcf_root_cover, multibart_bcf_cover)
    interval_len_results <- c(bcf_warmstart_interval_len, bcf_root_interval_len, 
                              multibart_bcf_len)
    runtime_results <- c(bcf_warmstart_runtime, bcf_root_runtime, multibart_bcf_runtime)
    
    # Output results as a row vector per simulation
    return(c(rmse_results, cover_results, interval_len_results, runtime_results))
}

# Capture the project directory using "here"
project_dir = here()

# Create "outputs / snapshots / datestamp" subdirectory, if doesn't exist
datestamp = format(Sys.time(), "%Y%m%d%H%M")
outputs_folder = file.path(project_dir, "tools", "simulations", "outputs")
ifelse(!dir.exists(outputs_folder), dir.create(outputs_folder), FALSE)
snapshots_timestamp_subfolder = file.path(outputs_folder, datestamp)
ifelse(!dir.exists(snapshots_timestamp_subfolder), dir.create(snapshots_timestamp_subfolder), FALSE)

# Column output names
method_names <- c("stochtree_bcf_warmstart", "stochtree_bcf", "multibart_bcf")
column_names <- c(
    paste0(method_names,"_rmse"),
    paste0(method_names,"_coverage"),
    paste0(method_names,"_interval_length"),
    paste0(method_names,"_runtime")
)

# Simulation config
n <- 500
n_sim <- 20

# Setup a parallel cluster
cl <- makeCluster(10)
registerDoParallel(cl)

# Run the simulation with tau fixed
result_fixed <- foreach(i = 1:n_sim, .combine = rbind, .packages = c("stochtree", "multibart")) %dopar% {
    set.seed(i)
    simulation_function(n, mu=mu2, tau=tau2, gfr_iter=20, burnin_iter=2000, 
                        mcmc_iter=1000, alpha=0.05, snr=3, sample_tau = F)
}

# Save results
file_name_fixed = "bcf_comparison_tau_fixed.csv"
snapshot_file_fixed = file.path(snapshots_timestamp_subfolder, file_name_fixed)
colnames(result_fixed) <- column_names
write.csv(result_fixed, snapshot_file_fixed, row.names=F)

# Run the simulation with tau random
result_random <- foreach(i = 1:n_sim, .combine = rbind, .packages = c("stochtree", "multibart")) %dopar% {
    set.seed(i)
    simulation_function(n, mu=mu2, tau=tau2, gfr_iter=20, burnin_iter=2000,
                        mcmc_iter=1000, alpha=0.05, snr=3, sample_tau=T)
}

# Save results
file_name_random = "bcf_comparison_tau_random.csv"
snapshot_file_random = file.path(snapshots_timestamp_subfolder, file_name_random)
colnames(result_random) <- column_names
write.csv(result_random, snapshot_file_random, row.names = F)

# Close the cluster
stopCluster(cl)

# RMSE
colMeans(result_fixed[,1:3])
colMeans(result_random[,1:3])

# Coverage
colMeans(result_fixed[,4:6])
colMeans(result_random[,4:6])

# Interval length
colMeans(result_fixed[,7:9])
colMeans(result_random[,7:9])

# Runtime
colMeans(result_fixed[,10:12])
colMeans(result_random[,10:12])
