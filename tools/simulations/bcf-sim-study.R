################################################################################
## Simulation study of BCF / warmstart BCF performance against a host of 
## similar causal effect estimators, in terms of 
##   (a) CATE test set RMSE
##   (b) CATE interval coverage
##   (c) CATE interval length
## 
## The estimators we compare are
##   (1) Warm-start BCF (stochtree)
##   (2) BCF (stochtree)
##   (3) S-Learner (with stochtree::bart for the learner)
##   (4) T-Learner (with stochtree::bart for the learners)
##   (5) GRF
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
        (test_lb <= tau_x[test_inds]) & 
            (test_ub >= tau_x[test_inds])
    )
    bcf_warmstart_cover <- mean(cover)
    bcf_warmstart_interval_len <- mean(test_ub - test_lb)
    
    # Regular BCF
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
        (test_lb <= tau_x[test_inds]) & 
            (test_ub >= tau_x[test_inds])
    )
    bcf_root_cover <- mean(cover)
    bcf_root_interval_len <- mean(test_ub - test_lb)
    
    # S-Learner
    num_gfr <- 0
    num_burnin <- burnin_iter
    num_mcmc <- mcmc_iter
    num_samples <- num_gfr + num_burnin + num_mcmc
    X_train_s_learner <- cbind(X_train, pi_train, Z_train)
    X_test_s_learner <- cbind(X_test, pi_test, Z_test)
    s_learner_warmstart <- stochtree::bart(
        X_train = X_train_s_learner, y_train = y_train, 
        X_test = X_test_s_learner, 
        num_gfr = num_gfr, num_burnin = num_burnin, num_mcmc = num_mcmc, 
        sample_tau = sample_tau, num_trees = 200
    )
    X_test_1 <- cbind(X_test, pi_test, 1)
    X_test_0 <- cbind(X_test, pi_test, 0)
    f1_samples <- predict(s_learner_warmstart, X_test = X_test_1)
    f0_samples <- predict(s_learner_warmstart, X_test = X_test_0)
    tauhat_s_learner <- f1_samples - f0_samples
    sample_inds <- (num_gfr + num_burnin + 1):num_samples
    s_learner_rmse <- sqrt(mean((rowMeans(tauhat_s_learner[,sample_inds]) - tau_test)^2))
    test_lb <- apply(tauhat_s_learner[,sample_inds], 1, quantile, alpha/2)
    test_ub <- apply(tauhat_s_learner[,sample_inds], 1, quantile, 1 - (alpha/2))
    cover <- (
        (test_lb <= tau_x[test_inds]) & 
            (test_ub >= tau_x[test_inds])
    )
    s_learner_cover <- mean(cover)
    s_learner_interval_len <- mean(test_ub - test_lb)
    
    # dbarts S-Learner
    num_gfr <- 0
    num_burnin <- burnin_iter
    num_mcmc <- mcmc_iter
    num_samples <- num_gfr + num_burnin + num_mcmc
    dbarts_s_learner = dbarts::bart(X_train_s_learner,y_train,X_test_s_learner,power=2.0,base=0.95,ntree=200,nskip=num_burnin,ndpost=num_mcmc,keeptrees=T)
    X_test_1 <- cbind(X_test, pi_test, 1)
    X_test_0 <- cbind(X_test, pi_test, 0)
    f1_samples <- predict(dbarts_s_learner, newdata = X_test_1)
    f0_samples <- predict(dbarts_s_learner, newdata = X_test_0)
    tauhat_dbarts_s_learner <- t(f1_samples - f0_samples)
    sample_inds <- (num_gfr + num_burnin + 1):num_samples
    dbarts_s_learner_rmse <- sqrt(mean((rowMeans(tauhat_dbarts_s_learner) - tau_test)^2))
    test_lb <- apply(tauhat_dbarts_s_learner, 1, quantile, alpha/2)
    test_ub <- apply(tauhat_dbarts_s_learner, 1, quantile, 1 - (alpha/2))
    cover <- (
        (test_lb <= tau_test) & (test_ub >= tau_test)
    )
    dbarts_s_learner_cover <- mean(cover)
    dbarts_s_learner_interval_len <- mean(test_ub - test_lb)
    
    # T-Learner
    num_gfr <- 0
    num_burnin <- burnin_iter
    num_mcmc <- mcmc_iter
    num_samples <- num_gfr + num_burnin + num_mcmc
    X_train_treated <- X_train[Z_train==1,]
    X_train_control <- X_train[Z_train==0,]
    y_train_treated <- y_train[Z_train==1]
    y_train_control <- y_train[Z_train==0]
    t_learner_warmstart_treated <- bart(
        X_train = X_train_treated, y_train = y_train_treated, 
        num_gfr = num_gfr, num_trees = 200, 
        num_burnin = num_burnin, num_mcmc = num_mcmc, sample_tau = sample_tau
    )
    t_learner_warmstart_control <- bart(
        X_train = X_train_control, y_train = y_train_control, 
        num_gfr = num_gfr, num_trees = 200, 
        num_burnin = num_burnin, num_mcmc = num_mcmc, sample_tau = sample_tau
    )
    sample_inds <- (num_gfr + num_burnin + 1):num_samples
    f1_samples <- predict(t_learner_warmstart_treated, X_test = X_test)
    f0_samples <- predict(t_learner_warmstart_control, X_test = X_test)
    tauhat_t_learner <- f1_samples - f0_samples
    t_learner_rmse <- sqrt(mean((rowMeans(tauhat_t_learner[,sample_inds]) - tau_test)^2))
    test_lb <- apply(tauhat_t_learner[,sample_inds], 1, quantile, alpha/2)
    test_ub <- apply(tauhat_t_learner[,sample_inds], 1, quantile, 1 - (alpha/2))
    cover <- (
        (test_lb <= tau_x[test_inds]) & 
            (test_ub >= tau_x[test_inds])
    )
    t_learner_cover <- mean(cover)
    t_learner_interval_len <- mean(test_ub - test_lb)
    
    # GRF
    grf_forest <- causal_forest(X_train, y_train, Z_train)
    grf_pred <- predict(grf_forest, X_test, estimate.variance = TRUE)
    grf_rmse <- sqrt(mean((grf_pred$predictions - tau_test)^2))
    z_score <- qnorm(1-alpha/2)
    test_lb <- grf_pred$predictions - z_score*sqrt(grf_pred$variance.estimates)
    test_ub <- grf_pred$predictions + z_score*sqrt(grf_pred$variance.estimates)
    cover <- (
        (test_lb <= tau_x[test_inds]) & 
            (test_ub >= tau_x[test_inds])
    )
    grf_cover <- mean(cover)
    grf_interval_len <- mean(test_ub - test_lb)
    
    # Aggregating all results
    rmse_results <- c(bcf_warmstart_rmse, bcf_root_rmse, s_learner_rmse, 
                      dbarts_s_learner_rmse, t_learner_rmse, grf_rmse)
    cover_results <- c(bcf_warmstart_cover, bcf_root_cover, s_learner_cover, 
                       dbarts_s_learner_cover, t_learner_cover, grf_cover)
    interval_len_results <- c(bcf_warmstart_interval_len, bcf_root_interval_len, 
                              s_learner_interval_len, dbarts_s_learner_interval_len, 
                              t_learner_interval_len, grf_interval_len)
    
  # Output results as a row vector per simulation
  return(c(rmse_results, cover_results, interval_len_results))
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
method_names <- c("bcf_warmstart", "bcf", "s_learner", "dbarts_s_learner", "t_learner", "grf")
column_names <- c(
    paste0(method_names,"_rmse"),
    paste0(method_names,"_coverage"),
    paste0(method_names,"_interval_length")
)

# Simulation config
n <- 500
n_sim <- 20

# Setup a parallel cluster
cl <- makeCluster(10)
registerDoParallel(cl)

# Run the simulation with tau fixed
result_fixed <- foreach(i = 1:n_sim, .combine = rbind, .packages = c("grf", "dbarts", "stochtree")) %dopar% {
    set.seed(i)
    simulation_function(n, mu=mu1, tau=tau2, gfr_iter=20, burnin_iter=2000, 
                        mcmc_iter=1000, alpha=0.05, snr=3, sample_tau = F)
}

# Save results
file_name_fixed = "bcf_simulation_tau_fixed.csv"
snapshot_file_fixed = file.path(snapshots_timestamp_subfolder, file_name_fixed)
colnames(result_fixed) <- column_names
write.csv(result_fixed, snapshot_file_fixed, row.names=F)

# Run the simulation with tau random
result_random <- foreach(i = 1:n_sim, .combine = rbind, .packages = c("grf", "dbarts", "stochtree")) %dopar% {
    set.seed(i)
    simulation_function(n, mu=mu1, tau=tau2, gfr_iter=20, burnin_iter=2000, 
                        mcmc_iter=1000, alpha=0.05, snr=3, sample_tau=T)
}

# Save results
file_name_random = "bcf_simulation_tau_random.csv"
snapshot_file_random = file.path(snapshots_timestamp_subfolder, file_name_random)
colnames(result_random) <- column_names
write.csv(result_random, snapshot_file_random, row.names = F)

# Close the cluster
stopCluster(cl)

# RMSE
metric_cols <- 1:6
colMeans(result_fixed[,metric_cols])
colMeans(result_random[,metric_cols])

# Coverage
metric_cols <- 7:12
colMeans(result_fixed[,metric_cols])
colMeans(result_random[,metric_cols])

# Interval length
metric_cols <- 13:18
colMeans(result_fixed[,metric_cols])
colMeans(result_random[,metric_cols])
