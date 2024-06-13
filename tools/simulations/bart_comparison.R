################################################################################
## Comparison of several stochastic tree ensemble implementations for supervised
## learning in terms of
##   (a) Run time (s)
##   (b) Train set RMSE
##   (c) Test set RMSE
## 
## The implementations we compare are
##   (1) Warm-start BART (StochTree)
##   (2) BART (StochTree)
##   (3) BART (BART)
##   (4) BART (dbarts)
################################################################################

# Setup
library(stochtree)
library(BART)
library(dbarts)
library(here)
library(foreach)
library(doParallel)
library(dplyr)
source("tools/debug/dgps.R")

# Generate dataset
generate_data <- function(dgp_name, n, p_x, p_w = NULL, snr = NULL, test_set_pct = 0.2) {
    # Dispatch the right DGP simulation function
    if (dgp_name == "partitioned_linear_model") {
        data_list <- dgp_prediction_partitioned_lm(n, p_x, p_w, snr)
    } else if (dgp_name == "step_function") {
        data_list <- dgp_prediction_step_function(n, p_x, snr)
    } else {
        stop(paste0("Invalid dgp_name: ", dgp_name))
    }
    
    # Unpack the data
    has_basis <- data_list$has_basis
    y <- data_list$y
    X <- data_list$X
    if (has_basis) {
        W <- data_list$W
    } else {
        W <- NULL
    }
    snr <- data_list$snr
    noise_sd <- data_list$noise_sd
    
    # Run test / train split
    n_test <- round(test_set_pct*n)
    n_train <- n - n_test
    test_inds <- sort(sample(1:n, n_test, replace = F))
    train_inds <- (1:n)[!((1:n) %in% test_inds)]
    
    # Split data into test and train sets
    X_test <- X[test_inds,]
    X_train <- X[train_inds,]
    if (has_basis) {
        W_test <- W[test_inds,]
        W_train <- W[train_inds,]
    } else {
        W_test <- NULL
        W_train <- NULL
    }
    y_test <- y[test_inds]
    y_train <- y[train_inds]
    
    # Standardize outcome separately for test and train
    y_bar_train <- mean(y_train)
    y_std_train <- sd(y_train)
    resid_train <- (y_train-y_bar_train)/y_std_train
    y_bar_test <- mean(y_test)
    y_std_test <- sd(y_test)
    resid_test <- (y_test-y_bar_test)/y_std_test
    
    return(list(
        resid_train = resid_train, resid_test = resid_test, 
        y_train = y_train, y_test = y_test, 
        X_train = X_train, X_test = X_test, 
        W_train = W_train, W_test = W_test, 
        y_bar_train = y_bar_train, y_bar_test = y_bar_test, 
        y_std_train = y_std_train, y_std_test = y_std_test, 
        snr = snr, noise_sd = noise_sd, n = n, 
        n_train = n_train, n_test = n_test
    ))
}

# Performance analysis functions for stochtree
stochtree_analysis <- function(resid_train, resid_test, y_train, y_test, 
                               X_train, X_test, y_bar_train, y_bar_test, 
                               y_std_train, y_std_test, n, n_train, n_test, 
                               num_gfr, num_burnin, num_mcmc_retained, 
                               W_train = NULL, W_test = NULL, random_seed = NULL) {
    # Model parameters
    ntree <- 200
    if (is.null(W_train)) {
        leaf_regression <- F
    } else {
        leaf_regression <- T
    }
    p_x <- ncol(X_train)
    tau_init <- var(y_train) / ntree
    param_list <- list(
        alpha = 0.95, beta = 2, min_samples_leaf = 1, num_trees = ntree, 
        cutpoint_grid_size = 100, global_variance_init = 1.0, tau_init = tau_init, 
        leaf_prior_scale = matrix(c(tau_init), ncol = 1), nu = 16, lambda = 0.25, 
        a_leaf = 3., b_leaf = 0.5 * tau_init, leaf_regression = leaf_regression, 
        feature_types = as.integer(rep(0, p_x)), var_weights = rep(1/p_x, p_x)
    )
    
    # Package the data
    data_list <- list(
        resid_train = resid_train, resid_test = resid_test, 
        y_train = y_train, y_test = y_test, X_train = X_train, X_test = X_test, 
        y_bar_train = y_bar_train, y_bar_test = y_bar_test, y_std_train = y_std_train, 
        y_std_test = y_std_test, W_train = W_train, W_test = W_test
    )
    
    return(dispatch_stochtree_run(num_gfr, num_burnin, num_mcmc_retained, param_list, data_list, random_seed))
}

# Performance analysis functions for stochtree
wrapped_bart_stochtree_analysis <- function(resid_train, resid_test, y_train, y_test, 
                                            X_train, X_test, y_bar_train, y_bar_test, 
                                            y_std_train, y_std_test, n, n_train, n_test, 
                                            num_gfr, num_burnin, num_mcmc_retained, 
                                            W_train = NULL, W_test = NULL, random_seed = NULL) {
    # Start timer
    start_time <- proc.time()
    
    # Run BART
    bart_model <- stochtree::bart(
        X_train = X_train, W_train = W_train, y_train = y_train, 
        X_test = X_test, W_test = W_test, num_trees = 200, num_gfr = num_gfr, 
        num_burnin = num_burnin, num_mcmc = num_mcmc_retained, sample_sigma = T, 
        sample_tau = T, random_seed = 1234
    )
    
    # End timer and measure run time
    end_time <- proc.time()
    runtime <- end_time[3] - start_time[3]
    
    # RMSEs
    num_samples <- num_gfr + num_burnin + num_mcmc_retained
    ypred_mean_train <- rowMeans(bart_model$y_hat_train[,(num_gfr+num_burnin+1):num_samples])
    ypred_mean_test <- rowMeans(bart_model$y_hat_test[,(num_gfr+num_burnin+1):num_samples])
    train_rmse <- sqrt(mean((ypred_mean_train - y_train)^2))
    test_rmse <- sqrt(mean((ypred_mean_test - y_test)^2))
    
    return(c(runtime,train_rmse,test_rmse))
}

# Performance analysis functions for wbart
wbart_analysis <- function(resid_train, resid_test, y_train, y_test, X_train, X_test, 
                           y_bar_train, y_bar_test, y_std_train, y_std_test, 
                           n, n_train, n_test, num_burnin, num_mcmc_retained, 
                           W_train = NULL, W_test = NULL, random_seed = NULL) {
    # Start timer
    start_time <- proc.time()
    
    # Run wbart from the BART (add W to X if W is present, since wbart doesn't support leaf regression)
    ntree <- 200
    alpha <- 0.95
    beta <- 2.0
    if (!is.null(W_train)) {X_train <- cbind(X_train, W_train)}
    if (!is.null(W_test)) {X_test <- cbind(X_test, W_test)}
    bartFit = wbart(X_train,y_train,X_test,power=beta,base=alpha,ntree=ntree,nskip=num_burnin,ndpost=num_mcmc_retained)
    
    # End timer and measure run time
    end_time <- proc.time()
    runtime <- end_time[3] - start_time[3]
    
    # RMSEs
    train_rmse <- sqrt(mean((bartFit$yhat.train.mean - y_train)^2))
    test_rmse <- sqrt(mean((bartFit$yhat.test.mean - y_test)^2))
    
    return(c(runtime,train_rmse,test_rmse))
}

# Performance analysis functions for dbart
dbarts_analysis <- function(resid_train, resid_test, y_train, y_test, X_train, X_test, 
                            y_bar_train, y_bar_test, y_std_train, y_std_test, 
                            n, n_train, n_test, num_burnin, num_mcmc_retained, 
                            W_train = NULL, W_test = NULL, random_seed = NULL) {
    # Start timer
    start_time <- proc.time()
    
    # Run wbart from the BART (add W to X if W is present, since wbart doesn't support leaf regression)
    ntree <- 200
    alpha <- 0.95
    beta <- 2.0
    if (!is.null(W_train)) {X_train <- cbind(X_train, W_train)}
    if (!is.null(W_test)) {X_test <- cbind(X_test, W_test)}
    bartFit = dbarts::bart(X_train,y_train,X_test,power=beta,base=alpha,ntree=ntree,nskip=num_burnin,ndpost=num_mcmc_retained,keeptrees=T)
    
    # End timer and measure run time
    end_time <- proc.time()
    runtime <- end_time[3] - start_time[3]
    
    # RMSEs
    train_rmse <- sqrt(mean((bartFit$yhat.train.mean - y_train)^2))
    test_rmse <- sqrt(mean((bartFit$yhat.test.mean - y_test)^2))
    
    return(c(runtime,train_rmse,test_rmse))
}

# Function that runs the simulation
simulation_function <- function(dgp_name, n=1000, p_x=10, p_w=1, snr=NULL, test_set_pct=0.2) {
    sim_data <- generate_data(dgp_name, n = n, p_x = p_x, p_w = p_w, snr = snr, test_set_pct = test_set_pct)
    wrapped_bart_warmstart_stochtree_results <- wrapped_bart_stochtree_analysis(
        sim_data$resid_train, sim_data$resid_test, sim_data$y_train, sim_data$y_test, 
        sim_data$X_train, sim_data$X_test, sim_data$y_bar_train, sim_data$y_bar_test, 
        sim_data$y_std_train, sim_data$y_std_test, sim_data$n, sim_data$n_train, sim_data$n_test, 
        num_gfr = 10, num_burnin = 0, num_mcmc_retained = 100, W_train = sim_data$W_train, 
        W_test = sim_data$W_test, random_seed = NULL
    )
    gc()
    wrapped_bart_mcmc_stochtree_results <- wrapped_bart_stochtree_analysis(
        sim_data$resid_train, sim_data$resid_test, sim_data$y_train, sim_data$y_test, 
        sim_data$X_train, sim_data$X_test, sim_data$y_bar_train, sim_data$y_bar_test, 
        sim_data$y_std_train, sim_data$y_std_test, sim_data$n, sim_data$n_train, sim_data$n_test, 
        num_gfr = 0, num_burnin = 2000, num_mcmc_retained = 2000, W_train = sim_data$W_train, 
        W_test = sim_data$W_test, random_seed = NULL
    )
    gc()
    mcmc_wbart_results <- wbart_analysis(
        sim_data$resid_train, sim_data$resid_test, sim_data$y_train, sim_data$y_test,
        sim_data$X_train, sim_data$X_test, sim_data$y_bar_train, sim_data$y_bar_test,
        sim_data$y_std_train, sim_data$y_std_test, sim_data$n, sim_data$n_train,
        sim_data$n_test, num_burnin = 2000, num_mcmc_retained = 2000,
        W_train = sim_data$W_train, W_test = sim_data$W_test, random_seed = NULL
    )
    gc()
    mcmc_dbarts_results <- dbarts_analysis(
        sim_data$resid_train, sim_data$resid_test, sim_data$y_train, sim_data$y_test,
        sim_data$X_train, sim_data$X_test, sim_data$y_bar_train, sim_data$y_bar_test,
        sim_data$y_std_train, sim_data$y_std_test, sim_data$n, sim_data$n_train,
        sim_data$n_test, num_burnin = 2000, num_mcmc_retained = 2000,
        W_train = sim_data$W_train, W_test = sim_data$W_test, random_seed = NULL
    )
    results <- rbind(wrapped_bart_warmstart_stochtree_results, 
                     wrapped_bart_mcmc_stochtree_results, 
                     mcmc_wbart_results, mcmc_dbarts_results)
    colnames(results) <- c("runtime", "train_rmse", "test_rmse")
    results_df <- data.frame(results)
    rownames(results_df) <- 1:nrow(results)
    results_df$snr <- sim_data$snr
    results_df$dgp <- dgp_name
    results_df$method <- c("stochtree_warmstart_bart", "stochtree_bart", 
                           "wbart", "dbarts")
    return(results_df)
}

# Simulation config
n_sim <- 20

# Setup a parallel cluster
cl <- makeCluster(10)
registerDoParallel(cl)

# Capture the project directory using "here"
project_dir = here()

# Create "outputs / snapshots / datestamp" subdirectory, if doesn't exist
datestamp = format(Sys.time(), "%Y%m%d%H%M")
outputs_folder = file.path(project_dir, "tools", "simulations", "outputs")
ifelse(!dir.exists(outputs_folder), dir.create(outputs_folder), FALSE)
snapshots_timestamp_subfolder = file.path(outputs_folder, datestamp)
ifelse(!dir.exists(snapshots_timestamp_subfolder), dir.create(snapshots_timestamp_subfolder), FALSE)

# Column names for file output
column_names <- c("Runtime","RMSE_Train","RMSE_Test","SNR","DGP","Method")

# Run the simulation with tau fixed
result_plm <- foreach(i = 1:n_sim, .combine = rbind, .packages = c("BART", "stochtree", "dbarts")) %dopar% {
    # set.seed(i)
    simulation_function("partitioned_linear_model", n=1000, p_x=10, p_w=1, snr=NULL, test_set_pct=0.2)
}

# Run the simulation with tau random
result_stpfn <- foreach(i = 1:n_sim, .combine = rbind, .packages = c("BART", "stochtree", "dbarts")) %dopar% {
    # set.seed(i)
    simulation_function("step_function", n=1000, p_x=10, p_w=1, snr=NULL, test_set_pct=0.2)
}

# Close the cluster
stopCluster(cl)

# Save results
file_name_plm = "bart_sims_plm.csv"
snapshot_file_plm = file.path(snapshots_timestamp_subfolder, file_name_plm)
colnames(result_plm) <- column_names
write.csv(result_plm, snapshot_file_plm, row.names=F)

# Save results
file_name_stpfn = "bart_sims_stpfn.csv"
snapshot_file_stpfn = file.path(snapshots_timestamp_subfolder, file_name_stpfn)
colnames(result_stpfn) <- column_names
write.csv(result_stpfn, snapshot_file_stpfn, row.names=F)

# Inspect results
results_df <- rbind(result_plm, result_stpfn)
results_df %>% group_by(DGP, Method) %>% summarize(test_rmse = mean(RMSE_Test), train_rmse = mean(RMSE_Train), 
                                                   runtime = mean(Runtime), snr = mean(SNR))
