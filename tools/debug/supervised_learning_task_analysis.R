################################################################################
## Run stochastic tree ensemble models on data for a supervised learning task 
## and inspect their performance in terms of:
##   (a) Run time (s)
##   (b) Train set RMSE
##   (c) Test set RMSE
################################################################################

# Setup
library(stochtree)
library(BART)
library(dbarts)
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
    # tau_init <- 0.1
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

dispatch_stochtree_run <- function(num_gfr, num_burnin, num_mcmc_retained, param_list, data_list, random_seed = NULL) {
    # Start timer
    start_time <- proc.time()
    
    # Data
    if (param_list$leaf_regression) {
        forest_dataset_train <- createForestDataset(data_list$X_train, data_list$W_train)
        forest_dataset_test <- createForestDataset(data_list$X_test, data_list$W_test)
        outcome_model_type <- 1
    } else {
        forest_dataset_train <- createForestDataset(data_list$X_train)
        forest_dataset_test <- createForestDataset(data_list$X_test)
        outcome_model_type <- 0
    }
    outcome_train <- createOutcome(data_list$resid_train)
    
    # Random number generator (std::mt19937)
    if (is.null(random_seed)) {random_seed = sample(1:10000,1,F)}
    rng <- createRNG(random_seed)
    
    # Sampling data structures
    forest_model <- createForestModel(forest_dataset_train, param_list$feature_types, param_list$num_trees, nrow(data_list$X_train), param_list$alpha, param_list$beta, param_list$min_samples_leaf)
    
    # Container of forest samples
    if (param_list$leaf_regression) {
        forest_samples <- createForestContainer(param_list$num_trees, 1, F)
    } else {
        forest_samples <- createForestContainer(param_list$num_trees, 1, T)
    }
    
    # Container of variance parameter samples
    num_samples <- num_gfr + num_burnin + num_mcmc_retained
    global_var_samples <- c(param_list$global_variance_init, rep(0, num_samples))
    leaf_scale_samples <- c(param_list$tau_init, rep(0, num_samples))
    
    # Run GFR (warm start) if specified
    if (num_gfr > 0){
        for (i in 1:num_gfr) {
            forest_model$sample_one_iteration(
                forest_dataset_train, outcome_train, forest_samples, rng, param_list$feature_types, 
                outcome_model_type, param_list$leaf_prior_scale, param_list$var_weights, 
                global_var_samples[i], param_list$cutpoint_grid_size, gfr = T
            )
            global_var_samples[i+1] <- sample_sigma2_one_iteration(outcome_train, rng, param_list$nu, param_list$lambda)
            leaf_scale_samples[i+1] <- sample_tau_one_iteration(forest_samples, rng, param_list$a_leaf, param_list$b_leaf, i-1)
            param_list$leaf_prior_scale[1,1] <- leaf_scale_samples[i+1]
        }
    }
    
    # Run MCMC
    for (i in (num_gfr+1):num_samples) {
        forest_model$sample_one_iteration(
            forest_dataset_train, outcome_train, forest_samples, rng, param_list$feature_types, 
            outcome_model_type, param_list$leaf_prior_scale, param_list$var_weights, 
            global_var_samples[i], param_list$cutpoint_grid_size, gfr = F
        )
        global_var_samples[i+1] <- sample_sigma2_one_iteration(outcome_train, rng, param_list$nu, param_list$lambda)
        # leaf_scale_samples[i+1] <- sample_tau_one_iteration(forest_samples, rng, param_list$a_leaf, param_list$b_leaf, i-1)
        # param_list$leaf_prior_scale[1,1] <- leaf_scale_samples[i+1]
    }
    
    # Forest predictions
    train_preds <- forest_samples$predict(forest_dataset_train)*data_list$y_std_train + data_list$y_bar_train
    test_preds <- forest_samples$predict(forest_dataset_test)*data_list$y_std_test + data_list$y_bar_test
    
    # End timer and measure run time
    end_time <- proc.time()
    runtime <- end_time[3] - start_time[3]
    
    # Global error variance
    sigma_samples <- sqrt(global_var_samples)*data_list$y_std_train
    
    # RMSEs (post-burnin)
    train_rmse <- sqrt(mean((rowMeans(train_preds[,(num_gfr+num_burnin+1):num_samples]) - data_list$y_train)^2))
    test_rmse <- sqrt(mean((rowMeans(test_preds[,(num_gfr+num_burnin+1):num_samples]) - data_list$y_test)^2))
    
    return(c(runtime,train_rmse,test_rmse))
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
        sample_tau = F, random_seed = 1234
        # random_seed = 1234, nu = 16
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

# Run the code
# DGP 1 - Run 1
dgp_name <- "partitioned_linear_model"
plm_data <- generate_data(dgp_name, n = 1000, p_x = 10, p_w = 1, snr = NULL, test_set_pct = 0.2)
warmstart_stochtree_results <- stochtree_analysis(
    plm_data$resid_train, plm_data$resid_test, plm_data$y_train, plm_data$y_test,
    plm_data$X_train, plm_data$X_test, plm_data$y_bar_train, plm_data$y_bar_test,
    plm_data$y_std_train, plm_data$y_std_test, plm_data$n, plm_data$n_train, plm_data$n_test,
    num_gfr = 10, num_burnin = 0, num_mcmc_retained = 100, W_train = plm_data$W_train,
    W_test = plm_data$W_test, random_seed = NULL
)
gc()
mcmc_stochtree_results <- stochtree_analysis(
    plm_data$resid_train, plm_data$resid_test, plm_data$y_train, plm_data$y_test,
    plm_data$X_train, plm_data$X_test, plm_data$y_bar_train, plm_data$y_bar_test,
    plm_data$y_std_train, plm_data$y_std_test, plm_data$n, plm_data$n_train, plm_data$n_test,
    num_gfr = 0, num_burnin = 2000, num_mcmc_retained = 2000, W_train = plm_data$W_train,
    W_test = plm_data$W_test, random_seed = NULL
)
gc()
wrapped_bart_warmstart_stochtree_results <- wrapped_bart_stochtree_analysis(
    plm_data$resid_train, plm_data$resid_test, plm_data$y_train, plm_data$y_test, 
    plm_data$X_train, plm_data$X_test, plm_data$y_bar_train, plm_data$y_bar_test, 
    plm_data$y_std_train, plm_data$y_std_test, plm_data$n, plm_data$n_train, plm_data$n_test, 
    num_gfr = 10, num_burnin = 0, num_mcmc_retained = 100, W_train = plm_data$W_train, 
    W_test = plm_data$W_test, random_seed = NULL
)
gc()
wrapped_bart_mcmc_stochtree_results <- wrapped_bart_stochtree_analysis(
    plm_data$resid_train, plm_data$resid_test, plm_data$y_train, plm_data$y_test, 
    plm_data$X_train, plm_data$X_test, plm_data$y_bar_train, plm_data$y_bar_test, 
    plm_data$y_std_train, plm_data$y_std_test, plm_data$n, plm_data$n_train, plm_data$n_test, 
    num_gfr = 0, num_burnin = 2000, num_mcmc_retained = 2000, W_train = plm_data$W_train, 
    W_test = plm_data$W_test, random_seed = NULL
)
gc()
mcmc_wbart_results <- wbart_analysis(
    plm_data$resid_train, plm_data$resid_test, plm_data$y_train, plm_data$y_test,
    plm_data$X_train, plm_data$X_test, plm_data$y_bar_train, plm_data$y_bar_test,
    plm_data$y_std_train, plm_data$y_std_test, plm_data$n, plm_data$n_train,
    plm_data$n_test, num_burnin = 2000, num_mcmc_retained = 2000,
    W_train = plm_data$W_train, W_test = plm_data$W_test, random_seed = NULL
)
results_dgp1a <- rbind(warmstart_stochtree_results, mcmc_stochtree_results, wrapped_bart_warmstart_stochtree_results, wrapped_bart_mcmc_stochtree_results, mcmc_wbart_results)
results_dgp1a <- cbind(results_dgp1a, plm_data$snr, dgp_name,
                      c("stochtree_warm_start", "stochtree_mcmc", "bart_stochtree_warm_start", "bart_stochtree_mcmc", "wbart_mcmc"))
cat("DGP 1 out of 2 - Run 1 out of 2\n")

# DGP 1 - Run 2
plm_data <- generate_data(dgp_name, n = 1000, p_x = 10, p_w = 1, snr = NULL, test_set_pct = 0.2)
warmstart_stochtree_results <- stochtree_analysis(
    plm_data$resid_train, plm_data$resid_test, plm_data$y_train, plm_data$y_test,
    plm_data$X_train, plm_data$X_test, plm_data$y_bar_train, plm_data$y_bar_test,
    plm_data$y_std_train, plm_data$y_std_test, plm_data$n, plm_data$n_train, plm_data$n_test,
    num_gfr = 10, num_burnin = 0, num_mcmc_retained = 100, W_train = plm_data$W_train,
    W_test = plm_data$W_test, random_seed = NULL
)
gc()
mcmc_stochtree_results <- stochtree_analysis(
    plm_data$resid_train, plm_data$resid_test, plm_data$y_train, plm_data$y_test,
    plm_data$X_train, plm_data$X_test, plm_data$y_bar_train, plm_data$y_bar_test,
    plm_data$y_std_train, plm_data$y_std_test, plm_data$n, plm_data$n_train, plm_data$n_test,
    num_gfr = 0, num_burnin = 2000, num_mcmc_retained = 2000, W_train = plm_data$W_train,
    W_test = plm_data$W_test, random_seed = NULL
)
gc()
wrapped_bart_warmstart_stochtree_results <- wrapped_bart_stochtree_analysis(
    plm_data$resid_train, plm_data$resid_test, plm_data$y_train, plm_data$y_test,
    plm_data$X_train, plm_data$X_test, plm_data$y_bar_train, plm_data$y_bar_test,
    plm_data$y_std_train, plm_data$y_std_test, plm_data$n, plm_data$n_train, plm_data$n_test,
    num_gfr = 10, num_burnin = 0, num_mcmc_retained = 100, W_train = plm_data$W_train,
    W_test = plm_data$W_test, random_seed = NULL
)
gc()
wrapped_bart_mcmc_stochtree_results <- wrapped_bart_stochtree_analysis(
    plm_data$resid_train, plm_data$resid_test, plm_data$y_train, plm_data$y_test,
    plm_data$X_train, plm_data$X_test, plm_data$y_bar_train, plm_data$y_bar_test,
    plm_data$y_std_train, plm_data$y_std_test, plm_data$n, plm_data$n_train, plm_data$n_test,
    num_gfr = 0, num_burnin = 2000, num_mcmc_retained = 2000, W_train = plm_data$W_train,
    W_test = plm_data$W_test, random_seed = NULL
)
gc()
mcmc_wbart_results <- wbart_analysis(
    plm_data$resid_train, plm_data$resid_test, plm_data$y_train, plm_data$y_test,
    plm_data$X_train, plm_data$X_test, plm_data$y_bar_train, plm_data$y_bar_test,
    plm_data$y_std_train, plm_data$y_std_test, plm_data$n, plm_data$n_train,
    plm_data$n_test, num_burnin = 2000, num_mcmc_retained = 2000,
    W_train = plm_data$W_train, W_test = plm_data$W_test, random_seed = NULL
)
results_dgp1b <- rbind(warmstart_stochtree_results, mcmc_stochtree_results, wrapped_bart_warmstart_stochtree_results, wrapped_bart_mcmc_stochtree_results, mcmc_wbart_results)
results_dgp1b <- cbind(results_dgp1b, plm_data$snr, dgp_name,
                       c("stochtree_warm_start", "stochtree_mcmc", "bart_stochtree_warm_start", "bart_stochtree_mcmc", "wbart_mcmc"))
cat("DGP 1 out of 2 - Run 2 out of 2\n")

# DGP 2 - Run 1
dgp_name <- "step_function"
stpfn_data <- generate_data(dgp_name, n = 1000, p_x = 10, p_w = NULL, snr = NULL, test_set_pct = 0.2)
warmstart_stochtree_results <- stochtree_analysis(
    stpfn_data$resid_train, stpfn_data$resid_test, stpfn_data$y_train, stpfn_data$y_test,
    stpfn_data$X_train, stpfn_data$X_test, stpfn_data$y_bar_train, stpfn_data$y_bar_test,
    stpfn_data$y_std_train, stpfn_data$y_std_test, stpfn_data$n, stpfn_data$n_train, stpfn_data$n_test,
    num_gfr = 10, num_burnin = 0, num_mcmc_retained = 100, W_train = stpfn_data$W_train,
    W_test = stpfn_data$W_test, random_seed = NULL
)
gc()
mcmc_stochtree_results <- stochtree_analysis(
    stpfn_data$resid_train, stpfn_data$resid_test, stpfn_data$y_train, stpfn_data$y_test,
    stpfn_data$X_train, stpfn_data$X_test, stpfn_data$y_bar_train, stpfn_data$y_bar_test,
    stpfn_data$y_std_train, stpfn_data$y_std_test, stpfn_data$n, stpfn_data$n_train, stpfn_data$n_test,
    num_gfr = 0, num_burnin = 2000, num_mcmc_retained = 2000, W_train = stpfn_data$W_train,
    W_test = stpfn_data$W_test, random_seed = NULL
)
gc()
wrapped_bart_warmstart_stochtree_results <- wrapped_bart_stochtree_analysis(
    stpfn_data$resid_train, stpfn_data$resid_test, stpfn_data$y_train, stpfn_data$y_test,
    stpfn_data$X_train, stpfn_data$X_test, stpfn_data$y_bar_train, stpfn_data$y_bar_test,
    stpfn_data$y_std_train, stpfn_data$y_std_test, stpfn_data$n, stpfn_data$n_train, stpfn_data$n_test,
    num_gfr = 10, num_burnin = 0, num_mcmc_retained = 100, W_train = stpfn_data$W_train,
    W_test = stpfn_data$W_test, random_seed = NULL
)
gc()
wrapped_bart_mcmc_stochtree_results <- wrapped_bart_stochtree_analysis(
    stpfn_data$resid_train, stpfn_data$resid_test, stpfn_data$y_train, stpfn_data$y_test,
    stpfn_data$X_train, stpfn_data$X_test, stpfn_data$y_bar_train, stpfn_data$y_bar_test,
    stpfn_data$y_std_train, stpfn_data$y_std_test, stpfn_data$n, stpfn_data$n_train, stpfn_data$n_test,
    num_gfr = 0, num_burnin = 2000, num_mcmc_retained = 2000, W_train = stpfn_data$W_train,
    W_test = stpfn_data$W_test, random_seed = NULL
)
gc()
mcmc_wbart_results <- wbart_analysis(
    stpfn_data$resid_train, stpfn_data$resid_test, stpfn_data$y_train, stpfn_data$y_test,
    stpfn_data$X_train, stpfn_data$X_test, stpfn_data$y_bar_train, stpfn_data$y_bar_test,
    stpfn_data$y_std_train, stpfn_data$y_std_test, stpfn_data$n, stpfn_data$n_train,
    stpfn_data$n_test, num_burnin = 2000, num_mcmc_retained = 2000,
    W_train = stpfn_data$W_train, W_test = stpfn_data$W_test, random_seed = NULL
)
results_dgp2a <- rbind(warmstart_stochtree_results, mcmc_stochtree_results, wrapped_bart_warmstart_stochtree_results, wrapped_bart_mcmc_stochtree_results, mcmc_wbart_results)
results_dgp2a <- cbind(results_dgp2a, stpfn_data$snr, dgp_name,
                      c("stochtree_warm_start", "stochtree_mcmc", "bart_stochtree_warm_start", "bart_stochtree_mcmc", "wbart_mcmc"))
cat("DGP 2 out of 2 - Run 1 out of 2\n")

# DGP 2 - Run 2
stpfn_data <- generate_data(dgp_name, n = 1000, p_x = 10, p_w = NULL, snr = NULL, test_set_pct = 0.2)
warmstart_stochtree_results <- stochtree_analysis(
    stpfn_data$resid_train, stpfn_data$resid_test, stpfn_data$y_train, stpfn_data$y_test,
    stpfn_data$X_train, stpfn_data$X_test, stpfn_data$y_bar_train, stpfn_data$y_bar_test,
    stpfn_data$y_std_train, stpfn_data$y_std_test, stpfn_data$n, stpfn_data$n_train, stpfn_data$n_test,
    num_gfr = 10, num_burnin = 0, num_mcmc_retained = 100, W_train = stpfn_data$W_train,
    W_test = stpfn_data$W_test, random_seed = NULL
)
gc()
mcmc_stochtree_results <- stochtree_analysis(
    stpfn_data$resid_train, stpfn_data$resid_test, stpfn_data$y_train, stpfn_data$y_test,
    stpfn_data$X_train, stpfn_data$X_test, stpfn_data$y_bar_train, stpfn_data$y_bar_test,
    stpfn_data$y_std_train, stpfn_data$y_std_test, stpfn_data$n, stpfn_data$n_train, stpfn_data$n_test,
    num_gfr = 0, num_burnin = 2000, num_mcmc_retained = 2000, W_train = stpfn_data$W_train,
    W_test = stpfn_data$W_test, random_seed = NULL
)
gc()
wrapped_bart_warmstart_stochtree_results <- wrapped_bart_stochtree_analysis(
    stpfn_data$resid_train, stpfn_data$resid_test, stpfn_data$y_train, stpfn_data$y_test,
    stpfn_data$X_train, stpfn_data$X_test, stpfn_data$y_bar_train, stpfn_data$y_bar_test,
    stpfn_data$y_std_train, stpfn_data$y_std_test, stpfn_data$n, stpfn_data$n_train, stpfn_data$n_test,
    num_gfr = 10, num_burnin = 0, num_mcmc_retained = 100, W_train = stpfn_data$W_train,
    W_test = stpfn_data$W_test, random_seed = NULL
)
gc()
wrapped_bart_mcmc_stochtree_results <- wrapped_bart_stochtree_analysis(
    stpfn_data$resid_train, stpfn_data$resid_test, stpfn_data$y_train, stpfn_data$y_test,
    stpfn_data$X_train, stpfn_data$X_test, stpfn_data$y_bar_train, stpfn_data$y_bar_test,
    stpfn_data$y_std_train, stpfn_data$y_std_test, stpfn_data$n, stpfn_data$n_train, stpfn_data$n_test,
    num_gfr = 0, num_burnin = 2000, num_mcmc_retained = 2000, W_train = stpfn_data$W_train,
    W_test = stpfn_data$W_test, random_seed = NULL
)
gc()
mcmc_wbart_results <- wbart_analysis(
    stpfn_data$resid_train, stpfn_data$resid_test, stpfn_data$y_train, stpfn_data$y_test,
    stpfn_data$X_train, stpfn_data$X_test, stpfn_data$y_bar_train, stpfn_data$y_bar_test,
    stpfn_data$y_std_train, stpfn_data$y_std_test, stpfn_data$n, stpfn_data$n_train,
    stpfn_data$n_test, num_burnin = 2000, num_mcmc_retained = 2000,
    W_train = stpfn_data$W_train, W_test = stpfn_data$W_test, random_seed = NULL
)
results_dgp2b <- rbind(warmstart_stochtree_results, mcmc_stochtree_results, wrapped_bart_warmstart_stochtree_results, wrapped_bart_mcmc_stochtree_results, mcmc_wbart_results)
results_dgp2b <- cbind(results_dgp2b, stpfn_data$snr, dgp_name,
                      c("stochtree_warm_start", "stochtree_mcmc", "bart_stochtree_warm_start", "bart_stochtree_mcmc", "wbart_mcmc"))
cat("DGP 2 out of 2 - Run 2 out of 2\n")

results_df <- data.frame(rbind(results_dgp1a, results_dgp1b, results_dgp2a, results_dgp2b))
colnames(results_df) <- c("runtime", "train_rmse", "test_rmse", "snr", "dgp", "model_type")
rownames(results_df) <- 1:nrow(results_df)
results_df <- results_df[,c("dgp", "model_type", "snr", "runtime", "train_rmse", "test_rmse")]
results_df
