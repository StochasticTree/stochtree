################################################################################
## Unwrapped version of `supervised_learning_task_analysis.R` for easier 
## inspection of outputs and stepping through with a debugger
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
    test_inds <- sort(sample(1:n, n_test, replace = FALSE))
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
wrapped_bart_stochtree_analysis <- function(resid_train, resid_test, y_train, y_test, 
                                            X_train, X_test, y_bar_train, y_bar_test, 
                                            y_std_train, y_std_test, n, n_train, n_test, 
                                            num_gfr, num_burnin, num_mcmc_retained, 
                                            W_train = NULL, W_test = NULL, random_seed = NULL) {
    # Start timer
    start_time <- proc.time()
    
    # Run BART
    bart_model <- stochtree::bart(
        X_train = X_train, leaf_basis_train = W_train, y_train = y_train, 
        X_test = X_test, leaf_basis_test = W_test, num_trees = 200, num_gfr = num_gfr, 
        num_burnin = num_burnin, num_mcmc = num_mcmc_retained, 
        sample_sigma = T, sample_tau = F, random_seed = 1234, nu = 3
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
    
    # Wrap the results and return them
    return(list(
        "model" = bart_model, 
        "metrics" = c(runtime,train_rmse,test_rmse)
    ))
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
    
    return(list(
        "model" = bartFit,
        "metrics" = c(runtime,train_rmse,test_rmse)
    ))
}

# Run a single iteration of both warmstart and MCMC
# dgp_name <- "partitioned_linear_model"
# plm_data <- generate_data(dgp_name, n = 1000, p_x = 10, p_w = 1, snr = NULL, test_set_pct = 0.2)
dgp_name <- "step_function"
sim_data <- generate_data(dgp_name, n = 1000, p_x = 10, p_w = NULL, snr = NULL, test_set_pct = 0.2)
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
    num_gfr = 0, num_burnin = 0, num_mcmc_retained = 4000, W_train = sim_data$W_train,
    W_test = sim_data$W_test, random_seed = NULL
)
gc()
mcmc_wbart_results <- wbart_analysis(
    sim_data$resid_train, sim_data$resid_test, sim_data$y_train, sim_data$y_test,
    sim_data$X_train, sim_data$X_test, sim_data$y_bar_train, sim_data$y_bar_test,
    sim_data$y_std_train, sim_data$y_std_test, sim_data$n, sim_data$n_train,
    sim_data$n_test, num_burnin = 0, num_mcmc_retained = 4000,
    W_train = sim_data$W_train, W_test = sim_data$W_test, random_seed = NULL
)

# Unpack models and compare
mod1 <- wrapped_bart_warmstart_stochtree_results$model
mod2 <- wrapped_bart_mcmc_stochtree_results$model
mod3 <- mcmc_wbart_results$model

# Plot results
sim_iter <- 20
par(mfrow = c(1,3))
plot(mod1$y_hat_train[,sim_iter], sim_data$y_train, main = "Scenario 1: StochTree Warmstart", xlab = "Predicted", ylab = "Actual"); abline(0,1,col="red",lty=3,lwd=2.5)
plot(mod2$y_hat_train[,sim_iter], sim_data$y_train, main = "Scenario 2: StochTree MCMC", xlab = "Predicted", ylab = "Actual"); abline(0,1,col="red",lty=3,lwd=2.5)
plot(mod3$yhat.train[sim_iter,], sim_data$y_train, main = "Scenario 3: BART MCMC", xlab = "Predicted", ylab = "Actual"); abline(0,1,col="red",lty=3,lwd=2.5)
