# Load libraries
library(stochtree)

# Define DGPs
dgp1 <- function(n, p, snr) {
    X <- matrix(runif(n*p), ncol = p)
    plm_term <- (
        ((0 <= X[,1]) & (0.25 > X[,1])) * (-7.5*X[,2]) + 
        ((0.25 <= X[,1]) & (0.5 > X[,1])) * (-2.5*X[,2]) + 
        ((0.5 <= X[,1]) & (0.75 > X[,1])) * (2.5*X[,2]) + 
        ((0.75 <= X[,1]) & (1 > X[,1])) * (7.5*X[,2])
    )
    trig_term <- (
        2*sin(X[,3]*2*pi) - 
        1.5*cos(X[,4]*2*pi)
    )
    f_XW <- plm_term + trig_term
    noise_sd <- sd(f_XW)/snr
    y <- f_XW + rnorm(n, 0, noise_sd)
    return(list(covariates = X, basis = NULL, outcome = y, conditional_mean = f_XW, 
                rfx_group_ids = NULL, rfx_basis = NULL))
}
dgp2 <- function(n, p, snr) {
    X <- matrix(runif(n*p), ncol = p)
    W <- matrix(runif(n*2), ncol = 2)
    plm_term <- (
        ((0 <= X[,1]) & (0.25 > X[,1])) * (-7.5*W[,1]) + 
        ((0.25 <= X[,1]) & (0.5 > X[,1])) * (-2.5*W[,1]) + 
        ((0.5 <= X[,1]) & (0.75 > X[,1])) * (2.5*W[,1]) + 
        ((0.75 <= X[,1]) & (1 > X[,1])) * (7.5*W[,1])
    )
    trig_term <- (
        2*sin(X[,3]*2*pi) - 
        1.5*cos(X[,4]*2*pi)
    )
    f_XW <- plm_term + trig_term
    noise_sd <- sd(f_XW)/snr
    y <- f_XW + rnorm(n, 0, noise_sd)
    return(list(covariates = X, basis = W, outcome = y, conditional_mean = f_XW, 
                rfx_group_ids = NULL, rfx_basis = NULL))
}
dgp3 <- function(n, p, snr) {
    X <- matrix(runif(n*p), ncol = p)
    plm_term <- (
        ((0 <= X[,1]) & (0.25 > X[,1])) * (-7.5*X[,2]) + 
        ((0.25 <= X[,1]) & (0.5 > X[,1])) * (-2.5*X[,2]) + 
        ((0.5 <= X[,1]) & (0.75 > X[,1])) * (2.5*X[,2]) + 
        ((0.75 <= X[,1]) & (1 > X[,1])) * (7.5*X[,2])
    )
    trig_term <- (
        2*sin(X[,3]*2*pi) - 
        1.5*cos(X[,4]*2*pi)
    )
    rfx_group_ids <- sample(1:3, size = n, replace = T)
    rfx_coefs <- t(matrix(c(-5, -3, -1, 5, 3, 1), nrow=2, byrow=TRUE))
    rfx_basis <- cbind(1, runif(n, -1, 1))
    rfx_term <- rowSums(rfx_coefs[rfx_group_ids,] * rfx_basis)
    f_XW <- plm_term + trig_term + rfx_term
    noise_sd <- sd(f_XW)/snr
    y <- f_XW + rnorm(n, 0, noise_sd)
    return(list(covariates = X, basis = NULL, outcome = y, conditional_mean = f_XW, 
                rfx_group_ids = rfx_group_ids, rfx_basis = rfx_basis))
}
dgp4 <- function(n, p, snr) {
    X <- matrix(runif(n*p), ncol = p)
    W <- matrix(runif(n*2), ncol = 2)
    plm_term <- (
        ((0 <= X[,1]) & (0.25 > X[,1])) * (-7.5*W[,1]) + 
        ((0.25 <= X[,1]) & (0.5 > X[,1])) * (-2.5*W[,1]) + 
        ((0.5 <= X[,1]) & (0.75 > X[,1])) * (2.5*W[,1]) + 
        ((0.75 <= X[,1]) & (1 > X[,1])) * (7.5*W[,1])
    )
    trig_term <- (
        2*sin(X[,3]*2*pi) - 
        1.5*cos(X[,4]*2*pi)
    )
    rfx_group_ids <- sample(1:3, size = n, replace = T)
    rfx_coefs <- t(matrix(c(-5, -3, -1, 5, 3, 1), nrow=2, byrow=TRUE))
    rfx_basis <- cbind(1, runif(n, -1, 1))
    rfx_term <- rowSums(rfx_coefs[rfx_group_ids,] * rfx_basis)
    f_XW <- plm_term + trig_term + rfx_term
    noise_sd <- sd(f_XW)/snr
    y <- f_XW + rnorm(n, 0, noise_sd)
    return(list(covariates = X, basis = W, outcome = y, conditional_mean = f_XW, 
                rfx_group_ids = rfx_group_ids, rfx_basis = rfx_basis))
}

# Test / train split utilities
compute_test_train_indices <- function(n, test_set_pct) {
    n_test <- round(test_set_pct*n)
    n_train <- n - n_test
    test_inds <- sort(sample(1:n, n_test, replace = FALSE))
    train_inds <- (1:n)[!((1:n) %in% test_inds)]
    return(list(test_inds = test_inds, train_inds = train_inds))
}
subset_data <- function(data, subset_inds) {
    if (is.matrix(data)) {
        return(data[subset_inds,])
    } else {
        return(data[subset_inds])
    }
}

# Capture command line arguments
args <- commandArgs(trailingOnly = T)
if (length(args) > 0){
    n_iter <- as.integer(args[1])
    n <- as.integer(args[2])
    p <- as.integer(args[3])
    num_gfr <- as.integer(args[4])
    num_mcmc <- as.integer(args[5])
    dgp_num <- as.integer(args[6])
    snr <- as.numeric(args[7])
    test_set_pct <- as.numeric(args[8])
    num_threads <- as.integer(args[9])
} else{
    # Default arguments
    n_iter <- 5
    n <- 1000
    p <- 5
    num_gfr <- 10
    num_mcmc <- 100
    dgp_num <- 1
    snr <- 2.0
    test_set_pct <- 0.2
    num_threads <- -1
}
cat("n_iter = ", n_iter, "\nn = ", n, "\np = ", p, "\nnum_gfr = ", num_gfr, 
    "\nnum_mcmc = ", num_mcmc, "\ndgp_num = ", dgp_num, "\nsnr = ", snr, 
    "\ntest_set_pct = ", test_set_pct, "\nnum_threads = ", num_threads, "\n", sep = "")

# Run the performance evaluation
results <- matrix(NA, nrow = n_iter, ncol = 4)
colnames(results) <- c("iter", "rmse", "coverage", "runtime")
for (i in 1:n_iter) {
    # Generate data
    if (dgp_num == 1) {
        data_list <- dgp1(n = n, p = p, snr = snr)
    } else if (dgp_num == 2) {
        data_list <- dgp2(n = n, p = p, snr = snr)
    } else if (dgp_num == 3) {
        data_list <- dgp3(n = n, p = p, snr = snr)
    } else if (dgp_num == 4) {
        data_list <- dgp4(n = n, p = p, snr = snr)
    } else {
        stop("Invalid DGP input")
    }
    covariates <- data_list[['covariates']]
    basis <- data_list[['basis']]
    conditional_mean <- data_list[['conditional_mean']]
    outcome <- data_list[['outcome']]
    rfx_group_ids <- data_list[['rfx_group_ids']]
    rfx_basis <- data_list[['rfx_basis']]
    
    # Split into train / test sets
    subset_inds_list <- compute_test_train_indices(n, test_set_pct)
    test_inds <- subset_inds_list$test_inds
    train_inds <- subset_inds_list$train_inds
    covariates_train <- subset_data(covariates, train_inds)
    covariates_test <- subset_data(covariates, test_inds)
    outcome_train <- subset_data(outcome, train_inds)
    outcome_test <- subset_data(outcome, test_inds)
    conditional_mean_train <- subset_data(conditional_mean, train_inds)
    conditional_mean_test <- subset_data(conditional_mean, test_inds)
    has_basis <- !is.null(basis)
    has_rfx <- !is.null(rfx_group_ids)
    if (has_basis) {
        basis_train <- subset_data(basis, train_inds)
        basis_test <- subset_data(basis, test_inds)
    } else {
        basis_train <- NULL
        basis_test <- NULL
    }
    if (has_rfx) {
        rfx_group_ids_train <- subset_data(rfx_group_ids, train_inds)
        rfx_group_ids_test <- subset_data(rfx_group_ids, test_inds)
        rfx_basis_train <- subset_data(rfx_basis, train_inds)
        rfx_basis_test <- subset_data(rfx_basis, test_inds)
    } else {
        rfx_group_ids_train <- NULL
        rfx_group_ids_test <- NULL
        rfx_basis_train <- NULL
        rfx_basis_test <- NULL
    }
    
    # Run (and time) BART
    bart_timing <- system.time({
        # Sample BART model
        general_params <- list(num_threads = num_threads)
        bart_model <- stochtree::bart(
            X_train = covariates_train, y_train = outcome_train, leaf_basis_train = basis_train, 
            rfx_group_ids_train = rfx_group_ids_train, rfx_basis_train = rfx_basis_train, 
            num_gfr = num_gfr, num_mcmc = num_mcmc, general_params = general_params
        )
        
        # Predict on the test set
        test_preds <- predict(
            bart_model, X = covariates_test, leaf_basis = basis_test, 
            rfx_group_ids = rfx_group_ids_test, rfx_basis = rfx_basis_test
        )
    })[3]
    
    # Compute test set evals
    y_hat_posterior <- test_preds$y_hat
    y_hat_posterior_mean <- rowMeans(y_hat_posterior)
    rmse_test <- sqrt(mean((y_hat_posterior_mean - outcome_test)^2))
    y_hat_posterior_quantile_025 <- apply(y_hat_posterior, 1, function(x) quantile(x, 0.025))
    y_hat_posterior_quantile_975 <- apply(y_hat_posterior, 1, function(x) quantile(x, 0.975))
    covered <- rep(NA, nrow(y_hat_posterior))
    for (j in 1:nrow(y_hat_posterior)) {
        covered[j] <- (
            (conditional_mean_test[j] >= y_hat_posterior_quantile_025[j]) & 
            (conditional_mean_test[j] <= y_hat_posterior_quantile_975[j])
        )
    }
    coverage_test <- mean(covered)

    # Store evaluations
    results[i,] <- c(i, rmse_test, coverage_test, bart_timing)
}

# Wrangle and save results to CSV
results_df <- data.frame(
    cbind(n, p, num_gfr, num_mcmc, dgp_num, snr, test_set_pct, num_threads, results)
)
snr_rounded <- as.integer(snr)
test_set_pct_rounded <- as.integer(test_set_pct*100)
num_threads_clean <- ifelse(num_threads < 0, 0, num_threads)
filename <- paste(
    "stochtree", "bart", "r", "n", n, "p", p, "num_gfr", num_gfr, "num_mcmc", num_mcmc, 
    "dgp_num", dgp_num, "snr", snr_rounded, "test_set_pct", test_set_pct_rounded, 
    "num_threads", num_threads_clean, sep = "_"
)
filename_full <- paste0("tools/regression/bart/stochtree_bart_r_results/", filename, ".csv")
write.csv(x = results_df, file = filename_full, row.names = F)
