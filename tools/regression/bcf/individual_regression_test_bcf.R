# Load libraries
library(stochtree)

# Define DGPs
dgp1 <- function(n, p, snr) {
    X <- matrix(rnorm(n*p), ncol = p)
    plm_term <- (
        ((0 <= X[,1]) & (0.25 > X[,1])) * (-7.5*X[,2]) + 
        ((0.25 <= X[,1]) & (0.5 > X[,1])) * (-2.5*X[,2]) + 
        ((0.5 <= X[,1]) & (0.75 > X[,1])) * (2.5*X[,2]) + 
        ((0.75 <= X[,1]) & (1 > X[,1])) * (7.5*X[,2])
    )
    trig_term <- (
        2*sin(X[,3]*2*pi) - 
        2*cos(X[,4]*2*pi)
    )
    mu_x <- plm_term + trig_term
    pi_x <- 0.8*pnorm((3*mu_x/sd(mu_x)) - 0.5*X[,1]) + 0.05 + runif(n)/10
    Z <- rbinom(n,1,pi_x)
    tau_x <- 1 + 2*X[,2]*X[,4]
    f_XZ <- mu_x + tau_x * Z
    noise_sd <- sd(f_XZ)/snr
    y <- f_XZ + rnorm(n, 0, noise_sd)
    return(list(covariates = X, treatment = Z, outcome = y, propensity = pi_x,
                prognostic_effect = mu_x, treatment_effect = tau_x, 
                conditional_mean = f_XZ, rfx_group_ids = NULL, rfx_basis = NULL))
}
dgp2 <- function(n, p, snr) {
    X <- matrix(runif(n*p), ncol = p)
    pi_x <- cbind(0.125 + 0.75 * X[, 1], 0.875 - 0.75 * X[, 2])
    mu_x <- pi_x[, 1] * 5 + pi_x[, 2] * 2 + 2 * X[, 3]
    tau_x <- cbind(X[, 2], X[, 3]) * 2
    Z <- matrix(NA_real_, nrow = n, ncol = ncol(pi_x))
    for (i in 1:ncol(pi_x)) {
        Z[, i] <- rbinom(n, 1, pi_x[, i])
    }
    f_XZ <- mu_x + rowSums(Z * tau_x)
    noise_sd <- sd(f_XZ)/snr
    y <- f_XZ + rnorm(n, 0, noise_sd)
    return(list(covariates = X, treatment = Z, outcome = y, propensity = pi_x,
                prognostic_effect = mu_x, treatment_effect = tau_x, 
                conditional_mean = f_XZ, rfx_group_ids = NULL, rfx_basis = NULL))
}
dgp3 <- function(n, p, snr) {
    X <- matrix(rnorm(n*p), ncol = p)
    plm_term <- (
        ((0 <= X[,1]) & (0.25 > X[,1])) * (-7.5*X[,2]) + 
            ((0.25 <= X[,1]) & (0.5 > X[,1])) * (-2.5*X[,2]) + 
            ((0.5 <= X[,1]) & (0.75 > X[,1])) * (2.5*X[,2]) + 
            ((0.75 <= X[,1]) & (1 > X[,1])) * (7.5*X[,2])
    )
    trig_term <- (
        2*sin(X[,3]*2*pi) - 
            2*cos(X[,4]*2*pi)
    )
    mu_x <- plm_term + trig_term
    pi_x <- 0.8*pnorm((3*mu_x/sd(mu_x)) - 0.5*X[,1]) + 0.05 + runif(n)/10
    Z <- rbinom(n,1,pi_x)
    tau_x <- 1 + 2*X[,2]*X[,4]
    rfx_group_ids <- sample(1:3, size = n, replace = T)
    rfx_coefs <- t(matrix(c(-5, -3, -1, 5, 3, 1), nrow=2, byrow=TRUE))
    rfx_basis <- cbind(1, runif(n, -1, 1))
    rfx_term <- rowSums(rfx_coefs[rfx_group_ids,] * rfx_basis)
    f_XZ <- mu_x + tau_x * Z + rfx_term
    noise_sd <- sd(f_XZ)/snr
    y <- f_XZ + rnorm(n, 0, noise_sd)
    return(list(covariates = X, treatment = Z, outcome = y, propensity = pi_x,
                prognostic_effect = mu_x, treatment_effect = tau_x, 
                conditional_mean = f_XZ, rfx_group_ids = rfx_group_ids, rfx_basis = rfx_basis))
}
dgp4 <- function(n, p, snr) {
    X <- matrix(runif(n*p), ncol = p)
    pi_x <- cbind(0.125 + 0.75 * X[, 1], 0.875 - 0.75 * X[, 2])
    mu_x <- pi_x[, 1] * 5 + pi_x[, 2] * 2 + 2 * X[, 3]
    tau_x <- cbind(X[, 2], X[, 3]) * 2
    Z <- matrix(NA_real_, nrow = n, ncol = ncol(pi_x))
    for (i in 1:ncol(pi_x)) {
        Z[, i] <- rbinom(n, 1, pi_x[, i])
    }
    rfx_group_ids <- sample(1:3, size = n, replace = T)
    rfx_coefs <- t(matrix(c(-5, -3, -1, 5, 3, 1), nrow=2, byrow=TRUE))
    rfx_basis <- cbind(1, runif(n, -1, 1))
    rfx_term <- rowSums(rfx_coefs[rfx_group_ids,] * rfx_basis)
    f_XZ <- mu_x + rowSums(Z * tau_x) + rfx_term
    noise_sd <- sd(f_XZ)/snr
    y <- f_XZ + rnorm(n, 0, noise_sd)
    return(list(covariates = X, treatment = Z, outcome = y, propensity = pi_x,
                prognostic_effect = mu_x, treatment_effect = tau_x, 
                conditional_mean = f_XZ, rfx_group_ids = rfx_group_ids, rfx_basis = rfx_basis))
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
results <- matrix(NA, nrow = n_iter, ncol = 6)
colnames(results) <- c("iter", "outcome_rmse", "outcome_coverage", "treatment_effect_rmse", "treatment_effect_coverage", "runtime")
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
    treatment <- data_list[['treatment']]
    propensity <- data_list[['propensity']]
    prognostic_effect <- data_list[['prognostic_effect']]
    treatment_effect <- data_list[['treatment_effect']]
    conditional_mean <- data_list[['conditional_mean']]
    outcome <- data_list[['outcome']]
    rfx_group_ids <- data_list[['rfx_group_ids']]
    rfx_basis <- data_list[['rfx_basis']]
    if (dgp_num %in% c(2,4)) {
        has_multivariate_treatment <- T
    } else {
        has_multivariate_treatment <- F
    }
    
    # Split into train / test sets
    subset_inds_list <- compute_test_train_indices(n, test_set_pct)
    test_inds <- subset_inds_list$test_inds
    train_inds <- subset_inds_list$train_inds
    covariates_train <- subset_data(covariates, train_inds)
    covariates_test <- subset_data(covariates, test_inds)
    treatment_train <- subset_data(treatment, train_inds)
    treatment_test <- subset_data(treatment, test_inds)
    propensity_train <- subset_data(propensity, train_inds)
    propensity_test <- subset_data(propensity, test_inds)
    outcome_train <- subset_data(outcome, train_inds)
    outcome_test <- subset_data(outcome, test_inds)
    prognostic_effect_train <- subset_data(prognostic_effect, train_inds)
    prognostic_effect_test <- subset_data(prognostic_effect, test_inds)
    treatment_effect_train <- subset_data(treatment_effect, train_inds)
    treatment_effect_test <- subset_data(treatment_effect, test_inds)
    conditional_mean_train <- subset_data(conditional_mean, train_inds)
    conditional_mean_test <- subset_data(conditional_mean, test_inds)
    has_rfx <- !is.null(rfx_group_ids)
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
    
    # Run (and time) BCF
    bcf_timing <- system.time({
        # Sample BCF model
        general_params <- list(num_threads = num_threads, adaptive_coding = F)
        prognostic_forest_params <- list(sample_sigma2_leaf = F)
        treatment_effect_forest_params <- list(sample_sigma2_leaf = F)
        bcf_model <- stochtree::bcf(
            X_train = covariates_train, Z_train = treatment_train, 
            propensity_train = propensity_train, y_train = outcome_train, 
            rfx_group_ids_train = rfx_group_ids_train, rfx_basis_train = rfx_basis_train, 
            num_gfr = num_gfr, num_mcmc = num_mcmc, general_params = general_params, 
            prognostic_forest_params = prognostic_forest_params, 
            treatment_effect_forest_params = treatment_effect_forest_params
        )
        
        # Predict on the test set
        test_preds <- predict(
            bcf_model, X = covariates_test, Z = treatment_test, propensity = propensity_test, 
            rfx_group_ids = rfx_group_ids_test, rfx_basis = rfx_basis_test
        )
    })[3]
    
    # Compute test set evals
    y_hat_posterior <- test_preds$y_hat
    y_hat_posterior_mean <- rowMeans(y_hat_posterior)
    tau_hat_posterior <- test_preds$tau_hat
    if (has_multivariate_treatment) tau_hat_posterior_mean <- apply(tau_hat_posterior, c(1,2), mean)
    else tau_hat_posterior_mean <- apply(tau_hat_posterior, 1, mean)
    y_hat_rmse_test <- sqrt(mean((y_hat_posterior_mean - outcome_test)^2))
    tau_hat_rmse_test <- sqrt(mean((tau_hat_posterior_mean - treatment_effect_test)^2))
    y_hat_posterior_quantile_025 <- apply(y_hat_posterior, 1, function(x) quantile(x, 0.025))
    y_hat_posterior_quantile_975 <- apply(y_hat_posterior, 1, function(x) quantile(x, 0.975))
    if (has_multivariate_treatment) {
        tau_hat_posterior_quantile_025 <- apply(tau_hat_posterior, c(1,2), function(x) quantile(x, 0.025))
        tau_hat_posterior_quantile_975 <- apply(tau_hat_posterior, c(1,2), function(x) quantile(x, 0.975))
    } else {
        tau_hat_posterior_quantile_025 <- apply(tau_hat_posterior, 1, function(x) quantile(x, 0.025))
        tau_hat_posterior_quantile_975 <- apply(tau_hat_posterior, 1, function(x) quantile(x, 0.975))
    }
    y_hat_covered <- rep(NA, nrow(y_hat_posterior))
    for (j in 1:nrow(y_hat_posterior)) {
        y_hat_covered[j] <- (
            (conditional_mean_test[j] >= y_hat_posterior_quantile_025[j]) & 
            (conditional_mean_test[j] <= y_hat_posterior_quantile_975[j])
        )
    }
    y_hat_coverage_test <- mean(y_hat_covered)
    if (has_multivariate_treatment) {
        tau_hat_covered <- matrix(NA_real_, nrow(tau_hat_posterior_mean), ncol(tau_hat_posterior_mean))
        for (j in 1:nrow(tau_hat_covered)) {
            for (k in 1:ncol(tau_hat_covered)) {
                tau_hat_covered[j,k] <- (
                    (treatment_effect_test[j,k] >= tau_hat_posterior_quantile_025[j,k]) & 
                    (treatment_effect_test[j,k] <= tau_hat_posterior_quantile_975[j,k])
                )
            }
        }
    } else {
        tau_hat_covered <- rep(NA, nrow(tau_hat_posterior))
        for (j in 1:nrow(tau_hat_posterior)) {
            tau_hat_covered[j] <- (
                (treatment_effect_test[j] >= tau_hat_posterior_quantile_025[j]) & 
                (treatment_effect_test[j] <= tau_hat_posterior_quantile_975[j])
            )
        }
    }
    tau_hat_coverage_test <- mean(tau_hat_covered)

    # Store evaluations
    results[i,] <- c(i, y_hat_rmse_test, y_hat_coverage_test, tau_hat_rmse_test, tau_hat_coverage_test, bcf_timing)
}

# Wrangle and save results to CSV
results_df <- data.frame(
    cbind(n, p, num_gfr, num_mcmc, dgp_num, snr, test_set_pct, num_threads, results)
)
snr_rounded <- as.integer(snr)
test_set_pct_rounded <- as.integer(test_set_pct*100)
num_threads_clean <- ifelse(num_threads < 0, 0, num_threads)
filename <- paste(
    "stochtree", "bcf", "r", "n", n, "p", p, "num_gfr", num_gfr, "num_mcmc", num_mcmc, 
    "dgp_num", dgp_num, "snr", snr_rounded, "test_set_pct", test_set_pct_rounded, 
    "num_threads", num_threads_clean, sep = "_"
)
filename_full <- paste0("tools/regression/bcf/stochtree_bcf_r_results/", filename, ".csv")
write.csv(x = results_df, file = filename_full, row.names = F)
