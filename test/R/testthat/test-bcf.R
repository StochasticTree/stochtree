test_that("MCMC BCF", {
    skip_on_cran()
    
    # Generate simulated data
    n <- 100
    p <- 5
    X <- matrix(runif(n*p), ncol = p)
    mu_X <- (
        ((0 <= X[,1]) & (0.25 > X[,1])) * (-7.5) + 
        ((0.25 <= X[,1]) & (0.5 > X[,1])) * (-2.5) + 
        ((0.5 <= X[,1]) & (0.75 > X[,1])) * (2.5) + 
        ((0.75 <= X[,1]) & (1 > X[,1])) * (7.5)
    )
    pi_X <- (
        ((0 <= X[,1]) & (0.25 > X[,1])) * (0.2) + 
        ((0.25 <= X[,1]) & (0.5 > X[,1])) * (0.4) + 
        ((0.5 <= X[,1]) & (0.75 > X[,1])) * (0.6) + 
        ((0.75 <= X[,1]) & (1 > X[,1])) * (0.8)
    )
    tau_X <- (
        ((0 <= X[,2]) & (0.25 > X[,2])) * (0.5) + 
        ((0.25 <= X[,2]) & (0.5 > X[,2])) * (1.0) + 
        ((0.5 <= X[,2]) & (0.75 > X[,2])) * (1.5) + 
        ((0.75 <= X[,2]) & (1 > X[,2])) * (2.0)
    )
    Z <- rbinom(n, 1, pi_X)
    noise_sd <- 1
    y <- mu_X + tau_X*Z + rnorm(n, 0, noise_sd)
    test_set_pct <- 0.2
    n_test <- round(test_set_pct*n)
    n_train <- n - n_test
    test_inds <- sort(sample(1:n, n_test, replace = FALSE))
    train_inds <- (1:n)[!((1:n) %in% test_inds)]
    X_test <- X[test_inds,]
    X_train <- X[train_inds,]
    Z_test <- Z[test_inds]
    Z_train <- Z[train_inds]
    pi_test <- pi[test_inds]
    pi_train <- pi[train_inds]
    mu_test <- mu_X[test_inds]
    mu_train <- mu_X[train_inds]
    tau_test <- tau_X[test_inds]
    tau_train <- tau_X[train_inds]
    y_test <- y[test_inds]
    y_train <- y[train_inds]
    
    # 1 chain, no thinning
    general_param_list <- list(num_chains = 1, keep_every = 1)
    expect_no_error(
        bcf_model <- bcf(X_train = X_train, y_train = y_train, Z_train = Z_train, 
                         propensity_train = pi_train, X_test = X_test, Z_test = Z_test, 
                         propensity_test = pi_test, num_gfr = 0, num_burnin = 10, 
                         num_mcmc = 10, general_params = general_param_list)
    )
    
    # 3 chains, no thinning
    general_param_list <- list(num_chains = 3, keep_every = 1)
    expect_no_error(
        bcf_model <- bcf(X_train = X_train, y_train = y_train, Z_train = Z_train, 
                         propensity_train = pi_train, X_test = X_test, Z_test = Z_test, 
                         propensity_test = pi_test, num_gfr = 0, num_burnin = 10, 
                         num_mcmc = 10, general_params = general_param_list)
    )
    
    # 1 chain, thinning
    general_param_list <- list(num_chains = 1, keep_every = 5)
    expect_no_error(
        bcf_model <- bcf(X_train = X_train, y_train = y_train, Z_train = Z_train, 
                         propensity_train = pi_train, X_test = X_test, Z_test = Z_test, 
                         propensity_test = pi_test, num_gfr = 0, num_burnin = 10, 
                         num_mcmc = 10, general_params = general_param_list)
    )
    
    # 3 chains, thinning
    general_param_list <- list(num_chains = 3, keep_every = 5)
    expect_no_error(
        bcf_model <- bcf(X_train = X_train, y_train = y_train, Z_train = Z_train, 
                         propensity_train = pi_train, X_test = X_test, Z_test = Z_test, 
                         propensity_test = pi_test, num_gfr = 0, num_burnin = 10, 
                         num_mcmc = 10, general_params = general_param_list)
    )
})

test_that("GFR BART", {
    skip_on_cran()
    
    # Generate simulated data
    n <- 100
    p <- 5
    X <- matrix(runif(n*p), ncol = p)
    mu_X <- (
        ((0 <= X[,1]) & (0.25 > X[,1])) * (-7.5) + 
            ((0.25 <= X[,1]) & (0.5 > X[,1])) * (-2.5) + 
            ((0.5 <= X[,1]) & (0.75 > X[,1])) * (2.5) + 
            ((0.75 <= X[,1]) & (1 > X[,1])) * (7.5)
    )
    pi_X <- (
        ((0 <= X[,1]) & (0.25 > X[,1])) * (0.2) + 
            ((0.25 <= X[,1]) & (0.5 > X[,1])) * (0.4) + 
            ((0.5 <= X[,1]) & (0.75 > X[,1])) * (0.6) + 
            ((0.75 <= X[,1]) & (1 > X[,1])) * (0.8)
    )
    tau_X <- (
        ((0 <= X[,2]) & (0.25 > X[,2])) * (0.5) + 
            ((0.25 <= X[,2]) & (0.5 > X[,2])) * (1.0) + 
            ((0.5 <= X[,2]) & (0.75 > X[,2])) * (1.5) + 
            ((0.75 <= X[,2]) & (1 > X[,2])) * (2.0)
    )
    Z <- rbinom(n, 1, pi_X)
    noise_sd <- 1
    y <- mu_X + tau_X*Z + rnorm(n, 0, noise_sd)
    test_set_pct <- 0.2
    n_test <- round(test_set_pct*n)
    n_train <- n - n_test
    test_inds <- sort(sample(1:n, n_test, replace = FALSE))
    train_inds <- (1:n)[!((1:n) %in% test_inds)]
    X_test <- X[test_inds,]
    X_train <- X[train_inds,]
    Z_test <- Z[test_inds]
    Z_train <- Z[train_inds]
    pi_test <- pi[test_inds]
    pi_train <- pi[train_inds]
    mu_test <- mu_X[test_inds]
    mu_train <- mu_X[train_inds]
    tau_test <- tau_X[test_inds]
    tau_train <- tau_X[train_inds]
    y_test <- y[test_inds]
    y_train <- y[train_inds]
    
    # 1 chain, no thinning
    general_param_list <- list(num_chains = 1, keep_every = 1)
    expect_no_error(
        bcf_model <- bcf(X_train = X_train, y_train = y_train, Z_train = Z_train, 
                         propensity_train = pi_train, X_test = X_test, Z_test = Z_test, 
                         propensity_test = pi_test, num_gfr = 10, num_burnin = 10, 
                         num_mcmc = 10, general_params = general_param_list)
    )
    
    # 3 chains, no thinning
    general_param_list <- list(num_chains = 3, keep_every = 1)
    expect_no_error(
        bcf_model <- bcf(X_train = X_train, y_train = y_train, Z_train = Z_train, 
                         propensity_train = pi_train, X_test = X_test, Z_test = Z_test, 
                         propensity_test = pi_test, num_gfr = 10, num_burnin = 10, 
                         num_mcmc = 10, general_params = general_param_list)
    )
    
    # 1 chain, thinning
    general_param_list <- list(num_chains = 1, keep_every = 5)
    expect_no_error(
        bcf_model <- bcf(X_train = X_train, y_train = y_train, Z_train = Z_train, 
                         propensity_train = pi_train, X_test = X_test, Z_test = Z_test, 
                         propensity_test = pi_test, num_gfr = 10, num_burnin = 10, 
                         num_mcmc = 10, general_params = general_param_list)
    )
    
    # 3 chains, thinning
    general_param_list <- list(num_chains = 3, keep_every = 5)
    expect_no_error(
        bcf_model <- bcf(X_train = X_train, y_train = y_train, Z_train = Z_train, 
                         propensity_train = pi_train, X_test = X_test, Z_test = Z_test, 
                         propensity_test = pi_test, num_gfr = 10, num_burnin = 10, 
                         num_mcmc = 10, general_params = general_param_list)
    )
    
    # Check for error when more chains than GFR forests
    general_param_list <- list(num_chains = 11, keep_every = 1)
    expect_error(
        bcf_model <- bcf(X_train = X_train, y_train = y_train, Z_train = Z_train, 
                         propensity_train = pi_train, X_test = X_test, Z_test = Z_test, 
                         propensity_test = pi_test, num_gfr = 10, num_burnin = 10, 
                         num_mcmc = 10, general_params = general_param_list)
    )
    
    # Check for error when more chains than GFR forests
    general_param_list <- list(num_chains = 11, keep_every = 5)
    expect_error(
        bcf_model <- bcf(X_train = X_train, y_train = y_train, Z_train = Z_train, 
                         propensity_train = pi_train, X_test = X_test, Z_test = Z_test, 
                         propensity_test = pi_test, num_gfr = 10, num_burnin = 10, 
                         num_mcmc = 10, general_params = general_param_list)
    )
})
