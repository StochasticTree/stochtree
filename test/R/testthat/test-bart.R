test_that("MCMC BART", {
    skip_on_cran()
    
    # Generate simulated data
    n <- 100
    p <- 5
    X <- matrix(runif(n*p), ncol = p)
    f_XW <- (
        ((0 <= X[,1]) & (0.25 > X[,1])) * (-7.5) + 
        ((0.25 <= X[,1]) & (0.5 > X[,1])) * (-2.5) + 
        ((0.5 <= X[,1]) & (0.75 > X[,1])) * (2.5) + 
        ((0.75 <= X[,1]) & (1 > X[,1])) * (7.5)
    )
    noise_sd <- 1
    y <- f_XW + rnorm(n, 0, noise_sd)
    test_set_pct <- 0.2
    n_test <- round(test_set_pct*n)
    n_train <- n - n_test
    test_inds <- sort(sample(1:n, n_test, replace = FALSE))
    train_inds <- (1:n)[!((1:n) %in% test_inds)]
    X_test <- X[test_inds,]
    X_train <- X[train_inds,]
    y_test <- y[test_inds]
    y_train <- y[train_inds]
    
    # 1 chain, no thinning
    general_param_list <- list(num_chains = 1, keep_every = 1)
    expect_no_error(
        bart_model <- bart(X_train = X_train, y_train = y_train, X_test = X_test, 
                           num_gfr = 0, num_burnin = 10, num_mcmc = 10, 
                           general_params = general_param_list)
    )
    
    # 3 chains, no thinning
    general_param_list <- list(num_chains = 3, keep_every = 1)
    expect_no_error(
        bart_model <- bart(X_train = X_train, y_train = y_train, X_test = X_test, 
                           num_gfr = 0, num_burnin = 10, num_mcmc = 10, 
                           general_params = general_param_list)
    )
    
    # 1 chain, thinning
    general_param_list <- list(num_chains = 1, keep_every = 5)
    expect_no_error(
        bart_model <- bart(X_train = X_train, y_train = y_train, X_test = X_test, 
                           num_gfr = 0, num_burnin = 10, num_mcmc = 10, 
                           general_params = general_param_list)
    )
    
    # 3 chains, thinning
    general_param_list <- list(num_chains = 3, keep_every = 5)
    expect_no_error(
        bart_model <- bart(X_train = X_train, y_train = y_train, X_test = X_test, 
                           num_gfr = 0, num_burnin = 10, num_mcmc = 10, 
                           general_params = general_param_list)
    )
    
    # Generate simulated data with a leaf basis
    n <- 100
    p <- 5
    p_w <- 2
    X <- matrix(runif(n*p), ncol = p)
    W <- matrix(runif(n*p_w), ncol = p_w)
    f_XW <- (
        ((0 <= X[,1]) & (0.25 > X[,1])) * (-7.5*W[,1]) + 
        ((0.25 <= X[,1]) & (0.5 > X[,1])) * (-2.5*W[,1]) + 
        ((0.5 <= X[,1]) & (0.75 > X[,1])) * (2.5*W[,1]) + 
        ((0.75 <= X[,1]) & (1 > X[,1])) * (7.5*W[,1])
    )
    noise_sd <- 1
    y <- f_XW + rnorm(n, 0, noise_sd)
    test_set_pct <- 0.2
    n_test <- round(test_set_pct*n)
    n_train <- n - n_test
    test_inds <- sort(sample(1:n, n_test, replace = FALSE))
    train_inds <- (1:n)[!((1:n) %in% test_inds)]
    X_test <- X[test_inds,]
    X_train <- X[train_inds,]
    W_test <- W[test_inds,]
    W_train <- W[train_inds,]
    y_test <- y[test_inds]
    y_train <- y[train_inds]
    
    # 3 chains, thinning, leaf regression
    general_param_list <- list(num_chains = 3, keep_every = 5)
    mean_forest_param_list <- list(sample_sigma2_leaf = FALSE)
    expect_no_error(
        bart_model <- bart(X_train = X_train, y_train = y_train, X_test = X_test, 
                           leaf_basis_train = W_train, leaf_basis_test = W_test, 
                           num_gfr = 0, num_burnin = 10, num_mcmc = 10, 
                           general_params = general_param_list, 
                           mean_forest_params = mean_forest_param_list)
    )
    
    # 3 chains, thinning, leaf regression with a scalar leaf scale
    general_param_list <- list(num_chains = 3, keep_every = 5)
    mean_forest_param_list <- list(sample_sigma2_leaf = FALSE, sigma2_leaf_init = 0.5)
    expect_no_error(
        bart_model <- bart(X_train = X_train, y_train = y_train, X_test = X_test, 
                           leaf_basis_train = W_train, leaf_basis_test = W_test, 
                           num_gfr = 0, num_burnin = 10, num_mcmc = 10, 
                           general_params = general_param_list, 
                           mean_forest_params = mean_forest_param_list)
    )
    
    # 3 chains, thinning, leaf regression with a scalar leaf scale, random leaf scale
    general_param_list <- list(num_chains = 3, keep_every = 5)
    mean_forest_param_list <- list(sample_sigma2_leaf = T, sigma2_leaf_init = 0.5)
    expect_warning(
        bart_model <- bart(X_train = X_train, y_train = y_train, X_test = X_test, 
                           leaf_basis_train = W_train, leaf_basis_test = W_test, 
                           num_gfr = 0, num_burnin = 10, num_mcmc = 10, 
                           general_params = general_param_list, 
                           mean_forest_params = mean_forest_param_list)
    )
})

test_that("GFR BART", {
    skip_on_cran()
    
    # Generate simulated data
    n <- 100
    p <- 5
    X <- matrix(runif(n*p), ncol = p)
    f_XW <- (
        ((0 <= X[,1]) & (0.25 > X[,1])) * (-7.5) + 
            ((0.25 <= X[,1]) & (0.5 > X[,1])) * (-2.5) + 
            ((0.5 <= X[,1]) & (0.75 > X[,1])) * (2.5) + 
            ((0.75 <= X[,1]) & (1 > X[,1])) * (7.5)
    )
    noise_sd <- 1
    y <- f_XW + rnorm(n, 0, noise_sd)
    test_set_pct <- 0.2
    n_test <- round(test_set_pct*n)
    n_train <- n - n_test
    test_inds <- sort(sample(1:n, n_test, replace = FALSE))
    train_inds <- (1:n)[!((1:n) %in% test_inds)]
    X_test <- X[test_inds,]
    X_train <- X[train_inds,]
    y_test <- y[test_inds]
    y_train <- y[train_inds]
    
    # 1 chain, no thinning
    general_param_list <- list(num_chains = 1, keep_every = 1)
    expect_no_error(
        bart_model <- bart(X_train = X_train, y_train = y_train, X_test = X_test, 
                           num_gfr = 10, num_burnin = 10, num_mcmc = 10, 
                           general_params = general_param_list)
    )
    
    # 3 chains, no thinning
    general_param_list <- list(num_chains = 3, keep_every = 1)
    expect_no_error(
        bart_model <- bart(X_train = X_train, y_train = y_train, X_test = X_test, 
                           num_gfr = 10, num_burnin = 10, num_mcmc = 10, 
                           general_params = general_param_list)
    )
    
    # 1 chain, thinning
    general_param_list <- list(num_chains = 1, keep_every = 5)
    expect_no_error(
        bart_model <- bart(X_train = X_train, y_train = y_train, X_test = X_test, 
                           num_gfr = 10, num_burnin = 10, num_mcmc = 10, 
                           general_params = general_param_list)
    )
    
    # 3 chains, thinning
    general_param_list <- list(num_chains = 3, keep_every = 5)
    expect_no_error(
        bart_model <- bart(X_train = X_train, y_train = y_train, X_test = X_test, 
                           num_gfr = 10, num_burnin = 10, num_mcmc = 10, 
                           general_params = general_param_list)
    )
    
    # Check for error when more chains than GFR forests
    general_param_list <- list(num_chains = 11, keep_every = 1)
    expect_error(
        bart_model <- bart(X_train = X_train, y_train = y_train, X_test = X_test, 
                           num_gfr = 10, num_burnin = 10, num_mcmc = 10, 
                           general_params = general_param_list)
    )
    
    # Check for error when more chains than GFR forests
    general_param_list <- list(num_chains = 11, keep_every = 5)
    expect_error(
        bart_model <- bart(X_train = X_train, y_train = y_train, X_test = X_test, 
                           num_gfr = 10, num_burnin = 10, num_mcmc = 10, 
                           general_params = general_param_list)
    )
})

test_that("Warmstart BART", {
    skip_on_cran()
    
    # Generate simulated data
    n <- 100
    p <- 5
    X <- matrix(runif(n*p), ncol = p)
    f_XW <- (
        ((0 <= X[,1]) & (0.25 > X[,1])) * (-7.5) + 
            ((0.25 <= X[,1]) & (0.5 > X[,1])) * (-2.5) + 
            ((0.5 <= X[,1]) & (0.75 > X[,1])) * (2.5) + 
            ((0.75 <= X[,1]) & (1 > X[,1])) * (7.5)
    )
    noise_sd <- 1
    y <- f_XW + rnorm(n, 0, noise_sd)
    test_set_pct <- 0.2
    n_test <- round(test_set_pct*n)
    n_train <- n - n_test
    test_inds <- sort(sample(1:n, n_test, replace = FALSE))
    train_inds <- (1:n)[!((1:n) %in% test_inds)]
    X_test <- X[test_inds,]
    X_train <- X[train_inds,]
    y_test <- y[test_inds]
    y_train <- y[train_inds]
    
    # Run a BART model with only GFR
    general_param_list <- list(num_chains = 1, keep_every = 1)
    bart_model <- bart(X_train = X_train, y_train = y_train, X_test = X_test, 
                       num_gfr = 10, num_burnin = 0, num_mcmc = 0, 
                       general_params = general_param_list)
    
    # Save to JSON string
    bart_model_json_string <- saveBARTModelToJsonString(bart_model)
    
    # Run a new BART chain from the existing (X)BART model
    general_param_list <- list(num_chains = 3, keep_every = 5)
    expect_no_error(
        bart_model <- bart(X_train = X_train, y_train = y_train, X_test = X_test, 
                           num_gfr = 0, num_burnin = 10, num_mcmc = 10, 
                           previous_model_json = bart_model_json_string, 
                           previous_model_warmstart_sample_num = 1, 
                           general_params = general_param_list)
    
    )
    
    # Generate simulated data with random effects
    n <- 100
    p <- 5
    X <- matrix(runif(n*p), ncol = p)
    f_XW <- (
        ((0 <= X[,1]) & (0.25 > X[,1])) * (-7.5) + 
        ((0.25 <= X[,1]) & (0.5 > X[,1])) * (-2.5) + 
        ((0.5 <= X[,1]) & (0.75 > X[,1])) * (2.5) + 
        ((0.75 <= X[,1]) & (1 > X[,1])) * (7.5)
    )
    rfx_group_ids <- sample(1:2, size = n, replace = TRUE)
    rfx_basis <- rep(1, n)
    rfx_coefs <- c(-5, 5)
    rfx_term <- rfx_coefs[rfx_group_ids] * rfx_basis
    noise_sd <- 1
    y <- f_XW + rfx_term + rnorm(n, 0, noise_sd)
    test_set_pct <- 0.2
    n_test <- round(test_set_pct*n)
    n_train <- n - n_test
    test_inds <- sort(sample(1:n, n_test, replace = FALSE))
    train_inds <- (1:n)[!((1:n) %in% test_inds)]
    X_test <- X[test_inds,]
    X_train <- X[train_inds,]
    rfx_group_ids_test <- rfx_group_ids[test_inds]
    rfx_group_ids_train <- rfx_group_ids[train_inds]
    rfx_basis_test <- rfx_basis[test_inds]
    rfx_basis_train <- rfx_basis[train_inds]
    y_test <- y[test_inds]
    y_train <- y[train_inds]
    
    # Run a BART model with only GFR
    general_param_list <- list(num_chains = 1, keep_every = 1)
    bart_model <- bart(X_train = X_train, y_train = y_train, X_test = X_test, 
                       rfx_group_ids_train = rfx_group_ids_train, 
                       rfx_group_ids_test = rfx_group_ids_test, 
                       rfx_basis_train = rfx_basis_train, 
                       rfx_basis_test = rfx_basis_test, 
                       num_gfr = 10, num_burnin = 0, num_mcmc = 0, 
                       general_params = general_param_list)
    
    # Save to JSON string
    bart_model_json_string <- saveBARTModelToJsonString(bart_model)
    
    # Run a new BART chain from the existing (X)BART model
    general_param_list <- list(num_chains = 4, keep_every = 5)
    expect_no_error(
        bart_model <- bart(X_train = X_train, y_train = y_train, X_test = X_test, 
                           rfx_group_ids_train = rfx_group_ids_train, 
                           rfx_group_ids_test = rfx_group_ids_test, 
                           rfx_basis_train = rfx_basis_train, 
                           rfx_basis_test = rfx_basis_test, 
                           num_gfr = 0, num_burnin = 10, num_mcmc = 10, 
                           previous_model_json = bart_model_json_string, 
                           previous_model_warmstart_sample_num = 1, 
                           general_params = general_param_list)
    )
})
