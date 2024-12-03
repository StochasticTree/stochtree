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
    param_list <- list(num_chains = 1, keep_every = 1)
    expect_no_error(
        bart_model <- bart(X_train = X_train, y_train = y_train, X_test = X_test, 
                           num_gfr = 0, num_burnin = 10, num_mcmc = 10, 
                           params = param_list)
    )
    
    # 3 chains, no thinning
    param_list <- list(num_chains = 3, keep_every = 1)
    expect_no_error(
        bart_model <- bart(X_train = X_train, y_train = y_train, X_test = X_test, 
                           num_gfr = 0, num_burnin = 10, num_mcmc = 10, 
                           params = param_list)
    )
    
    # 1 chain, thinning
    param_list <- list(num_chains = 1, keep_every = 5)
    expect_no_error(
        bart_model <- bart(X_train = X_train, y_train = y_train, X_test = X_test, 
                           num_gfr = 0, num_burnin = 10, num_mcmc = 10, 
                           params = param_list)
    )
    
    # 3 chains, thinning
    param_list <- list(num_chains = 3, keep_every = 5)
    expect_no_error(
        bart_model <- bart(X_train = X_train, y_train = y_train, X_test = X_test, 
                           num_gfr = 0, num_burnin = 10, num_mcmc = 10, 
                           params = param_list)
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
    param_list <- list(num_chains = 1, keep_every = 1)
    expect_no_error(
        bart_model <- bart(X_train = X_train, y_train = y_train, X_test = X_test, 
                           num_gfr = 10, num_burnin = 10, num_mcmc = 10, 
                           params = param_list)
    )
    
    # 3 chains, no thinning
    param_list <- list(num_chains = 3, keep_every = 1)
    expect_no_error(
        bart_model <- bart(X_train = X_train, y_train = y_train, X_test = X_test, 
                           num_gfr = 10, num_burnin = 10, num_mcmc = 10, 
                           params = param_list)
    )
    
    # 1 chain, thinning
    param_list <- list(num_chains = 1, keep_every = 5)
    expect_no_error(
        bart_model <- bart(X_train = X_train, y_train = y_train, X_test = X_test, 
                           num_gfr = 10, num_burnin = 10, num_mcmc = 10, 
                           params = param_list)
    )
    
    # 3 chains, thinning
    param_list <- list(num_chains = 3, keep_every = 5)
    expect_no_error(
        bart_model <- bart(X_train = X_train, y_train = y_train, X_test = X_test, 
                           num_gfr = 10, num_burnin = 10, num_mcmc = 10, 
                           params = param_list)
    )
    
    # Check for error when more chains than GFR forests
    param_list <- list(num_chains = 11, keep_every = 1)
    expect_error(
        bart_model <- bart(X_train = X_train, y_train = y_train, X_test = X_test, 
                           num_gfr = 10, num_burnin = 10, num_mcmc = 10, 
                           params = param_list)
    )
    
    # Check for error when more chains than GFR forests
    param_list <- list(num_chains = 11, keep_every = 5)
    expect_error(
        bart_model <- bart(X_train = X_train, y_train = y_train, X_test = X_test, 
                           num_gfr = 10, num_burnin = 10, num_mcmc = 10, 
                           params = param_list)
    )
})
