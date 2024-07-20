# Load libraries
library(stochtree)
library(rnn)

# Random seed
random_seed <- 1234
set.seed(random_seed)

# Fixed parameters
sample_size <- 10000
alpha <- 1.0
beta <- 0.1
ntree <- 50
num_iter <- 10
num_gfr <- 10
num_burnin <- 0
num_mcmc <- 10
min_samples_leaf <- 5
nu <- 3
lambda <- NULL
q <- 0.9
sigma2_init <- NULL
sample_tau <- F
sample_sigma <- T

# Generate data, choice of DGPs:
# (1) the "deep interaction" classification DGP
# (2) partitioned linear model (with split variables and basis included as BART covariates)
dgp_num <- 2
if (dgp_num == 1) {
    # Initial DGP setup
    n0 <- 50
    p <- 10
    n <- n0*(2^p)
    k <- 2
    p1 <- 20
    noise <- 0.1
    
    # Full factorial covariate reference frame
    xtemp <- as.data.frame(as.factor(rep(0:(2^p-1),n0)))
    xtemp1 <- rep(0:(2^p-1),n0)
    x <- t(sapply(xtemp1,function(j) as.numeric(int2bin(j,p))))
    X_superset <- x*abs(rnorm(length(x))) - (1-x)*abs(rnorm(length(x)))
    
    # Generate outcome
    M <- model.matrix(~.-1,data = xtemp)
    M <- cbind(rep(1,n),M)
    beta.true <- -10*abs(rnorm(ncol(M)))
    beta.true[1] <- 0.5
    non_zero_betas <- c(1,sample(1:ncol(M), p1-1))   
    beta.true[-non_zero_betas] <- 0      
    Y <- M %*% beta.true + rnorm(n, 0, noise)
    y_superset <- as.numeric(Y>0)
    
    # Downsample to desired n
    subset_inds <- order(sample(1:nrow(X_superset), sample_size, replace = F))
    X <- X_superset[subset_inds,]
    y <- y_superset[subset_inds]
} else if (dgp_num == 2) {
    p <- 10
    snr <- 2
    X <- matrix(runif(sample_size*p), ncol = p)
    f_X <- (
        ((0 <= X[,1]) & (0.25 > X[,1])) * (-7.5*X[,2]) +
            ((0.25 <= X[,1]) & (0.5 > X[,1])) * (-2.5*X[,2]) +
            ((0.5 <= X[,1]) & (0.75 > X[,1])) * (2.5*X[,2]) +
            ((0.75 <= X[,1]) & (1 > X[,1])) * (7.5*X[,2])
    )
    noise_sd <- sd(f_X) / snr
    y <- f_X + rnorm(sample_size, 0, noise_sd)
} else stop("dgp_num must be 1 or 2")

# Switch between 
# (1) the R-dispatched loop, 
# (2) the "generalized" C++ sampling loop, and 
# (3) the "streamlined" / "specialized" C++ sampling loop that only samples trees
# and sigma^2 (error variance parameter)
sampler_choice <- 3
system.time({
    if (sampler_choice == 1) {
        bart_obj <- stochtree::bart(
            X_train = X, y_train = y, alpha = alpha, beta = beta, 
            min_samples_leaf = min_samples_leaf, nu = nu, lambda = lambda, q = q, 
            sigma2_init = sigma2_init, num_trees = ntree, num_gfr = num_gfr, 
            num_burnin = num_burnin, num_mcmc = num_mcmc, sample_tau = sample_tau, 
            sample_sigma = sample_sigma, random_seed = random_seed
        )
        avg_md <- bart_obj$forests$average_max_depth()
    } else if (sampler_choice == 2) {
        bart_obj <- stochtree::bart_cpp_loop_generalized(
            X_train = X, y_train = y, alpha = alpha, beta = beta, 
            min_samples_leaf = min_samples_leaf, nu = nu, lambda = lambda, q = q, 
            sigma2_init = sigma2_init, num_trees = ntree, num_gfr = num_gfr, 
            num_burnin = num_burnin, num_mcmc = num_mcmc, sample_leaf_var = sample_tau, 
            sample_global_var = sample_sigma, random_seed = random_seed
        )
        avg_md <- average_max_depth_bart_generalized(bart_obj$bart_result)
    } else if (sampler_choice == 3) {
        bart_obj <- stochtree::bart_cpp_loop_specialized(
            X_train = X, y_train = y, alpha = alpha, beta = beta, 
            min_samples_leaf = min_samples_leaf, nu = nu, lambda = lambda, q = q, 
            sigma2_init = sigma2_init, num_trees = ntree, num_gfr = num_gfr, 
            num_burnin = num_burnin, num_mcmc = num_mcmc, random_seed = random_seed
        )
        avg_md <- average_max_depth_bart_specialized(bart_obj$bart_result)
    } else stop("sampler_choice must be 1, 2, or 3")
})

avg_md