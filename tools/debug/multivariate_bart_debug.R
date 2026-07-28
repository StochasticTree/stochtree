library(stochtree)

# Generate the data
n <- 500
p_x <- 10
p_w <- 2
snr <- 3
X <- matrix(runif(n*p_x), ncol = p_x)
W <- matrix(runif(n*p_w), ncol = p_w)
f_XW <- (
    ((0 <= X[,1]) & (0.25 > X[,1])) * (-7.5*W[,1]) + 
        ((0.25 <= X[,1]) & (0.5 > X[,1])) * (-2.5*W[,1]) + 
        ((0.5 <= X[,1]) & (0.75 > X[,1])) * (2.5*W[,1]) + 
        ((0.75 <= X[,1]) & (1 > X[,1])) * (7.5*W[,1])
)
noise_sd <- sd(f_XW) / snr
y <- f_XW + rnorm(n, 0, 1)*noise_sd

# Split data into test and train sets
test_set_pct <- 0.2
n_test <- round(test_set_pct*n)
n_train <- n - n_test
test_inds <- sort(sample(1:n, n_test, replace = FALSE))
train_inds <- (1:n)[!((1:n) %in% test_inds)]
X_test <- as.data.frame(X[test_inds,])
X_train <- as.data.frame(X[train_inds,])
W_test <- W[test_inds,]
W_train <- W[train_inds,]
y_test <- y[test_inds]
y_train <- y[train_inds]

# Sample BART model
num_gfr <- 10
num_burnin <- 0
num_mcmc <- 100
num_samples <- num_gfr + num_burnin + num_mcmc
bart_params <- list(sample_sigma_global = T, sample_sigma_leaf = F, num_trees_mean = 100)
bart_model_warmstart <- stochtree::bart(
    X_train = X_train, W_train = W_train, y_train = y_train, X_test = X_test, W_test = W_test, 
    num_gfr = num_gfr, num_burnin = num_burnin, num_mcmc = num_mcmc, 
    params = bart_params
)
