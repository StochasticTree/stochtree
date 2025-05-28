# Load libraries
library(profvis)
library(profmem)
library(stochtree)

# Generate data
n <- 10000
p <- 50
X <- matrix(runif(n*p), ncol = p)
f_XW <- (
    ((0 <= X[,1]) & (0.25 > X[,1])) * (-7.5) + 
    ((0.25 <= X[,1]) & (0.5 > X[,1])) * (-2.5) + 
    ((0.5 <= X[,1]) & (0.75 > X[,1])) * (2.5) + 
    ((0.75 <= X[,1]) & (1 > X[,1])) * (7.5)
)
noise_sd <- 1
y <- f_XW + rnorm(n, 0, noise_sd)

# Train test split
test_set_pct <- 0.2
n_test <- round(test_set_pct*n)
n_train <- n - n_test
test_inds <- sort(sample(1:n, n_test, replace = FALSE))
train_inds <- (1:n)[!((1:n) %in% test_inds)]
X_test <- X[test_inds,]
X_train <- X[train_inds,]
y_test <- y[test_inds]
y_train <- y[train_inds]

# Prepare sampler
num_gfr <- 500
num_mcmc <- 500

# Profile BART
profvis(
    bart_model <- bart(X_train = X_train, y_train = y_train, X_test = X_test, 
                       num_gfr = num_gfr, num_burnin = 0, num_mcmc = num_mcmc)
)

profmem(
    bart_model <- bart(X_train = X_train, y_train = y_train, X_test = X_test, 
                       num_gfr = 10, num_burnin = 0, num_mcmc = 10)
)
