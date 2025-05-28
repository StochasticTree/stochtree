library(microbenchmark)
library(stochtree)

# Generate data needed to train BART model
n <- 1000
p <- 5
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

# Run microbenchmark
microbenchmark(
    bart(X_train = X_train, y_train = y_train, X_test = X_test, num_gfr = 10, num_mcmc = 100)
)
