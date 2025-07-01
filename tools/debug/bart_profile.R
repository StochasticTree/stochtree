################################################################################
## Profiling BART on multiple platforms
################################################################################

library(stochtree)
Rprof()

start_time <- Sys.time()
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
test_set_pct <- 0.2
n_test <- round(test_set_pct*n)
n_train <- n - n_test
test_inds <- sort(sample(1:n, n_test, replace = FALSE))
train_inds <- (1:n)[!((1:n) %in% test_inds)]
X_test <- X[test_inds,]
X_train <- X[train_inds,]
y_test <- y[test_inds]
y_train <- y[train_inds]
general_params <- list(num_threads = 10)
bart_model <- bart(X_train = X_train, y_train = y_train, X_test = X_test, general_params = general_params)
end_time <- Sys.time()
print(paste("runtime:", end_time - start_time))

summaryRprof()
Rprof(NULL)
