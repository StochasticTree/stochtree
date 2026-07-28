# Load libraries
library(stochtree)
library(here)

# Load train and test data
from_file <- T
if (from_file) {
    project_dir <- here()
    train_set_path <- file.path(project_dir, "debug", "data", "heterosked_train.csv")
    test_set_path <- file.path(project_dir, "debug", "data", "heterosked_test.csv")
    train_df <- read.csv(train_set_path)
    test_df <- read.csv(test_set_path)
    y_train <- train_df[,1]
    y_test <- test_df[,1]
    X_train <- train_df[,2:11]
    X_test <- test_df[,2:11]
    f_x_train <- train_df[,12]
    f_x_test <- test_df[,12]
    s_x_train <- train_df[,13]
    s_x_test <- test_df[,13]
} else {
    n <- 500
    p_x <- 10
    X <- matrix(runif(n*p_x), ncol = p_x)
    f_XW <- 0
    s_XW <- (
        ((0 <= X[,1]) & (0.25 > X[,1])) * (0.5*X[,3]) + 
        ((0.25 <= X[,1]) & (0.5 > X[,1])) * (1*X[,3]) + 
        ((0.5 <= X[,1]) & (0.75 > X[,1])) * (2*X[,3]) + 
        ((0.75 <= X[,1]) & (1 > X[,1])) * (3*X[,3])
    )
    y <- f_XW + rnorm(n, 0, 1)*s_XW
    
    # Split data into test and train sets
    test_set_pct <- 0.2
    n_test <- round(test_set_pct*n)
    n_train <- n - n_test
    test_inds <- sort(sample(1:n, n_test, replace = FALSE))
    train_inds <- (1:n)[!((1:n) %in% test_inds)]
    X_test <- as.data.frame(X[test_inds,])
    X_train <- as.data.frame(X[train_inds,])
    W_test <- NULL
    W_train <- NULL
    y_test <- y[test_inds]
    y_train <- y[train_inds]
    f_x_test <- f_XW[test_inds]
    f_x_train <- f_XW[train_inds]
    s_x_test <- s_XW[test_inds]
    s_x_train <- s_XW[train_inds]
}

# Run BART
num_gfr <- 10
num_burnin <- 0
num_mcmc <- 200
num_samples <- num_gfr + num_burnin + num_mcmc
m <- 50
a_0 <- sqrt(1/2)
sigma0 <- 1/2
bart_model <- stochtree::bart(
    X_train = X_train, y_train = y_train, X_test = X_test,
    num_gfr = num_gfr, num_burnin = num_burnin, num_mcmc = num_mcmc,
    num_trees_mean = 0, num_trees_variance = m,
    alpha_mean = 0.8, beta_mean = 3, min_samples_leaf_mean = 5,
    max_depth_mean = 3, alpha_variance = 0.95, beta_variance = 1.25,
    min_samples_leaf_variance = 1, max_depth_variance = 10,
    sample_sigma = F, sample_tau = F, keep_gfr = T, sigma2_init = sigma0, 
    # a_forest = m/(a_0^2) + 1, b_forest = m/(a_0^2)
    a_forest = 3, b_forest = 2
)

s_x_hat_train <- rowMeans(bart_model$sigma_x_hat_train)
plot(s_x_hat_train, s_x_train, main = "Conditional std dev as a function of x", xlab = "Predicted", ylab = "Actual"); abline(0,1,col="red",lty=3,lwd=3)
sqrt(mean((s_x_hat_train - s_x_train)^2))

s_x_hat_test <- rowMeans(bart_model$sigma_x_hat_test)
plot(s_x_hat_test, s_x_test, main = "Conditional std dev as a function of x", xlab = "Predicted", ylab = "Actual"); abline(0,1,col="red",lty=3,lwd=3)
sqrt(mean((s_x_hat_test - s_x_test)^2))
