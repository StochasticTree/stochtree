library(stochtree)

# Generate the data
X_train <- seq(0,20,length=100)
X_test <- seq(0,20,length=99)
y_train <- (sin(pi*X_train/5) + 0.2*cos(4*pi*X_train/5)) * (X_train <= 9.6)
lin_train <- X_train>9.6; 
y_train[lin_train] <- -1 + X_train[lin_train]/10
y_train <- y_train + rnorm(length(y_train), sd=0.1)
y_test <- (sin(pi*X_test/5) + 0.2*cos(4*pi*X_test/5)) * (X_test <= 9.6)
lin_test <- X_test>9.6; 
y_test[lin_test] <- -1 + X_test[lin_test]/10

bart_model <- bart(X_train=X_train, y_train=y_train, X_test=X_test)
forest_kernel <- createForestKernel()
result_inds <- forest_kernel$compute_leaf_indices(
    covariates_train = X_train, covariates_test = X_test, 
    forest_container = bart_model$forests, 
    forest_num = bart_model$model_params$num_samples - 1
)
result_kernels <- forest_kernel$compute_kernel(
    covariates_train = X_train, covariates_test = X_test,
    forest_container = bart_model$forests,
    forest_num = bart_model$model_params$num_samples - 1
)

# GP
mu_tilde <- result_kernels$kernel_test_train %*% MASS::ginv(result_kernels$kernel_train) %*% y_train
Sigma_tilde <- result_kernels$kernel_test - result_kernels$kernel_test_train %*% MASS::ginv(result_kernels$kernel_train) %*% t(result_kernels$kernel_test_train)

B <- mvtnorm::rmvnorm(1000, mean = mu_tilde, sigma = Sigma_tilde)
yhat_mean_test <- colMeans(B)
plot(yhat_mean_test, y_test, xlab = "predicted", ylab = "actual", main = "Gaussian process")
abline(0,1,lwd=2.5,lty=3,col="red")

# m <- 200
# n <- length(X_train)
# dummies <- result[["leaf_indices_train"]]
# max_ix <- max(dummies)
# leaf_train <- Matrix::sparseMatrix(i=rep(1:n, m), 
#              j = dummies+1, 
#              x = rep(1, n*m),
#              dims = c(n, max_ix+1),
# )
# leaf_train <- as.matrix(leaf_train)
# check <- leaf_train %*% t(leaf_train)
