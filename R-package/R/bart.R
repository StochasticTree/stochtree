#' Samples from a BART model using the provided data
#'
#' @param y Outcome of interest
#' @param X Covariates used in the tree model
#' @param omega Basis functions used for leaf node regression
#' @param rfx_groups Vector of integer labels distinguishing groups for random effects
#' @param rfx_basis Basis functions used in the random effects regression. For simple group random effects, this is a vector of 1's.
#' @param num_samples Number of samples retained
#' @param num_burnin Number of "burn-in" (not retained) samples
#' @param num_trees Number of trees in the ensemble
#' @param nu Prior shape for global variance parameter (sigma^2)
#' @param lambda Prior scale for global variance parameter (sigma^2)
#' @param a_rfx Prior shape for rfx variance parameter(s)
#' @param b_rfx Prior scale for rfx variance parameter(s)
#' @param random_seed Seed for random number generator
#'
#' @return Object of class `bart_samples`
#' @export
#'
#' @examples
#' n <- 500
#' p <- 20
#' num_samples <- 50
#' num_burnin <- 10
#' num_trees <- 50
#' train_inds <- sample(1:n, round(n*0.8), replace = F)
#' test_inds <- (1:n)[!((1:n) %in% train_inds)]
#' X <- matrix(runif(n*p), ncol = p)
#' rfx_labels <- as.integer(rep(c(1,2), times = n/2))
#' rfx_basis <- rep(1, times = n)
#' omega <- runif(n)
#' betas <- c(-5, -10, 10, 5)
#' f_x_omega <- ifelse(
#'     (X[,1] >= 0) & (X[,1] < 0.25), betas[1] * omega,
#'     ifelse(
#'         (X[,1] >= 0.25) & (X[,1] < 0.5), betas[2] * omega,
#'         ifelse(
#'             (X[,1] >= 0.5) & (X[,1] < 0.75), betas[3] * omega, betas[4] * omega
#'         )
#'     )
#' )
#' rfx <- ifelse(rfx_labels == 1, -1, 5)
#' y <- f_x_omega + rfx + rnorm(n, 0, 1)
#' cutpoint_grid_size <- 20
#' nu <- 0.5
#' lambda <- 2.
#' a_rfx <- 1.
#' b_rfx <- 1.
#' result <- bart(y, X, omega, rfx_basis, rfx_labels, num_samples, num_burnin, num_trees, nu, lambda, a_rfx, b_rfx, random_seed = random_seed)
bart <- function(y, X, omega, rfx_basis, rfx_groups, num_samples, num_burnin, num_trees, nu, lambda, a_rfx, b_rfx, random_seed = -1) {
    num_rfx_groups <- length(unique(rfx_groups))
    ptr <- bart_sample_cpp(y, X, omega, rfx_basis, rfx_groups, num_rfx_groups, num_samples, num_burnin, num_trees, nu, lambda, a_rfx, b_rfx, random_seed)
    result <- list(ptr=ptr)
    class(result) <- "bart_samples"
    return(result)
}

#' Predicts from a sampled BART model and returns the result as a matrix
#'
#' @param bart Object returned by the `bart` function
#' @param X Covariates for which outcome will be predicted
#' @param omega Basis vector to use in leaf node predictions
#' @param rfx_basis Basis function for group random effects
#' @param rfx_groups Group indices for random effects
#' @param num_samples Number of retained samples in the original `bart` model
#'
#' @return Matrix with observations in rows and sampled BBART draws in columns
#' @export
#'
#' @examples
#' n <- 500
#' p <- 20
#' num_samples <- 50
#' num_burnin <- 10
#' num_trees <- 50
#' train_inds <- sample(1:n, round(n*0.8), replace = F)
#' test_inds <- (1:n)[!((1:n) %in% train_inds)]
#' X <- matrix(runif(n*p), ncol = p)
#' rfx_labels <- as.integer(rep(c(1,2), times = n/2))
#' rfx_basis <- rep(1, times = n)
#' omega <- runif(n)
#' betas <- c(-5, -10, 10, 5)
#' f_x_omega <- ifelse(
#'     (X[,1] >= 0) & (X[,1] < 0.25), betas[1] * omega,
#'     ifelse(
#'         (X[,1] >= 0.25) & (X[,1] < 0.5), betas[2] * omega,
#'         ifelse(
#'             (X[,1] >= 0.5) & (X[,1] < 0.75), betas[3] * omega, betas[4] * omega
#'         )
#'     )
#' )
#' rfx <- ifelse(rfx_labels == 1, -1, 5)
#' y <- f_x_omega + rfx + rnorm(n, 0, 1)
#' 
#' train_inds <- sample(1:n, round(n*0.8), replace = F)
#' test_inds <- (1:n)[!((1:n) %in% train_inds)]
#' Xtrain <- X[train_inds,]
#' Xtest <- X[test_inds,]
#' ytrain <- y[train_inds]
#' ytest <- y[test_inds]
#' omegatrain <- omega[train_inds]
#' omegatest <- omega[test_inds]
#' rfx_labels_train <- rfx_labels[train_inds]
#' rfx_labels_test <- rfx_labels[test_inds]
#' rfx_basis_train <- rfx_basis[train_inds]
#' rfx_basis_test <- rfx_basis[test_inds]
#' 
#' nu <- 0.5
#' lambda <- 2.
#' a_rfx <- 1.
#' b_rfx <- 1.
#' stochtree_samples <- bart(ytrain, Xtrain, omegatrain, rfx_basis_train, rfx_labels_train, num_samples, num_burnin, num_trees, nu, lambda, a_rfx, b_rfx, random_seed = random_seed)
#' stochtree_predictions <- predict(stochtree_samples, Xtest, omegatest, rfx_basis_test, rfx_labels_test, num_samples)
predict.bart_samples <- function(bart, X, omega, rfx_basis, rfx_groups, num_samples){
    result <- bart_predict_cpp(bart$ptr, X, omega, rfx_basis, rfx_groups, num_samples)
    return(result)
}
