#' Samples from an XBART model using the provided data
#'
#' @param y Outcome of interest
#' @param X Covariates used in the tree model
#' @param omega Basis functions used for leaf node regression
#' @param num_samples Number of samples retained
#' @param num_burnin Number of "burn-in" (not retained) samples
#' @param num_trees Number of trees in the ensemble
#' @param nu Prior scale for global variance parameter (sigma^2)
#' @param lambda Prior shape for global variance parameter (sigma^2)
#' @param a Prior scale for leaf node variance parameter (tau)
#' @param b Prior shape for leaf node variance parameter (tau)
#' @param cutpoint_grid_size Maximum number of cutpoints to consider at each split
#' @param random_seed Seed for random number generator
#'
#' @return Object of class `xbart_samples`
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
#' omega <- runif(n)
#' betas <- c(-5, -10, 10, 5)
#' y <- ifelse(
#'     (X[,1] >= 0) & (X[,1] < 0.25), betas[1] * omega,
#'     ifelse(
#'         (X[,1] >= 0.25) & (X[,1] < 0.5), betas[2] * omega,
#'         ifelse(
#'             (X[,1] >= 0.5) & (X[,1] < 0.75), betas[3] * omega, betas[4] * omega
#'         )
#'     )
#' ) + rnorm(n, 0, 1)
#' cutpoint_grid_size <- 20
#' nu <- 0.5
#' lambda <- 2.
#' a <- 1.
#' b <- 1.
#' result <- xbart(ytrain, Xtrain, omegatrain, num_samples, num_burnin, num_trees, nu, lambda, a, b, cutpoint_grid_size, random_seed = random_seed)
xbart <- function(y, X, omega, num_samples, num_burnin, num_trees, nu, lambda, a, b, cutpoint_grid_size, random_seed = -1){
    ptr <- xbart_sample_cpp(y, X, omega, num_samples, num_burnin, num_trees, nu, lambda, a, b, cutpoint_grid_size, random_seed)
    result <- list(ptr=ptr)
    class(result) <- "xbart_samples"
    return(result)
}

#' Predicts from a sampled XBART model and returns the result as a matrix
#'
#' @param xbart Object returned by the `xbart` function
#' @param X Covariates for which outcome will be predicted
#' @param omega Basis vector to use in leaf node predictions
#' @param num_samples Number of retained samples in the original `xbart` model
#'
#' @return Matrix with observations in rows and sampled XBART draws in columns
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
#' omega <- runif(n)
#' betas <- c(-5, -10, 10, 5)
#' y <- ifelse(
#'     (X[,1] >= 0) & (X[,1] < 0.25), betas[1] * omega,
#'     ifelse(
#'         (X[,1] >= 0.25) & (X[,1] < 0.5), betas[2] * omega,
#'         ifelse(
#'             (X[,1] >= 0.5) & (X[,1] < 0.75), betas[3] * omega, betas[4] * omega
#'         )
#'     )
#' ) + rnorm(n, 0, 1)
#' cutpoint_grid_size <- 20
#' nu <- 0.5
#' lambda <- 2.
#' a <- 1.
#' b <- 1.
#' stochtree_samples <- xbart(ytrain, Xtrain, omegatrain, num_samples, num_burnin, num_trees, nu, lambda, a, b, cutpoint_grid_size, random_seed = random_seed)
#' stochtree_predictions <- predict(stochtree_samples, Xtest, omegatest, num_samples)
predict.xbart_samples <- function(xbart, X, omega, num_samples){
    result <- xbart_predict_cpp(xbart$ptr, X, omega, num_samples)
    return(result)
}
