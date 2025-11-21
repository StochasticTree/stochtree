#' Sample one iteration of the (inverse gamma) global variance model
#'
#' @param residual Outcome class
#' @param dataset ForestDataset class
#' @param rng C++ random number generator
#' @param a Global variance shape parameter
#' @param b Global variance scale parameter
#' @return None
#' @export
#'
#' @examples
#' X <- matrix(runif(10*100), ncol = 10)
#' y <- -5 + 10*(X[,1] > 0.5) + rnorm(100)
#' y_std <- (y-mean(y))/sd(y)
#' forest_dataset <- createForestDataset(X)
#' outcome <- createOutcome(y_std)
#' rng <- createCppRNG(1234)
#' a <- 1.0
#' b <- 1.0
#' sigma2 <- sampleGlobalErrorVarianceOneIteration(outcome, forest_dataset, rng, a, b)
sampleGlobalErrorVarianceOneIteration <- function(
  residual,
  dataset,
  rng,
  a,
  b
) {
  return(sample_sigma2_one_iteration_cpp(
    residual$data_ptr,
    dataset$data_ptr,
    rng$rng_ptr,
    a,
    b
  ))
}

#' Sample one iteration of the leaf parameter variance model (only for univariate basis and constant leaf!)
#'
#' @param forest C++ forest
#' @param rng C++ random number generator
#' @param a Leaf variance shape parameter
#' @param b Leaf variance scale parameter
#' @return None
#' @export
#'
#' @examples
#' num_trees <- 100
#' leaf_dimension <- 1
#' is_leaf_constant <- TRUE
#' is_exponentiated <- FALSE
#' active_forest <- createForest(num_trees, leaf_dimension, is_leaf_constant, is_exponentiated)
#' rng <- createCppRNG(1234)
#' a <- 1.0
#' b <- 1.0
#' tau <- sampleLeafVarianceOneIteration(active_forest, rng, a, b)
sampleLeafVarianceOneIteration <- function(forest, rng, a, b) {
  return(sample_tau_one_iteration_cpp(forest$forest_ptr, rng$rng_ptr, a, b))
}
