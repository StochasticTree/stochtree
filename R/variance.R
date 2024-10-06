#' Sample one iteration of the (inverse gamma) global variance model
#'
#' @param residual Outcome class
#' @param dataset ForestDataset class
#' @param rng C++ random number generator
#' @param a Global variance shape parameter
#' @param b Global variance scale parameter
#'
#' @export
sample_sigma2_one_iteration <- function(residual, dataset, rng, a, b) {
    return(sample_sigma2_one_iteration_cpp(residual$data_ptr, dataset$data_ptr, rng$rng_ptr, a, b))
}

#' Sample one iteration of the leaf parameter variance model (only for univariate basis and constant leaf!)
#'
#' @param forest_samples Container of forest samples
#' @param rng C++ random number generator
#' @param a Leaf variance shape parameter
#' @param b Leaf variance scale parameter
#' @param sample_num Sample index
#'
#' @export
sample_tau_one_iteration <- function(forest_samples, rng, a, b, sample_num) {
    return(sample_tau_one_iteration_cpp(forest_samples$forest_container_ptr, rng$rng_ptr, a, b, sample_num))
}
