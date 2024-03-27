#' Sample one iteration of the global variance model
#'
#' @param residual Outcome class
#' @param rng C++ random number generator
#' @param nu Global variance shape parameter
#' @param lambda Constitutes the scale parameter for the global variance along with nu (i.e. scale is nu*lambda)
#'
#' @export
sample_sigma2_one_iteration <- function(residual, rng, nu, lambda) {
    return(sample_sigma2_one_iteration_cpp(residual$data_ptr, rng$rng_ptr, nu, lambda))
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
