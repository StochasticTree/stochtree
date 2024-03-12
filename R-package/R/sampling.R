#' Sample one iteration of the model
#'
#' @param data Dataset (covariates and optional basis and variance weights)
#' @param residual Outcome
#' @param forest_samples Container for forest samples
#' @param tracker Wrapper around convenience data structures used in sampling
#' @param split_prior Prior on tree splits
#' @param rng Pointer to C++ random number generator
#' @param feature_types Vector of integers mapping to feature types (0 = numeric, 1 = ordered categorical, 2 = unordered categorical)
#' @param leaf_model_int Integer indicating the type of leaf model (0 = constant, 1 = univariate regression, 2 = multivariate regression)
#' @param leaf_model_scale Value of leaf model scale parameter (passed as a matrix, handled as a double in C++ if the leaf model is univariate)
#' @param global_scale Value of global variance parameter
#' @param cutpoint_grid_size Maximum number of cutpoints to evaluate
#' @param gfr (Boolean) Whether or not the GFR sampler should be used (if false, MCMC is used)
#' 
#' @export
sample_model_one_iteration <- function(data, residual, forest_samples, tracker, split_prior, rng, 
                                       feature_types, leaf_model_int, leaf_model_scale, variable_weights, 
                                       global_scale, cutpoint_grid_size = 500, gfr = T) {
    if (gfr) {
        sample_gfr_one_iteration_cpp(
            data, residual, forest_samples, tracker, split_prior, 
            rng, feature_types, cutpoint_grid_size, leaf_model_scale, 
            variable_weights, global_scale, leaf_model_int
        )
    } else {
        sample_mcmc_one_iteration_cpp(
            data, residual, forest_samples, tracker, split_prior, 
            rng, feature_types, cutpoint_grid_size, leaf_model_scale, 
            variable_weights, global_scale, leaf_model_int
        ) 
    }
}

#' Sample one iteration of the global variance model
#'
#' @param residual Pointer to a C++ ColumnVector object storing residual
#' @param rng Pointer to a C++ random number generator
#' @param nu Global variance shape parameter
#' @param lambda Constitutes the scale parameter for the global variance along with nu (i.e. scale is nu*lambda)
#' 
#' @export
sample_sigma2_one_iteration <- function(residual, rng, nu, lambda) {
    return(sample_sigma2_one_iteration_cpp(residual, rng, nu, lambda))
}

#' Sample one iteration of the leaf parameter variance model (only for univariate basis and constant leaf!)
#'
#' @param forest_samples Pointer to container of forest samples
#' @param rng Pointer to a C++ random number generator
#' @param a Leaf variance shape parameter
#' @param b Leaf variance scale parameter
#' @param sample_num Sample index
#' 
#' @export
sample_tau_one_iteration <- function(forest_samples, rng, a, b, sample_num) {
    return(sample_tau_one_iteration_cpp(forest_samples, rng, a, b, sample_num))
}

#' Create a C++ random number generator and store its pointer (for replicability across sampling function calls)
#'
#' @param random_seed Seed for the RNG
#'
#' @return External pointer to std::mt19937 object
#' @export
#'
#' @examples
#' rng_ptr <- random_number_generator(101)
random_number_generator <- function(random_seed) {
    return(rng_cpp(random_seed))
}

#' Create a container for forest samples
#'
#' @param num_trees Number of trees
#' @param output_dimension Dimensionality of the outcome model
#' @param is_leaf_constant Whether leaf is constant
#'
#' @return External pointer to forest container object
#' @export
#'
#' @examples
#' forest_samples_ptr <- forest_container(100, 1, T)
forest_container <- function(num_trees, output_dimension = 1, is_leaf_constant = T) {
    return(forest_container_cpp(num_trees, output_dimension, is_leaf_constant))
}

#' Create a StochTree tree prior object
#'
#' @param alpha Split probability at root
#' @param beta Decay parameter for split probability
#' @param min_samples_leaf Minimum number of samples in a valid leaf
#'
#' @return External pointer to tree prior object
#' @export
#'
#' @examples
#' tree_prior_ptr <- tree_prior(0.95, 1.25, 10)
tree_prior <- function(alpha, beta, min_samples_leaf) {
    return(tree_prior_cpp(alpha, beta, min_samples_leaf))
}

#' Create a StochTree forest tracker object
#'
#' @param dataset Pointer to a stochtree dataset
#' @param feature_types Vector of integers mapping to feature types (0 = numeric, 1 = ordered categorical, 2 = unordered categorical)
#' @param num_trees Number of trees in the model
#' @param n Size of the training dataset
#'
#' @return External pointer to forest tracker object
#' @export
#'
#' @examples
#' n <- 100
#' p_X <- 10
#' p_W <- 1
#' X <- matrix(runif(n*p_X), ncol = p_X)
#' W <- matrix(runif(n*p_W), ncol = p_W)
#' data_ptr <- dataset(X, W)
#' feature_types <- rep(0, p_X)
#' gfr_sampler_ptr <- gfr_forest_sampler(data_ptr, feature_types, 100, n)
forest_tracker <- function(dataset, feature_types, num_trees, n) {
    return(forest_tracker_cpp(data_ptr, feature_types, num_trees, n))
}
