#' Class that wraps a C++ random number generator (for reproducibility)
#'
#' @description
#' Persists a C++ random number generator throughout an R session to 
#' ensure reproducibility from a given random seed. If no seed is provided, 
#' the C++ random number generator is initialized using `std::random_device`.

CppRNG <- R6::R6Class(
    classname = "CppRNG",
    cloneable = FALSE,
    public = list(
        
        #' @field rng_ptr External pointer to a C++ std::mt19937 class
        rng_ptr = NULL,

        #' @description
        #' Create a new CppRNG object.
        #' @param random_seed (Optional) random seed for sampling
        #' @return A new `CppRNG` object.
        initialize = function(random_seed = -1) {
            self$rng_ptr <- rng_cpp(random_seed)
        }
    )
)

#' Class that defines and samples a forest model
#'
#' @description
#' Hosts the C++ data structures needed to sample an ensemble of decision 
#' trees, and exposes functionality to run a forest sampler 
#' (using either MCMC or the grow-from-root algorithm).

ForestModel <- R6::R6Class(
    classname = "ForestModel",
    cloneable = FALSE,
    public = list(
        
        #' @field tracker_ptr External pointer to a C++ ForestTracker class
        tracker_ptr = NULL,
        
        #' @field tree_prior_ptr External pointer to a C++ TreePrior class
        tree_prior_ptr = NULL, 
        
        #' @description
        #' Create a new ForestModel object.
        #' @param forest_dataset `ForestDataset` object, used to initialize forest sampling data structures
        #' @param feature_types Feature types (integers where 0 = numeric, 1 = ordered categorical, 2 = unordered categorical)
        #' @param num_trees Number of trees in the forest being sampled
        #' @param n Number of observations in `forest_dataset`
        #' @param alpha Root node split probability in tree prior
        #' @param beta Depth prior penalty in tree prior
        #' @param min_samples_leaf Minimum number of samples in a tree leaf
        #' @param max_depth Maximum depth of any tree in an ensemble
        #' @return A new `ForestModel` object.
        initialize = function(forest_dataset, feature_types, num_trees, n, alpha, beta, min_samples_leaf, max_depth) {
            stopifnot(!is.null(forest_dataset$data_ptr))
            self$tracker_ptr <- forest_tracker_cpp(forest_dataset$data_ptr, feature_types, num_trees, n)
            self$tree_prior_ptr <- tree_prior_cpp(alpha, beta, min_samples_leaf, max_depth)
        }, 
        
        #' @description
        #' Run a single iteration of the forest sampling algorithm (MCMC or GFR)
        #' @param forest_dataset Dataset used to sample the forest
        #' @param residual Outcome used to sample the forest
        #' @param forest_samples Container of forest samples
        #' @param rng Wrapper around C++ random number generator
        #' @param feature_types Vector specifying the type of all p covariates in `forest_dataset` (0 = numeric, 1 = ordered categorical, 2 = unordered categorical)
        #' @param leaf_model_int Integer specifying the leaf model type (0 = constant leaf, 1 = univariate leaf regression, 2 = multivariate leaf regression)
        #' @param leaf_model_scale Scale parameter used in the leaf node model (should be a q x q matrix where q is the dimensionality of the basis and is only >1 when `leaf_model_int = 2`)
        #' @param variable_weights Vector specifying sampling probability for all p covariates in `forest_dataset`
        #' @param global_scale Global variance parameter
        #' @param cutpoint_grid_size (Optional) Number of unique cutpoints to consider (default: 500, currently only used when `GFR = TRUE`)
        #' @param gfr (Optional) Whether or not the forest should be sampled using the "grow-from-root" (GFR) algorithm
        #' @param pre_initialized (Optional) Whether or not the leaves are pre-initialized outside of the sampling loop (before any samples are drawn). In multi-forest implementations like BCF, this is true, though in the single-forest supervised learning implementation, we can let C++ do the initialization. Default: F.
        sample_one_iteration = function(forest_dataset, residual, forest_samples, rng, feature_types, 
                                        leaf_model_int, leaf_model_scale, variable_weights, 
                                        global_scale, cutpoint_grid_size = 500, gfr = T, 
                                        pre_initialized = F) {
            if (gfr) {
                sample_gfr_one_iteration_cpp(
                    forest_dataset$data_ptr, residual$data_ptr, 
                    forest_samples$forest_container_ptr, self$tracker_ptr, self$tree_prior_ptr, 
                    rng$rng_ptr, feature_types, cutpoint_grid_size, leaf_model_scale, 
                    variable_weights, global_scale, leaf_model_int, pre_initialized
                )
            } else {
                sample_mcmc_one_iteration_cpp(
                    forest_dataset$data_ptr, residual$data_ptr, 
                    forest_samples$forest_container_ptr, self$tracker_ptr, self$tree_prior_ptr,
                    rng$rng_ptr, feature_types, cutpoint_grid_size, leaf_model_scale, 
                    variable_weights, global_scale, leaf_model_int, pre_initialized
                ) 
            }
        }
    )
)

#' Create an R class that wraps a C++ random number generator
#'
#' @param random_seed (Optional) random seed for sampling
#'
#' @return `CppRng` object
#' @export
createRNG <- function(random_seed = -1){
    return(invisible((
        CppRNG$new(random_seed)
    )))
}

#' Create a forest model object
#'
#' @param forest_dataset `ForestDataset` object, used to initialize forest sampling data structures
#' @param feature_types Feature types (integers where 0 = numeric, 1 = ordered categorical, 2 = unordered categorical)
#' @param num_trees Number of trees in the forest being sampled
#' @param n Number of observations in `forest_dataset`
#' @param alpha Root node split probability in tree prior
#' @param beta Depth prior penalty in tree prior
#' @param min_samples_leaf Minimum number of samples in a tree leaf
#' @param max_depth Maximum depth of any tree in an ensemble
#'
#' @return `ForestModel` object
#' @export
createForestModel <- function(forest_dataset, feature_types, num_trees, n, alpha, beta, min_samples_leaf, max_depth) {
    return(invisible((
        ForestModel$new(forest_dataset, feature_types, num_trees, n, alpha, beta, min_samples_leaf, max_depth)
    )))
}

