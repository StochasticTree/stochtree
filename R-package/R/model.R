#' Class that wraps a C++ random number generator (for reproducibility)

CppRNG <- R6::R6Class(
    classname = "CppRNG",
    cloneable = FALSE,
    public = list(
        
        #' @field data_ptr External pointer to a C++ std::mt19937 class
        rng_ptr = NULL,

        #' @description
        #' Create a new CppRNG object.
        #' @param random_seed (Optional) random seed for sampling
        #' @return A new `CppRNG` object.
        initialize = function(random_seed) {
            self$rng_ptr <- rng_cpp(random_seed)
        }
    )
)

#' Class that defines and samples a forest model

ForestModel <- R6::R6Class(
    classname = "ForestModel",
    cloneable = FALSE,
    public = list(
        
        #' @field data_ptr External pointer to a C++ ForestTracker class
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
        #' @return A new `ForestModel` object.
        initialize = function(forest_dataset, feature_types, num_trees, n, alpha, beta, min_samples_leaf) {
            stopifnot(!is.null(forest_dataset$data_ptr))
            self$tracker_ptr <- forest_tracker_cpp(forest_dataset$data_ptr, feature_types, num_trees, n)
            self$tree_prior_ptr <- tree_prior_cpp(alpha, beta, min_samples_leaf)
        }, 
        
        #' @description
        #' Run a single iteration of the forest sampling algorithm (MCMC or GFR)
        #' @param basis Updated matrix of bases used to define a leaf regression
        #' @return 
        sample_one_iteration = function(forest_dataset, residual, forest_samples, rng, feature_types, 
                                        leaf_model_int, leaf_model_scale, variable_weights, 
                                        global_scale, cutpoint_grid_size = 500, gfr = T) {
            if (gfr) {
                sample_gfr_one_iteration_cpp(
                    forest_dataset$data_ptr, residual$data_ptr, 
                    forest_samples$forest_container_ptr, self$tracker_ptr, self$tree_prior_ptr, 
                    rng$rng_ptr, feature_types, cutpoint_grid_size, leaf_model_scale, 
                    variable_weights, global_scale, leaf_model_int
                )
            } else {
                sample_mcmc_one_iteration_cpp(
                    forest_dataset$data_ptr, residual$data_ptr, 
                    forest_samples$forest_container_ptr, self$tracker_ptr, self$tree_prior_ptr,
                    rng$rng_ptr, feature_types, cutpoint_grid_size, leaf_model_scale, 
                    variable_weights, global_scale, leaf_model_int
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
#'
#' @return `ForestModel` object
#' @export
createForestModel <- function(forest_dataset, feature_types, num_trees, n, alpha, beta, min_samples_leaf) {
    return(invisible((
        ForestModel$new(forest_dataset, feature_types, num_trees, n, alpha, beta, min_samples_leaf)
    )))
}

