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
    #' @param max_depth Maximum depth that any tree can reach
    #' @return A new `ForestModel` object.
    initialize = function(
      forest_dataset,
      feature_types,
      num_trees,
      n,
      alpha,
      beta,
      min_samples_leaf,
      max_depth = -1
    ) {
      stopifnot(!is.null(forest_dataset$data_ptr))
      self$tracker_ptr <- forest_tracker_cpp(
        forest_dataset$data_ptr,
        feature_types,
        num_trees,
        n
      )
      self$tree_prior_ptr <- tree_prior_cpp(
        alpha,
        beta,
        min_samples_leaf,
        max_depth
      )
    },

    #' @description
    #' Run a single iteration of the forest sampling algorithm (MCMC or GFR)
    #' @param forest_dataset Dataset used to sample the forest
    #' @param residual Outcome used to sample the forest
    #' @param forest_samples Container of forest samples
    #' @param active_forest "Active" forest updated by the sampler in each iteration
    #' @param rng Wrapper around C++ random number generator
    #' @param forest_model_config ForestModelConfig object containing forest model parameters and settings
    #' @param global_model_config GlobalModelConfig object containing global model parameters and settings
    #' @param num_threads Number of threads to use in the GFR and MCMC algorithms, as well as prediction. If OpenMP is not available on a user's system, this will default to `1`, otherwise to the maximum number of available threads.
    #' @param keep_forest (Optional) Whether the updated forest sample should be saved to `forest_samples`. Default: `TRUE`.
    #' @param gfr (Optional) Whether or not the forest should be sampled using the "grow-from-root" (GFR) algorithm. Default: `TRUE`.
    sample_one_iteration = function(
      forest_dataset,
      residual,
      forest_samples,
      active_forest,
      rng,
      forest_model_config,
      global_model_config,
      num_threads = -1,
      keep_forest = TRUE,
      gfr = TRUE
    ) {
      if (active_forest$is_empty()) {
        stop(
          "`active_forest` has not yet been initialized, which is necessary to run the sampler. Please set constant values for `active_forest`'s leaves using either the `set_root_leaves` or `prepare_for_sampler` methods."
        )
      }

      # Unpack parameters from model config object
      feature_types <- forest_model_config$feature_types
      sweep_update_indices <- forest_model_config$sweep_update_indices
      leaf_model_int <- forest_model_config$leaf_model_type
      leaf_model_scale <- forest_model_config$leaf_model_scale
      variable_weights <- forest_model_config$variable_weights
      a_forest <- forest_model_config$variance_forest_shape
      b_forest <- forest_model_config$variance_forest_scale
      global_scale <- global_model_config$global_error_variance
      cutpoint_grid_size <- forest_model_config$cutpoint_grid_size
      num_features_subsample <- forest_model_config$num_features_subsample

      # Default to empty integer vector if sweep_update_indices is NULL
      if (is.null(sweep_update_indices)) {
        # sweep_update_indices <- integer(0)
        sweep_update_indices <- 0:(forest_model_config$num_trees - 1)
      }

      # Detect changes to tree prior
      if (
        forest_model_config$alpha !=
          get_alpha_tree_prior_cpp(self$tree_prior_ptr)
      ) {
        update_alpha_tree_prior_cpp(
          self$tree_prior_ptr,
          forest_model_config$alpha
        )
      }
      if (
        forest_model_config$beta != get_beta_tree_prior_cpp(self$tree_prior_ptr)
      ) {
        update_beta_tree_prior_cpp(
          self$tree_prior_ptr,
          forest_model_config$beta
        )
      }
      if (
        forest_model_config$min_samples_leaf !=
          get_min_samples_leaf_tree_prior_cpp(self$tree_prior_ptr)
      ) {
        update_min_samples_leaf_tree_prior_cpp(
          self$tree_prior_ptr,
          forest_model_config$min_samples_leaf
        )
      }
      if (
        forest_model_config$max_depth !=
          get_max_depth_tree_prior_cpp(self$tree_prior_ptr)
      ) {
        update_max_depth_tree_prior_cpp(
          self$tree_prior_ptr,
          forest_model_config$max_depth
        )
      }

      # Run the sampler
      if (gfr) {
        sample_gfr_one_iteration_cpp(
          forest_dataset$data_ptr,
          residual$data_ptr,
          forest_samples$forest_container_ptr,
          active_forest$forest_ptr,
          self$tracker_ptr,
          self$tree_prior_ptr,
          rng$rng_ptr,
          sweep_update_indices,
          feature_types,
          cutpoint_grid_size,
          leaf_model_scale,
          variable_weights,
          a_forest,
          b_forest,
          global_scale,
          leaf_model_int,
          keep_forest,
          num_features_subsample,
          num_threads
        )
      } else {
        sample_mcmc_one_iteration_cpp(
          forest_dataset$data_ptr,
          residual$data_ptr,
          forest_samples$forest_container_ptr,
          active_forest$forest_ptr,
          self$tracker_ptr,
          self$tree_prior_ptr,
          rng$rng_ptr,
          sweep_update_indices,
          feature_types,
          cutpoint_grid_size,
          leaf_model_scale,
          variable_weights,
          a_forest,
          b_forest,
          global_scale,
          leaf_model_int,
          keep_forest,
          num_threads
        )
      }
    },

    #' @description
    #' Extract an internally-cached prediction of a forest on the training dataset in a sampler.
    #' @return Vector with as many elements as observations in the training dataset
    get_cached_forest_predictions = function() {
      get_cached_forest_predictions_cpp(self$tracker_ptr)
    },

    #' @description
    #' Propagates basis update through to the (full/partial) residual by iteratively
    #' (a) adding back in the previous prediction of each tree, (b) recomputing predictions
    #' for each tree (caching on the C++ side), (c) subtracting the new predictions from the residual.
    #'
    #' This is useful in cases where a basis (for e.g. leaf regression) is updated outside
    #' of a tree sampler (as with e.g. adaptive coding for binary treatment BCF).
    #' Once a basis has been updated, the overall "function" represented by a tree model has
    #' changed and this should be reflected through to the residual before the next sampling loop is run.
    #' @param dataset `ForestDataset` object storing the covariates and bases for a given forest
    #' @param outcome `Outcome` object storing the residuals to be updated based on forest predictions
    #' @param active_forest "Active" forest updated by the sampler in each iteration
    propagate_basis_update = function(dataset, outcome, active_forest) {
      stopifnot(!is.null(dataset$data_ptr))
      stopifnot(!is.null(outcome$data_ptr))
      stopifnot(!is.null(self$tracker_ptr))
      stopifnot(!is.null(active_forest$forest_ptr))

      propagate_basis_update_active_forest_cpp(
        dataset$data_ptr,
        outcome$data_ptr,
        active_forest$forest_ptr,
        self$tracker_ptr
      )
    },

    #' @description
    #' Update the current state of the outcome (i.e. partial residual) data by subtracting the current predictions of each tree.
    #' This function is run after the `Outcome` class's `update_data` method, which overwrites the partial residual with an entirely new stream of outcome data.
    #' @param residual Outcome used to sample the forest
    #' @return None
    propagate_residual_update = function(residual) {
      propagate_trees_column_vector_cpp(
        self$tracker_ptr,
        residual$data_ptr
      )
    },

    #' @description
    #' Update alpha in the tree prior
    #' @param alpha New value of alpha to be used
    #' @return None
    update_alpha = function(alpha) {
      update_alpha_tree_prior_cpp(self$tree_prior_ptr, alpha)
    },

    #' @description
    #' Update beta in the tree prior
    #' @param beta New value of beta to be used
    #' @return None
    update_beta = function(beta) {
      update_beta_tree_prior_cpp(self$tree_prior_ptr, beta)
    },

    #' @description
    #' Update min_samples_leaf in the tree prior
    #' @param min_samples_leaf New value of min_samples_leaf to be used
    #' @return None
    update_min_samples_leaf = function(min_samples_leaf) {
      update_min_samples_leaf_tree_prior_cpp(
        self$tree_prior_ptr,
        min_samples_leaf
      )
    },

    #' @description
    #' Update max_depth in the tree prior
    #' @param max_depth New value of max_depth to be used
    #' @return None
    update_max_depth = function(max_depth) {
      update_max_depth_tree_prior_cpp(self$tree_prior_ptr, max_depth)
    },

    #' @description
    #' Update alpha in the tree prior
    #' @return Value of alpha in the tree prior
    get_alpha = function() {
      get_alpha_tree_prior_cpp(self$tree_prior_ptr)
    },

    #' @description
    #' Update beta in the tree prior
    #' @return Value of beta in the tree prior
    get_beta = function() {
      get_beta_tree_prior_cpp(self$tree_prior_ptr)
    },

    #' @description
    #' Query min_samples_leaf in the tree prior
    #' @return Value of min_samples_leaf in the tree prior
    get_min_samples_leaf = function() {
      get_min_samples_leaf_tree_prior_cpp(self$tree_prior_ptr)
    },

    #' @description
    #' Query max_depth in the tree prior
    #' @return Value of max_depth in the tree prior
    get_max_depth = function() {
      get_max_depth_tree_prior_cpp(self$tree_prior_ptr)
    }
  )
)

#' Create an R class that wraps a C++ random number generator
#'
#' @param random_seed (Optional) random seed for sampling
#'
#' @return `CppRng` object
#' @export
#'
#' @examples
#' rng <- createCppRNG(1234)
#' rng <- createCppRNG()
createCppRNG <- function(random_seed = -1) {
  return(invisible((CppRNG$new(random_seed))))
}

#' Create a forest model object
#'
#' @param forest_dataset ForestDataset object, used to initialize forest sampling data structures
#' @param forest_model_config ForestModelConfig object containing forest model parameters and settings
#' @param global_model_config GlobalModelConfig object containing global model parameters and settings
#'
#' @return `ForestModel` object
#' @export
#'
#' @examples
#' num_trees <- 100
#' n <- 100
#' p <- 10
#' alpha <- 0.95
#' beta <- 2.0
#' min_samples_leaf <- 2
#' max_depth <- 10
#' feature_types <- as.integer(rep(0, p))
#' X <- matrix(runif(n*p), ncol = p)
#' forest_dataset <- createForestDataset(X)
#' forest_model_config <- createForestModelConfig(feature_types=feature_types,
#'                                                num_trees=num_trees, num_features=p,
#'                                                num_observations=n, alpha=alpha, beta=beta,
#'                                                min_samples_leaf=min_samples_leaf,
#'                                                max_depth=max_depth, leaf_model_type=1)
#' global_model_config <- createGlobalModelConfig(global_error_variance=1.0)
#' forest_model <- createForestModel(forest_dataset, forest_model_config, global_model_config)
createForestModel <- function(
  forest_dataset,
  forest_model_config,
  global_model_config
) {
  return(invisible(
    (ForestModel$new(
      forest_dataset,
      forest_model_config$feature_types,
      forest_model_config$num_trees,
      forest_model_config$num_observations,
      forest_model_config$alpha,
      forest_model_config$beta,
      forest_model_config$min_samples_leaf,
      forest_model_config$max_depth
    ))
  ))
}


#' Draw `sample_size` samples from `population_vector` without replacement, weighted by `sampling_probabilities`
#'
#' @param population_vector Vector from which to draw samples.
#' @param sampling_probabilities Vector of probabilities of drawing each element of `population_vector`.
#' @param sample_size Number of samples to draw from `population_vector`. Must be less than or equal to `length(population_vector)`
#'
#' @returns Vector of size `sample_size`
#' @export
#'
#' @examples
#' a <- as.integer(c(4,3,2,5,1,9,7))
#' p <- c(0.7,0.2,0.05,0.02,0.01,0.01,0.01)
#' num_samples <- 5
#' sample_without_replacement(a, p, num_samples)
sample_without_replacement <- function(
  population_vector,
  sampling_probabilities,
  sample_size
) {
  return(sample_without_replacement_integer_cpp(
    population_vector,
    sampling_probabilities,
    sample_size
  ))
}
