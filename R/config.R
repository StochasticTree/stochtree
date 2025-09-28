#' Object used to get / set parameters and other model configuration options
#' for a forest model in the "low-level" stochtree interface
#'
#' @description
#' The "low-level" stochtree interface enables a high degreee of sampler
#' customization, in which users employ R wrappers around C++ objects
#' like ForestDataset, Outcome, CppRng, and ForestModel to run the
#' Gibbs sampler of a BART model with custom modifications.
#' ForestModelConfig allows users to specify / query the parameters of a
#' forest model they wish to run.

ForestModelConfig <- R6::R6Class(
    classname = "ForestModelConfig",
    cloneable = FALSE,
    public = list(
        #' @field feature_types Vector of integer-coded feature types (integers where 0 = numeric, 1 = ordered categorical, 2 = unordered categorical)
        feature_types = NULL,

        #' @field sweep_update_indices Vector of trees to update in a sweep
        sweep_update_indices = NULL,

        #' @field num_trees Number of trees in the forest being sampled
        num_trees = NULL,

        #' @field num_features Number of features in training dataset
        num_features = NULL,

        #' @field num_observations Number of observations in training dataset
        num_observations = NULL,

        #' @field leaf_dimension Dimension of the leaf model
        leaf_dimension = NULL,

        #' @field alpha Root node split probability in tree prior
        alpha = NULL,

        #' @field beta Depth prior penalty in tree prior
        beta = NULL,

        #' @field min_samples_leaf Minimum number of samples in a tree leaf
        min_samples_leaf = NULL,

        #' @field max_depth Maximum depth of any tree in the ensemble in the model. Setting to `-1` does not enforce any depth limits on trees.
        max_depth = NULL,

        #' @field leaf_model_type Integer specifying the leaf model type (0 = constant leaf, 1 = univariate leaf regression, 2 = multivariate leaf regression)
        leaf_model_type = NULL,

        #' @field leaf_model_scale Scale parameter used in Gaussian leaf models
        leaf_model_scale = NULL,

        #' @field variable_weights Vector specifying sampling probability for all p covariates in ForestDataset
        variable_weights = NULL,

        #' @field variance_forest_shape Shape parameter for IG leaf models (applicable when `leaf_model_type = 3`)
        variance_forest_shape = NULL,

        #' @field variance_forest_scale Scale parameter for IG leaf models (applicable when `leaf_model_type = 3`)
        variance_forest_scale = NULL,

        #' @field cutpoint_grid_size Number of unique cutpoints to consider
        cutpoint_grid_size = NULL,

        #' @field num_features_subsample Number of features to subsample for the GFR algorithm
        num_features_subsample = NULL,

        #' Create a new ForestModelConfig object.
        #'
        #' @param feature_types Vector of integer-coded feature types (where 0 = numeric, 1 = ordered categorical, 2 = unordered categorical)
        #' @param sweep_update_indices Vector of (0-indexed) indices of trees to update in a sweep
        #' @param num_trees Number of trees in the forest being sampled
        #' @param num_features Number of features in training dataset
        #' @param num_observations Number of observations in training dataset
        #' @param variable_weights Vector specifying sampling probability for all p covariates in ForestDataset
        #' @param leaf_dimension Dimension of the leaf model (default: `1`)
        #' @param alpha Root node split probability in tree prior (default: `0.95`)
        #' @param beta Depth prior penalty in tree prior (default: `2.0`)
        #' @param min_samples_leaf Minimum number of samples in a tree leaf (default: `5`)
        #' @param max_depth Maximum depth of any tree in the ensemble in the model. Setting to `-1` does not enforce any depth limits on trees. Default: `-1`.
        #' @param leaf_model_type Integer specifying the leaf model type (0 = constant leaf, 1 = univariate leaf regression, 2 = multivariate leaf regression). Default: `0`.
        #' @param leaf_model_scale Scale parameter used in Gaussian leaf models (can either be a scalar or a q x q matrix, where q is the dimensionality of the basis and is only >1 when `leaf_model_int = 2`). Calibrated internally as `1/num_trees`, propagated along diagonal if needed for multivariate leaf models.
        #' @param variance_forest_shape Shape parameter for IG leaf models (applicable when `leaf_model_type = 3`). Default: `1`.
        #' @param variance_forest_scale Scale parameter for IG leaf models (applicable when `leaf_model_type = 3`). Default: `1`.
        #' @param cutpoint_grid_size Number of unique cutpoints to consider (default: `100`)
        #' @param num_features_subsample Number of features to subsample for the GFR algorithm
        #'
        #' @return A new ForestModelConfig object.
        initialize = function(
            feature_types = NULL,
            sweep_update_indices = NULL,
            num_trees = NULL,
            num_features = NULL,
            num_observations = NULL,
            variable_weights = NULL,
            leaf_dimension = 1,
            alpha = 0.95,
            beta = 2.0,
            min_samples_leaf = 5,
            max_depth = -1,
            leaf_model_type = 1,
            leaf_model_scale = NULL,
            variance_forest_shape = 1.0,
            variance_forest_scale = 1.0,
            cutpoint_grid_size = 100,
            num_features_subsample = NULL
        ) {
            if (is.null(feature_types)) {
                if (is.null(num_features)) {
                    stop(
                        "Neither of `num_features` nor `feature_types` (a vector from which `num_features` can be inferred) was provided. Please provide at least one of these inputs when creating a ForestModelConfig object."
                    )
                }
                warning(
                    "`feature_types` not provided, will be assumed to be numeric"
                )
                feature_types <- rep(0, num_features)
            } else {
                if (is.null(num_features)) {
                    num_features <- length(feature_types)
                }
            }
            if (is.null(variable_weights)) {
                warning(
                    "`variable_weights` not provided, will be assumed to be equal-weighted"
                )
                variable_weights <- rep(1 / num_features, num_features)
            }
            if (is.null(num_trees)) {
                stop("num_trees must be provided")
            }
            if (!is.null(sweep_update_indices)) {
                stopifnot(min(sweep_update_indices) >= 0)
                stopifnot(max(sweep_update_indices) < num_trees)
            }
            if (is.null(num_observations)) {
                stop("num_observations must be provided")
            }
            if (num_features != length(feature_types)) {
                stop("`feature_types` must have `num_features` total elements")
            }
            if (num_features != length(variable_weights)) {
                stop(
                    "`variable_weights` must have `num_features` total elements"
                )
            }
            self$feature_types <- feature_types
            self$sweep_update_indices <- sweep_update_indices
            self$variable_weights <- variable_weights
            self$num_trees <- num_trees
            self$num_features <- num_features
            self$num_observations <- num_observations
            self$leaf_dimension <- leaf_dimension
            self$alpha <- alpha
            self$beta <- beta
            self$min_samples_leaf <- min_samples_leaf
            self$max_depth <- max_depth
            self$variance_forest_shape <- variance_forest_shape
            self$variance_forest_scale <- variance_forest_scale
            self$cutpoint_grid_size <- cutpoint_grid_size
            if (is.null(num_features_subsample)) {
                num_features_subsample <- num_features
            }
            if (num_features_subsample > num_features) {
                stop(
                    "`num_features_subsample` cannot be larger than `num_features`"
                )
            }
            if (num_features_subsample <= 0) {
                stop("`num_features_subsample` must be at least 1")
            }
            self$num_features_subsample <- num_features_subsample

            if (!(as.integer(leaf_model_type) == leaf_model_type)) {
                stop("`leaf_model_type` must be an integer between 0 and 3")
                if ((leaf_model_type < 0) | (leaf_model_type > 3)) {
                    stop("`leaf_model_type` must be an integer between 0 and 3")
                }
            }
            self$leaf_model_type <- leaf_model_type

            if (is.null(leaf_model_scale)) {
                self$leaf_model_scale <- diag(1 / num_trees, leaf_dimension)
            } else if (is.matrix(leaf_model_scale)) {
                if (ncol(leaf_model_scale) != nrow(leaf_model_scale)) {
                    stop("`leaf_model_scale` must be a square matrix")
                }
                if (ncol(leaf_model_scale) != leaf_dimension) {
                    stop(
                        "`leaf_model_scale` must have `leaf_dimension` rows and columns"
                    )
                }
                self$leaf_model_scale <- leaf_model_scale
            } else {
                if (leaf_model_scale <= 0) {
                    stop(
                        "`leaf_model_scale` must be positive, if provided as scalar"
                    )
                }
                self$leaf_model_scale <- diag(leaf_model_scale, leaf_dimension)
            }
        },

        #' @description
        #' Update feature types
        #' @param feature_types Vector of integer-coded feature types (integers where 0 = numeric, 1 = ordered categorical, 2 = unordered categorical)
        update_feature_types = function(feature_types) {
            stopifnot(length(feature_types) == self$num_features)
            self$feature_types <- feature_types
        },

        #' @description
        #' Update sweep update indices
        #' @param sweep_update_indices Vector of (0-indexed) indices of trees to update in a sweep
        update_sweep_indices = function(sweep_update_indices) {
            if (!is.null(sweep_update_indices)) {
                stopifnot(min(sweep_update_indices) >= 0)
                stopifnot(max(sweep_update_indices) < self$num_trees)
            }
            self$sweep_update_indices <- sweep_update_indices
        },

        #' @description
        #' Update variable weights
        #' @param variable_weights Vector specifying sampling probability for all p covariates in ForestDataset
        update_variable_weights = function(variable_weights) {
            stopifnot(length(variable_weights) == self$num_features)
            self$variable_weights <- variable_weights
        },

        #' @description
        #' Update root node split probability in tree prior
        #' @param alpha Root node split probability in tree prior
        update_alpha = function(alpha) {
            self$alpha <- alpha
        },

        #' @description
        #' Update depth prior penalty in tree prior
        #' @param beta Depth prior penalty in tree prior
        update_beta = function(beta) {
            self$beta <- beta
        },

        #' @description
        #' Update minimum number of samples per leaf node in the tree prior
        #' @param min_samples_leaf Minimum number of samples in a tree leaf
        update_min_samples_leaf = function(min_samples_leaf) {
            self$min_samples_leaf <- min_samples_leaf
        },

        #' @description
        #' Update max depth in the tree prior
        #' @param max_depth Maximum depth of any tree in the ensemble in the model
        update_max_depth = function(max_depth) {
            self$max_depth <- max_depth
        },

        #' @description
        #' Update scale parameter used in Gaussian leaf models
        #' @param leaf_model_scale Scale parameter used in Gaussian leaf models
        update_leaf_model_scale = function(leaf_model_scale) {
            if (is.matrix(leaf_model_scale)) {
                if (ncol(leaf_model_scale) != nrow(leaf_model_scale)) {
                    stop("`leaf_model_scale` must be a square matrix")
                }
                if (ncol(leaf_model_scale) != self$leaf_dimension) {
                    stop(
                        "`leaf_model_scale` must have `leaf_dimension` rows and columns"
                    )
                }
                self$leaf_model_scale <- leaf_model_scale
            } else {
                if (leaf_model_scale <= 0) {
                    stop(
                        "`leaf_model_scale` must be positive, if provided as scalar"
                    )
                }
                self$leaf_model_scale <- diag(leaf_model_scale, leaf_dimension)
            }
        },

        #' @description
        #' Update shape parameter for IG leaf models
        #' @param variance_forest_shape Shape parameter for IG leaf models
        update_variance_forest_shape = function(variance_forest_shape) {
            self$variance_forest_shape <- variance_forest_shape
        },

        #' @description
        #' Update scale parameter for IG leaf models
        #' @param variance_forest_scale Scale parameter for IG leaf models
        update_variance_forest_scale = function(variance_forest_scale) {
            self$variance_forest_scale <- variance_forest_scale
        },

        #' @description
        #' Update number of unique cutpoints to consider
        #' @param cutpoint_grid_size Number of unique cutpoints to consider
        update_cutpoint_grid_size = function(cutpoint_grid_size) {
            self$cutpoint_grid_size <- cutpoint_grid_size
        },

        #' @description
        #' Update number of features to subsample for the GFR algorithm
        #' @param num_features_subsample Number of features to subsample for the GFR algorithm
        update_num_features_subsample = function(num_features_subsample) {
            if (num_features_subsample > self$num_features) {
                stop(
                    "`num_features_subsample` cannot be larger than `num_features`"
                )
            }
            if (num_features_subsample <= 0) {
                stop("`num_features_subsample` must at least 1")
            }
            self$num_features_subsample <- num_features_subsample
        },

        #' @description
        #' Query feature types for this ForestModelConfig object
        #' @returns Vector of integer-coded feature types (integers where 0 = numeric, 1 = ordered categorical, 2 = unordered categorical)
        get_feature_types = function() {
            return(self$feature_types)
        },

        #' @description
        #' Query sweep update indices for this ForestModelConfig object
        #' @returns Vector of (0-indexed) indices of trees to update in a sweep
        get_sweep_indices = function() {
            return(self$sweep_update_indices)
        },

        #' @description
        #' Query variable weights for this ForestModelConfig object
        #' @returns Vector specifying sampling probability for all p covariates in ForestDataset
        get_variable_weights = function() {
            return(self$variable_weights)
        },

        #' @description
        #' Query number of trees
        #' @returns Number of trees in a forest
        get_num_trees = function() {
            return(self$num_trees)
        },

        #' @description
        #' Query number of features
        #' @returns Number of features in a forest model training set
        get_num_features = function() {
            return(self$num_features)
        },

        #' @description
        #' Query number of observations
        #' @returns Number of observations in a forest model training set
        get_num_observations = function() {
            return(self$num_observations)
        },

        #' @description
        #' Query root node split probability in tree prior for this ForestModelConfig object
        #' @returns Root node split probability in tree prior
        get_alpha = function() {
            return(self$alpha)
        },

        #' @description
        #' Query depth prior penalty in tree prior for this ForestModelConfig object
        #' @returns Depth prior penalty in tree prior
        get_beta = function() {
            return(self$beta)
        },

        #' @description
        #' Query root node split probability in tree prior for this ForestModelConfig object
        #' @returns Minimum number of samples in a tree leaf
        get_min_samples_leaf = function() {
            return(self$min_samples_leaf)
        },

        #' @description
        #' Query root node split probability in tree prior for this ForestModelConfig object
        #' @returns Maximum depth of any tree in the ensemble in the model
        get_max_depth = function() {
            return(self$max_depth)
        },

        #' @description
        #' Query (integer-coded) type of leaf model
        #' @returns Integer coded leaf model type
        get_leaf_model_type = function() {
            return(self$leaf_model_type)
        },

        #' @description
        #' Query scale parameter used in Gaussian leaf models for this ForestModelConfig object
        #' @returns Scale parameter used in Gaussian leaf models
        get_leaf_model_scale = function() {
            return(self$leaf_model_scale)
        },

        #' @description
        #' Query shape parameter for IG leaf models for this ForestModelConfig object
        #' @returns Shape parameter for IG leaf models
        get_variance_forest_shape = function() {
            return(self$variance_forest_shape)
        },

        #' @description
        #' Query scale parameter for IG leaf models for this ForestModelConfig object
        #' @returns Scale parameter for IG leaf models
        get_variance_forest_scale = function() {
            return(self$variance_forest_scale)
        },

        #' @description
        #' Query number of unique cutpoints to consider for this ForestModelConfig object
        #' @returns Number of unique cutpoints to consider
        get_cutpoint_grid_size = function() {
            return(self$cutpoint_grid_size)
        },

        #' @description
        #' Query number of features to subsample for the GFR algorithm
        #' @returns Number of features to subsample for the GFR algorithm
        get_num_features_subsample = function() {
            return(self$num_features_subsample)
        }
    )
)

#' Object used to get / set global parameters and other global model
#' configuration options in the "low-level" stochtree interface
#'
#' @description
#' The "low-level" stochtree interface enables a high degreee of sampler
#' customization, in which users employ R wrappers around C++ objects
#' like ForestDataset, Outcome, CppRng, and ForestModel to run the
#' Gibbs sampler of a BART model with custom modifications.
#' GlobalModelConfig allows users to specify / query the global parameters
#' of a model they wish to run.

GlobalModelConfig <- R6::R6Class(
    classname = "GlobalModelConfig",
    cloneable = FALSE,
    public = list(
        #' @field global_error_variance Global error variance parameter
        global_error_variance = NULL,

        #' Create a new GlobalModelConfig object.
        #'
        #' @param global_error_variance Global error variance parameter (default: `1.0`)
        #'
        #' @return A new GlobalModelConfig object.
        initialize = function(global_error_variance = 1.0) {
            self$global_error_variance <- global_error_variance
        },

        #' @description
        #' Update global error variance parameter
        #' @param global_error_variance Global error variance parameter
        update_global_error_variance = function(global_error_variance) {
            self$global_error_variance <- global_error_variance
        },

        #' @description
        #' Query global error variance parameter for this GlobalModelConfig object
        #' @returns Global error variance parameter
        get_global_error_variance = function() {
            return(self$global_error_variance)
        }
    )
)

#' Create a forest model config object
#'
#' @param feature_types Vector of integer-coded feature types (integers where 0 = numeric, 1 = ordered categorical, 2 = unordered categorical)
#' @param sweep_update_indices Vector of (0-indexed) indices of trees to update in a sweep
#' @param num_trees Number of trees in the forest being sampled
#' @param num_features Number of features in training dataset
#' @param num_observations Number of observations in training dataset
#' @param variable_weights Vector specifying sampling probability for all p covariates in ForestDataset
#' @param leaf_dimension Dimension of the leaf model (default: `1`)
#' @param alpha Root node split probability in tree prior (default: `0.95`)
#' @param beta Depth prior penalty in tree prior (default: `2.0`)
#' @param min_samples_leaf Minimum number of samples in a tree leaf (default: `5`)
#' @param max_depth Maximum depth of any tree in the ensemble in the model. Setting to `-1` does not enforce any depth limits on trees. Default: `-1`.
#' @param leaf_model_type Integer specifying the leaf model type (0 = constant leaf, 1 = univariate leaf regression, 2 = multivariate leaf regression). Default: `0`.
#' @param leaf_model_scale Scale parameter used in Gaussian leaf models (can either be a scalar or a q x q matrix, where q is the dimensionality of the basis and is only >1 when `leaf_model_int = 2`). Calibrated internally as `1/num_trees`, propagated along diagonal if needed for multivariate leaf models.
#' @param variance_forest_shape Shape parameter for IG leaf models (applicable when `leaf_model_type = 3`). Default: `1`.
#' @param variance_forest_scale Scale parameter for IG leaf models (applicable when `leaf_model_type = 3`). Default: `1`.
#' @param cutpoint_grid_size Number of unique cutpoints to consider (default: `100`)
#' @param num_features_subsample Number of features to subsample for the GFR algorithm
#' @return ForestModelConfig object
#' @export
#'
#' @examples
#' config <- createForestModelConfig(num_trees = 10, num_features = 5, num_observations = 100)
createForestModelConfig <- function(
    feature_types = NULL,
    sweep_update_indices = NULL,
    num_trees = NULL,
    num_features = NULL,
    num_observations = NULL,
    variable_weights = NULL,
    leaf_dimension = 1,
    alpha = 0.95,
    beta = 2.0,
    min_samples_leaf = 5,
    max_depth = -1,
    leaf_model_type = 1,
    leaf_model_scale = NULL,
    variance_forest_shape = 1.0,
    variance_forest_scale = 1.0,
    cutpoint_grid_size = 100,
    num_features_subsample = NULL
) {
    return(invisible(
        (ForestModelConfig$new(
            feature_types,
            sweep_update_indices,
            num_trees,
            num_features,
            num_observations,
            variable_weights,
            leaf_dimension,
            alpha,
            beta,
            min_samples_leaf,
            max_depth,
            leaf_model_type,
            leaf_model_scale,
            variance_forest_shape,
            variance_forest_scale,
            cutpoint_grid_size,
            num_features_subsample
        ))
    ))
}

#' Create a global model config object
#'
#' @param global_error_variance Global error variance parameter (default: `1.0`)
#' @return GlobalModelConfig object
#' @export
#'
#' @examples
#' config <- createGlobalModelConfig(global_error_variance = 100)
createGlobalModelConfig <- function(global_error_variance = 1.0) {
    return(invisible((GlobalModelConfig$new(global_error_variance))))
}
