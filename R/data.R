#' Dataset used to sample a forest
#'
#' @description
#' A dataset consists of three matrices / vectors: covariates,
#' bases, and variance weights. Both the basis vector and variance
#' weights are optional.

ForestDataset <- R6::R6Class(
  classname = "ForestDataset",
  cloneable = FALSE,
  public = list(
    #' @field data_ptr External pointer to a C++ ForestDataset class
    data_ptr = NULL,

    #' @description
    #' Create a new ForestDataset object.
    #' @param covariates Matrix of covariates
    #' @param basis (Optional) Matrix of bases used to define a leaf regression
    #' @param variance_weights (Optional) Vector of observation-specific variance weights
    #' @return A new `ForestDataset` object.
    initialize = function(
      covariates,
      basis = NULL,
      variance_weights = NULL
    ) {
      self$data_ptr <- create_forest_dataset_cpp()
      forest_dataset_add_covariates_cpp(self$data_ptr, covariates)
      if (!is.null(basis)) {
        forest_dataset_add_basis_cpp(self$data_ptr, basis)
      }
      if (!is.null(variance_weights)) {
        forest_dataset_add_weights_cpp(self$data_ptr, variance_weights)
      }
    },

    #' @description
    #' Update basis matrix in a dataset
    #' @param basis Updated matrix of bases used to define a leaf regression
    update_basis = function(basis) {
      stopifnot(self$has_basis())
      forest_dataset_update_basis_cpp(self$data_ptr, basis)
    },

    #' @description
    #' Update variance_weights in a dataset
    #' @param variance_weights Updated vector of variance weights used to define individual variance / case weights
    #' @param exponentiate Whether or not input vector should be exponentiated before being written to the Dataset's variance weights. Default: F.
    update_variance_weights = function(variance_weights, exponentiate = F) {
      stopifnot(self$has_variance_weights())
      forest_dataset_update_var_weights_cpp(
        self$data_ptr,
        variance_weights,
        exponentiate
      )
    },

    #' @description
    #' Return number of observations in a `ForestDataset` object
    #' @return Observation count
    num_observations = function() {
      return(dataset_num_rows_cpp(self$data_ptr))
    },

    #' @description
    #' Return number of covariates in a `ForestDataset` object
    #' @return Covariate count
    num_covariates = function() {
      return(dataset_num_covariates_cpp(self$data_ptr))
    },

    #' @description
    #' Return number of bases in a `ForestDataset` object
    #' @return Basis count
    num_basis = function() {
      return(dataset_num_basis_cpp(self$data_ptr))
    },

    #' @description
    #' Return covariates as an R matrix
    #' @return Covariate data
    get_covariates = function() {
      return(forest_dataset_get_covariates_cpp(self$data_ptr))
    },

    #' @description
    #' Return bases as an R matrix
    #' @return Basis data
    get_basis = function() {
      return(forest_dataset_get_basis_cpp(self$data_ptr))
    },

    #' @description
    #' Return variance weights as an R vector
    #' @return Variance weight data
    get_variance_weights = function() {
      return(forest_dataset_get_variance_weights_cpp(self$data_ptr))
    },

    #' @description
    #' Whether or not a dataset has a basis matrix
    #' @return True if basis matrix is loaded, false otherwise
    has_basis = function() {
      return(dataset_has_basis_cpp(self$data_ptr))
    },

    #' @description
    #' Whether or not a dataset has variance weights
    #' @return True if variance weights are loaded, false otherwise
    has_variance_weights = function() {
      return(dataset_has_variance_weights_cpp(self$data_ptr))
    }
  )
)

#' Outcome / partial residual used to sample an additive model.
#'
#' @description
#' The outcome class is wrapper around a vector of (mutable)
#' outcomes for ML tasks (supervised learning, causal inference).
#' When an additive tree ensemble is sampled, the outcome used to
#' sample a specific model term is the "partial residual" consisting
#' of the outcome minus the predictions of every other model term
#' (trees, group random effects, etc...).

Outcome <- R6::R6Class(
  classname = "Outcome",
  cloneable = FALSE,
  public = list(
    #' @field data_ptr External pointer to a C++ Outcome class
    data_ptr = NULL,

    #' @description
    #' Create a new Outcome object.
    #' @param outcome Vector of outcome values
    #' @return A new `Outcome` object.
    initialize = function(outcome) {
      self$data_ptr <- create_column_vector_cpp(outcome)
    },

    #' @description
    #' Extract raw data in R from the underlying C++ object
    #' @return R vector containing (copy of) the values in `Outcome` object
    get_data = function() {
      return(get_residual_cpp(self$data_ptr))
    },

    #' @description
    #' Update the current state of the outcome (i.e. partial residual) data by adding the values of `update_vector`
    #' @param update_vector Vector to be added to outcome
    #' @return None
    add_vector = function(update_vector) {
      if (!is.numeric(update_vector)) {
        stop("update_vector must be a numeric vector or 2d matrix")
      } else {
        dim_vec <- dim(update_vector)
        if (!is.null(dim_vec)) {
          if (length(dim_vec) > 2) {
            stop(
              "if update_vector is provided as a matrix, it must be 2d"
            )
          }
          update_vector <- as.numeric(update_vector)
        }
      }
      add_to_column_vector_cpp(self$data_ptr, update_vector)
    },

    #' @description
    #' Update the current state of the outcome (i.e. partial residual) data by subtracting the values of `update_vector`
    #' @param update_vector Vector to be subtracted from outcome
    #' @return None
    subtract_vector = function(update_vector) {
      if (!is.numeric(update_vector)) {
        stop("update_vector must be a numeric vector or 2d matrix")
      } else {
        dim_vec <- dim(update_vector)
        if (!is.null(dim_vec)) {
          if (length(dim_vec) > 2) {
            stop(
              "if update_vector is provided as a matrix, it must be 2d"
            )
          }
          update_vector <- as.numeric(update_vector)
        }
      }
      subtract_from_column_vector_cpp(self$data_ptr, update_vector)
    },

    #' @description
    #' Update the current state of the outcome (i.e. partial residual) data by replacing each element with the elements of `new_vector`
    #' @param new_vector Vector from which to overwrite the current data
    #' @return None
    update_data = function(new_vector) {
      if (!is.numeric(new_vector)) {
        stop("update_vector must be a numeric vector or 2d matrix")
      } else {
        dim_vec <- dim(new_vector)
        if (!is.null(dim_vec)) {
          if (length(dim_vec) > 2) {
            stop(
              "if update_vector is provided as a matrix, it must be 2d"
            )
          }
          new_vector <- as.numeric(new_vector)
        }
      }
      overwrite_column_vector_cpp(self$data_ptr, new_vector)
    }
  )
)

#' Dataset used to sample a random effects model
#'
#' @description
#' A dataset consists of three matrices / vectors: group labels,
#' bases, and variance weights. Variance weights are optional.

RandomEffectsDataset <- R6::R6Class(
  classname = "RandomEffectsDataset",
  cloneable = FALSE,
  public = list(
    #' @field data_ptr External pointer to a C++ RandomEffectsDataset class
    data_ptr = NULL,

    #' @description
    #' Create a new RandomEffectsDataset object.
    #' @param group_labels Vector of group labels
    #' @param basis Matrix of bases used to define the random effects regression (for an intercept-only model, pass an array of ones)
    #' @param variance_weights (Optional) Vector of observation-specific variance weights
    #' @return A new `RandomEffectsDataset` object.
    initialize = function(group_labels, basis, variance_weights = NULL) {
      self$data_ptr <- create_rfx_dataset_cpp()
      rfx_dataset_add_group_labels_cpp(self$data_ptr, group_labels)
      rfx_dataset_add_basis_cpp(self$data_ptr, basis)
      if (!is.null(variance_weights)) {
        rfx_dataset_add_weights_cpp(self$data_ptr, variance_weights)
      }
    },

    #' @description
    #' Update basis matrix in a dataset
    #' @param basis Updated matrix of bases used to define random slopes / intercepts
    update_basis = function(basis) {
      stopifnot(self$has_basis())
      rfx_dataset_update_basis_cpp(self$data_ptr, basis)
    },

    #' @description
    #' Update variance_weights in a dataset
    #' @param variance_weights Updated vector of variance weights used to define individual variance / case weights
    #' @param exponentiate Whether or not input vector should be exponentiated before being written to the RandomEffectsDataset's variance weights. Default: F.
    update_variance_weights = function(variance_weights, exponentiate = F) {
      stopifnot(self$has_variance_weights())
      rfx_dataset_update_var_weights_cpp(
        self$data_ptr,
        variance_weights,
        exponentiate
      )
    },

    #' @description
    #' Return number of observations in a `RandomEffectsDataset` object
    #' @return Observation count
    num_observations = function() {
      return(rfx_dataset_num_rows_cpp(self$data_ptr))
    },

    #' @description
    #' Return dimension of the basis matrix in a `RandomEffectsDataset` object
    #' @return Basis vector count
    num_basis = function() {
      return(rfx_dataset_num_basis_cpp(self$data_ptr))
    },

    #' @description
    #' Return group labels as an R vector
    #' @return Group label data
    get_group_labels = function() {
      return(rfx_dataset_get_group_labels_cpp(self$data_ptr))
    },

    #' @description
    #' Return bases as an R matrix
    #' @return Basis data
    get_basis = function() {
      return(rfx_dataset_get_basis_cpp(self$data_ptr))
    },

    #' @description
    #' Return variance weights as an R vector
    #' @return Variance weight data
    get_variance_weights = function() {
      return(rfx_dataset_get_variance_weights_cpp(self$data_ptr))
    },

    #' @description
    #' Whether or not a dataset has group label indices
    #' @return True if group label vector is loaded, false otherwise
    has_group_labels = function() {
      return(rfx_dataset_has_group_labels_cpp(self$data_ptr))
    },

    #' @description
    #' Whether or not a dataset has a basis matrix
    #' @return True if basis matrix is loaded, false otherwise
    has_basis = function() {
      return(rfx_dataset_has_basis_cpp(self$data_ptr))
    },

    #' @description
    #' Whether or not a dataset has variance weights
    #' @return True if variance weights are loaded, false otherwise
    has_variance_weights = function() {
      return(rfx_dataset_has_variance_weights_cpp(self$data_ptr))
    }
  )
)

#' Create a forest dataset object
#'
#' @param covariates Matrix of covariates
#' @param basis (Optional) Matrix of bases used to define a leaf regression
#' @param variance_weights (Optional) Vector of observation-specific variance weights
#'
#' @return `ForestDataset` object
#' @export
#'
#' @examples
#' covariate_matrix <- matrix(runif(10*100), ncol = 10)
#' basis_matrix <- matrix(rnorm(3*100), ncol = 3)
#' weight_vector <- rnorm(100)
#' forest_dataset <- createForestDataset(covariate_matrix)
#' forest_dataset <- createForestDataset(covariate_matrix, basis_matrix)
#' forest_dataset <- createForestDataset(covariate_matrix, basis_matrix, weight_vector)
createForestDataset <- function(
  covariates,
  basis = NULL,
  variance_weights = NULL
) {
  return(invisible((ForestDataset$new(covariates, basis, variance_weights))))
}

#' Create an outcome object
#'
#' @param outcome Vector of outcome values
#'
#' @return `Outcome` object
#' @export
#'
#' @examples
#' X <- matrix(runif(10*100), ncol = 10)
#' y <- -5 + 10*(X[,1] > 0.5) + rnorm(100)
#' outcome <- createOutcome(y)
createOutcome <- function(outcome) {
  return(invisible((Outcome$new(outcome))))
}

#' Create a random effects dataset object
#'
#' @param group_labels Vector of group labels
#' @param basis Matrix of bases used to define the random effects regression (for an intercept-only model, pass an array of ones)
#' @param variance_weights (Optional) Vector of observation-specific variance weights
#'
#' @return `RandomEffectsDataset` object
#' @export
#'
#' @examples
#' rfx_group_ids <- sample(1:2, size = 100, replace = TRUE)
#' rfx_basis <- matrix(rnorm(3*100), ncol = 3)
#' weight_vector <- rnorm(100)
#' rfx_dataset <- createRandomEffectsDataset(rfx_group_ids, rfx_basis)
#' rfx_dataset <- createRandomEffectsDataset(rfx_group_ids, rfx_basis, weight_vector)
createRandomEffectsDataset <- function(
  group_labels,
  basis,
  variance_weights = NULL
) {
  return(invisible(
    (RandomEffectsDataset$new(group_labels, basis, variance_weights))
  ))
}
