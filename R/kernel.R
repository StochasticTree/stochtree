#' Compute vector of forest leaf indices
#' 
#' @description Compute and return a vector representation of a forest's leaf predictions for 
#' every observation in a dataset.
#' 
#' The vector has a "row-major" format that can be easily re-represented as 
#' as a CSR sparse matrix: elements are organized so that the first `n` elements 
#' correspond to leaf predictions for all `n` observations in a dataset for the 
#' first tree in an ensemble, the next `n` elements correspond to predictions for 
#' the second tree and so on. The "data" for each element corresponds to a uniquely 
#' mapped column index that corresponds to a single leaf of a single tree (i.e. 
#' if tree 1 has 3 leaves, its column indices range from 0 to 2, and then tree 2's 
#' leaf indices begin at 3, etc...).
#'
#' @param model_object Object of type `bartmodel` or `bcf` corresponding to a BART / BCF model with at least one forest sample
#' @param covariates Covariates to use for prediction. Must have the same dimensions / column types as the data used to train a forest.
#' @param forest_type Which forest to use from `model_object`. 
#' Valid inputs depend on the model type, and whether or not a given forest was sampled in that model.
#' 
#'   **1. BART**
#'
#'   - `'mean'`: Extracts leaf indices for the mean forest
#'   - `'variance'`: Extracts leaf indices for the variance forest
#' 
#'   **2. BCF**
#'
#'   - `'prognostic'`: Extracts leaf indices for the prognostic forest
#'   - `'treatment'`: Extracts leaf indices for the treatment effect forest
#'   - `'variance'`: Extracts leaf indices for the variance forest
#' 
#' @param forest_inds (Optional) Indices of the forest sample(s) for which to compute leaf indices. If not provided, 
#' this function will return leaf indices for every sample of a forest. 
#' This function uses 1-indexing, so the first forest sample corresponds to `forest_num = 1`, and so on.
#' @return List of vectors. Each vector is of size `num_obs * num_trees`, where `num_obs = nrow(covariates)` 
#' and `num_trees` is the number of trees in the relevant forest of `model_object`. 
#' @export
computeForestLeafIndices <- function(model_object, covariates, forest_type, forest_inds=NULL) {
    # Extract relevant forest container
    stopifnot(class(model_object) %in% c("bartmodel", "bcf"))
    model_type <- ifelse(class(model_object)=="bartmodel", "bart", "bcf")
    if (model_type == "bart") {
        stopifnot(forest_type %in% c("mean", "variance"))
        if (forest_type=="mean") {
            if (!model_object$model_params$include_mean_forest) {
                stop("Mean forest was not sampled in the bart model provided")
            }
            forest_container <- model_object$mean_forests
        } else if (forest_type=="variance") {
            if (!model_object$model_params$include_variance_forest) {
                stop("Variance forest was not sampled in the bart model provided")
            }
            forest_container <- model_object$variance_forests
        }
    } else {
        stopifnot(forest_type %in% c("prognostic", "treatment", "variance"))
        if (forest_type=="prognostic") {
            forest_container <- model_object$forests_mu
        } else if (forest_type=="treatment") {
            forest_container <- model_object$forests_tau
        } else if (forest_type=="variance") {
            if (!model_object$model_params$include_variance_forest) {
                stop("Variance forest was not sampled in the bcf model provided")
            }
            forest_container <- model_object$variance_forests
        }
    }
    
    # Preprocess covariates
    if ((!is.data.frame(covariates)) && (!is.matrix(covariates))) {
        stop("covariates must be a matrix or dataframe")
    }
    train_set_metadata <- model_object$train_set_metadata
    covariates_processed <- preprocessPredictionData(covariates, train_set_metadata)
    
    # Preprocess forest indices
    num_forests <- forest_container$num_samples()
    if (is.null(forest_inds)) {
        forest_inds <- as.integer(1:num_forests - 1)
    } else {
        stopifnot(all(forest_inds <= num_forests))
        stopifnot(all(forest_inds >= 1))
        forest_inds <- as.integer(forest_inds - 1)
    }
    
    # Compute leaf indices
    leaf_ind_matrix <- compute_leaf_indices_cpp(
        forest_container$forest_container_ptr, 
        covariates_processed, forest_inds
    )

    return(leaf_ind_matrix)
}

#' Compute vector of forest leaf scale parameters
#' 
#' @description Return each forest's leaf node scale parameters.
#' 
#' If leaf scale is not sampled for the forest in question, throws an error that the 
#' leaf model does not have a stochastic scale parameter.
#' 
#' @param model_object Object of type `bartmodel` or `bcf` corresponding to a BART / BCF model with at least one forest sample
#' @param forest_type Which forest to use from `model_object`. 
#' Valid inputs depend on the model type, and whether or not a given forest was sampled in that model.
#' 
#'   **1. BART**
#'
#'   - `'mean'`: Extracts leaf indices for the mean forest
#'   - `'variance'`: Extracts leaf indices for the variance forest
#' 
#'   **2. BCF**
#'
#'   - `'prognostic'`: Extracts leaf indices for the prognostic forest
#'   - `'treatment'`: Extracts leaf indices for the treatment effect forest
#'   - `'variance'`: Extracts leaf indices for the variance forest
#' 
#' @param forest_inds (Optional) Indices of the forest sample(s) for which to compute leaf indices. If not provided, 
#' this function will return leaf indices for every sample of a forest. 
#' This function uses 1-indexing, so the first forest sample corresponds to `forest_num = 1`, and so on.
#' @return Vector of size `length(forest_inds)` with the leaf scale parameter for each requested forest.
#' @export
computeForestLeafVariances <- function(model_object, forest_type, forest_inds=NULL) {
    # Extract relevant forest container
    stopifnot(class(model_object) %in% c("bartmodel", "bcf"))
    model_type <- ifelse(class(model_object)=="bartmodel", "bart", "bcf")
    if (model_type == "bart") {
        stopifnot(forest_type %in% c("mean", "variance"))
        if (forest_type=="mean") {
            if (!model_object$model_params$include_mean_forest) {
                stop("Mean forest was not sampled in the bart model provided")
            }
            if (model_object$model_params$sample_sigma_leaf == F) {
                stop("Leaf scale parameter was not sampled for the mean forest in the bart model provided")
            }
            leaf_scale_vector <- model_object$sigma2_leaf_samples
        } else if (forest_type=="variance") {
            if (!model_object$model_params$include_variance_forest) {
                stop("Variance forest was not sampled in the bart model provided")
            }
            stop("Leaf scale parameter was not sampled for the variance forest in the bart model provided")
        }
    } else {
        stopifnot(forest_type %in% c("prognostic", "treatment", "variance"))
        if (forest_type=="prognostic") {
            if (model_object$model_params$sample_sigma_leaf_mu == F) {
                stop("Leaf scale parameter was not sampled for the prognostic forest in the bcf model provided")
            }
            leaf_scale_vector <- model_object$sigma_leaf_mu_samples
        } else if (forest_type=="treatment") {
            if (model_object$model_params$sample_sigma_leaf_tau == F) {
                stop("Leaf scale parameter was not sampled for the treatment effect forest in the bcf model provided")
            }
            leaf_scale_vector <- model_object$sigma_leaf_tau_samples
        } else if (forest_type=="variance") {
            if (!model_object$model_params$include_variance_forest) {
                stop("Variance forest was not sampled in the bcf model provided")
            }
            stop("Leaf scale parameter was not sampled for the variance forest in the bcf model provided")
        }
    }
    
    # Preprocess forest indices
    num_forests <- forest_container$num_samples()
    if (is.null(forest_inds)) {
        forest_inds <- as.integer(1:num_forests)
    } else {
        stopifnot(all(forest_inds <= num_forests))
        stopifnot(all(forest_inds >= 1))
        forest_inds <- as.integer(forest_inds)
    }
    
    # Gather leaf scale parameters
    leaf_scale_params <- leaf_scale_vector[forest_inds]

    return(leaf_scale_params)
}

#' Compute and return the largest possible leaf index computable by `computeForestLeafIndices` for the forests in a designated forest sample container.
#'
#' @param model_object Object of type `bartmodel` or `bcf` corresponding to a BART / BCF model with at least one forest sample
#' @param covariates Covariates to use for prediction. Must have the same dimensions / column types as the data used to train a forest.
#' @param forest_type Which forest to use from `model_object`. 
#' Valid inputs depend on the model type, and whether or not a 
#' 
#'   **1. BART**
#'
#'   - `'mean'`: Extracts leaf indices for the mean forest
#'   - `'variance'`: Extracts leaf indices for the variance forest
#' 
#'   **2. BCF**
#'
#'   - `'prognostic'`: Extracts leaf indices for the prognostic forest
#'   - `'treatment'`: Extracts leaf indices for the treatment effect forest
#'   - `'variance'`: Extracts leaf indices for the variance forest
#' 
#' @param forest_inds (Optional) Indices of the forest sample(s) for which to compute leaf indices. If not provided, 
#' this function will return leaf indices for every sample of a forest. 
#' This function uses 1-indexing, so the first forest sample corresponds to `forest_num = 1`, and so on.
#' @return Vector containing the largest possible leaf index computable by `computeForestLeafIndices` for the forests in a designated forest sample container.
#' @export
computeMaxLeafIndex <- function(model_object, covariates, forest_type, forest_inds=NULL) {
    # Extract relevant forest container
    stopifnot(class(model_object) %in% c("bartmodel", "bcf"))
    model_type <- ifelse(class(model_object)=="bartmodel", "bart", "bcf")
    if (model_type == "bart") {
        stopifnot(forest_type %in% c("mean", "variance"))
        if (forest_type=="mean") {
            if (!model_object$model_params$include_mean_forest) {
                stop("Mean forest was not sampled in the bart model provided")
            }
            forest_container <- model_object$mean_forests
        } else if (forest_type=="variance") {
            if (!model_object$model_params$include_variance_forest) {
                stop("Variance forest was not sampled in the bart model provided")
            }
            forest_container <- model_object$variance_forests
        }
    } else {
        stopifnot(forest_type %in% c("prognostic", "treatment", "variance"))
        if (forest_type=="prognostic") {
            forest_container <- model_object$forests_mu
        } else if (forest_type=="treatment") {
            forest_container <- model_object$forests_tau
        } else if (forest_type=="variance") {
            if (!model_object$model_params$include_variance_forest) {
                stop("Variance forest was not sampled in the bcf model provided")
            }
            forest_container <- model_object$variance_forests
        }
    }
    
    # Preprocess forest indices
    num_forests <- forest_container$num_samples()
    if (is.null(forest_inds)) {
        forest_inds <- as.integer(1:num_forests - 1)
    } else {
        stopifnot(all(forest_inds <= num_forests))
        stopifnot(all(forest_inds >= 1))
        forest_inds <- as.integer(forest_inds - 1)
    }
    
    # Compute leaf indices
    output <- rep(NA, length(forest_inds))
    for (i in 1:length(forest_inds)) {
        output[i] <- forest_container_get_max_leaf_index_cpp(
            forest_container$forest_container_ptr,forest_inds[i]
        )
    }
    
    return(output)
}
