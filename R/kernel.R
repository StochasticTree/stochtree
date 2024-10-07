#' Class that provides functionality for statistical kernel definition and 
#' computation based on shared leaf membership of observations in a tree ensemble.
#'
#' @description
#' Computes leaf membership internally as a sparse matrix and also calculates a 
#' (dense) kernel based on the sparse matrix all in C++.

ForestKernel <- R6::R6Class(
    classname = "ForestKernel",
    cloneable = FALSE,
    public = list(
        
        #' @field forest_kernel_ptr External pointer to a C++ StochTree::ForestKernel class
        forest_kernel_ptr = NULL,

        #' @description
        #' Create a new ForestKernel object.
        #' @return A new `ForestKernel` object.
        initialize = function() {
            # Initialize forest kernel and return external pointer to the class
            self$forest_kernel_ptr <- forest_kernel_cpp()
        }, 
        
        #' @description
        #' Compute the leaf indices of each tree in the ensemble for every observation in a dataset. 
        #' Stores the result internally, which can be extracted from the class via a call to `get_leaf_indices`.
        #' @param covariates_train Matrix of training set covariates at which to compute leaf indices
        #' @param covariates_test (Optional) Matrix of test set covariates at which to compute leaf indices
        #' @param forest_container Object of type `ForestSamples`
        #' @param forest_num Index of the forest in forest_container to be assessed
        #' @return List of vectors. If `covariates_test = NULL` the list has one element (train set leaf indices), and 
        #' otherwise the list has two elements (train and test set leaf indices).
        compute_leaf_indices = function(covariates_train, covariates_test = NULL, forest_container, forest_num) {
            # Convert to matrix format if not provided as such
            if ((is.null(dim(covariates_train))) && (!is.null(covariates_train))) {
                covariates_train <- as.matrix(covariates_train)
            }
            if ((is.null(dim(covariates_test))) && (!is.null(covariates_test))) {
                covariates_test <- as.matrix(covariates_test)
            }
            
            # Compute the leaf indices
            result = list()
            if (is.null(covariates_test)) {
                forest_kernel_compute_leaf_indices_train_cpp(
                    self$forest_kernel_ptr, covariates_train, 
                    forest_container$forest_container_ptr, forest_num
                )
                result[["leaf_indices_train"]] = forest_kernel_get_train_leaf_indices_cpp(self$forest_kernel_ptr)
            } else {
                forest_kernel_compute_leaf_indices_train_test_cpp(
                    self$forest_kernel_ptr, covariates_train, covariates_test, 
                    forest_container$forest_container_ptr, forest_num
                )
                result[["leaf_indices_train"]] = forest_kernel_get_train_leaf_indices_cpp(self$forest_kernel_ptr)
                result[["leaf_indices_test"]] = forest_kernel_get_test_leaf_indices_cpp(self$forest_kernel_ptr)
            }
            return(result)
        }, 
        
        #' @description
        #' Compute the kernel implied by a tree ensemble. This function calls `compute_leaf_indices`, 
        #' so it is not necessary to call both. `compute_leaf_indices` is exposed at the class level 
        #' to allow for extracting the vector of leaf indices for an ensemble directly in R.
        #' @param covariates_train Matrix of training set covariates at which to assess ensemble kernel
        #' @param covariates_test (Optional) Matrix of test set covariates at which to assess ensemble kernel
        #' @param forest_container Object of type `ForestSamples`
        #' @param forest_num Index of the forest in forest_container to be assessed
        #' @return List of matrices. If `covariates_test = NULL`, the list contains 
        #' one `n_train` x `n_train` matrix, where `n_train = nrow(covariates_train)`. 
        #' This matrix is the kernel defined by `W_train %*% t(W_train)` where `W_train` 
        #' is a matrix with `n_train` rows and as many columns as there are total leaves in an ensemble. 
        #' If `covariates_test` is not `NULL`, the list contains two more matrices defined by 
        #' `W_test %*% t(W_train)` and `W_test %*% t(W_test)`.
        compute_kernel = function(covariates_train, covariates_test = NULL, forest_container, forest_num) {
            # Convert to matrix format if not provided as such
            if ((is.null(dim(covariates_train))) && (!is.null(covariates_train))) {
                covariates_train <- as.matrix(covariates_train)
            }
            if ((is.null(dim(covariates_test))) && (!is.null(covariates_test))) {
                covariates_test <- as.matrix(covariates_test)
            }
            
            # Compute the kernels
            num_trees <- forest_container$num_trees()
            n_train = nrow(covariates_train)
            kernel_train = matrix(0., nrow = n_train, ncol = n_train)
            inverse_kernel_train = matrix(0., nrow = n_train, ncol = n_train)
            if (is.null(covariates_test)) {
                result = forest_kernel_compute_kernel_train_cpp(
                    self$forest_kernel_ptr, covariates_train, 
                    forest_container$forest_container_ptr, forest_num
                )
                names(result) <- c("kernel_train")
            } else {
                n_test = nrow(covariates_test)
                kernel_test_train = matrix(0., nrow = n_test, ncol = n_train)
                kernel_test = matrix(0., nrow = n_test, ncol = n_test)
                result = forest_kernel_compute_kernel_train_test_cpp(
                    self$forest_kernel_ptr, covariates_train, covariates_test, 
                    forest_container$forest_container_ptr, forest_num
                )
                names(result) <- c("kernel_train", "kernel_test_train", "kernel_test")
            }
            
            # Divide each matrix by num_trees
            for (i in 1:length(result)) {result[[i]] <- result[[i]] / num_trees}
            
            return(result)
        }
    )
)

#' Create a `ForestKernel` object
#'
#' @return `ForestKernel` object
#' @export
createForestKernel <- function() {
    return(invisible((
        ForestKernel$new()
    )))
}

#' Compute a kernel from a tree ensemble, defined by the fraction 
#' of trees of an ensemble in which two observations fall into the 
#' same leaf.
#'
#' @param bart_model Object of type `bartmodel` corresponding to a BART model with at least one sample
#' @param X_train "Training" dataframe. In a traditional Gaussian process kriging context, this 
#' corresponds to the observations for which outcomes are observed.
#' @param X_test (Optional) "Test" dataframe. In a traditional Gaussian process kriging context, this 
#' corresponds to the observations for which outcomes are unobserved and must be estimated 
#' based on the kernels k(X_test,X_test), k(X_test,X_train), and k(X_train,X_train). If not provided, 
#' this function will only compute k(X_train, X_train).
#' @param forest_num (Optional) Index of the forest sample to use for kernel computation. If not provided, 
#' this function will use the last forest.
#' @param forest_type (Optional) Whether to compute the kernel from the mean or variance forest. Default: "mean". Specify "variance" for the variance forest. 
#' All other inputs are invalid. Must have sampled the relevant forest or an error will occur.
#' @return List of kernel matrices. If `X_test = NULL`, the list contains 
#' one `n_train` x `n_train` matrix, where `n_train = nrow(X_train)`. 
#' This matrix is the kernel defined by `W_train %*% t(W_train)` where `W_train` 
#' is a matrix with `n_train` rows and as many columns as there are total leaves in an ensemble. 
#' If `X_test` is not `NULL`, the list contains two more matrices defined by 
#' `W_test %*% t(W_train)` and `W_test %*% t(W_test)`.
#' @export
computeForestKernels <- function(bart_model, X_train, X_test=NULL, forest_num=NULL, forest_type="mean") {
    stopifnot(class(bart_model)=="bartmodel")
    if (forest_type=="mean") {
        if (!bart_model$model_params$include_mean_forest) {
            stop("Mean forest was not sampled in the bart model provided")
        }
    } else if (forest_type=="variance") {
        if (!bart_model$model_params$include_variance_forest) {
            stop("Variance forest was not sampled in the bart model provided")
        }
    } else {
        stop("Must provide either 'mean' or 'variance' for the `forest_type` parameter")
    }
    
    # Preprocess covariates
    if (!is.data.frame(X_train)) {
        stop("X_train must be a dataframe")
    }
    if (!is.data.frame(X_test)) {
        stop("X_test must be a dataframe")
    }
    train_set_metadata <- bart_model$train_set_metadata
    X_train <- preprocessPredictionDataFrame(X_train, train_set_metadata)
    X_test <- preprocessPredictionDataFrame(X_test, train_set_metadata)
    
    # Data checks
    stopifnot(bart_model$model_params$num_covariates == ncol(X_train))
    stopifnot(bart_model$model_params$num_covariates == ncol(X_test))

    # Initialize and compute kernel
    forest_kernel <- createForestKernel()
    num_samples <- bart_model$model_params$num_samples
    stopifnot(forest_num <= num_samples)
    sample_index <- ifelse(is.null(forest_num), num_samples-1, forest_num-1)
    if (forest_type=="mean") {
        return(forest_kernel$compute_kernel(
            covariates_train = X_train, covariates_test = X_test,
            forest_container = bart_model$mean_forests, forest_num = sample_index
        ))
    } else if (forest_type=="variance") {
        return(forest_kernel$compute_kernel(
            covariates_train = X_train, covariates_test = X_test,
            forest_container = bart_model$variance_forests, forest_num = sample_index
        ))
    }
}

#' Compute and return a vector representation of a forest's leaf predictions for 
#' every observation in a dataset.
#' The vector has a "column-major" format that can be easily re-represented as 
#' as a CSC sparse matrix: elements are organized so that the first `n` elements 
#' correspond to leaf predictions for all `n` observations in a dataset for the 
#' first tree in an ensemble, the next `n` elements correspond to predictions for 
#' the second tree and so on. The "data" for each element corresponds to a uniquely 
#' mapped column index that corresponds to a single leaf of a single tree (i.e. 
#' if tree 1 has 3 leaves, its column indices range from 0 to 2, and then tree 2's 
#' leaf indices begin at 3, etc...).
#' Users may pass a single dataset (which we refer to here as a "training set") 
#' or two datasets (which we refer to as "training and test sets"). This verbiage 
#' hints that one potential use-case for a matrix of leaf indices is to define a 
#' ensemble-based kernel for kriging.
#'
#' @param bart_model Object of type `bartmodel` corresponding to a BART model with at least one sample
#' @param X_train Matrix of "training" data. In a traditional Gaussian process kriging context, this 
#' corresponds to the observations for which outcomes are observed.
#' @param X_test (Optional) Matrix of "test" data. In a traditional Gaussian process kriging context, this 
#' corresponds to the observations for which outcomes are unobserved and must be estimated 
#' based on the kernels k(X_test,X_test), k(X_test,X_train), and k(X_train,X_train). If not provided, 
#' this function will only compute k(X_train, X_train).
#' @param forest_num (Optional) Index of the forest sample to use for kernel computation. If not provided, 
#' this function will use the last forest.
#' @param forest_type (Optional) Whether to compute the kernel from the mean or variance forest. Default: "mean". Specify "variance" for the variance forest. 
#' All other inputs are invalid. Must have sampled the relevant forest or an error will occur.
#' @return List of vectors. If `X_test = NULL`, the list contains 
#' one vector of length `n_train * num_trees`, where `n_train = nrow(X_train)` 
#' and `num_trees` is the number of trees in `bart_model`. If `X_test` is not `NULL`, 
#' the list contains another vector of length `n_test * num_trees`.
#' @export
computeForestLeafIndices <- function(bart_model, X_train, X_test=NULL, forest_num=NULL, forest_type="mean") {
    stopifnot(class(bart_model)=="bartmodel")
    if (forest_type=="mean") {
        if (!bart_model$model_params$include_mean_forest) {
            stop("Mean forest was not sampled in the bart model provided")
        }
    } else if (forest_type=="variance") {
        if (!bart_model$model_params$include_variance_forest) {
            stop("Variance forest was not sampled in the bart model provided")
        }
    } else {
        stop("Must provide either 'mean' or 'variance' for the `forest_type` parameter")
    }
    forest_kernel <- createForestKernel()
    num_samples <- bart_model$model_params$num_samples
    stopifnot(forest_num <= num_samples)
    sample_index <- ifelse(is.null(forest_num), num_samples-1, forest_num-1)
    if (forest_type == "mean") {
        return(forest_kernel$compute_leaf_indices(
            covariates_train = X_train, covariates_test = X_test,
            forest_container = bart_model$mean_forests, forest_num = sample_index
        ))
    } else if (forest_type == "variance") {
        return(forest_kernel$compute_leaf_indices(
            covariates_train = X_train, covariates_test = X_test,
            forest_container = bart_model$variance_forests, forest_num = sample_index
        ))
    }
}
