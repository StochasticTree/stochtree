#' Predict from a container of forest samples
#'
#' @param forest_samples Pointer to a ForestContainer object
#' @param dataset Pointer to a ForestDataset object
#'
#' @return Matrix of predictions
#' @export
predict_forest <- function(forest_samples, dataset) {
    predictions <- predict_forest_cpp(forest_samples, dataset)
    return(predictions)
}

#' Extract raw leaf values from a container of forest samples
#'
#' @param forest_samples Pointer to a ForestContainer object
#' @param dataset Pointer to a ForestDataset object
#'
#' @return If leaves are univariates, matrix of predictions with 
#' nrow equal to the number of observations in dataset and ncol 
#' equal to the number of samples drawn in forest_samples. 
#' if leaves are multivariate, an array of predictions with 
#' dimensions equal to (1) the number of observations in 
#' dataset, (2) the number of dimensions of the leaves in 
#' forest_samples, and (3) the number of samples drawn in 
#' forest_samples.
#' @export
predict_forest_raw <- function(forest_samples, dataset) {
    # Unpack dimensions
    output_dim <- forest_output_dimension(forest_samples)
    num_samples <- num_forest_samples(forest_samples)
    n <- num_dataset_rows(dataset)
    
    # Predict leaf values from forest
    predictions <- predict_forest_raw_cpp(forest_samples, dataset)
    
    # Extract results
    if (output_dim > 1) {
        output <- aperm(array(predictions, c(output_dim, n, num_samples)), c(2,1,3))
    } else {
        output <- predictions
    }
    return(output)
}
