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
