#' Create a StochTree dataset used for sampling forests
#'
#' @param covariates Matrix of covariates used for the tree splits
#' @param basis (Optional) matrix of basis functions used to define a regression model in the tree leaves
#' @param weights (Optional) vector of variance weights
#'
#' @return External pointer to the dataset
#' @export
#'
#' @examples
#' n <- 100
#' p_X <- 10
#' p_W <- 1
#' X <- matrix(runif(n*p_X), ncol = p_X)
#' W <- matrix(runif(n*p_W), ncol = p_W)
#' data_ptr <- create_forest_dataset(X, W)
create_forest_dataset <- function(covariates, basis = NULL, weights = NULL) {
    # Create external pointer to the C++ dataset object
    data_ptr <- create_forest_dataset_cpp()
    
    # Add covariates
    forest_dataset_add_covariates_cpp(data_ptr, covariates)
    
    # Add basis
    if (!is.null(basis)) {
        forest_dataset_add_basis_cpp(data_ptr, basis)
    }
    
    # Add weights
    if (!is.null(weights)) {
        forest_dataset_add_weights_cpp(data_ptr, weights)
    }
    
    return(data_ptr)
}

#' Create a StochTree C++ ColumVector object (most commonly this is used for the outcome variable)
#'
#' @param vector Vector of values
#'
#' @return External pointer to the ColumnVector object
#' @export
#'
#' @examples
#' n <- 100
#' x <- runif(n)
#' y <- x*10 + rnorm(n, 0, 0.5)
#' outcome_ptr <- create_column_vector(y)
create_column_vector <- function(vector) {
    col_vector_ptr <- create_column_vector_cpp(vector)
    return(col_vector_ptr)
}
