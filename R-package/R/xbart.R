#' Samples from an XBART model using the provided data
#'
#' @param model_matrix Matrix containing data (including outcome of interest and treatment if present)
#' @param params List of parameters of the form `list(param1=-1,...)`
#'
#' @return Object of class XBART
#' @export
#'
#' @examples
#' n <- 1000
#' p <- 5
#' X <- matrix(runif(n*p), ncol=p)
#' y <- X[,1] * 5 + X[,2] * 10 + rnorm(n, 0, 0.5)
#' model_matrix <- cbind(y, X)
#' param_list = list(label_column=0, num_trees=50, num_burnin=0, num_samples=20, min_samples_in_leaf=5)
#' result <- xbart(model_matrix, param_list)
xbart <- function(y, X, omega, num_samples, num_burnin, num_trees, nu, lambda, cutpoint_grid_size, random_seed = -1){
    ptr <- xbart_sample_cpp(y, X, omega, num_samples, num_burnin, num_trees, nu, lambda, cutpoint_grid_size, random_seed)
    result <- list(ptr=ptr)
    class(result) <- "xbart_samples"
    return(result)
}

#' Predicts from a sampled XBART model and returns the result as a matrix
#'
#' @param model_matrix Matrix containing prediction data (including outcome of interest and treatment if present)
#' @param params List of parameters of the form `list(param1=-1,...)`
#'
#' @return Matrix with observations in rows and sampled XBART draws in columns
#' @export
#'
#' @examples
#' n <- 2000
#' p <- 10
#' train_inds <- sample(1:n, round(n*0.5), replace = F)
#' test_inds <- (1:n)[!((1:n) %in% train_inds)]
#' X <- matrix(runif(n*p), ncol=p)
#' y <- X[,1] * 5 + X[,2] * 10 + rnorm(n, 0, 0.1)
#' model_matrix_train <- cbind(y[train_inds], X[train_inds,])
#' model_matrix_test <- cbind(y[test_inds], X[test_inds,])
#' param_list = list(label_column=0, num_trees=50, num_burnin=10, num_samples=20, min_samples_in_leaf=5)
#' xbart_samples <- xbart(model_matrix_train, param_list)
#' predictions <- predict(xbart_samples, model_matrix_test, param_list)
#' for (i in 1:ncol(predictions)) {plot(y[test_inds], predictions[,i], ylab = paste0("yhat_",i)); abline(0,1); Sys.sleep(0.25)}
#' plot(y[test_inds], rowMeans(predictions), ylab = "yhat_mean"); abline(0,1)
predict.xbart_samples <- function(xbart, X, omega, num_samples){
    result <- xbart_predict_cpp(xbart$ptr, X, omega, num_samples)
    return(result)
}
