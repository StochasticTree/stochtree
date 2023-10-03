#' Samples from a BART model using the provided data
#'
#' @param model_matrix Matrix containing data (including outcome of interest and treatment if present)
#' @param params List of parameters of the form `list(param1=-1,...)`
#'
#' @return Object of class BART
#' @export
#'
#' @examples
#' n <- 1000
#' p <- 5
#' X <- matrix(runif(n*p), ncol=p)
#' y <- X[,1] * 5 + X[,2] * 10 + rnorm(n, 0, 0.5)
#' model_matrix <- cbind(y, X)
#' param_list = list(label_column=0, num_trees=50, num_burnin=0, num_samples=20, min_samples_in_leaf=5)
#' result <- bart(model_matrix, param_list)
bart <- function(model_matrix, params = list()){
    params_string <- stochtree.params2str(params = params)
    y <- model_matrix[,params$label_column + 1]
    # ybar <- mean(y)
    # ysig <- sd(y)
    # y_std <- (y - ybar)/ysig
    # model_matrix[,params$label_column + 1] <- y_std
    ptr <- bart_sample_cpp(model_matrix, params_string)
    result <- list(ptr=ptr)
    # result <- list(ptr=ptr, ybar=ybar, ysig=ysig)
    class(result) <- "bart_samples"
    return(result)
}

#' Predicts from a sampled BART model and returns the result as a matrix
#'
#' @param model_matrix Matrix containing prediction data (including outcome of interest and treatment if present)
#' @param params List of parameters of the form `list(param1=-1,...)`
#'
#' @return Matrix with observations in rows and sampled BART draws in columns
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
#' bart_samples <- bart(model_matrix_train, param_list)
#' predictions <- predict(bart_samples, model_matrix_test, param_list)
#' for (i in 1:ncol(predictions)) {plot(y[test_inds], predictions[,i], ylab = paste0("yhat_",i)); abline(0,1); Sys.sleep(0.25)}
#' plot(y[test_inds], rowMeans(predictions), ylab = "yhat_mean"); abline(0,1)
predict.bart_samples <- function(bart, model_matrix, params = list()){
    params_string <- stochtree.params2str(params = params)
    # result <- bart_predict_cpp(bart$ptr, model_matrix, params_string)*bart$ysig + bart$ybar
    result <- bart_predict_cpp(bart$ptr, model_matrix, params_string)
    return(result)
}
