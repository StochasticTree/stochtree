#' @title Extract Random Effect Samples Generic Function
#' @description
#' Generic function for extracting random effect samples from a model object (BCF, BART, etc...)
#'
#' @param object Fitted model object from which to extract random effects
#' @param ... Other parameters to be used in random effects extraction
#' @return List of random effect samples
#' @export
#'
#' @examples
#' n <- 100
#' p <- 10
#' X <- matrix(runif(n*p), ncol = p)
#' rfx_group_ids <- sample(1:2, size = n, replace = TRUE)
#' rfx_basis <- rep(1.0, n)
#' y <- (-5 + 10*(X[,1] > 0.5)) + (-2*(rfx_group_ids==1)+2*(rfx_group_ids==2)) + rnorm(n)
#' bart_model <- bart(X_train=X, y_train=y, rfx_group_ids_train=rfx_group_ids,
#'                    rfx_basis_train = rfx_basis, num_gfr=0, num_mcmc=10)
#' rfx_samples <- getRandomEffectSamples(bart_model)
getRandomEffectSamples <- function(object, ...) {
  UseMethod("getRandomEffectSamples")
}

#' @title Extract Parameter Samples Generic Function
#' @description
#' Generic function for extracting parameter samples from a model object (BCF, BART, etc...)
#'
#' @param object Fitted model object from which to extract parameter samples
#' @param term Name of the parameter to extract (e.g., `"sigma2"`, `"y_hat_train"`, etc.)
#' @return Parameter sample array
#' @export
#'
#' @examples
#' n <- 100
#' p <- 10
#' X <- matrix(runif(n*p), ncol = p)
#' rfx_group_ids <- sample(1:2, size = n, replace = TRUE)
#' rfx_basis <- rep(1.0, n)
#' y <- (-5 + 10*(X[,1] > 0.5)) + (-2*(rfx_group_ids==1)+2*(rfx_group_ids==2)) + rnorm(n)
#' bart_model <- bart(X_train=X, y_train=y, rfx_group_ids_train=rfx_group_ids,
#'                    rfx_basis_train = rfx_basis, num_gfr=0, num_mcmc=10)
#' sigma2_samples <- extractParameter(bart_model, "sigma2")
extractParameter <- function(object, term) {
  UseMethod("extractParameter")
}

#' @title Extract Forest Samples Generic Function
#' @description
#' Generic function for extracting forest samples from a model object (BCF, BART, etc...)
#'
#' @param object Fitted model object from which to extract forest samples
#' @param term Name of the forest to extract (e.g., `"mean"`, `"variance"`, etc.)
#' @return Forest sample array
#' @export
#'
#' @examples
#' n <- 100
#' p <- 10
#' X <- matrix(runif(n*p), ncol = p)
#' rfx_group_ids <- sample(1:2, size = n, replace = TRUE)
#' rfx_basis <- rep(1.0, n)
#' y <- (-5 + 10*(X[,1] > 0.5)) + (-2*(rfx_group_ids==1)+2*(rfx_group_ids==2)) + rnorm(n)
#' bart_model <- bart(X_train=X, y_train=y, rfx_group_ids_train=rfx_group_ids,
#'                    rfx_basis_train = rfx_basis, num_gfr=0, num_mcmc=10)
#' mean_forest <- extractForest(bart_model, "mean")
extractForest <- function(object, term) {
  UseMethod("extractForest")
}

#' @title Continue Sampling a Model
#' @description
#' Continue sampling a model on new data.
#'
#' @param object Fitted model object to continue sampling
#' @param ... Other parameters to be used in sampling
#'
#' @return List of sampling outputs and a wrapper around the sampled forests (which can be used for in-memory prediction on new data, or serialized to JSON on disk).
#' @export
#'
#' @examples
#' n <- 100
#' p <- 5
#' X <- matrix(runif(n*p), ncol = p)
#' f_XW <- (
#'     ((0 <= X[,1]) & (0.25 > X[,1])) * (-7.5) +
#'     ((0.25 <= X[,1]) & (0.5 > X[,1])) * (-2.5) +
#'     ((0.5 <= X[,1]) & (0.75 > X[,1])) * (2.5) +
#'     ((0.75 <= X[,1]) & (1 > X[,1])) * (7.5)
#' )
#' noise_sd <- 1
#' y <- f_XW + rnorm(n, 0, noise_sd)
#' test_set_pct <- 0.2
#' n_test <- round(test_set_pct*n)
#' n_train <- n - n_test
#' test_inds <- sort(sample(1:n, n_test, replace = FALSE))
#' train_inds <- (1:n)[!((1:n) %in% test_inds)]
#' X_test <- X[test_inds,]
#' X_train <- X[train_inds,]
#' y_test <- y[test_inds]
#' y_train <- y[train_inds]
#' bart_model <- bart(X_train = X_train, y_train = y_train,
#'                    num_gfr = 10, num_burnin = 0, num_mcmc = 10)
#' continueSampling(bart_model, X_train = X_train, y_train = y_train,
#'                  num_gfr = 10, num_burnin = 0, num_mcmc = 10)
continueSampling <- function(object, ...) {
  UseMethod("continueSampling")
}
