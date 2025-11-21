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
