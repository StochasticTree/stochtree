#' Calibrate the scale parameter on an inverse gamma prior for the global error variance as in Chipman et al (2022)
#' 
#' Chipman, H., George, E., Hahn, R., McCulloch, R., Pratola, M. and Sparapani, R. (2022). Bayesian Additive Regression Trees, Computational Approaches. In Wiley StatsRef: Statistics Reference Online (eds N. Balakrishnan, T. Colton, B. Everitt, W. Piegorsch, F. Ruggeri and J.L. Teugels). https://doi.org/10.1002/9781118445112.stat08288
#'
#' @param y Outcome to be modeled using BART, BCF or another nonparametric ensemble method.
#' @param X Covariates to be used to partition trees in an ensemble or series of ensemble.
#' @param W (Optional) Basis used to define a "leaf regression" model for each decision tree. The "classic" BART model assumes a constant leaf parameter, which is equivalent to a "leaf regression" on a basis of all ones, though it is not necessary to pass a vector of ones, here or to the BART function. Default: `NULL`.
#' @param nu The shape parameter for the global error variance's IG prior. The scale parameter in the Sparapani et al (2021) parameterization is defined as `nu*lambda` where `lambda` is the output of this function. Default: `3`.
#' @param quant (Optional) Quantile of the inverse gamma prior distribution represented by a linear-regression-based overestimate of `sigma^2`. Default: `0.9`.
#' @param standardize (Optional) Whether or not outcome should be standardized (`(y-mean(y))/sd(y)`) before calibration of `lambda`. Default: `TRUE`.
#'
#' @return Value of `lambda` which determines the scale parameter of the global error variance prior (`sigma^2 ~ IG(nu,nu*lambda)`)
#' @export 
#'
#' @examples
#' n <- 100
#' p <- 5
#' X <- matrix(runif(n*p), ncol = p)
#' y <- 10*X[,1] - 20*X[,2] + rnorm(n)
#' nu <- 3
#' lambda <- calibrate_inverse_gamma_error_variance(y, X, nu = nu)
#' sigma2hat <- mean(resid(lm(y~X))^2)
#' mean(var(y)/rgamma(100000, nu, rate = nu*lambda) < sigma2hat)
calibrate_inverse_gamma_error_variance <- function(y, X, W = NULL, nu = 3, quant = 0.9, standardize = TRUE) {
    # Compute regression basis
    if (!is.null(W)) basis <- cbind(X, W)
    else basis <- X
    # Standardize outcome if requested
    if (standardize) y <- (y-mean(y))/sd(y)
    # Compute the "regression-based" overestimate of sigma^2
    sigma2hat <- mean(resid(lm(y~basis))^2)
    # Calibrate lambda based on the implied quantile of sigma2hat
    return((sigma2hat*qgamma(1-quant,nu))/nu)
}
