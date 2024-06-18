#' Generic function for extracting random effect samples from a model object (BCF, BART, etc...)
#' 
#' @param object Fitted model object from which to extract random effects
#' @param ... Other parameters to be used in random effects extraction
#' @return List of random effect samples
#' @export
getRandomEffectSamples <- function(object, ...) UseMethod("getRandomEffectSamples")
