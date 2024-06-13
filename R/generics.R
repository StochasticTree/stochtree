#' Generic function for extracting random effect samples from a model object (BCF, BART, etc...)
#' 
#' @return List of random effect samples
#' @export
getRandomEffectSamples <- function(object, ...) UseMethod("getRandomEffectSamples")
