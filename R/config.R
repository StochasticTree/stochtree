#' #' Dataset used to get / set parameters and other model configuration options
#' #' for the "low-level" stochtree interface
#' #'
#' #' @description
#' #' The "low-level" stochtree interface enables a high degreee of sampler 
#' #' customization, in which users employ R wrappers around C++ objects 
#' #' like ForestDataset, Outcome, CppRng, and ForestModel to run the 
#' #' Gibbs sampler of a BART model with custom modifications. 
#' #' ModelConfig allows users to specify / query the parameters of a 
#' #' tree model they wish to run.
#' 
#' ModelConfig <- R6::R6Class(
#'     classname = "ModelConfig",
#'     cloneable = FALSE,
#'     public = list(
#'         
#'         #' @field data_ptr External pointer to a C++ ModelConfig class
#'         data_ptr = NULL,
#'         
#'         #' @description
#'         #' Create a new ForestDataset object.
#'         #' @param covariates Matrix of covariates
#'         #' @param basis (Optional) Matrix of bases used to define a leaf regression
#'         #' @param variance_weights (Optional) Vector of observation-specific variance weights
#'         #' @return A new `ForestDataset` object.
#'         initialize = function(covariates, basis=NULL, variance_weights=NULL) {
#'             self$data_ptr <- create_forest_dataset_cpp()
#'             forest_dataset_add_covariates_cpp(self$data_ptr, covariates)
#'             if (!is.null(basis)) {
#'                 forest_dataset_add_basis_cpp(self$data_ptr, basis)
#'             }
#'             if (!is.null(variance_weights)) {
#'                 forest_dataset_add_weights_cpp(self$data_ptr, variance_weights)
#'             }
#'         }, 
#'         
#'         #' @description
#'         #' Update basis matrix in a dataset
#'         #' @param basis Updated matrix of bases used to define a leaf regression
#'         update_basis = function(basis) {
#'             stopifnot(self$has_basis())
#'             forest_dataset_update_basis_cpp(self$data_ptr, basis)
#'         }, 
#'         
#'         #' @description
#'         #' Return number of observations in a `ForestDataset` object
#'         #' @return Observation count
#'         num_observations = function() {
#'             return(dataset_num_rows_cpp(self$data_ptr))
#'         }, 
#'         
#'         #' @description
#'         #' Return number of covariates in a `ForestDataset` object
#'         #' @return Covariate count
#'         num_covariates = function() {
#'             return(dataset_num_covariates_cpp(self$data_ptr))
#'         }, 
#'         
#'         #' @description
#'         #' Return number of bases in a `ForestDataset` object
#'         #' @return Basis count
#'         num_basis = function() {
#'             return(dataset_num_basis_cpp(self$data_ptr))
#'         }, 
#'         
#'         #' @description
#'         #' Whether or not a dataset has a basis matrix
#'         #' @return True if basis matrix is loaded, false otherwise
#'         has_basis = function() {
#'             return(dataset_has_basis_cpp(self$data_ptr))
#'         }, 
#'         
#'         #' @description
#'         #' Whether or not a dataset has variance weights
#'         #' @return True if variance weights are loaded, false otherwise
#'         has_variance_weights = function() {
#'             return(dataset_has_variance_weights_cpp(self$data_ptr))
#'         }
#'     ),
#'     private = list(
#'         feature_types = NULL, 
#'         num_trees = NULL, 
#'         num_observations = NULL, 
#'         alpha = NULL, 
#'         beta = NULL, 
#'         min_samples_leaf = NULL, 
#'         max_depth = NULL, 
#'     )
#' )
#' 
#' #' Create an model config object
#' #'
#' #' @return `ModelConfig` object
#' #' @export
#' #' 
#' #' @examples
#' #' X <- matrix(runif(10*100), ncol = 10)
#' #' y <- -5 + 10*(X[,1] > 0.5) + rnorm(100)
#' #' config <- createModelConfig(y)
#' createModelConfig <- function(){
#'     return(invisible((
#'         createModelConfig$new()
#'     )))
#' }
