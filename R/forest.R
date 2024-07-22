#' Class that stores draws from an random ensemble of decision trees
#'
#' @description
#' Wrapper around a C++ container of tree ensembles

ForestSamples <- R6::R6Class(
    classname = "ForestSamples",
    cloneable = FALSE,
    public = list(
        
        #' @field forest_container_ptr External pointer to a C++ ForestContainer class
        forest_container_ptr = NULL,
        
        #' @description
        #' Create a new ForestContainer object.
        #' @param num_trees Number of trees
        #' @param output_dimension Dimensionality of the outcome model
        #' @param is_leaf_constant Whether leaf is constant
        #' @return A new `ForestContainer` object.
        initialize = function(num_trees, output_dimension=1, is_leaf_constant=F) {
            self$forest_container_ptr <- forest_container_cpp(num_trees, output_dimension, is_leaf_constant)
        }, 
        
        #' @description
        #' Create a new ForestContainer object from a json object
        #' @param json_object Object of class `CppJson`
        #' @param json_forest_label Label referring to a particular forest (i.e. "forest_0") in the overall json hierarchy
        #' @return A new `ForestContainer` object.
        load_from_json = function(json_object, json_forest_label) {
            self$forest_container_ptr <- forest_container_from_json_cpp(json_object$json_ptr, json_forest_label)
        }, 
        
        #' @description
        #' Predict every tree ensemble on every sample in `forest_dataset`
        #' @param forest_dataset `ForestDataset` R class
        #' @return matrix of predictions with as many rows as in forest_dataset 
        #' and as many columns as samples in the `ForestContainer`
        predict = function(forest_dataset) {
            stopifnot(!is.null(forest_dataset$data_ptr))
            return(predict_forest_cpp(self$forest_container_ptr, forest_dataset$data_ptr))
        }, 
        
        #' @description
        #' Predict "raw" leaf values (without being multiplied by basis) for every tree ensemble on every sample in `forest_dataset`
        #' @param forest_dataset `ForestDataset` R class
        #' @return Array of predictions for each observation in `forest_dataset` and 
        #' each sample in the `ForestSamples` class with each prediction having the 
        #' dimensionality of the forests' leaf model. In the case of a constant leaf model 
        #' or univariate leaf regression, this array is two-dimensional (number of observations, 
        #' number of forest samples). In the case of a multivariate leaf regression, 
        #' this array is three-dimension (number of observations, leaf model dimension, 
        #' number of samples).
        predict_raw = function(forest_dataset) {
            stopifnot(!is.null(forest_dataset$data_ptr))
            # Unpack dimensions
            output_dim <- output_dimension_forest_container_cpp(self$forest_container_ptr)
            num_samples <- num_samples_forest_container_cpp(self$forest_container_ptr)
            n <- dataset_num_rows_cpp(forest_dataset$data_ptr)
            
            # Predict leaf values from forest
            predictions <- predict_forest_raw_cpp(self$forest_container_ptr, forest_dataset$data_ptr)
            
            # Extract results
            if (output_dim > 1) {
                output <- aperm(array(predictions, c(output_dim, n, num_samples)), c(2,1,3))
            } else {
                output <- predictions
            }
            return(output)
        }, 
        
        #' @description
        #' Predict "raw" leaf values (without being multiplied by basis) for a specific forest on every sample in `forest_dataset`
        #' @param forest_dataset `ForestDataset` R class
        #' @param forest_num Index of the forest sample within the container
        #' @return matrix of predictions with as many rows as in forest_dataset 
        #' and as many columns as samples in the `ForestContainer`
        predict_raw_single_forest = function(forest_dataset, forest_num) {
            stopifnot(!is.null(forest_dataset$data_ptr))
            # Unpack dimensions
            output_dim <- output_dimension_forest_container_cpp(self$forest_container_ptr)
            n <- dataset_num_rows_cpp(forest_dataset$data_ptr)
            
            # Predict leaf values from forest
            output <- predict_forest_raw_single_forest_cpp(self$forest_container_ptr, forest_dataset$data_ptr, forest_num)
            return(output)
        }, 
        
        #' @description
        #' Set a constant predicted value for every tree in the ensemble. 
        #' Stops program if any tree is more than a root node. 
        #' @param forest_num Index of the forest sample within the container.
        #' @param leaf_value Constant leaf value(s) to be fixed for each tree in the ensemble indexed by `forest_num`. Can be either a single number or a vector, depending on the forest's leaf dimension.
        set_root_leaves = function(forest_num, leaf_value) {
            stopifnot(!is.null(self$forest_container_ptr))
            stopifnot(num_samples_forest_container_cpp(self$forest_container_ptr) == 0)
            
            # Set leaf values
            if (length(leaf_value) == 1) {
                stopifnot(output_dimension_forest_container_cpp(self$forest_container_ptr) == 1)
                set_leaf_value_forest_container_cpp(self$forest_container_ptr, leaf_value)
            } else if (length(leaf_value) > 1) {
                stopifnot(output_dimension_forest_container_cpp(self$forest_container_ptr) == length(leaf_value))
                set_leaf_vector_forest_container_cpp(self$forest_container_ptr, leaf_value)
            } else {
                stop("leaf_value must be a numeric value or vector of length >= 1")
            }
        }, 
        
        #' @description
        #' Adjusts residual based on the predictions of a forest 
        #' 
        #' This is typically run just once at the beginning of a forest sampling algorithm. 
        #' After trees are initialized with constant root node predictions, their root predictions are subtracted out of the residual.
        #' @param dataset `ForestDataset` object storing the covariates and bases for a given forest
        #' @param outcome `Outcome` object storing the residuals to be updated based on forest predictions
        #' @param forest_model `ForestModel` object storing tracking structures used in training / sampling
        #' @param requires_basis Whether or not a forest requires a basis for prediction
        #' @param forest_num Index of forest used to update residuals
        #' @param add Whether forest predictions should be added to or subtracted from residuals
        adjust_residual = function(dataset, outcome, forest_model, requires_basis, forest_num, add) {
            stopifnot(!is.null(dataset$data_ptr))
            stopifnot(!is.null(outcome$data_ptr))
            stopifnot(!is.null(forest_model$tracker_ptr))
            stopifnot(!is.null(self$forest_container_ptr))
            
            adjust_residual_forest_container_cpp(
                dataset$data_ptr, outcome$data_ptr, self$forest_container_ptr, 
                forest_model$tracker_ptr, requires_basis, forest_num, add
            )
        }, 
        
        #' @description
        #' Updates the residual used for training tree ensembles by iteratively 
        #' (a) adding back in the previous prediction of each tree, (b) recomputing predictions 
        #' for each tree (caching on the C++ side), (c) subtracting the new predictions from the residual.
        #' 
        #' This is useful in cases where a basis (for e.g. leaf regression) is updated outside 
        #' of a tree sampler (as with e.g. adaptive coding for binary treatment BCF). 
        #' Once a basis has been updated, the overall "function" represented by a tree model has 
        #' changed and this should be reflected through to the residual before the next sampling loop is run.
        #' @param dataset `ForestDataset` object storing the covariates and bases for a given forest
        #' @param outcome `Outcome` object storing the residuals to be updated based on forest predictions
        #' @param forest_model `ForestModel` object storing tracking structures used in training / sampling
        #' @param forest_num Index of forest used to update residuals (starting at 1, in R style)
        update_residual = function(dataset, outcome, forest_model, forest_num) {
            stopifnot(!is.null(dataset$data_ptr))
            stopifnot(!is.null(outcome$data_ptr))
            stopifnot(!is.null(forest_model$tracker_ptr))
            stopifnot(!is.null(self$forest_container_ptr))
            
            update_residual_forest_container_cpp(
                dataset$data_ptr, outcome$data_ptr, self$forest_container_ptr, 
                forest_model$tracker_ptr, forest_num - 1
            )
        }, 
        
        #' @description
        #' Store the trees and metadata of `ForestDataset` class in a json file
        #' @param json_filename Name of output json file (must end in ".json")
        save_json = function(json_filename) {
            invisible(json_save_forest_container_cpp(self$forest_container_ptr, json_filename))
        }, 
        
        #' @description
        #' Load trees and metadata for an ensemble from a json file. Note that 
        #' any trees and metadata already present in `ForestDataset` class will 
        #' be overwritten.
        #' @param json_filename Name of model input json file (must end in ".json")
        load_json = function(json_filename) {
            invisible(json_load_forest_container_cpp(self$forest_container_ptr, json_filename))
        }, 
        
        #' @description
        #' Return number of samples in a `ForestContainer` object
        #' @return Sample count
        num_samples = function() {
            return(num_samples_forest_container_cpp(self$forest_container_ptr))
        }, 
        
        #' @description
        #' Return number of trees in each ensemble of a `ForestContainer` object
        #' @return Tree count
        num_trees = function() {
            return(num_trees_forest_container_cpp(self$forest_container_ptr))
        }, 
        
        #' @description
        #' Return output dimension of trees in a `ForestContainer` object
        #' @return Leaf node parameter size
        output_dimension = function() {
            return(output_dimension_forest_container_cpp(self$forest_container_ptr))
        }
    )
)

#' Create a container of forest samples
#'
#' @param num_trees Number of trees
#' @param output_dimension Dimensionality of the outcome model
#' @param is_leaf_constant Whether leaf is constant
#'
#' @return `ForestSamples` object
#' @export
createForestContainer <- function(num_trees, output_dimension=1, is_leaf_constant=F) {
    return(invisible((
        ForestSamples$new(num_trees, output_dimension, is_leaf_constant)
    )))
}
