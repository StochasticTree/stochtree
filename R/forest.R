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
        #' @param leaf_dimension Dimensionality of the outcome model
        #' @param is_leaf_constant Whether leaf is constant
        #' @param is_exponentiated Whether forest predictions should be exponentiated before being returned
        #' @return A new `ForestContainer` object.
        initialize = function(
            num_trees,
            leaf_dimension = 1,
            is_leaf_constant = FALSE,
            is_exponentiated = FALSE
        ) {
            self$forest_container_ptr <- forest_container_cpp(
                num_trees,
                leaf_dimension,
                is_leaf_constant,
                is_exponentiated
            )
        },

        #' @description
        #' Collapse forests in this container by a pre-specified batch size.
        #' For example, if we have a container of twenty 10-tree forests, and we
        #' specify a `batch_size` of 5, then this method will yield four 50-tree
        #' forests. "Excess" forests remaining after the size of a forest container
        #' is divided by `batch_size` will be pruned from the beginning of the
        #' container (i.e. earlier sampled forests will be deleted). This method
        #' has no effect if `batch_size` is larger than the number of forests
        #' in a container.
        #' @param batch_size Number of forests to be collapsed into a single forest
        collapse = function(batch_size) {
            container_size <- self$num_samples()
            if ((batch_size <= container_size) && (batch_size > 1)) {
                reverse_container_inds <- seq(container_size, 1, -1)
                num_clean_batches <- container_size %/% batch_size
                batch_inds <- (reverse_container_inds -
                    (container_size -
                        (container_size %/% num_clean_batches) *
                            num_clean_batches) -
                    1) %/%
                    batch_size
                for (batch_ind in unique(batch_inds[batch_inds >= 0])) {
                    merge_forest_inds <- sort(
                        reverse_container_inds[batch_inds == batch_ind] - 1
                    )
                    num_merge_forests <- length(merge_forest_inds)
                    self$combine_forests(merge_forest_inds)
                    for (i in num_merge_forests:2) {
                        self$delete_sample(merge_forest_inds[i])
                    }
                    forest_scale_factor <- 1.0 / num_merge_forests
                    self$multiply_forest(
                        merge_forest_inds[1],
                        forest_scale_factor
                    )
                }
                if (min(batch_inds) < 0) {
                    delete_forest_inds <- sort(
                        reverse_container_inds[batch_inds < 0] - 1
                    )
                    for (i in length(delete_forest_inds):1) {
                        self$delete_sample(delete_forest_inds[i])
                    }
                }
            }
        },

        #' @description
        #' Merge specified forests into a single forest
        #' @param forest_inds Indices of forests to be combined (0-indexed)
        combine_forests = function(forest_inds) {
            stopifnot(max(forest_inds) < self$num_samples())
            stopifnot(min(forest_inds) >= 0)
            stopifnot(length(forest_inds) > 1)
            stopifnot(all(as.integer(forest_inds) == forest_inds))
            forest_inds_sorted <- as.integer(sort(forest_inds))
            combine_forests_forest_container_cpp(
                self$forest_container_ptr,
                forest_inds_sorted
            )
        },

        #' @description
        #' Add a constant value to every leaf of every tree of a given forest
        #' @param forest_index Index of forest whose leaves will be modified (0-indexed)
        #' @param constant_value Value to add to every leaf of every tree of the forest at `forest_index`
        add_to_forest = function(forest_index, constant_value) {
            stopifnot(forest_index < self$num_samples())
            stopifnot(forest_index >= 0)
            add_to_forest_forest_container_cpp(
                self$forest_container_ptr,
                forest_index,
                constant_value
            )
        },

        #' @description
        #' Multiply every leaf of every tree of a given forest by constant value
        #' @param forest_index Index of forest whose leaves will be modified (0-indexed)
        #' @param constant_multiple Value to multiply through by every leaf of every tree of the forest at `forest_index`
        multiply_forest = function(forest_index, constant_multiple) {
            stopifnot(forest_index < self$num_samples())
            stopifnot(forest_index >= 0)
            multiply_forest_forest_container_cpp(
                self$forest_container_ptr,
                forest_index,
                constant_multiple
            )
        },

        #' @description
        #' Create a new `ForestContainer` object from a json object
        #' @param json_object Object of class `CppJson`
        #' @param json_forest_label Label referring to a particular forest (i.e. "forest_0") in the overall json hierarchy
        #' @return A new `ForestContainer` object.
        load_from_json = function(json_object, json_forest_label) {
            self$forest_container_ptr <- forest_container_from_json_cpp(
                json_object$json_ptr,
                json_forest_label
            )
        },

        #' @description
        #' Append to a `ForestContainer` object from a json object
        #' @param json_object Object of class `CppJson`
        #' @param json_forest_label Label referring to a particular forest (i.e. "forest_0") in the overall json hierarchy
        #' @return None
        append_from_json = function(json_object, json_forest_label) {
            forest_container_append_from_json_cpp(
                self$forest_container_ptr,
                json_object$json_ptr,
                json_forest_label
            )
        },

        #' @description
        #' Create a new `ForestContainer` object from a json object
        #' @param json_string JSON string which parses into object of class `CppJson`
        #' @param json_forest_label Label referring to a particular forest (i.e. "forest_0") in the overall json hierarchy
        #' @return A new `ForestContainer` object.
        load_from_json_string = function(json_string, json_forest_label) {
            self$forest_container_ptr <- forest_container_from_json_string_cpp(
                json_string,
                json_forest_label
            )
        },

        #' @description
        #' Append to a `ForestContainer` object from a json object
        #' @param json_string JSON string which parses into object of class `CppJson`
        #' @param json_forest_label Label referring to a particular forest (i.e. "forest_0") in the overall json hierarchy
        #' @return None
        append_from_json_string = function(json_string, json_forest_label) {
            forest_container_append_from_json_string_cpp(
                self$forest_container_ptr,
                json_string,
                json_forest_label
            )
        },

        #' @description
        #' Predict every tree ensemble on every sample in `forest_dataset`
        #' @param forest_dataset `ForestDataset` R class
        #' @return matrix of predictions with as many rows as in forest_dataset
        #' and as many columns as samples in the `ForestContainer`
        predict = function(forest_dataset) {
            stopifnot(!is.null(forest_dataset$data_ptr))
            return(predict_forest_cpp(
                self$forest_container_ptr,
                forest_dataset$data_ptr
            ))
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
            output_dim <- leaf_dimension_forest_container_cpp(
                self$forest_container_ptr
            )
            num_samples <- num_samples_forest_container_cpp(
                self$forest_container_ptr
            )
            n <- dataset_num_rows_cpp(forest_dataset$data_ptr)

            # Predict leaf values from forest
            predictions <- predict_forest_raw_cpp(
                self$forest_container_ptr,
                forest_dataset$data_ptr
            )
            if (output_dim > 1) {
                dim(predictions) <- c(n, output_dim, num_samples)
            } else {
                dim(predictions) <- c(n, num_samples)
            }

            return(predictions)
        },

        #' @description
        #' Predict "raw" leaf values (without being multiplied by basis) for a specific forest on every sample in `forest_dataset`
        #' @param forest_dataset `ForestDataset` R class
        #' @param forest_num Index of the forest sample within the container
        #' @return matrix of predictions with as many rows as in forest_dataset
        #' and as many columns as dimensions in the leaves of trees in `ForestContainer`
        predict_raw_single_forest = function(forest_dataset, forest_num) {
            stopifnot(!is.null(forest_dataset$data_ptr))
            # Unpack dimensions
            output_dim <- leaf_dimension_forest_container_cpp(
                self$forest_container_ptr
            )
            n <- dataset_num_rows_cpp(forest_dataset$data_ptr)

            # Predict leaf values from forest
            output <- predict_forest_raw_single_forest_cpp(
                self$forest_container_ptr,
                forest_dataset$data_ptr,
                forest_num
            )
            return(output)
        },

        #' @description
        #' Predict "raw" leaf values (without being multiplied by basis) for a specific tree in a specific forest on every observation in `forest_dataset`
        #' @param forest_dataset `ForestDataset` R class
        #' @param forest_num Index of the forest sample within the container
        #' @param tree_num Index of the tree to be queried
        #' @return matrix of predictions with as many rows as in `forest_dataset`
        #' and as many columns as dimensions in the leaves of trees in `ForestContainer`
        predict_raw_single_tree = function(
            forest_dataset,
            forest_num,
            tree_num
        ) {
            stopifnot(!is.null(forest_dataset$data_ptr))

            # Predict leaf values from forest
            output <- predict_forest_raw_single_tree_cpp(
                self$forest_container_ptr,
                forest_dataset$data_ptr,
                forest_num,
                tree_num
            )
            return(output)
        },

        #' @description
        #' Set a constant predicted value for every tree in the ensemble.
        #' Stops program if any tree is more than a root node.
        #' @param forest_num Index of the forest sample within the container.
        #' @param leaf_value Constant leaf value(s) to be fixed for each tree in the ensemble indexed by `forest_num`. Can be either a single number or a vector, depending on the forest's leaf dimension.
        set_root_leaves = function(forest_num, leaf_value) {
            stopifnot(!is.null(self$forest_container_ptr))
            stopifnot(
                num_samples_forest_container_cpp(self$forest_container_ptr) == 0
            )

            # Set leaf values
            if (length(leaf_value) == 1) {
                stopifnot(
                    leaf_dimension_forest_container_cpp(
                        self$forest_container_ptr
                    ) ==
                        1
                )
                set_leaf_value_forest_container_cpp(
                    self$forest_container_ptr,
                    leaf_value
                )
            } else if (length(leaf_value) > 1) {
                stopifnot(
                    leaf_dimension_forest_container_cpp(
                        self$forest_container_ptr
                    ) ==
                        length(leaf_value)
                )
                set_leaf_vector_forest_container_cpp(
                    self$forest_container_ptr,
                    leaf_value
                )
            } else {
                stop(
                    "leaf_value must be a numeric value or vector of length >= 1"
                )
            }
        },

        #' @description
        #' Set a constant predicted value for every tree in the ensemble.
        #' Stops program if any tree is more than a root node.
        #' @param dataset `ForestDataset` Dataset class (covariates, basis, etc...)
        #' @param outcome `Outcome` Outcome class (residual / partial residual)
        #' @param forest_model `ForestModel` object storing tracking structures used in training / sampling
        #' @param leaf_model_int Integer value encoding the leaf model type (0 = constant gaussian, 1 = univariate gaussian, 2 = multivariate gaussian, 3 = log linear variance).
        #' @param leaf_value Constant leaf value(s) to be fixed for each tree in the ensemble indexed by `forest_num`. Can be either a single number or a vector, depending on the forest's leaf dimension.
        prepare_for_sampler = function(
            dataset,
            outcome,
            forest_model,
            leaf_model_int,
            leaf_value
        ) {
            stopifnot(!is.null(dataset$data_ptr))
            stopifnot(!is.null(outcome$data_ptr))
            stopifnot(!is.null(forest_model$tracker_ptr))
            stopifnot(!is.null(self$forest_container_ptr))
            stopifnot(
                num_samples_forest_container_cpp(self$forest_container_ptr) == 0
            )

            # Initialize the model
            initialize_forest_model_cpp(
                dataset$data_ptr,
                outcome$data_ptr,
                self$forest_container_ptr,
                forest_model$tracker_ptr,
                leaf_value,
                leaf_model_int
            )
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
        adjust_residual = function(
            dataset,
            outcome,
            forest_model,
            requires_basis,
            forest_num,
            add
        ) {
            stopifnot(!is.null(dataset$data_ptr))
            stopifnot(!is.null(outcome$data_ptr))
            stopifnot(!is.null(forest_model$tracker_ptr))
            stopifnot(!is.null(self$forest_container_ptr))

            adjust_residual_forest_container_cpp(
                dataset$data_ptr,
                outcome$data_ptr,
                self$forest_container_ptr,
                forest_model$tracker_ptr,
                requires_basis,
                forest_num,
                add
            )
        },

        #' @description
        #' Store the trees and metadata of `ForestDataset` class in a json file
        #' @param json_filename Name of output json file (must end in ".json")
        save_json = function(json_filename) {
            invisible(json_save_forest_container_cpp(
                self$forest_container_ptr,
                json_filename
            ))
        },

        #' @description
        #' Load trees and metadata for an ensemble from a json file. Note that
        #' any trees and metadata already present in `ForestDataset` class will
        #' be overwritten.
        #' @param json_filename Name of model input json file (must end in ".json")
        load_json = function(json_filename) {
            invisible(json_load_forest_container_cpp(
                self$forest_container_ptr,
                json_filename
            ))
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
        leaf_dimension = function() {
            return(leaf_dimension_forest_container_cpp(
                self$forest_container_ptr
            ))
        },

        #' @description
        #' Return constant leaf status of trees in a `ForestContainer` object
        #' @return `TRUE` if leaves are constant, `FALSE` otherwise
        is_constant_leaf = function() {
            return(is_constant_leaf_forest_container_cpp(
                self$forest_container_ptr
            ))
        },

        #' @description
        #' Return exponentiation status of trees in a `ForestContainer` object
        #' @return `TRUE` if leaf predictions must be exponentiated, `FALSE` otherwise
        is_exponentiated = function() {
            return(is_exponentiated_forest_container_cpp(
                self$forest_container_ptr
            ))
        },

        #' @description
        #' Add a new all-root ensemble to the container, with all of the leaves
        #' set to the value / vector provided
        #' @param leaf_value Value (or vector of values) to initialize root nodes in tree
        add_forest_with_constant_leaves = function(leaf_value) {
            if (length(leaf_value) > 1) {
                add_sample_vector_forest_container_cpp(
                    self$forest_container_ptr,
                    leaf_value
                )
            } else {
                add_sample_value_forest_container_cpp(
                    self$forest_container_ptr,
                    leaf_value
                )
            }
        },

        #' @description
        #' Add a numeric (i.e. `X[,i] <= c`) split to a given tree in the ensemble
        #' @param forest_num Index of the forest which contains the tree to be split
        #' @param tree_num Index of the tree to be split
        #' @param leaf_num Leaf to be split
        #' @param feature_num Feature that defines the new split
        #' @param split_threshold Value that defines the cutoff of the new split
        #' @param left_leaf_value Value (or vector of values) to assign to the newly created left node
        #' @param right_leaf_value Value (or vector of values) to assign to the newly created right node
        add_numeric_split_tree = function(
            forest_num,
            tree_num,
            leaf_num,
            feature_num,
            split_threshold,
            left_leaf_value,
            right_leaf_value
        ) {
            if (length(left_leaf_value) > 1) {
                add_numeric_split_tree_vector_forest_container_cpp(
                    self$forest_container_ptr,
                    forest_num,
                    tree_num,
                    leaf_num,
                    feature_num,
                    split_threshold,
                    left_leaf_value,
                    right_leaf_value
                )
            } else {
                add_numeric_split_tree_value_forest_container_cpp(
                    self$forest_container_ptr,
                    forest_num,
                    tree_num,
                    leaf_num,
                    feature_num,
                    split_threshold,
                    left_leaf_value,
                    right_leaf_value
                )
            }
        },

        #' @description
        #' Retrieve a vector of indices of leaf nodes for a given tree in a given forest
        #' @param forest_num Index of the forest which contains tree `tree_num`
        #' @param tree_num Index of the tree for which leaf indices will be retrieved
        get_tree_leaves = function(forest_num, tree_num) {
            return(get_tree_leaves_forest_container_cpp(
                self$forest_container_ptr,
                forest_num,
                tree_num
            ))
        },

        #' @description
        #' Retrieve a vector of split counts for every training set variable in a given tree in a given forest
        #' @param forest_num Index of the forest which contains tree `tree_num`
        #' @param tree_num Index of the tree for which split counts will be retrieved
        #' @param num_features Total number of features in the training set
        get_tree_split_counts = function(forest_num, tree_num, num_features) {
            return(get_tree_split_counts_forest_container_cpp(
                self$forest_container_ptr,
                forest_num,
                tree_num,
                num_features
            ))
        },

        #' @description
        #' Retrieve a vector of split counts for every training set variable in a given forest
        #' @param forest_num Index of the forest for which split counts will be retrieved
        #' @param num_features Total number of features in the training set
        get_forest_split_counts = function(forest_num, num_features) {
            return(get_forest_split_counts_forest_container_cpp(
                self$forest_container_ptr,
                forest_num,
                num_features
            ))
        },

        #' @description
        #' Retrieve a vector of split counts for every training set variable in a given forest, aggregated across ensembles and trees
        #' @param num_features Total number of features in the training set
        get_aggregate_split_counts = function(num_features) {
            return(get_overall_split_counts_forest_container_cpp(
                self$forest_container_ptr,
                num_features
            ))
        },

        #' @description
        #' Retrieve a vector of split counts for every training set variable in a given forest, reported separately for each ensemble and tree
        #' @param num_features Total number of features in the training set
        get_granular_split_counts = function(num_features) {
            n_samples <- self$num_samples()
            n_trees <- self$num_trees()
            output <- get_granular_split_count_array_forest_container_cpp(
                self$forest_container_ptr,
                num_features
            )
            dim(output) <- c(n_samples, n_trees, num_features)
            return(output)
        },

        #' @description
        #' Maximum depth of a specific tree in a specific ensemble in a `ForestSamples` object
        #' @param ensemble_num Ensemble number
        #' @param tree_num Tree index within ensemble `ensemble_num`
        #' @return Maximum leaf depth
        ensemble_tree_max_depth = function(ensemble_num, tree_num) {
            return(ensemble_tree_max_depth_forest_container_cpp(
                self$forest_container_ptr,
                ensemble_num,
                tree_num
            ))
        },

        #' @description
        #' Average the maximum depth of each tree in a given ensemble in a `ForestSamples` object
        #' @param ensemble_num Ensemble number
        #' @return Average maximum depth
        average_ensemble_max_depth = function(ensemble_num) {
            return(ensemble_average_max_depth_forest_container_cpp(
                self$forest_container_ptr,
                ensemble_num
            ))
        },

        #' @description
        #' Average the maximum depth of each tree in each ensemble in a `ForestContainer` object
        #' @return Average maximum depth
        average_max_depth = function() {
            return(average_max_depth_forest_container_cpp(
                self$forest_container_ptr
            ))
        },

        #' @description
        #' Number of leaves in a given ensemble in a `ForestSamples` object
        #' @param forest_num Index of the ensemble to be queried
        #' @return Count of leaves in the ensemble stored at `forest_num`
        num_forest_leaves = function(forest_num) {
            return(num_leaves_ensemble_forest_container_cpp(
                self$forest_container_ptr,
                forest_num
            ))
        },

        #' @description
        #' Sum of squared (raw) leaf values in a given ensemble in a `ForestSamples` object
        #' @param forest_num Index of the ensemble to be queried
        #' @return Average maximum depth
        sum_leaves_squared = function(forest_num) {
            return(sum_leaves_squared_ensemble_forest_container_cpp(
                self$forest_container_ptr,
                forest_num
            ))
        },

        #' @description
        #' Whether or not a given node of a given tree in a given forest in the `ForestSamples` is a leaf
        #' @param forest_num Index of the forest to be queried
        #' @param tree_num Index of the tree to be queried
        #' @param node_id Index of the node to be queried
        #' @return `TRUE` if node is a leaf, `FALSE` otherwise
        is_leaf_node = function(forest_num, tree_num, node_id) {
            return(is_leaf_node_forest_container_cpp(
                self$forest_container_ptr,
                forest_num,
                tree_num,
                node_id
            ))
        },

        #' @description
        #' Whether or not a given node of a given tree in a given forest in the `ForestSamples` is a numeric split node
        #' @param forest_num Index of the forest to be queried
        #' @param tree_num Index of the tree to be queried
        #' @param node_id Index of the node to be queried
        #' @return `TRUE` if node is a numeric split node, `FALSE` otherwise
        is_numeric_split_node = function(forest_num, tree_num, node_id) {
            return(is_numeric_split_node_forest_container_cpp(
                self$forest_container_ptr,
                forest_num,
                tree_num,
                node_id
            ))
        },

        #' @description
        #' Whether or not a given node of a given tree in a given forest in the `ForestSamples` is a categorical split node
        #' @param forest_num Index of the forest to be queried
        #' @param tree_num Index of the tree to be queried
        #' @param node_id Index of the node to be queried
        #' @return `TRUE` if node is a categorical split node, `FALSE` otherwise
        is_categorical_split_node = function(forest_num, tree_num, node_id) {
            return(is_categorical_split_node_forest_container_cpp(
                self$forest_container_ptr,
                forest_num,
                tree_num,
                node_id
            ))
        },

        #' @description
        #' Parent node of given node of a given tree in a given forest in a `ForestSamples` object
        #' @param forest_num Index of the forest to be queried
        #' @param tree_num Index of the tree to be queried
        #' @param node_id Index of the node to be queried
        #' @return Integer ID of the parent node
        parent_node = function(forest_num, tree_num, node_id) {
            return(parent_node_forest_container_cpp(
                self$forest_container_ptr,
                forest_num,
                tree_num,
                node_id
            ))
        },

        #' @description
        #' Left child node of given node of a given tree in a given forest in a `ForestSamples` object
        #' @param forest_num Index of the forest to be queried
        #' @param tree_num Index of the tree to be queried
        #' @param node_id Index of the node to be queried
        #' @return Integer ID of the left child node
        left_child_node = function(forest_num, tree_num, node_id) {
            return(left_child_node_forest_container_cpp(
                self$forest_container_ptr,
                forest_num,
                tree_num,
                node_id
            ))
        },

        #' @description
        #' Right child node of given node of a given tree in a given forest in a `ForestSamples` object
        #' @param forest_num Index of the forest to be queried
        #' @param tree_num Index of the tree to be queried
        #' @param node_id Index of the node to be queried
        #' @return Integer ID of the right child node
        right_child_node = function(forest_num, tree_num, node_id) {
            return(right_child_node_forest_container_cpp(
                self$forest_container_ptr,
                forest_num,
                tree_num,
                node_id
            ))
        },

        #' @description
        #' Depth of given node of a given tree in a given forest in a `ForestSamples` object, with 0 depth for the root node.
        #' @param forest_num Index of the forest to be queried
        #' @param tree_num Index of the tree to be queried
        #' @param node_id Index of the node to be queried
        #' @return Integer valued depth of the node
        node_depth = function(forest_num, tree_num, node_id) {
            return(node_depth_forest_container_cpp(
                self$forest_container_ptr,
                forest_num,
                tree_num,
                node_id
            ))
        },

        #' @description
        #' Split index of given node of a given tree in a given forest in a `ForestSamples` object. Returns `-1` is node is a leaf.
        #' @param forest_num Index of the forest to be queried
        #' @param tree_num Index of the tree to be queried
        #' @param node_id Index of the node to be queried
        #' @return Integer valued depth of the node
        node_split_index = function(forest_num, tree_num, node_id) {
            if (self$is_leaf_node(forest_num, tree_num, node_id)) {
                return(-1)
            } else {
                return(split_index_forest_container_cpp(
                    self$forest_container_ptr,
                    forest_num,
                    tree_num,
                    node_id
                ))
            }
        },

        #' @description
        #' Threshold that defines a numeric split for a given node of a given tree in a given forest in a `ForestSamples` object.
        #' Returns `Inf` if the node is a leaf or a categorical split node.
        #' @param forest_num Index of the forest to be queried
        #' @param tree_num Index of the tree to be queried
        #' @param node_id Index of the node to be queried
        #' @return Threshold defining a split for the node
        node_split_threshold = function(forest_num, tree_num, node_id) {
            if (
                self$is_leaf_node(forest_num, tree_num, node_id) ||
                    self$is_categorical_split_node(
                        forest_num,
                        tree_num,
                        node_id
                    )
            ) {
                return(Inf)
            } else {
                return(split_theshold_forest_container_cpp(
                    self$forest_container_ptr,
                    forest_num,
                    tree_num,
                    node_id
                ))
            }
        },

        #' @description
        #' Array of category indices that define a categorical split for a given node of a given tree in a given forest in a `ForestSamples` object.
        #' Returns `c(Inf)` if the node is a leaf or a numeric split node.
        #' @param forest_num Index of the forest to be queried
        #' @param tree_num Index of the tree to be queried
        #' @param node_id Index of the node to be queried
        #' @return Categories defining a split for the node
        node_split_categories = function(forest_num, tree_num, node_id) {
            if (
                self$is_leaf_node(forest_num, tree_num, node_id) ||
                    self$is_numeric_split_node(forest_num, tree_num, node_id)
            ) {
                return(c(Inf))
            } else {
                return(split_categories_forest_container_cpp(
                    self$forest_container_ptr,
                    forest_num,
                    tree_num,
                    node_id
                ))
            }
        },

        #' @description
        #' Leaf node value(s) for a given node of a given tree in a given forest in a `ForestSamples` object.
        #' Values are stale if the node is a split node.
        #' @param forest_num Index of the forest to be queried
        #' @param tree_num Index of the tree to be queried
        #' @param node_id Index of the node to be queried
        #' @return Vector (often univariate) of leaf values
        node_leaf_values = function(forest_num, tree_num, node_id) {
            return(leaf_values_forest_container_cpp(
                self$forest_container_ptr,
                forest_num,
                tree_num,
                node_id
            ))
        },

        #' @description
        #' Number of nodes in a given tree in a given forest in a `ForestSamples` object.
        #' @param forest_num Index of the forest to be queried
        #' @param tree_num Index of the tree to be queried
        #' @return Count of total tree nodes
        num_nodes = function(forest_num, tree_num) {
            return(num_nodes_forest_container_cpp(
                self$forest_container_ptr,
                forest_num,
                tree_num
            ))
        },

        #' @description
        #' Number of leaves in a given tree in a given forest in a `ForestSamples` object.
        #' @param forest_num Index of the forest to be queried
        #' @param tree_num Index of the tree to be queried
        #' @return Count of total tree leaves
        num_leaves = function(forest_num, tree_num) {
            return(num_leaves_forest_container_cpp(
                self$forest_container_ptr,
                forest_num,
                tree_num
            ))
        },

        #' @description
        #' Number of leaf parents (split nodes with two leaves as children) in a given tree in a given forest in a `ForestSamples` object.
        #' @param forest_num Index of the forest to be queried
        #' @param tree_num Index of the tree to be queried
        #' @return Count of total tree leaf parents
        num_leaf_parents = function(forest_num, tree_num) {
            return(num_leaf_parents_forest_container_cpp(
                self$forest_container_ptr,
                forest_num,
                tree_num
            ))
        },

        #' @description
        #' Number of split nodes in a given tree in a given forest in a `ForestSamples` object.
        #' @param forest_num Index of the forest to be queried
        #' @param tree_num Index of the tree to be queried
        #' @return Count of total tree split nodes
        num_split_nodes = function(forest_num, tree_num) {
            return(num_split_nodes_forest_container_cpp(
                self$forest_container_ptr,
                forest_num,
                tree_num
            ))
        },

        #' @description
        #' Array of node indices in a given tree in a given forest in a `ForestSamples` object.
        #' @param forest_num Index of the forest to be queried
        #' @param tree_num Index of the tree to be queried
        #' @return Indices of tree nodes
        nodes = function(forest_num, tree_num) {
            return(nodes_forest_container_cpp(
                self$forest_container_ptr,
                forest_num,
                tree_num
            ))
        },

        #' @description
        #' Array of leaf indices in a given tree in a given forest in a `ForestSamples` object.
        #' @param forest_num Index of the forest to be queried
        #' @param tree_num Index of the tree to be queried
        #' @return Indices of leaf nodes
        leaves = function(forest_num, tree_num) {
            return(leaves_forest_container_cpp(
                self$forest_container_ptr,
                forest_num,
                tree_num
            ))
        },

        #' @description
        #' Modify the ``ForestSamples`` object by removing the forest sample indexed by `forest_num
        #' @param forest_num Index of the forest to be removed
        delete_sample = function(forest_num) {
            return(remove_sample_forest_container_cpp(
                self$forest_container_ptr,
                forest_num
            ))
        }
    )
)

#' Class that stores a single ensemble of decision trees (often treated as the "active forest")
#'
#' @description
#' Wrapper around a C++ tree ensemble

Forest <- R6::R6Class(
    classname = "Forest",
    cloneable = FALSE,
    public = list(
        #' @field forest_ptr External pointer to a C++ TreeEnsemble class
        forest_ptr = NULL,

        #' @field internal_forest_is_empty Whether the forest has not yet been "initialized" such that its `predict` function can be called.
        internal_forest_is_empty = TRUE,

        #' @description
        #' Create a new Forest object.
        #' @param num_trees Number of trees in the forest
        #' @param leaf_dimension Dimensionality of the outcome model
        #' @param is_leaf_constant Whether leaf is constant
        #' @param is_exponentiated Whether forest predictions should be exponentiated before being returned
        #' @return A new `Forest` object.
        initialize = function(
            num_trees,
            leaf_dimension = 1,
            is_leaf_constant = FALSE,
            is_exponentiated = FALSE
        ) {
            self$forest_ptr <- active_forest_cpp(
                num_trees,
                leaf_dimension,
                is_leaf_constant,
                is_exponentiated
            )
            self$internal_forest_is_empty <- TRUE
        },

        #' @description
        #' Create a larger forest by merging the trees of this forest with those of another forest
        #' @param forest Forest to be merged into this forest
        merge_forest = function(forest) {
            stopifnot(self$leaf_dimension() == forest$leaf_dimension())
            stopifnot(self$is_constant_leaf() == forest$is_constant_leaf())
            stopifnot(self$is_exponentiated() == forest$is_exponentiated())
            forest_merge_cpp(self$forest_ptr, forest$forest_ptr)
        },

        #' @description
        #' Add a constant value to every leaf of every tree in an ensemble. If leaves are multi-dimensional, `constant_value` will be added to every dimension of the leaves.
        #' @param constant_value Value that will be added to every leaf of every tree
        add_constant = function(constant_value) {
            forest_add_constant_cpp(self$forest_ptr, constant_value)
        },

        #' @description
        #' Multiply every leaf of every tree by a constant value. If leaves are multi-dimensional, `constant_multiple` will be multiplied through every dimension of the leaves.
        #' @param constant_multiple Value that will be multiplied by every leaf of every tree
        multiply_constant = function(constant_multiple) {
            forest_multiply_constant_cpp(self$forest_ptr, constant_multiple)
        },

        #' @description
        #' Predict forest on every sample in `forest_dataset`
        #' @param forest_dataset `ForestDataset` R class
        #' @return vector of predictions with as many rows as in `forest_dataset`
        predict = function(forest_dataset) {
            stopifnot(!is.null(forest_dataset$data_ptr))
            stopifnot(!is.null(self$forest_ptr))
            return(predict_active_forest_cpp(
                self$forest_ptr,
                forest_dataset$data_ptr
            ))
        },

        #' @description
        #' Predict "raw" leaf values (without being multiplied by basis) for every sample in `forest_dataset`
        #' @param forest_dataset `ForestDataset` R class
        #' @return Array of predictions for each observation in `forest_dataset` and
        #' each sample in the `ForestSamples` class with each prediction having the
        #' dimensionality of the forests' leaf model. In the case of a constant leaf model
        #' or univariate leaf regression, this array is a vector (length is the number of
        #' observations). In the case of a multivariate leaf regression,
        #' this array is a matrix (number of observations by leaf model dimension,
        #' number of samples).
        predict_raw = function(forest_dataset) {
            stopifnot(!is.null(forest_dataset$data_ptr))
            # Unpack dimensions
            output_dim <- leaf_dimension_active_forest_cpp(self$forest_ptr)
            n <- dataset_num_rows_cpp(forest_dataset$data_ptr)

            # Predict leaf values from forest
            predictions <- predict_raw_active_forest_cpp(
                self$forest_ptr,
                forest_dataset$data_ptr
            )
            if (output_dim > 1) {
                dim(predictions) <- c(n, output_dim)
            }

            return(predictions)
        },

        #' @description
        #' Set a constant predicted value for every tree in the ensemble.
        #' Stops program if any tree is more than a root node.
        #' @param leaf_value Constant leaf value(s) to be fixed for each tree in the ensemble indexed by `forest_num`. Can be either a single number or a vector, depending on the forest's leaf dimension.
        set_root_leaves = function(leaf_value) {
            stopifnot(!is.null(self$forest_ptr))
            stopifnot(self$internal_forest_is_empty)

            # Set leaf values
            if (length(leaf_value) == 1) {
                stopifnot(
                    leaf_dimension_active_forest_cpp(self$forest_ptr) == 1
                )
                set_leaf_value_active_forest_cpp(self$forest_ptr, leaf_value)
            } else if (length(leaf_value) > 1) {
                stopifnot(
                    leaf_dimension_active_forest_cpp(self$forest_ptr) ==
                        length(leaf_value)
                )
                set_leaf_vector_active_forest_cpp(self$forest_ptr, leaf_value)
            } else {
                stop(
                    "leaf_value must be a numeric value or vector of length >= 1"
                )
            }

            self$internal_forest_is_empty = FALSE
        },

        #' @description
        #' Set a constant predicted value for every tree in the ensemble.
        #' Stops program if any tree is more than a root node.
        #' @param dataset `ForestDataset` Dataset class (covariates, basis, etc...)
        #' @param outcome `Outcome` Outcome class (residual / partial residual)
        #' @param forest_model `ForestModel` object storing tracking structures used in training / sampling
        #' @param leaf_model_int Integer value encoding the leaf model type (0 = constant gaussian, 1 = univariate gaussian, 2 = multivariate gaussian, 3 = log linear variance).
        #' @param leaf_value Constant leaf value(s) to be fixed for each tree in the ensemble indexed by `forest_num`. Can be either a single number or a vector, depending on the forest's leaf dimension.
        prepare_for_sampler = function(
            dataset,
            outcome,
            forest_model,
            leaf_model_int,
            leaf_value
        ) {
            stopifnot(!is.null(dataset$data_ptr))
            stopifnot(!is.null(outcome$data_ptr))
            stopifnot(!is.null(forest_model$tracker_ptr))
            stopifnot(!is.null(self$forest_ptr))
            stopifnot(self$internal_forest_is_empty)

            # Initialize the model
            initialize_forest_model_active_forest_cpp(
                dataset$data_ptr,
                outcome$data_ptr,
                self$forest_ptr,
                forest_model$tracker_ptr,
                leaf_value,
                leaf_model_int
            )

            self$internal_forest_is_empty = FALSE
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
        #' @param add Whether forest predictions should be added to or subtracted from residuals
        adjust_residual = function(
            dataset,
            outcome,
            forest_model,
            requires_basis,
            add
        ) {
            stopifnot(!is.null(dataset$data_ptr))
            stopifnot(!is.null(outcome$data_ptr))
            stopifnot(!is.null(forest_model$tracker_ptr))
            stopifnot(!is.null(self$forest_ptr))

            adjust_residual_active_forest_cpp(
                dataset$data_ptr,
                outcome$data_ptr,
                self$forest_ptr,
                forest_model$tracker_ptr,
                requires_basis,
                add
            )
        },

        #' @description
        #' Return number of trees in each ensemble of a `Forest` object
        #' @return Tree count
        num_trees = function() {
            return(num_trees_active_forest_cpp(self$forest_ptr))
        },

        #' @description
        #' Return output dimension of trees in a `Forest` object
        #' @return Leaf node parameter size
        leaf_dimension = function() {
            return(leaf_dimension_active_forest_cpp(self$forest_ptr))
        },

        #' @description
        #' Return constant leaf status of trees in a `Forest` object
        #' @return `TRUE` if leaves are constant, `FALSE` otherwise
        is_constant_leaf = function() {
            return(is_leaf_constant_active_forest_cpp(self$forest_ptr))
        },

        #' @description
        #' Return exponentiation status of trees in a `Forest` object
        #' @return `TRUE` if leaf predictions must be exponentiated, `FALSE` otherwise
        is_exponentiated = function() {
            return(is_exponentiated_active_forest_cpp(self$forest_ptr))
        },

        #' @description
        #' Add a numeric (i.e. `X[,i] <= c`) split to a given tree in the ensemble
        #' @param tree_num Index of the tree to be split
        #' @param leaf_num Leaf to be split
        #' @param feature_num Feature that defines the new split
        #' @param split_threshold Value that defines the cutoff of the new split
        #' @param left_leaf_value Value (or vector of values) to assign to the newly created left node
        #' @param right_leaf_value Value (or vector of values) to assign to the newly created right node
        add_numeric_split_tree = function(
            tree_num,
            leaf_num,
            feature_num,
            split_threshold,
            left_leaf_value,
            right_leaf_value
        ) {
            if (length(left_leaf_value) > 1) {
                add_numeric_split_tree_vector_active_forest_cpp(
                    self$forest_ptr,
                    tree_num,
                    leaf_num,
                    feature_num,
                    split_threshold,
                    left_leaf_value,
                    right_leaf_value
                )
            } else {
                add_numeric_split_tree_value_active_forest_cpp(
                    self$forest_ptr,
                    tree_num,
                    leaf_num,
                    feature_num,
                    split_threshold,
                    left_leaf_value,
                    right_leaf_value
                )
            }
        },

        #' @description
        #' Retrieve a vector of indices of leaf nodes for a given tree in a given forest
        #' @param tree_num Index of the tree for which leaf indices will be retrieved
        get_tree_leaves = function(tree_num) {
            return(get_tree_leaves_active_forest_cpp(self$forest_ptr, tree_num))
        },

        #' @description
        #' Retrieve a vector of split counts for every training set variable in a given tree in the forest
        #' @param tree_num Index of the tree for which split counts will be retrieved
        #' @param num_features Total number of features in the training set
        get_tree_split_counts = function(tree_num, num_features) {
            return(get_tree_split_counts_active_forest_cpp(
                self$forest_ptr,
                tree_num,
                num_features
            ))
        },

        #' @description
        #' Retrieve a vector of split counts for every training set variable in the forest
        #' @param num_features Total number of features in the training set
        get_forest_split_counts = function(num_features) {
            return(get_overall_split_counts_active_forest_cpp(
                self$forest_ptr,
                num_features
            ))
        },

        #' @description
        #' Maximum depth of a specific tree in the forest
        #' @param tree_num Tree index within forest
        #' @return Maximum leaf depth
        tree_max_depth = function(tree_num) {
            return(ensemble_tree_max_depth_active_forest_cpp(
                self$forest_ptr,
                tree_num
            ))
        },

        #' @description
        #' Average the maximum depth of each tree in the forest
        #' @return Average maximum depth
        average_max_depth = function() {
            return(ensemble_average_max_depth_active_forest_cpp(
                self$forest_ptr
            ))
        },

        #' @description
        #' When a forest object is created, it is "empty" in the sense that none
        #' of its component trees have leaves with values. There are two ways to
        #' "initialize" a Forest object. First, the `set_root_leaves()` method
        #' simply initializes every tree in the forest to a single node carrying
        #' the same (user-specified) leaf value. Second, the `prepare_for_sampler()`
        #' method initializes every tree in the forest to a single node with the
        #' same value and also propagates this information through to a ForestModel
        #' object, which must be synchronized with a Forest during a forest
        #' sampler loop.
        #' @return `TRUE` if a Forest has not yet been initialized with a constant
        #' root value, `FALSE` otherwise if the forest has already been
        #' initialized / grown.
        is_empty = function() {
            return(self$internal_forest_is_empty)
        }
    )
)

#' Create a container of forest samples
#'
#' @param num_trees Number of trees
#' @param leaf_dimension Dimensionality of the outcome model
#' @param is_leaf_constant Whether leaf is constant
#' @param is_exponentiated Whether forest predictions should be exponentiated before being returned
#'
#' @return `ForestSamples` object
#' @export
#'
#' @examples
#' num_trees <- 100
#' leaf_dimension <- 2
#' is_leaf_constant <- FALSE
#' is_exponentiated <- FALSE
#' forest_samples <- createForestSamples(num_trees, leaf_dimension, is_leaf_constant, is_exponentiated)
createForestSamples <- function(
    num_trees,
    leaf_dimension = 1,
    is_leaf_constant = FALSE,
    is_exponentiated = FALSE
) {
    return(invisible(
        (ForestSamples$new(
            num_trees,
            leaf_dimension,
            is_leaf_constant,
            is_exponentiated
        ))
    ))
}

#' Create a forest
#'
#' @param num_trees Number of trees in the forest
#' @param leaf_dimension Dimensionality of the outcome model
#' @param is_leaf_constant Whether leaf is constant
#' @param is_exponentiated Whether forest predictions should be exponentiated before being returned
#'
#' @return `Forest` object
#' @export
#'
#' @examples
#' num_trees <- 100
#' leaf_dimension <- 2
#' is_leaf_constant <- FALSE
#' is_exponentiated <- FALSE
#' forest <- createForest(num_trees, leaf_dimension, is_leaf_constant, is_exponentiated)
createForest <- function(
    num_trees,
    leaf_dimension = 1,
    is_leaf_constant = FALSE,
    is_exponentiated = FALSE
) {
    return(invisible(
        (Forest$new(
            num_trees,
            leaf_dimension,
            is_leaf_constant,
            is_exponentiated
        ))
    ))
}

#' Reset an active forest, either from a specific forest in a `ForestContainer`
#' or to an ensemble of single-node (i.e. root) trees
#'
#' @param active_forest Current active forest
#' @param forest_samples (Optional) Container of forest samples from which to re-initialize active forest. If not provided, active forest will be reset to an ensemble of single-node (i.e. root) trees.
#' @param forest_num (Optional) Index of forest samples from which to initialize active forest. If not provided, active forest will be reset to an ensemble of single-node (i.e. root) trees.
#' @return None
#' @export
#'
#' @examples
#' num_trees <- 100
#' leaf_dimension <- 1
#' is_leaf_constant <- TRUE
#' is_exponentiated <- FALSE
#' active_forest <- createForest(num_trees, leaf_dimension, is_leaf_constant, is_exponentiated)
#' forest_samples <- createForestSamples(num_trees, leaf_dimension, is_leaf_constant, is_exponentiated)
#' forest_samples$add_forest_with_constant_leaves(0.0)
#' forest_samples$add_numeric_split_tree(0, 0, 0, 0, 0.5, -1.0, 1.0)
#' forest_samples$add_numeric_split_tree(0, 1, 0, 1, 0.75, 3.4, 0.75)
#' active_forest$set_root_leaves(0.1)
#' resetActiveForest(active_forest, forest_samples, 0)
#' resetActiveForest(active_forest)
resetActiveForest <- function(
    active_forest,
    forest_samples = NULL,
    forest_num = NULL
) {
    if (is.null(forest_samples)) {
        root_reset_active_forest_cpp(active_forest$forest_ptr)
        active_forest$internal_forest_is_empty = TRUE
    } else {
        if (is.null(forest_num)) {
            stop(
                "`forest_num` must be specified if `forest_samples` is provided"
            )
        }
        reset_active_forest_cpp(
            active_forest$forest_ptr,
            forest_samples$forest_container_ptr,
            forest_num
        )
    }
}

#' Re-initialize a forest model (tracking data structures) from a specific forest in a `ForestContainer`
#'
#' @param forest_model Forest model with tracking data structures
#' @param forest Forest from which to re-initialize forest model
#' @param dataset Training dataset object
#' @param residual Residual which will also be updated
#' @param is_mean_model Whether the model being updated is a conditional mean model
#' @return None
#' @export
#'
#' @examples
#' n <- 100
#' p <- 10
#' num_trees <- 100
#' leaf_dimension <- 1
#' is_leaf_constant <- TRUE
#' is_exponentiated <- FALSE
#' alpha <- 0.95
#' beta <- 2.0
#' min_samples_leaf <- 2
#' max_depth <- 10
#' feature_types <- as.integer(rep(0, p))
#' leaf_model <- 0
#' sigma2 <- 1.0
#' leaf_scale <- as.matrix(1.0)
#' variable_weights <- rep(1/p, p)
#' a_forest <- 1
#' b_forest <- 1
#' cutpoint_grid_size <- 100
#' X <- matrix(runif(n*p), ncol = p)
#' forest_dataset <- createForestDataset(X)
#' y <- -5 + 10*(X[,1] > 0.5) + rnorm(n)
#' outcome <- createOutcome(y)
#' rng <- createCppRNG(1234)
#' global_model_config <- createGlobalModelConfig(global_error_variance=sigma2)
#' forest_model_config <- createForestModelConfig(feature_types=feature_types,
#'                                                num_trees=num_trees, num_observations=n,
#'                                                num_features=p, alpha=alpha, beta=beta,
#'                                                min_samples_leaf=min_samples_leaf,
#'                                                max_depth=max_depth,
#'                                                variable_weights=variable_weights,
#'                                                cutpoint_grid_size=cutpoint_grid_size,
#'                                                leaf_model_type=leaf_model,
#'                                                leaf_model_scale=leaf_scale)
#' forest_model <- createForestModel(forest_dataset, forest_model_config, global_model_config)
#' active_forest <- createForest(num_trees, leaf_dimension, is_leaf_constant, is_exponentiated)
#' forest_samples <- createForestSamples(num_trees, leaf_dimension,
#'                                       is_leaf_constant, is_exponentiated)
#' active_forest$prepare_for_sampler(forest_dataset, outcome, forest_model, 0, 0.)
#' forest_model$sample_one_iteration(
#'     forest_dataset, outcome, forest_samples, active_forest,
#'     rng, forest_model_config, global_model_config,
#'     keep_forest = TRUE, gfr = FALSE
#' )
#' resetActiveForest(active_forest, forest_samples, 0)
#' resetForestModel(forest_model, active_forest, forest_dataset, outcome, TRUE)
resetForestModel <- function(
    forest_model,
    forest,
    dataset,
    residual,
    is_mean_model
) {
    reset_forest_model_cpp(
        forest_model$tracker_ptr,
        forest$forest_ptr,
        dataset$data_ptr,
        residual$data_ptr,
        is_mean_model
    )
}
