#' Preprocess a parameter list, overriding defaults with any provided parameters.
#'
#' @param default_params List of parameters with default values set.
#' @param user_params (Optional) User-supplied overrides to `default_params`.
#' @noRd
#'
#' @return Parameter list with defaults overriden by values supplied in `user_params`
preprocessParams <- function(default_params, user_params = NULL) {
    # Override defaults from general_params
    if (!is.null(user_params)) {
        for (key in names(user_params)) {
            if (key %in% names(default_params)) {
                val <- user_params[[key]]
                if (!is.null(val)) default_params[[key]] <- val
            }
        }
    }

    # Return result
    return(default_params)
}

#' Preprocess covariates. DataFrames will be preprocessed based on their column
#' types. Matrices will be passed through assuming all columns are numeric.
#'
#' @param input_data Covariates, provided as either a dataframe or a matrix
#'
#' @return List with preprocessed (unmodified) data and details on the number of each type
#' of variable, unique categories associated with categorical variables, and the
#' vector of feature types needed for calls to BART and BCF.
#' @export
#'
#' @examples
#' cov_mat <- matrix(1:12, ncol = 3)
#' preprocess_list <- preprocessTrainData(cov_mat)
#' X <- preprocess_list$X
preprocessTrainData <- function(input_data) {
    # Input checks
    if ((!is.matrix(input_data)) && (!is.data.frame(input_data))) {
        stop("Covariates provided must be a dataframe or matrix")
    }

    # Routing the correct preprocessing function
    if (is.matrix(input_data)) {
        output <- preprocessTrainMatrix(input_data)
    } else {
        output <- preprocessTrainDataFrame(input_data)
    }

    return(output)
}

#' Preprocess covariates. DataFrames will be preprocessed based on their column
#' types. Matrices will be passed through assuming all columns are numeric.
#'
#' @param input_data Covariates, provided as either a dataframe or a matrix
#' @param metadata List containing information on variables, including train set
#' categories for categorical variables
#'
#' @return Preprocessed data with categorical variables appropriately handled
#' @export
#'
#' @examples
#' cov_df <- data.frame(x1 = 1:5, x2 = 5:1, x3 = 6:10)
#' metadata <- list(num_ordered_cat_vars = 0, num_unordered_cat_vars = 0,
#'                  num_numeric_vars = 3, numeric_vars = c("x1", "x2", "x3"))
#' X_preprocessed <- preprocessPredictionData(cov_df, metadata)
preprocessPredictionData <- function(input_data, metadata) {
    # Input checks
    if ((!is.matrix(input_data)) && (!is.data.frame(input_data))) {
        stop("Covariates provided must be a dataframe or matrix")
    }

    # Routing the correct preprocessing function
    if (is.matrix(input_data)) {
        X <- preprocessPredictionMatrix(input_data, metadata)
    } else {
        X <- preprocessPredictionDataFrame(input_data, metadata)
    }

    return(X)
}

#' Preprocess a matrix of covariate values, assuming all columns are numeric.
#' Returns a list including a matrix of preprocessed covariate values and associated tracking.
#'
#' @param input_matrix Covariate matrix.
#'
#' @return List with preprocessed (unmodified) data and details on the number of each type
#' of variable, unique categories associated with categorical variables, and the
#' vector of feature types needed for calls to BART and BCF.
#' @noRd
#'
#' @examples
#' cov_mat <- matrix(1:12, ncol = 3)
#' preprocess_list <- preprocessTrainMatrix(cov_mat)
#' X <- preprocess_list$X
preprocessTrainMatrix <- function(input_matrix) {
    # Input checks
    if (!is.matrix(input_matrix)) {
        stop("covariates provided must be a matrix")
    }

    # Unpack metadata (assuming all variables are numeric)
    names(input_matrix) <- paste0("x", 1:ncol(input_matrix))
    df_vars <- names(input_matrix)
    num_ordered_cat_vars <- 0
    num_unordered_cat_vars <- 0
    num_numeric_vars <- ncol(input_matrix)
    numeric_vars <- names(input_matrix)
    feature_types <- rep(0, ncol(input_matrix))

    # Unpack data
    X <- input_matrix

    # Aggregate results into a list
    metadata <- list(
        feature_types = feature_types,
        num_ordered_cat_vars = num_ordered_cat_vars,
        num_unordered_cat_vars = num_unordered_cat_vars,
        num_numeric_vars = num_numeric_vars,
        numeric_vars = numeric_vars,
        original_var_indices = 1:num_numeric_vars
    )
    output <- list(
        data = X,
        metadata = metadata
    )

    return(output)
}

#' Preprocess a matrix of covariate values, assuming all columns are numeric.
#'
#' @param input_matrix Covariate matrix.
#' @param metadata List containing information on variables, including train set
#' categories for categorical variables
#'
#' @return Preprocessed data with categorical variables appropriately preprocessed
#' @noRd
#'
#' @examples
#' cov_mat <- matrix(c(1:5, 5:1, 6:10), ncol = 3)
#' metadata <- list(num_ordered_cat_vars = 0, num_unordered_cat_vars = 0,
#'                  num_numeric_vars = 3, numeric_vars = c("x1", "x2", "x3"))
#' X_preprocessed <- preprocessPredictionMatrix(cov_mat, metadata)
preprocessPredictionMatrix <- function(input_matrix, metadata) {
    # Input checks
    if (!is.matrix(input_matrix)) {
        stop("covariates provided must be a matrix")
    }
    if (!(ncol(input_matrix) == metadata$num_numeric_vars)) {
        stop(
            "Prediction set covariates have inconsistent dimension from train set covariates"
        )
    }

    return(input_matrix)
}

#' Preprocess a dataframe of covariate values, converting categorical variables
#' to integers and one-hot encoding if need be. Returns a list including a
#' matrix of preprocessed covariate values and associated tracking.
#'
#' @param input_df Dataframe of covariates. Users must pre-process any
#' categorical variables as factors (ordered for ordered categorical).
#' @noRd
#'
#' @return List with preprocessed data and details on the number of each type
#' of variable, unique categories associated with categorical variables, and the
#' vector of feature types needed for calls to BART and BCF.
preprocessTrainDataFrame <- function(input_df) {
    # Input checks / details
    if (!is.data.frame(input_df)) {
        stop("covariates provided must be a data frame")
    }
    df_vars <- names(input_df)

    # Detect ordered and unordered categorical variables

    # First, ordered categorical: users must have explicitly
    # converted this to a factor with ordered = TRUE
    factor_mask <- sapply(input_df, is.factor)
    ordered_mask <- sapply(input_df, is.ordered)
    ordered_cat_matches <- factor_mask & ordered_mask
    ordered_cat_vars <- df_vars[ordered_cat_matches]
    ordered_cat_var_inds <- unname(which(ordered_cat_matches))
    num_ordered_cat_vars <- length(ordered_cat_vars)
    if (num_ordered_cat_vars > 0) {
        ordered_cat_df <- input_df[, ordered_cat_vars, drop = FALSE]
    }

    # Next, unordered categorical: we will convert character
    # columns but not integer columns (users must explicitly
    # convert these to factor)
    character_mask <- sapply(input_df, is.character)
    unordered_cat_matches <- (factor_mask & (!ordered_mask)) | character_mask
    unordered_cat_vars <- df_vars[unordered_cat_matches]
    unordered_cat_var_inds <- unname(which(unordered_cat_matches))
    num_unordered_cat_vars <- length(unordered_cat_vars)
    if (num_unordered_cat_vars > 0) {
        unordered_cat_df <- input_df[, unordered_cat_vars, drop = FALSE]
    }

    # Numeric variables
    numeric_matches <- (!ordered_cat_matches) & (!unordered_cat_matches)
    numeric_vars <- df_vars[numeric_matches]
    numeric_var_inds <- unname(which(numeric_matches))
    num_numeric_vars <- length(numeric_vars)
    if (num_numeric_vars > 0) {
        numeric_df <- input_df[, numeric_vars, drop = FALSE]
    }

    # Empty outputs
    X <- double(0)
    unordered_unique_levels <- list()
    ordered_unique_levels <- list()
    feature_types <- integer(0)
    original_var_indices <- integer(0)

    # First, extract the numeric covariates
    if (num_numeric_vars > 0) {
        Xnum <- double(0)
        for (i in 1:ncol(numeric_df)) {
            stopifnot(is.numeric(numeric_df[, i]))
            Xnum <- cbind(Xnum, numeric_df[, i])
        }
        X <- cbind(X, unname(Xnum))
        feature_types <- c(feature_types, rep(0, ncol(Xnum)))
        original_var_indices <- c(original_var_indices, numeric_var_inds)
    }

    # Next, run some simple preprocessing on the ordered categorical covariates
    if (num_ordered_cat_vars > 0) {
        Xordcat <- double(0)
        for (i in 1:ncol(ordered_cat_df)) {
            var_name <- names(ordered_cat_df)[i]
            preprocess_list <- orderedCatInitializeAndPreprocess(ordered_cat_df[,
                i
            ])
            ordered_unique_levels[[var_name]] <- preprocess_list$unique_levels
            Xordcat <- cbind(Xordcat, preprocess_list$x_preprocessed)
        }
        X <- cbind(X, unname(Xordcat))
        feature_types <- c(feature_types, rep(1, ncol(Xordcat)))
        original_var_indices <- c(original_var_indices, ordered_cat_var_inds)
    }

    # Finally, one-hot encode the unordered categorical covariates
    if (num_unordered_cat_vars > 0) {
        one_hot_mats <- list()
        for (i in 1:ncol(unordered_cat_df)) {
            var_name <- names(unordered_cat_df)[i]
            encode_list <- oneHotInitializeAndEncode(unordered_cat_df[, i])
            unordered_unique_levels[[var_name]] <- encode_list$unique_levels
            one_hot_mats[[var_name]] <- encode_list$Xtilde
            one_hot_var <- rep(
                unordered_cat_var_inds[i],
                ncol(encode_list$Xtilde)
            )
            original_var_indices <- c(original_var_indices, one_hot_var)
        }
        Xcat <- do.call(cbind, one_hot_mats)
        X <- cbind(X, unname(Xcat))
        feature_types <- c(feature_types, rep(1, ncol(Xcat)))
    }

    # Aggregate results into a list
    metadata <- list(
        feature_types = feature_types,
        num_ordered_cat_vars = num_ordered_cat_vars,
        num_unordered_cat_vars = num_unordered_cat_vars,
        num_numeric_vars = num_numeric_vars,
        original_var_indices = original_var_indices
    )
    if (num_ordered_cat_vars > 0) {
        metadata[["ordered_cat_vars"]] = ordered_cat_vars
        metadata[["ordered_unique_levels"]] = ordered_unique_levels
    }
    if (num_unordered_cat_vars > 0) {
        metadata[["unordered_cat_vars"]] = unordered_cat_vars
        metadata[["unordered_unique_levels"]] = unordered_unique_levels
    }
    if (num_numeric_vars > 0) {
        metadata[["numeric_vars"]] = numeric_vars
    }
    output <- list(
        data = X,
        metadata = metadata
    )

    return(output)
}

#' Preprocess a dataframe of covariate values, converting categorical variables
#' to integers and one-hot encoding if need be.
#'
#' @param input_df Dataframe of covariates. Users must pre-process any
#' categorical variables as factors (ordered for ordered categorical).
#' @param metadata List containing information on variables, including train set
#' categories for categorical variables
#'
#' @return Preprocessed data with categorical variables appropriately preprocessed
#' @noRd
#'
#' @examples
#' cov_df <- data.frame(x1 = 1:5, x2 = 5:1, x3 = 6:10)
#' metadata <- list(num_ordered_cat_vars = 0, num_unordered_cat_vars = 0,
#'                  num_numeric_vars = 3, numeric_vars = c("x1", "x2", "x3"))
#' X_preprocessed <- preprocessPredictionDataFrame(cov_df, metadata)
preprocessPredictionDataFrame <- function(input_df, metadata) {
    if (!is.data.frame(input_df)) {
        stop("covariates provided must be a data frame")
    }
    df_vars <- names(input_df)
    num_ordered_cat_vars <- metadata$num_ordered_cat_vars
    num_unordered_cat_vars <- metadata$num_unordered_cat_vars
    num_numeric_vars <- metadata$num_numeric_vars

    if (num_ordered_cat_vars > 0) {
        ordered_cat_vars <- metadata$ordered_cat_vars
        ordered_cat_df <- input_df[, ordered_cat_vars, drop = FALSE]
    }
    if (num_unordered_cat_vars > 0) {
        unordered_cat_vars <- metadata$unordered_cat_vars
        unordered_cat_df <- input_df[, unordered_cat_vars, drop = FALSE]
    }
    if (num_numeric_vars > 0) {
        numeric_vars <- metadata$numeric_vars
        numeric_df <- input_df[, numeric_vars, drop = FALSE]
    }

    # Empty outputs
    X <- double(0)

    # First, extract the numeric covariates
    if (num_numeric_vars > 0) {
        Xnum <- double(0)
        for (i in 1:ncol(numeric_df)) {
            stopifnot(is.numeric(numeric_df[, i]))
            Xnum <- cbind(Xnum, numeric_df[, i])
        }
        X <- cbind(X, unname(Xnum))
    }

    # Next, run some simple preprocessing on the ordered categorical covariates
    if (num_ordered_cat_vars > 0) {
        Xordcat <- double(0)
        for (i in 1:ncol(ordered_cat_df)) {
            var_name <- names(ordered_cat_df)[i]
            x_preprocessed <- orderedCatPreprocess(
                ordered_cat_df[, i],
                metadata$ordered_unique_levels[[var_name]]
            )
            Xordcat <- cbind(Xordcat, x_preprocessed)
        }
        X <- cbind(X, unname(Xordcat))
    }

    # Finally, one-hot encode the unordered categorical covariates
    if (num_unordered_cat_vars > 0) {
        one_hot_mats <- list()
        for (i in 1:ncol(unordered_cat_df)) {
            var_name <- names(unordered_cat_df)[i]
            Xtilde <- oneHotEncode(
                unordered_cat_df[, i],
                metadata$unordered_unique_levels[[var_name]]
            )
            one_hot_mats[[var_name]] <- Xtilde
        }
        Xcat <- do.call(cbind, one_hot_mats)
        X <- cbind(X, unname(Xcat))
    }

    return(X)
}

#' Convert the persistent aspects of a covariate preprocessor to (in-memory) C++ JSON object
#'
#' @param object List containing information on variables, including train set
#' categories for categorical variables
#'
#' @return wrapper around in-memory C++ JSON object
#' @export
#'
#' @examples
#' cov_mat <- matrix(1:12, ncol = 3)
#' preprocess_list <- preprocessTrainData(cov_mat)
#' preprocessor_json <- convertPreprocessorToJson(preprocess_list$metadata)
convertPreprocessorToJson <- function(object) {
    jsonobj <- createCppJson()
    if (is.null(object$feature_types)) {
        stop("This covariate preprocessor has not yet been fit")
    }

    # Add internal scalars
    jsonobj$add_integer("num_numeric_vars", object$num_numeric_vars)
    jsonobj$add_integer("num_ordered_cat_vars", object$num_ordered_cat_vars)
    jsonobj$add_integer("num_unordered_cat_vars", object$num_unordered_cat_vars)

    # Add internal vectors
    jsonobj$add_vector("feature_types", object$feature_types)
    jsonobj$add_vector("original_var_indices", object$original_var_indices)
    if (object$num_numeric_vars > 0) {
        jsonobj$add_string_vector("numeric_vars", object$numeric_vars)
    }
    if (object$num_ordered_cat_vars > 0) {
        jsonobj$add_string_vector("ordered_cat_vars", object$ordered_cat_vars)
        for (i in 1:object$num_ordered_cat_vars) {
            var_key <- names(object$ordered_unique_levels)[i]
            jsonobj$add_string(
                paste0("key_", i),
                var_key,
                "ordered_unique_level_keys"
            )
            jsonobj$add_string_vector(
                var_key,
                object$ordered_unique_levels[[i]],
                "ordered_unique_levels"
            )
        }
    }
    if (object$num_unordered_cat_vars > 0) {
        jsonobj$add_string_vector(
            "unordered_cat_vars",
            object$unordered_cat_vars
        )
        for (i in 1:object$num_unordered_cat_vars) {
            var_key <- names(object$unordered_unique_levels)[i]
            jsonobj$add_string(
                paste0("key_", i),
                var_key,
                "unordered_unique_level_keys"
            )
            jsonobj$add_string_vector(
                var_key,
                object$unordered_unique_levels[[i]],
                "unordered_unique_levels"
            )
        }
    }

    return(jsonobj)
}

#' Convert the persistent aspects of a covariate preprocessor to (in-memory) JSON string
#'
#' @param object List containing information on variables, including train set
#' categories for categorical variables
#'
#' @return in-memory JSON string
#' @export
#'
#' @examples
#' cov_mat <- matrix(1:12, ncol = 3)
#' preprocess_list <- preprocessTrainData(cov_mat)
#' preprocessor_json_string <- savePreprocessorToJsonString(preprocess_list$metadata)
savePreprocessorToJsonString <- function(object) {
    # Convert to Json
    jsonobj <- convertPreprocessorToJson(object)

    # Dump to string
    return(jsonobj$return_json_string())
}

#' Reload a covariate preprocessor object from a JSON string containing a serialized preprocessor
#'
#' @param json_object in-memory wrapper around JSON C++ object containing covariate preprocessor metadata
#'
#' @returns Preprocessor object that can be used with the `preprocessPredictionData` function
#' @export
#'
#' @examples
#' cov_mat <- matrix(1:12, ncol = 3)
#' preprocess_list <- preprocessTrainData(cov_mat)
#' preprocessor_json <- convertPreprocessorToJson(preprocess_list$metadata)
#' preprocessor_roundtrip <- createPreprocessorFromJson(preprocessor_json)
createPreprocessorFromJson <- function(json_object) {
    # Initialize the metadata list
    metadata <- list()

    # Unpack internal scalars
    metadata[["num_numeric_vars"]] <- json_object$get_integer(
        "num_numeric_vars"
    )
    metadata[["num_ordered_cat_vars"]] <- json_object$get_integer(
        "num_ordered_cat_vars"
    )
    metadata[["num_unordered_cat_vars"]] <- json_object$get_integer(
        "num_unordered_cat_vars"
    )

    # Unpack internal vectors
    metadata[["feature_types"]] <- json_object$get_vector("feature_types")
    metadata[["original_var_indices"]] <- json_object$get_vector(
        "original_var_indices"
    )
    if (metadata$num_numeric_vars > 0) {
        metadata[["numeric_vars"]] <- json_object$get_string_vector(
            "numeric_vars"
        )
    }
    if (metadata$num_ordered_cat_vars > 0) {
        metadata[["ordered_cat_vars"]] <- json_object$get_string_vector(
            "ordered_cat_vars"
        )
        ordered_unique_levels <- list()
        for (i in 1:metadata$num_ordered_cat_vars) {
            var_key <- json_object$get_string(
                paste0("key_", i),
                "ordered_unique_level_keys"
            )
            ordered_unique_levels[[var_key]] <- json_object$get_string_vector(
                var_key,
                "ordered_unique_levels"
            )
        }
        metadata[["ordered_unique_levels"]] <- ordered_unique_levels
    }
    if (metadata$num_unordered_cat_vars > 0) {
        metadata[["unordered_cat_vars"]] <- json_object$get_string_vector(
            "unordered_cat_vars"
        )
        unordered_unique_levels <- list()
        for (i in 1:metadata$num_unordered_cat_vars) {
            var_key <- json_object$get_string(
                paste0("key_", i),
                "unordered_unique_level_keys"
            )
            unordered_unique_levels[[var_key]] <- json_object$get_string_vector(
                var_key,
                "unordered_unique_levels"
            )
        }
        metadata[["unordered_unique_levels"]] <- unordered_unique_levels
    }

    return(metadata)
}

#' Reload a covariate preprocessor object from a JSON string containing a serialized preprocessor
#'
#' @param json_string in-memory JSON string containing covariate preprocessor metadata
#'
#' @return Preprocessor object that can be used with the `preprocessPredictionData` function
#' @export
#'
#' @examples
#' cov_mat <- matrix(1:12, ncol = 3)
#' preprocess_list <- preprocessTrainData(cov_mat)
#' preprocessor_json_string <- savePreprocessorToJsonString(preprocess_list$metadata)
#' preprocessor_roundtrip <- createPreprocessorFromJsonString(preprocessor_json_string)
createPreprocessorFromJsonString <- function(json_string) {
    # Load a `CppJson` object from string
    preprocessor_json <- createCppJsonString(json_string)

    # Create and return the BCF object
    preprocessor_object <- createPreprocessorFromJson(preprocessor_json)

    return(preprocessor_object)
}

#' Preprocess a dataframe of covariate values, converting categorical variables
#' to integers and one-hot encoding if need be. Returns a list including a
#' matrix of preprocessed covariate values and associated tracking.
#'
#' @param input_data Dataframe or matrix of covariates. Users may pre-process any
#' categorical variables as factors but it is not necessary.
#' @param ordered_cat_vars (Optional) Vector of names of ordered categorical variables, or vector of column indices if `input_data` is a matrix.
#' @param unordered_cat_vars (Optional) Vector of names of unordered categorical variables, or vector of column indices if `input_data` is a matrix.
#'
#' @return List with preprocessed data and details on the number of each type
#' of variable, unique categories associated with categorical variables, and the
#' vector of feature types needed for calls to BART and BCF.
#' @noRd
#'
#' @examples
#' cov_df <- data.frame(x1 = 1:5, x2 = 5:1, x3 = 6:10)
#' preprocess_list <- createForestCovariates(cov_df)
#' X <- preprocess_list$X
createForestCovariates <- function(
    input_data,
    ordered_cat_vars = NULL,
    unordered_cat_vars = NULL
) {
    if (is.matrix(input_data)) {
        input_df <- as.data.frame(input_data)
        names(input_df) <- paste0("x", 1:ncol(input_data))
        if (!is.null(ordered_cat_vars)) {
            if (is.numeric(ordered_cat_vars)) {
                ordered_cat_vars <- paste0("x", as.integer(ordered_cat_vars))
            }
        }
        if (!is.null(unordered_cat_vars)) {
            if (is.numeric(unordered_cat_vars)) {
                unordered_cat_vars <- paste0(
                    "x",
                    as.integer(unordered_cat_vars)
                )
            }
        }
    } else if (is.data.frame(input_data)) {
        input_df <- input_data
    } else {
        stop("input_data must be either a matrix or a data frame")
    }
    df_vars <- names(input_df)
    if (is.null(ordered_cat_vars)) {
        ordered_cat_matches <- rep(FALSE, length(df_vars))
    } else {
        ordered_cat_matches <- df_vars %in% ordered_cat_vars
    }
    if (is.null(unordered_cat_vars)) {
        unordered_cat_matches <- rep(FALSE, length(df_vars))
    } else {
        unordered_cat_matches <- df_vars %in% unordered_cat_vars
    }
    numeric_matches <- ((!ordered_cat_matches) & (!unordered_cat_matches))
    ordered_cat_vars <- df_vars[ordered_cat_matches]
    unordered_cat_vars <- df_vars[unordered_cat_matches]
    numeric_vars <- df_vars[numeric_matches]
    num_ordered_cat_vars <- length(ordered_cat_vars)
    num_unordered_cat_vars <- length(unordered_cat_vars)
    num_numeric_vars <- length(numeric_vars)
    if (num_ordered_cat_vars > 0) {
        ordered_cat_df <- input_df[, ordered_cat_vars, drop = FALSE]
    }
    if (num_unordered_cat_vars > 0) {
        unordered_cat_df <- input_df[, unordered_cat_vars, drop = FALSE]
    }
    if (num_numeric_vars > 0) {
        numeric_df <- input_df[, numeric_vars, drop = FALSE]
    }

    # Empty outputs
    X <- double(0)
    unordered_unique_levels <- list()
    ordered_unique_levels <- list()
    feature_types <- integer(0)

    # First, extract the numeric covariates
    if (num_numeric_vars > 0) {
        Xnum <- double(0)
        for (i in 1:ncol(numeric_df)) {
            stopifnot(is.numeric(numeric_df[, i]))
            Xnum <- cbind(Xnum, numeric_df[, i])
        }
        X <- cbind(X, unname(Xnum))
        feature_types <- c(feature_types, rep(0, ncol(Xnum)))
    }

    # Next, run some simple preprocessing on the ordered categorical covariates
    if (num_ordered_cat_vars > 0) {
        Xordcat <- double(0)
        for (i in 1:ncol(ordered_cat_df)) {
            var_name <- names(ordered_cat_df)[i]
            preprocess_list <- orderedCatInitializeAndPreprocess(ordered_cat_df[,
                i
            ])
            ordered_unique_levels[[var_name]] <- preprocess_list$unique_levels
            Xordcat <- cbind(Xordcat, preprocess_list$x_preprocessed)
        }
        X <- cbind(X, unname(Xordcat))
        feature_types <- c(feature_types, rep(1, ncol(Xordcat)))
    }

    # Finally, one-hot encode the unordered categorical covariates
    if (num_unordered_cat_vars > 0) {
        one_hot_mats <- list()
        for (i in 1:ncol(unordered_cat_df)) {
            var_name <- names(unordered_cat_df)[i]
            encode_list <- oneHotInitializeAndEncode(unordered_cat_df[, i])
            unordered_unique_levels[[var_name]] <- encode_list$unique_levels
            one_hot_mats[[var_name]] <- encode_list$Xtilde
        }
        Xcat <- do.call(cbind, one_hot_mats)
        X <- cbind(X, unname(Xcat))
        feature_types <- c(feature_types, rep(1, ncol(Xcat)))
    }

    # Aggregate results into a list
    metadata <- list(
        feature_types = feature_types,
        num_ordered_cat_vars = num_ordered_cat_vars,
        num_unordered_cat_vars = num_unordered_cat_vars,
        num_numeric_vars = num_numeric_vars
    )
    if (num_ordered_cat_vars > 0) {
        metadata[["ordered_cat_vars"]] = ordered_cat_vars
        metadata[["ordered_unique_levels"]] = ordered_unique_levels
    }
    if (num_unordered_cat_vars > 0) {
        metadata[["unordered_cat_vars"]] = unordered_cat_vars
        metadata[["unordered_unique_levels"]] = unordered_unique_levels
    }
    if (num_numeric_vars > 0) {
        metadata[["numeric_vars"]] = numeric_vars
    }
    output <- list(
        data = X,
        metadata = metadata
    )

    return(output)
}

#' Preprocess a dataframe of covariate values, converting categorical variables
#' to integers and one-hot encoding if need be. Returns a list including a
#' matrix of preprocessed covariate values and associated tracking.
#'
#' @param input_data Dataframe or matrix of covariates. Users may pre-process any
#' categorical variables as factors but it is not necessary.
#' @param metadata List containing information on variables, including train set
#' categories for categorical variables
#'
#' @return Preprocessed data with categorical variables appropriately preprocessed
#' @noRd
#'
#' @examples
#' cov_df <- data.frame(x1 = 1:5, x2 = 5:1, x3 = 6:10)
#' metadata <- list(num_ordered_cat_vars = 0, num_unordered_cat_vars = 0,
#'                  num_numeric_vars = 3, numeric_vars = c("x1", "x2", "x3"))
#' X_preprocessed <- createForestCovariatesFromMetadata(cov_df, metadata)
createForestCovariatesFromMetadata <- function(input_data, metadata) {
    if (is.matrix(input_data)) {
        input_df <- as.data.frame(input_data)
        names(input_df) <- paste0("x", 1:ncol(input_data))
    } else if (is.data.frame(input_data)) {
        input_df <- input_data
    } else {
        stop("input_data must be either a matrix or a data frame")
    }
    df_vars <- names(input_df)
    num_ordered_cat_vars <- metadata$num_ordered_cat_vars
    num_unordered_cat_vars <- metadata$num_unordered_cat_vars
    num_numeric_vars <- metadata$num_numeric_vars

    if (num_ordered_cat_vars > 0) {
        ordered_cat_vars <- metadata$ordered_cat_vars
        ordered_cat_df <- input_df[, ordered_cat_vars, drop = FALSE]
    }
    if (num_unordered_cat_vars > 0) {
        unordered_cat_vars <- metadata$unordered_cat_vars
        unordered_cat_df <- input_df[, unordered_cat_vars, drop = FALSE]
    }
    if (num_numeric_vars > 0) {
        numeric_vars <- metadata$numeric_vars
        numeric_df <- input_df[, numeric_vars, drop = FALSE]
    }

    # Empty outputs
    X <- double(0)

    # First, extract the numeric covariates
    if (num_numeric_vars > 0) {
        Xnum <- double(0)
        for (i in 1:ncol(numeric_df)) {
            stopifnot(is.numeric(numeric_df[, i]))
            Xnum <- cbind(Xnum, numeric_df[, i])
        }
        X <- cbind(X, unname(Xnum))
    }

    # Next, run some simple preprocessing on the ordered categorical covariates
    if (num_ordered_cat_vars > 0) {
        Xordcat <- double(0)
        for (i in 1:ncol(ordered_cat_df)) {
            var_name <- names(ordered_cat_df)[i]
            x_preprocessed <- orderedCatPreprocess(
                ordered_cat_df[, i],
                metadata$ordered_unique_levels[[var_name]]
            )
            Xordcat <- cbind(Xordcat, x_preprocessed)
        }
        X <- cbind(X, unname(Xordcat))
    }

    # Finally, one-hot encode the unordered categorical covariates
    if (num_unordered_cat_vars > 0) {
        one_hot_mats <- list()
        for (i in 1:ncol(unordered_cat_df)) {
            var_name <- names(unordered_cat_df)[i]
            Xtilde <- oneHotEncode(
                unordered_cat_df[, i],
                metadata$unordered_unique_levels[[var_name]]
            )
            one_hot_mats[[var_name]] <- Xtilde
        }
        Xcat <- do.call(cbind, one_hot_mats)
        X <- cbind(X, unname(Xcat))
    }

    return(X)
}

#' Convert a vector of unordered categorical data (either numeric or character
#' labels) to a "one-hot" encoded matrix in which a 1 in a column indicates
#' the presence of the relevant category.
#'
#' To allow for prediction on "unseen" categories in a test dataset, this
#' procedure pads the one-hot matrix with a blank "other" column.
#' Test set observations that contain categories not in `levels(factor(x_input))`
#' will all be mapped to this column.
#'
#' @param x_input Vector of unordered categorical data (typically either strings
#' integers, but this function also accepts floating point data).
#'
#' @return List containing a binary one-hot matrix and the unique levels of the
#' input variable. These unique levels are used in the BCF and BART functions.
#' @noRd
#'
#' @examples
#' x <- c("a","c","b","c","d","a","c","a","b","d")
#' x_onehot <- oneHotInitializeAndEncode(x)
oneHotInitializeAndEncode <- function(x_input) {
    stopifnot((is.null(dim(x_input)) && length(x_input) > 0))
    if (is.factor(x_input) && is.ordered(x_input)) {
        warning("One-hot encoding an ordered categorical variable")
    }
    x_factor <- factor(x_input)
    unique_levels <- levels(x_factor)
    Xtilde <- cbind(unname(model.matrix(~ 0 + x_factor)), 0)
    output <- list(Xtilde = Xtilde, unique_levels = unique_levels)
    return(output)
}

#' Convert a vector of unordered categorical data (either numeric or character
#' labels) to a "one-hot" encoded matrix in which a 1 in a column indicates
#' the presence of the relevant category.
#'
#' This procedure assumes that a reference set of observations for this variable
#' (typically a training set that was used to sample a forest) has already been
#' one-hot encoded and that the unique levels of the training set variable are
#' available (and passed as `unique_levels`). Test set observations that contain
#' categories not in `unique_levels` will all be mapped to the last column of
#' this matrix
#'
#' @param x_input Vector of unordered categorical data (typically either strings
#' integers, but this function also accepts floating point data).
#' @param unique_levels Unique values of the categorical variable used to create
#' the initial one-hot matrix (typically a training set)
#'
#' @return Binary one-hot matrix
#' @noRd
#'
#' @examples
#' x <- sample(1:8, 100, TRUE)
#' x_test <- sample(1:9, 10, TRUE)
#' x_onehot <- oneHotEncode(x_test, levels(factor(x)))
oneHotEncode <- function(x_input, unique_levels) {
    stopifnot((is.null(dim(x_input)) && length(x_input) > 0))
    stopifnot((is.null(dim(unique_levels)) && length(unique_levels) > 0))
    num_unique_levels <- length(unique_levels)
    in_sample <- x_input %in% unique_levels
    out_of_sample <- !(x_input %in% unique_levels)
    has_out_of_sample <- sum(out_of_sample) > 0
    if (has_out_of_sample) {
        x_factor_insample <- factor(x_input[in_sample], levels = unique_levels)
        Xtilde <- matrix(
            0,
            nrow = length(x_input),
            ncol = num_unique_levels + 1
        )
        Xtilde_insample <- cbind(
            unname(model.matrix(~ 0 + x_factor_insample)),
            0
        )
        Xtilde_out_of_sample <- cbind(
            matrix(0, nrow = sum(out_of_sample), ncol = num_unique_levels),
            1
        )
        Xtilde[in_sample, ] <- Xtilde_insample
        Xtilde[out_of_sample, ] <- Xtilde_out_of_sample
    } else {
        x_factor <- factor(x_input, levels = unique_levels)
        Xtilde <- cbind(unname(model.matrix(~ 0 + x_factor)), 0)
    }
    return(Xtilde)
}

#' Run some simple preprocessing of ordered categorical variables, converting
#' ordered levels to integers if necessary, and storing the unique levels of a
#' variable.
#'
#' @param x_input Vector of ordered categorical data. If the data is not already
#' stored as an ordered factor, it will be converted to one using the default
#' sort order.
#'
#' @return List containing a preprocessed vector of integer-converted ordered
#' categorical observations and the unique level of the original ordered
#' categorical feature.
#' @noRd
#'
#' @examples
#' x <- c("1. Strongly disagree", "3. Neither agree nor disagree", "2. Disagree",
#'        "4. Agree", "3. Neither agree nor disagree", "5. Strongly agree", "4. Agree")
#' preprocess_list <- orderedCatInitializeAndPreprocess(x)
#' x_preprocessed <- preprocess_list$x_preprocessed
orderedCatInitializeAndPreprocess <- function(x_input) {
    stopifnot((is.null(dim(x_input)) && length(x_input) > 0))
    already_ordered_factor <- (is.factor(x_input)) && (is.ordered(x_input))
    if (already_ordered_factor) {
        x_preprocessed <- as.integer(x_input)
        unique_levels <- levels(x_input)
    } else {
        x_factor <- factor(x_input, ordered = TRUE)
        x_preprocessed <- as.integer(x_factor)
        unique_levels <- levels(x_factor)
    }
    return(list(x_preprocessed = x_preprocessed, unique_levels = unique_levels))
}

#' Run some simple preprocessing of ordered categorical variables, converting
#' ordered levels to integers if necessary, and storing the unique levels of a
#' variable.
#'
#' @param x_input Vector of ordered categorical data. If the data is not already
#' stored as an ordered factor, it will be converted to one using the default
#' sort order.
#' @param unique_levels Vector of unique levels for a categorical feature.
#' @param var_name (Optional) Name of variable.
#'
#' @return List containing a preprocessed vector of integer-converted ordered
#' categorical observations and the unique level of the original ordered
#' categorical feature.
#' @noRd
#'
#' @examples
#' x_levels <- c("1. Strongly disagree", "2. Disagree",
#'               "3. Neither agree nor disagree",
#'               "4. Agree", "5. Strongly agree")
#' x <- c("1. Strongly disagree", "3. Neither agree nor disagree", "2. Disagree",
#'        "4. Agree", "3. Neither agree nor disagree", "5. Strongly agree", "4. Agree")
#' x_processed <- orderedCatPreprocess(x, x_levels)
orderedCatPreprocess <- function(x_input, unique_levels, var_name = NULL) {
    stopifnot((is.null(dim(x_input)) && length(x_input) > 0))
    stopifnot((is.null(dim(unique_levels)) && length(unique_levels) > 0))
    already_ordered_factor <- (is.factor(x_input)) && (is.ordered(x_input))
    if (already_ordered_factor) {
        # Run time checks
        levels_not_in_reflist <- !(levels(x_input) %in% unique_levels)
        if (sum(levels_not_in_reflist) > 0) {
            if (!is.null(var_name)) {
                warning_message <- paste0(
                    "Variable ",
                    var_name,
                    " includes ordered categorical levels not included in the original training set"
                )
            } else {
                warning_message <- paste0(
                    "Variable includes ordered categorical levels not included in the original training set"
                )
            }
            warning(warning_message)
        }
        # Preprocessing
        x_string <- as.character(x_input)
        x_factor <- factor(x_string, unique_levels, ordered = TRUE)
        x_preprocessed <- as.integer(x_factor)
        x_preprocessed[is.na(x_preprocessed)] <- length(unique_levels) + 1
    } else {
        x_factor <- factor(x_input, ordered = TRUE)
        # Run time checks
        levels_not_in_reflist <- !(levels(x_factor) %in% unique_levels)
        if (sum(levels_not_in_reflist) > 0) {
            if (!is.null(var_name)) {
                warning_message <- paste0(
                    "Variable ",
                    var_name,
                    " includes ordered categorical levels not included in the original training set"
                )
            } else {
                warning_message <- paste0(
                    "Variable includes ordered categorical levels not included in the original training set"
                )
            }
            warning(warning_message)
        }
        # Preprocessing
        x_string <- as.character(x_input)
        x_factor <- factor(x_string, unique_levels, ordered = TRUE)
        x_preprocessed <- as.integer(x_factor)
        x_preprocessed[is.na(x_preprocessed)] <- length(unique_levels) + 1
    }
    return(x_preprocessed)
}

#' Convert scalar input to vector of dimension `output_size`,
#' or check that input array is equivalent to a vector of dimension `output_size`.
#'
#' @param input Input to be converted to a vector (or passed through as-is)
#' @param output_size Intended size of the output vector
#' @return A vector of length `output_size`
#' @export
expand_dims_1d <- function(input, output_size) {
    if (length(input) == 1) {
        output <- rep(input, output_size)
    } else if (is.numeric(input)) {
        if (length(input) != output_size) {
            stop("`input` must be a 1D numpy array with `output_size` elements")
        }
        output <- input
    } else {
        stop(
            "`input` must be either a 1D numpy array or a scalar that can be repeated `output_size` times"
        )
    }
    return(output)
}

#' Ensures that input is propagated appropriately to a matrix of dimension `output_rows` x `output_cols`.
#' Handles the following cases:
#'  1. `input` is a scalar: output is simply a (`output_rows`, `output_cols`) matrix with `input` repeated for each element
#'  2. `input` is a vector of length `output_rows`: output is a (`output_rows`, `output_cols`) array with `input` broadcast across each of `output_cols` columns
#'  3. `input` is a vector of length `output_cols`: output is a (`output_rows`, `output_cols`) array with `input` broadcast across each of `output_rows` rows
#'  4. `input` is a matrix of dimension (`output_rows`, `output_cols`): input is passed through as-is
#' All other cases throw an error.
#'
#' @param input Input to be converted to a matrix (or passed through as-is)
#' @param output_rows Intended number of rows in the output array
#' @param output_cols Intended number of columns in the output array
#' @return A matrix of dimension `output_rows` x `output_cols`
#' @export
expand_dims_2d <- function(input, output_rows, output_cols) {
    if (length(input) == 1) {
        output <- matrix(
            rep(input, output_rows * output_cols),
            ncol = output_cols
        )
    } else if (is.numeric(input)) {
        if (length(input) == output_cols) {
            output <- matrix(
                rep(input, output_rows),
                nrow = output_rows,
                byrow = T
            )
        } else if (length(input) == output_rows) {
            output <- matrix(
                rep(input, output_cols),
                ncol = output_cols,
                byrow = F
            )
        } else {
            stop(
                "If `input` is a vector, it must either contain `output_rows` or `output_cols` elements"
            )
        }
    } else if (is.matrix(input)) {
        if (nrow(input) != output_rows) {
            stop("`input` must be a matrix with `output_rows` rows")
        }
        if (ncol(input) != output_cols) {
            stop("`input` must be a matrix with `output_cols` columns")
        }
        output <- input
    } else {
        stop("`input` must be either a matrix, vector or a scalar")
    }
    return(output)
}

#' Convert scalar input to square matrix of dimension `output_size` x `output_size` with `input` along the diagonal,
#' or check that input array is equivalent to a square matrix of dimension `output_size` x `output_size`.
#'
#' @param input Input to be converted to a square matrix (or passed through as-is)
#' @param output_size Intended row and column dimension of the square output matrix
#' @return A square matrix of dimension `output_size` x `output_size`
#' @export
expand_dims_2d_diag <- function(input, output_size) {
    if (length(input) == 1) {
        output <- as.matrix(diag(input, output_size))
    } else if (is.matrix(input)) {
        if (nrow(input) != ncol(input)) {
            stop("`input` must be a square matrix")
        }
        if (nrow(input) != output_size) {
            stop(
                "`input` must be a square matrix with `output_size` rows and columns"
            )
        }
        output <- input
    } else {
        stop("`input` must be either a square matrix or a scalar")
    }
    return(output)
}
