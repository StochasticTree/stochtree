#' Preprocess BART parameter list. Override defaults with any provided parameters.
#'
#' @param params Parameter list
#'
#' @return Parameter list with defaults overriden by values supplied in `params`
#' @export
preprocessBartParams <- function(params) {
    # Default parameter values
    processed_params <- list(
        cutpoint_grid_size = 100, sigma_leaf_init = NULL, 
        alpha_mean = 0.95, beta_mean = 2.0, 
        min_samples_leaf_mean = 5, max_depth_mean = 10, 
        alpha_variance = 0.95, beta_variance = 2.0, 
        min_samples_leaf_variance = 5, max_depth_variance = 10, 
        a_global = 0, b_global = 0, a_leaf = 3, b_leaf = NULL, 
        a_forest = NULL, b_forest = NULL, variance_scale = 1, 
        sigma2_init = NULL, variance_forest_init = NULL, 
        pct_var_sigma2_init = 1, pct_var_variance_forest_init = 1, 
        variable_weights_mean = NULL, variable_weights_variance = NULL, 
        num_trees_mean = 200, num_trees_variance = 0, 
        sample_sigma_global = T, sample_sigma_leaf = F, 
        random_seed = -1, keep_burnin = F, keep_gfr = F, keep_every = 1, 
        num_chains = 1, standardize = T, verbose = F
    )
    
    # Override defaults
    for (key in names(params)) {
        if (!key %in% names(processed_params)) {
            stop("Variable ", key, " is not a valid BART model parameter")
        }
        val <- params[[key]]
        if (!is.null(val)) processed_params[[key]] <- val
    }
    
    # Return result
    return(processed_params)
}

#' Preprocess BCF parameter list. Override defaults with any provided parameters.
#'
#' @param params Parameter list
#'
#' @return Parameter list with defaults overriden by values supplied in `params`
#' @export
preprocessBcfParams <- function(params) {
    # Default parameter values
    processed_params <- list(
        cutpoint_grid_size = 100, sigma_leaf_mu = NULL, sigma_leaf_tau = NULL, 
        alpha_mu = 0.95, alpha_tau = 0.25, alpha_variance = 0.95, 
        beta_mu = 2.0, beta_tau = 3.0, beta_variance = 2.0, 
        min_samples_leaf_mu = 5, min_samples_leaf_tau = 5, min_samples_leaf_variance = 5, 
        max_depth_mu = 10, max_depth_tau = 5, max_depth_variance = 10, 
        a_global = 0, b_global = 0, a_leaf_mu = 3, a_leaf_tau = 3, b_leaf_mu = NULL, 
        b_leaf_tau = NULL, a_forest = NULL, b_forest = NULL, sigma2_init = NULL, 
        variance_forest_init = NULL, pct_var_sigma2_init = 1, pct_var_variance_forest_init = 1, 
        variable_weights = NULL, keep_vars_mu = NULL, drop_vars_mu = NULL, 
        keep_vars_tau = NULL, drop_vars_tau = NULL, keep_vars_variance = NULL, 
        drop_vars_variance = NULL, num_trees_mu = 250, num_trees_tau = 50, 
        num_trees_variance = 0, num_gfr = 5, num_burnin = 0, num_mcmc = 100, 
        sample_sigma_global = T, sample_sigma_leaf_mu = T, sample_sigma_leaf_tau = F, 
        propensity_covariate = "mu", adaptive_coding = T, b_0 = -0.5, b_1 = 0.5, 
        rfx_prior_var = NULL, random_seed = -1, keep_burnin = F, keep_gfr = F, 
        keep_every = 1, num_chains = 1, standardize = T, verbose = F
    )
    
    # Override defaults
    for (key in names(params)) {
        if (!key %in% names(processed_params)) {
            stop("Variable ", key, " is not a valid BART model parameter")
        }
        val <- params[[key]]
        if (!is.null(val)) processed_params[[key]] <- val
    }
    
    # Return result
    return(processed_params)
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
#' @export
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
#' @export
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
        stop("Prediction set covariates have inconsistent dimension from train set covariates")
    }
    
    return(input_matrix)
}

#' Preprocess a dataframe of covariate values, converting categorical variables 
#' to integers and one-hot encoding if need be. Returns a list including a 
#' matrix of preprocessed covariate values and associated tracking.
#'
#' @param input_df Dataframe of covariates. Users must pre-process any 
#' categorical variables as factors (ordered for ordered categorical).
#'
#' @return List with preprocessed data and details on the number of each type 
#' of variable, unique categories associated with categorical variables, and the 
#' vector of feature types needed for calls to BART and BCF.
#' @export
#'
#' @examples
#' cov_df <- data.frame(x1 = 1:5, x2 = 5:1, x3 = 6:10)
#' preprocess_list <- preprocessTrainDataFrame(cov_df)
#' X <- preprocess_list$X
preprocessTrainDataFrame <- function(input_df) {
    # Input checks / details
    if (!is.data.frame(input_df)) {
        stop("covariates provided must be a data frame")
    }
    df_vars <- names(input_df)
    
    # Detect ordered and unordered categorical variables
    
    # First, ordered categorical: users must have explicitly 
    # converted this to a factor with ordered = T
    factor_mask <- sapply(input_df, is.factor)
    ordered_mask <- sapply(input_df, is.ordered)
    ordered_cat_matches <- factor_mask & ordered_mask
    ordered_cat_vars <- df_vars[ordered_cat_matches]
    ordered_cat_var_inds <- unname(which(ordered_cat_matches))
    num_ordered_cat_vars <- length(ordered_cat_vars)
    if (num_ordered_cat_vars > 0) ordered_cat_df <- input_df[,ordered_cat_vars,drop=F]
    
    # Next, unordered categorical: we will convert character 
    # columns but not integer columns (users must explicitly 
    # convert these to factor)
    character_mask <- sapply(input_df, is.character)
    unordered_cat_matches <- (factor_mask & (!ordered_mask)) | character_mask
    unordered_cat_vars <- df_vars[unordered_cat_matches]
    unordered_cat_var_inds <- unname(which(unordered_cat_matches))
    num_unordered_cat_vars <- length(unordered_cat_vars)
    if (num_unordered_cat_vars > 0) unordered_cat_df <- input_df[,unordered_cat_vars,drop=F]
    
    # Numeric variables
    numeric_matches <- (!ordered_cat_matches) & (!unordered_cat_matches)
    numeric_vars <- df_vars[numeric_matches]
    numeric_var_inds <- unname(which(numeric_matches))
    num_numeric_vars <- length(numeric_vars)
    if (num_numeric_vars > 0) numeric_df <- input_df[,numeric_vars,drop=F]
    
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
            stopifnot(is.numeric(numeric_df[,i]))
            Xnum <- cbind(Xnum, numeric_df[,i])
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
            preprocess_list <- orderedCatInitializeAndPreprocess(ordered_cat_df[,i])
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
            encode_list <- oneHotInitializeAndEncode(unordered_cat_df[,i])
            unordered_unique_levels[[var_name]] <- encode_list$unique_levels
            one_hot_mats[[var_name]] <- encode_list$Xtilde
            one_hot_var <- rep(unordered_cat_var_inds[i], ncol(encode_list$Xtilde))
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
    if (num_numeric_vars > 0) metadata[["numeric_vars"]] = numeric_vars
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
#' @export
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
        ordered_cat_df <- input_df[,ordered_cat_vars,drop=F]
    }
    if (num_unordered_cat_vars > 0) {
        unordered_cat_vars <- metadata$unordered_cat_vars
        unordered_cat_df <- input_df[,unordered_cat_vars,drop=F]
    }
    if (num_numeric_vars > 0) {
        numeric_vars <- metadata$numeric_vars
        numeric_df <- input_df[,numeric_vars,drop=F]
    }
    
    # Empty outputs
    X <- double(0)
    
    # First, extract the numeric covariates
    if (num_numeric_vars > 0) {
        Xnum <- double(0)
        for (i in 1:ncol(numeric_df)) {
            stopifnot(is.numeric(numeric_df[,i]))
            Xnum <- cbind(Xnum, numeric_df[,i])
        }
        X <- cbind(X, unname(Xnum))
    }
    
    # Next, run some simple preprocessing on the ordered categorical covariates
    if (num_ordered_cat_vars > 0) {
        Xordcat <- double(0)
        for (i in 1:ncol(ordered_cat_df)) {
            var_name <- names(ordered_cat_df)[i]
            x_preprocessed <- orderedCatPreprocess(ordered_cat_df[,i], metadata$ordered_unique_levels[[var_name]])
            Xordcat <- cbind(Xordcat, x_preprocessed)
        }
        X <- cbind(X, unname(Xordcat))
    }
    
    # Finally, one-hot encode the unordered categorical covariates
    if (num_unordered_cat_vars > 0) {
        one_hot_mats <- list()
        for (i in 1:ncol(unordered_cat_df)) {
            var_name <- names(unordered_cat_df)[i]
            Xtilde <- oneHotEncode(unordered_cat_df[,i], metadata$unordered_unique_levels[[var_name]])
            one_hot_mats[[var_name]] <- Xtilde
        }
        Xcat <- do.call(cbind, one_hot_mats)
        X <- cbind(X, unname(Xcat))
    }
    
    return(X)
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
#' @export
#'
#' @examples
#' cov_df <- data.frame(x1 = 1:5, x2 = 5:1, x3 = 6:10)
#' preprocess_list <- createForestCovariates(cov_df)
#' X <- preprocess_list$X
createForestCovariates <- function(input_data, ordered_cat_vars = NULL, unordered_cat_vars = NULL) {
    if (is.matrix(input_data)) {
        input_df <- as.data.frame(input_data)
        names(input_df) <- paste0("x", 1:ncol(input_data))
        if (!is.null(ordered_cat_vars)) {
            if (is.numeric(ordered_cat_vars)) ordered_cat_vars <- paste0("x", as.integer(ordered_cat_vars))
        }
        if (!is.null(unordered_cat_vars)) {
            if (is.numeric(unordered_cat_vars)) unordered_cat_vars <- paste0("x", as.integer(unordered_cat_vars))
        }
    } else if (is.data.frame(input_data)) {
        input_df <- input_data
    } else {
        stop("input_data must be either a matrix or a data frame")
    }
    df_vars <- names(input_df)
    if (is.null(ordered_cat_vars)) ordered_cat_matches <- rep(F, length(df_vars))
    else ordered_cat_matches <- df_vars %in% ordered_cat_vars
    if (is.null(unordered_cat_vars)) unordered_cat_matches <- rep(F, length(df_vars))
    else unordered_cat_matches <- df_vars %in% unordered_cat_vars
    numeric_matches <- ((!ordered_cat_matches) & (!unordered_cat_matches))
    ordered_cat_vars <- df_vars[ordered_cat_matches]
    unordered_cat_vars <- df_vars[unordered_cat_matches]
    numeric_vars <- df_vars[numeric_matches]
    num_ordered_cat_vars <- length(ordered_cat_vars)
    num_unordered_cat_vars <- length(unordered_cat_vars)
    num_numeric_vars <- length(numeric_vars)
    if (num_ordered_cat_vars > 0) ordered_cat_df <- input_df[,ordered_cat_vars,drop=F]
    if (num_unordered_cat_vars > 0) unordered_cat_df <- input_df[,unordered_cat_vars,drop=F]
    if (num_numeric_vars > 0) numeric_df <- input_df[,numeric_vars,drop=F]
    
    # Empty outputs
    X <- double(0)
    unordered_unique_levels <- list()
    ordered_unique_levels <- list()
    feature_types <- integer(0)
    
    # First, extract the numeric covariates
    if (num_numeric_vars > 0) {
        Xnum <- double(0)
        for (i in 1:ncol(numeric_df)) {
            stopifnot(is.numeric(numeric_df[,i]))
            Xnum <- cbind(Xnum, numeric_df[,i])
        }
        X <- cbind(X, unname(Xnum))
        feature_types <- c(feature_types, rep(0, ncol(Xnum)))
    }
    
    # Next, run some simple preprocessing on the ordered categorical covariates
    if (num_ordered_cat_vars > 0) {
        Xordcat <- double(0)
        for (i in 1:ncol(ordered_cat_df)) {
            var_name <- names(ordered_cat_df)[i]
            preprocess_list <- orderedCatInitializeAndPreprocess(ordered_cat_df[,i])
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
            encode_list <- oneHotInitializeAndEncode(unordered_cat_df[,i])
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
    if (num_numeric_vars > 0) metadata[["numeric_vars"]] = numeric_vars
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
#' @export
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
        ordered_cat_df <- input_df[,ordered_cat_vars,drop=F]
    }
    if (num_unordered_cat_vars > 0) {
        unordered_cat_vars <- metadata$unordered_cat_vars
        unordered_cat_df <- input_df[,unordered_cat_vars,drop=F]
    }
    if (num_numeric_vars > 0) {
        numeric_vars <- metadata$numeric_vars
        numeric_df <- input_df[,numeric_vars,drop=F]
    }
        
    # Empty outputs
    X <- double(0)

    # First, extract the numeric covariates
    if (num_numeric_vars > 0) {
        Xnum <- double(0)
        for (i in 1:ncol(numeric_df)) {
            stopifnot(is.numeric(numeric_df[,i]))
            Xnum <- cbind(Xnum, numeric_df[,i])
        }
        X <- cbind(X, unname(Xnum))
    }
    
    # Next, run some simple preprocessing on the ordered categorical covariates
    if (num_ordered_cat_vars > 0) {
        Xordcat <- double(0)
        for (i in 1:ncol(ordered_cat_df)) {
            var_name <- names(ordered_cat_df)[i]
            x_preprocessed <- orderedCatPreprocess(ordered_cat_df[,i], metadata$ordered_unique_levels[[var_name]])
            Xordcat <- cbind(Xordcat, x_preprocessed)
        }
        X <- cbind(X, unname(Xordcat))
    }
    
    # Finally, one-hot encode the unordered categorical covariates
    if (num_unordered_cat_vars > 0) {
        one_hot_mats <- list()
        for (i in 1:ncol(unordered_cat_df)) {
            var_name <- names(unordered_cat_df)[i]
            Xtilde <- oneHotEncode(unordered_cat_df[,i], metadata$unordered_unique_levels[[var_name]])
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
#' @export
#'
#' @examples
#' x <- c("a","c","b","c","d","a","c","a","b","d")
#' x_onehot <- oneHotInitializeAndEncode(x)
oneHotInitializeAndEncode <- function(x_input) {
    stopifnot((is.null(dim(x_input)) && length(x_input) > 0))
    if (is.factor(x_input) && is.ordered(x_input)) warning("One-hot encoding an ordered categorical variable")
    x_factor <- factor(x_input)
    unique_levels <- levels(x_factor)
    Xtilde <- cbind(unname(model.matrix(~0+x_factor)), 0)
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
#' @export
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
        Xtilde <- matrix(0, nrow = length(x_input), ncol = num_unique_levels + 1)
        Xtilde_insample <- cbind(unname(model.matrix(~0+x_factor_insample)), 0)
        Xtilde_out_of_sample <- cbind(matrix(0, nrow=sum(out_of_sample), ncol=num_unique_levels), 1)
        Xtilde[in_sample,] <- Xtilde_insample
        Xtilde[out_of_sample,] <- Xtilde_out_of_sample
    } else {
        x_factor <- factor(x_input, levels = unique_levels)
        Xtilde <- cbind(unname(model.matrix(~0+x_factor)), 0)
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
#' @export
#'
#' @examples
#' x <- c("1. Strongly disagree", "3. Neither agree nor disagree", "2. Disagree", "4. Agree", "3. Neither agree nor disagree", "5. Strongly agree", "4. Agree")
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
#' @export
#'
#' @examples
#' x_levels <- c("1. Strongly disagree", "2. Disagree", "3. Neither agree nor disagree", 
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
            if (!is.null(var_name)) warning_message <- paste0("Variable ", var_name, " includes ordered categorical levels not included in the original training set")
            else warning_message <- paste0("Variable includes ordered categorical levels not included in the original training set")
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
            if (!is.null(var_name)) warning_message <- paste0("Variable ", var_name, " includes ordered categorical levels not included in the original training set")
            else warning_message <- paste0("Variable includes ordered categorical levels not included in the original training set")
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
