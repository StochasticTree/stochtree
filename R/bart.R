#' Run the BART algorithm for supervised learning. 
#'
#' @param X_train Covariates used to split trees in the ensemble. May be provided either as a dataframe or a matrix. 
#' Matrix covariates will be assumed to be all numeric. Covariates passed as a dataframe will be 
#' preprocessed based on the variable types (e.g. categorical columns stored as unordered factors will be one-hot encoded, 
#' categorical columns stored as ordered factors will passed as integers to the core algorithm, along with the metadata 
#' that the column is ordered categorical).
#' @param y_train Outcome to be modeled by the ensemble.
#' @param W_train (Optional) Bases used to define a regression model `y ~ W` in 
#' each leaf of each regression tree. By default, BART assumes constant leaf node 
#' parameters, implicitly regressing on a constant basis of ones (i.e. `y ~ 1`).
#' @param group_ids_train (Optional) Group labels used for an additive random effects model.
#' @param rfx_basis_train (Optional) Basis for "random-slope" regression in an additive random effects model.
#' If `group_ids_train` is provided with a regression basis, an intercept-only random effects model 
#' will be estimated.
#' @param X_test (Optional) Test set of covariates used to define "out of sample" evaluation data. 
#' May be provided either as a dataframe or a matrix, but the format of `X_test` must be consistent with 
#' that of `X_train`.
#' @param W_test (Optional) Test set of bases used to define "out of sample" evaluation data. 
#' While a test set is optional, the structure of any provided test set must match that 
#' of the training set (i.e. if both X_train and W_train are provided, then a test set must 
#' consist of X_test and W_test with the same number of columns).
#' @param group_ids_test (Optional) Test set group labels used for an additive random effects model. 
#' We do not currently support (but plan to in the near future), test set evaluation for group labels
#' that were not in the training set.
#' @param rfx_basis_test (Optional) Test set basis for "random-slope" regression in additive random effects model.
#' @param num_gfr Number of "warm-start" iterations run using the grow-from-root algorithm (He and Hahn, 2021). Default: 5.
#' @param num_burnin Number of "burn-in" iterations of the MCMC sampler. Default: 0.
#' @param num_mcmc Number of "retained" iterations of the MCMC sampler. Default: 100.
#' @param previous_model_json (Optional) JSON string containing a previous BART model. This can be used to "continue" a sampler interactively after inspecting the samples or to run parallel chains "warm-started" from existing forest samples. Default: `NULL`.
#' @param warmstart_sample_num (Optional) Sample number from `previous_model_json` that will be used to warmstart this BART sampler. One-indexed (so that the first sample is used for warm-start by setting `warmstart_sample_num = 1`). Default: `NULL`.
#' @param params The list of model parameters, each of which has a default value.
#'
#'   **1. Global Parameters**
#'
#'   - `cutpoint_grid_size` Maximum size of the "grid" of potential cutpoints to consider. Default: `100`.
#'   - `sigma2_init` Starting value of global error variance parameter. Calibrated internally as `pct_var_sigma2_init*var((y-mean(y))/sd(y))` if not set.
#'   - `pct_var_sigma2_init` Percentage of standardized outcome variance used to initialize global error variance parameter. Default: `1`. Superseded by `sigma2_init`.
#'   - `variance_scale` Variance after the data have been scaled. Default: `1`.
#'   - `a_global` Shape parameter in the `IG(a_global, b_global)` global error variance model. Default: `0`.
#'   - `b_global` Scale parameter in the `IG(a_global, b_global)` global error variance model. Default: `0`.
#'   - `random_seed` Integer parameterizing the C++ random number generator. If not specified, the C++ random number generator is seeded according to `std::random_device`.
#'   - `sample_sigma_global` Whether or not to update the `sigma^2` global error variance parameter based on `IG(a_global, b_global)`. Default: `TRUE`.
#'   - `keep_burnin` Whether or not "burnin" samples should be included in cached predictions. Default `FALSE`. Ignored if `num_mcmc = 0`.
#'   - `keep_gfr` Whether or not "grow-from-root" samples should be included in cached predictions. Default `FALSE`. Ignored if `num_mcmc = 0`.
#'   - `standardize` Whether or not to standardize the outcome (and store the offset / scale in the model object). Default: `TRUE`.
#'   - `keep_every` How many iterations of the burned-in MCMC sampler should be run before forests and parameters are retained. Default `1`. Setting `keep_every <- k` for some `k > 1` will "thin" the MCMC samples by retaining every `k`-th sample, rather than simply every sample. This can reduce the autocorrelation of the MCMC samples.
#'   - `num_chains` How many independent MCMC chains should be sampled. If `num_mcmc = 0`, this is ignored. If `num_gfr = 0`, then each chain is run from root for `num_mcmc * keep_every + num_burnin` iterations, with `num_mcmc` samples retained. If `num_gfr > 0`, each MCMC chain will be initialized from a separate GFR ensemble, with the requirement that `num_gfr >= num_chains`. Default: `1`.
#'   - `verbose` Whether or not to print progress during the sampling loops. Default: `FALSE`.
#'
#'   **2. Mean Forest Parameters**
#'   
#'   - `num_trees_mean` Number of trees in the ensemble for the conditional mean model. Default: `200`. If `num_trees_mean = 0`, the conditional mean will not be modeled using a forest, and the function will only proceed if `num_trees_variance > 0`.
#'   - `sample_sigma_leaf` Whether or not to update the `tau` leaf scale variance parameter based on `IG(a_leaf, b_leaf)`. Cannot (currently) be set to true if `ncol(W_train)>1`. Default: `FALSE`.
#'
#'   **2.1. Tree Prior Parameters**
#'   
#'   - `alpha_mean` Prior probability of splitting for a tree of depth 0 in the mean model. Tree split prior combines `alpha_mean` and `beta_mean` via `alpha_mean*(1+node_depth)^-beta_mean`. Default: `0.95`.
#'   - `beta_mean` Exponent that decreases split probabilities for nodes of depth > 0 in the mean model. Tree split prior combines `alpha_mean` and `beta_mean` via `alpha_mean*(1+node_depth)^-beta_mean`. Default: `2`.
#'   - `min_samples_leaf_mean` Minimum allowable size of a leaf, in terms of training samples, in the mean model. Default: `5`.
#'   - `max_depth_mean` Maximum depth of any tree in the ensemble in the mean model. Default: `10`. Can be overridden with ``-1`` which does not enforce any depth limits on trees.
#'
#'   **2.2. Leaf Model Parameters**
#'   
#'   - `variable_weights_mean` Numeric weights reflecting the relative probability of splitting on each variable in the mean forest. Does not need to sum to 1 but cannot be negative. Defaults to `rep(1/ncol(X_train), ncol(X_train))` if not set here.
#'   - `sigma_leaf_init` Starting value of leaf node scale parameter. Calibrated internally as `1/num_trees_mean` if not set here.
#'   - `a_leaf` Shape parameter in the `IG(a_leaf, b_leaf)` leaf node parameter variance model. Default: `3`.
#'   - `b_leaf` Scale parameter in the `IG(a_leaf, b_leaf)` leaf node parameter variance model. Calibrated internally as `0.5/num_trees_mean` if not set here.
#'
#'   **3. Conditional Variance Forest Parameters**
#'   
#'   - `num_trees_variance` Number of trees in the ensemble for the conditional variance model. Default: `0`. Variance is only modeled using a tree / forest if `num_trees_variance > 0`.
#'   - `variance_forest_init` Starting value of root forest prediction in conditional (heteroskedastic) error variance model. Calibrated internally as `log(pct_var_variance_forest_init*var((y-mean(y))/sd(y)))/num_trees_variance` if not set.
#'   - `pct_var_variance_forest_init` Percentage of standardized outcome variance used to initialize global error variance parameter. Default: `1`. Superseded by `variance_forest_init`.
#'
#'   **3.1. Tree Prior Parameters**
#'   
#'   - `alpha_variance` Prior probability of splitting for a tree of depth 0 in the variance model. Tree split prior combines `alpha_variance` and `beta_variance` via `alpha_variance*(1+node_depth)^-beta_variance`. Default: `0.95`.
#'   - `beta_variance` Exponent that decreases split probabilities for nodes of depth > 0 in the variance model. Tree split prior combines `alpha_variance` and `beta_variance` via `alpha_variance*(1+node_depth)^-beta_variance`. Default: `2`.
#'   - `min_samples_leaf_variance` Minimum allowable size of a leaf, in terms of training samples, in the variance model. Default: `5`.
#'   - `max_depth_variance` Maximum depth of any tree in the ensemble in the variance model. Default: `10`. Can be overridden with ``-1`` which does not enforce any depth limits on trees.
#'
#'   **3.2. Leaf Model Parameters**
#'   
#'   - `variable_weights_variance` Numeric weights reflecting the relative probability of splitting on each variable in the variance forest. Does not need to sum to 1 but cannot be negative. Defaults to `rep(1/ncol(X_train), ncol(X_train))` if not set here.
#'   - `sigma_leaf_init` Starting value of leaf node scale parameter. Calibrated internally as `1/num_trees_mean` if not set here.
#'   - `a_forest` Shape parameter in the `IG(a_forest, b_forest)` conditional error variance model (which is only sampled if `num_trees_variance > 0`). Calibrated internally as `num_trees_variance / 1.5^2 + 0.5` if not set.
#'   - `b_forest` Scale parameter in the `IG(a_forest, b_forest)` conditional error variance model (which is only sampled if `num_trees_variance > 0`). Calibrated internally as `num_trees_variance / 1.5^2` if not set.
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
#' bart_model <- bart(X_train = X_train, y_train = y_train, X_test = X_test)
#' # plot(rowMeans(bart_model$y_hat_test), y_test, xlab = "predicted", ylab = "actual")
#' # abline(0,1,col="red",lty=3,lwd=3)
bart <- function(X_train, y_train, W_train = NULL, group_ids_train = NULL, 
                 rfx_basis_train = NULL, X_test = NULL, W_test = NULL, 
                 group_ids_test = NULL, rfx_basis_test = NULL, 
                 num_gfr = 5, num_burnin = 0, num_mcmc = 100, 
                 previous_model_json = NULL, warmstart_sample_num = NULL, 
                 params = list()) {
    # Extract BART parameters
    bart_params <- preprocessBartParams(params)
    cutpoint_grid_size <- bart_params$cutpoint_grid_size
    sigma_leaf_init <- bart_params$sigma_leaf_init
    alpha_mean <- bart_params$alpha_mean
    beta_mean <- bart_params$beta_mean
    min_samples_leaf_mean <- bart_params$min_samples_leaf_mean
    max_depth_mean <- bart_params$max_depth_mean
    alpha_variance <- bart_params$alpha_variance
    beta_variance <- bart_params$beta_variance
    min_samples_leaf_variance <- bart_params$min_samples_leaf_variance
    max_depth_variance <- bart_params$max_depth_variance
    a_global <- bart_params$a_global
    b_global <- bart_params$b_global
    a_leaf <- bart_params$a_leaf
    b_leaf <- bart_params$b_leaf
    a_forest <- bart_params$a_forest
    b_forest <- bart_params$b_forest
    variance_scale <- bart_params$variance_scale
    sigma2_init <- bart_params$sigma2_init
    variance_forest_init <- bart_params$variance_forest_init
    pct_var_sigma2_init <- bart_params$pct_var_sigma2_init
    pct_var_variance_forest_init <- bart_params$pct_var_variance_forest_init
    variable_weights_mean <- bart_params$variable_weights_mean
    variable_weights_variance <- bart_params$variable_weights_variance
    num_trees_mean <- bart_params$num_trees_mean
    num_trees_variance <- bart_params$num_trees_variance
    sample_sigma_global <- bart_params$sample_sigma_global
    sample_sigma_leaf <- bart_params$sample_sigma_leaf
    random_seed <- bart_params$random_seed
    keep_burnin <- bart_params$keep_burnin
    keep_gfr <- bart_params$keep_gfr
    standardize <- bart_params$standardize
    keep_every <- bart_params$keep_every
    num_chains <- bart_params$num_chains
    verbose <- bart_params$verbose
    
    # Check if there are enough GFR samples to seed num_chains samplers
    if (num_gfr > 0) {
        if (num_chains > num_gfr) {
            stop("num_chains > num_gfr, meaning we do not have enough GFR samples to seed num_chains distinct MCMC chains")
        }
    }
    
    # Override keep_gfr if there are no MCMC samples
    if (num_mcmc == 0) keep_gfr <- T
    
    # Check if previous model JSON is provided and parse it if so
    # TODO: check that warmstart_sample_num is <= the number of samples in this previous model
    has_prev_model <- !is.null(previous_model_json)
    if (has_prev_model) {
        previous_bart_model <- createBARTModelFromJsonString(previous_model_json)
        previous_y_bar <- previous_bart_model$model_params$outcome_mean
        previous_y_scale <- previous_bart_model$model_params$outcome_scale
        previous_var_scale <- previous_bart_model$model_params$variance_scale
        if (previous_bart_model$model_params$include_mean_forest) {
            previous_forest_samples_mean <- previous_bart_model$mean_forests
        } else previous_forest_samples_mean <- NULL
        if (previous_bart_model$model_params$include_mean_forest) {
            previous_forest_samples_variance <- previous_bart_model$variance_forests
        } else previous_forest_samples_variance <- NULL
        if (previous_bart_model$model_params$sample_sigma_global) {
            previous_global_var_samples <- previous_bart_model$sigma2_global_samples*(
                previous_var_scale / (previous_y_scale*previous_y_scale)
            )
        } else previous_global_var_samples <- NULL
        if (previous_bart_model$model_params$sample_sigma_leaf) {
            previous_leaf_var_samples <- previous_bart_model$sigma2_leaf_samples
        } else previous_leaf_var_samples <- NULL
        if (previous_bart_model$model_params$has_rfx) {
            previous_rfx_samples <- previous_bart_model$rfx_samples
        } else previous_rfx_samples <- NULL
    } else {
        previous_y_bar <- NULL
        previous_y_scale <- NULL
        previous_var_scale <- NULL
        previous_global_var_samples <- NULL
        previous_leaf_var_samples <- NULL
        previous_rfx_samples <- NULL
        previous_forest_samples_mean <- NULL
        previous_forest_samples_variance <- NULL
    }
    
    # Determine whether conditional mean, variance, or both will be modeled
    if (num_trees_variance > 0) include_variance_forest = T
    else include_variance_forest = F
    if (num_trees_mean > 0) include_mean_forest = T
    else include_mean_forest = F
    
    # Set the variance forest priors if not set
    if (include_variance_forest) {
        a_0 <- 1.5
        if (is.null(a_forest)) a_forest <- num_trees_variance / (a_0^2) + 0.5
        if (is.null(b_forest)) b_forest <- num_trees_variance / (a_0^2)
    } else {
        a_forest <- 1.
        b_forest <- 1.
    }
    
    # Override tau sampling if there is no mean forest
    if (!include_mean_forest) sample_sigma_leaf <- F
    
    # Variable weight preprocessing (and initialization if necessary)
    if (include_mean_forest) {
        if (is.null(variable_weights_mean)) {
            variable_weights_mean = rep(1/ncol(X_train), ncol(X_train))
        }
        if (any(variable_weights_mean < 0)) {
            stop("variable_weights_mean cannot have any negative weights")
        }
    }
    if (include_variance_forest) {
        if (is.null(variable_weights_variance)) {
            variable_weights_variance = rep(1/ncol(X_train), ncol(X_train))
        }
        if (any(variable_weights_variance < 0)) {
            stop("variable_weights_variance cannot have any negative weights")
        }
    }
    
    # Preprocess covariates
    if ((!is.data.frame(X_train)) && (!is.matrix(X_train))) {
        stop("X_train must be a matrix or dataframe")
    }
    if (!is.null(X_test)){
        if ((!is.data.frame(X_test)) && (!is.matrix(X_test))) {
            stop("X_test must be a matrix or dataframe")
        }
    }
    if ((ncol(X_train) != length(variable_weights_mean)) && (include_mean_forest)) {
        stop("length(variable_weights_mean) must equal ncol(X_train)")
    }
    if ((ncol(X_train) != length(variable_weights_variance)) && (include_variance_forest)) {
        stop("length(variable_weights_variance) must equal ncol(X_train)")
    }
    train_cov_preprocess_list <- preprocessTrainData(X_train)
    X_train_metadata <- train_cov_preprocess_list$metadata
    X_train <- train_cov_preprocess_list$data
    original_var_indices <- X_train_metadata$original_var_indices
    feature_types <- X_train_metadata$feature_types
    if (!is.null(X_test)) X_test <- preprocessPredictionData(X_test, X_train_metadata)
    
    # Update variable weights
    variable_weights_adj <- 1/sapply(original_var_indices, function(x) sum(original_var_indices == x))
    if (include_mean_forest) {
        variable_weights_mean <- variable_weights_mean[original_var_indices]*variable_weights_adj
    }
    if (include_variance_forest) {
        variable_weights_variance <- variable_weights_variance[original_var_indices]*variable_weights_adj
    }
    
    # Convert all input data to matrices if not already converted
    if ((is.null(dim(W_train))) && (!is.null(W_train))) {
        W_train <- as.matrix(W_train)
    }
    if ((is.null(dim(W_test))) && (!is.null(W_test))) {
        W_test <- as.matrix(W_test)
    }
    if ((is.null(dim(rfx_basis_train))) && (!is.null(rfx_basis_train))) {
        rfx_basis_train <- as.matrix(rfx_basis_train)
    }
    if ((is.null(dim(rfx_basis_test))) && (!is.null(rfx_basis_test))) {
        rfx_basis_test <- as.matrix(rfx_basis_test)
    }
    
    # Recode group IDs to integer vector (if passed as, for example, a vector of county names, etc...)
    has_rfx <- F
    has_rfx_test <- F
    if (!is.null(group_ids_train)) {
        group_ids_factor <- factor(group_ids_train)
        group_ids_train <- as.integer(group_ids_factor)
        has_rfx <- T
        if (!is.null(group_ids_test)) {
            group_ids_factor_test <- factor(group_ids_test, levels = levels(group_ids_factor))
            if (sum(is.na(group_ids_factor_test)) > 0) {
                stop("All random effect group labels provided in group_ids_test must be present in group_ids_train")
            }
            group_ids_test <- as.integer(group_ids_factor_test)
            has_rfx_test <- T
        }
    }
    
    # Data consistency checks
    if ((!is.null(X_test)) && (ncol(X_test) != ncol(X_train))) {
        stop("X_train and X_test must have the same number of columns")
    }
    if ((!is.null(W_test)) && (ncol(W_test) != ncol(W_train))) {
        stop("W_train and W_test must have the same number of columns")
    }
    if ((!is.null(W_train)) && (nrow(W_train) != nrow(X_train))) {
        stop("W_train and X_train must have the same number of rows")
    }
    if ((!is.null(W_test)) && (nrow(W_test) != nrow(X_test))) {
        stop("W_test and X_test must have the same number of rows")
    }
    if (nrow(X_train) != length(y_train)) {
        stop("X_train and y_train must have the same number of observations")
    }
    if ((!is.null(rfx_basis_test)) && (ncol(rfx_basis_test) != ncol(rfx_basis_train))) {
        stop("rfx_basis_train and rfx_basis_test must have the same number of columns")
    }
    if (!is.null(group_ids_train)) {
        if (!is.null(group_ids_test)) {
            if ((!is.null(rfx_basis_train)) && (is.null(rfx_basis_test))) {
                stop("rfx_basis_train is provided but rfx_basis_test is not provided")
            }
        }
    }
    
    # Fill in rfx basis as a vector of 1s (random intercept) if a basis not provided 
    has_basis_rfx <- F
    num_basis_rfx <- 0
    if (has_rfx) {
        if (is.null(rfx_basis_train)) {
            rfx_basis_train <- matrix(rep(1,nrow(X_train)), nrow = nrow(X_train), ncol = 1)
        } else {
            has_basis_rfx <- T
            num_basis_rfx <- ncol(rfx_basis_train)
        }
        num_rfx_groups <- length(unique(group_ids_train))
        num_rfx_components <- ncol(rfx_basis_train)
        if (num_rfx_groups == 1) warning("Only one group was provided for random effect sampling, so the 'redundant parameterization' is likely overkill")
    }
    if (has_rfx_test) {
        if (is.null(rfx_basis_test)) {
            if (!is.null(rfx_basis_train)) {
                stop("Random effects basis provided for training set, must also be provided for the test set")
            }
            rfx_basis_test <- matrix(rep(1,nrow(X_test)), nrow = nrow(X_test), ncol = 1)
        }
    }

    # Convert y_train to numeric vector if not already converted
    if (!is.null(dim(y_train))) {
        y_train <- as.matrix(y_train)
    }
    
    # Determine whether a basis vector is provided
    has_basis = !is.null(W_train)
    
    # Determine whether a test set is provided
    has_test = !is.null(X_test)

    # Standardize outcome separately for test and train
    if (standardize) {
        y_bar_train <- mean(y_train)
        y_std_train <- sd(y_train)
    } else {
        y_bar_train <- 0
        y_std_train <- 1
    }
    resid_train <- (y_train-y_bar_train)/y_std_train
    resid_train <- resid_train*sqrt(variance_scale)
    
    # Compute initial value of root nodes in mean forest
    init_val_mean <- mean(resid_train)

    # Calibrate priors for sigma^2 and tau
    if (is.null(sigma2_init)) sigma2_init <- pct_var_sigma2_init*var(resid_train)
    if (is.null(variance_forest_init)) variance_forest_init <- pct_var_variance_forest_init*var(resid_train)
    if (is.null(b_leaf)) b_leaf <- var(resid_train)/(2*num_trees_mean)
    if (has_basis) {
        if (ncol(W_train) > 1) {
            if (is.null(sigma_leaf_init)) sigma_leaf_init <- diag(var(resid_train)/(num_trees_mean), ncol(W_train))
            current_leaf_scale <- sigma_leaf_init
        } else {
            if (is.null(sigma_leaf_init)) sigma_leaf_init <- var(resid_train)/(num_trees_mean)
            current_leaf_scale <- as.matrix(sigma_leaf_init)
        }
    } else {
        if (is.null(sigma_leaf_init)) sigma_leaf_init <- var(resid_train)/(num_trees_mean)
        current_leaf_scale <- as.matrix(sigma_leaf_init)
    }
    current_sigma2 <- sigma2_init

    # Determine leaf model type
    if (!has_basis) leaf_model_mean_forest <- 0
    else if (ncol(W_train) == 1) leaf_model_mean_forest <- 1
    else if (ncol(W_train) > 1) leaf_model_mean_forest <- 2
    else stop("W_train passed must be a matrix with at least 1 column")

    # Set variance leaf model type (currently only one option)
    leaf_model_variance_forest <- 3
    
    # Unpack model type info
    if (leaf_model_mean_forest == 0) {
        output_dimension = 1
        is_leaf_constant = T
        leaf_regression = F
    } else if (leaf_model_mean_forest == 1) {
        stopifnot(has_basis)
        stopifnot(ncol(W_train) == 1)
        output_dimension = 1
        is_leaf_constant = F
        leaf_regression = T
    } else if (leaf_model_mean_forest == 2) {
        stopifnot(has_basis)
        stopifnot(ncol(W_train) > 1)
        output_dimension = ncol(W_train)
        is_leaf_constant = F
        leaf_regression = T
        if (sample_sigma_leaf) {
            stop("Sampling leaf scale not yet supported for multivariate leaf models")
        }
    }
    
    # Data
    if (leaf_regression) {
        forest_dataset_train <- createForestDataset(X_train, W_train)
        if (has_test) forest_dataset_test <- createForestDataset(X_test, W_test)
        requires_basis <- T
    } else {
        forest_dataset_train <- createForestDataset(X_train)
        if (has_test) forest_dataset_test <- createForestDataset(X_test)
        requires_basis <- F
    }
    outcome_train <- createOutcome(resid_train)
    
    # Random number generator (std::mt19937)
    if (is.null(random_seed)) random_seed = sample(1:10000,1,F)
    rng <- createRNG(random_seed)
    
    # Sampling data structures
    feature_types <- as.integer(feature_types)
    if (include_mean_forest) {
        forest_model_mean <- createForestModel(forest_dataset_train, feature_types, num_trees_mean, nrow(X_train), alpha_mean, beta_mean, min_samples_leaf_mean, max_depth_mean)
    }
    if (include_variance_forest) {
        forest_model_variance <- createForestModel(forest_dataset_train, feature_types, num_trees_variance, nrow(X_train), alpha_variance, beta_variance, min_samples_leaf_variance, max_depth_variance)
    }
    
    # Container of forest samples
    if (include_mean_forest) {
        forest_samples_mean <- createForestContainer(num_trees_mean, output_dimension, is_leaf_constant, FALSE)
        active_forest_mean <- createForest(num_trees_mean, output_dimension, is_leaf_constant, FALSE)
    }
    if (include_variance_forest) {
        forest_samples_variance <- createForestContainer(num_trees_variance, 1, TRUE, TRUE)
        active_forest_variance <- createForest(num_trees_variance, 1, TRUE, TRUE)
    }
    
    # Random effects initialization 
    if (has_rfx) {
        # Prior parameters
        if (num_rfx_components == 1) {
            alpha_init <- c(1)
        } else if (num_rfx_components > 1) {
            alpha_init <- c(1,rep(0,num_rfx_components-1))
        } else {
            stop("There must be at least 1 random effect component")
        }
        xi_init <- matrix(rep(alpha_init, num_rfx_groups),num_rfx_components,num_rfx_groups)
        sigma_alpha_init <- diag(1,num_rfx_components,num_rfx_components)
        sigma_xi_init <- diag(1,num_rfx_components,num_rfx_components)
        sigma_xi_shape <- 1
        sigma_xi_scale <- 1
        
        # Random effects data structure and storage container
        rfx_dataset_train <- createRandomEffectsDataset(group_ids_train, rfx_basis_train)
        rfx_tracker_train <- createRandomEffectsTracker(group_ids_train)
        rfx_model <- createRandomEffectsModel(num_rfx_components, num_rfx_groups)
        rfx_model$set_working_parameter(alpha_init)
        rfx_model$set_group_parameters(xi_init)
        rfx_model$set_working_parameter_cov(sigma_alpha_init)
        rfx_model$set_group_parameter_cov(sigma_xi_init)
        rfx_model$set_variance_prior_shape(sigma_xi_shape)
        rfx_model$set_variance_prior_scale(sigma_xi_scale)
        rfx_samples <- createRandomEffectSamples(num_rfx_components, num_rfx_groups, rfx_tracker_train)
    }

    # Container of variance parameter samples
    num_actual_mcmc_iter <- num_mcmc * keep_every
    num_samples <- num_gfr + num_burnin + num_actual_mcmc_iter
    # Delete GFR samples from these containers after the fact if desired
    # num_retained_samples <- ifelse(keep_gfr, num_gfr, 0) + ifelse(keep_burnin, num_burnin, 0) + num_mcmc
    num_retained_samples <- num_gfr + ifelse(keep_burnin, num_burnin, 0) + num_mcmc * num_chains
    if (sample_sigma_global) global_var_samples <- rep(NA, num_retained_samples)
    if (sample_sigma_leaf) leaf_scale_samples <- rep(NA, num_retained_samples)
    sample_counter <- 0
    
    # Initialize the leaves of each tree in the mean forest
    if (include_mean_forest) {
        if (requires_basis) init_values_mean_forest <- rep(0., ncol(W_train))
        else init_values_mean_forest <- 0.
        active_forest_mean$prepare_for_sampler(forest_dataset_train, outcome_train, forest_model_mean, leaf_model_mean_forest, init_values_mean_forest)
    }

    # Initialize the leaves of each tree in the variance forest
    if (include_variance_forest) {
        active_forest_variance$prepare_for_sampler(forest_dataset_train, outcome_train, forest_model_variance, leaf_model_variance_forest, variance_forest_init)
    }
    
    # Run GFR (warm start) if specified
    if (num_gfr > 0){
        for (i in 1:num_gfr) {
            # Keep all GFR samples at this stage -- remove from ForestSamples after MCMC
            # keep_sample <- ifelse(keep_gfr, T, F)
            keep_sample <- T
            if (keep_sample) sample_counter <- sample_counter + 1
            # Print progress
            if (verbose) {
                if ((i %% 10 == 0) || (i == num_gfr)) {
                    cat("Sampling", i, "out of", num_gfr, "XBART (grow-from-root) draws\n")
                }
            }
            
            if (include_mean_forest) {
                forest_model_mean$sample_one_iteration(
                    forest_dataset_train, outcome_train, forest_samples_mean, active_forest_mean, 
                    rng, feature_types, leaf_model_mean_forest, current_leaf_scale, variable_weights_mean, 
                    a_forest, b_forest, current_sigma2, cutpoint_grid_size, keep_forest = keep_sample, gfr = T, pre_initialized = T
                )
            }
            if (include_variance_forest) {
                forest_model_variance$sample_one_iteration(
                    forest_dataset_train, outcome_train, forest_samples_variance, active_forest_variance, 
                    rng, feature_types, leaf_model_variance_forest, current_leaf_scale, variable_weights_variance, 
                    a_forest, b_forest, current_sigma2, cutpoint_grid_size, keep_forest = keep_sample, gfr = T, pre_initialized = T
                )
            }
            if (sample_sigma_global) {
                current_sigma2 <- sample_sigma2_one_iteration(outcome_train, forest_dataset_train, rng, a_global, b_global)
                if (keep_sample) global_var_samples[sample_counter] <- current_sigma2
            }
            if (sample_sigma_leaf) {
                leaf_scale_double <- sample_tau_one_iteration(active_forest_mean, rng, a_leaf, b_leaf)
                current_leaf_scale <- as.matrix(leaf_scale_double)
                if (keep_sample) leaf_scale_samples[sample_counter] <- leaf_scale_double
            }
            if (has_rfx) {
                rfx_model$sample_random_effect(rfx_dataset_train, outcome_train, rfx_tracker_train, rfx_samples, keep_sample, current_sigma2, rng)
            }
        }
    }
    
    # Run MCMC
    if (num_burnin + num_mcmc > 0) {
        for (chain_num in 1:num_chains) {
            if (num_gfr > 0) {
                # Reset state of active_forest and forest_model based on a previous GFR sample
                forest_ind <- num_gfr - chain_num
                if (include_mean_forest) {
                    resetActiveForest(active_forest_mean, forest_samples_mean, forest_ind)
                    resetForestModel(forest_model_mean, active_forest_mean, forest_dataset_train, outcome_train, TRUE)
                    if (sample_sigma_leaf) {
                        leaf_scale_double <- leaf_scale_samples[forest_ind + 1]
                        current_leaf_scale <- as.matrix(leaf_scale_double)
                    }
                }
                if (include_variance_forest) {
                    resetActiveForest(active_forest_variance, forest_samples_variance, forest_ind)
                    resetForestModel(forest_model_variance, active_forest_variance, forest_dataset_train, outcome_train, FALSE)
                }
                if (has_rfx) {
                    resetRandomEffectsModel(rfx_model, rfx_samples, forest_ind, sigma_alpha_init)
                    resetRandomEffectsTracker(rfx_tracker_train, rfx_model, rfx_dataset_train, outcome_train, rfx_samples)
                }
                if (sample_sigma_global) current_sigma2 <- global_var_samples[forest_ind + 1]
            } else if (has_prev_model) {
                if (include_mean_forest) {
                    resetActiveForest(active_forest_mean, previous_forest_samples_mean, warmstart_sample_num - 1)
                    resetForestModel(forest_model_mean, active_forest_mean, forest_dataset_train, outcome_train, TRUE)
                    if (sample_sigma_leaf && (!is.null(previous_leaf_var_samples))) {
                        leaf_scale_double <- previous_leaf_var_samples[warmstart_sample_num]
                        current_leaf_scale <- as.matrix(leaf_scale_double)
                    }
                }
                if (include_variance_forest) {
                    resetActiveForest(active_forest_variance, previous_forest_samples_variance, warmstart_sample_num - 1)
                    resetForestModel(forest_model_variance, active_forest_variance, forest_dataset_train, outcome_train, FALSE)
                }
                # TODO: also initialize from previous RFX samples
                # if (has_rfx) {
                #     rootResetRandomEffectsModel(rfx_model, alpha_init, xi_init, sigma_alpha_init,
                #                                 sigma_xi_init, sigma_xi_shape, sigma_xi_scale)
                #     rootResetRandomEffectsTracker(rfx_tracker_train, rfx_model, rfx_dataset_train, outcome_train)
                # }
                if (sample_sigma_global) {
                    if (!is.null(previous_global_var_samples)) {
                        current_sigma2 <- previous_global_var_samples[warmstart_sample_num]
                    }
                }
            } else {
                if (include_mean_forest) {
                    rootResetActiveForest(active_forest_mean)
                    active_forest_mean$set_root_leaves(init_values_mean_forest / num_trees_mean)
                    resetForestModel(forest_model_mean, active_forest_mean, forest_dataset_train, outcome_train, TRUE)
                    if (sample_sigma_leaf) {
                        current_leaf_scale <- as.matrix(sigma_leaf_init)
                    }
                }
                if (include_variance_forest) {
                    rootResetActiveForest(active_forest_variance)
                    active_forest_variance$set_root_leaves(log(variance_forest_init) / num_trees_variance)
                    resetForestModel(forest_model_variance, active_forest_variance, forest_dataset_train, outcome_train, FALSE)
                }
                if (has_rfx) {
                    rootResetRandomEffectsModel(rfx_model, alpha_init, xi_init, sigma_alpha_init,
                                                sigma_xi_init, sigma_xi_shape, sigma_xi_scale)
                    rootResetRandomEffectsTracker(rfx_tracker_train, rfx_model, rfx_dataset_train, outcome_train)
                }
                if (sample_sigma_global) current_sigma2 <- sigma2_init
            }
            for (i in (num_gfr+1):num_samples) {
                is_mcmc <- i > (num_gfr + num_burnin)
                if (is_mcmc) {
                    mcmc_counter <- i - (num_gfr + num_burnin)
                    if (mcmc_counter %% keep_every == 0) keep_sample <- T
                    else keep_sample <- F
                } else {
                    if (keep_burnin) keep_sample <- T
                    else keep_sample <- F
                }
                if (keep_sample) sample_counter <- sample_counter + 1
                # Print progress
                if (verbose) {
                    if (num_burnin > 0) {
                        if (((i - num_gfr) %% 100 == 0) || ((i - num_gfr) == num_burnin)) {
                            cat("Sampling", i - num_gfr, "out of", num_burnin, "BART burn-in draws; Chain number ", chain_num, "\n")
                        }
                    }
                    if (num_mcmc > 0) {
                        if (((i - num_gfr - num_burnin) %% 100 == 0) || (i == num_samples)) {
                            cat("Sampling", i - num_burnin - num_gfr, "out of", num_mcmc, "BART MCMC draws; Chain number ", chain_num, "\n")
                        }
                    }
                }
                
                if (include_mean_forest) {
                    forest_model_mean$sample_one_iteration(
                        forest_dataset_train, outcome_train, forest_samples_mean, active_forest_mean, 
                        rng, feature_types, leaf_model_mean_forest, current_leaf_scale, variable_weights_mean, 
                        a_forest, b_forest, current_sigma2, cutpoint_grid_size, keep_forest = keep_sample, gfr = F, pre_initialized = T
                    )
                }
                if (include_variance_forest) {
                    forest_model_variance$sample_one_iteration(
                        forest_dataset_train, outcome_train, forest_samples_variance, active_forest_variance, 
                        rng, feature_types, leaf_model_variance_forest, current_leaf_scale, variable_weights_variance, 
                        a_forest, b_forest, current_sigma2, cutpoint_grid_size, keep_forest = keep_sample, gfr = F, pre_initialized = T
                    )
                }
                if (sample_sigma_global) {
                    current_sigma2 <- sample_sigma2_one_iteration(outcome_train, forest_dataset_train, rng, a_global, b_global)
                    if (keep_sample) global_var_samples[sample_counter] <- current_sigma2
                }
                if (sample_sigma_leaf) {
                    leaf_scale_double <- sample_tau_one_iteration(active_forest_mean, rng, a_leaf, b_leaf)
                    current_leaf_scale <- as.matrix(leaf_scale_double)
                    if (keep_sample) leaf_scale_samples[sample_counter] <- leaf_scale_double
                }
                if (has_rfx) {
                    rfx_model$sample_random_effect(rfx_dataset_train, outcome_train, rfx_tracker_train, rfx_samples, keep_sample, current_sigma2, rng)
                }
            }
        }
    }
    
    # Remove GFR samples if they are not to be retained
    if ((!keep_gfr) && (num_gfr > 0)) {
        for (i in 1:num_gfr) {
            if (include_mean_forest) {
                forest_samples_mean$delete_sample(i-1)
            }
            if (include_variance_forest) {
                forest_samples_variance$delete_sample(i-1)
            }
            if (has_rfx) {
                rfx_samples$delete_sample(i-1)
            }
        }
        if (sample_sigma_global) {
            global_var_samples <- global_var_samples[(num_gfr+1):length(global_var_samples)]
        }
        if (sample_sigma_leaf) {
            leaf_scale_samples <- leaf_scale_samples[(num_gfr+1):length(leaf_scale_samples)]
        }
        num_retained_samples <- num_retained_samples - num_gfr
    }

    # Mean forest predictions
    if (include_mean_forest) {
        y_hat_train <- forest_samples_mean$predict(forest_dataset_train)*y_std_train/sqrt(variance_scale) + y_bar_train
        if (has_test) y_hat_test <- forest_samples_mean$predict(forest_dataset_test)*y_std_train/sqrt(variance_scale) + y_bar_train
    }
    
    # Variance forest predictions
    if (include_variance_forest) {
        sigma_x_hat_train <- forest_samples_variance$predict(forest_dataset_train)
        if (has_test) sigma_x_hat_test <- forest_samples_variance$predict(forest_dataset_test)
    }
    
    # Random effects predictions
    if (has_rfx) {
        rfx_preds_train <- rfx_samples$predict(group_ids_train, rfx_basis_train)*y_std_train/sqrt(variance_scale)
        y_hat_train <- y_hat_train + rfx_preds_train
    }
    if ((has_rfx_test) && (has_test)) {
        rfx_preds_test <- rfx_samples$predict(group_ids_test, rfx_basis_test)*y_std_train/sqrt(variance_scale)
        y_hat_test <- y_hat_test + rfx_preds_test
    }

    # Global error variance
    if (sample_sigma_global) sigma2_samples <- global_var_samples*(y_std_train^2)/variance_scale
    
    # Leaf parameter variance
    if (sample_sigma_leaf) tau_samples <- leaf_scale_samples
    
    # Rescale variance forest prediction by global sigma2 (sampled or constant)
    if (include_variance_forest) {
        if (sample_sigma_global) {
            sigma_x_hat_train <- sapply(1:num_retained_samples, function(i) sqrt(sigma_x_hat_train[,i]*sigma2_samples[i]))
            if (has_test) sigma_x_hat_test <- sapply(1:num_retained_samples, function(i) sqrt(sigma_x_hat_test[,i]*sigma2_samples[i]))
        } else {
            sigma_x_hat_train <- sqrt(sigma_x_hat_train*sigma2_init)*y_std_train/sqrt(variance_scale)
            if (has_test) sigma_x_hat_test <- sqrt(sigma_x_hat_test*sigma2_init)*y_std_train/sqrt(variance_scale)
        }
    }
    
    # Return results as a list
    # TODO: store variance_scale and propagate through predict function
    # TODO: refactor out the "num_retained_samples" variable now that we burn-in/thin correctly
    model_params <- list(
        "sigma2_init" = sigma2_init, 
        "sigma_leaf_init" = sigma_leaf_init,
        "a_global" = a_global,
        "b_global" = b_global, 
        "a_leaf" = a_leaf, 
        "b_leaf" = b_leaf,
        "a_forest" = a_forest, 
        "b_forest" = b_forest,
        "outcome_mean" = y_bar_train,
        "outcome_scale" = y_std_train, 
        "standardize" = standardize, 
        "output_dimension" = output_dimension,
        "is_leaf_constant" = is_leaf_constant,
        "leaf_regression" = leaf_regression,
        "requires_basis" = requires_basis, 
        "num_covariates" = ncol(X_train), 
        "num_basis" = ifelse(is.null(W_train),0,ncol(W_train)), 
        "num_samples" = num_retained_samples, 
        "num_gfr" = num_gfr, 
        "num_burnin" = num_burnin, 
        "num_mcmc" = num_mcmc, 
        "keep_every" = keep_every,
        "num_chains" = num_chains,
        "has_basis" = !is.null(W_train), 
        "has_rfx" = has_rfx, 
        "has_rfx_basis" = has_basis_rfx, 
        "num_rfx_basis" = num_basis_rfx, 
        "sample_sigma_global" = sample_sigma_global,
        "sample_sigma_leaf" = sample_sigma_leaf,
        "include_mean_forest" = include_mean_forest,
        "include_variance_forest" = include_variance_forest,
        "variance_scale" = variance_scale
    )
    result <- list(
        "model_params" = model_params, 
        "train_set_metadata" = X_train_metadata
    )
    if (include_mean_forest) {
        result[["mean_forests"]] = forest_samples_mean
        result[["y_hat_train"]] = y_hat_train
        if (has_test) result[["y_hat_test"]] = y_hat_test
    }
    if (include_variance_forest) {
        result[["variance_forests"]] = forest_samples_variance
        result[["sigma_x_hat_train"]] = sigma_x_hat_train
        if (has_test) result[["sigma_x_hat_test"]] = sigma_x_hat_test
    }
    if (sample_sigma_global) result[["sigma2_global_samples"]] = sigma2_samples
    if (sample_sigma_leaf) result[["sigma2_leaf_samples"]] = tau_samples
    if (has_rfx) {
        result[["rfx_samples"]] = rfx_samples
        result[["rfx_preds_train"]] = rfx_preds_train
        result[["rfx_unique_group_ids"]] = levels(group_ids_factor)
    }
    if ((has_rfx_test) && (has_test)) result[["rfx_preds_test"]] = rfx_preds_test
    class(result) <- "bartmodel"
    
    # Clean up classes with external pointers to C++ data structures
    if (include_mean_forest) rm(forest_model_mean)
    if (include_variance_forest) rm(forest_model_variance)
    rm(forest_dataset_train)
    if (has_test) rm(forest_dataset_test)
    if (has_rfx) rm(rfx_dataset_train, rfx_tracker_train, rfx_model)
    rm(outcome_train)
    rm(rng)
    
    return(result)
}

#' Predict from a sampled BART model on new data
#'
#' @param bart Object of type `bart` containing draws of a regression forest and associated sampling outputs.
#' @param X_test Covariates used to determine tree leaf predictions for each observation. Must be passed as a matrix or dataframe.
#' @param W_test (Optional) Bases used for prediction (by e.g. dot product with leaf values). Default: `NULL`.
#' @param group_ids_test (Optional) Test set group labels used for an additive random effects model. 
#' We do not currently support (but plan to in the near future), test set evaluation for group labels
#' that were not in the training set.
#' @param rfx_basis_test (Optional) Test set basis for "random-slope" regression in additive random effects model.
#'
#' @return List of prediction matrices. If model does not have random effects, the list has one element -- the predictions from the forest. 
#' If the model does have random effects, the list has three elements -- forest predictions, random effects predictions, and their sum (`y_hat`).
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
#' bart_model <- bart(X_train = X_train, y_train = y_train)
#' y_hat_test <- predict(bart_model, X_test)
#' # plot(rowMeans(y_hat_test), y_test, xlab = "predicted", ylab = "actual")
#' # abline(0,1,col="red",lty=3,lwd=3)
predict.bartmodel <- function(bart, X_test, W_test = NULL, group_ids_test = NULL, rfx_basis_test = NULL){
    # Preprocess covariates
    if ((!is.data.frame(X_test)) && (!is.matrix(X_test))) {
        stop("X_test must be a matrix or dataframe")
    }
    train_set_metadata <- bart$train_set_metadata
    X_test <- preprocessPredictionData(X_test, train_set_metadata)
    
    # Convert all input data to matrices if not already converted
    if ((is.null(dim(W_test))) && (!is.null(W_test))) {
        W_test <- as.matrix(W_test)
    }
    if ((is.null(dim(rfx_basis_test))) && (!is.null(rfx_basis_test))) {
        rfx_basis_test <- as.matrix(rfx_basis_test)
    }
    
    # Data checks
    if ((bart$model_params$requires_basis) && (is.null(W_test))) {
        stop("Basis (W_test) must be provided for this model")
    }
    if ((!is.null(W_test)) && (nrow(X_test) != nrow(W_test))) {
        stop("X_test and W_test must have the same number of rows")
    }
    if (bart$model_params$num_covariates != ncol(X_test)) {
        stop("X_test and W_test must have the same number of rows")
    }
    if ((bart$model_params$has_rfx) && (is.null(group_ids_test))) {
        stop("Random effect group labels (group_ids_test) must be provided for this model")
    }
    if ((bart$model_params$has_rfx_basis) && (is.null(rfx_basis_test))) {
        stop("Random effects basis (rfx_basis_test) must be provided for this model")
    }
    if ((bart$model_params$num_rfx_basis > 0) && (ncol(rfx_basis_test) != bart$model_params$num_rfx_basis)) {
        stop("Random effects basis has a different dimension than the basis used to train this model")
    }
    
    # Recode group IDs to integer vector (if passed as, for example, a vector of county names, etc...)
    has_rfx <- F
    if (!is.null(group_ids_test)) {
        rfx_unique_group_ids <- bcf$rfx_unique_group_ids
        group_ids_factor_test <- factor(group_ids_test, levels = rfx_unique_group_ids)
        if (sum(is.na(group_ids_factor_test)) > 0) {
            stop("All random effect group labels provided in group_ids_test must be present in group_ids_train")
        }
        group_ids_test <- as.integer(group_ids_factor_test)
        has_rfx <- T
    }
    
    # Produce basis for the "intercept-only" random effects case
    if ((bart$model_params$has_rfx) && (is.null(rfx_basis_test))) {
        rfx_basis_test <- matrix(rep(1, nrow(X_test)), ncol = 1)
    }
    
    # Create prediction dataset
    if (!is.null(W_test)) prediction_dataset <- createForestDataset(X_test, W_test)
    else prediction_dataset <- createForestDataset(X_test)
    
    # Compute mean forest predictions
    num_samples <- bart$model_params$num_samples
    variance_scale <- bart$model_params$variance_scale
    y_std <- bart$model_params$outcome_scale
    y_bar <- bart$model_params$outcome_mean
    sigma2_init <- bart$model_params$sigma2_init
    if (bart$model_params$include_mean_forest) {
        mean_forest_predictions <- bart$mean_forests$predict(prediction_dataset)*y_std/sqrt(variance_scale) + y_bar
    }
    
    # Compute variance forest predictions
    if (bart$model_params$include_variance_forest) {
        s_x_raw <- bart$variance_forests$predict(prediction_dataset)
    }
    
    # Compute rfx predictions (if needed)
    if (bart$model_params$has_rfx) {
        rfx_predictions <- bart$rfx_samples$predict(group_ids_test, rfx_basis_test)*y_std/sqrt(variance_scale)
    }
    
    # Scale variance forest predictions
    if (bart$model_params$include_variance_forest) {
        if (bart$model_params$sample_sigma_global) {
            sigma2_samples <- bart$sigma2_global_samples
            variance_forest_predictions <- sapply(1:num_samples, function(i) sqrt(s_x_raw[,i]*sigma2_samples[i]))
        } else {
            variance_forest_predictions <- sqrt(s_x_raw*sigma2_init)*y_std/sqrt(variance_scale)
        }
    }

    if ((bart$model_params$include_mean_forest) && (bart$model_params$has_rfx)) {
        y_hat <- mean_forest_predictions + rfx_predictions
    } else if ((bart$model_params$include_mean_forest) && (!bart$model_params$has_rfx)) {
        y_hat <- mean_forest_predictions
    } else if ((!bart$model_params$include_mean_forest) && (bart$model_params$has_rfx)) {
        y_hat <- rfx_predictions
    } 
    
    result <- list()
    if ((bart$model_params$has_rfx) || (bart$model_params$include_mean_forest)) {
        result[["y_hat"]] = y_hat
    }
    if (bart$model_params$include_mean_forest) {
        result[["mean_forest_predictions"]] = mean_forest_predictions
    }
    if (bart$model_params$has_rfx) {
        result[["rfx_predictions"]] = rfx_predictions
    }
    if (bart$model_params$include_variance_forest) {
        result[["variance_forest_predictions"]] = variance_forest_predictions
    }
    return(result)
}

#' Extract raw sample values for each of the random effect parameter terms.
#'
#' @param object Object of type `bcf` containing draws of a Bayesian causal forest model and associated sampling outputs.
#' @param ... Other parameters to be used in random effects extraction
#' @return List of arrays. The alpha array has dimension (`num_components`, `num_samples`) and is simply a vector if `num_components = 1`.
#' The xi and beta arrays have dimension (`num_components`, `num_groups`, `num_samples`) and is simply a matrix if `num_components = 1`.
#' The sigma array has dimension (`num_components`, `num_samples`) and is simply a vector if `num_components = 1`.
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
#' snr <- 3
#' group_ids <- rep(c(1,2), n %/% 2)
#' rfx_coefs <- matrix(c(-1, -1, 1, 1), nrow=2, byrow=TRUE)
#' rfx_basis <- cbind(1, runif(n, -1, 1))
#' rfx_term <- rowSums(rfx_coefs[group_ids,] * rfx_basis)
#' E_y <- f_XW + rfx_term
#' y <- E_y + rnorm(n, 0, 1)*(sd(E_y)/snr)
#' test_set_pct <- 0.2
#' n_test <- round(test_set_pct*n)
#' n_train <- n - n_test
#' test_inds <- sort(sample(1:n, n_test, replace = FALSE))
#' train_inds <- (1:n)[!((1:n) %in% test_inds)]
#' X_test <- X[test_inds,]
#' X_train <- X[train_inds,]
#' y_test <- y[test_inds]
#' y_train <- y[train_inds]
#' group_ids_test <- group_ids[test_inds]
#' group_ids_train <- group_ids[train_inds]
#' rfx_basis_test <- rfx_basis[test_inds,]
#' rfx_basis_train <- rfx_basis[train_inds,]
#' rfx_term_test <- rfx_term[test_inds]
#' rfx_term_train <- rfx_term[train_inds]
#' bart_model <- bart(X_train = X_train, y_train = y_train, 
#'                    group_ids_train = group_ids_train, rfx_basis_train = rfx_basis_train, 
#'                    X_test = X_test, group_ids_test = group_ids_test, rfx_basis_test = rfx_basis_test, 
#'                    num_gfr = 100, num_burnin = 0, num_mcmc = 100)
#' rfx_samples <- getRandomEffectSamples(bart_model)
getRandomEffectSamples.bartmodel <- function(object, ...){
    result = list()
    
    if (!object$model_params$has_rfx) {
        warning("This model has no RFX terms, returning an empty list")
        return(result)
    }
    
    # Extract the samples
    result <- object$rfx_samples$extract_parameter_samples()
    
    # Scale by sd(y_train)
    result$beta_samples <- result$beta_samples*object$model_params$outcome_scale
    result$xi_samples <- result$xi_samples*object$model_params$outcome_scale
    result$alpha_samples <- result$alpha_samples*object$model_params$outcome_scale
    result$sigma_samples <- result$sigma_samples*(object$model_params$outcome_scale^2)
    
    return(result)
}

#' Convert the persistent aspects of a BART model to (in-memory) JSON
#'
#' @param object Object of type `bartmodel` containing draws of a BART model and associated sampling outputs.
#'
#' @return Object of type `CppJson`
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
#' bart_model <- bart(X_train = X_train, y_train = y_train)
#' # bart_json <- convertBARTModelToJson(bart_model)
convertBARTModelToJson <- function(object){
    jsonobj <- createCppJson()
    
    if (is.null(object$model_params)) {
        stop("This BCF model has not yet been sampled")
    }

    # Add the forests
    if (object$model_params$include_mean_forest) {
        jsonobj$add_forest(object$mean_forests)
    }
    if (object$model_params$include_variance_forest) {
        jsonobj$add_forest(object$variance_forests)
    }

    # Add metadata
    jsonobj$add_scalar("num_numeric_vars", object$train_set_metadata$num_numeric_vars)
    jsonobj$add_scalar("num_ordered_cat_vars", object$train_set_metadata$num_ordered_cat_vars)
    jsonobj$add_scalar("num_unordered_cat_vars", object$train_set_metadata$num_unordered_cat_vars)
    if (object$train_set_metadata$num_numeric_vars > 0) {
        jsonobj$add_string_vector("numeric_vars", object$train_set_metadata$numeric_vars)
    }
    if (object$train_set_metadata$num_ordered_cat_vars > 0) {
        jsonobj$add_string_vector("ordered_cat_vars", object$train_set_metadata$ordered_cat_vars)
        jsonobj$add_string_list("ordered_unique_levels", object$train_set_metadata$ordered_unique_levels)
    }
    if (object$train_set_metadata$num_unordered_cat_vars > 0) {
        jsonobj$add_string_vector("unordered_cat_vars", object$train_set_metadata$unordered_cat_vars)
        jsonobj$add_string_list("unordered_unique_levels", object$train_set_metadata$unordered_unique_levels)
    }
    
    # Add global parameters
    jsonobj$add_scalar("variance_scale", object$model_params$variance_scale)
    jsonobj$add_scalar("outcome_scale", object$model_params$outcome_scale)
    jsonobj$add_scalar("outcome_mean", object$model_params$outcome_mean)
    jsonobj$add_boolean("standardize", object$model_params$standardize)
    jsonobj$add_scalar("sigma2_init", object$model_params$sigma2_init)
    jsonobj$add_boolean("sample_sigma_global", object$model_params$sample_sigma_global)
    jsonobj$add_boolean("sample_sigma_leaf", object$model_params$sample_sigma_leaf)
    jsonobj$add_boolean("include_mean_forest", object$model_params$include_mean_forest)
    jsonobj$add_boolean("include_variance_forest", object$model_params$include_variance_forest)
    jsonobj$add_boolean("has_rfx", object$model_params$has_rfx)
    jsonobj$add_boolean("has_rfx_basis", object$model_params$has_rfx_basis)
    jsonobj$add_scalar("num_rfx_basis", object$model_params$num_rfx_basis)
    jsonobj$add_scalar("num_gfr", object$model_params$num_gfr)
    jsonobj$add_scalar("num_burnin", object$model_params$num_burnin)
    jsonobj$add_scalar("num_mcmc", object$model_params$num_mcmc)
    jsonobj$add_scalar("num_samples", object$model_params$num_samples)
    jsonobj$add_scalar("num_covariates", object$model_params$num_covariates)
    jsonobj$add_scalar("num_basis", object$model_params$num_basis)
    jsonobj$add_scalar("num_chains", object$model_params$num_chains)
    jsonobj$add_scalar("keep_every", object$model_params$keep_every)
    jsonobj$add_boolean("requires_basis", object$model_params$requires_basis)
    if (object$model_params$sample_sigma_global) {
        jsonobj$add_vector("sigma2_global_samples", object$sigma2_global_samples, "parameters")
    }
    if (object$model_params$sample_sigma_leaf) {
        jsonobj$add_vector("sigma2_leaf_samples", object$sigma2_leaf_samples, "parameters")
    }

    # Add random effects (if present)
    if (object$model_params$has_rfx) {
        jsonobj$add_random_effects(object$rfx_samples)
        jsonobj$add_string_vector("rfx_unique_group_ids", object$rfx_unique_group_ids)
    }
    
    return(jsonobj)
}

#' Convert in-memory BART model objects (forests, random effects, vectors) to in-memory JSON. 
#' This function is primarily a convenience function for serialization / deserialization in a parallel BART sampler.
#'
#' @param param_list List containing high-level model state parameters
#' @param mean_forest Container of conditional mean forest samples (optional). Default: `NULL`.
#' @param variance_forest Container of conditional variance forest samples (optional). Default: `NULL`.
#' @param rfx_samples Container of random effect samples (optional). Default: `NULL`.
#' @param global_variance_samples Vector of global error variance samples (optional). Default: `NULL`.
#' @param local_variance_samples Vector of leaf scale samples (optional). Default: `NULL`.
#'
#' @return Object of type `CppJson`
convertBARTStateToJson <- function(param_list, mean_forest = NULL, variance_forest = NULL, 
                                   rfx_samples = NULL, global_variance_samples = NULL, 
                                   local_variance_samples = NULL) {
    # Initialize JSON object
    jsonobj <- createCppJson()
    
    # Add global parameters
    jsonobj$add_scalar("variance_scale", param_list$variance_scale)
    jsonobj$add_scalar("outcome_scale", param_list$outcome_scale)
    jsonobj$add_scalar("outcome_mean", param_list$outcome_mean)
    jsonobj$add_boolean("standardize", param_list$standardize)
    jsonobj$add_scalar("sigma2_init", param_list$sigma2_init)
    jsonobj$add_boolean("sample_sigma_global", param_list$sample_sigma_global)
    jsonobj$add_boolean("sample_sigma_leaf", param_list$sample_sigma_leaf)
    jsonobj$add_boolean("include_mean_forest", param_list$include_mean_forest)
    jsonobj$add_boolean("include_variance_forest", param_list$include_variance_forest)
    jsonobj$add_boolean("has_rfx", param_list$has_rfx)
    jsonobj$add_boolean("has_rfx_basis", param_list$has_rfx_basis)
    jsonobj$add_scalar("num_rfx_basis", param_list$num_rfx_basis)
    jsonobj$add_scalar("num_gfr", param_list$num_gfr)
    jsonobj$add_scalar("num_burnin", param_list$num_burnin)
    jsonobj$add_scalar("num_mcmc", param_list$num_mcmc)
    jsonobj$add_scalar("num_covariates", param_list$num_covariates)
    jsonobj$add_scalar("num_basis", param_list$num_basis)
    jsonobj$add_scalar("keep_every", param_list$keep_every)
    jsonobj$add_boolean("requires_basis", param_list$requires_basis)
    
    # Add the forests
    if (param_list$include_mean_forest) {
        jsonobj$add_forest(mean_forest)
    }
    if (param_list$include_variance_forest) {
        jsonobj$add_forest(object$variance_forests)
    }
    
    # Add sampled parameters
    if (param_list$sample_sigma_global) {
        jsonobj$add_vector("sigma2_global_samples", global_variance_samples, "parameters")
    }
    if (param_list$sample_sigma_leaf) {
        jsonobj$add_vector("sigma2_leaf_samples", local_variance_samples, "parameters")
    }
    
    # Add random effects
    if (param_list$has_rfx) {
        jsonobj$add_random_effects(rfx_samples)
        jsonobj$add_string_vector("rfx_unique_group_ids", param_list$rfx_unique_group_ids)
    }
    
    return(jsonobj)
}

#' Convert the persistent aspects of a BART model to (in-memory) JSON and save to a file
#'
#' @param object Object of type `bartmodel` containing draws of a BART model and associated sampling outputs.
#' @param filename String of filepath, must end in ".json"
#'
#' @return NULL
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
#' bart_model <- bart(X_train = X_train, y_train = y_train)
#' # saveBARTModelToJsonFile(bart_model, "test.json")
saveBARTModelToJsonFile <- function(object, filename){
    # Convert to Json
    jsonobj <- convertBARTModelToJson(object)
    
    # Save to file
    jsonobj$save_file(filename)
}

#' Convert the persistent aspects of a BART model to (in-memory) JSON string
#'
#' @param object Object of type `bartmodel` containing draws of a BART model and associated sampling outputs.
#' @return JSON string
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
#' bart_model <- bart(X_train = X_train, y_train = y_train)
#' # saveBARTModelToJsonString(bart_model)
saveBARTModelToJsonString <- function(object){
    # Convert to Json
    jsonobj <- convertBARTModelToJson(object)
    
    # Dump to string
    return(jsonobj$return_json_string())
}

#' Convert an (in-memory) JSON representation of a BART model to a BART model object 
#' which can be used for prediction, etc...
#'
#' @param json_object Object of type `CppJson` containing Json representation of a BART model
#'
#' @return Object of type `bartmodel`
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
#' bart_model <- bart(X_train = X_train, y_train = y_train)
#' # bart_json <- convertBARTModelToJson(bart_model)
#' # bart_model_roundtrip <- createBARTModelFromJson(bart_json)
createBARTModelFromJson <- function(json_object){
    # Initialize the BCF model
    output <- list()
    
    # Unpack the forests
    include_mean_forest <- json_object$get_boolean("include_mean_forest")
    include_variance_forest <- json_object$get_boolean("include_variance_forest")
    if (include_mean_forest) {
        output[["mean_forests"]] <- loadForestContainerJson(json_object, "forest_0")
        if (include_variance_forest) {
            output[["variance_forests"]] <- loadForestContainerJson(json_object, "forest_1")
        }
    } else {
        output[["variance_forests"]] <- loadForestContainerJson(json_object, "forest_0")
    }

    # Unpack metadata
    train_set_metadata = list()
    train_set_metadata[["num_numeric_vars"]] <- json_object$get_scalar("num_numeric_vars")
    train_set_metadata[["num_ordered_cat_vars"]] <- json_object$get_scalar("num_ordered_cat_vars")
    train_set_metadata[["num_unordered_cat_vars"]] <- json_object$get_scalar("num_unordered_cat_vars")
    if (train_set_metadata[["num_numeric_vars"]] > 0) {
        train_set_metadata[["numeric_vars"]] <- json_object$get_string_vector("numeric_vars")
    }
    if (train_set_metadata[["num_ordered_cat_vars"]] > 0) {
        train_set_metadata[["ordered_cat_vars"]] <- json_object$get_string_vector("ordered_cat_vars")
        train_set_metadata[["ordered_unique_levels"]] <- json_object$get_string_list("ordered_unique_levels", train_set_metadata[["ordered_cat_vars"]])
    }
    if (train_set_metadata[["num_unordered_cat_vars"]] > 0) {
        train_set_metadata[["unordered_cat_vars"]] <- json_object$get_string_vector("unordered_cat_vars")
        train_set_metadata[["unordered_unique_levels"]] <- json_object$get_string_list("unordered_unique_levels", train_set_metadata[["unordered_cat_vars"]])
    }
    output[["train_set_metadata"]] <- train_set_metadata
    
    # Unpack model params
    model_params = list()
    model_params[["variance_scale"]] <- json_object$get_scalar("variance_scale")
    model_params[["outcome_scale"]] <- json_object$get_scalar("outcome_scale")
    model_params[["outcome_mean"]] <- json_object$get_scalar("outcome_mean")
    model_params[["standardize"]] <- json_object$get_boolean("standardize")
    model_params[["sigma2_init"]] <- json_object$get_scalar("sigma2_init")
    model_params[["sample_sigma_global"]] <- json_object$get_boolean("sample_sigma_global")
    model_params[["sample_sigma_leaf"]] <- json_object$get_boolean("sample_sigma_leaf")
    model_params[["include_mean_forest"]] <- include_mean_forest
    model_params[["include_variance_forest"]] <- include_variance_forest
    model_params[["has_rfx"]] <- json_object$get_boolean("has_rfx")
    model_params[["has_rfx_basis"]] <- json_object$get_boolean("has_rfx_basis")
    model_params[["num_rfx_basis"]] <- json_object$get_scalar("num_rfx_basis")
    model_params[["num_gfr"]] <- json_object$get_scalar("num_gfr")
    model_params[["num_burnin"]] <- json_object$get_scalar("num_burnin")
    model_params[["num_mcmc"]] <- json_object$get_scalar("num_mcmc")
    model_params[["num_samples"]] <- json_object$get_scalar("num_samples")
    model_params[["num_covariates"]] <- json_object$get_scalar("num_covariates")
    model_params[["num_basis"]] <- json_object$get_scalar("num_basis")
    model_params[["num_chains"]] <- json_object$get_scalar("num_chains")
    model_params[["keep_every"]] <- json_object$get_scalar("keep_every")
    model_params[["requires_basis"]] <- json_object$get_boolean("requires_basis")
    output[["model_params"]] <- model_params
    
    # Unpack sampled parameters
    if (model_params[["sample_sigma_global"]]) {
        output[["sigma2_global_samples"]] <- json_object$get_vector("sigma2_global_samples", "parameters")
    }
    if (model_params[["sample_sigma_leaf"]]) {
        output[["sigma2_leaf_samples"]] <- json_object$get_vector("sigma2_leaf_samples", "parameters")
    }

    # Unpack random effects
    if (model_params[["has_rfx"]]) {
        output[["rfx_unique_group_ids"]] <- json_object$get_string_vector("rfx_unique_group_ids")
        output[["rfx_samples"]] <- loadRandomEffectSamplesJson(json_object, 0)
    }
    
    class(output) <- "bartmodel"
    return(output)
}

#' Convert a JSON file containing sample information on a trained BART model 
#' to a BART model object which can be used for prediction, etc...
#'
#' @param json_filename String of filepath, must end in ".json"
#'
#' @return Object of type `bartmodel`
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
#' bart_model <- bart(X_train = X_train, y_train = y_train)
#' # saveBARTModelToJsonFile(bart_model, "test.json")
#' # bart_model_roundtrip <- createBARTModelFromJsonFile("test.json")
createBARTModelFromJsonFile <- function(json_filename){
    # Load a `CppJson` object from file
    bart_json <- createCppJsonFile(json_filename)
    
    # Create and return the BART object
    bart_object <- createBARTModelFromJson(bart_json)
    
    return(bart_object)
}

#' Convert a JSON string containing sample information on a trained BART model 
#' to a BART model object which can be used for prediction, etc...
#'
#' @param json_string JSON string dump
#'
#' @return Object of type `bartmodel`
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
#' bart_model <- bart(X_train = X_train, y_train = y_train)
#' # bart_json <- saveBARTModelToJsonString(bart_model)
#' # bart_model_roundtrip <- createBARTModelFromJsonString(bart_json)
#' # y_hat_mean_roundtrip <- rowMeans(predict(bart_model_roundtrip, X_train)$y_hat)
#' # plot(rowMeans(bart_model$y_hat_train), y_hat_mean_roundtrip, 
#' #      xlab = "original", ylab = "roundtrip")
createBARTModelFromJsonString <- function(json_string){
    # Load a `CppJson` object from string
    bart_json <- createCppJsonString(json_string)
    
    # Create and return the BART object
    bart_object <- createBARTModelFromJson(bart_json)
    
    return(bart_object)
}

#' Convert a list of (in-memory) JSON representations of a BART model to a single combined BART model object 
#' which can be used for prediction, etc...
#'
#' @param json_object_list List of objects of type `CppJson` containing Json representation of a BART model
#'
#' @return Object of type `bartmodel`
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
#' bart_model <- bart(X_train = X_train, y_train = y_train)
#' # bart_json <- list(convertBARTModelToJson(bart_model))
#' # bart_model_roundtrip <- createBARTModelFromCombinedJson(bart_json)
createBARTModelFromCombinedJson <- function(json_object_list){
    # Initialize the BCF model
    output <- list()

    # For scalar / preprocessing details which aren't sample-dependent, 
    # defer to the first json
    json_object_default <- json_object_list[[1]]
    
    # Unpack the forests
    include_mean_forest <- json_object_default$get_boolean("include_mean_forest")
    include_variance_forest <- json_object_default$get_boolean("include_variance_forest")
    if (include_mean_forest) {
        output[["mean_forests"]] <- loadForestContainerCombinedJson(json_object_list, "forest_0")
        if (include_variance_forest) {
            output[["variance_forests"]] <- loadForestContainerCombinedJson(json_object_list, "forest_1")
        }
    } else {
        output[["variance_forests"]] <- loadForestContainerCombinedJson(json_object_list, "forest_0")
    }
    
    # Unpack metadata
    train_set_metadata = list()
    train_set_metadata[["num_numeric_vars"]] <- json_object_default$get_scalar("num_numeric_vars")
    train_set_metadata[["num_ordered_cat_vars"]] <- json_object_default$get_scalar("num_ordered_cat_vars")
    train_set_metadata[["num_unordered_cat_vars"]] <- json_object_default$get_scalar("num_unordered_cat_vars")
    if (train_set_metadata[["num_numeric_vars"]] > 0) {
        train_set_metadata[["numeric_vars"]] <- json_object_default$get_string_vector("numeric_vars")
    }
    if (train_set_metadata[["num_ordered_cat_vars"]] > 0) {
        train_set_metadata[["ordered_cat_vars"]] <- json_object_default$get_string_vector("ordered_cat_vars")
        train_set_metadata[["ordered_unique_levels"]] <- json_object_default$get_string_list("ordered_unique_levels", train_set_metadata[["ordered_cat_vars"]])
    }
    if (train_set_metadata[["num_unordered_cat_vars"]] > 0) {
        train_set_metadata[["unordered_cat_vars"]] <- json_object_default$get_string_vector("unordered_cat_vars")
        train_set_metadata[["unordered_unique_levels"]] <- json_object_default$get_string_list("unordered_unique_levels", train_set_metadata[["unordered_cat_vars"]])
    }
    output[["train_set_metadata"]] <- train_set_metadata

    # Unpack model params
    model_params = list()
    model_params[["outcome_scale"]] <- json_object_default$get_scalar("outcome_scale")
    model_params[["outcome_mean"]] <- json_object_default$get_scalar("outcome_mean")
    model_params[["standardize"]] <- json_object_default$get_boolean("standardize")
    model_params[["sigma2_init"]] <- json_object_default$get_scalar("sigma2_init")
    model_params[["sample_sigma_global"]] <- json_object$get_boolean("sample_sigma_global")
    model_params[["sample_sigma_leaf"]] <- json_object$get_boolean("sample_sigma_leaf")
    model_params[["include_mean_forest"]] <- include_mean_forest
    model_params[["include_variance_forest"]] <- include_variance_forest
    model_params[["has_rfx"]] <- json_object_default$get_boolean("has_rfx")
    model_params[["has_rfx_basis"]] <- json_object_default$get_boolean("has_rfx_basis")
    model_params[["num_rfx_basis"]] <- json_object_default$get_scalar("num_rfx_basis")
    model_params[["num_covariates"]] <- json_object_default$get_scalar("num_covariates")
    model_params[["num_basis"]] <- json_object_default$get_scalar("num_basis")
    model_params[["requires_basis"]] <- json_object_default$get_boolean("requires_basis")
    model_params[["num_chains"]] <- json_object_default$get_scalar("num_chains")
    model_params[["keep_every"]] <- json_object_default$get_scalar("keep_every")

    # Combine values that are sample-specific
    for (i in 1:length(json_object_list)) {
        json_object <- json_object_list[[i]]
        if (i == 1) {
            model_params[["num_gfr"]] <- json_object$get_scalar("num_gfr")
            model_params[["num_burnin"]] <- json_object$get_scalar("num_burnin")
            model_params[["num_mcmc"]] <- json_object$get_scalar("num_mcmc")
            model_params[["num_samples"]] <- json_object$get_scalar("num_samples")
        } else {
            prev_json <- json_object_list[[i-1]]
            model_params[["num_gfr"]] <- model_params[["num_gfr"]] + json_object$get_scalar("num_gfr")
            model_params[["num_burnin"]] <- model_params[["num_burnin"]] + json_object$get_scalar("num_burnin")
            model_params[["num_mcmc"]] <- model_params[["num_mcmc"]] + json_object$get_scalar("num_mcmc")
            model_params[["num_samples"]] <- model_params[["num_samples"]] + json_object$get_scalar("num_samples")
        }
    }
    output[["model_params"]] <- model_params
    
    # Unpack sampled parameters
    if (model_params[["sample_sigma_global"]]) {
        for (i in 1:length(json_object_list)) {
            json_object <- json_object_list[[i]]
            if (i == 1) {
                output[["sigma2_global_samples"]] <- json_object$get_vector("sigma2_global_samples", "parameters")
            } else {
                output[["sigma2_global_samples"]] <- c(output[["sigma2_global_samples"]], json_object$get_vector("sigma2_global_samples", "parameters"))
            }
        }
    }
    if (model_params[["sample_sigma_leaf"]]) {
        for (i in 1:length(json_object_list)) {
            json_object <- json_object_list[[i]]
            if (i == 1) {
                output[["sigma2_leaf_samples"]] <- json_object$get_vector("sigma2_leaf_samples", "parameters")
            } else {
                output[["sigma2_leaf_samples"]] <- c(output[["sigma2_leaf_samples"]], json_object$get_vector("sigma2_leaf_samples", "parameters"))
            }
        }
    }
    
    # Unpack random effects
    if (model_params[["has_rfx"]]) {
        output[["rfx_unique_group_ids"]] <- json_object_default$get_string_vector("rfx_unique_group_ids")
        output[["rfx_samples"]] <- loadRandomEffectSamplesCombinedJson(json_object_list, 0)
    }
    
    class(output) <- "bartmodel"
    return(output)
}

#' Convert a list of (in-memory) JSON strings that represent BART models to a single combined BART model object 
#' which can be used for prediction, etc...
#'
#' @param json_string_list List of JSON strings which can be parsed to objects of type `CppJson` containing Json representation of a BART model
#'
#' @return Object of type `bartmodel`
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
#' bart_model <- bart(X_train = X_train, y_train = y_train)
#' # bart_json_string_list <- list(saveBARTModelToJsonString(bart_model))
#' # bart_model_roundtrip <- createBARTModelFromCombinedJsonString(bart_json_string_list)
createBARTModelFromCombinedJsonString <- function(json_string_list){
    # Initialize the BCF model
    output <- list()
    
    # Convert JSON strings
    json_object_list <- list()
    for (i in 1:length(json_string_list)) {
        json_string <- json_string_list[[i]]
        json_object_list[[i]] <- createCppJsonString(json_string)
    }
    
    # For scalar / preprocessing details which aren't sample-dependent, 
    # defer to the first json
    json_object_default <- json_object_list[[1]]
    
    # Unpack the forests
    include_mean_forest <- json_object_default$get_boolean("include_mean_forest")
    include_variance_forest <- json_object_default$get_boolean("include_variance_forest")
    if (include_mean_forest) {
        output[["mean_forests"]] <- loadForestContainerCombinedJson(json_object_list, "forest_0")
        if (include_variance_forest) {
            output[["variance_forests"]] <- loadForestContainerCombinedJson(json_object_list, "forest_1")
        }
    } else {
        output[["variance_forests"]] <- loadForestContainerCombinedJson(json_object_list, "forest_0")
    }
    
    # Unpack metadata
    train_set_metadata = list()
    train_set_metadata[["num_numeric_vars"]] <- json_object_default$get_scalar("num_numeric_vars")
    train_set_metadata[["num_ordered_cat_vars"]] <- json_object_default$get_scalar("num_ordered_cat_vars")
    train_set_metadata[["num_unordered_cat_vars"]] <- json_object_default$get_scalar("num_unordered_cat_vars")
    if (train_set_metadata[["num_numeric_vars"]] > 0) {
        train_set_metadata[["numeric_vars"]] <- json_object_default$get_string_vector("numeric_vars")
    }
    if (train_set_metadata[["num_ordered_cat_vars"]] > 0) {
        train_set_metadata[["ordered_cat_vars"]] <- json_object_default$get_string_vector("ordered_cat_vars")
        train_set_metadata[["ordered_unique_levels"]] <- json_object_default$get_string_list("ordered_unique_levels", train_set_metadata[["ordered_cat_vars"]])
    }
    if (train_set_metadata[["num_unordered_cat_vars"]] > 0) {
        train_set_metadata[["unordered_cat_vars"]] <- json_object_default$get_string_vector("unordered_cat_vars")
        train_set_metadata[["unordered_unique_levels"]] <- json_object_default$get_string_list("unordered_unique_levels", train_set_metadata[["unordered_cat_vars"]])
    }
    output[["train_set_metadata"]] <- train_set_metadata

    # Unpack model params
    model_params = list()
    model_params[["variance_scale"]] <- json_object_default$get_scalar("variance_scale")
    model_params[["outcome_scale"]] <- json_object_default$get_scalar("outcome_scale")
    model_params[["outcome_mean"]] <- json_object_default$get_scalar("outcome_mean")
    model_params[["standardize"]] <- json_object_default$get_boolean("standardize")
    model_params[["sigma2_init"]] <- json_object_default$get_scalar("sigma2_init")
    model_params[["sample_sigma_global"]] <- json_object_default$get_boolean("sample_sigma_global")
    model_params[["sample_sigma_leaf"]] <- json_object_default$get_boolean("sample_sigma_leaf")
    model_params[["include_mean_forest"]] <- include_mean_forest
    model_params[["include_variance_forest"]] <- include_variance_forest
    model_params[["has_rfx"]] <- json_object_default$get_boolean("has_rfx")
    model_params[["has_rfx_basis"]] <- json_object_default$get_boolean("has_rfx_basis")
    model_params[["num_rfx_basis"]] <- json_object_default$get_scalar("num_rfx_basis")
    model_params[["num_covariates"]] <- json_object_default$get_scalar("num_covariates")
    model_params[["num_basis"]] <- json_object_default$get_scalar("num_basis")
    model_params[["num_chains"]] <- json_object_default$get_scalar("num_chains")
    model_params[["keep_every"]] <- json_object_default$get_scalar("keep_every")
    model_params[["requires_basis"]] <- json_object_default$get_boolean("requires_basis")
    
    # Combine values that are sample-specific
    for (i in 1:length(json_object_list)) {
        json_object <- json_object_list[[i]]
        if (i == 1) {
            model_params[["num_gfr"]] <- json_object$get_scalar("num_gfr")
            model_params[["num_burnin"]] <- json_object$get_scalar("num_burnin")
            model_params[["num_mcmc"]] <- json_object$get_scalar("num_mcmc")
            model_params[["num_samples"]] <- json_object$get_scalar("num_samples")
        } else {
            prev_json <- json_object_list[[i-1]]
            model_params[["num_gfr"]] <- model_params[["num_gfr"]] + json_object$get_scalar("num_gfr")
            model_params[["num_burnin"]] <- model_params[["num_burnin"]] + json_object$get_scalar("num_burnin")
            model_params[["num_mcmc"]] <- model_params[["num_mcmc"]] + json_object$get_scalar("num_mcmc")
            model_params[["num_samples"]] <- model_params[["num_samples"]] + json_object$get_scalar("num_samples")
        }
    }
    output[["model_params"]] <- model_params
    
    # Unpack sampled parameters
    if (model_params[["sample_sigma_global"]]) {
        for (i in 1:length(json_object_list)) {
            json_object <- json_object_list[[i]]
            if (i == 1) {
                output[["sigma2_global_samples"]] <- json_object$get_vector("sigma2_global_samples", "parameters")
            } else {
                output[["sigma2_global_samples"]] <- c(output[["sigma2_global_samples"]], json_object$get_vector("sigma2_global_samples", "parameters"))
            }
        }
    }
    if (model_params[["sample_sigma_leaf"]]) {
        for (i in 1:length(json_object_list)) {
            json_object <- json_object_list[[i]]
            if (i == 1) {
                output[["sigma2_leaf_samples"]] <- json_object$get_vector("sigma2_leaf_samples", "parameters")
            } else {
                output[["sigma2_leaf_samples"]] <- c(output[["sigma2_leaf_samples"]], json_object$get_vector("sigma2_leaf_samples", "parameters"))
            }
        }
    }
    
    # Unpack random effects
    if (model_params[["has_rfx"]]) {
        output[["rfx_unique_group_ids"]] <- json_object_default$get_string_vector("rfx_unique_group_ids")
        output[["rfx_samples"]] <- loadRandomEffectSamplesCombinedJson(json_object_list, 0)
    }
    
    class(output) <- "bartmodel"
    return(output)
}
