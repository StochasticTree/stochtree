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
#' @param cutpoint_grid_size Maximum size of the "grid" of potential cutpoints to consider. Default: 100.
#' @param tau_init Starting value of leaf node scale parameter. Calibrated internally as `1/num_trees_mean` if not set here.
#' @param leaf_model Model to use in the leaves, coded as integer with (0 = constant leaf, 1 = univariate leaf regression, 2 = multivariate leaf regression). Default: 0.
#' @param alpha_mean Prior probability of splitting for a tree of depth 0 in the mean model. Tree split prior combines `alpha_mean` and `beta_mean` via `alpha_mean*(1+node_depth)^-beta_mean`.
#' @param beta_mean Exponent that decreases split probabilities for nodes of depth > 0 in the mean model. Tree split prior combines `alpha_mean` and `beta_mean` via `alpha_mean*(1+node_depth)^-beta_mean`.
#' @param min_samples_leaf_mean Minimum allowable size of a leaf, in terms of training samples, in the mean model. Default: 5.
#' @param max_depth_mean Maximum depth of any tree in the ensemble in the mean model. Default: 10. Can be overriden with ``-1`` which does not enforce any depth limits on trees.
#' @param alpha_variance Prior probability of splitting for a tree of depth 0 in the variance model. Tree split prior combines `alpha_variance` and `beta_variance` via `alpha_variance*(1+node_depth)^-beta_variance`.
#' @param beta_variance Exponent that decreases split probabilities for nodes of depth > 0 in the variance model. Tree split prior combines `alpha_variance` and `beta_variance` via `alpha_variance*(1+node_depth)^-beta_variance`.
#' @param min_samples_leaf_variance Minimum allowable size of a leaf, in terms of training samples, in the variance model. Default: 5.
#' @param max_depth_variance Maximum depth of any tree in the ensemble in the variance model. Default: 10. Can be overriden with ``-1`` which does not enforce any depth limits on trees.
#' @param a_global Shape parameter in the `IG(a_global, b_global)` global error variance model. Default: 0.
#' @param b_global Scale parameter in the `IG(a_global, b_global)` global error variance model. Default: 0.
#' @param a_leaf Shape parameter in the `IG(a_leaf, b_leaf)` leaf node parameter variance model. Default: 3.
#' @param b_leaf Scale parameter in the `IG(a_leaf, b_leaf)` leaf node parameter variance model. Calibrated internally as `0.5/num_trees_mean` if not set here.
#' @param a_forest Shape parameter in the `IG(a_forest, b_forest)` conditional error variance model (which is only sampled if `include_variance_forest = T`). Calibrated internally as `num_trees_variance / 1.5^2 + 0.5` if not set.
#' @param b_forest Scale parameter in the `IG(a_forest, b_forest)` conditional error variance model (which is only sampled if `include_variance_forest = T`). Calibrated internally as `num_trees_variance / 1.5^2` if not set.
#' @param q Quantile used to calibrated `lambda` as in Sparapani et al (2021). Default: 0.9.
#' @param sigma2_init Starting value of global error variance parameter. Calibrated internally as `pct_var_sigma2_init*var((y-mean(y))/sd(y))` if not set.
#' @param variance_forest_init Starting value of root forest prediction in conditional (heteroskedastic) error variance model. Calibrated internally as `log(pct_var_variance_forest_init*var((y-mean(y))/sd(y)))/num_trees_variance` if not set.
#' @param pct_var_sigma2_init Percentage of standardized outcome variance used to initialize global error variance parameter. Default: 0.25. Superseded by `sigma2_init`.
#' @param pct_var_variance_forest_init Percentage of standardized outcome variance used to initialize global error variance parameter. Default: 1. Superseded by `variance_forest_init`.
#' @param variable_weights_mean Numeric weights reflecting the relative probability of splitting on each variable in the mean forest. Does not need to sum to 1 but cannot be negative. Defaults to `rep(1/ncol(X_train), ncol(X_train))` if not set here.
#' @param variable_weights_variance Numeric weights reflecting the relative probability of splitting on each variable in the variance forest. Does not need to sum to 1 but cannot be negative. Defaults to `rep(1/ncol(X_train), ncol(X_train))` if not set here.
#' @param num_trees_mean Number of trees in the ensemble for the conditional mean model. Default: 200. If `num_trees_mean = 0`, the conditional mean will not be modeled using a forest and the function will only proceed if `num_trees_variance > 0`.
#' @param num_trees_variance Number of trees in the ensemble for the conditional variance model. Default: 0. Variance is only modeled using a tree / forest if `num_trees_variance > 0`.
#' @param num_gfr Number of "warm-start" iterations run using the grow-from-root algorithm (He and Hahn, 2021). Default: 5.
#' @param num_burnin Number of "burn-in" iterations of the MCMC sampler. Default: 0.
#' @param num_mcmc Number of "retained" iterations of the MCMC sampler. Default: 100.
#' @param sample_sigma Whether or not to update the `sigma^2` global error variance parameter based on `IG(a_globa, b_global)`. Default: T.
#' @param sample_tau Whether or not to update the `tau` leaf scale variance parameter based on `IG(a_leaf, b_leaf)`. Cannot (currently) be set to true if `ncol(W_train)>1`. Default: T.
#' @param random_seed Integer parameterizing the C++ random number generator. If not specified, the C++ random number generator is seeded according to `std::random_device`.
#' @param keep_burnin Whether or not "burnin" samples should be included in cached predictions. Default FALSE. Ignored if num_mcmc = 0.
#' @param keep_gfr Whether or not "grow-from-root" samples should be included in cached predictions. Default TRUE. Ignored if num_mcmc = 0.
#' @param verbose Whether or not to print progress during the sampling loops. Default: FALSE.
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
                 cutpoint_grid_size = 100, tau_init = NULL, 
                 alpha_mean = 0.95, beta_mean = 2.0, min_samples_leaf_mean = 5, 
                 max_depth_mean = 10, alpha_variance = 0.95, beta_variance = 2.0, 
                 min_samples_leaf_variance = 5, max_depth_variance = 10, 
                 a_global = 0, b_global = 0, a_leaf = 3, b_leaf = NULL, 
                 a_forest = NULL, b_forest = NULL, q = 0.9, sigma2_init = NULL, 
                 variance_forest_init = NULL, pct_var_sigma2_init = 1, 
                 pct_var_variance_forest_init = 1, variable_weights_mean = NULL, 
                 variable_weights_variance = NULL, num_trees_mean = 200, num_trees_variance = 20, 
                 num_gfr = 5, num_burnin = 0, num_mcmc = 100, sample_sigma = T, 
                 sample_tau = T, random_seed = -1, keep_burnin = F, 
                 keep_gfr = F, verbose = F) {
    # Determine whether conditional mean, variance, or both will be modeled
    if (num_trees_variance > 0) include_variance_forest = T
    else include_variance_forest = F
    if (num_trees_mean > 0) include_mean_forest = T
    else include_mean_forest = F
    
    # Override tau sampling if there is no mean forest
    if (!include_mean_forest) sample_tau <- F
    
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
    y_bar_train <- mean(y_train)
    y_std_train <- sd(y_train)
    resid_train <- (y_train-y_bar_train)/y_std_train
    
    # Compute initial value of root nodes in mean forest
    init_val_mean <- mean(resid_train)

    # Calibrate priors for sigma^2 and tau
    if (is.null(sigma2_init)) sigma2_init <- pct_var_sigma2_init*var(resid_train)
    if (is.null(variance_forest_init)) variance_forest_init <- pct_var_variance_forest_init*var(resid_train)
    if (is.null(b_leaf)) b_leaf <- var(resid_train)/(2*num_trees_mean)
    if (is.null(tau_init)) tau_init <- var(resid_train)/(num_trees_mean)
    current_leaf_scale <- as.matrix(tau_init)
    current_sigma2 <- sigma2_init
    a_0 <- 1.5
    a_forest <- num_trees_variance/(a_0^2) + 0.5
    b_forest <- num_trees_variance/(a_0^2)
    
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
        if (sample_tau) {
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
    }
    if (include_variance_forest) {
        forest_samples_variance <- createForestContainer(num_trees_variance, 1, TRUE, TRUE)
    }
    
    # Random effects prior parameters
    if (has_rfx) {
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
    }

    # Random effects data structure and storage container
    if (has_rfx) {
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
    num_samples <- num_gfr + num_burnin + num_mcmc
    if (sample_sigma) global_var_samples <- rep(0, num_samples)
    if (sample_tau) leaf_scale_samples <- rep(0, num_samples)
    
    # Initialize the leaves of each tree in the mean forest
    if (include_mean_forest) {
        if (requires_basis) init_values_mean_forest <- rep(0., ncol(W_train))
        else init_values_mean_forest <- 0.
        forest_samples_mean$prepare_for_sampler(forest_dataset_train, outcome_train, forest_model_mean, leaf_model_mean_forest, init_values_mean_forest)
    }

    # Initialize the leaves of each tree in the variance forest
    if (include_variance_forest) {
        forest_samples_variance$prepare_for_sampler(forest_dataset_train, outcome_train, forest_model_variance, leaf_model_variance_forest, variance_forest_init)
    }
    
    # Run GFR (warm start) if specified
    if (num_gfr > 0){
        gfr_indices = 1:num_gfr
        for (i in 1:num_gfr) {
            # Print progress
            if (verbose) {
                if ((i %% 10 == 0) || (i == num_gfr)) {
                    cat("Sampling", i, "out of", num_gfr, "XBART (grow-from-root) draws\n")
                }
            }
            
            
            if (include_mean_forest) {
                forest_model_mean$sample_one_iteration(
                    forest_dataset_train, outcome_train, forest_samples_mean, rng, feature_types, 
                    leaf_model_mean_forest, current_leaf_scale, variable_weights_mean, 
                    a_forest, b_forest, current_sigma2, cutpoint_grid_size, gfr = T, pre_initialized = T
                )
            }
            if (include_variance_forest) {
                forest_model_variance$sample_one_iteration(
                    forest_dataset_train, outcome_train, forest_samples_variance, rng, feature_types, 
                    leaf_model_variance_forest, current_leaf_scale, variable_weights_variance, 
                    a_forest, b_forest, current_sigma2, cutpoint_grid_size, gfr = T, pre_initialized = T
                )
            }
            if (sample_sigma) {
                global_var_samples[i] <- sample_sigma2_one_iteration(outcome_train, forest_dataset_train, rng, a_global, b_global)
                current_sigma2 <- global_var_samples[i]
            }
            if (sample_tau) {
                leaf_scale_samples[i] <- sample_tau_one_iteration(forest_samples_mean, rng, a_leaf, b_leaf, i-1)
                current_leaf_scale <- as.matrix(leaf_scale_samples[i])
            }
            if (has_rfx) {
                rfx_model$sample_random_effect(rfx_dataset_train, outcome_train, rfx_tracker_train, rfx_samples, current_sigma2, rng)
            }
        }
    }
    
    # Run MCMC
    if (num_burnin + num_mcmc > 0) {
        if (num_burnin > 0) {
            burnin_indices = (num_gfr+1):(num_gfr+num_burnin)
        }
        if (num_mcmc > 0) {
            mcmc_indices = (num_gfr+num_burnin+1):(num_gfr+num_burnin+num_mcmc)
        }
        for (i in (num_gfr+1):num_samples) {
            # Print progress
            if (verbose) {
                if (num_burnin > 0) {
                    if (((i - num_gfr) %% 100 == 0) || ((i - num_gfr) == num_burnin)) {
                        cat("Sampling", i - num_gfr, "out of", num_burnin, "BART burn-in draws\n")
                    }
                }
                if (num_mcmc > 0) {
                    if (((i - num_gfr - num_burnin) %% 100 == 0) || (i == num_samples)) {
                        cat("Sampling", i - num_burnin - num_gfr, "out of", num_mcmc, "BART MCMC draws\n")
                    }
                }
            }
            
            if (include_mean_forest) {
                forest_model_mean$sample_one_iteration(
                    forest_dataset_train, outcome_train, forest_samples_mean, rng, feature_types, 
                    leaf_model_mean_forest, current_leaf_scale, variable_weights_mean, 
                    a_forest, b_forest, current_sigma2, cutpoint_grid_size, gfr = F, pre_initialized = T
                )
            }
            if (include_variance_forest) {
                forest_model_variance$sample_one_iteration(
                    forest_dataset_train, outcome_train, forest_samples_variance, rng, feature_types, 
                    leaf_model_variance_forest, current_leaf_scale, variable_weights_variance, 
                    a_forest, b_forest, current_sigma2, cutpoint_grid_size, gfr = F, pre_initialized = T
                )
            }
            if (sample_sigma) {
                global_var_samples[i] <- sample_sigma2_one_iteration(outcome_train, forest_dataset_train, rng, a_global, b_global)
                current_sigma2 <- global_var_samples[i]
            }
            if (sample_tau) {
                leaf_scale_samples[i] <- sample_tau_one_iteration(forest_samples_mean, rng, a_leaf, b_leaf, i-1)
                current_leaf_scale <- as.matrix(leaf_scale_samples[i])
            }
            if (has_rfx) {
                rfx_model$sample_random_effect(rfx_dataset_train, outcome_train, rfx_tracker_train, rfx_samples, current_sigma2, rng)
            }
        }
    }
    
    # Mean forest predictions
    if (include_mean_forest) {
        y_hat_train <- forest_samples_mean$predict(forest_dataset_train)*y_std_train + y_bar_train
        if (has_test) y_hat_test <- forest_samples_mean$predict(forest_dataset_test)*y_std_train + y_bar_train
    }
    
    # Variance forest predictions
    if (include_variance_forest) {
        sigma_x_hat_train <- forest_samples_variance$predict(forest_dataset_train)
        if (has_test) sigma_x_hat_test <- forest_samples_variance$predict(forest_dataset_test)
    }
    
    # Random effects predictions
    if (has_rfx) {
        rfx_preds_train <- rfx_samples$predict(group_ids_train, rfx_basis_train)*y_std_train
        y_hat_train <- y_hat_train + rfx_preds_train
    }
    if ((has_rfx_test) && (has_test)) {
        rfx_preds_test <- rfx_samples$predict(group_ids_test, rfx_basis_test)*y_std_train
        y_hat_test <- y_hat_test + rfx_preds_test
    }
    
    # Compute retention indices
    if (num_mcmc > 0) {
        keep_indices = mcmc_indices
        if (keep_gfr) keep_indices <- c(gfr_indices, keep_indices)
        if (keep_burnin) keep_indices <- c(burnin_indices, keep_indices)
    } else {
        if ((num_gfr > 0) && (num_burnin > 0)) {
            # Override keep_gfr = FALSE since there are no MCMC samples
            # Don't retain both GFR and burnin samples
            keep_indices = gfr_indices
        } else if ((num_gfr <= 0) && (num_burnin > 0)) {
            # Override keep_burnin = FALSE since there are no MCMC or GFR samples
            keep_indices = burnin_indices
        } else if ((num_gfr > 0) && (num_burnin <= 0)) {
            # Override keep_gfr = FALSE since there are no MCMC samples
            keep_indices = gfr_indices
        } else {
            stop("There are no samples to retain!")
        } 
    }
    
    # Subset forest and RFX predictions
    if (include_mean_forest) {
        y_hat_train <- y_hat_train[,keep_indices]
        if (has_test) y_hat_test <- y_hat_test[,keep_indices]
    }
    if (include_variance_forest) {
        sigma_x_hat_train <- sigma_x_hat_train[,keep_indices]
        if (has_test) sigma_x_hat_test <- sigma_x_hat_test[,keep_indices]
    }
    if (has_rfx) {
        rfx_preds_train <- rfx_preds_train[,keep_indices]
        if (has_test) rfx_preds_test <- rfx_preds_test[,keep_indices]
    }

    # Global error variance
    if (sample_sigma) sigma2_samples <- global_var_samples[keep_indices]*(y_std_train^2)
    
    # Leaf parameter variance
    if (sample_tau) tau_samples <- leaf_scale_samples[keep_indices]
    
    # Rescale variance forest prediction by sigma2_samples
    if (include_variance_forest) {
        if (sample_sigma) {
            sigma_x_hat_train <- sapply(1:length(keep_indices), function(i) sqrt(sigma_x_hat_train[,i]*sigma2_samples[i]))
            if (has_test) sigma_x_hat_test <- sapply(1:length(keep_indices), function(i) sqrt(sigma_x_hat_test[,i]*sigma2_samples[i]))
        } else {
            sigma_x_hat_train <- sqrt(sigma_x_hat_train*sigma2_init)
            if (has_test) sigma_x_hat_test <- sqrt(sigma_x_hat_test*sigma2_init)
        }
    }
    
    # Return results as a list
    model_params <- list(
        "sigma2_init" = sigma2_init, 
        "a_global" = a_global,
        "b_global" = b_global, 
        "tau_init" = tau_init,
        "a_leaf" = a_leaf, 
        "b_leaf" = b_leaf,
        "a_forest" = a_forest, 
        "b_forest" = b_forest,
        "outcome_mean" = y_bar_train,
        "outcome_scale" = y_std_train, 
        "output_dimension" = output_dimension,
        "is_leaf_constant" = is_leaf_constant,
        "leaf_regression" = leaf_regression,
        "requires_basis" = requires_basis, 
        "num_covariates" = ncol(X_train), 
        "num_basis" = ifelse(is.null(W_train),0,ncol(W_train)), 
        "num_samples" = num_samples, 
        "num_gfr" = num_gfr, 
        "num_burnin" = num_burnin, 
        "num_mcmc" = num_mcmc, 
        "has_basis" = !is.null(W_train), 
        "has_rfx" = has_rfx, 
        "has_rfx_basis" = has_basis_rfx, 
        "num_rfx_basis" = num_basis_rfx, 
        "sample_sigma" = sample_sigma,
        "sample_tau" = sample_tau,
        "include_mean_forest" = include_mean_forest,
        "include_variance_forest" = include_variance_forest
    )
    result <- list(
        "model_params" = model_params, 
        "train_set_metadata" = X_train_metadata,
        "keep_indices" = keep_indices
    )
    if (include_mean_forest) {
        result[["mean_forests"]] = forest_samples_mean
        result[["y_hat_train"]] = y_hat_train
        if (has_test) result[["y_hat_test"]] = y_hat_test
    }
    if (include_variance_forest) {
        result[["var_forests"]] = forest_samples_variance
        result[["sigma_x_hat_train"]] = sigma_x_hat_train
        if (has_test) result[["sigma_x_hat_test"]] = sigma_x_hat_test
    }
    if (sample_sigma) result[["sigma2_samples"]] = sigma2_samples
    if (sample_tau) result[["tau_samples"]] = tau_samples
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
#' @param predict_all (Optional) Whether to predict the model for all of the samples in the stored objects or the subset of burnt-in / GFR samples as specified at training time. Default FALSE.
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
predict.bartmodel <- function(bart, X_test, W_test = NULL, group_ids_test = NULL, rfx_basis_test = NULL, predict_all = F){
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
    y_std <- bart$model_params$outcome_scale
    y_bar <- bart$model_params$outcome_mean
    mean_forest_predictions <- bart$mean_forests$predict(prediction_dataset)*y_std + y_bar
    
    # Compute variance forest predictions
    if (bart$model_params$include_variance_forest) {
        var_forest_predictions <- bart$variance_forests$predict(prediction_dataset)*(y_std^2)
    }
    
    # Compute rfx predictions (if needed)
    if (bart$model_params$has_rfx) {
        rfx_predictions <- bart$rfx_samples$predict(group_ids_test, rfx_basis_test)*y_std
    }
    
    # Restrict predictions to the "retained" samples (if applicable)
    if (!predict_all) {
        keep_indices = bart$keep_indices
        mean_forest_predictions <- mean_forest_predictions[,keep_indices]
        if (bart$model_params$include_variance_forest) {
            variance_forest_predictions <- variance_forest_predictions[,keep_indices]
        }
        if (bart$model_params$has_rfx) rfx_predictions <- rfx_predictions[,keep_indices]
    }
    
    if (bart$model_params$has_rfx) {
        y_hat <- forest_predictions + rfx_predictions
        result <- list(
            "mean_forest_predictions" = mean_forest_predictions, 
            "variance_forest_predictions" = variance_forest_predictions, 
            "rfx_predictions" = rfx_predictions, 
            "y_hat" = y_hat
        )
        return(result)
    } else {
        result <- list(
            "y_hat" = mean_forest_predictions, 
            "variance_forest_predictions" = variance_forest_predictions
        )
        return(result)
    }
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
#'                    num_gfr = 100, num_burnin = 0, num_mcmc = 100, sample_tau = TRUE)
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
