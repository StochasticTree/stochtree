#' Run the Bayesian Causal Forest (BCF) algorithm for regularized causal effect estimation. 
#'
#' @param X_train Covariates used to split trees in the ensemble. May be provided either as a dataframe or a matrix. 
#' Matrix covariates will be assumed to be all numeric. Covariates passed as a dataframe will be 
#' preprocessed based on the variable types (e.g. categorical columns stored as unordered factors will be one-hot encoded, 
#' categorical columns stored as ordered factors will passed as integers to the core algorithm, along with the metadata 
#' that the column is ordered categorical).
#' @param Z_train Vector of (continuous or binary) treatment assignments.
#' @param y_train Outcome to be modeled by the ensemble.
#' @param pi_train (Optional) Vector of propensity scores. If not provided, this will be estimated from the data.
#' @param group_ids_train (Optional) Group labels used for an additive random effects model.
#' @param rfx_basis_train (Optional) Basis for "random-slope" regression in an additive random effects model.
#' If `group_ids_train` is provided with a regression basis, an intercept-only random effects model 
#' will be estimated.
#' @param X_test (Optional) Test set of covariates used to define "out of sample" evaluation data. 
#' May be provided either as a dataframe or a matrix, but the format of `X_test` must be consistent with 
#' that of `X_train`.
#' @param Z_test (Optional) Test set of (continuous or binary) treatment assignments.
#' @param pi_test (Optional) Vector of propensity scores. If not provided, this will be estimated from the data.
#' @param group_ids_test (Optional) Test set group labels used for an additive random effects model. 
#' We do not currently support (but plan to in the near future), test set evaluation for group labels
#' that were not in the training set.
#' @param rfx_basis_test (Optional) Test set basis for "random-slope" regression in additive random effects model.
#' @param num_gfr Number of "warm-start" iterations run using the grow-from-root algorithm (He and Hahn, 2021). Default: 5.
#' @param num_burnin Number of "burn-in" iterations of the MCMC sampler. Default: 0.
#' @param num_mcmc Number of "retained" iterations of the MCMC sampler. Default: 100.
#' @param previous_model_json (Optional) JSON string containing a previous BCF model. This can be used to "continue" a sampler interactively after inspecting the samples or to run parallel chains "warm-started" from existing forest samples. Default: `NULL`.
#' @param warmstart_sample_num (Optional) Sample number from `previous_model_json` that will be used to warmstart this BCF sampler. One-indexed (so that the first sample is used for warm-start by setting `warmstart_sample_num = 1`). Default: `NULL`.
#' @param params The list of model parameters, each of which has a default value.
#' 
#'   **1. Global Parameters**
#'
#'   - `cutpoint_grid_size` Maximum size of the "grid" of potential cutpoints to consider. Default: `100`.
#'   - `a_global` Shape parameter in the `IG(a_global, b_global)` global error variance model. Default: `0`.
#'   - `b_global` Scale parameter in the `IG(a_global, b_global)` global error variance model. Default: `0`.
#'   - `sigma2_init` Starting value of global error variance parameter. Calibrated internally as `pct_var_sigma2_init*var((y-mean(y))/sd(y))` if not set.
#'   - `pct_var_sigma2_init` Percentage of standardized outcome variance used to initialize global error variance parameter. Default: `1`. Superseded by `sigma2_init`.
#'   - `variable_weights` Numeric weights reflecting the relative probability of splitting on each variable. Does not need to sum to 1 but cannot be negative. Defaults to `rep(1/ncol(X_train), ncol(X_train))` if not set here. Note that if the propensity score is included as a covariate in either forest, its weight will default to `1/ncol(X_train)`. A workaround if you wish to provide a custom weight for the propensity score is to include it as a column in `X_train` and then set `propensity_covariate` to `'none'` adjust `keep_vars_mu`, `keep_vars_tau` and `keep_vars_variance` accordingly.
#'   - `propensity_covariate` Whether to include the propensity score as a covariate in either or both of the forests. Enter `"none"` for neither, `"mu"` for the prognostic forest, `"tau"` for the treatment forest, and `"both"` for both forests. If this is not `"none"` and a propensity score is not provided, it will be estimated from (`X_train`, `Z_train`) using `stochtree::bart()`. Default: `"mu"`.
#'   - `adaptive_coding` Whether or not to use an "adaptive coding" scheme in which a binary treatment variable is not coded manually as (0,1) or (-1,1) but learned via parameters `b_0` and `b_1` that attach to the outcome model `[b_0 (1-Z) + b_1 Z] tau(X)`. This is ignored when Z is not binary. Default: `TRUE`.
#'   - `b_0` Initial value of the "control" group coding parameter. This is ignored when Z is not binary. Default: `-0.5`.
#'   - `b_1` Initial value of the "treatment" group coding parameter. This is ignored when Z is not binary. Default: `0.5`.
#'   - `random_seed` Integer parameterizing the C++ random number generator. If not specified, the C++ random number generator is seeded according to `std::random_device`.
#'   - `keep_burnin` Whether or not "burnin" samples should be included in cached predictions. Default `FALSE`. Ignored if `num_mcmc = 0`.
#'   - `keep_gfr` Whether or not "grow-from-root" samples should be included in cached predictions. Default `FALSE`. Ignored if `num_mcmc = 0`.
#'   - `standardize` Whether or not to standardize the outcome (and store the offset / scale in the model object). Default: `TRUE`.
#'   - `keep_every` How many iterations of the burned-in MCMC sampler should be run before forests and parameters are retained. Default `1`. Setting `keep_every <- k` for some `k > 1` will "thin" the MCMC samples by retaining every `k`-th sample, rather than simply every sample. This can reduce the autocorrelation of the MCMC samples.
#'   - `num_chains` How many independent MCMC chains should be sampled. If `num_mcmc = 0`, this is ignored. If `num_gfr = 0`, then each chain is run from root for `num_mcmc * keep_every + num_burnin` iterations, with `num_mcmc` samples retained. If `num_gfr > 0`, each MCMC chain will be initialized from a separate GFR ensemble, with the requirement that `num_gfr >= num_chains`. Default: `1`.
#'   - `verbose` Whether or not to print progress during the sampling loops. Default: `FALSE`.
#'   - `sample_sigma_global` Whether or not to update the `sigma^2` global error variance parameter based on `IG(a_global, b_global)`. Default: `TRUE`.
#' 
#'   **2. Prognostic Forest Parameters**
#'   
#'   - `num_trees_mu` Number of trees in the prognostic forest. Default: `200`.
#'   - `sample_sigma_leaf_mu` Whether or not to update the `sigma_leaf_mu` leaf scale variance parameter in the prognostic forest based on `IG(a_leaf_mu, b_leaf_mu)`. Default: `TRUE`.
#'
#'   **2.1. Tree Prior Parameters**
#'   
#'   - `alpha_mu` Prior probability of splitting for a tree of depth 0 for the prognostic forest. Tree split prior combines `alpha` and `beta` via `alpha_mu*(1+node_depth)^-beta_mu`. Default: `0.95`.
#'   - `beta_mu` Exponent that decreases split probabilities for nodes of depth > 0 for the prognostic forest. Tree split prior combines `alpha` and `beta` via `alpha_mu*(1+node_depth)^-beta_mu`. Default: `2.0`.
#'   - `min_samples_leaf_mu` Minimum allowable size of a leaf, in terms of training samples, for the prognostic forest. Default: `5`.
#'   - `max_depth_mu` Maximum depth of any tree in the mu ensemble. Default: `10`. Can be overridden with `-1` which does not enforce any depth limits on trees.
#'
#'   **2.2. Leaf Model Parameters**
#'   
#'   - `keep_vars_mu` Vector of variable names or column indices denoting variables that should be included in the prognostic (`mu(X)`) forest. Default: `NULL`.
#'   - `drop_vars_mu` Vector of variable names or column indices denoting variables that should be excluded from the prognostic (`mu(X)`) forest. Default: `NULL`. If both `drop_vars_mu` and `keep_vars_mu` are set, `drop_vars_mu` will be ignored.
#'   - `sigma_leaf_mu` Starting value of leaf node scale parameter for the prognostic forest. Calibrated internally as `1/num_trees_mu` if not set here.
#'   - `a_leaf_mu` Shape parameter in the `IG(a_leaf_mu, b_leaf_mu)` leaf node parameter variance model for the prognostic forest. Default: `3`.
#'   - `b_leaf_mu` Scale parameter in the `IG(a_leaf_mu, b_leaf_mu)` leaf node parameter variance model for the prognostic forest. Calibrated internally as `0.5/num_trees` if not set here.
#'
#'   **3. Treatment Effect Forest Parameters**
#'   
#'   - `num_trees_tau` Number of trees in the treatment effect forest. Default: `50`.
#'   - `sample_sigma_leaf_tau` Whether or not to update the `sigma_leaf_tau` leaf scale variance parameter in the treatment effect forest based on `IG(a_leaf_tau, b_leaf_tau)`. Default: `TRUE`.
#'
#'   **3.1. Tree Prior Parameters**
#'   
#'   - `alpha_tau` Prior probability of splitting for a tree of depth 0 for the treatment effect forest. Tree split prior combines `alpha` and `beta` via `alpha_tau*(1+node_depth)^-beta_tau`. Default: `0.25`.
#'   - `beta_tau` Exponent that decreases split probabilities for nodes of depth > 0 for the treatment effect forest. Tree split prior combines `alpha` and `beta` via `alpha_tau*(1+node_depth)^-beta_tau`. Default: `3.0`.
#'   - `min_samples_leaf_tau` Minimum allowable size of a leaf, in terms of training samples, for the treatment effect forest. Default: `5`.
#'   - `max_depth_tau` Maximum depth of any tree in the tau ensemble. Default: `5`. Can be overridden with `-1` which does not enforce any depth limits on trees.
#'
#'   **3.2. Leaf Model Parameters**
#'   
#'   - `a_leaf_tau` Shape parameter in the `IG(a_leaf, b_leaf)` leaf node parameter variance model for the treatment effect forest. Default: `3`.
#'   - `b_leaf_tau` Scale parameter in the `IG(a_leaf, b_leaf)` leaf node parameter variance model for the treatment effect forest. Calibrated internally as `0.5/num_trees` if not set here.
#'   - `keep_vars_tau` Vector of variable names or column indices denoting variables that should be included in the treatment effect (`tau(X)`) forest. Default: `NULL`.
#'   - `drop_vars_tau` Vector of variable names or column indices denoting variables that should be excluded from the treatment effect (`tau(X)`) forest. Default: `NULL`. If both `drop_vars_tau` and `keep_vars_tau` are set, `drop_vars_tau` will be ignored.
#'
#'   **4. Conditional Variance Forest Parameters**
#'   
#'   - `num_trees_variance` Number of trees in the (optional) conditional variance forest model. Default: `0`.
#'   - `variance_forest_init` Starting value of root forest prediction in conditional (heteroskedastic) error variance model. Calibrated internally as `log(pct_var_variance_forest_init*var((y-mean(y))/sd(y)))/num_trees_variance` if not set.
#'   - `pct_var_variance_forest_init` Percentage of standardized outcome variance used to initialize global error variance parameter. Default: `1`. Superseded by `variance_forest_init`.
#'
#'   **4.1. Tree Prior Parameters**
#'   
#'   - `alpha_variance` Prior probability of splitting for a tree of depth 0 in the (optional) conditional variance model. Tree split prior combines `alpha_variance` and `beta_variance` via `alpha_variance*(1+node_depth)^-beta_variance`. Default: `0.95`.
#'   - `beta_variance` Exponent that decreases split probabilities for nodes of depth > 0 in the (optional) conditional variance model. Tree split prior combines `alpha_variance` and `beta_variance` via `alpha_variance*(1+node_depth)^-beta_variance`. Default: `2.0`.
#'   - `min_samples_leaf_variance` Minimum allowable size of a leaf, in terms of training samples, in the (optional) conditional variance model. Default: `5`.
#'   - `max_depth_variance` Maximum depth of any tree in the ensemble in the (optional) conditional variance model. Default: `10`. Can be overridden with `-1` which does not enforce any depth limits on trees.
#'
#'   **4.2. Leaf Model Parameters**
#'   
#'   - `a_forest` Shape parameter in the `IG(a_forest, b_forest)` conditional error variance model (which is only sampled if `num_trees_variance > 0`). Calibrated internally as `num_trees_variance / 1.5^2 + 0.5` if not set.
#'   - `b_forest` Scale parameter in the `IG(a_forest, b_forest)` conditional error variance model (which is only sampled if `num_trees_variance > 0`). Calibrated internally as `num_trees_variance / 1.5^2` if not set.
#'   - `keep_vars_variance` Vector of variable names or column indices denoting variables that should be included in the (optional) conditional variance forest. Default: `NULL`.
#'   - `drop_vars_variance` Vector of variable names or column indices denoting variables that should be excluded from the (optional) conditional variance forest. Default: NULL. If both `drop_vars_variance` and `keep_vars_variance` are set, `drop_vars_variance` will be ignored.
#'   
#'   **5. Random Effects Parameters**
#'   
#'   - `rfx_prior_var` Prior on the (diagonals of the) covariance of the additive group-level random regression coefficients. Must be a vector of length `ncol(rfx_basis_train)`. Default: `rep(1, ncol(rfx_basis_train))`
#'
#' @return List of sampling outputs and a wrapper around the sampled forests (which can be used for in-memory prediction on new data, or serialized to JSON on disk).
#' @export
#'
#' @examples
#' n <- 500
#' x1 <- rnorm(n)
#' x2 <- rnorm(n)
#' x3 <- rnorm(n)
#' x4 <- as.numeric(rbinom(n,1,0.5))
#' x5 <- as.numeric(sample(1:3,n,replace=TRUE))
#' X <- cbind(x1,x2,x3,x4,x5)
#' p <- ncol(X)
#' g <- function(x) {ifelse(x[,5]==1,2,ifelse(x[,5]==2,-1,4))}
#' mu1 <- function(x) {1+g(x)+x[,1]*x[,3]}
#' mu2 <- function(x) {1+g(x)+6*abs(x[,3]-1)}
#' tau1 <- function(x) {rep(3,nrow(x))}
#' tau2 <- function(x) {1+2*x[,2]*x[,4]}
#' mu_x <- mu1(X)
#' tau_x <- tau2(X)
#' pi_x <- 0.8*pnorm((3*mu_x/sd(mu_x)) - 0.5*X[,1]) + 0.05 + runif(n)/10
#' Z <- rbinom(n,1,pi_x)
#' E_XZ <- mu_x + Z*tau_x
#' snr <- 4
#' y <- E_XZ + rnorm(n, 0, 1)*(sd(E_XZ)/snr)
#' X <- as.data.frame(X)
#' X$x4 <- factor(X$x4, ordered = TRUE)
#' X$x5 <- factor(X$x5, ordered = TRUE)
#' test_set_pct <- 0.2
#' n_test <- round(test_set_pct*n)
#' n_train <- n - n_test
#' test_inds <- sort(sample(1:n, n_test, replace = FALSE))
#' train_inds <- (1:n)[!((1:n) %in% test_inds)]
#' X_test <- X[test_inds,]
#' X_train <- X[train_inds,]
#' pi_test <- pi_x[test_inds]
#' pi_train <- pi_x[train_inds]
#' Z_test <- Z[test_inds]
#' Z_train <- Z[train_inds]
#' y_test <- y[test_inds]
#' y_train <- y[train_inds]
#' mu_test <- mu_x[test_inds]
#' mu_train <- mu_x[train_inds]
#' tau_test <- tau_x[test_inds]
#' tau_train <- tau_x[train_inds]
#' bcf_model <- bcf(X_train = X_train, Z_train = Z_train, y_train = y_train, pi_train = pi_train, 
#'                  X_test = X_test, Z_test = Z_test, pi_test = pi_test)
#' # plot(rowMeans(bcf_model$mu_hat_test), mu_test, xlab = "predicted", 
#' #      ylab = "actual", main = "Prognostic function")
#' # abline(0,1,col="red",lty=3,lwd=3)
#' # plot(rowMeans(bcf_model$tau_hat_test), tau_test, xlab = "predicted", 
#' #      ylab = "actual", main = "Treatment effect")
#' # abline(0,1,col="red",lty=3,lwd=3)
bcf <- function(X_train, Z_train, y_train, pi_train = NULL, group_ids_train = NULL, 
                rfx_basis_train = NULL, X_test = NULL, Z_test = NULL, pi_test = NULL, 
                group_ids_test = NULL, rfx_basis_test = NULL, num_gfr = 5, 
                num_burnin = 0, num_mcmc = 100, previous_model_json = NULL, 
                warmstart_sample_num = NULL, params = list()) {
    # Extract BCF parameters
    bcf_params <- preprocessBcfParams(params)
    cutpoint_grid_size <- bcf_params$cutpoint_grid_size
    sigma_leaf_mu <- bcf_params$sigma_leaf_mu
    sigma_leaf_tau <- bcf_params$sigma_leaf_tau
    alpha_mu <- bcf_params$alpha_mu
    alpha_tau <- bcf_params$alpha_tau
    alpha_variance <- bcf_params$alpha_variance
    beta_mu <- bcf_params$beta_mu
    beta_tau <- bcf_params$beta_tau
    beta_variance <- bcf_params$beta_variance
    min_samples_leaf_mu <- bcf_params$min_samples_leaf_mu
    min_samples_leaf_tau <- bcf_params$min_samples_leaf_tau
    min_samples_leaf_variance <- bcf_params$min_samples_leaf_variance
    max_depth_mu <- bcf_params$max_depth_mu
    max_depth_tau <- bcf_params$max_depth_tau
    max_depth_variance <- bcf_params$max_depth_variance
    a_global <- bcf_params$a_global
    b_global <- bcf_params$b_global
    a_leaf_mu <- bcf_params$a_leaf_mu
    a_leaf_tau <- bcf_params$a_leaf_tau
    b_leaf_mu <- bcf_params$b_leaf_mu
    b_leaf_tau <- bcf_params$b_leaf_tau
    a_forest <- bcf_params$a_forest
    b_forest <- bcf_params$b_forest
    sigma2_init <- bcf_params$sigma2_init
    variance_forest_init <- bcf_params$variance_forest_init
    pct_var_sigma2_init <- bcf_params$pct_var_sigma2_init
    pct_var_variance_forest_init <- bcf_params$pct_var_variance_forest_init
    variable_weights <- bcf_params$variable_weights
    keep_vars_mu <- bcf_params$keep_vars_mu
    drop_vars_mu <- bcf_params$drop_vars_mu
    keep_vars_tau <- bcf_params$keep_vars_tau
    drop_vars_tau <- bcf_params$drop_vars_tau
    keep_vars_variance <- bcf_params$keep_vars_variance
    drop_vars_variance <- bcf_params$drop_vars_variance
    num_trees_mu <- bcf_params$num_trees_mu
    num_trees_tau <- bcf_params$num_trees_tau
    num_trees_variance <- bcf_params$num_trees_variance
    sample_sigma_global <- bcf_params$sample_sigma_global
    sample_sigma_leaf_mu <- bcf_params$sample_sigma_leaf_mu
    sample_sigma_leaf_tau <- bcf_params$sample_sigma_leaf_tau
    propensity_covariate <- bcf_params$propensity_covariate
    adaptive_coding <- bcf_params$adaptive_coding
    b_0 <- bcf_params$b_0
    b_1 <- bcf_params$b_1
    rfx_prior_var <- bcf_params$rfx_prior_var
    random_seed <- bcf_params$random_seed
    keep_burnin <- bcf_params$keep_burnin
    keep_gfr <- bcf_params$keep_gfr
    standardize <- bcf_params$standardize
    keep_every <- bcf_params$keep_every
    num_chains <- bcf_params$num_chains
    verbose <- bcf_params$verbose
    
    # Override keep_gfr if there are no MCMC samples
    if (num_mcmc == 0) keep_gfr <- T
    
    # Check if previous model JSON is provided and parse it if so
    # TODO: check that warmstart_sample_num is <= the number of samples in this previous model
    has_prev_model <- !is.null(previous_model_json)
    if (has_prev_model) {
        previous_bcf_model <- createBCFModelFromJsonString(previous_model_json)
        previous_y_bar <- previous_bcf_model$model_params$outcome_mean
        previous_y_scale <- previous_bcf_model$model_params$outcome_scale
        previous_var_scale <- previous_bcf_model$model_params$variance_scale
        previous_forest_samples_mu <- previous_bcf_model$forests_mu
        previous_forest_samples_tau <- previous_bcf_model$forests_tau
        if (previous_bcf_model$model_params$include_variance_forest) {
            previous_forest_samples_variance <- previous_bcf_model$forests_variance
        } else previous_forest_samples_variance <- NULL
        if (previous_bcf_model$model_params$sample_sigma_global) {
            previous_global_var_samples <- previous_bcf_model$sigma2_samples*(
                previous_var_scale / (previous_y_scale*previous_y_scale)
            )
        } else previous_global_var_samples <- NULL
        if (previous_bcf_model$model_params$sample_sigma_leaf_mu) {
            previous_leaf_var_mu_samples <- previous_bcf_model$sigma_leaf_mu_samples
        } else previous_leaf_var_mu_samples <- NULL
        if (previous_bcf_model$model_params$sample_sigma_leaf_tau) {
            previous_leaf_var_tau_samples <- previous_bcf_model$sigma_leaf_tau_samples
        } else previous_leaf_var_tau_samples <- NULL
        if (previous_bcf_model$model_params$has_rfx) {
            previous_rfx_samples <- previous_bcf_model$rfx_samples
        } else previous_rfx_samples <- NULL
        if (previous_bcf_model$model_params$adaptive_coding) {
            previous_b_1_samples <- previous_bcf_model$b_1_samples
            previous_b_0_samples <- previous_bcf_model$b_0_samples
        } else {
            previous_b_1_samples <- NULL
            previous_b_0_samples <- NULL
        }
    } else {
        previous_y_bar <- NULL
        previous_y_scale <- NULL
        previous_var_scale <- NULL
        previous_global_var_samples <- NULL
        previous_leaf_var_mu_samples <- NULL
        previous_leaf_var_tau_samples <- NULL
        previous_rfx_samples <- NULL
        previous_forest_samples_mu <- NULL
        previous_forest_samples_tau <- NULL
        previous_forest_samples_variance <- NULL
        previous_b_1_samples <- NULL
        previous_b_0_samples <- NULL
    }
    
    # Determine whether conditional variance will be modeled
    if (num_trees_variance > 0) include_variance_forest = T
    else include_variance_forest = F

    # Set the variance forest priors if not set
    if (include_variance_forest) {
        a_0 <- 1.5
        if (is.null(a_forest)) a_forest <- num_trees_variance / (a_0^2) + 0.5
        if (is.null(b_forest)) b_forest <- num_trees_variance / (a_0^2)
    } else {
        a_forest <- 1.
        b_forest <- 1.
    }
    
    # Variable weight preprocessing (and initialization if necessary)
    if (is.null(variable_weights)) {
        variable_weights = rep(1/ncol(X_train), ncol(X_train))
    }
    if (any(variable_weights < 0)) {
        stop("variable_weights cannot have any negative weights")
    }
    
    # Check covariates are matrix or dataframe
    if ((!is.data.frame(X_train)) && (!is.matrix(X_train))) {
        stop("X_train must be a matrix or dataframe")
    }
    if (!is.null(X_test)){
        if ((!is.data.frame(X_test)) && (!is.matrix(X_test))) {
            stop("X_test must be a matrix or dataframe")
        }
    }
    num_cov_orig <- ncol(X_train)
    
    # Standardize the keep variable lists to numeric indices
    if (!is.null(keep_vars_mu)) {
        if (is.character(keep_vars_mu)) {
            if (!all(keep_vars_mu %in% names(X_train))) {
                stop("keep_vars_mu includes some variable names that are not in X_train")
            }
            variable_subset_mu <- unname(which(names(X_train) %in% keep_vars_mu))
        } else {
            if (any(keep_vars_mu > ncol(X_train))) {
                stop("keep_vars_mu includes some variable indices that exceed the number of columns in X_train")
            }
            if (any(keep_vars_mu < 0)) {
                stop("keep_vars_mu includes some negative variable indices")
            }
            variable_subset_mu <- keep_vars_mu
        }
    } else if ((is.null(keep_vars_mu)) && (!is.null(drop_vars_mu))) {
        if (is.character(drop_vars_mu)) {
            if (!all(drop_vars_mu %in% names(X_train))) {
                stop("drop_vars_mu includes some variable names that are not in X_train")
            }
            variable_subset_mu <- unname(which(!(names(X_train) %in% drop_vars_mu)))
        } else {
            if (any(drop_vars_mu > ncol(X_train))) {
                stop("drop_vars_mu includes some variable indices that exceed the number of columns in X_train")
            }
            if (any(drop_vars_mu < 0)) {
                stop("drop_vars_mu includes some negative variable indices")
            }
            variable_subset_mu <- (1:ncol(X_train))[!(1:ncol(X_train) %in% drop_vars_mu)]
        }
    } else {
        variable_subset_mu <- 1:ncol(X_train)
    }
    if (!is.null(keep_vars_tau)) {
        if (is.character(keep_vars_tau)) {
            if (!all(keep_vars_tau %in% names(X_train))) {
                stop("keep_vars_tau includes some variable names that are not in X_train")
            }
            variable_subset_tau <- unname(which(names(X_train) %in% keep_vars_tau))
        } else {
            if (any(keep_vars_tau > ncol(X_train))) {
                stop("keep_vars_tau includes some variable indices that exceed the number of columns in X_train")
            }
            if (any(keep_vars_tau < 0)) {
                stop("keep_vars_tau includes some negative variable indices")
            }
            variable_subset_tau <- keep_vars_tau
        }
    } else if ((is.null(keep_vars_tau)) && (!is.null(drop_vars_tau))) {
        if (is.character(drop_vars_tau)) {
            if (!all(drop_vars_tau %in% names(X_train))) {
                stop("drop_vars_tau includes some variable names that are not in X_train")
            }
            variable_subset_tau <- unname(which(!(names(X_train) %in% drop_vars_tau)))
        } else {
            if (any(drop_vars_tau > ncol(X_train))) {
                stop("drop_vars_tau includes some variable indices that exceed the number of columns in X_train")
            }
            if (any(drop_vars_tau < 0)) {
                stop("drop_vars_tau includes some negative variable indices")
            }
            variable_subset_tau <- (1:ncol(X_train))[!(1:ncol(X_train) %in% drop_vars_tau)]
        }
    } else {
        variable_subset_tau <- 1:ncol(X_train)
    }
    if (!is.null(keep_vars_variance)) {
        if (is.character(keep_vars_variance)) {
            if (!all(keep_vars_variance %in% names(X_train))) {
                stop("keep_vars_variance includes some variable names that are not in X_train")
            }
            variable_subset_variance <- unname(which(names(X_train) %in% keep_vars_variance))
        } else {
            if (any(keep_vars_variance > ncol(X_train))) {
                stop("keep_vars_variance includes some variable indices that exceed the number of columns in X_train")
            }
            if (any(keep_vars_variance < 0)) {
                stop("keep_vars_variance includes some negative variable indices")
            }
            variable_subset_variance <- keep_vars_variance
        }
    } else if ((is.null(keep_vars_variance)) && (!is.null(drop_vars_variance))) {
        if (is.character(drop_vars_variance)) {
            if (!all(drop_vars_variance %in% names(X_train))) {
                stop("drop_vars_variance includes some variable names that are not in X_train")
            }
            variable_subset_variance <- unname(which(!(names(X_train) %in% drop_vars_variance)))
        } else {
            if (any(drop_vars_variance > ncol(X_train))) {
                stop("drop_vars_variance includes some variable indices that exceed the number of columns in X_train")
            }
            if (any(drop_vars_variance < 0)) {
                stop("drop_vars_variance includes some negative variable indices")
            }
            variable_subset_variance <- (1:ncol(X_train))[!(1:ncol(X_train) %in% drop_vars_variance)]
        }
    } else {
        variable_subset_variance <- 1:ncol(X_train)
    }
    
    # Preprocess covariates
    train_cov_preprocess_list <- preprocessTrainData(X_train)
    X_train_metadata <- train_cov_preprocess_list$metadata
    X_train_raw <- X_train
    X_train <- train_cov_preprocess_list$data
    original_var_indices <- X_train_metadata$original_var_indices
    feature_types <- X_train_metadata$feature_types
    X_test_raw <- X_test
    if (!is.null(X_test)) X_test <- preprocessPredictionData(X_test, X_train_metadata)
    
    # Convert all input data to matrices if not already converted
    if ((is.null(dim(Z_train))) && (!is.null(Z_train))) {
        Z_train <- as.matrix(as.numeric(Z_train))
    }
    if ((is.null(dim(pi_train))) && (!is.null(pi_train))) {
        pi_train <- as.matrix(pi_train)
    }
    if ((is.null(dim(Z_test))) && (!is.null(Z_test))) {
        Z_test <- as.matrix(as.numeric(Z_test))
    }
    if ((is.null(dim(pi_test))) && (!is.null(pi_test))) {
        pi_test <- as.matrix(pi_test)
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
    
    # Check that outcome and treatment are numeric
    if (!is.numeric(y_train)) stop("y_train must be numeric")
    if (!is.numeric(Z_train)) stop("Z_train must be numeric")
    if (!is.null(Z_test)) {
        if (!is.numeric(Z_test)) stop("Z_test must be numeric")
    }

    # Data consistency checks
    if ((!is.null(X_test)) && (ncol(X_test) != ncol(X_train))) {
        stop("X_train and X_test must have the same number of columns")
    }
    if ((!is.null(Z_test)) && (ncol(Z_test) != ncol(Z_train))) {
        stop("Z_train and Z_test must have the same number of columns")
    }
    if ((!is.null(Z_train)) && (nrow(Z_train) != nrow(X_train))) {
        stop("Z_train and X_train must have the same number of rows")
    }
    if ((!is.null(pi_train)) && (nrow(pi_train) != nrow(X_train))) {
        stop("pi_train and X_train must have the same number of rows")
    }
    if ((!is.null(Z_test)) && (nrow(Z_test) != nrow(X_test))) {
        stop("Z_test and X_test must have the same number of rows")
    }
    if ((!is.null(pi_test)) && (nrow(pi_test) != nrow(X_test))) {
        stop("pi_test and X_test must have the same number of rows")
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

    # Random effects covariance prior
    if (has_rfx) {
        if (is.null(rfx_prior_var)) {
            rfx_prior_var <- rep(1, ncol(rfx_basis_train))
        } else {
            if ((!is.integer(rfx_prior_var)) && (!is.numeric(rfx_prior_var))) stop("rfx_prior_var must be a numeric vector")
            if (length(rfx_prior_var) != ncol(rfx_basis_train)) stop("length(rfx_prior_var) must equal ncol(rfx_basis_train)")
        }
    }
    
    # Update variable weights
    variable_weights_adj <- 1/sapply(original_var_indices, function(x) sum(original_var_indices == x))
    variable_weights <- variable_weights[original_var_indices]*variable_weights_adj
    
    # Create mu and tau (and variance) specific variable weights with weights zeroed out for excluded variables
    variable_weights_variance <- variable_weights_tau <- variable_weights_mu <- variable_weights
    variable_weights_mu[!(original_var_indices %in% variable_subset_mu)] <- 0
    variable_weights_tau[!(original_var_indices %in% variable_subset_tau)] <- 0
    if (include_variance_forest) {
        variable_weights_variance[!(original_var_indices %in% variable_subset_variance)] <- 0
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
    
    # Check that number of samples are all nonnegative
    stopifnot(num_gfr >= 0)
    stopifnot(num_burnin >= 0)
    stopifnot(num_mcmc >= 0)

    # Determine whether a test set is provided
    has_test = !is.null(X_test)
    
    # Convert y_train to numeric vector if not already converted
    if (!is.null(dim(y_train))) {
        y_train <- as.matrix(y_train)
    }
    
    # Check whether treatment is binary (specifically 0-1 binary)
    binary_treatment <- length(unique(Z_train)) == 2
    if (binary_treatment) {
        unique_treatments <- sort(unique(Z_train))
        if (!(all(unique_treatments == c(0,1)))) binary_treatment <- F
    }
    
    # Adaptive coding will be ignored for continuous / ordered categorical treatments
    if ((!binary_treatment) && (adaptive_coding)) {
        adaptive_coding <- F
    }
    
    # Check if propensity_covariate is one of the required inputs
    if (!(propensity_covariate %in% c("mu","tau","both","none"))) {
        stop("propensity_covariate must equal one of 'none', 'mu', 'tau', or 'both'")
    }
    
    # Estimate if pre-estimated propensity score is not provided
    if ((is.null(pi_train)) && (propensity_covariate != "none")) {
        # Estimate using the last of several iterations of GFR BART
        num_burnin <- 10
        num_total <- 50
        bart_model_propensity <- bart(X_train = X_train_raw, y_train = as.numeric(Z_train), X_test = X_test_raw, 
                                      num_gfr = num_total, num_burnin = 0, num_mcmc = 0)
        pi_train <- rowMeans(bart_model_propensity$y_hat_train[,(num_burnin+1):num_total])
        if ((is.null(dim(pi_train))) && (!is.null(pi_train))) {
            pi_train <- as.matrix(pi_train)
        }
        if (has_test) {
            pi_test <- rowMeans(bart_model_propensity$y_hat_test[,(num_burnin+1):num_total])
            if ((is.null(dim(pi_test))) && (!is.null(pi_test))) {
                pi_test <- as.matrix(pi_test)
            }
        }
    }

    if (has_test) {
        if (is.null(pi_test)) stop("Propensity score must be provided for the test set if provided for the training set")
    }
    
    # Update feature_types and covariates
    feature_types <- as.integer(feature_types)
    if (propensity_covariate != "none") {
        feature_types <- as.integer(c(feature_types,0))
        X_train <- cbind(X_train, pi_train)
        if (propensity_covariate == "mu") {
            variable_weights_mu <- c(variable_weights_mu, rep(1./num_cov_orig, ncol(pi_train)))
            variable_weights_tau <- c(variable_weights_tau, 0)
            if (include_variance_forest) variable_weights_variance <- c(variable_weights_variance, 0)
        } else if (propensity_covariate == "tau") {
            variable_weights_mu <- c(variable_weights_mu, 0)
            variable_weights_tau <- c(variable_weights_tau, rep(1./num_cov_orig, ncol(pi_train)))
            if (include_variance_forest) variable_weights_variance <- c(variable_weights_variance, 0)
        } else if (propensity_covariate == "both") {
            variable_weights_mu <- c(variable_weights_mu, rep(1./num_cov_orig, ncol(pi_train)))
            variable_weights_tau <- c(variable_weights_tau, rep(1./num_cov_orig, ncol(pi_train)))
            if (include_variance_forest) variable_weights_variance <- c(variable_weights_variance, 0)
        }
        if (has_test) X_test <- cbind(X_test, pi_test)
    }
    
    # Renormalize variable weights
    variable_weights_mu <- variable_weights_mu / sum(variable_weights_mu)
    variable_weights_tau <- variable_weights_tau / sum(variable_weights_tau)
    if (include_variance_forest) {
        variable_weights_variance <- variable_weights_variance / sum(variable_weights_variance)
    }
    
    # Standardize outcome separately for test and train
    if (standardize) {
        y_bar_train <- mean(y_train)
        y_std_train <- sd(y_train)
    } else {
        y_bar_train <- 0
        y_std_train <- 1
    }
    resid_train <- (y_train-y_bar_train)/y_std_train
    
    # Calibrate priors for global sigma^2 and sigma_leaf_mu / sigma_leaf_tau
    if (is.null(sigma2_init)) sigma2_init <- pct_var_sigma2_init*var(resid_train)
    if (is.null(variance_forest_init)) variance_forest_init <- pct_var_variance_forest_init*var(resid_train)
    if (is.null(b_leaf_mu)) b_leaf_mu <- var(resid_train)/(num_trees_mu)
    if (is.null(b_leaf_tau)) b_leaf_tau <- var(resid_train)/(2*num_trees_tau)
    if (is.null(sigma_leaf_mu)) sigma_leaf_mu <- var(resid_train)/(num_trees_mu)
    if (is.null(sigma_leaf_tau)) sigma_leaf_tau <- var(resid_train)/(2*num_trees_tau)
    current_sigma2 <- sigma2_init
    current_leaf_scale_mu <- as.matrix(sigma_leaf_mu)
    current_leaf_scale_tau <- as.matrix(sigma_leaf_tau)
    
    # Set variance leaf model type (currently only one option)
    leaf_model_variance_forest <- 3
    
    # Random effects prior parameters
    if (has_rfx) {
        # Initialize the working parameter to 1
        if (num_rfx_components < 1) {
            stop("There must be at least 1 random effect component")
        }
        alpha_init <- rep(1,num_rfx_components)
        # Initialize each group parameter based on a regression of outcome on basis in that grou
        xi_init <- matrix(0,num_rfx_components,num_rfx_groups)
        for (i in 1:num_rfx_groups) {
            group_subset_indices <- group_ids_train == i
            basis_group <- rfx_basis_train[group_subset_indices,]
            resid_group <- resid_train[group_subset_indices]
            rfx_group_model <- lm(resid_group ~ 0+basis_group)
            xi_init[,i] <- unname(coef(rfx_group_model))
        }
        sigma_alpha_init <- diag(1,num_rfx_components,num_rfx_components)
        sigma_xi_init <- diag(rfx_prior_var)
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
    num_actual_mcmc_iter <- num_mcmc * keep_every
    num_samples <- num_gfr + num_burnin + num_actual_mcmc_iter
    # Delete GFR samples from these containers after the fact if desired
    # num_retained_samples <- ifelse(keep_gfr, num_gfr, 0) + ifelse(keep_burnin, num_burnin, 0) + num_mcmc
    num_retained_samples <- num_gfr + ifelse(keep_burnin, num_burnin, 0) + num_mcmc * num_chains
    if (sample_sigma_global) global_var_samples <- rep(NA, num_retained_samples)
    if (sample_sigma_leaf_mu) leaf_scale_mu_samples <- rep(NA, num_retained_samples)
    if (sample_sigma_leaf_tau) leaf_scale_tau_samples <- rep(NA, num_retained_samples)
    sample_counter <- 0

    # Prepare adaptive coding structure
    if ((!is.numeric(b_0)) || (!is.numeric(b_1)) || (length(b_0) > 1) || (length(b_1) > 1)) {
        stop("b_0 and b_1 must be single numeric values")
    }
    if (adaptive_coding) {
        b_0_samples <- rep(NA, num_retained_samples)
        b_1_samples <- rep(NA, num_retained_samples)
        current_b_0 <- b_0
        current_b_1 <- b_1
        tau_basis_train <- (1-Z_train)*current_b_0 + Z_train*current_b_1
        if (has_test) tau_basis_test <- (1-Z_test)*current_b_0 + Z_test*current_b_1
    } else {
        tau_basis_train <- Z_train
        if (has_test) tau_basis_test <- Z_test
    }
    
    # Data
    forest_dataset_train <- createForestDataset(X_train, tau_basis_train)
    if (has_test) forest_dataset_test <- createForestDataset(X_test, tau_basis_test)
    outcome_train <- createOutcome(resid_train)
    
    # Random number generator (std::mt19937)
    if (is.null(random_seed)) random_seed = sample(1:10000,1,F)
    rng <- createRNG(random_seed)
    
    # Sampling data structures
    forest_model_mu <- createForestModel(forest_dataset_train, feature_types, num_trees_mu, nrow(X_train), alpha_mu, beta_mu, min_samples_leaf_mu, max_depth_mu)
    forest_model_tau <- createForestModel(forest_dataset_train, feature_types, num_trees_tau, nrow(X_train), alpha_tau, beta_tau, min_samples_leaf_tau, max_depth_tau)
    if (include_variance_forest) {
        forest_model_variance <- createForestModel(forest_dataset_train, feature_types, num_trees_variance, nrow(X_train), alpha_variance, beta_variance, min_samples_leaf_variance, max_depth_variance)
    }
    
    # Container of forest samples
    forest_samples_mu <- createForestContainer(num_trees_mu, 1, T)
    forest_samples_tau <- createForestContainer(num_trees_tau, 1, F)
    active_forest_mu <- createForest(num_trees_mu, 1, T)
    active_forest_tau <- createForest(num_trees_tau, 1, F)
    if (include_variance_forest) {
        forest_samples_variance <- createForestContainer(num_trees_variance, 1, TRUE, TRUE)
        active_forest_variance <- createForest(num_trees_variance, 1, TRUE, TRUE)
    }
    
    # Initialize the leaves of each tree in the prognostic forest
    init_mu <- mean(resid_train)
    active_forest_mu$prepare_for_sampler(forest_dataset_train, outcome_train, forest_model_mu, 0, init_mu)
    active_forest_mu$adjust_residual(forest_dataset_train, outcome_train, forest_model_mu, F, F)

    # Initialize the leaves of each tree in the treatment effect forest
    init_tau <- 0.
    active_forest_tau$prepare_for_sampler(forest_dataset_train, outcome_train, forest_model_tau, 1, init_tau)
    active_forest_tau$adjust_residual(forest_dataset_train, outcome_train, forest_model_tau, T, F)

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
                    cat("Sampling", i, "out of", num_gfr, "XBCF (grow-from-root) draws\n")
                }
            }
            
            # Sample the prognostic forest
            forest_model_mu$sample_one_iteration(
                forest_dataset_train, outcome_train, forest_samples_mu, active_forest_mu, 
                rng, feature_types, 0, current_leaf_scale_mu, variable_weights_mu, a_forest, b_forest, 
                current_sigma2, cutpoint_grid_size, keep_forest = keep_sample, gfr = T, pre_initialized = T
            )
            
            # Sample variance parameters (if requested)
            if (sample_sigma_global) {
                current_sigma2 <- sample_sigma2_one_iteration(outcome_train, forest_dataset_train, rng, a_global, b_global)
            }
            if (sample_sigma_leaf_mu) {
                leaf_scale_mu_double <- sample_tau_one_iteration(active_forest_mu, rng, a_leaf_mu, b_leaf_mu)
                current_leaf_scale_mu <- as.matrix(leaf_scale_mu_double)
                if (keep_sample) leaf_scale_mu_samples[sample_counter] <- leaf_scale_mu_double
            }
            
            # Sample the treatment forest
            forest_model_tau$sample_one_iteration(
                forest_dataset_train, outcome_train, forest_samples_tau, active_forest_tau,
                rng, feature_types, 1, current_leaf_scale_tau, variable_weights_tau, a_forest, b_forest, 
                current_sigma2, cutpoint_grid_size, keep_forest = keep_sample, gfr = T, pre_initialized = T
            )
            
            # Sample coding parameters (if requested)
            if (adaptive_coding) {
                # Estimate mu(X) and tau(X) and compute y - mu(X)
                mu_x_raw_train <- active_forest_mu$predict_raw(forest_dataset_train)
                tau_x_raw_train <- active_forest_tau$predict_raw(forest_dataset_train)
                partial_resid_mu_train <- resid_train - mu_x_raw_train
                if (has_rfx) {
                    rfx_preds_train <- rfx_model$predict(rfx_dataset_train, rfx_tracker_train)
                    partial_resid_mu_train <- partial_resid_mu_train - rfx_preds_train
                }
                
                # Compute sufficient statistics for regression of y - mu(X) on [tau(X)(1-Z), tau(X)Z]
                s_tt0 <- sum(tau_x_raw_train*tau_x_raw_train*(Z_train==0))
                s_tt1 <- sum(tau_x_raw_train*tau_x_raw_train*(Z_train==1))
                s_ty0 <- sum(tau_x_raw_train*partial_resid_mu_train*(Z_train==0))
                s_ty1 <- sum(tau_x_raw_train*partial_resid_mu_train*(Z_train==1))
                
                # Sample b0 (coefficient on tau(X)(1-Z)) and b1 (coefficient on tau(X)Z)
                current_b_0 <- rnorm(1, (s_ty0/(s_tt0 + 2*current_sigma2)), sqrt(current_sigma2/(s_tt0 + 2*current_sigma2)))
                current_b_1 <- rnorm(1, (s_ty1/(s_tt1 + 2*current_sigma2)), sqrt(current_sigma2/(s_tt1 + 2*current_sigma2)))
                
                # Update basis for the leaf regression
                tau_basis_train <- (1-Z_train)*current_b_0 + Z_train*current_b_1
                forest_dataset_train$update_basis(tau_basis_train)
                if (keep_sample) {
                    b_0_samples[sample_counter] <- current_b_0
                    b_1_samples[sample_counter] <- current_b_1
                }
                if (has_test) {
                    tau_basis_test <- (1-Z_test)*current_b_0 + Z_test*current_b_1
                    forest_dataset_test$update_basis(tau_basis_test)
                }
                
                # Update leaf predictions and residual
                forest_model_tau$propagate_basis_update(forest_dataset_train, outcome_train, active_forest_tau)
            }
            
            # Sample variance parameters (if requested)
            if (include_variance_forest) {
                forest_model_variance$sample_one_iteration(
                    forest_dataset_train, outcome_train, forest_samples_variance, active_forest_variance, 
                    rng, feature_types, leaf_model_variance_forest, current_leaf_scale_mu, variable_weights_variance, 
                    a_forest, b_forest, current_sigma2, cutpoint_grid_size, keep_forest = keep_sample, gfr = T, pre_initialized = T
                )
            }
            if (sample_sigma_global) {
                current_sigma2 <- sample_sigma2_one_iteration(outcome_train, forest_dataset_train, rng, a_global, b_global)
                if (keep_sample) global_var_samples[sample_counter] <- current_sigma2
            }
            if (sample_sigma_leaf_tau) {
                leaf_scale_tau_double <- sample_tau_one_iteration(active_forest_tau, rng, a_leaf_tau, b_leaf_tau)
                current_leaf_scale_tau <- as.matrix(leaf_scale_tau_double)
                if (keep_sample) leaf_scale_tau_samples[sample_counter] <- leaf_scale_tau_double
            }
            
            # Sample random effects parameters (if requested)
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
                resetActiveForest(active_forest_mu, forest_samples_mu, forest_ind)
                resetForestModel(forest_model_mu, active_forest_mu, forest_dataset_train, outcome_train, TRUE)
                resetActiveForest(active_forest_tau, forest_samples_tau, forest_ind)
                resetForestModel(forest_model_tau, active_forest_tau, forest_dataset_train, outcome_train, TRUE)
                if (sample_sigma_leaf_mu) {
                    leaf_scale_mu_double <- leaf_scale_mu_samples[forest_ind + 1]
                    current_leaf_scale_mu <- as.matrix(leaf_scale_mu_double)
                }
                if (sample_sigma_leaf_tau) {
                    leaf_scale_tau_double <- leaf_scale_tau_samples[forest_ind + 1]
                    current_leaf_scale_tau <- as.matrix(leaf_scale_tau_double)
                }
                if (include_variance_forest) {
                    resetActiveForest(active_forest_variance, forest_samples_variance, forest_ind)
                    resetForestModel(forest_model_variance, active_forest_variance, forest_dataset_train, outcome_train, FALSE)
                }
                if (has_rfx) {
                    resetRandomEffectsModel(rfx_model, rfx_samples, forest_ind, sigma_alpha_init)
                    resetRandomEffectsTracker(rfx_tracker_train, rfx_model, rfx_dataset_train, outcome_train, rfx_samples)
                }
                if (adaptive_coding) {
                    current_b_1 <- b_1_samples[forest_ind + 1]
                    current_b_0 <- b_0_samples[forest_ind + 1]
                    tau_basis_train <- (1-Z_train)*current_b_0 + Z_train*current_b_1
                    forest_dataset_train$update_basis(tau_basis_train)
                    if (has_test) {
                        tau_basis_test <- (1-Z_test)*current_b_0 + Z_test*current_b_1
                        forest_dataset_test$update_basis(tau_basis_test)
                    }
                    forest_model_tau$propagate_basis_update(forest_dataset_train, outcome_train, active_forest_tau)
                }
                if (sample_sigma_global) current_sigma2 <- global_var_samples[forest_ind + 1]
            } else if (has_prev_model) {
                resetActiveForest(active_forest_mu, previous_forest_samples_mu, warmstart_sample_num - 1)
                resetForestModel(forest_model_mu, active_forest_mu, forest_dataset_train, outcome_train, TRUE)
                resetActiveForest(active_forest_tau, previous_forest_samples_tau, warmstart_sample_num - 1)
                resetForestModel(forest_model_tau, active_forest_tau, forest_dataset_train, outcome_train, TRUE)
                if (include_variance_forest) {
                    resetActiveForest(active_forest_variance, previous_forest_samples_variance, warmstart_sample_num - 1)
                    resetForestModel(forest_model_variance, active_forest_variance, forest_dataset_train, outcome_train, FALSE)
                }
                if (sample_sigma_leaf_mu && (!is.null(previous_leaf_var_mu_samples))) {
                    leaf_scale_mu_double <- previous_leaf_var_mu_samples[warmstart_sample_num]
                    current_leaf_scale_mu <- as.matrix(leaf_scale_mu_double)
                }
                if (sample_sigma_leaf_tau && (!is.null(previous_leaf_var_tau_samples))) {
                    leaf_scale_tau_double <- previous_leaf_var_tau_samples[warmstart_sample_num]
                    current_leaf_scale_tau <- as.matrix(leaf_scale_tau_double)
                }
                if (adaptive_coding) {
                    if (!is.null(previous_b_1_samples)) {
                        current_b_1 <- previous_b_1_samples[warmstart_sample_num]
                    }
                    if (!is.null(previous_b_0_samples)) {
                        current_b_0 <- previous_b_0_samples[warmstart_sample_num]
                    }
                    tau_basis_train <- (1-Z_train)*current_b_0 + Z_train*current_b_1
                    forest_dataset_train$update_basis(tau_basis_train)
                    if (has_test) {
                        tau_basis_test <- (1-Z_test)*current_b_0 + Z_test*current_b_1
                        forest_dataset_test$update_basis(tau_basis_test)
                    }
                    forest_model_tau$propagate_basis_update(forest_dataset_train, outcome_train, active_forest_tau)
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
                rootResetActiveForest(active_forest_mu)
                active_forest_mu$set_root_leaves(init_mu / num_trees_mu)
                resetForestModel(forest_model_mu, active_forest_mu, forest_dataset_train, outcome_train, TRUE)
                rootResetActiveForest(active_forest_tau)
                active_forest_tau$set_root_leaves(init_tau / num_trees_tau)
                resetForestModel(forest_model_tau, active_forest_tau, forest_dataset_train, outcome_train, TRUE)
                if (sample_sigma_leaf_mu) {
                    current_leaf_scale_mu <- as.matrix(sigma_leaf_mu)
                }
                if (sample_sigma_leaf_tau) {
                    current_leaf_scale_tau <- as.matrix(sigma_leaf_tau)
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
                if (adaptive_coding) {
                    current_b_1 <- b_1
                    current_b_0 <- b_0
                    tau_basis_train <- (1-Z_train)*current_b_0 + Z_train*current_b_1
                    forest_dataset_train$update_basis(tau_basis_train)
                    if (has_test) {
                        tau_basis_test <- (1-Z_test)*current_b_0 + Z_test*current_b_1
                        forest_dataset_test$update_basis(tau_basis_test)
                    }
                    forest_model_tau$propagate_basis_update(forest_dataset_train, outcome_train, active_forest_tau)
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
                            cat("Sampling", i - num_gfr, "out of", num_gfr, "BCF burn-in draws\n")
                        }
                    }
                    if (num_mcmc > 0) {
                        if (((i - num_gfr - num_burnin) %% 100 == 0) || (i == num_samples)) {
                            cat("Sampling", i - num_burnin - num_gfr, "out of", num_mcmc, "BCF MCMC draws\n")
                        }
                    }
                }
                
                # Sample the prognostic forest
                forest_model_mu$sample_one_iteration(
                    forest_dataset_train, outcome_train, forest_samples_mu, active_forest_mu, 
                    rng, feature_types, 0, current_leaf_scale_mu, variable_weights_mu, a_forest, b_forest, 
                    current_sigma2, cutpoint_grid_size, keep_forest = keep_sample, gfr = F, pre_initialized = T
                )
                
                # Sample variance parameters (if requested)
                if (sample_sigma_global) {
                    current_sigma2 <- sample_sigma2_one_iteration(outcome_train, forest_dataset_train, rng, a_global, b_global)
                }
                if (sample_sigma_leaf_mu) {
                    leaf_scale_mu_double <- sample_tau_one_iteration(active_forest_mu, rng, a_leaf_mu, b_leaf_mu)
                    current_leaf_scale_mu <- as.matrix(leaf_scale_mu_double)
                    if (keep_sample) leaf_scale_mu_samples[sample_counter] <- leaf_scale_mu_double
                }
                
                # Sample the treatment forest
                forest_model_tau$sample_one_iteration(
                    forest_dataset_train, outcome_train, forest_samples_tau, active_forest_tau, 
                    rng, feature_types, 1, current_leaf_scale_tau, variable_weights_tau, a_forest, b_forest, 
                    current_sigma2, cutpoint_grid_size, keep_forest = keep_sample, gfr = F, pre_initialized = T
                )
                
                # Sample coding parameters (if requested)
                if (adaptive_coding) {
                    # Estimate mu(X) and tau(X) and compute y - mu(X)
                    mu_x_raw_train <- active_forest_mu$predict_raw(forest_dataset_train)
                    tau_x_raw_train <- active_forest_tau$predict_raw(forest_dataset_train)
                    partial_resid_mu_train <- resid_train - mu_x_raw_train
                    if (has_rfx) {
                        rfx_preds_train <- rfx_model$predict(rfx_dataset_train, rfx_tracker_train)
                        partial_resid_mu_train <- partial_resid_mu_train - rfx_preds_train
                    }
                    
                    # Compute sufficient statistics for regression of y - mu(X) on [tau(X)(1-Z), tau(X)Z]
                    s_tt0 <- sum(tau_x_raw_train*tau_x_raw_train*(Z_train==0))
                    s_tt1 <- sum(tau_x_raw_train*tau_x_raw_train*(Z_train==1))
                    s_ty0 <- sum(tau_x_raw_train*partial_resid_mu_train*(Z_train==0))
                    s_ty1 <- sum(tau_x_raw_train*partial_resid_mu_train*(Z_train==1))
                    
                    # Sample b0 (coefficient on tau(X)(1-Z)) and b1 (coefficient on tau(X)Z)
                    current_b_0 <- rnorm(1, (s_ty0/(s_tt0 + 2*current_sigma2)), sqrt(current_sigma2/(s_tt0 + 2*current_sigma2)))
                    current_b_1 <- rnorm(1, (s_ty1/(s_tt1 + 2*current_sigma2)), sqrt(current_sigma2/(s_tt1 + 2*current_sigma2)))
                    
                    # Update basis for the leaf regression
                    tau_basis_train <- (1-Z_train)*current_b_0 + Z_train*current_b_1
                    forest_dataset_train$update_basis(tau_basis_train)
                    if (keep_sample) {
                        b_0_samples[sample_counter] <- current_b_0
                        b_1_samples[sample_counter] <- current_b_1
                    }
                    if (has_test) {
                        tau_basis_test <- (1-Z_test)*current_b_0 + Z_test*current_b_1
                        forest_dataset_test$update_basis(tau_basis_test)
                    }
                    
                    # Update leaf predictions and residual
                    forest_model_tau$propagate_basis_update(forest_dataset_train, outcome_train, active_forest_tau)
                }
                
                # Sample variance parameters (if requested)
                if (include_variance_forest) {
                    forest_model_variance$sample_one_iteration(
                        forest_dataset_train, outcome_train, forest_samples_variance, active_forest_variance, 
                        rng, feature_types, leaf_model_variance_forest, current_leaf_scale_mu, variable_weights_variance, 
                        a_forest, b_forest, current_sigma2, cutpoint_grid_size, keep_forest = keep_sample, gfr = F, pre_initialized = T
                    )
                }
                if (sample_sigma_global) {
                    current_sigma2 <- sample_sigma2_one_iteration(outcome_train, forest_dataset_train, rng, a_global, b_global)
                    if (keep_sample) global_var_samples[sample_counter] <- current_sigma2
                }
                if (sample_sigma_leaf_tau) {
                    leaf_scale_tau_double <- sample_tau_one_iteration(active_forest_tau, rng, a_leaf_tau, b_leaf_tau)
                    current_leaf_scale_tau <- as.matrix(leaf_scale_tau_double)
                    if (keep_sample) leaf_scale_tau_samples[sample_counter] <- leaf_scale_tau_double
                }
                
                # Sample random effects parameters (if requested)
                if (has_rfx) {
                    rfx_model$sample_random_effect(rfx_dataset_train, outcome_train, rfx_tracker_train, rfx_samples, keep_sample, current_sigma2, rng)
                }
            }
        }
    }
    
    # Remove GFR samples if they are not to be retained
    if ((!keep_gfr) && (num_gfr > 0)) {
        for (i in 1:num_gfr) {
            forest_samples_mu$delete_sample(i-1)
            forest_samples_tau$delete_sample(i-1)
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
        if (sample_sigma_leaf_mu) {
            leaf_scale_mu_samples <- leaf_scale_mu_samples[(num_gfr+1):length(leaf_scale_mu_samples)]
        }
        if (sample_sigma_leaf_tau) {
            leaf_scale_tau_samples <- leaf_scale_tau_samples[(num_gfr+1):length(leaf_scale_tau_samples)]
        }
        num_retained_samples <- num_retained_samples - num_gfr
    }

    # Forest predictions
    mu_hat_train <- forest_samples_mu$predict(forest_dataset_train)*y_std_train + y_bar_train
    if (adaptive_coding) {
        tau_hat_train_raw <- forest_samples_tau$predict_raw(forest_dataset_train)
        tau_hat_train <- t(t(tau_hat_train_raw) * (b_1_samples - b_0_samples))*y_std_train
    } else {
        tau_hat_train <- forest_samples_tau$predict_raw(forest_dataset_train)*y_std_train
    }
    y_hat_train <- mu_hat_train + tau_hat_train * as.numeric(Z_train)
    if (has_test) {
        mu_hat_test <- forest_samples_mu$predict(forest_dataset_test)*y_std_train + y_bar_train
        if (adaptive_coding) {
            tau_hat_test_raw <- forest_samples_tau$predict_raw(forest_dataset_test)
            tau_hat_test <- t(t(tau_hat_test_raw) * (b_1_samples - b_0_samples))*y_std_train
        } else {
            tau_hat_test <- forest_samples_tau$predict_raw(forest_dataset_test)*y_std_train
        }
        y_hat_test <- mu_hat_test + tau_hat_test * as.numeric(Z_test)
    }
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
    
    # Global error variance
    if (sample_sigma_global) sigma2_samples <- global_var_samples*(y_std_train^2)
    
    # Leaf parameter variance for prognostic forest
    if (sample_sigma_leaf_mu) sigma_leaf_mu_samples <- leaf_scale_mu_samples
    
    # Leaf parameter variance for treatment effect forest
    if (sample_sigma_leaf_tau) sigma_leaf_tau_samples <- leaf_scale_tau_samples
    
    # Rescale variance forest prediction by global sigma2 (sampled or constant)
    if (include_variance_forest) {
        if (sample_sigma_global) {
            sigma_x_hat_train <- sapply(1:num_retained_samples, function(i) sqrt(sigma_x_hat_train[,i]*sigma2_samples[i]))
            if (has_test) sigma_x_hat_test <- sapply(1:num_retained_samples, function(i) sqrt(sigma_x_hat_test[,i]*sigma2_samples[i]))
        } else {
            sigma_x_hat_train <- sqrt(sigma_x_hat_train*sigma2_init)*y_std_train
            if (has_test) sigma_x_hat_test <- sqrt(sigma_x_hat_test*sigma2_init)*y_std_train
        }
    }
    
    # Return results as a list
    if (include_variance_forest) {
        num_variance_covariates <- sum(variable_weights_variance > 0)
    } else {
        num_variance_covariates <- 0
    }
    model_params <- list(
        "initial_sigma2" = sigma2_init, 
        "initial_sigma_leaf_mu" = sigma_leaf_mu,
        "initial_sigma_leaf_tau" = sigma_leaf_tau,
        "initial_b_0" = b_0,
        "initial_b_1" = b_1,
        "a_global" = a_global,
        "b_global" = b_global,
        "a_leaf_mu" = a_leaf_mu, 
        "b_leaf_mu" = b_leaf_mu,
        "a_leaf_tau" = a_leaf_tau, 
        "b_leaf_tau" = b_leaf_tau,
        "a_forest" = a_forest, 
        "b_forest" = b_forest,
        "outcome_mean" = y_bar_train,
        "outcome_scale" = y_std_train,
        "standardize" = standardize, 
        "num_covariates" = num_cov_orig,
        "num_prognostic_covariates" = sum(variable_weights_mu > 0),
        "num_treatment_covariates" = sum(variable_weights_tau > 0),
        "num_variance_covariates" = num_variance_covariates,
        "treatment_dim" = ncol(Z_train), 
        "propensity_covariate" = propensity_covariate, 
        "binary_treatment" = binary_treatment, 
        "adaptive_coding" = adaptive_coding, 
        "num_samples" = num_retained_samples, 
        "num_gfr" = num_gfr, 
        "num_burnin" = num_burnin, 
        "num_mcmc" = num_mcmc, 
        "keep_every" = keep_every,
        "num_chains" = num_chains,
        "has_rfx" = has_rfx, 
        "has_rfx_basis" = has_basis_rfx, 
        "num_rfx_basis" = num_basis_rfx, 
        "include_variance_forest" = include_variance_forest, 
        "sample_sigma_global" = sample_sigma_global,
        "sample_sigma_leaf_mu" = sample_sigma_leaf_mu,
        "sample_sigma_leaf_tau" = sample_sigma_leaf_tau
    )
    result <- list(
        "forests_mu" = forest_samples_mu, 
        "forests_tau" = forest_samples_tau, 
        "model_params" = model_params, 
        "mu_hat_train" = mu_hat_train, 
        "tau_hat_train" = tau_hat_train, 
        "y_hat_train" = y_hat_train, 
        "train_set_metadata" = X_train_metadata
    )
    if (has_test) result[["mu_hat_test"]] = mu_hat_test
    if (has_test) result[["tau_hat_test"]] = tau_hat_test
    if (has_test) result[["y_hat_test"]] = y_hat_test
    if (include_variance_forest) {
        result[["forests_variance"]] = forest_samples_variance
        result[["sigma_x_hat_train"]] = sigma_x_hat_train
        if (has_test) result[["sigma_x_hat_test"]] = sigma_x_hat_test
    }
    if (sample_sigma_global) result[["sigma2_samples"]] = sigma2_samples
    if (sample_sigma_leaf_mu) result[["sigma_leaf_mu_samples"]] = sigma_leaf_mu_samples
    if (sample_sigma_leaf_tau) result[["sigma_leaf_tau_samples"]] = sigma_leaf_tau_samples
    if (adaptive_coding) {
        result[["b_0_samples"]] = b_0_samples
        result[["b_1_samples"]] = b_1_samples
    }
    if (has_rfx) {
        result[["rfx_samples"]] = rfx_samples
        result[["rfx_preds_train"]] = rfx_preds_train
        result[["rfx_unique_group_ids"]] = levels(group_ids_factor)
    }
    if ((has_rfx_test) && (has_test)) result[["rfx_preds_test"]] = rfx_preds_test
    class(result) <- "bcf"
    
    return(result)
}

#' Predict from a sampled BCF model on new data
#'
#' @param bcf Object of type `bcf` containing draws of a Bayesian causal forest model and associated sampling outputs.
#' @param X_test Covariates used to determine tree leaf predictions for each observation. Must be passed as a matrix or dataframe.
#' @param Z_test Treatments used for prediction.
#' @param pi_test (Optional) Propensities used for prediction.
#' @param group_ids_test (Optional) Test set group labels used for an additive random effects model. 
#' We do not currently support (but plan to in the near future), test set evaluation for group labels
#' that were not in the training set.
#' @param rfx_basis_test (Optional) Test set basis for "random-slope" regression in additive random effects model.
#'
#' @return List of 3-5 `nrow(X_test)` by `bcf$num_samples` matrices: prognostic function estimates, treatment effect estimates, (optionally) random effects predictions, (optionally) variance forest predictions, and outcome predictions.
#' @export
#'
#' @examples
#' n <- 500
#' x1 <- rnorm(n)
#' x2 <- rnorm(n)
#' x3 <- rnorm(n)
#' x4 <- as.numeric(rbinom(n,1,0.5))
#' x5 <- as.numeric(sample(1:3,n,replace=TRUE))
#' X <- cbind(x1,x2,x3,x4,x5)
#' p <- ncol(X)
#' g <- function(x) {ifelse(x[,5]==1,2,ifelse(x[,5]==2,-1,4))}
#' mu1 <- function(x) {1+g(x)+x[,1]*x[,3]}
#' mu2 <- function(x) {1+g(x)+6*abs(x[,3]-1)}
#' tau1 <- function(x) {rep(3,nrow(x))}
#' tau2 <- function(x) {1+2*x[,2]*x[,4]}
#' mu_x <- mu1(X)
#' tau_x <- tau2(X)
#' pi_x <- 0.8*pnorm((3*mu_x/sd(mu_x)) - 0.5*X[,1]) + 0.05 + runif(n)/10
#' Z <- rbinom(n,1,pi_x)
#' E_XZ <- mu_x + Z*tau_x
#' snr <- 4
#' y <- E_XZ + rnorm(n, 0, 1)*(sd(E_XZ)/snr)
#' X <- as.data.frame(X)
#' X$x4 <- factor(X$x4, ordered = TRUE)
#' X$x5 <- factor(X$x5, ordered = TRUE)
#' test_set_pct <- 0.2
#' n_test <- round(test_set_pct*n)
#' n_train <- n - n_test
#' test_inds <- sort(sample(1:n, n_test, replace = FALSE))
#' train_inds <- (1:n)[!((1:n) %in% test_inds)]
#' X_test <- X[test_inds,]
#' X_train <- X[train_inds,]
#' pi_test <- pi_x[test_inds]
#' pi_train <- pi_x[train_inds]
#' Z_test <- Z[test_inds]
#' Z_train <- Z[train_inds]
#' y_test <- y[test_inds]
#' y_train <- y[train_inds]
#' mu_test <- mu_x[test_inds]
#' mu_train <- mu_x[train_inds]
#' tau_test <- tau_x[test_inds]
#' tau_train <- tau_x[train_inds]
#' bcf_model <- bcf(X_train = X_train, Z_train = Z_train, y_train = y_train, pi_train = pi_train)
#' preds <- predict(bcf_model, X_test, Z_test, pi_test)
#' # plot(rowMeans(preds$mu_hat), mu_test, xlab = "predicted",
#' #      ylab = "actual", main = "Prognostic function")
#' # abline(0,1,col="red",lty=3,lwd=3)
#' # plot(rowMeans(preds$tau_hat), tau_test, xlab = "predicted", 
#' #      ylab = "actual", main = "Treatment effect")
#' # abline(0,1,col="red",lty=3,lwd=3)
predict.bcf <- function(bcf, X_test, Z_test, pi_test = NULL, group_ids_test = NULL, rfx_basis_test = NULL){
    # Preprocess covariates
    if ((!is.data.frame(X_test)) && (!is.matrix(X_test))) {
        stop("X_test must be a matrix or dataframe")
    }
    train_set_metadata <- bcf$train_set_metadata
    X_test <- preprocessPredictionData(X_test, train_set_metadata)
    
    # Convert all input data to matrices if not already converted
    if ((is.null(dim(Z_test))) && (!is.null(Z_test))) {
        Z_test <- as.matrix(as.numeric(Z_test))
    }
    if ((is.null(dim(pi_test))) && (!is.null(pi_test))) {
        pi_test <- as.matrix(pi_test)
    }
    if ((is.null(dim(rfx_basis_test))) && (!is.null(rfx_basis_test))) {
        rfx_basis_test <- as.matrix(rfx_basis_test)
    }
    
    # Data checks
    if ((bcf$model_params$propensity_covariate != "none") && (is.null(pi_test))) {
        stop("pi_test must be provided for this model")
    }
    if (nrow(X_test) != nrow(Z_test)) {
        stop("X_test and Z_test must have the same number of rows")
    }
    if (bcf$model_params$num_covariates != ncol(X_test)) {
        stop("X_test and must have the same number of columns as the covariates used to train the model")
    }
    if ((bcf$model_params$has_rfx) && (is.null(group_ids_test))) {
        stop("Random effect group labels (group_ids_test) must be provided for this model")
    }
    if ((bcf$model_params$has_rfx_basis) && (is.null(rfx_basis_test))) {
        stop("Random effects basis (rfx_basis_test) must be provided for this model")
    }
    if ((bcf$model_params$num_rfx_basis > 0) && (ncol(rfx_basis_test) != bcf$model_params$num_rfx_basis)) {
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
    if ((bcf$model_params$has_rfx) && (is.null(rfx_basis_test))) {
        rfx_basis_test <- matrix(rep(1, nrow(X_test)), ncol = 1)
    }
    
    # Add propensities to any covariate set
    if (bcf$model_params$propensity_covariate == "both") {
        X_test_mu <- cbind(X_test, pi_test)
        X_test_tau <- cbind(X_test, pi_test)
    } else if (bcf$model_params$propensity_covariate == "mu") {
        X_test_mu <- cbind(X_test, pi_test)
        X_test_tau <- X_test
    } else if (bcf$model_params$propensity_covariate == "tau") {
        X_test_mu <- X_test
        X_test_tau <- cbind(X_test, pi_test)
    }
    
    # Create prediction datasets
    prediction_dataset_mu <- createForestDataset(X_test_mu)
    prediction_dataset_tau <- createForestDataset(X_test_tau, Z_test)

    # Compute forest predictions
    num_samples <- bcf$model_params$num_samples
    y_std <- bcf$model_params$outcome_scale
    y_bar <- bcf$model_params$outcome_mean
    initial_sigma2 <- bcf$model_params$initial_sigma2
    mu_hat_test <- bcf$forests_mu$predict(prediction_dataset_mu)*y_std + y_bar
    if (bcf$model_params$adaptive_coding) {
        tau_hat_test_raw <- bcf$forests_tau$predict_raw(prediction_dataset_tau)
        tau_hat_test <- t(t(tau_hat_test_raw) * (bcf$b_1_samples - bcf$b_0_samples))*y_std
    } else {
        tau_hat_test <- bcf$forests_tau$predict_raw(prediction_dataset_tau)*y_std
    }
    if (bcf$model_params$include_variance_forest) {
        s_x_raw <- bcf$variance_forests$predict(prediction_dataset)
    }
    
    # Compute rfx predictions (if needed)
    if (bcf$model_params$has_rfx) {
        rfx_predictions <- bcf$rfx_samples$predict(group_ids_test, rfx_basis_test)*y_std
    }
    
    # Compute overall "y_hat" predictions
    y_hat_test <- mu_hat_test + tau_hat_test * as.numeric(Z_test)
    if (bcf$model_params$has_rfx) y_hat_test <- y_hat_test + rfx_predictions
    
    # Scale variance forest predictions
    if (bcf$model_params$include_variance_forest) {
        if (bcf$model_params$sample_sigma_global) {
            sigma2_samples <- bcf$sigma2_global_samples
            variance_forest_predictions <- sapply(1:num_samples, function(i) sqrt(s_x_raw[,i]*sigma2_samples[i]))
        } else {
            variance_forest_predictions <- sqrt(s_x_raw*initial_sigma2)*y_std
        }
    }

    result <- list(
        "mu_hat" = mu_hat_test, 
        "tau_hat" = tau_hat_test, 
        "y_hat" = y_hat_test
    )
    if (bcf$model_params$has_rfx) {
        result[["rfx_predictions"]] = rfx_predictions
    }
    if (bcf$model_params$include_variance_forest) {
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
#' n <- 500
#' x1 <- rnorm(n)
#' x2 <- rnorm(n)
#' x3 <- rnorm(n)
#' x4 <- as.numeric(rbinom(n,1,0.5))
#' x5 <- as.numeric(sample(1:3,n,replace=TRUE))
#' X <- cbind(x1,x2,x3,x4,x5)
#' p <- ncol(X)
#' g <- function(x) {ifelse(x[,5]==1,2,ifelse(x[,5]==2,-1,4))}
#' mu1 <- function(x) {1+g(x)+x[,1]*x[,3]}
#' mu2 <- function(x) {1+g(x)+6*abs(x[,3]-1)}
#' tau1 <- function(x) {rep(3,nrow(x))}
#' tau2 <- function(x) {1+2*x[,2]*x[,4]}
#' mu_x <- mu1(X)
#' tau_x <- tau2(X)
#' pi_x <- 0.8*pnorm((3*mu_x/sd(mu_x)) - 0.5*X[,1]) + 0.05 + runif(n)/10
#' Z <- rbinom(n,1,pi_x)
#' E_XZ <- mu_x + Z*tau_x
#' snr <- 3
#' group_ids <- rep(c(1,2), n %/% 2)
#' rfx_coefs <- matrix(c(-1, -1, 1, 1), nrow=2, byrow=TRUE)
#' rfx_basis <- cbind(1, runif(n, -1, 1))
#' rfx_term <- rowSums(rfx_coefs[group_ids,] * rfx_basis)
#' y <- E_XZ + rfx_term + rnorm(n, 0, 1)*(sd(E_XZ)/snr)
#' X <- as.data.frame(X)
#' X$x4 <- factor(X$x4, ordered = TRUE)
#' X$x5 <- factor(X$x5, ordered = TRUE)
#' test_set_pct <- 0.2
#' n_test <- round(test_set_pct*n)
#' n_train <- n - n_test
#' test_inds <- sort(sample(1:n, n_test, replace = FALSE))
#' train_inds <- (1:n)[!((1:n) %in% test_inds)]
#' X_test <- X[test_inds,]
#' X_train <- X[train_inds,]
#' pi_test <- pi_x[test_inds]
#' pi_train <- pi_x[train_inds]
#' Z_test <- Z[test_inds]
#' Z_train <- Z[train_inds]
#' y_test <- y[test_inds]
#' y_train <- y[train_inds]
#' mu_test <- mu_x[test_inds]
#' mu_train <- mu_x[train_inds]
#' tau_test <- tau_x[test_inds]
#' tau_train <- tau_x[train_inds]
#' group_ids_test <- group_ids[test_inds]
#' group_ids_train <- group_ids[train_inds]
#' rfx_basis_test <- rfx_basis[test_inds,]
#' rfx_basis_train <- rfx_basis[train_inds,]
#' rfx_term_test <- rfx_term[test_inds]
#' rfx_term_train <- rfx_term[train_inds]
#' bcf_params <- list(sample_sigma_leaf_mu = TRUE, sample_sigma_leaf_tau = FALSE)
#' bcf_model <- bcf(X_train = X_train, Z_train = Z_train, y_train = y_train, 
#'                  pi_train = pi_train, group_ids_train = group_ids_train, 
#'                  rfx_basis_train = rfx_basis_train, X_test = X_test, 
#'                  Z_test = Z_test, pi_test = pi_test, group_ids_test = group_ids_test,
#'                  rfx_basis_test = rfx_basis_test, 
#'                  num_gfr = 100, num_burnin = 0, num_mcmc = 100, 
#'                  params = bcf_params)
#' rfx_samples <- getRandomEffectSamples(bcf_model)
getRandomEffectSamples.bcf <- function(object, ...){
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

#' Convert the persistent aspects of a BCF model to (in-memory) JSON
#'
#' @param object Object of type `bcf` containing draws of a Bayesian causal forest model and associated sampling outputs.
#'
#' @return Object of type `CppJson`
#' @export
#'
#' @examples
#' n <- 500
#' x1 <- rnorm(n)
#' x2 <- rnorm(n)
#' x3 <- rnorm(n)
#' x4 <- as.numeric(rbinom(n,1,0.5))
#' x5 <- as.numeric(sample(1:3,n,replace=TRUE))
#' X <- cbind(x1,x2,x3,x4,x5)
#' p <- ncol(X)
#' g <- function(x) {ifelse(x[,5]==1,2,ifelse(x[,5]==2,-1,4))}
#' mu1 <- function(x) {1+g(x)+x[,1]*x[,3]}
#' mu2 <- function(x) {1+g(x)+6*abs(x[,3]-1)}
#' tau1 <- function(x) {rep(3,nrow(x))}
#' tau2 <- function(x) {1+2*x[,2]*x[,4]}
#' mu_x <- mu1(X)
#' tau_x <- tau2(X)
#' pi_x <- 0.8*pnorm((3*mu_x/sd(mu_x)) - 0.5*X[,1]) + 0.05 + runif(n)/10
#' Z <- rbinom(n,1,pi_x)
#' E_XZ <- mu_x + Z*tau_x
#' snr <- 3
#' group_ids <- rep(c(1,2), n %/% 2)
#' rfx_coefs <- matrix(c(-1, -1, 1, 1), nrow=2, byrow=TRUE)
#' rfx_basis <- cbind(1, runif(n, -1, 1))
#' rfx_term <- rowSums(rfx_coefs[group_ids,] * rfx_basis)
#' y <- E_XZ + rfx_term + rnorm(n, 0, 1)*(sd(E_XZ)/snr)
#' X <- as.data.frame(X)
#' X$x4 <- factor(X$x4, ordered = TRUE)
#' X$x5 <- factor(X$x5, ordered = TRUE)
#' test_set_pct <- 0.2
#' n_test <- round(test_set_pct*n)
#' n_train <- n - n_test
#' test_inds <- sort(sample(1:n, n_test, replace = FALSE))
#' train_inds <- (1:n)[!((1:n) %in% test_inds)]
#' X_test <- X[test_inds,]
#' X_train <- X[train_inds,]
#' pi_test <- pi_x[test_inds]
#' pi_train <- pi_x[train_inds]
#' Z_test <- Z[test_inds]
#' Z_train <- Z[train_inds]
#' y_test <- y[test_inds]
#' y_train <- y[train_inds]
#' mu_test <- mu_x[test_inds]
#' mu_train <- mu_x[train_inds]
#' tau_test <- tau_x[test_inds]
#' tau_train <- tau_x[train_inds]
#' group_ids_test <- group_ids[test_inds]
#' group_ids_train <- group_ids[train_inds]
#' rfx_basis_test <- rfx_basis[test_inds,]
#' rfx_basis_train <- rfx_basis[train_inds,]
#' rfx_term_test <- rfx_term[test_inds]
#' rfx_term_train <- rfx_term[train_inds]
#' bcf_params <- list(sample_sigma_leaf_mu = TRUE, sample_sigma_leaf_tau = FALSE)
#' bcf_model <- bcf(X_train = X_train, Z_train = Z_train, y_train = y_train, 
#'                  pi_train = pi_train, group_ids_train = group_ids_train, 
#'                  rfx_basis_train = rfx_basis_train, X_test = X_test, 
#'                  Z_test = Z_test, pi_test = pi_test, group_ids_test = group_ids_test,
#'                  rfx_basis_test = rfx_basis_test, 
#'                  num_gfr = 100, num_burnin = 0, num_mcmc = 100, 
#'                  params = bcf_params)
#' # bcf_json <- convertBCFModelToJson(bcf_model)
convertBCFModelToJson <- function(object){
    jsonobj <- createCppJson()
    
    if (is.null(object$model_params)) {
        stop("This BCF model has not yet been sampled")
    }

    # Add the forests
    jsonobj$add_forest(object$forests_mu)
    jsonobj$add_forest(object$forests_tau)
    if (object$model_params$include_variance_forest) {
        jsonobj$add_forest(object$forests_variance)
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
    jsonobj$add_scalar("outcome_scale", object$model_params$outcome_scale)
    jsonobj$add_scalar("outcome_mean", object$model_params$outcome_mean)
    jsonobj$add_boolean("standardize", object$model_params$standardize)
    jsonobj$add_scalar("initial_sigma2", object$model_params$initial_sigma2)
    jsonobj$add_boolean("sample_sigma_global", object$model_params$sample_sigma_global)
    jsonobj$add_boolean("sample_sigma_leaf_mu", object$model_params$sample_sigma_leaf_mu)
    jsonobj$add_boolean("sample_sigma_leaf_tau", object$model_params$sample_sigma_leaf_tau)
    jsonobj$add_boolean("include_variance_forest", object$model_params$include_variance_forest)
    jsonobj$add_string("propensity_covariate", object$model_params$propensity_covariate)
    jsonobj$add_boolean("has_rfx", object$model_params$has_rfx)
    jsonobj$add_boolean("has_rfx_basis", object$model_params$has_rfx_basis)
    jsonobj$add_scalar("num_rfx_basis", object$model_params$num_rfx_basis)
    jsonobj$add_boolean("adaptive_coding", object$model_params$adaptive_coding)
    jsonobj$add_scalar("num_gfr", object$model_params$num_gfr)
    jsonobj$add_scalar("num_burnin", object$model_params$num_burnin)
    jsonobj$add_scalar("num_mcmc", object$model_params$num_mcmc)
    jsonobj$add_scalar("num_samples", object$model_params$num_samples)
    jsonobj$add_scalar("keep_every", object$model_params$keep_every)
    jsonobj$add_scalar("num_chains", object$model_params$num_chains)
    jsonobj$add_scalar("num_covariates", object$model_params$num_covariates)
    if (object$model_params$sample_sigma_global) {
        jsonobj$add_vector("sigma2_samples", object$sigma2_samples, "parameters")
    }
    if (object$model_params$sample_sigma_leaf_mu) {
        jsonobj$add_vector("sigma_leaf_mu_samples", object$sigma_leaf_mu_samples, "parameters")
    }
    if (object$model_params$sample_sigma_leaf_tau) {
        jsonobj$add_vector("sigma_leaf_tau_samples", object$sigma_leaf_tau_samples, "parameters")
    }
    if (object$model_params$adaptive_coding) {
        jsonobj$add_vector("b_1_samples", object$b_1_samples, "parameters")
        jsonobj$add_vector("b_0_samples", object$b_0_samples, "parameters")
    }

    # Add random effects (if present)
    if (object$model_params$has_rfx) {
        jsonobj$add_random_effects(object$rfx_samples)
        jsonobj$add_string_vector("rfx_unique_group_ids", object$rfx_unique_group_ids)
    }
    
    return(jsonobj)
}

#' Convert the persistent aspects of a BCF model to (in-memory) JSON and save to a file
#'
#' @param object Object of type `bcf` containing draws of a Bayesian causal forest model and associated sampling outputs.
#' @param filename String of filepath, must end in ".json"
#'
#' @return NULL
#' @export
#'
#' @examples
#' n <- 500
#' x1 <- rnorm(n)
#' x2 <- rnorm(n)
#' x3 <- rnorm(n)
#' x4 <- as.numeric(rbinom(n,1,0.5))
#' x5 <- as.numeric(sample(1:3,n,replace=TRUE))
#' X <- cbind(x1,x2,x3,x4,x5)
#' p <- ncol(X)
#' g <- function(x) {ifelse(x[,5]==1,2,ifelse(x[,5]==2,-1,4))}
#' mu1 <- function(x) {1+g(x)+x[,1]*x[,3]}
#' mu2 <- function(x) {1+g(x)+6*abs(x[,3]-1)}
#' tau1 <- function(x) {rep(3,nrow(x))}
#' tau2 <- function(x) {1+2*x[,2]*x[,4]}
#' mu_x <- mu1(X)
#' tau_x <- tau2(X)
#' pi_x <- 0.8*pnorm((3*mu_x/sd(mu_x)) - 0.5*X[,1]) + 0.05 + runif(n)/10
#' Z <- rbinom(n,1,pi_x)
#' E_XZ <- mu_x + Z*tau_x
#' snr <- 3
#' group_ids <- rep(c(1,2), n %/% 2)
#' rfx_coefs <- matrix(c(-1, -1, 1, 1), nrow=2, byrow=TRUE)
#' rfx_basis <- cbind(1, runif(n, -1, 1))
#' rfx_term <- rowSums(rfx_coefs[group_ids,] * rfx_basis)
#' y <- E_XZ + rfx_term + rnorm(n, 0, 1)*(sd(E_XZ)/snr)
#' X <- as.data.frame(X)
#' X$x4 <- factor(X$x4, ordered = TRUE)
#' X$x5 <- factor(X$x5, ordered = TRUE)
#' test_set_pct <- 0.2
#' n_test <- round(test_set_pct*n)
#' n_train <- n - n_test
#' test_inds <- sort(sample(1:n, n_test, replace = FALSE))
#' train_inds <- (1:n)[!((1:n) %in% test_inds)]
#' X_test <- X[test_inds,]
#' X_train <- X[train_inds,]
#' pi_test <- pi_x[test_inds]
#' pi_train <- pi_x[train_inds]
#' Z_test <- Z[test_inds]
#' Z_train <- Z[train_inds]
#' y_test <- y[test_inds]
#' y_train <- y[train_inds]
#' mu_test <- mu_x[test_inds]
#' mu_train <- mu_x[train_inds]
#' tau_test <- tau_x[test_inds]
#' tau_train <- tau_x[train_inds]
#' group_ids_test <- group_ids[test_inds]
#' group_ids_train <- group_ids[train_inds]
#' rfx_basis_test <- rfx_basis[test_inds,]
#' rfx_basis_train <- rfx_basis[train_inds,]
#' rfx_term_test <- rfx_term[test_inds]
#' rfx_term_train <- rfx_term[train_inds]
#' bcf_params <- list(sample_sigma_leaf_mu = TRUE, sample_sigma_leaf_tau = FALSE)
#' bcf_model <- bcf(X_train = X_train, Z_train = Z_train, y_train = y_train, 
#'                  pi_train = pi_train, group_ids_train = group_ids_train, 
#'                  rfx_basis_train = rfx_basis_train, X_test = X_test, 
#'                  Z_test = Z_test, pi_test = pi_test, group_ids_test = group_ids_test,
#'                  rfx_basis_test = rfx_basis_test, 
#'                  num_gfr = 100, num_burnin = 0, num_mcmc = 100, 
#'                  params = bcf_params)
#' # saveBCFModelToJsonFile(bcf_model, "test.json")
saveBCFModelToJsonFile <- function(object, filename){
    # Convert to Json
    jsonobj <- convertBCFModelToJson(object)
    
    # Save to file
    jsonobj$save_file(filename)
}

#' Convert the persistent aspects of a BCF model to (in-memory) JSON string
#'
#' @param object Object of type `bcf` containing draws of a Bayesian causal forest model and associated sampling outputs.
#' @return JSON string
#' @export
#'
#' @examples
#' n <- 500
#' x1 <- rnorm(n)
#' x2 <- rnorm(n)
#' x3 <- rnorm(n)
#' x4 <- as.numeric(rbinom(n,1,0.5))
#' x5 <- as.numeric(sample(1:3,n,replace=TRUE))
#' X <- cbind(x1,x2,x3,x4,x5)
#' p <- ncol(X)
#' g <- function(x) {ifelse(x[,5]==1,2,ifelse(x[,5]==2,-1,4))}
#' mu1 <- function(x) {1+g(x)+x[,1]*x[,3]}
#' mu2 <- function(x) {1+g(x)+6*abs(x[,3]-1)}
#' tau1 <- function(x) {rep(3,nrow(x))}
#' tau2 <- function(x) {1+2*x[,2]*x[,4]}
#' mu_x <- mu1(X)
#' tau_x <- tau2(X)
#' pi_x <- 0.8*pnorm((3*mu_x/sd(mu_x)) - 0.5*X[,1]) + 0.05 + runif(n)/10
#' Z <- rbinom(n,1,pi_x)
#' E_XZ <- mu_x + Z*tau_x
#' snr <- 3
#' group_ids <- rep(c(1,2), n %/% 2)
#' rfx_coefs <- matrix(c(-1, -1, 1, 1), nrow=2, byrow=TRUE)
#' rfx_basis <- cbind(1, runif(n, -1, 1))
#' rfx_term <- rowSums(rfx_coefs[group_ids,] * rfx_basis)
#' y <- E_XZ + rfx_term + rnorm(n, 0, 1)*(sd(E_XZ)/snr)
#' X <- as.data.frame(X)
#' X$x4 <- factor(X$x4, ordered = TRUE)
#' X$x5 <- factor(X$x5, ordered = TRUE)
#' test_set_pct <- 0.2
#' n_test <- round(test_set_pct*n)
#' n_train <- n - n_test
#' test_inds <- sort(sample(1:n, n_test, replace = FALSE))
#' train_inds <- (1:n)[!((1:n) %in% test_inds)]
#' X_test <- X[test_inds,]
#' X_train <- X[train_inds,]
#' pi_test <- pi_x[test_inds]
#' pi_train <- pi_x[train_inds]
#' Z_test <- Z[test_inds]
#' Z_train <- Z[train_inds]
#' y_test <- y[test_inds]
#' y_train <- y[train_inds]
#' mu_test <- mu_x[test_inds]
#' mu_train <- mu_x[train_inds]
#' tau_test <- tau_x[test_inds]
#' tau_train <- tau_x[train_inds]
#' group_ids_test <- group_ids[test_inds]
#' group_ids_train <- group_ids[train_inds]
#' rfx_basis_test <- rfx_basis[test_inds,]
#' rfx_basis_train <- rfx_basis[train_inds,]
#' rfx_term_test <- rfx_term[test_inds]
#' rfx_term_train <- rfx_term[train_inds]
#' bcf_params <- list(sample_sigma_leaf_mu = TRUE, sample_sigma_leaf_tau = FALSE)
#' bcf_model <- bcf(X_train = X_train, Z_train = Z_train, y_train = y_train, 
#'                  pi_train = pi_train, group_ids_train = group_ids_train, 
#'                  rfx_basis_train = rfx_basis_train, X_test = X_test, 
#'                  Z_test = Z_test, pi_test = pi_test, group_ids_test = group_ids_test,
#'                  rfx_basis_test = rfx_basis_test, 
#'                  num_gfr = 100, num_burnin = 0, num_mcmc = 100, 
#'                  params = bcf_params)
#' # saveBCFModelToJsonString(bcf_model)
saveBCFModelToJsonString <- function(object){
    # Convert to Json
    jsonobj <- convertBCFModelToJson(object)
    
    # Dump to string
    return(jsonobj$return_json_string())
}

#' Convert an (in-memory) JSON representation of a BCF model to a BCF model object 
#' which can be used for prediction, etc...
#'
#' @param json_object Object of type `CppJson` containing Json representation of a BCF model
#'
#' @return Object of type `bcf`
#' @export
#'
#' @examples
#' n <- 500
#' x1 <- rnorm(n)
#' x2 <- rnorm(n)
#' x3 <- rnorm(n)
#' x4 <- as.numeric(rbinom(n,1,0.5))
#' x5 <- as.numeric(sample(1:3,n,replace=TRUE))
#' X <- cbind(x1,x2,x3,x4,x5)
#' p <- ncol(X)
#' g <- function(x) {ifelse(x[,5]==1,2,ifelse(x[,5]==2,-1,4))}
#' mu1 <- function(x) {1+g(x)+x[,1]*x[,3]}
#' mu2 <- function(x) {1+g(x)+6*abs(x[,3]-1)}
#' tau1 <- function(x) {rep(3,nrow(x))}
#' tau2 <- function(x) {1+2*x[,2]*x[,4]}
#' mu_x <- mu1(X)
#' tau_x <- tau2(X)
#' pi_x <- 0.8*pnorm((3*mu_x/sd(mu_x)) - 0.5*X[,1]) + 0.05 + runif(n)/10
#' Z <- rbinom(n,1,pi_x)
#' E_XZ <- mu_x + Z*tau_x
#' snr <- 3
#' group_ids <- rep(c(1,2), n %/% 2)
#' rfx_coefs <- matrix(c(-1, -1, 1, 1), nrow=2, byrow=TRUE)
#' rfx_basis <- cbind(1, runif(n, -1, 1))
#' rfx_term <- rowSums(rfx_coefs[group_ids,] * rfx_basis)
#' y <- E_XZ + rfx_term + rnorm(n, 0, 1)*(sd(E_XZ)/snr)
#' X <- as.data.frame(X)
#' X$x4 <- factor(X$x4, ordered = TRUE)
#' X$x5 <- factor(X$x5, ordered = TRUE)
#' test_set_pct <- 0.2
#' n_test <- round(test_set_pct*n)
#' n_train <- n - n_test
#' test_inds <- sort(sample(1:n, n_test, replace = FALSE))
#' train_inds <- (1:n)[!((1:n) %in% test_inds)]
#' X_test <- X[test_inds,]
#' X_train <- X[train_inds,]
#' pi_test <- pi_x[test_inds]
#' pi_train <- pi_x[train_inds]
#' Z_test <- Z[test_inds]
#' Z_train <- Z[train_inds]
#' y_test <- y[test_inds]
#' y_train <- y[train_inds]
#' mu_test <- mu_x[test_inds]
#' mu_train <- mu_x[train_inds]
#' tau_test <- tau_x[test_inds]
#' tau_train <- tau_x[train_inds]
#' group_ids_test <- group_ids[test_inds]
#' group_ids_train <- group_ids[train_inds]
#' rfx_basis_test <- rfx_basis[test_inds,]
#' rfx_basis_train <- rfx_basis[train_inds,]
#' rfx_term_test <- rfx_term[test_inds]
#' rfx_term_train <- rfx_term[train_inds]
#' bcf_params <- list(sample_sigma_leaf_mu = TRUE, sample_sigma_leaf_tau = FALSE)
#' bcf_model <- bcf(X_train = X_train, Z_train = Z_train, y_train = y_train, 
#'                  pi_train = pi_train, group_ids_train = group_ids_train, 
#'                  rfx_basis_train = rfx_basis_train, X_test = X_test, 
#'                  Z_test = Z_test, pi_test = pi_test, group_ids_test = group_ids_test,
#'                  rfx_basis_test = rfx_basis_test, 
#'                  num_gfr = 100, num_burnin = 0, num_mcmc = 100, 
#'                  params = bcf_params)
#' # bcf_json <- convertBCFModelToJson(bcf_model)
#' # bcf_model_roundtrip <- createBCFModelFromJson(bcf_json)
createBCFModelFromJson <- function(json_object){
    # Initialize the BCF model
    output <- list()
    
    # Unpack the forests
    output[["forests_mu"]] <- loadForestContainerJson(json_object, "forest_0")
    output[["forests_tau"]] <- loadForestContainerJson(json_object, "forest_1")
    include_variance_forest <- json_object$get_boolean("include_variance_forest")
    if (include_variance_forest) {
        output[["forests_variance"]] <- loadForestContainerJson(json_object, "forest_2")
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
    model_params[["outcome_scale"]] <- json_object$get_scalar("outcome_scale")
    model_params[["outcome_mean"]] <- json_object$get_scalar("outcome_mean")
    model_params[["standardize"]] <- json_object$get_boolean("standardize")
    model_params[["initial_sigma2"]] <- json_object$get_scalar("initial_sigma2")
    model_params[["sample_sigma_global"]] <- json_object$get_boolean("sample_sigma_global")
    model_params[["sample_sigma_leaf_mu"]] <- json_object$get_boolean("sample_sigma_leaf_mu")
    model_params[["sample_sigma_leaf_tau"]] <- json_object$get_boolean("sample_sigma_leaf_tau")
    model_params[["include_variance_forest"]] <- include_variance_forest
    model_params[["propensity_covariate"]] <- json_object$get_string("propensity_covariate")
    model_params[["has_rfx"]] <- json_object$get_boolean("has_rfx")
    model_params[["has_rfx_basis"]] <- json_object$get_boolean("has_rfx_basis")
    model_params[["num_rfx_basis"]] <- json_object$get_scalar("num_rfx_basis")
    model_params[["adaptive_coding"]] <- json_object$get_boolean("adaptive_coding")
    model_params[["num_gfr"]] <- json_object$get_scalar("num_gfr")
    model_params[["num_burnin"]] <- json_object$get_scalar("num_burnin")
    model_params[["num_mcmc"]] <- json_object$get_scalar("num_mcmc")
    model_params[["num_samples"]] <- json_object$get_scalar("num_samples")
    model_params[["num_covariates"]] <- json_object$get_scalar("num_covariates")
    output[["model_params"]] <- model_params
    
    # Unpack sampled parameters
    if (model_params[["sample_sigma_global"]]) {
        output[["sigma2_samples"]] <- json_object$get_vector("sigma2_samples", "parameters")
    }
    if (model_params[["sample_sigma_leaf_mu"]]) {
        output[["sigma_leaf_mu_samples"]] <- json_object$get_vector("sigma_leaf_mu_samples", "parameters")
    }
    if (model_params[["sample_sigma_leaf_tau"]]) {
        output[["sigma_leaf_tau_samples"]] <- json_object$get_vector("sigma_leaf_tau_samples", "parameters")
    }
    if (model_params[["adaptive_coding"]]) {
        output[["b_1_samples"]] <- json_object$get_vector("b_1_samples", "parameters")
        output[["b_0_samples"]] <- json_object$get_vector("b_0_samples", "parameters")
    }
    
    # Unpack random effects
    if (model_params[["has_rfx"]]) {
        output[["rfx_unique_group_ids"]] <- json_object$get_string_vector("rfx_unique_group_ids")
        output[["rfx_samples"]] <- loadRandomEffectSamplesJson(json_object, 0)
    }
    
    class(output) <- "bcf"
    return(output)
}

#' Convert a JSON file containing sample information on a trained BCF model 
#' to a BCF model object which can be used for prediction, etc...
#'
#' @param json_filename String of filepath, must end in ".json"
#'
#' @return Object of type `bcf`
#' @export
#'
#' @examples
#' n <- 500
#' x1 <- rnorm(n)
#' x2 <- rnorm(n)
#' x3 <- rnorm(n)
#' x4 <- as.numeric(rbinom(n,1,0.5))
#' x5 <- as.numeric(sample(1:3,n,replace=TRUE))
#' X <- cbind(x1,x2,x3,x4,x5)
#' p <- ncol(X)
#' g <- function(x) {ifelse(x[,5]==1,2,ifelse(x[,5]==2,-1,4))}
#' mu1 <- function(x) {1+g(x)+x[,1]*x[,3]}
#' mu2 <- function(x) {1+g(x)+6*abs(x[,3]-1)}
#' tau1 <- function(x) {rep(3,nrow(x))}
#' tau2 <- function(x) {1+2*x[,2]*x[,4]}
#' mu_x <- mu1(X)
#' tau_x <- tau2(X)
#' pi_x <- 0.8*pnorm((3*mu_x/sd(mu_x)) - 0.5*X[,1]) + 0.05 + runif(n)/10
#' Z <- rbinom(n,1,pi_x)
#' E_XZ <- mu_x + Z*tau_x
#' snr <- 3
#' group_ids <- rep(c(1,2), n %/% 2)
#' rfx_coefs <- matrix(c(-1, -1, 1, 1), nrow=2, byrow=TRUE)
#' rfx_basis <- cbind(1, runif(n, -1, 1))
#' rfx_term <- rowSums(rfx_coefs[group_ids,] * rfx_basis)
#' y <- E_XZ + rfx_term + rnorm(n, 0, 1)*(sd(E_XZ)/snr)
#' X <- as.data.frame(X)
#' X$x4 <- factor(X$x4, ordered = TRUE)
#' X$x5 <- factor(X$x5, ordered = TRUE)
#' test_set_pct <- 0.2
#' n_test <- round(test_set_pct*n)
#' n_train <- n - n_test
#' test_inds <- sort(sample(1:n, n_test, replace = FALSE))
#' train_inds <- (1:n)[!((1:n) %in% test_inds)]
#' X_test <- X[test_inds,]
#' X_train <- X[train_inds,]
#' pi_test <- pi_x[test_inds]
#' pi_train <- pi_x[train_inds]
#' Z_test <- Z[test_inds]
#' Z_train <- Z[train_inds]
#' y_test <- y[test_inds]
#' y_train <- y[train_inds]
#' mu_test <- mu_x[test_inds]
#' mu_train <- mu_x[train_inds]
#' tau_test <- tau_x[test_inds]
#' tau_train <- tau_x[train_inds]
#' group_ids_test <- group_ids[test_inds]
#' group_ids_train <- group_ids[train_inds]
#' rfx_basis_test <- rfx_basis[test_inds,]
#' rfx_basis_train <- rfx_basis[train_inds,]
#' rfx_term_test <- rfx_term[test_inds]
#' rfx_term_train <- rfx_term[train_inds]
#' bcf_params <- list(sample_sigma_leaf_mu = TRUE, sample_sigma_leaf_tau = FALSE)
#' bcf_model <- bcf(X_train = X_train, Z_train = Z_train, y_train = y_train, 
#'                  pi_train = pi_train, group_ids_train = group_ids_train, 
#'                  rfx_basis_train = rfx_basis_train, X_test = X_test, 
#'                  Z_test = Z_test, pi_test = pi_test, group_ids_test = group_ids_test,
#'                  rfx_basis_test = rfx_basis_test, 
#'                  num_gfr = 100, num_burnin = 0, num_mcmc = 100, 
#'                  params = bcf_params)
#' # saveBCFModelToJsonFile(bcf_model, "test.json")
#' # bcf_model_roundtrip <- createBCFModelFromJsonFile("test.json")
createBCFModelFromJsonFile <- function(json_filename){
    # Load a `CppJson` object from file
    bcf_json <- createCppJsonFile(json_filename)
    
    # Create and return the BCF object
    bcf_object <- createBCFModelFromJson(bcf_json)
    
    return(bcf_object)
}

#' Convert a JSON string containing sample information on a trained BCF model 
#' to a BCF model object which can be used for prediction, etc...
#'
#' @param json_string JSON string dump
#'
#' @return Object of type `bcf`
#' @export
#'
#' @examples
#' n <- 500
#' x1 <- rnorm(n)
#' x2 <- rnorm(n)
#' x3 <- rnorm(n)
#' x4 <- as.numeric(rbinom(n,1,0.5))
#' x5 <- as.numeric(sample(1:3,n,replace=TRUE))
#' X <- cbind(x1,x2,x3,x4,x5)
#' p <- ncol(X)
#' g <- function(x) {ifelse(x[,5]==1,2,ifelse(x[,5]==2,-1,4))}
#' mu1 <- function(x) {1+g(x)+x[,1]*x[,3]}
#' mu2 <- function(x) {1+g(x)+6*abs(x[,3]-1)}
#' tau1 <- function(x) {rep(3,nrow(x))}
#' tau2 <- function(x) {1+2*x[,2]*x[,4]}
#' mu_x <- mu1(X)
#' tau_x <- tau2(X)
#' pi_x <- 0.8*pnorm((3*mu_x/sd(mu_x)) - 0.5*X[,1]) + 0.05 + runif(n)/10
#' Z <- rbinom(n,1,pi_x)
#' E_XZ <- mu_x + Z*tau_x
#' snr <- 3
#' group_ids <- rep(c(1,2), n %/% 2)
#' rfx_coefs <- matrix(c(-1, -1, 1, 1), nrow=2, byrow=TRUE)
#' rfx_basis <- cbind(1, runif(n, -1, 1))
#' rfx_term <- rowSums(rfx_coefs[group_ids,] * rfx_basis)
#' y <- E_XZ + rfx_term + rnorm(n, 0, 1)*(sd(E_XZ)/snr)
#' X <- as.data.frame(X)
#' X$x4 <- factor(X$x4, ordered = TRUE)
#' X$x5 <- factor(X$x5, ordered = TRUE)
#' test_set_pct <- 0.2
#' n_test <- round(test_set_pct*n)
#' n_train <- n - n_test
#' test_inds <- sort(sample(1:n, n_test, replace = FALSE))
#' train_inds <- (1:n)[!((1:n) %in% test_inds)]
#' X_test <- X[test_inds,]
#' X_train <- X[train_inds,]
#' pi_test <- pi_x[test_inds]
#' pi_train <- pi_x[train_inds]
#' Z_test <- Z[test_inds]
#' Z_train <- Z[train_inds]
#' y_test <- y[test_inds]
#' y_train <- y[train_inds]
#' mu_test <- mu_x[test_inds]
#' mu_train <- mu_x[train_inds]
#' tau_test <- tau_x[test_inds]
#' tau_train <- tau_x[train_inds]
#' group_ids_test <- group_ids[test_inds]
#' group_ids_train <- group_ids[train_inds]
#' rfx_basis_test <- rfx_basis[test_inds,]
#' rfx_basis_train <- rfx_basis[train_inds,]
#' rfx_term_test <- rfx_term[test_inds]
#' rfx_term_train <- rfx_term[train_inds]
#' bcf_model <- bcf(X_train = X_train, Z_train = Z_train, y_train = y_train, 
#'                  pi_train = pi_train, group_ids_train = group_ids_train, 
#'                  rfx_basis_train = rfx_basis_train, X_test = X_test, 
#'                  Z_test = Z_test, pi_test = pi_test, group_ids_test = group_ids_test,
#'                  rfx_basis_test = rfx_basis_test, 
#'                  num_gfr = 100, num_burnin = 0, num_mcmc = 100)
#' # bcf_json <- saveBCFModelToJsonString(bcf_model)
#' # bcf_model_roundtrip <- createBCFModelFromJsonString(bcf_json)
createBCFModelFromJsonString <- function(json_string){
    # Load a `CppJson` object from string
    bcf_json <- createCppJsonString(json_string)
    
    # Create and return the BCF object
    bcf_object <- createBCFModelFromJson(bcf_json)
    
    return(bcf_object)
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
#' x1 <- rnorm(n)
#' x2 <- rnorm(n)
#' x3 <- rnorm(n)
#' x4 <- rnorm(n)
#' x5 <- rnorm(n)
#' X <- cbind(x1,x2,x3,x4,x5)
#' p <- ncol(X)
#' g <- function(x) {ifelse(x[,5] < -0.44,2,ifelse(x[,5] < 0.44,-1,4))}
#' mu1 <- function(x) {1+g(x)+x[,1]*x[,3]}
#' mu2 <- function(x) {1+g(x)+6*abs(x[,3]-1)}
#' tau1 <- function(x) {rep(3,nrow(x))}
#' tau2 <- function(x) {1+2*x[,2]*(x[,4] > 0)}
#' mu_x <- mu1(X)
#' tau_x <- tau2(X)
#' pi_x <- 0.8*pnorm((3*mu_x/sd(mu_x)) - 0.5*X[,1]) + 0.05 + runif(n)/10
#' Z <- rbinom(n,1,pi_x)
#' E_XZ <- mu_x + Z*tau_x
#' snr <- 3
#' group_ids <- rep(c(1,2), n %/% 2)
#' rfx_coefs <- matrix(c(-1, -1, 1, 1), nrow=2, byrow=TRUE)
#' rfx_basis <- cbind(1, runif(n, -1, 1))
#' rfx_term <- rowSums(rfx_coefs[group_ids,] * rfx_basis)
#' y <- E_XZ + rfx_term + rnorm(n, 0, 1)*(sd(E_XZ)/snr)
#' X <- as.data.frame(X)
#' X$x4 <- factor(X$x4, ordered = TRUE)
#' X$x5 <- factor(X$x5, ordered = TRUE)
#' test_set_pct <- 0.2
#' n_test <- round(test_set_pct*n)
#' n_train <- n - n_test
#' test_inds <- sort(sample(1:n, n_test, replace = FALSE))
#' train_inds <- (1:n)[!((1:n) %in% test_inds)]
#' X_test <- X[test_inds,]
#' X_train <- X[train_inds,]
#' pi_test <- pi_x[test_inds]
#' pi_train <- pi_x[train_inds]
#' Z_test <- Z[test_inds]
#' Z_train <- Z[train_inds]
#' y_test <- y[test_inds]
#' y_train <- y[train_inds]
#' mu_test <- mu_x[test_inds]
#' mu_train <- mu_x[train_inds]
#' tau_test <- tau_x[test_inds]
#' tau_train <- tau_x[train_inds]
#' group_ids_test <- group_ids[test_inds]
#' group_ids_train <- group_ids[train_inds]
#' rfx_basis_test <- rfx_basis[test_inds,]
#' rfx_basis_train <- rfx_basis[train_inds,]
#' rfx_term_test <- rfx_term[test_inds]
#' rfx_term_train <- rfx_term[train_inds]
#' bcf_model <- bcf(X_train = X_train, Z_train = Z_train, y_train = y_train, 
#'                  pi_train = pi_train, group_ids_train = group_ids_train, 
#'                  rfx_basis_train = rfx_basis_train, X_test = X_test, 
#'                  Z_test = Z_test, pi_test = pi_test, group_ids_test = group_ids_test,
#'                  rfx_basis_test = rfx_basis_test, 
#'                  num_gfr = 100, num_burnin = 0, num_mcmc = 100)
#' # bcf_json_string_list <- list(saveBCFModelToJsonString(bcf_model))
#' # bcf_model_roundtrip <- createBCFModelFromCombinedJsonString(bcf_json_string_list)
createBCFModelFromCombinedJsonString <- function(json_string_list){
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
    output[["forests_mu"]] <- loadForestContainerCombinedJson(json_object_list, "forest_0")
    output[["forests_tau"]] <- loadForestContainerCombinedJson(json_object_list, "forest_1")
    include_variance_forest <- json_object_default$get_boolean("include_variance_forest")
    if (include_variance_forest) {
        output[["forests_variance"]] <- loadForestContainerCombinedJson(json_object_list, "forest_2")
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
    model_params[["initial_sigma2"]] <- json_object_default$get_scalar("initial_sigma2")
    model_params[["sample_sigma_global"]] <- json_object_default$get_boolean("sample_sigma_global")
    model_params[["sample_sigma_leaf_mu"]] <- json_object_default$get_boolean("sample_sigma_leaf_mu")
    model_params[["sample_sigma_leaf_tau"]] <- json_object_default$get_boolean("sample_sigma_leaf_tau")
    model_params[["include_variance_forest"]] <- include_variance_forest
    model_params[["propensity_covariate"]] <- json_object_default$get_string("propensity_covariate")
    model_params[["has_rfx"]] <- json_object_default$get_boolean("has_rfx")
    model_params[["has_rfx_basis"]] <- json_object_default$get_boolean("has_rfx_basis")
    model_params[["num_rfx_basis"]] <- json_object_default$get_scalar("num_rfx_basis")
    model_params[["num_covariates"]] <- json_object_default$get_scalar("num_covariates")
    model_params[["num_chains"]] <- json_object_default$get_scalar("num_chains")
    model_params[["keep_every"]] <- json_object_default$get_scalar("keep_every")
    model_params[["adaptive_coding"]] <- json_object_default$get_boolean("adaptive_coding")
    
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
                output[["sigma2_samples"]] <- json_object$get_vector("sigma2_samples", "parameters")
            } else {
                output[["sigma2_samples"]] <- c(output[["sigma2_samples"]], json_object$get_vector("sigma2_samples", "parameters"))
            }
        }
    }
    if (model_params[["sample_sigma_leaf_mu"]]) {
        for (i in 1:length(json_object_list)) {
            json_object <- json_object_list[[i]]
            if (i == 1) {
                output[["sigma_leaf_mu_samples"]] <- json_object$get_vector("sigma_leaf_mu_samples", "parameters")
            } else {
                output[["sigma_leaf_mu_samples"]] <- c(output[["sigma_leaf_mu_samples"]], json_object$get_vector("sigma_leaf_mu_samples", "parameters"))
            }
        }
    }
    if (model_params[["sample_sigma_leaf_tau"]]) {
        for (i in 1:length(json_object_list)) {
            json_object <- json_object_list[[i]]
            if (i == 1) {
                output[["sigma_leaf_tau_samples"]] <- json_object$get_vector("sigma_leaf_tau_samples", "parameters")
            } else {
                output[["sigma_leaf_tau_samples"]] <- c(output[["sigma_leaf_tau_samples"]], json_object$get_vector("sigma_leaf_tau_samples", "parameters"))
            }
        }
    }
    if (model_params[["sample_sigma_leaf_tau"]]) {
        for (i in 1:length(json_object_list)) {
            json_object <- json_object_list[[i]]
            if (i == 1) {
                output[["sigma_leaf_tau_samples"]] <- json_object$get_vector("sigma_leaf_tau_samples", "parameters")
            } else {
                output[["sigma_leaf_tau_samples"]] <- c(output[["sigma_leaf_tau_samples"]], json_object$get_vector("sigma_leaf_tau_samples", "parameters"))
            }
        }
    }
    if (model_params[["adaptive_coding"]]) {
        for (i in 1:length(json_object_list)) {
            json_object <- json_object_list[[i]]
            if (i == 1) {
                output[["b_1_samples"]] <- json_object$get_vector("b_1_samples", "parameters")
                output[["b_0_samples"]] <- json_object$get_vector("b_0_samples", "parameters")
            } else {
                output[["b_1_samples"]] <- c(output[["b_1_samples"]], json_object$get_vector("b_1_samples", "parameters"))
                output[["b_0_samples"]] <- c(output[["b_0_samples"]], json_object$get_vector("b_0_samples", "parameters"))
            }
        }
    }
    
    # Unpack random effects
    if (model_params[["has_rfx"]]) {
        output[["rfx_unique_group_ids"]] <- json_object_default$get_string_vector("rfx_unique_group_ids")
        output[["rfx_samples"]] <- loadRandomEffectSamplesCombinedJson(json_object_list, 0)
    }
    
    class(output) <- "bcf"
    return(output)
}

