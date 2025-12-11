#' Run the BART algorithm for supervised learning.
#'
#' @param X_train Covariates used to split trees in the ensemble. May be provided either as a dataframe or a matrix.
#' Matrix covariates will be assumed to be all numeric. Covariates passed as a dataframe will be
#' preprocessed based on the variable types (e.g. categorical columns stored as unordered factors will be one-hot encoded,
#' categorical columns stored as ordered factors will passed as integers to the core algorithm, along with the metadata
#' that the column is ordered categorical).
#' @param y_train Outcome to be modeled by the ensemble.
#' @param leaf_basis_train (Optional) Bases used to define a regression model `y ~ W` in
#' each leaf of each regression tree. By default, BART assumes constant leaf node
#' parameters, implicitly regressing on a constant basis of ones (i.e. `y ~ 1`).
#' @param rfx_group_ids_train (Optional) Group labels used for an additive random effects model.
#' @param rfx_basis_train (Optional) Basis for "random-slope" regression in an additive random effects model.
#' If `rfx_group_ids_train` is provided with a regression basis, an intercept-only random effects model
#' will be estimated.
#' @param X_test (Optional) Test set of covariates used to define "out of sample" evaluation data.
#' May be provided either as a dataframe or a matrix, but the format of `X_test` must be consistent with
#' that of `X_train`.
#' @param leaf_basis_test (Optional) Test set of bases used to define "out of sample" evaluation data.
#' While a test set is optional, the structure of any provided test set must match that
#' of the training set (i.e. if both `X_train` and `leaf_basis_train` are provided, then a test set must
#' consist of `X_test` and `leaf_basis_test` with the same number of columns).
#' @param rfx_group_ids_test (Optional) Test set group labels used for an additive random effects model.
#' We do not currently support (but plan to in the near future), test set evaluation for group labels
#' that were not in the training set.
#' @param rfx_basis_test (Optional) Test set basis for "random-slope" regression in additive random effects model.
#' @param num_gfr Number of "warm-start" iterations run using the grow-from-root algorithm (He and Hahn, 2021). Default: 5.
#' @param num_burnin Number of "burn-in" iterations of the MCMC sampler. Default: 0.
#' @param num_mcmc Number of "retained" iterations of the MCMC sampler. Default: 100.
#' @param previous_model_json (Optional) JSON string containing a previous BART model. This can be used to "continue" a sampler interactively after inspecting the samples or to run parallel chains "warm-started" from existing forest samples. Default: `NULL`.
#' @param previous_model_warmstart_sample_num (Optional) Sample number from `previous_model_json` that will be used to warmstart this BART sampler. One-indexed (so that the first sample is used for warm-start by setting `previous_model_warmstart_sample_num = 1`). Default: `NULL`.  If `num_chains` in the `general_params` list is > 1, then each successive chain will be initialized from a different sample, counting backwards from `previous_model_warmstart_sample_num`. That is, if `previous_model_warmstart_sample_num = 10` and `num_chains = 4`, then chain 1 will be initialized from sample 10, chain 2 from sample 9, chain 3 from sample 8, and chain 4 from sample 7. If `previous_model_json` is provided but `previous_model_warmstart_sample_num` is NULL, the last sample in the previous model will be used to initialize the first chain, counting backwards as noted before. If more chains are requested than there are samples in `previous_model_json`, a warning will be raised and only the last sample will be used.
#' @param general_params (Optional) A list of general (non-forest-specific) model parameters, each of which has a default value processed internally, so this argument list is optional.
#'
#'   - `cutpoint_grid_size` Maximum size of the "grid" of potential cutpoints to consider in the GFR algorithm. Default: `100`.
#'   - `standardize` Whether or not to standardize the outcome (and store the offset / scale in the model object). Default: `TRUE`.
#'   - `sample_sigma2_global` Whether or not to update the `sigma^2` global error variance parameter based on `IG(sigma2_global_shape, sigma2_global_scale)`. Default: `TRUE`.
#'   - `sigma2_global_init` Starting value of global error variance parameter. Calibrated internally as `1.0*var(y_train)`, where `y_train` is the possibly standardized outcome, if not set.
#'   - `sigma2_global_shape` Shape parameter in the `IG(sigma2_global_shape, sigma2_global_scale)` global error variance model. Default: `0`.
#'   - `sigma2_global_scale` Scale parameter in the `IG(sigma2_global_shape, sigma2_global_scale)` global error variance model. Default: `0`.
#'   - `variable_weights` Numeric weights reflecting the relative probability of splitting on each variable. Does not need to sum to 1 but cannot be negative. Defaults to `rep(1/ncol(X_train), ncol(X_train))` if not set here. Note that if the propensity score is included as a covariate in either forest, its weight will default to `1/ncol(X_train)`.
#'   - `random_seed` Integer parameterizing the C++ random number generator. If not specified, the C++ random number generator is seeded according to `std::random_device`.
#'   - `keep_burnin` Whether or not "burnin" samples should be included in the stored samples of forests and other parameters. Default `FALSE`. Ignored if `num_mcmc = 0`.
#'   - `keep_gfr` Whether or not "grow-from-root" samples should be included in the stored samples of forests and other parameters. Default `FALSE`. Ignored if `num_mcmc = 0`.
#'   - `keep_every` How many iterations of the burned-in MCMC sampler should be run before forests and parameters are retained. Default `1`. Setting `keep_every <- k` for some `k > 1` will "thin" the MCMC samples by retaining every `k`-th sample, rather than simply every sample. This can reduce the autocorrelation of the MCMC samples.
#'   - `num_chains` How many independent MCMC chains should be sampled. If `num_mcmc = 0`, this is ignored. If `num_gfr = 0`, then each chain is run from root for `num_mcmc * keep_every + num_burnin` iterations, with `num_mcmc` samples retained. If `num_gfr > 0`, each MCMC chain will be initialized from a separate GFR ensemble, with the requirement that `num_gfr >= num_chains`. Default: `1`. Note that if `num_chains > 1`, the returned model object will contain samples from all chains, stored consecutively. That is, if there are 4 chains with 100 samples each, the first 100 samples will be from chain 1, the next 100 samples will be from chain 2, etc... For more detail on working with multi-chain BART models, see [the multi chain vignette](https://stochtree.ai/R_docs/pkgdown/articles/MultiChain.html).
#'   - `verbose` Whether or not to print progress during the sampling loops. Default: `FALSE`.
#'   - `probit_outcome_model` Whether or not the outcome should be modeled as explicitly binary via a probit link. If `TRUE`, `y` must only contain the values `0` and `1`. Default: `FALSE`.
#'   - `num_threads` Number of threads to use in the GFR and MCMC algorithms, as well as prediction. If OpenMP is not available on a user's setup, this will default to `1`, otherwise to the maximum number of available threads.
#'
#' @param mean_forest_params (Optional) A list of mean forest model parameters, each of which has a default value processed internally, so this argument list is optional.
#'
#'   - `num_trees` Number of trees in the ensemble for the conditional mean model. Default: `200`. If `num_trees = 0`, the conditional mean will not be modeled using a forest, and the function will only proceed if `num_trees > 0` for the variance forest.
#'   - `alpha` Prior probability of splitting for a tree of depth 0 in the mean model. Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`. Default: `0.95`.
#'   - `beta` Exponent that decreases split probabilities for nodes of depth > 0 in the mean model. Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`. Default: `2`.
#'   - `min_samples_leaf` Minimum allowable size of a leaf, in terms of training samples, in the mean model. Default: `5`.
#'   - `max_depth` Maximum depth of any tree in the ensemble in the mean model. Default: `10`. Can be overridden with ``-1`` which does not enforce any depth limits on trees.
#'   - `sample_sigma2_leaf` Whether or not to update the leaf scale variance parameter based on `IG(sigma2_leaf_shape, sigma2_leaf_scale)`. Cannot (currently) be set to true if `ncol(leaf_basis_train)>1`. Default: `FALSE`.
#'   - `sigma2_leaf_init` Starting value of leaf node scale parameter. Calibrated internally as `1/num_trees` if not set here.
#'   - `sigma2_leaf_shape` Shape parameter in the `IG(sigma2_leaf_shape, sigma2_leaf_scale)` leaf node parameter variance model. Default: `3`.
#'   - `sigma2_leaf_scale` Scale parameter in the `IG(sigma2_leaf_shape, sigma2_leaf_scale)` leaf node parameter variance model. Calibrated internally as `0.5/num_trees` if not set here.
#'   - `keep_vars` Vector of variable names or column indices denoting variables that should be included in the forest. Default: `NULL`.
#'   - `drop_vars` Vector of variable names or column indices denoting variables that should be excluded from the forest. Default: `NULL`. If both `drop_vars` and `keep_vars` are set, `drop_vars` will be ignored.
#'   - `num_features_subsample` How many features to subsample when growing each tree for the GFR algorithm. Defaults to the number of features in the training dataset.
#'
#' @param variance_forest_params (Optional) A list of variance forest model parameters, each of which has a default value processed internally, so this argument list is optional.
#'
#'   - `num_trees` Number of trees in the ensemble for the conditional variance model. Default: `0`. Variance is only modeled using a tree / forest if `num_trees > 0`.
#'   - `alpha` Prior probability of splitting for a tree of depth 0 in the variance model. Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`. Default: `0.95`.
#'   - `beta` Exponent that decreases split probabilities for nodes of depth > 0 in the variance model. Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`. Default: `2`.
#'   - `min_samples_leaf` Minimum allowable size of a leaf, in terms of training samples, in the variance model. Default: `5`.
#'   - `max_depth` Maximum depth of any tree in the ensemble in the variance model. Default: `10`. Can be overridden with ``-1`` which does not enforce any depth limits on trees.
#'   - `leaf_prior_calibration_param` Hyperparameter used to calibrate the `IG(var_forest_prior_shape, var_forest_prior_scale)` conditional error variance model. If `var_forest_prior_shape` and `var_forest_prior_scale` are not set below, this calibration parameter is used to set these values to `num_trees / leaf_prior_calibration_param^2 + 0.5` and `num_trees / leaf_prior_calibration_param^2`, respectively. Default: `1.5`.
#'   - `var_forest_leaf_init` Starting value of root forest prediction in conditional (heteroskedastic) error variance model. Calibrated internally as `log(0.6*var(y_train))/num_trees`, where `y_train` is the possibly standardized outcome, if not set.
#'   - `var_forest_prior_shape` Shape parameter in the `IG(var_forest_prior_shape, var_forest_prior_scale)` conditional error variance model (which is only sampled if `num_trees > 0`). Calibrated internally as `num_trees / leaf_prior_calibration_param^2 + 0.5` if not set.
#'   - `var_forest_prior_scale` Scale parameter in the `IG(var_forest_prior_shape, var_forest_prior_scale)` conditional error variance model (which is only sampled if `num_trees > 0`). Calibrated internally as `num_trees / leaf_prior_calibration_param^2` if not set.
#'   - `keep_vars` Vector of variable names or column indices denoting variables that should be included in the forest. Default: `NULL`.
#'   - `drop_vars` Vector of variable names or column indices denoting variables that should be excluded from the forest. Default: `NULL`. If both `drop_vars` and `keep_vars` are set, `drop_vars` will be ignored.
#'   - `num_features_subsample` How many features to subsample when growing each tree for the GFR algorithm. Defaults to the number of features in the training dataset.
#'
#' @param random_effects_params (Optional) A list of random effects model parameters, each of which has a default value processed internally, so this argument list is optional.
#'
#'   - `model_spec` Specification of the random effects model. Options are "custom" and "intercept_only". If "custom" is specified, then a user-provided basis must be passed through `rfx_basis_train`. If "intercept_only" is specified, a random effects basis of all ones will be dispatched internally at sampling and prediction time. If "intercept_plus_treatment" is specified, a random effects basis that combines an "intercept" basis of all ones with the treatment variable (`Z_train`) will be dispatched internally at sampling and prediction time. Default: "custom". If "intercept_only" is specified, `rfx_basis_train` and `rfx_basis_test` (if provided) will be ignored.
#'   - `working_parameter_prior_mean` Prior mean for the random effects "working parameter". Default: `NULL`. Must be a vector whose dimension matches the number of random effects bases, or a scalar value that will be expanded to a vector.
#'   - `group_parameters_prior_mean` Prior mean for the random effects "group parameters." Default: `NULL`. Must be a vector whose dimension matches the number of random effects bases, or a scalar value that will be expanded to a vector.
#'   - `working_parameter_prior_cov` Prior covariance matrix for the random effects "working parameter." Default: `NULL`. Must be a square matrix whose dimension matches the number of random effects bases, or a scalar value that will be expanded to a diagonal matrix.
#'   - `group_parameter_prior_cov` Prior covariance matrix for the random effects "group parameters." Default: `NULL`. Must be a square matrix whose dimension matches the number of random effects bases, or a scalar value that will be expanded to a diagonal matrix.
#'   - `variance_prior_shape` Shape parameter for the inverse gamma prior on the variance of the random effects "group parameter." Default: `1`.
#'   - `variance_prior_scale` Scale parameter for the inverse gamma prior on the variance of the random effects "group parameter." Default: `1`.
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
#'
#' bart_model <- bart(X_train = X_train, y_train = y_train, X_test = X_test,
#'                    num_gfr = 10, num_burnin = 0, num_mcmc = 10)
bart <- function(
  X_train,
  y_train,
  leaf_basis_train = NULL,
  rfx_group_ids_train = NULL,
  rfx_basis_train = NULL,
  X_test = NULL,
  leaf_basis_test = NULL,
  rfx_group_ids_test = NULL,
  rfx_basis_test = NULL,
  num_gfr = 5,
  num_burnin = 0,
  num_mcmc = 100,
  previous_model_json = NULL,
  previous_model_warmstart_sample_num = NULL,
  general_params = list(),
  mean_forest_params = list(),
  variance_forest_params = list(),
  random_effects_params = list()
) {
  # Update general BART parameters
  general_params_default <- list(
    cutpoint_grid_size = 100,
    standardize = TRUE,
    sample_sigma2_global = TRUE,
    sigma2_global_init = NULL,
    sigma2_global_shape = 0,
    sigma2_global_scale = 0,
    variable_weights = NULL,
    random_seed = -1,
    keep_burnin = FALSE,
    keep_gfr = FALSE,
    keep_every = 1,
    num_chains = 1,
    verbose = FALSE,
    probit_outcome_model = FALSE,
    num_threads = -1
  )
  general_params_updated <- preprocessParams(
    general_params_default,
    general_params
  )

  # Update mean forest BART parameters
  mean_forest_params_default <- list(
    num_trees = 200,
    alpha = 0.95,
    beta = 2.0,
    min_samples_leaf = 5,
    max_depth = 10,
    sample_sigma2_leaf = TRUE,
    sigma2_leaf_init = NULL,
    sigma2_leaf_shape = 3,
    sigma2_leaf_scale = NULL,
    keep_vars = NULL,
    drop_vars = NULL,
    num_features_subsample = NULL
  )
  mean_forest_params_updated <- preprocessParams(
    mean_forest_params_default,
    mean_forest_params
  )

  # Update variance forest BART parameters
  variance_forest_params_default <- list(
    num_trees = 0,
    alpha = 0.95,
    beta = 2.0,
    min_samples_leaf = 5,
    max_depth = 10,
    leaf_prior_calibration_param = 1.5,
    var_forest_leaf_init = NULL,
    var_forest_prior_shape = NULL,
    var_forest_prior_scale = NULL,
    keep_vars = NULL,
    drop_vars = NULL,
    num_features_subsample = NULL
  )
  variance_forest_params_updated <- preprocessParams(
    variance_forest_params_default,
    variance_forest_params
  )

  # Update rfx parameters
  rfx_params_default <- list(
    model_spec = "custom",
    working_parameter_prior_mean = NULL,
    group_parameter_prior_mean = NULL,
    working_parameter_prior_cov = NULL,
    group_parameter_prior_cov = NULL,
    variance_prior_shape = 1,
    variance_prior_scale = 1
  )
  rfx_params_updated <- preprocessParams(
    rfx_params_default,
    random_effects_params
  )

  ### Unpack all parameter values
  # 1. General parameters
  cutpoint_grid_size <- general_params_updated$cutpoint_grid_size
  standardize <- general_params_updated$standardize
  sample_sigma2_global <- general_params_updated$sample_sigma2_global
  sigma2_init <- general_params_updated$sigma2_global_init
  a_global <- general_params_updated$sigma2_global_shape
  b_global <- general_params_updated$sigma2_global_scale
  variable_weights <- general_params_updated$variable_weights
  random_seed <- general_params_updated$random_seed
  keep_burnin <- general_params_updated$keep_burnin
  keep_gfr <- general_params_updated$keep_gfr
  keep_every <- general_params_updated$keep_every
  num_chains <- general_params_updated$num_chains
  verbose <- general_params_updated$verbose
  probit_outcome_model <- general_params_updated$probit_outcome_model
  num_threads <- general_params_updated$num_threads

  # 2. Mean forest parameters
  num_trees_mean <- mean_forest_params_updated$num_trees
  alpha_mean <- mean_forest_params_updated$alpha
  beta_mean <- mean_forest_params_updated$beta
  min_samples_leaf_mean <- mean_forest_params_updated$min_samples_leaf
  max_depth_mean <- mean_forest_params_updated$max_depth
  sample_sigma2_leaf <- mean_forest_params_updated$sample_sigma2_leaf
  sigma2_leaf_init <- mean_forest_params_updated$sigma2_leaf_init
  a_leaf <- mean_forest_params_updated$sigma2_leaf_shape
  b_leaf <- mean_forest_params_updated$sigma2_leaf_scale
  keep_vars_mean <- mean_forest_params_updated$keep_vars
  drop_vars_mean <- mean_forest_params_updated$drop_vars
  num_features_subsample_mean <- mean_forest_params_updated$num_features_subsample

  # 3. Variance forest parameters
  num_trees_variance <- variance_forest_params_updated$num_trees
  alpha_variance <- variance_forest_params_updated$alpha
  beta_variance <- variance_forest_params_updated$beta
  min_samples_leaf_variance <- variance_forest_params_updated$min_samples_leaf
  max_depth_variance <- variance_forest_params_updated$max_depth
  a_0 <- variance_forest_params_updated$leaf_prior_calibration_param
  variance_forest_init <- variance_forest_params_updated$var_forest_leaf_init
  a_forest <- variance_forest_params_updated$var_forest_prior_shape
  b_forest <- variance_forest_params_updated$var_forest_prior_scale
  keep_vars_variance <- variance_forest_params_updated$keep_vars
  drop_vars_variance <- variance_forest_params_updated$drop_vars
  num_features_subsample_variance <- variance_forest_params_updated$num_features_subsample

  # 4. RFX parameters
  rfx_model_spec <- rfx_params_updated$model_spec
  rfx_working_parameter_prior_mean <- rfx_params_updated$working_parameter_prior_mean
  rfx_group_parameter_prior_mean <- rfx_params_updated$group_parameter_prior_mean
  rfx_working_parameter_prior_cov <- rfx_params_updated$working_parameter_prior_cov
  rfx_group_parameter_prior_cov <- rfx_params_updated$group_parameter_prior_cov
  rfx_variance_prior_shape <- rfx_params_updated$variance_prior_shape
  rfx_variance_prior_scale <- rfx_params_updated$variance_prior_scale

  # Set a function-scoped RNG if user provided a random seed
  custom_rng <- random_seed >= 0
  if (custom_rng) {
    # Store original global environment RNG state
    original_global_seed <- .Random.seed
    # Set new seed and store associated RNG state
    set.seed(random_seed)
    function_scoped_seed <- .Random.seed
  }

  # Check if there are enough GFR samples to seed num_chains samplers
  if (num_gfr > 0) {
    if (num_chains > num_gfr) {
      stop(
        "num_chains > num_gfr, meaning we do not have enough GFR samples to seed num_chains distinct MCMC chains"
      )
    }
  }

  # Override keep_gfr if there are no MCMC samples
  if (num_mcmc == 0) {
    keep_gfr <- TRUE
  }

  # Check if previous model JSON is provided and parse it if so
  has_prev_model <- !is.null(previous_model_json)
  has_prev_model_index <- !is.null(previous_model_warmstart_sample_num)
  if (has_prev_model) {
    previous_bart_model <- createBARTModelFromJsonString(
      previous_model_json
    )
    prev_num_samples <- previous_bart_model$model_params$num_samples
    if (!has_prev_model_index) {
      previous_model_warmstart_sample_num <- prev_num_samples
      warning(
        "`previous_model_warmstart_sample_num` was not provided alongside `previous_model_json`, so it will be set to the number of samples available in `previous_model_json`"
      )
    } else {
      if (previous_model_warmstart_sample_num < 1) {
        stop(
          "`previous_model_warmstart_sample_num` must be a positive integer"
        )
      }
      if (previous_model_warmstart_sample_num > prev_num_samples) {
        stop(
          "`previous_model_warmstart_sample_num` exceeds the number of samples in `previous_model_json`"
        )
      }
    }
    previous_model_decrement <- T
    if (num_chains > previous_model_warmstart_sample_num) {
      warning(
        "The number of chains being sampled exceeds the number of previous model samples available from the requested position in `previous_model_json`. All chains will be initialized from the same sample."
      )
      previous_model_decrement <- F
    }
    previous_y_bar <- previous_bart_model$model_params$outcome_mean
    previous_y_scale <- previous_bart_model$model_params$outcome_scale
    if (previous_bart_model$model_params$include_mean_forest) {
      previous_forest_samples_mean <- previous_bart_model$mean_forests
    } else {
      previous_forest_samples_mean <- NULL
    }
    if (previous_bart_model$model_params$include_variance_forest) {
      previous_forest_samples_variance <- previous_bart_model$variance_forests
    } else {
      previous_forest_samples_variance <- NULL
    }
    if (previous_bart_model$model_params$sample_sigma2_global) {
      previous_global_var_samples <- previous_bart_model$sigma2_global_samples /
        (previous_y_scale * previous_y_scale)
    } else {
      previous_global_var_samples <- NULL
    }
    if (previous_bart_model$model_params$sample_sigma2_leaf) {
      previous_leaf_var_samples <- previous_bart_model$sigma2_leaf_samples
    } else {
      previous_leaf_var_samples <- NULL
    }
    if (previous_bart_model$model_params$has_rfx) {
      previous_rfx_samples <- previous_bart_model$rfx_samples
    } else {
      previous_rfx_samples <- NULL
    }
    previous_model_num_samples <- previous_bart_model$model_params$num_samples
    if (previous_model_warmstart_sample_num > previous_model_num_samples) {
      stop(
        "`previous_model_warmstart_sample_num` exceeds the number of samples in `previous_model_json`"
      )
    }
  } else {
    previous_y_bar <- NULL
    previous_y_scale <- NULL
    previous_global_var_samples <- NULL
    previous_leaf_var_samples <- NULL
    previous_rfx_samples <- NULL
    previous_forest_samples_mean <- NULL
    previous_forest_samples_variance <- NULL
    previous_model_num_samples <- 0
  }

  # Determine whether conditional mean, variance, or both will be modeled
  if (num_trees_variance > 0) {
    include_variance_forest = TRUE
  } else {
    include_variance_forest = FALSE
  }
  if (num_trees_mean > 0) {
    include_mean_forest = TRUE
  } else {
    include_mean_forest = FALSE
  }

  # Set the variance forest priors if not set
  if (include_variance_forest) {
    if (is.null(a_forest)) {
      a_forest <- num_trees_variance / (a_0^2) + 0.5
    }
    if (is.null(b_forest)) b_forest <- num_trees_variance / (a_0^2)
  } else {
    a_forest <- 1.
    b_forest <- 1.
  }

  # Override tau sampling if there is no mean forest
  if (!include_mean_forest) {
    sample_sigma2_leaf <- FALSE
  }

  # Variable weight preprocessing (and initialization if necessary)
  if (is.null(variable_weights)) {
    variable_weights = rep(1 / ncol(X_train), ncol(X_train))
  }
  if (any(variable_weights < 0)) {
    stop("variable_weights cannot have any negative weights")
  }

  # Check covariates are matrix or dataframe
  if ((!is.data.frame(X_train)) && (!is.matrix(X_train))) {
    stop("X_train must be a matrix or dataframe")
  }
  if (!is.null(X_test)) {
    if ((!is.data.frame(X_test)) && (!is.matrix(X_test))) {
      stop("X_test must be a matrix or dataframe")
    }
  }
  num_cov_orig <- ncol(X_train)

  # Raise a warning if the data have ties and only GFR is being run
  if ((num_gfr > 0) && (num_mcmc == 0) && (num_burnin == 0)) {
    num_values <- nrow(X_train)
    max_grid_size <- ifelse(
      num_values > cutpoint_grid_size,
      floor(num_values / cutpoint_grid_size),
      1
    )
    x_is_df <- is.data.frame(X_train)
    covs_warning_1 <- NULL
    covs_warning_2 <- NULL
    covs_warning_3 <- NULL
    covs_warning_4 <- NULL
    for (i in 1:num_cov_orig) {
      # Skip check for variables that are treated as categorical
      x_numeric <- T
      if (x_is_df) {
        if (is.factor(X_train[, i])) {
          x_numeric <- F
        }
      }
      if (x_numeric) {
        # Determine the number of unique values
        num_unique_values <- length(unique(X_train[, i]))

        # Determine a "name" for the covariate
        cov_name <- ifelse(
          is.null(colnames(X_train)),
          paste0("X", i),
          colnames(X_train)[i]
        )

        # Check for a small relative number of unique values
        unique_full_ratio <- num_unique_values / num_values
        if (unique_full_ratio < 0.2) {
          covs_warning_1 <- c(covs_warning_1, cov_name)
        }

        # Check for a small absolute number of unique values
        if (num_values > 100) {
          if (num_unique_values < 20) {
            covs_warning_2 <- c(covs_warning_2, cov_name)
          }
        }

        # Check for a large number of duplicates of any individual value
        x_j_hist <- table(X_train[, i])
        if (any(x_j_hist > 2 * max_grid_size)) {
          covs_warning_3 <- c(covs_warning_3, cov_name)
        }

        # Check for binary variables
        if (num_unique_values == 2) {
          covs_warning_4 <- c(covs_warning_4, cov_name)
        }
      }
    }

    if (!is.null(covs_warning_1)) {
      warning(
        paste0(
          "Covariate(s) ",
          paste(covs_warning_1, collapse = ", "),
          " have a ratio of unique to overall observations of less than 0.2. ",
          "This might present some issues with the grow-from-root (GFR) algorithm. ",
          "Consider running with `num_mcmc > 0` and `num_burnin > 0` to improve your model's performance."
        )
      )
    }

    if (!is.null(covs_warning_2)) {
      warning(
        paste0(
          "Covariate(s) ",
          paste(covs_warning_2, collapse = ", "),
          " have fewer than 20 unique values. ",
          "This might present some issues with the grow-from-root (GFR) algorithm. ",
          "Consider running with `num_mcmc > 0` and `num_burnin > 0` to improve your model's performance."
        )
      )
    }

    if (!is.null(covs_warning_3)) {
      warning(
        paste0(
          "Covariates ",
          paste(covs_warning_3, collapse = ", "),
          " have some observed values with more than ",
          2 * max_grid_size,
          " repeated observations. ",
          "This might present some issues with the grow-from-root (GFR) algorithm. ",
          "Consider running with `num_mcmc > 0` and `num_burnin > 0` to improve your model's performance."
        )
      )
    }

    if (!is.null(covs_warning_4)) {
      warning(
        paste0(
          "Covariates ",
          paste(covs_warning_4, collapse = ", "),
          " appear to be binary but are currently treated by stochtree as continuous. ",
          "This might present some issues with the grow-from-root (GFR) algorithm. ",
          "Consider converting binary variables to ordered factor (i.e. `factor(..., ordered = T)`."
        )
      )
    }
  }

  # Standardize the keep variable lists to numeric indices
  if (!is.null(keep_vars_mean)) {
    if (is.character(keep_vars_mean)) {
      if (!all(keep_vars_mean %in% names(X_train))) {
        stop(
          "keep_vars_mean includes some variable names that are not in X_train"
        )
      }
      variable_subset_mean <- unname(which(
        names(X_train) %in% keep_vars_mean
      ))
    } else {
      if (any(keep_vars_mean > ncol(X_train))) {
        stop(
          "keep_vars_mean includes some variable indices that exceed the number of columns in X_train"
        )
      }
      if (any(keep_vars_mean < 0)) {
        stop("keep_vars_mean includes some negative variable indices")
      }
      variable_subset_mean <- keep_vars_mean
    }
  } else if ((is.null(keep_vars_mean)) && (!is.null(drop_vars_mean))) {
    if (is.character(drop_vars_mean)) {
      if (!all(drop_vars_mean %in% names(X_train))) {
        stop(
          "drop_vars_mean includes some variable names that are not in X_train"
        )
      }
      variable_subset_mean <- unname(which(
        !(names(X_train) %in% drop_vars_mean)
      ))
    } else {
      if (any(drop_vars_mean > ncol(X_train))) {
        stop(
          "drop_vars_mean includes some variable indices that exceed the number of columns in X_train"
        )
      }
      if (any(drop_vars_mean < 0)) {
        stop("drop_vars_mean includes some negative variable indices")
      }
      variable_subset_mean <- (1:ncol(X_train))[
        !(1:ncol(X_train) %in% drop_vars_mean)
      ]
    }
  } else {
    variable_subset_mean <- 1:ncol(X_train)
  }
  if (!is.null(keep_vars_variance)) {
    if (is.character(keep_vars_variance)) {
      if (!all(keep_vars_variance %in% names(X_train))) {
        stop(
          "keep_vars_variance includes some variable names that are not in X_train"
        )
      }
      variable_subset_variance <- unname(which(
        names(X_train) %in% keep_vars_variance
      ))
    } else {
      if (any(keep_vars_variance > ncol(X_train))) {
        stop(
          "keep_vars_variance includes some variable indices that exceed the number of columns in X_train"
        )
      }
      if (any(keep_vars_variance < 0)) {
        stop(
          "keep_vars_variance includes some negative variable indices"
        )
      }
      variable_subset_variance <- keep_vars_variance
    }
  } else if ((is.null(keep_vars_variance)) && (!is.null(drop_vars_variance))) {
    if (is.character(drop_vars_variance)) {
      if (!all(drop_vars_variance %in% names(X_train))) {
        stop(
          "drop_vars_variance includes some variable names that are not in X_train"
        )
      }
      variable_subset_variance <- unname(which(
        !(names(X_train) %in% drop_vars_variance)
      ))
    } else {
      if (any(drop_vars_variance > ncol(X_train))) {
        stop(
          "drop_vars_variance includes some variable indices that exceed the number of columns in X_train"
        )
      }
      if (any(drop_vars_variance < 0)) {
        stop(
          "drop_vars_variance includes some negative variable indices"
        )
      }
      variable_subset_variance <- (1:ncol(X_train))[
        !(1:ncol(X_train) %in% drop_vars_variance)
      ]
    }
  } else {
    variable_subset_variance <- 1:ncol(X_train)
  }

  # Preprocess covariates
  if ((!is.data.frame(X_train)) && (!is.matrix(X_train))) {
    stop("X_train must be a matrix or dataframe")
  }
  if (!is.null(X_test)) {
    if ((!is.data.frame(X_test)) && (!is.matrix(X_test))) {
      stop("X_test must be a matrix or dataframe")
    }
  }
  if (ncol(X_train) != length(variable_weights)) {
    stop("length(variable_weights) must equal ncol(X_train)")
  }
  train_cov_preprocess_list <- preprocessTrainData(X_train)
  X_train_metadata <- train_cov_preprocess_list$metadata
  X_train <- train_cov_preprocess_list$data
  original_var_indices <- X_train_metadata$original_var_indices
  feature_types <- X_train_metadata$feature_types
  if (!is.null(X_test)) {
    X_test <- preprocessPredictionData(X_test, X_train_metadata)
  }

  # Update variable weights
  variable_weights_mean <- variable_weights_variance <- variable_weights
  variable_weights_adj <- 1 /
    sapply(original_var_indices, function(x) sum(original_var_indices == x))
  if (include_mean_forest) {
    variable_weights_mean <- variable_weights_mean[original_var_indices] *
      variable_weights_adj
    variable_weights_mean[
      !(original_var_indices %in% variable_subset_mean)
    ] <- 0
  }
  if (include_variance_forest) {
    variable_weights_variance <- variable_weights_variance[
      original_var_indices
    ] *
      variable_weights_adj
    variable_weights_variance[
      !(original_var_indices %in% variable_subset_variance)
    ] <- 0
  }

  # Set num_features_subsample to default, ncol(X_train), if not already set
  if (is.null(num_features_subsample_mean)) {
    num_features_subsample_mean <- ncol(X_train)
  }
  if (is.null(num_features_subsample_variance)) {
    num_features_subsample_variance <- ncol(X_train)
  }

  # Convert all input data to matrices if not already converted
  if ((is.null(dim(leaf_basis_train))) && (!is.null(leaf_basis_train))) {
    leaf_basis_train <- as.matrix(leaf_basis_train)
  }
  if ((is.null(dim(leaf_basis_test))) && (!is.null(leaf_basis_test))) {
    leaf_basis_test <- as.matrix(leaf_basis_test)
  }
  if ((is.null(dim(rfx_basis_train))) && (!is.null(rfx_basis_train))) {
    rfx_basis_train <- as.matrix(rfx_basis_train)
  }
  if ((is.null(dim(rfx_basis_test))) && (!is.null(rfx_basis_test))) {
    rfx_basis_test <- as.matrix(rfx_basis_test)
  }

  # Recode group IDs to integer vector (if passed as, for example, a vector of county names, etc...)
  has_rfx <- FALSE
  has_rfx_test <- FALSE
  if (!is.null(rfx_group_ids_train)) {
    group_ids_factor <- factor(rfx_group_ids_train)
    rfx_group_ids_train <- as.integer(group_ids_factor)
    has_rfx <- TRUE
    if (!is.null(rfx_group_ids_test)) {
      group_ids_factor_test <- factor(
        rfx_group_ids_test,
        levels = levels(group_ids_factor)
      )
      if (sum(is.na(group_ids_factor_test)) > 0) {
        stop(
          "All random effect group labels provided in rfx_group_ids_test must be present in rfx_group_ids_train"
        )
      }
      rfx_group_ids_test <- as.integer(group_ids_factor_test)
      has_rfx_test <- TRUE
    }
  }

  # Data consistency checks
  if ((!is.null(X_test)) && (ncol(X_test) != ncol(X_train))) {
    stop("X_train and X_test must have the same number of columns")
  }
  if (
    (!is.null(leaf_basis_test)) &&
      (ncol(leaf_basis_test) != ncol(leaf_basis_train))
  ) {
    stop(
      "leaf_basis_train and leaf_basis_test must have the same number of columns"
    )
  }
  if (
    (!is.null(leaf_basis_train)) &&
      (nrow(leaf_basis_train) != nrow(X_train))
  ) {
    stop("leaf_basis_train and X_train must have the same number of rows")
  }
  if ((!is.null(leaf_basis_test)) && (nrow(leaf_basis_test) != nrow(X_test))) {
    stop("leaf_basis_test and X_test must have the same number of rows")
  }
  if (nrow(X_train) != length(y_train)) {
    stop("X_train and y_train must have the same number of observations")
  }
  if (
    (!is.null(rfx_basis_test)) &&
      (ncol(rfx_basis_test) != ncol(rfx_basis_train))
  ) {
    stop(
      "rfx_basis_train and rfx_basis_test must have the same number of columns"
    )
  }
  if (!is.null(rfx_group_ids_train)) {
    if (!is.null(rfx_group_ids_test)) {
      if ((!is.null(rfx_basis_train)) && (is.null(rfx_basis_test))) {
        stop(
          "rfx_basis_train is provided but rfx_basis_test is not provided"
        )
      }
    }
  }

  # Handle the rfx basis matrices
  has_basis_rfx <- FALSE
  num_basis_rfx <- 0
  if (has_rfx) {
    if (rfx_model_spec == "custom") {
      if (is.null(rfx_basis_train)) {
        stop(
          "A user-provided basis (`rfx_basis_train`) must be provided when the random effects model spec is 'custom'"
        )
      }
      has_basis_rfx <- TRUE
      num_basis_rfx <- ncol(rfx_basis_train)
    } else if (rfx_model_spec == "intercept_only") {
      rfx_basis_train <- matrix(
        rep(1, nrow(X_train)),
        nrow = nrow(X_train),
        ncol = 1
      )
      has_basis_rfx <- TRUE
      num_basis_rfx <- 1
    }
    num_rfx_groups <- length(unique(rfx_group_ids_train))
    num_rfx_components <- ncol(rfx_basis_train)
    if (num_rfx_groups == 1) {
      warning(
        "Only one group was provided for random effect sampling, so the random effects model is likely overkill"
      )
    }
  }
  if (has_rfx_test) {
    if (rfx_model_spec == "custom") {
      if (is.null(rfx_basis_test)) {
        stop(
          "A user-provided basis (`rfx_basis_test`) must be provided when the random effects model spec is 'custom'"
        )
      }
    } else if (rfx_model_spec == "intercept_only") {
      rfx_basis_test <- matrix(
        rep(1, nrow(X_test)),
        nrow = nrow(X_test),
        ncol = 1
      )
    }
  }

  # Convert y_train to numeric vector if not already converted
  if (!is.null(dim(y_train))) {
    y_train <- as.matrix(y_train)
  }

  # Determine whether a basis vector is provided
  has_basis = !is.null(leaf_basis_train)

  # Determine whether a test set is provided
  has_test = !is.null(X_test)

  # Preliminary runtime checks for probit link
  if (!include_mean_forest) {
    probit_outcome_model <- FALSE
  }
  if (probit_outcome_model) {
    if (!(length(unique(y_train)) == 2)) {
      stop(
        "You specified a probit outcome model, but supplied an outcome with more than 2 unique values"
      )
    }
    unique_outcomes <- sort(unique(y_train))
    if (!(all(unique_outcomes == c(0, 1)))) {
      stop(
        "You specified a probit outcome model, but supplied an outcome with 2 unique values other than 0 and 1"
      )
    }
    if (include_variance_forest) {
      stop("We do not support heteroskedasticity with a probit link")
    }
    if (sample_sigma2_global) {
      warning(
        "Global error variance will not be sampled with a probit link as it is fixed at 1"
      )
      sample_sigma2_global <- F
    }
  }

  # Runtime checks for variance forest
  if (include_variance_forest) {
    if (sample_sigma2_global) {
      warning(
        "Global error variance will not be sampled with a heteroskedasticity forest"
      )
      sample_sigma2_global <- F
    }
  }

  # Handle standardization, prior calibration, and initialization of forest
  # differently for binary and continuous outcomes
  if (probit_outcome_model) {
    # Compute a probit-scale offset and fix scale to 1
    y_bar_train <- qnorm(mean(y_train))
    y_std_train <- 1

    # Set a pseudo outcome by subtracting mean(y_train) from y_train
    resid_train <- y_train - mean(y_train)

    # Set initial values of root nodes to 0.0 (in probit scale)
    init_val_mean <- 0.0

    # Calibrate priors for sigma^2 and tau
    # Set sigma2_init to 1, ignoring default provided
    sigma2_init <- 1.0
    # Skip variance_forest_init, since variance forests are not supported with probit link
    b_leaf <- 1 / (num_trees_mean)
    if (has_basis) {
      if (ncol(leaf_basis_train) > 1) {
        if (is.null(sigma2_leaf_init)) {
          sigma2_leaf_init <- diag(
            2 / (num_trees_mean),
            ncol(leaf_basis_train)
          )
        }
        if (!is.matrix(sigma2_leaf_init)) {
          current_leaf_scale <- as.matrix(diag(
            sigma2_leaf_init,
            ncol(leaf_basis_train)
          ))
        } else {
          current_leaf_scale <- sigma2_leaf_init
        }
      } else {
        if (is.null(sigma2_leaf_init)) {
          sigma2_leaf_init <- as.matrix(2 / (num_trees_mean))
        }
        if (!is.matrix(sigma2_leaf_init)) {
          current_leaf_scale <- as.matrix(diag(sigma2_leaf_init, 1))
        } else {
          current_leaf_scale <- sigma2_leaf_init
        }
      }
    } else {
      if (is.null(sigma2_leaf_init)) {
        sigma2_leaf_init <- as.matrix(2 / (num_trees_mean))
      }
      if (!is.matrix(sigma2_leaf_init)) {
        current_leaf_scale <- as.matrix(diag(sigma2_leaf_init, 1))
      } else {
        current_leaf_scale <- sigma2_leaf_init
      }
    }
    current_sigma2 <- sigma2_init
  } else {
    # Only standardize if user requested
    if (standardize) {
      y_bar_train <- mean(y_train)
      y_std_train <- sd(y_train)
    } else {
      y_bar_train <- 0
      y_std_train <- 1
    }

    # Compute standardized outcome
    resid_train <- (y_train - y_bar_train) / y_std_train

    # Compute initial value of root nodes in mean forest
    init_val_mean <- mean(resid_train)

    # Calibrate priors for sigma^2 and tau
    if (is.null(sigma2_init)) {
      sigma2_init <- 1.0 * var(resid_train)
    }
    if (is.null(variance_forest_init)) {
      variance_forest_init <- 1.0 * var(resid_train)
    }
    if (is.null(b_leaf)) {
      b_leaf <- var(resid_train) / (2 * num_trees_mean)
    }
    if (has_basis) {
      if (ncol(leaf_basis_train) > 1) {
        if (is.null(sigma2_leaf_init)) {
          sigma2_leaf_init <- diag(
            2 * var(resid_train) / (num_trees_mean),
            ncol(leaf_basis_train)
          )
        }
        if (!is.matrix(sigma2_leaf_init)) {
          current_leaf_scale <- as.matrix(diag(
            sigma2_leaf_init,
            ncol(leaf_basis_train)
          ))
        } else {
          current_leaf_scale <- sigma2_leaf_init
        }
      } else {
        if (is.null(sigma2_leaf_init)) {
          sigma2_leaf_init <- as.matrix(
            2 * var(resid_train) / (num_trees_mean)
          )
        }
        if (!is.matrix(sigma2_leaf_init)) {
          current_leaf_scale <- as.matrix(diag(sigma2_leaf_init, 1))
        } else {
          current_leaf_scale <- sigma2_leaf_init
        }
      }
    } else {
      if (is.null(sigma2_leaf_init)) {
        sigma2_leaf_init <- as.matrix(
          2 * var(resid_train) / (num_trees_mean)
        )
      }
      if (!is.matrix(sigma2_leaf_init)) {
        current_leaf_scale <- as.matrix(diag(sigma2_leaf_init, 1))
      } else {
        current_leaf_scale <- sigma2_leaf_init
      }
    }
    current_sigma2 <- sigma2_init
  }

  # Determine leaf model type
  if (!has_basis) {
    leaf_model_mean_forest <- 0
  } else if (ncol(leaf_basis_train) == 1) {
    leaf_model_mean_forest <- 1
  } else if (ncol(leaf_basis_train) > 1) {
    leaf_model_mean_forest <- 2
  } else {
    stop("leaf_basis_train passed must be a matrix with at least 1 column")
  }

  # Set variance leaf model type (currently only one option)
  leaf_model_variance_forest <- 3

  # Unpack model type info
  if (leaf_model_mean_forest == 0) {
    leaf_dimension = 1
    is_leaf_constant = TRUE
    leaf_regression = FALSE
  } else if (leaf_model_mean_forest == 1) {
    stopifnot(has_basis)
    stopifnot(ncol(leaf_basis_train) == 1)
    leaf_dimension = 1
    is_leaf_constant = FALSE
    leaf_regression = TRUE
  } else if (leaf_model_mean_forest == 2) {
    stopifnot(has_basis)
    stopifnot(ncol(leaf_basis_train) > 1)
    leaf_dimension = ncol(leaf_basis_train)
    is_leaf_constant = FALSE
    leaf_regression = TRUE
    if (sample_sigma2_leaf) {
      warning(
        "Sampling leaf scale not yet supported for multivariate leaf models, so the leaf scale parameter will not be sampled in this model."
      )
      sample_sigma2_leaf <- FALSE
    }
  }

  # Data
  if (leaf_regression) {
    forest_dataset_train <- createForestDataset(X_train, leaf_basis_train)
    if (has_test) {
      forest_dataset_test <- createForestDataset(X_test, leaf_basis_test)
    }
    requires_basis <- TRUE
  } else {
    forest_dataset_train <- createForestDataset(X_train)
    if (has_test) {
      forest_dataset_test <- createForestDataset(X_test)
    }
    requires_basis <- FALSE
  }
  outcome_train <- createOutcome(resid_train)

  # Random number generator (std::mt19937)
  if (is.null(random_seed)) {
    random_seed = sample(1:10000, 1, FALSE)
  }
  rng <- createCppRNG(random_seed)

  # Sampling data structures
  feature_types <- as.integer(feature_types)
  global_model_config <- createGlobalModelConfig(
    global_error_variance = current_sigma2
  )
  if (include_mean_forest) {
    forest_model_config_mean <- createForestModelConfig(
      feature_types = feature_types,
      num_trees = num_trees_mean,
      num_features = ncol(X_train),
      num_observations = nrow(X_train),
      variable_weights = variable_weights_mean,
      leaf_dimension = leaf_dimension,
      alpha = alpha_mean,
      beta = beta_mean,
      min_samples_leaf = min_samples_leaf_mean,
      max_depth = max_depth_mean,
      leaf_model_type = leaf_model_mean_forest,
      leaf_model_scale = current_leaf_scale,
      cutpoint_grid_size = cutpoint_grid_size,
      num_features_subsample = num_features_subsample_mean
    )
    forest_model_mean <- createForestModel(
      forest_dataset_train,
      forest_model_config_mean,
      global_model_config
    )
  }
  if (include_variance_forest) {
    forest_model_config_variance <- createForestModelConfig(
      feature_types = feature_types,
      num_trees = num_trees_variance,
      num_features = ncol(X_train),
      num_observations = nrow(X_train),
      variable_weights = variable_weights_variance,
      leaf_dimension = 1,
      alpha = alpha_variance,
      beta = beta_variance,
      min_samples_leaf = min_samples_leaf_variance,
      max_depth = max_depth_variance,
      leaf_model_type = leaf_model_variance_forest,
      cutpoint_grid_size = cutpoint_grid_size,
      num_features_subsample = num_features_subsample_variance
    )
    forest_model_variance <- createForestModel(
      forest_dataset_train,
      forest_model_config_variance,
      global_model_config
    )
  }

  # Container of forest samples
  if (include_mean_forest) {
    forest_samples_mean <- createForestSamples(
      num_trees_mean,
      leaf_dimension,
      is_leaf_constant,
      FALSE
    )
    active_forest_mean <- createForest(
      num_trees_mean,
      leaf_dimension,
      is_leaf_constant,
      FALSE
    )
  }
  if (include_variance_forest) {
    forest_samples_variance <- createForestSamples(
      num_trees_variance,
      1,
      TRUE,
      TRUE
    )
    active_forest_variance <- createForest(
      num_trees_variance,
      1,
      TRUE,
      TRUE
    )
  }

  # Random effects initialization
  if (has_rfx) {
    # Prior parameters
    if (is.null(rfx_working_parameter_prior_mean)) {
      if (num_rfx_components == 1) {
        alpha_init <- c(0)
      } else if (num_rfx_components > 1) {
        alpha_init <- rep(0, num_rfx_components)
      } else {
        stop("There must be at least 1 random effect component")
      }
    } else {
      alpha_init <- expand_dims_1d(
        rfx_working_parameter_prior_mean,
        num_rfx_components
      )
    }

    if (is.null(rfx_group_parameter_prior_mean)) {
      xi_init <- matrix(
        rep(alpha_init, num_rfx_groups),
        num_rfx_components,
        num_rfx_groups
      )
    } else {
      xi_init <- expand_dims_2d(
        rfx_group_parameter_prior_mean,
        num_rfx_components,
        num_rfx_groups
      )
    }

    if (is.null(rfx_working_parameter_prior_cov)) {
      sigma_alpha_init <- diag(1, num_rfx_components, num_rfx_components)
    } else {
      sigma_alpha_init <- expand_dims_2d_diag(
        rfx_working_parameter_prior_cov,
        num_rfx_components
      )
    }

    if (is.null(rfx_group_parameter_prior_cov)) {
      sigma_xi_init <- diag(1, num_rfx_components, num_rfx_components)
    } else {
      sigma_xi_init <- expand_dims_2d_diag(
        rfx_group_parameter_prior_cov,
        num_rfx_components
      )
    }

    sigma_xi_shape <- rfx_variance_prior_shape
    sigma_xi_scale <- rfx_variance_prior_scale

    # Random effects data structure and storage container
    rfx_dataset_train <- createRandomEffectsDataset(
      rfx_group_ids_train,
      rfx_basis_train
    )
    rfx_tracker_train <- createRandomEffectsTracker(rfx_group_ids_train)
    rfx_model <- createRandomEffectsModel(
      num_rfx_components,
      num_rfx_groups
    )
    rfx_model$set_working_parameter(alpha_init)
    rfx_model$set_group_parameters(xi_init)
    rfx_model$set_working_parameter_cov(sigma_alpha_init)
    rfx_model$set_group_parameter_cov(sigma_xi_init)
    rfx_model$set_variance_prior_shape(sigma_xi_shape)
    rfx_model$set_variance_prior_scale(sigma_xi_scale)
    rfx_samples <- createRandomEffectSamples(
      num_rfx_components,
      num_rfx_groups,
      rfx_tracker_train
    )
  }

  # Container of variance parameter samples
  num_actual_mcmc_iter <- num_mcmc * keep_every
  num_samples <- num_gfr + num_burnin + num_actual_mcmc_iter
  # Delete GFR samples from these containers after the fact if desired
  # num_retained_samples <- ifelse(keep_gfr, num_gfr, 0) + ifelse(keep_burnin, num_burnin, 0) + num_mcmc
  num_retained_samples <- num_gfr +
    ifelse(keep_burnin, num_burnin, 0) +
    num_mcmc * num_chains
  if (sample_sigma2_global) {
    global_var_samples <- rep(NA, num_retained_samples)
  }
  if (sample_sigma2_leaf) {
    leaf_scale_samples <- rep(NA, num_retained_samples)
  }
  if (include_mean_forest) {
    mean_forest_pred_train <- matrix(
      NA_real_,
      nrow(X_train),
      num_retained_samples
    )
  }
  if (include_variance_forest) {
    variance_forest_pred_train <- matrix(
      NA_real_,
      nrow(X_train),
      num_retained_samples
    )
  }
  sample_counter <- 0

  # Initialize the leaves of each tree in the mean forest
  if (include_mean_forest) {
    if (requires_basis) {
      init_values_mean_forest <- rep(0., ncol(leaf_basis_train))
    } else {
      init_values_mean_forest <- 0.
    }
    active_forest_mean$prepare_for_sampler(
      forest_dataset_train,
      outcome_train,
      forest_model_mean,
      leaf_model_mean_forest,
      init_values_mean_forest
    )
    active_forest_mean$adjust_residual(
      forest_dataset_train,
      outcome_train,
      forest_model_mean,
      requires_basis,
      FALSE
    )
  }

  # Initialize the leaves of each tree in the variance forest
  if (include_variance_forest) {
    active_forest_variance$prepare_for_sampler(
      forest_dataset_train,
      outcome_train,
      forest_model_variance,
      leaf_model_variance_forest,
      variance_forest_init
    )
  }

  # Run GFR (warm start) if specified
  if (num_gfr > 0) {
    for (i in 1:num_gfr) {
      # Keep all GFR samples at this stage -- remove from ForestSamples after MCMC
      # keep_sample <- ifelse(keep_gfr, TRUE, FALSE)
      keep_sample <- TRUE
      if (keep_sample) {
        sample_counter <- sample_counter + 1
      }
      # Print progress
      if (verbose) {
        if ((i %% 10 == 0) || (i == num_gfr)) {
          cat(
            "Sampling",
            i,
            "out of",
            num_gfr,
            "XBART (grow-from-root) draws\n"
          )
        }
      }

      if (include_mean_forest) {
        if (probit_outcome_model) {
          # Sample latent probit variable, z | -
          outcome_pred <- active_forest_mean$predict(
            forest_dataset_train
          )
          if (has_rfx) {
            rfx_pred <- rfx_model$predict(
              rfx_dataset_train,
              rfx_tracker_train
            )
            outcome_pred <- outcome_pred + rfx_pred
          }
          mu0 <- outcome_pred[y_train == 0]
          mu1 <- outcome_pred[y_train == 1]
          u0 <- runif(sum(y_train == 0), 0, pnorm(0 - mu0))
          u1 <- runif(sum(y_train == 1), pnorm(0 - mu1), 1)
          resid_train[y_train == 0] <- mu0 + qnorm(u0)
          resid_train[y_train == 1] <- mu1 + qnorm(u1)

          # Update outcome
          outcome_train$update_data(resid_train - outcome_pred)
        }

        # Sample mean forest
        forest_model_mean$sample_one_iteration(
          forest_dataset = forest_dataset_train,
          residual = outcome_train,
          forest_samples = forest_samples_mean,
          active_forest = active_forest_mean,
          rng = rng,
          forest_model_config = forest_model_config_mean,
          global_model_config = global_model_config,
          num_threads = num_threads,
          keep_forest = keep_sample,
          gfr = TRUE
        )

        # Cache train set predictions since they are already computed during sampling
        if (keep_sample) {
          mean_forest_pred_train[,
            sample_counter
          ] <- forest_model_mean$get_cached_forest_predictions()
        }
      }
      if (include_variance_forest) {
        forest_model_variance$sample_one_iteration(
          forest_dataset = forest_dataset_train,
          residual = outcome_train,
          forest_samples = forest_samples_variance,
          active_forest = active_forest_variance,
          rng = rng,
          forest_model_config = forest_model_config_variance,
          global_model_config = global_model_config,
          keep_forest = keep_sample,
          gfr = TRUE
        )

        # Cache train set predictions since they are already computed during sampling
        if (keep_sample) {
          variance_forest_pred_train[,
            sample_counter
          ] <- forest_model_variance$get_cached_forest_predictions()
        }
      }
      if (sample_sigma2_global) {
        current_sigma2 <- sampleGlobalErrorVarianceOneIteration(
          outcome_train,
          forest_dataset_train,
          rng,
          a_global,
          b_global
        )
        if (keep_sample) {
          global_var_samples[sample_counter] <- current_sigma2
        }
        global_model_config$update_global_error_variance(current_sigma2)
      }
      if (sample_sigma2_leaf) {
        leaf_scale_double <- sampleLeafVarianceOneIteration(
          active_forest_mean,
          rng,
          a_leaf,
          b_leaf
        )
        current_leaf_scale <- as.matrix(leaf_scale_double)
        if (keep_sample) {
          leaf_scale_samples[sample_counter] <- leaf_scale_double
        }
        forest_model_config_mean$update_leaf_model_scale(
          current_leaf_scale
        )
      }
      if (has_rfx) {
        rfx_model$sample_random_effect(
          rfx_dataset_train,
          outcome_train,
          rfx_tracker_train,
          rfx_samples,
          keep_sample,
          current_sigma2,
          rng
        )
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
          resetActiveForest(
            active_forest_mean,
            forest_samples_mean,
            forest_ind
          )
          resetForestModel(
            forest_model_mean,
            active_forest_mean,
            forest_dataset_train,
            outcome_train,
            TRUE
          )
          if (sample_sigma2_leaf) {
            leaf_scale_double <- leaf_scale_samples[forest_ind + 1]
            current_leaf_scale <- as.matrix(leaf_scale_double)
            forest_model_config_mean$update_leaf_model_scale(
              current_leaf_scale
            )
          }
        }
        if (include_variance_forest) {
          resetActiveForest(
            active_forest_variance,
            forest_samples_variance,
            forest_ind
          )
          resetForestModel(
            forest_model_variance,
            active_forest_variance,
            forest_dataset_train,
            outcome_train,
            FALSE
          )
        }
        if (has_rfx) {
          resetRandomEffectsModel(
            rfx_model,
            rfx_samples,
            forest_ind,
            sigma_alpha_init
          )
          resetRandomEffectsTracker(
            rfx_tracker_train,
            rfx_model,
            rfx_dataset_train,
            outcome_train,
            rfx_samples
          )
        }
        if (sample_sigma2_global) {
          current_sigma2 <- global_var_samples[forest_ind + 1]
          global_model_config$update_global_error_variance(
            current_sigma2
          )
        }
      } else if (has_prev_model) {
        warmstart_index <- ifelse(
          previous_model_decrement,
          previous_model_warmstart_sample_num - chain_num + 1,
          previous_model_warmstart_sample_num
        )
        if (include_mean_forest) {
          resetActiveForest(
            active_forest_mean,
            previous_forest_samples_mean,
            warmstart_index - 1
          )
          resetForestModel(
            forest_model_mean,
            active_forest_mean,
            forest_dataset_train,
            outcome_train,
            TRUE
          )
          if (
            sample_sigma2_leaf &&
              (!is.null(previous_leaf_var_samples))
          ) {
            leaf_scale_double <- previous_leaf_var_samples[
              warmstart_index
            ]
            current_leaf_scale <- as.matrix(leaf_scale_double)
            forest_model_config_mean$update_leaf_model_scale(
              current_leaf_scale
            )
          }
        }
        if (include_variance_forest) {
          resetActiveForest(
            active_forest_variance,
            previous_forest_samples_variance,
            warmstart_index - 1
          )
          resetForestModel(
            forest_model_variance,
            active_forest_variance,
            forest_dataset_train,
            outcome_train,
            FALSE
          )
        }
        if (has_rfx) {
          if (is.null(previous_rfx_samples)) {
            warning(
              "`previous_model_json` did not have any random effects samples, so the RFX sampler will be run from scratch while the forests and any other parameters are warm started"
            )
            rootResetRandomEffectsModel(
              rfx_model,
              alpha_init,
              xi_init,
              sigma_alpha_init,
              sigma_xi_init,
              sigma_xi_shape,
              sigma_xi_scale
            )
            rootResetRandomEffectsTracker(
              rfx_tracker_train,
              rfx_model,
              rfx_dataset_train,
              outcome_train
            )
          } else {
            resetRandomEffectsModel(
              rfx_model,
              previous_rfx_samples,
              warmstart_index - 1,
              sigma_alpha_init
            )
            resetRandomEffectsTracker(
              rfx_tracker_train,
              rfx_model,
              rfx_dataset_train,
              outcome_train,
              rfx_samples
            )
          }
        }
        if (sample_sigma2_global) {
          if (!is.null(previous_global_var_samples)) {
            current_sigma2 <- previous_global_var_samples[
              warmstart_index
            ]
            global_model_config$update_global_error_variance(
              current_sigma2
            )
          }
        }
      } else {
        if (include_mean_forest) {
          resetActiveForest(active_forest_mean)
          active_forest_mean$set_root_leaves(
            init_values_mean_forest / num_trees_mean
          )
          resetForestModel(
            forest_model_mean,
            active_forest_mean,
            forest_dataset_train,
            outcome_train,
            TRUE
          )
          if (sample_sigma2_leaf) {
            current_leaf_scale <- as.matrix(sigma2_leaf_init)
            forest_model_config_mean$update_leaf_model_scale(
              current_leaf_scale
            )
          }
        }
        if (include_variance_forest) {
          resetActiveForest(active_forest_variance)
          active_forest_variance$set_root_leaves(
            log(variance_forest_init) / num_trees_variance
          )
          resetForestModel(
            forest_model_variance,
            active_forest_variance,
            forest_dataset_train,
            outcome_train,
            FALSE
          )
        }
        if (has_rfx) {
          rootResetRandomEffectsModel(
            rfx_model,
            alpha_init,
            xi_init,
            sigma_alpha_init,
            sigma_xi_init,
            sigma_xi_shape,
            sigma_xi_scale
          )
          rootResetRandomEffectsTracker(
            rfx_tracker_train,
            rfx_model,
            rfx_dataset_train,
            outcome_train
          )
        }
        if (sample_sigma2_global) {
          current_sigma2 <- sigma2_init
          global_model_config$update_global_error_variance(
            current_sigma2
          )
        }
      }
      for (i in (num_gfr + 1):num_samples) {
        is_mcmc <- i > (num_gfr + num_burnin)
        if (is_mcmc) {
          mcmc_counter <- i - (num_gfr + num_burnin)
          if (mcmc_counter %% keep_every == 0) {
            keep_sample <- TRUE
          } else {
            keep_sample <- FALSE
          }
        } else {
          if (keep_burnin) {
            keep_sample <- TRUE
          } else {
            keep_sample <- FALSE
          }
        }
        if (keep_sample) {
          sample_counter <- sample_counter + 1
        }
        # Print progress
        if (verbose) {
          if (num_burnin > 0) {
            if (
              ((i - num_gfr) %% 100 == 0) ||
                ((i - num_gfr) == num_burnin)
            ) {
              cat(
                "Sampling",
                i - num_gfr,
                "out of",
                num_burnin,
                "BART burn-in draws; Chain number ",
                chain_num,
                "\n"
              )
            }
          }
          if (num_mcmc > 0) {
            if (
              ((i - num_gfr - num_burnin) %% 100 == 0) ||
                (i == num_samples)
            ) {
              cat(
                "Sampling",
                i - num_burnin - num_gfr,
                "out of",
                num_mcmc,
                "BART MCMC draws; Chain number ",
                chain_num,
                "\n"
              )
            }
          }
        }

        if (include_mean_forest) {
          if (probit_outcome_model) {
            # Sample latent probit variable, z | -
            outcome_pred <- active_forest_mean$predict(
              forest_dataset_train
            )
            if (has_rfx) {
              rfx_pred <- rfx_model$predict(
                rfx_dataset_train,
                rfx_tracker_train
              )
              outcome_pred <- outcome_pred + rfx_pred
            }
            mu0 <- outcome_pred[y_train == 0]
            mu1 <- outcome_pred[y_train == 1]
            u0 <- runif(sum(y_train == 0), 0, pnorm(0 - mu0))
            u1 <- runif(sum(y_train == 1), pnorm(0 - mu1), 1)
            resid_train[y_train == 0] <- mu0 + qnorm(u0)
            resid_train[y_train == 1] <- mu1 + qnorm(u1)

            # Update outcome
            outcome_train$update_data(resid_train - outcome_pred)
          }

          forest_model_mean$sample_one_iteration(
            forest_dataset = forest_dataset_train,
            residual = outcome_train,
            forest_samples = forest_samples_mean,
            active_forest = active_forest_mean,
            rng = rng,
            forest_model_config = forest_model_config_mean,
            global_model_config = global_model_config,
            keep_forest = keep_sample,
            gfr = FALSE
          )

          # Cache train set predictions since they are already computed during sampling
          if (keep_sample) {
            mean_forest_pred_train[,
              sample_counter
            ] <- forest_model_mean$get_cached_forest_predictions()
          }
        }
        if (include_variance_forest) {
          forest_model_variance$sample_one_iteration(
            forest_dataset = forest_dataset_train,
            residual = outcome_train,
            forest_samples = forest_samples_variance,
            active_forest = active_forest_variance,
            rng = rng,
            forest_model_config = forest_model_config_variance,
            global_model_config = global_model_config,
            keep_forest = keep_sample,
            gfr = FALSE
          )

          # Cache train set predictions since they are already computed during sampling
          if (keep_sample) {
            variance_forest_pred_train[,
              sample_counter
            ] <- forest_model_variance$get_cached_forest_predictions()
          }
        }
        if (sample_sigma2_global) {
          current_sigma2 <- sampleGlobalErrorVarianceOneIteration(
            outcome_train,
            forest_dataset_train,
            rng,
            a_global,
            b_global
          )
          if (keep_sample) {
            global_var_samples[sample_counter] <- current_sigma2
          }
          global_model_config$update_global_error_variance(
            current_sigma2
          )
        }
        if (sample_sigma2_leaf) {
          leaf_scale_double <- sampleLeafVarianceOneIteration(
            active_forest_mean,
            rng,
            a_leaf,
            b_leaf
          )
          current_leaf_scale <- as.matrix(leaf_scale_double)
          if (keep_sample) {
            leaf_scale_samples[sample_counter] <- leaf_scale_double
          }
          forest_model_config_mean$update_leaf_model_scale(
            current_leaf_scale
          )
        }
        if (has_rfx) {
          rfx_model$sample_random_effect(
            rfx_dataset_train,
            outcome_train,
            rfx_tracker_train,
            rfx_samples,
            keep_sample,
            current_sigma2,
            rng
          )
        }
      }
    }
  }

  # Remove GFR samples if they are not to be retained
  if ((!keep_gfr) && (num_gfr > 0)) {
    for (i in 1:num_gfr) {
      if (include_mean_forest) {
        forest_samples_mean$delete_sample(0)
      }
      if (include_variance_forest) {
        forest_samples_variance$delete_sample(0)
      }
      if (has_rfx) {
        rfx_samples$delete_sample(0)
      }
    }
    if (include_mean_forest) {
      mean_forest_pred_train <- mean_forest_pred_train[,
        (num_gfr + 1):ncol(mean_forest_pred_train)
      ]
    }
    if (include_variance_forest) {
      variance_forest_pred_train <- variance_forest_pred_train[,
        (num_gfr + 1):ncol(variance_forest_pred_train)
      ]
    }
    if (sample_sigma2_global) {
      global_var_samples <- global_var_samples[
        (num_gfr + 1):length(global_var_samples)
      ]
    }
    if (sample_sigma2_leaf) {
      leaf_scale_samples <- leaf_scale_samples[
        (num_gfr + 1):length(leaf_scale_samples)
      ]
    }
    num_retained_samples <- num_retained_samples - num_gfr
  }

  # Mean forest predictions
  if (include_mean_forest) {
    # y_hat_train <- forest_samples_mean$predict(forest_dataset_train)*y_std_train + y_bar_train
    y_hat_train <- mean_forest_pred_train * y_std_train + y_bar_train
    if (has_test) {
      y_hat_test <- forest_samples_mean$predict(forest_dataset_test) *
        y_std_train +
        y_bar_train
    }
  }

  # Variance forest predictions
  if (include_variance_forest) {
    # sigma2_x_hat_train <- forest_samples_variance$predict(forest_dataset_train)
    sigma2_x_hat_train <- exp(variance_forest_pred_train)
    if (has_test) {
      sigma2_x_hat_test <- forest_samples_variance$predict(
        forest_dataset_test
      )
    }
  }

  # Random effects predictions
  if (has_rfx) {
    rfx_preds_train <- rfx_samples$predict(
      rfx_group_ids_train,
      rfx_basis_train
    ) *
      y_std_train
    y_hat_train <- y_hat_train + rfx_preds_train
  }
  if ((has_rfx_test) && (has_test)) {
    rfx_preds_test <- rfx_samples$predict(
      rfx_group_ids_test,
      rfx_basis_test
    ) *
      y_std_train
    y_hat_test <- y_hat_test + rfx_preds_test
  }

  # Global error variance
  if (sample_sigma2_global) {
    sigma2_global_samples <- global_var_samples * (y_std_train^2)
  }

  # Leaf parameter variance
  if (sample_sigma2_leaf) {
    tau_samples <- leaf_scale_samples
  }

  # Rescale variance forest prediction by global sigma2 (sampled or constant)
  if (include_variance_forest) {
    if (sample_sigma2_global) {
      sigma2_x_hat_train <- sapply(1:num_retained_samples, function(i) {
        sigma2_x_hat_train[, i] * sigma2_global_samples[i]
      })
      if (has_test) {
        sigma2_x_hat_test <- sapply(
          1:num_retained_samples,
          function(i) {
            sigma2_x_hat_test[, i] * sigma2_global_samples[i]
          }
        )
      }
    } else {
      sigma2_x_hat_train <- sigma2_x_hat_train *
        sigma2_init *
        y_std_train *
        y_std_train
      if (has_test) {
        sigma2_x_hat_test <- sigma2_x_hat_test *
          sigma2_init *
          y_std_train *
          y_std_train
      }
    }
  }

  # Return results as a list
  model_params <- list(
    "sigma2_init" = sigma2_init,
    "sigma2_leaf_init" = sigma2_leaf_init,
    "a_global" = a_global,
    "b_global" = b_global,
    "a_leaf" = a_leaf,
    "b_leaf" = b_leaf,
    "a_forest" = a_forest,
    "b_forest" = b_forest,
    "outcome_mean" = y_bar_train,
    "outcome_scale" = y_std_train,
    "standardize" = standardize,
    "leaf_dimension" = leaf_dimension,
    "is_leaf_constant" = is_leaf_constant,
    "leaf_regression" = leaf_regression,
    "requires_basis" = requires_basis,
    "num_covariates" = num_cov_orig,
    "num_basis" = ifelse(
      is.null(leaf_basis_train),
      0,
      ncol(leaf_basis_train)
    ),
    "num_samples" = num_retained_samples,
    "num_gfr" = num_gfr,
    "num_burnin" = num_burnin,
    "num_mcmc" = num_mcmc,
    "keep_every" = keep_every,
    "num_chains" = num_chains,
    "has_basis" = !is.null(leaf_basis_train),
    "has_rfx" = has_rfx,
    "has_rfx_basis" = has_basis_rfx,
    "num_rfx_basis" = num_basis_rfx,
    "sample_sigma2_global" = sample_sigma2_global,
    "sample_sigma2_leaf" = sample_sigma2_leaf,
    "include_mean_forest" = include_mean_forest,
    "include_variance_forest" = include_variance_forest,
    "probit_outcome_model" = probit_outcome_model,
    "rfx_model_spec" = rfx_model_spec
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
    result[["sigma2_x_hat_train"]] = sigma2_x_hat_train
    if (has_test) result[["sigma2_x_hat_test"]] = sigma2_x_hat_test
  }
  if (sample_sigma2_global) {
    result[["sigma2_global_samples"]] = sigma2_global_samples
  }
  if (sample_sigma2_leaf) {
    result[["sigma2_leaf_samples"]] = tau_samples
  }
  if (has_rfx) {
    result[["rfx_samples"]] = rfx_samples
    result[["rfx_preds_train"]] = rfx_preds_train
    result[["rfx_unique_group_ids"]] = levels(group_ids_factor)
  }
  if ((has_rfx_test) && (has_test)) {
    result[["rfx_preds_test"]] = rfx_preds_test
  }
  class(result) <- "bartmodel"

  # Clean up classes with external pointers to C++ data structures
  if (include_mean_forest) {
    rm(forest_model_mean)
  }
  if (include_variance_forest) {
    rm(forest_model_variance)
  }
  rm(forest_dataset_train)
  if (has_test) {
    rm(forest_dataset_test)
  }
  if (has_rfx) {
    rm(rfx_dataset_train, rfx_tracker_train, rfx_model)
  }
  rm(outcome_train)
  rm(rng)

  # Restore global RNG state if user provided a random seed
  if (custom_rng) {
    .Random.seed <- original_global_seed
  }

  return(result)
}

#' Predict from a sampled BART model on new data
#'
#' @param object Object of type `bart` containing draws of a regression forest and associated sampling outputs.
#' @param X Covariates used to determine tree leaf predictions for each observation. Must be passed as a matrix or dataframe.
#' @param leaf_basis (Optional) Bases used for prediction (by e.g. dot product with leaf values). Default: `NULL`.
#' @param rfx_group_ids (Optional) Test set group labels used for an additive random effects model.
#' We do not currently support (but plan to in the near future), test set evaluation for group labels
#' that were not in the training set.
#' @param rfx_basis (Optional) Test set basis for "random-slope" regression in additive random effects model.
#' @param type (Optional) Type of prediction to return. Options are "mean", which averages the predictions from every draw of a BART model, and "posterior", which returns the entire matrix of posterior predictions. Default: "posterior".
#' @param terms (Optional) Which model terms to include in the prediction. This can be a single term or a list of model terms. Options include "y_hat", "mean_forest", "rfx", "variance_forest", or "all". If a model doesn't have mean forest, random effects, or variance forest predictions, but one of those terms is request, the request will simply be ignored. If none of the requested terms are present in a model, this function will return `NULL` along with a warning. Default: "all".
#' @param scale (Optional) Scale of mean function predictions. Options are "linear", which returns predictions on the original scale of the mean forest / RFX terms, and "probability", which transforms predictions into a probability of observing `y == 1`. "probability" is only valid for models fit with a probit outcome model. Default: "linear".
#' @param ... (Optional) Other prediction parameters.
#'
#' @return List of prediction matrices or single prediction matrix / vector, depending on the terms requested.
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
#' bart_model <- bart(X_train = X_train, y_train = y_train,
#'                    num_gfr = 10, num_burnin = 0, num_mcmc = 10)
#' y_hat_test <- predict(bart_model, X=X_test)$y_hat
predict.bartmodel <- function(
  object,
  X,
  leaf_basis = NULL,
  rfx_group_ids = NULL,
  rfx_basis = NULL,
  type = "posterior",
  terms = "all",
  scale = "linear",
  ...
) {
  # Handle mean function scale
  if (!is.character(scale)) {
    stop("scale must be a string or character vector")
  }
  if (!(scale %in% c("linear", "probability"))) {
    stop("scale must either be 'linear' or 'probability'")
  }
  is_probit <- object$model_params$probit_outcome_model
  if ((scale == "probability") && (!is_probit)) {
    stop(
      "scale cannot be 'probability' for models not fit with a probit outcome model"
    )
  }
  probability_scale <- scale == "probability"

  # Handle prediction type
  if (!is.character(type)) {
    stop("type must be a string or character vector")
  }
  if (!(type %in% c("mean", "posterior"))) {
    stop("type must either be 'mean' or 'posterior'")
  }
  predict_mean <- type == "mean"

  # Handle prediction terms
  rfx_model_spec <- object$model_params$rfx_model_spec
  rfx_intercept <- rfx_model_spec == "intercept_only"
  if (!is.character(terms)) {
    stop("type must be a string or character vector")
  }
  num_terms <- length(terms)
  has_mean_forest <- object$model_params$include_mean_forest
  has_variance_forest <- object$model_params$include_variance_forest
  has_rfx <- object$model_params$has_rfx
  has_y_hat <- has_mean_forest || has_rfx
  predict_y_hat <- (((has_y_hat) && ("y_hat" %in% terms)) ||
    ((has_y_hat) && ("all" %in% terms)))
  predict_mean_forest <- (((has_mean_forest) && ("mean_forest" %in% terms)) ||
    ((has_mean_forest) && ("all" %in% terms)))
  predict_rfx <- (((has_rfx) && ("rfx" %in% terms)) ||
    ((has_rfx) && ("all" %in% terms)))
  predict_variance_forest <- (((has_variance_forest) &&
    ("variance_forest" %in% terms)) ||
    ((has_variance_forest) && ("all" %in% terms)))
  predict_count <- sum(c(
    predict_y_hat,
    predict_mean_forest,
    predict_rfx,
    predict_variance_forest
  ))
  if (predict_count == 0) {
    warning(paste0(
      "None of the requested model terms, ",
      paste(terms, collapse = ", "),
      ", were fit in this model"
    ))
    return(NULL)
  }
  predict_rfx_intermediate <- (predict_y_hat && has_rfx)
  predict_mean_forest_intermediate <- (predict_y_hat && has_mean_forest)

  # Check that we have at least one term to predict on probability scale
  if (
    probability_scale &&
      !predict_y_hat &&
      !predict_mean_forest &&
      !predict_rfx
  ) {
    stop(
      "scale can only be 'probability' if at least one mean term is requested"
    )
  }

  # Check that covariates are matrix or data frame
  if ((!is.data.frame(X)) && (!is.matrix(X))) {
    stop("X must be a matrix or dataframe")
  }

  # Convert all input data to matrices if not already converted
  if ((is.null(dim(leaf_basis))) && (!is.null(leaf_basis))) {
    leaf_basis <- as.matrix(leaf_basis)
  }
  if ((is.null(dim(rfx_basis))) && (!is.null(rfx_basis))) {
    if (predict_rfx) rfx_basis <- as.matrix(rfx_basis)
  }

  # Data checks
  if ((object$model_params$requires_basis) && (is.null(leaf_basis))) {
    stop("Basis (leaf_basis) must be provided for this model")
  }
  if ((!is.null(leaf_basis)) && (nrow(X) != nrow(leaf_basis))) {
    stop("X and leaf_basis must have the same number of rows")
  }
  if (object$model_params$num_covariates != ncol(X)) {
    stop(
      "X must contain the same number of columns as the BART model's training dataset"
    )
  }
  if ((predict_rfx) && (is.null(rfx_group_ids))) {
    stop(
      "Random effect group labels (rfx_group_ids) must be provided for this model"
    )
  }
  if ((predict_rfx) && (is.null(rfx_basis)) && (!rfx_intercept)) {
    stop("Random effects basis (rfx_basis) must be provided for this model")
  }
  if ((object$model_params$num_rfx_basis > 0) && (!rfx_intercept)) {
    if (ncol(rfx_basis) != object$model_params$num_rfx_basis) {
      stop(
        "Random effects basis has a different dimension than the basis used to train this model"
      )
    }
  }

  # Preprocess covariates
  train_set_metadata <- object$train_set_metadata
  X <- preprocessPredictionData(X, train_set_metadata)

  # Recode group IDs to integer vector (if passed as, for example, a vector of county names, etc...)
  if (predict_rfx) {
    if (!is.null(rfx_group_ids)) {
      rfx_unique_group_ids <- object$rfx_unique_group_ids
      group_ids_factor <- factor(rfx_group_ids, levels = rfx_unique_group_ids)
      if (sum(is.na(group_ids_factor)) > 0) {
        stop(
          "All random effect group labels provided in rfx_group_ids must have been present in rfx_group_ids_train"
        )
      }
      rfx_group_ids <- as.integer(group_ids_factor)
    }
  }

  # Handle RFX model specification
  if (has_rfx) {
    if (object$model_params$rfx_model_spec == "custom") {
      if (is.null(rfx_basis)) {
        stop(
          "A user-provided basis (`rfx_basis`) must be provided when the model was sampled with a random effects model spec set to 'custom'"
        )
      }
    } else if (object$model_params$rfx_model_spec == "intercept_only") {
      # Only construct a basis if user-provided basis missing
      if (is.null(rfx_basis)) {
        rfx_basis <- matrix(
          rep(1, nrow(X)),
          nrow = nrow(X),
          ncol = 1
        )
      }
    }
  }

  # Create prediction dataset
  if (!is.null(leaf_basis)) {
    prediction_dataset <- createForestDataset(X, leaf_basis)
  } else {
    prediction_dataset <- createForestDataset(X)
  }

  # Compute variance forest predictions
  if (predict_variance_forest) {
    s_x_raw <- object$variance_forests$predict(prediction_dataset)
  }

  # Scale variance forest predictions
  num_samples <- object$model_params$num_samples
  y_std <- object$model_params$outcome_scale
  y_bar <- object$model_params$outcome_mean
  sigma2_init <- object$model_params$sigma2_init
  if (predict_variance_forest) {
    if (object$model_params$sample_sigma2_global) {
      sigma2_global_samples <- object$sigma2_global_samples
      variance_forest_predictions <- sapply(1:num_samples, function(i) {
        s_x_raw[, i] * sigma2_global_samples[i]
      })
    } else {
      variance_forest_predictions <- s_x_raw * sigma2_init * y_std * y_std
    }
    if (predict_mean) {
      variance_forest_predictions <- rowMeans(variance_forest_predictions)
    }
  }

  # Compute mean forest predictions
  if (predict_mean_forest || predict_mean_forest_intermediate) {
    mean_forest_predictions <- object$mean_forests$predict(
      prediction_dataset
    ) *
      y_std +
      y_bar
  }

  # Compute rfx predictions (if needed)
  if (predict_rfx || predict_rfx_intermediate) {
    if (!is.null(rfx_basis)) {
      rfx_predictions <- object$rfx_samples$predict(
        rfx_group_ids,
        rfx_basis
      ) *
        y_std
    } else {
      # Sanity check -- this branch should only occur if rfx_model_spec == "intercept_only"
      if (!rfx_intercept) {
        stop(
          "rfx_basis must be provided for random effects models with random slopes"
        )
      }

      # Extract the raw RFX samples and scale by train set outcome standard deviation
      rfx_param_list <- object$rfx_samples$extract_parameter_samples()
      rfx_beta_draws <- rfx_param_list$beta_samples * y_std

      # Promote to an array with consistent dimensions when there's one rfx term
      if (length(dim(rfx_beta_draws)) == 2) {
        dim(rfx_beta_draws) <- c(1, dim(rfx_beta_draws))
      }

      # Construct a matrix with the appropriate group random effects arranged for each observation
      rfx_predictions_raw <- array(
        NA,
        dim = c(
          nrow(X),
          ncol(rfx_basis),
          object$model_params$num_samples
        )
      )
      for (i in 1:nrow(X)) {
        rfx_predictions_raw[i, , ] <-
          rfx_beta_draws[, rfx_group_ids[i], ]
      }

      # Intercept-only model, so the random effect prediction is simply the
      # value of the respective group's intercept coefficient for each observation
      rfx_predictions = rfx_predictions_raw[, 1, ]
    }
  }

  # Combine into y hat predictions
  if (probability_scale) {
    if (predict_y_hat && has_mean_forest && has_rfx) {
      y_hat <- pnorm(mean_forest_predictions + rfx_predictions)
      mean_forest_predictions <- pnorm(mean_forest_predictions)
      rfx_predictions <- pnorm(rfx_predictions)
    } else if (predict_y_hat && has_mean_forest) {
      y_hat <- pnorm(mean_forest_predictions)
      mean_forest_predictions <- pnorm(mean_forest_predictions)
    } else if (predict_y_hat && has_rfx) {
      y_hat <- pnorm(rfx_predictions)
      rfx_predictions <- pnorm(rfx_predictions)
    }
  } else {
    if (predict_y_hat && has_mean_forest && has_rfx) {
      y_hat <- mean_forest_predictions + rfx_predictions
    } else if (predict_y_hat && has_mean_forest) {
      y_hat <- mean_forest_predictions
    } else if (predict_y_hat && has_rfx) {
      y_hat <- rfx_predictions
    }
  }

  # Collapse to posterior mean predictions if requested
  if (predict_mean) {
    if (predict_mean_forest) {
      mean_forest_predictions <- rowMeans(mean_forest_predictions)
    }
    if (predict_rfx) {
      rfx_predictions <- rowMeans(rfx_predictions)
    }
    if (predict_y_hat) {
      y_hat <- rowMeans(y_hat)
    }
  }

  if (predict_count == 1) {
    if (predict_y_hat) {
      return(y_hat)
    } else if (predict_mean_forest) {
      return(mean_forest_predictions)
    } else if (predict_rfx) {
      return(rfx_predictions)
    } else if (predict_variance_forest) {
      return(variance_forest_predictions)
    }
  } else {
    result <- list()
    if (predict_y_hat) {
      result[["y_hat"]] = y_hat
    } else {
      result[["y_hat"]] <- NULL
    }
    if (predict_mean_forest) {
      result[["mean_forest_predictions"]] = mean_forest_predictions
    } else {
      result[["mean_forest_predictions"]] <- NULL
    }
    if (predict_rfx) {
      result[["rfx_predictions"]] = rfx_predictions
    } else {
      result[["rfx_predictions"]] <- NULL
    }
    if (predict_variance_forest) {
      result[["variance_forest_predictions"]] = variance_forest_predictions
    } else {
      result[["variance_forest_predictions"]] <- NULL
    }
    return(result)
  }
}

#' Extract raw sample values for each of the random effect parameter terms.
#'
#' @param object Object of type `bartmodel` containing draws of a BART model and associated sampling outputs.
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
#' rfx_group_ids_test <- group_ids[test_inds]
#' rfx_group_ids_train <- group_ids[train_inds]
#' rfx_basis_test <- rfx_basis[test_inds,]
#' rfx_basis_train <- rfx_basis[train_inds,]
#' rfx_term_test <- rfx_term[test_inds]
#' rfx_term_train <- rfx_term[train_inds]
#' bart_model <- bart(X_train = X_train, y_train = y_train, X_test = X_test,
#'                    rfx_group_ids_train = rfx_group_ids_train,
#'                    rfx_group_ids_test = rfx_group_ids_test,
#'                    rfx_basis_train = rfx_basis_train,
#'                    rfx_basis_test = rfx_basis_test,
#'                    num_gfr = 10, num_burnin = 0, num_mcmc = 10)
#' rfx_samples <- getRandomEffectSamples(bart_model)
getRandomEffectSamples.bartmodel <- function(object, ...) {
  result = list()

  if (!object$model_params$has_rfx) {
    warning("This model has no RFX terms, returning an empty list")
    return(result)
  }

  # Extract the samples
  result <- object$rfx_samples$extract_parameter_samples()

  # Scale by sd(y_train)
  result$beta_samples <- result$beta_samples *
    object$model_params$outcome_scale
  result$xi_samples <- result$xi_samples * object$model_params$outcome_scale
  result$alpha_samples <- result$alpha_samples *
    object$model_params$outcome_scale
  result$sigma_samples <- result$sigma_samples *
    (object$model_params$outcome_scale^2)

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
#' bart_model <- bart(X_train = X_train, y_train = y_train,
#'                    num_gfr = 10, num_burnin = 0, num_mcmc = 10)
#' bart_json <- saveBARTModelToJson(bart_model)
saveBARTModelToJson <- function(object) {
  jsonobj <- createCppJson()

  if (!inherits(object, "bartmodel")) {
    stop("`object` must be a BART model")
  }

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
  jsonobj$add_scalar(
    "num_numeric_vars",
    object$train_set_metadata$num_numeric_vars
  )
  jsonobj$add_scalar(
    "num_ordered_cat_vars",
    object$train_set_metadata$num_ordered_cat_vars
  )
  jsonobj$add_scalar(
    "num_unordered_cat_vars",
    object$train_set_metadata$num_unordered_cat_vars
  )
  if (object$train_set_metadata$num_numeric_vars > 0) {
    jsonobj$add_string_vector(
      "numeric_vars",
      object$train_set_metadata$numeric_vars
    )
  }
  if (object$train_set_metadata$num_ordered_cat_vars > 0) {
    jsonobj$add_string_vector(
      "ordered_cat_vars",
      object$train_set_metadata$ordered_cat_vars
    )
    jsonobj$add_string_list(
      "ordered_unique_levels",
      object$train_set_metadata$ordered_unique_levels
    )
  }
  if (object$train_set_metadata$num_unordered_cat_vars > 0) {
    jsonobj$add_string_vector(
      "unordered_cat_vars",
      object$train_set_metadata$unordered_cat_vars
    )
    jsonobj$add_string_list(
      "unordered_unique_levels",
      object$train_set_metadata$unordered_unique_levels
    )
  }

  # Add global parameters
  jsonobj$add_scalar("outcome_scale", object$model_params$outcome_scale)
  jsonobj$add_scalar("outcome_mean", object$model_params$outcome_mean)
  jsonobj$add_boolean("standardize", object$model_params$standardize)
  jsonobj$add_scalar("sigma2_init", object$model_params$sigma2_init)
  jsonobj$add_boolean(
    "sample_sigma2_global",
    object$model_params$sample_sigma2_global
  )
  jsonobj$add_boolean(
    "sample_sigma2_leaf",
    object$model_params$sample_sigma2_leaf
  )
  jsonobj$add_boolean(
    "include_mean_forest",
    object$model_params$include_mean_forest
  )
  jsonobj$add_boolean(
    "include_variance_forest",
    object$model_params$include_variance_forest
  )
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
  jsonobj$add_boolean(
    "probit_outcome_model",
    object$model_params$probit_outcome_model
  )
  jsonobj$add_string(
    "rfx_model_spec",
    object$model_params$rfx_model_spec
  )
  if (object$model_params$sample_sigma2_global) {
    jsonobj$add_vector(
      "sigma2_global_samples",
      object$sigma2_global_samples,
      "parameters"
    )
  }
  if (object$model_params$sample_sigma2_leaf) {
    jsonobj$add_vector(
      "sigma2_leaf_samples",
      object$sigma2_leaf_samples,
      "parameters"
    )
  }

  # Add random effects (if present)
  if (object$model_params$has_rfx) {
    jsonobj$add_random_effects(object$rfx_samples)
    jsonobj$add_string_vector(
      "rfx_unique_group_ids",
      object$rfx_unique_group_ids
    )
  }

  # Add covariate preprocessor metadata
  preprocessor_metadata_string <- savePreprocessorToJsonString(
    object$train_set_metadata
  )
  jsonobj$add_string("preprocessor_metadata", preprocessor_metadata_string)

  return(jsonobj)
}

#' Convert the persistent aspects of a BART model to (in-memory) JSON and save to a file
#'
#' @param object Object of type `bartmodel` containing draws of a BART model and associated sampling outputs.
#' @param filename String of filepath, must end in ".json"
#'
#' @return None
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
#' bart_model <- bart(X_train = X_train, y_train = y_train,
#'                    num_gfr = 10, num_burnin = 0, num_mcmc = 10)
#' tmpjson <- tempfile(fileext = ".json")
#' saveBARTModelToJsonFile(bart_model, file.path(tmpjson))
#' unlink(tmpjson)
saveBARTModelToJsonFile <- function(object, filename) {
  # Convert to Json
  jsonobj <- saveBARTModelToJson(object)

  # Save to file
  jsonobj$save_file(filename)
}

#' Convert the persistent aspects of a BART model to (in-memory) JSON string
#'
#' @param object Object of type `bartmodel` containing draws of a BART model and associated sampling outputs.
#' @return in-memory JSON string
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
#' bart_model <- bart(X_train = X_train, y_train = y_train,
#'                    num_gfr = 10, num_burnin = 0, num_mcmc = 10)
#' bart_json_string <- saveBARTModelToJsonString(bart_model)
saveBARTModelToJsonString <- function(object) {
  # Convert to Json
  jsonobj <- saveBARTModelToJson(object)

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
#' bart_model <- bart(X_train = X_train, y_train = y_train,
#'                    num_gfr = 10, num_burnin = 0, num_mcmc = 10)
#' bart_json <- saveBARTModelToJson(bart_model)
#' bart_model_roundtrip <- createBARTModelFromJson(bart_json)
createBARTModelFromJson <- function(json_object) {
  # Initialize the BCF model
  output <- list()

  # Unpack the forests
  include_mean_forest <- json_object$get_boolean("include_mean_forest")
  include_variance_forest <- json_object$get_boolean(
    "include_variance_forest"
  )
  if (include_mean_forest) {
    output[["mean_forests"]] <- loadForestContainerJson(
      json_object,
      "forest_0"
    )
    if (include_variance_forest) {
      output[["variance_forests"]] <- loadForestContainerJson(
        json_object,
        "forest_1"
      )
    }
  } else {
    output[["variance_forests"]] <- loadForestContainerJson(
      json_object,
      "forest_0"
    )
  }

  # Unpack metadata
  train_set_metadata = list()
  train_set_metadata[["num_numeric_vars"]] <- json_object$get_scalar(
    "num_numeric_vars"
  )
  train_set_metadata[["num_ordered_cat_vars"]] <- json_object$get_scalar(
    "num_ordered_cat_vars"
  )
  train_set_metadata[["num_unordered_cat_vars"]] <- json_object$get_scalar(
    "num_unordered_cat_vars"
  )
  if (train_set_metadata[["num_numeric_vars"]] > 0) {
    train_set_metadata[["numeric_vars"]] <- json_object$get_string_vector(
      "numeric_vars"
    )
  }
  if (train_set_metadata[["num_ordered_cat_vars"]] > 0) {
    train_set_metadata[[
      "ordered_cat_vars"
    ]] <- json_object$get_string_vector("ordered_cat_vars")
    train_set_metadata[[
      "ordered_unique_levels"
    ]] <- json_object$get_string_list(
      "ordered_unique_levels",
      train_set_metadata[["ordered_cat_vars"]]
    )
  }
  if (train_set_metadata[["num_unordered_cat_vars"]] > 0) {
    train_set_metadata[[
      "unordered_cat_vars"
    ]] <- json_object$get_string_vector("unordered_cat_vars")
    train_set_metadata[[
      "unordered_unique_levels"
    ]] <- json_object$get_string_list(
      "unordered_unique_levels",
      train_set_metadata[["unordered_cat_vars"]]
    )
  }
  output[["train_set_metadata"]] <- train_set_metadata

  # Unpack model params
  model_params = list()
  model_params[["outcome_scale"]] <- json_object$get_scalar("outcome_scale")
  model_params[["outcome_mean"]] <- json_object$get_scalar("outcome_mean")
  model_params[["standardize"]] <- json_object$get_boolean("standardize")
  model_params[["sigma2_init"]] <- json_object$get_scalar("sigma2_init")
  model_params[["sample_sigma2_global"]] <- json_object$get_boolean(
    "sample_sigma2_global"
  )
  model_params[["sample_sigma2_leaf"]] <- json_object$get_boolean(
    "sample_sigma2_leaf"
  )
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
  model_params[["requires_basis"]] <- json_object$get_boolean(
    "requires_basis"
  )
  model_params[["probit_outcome_model"]] <- json_object$get_boolean(
    "probit_outcome_model"
  )
  model_params[["rfx_model_spec"]] <- json_object$get_string(
    "rfx_model_spec"
  )

  output[["model_params"]] <- model_params

  # Unpack sampled parameters
  if (model_params[["sample_sigma2_global"]]) {
    output[["sigma2_global_samples"]] <- json_object$get_vector(
      "sigma2_global_samples",
      "parameters"
    )
  }
  if (model_params[["sample_sigma2_leaf"]]) {
    output[["sigma2_leaf_samples"]] <- json_object$get_vector(
      "sigma2_leaf_samples",
      "parameters"
    )
  }

  # Unpack random effects
  if (model_params[["has_rfx"]]) {
    output[["rfx_unique_group_ids"]] <- json_object$get_string_vector(
      "rfx_unique_group_ids"
    )
    output[["rfx_samples"]] <- loadRandomEffectSamplesJson(json_object, 0)
  }

  # Unpack covariate preprocessor
  preprocessor_metadata_string <- json_object$get_string(
    "preprocessor_metadata"
  )
  output[["train_set_metadata"]] <- createPreprocessorFromJsonString(
    preprocessor_metadata_string
  )

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
#' bart_model <- bart(X_train = X_train, y_train = y_train,
#'                    num_gfr = 10, num_burnin = 0, num_mcmc = 10)
#' tmpjson <- tempfile(fileext = ".json")
#' saveBARTModelToJsonFile(bart_model, file.path(tmpjson))
#' bart_model_roundtrip <- createBARTModelFromJsonFile(file.path(tmpjson))
#' unlink(tmpjson)
createBARTModelFromJsonFile <- function(json_filename) {
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
#' bart_model <- bart(X_train = X_train, y_train = y_train,
#'                    num_gfr = 10, num_burnin = 0, num_mcmc = 10)
#' bart_json <- saveBARTModelToJsonString(bart_model)
#' bart_model_roundtrip <- createBARTModelFromJsonString(bart_json)
#' y_hat_mean_roundtrip <- rowMeans(predict(bart_model_roundtrip, X=X_train)$y_hat)
createBARTModelFromJsonString <- function(json_string) {
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
#' bart_model <- bart(X_train = X_train, y_train = y_train,
#'                    num_gfr = 10, num_burnin = 0, num_mcmc = 10)
#' bart_json <- list(saveBARTModelToJson(bart_model))
#' bart_model_roundtrip <- createBARTModelFromCombinedJson(bart_json)
createBARTModelFromCombinedJson <- function(json_object_list) {
  # Initialize the BCF model
  output <- list()

  # For scalar / preprocessing details which aren't sample-dependent,
  # defer to the first json
  json_object_default <- json_object_list[[1]]

  # Unpack the forests
  include_mean_forest <- json_object_default$get_boolean(
    "include_mean_forest"
  )
  include_variance_forest <- json_object_default$get_boolean(
    "include_variance_forest"
  )
  if (include_mean_forest) {
    output[["mean_forests"]] <- loadForestContainerCombinedJson(
      json_object_list,
      "forest_0"
    )
    if (include_variance_forest) {
      output[["variance_forests"]] <- loadForestContainerCombinedJson(
        json_object_list,
        "forest_1"
      )
    }
  } else {
    output[["variance_forests"]] <- loadForestContainerCombinedJson(
      json_object_list,
      "forest_0"
    )
  }

  # Unpack metadata
  train_set_metadata = list()
  train_set_metadata[["num_numeric_vars"]] <- json_object_default$get_scalar(
    "num_numeric_vars"
  )
  train_set_metadata[[
    "num_ordered_cat_vars"
  ]] <- json_object_default$get_scalar("num_ordered_cat_vars")
  train_set_metadata[[
    "num_unordered_cat_vars"
  ]] <- json_object_default$get_scalar("num_unordered_cat_vars")
  if (train_set_metadata[["num_numeric_vars"]] > 0) {
    train_set_metadata[[
      "numeric_vars"
    ]] <- json_object_default$get_string_vector("numeric_vars")
  }
  if (train_set_metadata[["num_ordered_cat_vars"]] > 0) {
    train_set_metadata[[
      "ordered_cat_vars"
    ]] <- json_object_default$get_string_vector("ordered_cat_vars")
    train_set_metadata[[
      "ordered_unique_levels"
    ]] <- json_object_default$get_string_list(
      "ordered_unique_levels",
      train_set_metadata[["ordered_cat_vars"]]
    )
  }
  if (train_set_metadata[["num_unordered_cat_vars"]] > 0) {
    train_set_metadata[[
      "unordered_cat_vars"
    ]] <- json_object_default$get_string_vector("unordered_cat_vars")
    train_set_metadata[[
      "unordered_unique_levels"
    ]] <- json_object_default$get_string_list(
      "unordered_unique_levels",
      train_set_metadata[["unordered_cat_vars"]]
    )
  }
  output[["train_set_metadata"]] <- train_set_metadata

  # Unpack model params
  model_params = list()
  model_params[["outcome_scale"]] <- json_object_default$get_scalar(
    "outcome_scale"
  )
  model_params[["outcome_mean"]] <- json_object_default$get_scalar(
    "outcome_mean"
  )
  model_params[["standardize"]] <- json_object_default$get_boolean(
    "standardize"
  )
  model_params[["sigma2_init"]] <- json_object_default$get_scalar(
    "sigma2_init"
  )
  model_params[["sample_sigma2_global"]] <- json_object_default$get_boolean(
    "sample_sigma2_global"
  )
  model_params[["sample_sigma2_leaf"]] <- json_object_default$get_boolean(
    "sample_sigma2_leaf"
  )
  model_params[["include_mean_forest"]] <- include_mean_forest
  model_params[["include_variance_forest"]] <- include_variance_forest
  model_params[["has_rfx"]] <- json_object_default$get_boolean("has_rfx")
  model_params[["has_rfx_basis"]] <- json_object_default$get_boolean(
    "has_rfx_basis"
  )
  model_params[["num_rfx_basis"]] <- json_object_default$get_scalar(
    "num_rfx_basis"
  )
  model_params[["num_covariates"]] <- json_object_default$get_scalar(
    "num_covariates"
  )
  model_params[["num_basis"]] <- json_object_default$get_scalar("num_basis")
  model_params[["requires_basis"]] <- json_object_default$get_boolean(
    "requires_basis"
  )
  model_params[["probit_outcome_model"]] <- json_object_default$get_boolean(
    "probit_outcome_model"
  )
  model_params[["rfx_model_spec"]] <- json_object_default$get_string(
    "rfx_model_spec"
  )
  model_params[["num_chains"]] <- json_object_default$get_scalar("num_chains")
  model_params[["keep_every"]] <- json_object_default$get_scalar("keep_every")

  # Combine values that are sample-specific
  for (i in 1:length(json_object_list)) {
    json_object <- json_object_list[[i]]
    if (i == 1) {
      model_params[["num_gfr"]] <- json_object$get_scalar("num_gfr")
      model_params[["num_burnin"]] <- json_object$get_scalar("num_burnin")
      model_params[["num_mcmc"]] <- json_object$get_scalar("num_mcmc")
      model_params[["num_samples"]] <- json_object$get_scalar(
        "num_samples"
      )
    } else {
      prev_json <- json_object_list[[i - 1]]
      model_params[["num_gfr"]] <- model_params[["num_gfr"]] +
        json_object$get_scalar("num_gfr")
      model_params[["num_burnin"]] <- model_params[["num_burnin"]] +
        json_object$get_scalar("num_burnin")
      model_params[["num_mcmc"]] <- model_params[["num_mcmc"]] +
        json_object$get_scalar("num_mcmc")
      model_params[["num_samples"]] <- model_params[["num_samples"]] +
        json_object$get_scalar("num_samples")
    }
  }
  output[["model_params"]] <- model_params

  # Unpack sampled parameters
  if (model_params[["sample_sigma2_global"]]) {
    for (i in 1:length(json_object_list)) {
      json_object <- json_object_list[[i]]
      if (i == 1) {
        output[["sigma2_global_samples"]] <- json_object$get_vector(
          "sigma2_global_samples",
          "parameters"
        )
      } else {
        output[["sigma2_global_samples"]] <- c(
          output[["sigma2_global_samples"]],
          json_object$get_vector(
            "sigma2_global_samples",
            "parameters"
          )
        )
      }
    }
  }
  if (model_params[["sample_sigma2_leaf"]]) {
    for (i in 1:length(json_object_list)) {
      json_object <- json_object_list[[i]]
      if (i == 1) {
        output[["sigma2_leaf_samples"]] <- json_object$get_vector(
          "sigma2_leaf_samples",
          "parameters"
        )
      } else {
        output[["sigma2_leaf_samples"]] <- c(
          output[["sigma2_leaf_samples"]],
          json_object$get_vector("sigma2_leaf_samples", "parameters")
        )
      }
    }
  }

  # Unpack random effects
  if (model_params[["has_rfx"]]) {
    output[[
      "rfx_unique_group_ids"
    ]] <- json_object_default$get_string_vector("rfx_unique_group_ids")
    output[["rfx_samples"]] <- loadRandomEffectSamplesCombinedJson(
      json_object_list,
      0
    )
  }

  # Unpack covariate preprocessor
  preprocessor_metadata_string <- json_object$get_string(
    "preprocessor_metadata"
  )
  output[["train_set_metadata"]] <- createPreprocessorFromJsonString(
    preprocessor_metadata_string
  )

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
#' bart_model <- bart(X_train = X_train, y_train = y_train,
#'                    num_gfr = 10, num_burnin = 0, num_mcmc = 10)
#' bart_json_string_list <- list(saveBARTModelToJsonString(bart_model))
#' bart_model_roundtrip <- createBARTModelFromCombinedJsonString(bart_json_string_list)
createBARTModelFromCombinedJsonString <- function(json_string_list) {
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
  include_mean_forest <- json_object_default$get_boolean(
    "include_mean_forest"
  )
  include_variance_forest <- json_object_default$get_boolean(
    "include_variance_forest"
  )
  if (include_mean_forest) {
    output[["mean_forests"]] <- loadForestContainerCombinedJson(
      json_object_list,
      "forest_0"
    )
    if (include_variance_forest) {
      output[["variance_forests"]] <- loadForestContainerCombinedJson(
        json_object_list,
        "forest_1"
      )
    }
  } else {
    output[["variance_forests"]] <- loadForestContainerCombinedJson(
      json_object_list,
      "forest_0"
    )
  }

  # Unpack metadata
  train_set_metadata = list()
  train_set_metadata[["num_numeric_vars"]] <- json_object_default$get_scalar(
    "num_numeric_vars"
  )
  train_set_metadata[[
    "num_ordered_cat_vars"
  ]] <- json_object_default$get_scalar("num_ordered_cat_vars")
  train_set_metadata[[
    "num_unordered_cat_vars"
  ]] <- json_object_default$get_scalar("num_unordered_cat_vars")
  if (train_set_metadata[["num_numeric_vars"]] > 0) {
    train_set_metadata[[
      "numeric_vars"
    ]] <- json_object_default$get_string_vector("numeric_vars")
  }
  if (train_set_metadata[["num_ordered_cat_vars"]] > 0) {
    train_set_metadata[[
      "ordered_cat_vars"
    ]] <- json_object_default$get_string_vector("ordered_cat_vars")
    train_set_metadata[[
      "ordered_unique_levels"
    ]] <- json_object_default$get_string_list(
      "ordered_unique_levels",
      train_set_metadata[["ordered_cat_vars"]]
    )
  }
  if (train_set_metadata[["num_unordered_cat_vars"]] > 0) {
    train_set_metadata[[
      "unordered_cat_vars"
    ]] <- json_object_default$get_string_vector("unordered_cat_vars")
    train_set_metadata[[
      "unordered_unique_levels"
    ]] <- json_object_default$get_string_list(
      "unordered_unique_levels",
      train_set_metadata[["unordered_cat_vars"]]
    )
  }
  output[["train_set_metadata"]] <- train_set_metadata

  # Unpack model params
  model_params = list()
  model_params[["outcome_scale"]] <- json_object_default$get_scalar(
    "outcome_scale"
  )
  model_params[["outcome_mean"]] <- json_object_default$get_scalar(
    "outcome_mean"
  )
  model_params[["standardize"]] <- json_object_default$get_boolean(
    "standardize"
  )
  model_params[["sigma2_init"]] <- json_object_default$get_scalar(
    "sigma2_init"
  )
  model_params[["sample_sigma2_global"]] <- json_object_default$get_boolean(
    "sample_sigma2_global"
  )
  model_params[["sample_sigma2_leaf"]] <- json_object_default$get_boolean(
    "sample_sigma2_leaf"
  )
  model_params[["include_mean_forest"]] <- include_mean_forest
  model_params[["include_variance_forest"]] <- include_variance_forest
  model_params[["has_rfx"]] <- json_object_default$get_boolean("has_rfx")
  model_params[["has_rfx_basis"]] <- json_object_default$get_boolean(
    "has_rfx_basis"
  )
  model_params[["num_rfx_basis"]] <- json_object_default$get_scalar(
    "num_rfx_basis"
  )
  model_params[["num_covariates"]] <- json_object_default$get_scalar(
    "num_covariates"
  )
  model_params[["num_basis"]] <- json_object_default$get_scalar("num_basis")
  model_params[["num_chains"]] <- json_object_default$get_scalar("num_chains")
  model_params[["keep_every"]] <- json_object_default$get_scalar("keep_every")
  model_params[["requires_basis"]] <- json_object_default$get_boolean(
    "requires_basis"
  )
  model_params[["probit_outcome_model"]] <- json_object_default$get_boolean(
    "probit_outcome_model"
  )
  model_params[["rfx_model_spec"]] <- json_object_default$get_string(
    "rfx_model_spec"
  )

  # Combine values that are sample-specific
  for (i in 1:length(json_object_list)) {
    json_object <- json_object_list[[i]]
    if (i == 1) {
      model_params[["num_gfr"]] <- json_object$get_scalar("num_gfr")
      model_params[["num_burnin"]] <- json_object$get_scalar("num_burnin")
      model_params[["num_mcmc"]] <- json_object$get_scalar("num_mcmc")
      model_params[["num_samples"]] <- json_object$get_scalar(
        "num_samples"
      )
    } else {
      prev_json <- json_object_list[[i - 1]]
      model_params[["num_gfr"]] <- model_params[["num_gfr"]] +
        json_object$get_scalar("num_gfr")
      model_params[["num_burnin"]] <- model_params[["num_burnin"]] +
        json_object$get_scalar("num_burnin")
      model_params[["num_mcmc"]] <- model_params[["num_mcmc"]] +
        json_object$get_scalar("num_mcmc")
      model_params[["num_samples"]] <- model_params[["num_samples"]] +
        json_object$get_scalar("num_samples")
    }
  }
  output[["model_params"]] <- model_params

  # Unpack sampled parameters
  if (model_params[["sample_sigma2_global"]]) {
    for (i in 1:length(json_object_list)) {
      json_object <- json_object_list[[i]]
      if (i == 1) {
        output[["sigma2_global_samples"]] <- json_object$get_vector(
          "sigma2_global_samples",
          "parameters"
        )
      } else {
        output[["sigma2_global_samples"]] <- c(
          output[["sigma2_global_samples"]],
          json_object$get_vector(
            "sigma2_global_samples",
            "parameters"
          )
        )
      }
    }
  }
  if (model_params[["sample_sigma2_leaf"]]) {
    for (i in 1:length(json_object_list)) {
      json_object <- json_object_list[[i]]
      if (i == 1) {
        output[["sigma2_leaf_samples"]] <- json_object$get_vector(
          "sigma2_leaf_samples",
          "parameters"
        )
      } else {
        output[["sigma2_leaf_samples"]] <- c(
          output[["sigma2_leaf_samples"]],
          json_object$get_vector("sigma2_leaf_samples", "parameters")
        )
      }
    }
  }

  # Unpack random effects
  if (model_params[["has_rfx"]]) {
    output[[
      "rfx_unique_group_ids"
    ]] <- json_object_default$get_string_vector("rfx_unique_group_ids")
    output[["rfx_samples"]] <- loadRandomEffectSamplesCombinedJson(
      json_object_list,
      0
    )
  }

  # Unpack covariate preprocessor
  preprocessor_metadata_string <- json_object_default$get_string(
    "preprocessor_metadata"
  )
  output[["train_set_metadata"]] <- createPreprocessorFromJsonString(
    preprocessor_metadata_string
  )

  class(output) <- "bartmodel"
  return(output)
}
