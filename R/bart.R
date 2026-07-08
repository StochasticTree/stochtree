#' BART Serialization Routines
#' @name BARTSerialization
#' @description
#' BART models contains external pointers to C++ objects, which means they cannot
#' be correctly serialized to `.Rds` from an R session in their default state.
#' This group of serialization functions allow us to convert between C++ data structures and a persistent JSON
#' representation. The `CppJson` class wraps a performant C++ JSON API, and the functions
#' `saveBARTModelToJson` and `createBARTModelFromJson` save to and load from this format.
#' This representation, of course, also relies on external C++ pointers, so in order to
#' save and reload BART models across sessions, we provide two other interfaces.
#'
#' `saveBARTModelToJsonString` converts a BART model to an in-memory string containing the model's
#' JSON representation and `createBARTModelFromJsonString` converts this representation back to a BART model object.
#'
#' `saveBARTModelToJsonFile` and `createBARTModelFromJsonFile` save or reload a BART model
#' directly to / from a `.json` file.
#'
#' Finally, for cases in which multiple BART models have been sampled (for instance, multiple processes
#' run via `doParallel`), we offer `createBARTModelFromCombinedJson` and `createBARTModelFromCombinedJsonString` for
#' loading a new combined BART model from a list of BART JSON objects / strings.
#' @returns
#' `saveBARTModelToJson` return an object of type `CppJson`.
#' `saveBARTModelToJsonString` returns a string dump of the BART model's JSON representation.
#' `saveBARTModelToJsonFile` returns nothing, but writes to the provided filename.
#'
#' `createBARTModelFromJson`, `createBARTModelFromJsonFile`, `createBARTModelFromJsonString`,
#' `createBARTModelFromCombinedJson`, and `createBARTModelFromCombinedJsonString` all return
#' objects of type `bartmodel`.
#' @examples
#' # Generate data
#' n <- 100
#' p <- 5
#' X <- matrix(runif(n*p), ncol = p)
#' y <- X[,1] + rnorm(n, 0, 1)
#'
#' # Sample BART model
#' bart_model <- bart(X_train = X, y_train = y, num_gfr = 0,
#'                    num_burnin = 0, num_mcmc = 10)
#'
#' # Save to in-memory JSON
#' bart_json <- saveBARTModelToJson(bart_model)
#' # Save to JSON string
#' bart_json_string <- saveBARTModelToJsonString(bart_model)
#' # Save to JSON file
#' tmpjson <- tempfile(fileext = ".json")
#' saveBARTModelToJsonFile(bart_model, file.path(tmpjson))
#'
#' # Reload BART model from in-memory JSON object
#' bart_model_roundtrip <- createBARTModelFromJson(bart_json)
#' # Reload BART model from JSON string
#' bart_model_roundtrip <- createBARTModelFromJsonString(bart_json_string)
#' # Reload BART model from JSON file
#' bart_model_roundtrip <- createBARTModelFromJsonFile(file.path(tmpjson))
#' unlink(tmpjson)
#' # Reload BART model from list of JSON objects
#' bart_model_roundtrip <- createBARTModelFromCombinedJson(list(bart_json))
#' # Reload BART model from list of JSON strings
#' bart_model_roundtrip <- createBARTModelFromCombinedJsonString(list(bart_json_string))
#'
NULL
#> NULL

#' @title Run BART for Supervised Learning
#' @description
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
#' @param observation_weights_train (Optional) Numeric vector of observation weights of length `nrow(X_train)`. Weights are
#'   applied as `y_i | - ~ N(mu(X_i), sigma^2 / w_i)`, so larger weights increase an observation's influence on the fit.
#'   All weights must be non-negative. Default: `NULL` (all observations equally weighted). Compatible with Gaussian
#'   (continuous/identity) and probit outcome models; not compatible with cloglog link functions. Note: these are
#'   referred to internally in the C++ layer as "variance weights" (`var_weights`), since they scale the residual variance.
#' @param observation_weights Deprecated alias for `observation_weights_train`; will be removed in a future release.
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
#'   - `num_chains` How many independent MCMC chains should be sampled. If `num_mcmc = 0`, this is ignored. If `num_gfr = 0`, then each chain is run from root for `num_mcmc * keep_every + num_burnin` iterations, with `num_mcmc` samples retained. If `num_gfr > 0`, each MCMC chain will be initialized from a separate GFR ensemble, with the requirement that `num_gfr >= num_chains`. Default: `1`. Note that if `num_chains > 1`, the returned model object will contain samples from all chains, stored consecutively. That is, if there are 4 chains with 100 samples each, the first 100 samples will be from chain 1, the next 100 samples will be from chain 2, etc... For more detail on working with multi-chain BART models, see [the multi chain vignette](https://stochtree.ai/vignettes/multi-chain.html).
#'   - `verbose` Whether or not to print progress during the sampling loops. Default: `FALSE`.
#'   - `outcome_model` A structured `OutcomeModel` object that specifies the outcome type and desired link function. This argument pre-empts the legacy (deprecated) `probit_outcome_model` option. Default: `OutcomeModel(outcome='continuous', link='identity')`.
#'   - `probit_outcome_model` Deprecated in favor of `outcome_model`. Whether or not the outcome should be modeled as explicitly binary via a probit link. If `TRUE`, `y` must only contain the values `0` and `1`. Default: `FALSE`.
#'   - `num_threads` Number of threads to use in the GFR and MCMC algorithms, as well as prediction. Defaults to `1` (single-threaded). Set to `-1` to use the maximum number of available threads, or a positive integer for a specific count. OpenMP must be available for values other than `1`.
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
  observation_weights_train = NULL,
  observation_weights = NULL,
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
    outcome_model = OutcomeModel(outcome = "continuous", link = "identity"),
    probit_outcome_model = FALSE,
    num_threads = 1
  )
  general_params_updated <- preprocessParams(
    general_params_default,
    general_params
  )
  # TODO: think about validation and deprecation flow for probit_outcome_model

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
    num_features_subsample = NULL,
    cloglog_leaf_prior_shape = 2.0,
    cloglog_leaf_prior_scale = 2.0
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
  outcome_model <- general_params_updated$outcome_model
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
  cloglog_leaf_prior_shape <- mean_forest_params_updated$cloglog_leaf_prior_shape
  cloglog_leaf_prior_scale <- mean_forest_params_updated$cloglog_leaf_prior_scale

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

  # Raise a deprecation warning to use `outcome_model` if `probit_outcome_model = TRUE` is specified
  if (probit_outcome_model) {
    warning(
      "Specifying a probit link through `general_params = list(probit_outcome_model = TRUE)` is deprecated and will be removed in a future version. Please use `general_params = list(outcome_model = OutcomeModel(outcome = 'binary', link = 'probit'))` instead."
    )
  }

  # Unpack outcome model details
  link_is_linear <- FALSE
  link_is_probit <- FALSE
  link_is_cloglog <- FALSE
  outcome_is_continuous <- FALSE
  outcome_is_binary <- FALSE
  outcome_is_ordinal <- FALSE
  if (
    outcome_model$outcome == "continuous" && outcome_model$link == "identity"
  ) {
    link_is_linear <- TRUE
    outcome_is_continuous <- TRUE
  } else if (
    outcome_model$outcome == "binary" && outcome_model$link == "probit"
  ) {
    link_is_probit <- TRUE
    outcome_is_binary <- TRUE
  } else if (
    outcome_model$outcome == "binary" && outcome_model$link == "cloglog"
  ) {
    link_is_cloglog <- TRUE
    outcome_is_binary <- TRUE
  } else if (
    outcome_model$outcome == "ordinal" && outcome_model$link == "cloglog"
  ) {
    link_is_cloglog <- TRUE
    outcome_is_ordinal <- TRUE
  } else {
    stop(paste0(
      "Invalid outcome model specification, outcome = ",
      outcome_model$outcome,
      ", link = ",
      outcome_model$link
    ))
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
      previous_forest_samples_mean <- previous_bart_model$samples$materialize_mean_forest()
    } else {
      previous_forest_samples_mean <- NULL
    }
    if (previous_bart_model$model_params$include_variance_forest) {
      previous_forest_samples_variance <- previous_bart_model$samples$materialize_variance_forest()
    } else {
      previous_forest_samples_variance <- NULL
    }
    if (previous_bart_model$model_params$sample_sigma2_global) {
      previous_global_var_samples <- previous_bart_model$samples$global_var_samples() /
        (previous_y_scale * previous_y_scale)
    } else {
      previous_global_var_samples <- NULL
    }
    if (previous_bart_model$model_params$sample_sigma2_leaf) {
      previous_leaf_var_samples <- previous_bart_model$samples$leaf_scale_samples()
    } else {
      previous_leaf_var_samples <- NULL
    }
    if (previous_bart_model$model_params$has_rfx) {
      previous_rfx_samples <- previous_bart_model$rfx_samples
    } else {
      previous_rfx_samples <- NULL
    }
    if (previous_bart_model$model_params$outcome_model$link == "cloglog") {
      previous_cloglog_cutpoint_samples <- previous_bart_model$samples$cloglog_cutpoint_samples()
      previous_cloglog_num_categories <- previous_bart_model$cloglog_num_categories
    } else {
      previous_cloglog_cutpoint_samples <- NULL
      previous_cloglog_num_categories <- 0
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
    previous_cloglog_cutpoint_samples <- NULL
    previous_cloglog_num_categories <- 0
    previous_model_num_samples <- 0
  }

  # Determine whether conditional mean, variance, or both will be modeled
  if (num_trees_variance > 0) {
    include_variance_forest <- TRUE
  } else {
    include_variance_forest <- FALSE
  }
  if (num_trees_mean > 0) {
    include_mean_forest <- TRUE
  } else {
    include_mean_forest <- FALSE
  }

  # `observation_weights` was renamed to `observation_weights_train`; honor the
  # deprecated argument for one release cycle.
  if (!is.null(observation_weights)) {
    warning(
      "`observation_weights` is deprecated and will be removed in a future release; use `observation_weights_train` instead."
    )
    if (is.null(observation_weights_train)) {
      observation_weights_train <- observation_weights
    }
  }

  # observation_weights_train compatibility checks
  if (!is.null(observation_weights_train)) {
    if (link_is_cloglog) {
      stop(
        "observation_weights_train are not compatible with cloglog link functions."
      )
    }
    if (include_variance_forest) {
      stop(
        "observation_weights_train are not compatible with a variance forest model. ",
        "Use either observation_weights_train or a variance forest, not both."
      )
    }
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
    variable_weights <- rep(1 / ncol(X_train), ncol(X_train))
  }
  if (any(variable_weights < 0)) {
    stop("variable_weights cannot have any negative weights")
  }

  # Observation weight validation
  if (!is.null(observation_weights_train)) {
    if (!is.numeric(observation_weights_train)) {
      stop("observation_weights_train must be a numeric vector")
    }
    if (length(observation_weights_train) != nrow(X_train)) {
      stop("length(observation_weights_train) must equal nrow(X_train)")
    }
    if (any(observation_weights_train < 0)) {
      stop("observation_weights_train cannot have any negative values")
    }
    if (all(observation_weights_train == 0) && num_gfr > 0) {
      stop(
        "observation_weights_train are all zero (prior sampling mode) but num_gfr > 0. ",
        "GFR warm-start is data-dependent and ill-defined with zero weights. ",
        "Set num_gfr = 0 when using all-zero observation_weights_train."
      )
    }
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
  variable_subset_mean <- resolveVariableSubset(
    keep_vars_mean,
    drop_vars_mean,
    X_train,
    "mean"
  )
  variable_subset_variance <- resolveVariableSubset(
    keep_vars_variance,
    drop_vars_variance,
    X_train,
    "variance"
  )

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

  # Update variable weights (expand per-variable weights to the preprocessed feature space)
  variable_weights_mean <- variable_weights_variance <- variable_weights
  if (include_mean_forest) {
    variable_weights_mean <- expandVariableWeights(
      variable_weights,
      original_var_indices,
      variable_subset_mean
    )
  }
  if (include_variance_forest) {
    variable_weights_variance <- expandVariableWeights(
      variable_weights,
      original_var_indices,
      variable_subset_variance
    )
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
  has_basis <- !is.null(leaf_basis_train)

  # Determine whether a test set is provided
  has_test <- !is.null(X_test)

  # Preliminary runtime checks for probit link
  if (!include_mean_forest) {
    link_is_probit <- FALSE
    # TODO: think about allowing binary models with probit link for homoskedastic RFX-only models?
  }
  if (link_is_probit) {
    if (!(length(unique(y_train)) == 2)) {
      stop(
        "You specified a probit link, but supplied an outcome with more than 2 unique values. Probit is only currently supported for binary outcomes."
      )
    }
    unique_outcomes <- sort(unique(y_train))
    if (!(all(unique_outcomes == c(0, 1)))) {
      stop(
        "You specified a probit link, but supplied an outcome with 2 unique values other than 0 and 1"
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

  # Preliminary runtime checks for cloglog link
  if (!include_mean_forest) {
    link_is_cloglog <- FALSE
    # TODO: think about allowing binary models with cloglog link for homoskedastic RFX-only models?
  }
  if (link_is_cloglog) {
    if (!all(as.integer(y_train) == y_train)) {
      stop(
        "You specified a cloglog link, but supplied an outcome with non-integer values. Cloglog is only currently supported for integer outcomes."
      )
    }
    unique_outcomes <- sort(unique(y_train))
    if (!(min(unique_outcomes) %in% c(0, 1))) {
      stop(
        "You specified a cloglog link, but supplied an integer outcome that does not start with 0 or 1. Please remap / shift the outcomes so that the smallest category label is either 0 or 1."
      )
    }
    if (!all(diff(unique_outcomes) == 1)) {
      stop(
        "You specified a cloglog link, but supplied an integer outcome that is not a sequence of consecutive integers"
      )
    }
    if (include_variance_forest) {
      stop("We do not support heteroskedasticity with a cloglog link")
    }
    if (has_basis) {
      stop("We do not support leaf basis regression with a cloglog link")
    }
    if (sample_sigma2_global) {
      warning(
        "Global error variance will not be sampled with a cloglog link"
      )
      sample_sigma2_global <- F
    }
    if (sample_sigma2_leaf) {
      warning(
        "Leaf scale parameter will not be sampled with a cloglog link"
      )
      sample_sigma2_leaf <- F
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

  # Determine leaf model type
  if ((!has_basis) && (!link_is_cloglog)) {
    leaf_model_mean_forest <- 0
  } else if ((!has_basis) && (link_is_cloglog)) {
    leaf_model_mean_forest <- 4
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
    leaf_dimension <- 1
    is_leaf_constant <- TRUE
    leaf_regression <- FALSE
  } else if (leaf_model_mean_forest == 1) {
    stopifnot(has_basis)
    stopifnot(ncol(leaf_basis_train) == 1)
    leaf_dimension <- 1
    is_leaf_constant <- FALSE
    leaf_regression <- TRUE
  } else if (leaf_model_mean_forest == 2) {
    stopifnot(has_basis)
    stopifnot(ncol(leaf_basis_train) > 1)
    leaf_dimension <- ncol(leaf_basis_train)
    is_leaf_constant <- FALSE
    leaf_regression <- TRUE
    if (sample_sigma2_leaf) {
      warning(
        "Sampling leaf scale not yet supported for multivariate leaf models, so the leaf scale parameter will not be sampled in this model."
      )
      sample_sigma2_leaf <- FALSE
    }
  } else if (leaf_model_mean_forest == 4) {
    leaf_dimension <- 1
    is_leaf_constant <- TRUE
    leaf_regression <- FALSE
  }

  cloglog_num_categories <- ifelse(
    link_is_cloglog,
    max(y_train - min(y_train)) + 1,
    0
  )
  model_params_r <- list(
    "a_global" = a_global,
    "b_global" = b_global,
    "a_leaf" = a_leaf,
    "standardize" = standardize,
    "leaf_dimension" = leaf_dimension,
    "is_leaf_constant" = is_leaf_constant,
    "leaf_regression" = leaf_regression,
    "requires_basis" = leaf_regression,
    "num_covariates" = num_cov_orig,
    "num_basis" = ifelse(
      is.null(leaf_basis_train),
      0,
      ncol(leaf_basis_train)
    ),
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
    "outcome_model" = outcome_model,
    "probit_outcome_model" = probit_outcome_model,
    "cloglog_num_categories" = cloglog_num_categories,
    "rfx_model_spec" = rfx_model_spec
  )

  # Expand dimensions on RFX prior parameters if provided
  # Working parameter (should be expanded to a vector if provided as a scalar)
  if (!is.null(rfx_working_parameter_prior_mean)) {
    rfx_working_parameter_prior_mean <- expand_dims_1d(
      rfx_working_parameter_prior_mean,
      num_rfx_components
    )
  }

  # Group parameter (should be expanded to a matrix if provided as a scalar)
  if (!is.null(rfx_group_parameter_prior_mean)) {
    rfx_group_parameter_prior_mean <- expand_dims_2d(
      rfx_group_parameter_prior_mean,
      num_rfx_components,
      num_rfx_groups
    )
  }

  # Working parameter (should be expanded to a diagonal matrix if provided as a scalar)
  if (!is.null(rfx_working_parameter_prior_cov)) {
    rfx_working_parameter_prior_cov <- expand_dims_2d_diag(
      rfx_working_parameter_prior_cov,
      num_rfx_components
    )
  }

  # Group parameter (should be expanded to a diagonal matrix if provided as a scalar)
  if (!is.null(rfx_group_parameter_prior_cov)) {
    rfx_group_parameter_prior_cov <- expand_dims_2d_diag(
      rfx_group_parameter_prior_cov,
      num_rfx_components
    )
  }

  # Specify the BART config
  bart_config <- list(
    "standardize_outcome" = standardize,
    "num_threads" = num_threads,
    "verbose" = verbose,
    "cutpoint_grid_size" = cutpoint_grid_size,
    "link_function" = ifelse(
      outcome_model$link == "identity",
      0,
      ifelse(outcome_model$link == "probit", 1, 2)
    ),
    "outcome_type" = ifelse(
      outcome_model$outcome == "continuous",
      0,
      ifelse(outcome_model$outcome == "binary", 1, 2)
    ),
    "random_seed" = random_seed,
    "keep_gfr" = keep_gfr,
    "keep_burnin" = keep_burnin,
    "a_sigma2_global" = a_global,
    "b_sigma2_global" = b_global,
    "sigma2_global_init" = sigma2_init,
    "sample_sigma2_global" = sample_sigma2_global,
    "num_trees_mean" = num_trees_mean,
    "alpha_mean" = alpha_mean,
    "beta_mean" = beta_mean,
    "min_samples_leaf_mean" = min_samples_leaf_mean,
    "max_depth_mean" = max_depth_mean,
    "leaf_constant_mean" = is_leaf_constant,
    "leaf_dim_mean" = leaf_dimension,
    "exponentiated_leaf_mean" = FALSE,
    "num_features_subsample_mean" = num_features_subsample_mean,
    "a_sigma2_mean" = a_leaf,
    "b_sigma2_mean" = b_leaf,
    "sigma2_mean_init" = if (is.matrix(sigma2_leaf_init)) {
      NULL
    } else {
      sigma2_leaf_init
    },
    "sample_sigma2_leaf_mean" = sample_sigma2_leaf,
    "mean_leaf_model_type" = leaf_model_mean_forest,
    "sigma2_leaf_mean_matrix" = if (is.matrix(sigma2_leaf_init)) {
      as.numeric(sigma2_leaf_init)
    } else {
      NULL
    },
    "num_classes_cloglog" = cloglog_num_categories,
    "cloglog_leaf_prior_shape" = cloglog_leaf_prior_shape,
    "cloglog_leaf_prior_scale" = cloglog_leaf_prior_scale,
    "cloglog_cutpoint_0" = 0,
    "num_trees_variance" = num_trees_variance,
    "leaf_prior_calibration_param" = a_0,
    "shape_variance_forest" = a_forest,
    "scale_variance_forest" = b_forest,
    "variance_forest_leaf_init" = variance_forest_init,
    "alpha_variance" = alpha_variance,
    "beta_variance" = beta_variance,
    "min_samples_leaf_variance" = min_samples_leaf_variance,
    "max_depth_variance" = max_depth_variance,
    "leaf_constant_variance" = TRUE,
    "leaf_dim_variance" = 1,
    "exponentiated_leaf_variance" = TRUE,
    "num_features_subsample_variance" = num_features_subsample_variance,
    "feature_types" = as.integer(feature_types),
    "sweep_update_indices_mean" = if (num_trees_mean > 0) {
      0:(num_trees_mean - 1)
    } else {
      NULL
    },
    "sweep_update_indices_variance" = if (num_trees_variance > 0) {
      0:(num_trees_variance - 1)
    } else {
      NULL
    },
    "var_weights_mean" = variable_weights_mean,
    "var_weights_variance" = variable_weights_variance,
    "has_random_effects" = has_rfx,
    "rfx_model_spec" = if (has_rfx) {
      ifelse(
        rfx_model_spec == "custom",
        0,
        ifelse(rfx_model_spec == "intercept_only", 1, NULL)
      )
    } else {
      NULL
    },
    "rfx_working_parameter_mean_prior" = if (has_rfx) {
      rfx_working_parameter_prior_mean
    } else {
      NULL
    },
    "rfx_working_parameter_cov_prior" = if (has_rfx) {
      rfx_working_parameter_prior_cov
    } else {
      NULL
    },
    "rfx_group_parameter_mean_prior" = if (has_rfx) {
      rfx_group_parameter_prior_mean
    } else {
      NULL
    },
    "rfx_group_parameter_cov_prior" = if (has_rfx) {
      rfx_group_parameter_prior_cov
    } else {
      NULL
    },
    "rfx_variance_prior_shape" = if (has_rfx) {
      rfx_variance_prior_shape
    } else {
      NULL
    },
    "rfx_variance_prior_scale" = if (has_rfx) {
      rfx_variance_prior_scale
    } else {
      NULL
    }
  )

  bart_samples <- BARTSamples$new()
  bart_metadata <- bart_sample_cpp(
    samples = bart_samples$samples_ptr,
    X_train = X_train,
    y_train = if (link_is_cloglog) {
      as.numeric(y_train - min(y_train))
    } else {
      y_train
    },
    X_test = if (exists("X_test")) X_test else NULL,
    n_train = nrow(X_train),
    n_test = if (!is.null(X_test)) nrow(X_test) else 0L,
    p = ncol(X_train),
    basis_train = if (exists("leaf_basis_train")) leaf_basis_train else NULL,
    basis_test = if (exists("leaf_basis_test")) leaf_basis_test else NULL,
    basis_dim = if (!is.null(leaf_basis_train)) {
      ncol(leaf_basis_train)
    } else {
      0L
    },
    obs_weights_train = observation_weights_train,
    obs_weights_test = NULL,
    rfx_group_ids_train = if (exists("rfx_group_ids_train")) {
      rfx_group_ids_train
    } else {
      NULL
    },
    rfx_group_ids_test = if (exists("rfx_group_ids_test")) {
      rfx_group_ids_test
    } else {
      NULL
    },
    rfx_basis_train = if (exists("rfx_basis_train")) {
      rfx_basis_train
    } else {
      NULL
    },
    rfx_basis_test = if (exists("rfx_basis_test")) rfx_basis_test else NULL,
    rfx_num_groups = if (exists("num_rfx_groups")) {
      as.integer(num_rfx_groups)
    } else {
      0L
    },
    rfx_basis_dim = as.integer(num_basis_rfx),
    num_gfr = as.integer(num_gfr),
    num_burnin = as.integer(num_burnin),
    keep_every = as.integer(keep_every),
    num_mcmc = as.integer(num_mcmc),
    num_chains = as.integer(num_chains),
    config_input = bart_config
  )
  result <- list()
  model_params_cpp <- list(
    "sigma2_init" = bart_metadata[["sigma2_global_init"]],
    "sigma2_leaf_init" = bart_metadata[["sigma2_mean_init"]],
    "b_leaf" = bart_metadata[["b_sigma2_mean"]],
    "a_forest" = bart_metadata[["shape_variance_forest"]],
    "b_forest" = bart_metadata[["scale_variance_forest"]],
    "outcome_mean" = bart_samples$y_bar(),
    "outcome_scale" = bart_samples$y_std(),
    "num_samples" = bart_samples$num_samples(),
    # Final RNG state, so continueSampling() can resume the stream by default
    "rng_state" = bart_metadata[["rng_state"]]
  )
  model_params <- c(model_params_r, model_params_cpp)
  result[["model_params"]] <- model_params
  result[["train_set_metadata"]] <- X_train_metadata
  result[["samples"]] <- bart_samples
  # Cache the resolved sampler config so continueSampling() can reconstruct the sampler. This is a
  # runtime-only field (not serialized), so continuation is unavailable for deserialized models.
  result[["bart_config"]] <- bart_config
  # Cache the raw per-variable split weights and the resolved variable subsets so continueSampling()
  # can re-resolve split-variable weights when keep_vars / drop_vars / variable_weights are changed.
  result[["continuation_state"]] <- list(
    variable_weights = variable_weights,
    variable_subset_mean = variable_subset_mean,
    variable_subset_variance = variable_subset_variance
  )

  # Unpack RFX samples
  if (has_rfx) {
    # Only need to store unique group IDs, everything else stored in BARTSamples
    result[["rfx_unique_group_ids"]] <- levels(group_ids_factor)
  }

  class(result) <- "bartmodel"

  return(result)
}

#' Guard accessor for a `bartmodel`'s removed direct-access forest and parameter / prediction fields.
#'
#' The sampled forests / parameters / predictions are owned by a single `BARTSamples` object stored in `object$samples`.
#' They are no longer stored or accessible via `$mean_forests` / `$variance_forests`.
#' Similarly, sampled parameter vectors are no longer stored or accessible via `$sigma2_global_samples` / `$sigma2_leaf_samples`
#' and cached predictions are not accessible via `$y_hat_train` / `$y_hat_test` / `$sigma2_x_hat_train` / `$sigma2_x_hat_test`.
#' Accessing any of these model terms by name raises an error pointing at the supported extraction path.
#' @noRd
#' @export
`$.bartmodel` <- function(x, name) {
  if (identical(name, "mean_forests") || identical(name, "variance_forests")) {
    stop(
      sprintf(
        paste0(
          "`bartmodel$%s` has been removed. The sampled forests are owned by `model$bart_samples`; ",
          "you can extract a standalone copy with `extractForest()`."
        ),
        name
      ),
      call. = FALSE
    )
  } else if (
    identical(name, "sigma2_global_samples") ||
      identical(name, "sigma2_leaf_samples") ||
      identical(name, "cloglog_cutpoint_samples") ||
      identical(name, "y_hat_train") ||
      identical(name, "y_hat_test") ||
      identical(name, "sigma2_x_hat_train") ||
      identical(name, "sigma2_x_hat_test")
  ) {
    stop(
      sprintf(
        paste0(
          "`bartmodel$%s` has been removed. The parameters are stored in a C++ BARTSamples object; ",
          "you can extract a standalone copy with `extractParameter()`."
        ),
        name
      ),
      call. = FALSE
    )
  } else {
    .subset2(x, name)
  }
}

#' @title Predict from a BART Model
#' @description
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
#' @param scale (Optional) Scale of mean function predictions. Options are "linear", which returns predictions on the original scale of the mean forest / RFX terms, "probability", which transforms predictions into class probabilities for models with discrete outcomes, and "class", which returns predicted outcome categories for discrete outcome models. "probability" is only valid for outcome models with `outcome == 'binary'` or `outcome == 'ordinal'`. For binary outcomes, this will return the probability that `y == 1`, and for ordinal outcomes, this will return probabilities for each outcome label. Default: "linear".
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
  if (!(scale %in% c("linear", "probability", "class"))) {
    stop("scale must either be 'linear', 'probability', or 'class'")
  }
  outcome_model <- object$model_params$outcome_model
  is_probit <- (outcome_model$link == "probit" &&
    outcome_model$outcome == "binary")
  is_binary_cloglog <- (outcome_model$link == "cloglog" &&
    outcome_model$outcome == "binary")
  is_ordinal_cloglog <- (outcome_model$link == "cloglog" &&
    outcome_model$outcome == "ordinal")
  is_cloglog <- is_binary_cloglog || is_ordinal_cloglog
  if ((scale == "probability") && (!(is_probit || is_cloglog))) {
    stop(
      "scale cannot be 'probability' for models not fit with a probit or cloglog outcome model"
    )
  }
  if ((scale == "class") && (!(is_probit || is_cloglog))) {
    stop(
      "scale cannot be 'class' for models not fit with a probit or cloglog outcome model"
    )
  }
  probability_scale <- scale == "probability"
  class_scale <- scale == "class"

  # Handle prediction type
  if (!is.character(type)) {
    stop("type must be a string or character vector")
  }
  if (!(type %in% c("mean", "posterior"))) {
    stop("type must either be 'mean' or 'posterior'")
  }
  predict_mean <- type == "mean"
  if (predict_mean && class_scale) {
    stop("Posterior mean predictions are not supported for scale = 'class'")
  }

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
  if (class_scale) {
    if (!((predict_count == 1) && (predict_y_hat))) {
      stop("Class scale can only be used with y_hat predictions")
    }
  }

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

  # Read the forests through borrowed (non-owning) pointers into the single-owner
  # samples object -- no deep copy, and avoids tripping the deprecated $mean_forests
  # accessor's warning on internal prediction.
  bart_samples <- object$samples
  bart_metadata_list <- list(
    num_samples = as.integer(object$model_params$num_samples),
    y_bar = as.double(object$model_params$outcome_mean),
    y_std = as.double(object$model_params$outcome_scale),
    include_variance_forest = has_variance_forest,
    has_rfx = has_rfx,
    rfx_model_spec = if (has_rfx) {
      object$model_params$rfx_model_spec
    } else {
      ""
    },
    link_function = object$model_params$outcome_model$link,
    outcome_type = object$model_params$outcome_model$outcome,
    cloglog_num_classes = if (
      !is.null(object$model_params$num_classes_cloglog)
    ) {
      as.integer(object$model_params$num_classes_cloglog)
    } else if (!is.null(object$model_params$cloglog_num_categories)) {
      as.integer(object$model_params$cloglog_num_categories)
    } else {
      0L
    }
  )

  # Dimensions and integer-coded scale needed by the C++ predict path
  n <- nrow(X)
  p <- ncol(X)
  num_basis <- if (!is.null(leaf_basis)) ncol(leaf_basis) else 0L
  rfx_num_groups <- if (!is.null(rfx_group_ids)) {
    length(unique(rfx_group_ids))
  } else {
    0L
  }
  rfx_basis_dim <- if (!is.null(rfx_basis)) ncol(rfx_basis) else 0L
  scale_int <- switch(scale, "linear" = 0L, "probability" = 1L, "class" = 2L)

  output <- bart_predict_cpp(
    bart_samples_ptr = bart_samples$samples_ptr,
    bart_model_metadata = bart_metadata_list,
    X = X,
    leaf_basis = leaf_basis,
    n = n,
    p = p,
    num_basis = num_basis,
    obs_weights = NULL,
    rfx_group_ids = rfx_group_ids,
    rfx_basis = rfx_basis,
    rfx_num_groups = rfx_num_groups,
    rfx_basis_dim = rfx_basis_dim,
    posterior = type == "posterior",
    scale = scale_int,
    predict_y_hat = predict_y_hat,
    predict_mean_forest = predict_mean_forest,
    predict_variance_forest = predict_variance_forest,
    predict_random_effects = predict_rfx
  )
  # Reshape flat C++ output vectors to matrices (n x num_samples) and rename
  # fields to match the R predict path.  For type="mean", num_samples_output=1
  # so we drop the trailing singleton to return a plain vector.
  num_samples_raw <- as.integer(object$model_params$num_samples)
  num_samples_output <- if (type == "posterior") num_samples_raw else 1L
  reshape_cpp_pred_2d <- function(v, dim1, dim2) {
    if (is.null(v)) {
      return(NULL)
    }
    if (dim2 == 1L) {
      return(as.vector(v))
    }
    m <- v
    dim(m) <- c(dim1, dim2)
    m
  }
  reshape_cpp_pred_3d <- function(v, dim1, dim2, dim3) {
    if (is.null(v)) {
      return(NULL)
    }
    a <- v
    dim(a) <- c(dim1, dim2, dim3)
    a
  }
  cloglog_num_classes_out <- if (
    !is.null(object$model_params$cloglog_num_categories)
  ) {
    as.integer(object$model_params$cloglog_num_categories)
  } else if (!is.null(object$model_params$num_classes_cloglog)) {
    as.integer(object$model_params$num_classes_cloglog)
  } else {
    0L
  }
  result <- list(
    y_hat = if (is_ordinal_cloglog && probability_scale) {
      reshape_cpp_pred_3d(
        output$y_hat,
        n,
        cloglog_num_classes_out,
        num_samples_output
      )
    } else if (is_ordinal_cloglog && class_scale) {
      # C++ class_transform_multiclass uses 0-indexed labels; match slow path (which.max = 1-indexed)
      reshape_cpp_pred_2d(output$y_hat, n, num_samples_output) + 1L
    } else {
      reshape_cpp_pred_2d(output$y_hat, n, num_samples_output)
    },
    mean_forest_predictions = if (is_ordinal_cloglog && probability_scale) {
      reshape_cpp_pred_3d(
        output$mean_forest_predictions,
        n,
        cloglog_num_classes_out,
        num_samples_output
      )
    } else {
      reshape_cpp_pred_2d(
        output$mean_forest_predictions,
        n,
        num_samples_output
      )
    },
    rfx_predictions = reshape_cpp_pred_2d(
      output$rfx_predictions,
      n,
      num_samples_output
    ),
    variance_forest_predictions = reshape_cpp_pred_2d(
      output$variance_forest_predictions,
      n,
      num_samples_output
    )
  )
  if (predict_count == 1) {
    if (predict_y_hat) {
      return(result[["y_hat"]])
    }
    if (predict_mean_forest) {
      return(result[["mean_forest_predictions"]])
    }
    if (predict_rfx) {
      return(result[["rfx_predictions"]])
    }
    if (predict_variance_forest) {
      return(result[["variance_forest_predictions"]])
    }
  }
  return(result)
}


#' @title Continue Sampling a BART Model
#' @description
#' Continue sampling from an already-fit BART model, adding more draws of the posterior to the `bartmodel` object.
#' Model terms (mean forest, variance forest, random effects, parametric terms) are initialized from their last retained sample.
#' Training data must be passed anew to this function as the `bartmodel` object does not retain it.
#' This function can only run on a model that has been sampled in session via the `bart()` function.
#' Models that have been serialized to JSON can be passed directly to the `bart()` function to achieve a similar goal of drawing
#' more posterior samples from a given model, though in that workflow the JSON samples are not included in the resulting `bartmodel` object.
#'
#' The model specification interface mirrors that of `bart()`, but some parameters are considered fixed and thus cannot be changed in the user-provided parameter lists (i.e., `global_params`,
#' `mean_forest_params`, `variance_forest_params`, `rfx_params`). Any model specifications that define the relevant models terms, number of trees, outcome scale and distribution, or
#' calibration / initialization (e.g. `num_trees`, `sample_sigma2_global`, `sample_sigma2_leaf`, `sigma2_global_init`, `sigma2_leaf_init`, `standardize`, `outcome_model`, `num_chains`)
#' cannot be changed. Specifying any of these components in a parameter list triggers a warning and will be ignored. Prior hyperparameters and sampling knobs that only impact future draws
#' *can* be modified (see the parameter lists below), though this means the continued draws target a slightly different posterior. This may be desirable in some use cases, but it's important
#' to be thoughtful about any such parameter changes.
#'
#' @param object Fitted model object to continue sampling. Must be of class `bartmodel` and must be have been obtained by running the `bart()` function in session (this function is unavailable for models deserialized from JSON).
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
#' @param observation_weights_train (Optional) Numeric vector of observation weights of length `nrow(X_train)`. Weights are
#'   applied as `y_i | - ~ N(mu(X_i), sigma^2 / w_i)`, so larger weights increase an observation's influence on the fit.
#'   All weights must be non-negative. Default: `NULL` (all observations equally weighted). Compatible with Gaussian
#'   (continuous/identity) and probit outcome models; not compatible with cloglog link functions. Note: these are
#'   referred to internally in the C++ layer as "variance weights" (`var_weights`), since they scale the residual variance.
#' @param observation_weights Deprecated alias for `observation_weights_train`; will be removed in a future release.
#' @param num_gfr Number of "warm-start" iterations run using the grow-from-root algorithm (He and Hahn, 2021). Default: 5.
#' @param num_burnin Number of "burn-in" iterations of the MCMC sampler. Default: 0.
#' @param num_mcmc Number of "retained" iterations of the MCMC sampler. Default: 100.
#' @param general_params (Optional) A list of general (non-forest-specific) model parameters. Only the following keys can be modified by `continueSampling`; any other parameters specified
#' prompt a warning and are ignored. Unspecified keys default to their value set in `bart()`.
#'
#'   - `cutpoint_grid_size` Maximum size of the "grid" of potential cutpoints to consider in the GFR algorithm. Default: `100`.
#'   - `sigma2_global_shape` Shape parameter in the `IG(sigma2_global_shape, sigma2_global_scale)` global error variance model. Default: `0`.
#'   - `sigma2_global_scale` Scale parameter in the `IG(sigma2_global_shape, sigma2_global_scale)` global error variance model. Default: `0`.
#'   - `variable_weights` Numeric weights reflecting the relative probability of splitting on each variable. Does not need to sum to 1 but cannot be negative. Defaults to `rep(1/ncol(X_train), ncol(X_train))` if not set here. Note that if the propensity score is included as a covariate in either forest, its weight will default to `1/ncol(X_train)`.
#'   - `random_seed` Integer parameterizing the C++ random number generator. If not specified, the C++ random number generator is seeded according to `std::random_device`.
#'   - `keep_burnin` Whether or not "burnin" samples should be included in the stored samples of forests and other parameters. Default `FALSE`. Ignored if `num_mcmc = 0`.
#'   - `keep_gfr` Whether or not "grow-from-root" samples should be included in the stored samples of forests and other parameters. Default `FALSE`. Ignored if `num_mcmc = 0`.
#'   - `keep_every` How many iterations of the burned-in MCMC sampler should be run before forests and parameters are retained. Default `1`. Setting `keep_every <- k` for some `k > 1` will "thin" the MCMC samples by retaining every `k`-th sample, rather than simply every sample. This can reduce the autocorrelation of the MCMC samples.
#'   - `num_chains` How many independent MCMC chains should be sampled. If `num_mcmc = 0`, this is ignored. If `num_gfr = 0`, then each chain is run from root for `num_mcmc * keep_every + num_burnin` iterations, with `num_mcmc` samples retained. If `num_gfr > 0`, each MCMC chain will be initialized from a separate GFR ensemble, with the requirement that `num_gfr >= num_chains`. Default: `1`. Note that if `num_chains > 1`, the returned model object will contain samples from all chains, stored consecutively. That is, if there are 4 chains with 100 samples each, the first 100 samples will be from chain 1, the next 100 samples will be from chain 2, etc... For more detail on working with multi-chain BART models, see [the multi chain vignette](https://stochtree.ai/vignettes/multi-chain.html).
#'   - `verbose` Whether or not to print progress during the sampling loops. Default: `FALSE`.
#'   - `num_threads` Number of threads to use in the GFR and MCMC algorithms, as well as prediction. Defaults to `1` (single-threaded). Set to `-1` to use the maximum number of available threads, or a positive integer for a specific count. OpenMP must be available for values other than `1`.
#'
#' @param mean_forest_params (Optional) A list of mean forest model parameters. Only the following keys can be modified by `continueSampling`; any other parameters specified
#' prompt a warning and are ignored. Unspecified keys default to their value set in `bart()`.
#'
#'   - `alpha` Prior probability of splitting for a tree of depth 0 in the mean model. Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`. Default: `0.95`.
#'   - `beta` Exponent that decreases split probabilities for nodes of depth > 0 in the mean model. Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`. Default: `2`.
#'   - `min_samples_leaf` Minimum allowable size of a leaf, in terms of training samples, in the mean model. Default: `5`.
#'   - `max_depth` Maximum depth of any tree in the ensemble in the mean model. Default: `10`. Can be overridden with ``-1`` which does not enforce any depth limits on trees.
#'   - `sigma2_leaf_shape` Shape parameter in the `IG(sigma2_leaf_shape, sigma2_leaf_scale)` leaf node parameter variance model. Default: `3`.
#'   - `sigma2_leaf_scale` Scale parameter in the `IG(sigma2_leaf_shape, sigma2_leaf_scale)` leaf node parameter variance model. Calibrated internally as `0.5/num_trees` if not set here.
#'   - `keep_vars` Vector of variable names or column indices denoting variables that should be included in the forest. Default: `NULL`.
#'   - `drop_vars` Vector of variable names or column indices denoting variables that should be excluded from the forest. Default: `NULL`. If both `drop_vars` and `keep_vars` are set, `drop_vars` will be ignored.
#'   - `num_features_subsample` How many features to subsample when growing each tree for the GFR algorithm. Defaults to the number of features in the training dataset.
#'
#' @param variance_forest_params (Optional) A list of variance forest model parameters. Only the following keys can be modified by `continueSampling`; any other parameters specified
#' prompt a warning and are ignored. Unspecified keys default to their value set in `bart()`.
#'
#'   - `alpha` Prior probability of splitting for a tree of depth 0 in the variance model. Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`. Default: `0.95`.
#'   - `beta` Exponent that decreases split probabilities for nodes of depth > 0 in the variance model. Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`. Default: `2`.
#'   - `min_samples_leaf` Minimum allowable size of a leaf, in terms of training samples, in the variance model. Default: `5`.
#'   - `max_depth` Maximum depth of any tree in the ensemble in the variance model. Default: `10`. Can be overridden with ``-1`` which does not enforce any depth limits on trees.
#'   - `var_forest_prior_shape` Shape parameter in the `IG(var_forest_prior_shape, var_forest_prior_scale)` conditional error variance model (which is only sampled if `num_trees > 0`). Calibrated internally as `num_trees / leaf_prior_calibration_param^2 + 0.5` if not set.
#'   - `var_forest_prior_scale` Scale parameter in the `IG(var_forest_prior_shape, var_forest_prior_scale)` conditional error variance model (which is only sampled if `num_trees > 0`). Calibrated internally as `num_trees / leaf_prior_calibration_param^2` if not set.
#'   - `keep_vars` Vector of variable names or column indices denoting variables that should be included in the forest. Default: `NULL`.
#'   - `drop_vars` Vector of variable names or column indices denoting variables that should be excluded from the forest. Default: `NULL`. If both `drop_vars` and `keep_vars` are set, `drop_vars` will be ignored.
#'   - `num_features_subsample` How many features to subsample when growing each tree for the GFR algorithm. Defaults to the number of features in the training dataset.
#'
#' @param random_effects_params (Optional) A list of random effects model parameters. Only the following keys can be modified by `continueSampling`; any other parameters specified
#' prompt a warning and are ignored. Unspecified keys default to their value set in `bart()`.
#'
#'   - `variance_prior_shape` Shape parameter for the inverse gamma prior on the variance of the random effects "group parameter." Default: `1`.
#'   - `variance_prior_scale` Scale parameter for the inverse gamma prior on the variance of the random effects "group parameter." Default: `1`.
#'
#'
#' @param ... Other parameters (ignored).
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
#' bart_model <- bart(X_train = X_train, y_train = y_train,
#'                    num_gfr = 10, num_burnin = 0, num_mcmc = 10)
#' continueSampling(bart_model, X_train = X_train, y_train = y_train,
#'                  num_gfr = 10, num_burnin = 0, num_mcmc = 10)
continueSampling.bartmodel <- function(
  object,
  X_train,
  y_train,
  leaf_basis_train = NULL,
  rfx_group_ids_train = NULL,
  rfx_basis_train = NULL,
  X_test = NULL,
  leaf_basis_test = NULL,
  rfx_group_ids_test = NULL,
  rfx_basis_test = NULL,
  observation_weights_train = NULL,
  observation_weights = NULL,
  num_gfr = 5,
  num_burnin = 0,
  num_mcmc = 100,
  previous_model_json = NULL,
  previous_model_warmstart_sample_num = NULL,
  general_params = list(),
  mean_forest_params = list(),
  variance_forest_params = list(),
  random_effects_params = list(),
  ...
) {
  # Input checks
  if (!inherits(object, "bartmodel")) {
    stop("object must be a bartmodel")
  }
  if (!object$model_params$include_mean_forest) {
    stop("Continued sampling currently requires a mean forest")
  }
  bart_config <- object[["bart_config"]]
  if (is.null(bart_config)) {
    stop(
      "Cannot continue sampling: the cached sampler configuration is unavailable. ",
      "Continuation is not supported for deserialized models (a model loaded from JSON)."
    )
  }
  if (object$model_params$outcome_model$link != "identity") {
    stop(
      "Continued sampling is not yet supported for probit or cloglog link functions"
    )
  }

  # Preprocessing user-supplied parameters for continuation and merging with existing config
  overlayContinuationParams <- function(
    config,
    user_params,
    mapping,
    list_name
  ) {
    if (length(user_params) == 0) {
      return(config)
    }
    unknown <- setdiff(names(user_params), names(mapping))
    if (length(unknown) > 0) {
      warning(
        "The following ",
        list_name,
        " cannot be changed on continuation and will be ignored: ",
        paste(unknown, collapse = ", ")
      )
    }
    for (key in intersect(names(user_params), names(mapping))) {
      val <- user_params[[key]]
      if (!is.null(val)) config[[mapping[[key]]]] <- val
    }
    config
  }

  # Changeable keys, mapped from user-facing names to internal bart_config keys.
  general_map <- list(
    random_seed = "random_seed",
    keep_gfr = "keep_gfr",
    keep_burnin = "keep_burnin",
    cutpoint_grid_size = "cutpoint_grid_size",
    sigma2_global_shape = "a_sigma2_global",
    sigma2_global_scale = "b_sigma2_global",
    num_threads = "num_threads",
    verbose = "verbose"
  )
  mean_forest_map <- list(
    alpha = "alpha_mean",
    beta = "beta_mean",
    min_samples_leaf = "min_samples_leaf_mean",
    max_depth = "max_depth_mean",
    num_features_subsample = "num_features_subsample_mean",
    sigma2_leaf_shape = "a_sigma2_mean",
    sigma2_leaf_scale = "b_sigma2_mean"
  )
  variance_forest_map <- list(
    alpha = "alpha_variance",
    beta = "beta_variance",
    min_samples_leaf = "min_samples_leaf_variance",
    max_depth = "max_depth_variance",
    num_features_subsample = "num_features_subsample_variance",
    var_forest_prior_shape = "shape_variance_forest",
    var_forest_prior_scale = "scale_variance_forest"
  )
  random_effects_map <- list(
    variance_prior_shape = "rfx_variance_prior_shape",
    variance_prior_scale = "rfx_variance_prior_scale"
  )

  # Temporarily store copies and remove user-provided parameters
  user_variable_weights <- general_params[["variable_weights"]]
  keep_vars_mean <- mean_forest_params[["keep_vars"]]
  drop_vars_mean <- mean_forest_params[["drop_vars"]]
  keep_vars_variance <- variance_forest_params[["keep_vars"]]
  drop_vars_variance <- variance_forest_params[["drop_vars"]]
  user_keep_every <- general_params[["keep_every"]]
  general_params[["variable_weights"]] <- NULL
  general_params[["keep_every"]] <- NULL
  mean_forest_params[["keep_vars"]] <- NULL
  mean_forest_params[["drop_vars"]] <- NULL
  variance_forest_params[["keep_vars"]] <- NULL
  variance_forest_params[["drop_vars"]] <- NULL

  # Fold any user-provided parameters into the config stored with the BART model
  config <- bart_config
  config <- overlayContinuationParams(
    config,
    general_params,
    general_map,
    "general_params"
  )
  config <- overlayContinuationParams(
    config,
    mean_forest_params,
    mean_forest_map,
    "mean_forest_params"
  )
  config <- overlayContinuationParams(
    config,
    variance_forest_params,
    variance_forest_map,
    "variance_forest_params"
  )
  config <- overlayContinuationParams(
    config,
    random_effects_params,
    random_effects_map,
    "random_effects_params"
  )

  # Handle overrides to keep_every / keep_gfr
  keep_every <- if (!is.null(user_keep_every)) user_keep_every else 1
  keep_gfr <- if (!is.null(general_params$keep_gfr)) {
    general_params$keep_gfr
  } else {
    TRUE
  }

  # Preprocessing training data
  train_set_metadata <- object$train_set_metadata
  # Keep the raw (pre-preprocessing) covariates so keep_vars / drop_vars can be resolved against the
  # original variable names / count, exactly as bart() does.
  X_train_raw <- X_train
  X_train <- preprocessPredictionData(X_train, train_set_metadata)
  if (ncol(X_train) != object$model_params$num_covariates) {
    stop(sprintf(
      "Re-supplied covariates have %d columns; model expects %d",
      ncol(X_train),
      object$model_params$num_covariates
    ))
  }
  y_train <- as.numeric(y_train)
  if (nrow(X_train) != length(y_train)) {
    stop("X_train and y_train have differing numbers of observations")
  }

  # Update split-variable weights if variable_weights / keep_vars / drop_vars changed.
  cstate <- object$continuation_state
  if (is.null(cstate)) {
    stop(
      "Cannot continue sampling: cached continuation state is unavailable for this model."
    )
  }
  variable_weights <- if (!is.null(user_variable_weights)) {
    user_variable_weights
  } else {
    cstate$variable_weights
  }
  if (length(variable_weights) != object$model_params$num_covariates) {
    stop(sprintf(
      "variable_weights must have length %d (the number of covariates)",
      object$model_params$num_covariates
    ))
  }
  if (any(variable_weights < 0)) {
    stop("variable_weights cannot have any negative weights")
  }
  variable_subset_mean <- if (
    !is.null(keep_vars_mean) || !is.null(drop_vars_mean)
  ) {
    resolveVariableSubset(keep_vars_mean, drop_vars_mean, X_train_raw, "mean")
  } else {
    cstate$variable_subset_mean
  }
  variable_subset_variance <- if (
    !is.null(keep_vars_variance) || !is.null(drop_vars_variance)
  ) {
    resolveVariableSubset(
      keep_vars_variance,
      drop_vars_variance,
      X_train_raw,
      "variance"
    )
  } else {
    cstate$variable_subset_variance
  }
  original_var_indices <- train_set_metadata$original_var_indices
  if (object$model_params$include_mean_forest) {
    config[["var_weights_mean"]] <- expandVariableWeights(
      variable_weights,
      original_var_indices,
      variable_subset_mean
    )
  }
  if (object$model_params$include_variance_forest) {
    config[["var_weights_variance"]] <- expandVariableWeights(
      variable_weights,
      original_var_indices,
      variable_subset_variance
    )
  }

  if (object$model_params$has_basis && is.null(leaf_basis_train)) {
    stop(
      "This model was fit with a leaf basis; leaf_basis_train must be supplied to continue sampling"
    )
  }

  # Re-supplied random effects data, encoded with the model's stored factor levels so that the
  # group ids map to the same 0-indexed categories as the original fit.
  has_rfx <- object$model_params$has_rfx
  rfx_intercept <- (object$model_params$rfx_model_spec == "intercept_only")
  if (has_rfx && is.null(rfx_group_ids_train)) {
    stop(
      "This model was fit with random effects; rfx_group_ids_train must be supplied to continue sampling"
    )
  }
  if (has_rfx && is.null(rfx_basis_train) && !rfx_intercept) {
    stop(
      "This model was fit with a non-intercept-only random effects model; rfx_basis_train must be supplied to continue sampling"
    )
  }
  rfx_num_groups <- 0L
  rfx_basis_dim <- 0L
  if (has_rfx) {
    group_ids_factor <- factor(
      rfx_group_ids_train,
      levels = object$rfx_unique_group_ids
    )
    if (any(is.na(group_ids_factor))) {
      stop(
        "rfx_group_ids_train contains group labels not present in the fitted model"
      )
    }
    rfx_group_ids_train <- as.integer(group_ids_factor)
    rfx_num_groups <- length(object$rfx_unique_group_ids)
    rfx_basis_dim <- as.integer(object$model_params$num_rfx_basis)
  }

  # RNG state can be overriden by a new (non-default) user-provided seed, otherwise it resumes from the model's available RNG state.
  override_seed <- !is.null(general_params$random_seed)
  rng_state_in <- if (
    !override_seed && !is.null(object$model_params$rng_state)
  ) {
    object$model_params$rng_state
  } else {
    ""
  }

  # Continue sampling the BART model in C++, augmenting the pointer stored in object$samples and returning updated metadata about the new total number of samples and the new RNG state
  bart_samples <- object$samples
  num_history <- bart_samples$num_samples()
  bart_metadata <- bart_continue_sample_cpp(
    samples = bart_samples$samples_ptr,
    X_train = X_train,
    y_train = y_train,
    X_test = NULL,
    n_train = nrow(X_train),
    n_test = 0L,
    p = ncol(X_train),
    basis_train = leaf_basis_train,
    basis_test = NULL,
    basis_dim = if (!is.null(leaf_basis_train)) ncol(leaf_basis_train) else 0L,
    obs_weights_train = NULL,
    obs_weights_test = NULL,
    rfx_group_ids_train = if (has_rfx) rfx_group_ids_train else NULL,
    rfx_group_ids_test = NULL,
    rfx_basis_train = if (has_rfx && !rfx_intercept) rfx_basis_train else NULL,
    rfx_basis_test = NULL,
    rfx_num_groups = as.integer(rfx_num_groups),
    rfx_basis_dim = as.integer(rfx_basis_dim),
    num_gfr = as.integer(num_gfr),
    num_burnin = as.integer(num_burnin),
    keep_every = as.integer(keep_every),
    num_mcmc = as.integer(num_mcmc),
    keep_gfr = keep_gfr,
    rng_state_in = rng_state_in,
    override_seed = override_seed,
    config_input = config
  )

  # Update model metadata
  object$model_params$num_samples <- bart_samples$num_samples()
  object$model_params$num_mcmc <- object$model_params$num_mcmc + num_mcmc
  if (keep_gfr) {
    object$model_params$num_gfr <- object$model_params$num_gfr + num_gfr
  }
  object$model_params$rng_state <- bart_metadata[["rng_state"]]

  return(object)
}

#' @title Print Summary of BART Model
#' @description Prints a summary of the BART model, including the model terms and their specifications.
#' @param x The BART model object
#' @param ... Additional arguments
#' @export
#' @return BART model object unchanged after printing summary
print.bartmodel <- function(x, ...) {
  # What type of model was run
  model_terms <- c()
  if (x$model_params$include_mean_forest) {
    model_terms <- c(model_terms, "mean forest")
  }
  if (x$model_params$include_variance_forest) {
    model_terms <- c(model_terms, "variance forest")
  }
  if (x$model_params$has_rfx) {
    model_terms <- c(model_terms, "additive random effects")
  }
  if (x$model_params$sample_sigma2_global) {
    model_terms <- c(model_terms, "global error variance model")
  }
  if (x$model_params$sample_sigma2_leaf) {
    model_terms <- c(model_terms, "mean forest leaf scale model")
  }
  if (length(model_terms) > 2) {
    summary_message <- paste0(
      "stochtree::bart() run with ",
      paste0(
        paste0(model_terms[1:(length(model_terms) - 1)], collapse = ", "),
        ", and ",
        model_terms[length(model_terms)]
      )
    )
  } else if (length(model_terms) == 2) {
    summary_message <- paste0(
      "stochtree::bart() run with ",
      paste0(model_terms, collapse = " and ")
    )
  } else {
    summary_message <- paste0("stochtree::bart() run with ", model_terms)
  }

  # Outcome and leaf model details
  outcome_model <- x$model_params$outcome_model
  is_probit <- (outcome_model$link == "probit" &&
    outcome_model$outcome == "binary")
  is_binary_cloglog <- (outcome_model$link == "cloglog" &&
    outcome_model$outcome == "binary")
  is_ordinal_cloglog <- (outcome_model$link == "cloglog" &&
    outcome_model$outcome == "ordinal")
  if (is_ordinal_cloglog) {
    num_categories <- x$model_params$cloglog_num_categories
    outcome_model_summary <- paste0(
      "Ordinal outcome with ",
      num_categories,
      " categories was modeled with a complementary log-log (cloglog) link function"
    )
  } else if (is_binary_cloglog) {
    outcome_model_summary <- paste0(
      "Binary outcome was modeled with a complementary log-log (cloglog) link function"
    )
  } else if (is_probit) {
    outcome_model_summary <- paste0(
      "Binary outcome was modeled with a probit link function"
    )
  } else {
    outcome_model_summary <- paste0(
      "Continuous outcome was modeled as Gaussian"
    )
  }
  if (x$model_params$leaf_regression) {
    summary_message <- paste0(
      summary_message,
      "\n",
      outcome_model_summary,
      " with a leaf regression prior with ",
      x$model_params$leaf_dimension,
      " bases for the mean forest"
    )
  } else if (x$model_params$include_mean_forest) {
    summary_message <- paste0(
      summary_message,
      "\n",
      outcome_model_summary,
      " with a constant leaf prior for the mean forest"
    )
  } else {
    summary_message <- paste0(
      summary_message,
      "\n",
      outcome_model_summary,
    )
  }

  # Standardization
  if (x$model_params$standardize) {
    summary_message <- paste0(
      summary_message,
      "\n",
      "Outcome was standardized"
    )
  }

  # Random effects details
  if (x$model_params$has_rfx) {
    if (x$model_params$rfx_model_spec == "custom") {
      summary_message <- paste0(
        summary_message,
        "\n",
        "Random effects were fit with a user-supplied basis"
      )
    } else if (x$model_params$rfx_model_spec == "intercept_only") {
      summary_message <- paste0(
        summary_message,
        "\n",
        "Random effects were fit with an 'intercept-only' parameterization"
      )
    }
  }

  # Sampler details
  summary_message <- paste0(
    summary_message,
    "\n",
    "The sampler was run for ",
    x$model_params$num_gfr,
    " GFR iterations, with ",
    x$model_params$num_chains,
    ifelse(
      x$model_params$num_chains == 1,
      " chain of ",
      " chains of "
    ),
    x$model_params$num_burnin,
    " burn-in iterations and ",
    x$model_params$num_mcmc,
    " MCMC iterations, ",
    ifelse(
      x$model_params$keep_every == 1,
      "retaining every iteration (i.e. no thinning)",
      paste0(
        "retaining every ",
        x$model_params$keep_every,
        "th iteration (i.e. thinning)"
      )
    )
  )

  # Print the model details
  cat(summary_message, "\n")

  # Return bart_model invisibly
  invisible(x)
}

#' @title Summarize BART Model Fit and Parameters
#' @description Summarize a BART fit with a description of the model that was fit and numeric summaries of any sampled quantities.
#' @param object The BART model object
#' @param ... Additional arguments
#' @export
#' @return BART model object unchanged after summarizing
summary.bartmodel <- function(object, ...) {
  # First, print the BART model
  tmp <- print(object)

  # Summarize any sampled quantities

  # Global error scale
  if (object$model_params$sample_sigma2_global) {
    sigma2_samples <- object$samples$global_var_samples()
    n_samples <- length(sigma2_samples)
    mean_sigma2 <- mean(sigma2_samples)
    sd_sigma2 <- sd(sigma2_samples)
    quantiles_sigma2 <- quantile(
      sigma2_samples,
      probs = c(0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
    )
    cat(sprintf(
      "Summary of sigma^2 posterior: \n%d samples, mean = %.3f, standard deviation = %.3f, quantiles:\n",
      n_samples,
      mean_sigma2,
      sd_sigma2
    ))
    print(quantiles_sigma2)
  }

  # Leaf scale
  if (object$model_params$sample_sigma2_leaf) {
    sigma2_leaf_samples <- object$samples$leaf_scale_samples()
    n_samples <- length(sigma2_leaf_samples)
    mean_sigma2 <- mean(sigma2_leaf_samples)
    sd_sigma2 <- sd(sigma2_leaf_samples)
    quantiles_sigma2 <- quantile(
      sigma2_leaf_samples,
      probs = c(0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
    )
    cat(sprintf(
      "Summary of leaf scale posterior: \n%d samples, mean = %.3f, standard deviation = %.3f, quantiles:\n",
      n_samples,
      mean_sigma2,
      sd_sigma2
    ))
    print(quantiles_sigma2)
  }

  # Determine whether outcome model is binary / ordinal
  outcome_model <- object$model_params$outcome_model
  is_probit <- (outcome_model$link == "probit" &&
    outcome_model$outcome == "binary")
  is_binary_cloglog <- (outcome_model$link == "cloglog" &&
    outcome_model$outcome == "binary")
  is_ordinal_cloglog <- (outcome_model$link == "cloglog" &&
    outcome_model$outcome == "ordinal")
  non_continuous_outcome <- (is_probit ||
    is_binary_cloglog ||
    is_ordinal_cloglog)

  # In-sample predictions
  if (object$samples$has_yhat_train()) {
    y_hat_train_mean <- rowMeans(object$samples$y_hat_train())
    n_y_hat_train <- length(y_hat_train_mean)
    mean_y_hat_train <- mean(y_hat_train_mean)
    sd_y_hat_train <- sd(y_hat_train_mean)
    quantiles_y_hat_train <- quantile(
      y_hat_train_mean,
      probs = c(0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
    )
    if (non_continuous_outcome) {
      summary_text <- "Summary of in-sample inverse-link-scale posterior predictions: \n%d observations, mean = %.3f, standard deviation = %.3f, quantiles:\n"
    } else {
      summary_text <- "Summary of in-sample posterior mean predictions: \n%d observations, mean = %.3f, standard deviation = %.3f, quantiles:\n"
    }
    cat(sprintf(
      summary_text,
      n_y_hat_train,
      mean_y_hat_train,
      sd_y_hat_train
    ))
    print(quantiles_y_hat_train)
  }

  # Test-set predictions
  if (object$samples$has_yhat_test()) {
    y_hat_test_mean <- rowMeans(object$samples$y_hat_test())
    n_y_hat_test <- length(y_hat_test_mean)
    mean_y_hat_test <- mean(y_hat_test_mean)
    sd_y_hat_test <- sd(y_hat_test_mean)
    quantiles_y_hat_test <- quantile(
      y_hat_test_mean,
      probs = c(0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
    )
    if (non_continuous_outcome) {
      summary_text <- "Summary of test-set inverse-link-scale posterior predictions: \n%d observations, mean = %.3f, standard deviation = %.3f, quantiles:\n"
    } else {
      summary_text <- "Summary of test-set posterior mean predictions: \n%d observations, mean = %.3f, standard deviation = %.3f, quantiles:\n"
    }
    cat(sprintf(
      summary_text,
      n_y_hat_test,
      mean_y_hat_test,
      sd_y_hat_test
    ))
    print(quantiles_y_hat_test)
  }

  # Random effects
  if (object$model_params$has_rfx) {
    rfx_samples <- getRandomEffectSamples(object)
    rfx_beta_samples <- rfx_samples$beta_samples
    if (length(dim(rfx_beta_samples)) > 2) {
      cat(
        "Random effects summary of variance components across groups and posterior draws:\n"
      )
      rfx_component_means <- apply(rfx_beta_samples, 1, mean)
      rfx_component_sds <- apply(rfx_beta_samples, 1, sd)
      cat("Variance component means: ", rfx_component_means, "\n")
      cat("Variance component standard deviations: ", rfx_component_sds, "\n")
      quantile_summary <- t(apply(
        rfx_beta_samples,
        1,
        quantile,
        probs = c(0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
      ))
      cat("Variance component quantiles:\n")
      print(quantile_summary)
    } else {
      cat(
        "Random effects summary of variance components across groups and posterior draws:\n"
      )
      rfx_component_means <- mean(rfx_beta_samples)
      rfx_component_sds <- sd(rfx_beta_samples)
      cat("Random effects overall mean: ", rfx_component_means, "\n")
      cat(
        "Random effects overall standard deviation: ",
        rfx_component_sds,
        "\n"
      )
      cat("Random effects overall quantiles:\n")
      quantile_summary <- quantile(
        rfx_beta_samples,
        probs = c(0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
      )
      cat("Random effects overall quantiles:\n")
      print(quantile_summary)
    }
  }

  # Return bart_model invisibly
  invisible(object)
}

#' @title Plot BART Model Fit
#' @description Plot the BART model fit and any relevant sampled quantities. This will default to a traceplot of the global error scale and the in-sample mean forest predictions for the first train set observation. Since `stochtree::bart()` is flexible and it's possible to sample a model with a fixed global error scale and no mean forest, this procedure is adaptive and will attempt to plot a trace of whichever model terms are included if these two default terms are omitted.
#' @param x The BART model object
#' @param ... Additional arguments
#' @export
#' @return BART model object unchanged after summarizing
plot.bartmodel <- function(x, ...) {
  # Check if model has global error scale samples
  has_sigma2_samples <- x$model_params$sample_sigma2_global
  has_mean_forest_preds <- x$samples$has_yhat_train()

  # Check if model is ordinal / binary
  is_probit <- (x$model_params$outcome_model$link == "probit" &&
    x$model_params$outcome_model$outcome == "binary")
  is_binary_cloglog <- (x$model_params$outcome_model$link == "cloglog" &&
    x$model_params$outcome_model$outcome == "binary")
  is_ordinal_cloglog <- (x$model_params$outcome_model$link == "cloglog" &&
    x$model_params$outcome_model$outcome == "ordinal")
  non_continuous_outcome <- (is_probit ||
    is_binary_cloglog ||
    is_ordinal_cloglog)

  # First try combinations of sigma2 and mean forest predictions
  if (has_sigma2_samples || has_mean_forest_preds) {
    if (has_sigma2_samples) {
      plot(
        x$samples$global_var_samples(),
        type = "l",
        ylab = "Sigma^2",
        main = "Global error scale traceplot"
      )
    } else if (has_mean_forest_preds) {
      if (non_continuous_outcome) {
        plot_text <- "In-sample inverse-link-scale prediction trace for the first train set observation"
      } else {
        plot_text <- "In-sample mean function trace for the first train set observation"
      }
      plot(
        x$samples$y_hat_train()[1, ],
        type = "l",
        ylab = "Predictions",
        main = plot_text
      )
    }
  } else {
    stop(
      "This model does not have enough model terms / parameter traces to produce stochtree's default plots. See `predict.bartmodel()` for examples of how to further investigate your model."
    )
  }

  # Return x invisibly
  invisible(x)
}

#' @title Extract Random Effects Samples from BART Model
#' @description
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
  result <- list()

  if (!object$model_params$has_rfx) {
    warning("This model has no RFX terms, returning an empty list")
    return(result)
  }

  # Extract the samples
  rfx_samples <- object$samples$materialize_rfx_samples()
  result <- rfx_samples$extract_parameter_samples()

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

#' @title Extract BART Forests
#' @description Extract a forest from a BART model by name.
#' If the requested forest type is not found, an error is thrown.
#' The following conventions are used for forest:
#' - Mean forest: `"mean"`, `"mean_forest"`
#' - Variance forest: `"variance"`, `"variance_forest"`
#'
#' @param object Object of type `bartmodel` containing draws of a BART model and associated sampling outputs.
#' @param term Name of the forest to extract (e.g., `"mean"`, `"variance"`, etc.)
#' @return Object of class ForestSamples containing a deep copy of the requested forest samples.
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
#' mean_forest <- extractForest(bart_model, "mean")
extractForest.bartmodel <- function(object, term) {
  if (term %in% c("mean", "mean_forest")) {
    if (object$samples$has_mean_forest()) {
      return(object$samples$materialize_mean_forest())
    } else {
      stop("This model does not have a mean forest")
    }
  }

  if (term %in% c("variance", "variance_forest")) {
    if (object$samples$has_variance_forest()) {
      return(object$samples$materialize_variance_forest())
    } else {
      stop("This model does not have a variance forest")
    }
  }

  stop(paste0("term ", term, " is not a valid BART forest term"))
}

#' @title Extract BART Parameter Samples
#' @description Extract a vector, matrix or array of parameter samples from a BART model by name.
#' Random effects are handled by a separate `getRandomEffectSamples` function due to the complexity of the random effects parameters.
#' If the requested model term is not found, an error is thrown.
#' The following conventions are used for parameter names:
#' - Global error variance: `"sigma2"`, `"global_error_scale"`, `"sigma2_global"`
#' - Leaf scale: `"sigma2_leaf"`, `"leaf_scale"`
#' - In-sample mean function predictions: `"y_hat_train"`
#' - Test set mean function predictions: `"y_hat_test"`
#' - In-sample variance forest predictions: `"sigma2_x_train"`, `"var_x_train"`
#' - Test set variance forest predictions: `"sigma2_x_test"`, `"var_x_test"`
#' - Ordinal model cutpoints (valid only for ordinal cloglog models): `"cloglog_cutpoints"`, `"cutpoints"`
#'
#' @param object Object of type `bartmodel` containing draws of a BART model and associated sampling outputs.
#' @param term Name of the parameter to extract (e.g., `"sigma2"`, `"y_hat_train"`, etc.)
#' @return Array of parameter samples. If the underlying parameter is a scalar, this will be a vector of length `num_samples`.
#' If the underlying parameter is vector-valued, this will be (`parameter_dimension` x `num_samples`) matrix, and if the underlying
#' parameter is multidimensional, this will be an array of dimension (`parameter_dimension_1` x `parameter_dimension_2` x ... x `num_samples`).
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
#' sigma2_samples <- extractParameter(bart_model, "sigma2")
extractParameter.bartmodel <- function(object, term) {
  if (term %in% c("sigma2", "global_error_scale", "sigma2_global")) {
    if (object$samples$has_global_var_samples()) {
      return(object$samples$global_var_samples())
    } else {
      stop("This model does not have global variance parameter samples")
    }
  }

  if (term %in% c("sigma2_leaf", "leaf_scale")) {
    if (object$samples$has_leaf_scale_samples()) {
      return(object$samples$leaf_scale_samples())
    } else {
      stop("This model does not have leaf variance parameter samples")
    }
  }

  if (term %in% c("y_hat_train")) {
    if (object$samples$has_mean_forest_predictions_train()) {
      return(object$samples$mean_forest_predictions_train())
    } else {
      stop(
        "This model does not have in-sample mean function prediction samples"
      )
    }
  }

  if (term %in% c("y_hat_test")) {
    if (object$samples$has_mean_forest_predictions_test()) {
      return(object$samples$mean_forest_predictions_test())
    } else {
      stop("This model does not have test set mean function prediction samples")
    }
  }

  if (term %in% c("sigma2_x_train", "var_x_train")) {
    if (object$samples$has_variance_forest_predictions_train()) {
      return(object$samples$variance_forest_predictions_train())
    } else {
      stop("This model does not have in-sample variance forest predictions")
    }
  }

  if (term %in% c("sigma2_x_test", "var_x_test")) {
    if (object$samples$has_variance_forest_predictions_test()) {
      return(object$samples$variance_forest_predictions_test())
    } else {
      stop("This model does not have test set variance forest predictions")
    }
  }

  if (term %in% c("cloglog_cutpoints", "cutpoints")) {
    if (object$samples$has_cloglog_cutpoint_samples()) {
      return(object$samples$cloglog_cutpoint_samples())
    } else {
      stop("This model does not have ordinal cutpoint samples")
    }
  }

  stop(paste0("term ", term, " is not a valid BART model term"))
}

#' @title Convert BART Model to JSON
#' @rdname BARTSerialization
#' @param object Object of type `bartmodel` containing draws of a BART model and associated sampling outputs.
#' @export
saveBARTModelToJson <- function(object) {
  jsonobj <- createCppJson()

  if (!inherits(object, "bartmodel")) {
    stop("`object` must be a BART model")
  }

  if (is.null(object$model_params)) {
    stop("This BCF model has not yet been sampled")
  }

  # Add the samples C++ object to the JSON object, which will handle serialization of all the sampled parameters and predictions in a cross-platform way.
  bart_samples <- object$samples
  bart_samples$append_to_json(jsonobj)

  # Add version stamp and global parameters
  jsonobj$add_string("stochtree_version", getStochtreeVersion())
  jsonobj$add_string("platform", "R")
  jsonobj$add_integer("schema_version", STOCHTREE_SCHEMA_VERSION)
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
  jsonobj$add_string(
    "outcome",
    object$model_params$outcome_model$outcome,
    "outcome_model"
  )
  jsonobj$add_string(
    "link",
    object$model_params$outcome_model$link,
    "outcome_model"
  )
  jsonobj$add_boolean(
    "probit_outcome_model",
    object$model_params$probit_outcome_model
  )
  jsonobj$add_string(
    "rfx_model_spec",
    object$model_params$rfx_model_spec
  )
  if (object$model_params$outcome_model$link == "cloglog") {
    jsonobj$add_scalar(
      "cloglog_num_categories",
      object$model_params$cloglog_num_categories
    )
  }

  # Add random effects (if present)
  if (object$model_params$has_rfx) {
    jsonobj$add_string_vector(
      "rfx_unique_group_ids",
      object$rfx_unique_group_ids,
      subfolder_name = "random_effects"
    )
    # Cross-platform compatible on the rfx axis iff the group id levels are
    # integer-valued (Python supports only integer group ids).
    rfx_compatible <- all(grepl("^-?[0-9]+$", object$rfx_unique_group_ids))
    jsonobj$add_boolean(
      "cross_platform_compatible",
      rfx_compatible,
      subfolder_name = "random_effects"
    )
  }

  # Add covariate preprocessor metadata
  preprocessor_metadata_string <- savePreprocessorToJsonString(
    object$train_set_metadata
  )
  jsonobj$add_string("covariate_preprocessor", preprocessor_metadata_string)

  return(jsonobj)
}

#' @title Save BART Model to JSON File
#' @rdname BARTSerialization
#' @param object Object of type `bartmodel` containing draws of a BART model and associated sampling outputs.
#' @param filename String of filepath, must end in ".json"
#'
#' @export
saveBARTModelToJsonFile <- function(object, filename) {
  # Convert to Json
  jsonobj <- saveBARTModelToJson(object)

  # Save to file
  jsonobj$save_file(filename)
}

#' @title Convert BART Model to JSON String
#' @rdname BARTSerialization
#' @param object Object of type `bartmodel` containing draws of a BART model and associated sampling outputs.
#' @export
saveBARTModelToJsonString <- function(object) {
  # Convert to Json
  jsonobj <- saveBARTModelToJson(object)

  # Dump to string
  return(jsonobj$return_json_string())
}

# Reconstruct mean-forest leaf-model metadata that is not serialized directly.
# These fields are fully determined by num_basis (a basis/regression leaf model
# has num_basis > 0), so we derive them on load rather than store redundant
# copies. Restoring them keeps print()/summary()/predict() working on a
# reloaded model (they were previously NULL and errored).
.reconstructBartLeafModelFields <- function(model_params) {
  num_basis <- model_params[["num_basis"]]
  has_basis <- isTRUE(num_basis > 0)
  model_params[["has_basis"]] <- has_basis
  model_params[["leaf_regression"]] <- has_basis
  model_params[["is_leaf_constant"]] <- !has_basis
  model_params[["leaf_dimension"]] <- if (has_basis) num_basis else 1
  model_params
}

# In-place v0 -> v1 migration for a BART model envelope: positional forest keys
# (forests/forest_0, ...) -> named keys (mean_forest / variance_forest), driven by
# the include_*_forest flags (unchanged across v0/v1).
.migrateBartJsonV0ToV1 <- function(json_object, loaded_version) {
  json_object$add_string("platform", inferPlatformV0(json_object, "R"))
  include_mean <- json_object$get_boolean_or_default(
    "include_mean_forest",
    FALSE
  )
  include_variance <- json_object$get_boolean_or_default(
    "include_variance_forest",
    FALSE
  )
  if (include_mean) {
    json_object$rename_field(
      "forest_0",
      "mean_forest",
      subfolder_name = "forests"
    )
    if (include_variance) {
      json_object$rename_field(
        "forest_1",
        "variance_forest",
        subfolder_name = "forests"
      )
    }
  } else if (include_variance) {
    json_object$rename_field(
      "forest_0",
      "variance_forest",
      subfolder_name = "forests"
    )
  }
  # R's legacy preprocessor key -> unified v1 key (no-op for Python v0 JSON,
  # which already uses `covariate_preprocessor`).
  json_object$rename_field("preprocessor_metadata", "covariate_preprocessor")
  # Relocate R's top-level rfx unique group ids into the random_effects subfolder
  # (no-op for Python v0 JSON, which never wrote this field).
  if (json_object$contains("rfx_unique_group_ids")) {
    .rfx_uids <- json_object$get_string_vector("rfx_unique_group_ids")
    json_object$add_string_vector(
      "rfx_unique_group_ids",
      .rfx_uids,
      subfolder_name = "random_effects"
    )
    json_object$erase_field("rfx_unique_group_ids")
  }
}

#' @description Create a BARTSamples object from JSON
#' @noRd
createBARTSamplesFromJson <- function(json) {
  bart_samples <- BARTSamples$new()
  bart_samples$from_json(json)
  bart_samples
}

#' @title Convert JSON to BART Model
#' @rdname BARTSerialization
#' @param json_object Object of type `CppJson` containing Json representation of a BART model
#' @export
createBARTModelFromJson <- function(json_object) {
  # Initialize the BCF model
  output <- list()

  # Helpers for optional-field presence checks
  .ver <- inferStochtreeJsonVersion(json_object)
  resolveSchemaVersion(json_object, migrate = .migrateBartJsonV0ToV1)
  cross_platform <- enforceCrossPlatformGate(json_object, "R")
  has_field <- function(name) {
    json_contains_field_cpp(json_object$json_ptr, name)
  }
  has_subfolder_field <- function(subfolder, name) {
    json_contains_field_subfolder_cpp(json_object$json_ptr, subfolder, name)
  }

  # Unpack model params
  include_mean_forest <- json_object$get_boolean(
    "include_mean_forest"
  )
  include_variance_forest <- json_object$get_boolean(
    "include_variance_forest"
  )
  model_params <- list()
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

  if (has_field("has_rfx_basis")) {
    model_params[["has_rfx_basis"]] <- json_object$get_boolean("has_rfx_basis")
    model_params[["num_rfx_basis"]] <- json_object$get_scalar("num_rfx_basis")
  } else {
    model_params[["has_rfx_basis"]] <- FALSE
    model_params[["num_rfx_basis"]] <- 1
    warning(paste0(
      "Fields 'has_rfx_basis' and 'num_rfx_basis' not found in JSON (model appears to have been ",
      "serialized under stochtree ",
      .ver,
      "). Defaulting to FALSE / 1. ",
      "Re-save your model to suppress this warning."
    ))
  }

  model_params[["num_gfr"]] <- json_object$get_scalar("num_gfr")
  model_params[["num_burnin"]] <- json_object$get_scalar("num_burnin")
  model_params[["num_mcmc"]] <- json_object$get_scalar("num_mcmc")
  model_params[["num_samples"]] <- json_object$get_scalar("num_samples")
  model_params[["num_covariates"]] <- if (has_field("num_covariates")) {
    json_object$get_scalar("num_covariates")
  } else {
    NA_real_
  }
  model_params[["num_basis"]] <- json_object$get_scalar("num_basis")
  model_params <- .reconstructBartLeafModelFields(model_params)
  model_params[["requires_basis"]] <- json_object$get_boolean("requires_basis")

  if (has_field("num_chains")) {
    model_params[["num_chains"]] <- json_object$get_scalar("num_chains")
  } else {
    model_params[["num_chains"]] <- 1
    warning(paste0(
      "Field 'num_chains' not found in JSON (model appears to have been serialized under stochtree ",
      .ver,
      "). Defaulting to 1. Re-save your model to suppress this warning."
    ))
  }

  if (has_field("keep_every")) {
    model_params[["keep_every"]] <- json_object$get_scalar("keep_every")
  } else {
    model_params[["keep_every"]] <- 1
    warning(paste0(
      "Field 'keep_every' not found in JSON (model appears to have been serialized under stochtree ",
      .ver,
      "). Defaulting to 1. Re-save your model to suppress this warning."
    ))
  }

  model_params[["probit_outcome_model"]] <- if (
    has_field("probit_outcome_model")
  ) {
    json_object$get_boolean("probit_outcome_model")
  } else {
    FALSE
  }

  if (
    has_subfolder_field("outcome_model", "outcome") &&
      has_subfolder_field("outcome_model", "link")
  ) {
    outcome_model_outcome <- json_object$get_string("outcome", "outcome_model")
    outcome_model_link <- json_object$get_string("link", "outcome_model")
  } else {
    outcome_model_outcome <- "continuous"
    outcome_model_link <- "identity"
    warning(paste0(
      "Fields 'outcome' and 'link' not found under 'outcome_model' in JSON (model appears to have ",
      "been serialized under stochtree ",
      .ver,
      "). Defaulting to outcome='continuous', ",
      "link='identity'. Re-save your model to suppress this warning."
    ))
  }
  model_params[["outcome_model"]] <- OutcomeModel(
    outcome = outcome_model_outcome,
    link = outcome_model_link
  )

  if (has_field("rfx_model_spec")) {
    model_params[["rfx_model_spec"]] <- json_object$get_string("rfx_model_spec")
  } else {
    model_params[["rfx_model_spec"]] <- ""
    if (model_params[["has_rfx"]]) {
      warning(paste0(
        "Field 'rfx_model_spec' not found in JSON (model appears to have been serialized under ",
        "stochtree ",
        .ver,
        "). Defaulting to ''. Re-save your model to suppress this warning."
      ))
    }
  }
  if (model_params[["outcome_model"]]$link == "cloglog") {
    cloglog_num_categories <- json_object$get_scalar("cloglog_num_categories")
    model_params[["cloglog_num_categories"]] <- cloglog_num_categories
  } else {
    model_params[["cloglog_num_categories"]] <- 0
  }

  output[["model_params"]] <- model_params

  # Unpack samples
  output[["samples"]] <- createBARTSamplesFromJson(json_object)

  # Unpack random effects group IDs
  if (model_params[["has_rfx"]]) {
    output[["rfx_unique_group_ids"]] <- resolveRfxUniqueGroupIds(
      json_object,
      output[["rfx_samples"]]$materialize_rfx()
    )
  }

  # Unpack covariate preprocessor
  if (cross_platform) {
    # Identity metadata for the cross-platform all-numeric path (gate enforced);
    # the foreign native preprocessor is not reconstructed.
    output[["train_set_metadata"]] <- buildIdentityPreprocessorMetadata(
      json_object
    )
  } else if (has_field("covariate_preprocessor")) {
    preprocessor_metadata_string <- json_object$get_string(
      "covariate_preprocessor"
    )
    output[["train_set_metadata"]] <- createPreprocessorFromJsonString(
      preprocessor_metadata_string
    )
  } else {
    output[["train_set_metadata"]] <- NULL
    warning(paste0(
      "Field 'covariate_preprocessor' not found in JSON (model appears to have been serialized ",
      "under stochtree ",
      .ver,
      "). DataFrame covariates will not be supported for prediction. ",
      "Re-save your model to suppress this warning."
    ))
  }

  class(output) <- "bartmodel"
  return(output)
}

#' @title Convert JSON File to BART Model
#' @rdname BARTSerialization
#' @param json_filename String of filepath, must end in ".json"
#' @export
createBARTModelFromJsonFile <- function(json_filename) {
  # Load a `CppJson` object from file
  bart_json <- createCppJsonFile(json_filename)

  # Create and return the BART object
  bart_object <- createBARTModelFromJson(bart_json)

  return(bart_object)
}

#' @title Convert JSON String to BART Model
#' @rdname BARTSerialization
#' @param json_string JSON string dump
#' @export
createBARTModelFromJsonString <- function(json_string) {
  # Load a `CppJson` object from string
  bart_json <- createCppJsonString(json_string)

  # Create and return the BART object
  bart_object <- createBARTModelFromJson(bart_json)

  return(bart_object)
}

#' @title Convert JSON List to Single BART Model
#' @rdname BARTSerialization
#' @param json_object_list List of objects of type `CppJson` containing Json representation of a BART model
#' @export
createBARTModelFromCombinedJson <- function(json_object_list) {
  # Initialize the BCF model
  output <- list()

  # For scalar / preprocessing details which aren't sample-dependent,
  # defer to the first json
  json_object_default <- json_object_list[[1]]

  # Helpers for optional-field presence checks
  .ver <- inferStochtreeJsonVersion(json_object_default)
  for (.jo in json_object_list) {
    resolveSchemaVersion(.jo, migrate = .migrateBartJsonV0ToV1)
  }
  cross_platform <- enforceCrossPlatformGate(json_object_default, "R")
  has_field <- function(name) {
    json_contains_field_cpp(json_object_default$json_ptr, name)
  }
  has_subfolder_field <- function(subfolder, name) {
    json_contains_field_subfolder_cpp(
      json_object_default$json_ptr,
      subfolder,
      name
    )
  }

  # Unpack model params
  include_mean_forest <- json_object_default$get_boolean(
    "include_mean_forest"
  )
  include_variance_forest <- json_object_default$get_boolean(
    "include_variance_forest"
  )
  model_params <- list()
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

  if (has_field("has_rfx_basis")) {
    model_params[["has_rfx_basis"]] <- json_object_default$get_boolean(
      "has_rfx_basis"
    )
    model_params[["num_rfx_basis"]] <- json_object_default$get_scalar(
      "num_rfx_basis"
    )
  } else {
    model_params[["has_rfx_basis"]] <- FALSE
    model_params[["num_rfx_basis"]] <- 1
    warning(paste0(
      "Fields 'has_rfx_basis' and 'num_rfx_basis' not found in JSON (model appears to have been ",
      "serialized under stochtree ",
      .ver,
      "). Defaulting to FALSE / 1. ",
      "Re-save your model to suppress this warning."
    ))
  }

  model_params[["num_covariates"]] <- if (has_field("num_covariates")) {
    json_object_default$get_scalar("num_covariates")
  } else {
    NA_real_
  }
  model_params[["num_basis"]] <- json_object_default$get_scalar("num_basis")
  model_params <- .reconstructBartLeafModelFields(model_params)
  model_params[["requires_basis"]] <- json_object_default$get_boolean(
    "requires_basis"
  )

  model_params[["probit_outcome_model"]] <- if (
    has_field("probit_outcome_model")
  ) {
    json_object_default$get_boolean("probit_outcome_model")
  } else {
    FALSE
  }

  if (
    has_subfolder_field("outcome_model", "outcome") &&
      has_subfolder_field("outcome_model", "link")
  ) {
    outcome_model_outcome <- json_object_default$get_string(
      "outcome",
      "outcome_model"
    )
    outcome_model_link <- json_object_default$get_string(
      "link",
      "outcome_model"
    )
  } else {
    outcome_model_outcome <- "continuous"
    outcome_model_link <- "identity"
    warning(paste0(
      "Fields 'outcome' and 'link' not found under 'outcome_model' in JSON (model appears to have ",
      "been serialized under stochtree ",
      .ver,
      "). Defaulting to outcome='continuous', ",
      "link='identity'. Re-save your model to suppress this warning."
    ))
  }
  model_params[["outcome_model"]] <- OutcomeModel(
    outcome = outcome_model_outcome,
    link = outcome_model_link
  )

  if (has_field("rfx_model_spec")) {
    model_params[["rfx_model_spec"]] <- json_object_default$get_string(
      "rfx_model_spec"
    )
  } else {
    model_params[["rfx_model_spec"]] <- ""
    if (model_params[["has_rfx"]]) {
      warning(paste0(
        "Field 'rfx_model_spec' not found in JSON (model appears to have been serialized under ",
        "stochtree ",
        .ver,
        "). Defaulting to ''. Re-save your model to suppress this warning."
      ))
    }
  }

  if (has_field("num_chains")) {
    model_params[["num_chains"]] <- json_object_default$get_scalar("num_chains")
  } else {
    model_params[["num_chains"]] <- 1
    warning(paste0(
      "Field 'num_chains' not found in JSON (model appears to have been serialized under stochtree ",
      .ver,
      "). Defaulting to 1. Re-save your model to suppress this warning."
    ))
  }

  if (has_field("keep_every")) {
    model_params[["keep_every"]] <- json_object_default$get_scalar("keep_every")
  } else {
    model_params[["keep_every"]] <- 1
    warning(paste0(
      "Field 'keep_every' not found in JSON (model appears to have been serialized under stochtree ",
      .ver,
      "). Defaulting to 1. Re-save your model to suppress this warning."
    ))
  }
  if (model_params[["outcome_model"]]$link == "cloglog") {
    cloglog_num_categories <- json_object_default$get_scalar(
      "cloglog_num_categories"
    )
    model_params[["cloglog_num_categories"]] <- cloglog_num_categories
  } else {
    model_params[["cloglog_num_categories"]] <- 0
  }

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

  # Unpack samples
  for (i in 1:length(json_object_list)) {
    json_object <- json_object_list[[i]]
    if (i == 1) {
      combined_samples <- createBARTSamplesFromJson(json_object)
    } else {
      additional_samples <- createBARTSamplesFromJson(json_object)
      combined_samples$merge(additional_samples)
    }
  }
  output[["samples"]] <- combined_samples

  # Unpack random effects group IDs
  if (model_params[["has_rfx"]]) {
    output[["rfx_unique_group_ids"]] <- resolveRfxUniqueGroupIds(
      json_object_default,
      output[["rfx_samples"]] ## TODO: write materialize wrapper for RFX
    )
  }

  # Unpack covariate preprocessor
  if (cross_platform) {
    # Identity metadata for the cross-platform all-numeric path (gate enforced);
    # the foreign native preprocessor is not reconstructed.
    output[["train_set_metadata"]] <- buildIdentityPreprocessorMetadata(
      json_object_default
    )
  } else if (has_field("covariate_preprocessor")) {
    preprocessor_metadata_string <- json_object_default$get_string(
      "covariate_preprocessor"
    )
    output[["train_set_metadata"]] <- createPreprocessorFromJsonString(
      preprocessor_metadata_string
    )
  } else {
    output[["train_set_metadata"]] <- NULL
    warning(paste0(
      "Field 'covariate_preprocessor' not found in JSON (model appears to have been serialized ",
      "under stochtree ",
      .ver,
      "). DataFrame covariates will not be supported for prediction. ",
      "Re-save your model to suppress this warning."
    ))
  }

  class(output) <- "bartmodel"
  return(output)
}

#' @title Convert JSON String List to Single BART Model
#' @rdname BARTSerialization
#' @param json_string_list List of JSON strings which can be parsed to objects of type `CppJson` containing Json representation of a BART model
#' @export
createBARTModelFromCombinedJsonString <- function(json_string_list) {
  # Convert JSON strings
  json_object_list <- list()
  for (i in 1:length(json_string_list)) {
    json_string <- json_string_list[[i]]
    json_object_list[[i]] <- createCppJsonString(json_string)
  }

  # Create BART model from list of JSON objects
  createBARTModelFromCombinedJson(json_object_list)
}
