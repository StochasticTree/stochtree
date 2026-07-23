#' BCF Serialization Routines
#' @name BCFSerialization
#' @description
#' BCF models contains external pointers to C++ objects, which means they cannot
#' be correctly serialized to `.Rds` from an R session in their default state.
#' These functions allow us to convert between C++ data structures and a persistent JSON
#' representation. The `CppJson` class wraps a performant C++ JSON API, and the functions
#' `saveBCFModelToJson` and `createBCFModelFromJson` save to and load from this format.
#' This representation, of course, also relies on external C++ pointers, so in order to
#' save and reload BCF models across sessions, we provide two other interfaces.
#'
#' `saveBCFModelToJsonFile` and `createBCFModelFromJsonFile` save or reload a BCF model's JSON
#' representation directly to / from a `.json` file.
#'
#' `saveBCFModelToJsonString` and `createBCFModelFromJsonString` handle in-memory strings containing JSON data,
#' which can be written to disk or passed between processes.
#'
#' Finally, for cases in which multiple BCF models have been sampled (for instance, sampled in multiple processes
#' via `doParallel`), we offer `createBCFModelFromCombinedJson` and `createBCFModelFromCombinedJsonString` for
#' loading a new combined BCF model from a list of BCF JSON objects or strings.
#' @returns
#' `saveBCFModelToJson` return an object of type `CppJson`.
#' `saveBCFModelToJsonFile` returns nothing, but writes to the provided filename.
#' `saveBCFModelToJsonString` returns a string dump of the BCF model's JSON representation.
#'
#' `createBCFModelFromJson`, `createBCFModelFromJsonFile`, `createBCFModelFromJsonString`,
#' `createBCFModelFromCombinedJson`, and `createBCFModelFromCombinedJsonString` all return
#' objects of type `bcfmodel`.
#' @examples
#' # Generate data
#' n <- 100
#' p <- 5
#' X <- matrix(runif(n*p), ncol = p)
#' pi_X <- runif(n, 0.3, 0.7)
#' Z <- rbinom(n, p = pi_X, size = 1)
#' y <- X[,1] + Z + rnorm(n, 0, 1)
#'
#' # Sample BCF model
#' bcf_model <- bcf(X_train = X, Z_train = Z, propensity_train = pi_X, y_train = y,
#'                  num_gfr = 0, num_burnin = 0, num_mcmc = 10)
#'
#' # Save to in-memory JSON
#' bcf_json <- saveBCFModelToJson(bcf_model)
#' # Save to JSON string
#' bcf_json_string <- saveBCFModelToJsonString(bcf_model)
#' # Save to JSON file
#' tmpjson <- tempfile(fileext = ".json")
#' saveBCFModelToJsonFile(bcf_model, file.path(tmpjson))
#'
#' # Reload BCF model from in-memory JSON object
#' bcf_model_roundtrip <- createBCFModelFromJson(bcf_json)
#' # Reload BCF model from JSON string
#' bcf_model_roundtrip <- createBCFModelFromJsonString(bcf_json_string)
#' # Reload BCF model from JSON file
#' bcf_model_roundtrip <- createBCFModelFromJsonFile(file.path(tmpjson))
#' unlink(tmpjson)
#' # Reload BCF model from list of JSON objects
#' bcf_model_roundtrip <- createBCFModelFromCombinedJson(list(bcf_json))
#' # Reload BCF model from list of JSON strings
#' bcf_model_roundtrip <- createBCFModelFromCombinedJsonString(list(bcf_json_string))
#'
NULL
#> NULL

#' @title Run BCF for Causal Effect Estimation
#' @description
#' Run the Bayesian Causal Forest (BCF) algorithm for regularized causal effect estimation.
#'
#' @param X_train Covariates used to split trees in the ensemble. May be provided either as a dataframe or a matrix.
#' Matrix covariates will be assumed to be all numeric. Covariates passed as a dataframe will be
#' preprocessed based on the variable types (e.g. categorical columns stored as unordered factors will be one-hot encoded,
#' categorical columns stored as ordered factors will passed as integers to the core algorithm, along with the metadata
#' that the column is ordered categorical).
#' @param Z_train Vector of (continuous or binary) treatment assignments.
#' @param y_train Outcome to be modeled by the ensemble.
#' @param propensity_train (Optional) Vector of propensity scores. If not provided, this will be estimated from the data.
#'   If `NULL` and `previous_model_json` is provided with an internally estimated propensity model, that model's
#'   propensity estimates are re-used rather than re-fitted.
#' @param rfx_group_ids_train (Optional) Group labels used for an additive random effects model.
#' @param rfx_basis_train (Optional) Basis for "random-slope" regression in an additive random effects model.
#' If `rfx_group_ids_train` is provided with a regression basis, an intercept-only random effects model
#' will be estimated.
#' @param X_test (Optional) Test set of covariates used to define "out of sample" evaluation data.
#' May be provided either as a dataframe or a matrix, but the format of `X_test` must be consistent with
#' that of `X_train`.
#' @param Z_test (Optional) Test set of (continuous or binary) treatment assignments.
#' @param propensity_test (Optional) Vector of propensity scores. If not provided, this will be estimated from the data.
#' @param rfx_group_ids_test (Optional) Test set group labels used for an additive random effects model.
#' We do not currently support (but plan to in the near future), test set evaluation for group labels
#' that were not in the training set.
#' @param rfx_basis_test (Optional) Test set basis for "random-slope" regression in additive random effects model.
#' @param observation_weights_train (Optional) Numeric vector of observation weights of length `nrow(X_train)`. Weights are
#'   applied as `y_i | - ~ N(mu(X_i), sigma^2 / w_i)`, so larger weights increase an observation's influence on the fit.
#'   All weights must be non-negative. Default: `NULL` (all observations equally weighted). Applied to both the
#'   prognostic and treatment effect forests. Compatible with Gaussian (continuous/identity) and probit outcome models;
#'   not compatible with cloglog link functions.
#' @param observation_weights Deprecated alias for `observation_weights_train`; will be removed in a future release.
#' @param num_gfr Number of "warm-start" iterations run using the grow-from-root algorithm (He and Hahn, 2021). Default: 5.
#' @param num_burnin Number of "burn-in" iterations of the MCMC sampler. Default: 0.
#' @param num_mcmc Number of "retained" iterations of the MCMC sampler. Default: 100.
#' @param previous_model_json (Optional) JSON string containing a previous BCF model. This can be used to "continue" a
#'   sampler interactively after inspecting the samples or to run parallel chains "warm-started" from existing forest
#'   samples. If the previous model used an internally estimated propensity score (i.e. `propensity_train` was not
#'   supplied to that run), the fitted propensity model is carried forward and re-used rather than being re-estimated.
#'   This ensures that multi-chain warm-starts remain consistent with the propensity scores used in the initial run.
#'   Default: `NULL`.
#' @param previous_model_warmstart_sample_num (Optional) Sample number from `previous_model_json` that will be used to warmstart this BCF sampler. One-indexed (so that the first sample is used for warm-start by setting `previous_model_warmstart_sample_num = 1`). Default: `NULL`. If `num_chains` in the `general_params` list is > 1, then each successive chain will be initialized from a different sample, counting backwards from `previous_model_warmstart_sample_num`. That is, if `previous_model_warmstart_sample_num = 10` and `num_chains = 4`, then chain 1 will be initialized from sample 10, chain 2 from sample 9, chain 3 from sample 8, and chain 4 from sample 7. If `previous_model_json` is provided but `previous_model_warmstart_sample_num` is NULL, the last sample in the previous model will be used to initialize the first chain, counting backwards as noted before. If more chains are requested than there are samples in `previous_model_json`, a warning will be raised and only the last sample will be used.
#' @param general_params (Optional) A list of general (non-forest-specific) model parameters, each of which has a default value processed internally, so this argument list is optional.
#'
#'   - `cutpoint_grid_size` Maximum size of the "grid" of potential cutpoints to consider in the GFR algorithm. Default: `100`.
#'   - `standardize` Whether or not to standardize the outcome (and store the offset / scale in the model object). Default: `TRUE`.
#'   - `sample_sigma2_global` Whether or not to update the `sigma^2` global error variance parameter based on `IG(sigma2_global_shape, sigma2_global_scale)`. Default: `TRUE`.
#'   - `sigma2_global_init` Starting value of global error variance parameter. Calibrated internally as `1.0*var((y_train-mean(y_train))/sd(y_train))` if not set.
#'   - `sigma2_global_shape` Shape parameter in the `IG(sigma2_global_shape, sigma2_global_scale)` global error variance model. Default: `0`.
#'   - `sigma2_global_scale` Scale parameter in the `IG(sigma2_global_shape, sigma2_global_scale)` global error variance model. Default: `0`.
#'   - `variable_weights` Numeric weights reflecting the relative probability of splitting on each variable. Does not need to sum to 1 but cannot be negative. Defaults to `rep(1/ncol(X_train), ncol(X_train))` if not set here. Note that if the propensity score is included as a covariate in either forest, its weight will default to `1/ncol(X_train)`. A workaround if you wish to provide a custom weight for the propensity score is to include it as a column in `X_train` and then set `propensity_covariate` to `'none'` adjust `keep_vars` accordingly for the `prognostic` or `treatment_effect` forests.
#'   - `propensity_covariate` Whether to include the propensity score as a covariate in either or both of the forests. Enter `"none"` for neither, `"prognostic"` for the prognostic forest, `"treatment_effect"` for the treatment forest, and `"both"` for both forests. If this is not `"none"` and a propensity score is not provided, it will be estimated from (`X_train`, `Z_train`) using `stochtree::bart()`. Default: `"prognostic"`.
#'   - `adaptive_coding` Whether or not to use an "adaptive coding" scheme in which a binary treatment variable is not coded manually as (0,1) or (-1,1) but learned via parameters `b_0` and `b_1` that attach to the outcome model `[b_0 (1-Z) + b_1 Z] tau(X)`. This is ignored when Z is not binary. Default: `FALSE`.
#'   - `control_coding_init` Initial value of the "control" group coding parameter. This is ignored when Z is not binary. Default: `-0.5`.
#'   - `treated_coding_init` Initial value of the "treatment" group coding parameter. This is ignored when Z is not binary. Default: `0.5`.
#'   - `rfx_prior_var` Prior on the (diagonals of the) covariance of the additive group-level random regression coefficients. Must be a vector of length `ncol(rfx_basis_train)`. Default: `rep(1, ncol(rfx_basis_train))`
#'   - `random_seed` Integer parameterizing the C++ random number generator. If not specified, the C++ random number generator is seeded according to `std::random_device`.
#'   - `keep_burnin` Whether or not "burnin" samples should be included in the stored samples of forests and other parameters. Default `FALSE`. Ignored if `num_mcmc = 0`.
#'   - `keep_gfr` Whether or not "grow-from-root" samples should be included in the stored samples of forests and other parameters. Default `FALSE`. Ignored if `num_mcmc = 0`.
#'   - `keep_every` How many iterations of the burned-in MCMC sampler should be run before forests and parameters are retained. Default `1`. Setting `keep_every <- k` for some `k > 1` will "thin" the MCMC samples by retaining every `k`-th sample, rather than simply every sample. This can reduce the autocorrelation of the MCMC samples.
#'   - `num_chains` How many independent MCMC chains should be sampled. If `num_mcmc = 0`, this is ignored. If `num_gfr = 0`, then each chain is run from root for `num_mcmc * keep_every + num_burnin` iterations, with `num_mcmc` samples retained. If `num_gfr > 0`, each MCMC chain will be initialized from a separate GFR ensemble, with the requirement that `num_gfr >= num_chains`. Default: `1`. Note that if `num_chains > 1`, the returned model object will contain samples from all chains, stored consecutively. That is, if there are 4 chains with 100 samples each, the first 100 samples will be from chain 1, the next 100 samples will be from chain 2, etc... For more detail on working with multi-chain BCF models, see [the multi chain vignette](https://stochtree.ai/vignettes/multi-chain.html).
#'   - `verbose` Whether or not to print progress during the sampling loops. Default: `FALSE`.
#'   - `outcome_model` A structured `OutcomeModel` object that specifies the outcome type and desired link function. This argument pre-empts the legacy (deprecated) `probit_outcome_model` option. Default: `OutcomeModel(outcome='continuous', link='identity')`.
#'   - `probit_outcome_model` Deprecated in favor of `outcome_model`. Whether or not the outcome should be modeled as explicitly binary via a probit link. If `TRUE`, `y` must only contain the values `0` and `1`. Default: `FALSE`.
#'   - `num_threads` Number of threads to use in the GFR and MCMC algorithms, as well as prediction. Defaults to `1` (single-threaded). Set to `-1` to use the maximum number of available threads, or a positive integer for a specific count. OpenMP must be available for values other than `1`.
#'
#' @param prognostic_forest_params (Optional) A list of prognostic forest model parameters, each of which has a default value processed internally, so this argument list is optional.
#'
#'   - `num_trees` Number of trees in the ensemble for the prognostic forest. Default: `250`. Must be a positive integer.
#'   - `alpha` Prior probability of splitting for a tree of depth 0 in the prognostic forest. Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`. Default: `0.95`.
#'   - `beta` Exponent that decreases split probabilities for nodes of depth > 0 in the prognostic forest. Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`. Default: `2`.
#'   - `min_samples_leaf` Minimum allowable size of a leaf, in terms of training samples, in the prognostic forest. Default: `5`.
#'   - `max_depth` Maximum depth of any tree in the ensemble in the prognostic forest. Default: `10`. Can be overridden with ``-1`` which does not enforce any depth limits on trees.
#'   - `variable_weights` Numeric weights reflecting the relative probability of splitting on each variable in the prognostic forest. Does not need to sum to 1 but cannot be negative. Defaults to `rep(1/ncol(X_train), ncol(X_train))` if not set here.
#'   - `sample_sigma2_leaf` Whether or not to update the leaf scale variance parameter based on `IG(sigma2_leaf_shape, sigma2_leaf_scale)`.
#'   - `sigma2_leaf_init` Starting value of leaf node scale parameter. Calibrated internally as `1/num_trees` if not set here.
#'   - `sigma2_leaf_shape` Shape parameter in the `IG(sigma2_leaf_shape, sigma2_leaf_scale)` leaf node parameter variance model. Default: `3`.
#'   - `sigma2_leaf_scale` Scale parameter in the `IG(sigma2_leaf_shape, sigma2_leaf_scale)` leaf node parameter variance model. Calibrated internally as `0.5/num_trees` if not set here.
#'   - `keep_vars` Vector of variable names or column indices denoting variables that should be included in the forest. Default: `NULL`.
#'   - `drop_vars` Vector of variable names or column indices denoting variables that should be excluded from the forest. Default: `NULL`. If both `drop_vars` and `keep_vars` are set, `drop_vars` will be ignored.
#'   - `num_features_subsample` How many features to subsample when growing each tree for the GFR algorithm. Defaults to the number of features in the training dataset.
#'
#' @param treatment_effect_forest_params (Optional) A list of treatment effect forest model parameters, each of which has a default value processed internally, so this argument list is optional.
#'
#'   - `num_trees` Number of trees in the ensemble for the treatment effect forest. Default: `100`. Must be a positive integer.
#'   - `alpha` Prior probability of splitting for a tree of depth 0 in the treatment effect forest. Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`. Default: `0.25`.
#'   - `beta` Exponent that decreases split probabilities for nodes of depth > 0 in the treatment effect forest. Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`. Default: `3`.
#'   - `min_samples_leaf` Minimum allowable size of a leaf, in terms of training samples, in the treatment effect forest. Default: `5`.
#'   - `max_depth` Maximum depth of any tree in the ensemble in the treatment effect forest. Default: `5`. Can be overridden with ``-1`` which does not enforce any depth limits on trees.
#'   - `variable_weights` Numeric weights reflecting the relative probability of splitting on each variable in the treatment effect forest. Does not need to sum to 1 but cannot be negative. Defaults to `rep(1/ncol(X_train), ncol(X_train))` if not set here.
#'   - `sample_sigma2_leaf` Whether or not to update the leaf scale variance parameter based on `IG(sigma2_leaf_shape, sigma2_leaf_scale)`. Cannot (currently) be set to true if `ncol(Z_train)>1`. Default: `FALSE`.
#'   - `sigma2_leaf_init` Starting value of leaf node scale parameter. Calibrated internally as `0.5 * var(y)/num_trees` if not set here (`0.5 / num_trees` if `y` is continuous and `standardize = TRUE` in the `general_params` list).
#'   - `sigma2_leaf_shape` Shape parameter in the `IG(sigma2_leaf_shape, sigma2_leaf_scale)` leaf node parameter variance model. Default: `3`.
#'   - `sigma2_leaf_scale` Scale parameter in the `IG(sigma2_leaf_shape, sigma2_leaf_scale)` leaf node parameter variance model. Calibrated internally as `0.5/num_trees` if not set here.
#'   - `delta_max` Maximum plausible conditional distributional treatment effect (i.e. P(Y(1) = 1 | X) - P(Y(0) = 1 | X)) when the outcome is binary. Only used when the outcome is specified as a probit model in `general_params`. Must be > 0 and < 1. Default: `0.9`. Ignored if `sigma2_leaf_init` is set directly, as this parameter is used to calibrate `sigma2_leaf_init`.
#'   - `keep_vars` Vector of variable names or column indices denoting variables that should be included in the forest. Default: `NULL`.
#'   - `drop_vars` Vector of variable names or column indices denoting variables that should be excluded from the forest. Default: `NULL`. If both `drop_vars` and `keep_vars` are set, `drop_vars` will be ignored.
#'   - `num_features_subsample` How many features to subsample when growing each tree for the GFR algorithm. Defaults to the number of features in the training dataset.
#'   - `sample_intercept` Whether to sample a global treatment effect intercept `tau_0` so the full CATE is `tau_0 + tau(X)`. Default: `TRUE`. Compatible with `adaptive_coding = TRUE`, in which case the recoded treatment basis is used.
#'   - `tau_0_prior_var` Variance of the normal prior on `tau_0` (a scalar applied to each treatment dimension independently). Auto-calibrated to outcome variance when `NULL` and outcome is continuous. Only used when `sample_intercept = TRUE`.
#'
#' @param variance_forest_params (Optional) A list of variance forest model parameters, each of which has a default value processed internally, so this argument list is optional.
#'
#'   - `num_trees` Number of trees in the ensemble for the conditional variance model. Default: `0`. Variance is only modeled using a tree / forest if `num_trees > 0`.
#'   - `alpha` Prior probability of splitting for a tree of depth 0 in the variance model. Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`. Default: `0.95`.
#'   - `beta` Exponent that decreases split probabilities for nodes of depth > 0 in the variance model. Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`. Default: `2`.
#'   - `min_samples_leaf` Minimum allowable size of a leaf, in terms of training samples, in the variance model. Default: `5`.
#'   - `max_depth` Maximum depth of any tree in the ensemble in the variance model. Default: `10`. Can be overridden with ``-1`` which does not enforce any depth limits on trees.
#'   - `leaf_prior_calibration_param` Hyperparameter used to calibrate the `IG(var_forest_prior_shape, var_forest_prior_scale)` conditional error variance model. If `var_forest_prior_shape` and `var_forest_prior_scale` are not set below, this calibration parameter is used to set these values to `num_trees / leaf_prior_calibration_param^2 + 0.5` and `num_trees / leaf_prior_calibration_param^2`, respectively. Default: `1.5`.
#'   - `variance_forest_init` Starting value of root forest prediction in conditional (heteroskedastic) error variance model. Calibrated internally as `log(0.6*var((y_train-mean(y_train))/sd(y_train)))/num_trees` if not set.
#'   - `var_forest_prior_shape` Shape parameter in the `IG(var_forest_prior_shape, var_forest_prior_scale)` conditional error variance model (which is only sampled if `num_trees > 0`). Calibrated internally as `num_trees / 1.5^2 + 0.5` if not set.
#'   - `var_forest_prior_scale` Scale parameter in the `IG(var_forest_prior_shape, var_forest_prior_scale)` conditional error variance model (which is only sampled if `num_trees > 0`). Calibrated internally as `num_trees / 1.5^2` if not set.
#'   - `keep_vars` Vector of variable names or column indices denoting variables that should be included in the forest. Default: `NULL`.
#'   - `drop_vars` Vector of variable names or column indices denoting variables that should be excluded from the forest. Default: `NULL`. If both `drop_vars` and `keep_vars` are set, `drop_vars` will be ignored.
#'   - `num_features_subsample` How many features to subsample when growing each tree for the GFR algorithm. Defaults to the number of features in the training dataset.
#'
#' @param random_effects_params (Optional) A list of random effects model parameters, each of which has a default value processed internally, so this argument list is optional.
#'
#'   - `model_spec` Specification of the random effects model. Options are "custom", "intercept_only", and "intercept_plus_treatment". If "custom" is specified, then a user-provided basis must be passed through `rfx_basis_train`. If "intercept_only" is specified, a random effects basis of all ones will be dispatched internally at sampling and prediction time. If "intercept_plus_treatment" is specified, a random effects basis that combines an "intercept" basis of all ones with the treatment variable (`Z_train`) will be dispatched internally at sampling and prediction time. Default: "custom". If either "intercept_only" or "intercept_plus_treatment" is specified, `rfx_basis_train` and `rfx_basis_test` (if provided) will be ignored.
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
#' n <- 500
#' p <- 5
#' X <- matrix(runif(n*p), ncol = p)
#' mu_x <- (
#'     ((0 <= X[,1]) & (0.25 > X[,1])) * (-7.5) +
#'     ((0.25 <= X[,1]) & (0.5 > X[,1])) * (-2.5) +
#'     ((0.5 <= X[,1]) & (0.75 > X[,1])) * (2.5) +
#'     ((0.75 <= X[,1]) & (1 > X[,1])) * (7.5)
#' )
#' pi_x <- (
#'     ((0 <= X[,1]) & (0.25 > X[,1])) * (0.2) +
#'     ((0.25 <= X[,1]) & (0.5 > X[,1])) * (0.4) +
#'     ((0.5 <= X[,1]) & (0.75 > X[,1])) * (0.6) +
#'     ((0.75 <= X[,1]) & (1 > X[,1])) * (0.8)
#' )
#' tau_x <- (
#'     ((0 <= X[,2]) & (0.25 > X[,2])) * (0.5) +
#'     ((0.25 <= X[,2]) & (0.5 > X[,2])) * (1.0) +
#'     ((0.5 <= X[,2]) & (0.75 > X[,2])) * (1.5) +
#'     ((0.75 <= X[,2]) & (1 > X[,2])) * (2.0)
#' )
#' Z <- rbinom(n, 1, pi_x)
#' noise_sd <- 1
#' y <- mu_x + tau_x*Z + rnorm(n, 0, noise_sd)
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
#' bcf_model <- bcf(X_train = X_train, Z_train = Z_train, y_train = y_train,
#'                  propensity_train = pi_train, X_test = X_test, Z_test = Z_test,
#'                  propensity_test = pi_test, num_gfr = 10,
#'                  num_burnin = 0, num_mcmc = 10)
bcf <- function(
  X_train,
  Z_train,
  y_train,
  propensity_train = NULL,
  rfx_group_ids_train = NULL,
  rfx_basis_train = NULL,
  X_test = NULL,
  Z_test = NULL,
  propensity_test = NULL,
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
  prognostic_forest_params = list(),
  treatment_effect_forest_params = list(),
  variance_forest_params = list(),
  random_effects_params = list()
) {
  # Update general BCF parameters
  general_params_default <- list(
    cutpoint_grid_size = 100,
    standardize = TRUE,
    sample_sigma2_global = TRUE,
    sigma2_global_init = NULL,
    sigma2_global_shape = 0,
    sigma2_global_scale = 0,
    variable_weights = NULL,
    propensity_covariate = "prognostic",
    adaptive_coding = FALSE,
    control_coding_init = -0.5,
    treated_coding_init = 0.5,
    rfx_prior_var = NULL,
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

  # Update mu forest BCF parameters
  prognostic_forest_params_default <- list(
    num_trees = 250,
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
  prognostic_forest_params_updated <- preprocessParams(
    prognostic_forest_params_default,
    prognostic_forest_params
  )

  # Update tau forest BCF parameters
  treatment_effect_forest_params_default <- list(
    num_trees = 100,
    alpha = 0.25,
    beta = 3.0,
    min_samples_leaf = 5,
    max_depth = 5,
    sample_sigma2_leaf = FALSE,
    sigma2_leaf_init = NULL,
    sigma2_leaf_shape = 3,
    sigma2_leaf_scale = NULL,
    keep_vars = NULL,
    drop_vars = NULL,
    delta_max = 0.9,
    num_features_subsample = NULL,
    sample_intercept = TRUE,
    tau_0_prior_var = NULL
  )
  treatment_effect_forest_params_updated <- preprocessParams(
    treatment_effect_forest_params_default,
    treatment_effect_forest_params
  )

  # Update variance forest BCF parameters
  variance_forest_params_default <- list(
    num_trees = 0,
    alpha = 0.95,
    beta = 2.0,
    min_samples_leaf = 5,
    max_depth = 10,
    leaf_prior_calibration_param = 1.5,
    variance_forest_init = NULL,
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

  # Update random effects parameters
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
  propensity_covariate <- general_params_updated$propensity_covariate
  adaptive_coding <- general_params_updated$adaptive_coding
  b_0 <- general_params_updated$control_coding_init
  b_1 <- general_params_updated$treated_coding_init
  rfx_prior_var <- general_params_updated$rfx_prior_var
  random_seed <- general_params_updated$random_seed
  keep_burnin <- general_params_updated$keep_burnin
  keep_gfr <- general_params_updated$keep_gfr
  keep_every <- general_params_updated$keep_every
  num_chains <- general_params_updated$num_chains
  verbose <- general_params_updated$verbose
  outcome_model <- general_params_updated$outcome_model
  probit_outcome_model <- general_params_updated$probit_outcome_model
  num_threads <- general_params_updated$num_threads

  # 2. Mu forest parameters
  num_trees_mu <- prognostic_forest_params_updated$num_trees
  alpha_mu <- prognostic_forest_params_updated$alpha
  beta_mu <- prognostic_forest_params_updated$beta
  min_samples_leaf_mu <- prognostic_forest_params_updated$min_samples_leaf
  max_depth_mu <- prognostic_forest_params_updated$max_depth
  sample_sigma2_leaf_mu <- prognostic_forest_params_updated$sample_sigma2_leaf
  sigma2_leaf_mu <- prognostic_forest_params_updated$sigma2_leaf_init
  a_leaf_mu <- prognostic_forest_params_updated$sigma2_leaf_shape
  b_leaf_mu <- prognostic_forest_params_updated$sigma2_leaf_scale
  keep_vars_mu <- prognostic_forest_params_updated$keep_vars
  drop_vars_mu <- prognostic_forest_params_updated$drop_vars
  num_features_subsample_mu <- prognostic_forest_params_updated$num_features_subsample

  # 3. Tau forest parameters
  num_trees_tau <- treatment_effect_forest_params_updated$num_trees
  alpha_tau <- treatment_effect_forest_params_updated$alpha
  beta_tau <- treatment_effect_forest_params_updated$beta
  min_samples_leaf_tau <- treatment_effect_forest_params_updated$min_samples_leaf
  max_depth_tau <- treatment_effect_forest_params_updated$max_depth
  sample_sigma2_leaf_tau <- treatment_effect_forest_params_updated$sample_sigma2_leaf
  sigma2_leaf_tau <- treatment_effect_forest_params_updated$sigma2_leaf_init
  a_leaf_tau <- treatment_effect_forest_params_updated$sigma2_leaf_shape
  b_leaf_tau <- treatment_effect_forest_params_updated$sigma2_leaf_scale
  keep_vars_tau <- treatment_effect_forest_params_updated$keep_vars
  drop_vars_tau <- treatment_effect_forest_params_updated$drop_vars
  delta_max <- treatment_effect_forest_params_updated$delta_max
  num_features_subsample_tau <- treatment_effect_forest_params_updated$num_features_subsample
  sample_tau_0 <- treatment_effect_forest_params_updated$sample_intercept
  tau_0_prior_var <- treatment_effect_forest_params_updated$tau_0_prior_var

  # 4. Variance forest parameters
  num_trees_variance <- variance_forest_params_updated$num_trees
  alpha_variance <- variance_forest_params_updated$alpha
  beta_variance <- variance_forest_params_updated$beta
  min_samples_leaf_variance <- variance_forest_params_updated$min_samples_leaf
  max_depth_variance <- variance_forest_params_updated$max_depth
  a_0 <- variance_forest_params_updated$leaf_prior_calibration_param
  variance_forest_init <- variance_forest_params_updated$variance_forest_init
  a_forest <- variance_forest_params_updated$var_forest_prior_shape
  b_forest <- variance_forest_params_updated$var_forest_prior_scale
  keep_vars_variance <- variance_forest_params_updated$keep_vars
  drop_vars_variance <- variance_forest_params_updated$drop_vars
  num_features_subsample_variance <- variance_forest_params_updated$num_features_subsample

  # 5. Random effects parameters
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

  # Handle random effects specification
  if (!is.character(rfx_model_spec)) {
    stop("rfx_model_spec must be a string or character vector")
  }
  if (
    !(rfx_model_spec %in%
      c("custom", "intercept_only", "intercept_plus_treatment"))
  ) {
    stop(
      "rfx_model_spec must either be 'custom', 'intercept_only', or 'intercept_plus_treatment'"
    )
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
  # Warm-start from a serialized previous model. The previous model is deserialized here via the
  # version/platform-aware constructor; its in-memory BCFSamples pointer is passed to the C++ sampler,
  # which reconstitutes the active forests (+ scalars / tau_0 / adaptive-coding / rfx) from the requested
  # retained sample. Same-scale is assumed. The feature-space compatibility guard runs below, after the
  # new run's covariates are preprocessed.
  has_prev_model <- !is.null(previous_model_json)
  has_prev_model_index <- !is.null(previous_model_warmstart_sample_num)
  previous_bcf_model <- NULL
  if (has_prev_model) {
    previous_bcf_model <- createBCFModelFromJsonString(previous_model_json)
    prev_num_samples <- previous_bcf_model$model_params$num_samples
    if (!has_prev_model_index) {
      previous_model_warmstart_sample_num <- prev_num_samples
      warning(
        "`previous_model_warmstart_sample_num` was not provided alongside `previous_model_json`, so it will be set to the number of samples available in `previous_model_json`"
      )
    } else {
      if (previous_model_warmstart_sample_num < 1) {
        stop("`previous_model_warmstart_sample_num` must be a positive integer")
      }
      if (previous_model_warmstart_sample_num > prev_num_samples) {
        stop("`previous_model_warmstart_sample_num` exceeds the number of samples in `previous_model_json`")
      }
    }
    # Multi-chain warm-start: each chain past the first is seeded from a successively earlier sample
    # of the previous model (counting backwards from previous_model_warmstart_sample_num). If more
    # chains are requested than there are samples at/below that index, the C++ sampler clamps to the
    # earliest available sample; warn so the user knows chains will share a starting point.
    if (num_chains > previous_model_warmstart_sample_num) {
      warning(sprintf(
        "`num_chains` (%d) exceeds `previous_model_warmstart_sample_num` (%d); chains beyond sample 1 will all be warm-started from the earliest available sample.",
        num_chains, previous_model_warmstart_sample_num
      ))
    }
  }

  # Determine whether conditional variance will be modeled
  if (num_trees_variance > 0) {
    include_variance_forest <- TRUE
  } else {
    include_variance_forest <- FALSE
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

  # observation_weights_train validation and compatibility checks
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

  # Variable weight preprocessing (and initialization if necessary)
  if (is.null(variable_weights)) {
    variable_weights <- rep(1 / ncol(X_train), ncol(X_train))
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

  # Check delta_max is valid
  if ((delta_max <= 0) || (delta_max >= 1)) {
    stop("delta_max must be > 0 and < 1")
  }

  # Standardize the keep variable lists to numeric indices
  if (!is.null(keep_vars_mu)) {
    if (is.character(keep_vars_mu)) {
      if (!all(keep_vars_mu %in% names(X_train))) {
        stop(
          "keep_vars_mu includes some variable names that are not in X_train"
        )
      }
      variable_subset_mu <- unname(which(
        names(X_train) %in% keep_vars_mu
      ))
    } else {
      if (any(keep_vars_mu > ncol(X_train))) {
        stop(
          "keep_vars_mu includes some variable indices that exceed the number of columns in X_train"
        )
      }
      if (any(keep_vars_mu < 0)) {
        stop("keep_vars_mu includes some negative variable indices")
      }
      variable_subset_mu <- keep_vars_mu
    }
  } else if ((is.null(keep_vars_mu)) && (!is.null(drop_vars_mu))) {
    if (is.character(drop_vars_mu)) {
      if (!all(drop_vars_mu %in% names(X_train))) {
        stop(
          "drop_vars_mu includes some variable names that are not in X_train"
        )
      }
      variable_subset_mu <- unname(which(
        !(names(X_train) %in% drop_vars_mu)
      ))
    } else {
      if (any(drop_vars_mu > ncol(X_train))) {
        stop(
          "drop_vars_mu includes some variable indices that exceed the number of columns in X_train"
        )
      }
      if (any(drop_vars_mu < 0)) {
        stop("drop_vars_mu includes some negative variable indices")
      }
      variable_subset_mu <- (1:ncol(X_train))[
        !(1:ncol(X_train) %in% drop_vars_mu)
      ]
    }
  } else {
    variable_subset_mu <- 1:ncol(X_train)
  }
  if (!is.null(keep_vars_tau)) {
    if (is.character(keep_vars_tau)) {
      if (!all(keep_vars_tau %in% names(X_train))) {
        stop(
          "keep_vars_tau includes some variable names that are not in X_train"
        )
      }
      variable_subset_tau <- unname(which(
        names(X_train) %in% keep_vars_tau
      ))
    } else {
      if (any(keep_vars_tau > ncol(X_train))) {
        stop(
          "keep_vars_tau includes some variable indices that exceed the number of columns in X_train"
        )
      }
      if (any(keep_vars_tau < 0)) {
        stop("keep_vars_tau includes some negative variable indices")
      }
      variable_subset_tau <- keep_vars_tau
    }
  } else if ((is.null(keep_vars_tau)) && (!is.null(drop_vars_tau))) {
    if (is.character(drop_vars_tau)) {
      if (!all(drop_vars_tau %in% names(X_train))) {
        stop(
          "drop_vars_tau includes some variable names that are not in X_train"
        )
      }
      variable_subset_tau <- unname(which(
        !(names(X_train) %in% drop_vars_tau)
      ))
    } else {
      if (any(drop_vars_tau > ncol(X_train))) {
        stop(
          "drop_vars_tau includes some variable indices that exceed the number of columns in X_train"
        )
      }
      if (any(drop_vars_tau < 0)) {
        stop("drop_vars_tau includes some negative variable indices")
      }
      variable_subset_tau <- (1:ncol(X_train))[
        !(1:ncol(X_train) %in% drop_vars_tau)
      ]
    }
  } else {
    variable_subset_tau <- 1:ncol(X_train)
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
  if (ncol(X_train) != length(variable_weights)) {
    stop("length(variable_weights) must equal ncol(X_train)")
  }
  train_cov_preprocess_list <- preprocessTrainData(X_train)
  X_train_metadata <- train_cov_preprocess_list$metadata
  X_train_raw <- X_train
  X_train <- train_cov_preprocess_list$data
  original_var_indices <- X_train_metadata$original_var_indices
  feature_types <- X_train_metadata$feature_types
  X_test_raw <- X_test

  # Feature-space compatibility guard for a previous-model warm-start: the previous model's forests
  # split on preprocessed feature indices, so the new run must produce the same preprocessed layout
  # (feature count + per-feature original-variable mapping), or those split indices would point at the
  # wrong features. `original_var_indices` encodes the expansion mapping (before the propensity column
  # is appended, which both models append identically), so comparing it catches both a covariate-count
  # mismatch and a categorical-expansion mismatch.
  if (has_prev_model) {
    prev_var_indices <- previous_bcf_model$train_set_metadata$original_var_indices
    if (!identical(as.integer(prev_var_indices), as.integer(original_var_indices))) {
      stop(
        "`previous_model_json` was fit on a different covariate structure than the current data ",
        "(preprocessed feature layout does not match). Warm-start requires the same covariates, ",
        "types, and categorical levels."
      )
    }
  }
  if (!is.null(X_test)) {
    X_test <- preprocessPredictionData(X_test, X_train_metadata)
  }

  # Handle factor-valued treatment vectors before any numeric conversion.
  # as.numeric() on a factor returns level indices (1, 2, ...), not 0/1, so
  # factors must be explicitly converted to 0/1 first.
  if (is.factor(Z_train)) {
    lvls <- levels(Z_train)
    if (length(lvls) != 2) {
      stop("Factor Z_train must have exactly 2 levels for binary treatment")
    }
    message(
      "Z_train is a factor; converting to 0/1 using level order: ",
      lvls[1],
      " = 0, ",
      lvls[2],
      " = 1"
    )
    Z_train <- as.integer(Z_train) - 1L
  }
  if (!is.null(Z_test) && is.factor(Z_test)) {
    lvls <- levels(Z_test)
    if (length(lvls) != 2) {
      stop("Factor Z_test must have exactly 2 levels for binary treatment")
    }
    message(
      "Z_test is a factor; converting to 0/1 using level order: ",
      lvls[1],
      " = 0, ",
      lvls[2],
      " = 1"
    )
    Z_test <- as.integer(Z_test) - 1L
  }

  # Check that all inputs are numeric before matrix conversions
  if (!is.numeric(y_train)) {
    stop("y_train must be numeric")
  }
  if (!is.numeric(Z_train)) {
    stop("Z_train must be numeric")
  }
  if (!is.null(Z_test) && !is.numeric(Z_test)) {
    stop("Z_test must be numeric")
  }
  if (!is.null(propensity_train) && !is.numeric(propensity_train)) {
    stop("propensity_train must be numeric")
  }
  if (!is.null(propensity_test) && !is.numeric(propensity_test)) {
    stop("propensity_test must be numeric")
  }
  if (!is.null(rfx_basis_train) && !is.numeric(rfx_basis_train)) {
    stop("rfx_basis_train must be numeric")
  }
  if (!is.null(rfx_basis_test) && !is.numeric(rfx_basis_test)) {
    stop("rfx_basis_test must be numeric")
  }

  # Convert all input data to matrices if not already converted
  Z_col <- ifelse(is.null(dim(Z_train)), 1, ncol(Z_train))
  Z_train <- matrix(as.numeric(Z_train), ncol = Z_col)
  if ((is.null(dim(propensity_train))) && (!is.null(propensity_train))) {
    propensity_train <- as.matrix(propensity_train)
  }
  if (!is.null(Z_test)) {
    Z_test <- matrix(as.numeric(Z_test), ncol = Z_col)
  }
  if ((is.null(dim(propensity_test))) && (!is.null(propensity_test))) {
    propensity_test <- as.matrix(propensity_test)
  }
  if ((is.null(dim(rfx_basis_train))) && (!is.null(rfx_basis_train))) {
    rfx_basis_train <- as.matrix(rfx_basis_train)
  }
  if ((is.null(dim(rfx_basis_test))) && (!is.null(rfx_basis_test))) {
    rfx_basis_test <- as.matrix(rfx_basis_test)
  }

  # Convert y_train to a vector if passed as a one-column matrix
  if (is.matrix(y_train)) {
    if (ncol(y_train) > 1) {
      stop("y_train must be a numeric vector or a one-column matrix")
    }
    y_train <- as.numeric(y_train)
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
  if ((!is.null(Z_test)) && (ncol(Z_test) != ncol(Z_train))) {
    stop("Z_train and Z_test must have the same number of columns")
  }
  if ((!is.null(Z_train)) && (nrow(Z_train) != nrow(X_train))) {
    stop("Z_train and X_train must have the same number of rows")
  }
  if (
    (!is.null(propensity_train)) &&
      (nrow(propensity_train) != nrow(X_train))
  ) {
    stop("propensity_train and X_train must have the same number of rows")
  }
  if ((!is.null(Z_test)) && (nrow(Z_test) != nrow(X_test))) {
    stop("Z_test and X_test must have the same number of rows")
  }
  if ((!is.null(propensity_test)) && (nrow(propensity_test) != nrow(X_test))) {
    stop("propensity_test and X_test must have the same number of rows")
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

  # # Stop if multivariate treatment is provided
  # if (ncol(Z_train) > 1) stop("Multivariate treatments are not currently supported")

  # Handle multivariate treatment
  has_multivariate_treatment <- ncol(Z_train) > 1
  if (has_multivariate_treatment) {
    # Disable internal propensity model and leaf scale sampling if treatment is multivariate
    if (is.null(propensity_train)) {
      if (propensity_covariate != "none") {
        warning(
          "No propensities were provided for the multivariate treatment; an internal propensity model will not be fitted to the multivariate treatment and propensity_covariate will be set to 'none'"
        )
        propensity_covariate <- "none"
      }
    }
    if (sample_sigma2_leaf_tau) {
      warning(
        "Sampling leaf scale not yet supported for multivariate leaf models, so the leaf scale parameter will not be sampled for the treatment forest in this model."
      )
      sample_sigma2_leaf_tau <- FALSE
    }
  }

  # Preserve the raw per-original-variable weights (used below to derive the per-forest weights and
  # cached in continuation_state so continueSampling.bcfmodel() can re-derive them if the user
  # changes variable_weights / keep_vars / drop_vars).
  variable_weights_raw <- variable_weights

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
    } else if (rfx_model_spec == "intercept_plus_treatment") {
      if (has_multivariate_treatment) {
        warning(
          "Random effects `intercept_plus_treatment` specification is not currently implemented for multivariate treatments. This model will be fit under the `intercept_only` specification instead. Please provide a custom `rfx_basis_train` if you wish to have random slopes on multivariate treatment variables."
        )
        rfx_model_spec <- "intercept_only"
      }
    }
    if (is.null(rfx_basis_train)) {
      if (rfx_model_spec == "intercept_only") {
        rfx_basis_train <- matrix(
          rep(1, nrow(X_train)),
          nrow = nrow(X_train),
          ncol = 1
        )
        has_basis_rfx <- TRUE
        num_basis_rfx <- 1
      } else {
        rfx_basis_train <- cbind(
          rep(1, nrow(X_train)),
          Z_train
        )
        has_basis_rfx <- TRUE
        num_basis_rfx <- 1 + ncol(Z_train)
      }
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
    } else if (rfx_model_spec == "intercept_plus_treatment") {
      rfx_basis_test <- cbind(
        rep(1, nrow(X_test)),
        Z_test
      )
    }
  }

  # Random effects covariance prior
  if (has_rfx) {
    if (is.null(rfx_prior_var)) {
      rfx_prior_var <- rep(1, ncol(rfx_basis_train))
    } else {
      if ((!is.integer(rfx_prior_var)) && (!is.numeric(rfx_prior_var))) {
        stop("rfx_prior_var must be a numeric vector")
      }
      if (length(rfx_prior_var) != ncol(rfx_basis_train)) {
        stop("length(rfx_prior_var) must equal ncol(rfx_basis_train)")
      }
    }
  }

  # Check that number of samples are all nonnegative
  stopifnot(num_gfr >= 0)
  stopifnot(num_burnin >= 0)
  stopifnot(num_mcmc >= 0)

  # Determine whether a test set is provided
  has_test <- !is.null(X_test)

  # Convert y_train to numeric vector if not already converted
  if (!is.null(dim(y_train))) {
    y_train <- as.matrix(y_train)
  }

  # Check whether treatment is binary and univariate (specifically 0-1 binary)
  binary_treatment <- FALSE
  if (!has_multivariate_treatment) {
    binary_treatment <- length(unique(Z_train)) == 2
    if (binary_treatment) {
      unique_treatments <- sort(unique(Z_train))
      if (!(all(unique_treatments == c(0, 1)))) binary_treatment <- FALSE
    }
  }

  # Adaptive coding will be ignored for continuous / ordered categorical treatments
  if ((!binary_treatment) && (adaptive_coding)) {
    warning(
      "Adaptive coding is only compatible with binary (univariate) treatment and, as a result, will be ignored in sampling this model"
    )
    adaptive_coding <- FALSE
  }

  # Validate tau_0_prior_var if sample_tau_0 is TRUE
  if (sample_tau_0 && !is.null(tau_0_prior_var)) {
    if (
      !is.numeric(tau_0_prior_var) ||
        length(tau_0_prior_var) != 1 ||
        tau_0_prior_var <= 0
    ) {
      stop("tau_0_prior_var must be a single positive numeric value")
    }
  }

  # Check if propensity_covariate is one of the required inputs
  if (
    !(propensity_covariate %in%
      c("prognostic", "treatment_effect", "both", "none"))
  ) {
    stop(
      "propensity_covariate must equal one of 'none', 'prognostic', 'treatment_effect', or 'both'"
    )
  }

  # Estimate if pre-estimated propensity score is not provided
  internal_propensity_model <- FALSE
  if ((is.null(propensity_train)) && (propensity_covariate != "none")) {
    internal_propensity_model <- TRUE
    if (
      has_prev_model &&
        previous_bcf_model$model_params$internal_propensity_model
    ) {
      # Reuse the propensity model from the warm-started BCF model rather than
      # re-fitting from scratch. Training propensities come from the previous
      # model's stored predictions; test propensities are re-predicted on the
      # (potentially new) test set.
      bart_model_propensity <- previous_bcf_model$bart_propensity_model
      propensity_train <- predict(
        bart_model_propensity,
        X = X_train,
        terms = "y_hat",
        type = "mean"
      )
      if ((is.null(dim(propensity_train))) && (!is.null(propensity_train))) {
        propensity_train <- as.matrix(propensity_train)
      }
      if (has_test) {
        propensity_test <- predict(
          bart_model_propensity,
          X = X_test,
          terms = "y_hat",
          type = "mean"
        )
        if ((is.null(dim(propensity_test))) && (!is.null(propensity_test))) {
          propensity_test <- as.matrix(propensity_test)
        }
      }
    } else {
      # Estimate using the last of several iterations of GFR BART
      num_gfr_propensity <- 10
      num_burnin_propensity <- 0
      num_mcmc_propensity <- 10
      bart_model_propensity <- bart(
        X_train = X_train,
        y_train = as.numeric(Z_train),
        X_test = X_test,
        num_gfr = num_gfr_propensity,
        num_burnin = num_burnin_propensity,
        num_mcmc = num_mcmc_propensity,
        general_params = list(random_seed = random_seed)
      )
      propensity_train <- predict(
        bart_model_propensity,
        X = X_train,
        terms = "y_hat",
        type = "mean"
      )
      if ((is.null(dim(propensity_train))) && (!is.null(propensity_train))) {
        propensity_train <- as.matrix(propensity_train)
      }
      if (has_test) {
        propensity_test <- predict(
          bart_model_propensity,
          X = X_test,
          terms = "y_hat",
          type = "mean"
        )
        if ((is.null(dim(propensity_test))) && (!is.null(propensity_test))) {
          propensity_test <- as.matrix(propensity_test)
        }
      }
    }
  }

  if (has_test && !is.null(propensity_train)) {
    if (is.null(propensity_test)) {
      stop(
        "Propensity score must be provided for the test set if provided for the training set"
      )
    }
  }

  # Update feature_types and covariates
  feature_types <- as.integer(feature_types)
  ncol_propensity <- if (propensity_covariate != "none") {
    ncol(propensity_train)
  } else {
    0
  }
  if (propensity_covariate != "none") {
    feature_types <- as.integer(c(
      feature_types,
      rep(0, ncol_propensity)
    ))
    X_train <- cbind(X_train, propensity_train)
    if (has_test) X_test <- cbind(X_test, propensity_test)
  }

  # Derive the per-forest split-variable weights (expand across preprocessed features, append
  # propensity-column weights, renormalize). Shared with continueSampling.bcfmodel() via the helper.
  variable_weights_list <- computeBCFForestWeights(
    variable_weights = variable_weights_raw,
    original_var_indices = original_var_indices,
    variable_subset_mu = variable_subset_mu,
    variable_subset_tau = variable_subset_tau,
    variable_subset_variance = variable_subset_variance,
    propensity_covariate = propensity_covariate,
    ncol_propensity = ncol_propensity,
    num_cov_orig = num_cov_orig,
    include_variance_forest = include_variance_forest
  )
  variable_weights_mu <- variable_weights_list$mu
  variable_weights_tau <- variable_weights_list$tau
  variable_weights_variance <- variable_weights_list$variance

  # Set num_features_subsample to default, ncol(X_train), if not already set
  if (is.null(num_features_subsample_mu)) {
    num_features_subsample_mu <- ncol(X_train)
  }
  if (is.null(num_features_subsample_tau)) {
    num_features_subsample_tau <- ncol(X_train)
  }
  if (is.null(num_features_subsample_variance)) {
    num_features_subsample_variance <- ncol(X_train)
  }

  # Preliminary runtime checks for probit link
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
    # Calibrate the treatment effect leaf scale from delta_max when the user has not
    # set sigma2_leaf_init directly for the treatment forest.
    if (is.null(sigma2_leaf_tau)) {
      # Prior calibrated so that P(abs(tau(X)) < delta_max / dnorm(0)) = p (p = 0.6827)
      p <- 0.6827
      q_quantile <- qnorm((p + 1) / 2)
      sigma2_leaf_tau_scalar <- ((delta_max / (q_quantile * dnorm(0)))^2) /
        num_trees_tau
      if (has_multivariate_treatment) {
        sigma2_leaf_tau <- diag(sigma2_leaf_tau_scalar, ncol(Z_train))
      } else {
        sigma2_leaf_tau <- sigma2_leaf_tau_scalar
      }
    }
  }

  # Runtime checks for variance forest
  if (include_variance_forest) {
    if (sample_sigma2_global) {
      warning(
        "Global error variance will not be sampled with a heteroskedasticity"
      )
      sample_sigma2_global <- F
    }
  }

  # Set mu and tau leaf models / dimensions
  leaf_model_mu_forest <- 0
  leaf_dimension_mu_forest <- 1
  if (has_multivariate_treatment) {
    leaf_model_tau_forest <- 2
    leaf_dimension_tau_forest <- ncol(Z_train)
  } else {
    leaf_model_tau_forest <- 1
    leaf_dimension_tau_forest <- 1
  }

  # Set variance leaf model type (currently only one option)
  leaf_model_variance_forest <- 3
  leaf_dimension_variance_forest <- 1

  # Model params set without calibration / initialization
  if (include_variance_forest) {
    num_variance_covariates <- sum(variable_weights_variance > 0)
  } else {
    num_variance_covariates <- 0
  }
  model_params_r <- list(
    "initial_b_0" = b_0,
    "initial_b_1" = b_1,
    "a_global" = a_global,
    "b_global" = b_global,
    "a_leaf_mu" = a_leaf_mu,
    "a_leaf_tau" = a_leaf_tau,
    "standardize" = standardize,
    "num_covariates" = num_cov_orig,
    "num_prognostic_covariates" = sum(variable_weights_mu > 0),
    "num_treatment_covariates" = sum(variable_weights_tau > 0),
    "num_variance_covariates" = num_variance_covariates,
    "treatment_dim" = ncol(Z_train),
    "propensity_covariate" = propensity_covariate,
    "binary_treatment" = binary_treatment,
    "multivariate_treatment" = has_multivariate_treatment,
    "adaptive_coding" = adaptive_coding,
    "internal_propensity_model" = internal_propensity_model,
    "num_gfr" = num_gfr,
    "num_burnin" = num_burnin,
    "num_mcmc" = num_mcmc,
    "keep_every" = keep_every,
    "num_chains" = num_chains,
    "has_test" = has_test,
    "has_rfx" = has_rfx,
    "has_rfx_basis" = has_basis_rfx,
    "num_rfx_basis" = num_basis_rfx,
    "include_variance_forest" = include_variance_forest,
    "sample_sigma2_global" = sample_sigma2_global,
    "sample_sigma2_leaf_mu" = sample_sigma2_leaf_mu,
    "sample_sigma2_leaf_tau" = sample_sigma2_leaf_tau,
    "probit_outcome_model" = probit_outcome_model,
    "outcome_model" = outcome_model,
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

  # Specify the BCF config
  bcf_config <- list(
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
    "adaptive_coding" = adaptive_coding,
    "b_0_init" = b_0,
    "b_1_init" = b_1,
    "a_sigma2_global" = a_global,
    "b_sigma2_global" = b_global,
    "sigma2_global_init" = sigma2_init,
    "sample_sigma2_global" = sample_sigma2_global,
    "num_trees_mu" = num_trees_mu,
    "alpha_mu" = alpha_mu,
    "beta_mu" = beta_mu,
    "min_samples_leaf_mu" = min_samples_leaf_mu,
    "max_depth_mu" = max_depth_mu,
    "leaf_constant_mu" = TRUE,
    "leaf_dim_mu" = leaf_dimension_mu_forest,
    "exponentiated_leaf_mu" = FALSE,
    "num_features_subsample_mu" = num_features_subsample_mu,
    "a_sigma2_mu" = a_leaf_mu,
    "b_sigma2_mu" = b_leaf_mu,
    "sigma2_mu_init" = if (is.matrix(sigma2_leaf_mu)) {
      NULL
    } else {
      sigma2_leaf_mu
    },
    "sample_sigma2_leaf_mu" = sample_sigma2_leaf_mu,
    "mean_leaf_model_type" = leaf_model_mu_forest,
    "sigma2_leaf_mu_matrix" = if (is.matrix(sigma2_leaf_mu)) {
      as.numeric(sigma2_leaf_mu)
    } else {
      NULL
    },
    "num_trees_tau" = num_trees_tau,
    "alpha_tau" = alpha_tau,
    "beta_tau" = beta_tau,
    "min_samples_leaf_tau" = min_samples_leaf_tau,
    "max_depth_tau" = max_depth_tau,
    "leaf_constant_tau" = FALSE,
    "leaf_dim_tau" = leaf_dimension_tau_forest,
    "exponentiated_leaf_tau" = FALSE,
    "num_features_subsample_tau" = num_features_subsample_tau,
    "a_sigma2_tau" = a_leaf_tau,
    "b_sigma2_tau" = b_leaf_tau,
    "sigma2_tau_init" = if (is.matrix(sigma2_leaf_tau)) {
      NULL
    } else {
      sigma2_leaf_tau
    },
    "sample_sigma2_leaf_tau" = sample_sigma2_leaf_tau,
    "tau_leaf_model_type" = leaf_model_tau_forest,
    "sigma2_leaf_tau_matrix" = if (is.matrix(sigma2_leaf_tau)) {
      as.numeric(sigma2_leaf_tau)
    } else {
      NULL
    },
    "sample_tau_0" = sample_tau_0,
    "tau_0_prior_var_scalar" = if (is.matrix(tau_0_prior_var)) {
      NULL
    } else {
      tau_0_prior_var
    },
    "tau_0_prior_var_multivariate" = if (is.matrix(tau_0_prior_var)) {
      as.numeric(tau_0_prior_var)
    } else {
      NULL
    },
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
    "leaf_dim_variance" = leaf_dimension_variance_forest,
    "exponentiated_leaf_variance" = TRUE,
    "num_features_subsample_variance" = num_features_subsample_variance,
    "feature_types" = as.integer(feature_types),
    "sweep_update_indices_mu" = if (num_trees_mu > 0) {
      0:(num_trees_mu - 1)
    } else {
      NULL
    },
    "sweep_update_indices_tau" = if (num_trees_tau > 0) {
      0:(num_trees_tau - 1)
    } else {
      NULL
    },
    "sweep_update_indices_variance" = if (num_trees_variance > 0) {
      0:(num_trees_variance - 1)
    } else {
      NULL
    },
    "var_weights_mu" = variable_weights_mu,
    "var_weights_tau" = variable_weights_tau,
    "var_weights_variance" = variable_weights_variance,
    "has_random_effects" = has_rfx,
    "rfx_model_spec" = if (has_rfx) {
      ifelse(
        rfx_model_spec == "custom",
        0,
        ifelse(
          rfx_model_spec == "intercept_only",
          1,
          ifelse(rfx_model_spec == "intercept_plus_treatment", 2, NULL)
        )
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

  bcf_samples <- BCFSamples$new()
  bcf_metadata <- bcf_sample_cpp(
    samples = bcf_samples$samples_ptr,
    X_train = X_train,
    Z_train = Z_train,
    y_train = y_train,
    X_test = if (exists("X_test")) X_test else NULL,
    Z_test = if (exists("Z_test")) Z_test else NULL,
    n_train = nrow(X_train),
    n_test = if (!is.null(X_test)) nrow(X_test) else 0L,
    p = ncol(X_train),
    treatment_dim = ncol(Z_train),
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
    adaptive_coding = adaptive_coding,
    warmstart_samples = if (has_prev_model) {
      previous_bcf_model$samples$samples_ptr
    } else {
      NULL
    },
    warmstart_sample_num = if (has_prev_model) {
      as.integer(previous_model_warmstart_sample_num)
    } else {
      0L
    },
    config_input = bcf_config
  )
  result <- list()
  model_params_cpp <- list(
    "initial_sigma2" = bcf_metadata[["sigma2_global_init"]],
    "sigma2_leaf_mu" = bcf_metadata[["sigma2_mu_init"]],
    "sigma2_leaf_tau" = bcf_metadata[["sigma2_tau_init"]],
    "b_leaf_mu" = bcf_metadata[["b_sigma2_mu"]],
    "b_leaf_tau" = bcf_metadata[["b_sigma2_tau"]],
    "a_forest" = bcf_metadata[["shape_variance_forest"]],
    "b_forest" = bcf_metadata[["scale_variance_forest"]],
    "outcome_mean" = bcf_samples$y_bar(),
    "outcome_scale" = bcf_samples$y_std(),
    "num_samples" = bcf_samples$num_samples(),
    "sample_tau_0" = sample_tau_0,
    "tau_0_prior_var" = if (sample_tau_0) tau_0_prior_var else NULL,
    "rng_state" = bcf_metadata[["rng_state"]]
  )
  model_params <- c(model_params_r, model_params_cpp)
  result[["model_params"]] <- model_params
  result[["train_set_metadata"]] <- X_train_metadata
  result[["samples"]] <- bcf_samples
  result[["bcf_config"]] <- bcf_config
  result[["continuation_state"]] <- list(
    variable_weights = variable_weights_raw,
    variable_subset_mu = variable_subset_mu,
    variable_subset_tau = variable_subset_tau,
    variable_subset_variance = variable_subset_variance
  )

  # Unpack RFX samples
  if (has_rfx) {
    # Only need to store unique group IDs, everything else stored in BARTSamples
    result[["rfx_unique_group_ids"]] <- levels(group_ids_factor)
  }

  if (internal_propensity_model) {
    result[["bart_propensity_model"]] <- bart_model_propensity
  }

  class(result) <- "bcfmodel"

  return(result)
}

#' Guard accessor for a `bcfmodel`'s removed direct-forest fields.
#'
#' The sampled forests / parameters / predictions are owned by a single `BCFSamples` object stored in `object$samples`.
#' They are no longer stored or accessible via `$forests_mu` / `$forests_tau` / `$forests_variance`.
#' Similarly, sampled parameter vectors are no longer stored or accessible via `$sigma2_global_samples` / `$sigma2_leaf_mu_samples` / `$sigma2_leaf_tau_samples`
#' and cached predictions are no longer accessible via `$y_hat_train` / `$y_hat_test` / `$mu_hat_train` / `$mu_hat_test` / `$tau_hat_train` / `$tau_hat_test` /
#' `$sigma2_x_hat_train` / `$sigma2_x_hat_test`. Accessing any of these model terms by name raises an error pointing at the supported extraction path.
#' @noRd
#' @export
`$.bcfmodel` <- function(x, name) {
  if (
    identical(name, "forests_mu") ||
      identical(name, "forests_tau") ||
      identical(name, "forests_variance")
  ) {
    stop(
      sprintf(
        paste0(
          "`bcfmodel$%s` has been removed. The sampled forests are owned by `model$samples`; ",
          "extract a standalone copy with `extractForest()`."
        ),
        name
      ),
      call. = FALSE
    )
  } else if (
    identical(name, "sigma2_global_samples") ||
      identical(name, "sigma2_leaf_mu_samples") ||
      identical(name, "sigma2_leaf_tau_samples") ||
      identical(name, "b_0_samples") ||
      identical(name, "b_1_samples") ||
      identical(name, "tau_0_samples") ||
      identical(name, "y_hat_train") ||
      identical(name, "y_hat_test") ||
      identical(name, "mu_hat_train") ||
      identical(name, "mu_hat_test") ||
      identical(name, "tau_hat_train") ||
      identical(name, "tau_hat_test") ||
      identical(name, "sigma2_x_hat_train") ||
      identical(name, "sigma2_x_hat_test")
  ) {
    stop(
      sprintf(
        paste0(
          "`bcfmodel$%s` has been removed. The parameters are stored in a C++ BCFSamples object; ",
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

#' @title Predict from BCF Model
#' @description
#' Predict from a sampled BCF model on new data
#'
#' @param object Object of type `bcfmodel` containing draws of a Bayesian causal forest model and associated sampling outputs.
#' @param X Covariates used to determine tree leaf predictions for each observation. Must be passed as a matrix or dataframe.
#' @param Z Treatments used for prediction.
#' @param propensity (Optional) Propensities used for prediction.
#' @param rfx_group_ids (Optional) Test set group labels used for an additive random effects model.
#' We do not currently support (but plan to in the near future), test set evaluation for group labels
#' that were not in the training set.
#' @param rfx_basis (Optional) Test set basis for "random-slope" regression in additive random effects model. If the model was sampled with a random effects `model_spec` of "intercept_only" or "intercept_plus_treatment", this is optional, but if it is provided, it will be used.
#' @param type (Optional) Type of prediction to return. Options are "mean", which averages the predictions from every draw of a BCF model, and "posterior", which returns the entire matrix of posterior predictions. Default: "posterior".
#' @param terms (Optional) Which model terms to include in the prediction. Options include `"y_hat"`, `"prognostic_function"`, `"mu"`, `"cate"`, `"tau"`, `"rfx"`, `"variance_forest"`, or `"all"`.
#'
#'   The treatment effect terms follow a three-level hierarchy:
#'   \itemize{
#'     \item `"tau"` returns `tau_0 + tau(X)`: the parametric treatment intercept (if sampled) plus the treatment forest. This matches `model$tau_hat_train` / `model$tau_hat_test`.
#'     \item `"cate"` additionally folds in the random slope on treatment when random effects are fit with `rfx_model_spec = "intercept_plus_treatment"`; otherwise it is identical to `"tau"`.
#'     \item The raw forest-only component (without `tau_0`) is not directly returned by this method; extract the treatment forest with `model$samples$materialize_tau_forest()` to access it.
#'   }
#'
#'   Similarly for the prognostic term: `"mu"` returns the prognostic forest only, while `"prognostic_function"` additionally folds in the random intercept when `rfx_model_spec` is `"intercept_only"` or `"intercept_plus_treatment"`; otherwise the two are identical.
#'
#'   If a model doesn't have random effects or variance forest predictions but one of those terms is requested, the request will simply be ignored. If none of the requested terms are present, this function will return `NULL` along with a warning. Default: `"all"`.
#' @param scale (Optional) Scale of mean function predictions. Options are "linear", which returns predictions on the original scale of the mean forest / RFX terms, and "probability", which transforms predictions into a probability of observing `y == 1`. "probability" is only valid for models fit with a probit outcome model. Default: "linear".
#' @param ... (Optional) Other prediction parameters.
#'
#' @return List of prediction matrices or single prediction matrix / vector, depending on the terms requested.
#' @export
#'
#' @examples
#' n <- 500
#' p <- 5
#' X <- matrix(runif(n*p), ncol = p)
#' mu_x <- (
#'     ((0 <= X[,1]) & (0.25 > X[,1])) * (-7.5) +
#'     ((0.25 <= X[,1]) & (0.5 > X[,1])) * (-2.5) +
#'     ((0.5 <= X[,1]) & (0.75 > X[,1])) * (2.5) +
#'     ((0.75 <= X[,1]) & (1 > X[,1])) * (7.5)
#' )
#' pi_x <- (
#'     ((0 <= X[,1]) & (0.25 > X[,1])) * (0.2) +
#'     ((0.25 <= X[,1]) & (0.5 > X[,1])) * (0.4) +
#'     ((0.5 <= X[,1]) & (0.75 > X[,1])) * (0.6) +
#'     ((0.75 <= X[,1]) & (1 > X[,1])) * (0.8)
#' )
#' tau_x <- (
#'     ((0 <= X[,2]) & (0.25 > X[,2])) * (0.5) +
#'     ((0.25 <= X[,2]) & (0.5 > X[,2])) * (1.0) +
#'     ((0.5 <= X[,2]) & (0.75 > X[,2])) * (1.5) +
#'     ((0.75 <= X[,2]) & (1 > X[,2])) * (2.0)
#' )
#' Z <- rbinom(n, 1, pi_x)
#' noise_sd <- 1
#' y <- mu_x + tau_x*Z + rnorm(n, 0, noise_sd)
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
#' bcf_model <- bcf(X_train = X_train, Z_train = Z_train, y_train = y_train,
#'                  propensity_train = pi_train, num_gfr = 10,
#'                  num_burnin = 0, num_mcmc = 10)
#' preds <- predict(bcf_model, X_test, Z_test, pi_test)
predict.bcfmodel <- function(
  object,
  X,
  Z,
  propensity = NULL,
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
  is_probit <- object$model_params$outcome_model$link == "probit"
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
    stop("type must either be 'mean' or 'posterior")
  }
  predict_mean <- type == "mean"

  # Warn users about CATE / prognostic function when rfx_model_spec is "custom"
  if (object$model_params$has_rfx) {
    if (object$model_params$rfx_model_spec == "custom") {
      if (("prognostic_function" %in% terms) || ("cate" %in% terms)) {
        warning(paste0(
          "This BCF model was fit with a custom random effects model specification (i.e. a user-provided basis). ",
          "As a result, 'prognostic_function' and 'cate' refer only to the prognostic ('mu') ",
          "and treatment effect 'tau' forests, respectively, and do not include any random ",
          "effects contributions. If your user-provided random effects basis includes a random intercept or a ",
          "random slope on the treatment variable, you will need to compute the prognostic or CATE functions manually by predicting ",
          "'yhat' for different covariate and rfx_basis values."
        ))
      }
    }
  }

  # Handle prediction terms
  rfx_model_spec <- object$model_params$rfx_model_spec
  rfx_intercept_only <- rfx_model_spec == "intercept_only"
  rfx_intercept_plus_treatment <- rfx_model_spec == "intercept_plus_treatment"
  rfx_intercept <- rfx_intercept_only || rfx_intercept_plus_treatment
  mu_prog_separate <- ifelse(rfx_intercept, TRUE, FALSE)
  tau_cate_separate <- ifelse(rfx_intercept_plus_treatment, TRUE, FALSE)
  if (!is.character(terms)) {
    stop("type must be a string or character vector")
  }
  for (term in terms) {
    if (
      !(term %in%
        c(
          "y_hat",
          "prognostic_function",
          "mu",
          "cate",
          "tau",
          "rfx",
          "variance_forest",
          "all"
        ))
    ) {
      warning(paste0(
        "Term '",
        term,
        "' was requested. Valid terms are 'y_hat', 'prognostic_function', 'mu', 'cate', 'tau', 'rfx', 'variance_forest', and 'all'.",
        " This term will be ignored and prediction will only proceed if other requested terms are available in the model."
      ))
    }
  }

  num_terms <- length(terms)
  has_mu_forest <- T
  has_tau_forest <- T
  has_variance_forest <- object$model_params$include_variance_forest
  has_rfx <- object$model_params$has_rfx
  has_y_hat <- T
  predict_y_hat <- (((has_y_hat) && ("y_hat" %in% terms)) ||
    ((has_y_hat) && ("all" %in% terms)))
  predict_mu_forest <- (((has_mu_forest) && ("all" %in% terms)) ||
    ((has_mu_forest) && ("mu" %in% terms)))
  predict_tau_forest <- (((has_tau_forest) && ("tau" %in% terms)) ||
    ((has_tau_forest) && ("all" %in% terms)))
  predict_prog_function <- (((has_mu_forest) &&
    ("prognostic_function" %in% terms)) ||
    ((has_mu_forest) && ("all" %in% terms)))
  predict_cate_function <- (((has_tau_forest) && ("cate" %in% terms)) ||
    ((has_tau_forest) && ("all" %in% terms)))
  predict_rfx <- (((has_rfx) && ("rfx" %in% terms)) ||
    ((has_rfx) && ("all" %in% terms)))
  predict_variance_forest <- (((has_variance_forest) &&
    ("variance_forest" %in% terms)) ||
    ((has_variance_forest) && ("all" %in% terms)))
  predict_count <- sum(c(
    predict_y_hat,
    predict_mu_forest,
    predict_prog_function,
    predict_tau_forest,
    predict_cate_function,
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
  predict_rfx_raw <- ((predict_prog_function && has_rfx && rfx_intercept) ||
    (predict_cate_function && has_rfx && rfx_intercept_plus_treatment))
  predict_mu_forest_intermediate <- ((predict_y_hat || predict_prog_function) &&
    has_mu_forest)
  predict_tau_forest_intermediate <- ((predict_y_hat ||
    predict_cate_function ||
    (object$model_params$adaptive_coding &&
      (predict_mu_forest || predict_prog_function))) &&
    has_tau_forest)

  # Make sure covariates are matrix or data frame
  if ((!is.data.frame(X)) && (!is.matrix(X))) {
    stop("X must be a matrix or dataframe")
  }

  # Handle factor-valued treatment before numeric conversion
  if (is.factor(Z)) {
    lvls <- levels(Z)
    if (length(lvls) != 2) {
      stop("Factor Z must have exactly 2 levels for binary treatment")
    }
    warning(
      "Z is a factor; recoding to 0/1 using level order: ",
      lvls[1],
      " = 0, ",
      lvls[2],
      " = 1"
    )
    Z <- as.integer(Z) - 1L
  }

  # Convert all input data to matrices if not already converted
  if ((is.null(dim(Z))) && (!is.null(Z))) {
    Z <- as.matrix(as.numeric(Z))
  }
  if ((is.null(dim(propensity))) && (!is.null(propensity))) {
    propensity <- as.matrix(propensity)
  }
  if ((is.null(dim(rfx_basis))) && (!is.null(rfx_basis))) {
    rfx_basis <- as.matrix(rfx_basis)
  }

  # Data checks
  if (nrow(X) != nrow(Z)) {
    stop("X and Z must have the same number of rows")
  }
  if (object$model_params$num_covariates != ncol(X)) {
    stop(
      "X and must have the same number of columns as the covariates used to train the model"
    )
  }
  if ((object$model_params$has_rfx) && (is.null(rfx_group_ids))) {
    stop(
      "Random effect group labels (rfx_group_ids) must be provided for this model"
    )
  }
  if ((object$model_params$has_rfx_basis) && (is.null(rfx_basis))) {
    if (object$model_params$rfx_model_spec == "custom") {
      stop("Random effects basis (rfx_basis) must be provided for this model")
    }
  }
  if ((object$model_params$num_rfx_basis > 0) && (!is.null(rfx_basis))) {
    if (ncol(rfx_basis) != object$model_params$num_rfx_basis) {
      stop(
        "Random effects basis has a different dimension than the basis used to train this model"
      )
    }
  }

  # Preprocess covariates before any prediction calls that depend on X being
  # a numeric matrix (e.g. the internal propensity BART model was trained on a
  # preprocessed matrix, so it expects a matrix, not the raw data frame)
  train_set_metadata <- object$train_set_metadata
  X <- preprocessPredictionData(X, train_set_metadata)

  # Compute propensity score using the internal bart model
  if (
    (object$model_params$propensity_covariate != "none") &&
      (is.null(propensity))
  ) {
    if (!object$model_params$internal_propensity_model) {
      stop("propensity must be provided for this model")
    }
    propensity <- rowMeans(predict(object$bart_propensity_model, X)$y_hat)
  }

  # Recode group IDs to integer vector (if passed as, for example, a vector of county names, etc...)
  if (!is.null(rfx_group_ids)) {
    rfx_unique_group_ids <- object$rfx_unique_group_ids
    group_ids_factor <- factor(rfx_group_ids, levels = rfx_unique_group_ids)
    if (sum(is.na(group_ids_factor)) > 0) {
      stop(
        "All random effect group labels provided in rfx_group_ids must have been present at sampling time"
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
    } else if (
      object$model_params$rfx_model_spec == "intercept_plus_treatment"
    ) {
      # Only construct a basis if user-provided basis missing
      if (is.null(rfx_basis)) {
        rfx_basis <- cbind(
          rep(1, nrow(X)),
          Z
        )
      }
    }
  }

  # Add propensities to covariate set if necessary
  X_combined <- X
  if (object$model_params$propensity_covariate != "none") {
    X_combined <- cbind(X, propensity)
  }

  # Dimensions needed by both the C++ and R predict paths
  n <- nrow(X_combined)
  p <- ncol(X_combined)
  treatment_dim <- ncol(Z)
  obs_weights <- NULL
  rfx_num_groups <- if (!is.null(rfx_group_ids)) {
    length(unique(rfx_group_ids))
  } else {
    0L
  }
  rfx_basis_dim <- if (!is.null(rfx_basis)) ncol(rfx_basis) else 0L

  scale_int <- switch(
    scale,
    "linear" = 0L,
    "probability" = 1L,
    "class" = 2L,
    0L
  )

  # Build a flat list of model components for bcf_predict_cpp, since the bcfmodel
  # object uses R6 wrappers and nested model_params that C++ cannot navigate directly.
  has_variance_forest_model <- isTRUE(
    object$model_params$include_variance_forest
  )
  # Read forests through borrowed (non-owning) pointers into the single-owner
  # samples object -- no deep copy, no deprecated-accessor error.
  bcf_samples <- object$samples
  has_rfx_model <- isTRUE(object$model_params$has_rfx)
  bcf_metadata_list <- list(
    num_samples = as.integer(object$model_params$num_samples),
    y_bar = as.double(object$model_params$outcome_mean),
    y_std = as.double(object$model_params$outcome_scale),
    include_variance_forest = has_variance_forest_model,
    has_rfx = has_rfx_model,
    rfx_model_spec = if (has_rfx_model) {
      object$model_params$rfx_model_spec
    } else {
      ""
    },
    adaptive_coding = isTRUE(object$model_params$adaptive_coding),
    sample_tau_0 = isTRUE(object$model_params$sample_tau_0)
  )

  output <- bcf_predict_cpp(
    bcf_samples_ptr = bcf_samples$samples_ptr,
    bcf_model_metadata = bcf_metadata_list,
    X = X_combined,
    Z = Z,
    n = n,
    p = p,
    treatment_dim = treatment_dim,
    obs_weights = obs_weights,
    rfx_group_ids = rfx_group_ids,
    rfx_basis = rfx_basis,
    rfx_num_groups = rfx_num_groups,
    rfx_basis_dim = rfx_basis_dim,
    posterior = type == "posterior",
    scale = scale_int,
    predict_y_hat = predict_y_hat,
    predict_mu_x = predict_mu_forest,
    predict_tau_x = predict_tau_forest,
    predict_prognostic_function = predict_prog_function,
    predict_cate = predict_cate_function,
    predict_conditional_variance = predict_variance_forest,
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
    if (dim2 == 1L && dim3 == 1L) {
      return(as.vector(v))
    }
    if (dim3 == 1L) {
      m <- v
      dim(m) <- c(dim1, dim2)
      return(m)
    }
    if (dim2 == 1L) {
      m <- v
      dim(m) <- c(dim1, dim3)
      return(m)
    }
    a <- v
    dim(a) <- c(dim1, dim2, dim3)
    a
  }
  result <- list(
    y_hat = reshape_cpp_pred_2d(output$y_hat, n, num_samples_output),
    mu_hat = reshape_cpp_pred_2d(output$mu_x, n, num_samples_output),
    tau_hat = reshape_cpp_pred_3d(
      output$tau_x,
      n,
      treatment_dim,
      num_samples_output
    ),
    prognostic_function = reshape_cpp_pred_2d(
      output$prognostic_function,
      n,
      num_samples_output
    ),
    cate = reshape_cpp_pred_3d(
      output$cate,
      n,
      treatment_dim,
      num_samples_output
    ),
    rfx_predictions = reshape_cpp_pred_2d(
      output$random_effects,
      n,
      num_samples_output
    ),
    variance_forest_predictions = reshape_cpp_pred_2d(
      output$conditional_variance,
      n,
      num_samples_output
    )
  )
  if (predict_count == 1L) {
    if (predict_y_hat) {
      return(result[["y_hat"]])
    }
    if (predict_mu_forest) {
      return(result[["mu_hat"]])
    }
    if (predict_prog_function) {
      return(result[["prognostic_function"]])
    }
    if (predict_tau_forest) {
      return(result[["tau_hat"]])
    }
    if (predict_cate_function) {
      return(result[["cate"]])
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

#' @title Print Summary of BCF Model
#' @description Prints a summary of the BCF model, including the model terms and their specifications.
#' @param x The BCF model object
#' @param ... Additional arguments (currently unused)
#' @export
#' @return BCF model object unchanged after printing summary
print.bcfmodel <- function(x, ...) {
  # What type of model was run
  model_terms <- c("prognostic forest", "treatment effect forest")
  if (x$model_params$include_variance_forest) {
    model_terms <- c(model_terms, "variance forest")
  }
  if (x$model_params$has_rfx) {
    model_terms <- c(model_terms, "additive random effects")
  }
  if (x$model_params$sample_sigma2_global) {
    model_terms <- c(model_terms, "global error variance model")
  }
  if (x$model_params$sample_sigma2_leaf_mu) {
    model_terms <- c(model_terms, "prognostic forest leaf scale model")
  }
  if (x$model_params$sample_sigma2_leaf_tau) {
    model_terms <- c(model_terms, "treatment effect forest leaf scale model")
  }
  if (x$model_params$sample_tau_0) {
    model_terms <- c(model_terms, "treatment effect intercept model")
  }
  if (length(model_terms) > 2) {
    summary_message <- paste0(
      "stochtree::bcf() run with ",
      paste0(
        paste0(model_terms[1:(length(model_terms) - 1)], collapse = ", "),
        ", and ",
        model_terms[length(model_terms)]
      )
    )
  } else if (length(model_terms) == 2) {
    summary_message <- paste0(
      "stochtree::bcf() run with ",
      paste0(model_terms, collapse = " and ")
    )
  } else {
    summary_message <- paste0("stochtree::bcf() run with ", model_terms)
  }

  # Outcome details
  is_probit <- x$model_params$outcome_model$link == "probit"
  summary_message <- paste0(
    summary_message,
    "\n",
    "Outcome was modeled ",
    ifelse(
      is_probit,
      "with a probit link",
      "as gaussian"
    )
  )

  # Treatment details
  if (x$model_params$binary_treatment) {
    summary_message <- paste0(
      summary_message,
      "\n",
      "Treatment was binary and ",
      ifelse(
        x$model_params$adaptive_coding,
        "its effect was estimated with adaptive coding",
        "its effect was estimated with default coding"
      )
    )
  } else if (x$model_params$multivariate_treatment) {
    summary_message <- paste0(
      summary_message,
      "\n",
      "Treatment was multivariate with ",
      x$model_params$treatment_dim,
      " dimensions"
    )
  } else {
    summary_message <- paste0(
      summary_message,
      "\n",
      "Treatment was univariate but not binary"
    )
  }

  # Standardization
  if (x$model_params$standardize) {
    summary_message <- paste0(
      summary_message,
      "\n",
      "outcome was standardized"
    )
  }

  # Internal propensity model
  if (x$model_params$propensity_covariate == "none") {
    summary_message <- paste0(
      summary_message,
      "\n",
      "Propensity scores were not used in either forest of the model"
    )
  } else {
    if (x$model_params$internal_propensity_model) {
      summary_message <- paste0(
        summary_message,
        "\n",
        "An internal propensity model was fit using stochtree::bart() in lieu of user-provided propensity scores"
      )
    } else {
      summary_message <- paste0(
        summary_message,
        "\n",
        "User-provided propensity scores were included in the model"
      )
    }
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
    } else if (x$model_params$rfx_model_spec == "intercept_plus_treatment") {
      summary_message <- paste0(
        summary_message,
        "\n",
        "Random effects were fit with an 'intercept-plus-treatment' parameterization"
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
    ifelse(x$model_params$num_chains == 1, " chain of ", " chains of "),
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

  # Return bcf model invisibly
  invisible(x)
}

#' @title Summarize BCF Model
#' @description Summarize a BCF fit with a description of the model that was fit and numeric summaries of any sampled quantities.
#' @param object The BCF model object
#' @param ... Additional arguments
#' @export
#' @return BCF model object unchanged after summarizing
summary.bcfmodel <- function(object, ...) {
  # First, print the BCF model
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

  # Leaf scale for the prognostic forest
  if (object$model_params$sample_sigma2_leaf_mu) {
    sigma2_leaf_samples <- object$samples$leaf_scale_mu_samples()
    n_samples <- length(sigma2_leaf_samples)
    mean_sigma2 <- mean(sigma2_leaf_samples)
    sd_sigma2 <- sd(sigma2_leaf_samples)
    quantiles_sigma2 <- quantile(
      sigma2_leaf_samples,
      probs = c(0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
    )
    cat(sprintf(
      "Summary of prognostic forest leaf scale posterior: \n%d samples, mean = %.3f, standard deviation = %.3f, quantiles:\n",
      n_samples,
      mean_sigma2,
      sd_sigma2
    ))
    print(quantiles_sigma2)
  }

  # Leaf scale for the treatment effect forest
  if (object$model_params$sample_sigma2_leaf_tau) {
    sigma2_leaf_samples <- object$samples$leaf_scale_tau_samples()
    n_samples <- length(sigma2_leaf_samples)
    mean_sigma2 <- mean(sigma2_leaf_samples)
    sd_sigma2 <- sd(sigma2_leaf_samples)
    quantiles_sigma2 <- quantile(
      sigma2_leaf_samples,
      probs = c(0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
    )
    cat(sprintf(
      "Summary of treatment effect forest leaf scale posterior: \n%d samples, mean = %.3f, standard deviation = %.3f, quantiles:\n",
      n_samples,
      mean_sigma2,
      sd_sigma2
    ))
    print(quantiles_sigma2)
  }

  # Adaptive coding parameters
  if (object$model_params$adaptive_coding) {
    b0_samples <- object$samples$b0_samples()
    b1_samples <- object$samples$b1_samples()
    n_samples <- length(b0_samples)
    mean_b0 <- mean(b0_samples)
    mean_b1 <- mean(b1_samples)
    sd_b0 <- sd(b0_samples)
    sd_b1 <- sd(b1_samples)
    quantiles_b0 <- quantile(
      b0_samples,
      probs = c(0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
    )
    quantiles_b1 <- quantile(
      b1_samples,
      probs = c(0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
    )
    cat(sprintf(
      "Summary of adaptive coding parameters: \n%d samples, mean (control) = %.3f, mean (treated) = %.3f, standard deviation (control) = %.3f, standard deviation (treated) = %.3f\n",
      n_samples,
      mean_b0,
      mean_b1,
      sd_b0,
      sd_b1
    ))
    cat("quantiles (control):\n")
    print(quantiles_b0)
    cat("quantiles (treated):\n")
    print(quantiles_b1)
  }

  # Treatment effect intercept (tau_0)
  tau_0_samples <- object$samples$tau_0_samples()
  if (object$model_params$sample_tau_0 && length(tau_0_samples) > 0) {
    tau_0_vec <- as.numeric(tau_0_samples)
    n_samples <- if (is.matrix(tau_0_samples)) {
      ncol(tau_0_samples)
    } else {
      length(tau_0_samples)
    }
    mean_tau_0 <- mean(tau_0_vec)
    sd_tau_0 <- sd(tau_0_vec)
    quantiles_tau_0 <- quantile(
      tau_0_vec,
      probs = c(0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
    )
    cat(sprintf(
      "Summary of treatment effect intercept (tau_0) posterior: \n%d samples, mean = %.3f, standard deviation = %.3f, quantiles:\n",
      n_samples,
      mean_tau_0,
      sd_tau_0
    ))
    print(quantiles_tau_0)
  }

  # In-sample predictions
  y_hat_train <- object$samples$y_hat_train()
  if (length(y_hat_train) > 0) {
    y_hat_train_mean <- rowMeans(y_hat_train)
    n_y_hat_train <- length(y_hat_train_mean)
    mean_y_hat_train <- mean(y_hat_train_mean)
    sd_y_hat_train <- sd(y_hat_train_mean)
    quantiles_y_hat_train <- quantile(
      y_hat_train_mean,
      probs = c(0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
    )
    cat(sprintf(
      "Summary of in-sample posterior mean predictions: \n%d observations, mean = %.3f, standard deviation = %.3f, quantiles:\n",
      n_y_hat_train,
      mean_y_hat_train,
      sd_y_hat_train
    ))
    print(quantiles_y_hat_train)
  }

  # Test-set predictions
  y_hat_test <- object$samples$y_hat_test()
  if (length(y_hat_test) > 0) {
    y_hat_test_mean <- rowMeans(y_hat_test)
    n_y_hat_test <- length(y_hat_test_mean)
    mean_y_hat_test <- mean(y_hat_test_mean)
    sd_y_hat_test <- sd(y_hat_test_mean)
    quantiles_y_hat_test <- quantile(
      y_hat_test_mean,
      probs = c(0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
    )
    cat(sprintf(
      "Summary of test-set posterior mean predictions: \n%d observations, mean = %.3f, standard deviation = %.3f, quantiles:\n",
      n_y_hat_test,
      mean_y_hat_test,
      sd_y_hat_test
    ))
    print(quantiles_y_hat_test)
  }

  # In-sample treatment effect function estimates
  tau_hat_train <- object$samples$tau_forest_predictions_train()
  if (length(tau_hat_train) > 0) {
    if (!object$model_params$multivariate_treatment) {
      tau_hat_train_mean <- rowMeans(tau_hat_train)
      n_tau_hat_train <- length(tau_hat_train_mean)
      mean_tau_hat_train <- mean(tau_hat_train_mean)
      sd_tau_hat_train <- sd(tau_hat_train_mean)
      quantiles_tau_hat_train <- quantile(
        tau_hat_train_mean,
        probs = c(0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
      )
      cat(sprintf(
        "Summary of in-sample posterior mean CATEs: \n%d observations, mean = %.3f, standard deviation = %.3f, quantiles:\n",
        n_tau_hat_train,
        mean_tau_hat_train,
        sd_tau_hat_train
      ))
      print(quantiles_tau_hat_train)
    }
  }

  # Test set treatment effect function estimates
  tau_hat_test <- object$samples$tau_forest_predictions_test()
  if (length(tau_hat_test) > 0) {
    if (!object$model_params$multivariate_treatment) {
      tau_hat_test_mean <- rowMeans(tau_hat_test)
      n_tau_hat_test <- length(tau_hat_test_mean)
      mean_tau_hat_test <- mean(tau_hat_test_mean)
      sd_tau_hat_test <- sd(tau_hat_test_mean)
      quantiles_tau_hat_test <- quantile(
        tau_hat_test_mean,
        probs = c(0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
      )
      cat(sprintf(
        "Summary of test-set posterior mean CATEs: \n%d observations, mean = %.3f, standard deviation = %.3f, quantiles:\n",
        n_tau_hat_test,
        mean_tau_hat_test,
        sd_tau_hat_test
      ))
      print(quantiles_tau_hat_test)
    }
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
      quantile_summary <- quantile(
        rfx_beta_samples,
        probs = c(0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
      )
      cat("Random effects overall quantiles:\n")
      print(quantile_summary)
    }
  }

  # Return bcf model invisibly
  invisible(object)
}

#' @title Plot BCF Model
#' @description Plot the BCF model fit and any relevant sampled quantities. This will default to a traceplot of the global error scale and the in-sample mean forest predictions for the first train set observation. Since `stochtree::bcf()` is flexible and it's possible to sample a model with a fixed global error scale and no mean forest, this procedure will throw an error if these two default terms are omitted.
#' @param x The BCF model object
#' @param ... Additional arguments
#' @export
#' @return BCF model object unchanged after summarizing
plot.bcfmodel <- function(x, ...) {
  # Check if model has global error scale samples
  has_sigma2_samples <- x$model_params$sample_sigma2_global
  has_mean_forest_preds <- length(x$samples$y_hat_train()) > 0

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
      plot(
        x$samples$y_hat_train()[1, ],
        type = "l",
        ylab = "Predictions",
        main = "In-sample mean function trace for the first train set observation"
      )
    }
  } else {
    stop(
      "This model does not have enough model terms / parameter traces to produce stochtree's default plots. See `predict.bcfmodel()` for examples of how to further investigate your model."
    )
  }

  # Return bcf model invisibly
  invisible(x)
}

#' @title Extract BCF Parameter Samples
#' @description Extract a vector, matrix or array of parameter samples from a BCF model by name.
#' Random effects are handled by a separate `getRandomEffectSamples` function due to the complexity of the random effects parameters.
#' If the requested model term is not found, an error is thrown.
#' The following conventions are used for parameter names:
#' - Global error variance: `"sigma2"`, `"global_error_scale"`, `"sigma2_global"`
#' - Prognostic forest leaf scale: `"sigma2_leaf_mu"`, `"leaf_scale_mu"`, `"mu_leaf_scale"`
#' - Treatment effect forest leaf scale: `"sigma2_leaf_tau"`, `"leaf_scale_tau"`, `"tau_leaf_scale"`
#' - Adaptive coding parameters: `"adaptive_coding"` (returns both the control and treated parameters jointly, with control in the first row and treated in the second row)
#' - In-sample mean function predictions: `"y_hat_train"`
#' - Test set mean function predictions: `"y_hat_test"`
#' - In-sample treatment effect forest predictions: `"tau_hat_train"`
#' - Test set treatment effect forest predictions: `"tau_hat_test"`
#' - Treatment effect intercept: `"tau_0"`, `"treatment_intercept"`, `"tau_intercept"`
#' - In-sample variance forest predictions: `"sigma2_x_train"`, `"var_x_train"`
#' - Test set variance forest predictions: `"sigma2_x_test"`, `"var_x_test"`
#'
#' @param object Object of type `bcfmodel` containing draws of a BCF model and associated sampling outputs.
#' @param term Name of the parameter to extract (e.g., `"sigma2"`, `"y_hat_train"`, etc.)
#' @return Array of parameter samples. If the underlying parameter is a scalar, this will be a vector of length `num_samples`.
#' If the underlying parameter is vector-valued, this will be (`parameter_dimension` x `num_samples`) matrix, and if the underlying
#' parameter is multidimensional, this will be an array of dimension (`parameter_dimension_1` x `parameter_dimension_2` x ... x `num_samples`).
#' @export
#'
#' @examples
#' n <- 500
#' p <- 5
#' X <- matrix(runif(n*p), ncol = p)
#' mu_x <- (
#'     ((0 <= X[,1]) & (0.25 > X[,1])) * (-7.5) +
#'     ((0.25 <= X[,1]) & (0.5 > X[,1])) * (-2.5) +
#'     ((0.5 <= X[,1]) & (0.75 > X[,1])) * (2.5) +
#'     ((0.75 <= X[,1]) & (1 > X[,1])) * (7.5)
#' )
#' pi_x <- (
#'     ((0 <= X[,1]) & (0.25 > X[,1])) * (0.2) +
#'     ((0.25 <= X[,1]) & (0.5 > X[,1])) * (0.4) +
#'     ((0.5 <= X[,1]) & (0.75 > X[,1])) * (0.6) +
#'     ((0.75 <= X[,1]) & (1 > X[,1])) * (0.8)
#' )
#' tau_x <- (
#'     ((0 <= X[,2]) & (0.25 > X[,2])) * (0.5) +
#'     ((0.25 <= X[,2]) & (0.5 > X[,2])) * (1.0) +
#'     ((0.5 <= X[,2]) & (0.75 > X[,2])) * (1.5) +
#'     ((0.75 <= X[,2]) & (1 > X[,2])) * (2.0)
#' )
#' Z <- rbinom(n, 1, pi_x)
#' E_XZ <- mu_x + Z*tau_x
#' snr <- 3
#' rfx_group_ids <- rep(c(1,2), n %/% 2)
#' rfx_coefs <- matrix(c(-1, -1, 1, 1), nrow=2, byrow=TRUE)
#' rfx_basis <- cbind(1, runif(n, -1, 1))
#' rfx_term <- rowSums(rfx_coefs[rfx_group_ids,] * rfx_basis)
#' y <- E_XZ + rfx_term + rnorm(n, 0, 1)*(sd(E_XZ)/snr)
#' bcf_model <- bcf(X_train = X, y_train = y, Z_train = Z,
#'                  rfx_group_ids_train = rfx_group_ids,
#'                  rfx_basis_train = rfx_basis,
#'                  num_gfr = 10, num_burnin = 0, num_mcmc = 10)
#' sigma2_samples <- extractParameter(bcf_model, "sigma2")
extractParameter.bcfmodel <- function(object, term) {
  if (term %in% c("sigma2", "global_error_scale", "sigma2_global")) {
    s <- object$samples$global_var_samples()
    if (length(s) > 0) {
      return(s)
    } else {
      stop("This model does not have global variance parameter samples")
    }
  }

  if (term %in% c("sigma2_leaf_mu", "leaf_scale_mu", "mu_leaf_scale")) {
    s <- object$samples$leaf_scale_mu_samples()
    if (length(s) > 0) {
      return(s)
    } else {
      stop(
        "This model does not have prognostic forest leaf variance parameter samples"
      )
    }
  }

  if (term %in% c("sigma2_leaf_tau", "leaf_scale_tau", "tau_leaf_scale")) {
    s <- object$samples$leaf_scale_tau_samples()
    if (length(s) > 0) {
      return(s)
    } else {
      stop(
        "This model does not have treatment effect forest leaf variance parameter samples"
      )
    }
  }

  if (term %in% c("adaptive_coding")) {
    if (object$model_params$adaptive_coding) {
      b0_samples <- object$samples$b0_samples()
      b1_samples <- object$samples$b1_samples()
      return(unname(rbind(b0_samples, b1_samples)))
    } else {
      stop("This model does not have adaptive coding parameter samples")
    }
  }

  if (term %in% c("y_hat_train")) {
    preds <- object$samples$y_hat_train()
    if (length(preds) > 0) {
      return(preds)
    } else {
      stop(
        "This model does not have in-sample mean function prediction samples"
      )
    }
  }

  if (term %in% c("y_hat_test")) {
    preds <- object$samples$y_hat_test()
    if (length(preds) > 0) {
      return(preds)
    } else {
      stop("This model does not have test set mean function prediction samples")
    }
  }

  if (term %in% c("mu_hat_train", "prognostic_function_train")) {
    preds <- object$samples$mu_forest_predictions_train()
    if (length(preds) > 0) {
      return(preds)
    } else {
      stop(
        "This model does not have in-sample prognostic function predictions"
      )
    }
  }

  if (term %in% c("mu_hat_test", "prognostic_function_test")) {
    preds <- object$samples$mu_forest_predictions_test()
    if (length(preds) > 0) {
      return(preds)
    } else {
      stop(
        "This model does not have test set prognostic function predictions"
      )
    }
  }

  # tau_hat / cate is the FULL CATE: the sampler folds tau_0 (and the adaptive-coding
  # (b1 - b0) scaling) into tau_forest_predictions before caching, so these accessors already
  # return tau_0 + tau(x) -- do NOT add tau_0 again here (see bcf_sampler.cpp postprocess / GH #376).
  if (term %in% c("tau_hat_train", "cate_train")) {
    preds <- object$samples$tau_forest_predictions_train()
    if (length(preds) > 0) {
      return(preds)
    } else {
      stop(
        "This model does not have in-sample treatment effect forest predictions"
      )
    }
  }

  if (term %in% c("tau_hat_test", "cate_test")) {
    preds <- object$samples$tau_forest_predictions_test()
    if (length(preds) > 0) {
      return(preds)
    } else {
      stop(
        "This model does not have test set treatment effect forest predictions"
      )
    }
  }

  if (term %in% c("sigma2_x_train", "sigma2_x_hat_train", "var_x_train")) {
    preds <- object$samples$variance_forest_predictions_train()
    if (length(preds) > 0) {
      return(preds)
    } else {
      stop("This model does not have in-sample variance forest predictions")
    }
  }

  if (term %in% c("sigma2_x_test", "sigma2_x_hat_test", "var_x_test")) {
    preds <- object$samples$variance_forest_predictions_test()
    if (length(preds) > 0) {
      return(preds)
    } else {
      stop("This model does not have test set variance forest predictions")
    }
  }

  if (term %in% c("tau_0", "treatment_intercept", "tau_intercept")) {
    s <- object$samples$tau_0_samples()
    if (length(s) > 0) {
      return(s)
    } else {
      stop(
        "This model does not have treatment effect intercept (tau_0) samples"
      )
    }
  }

  stop(paste0("term ", term, " is not a valid BCF model term"))
}

#' @title Extract Random Effect Samples from BCF Model
#' @description
#' Extract raw sample values for each of the random effect parameter terms.
#'
#' @param object Object of type `bcfmodel` containing draws of a Bayesian causal forest model and associated sampling outputs.
#' @param ... Other parameters to be used in random effects extraction
#' @return List of arrays. The alpha array has dimension (`num_components`, `num_samples`) and is simply a vector if `num_components = 1`.
#' The xi and beta arrays have dimension (`num_components`, `num_groups`, `num_samples`) and is simply a matrix if `num_components = 1`.
#' The sigma array has dimension (`num_components`, `num_samples`) and is simply a vector if `num_components = 1`.
#' @export
#'
#' @examples
#' n <- 500
#' p <- 5
#' X <- matrix(runif(n*p), ncol = p)
#' mu_x <- (
#'     ((0 <= X[,1]) & (0.25 > X[,1])) * (-7.5) +
#'     ((0.25 <= X[,1]) & (0.5 > X[,1])) * (-2.5) +
#'     ((0.5 <= X[,1]) & (0.75 > X[,1])) * (2.5) +
#'     ((0.75 <= X[,1]) & (1 > X[,1])) * (7.5)
#' )
#' pi_x <- (
#'     ((0 <= X[,1]) & (0.25 > X[,1])) * (0.2) +
#'     ((0.25 <= X[,1]) & (0.5 > X[,1])) * (0.4) +
#'     ((0.5 <= X[,1]) & (0.75 > X[,1])) * (0.6) +
#'     ((0.75 <= X[,1]) & (1 > X[,1])) * (0.8)
#' )
#' tau_x <- (
#'     ((0 <= X[,2]) & (0.25 > X[,2])) * (0.5) +
#'     ((0.25 <= X[,2]) & (0.5 > X[,2])) * (1.0) +
#'     ((0.5 <= X[,2]) & (0.75 > X[,2])) * (1.5) +
#'     ((0.75 <= X[,2]) & (1 > X[,2])) * (2.0)
#' )
#' Z <- rbinom(n, 1, pi_x)
#' E_XZ <- mu_x + Z*tau_x
#' snr <- 3
#' rfx_group_ids <- rep(c(1,2), n %/% 2)
#' rfx_coefs <- matrix(c(-1, -1, 1, 1), nrow=2, byrow=TRUE)
#' rfx_basis <- cbind(1, runif(n, -1, 1))
#' rfx_term <- rowSums(rfx_coefs[rfx_group_ids,] * rfx_basis)
#' y <- E_XZ + rfx_term + rnorm(n, 0, 1)*(sd(E_XZ)/snr)
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
#' rfx_group_ids_test <- rfx_group_ids[test_inds]
#' rfx_group_ids_train <- rfx_group_ids[train_inds]
#' rfx_basis_test <- rfx_basis[test_inds,]
#' rfx_basis_train <- rfx_basis[train_inds,]
#' rfx_term_test <- rfx_term[test_inds]
#' rfx_term_train <- rfx_term[train_inds]
#' mu_params <- list(sample_sigma2_leaf = TRUE)
#' tau_params <- list(sample_sigma2_leaf = FALSE)
#' bcf_model <- bcf(X_train = X_train, Z_train = Z_train, y_train = y_train,
#'                  propensity_train = pi_train,
#'                  rfx_group_ids_train = rfx_group_ids_train,
#'                  rfx_basis_train = rfx_basis_train, X_test = X_test,
#'                  Z_test = Z_test, propensity_test = pi_test,
#'                  rfx_group_ids_test = rfx_group_ids_test,
#'                  rfx_basis_test = rfx_basis_test,
#'                  num_gfr = 10, num_burnin = 0, num_mcmc = 10,
#'                  prognostic_forest_params = mu_params,
#'                  treatment_effect_forest_params = tau_params)
#' rfx_samples <- extractRandomEffectSamples(bcf_model)
extractRandomEffectSamples.bcfmodel <- function(object, ...) {
  result <- list()

  if (!object$model_params$has_rfx) {
    warning("This model has no RFX terms, returning an empty list")
    return(result)
  }

  # Extract the samples
  rfx_samples <- object$samples$materialize_rfx()
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

#' @title Extract Random Effects Samples from BCF Model (legacy alias)
#' @description Legacy alias for [extractRandomEffectSamples()]; delegates to it.
#' @param object Object of type `bcfmodel` containing draws of a BCF model and associated sampling outputs.
#' @param ... Other parameters to be used in random effects extraction
#' @return List of random effect samples (see [extractRandomEffectSamples()]).
#' @export
#' @examples
#' n <- 100
#' p <- 5
#' X <- matrix(runif(n*p), ncol = p)
#' pi_x <- 0.25 + 0.5*X[,1]
#' Z <- rbinom(n, 1, pi_x)
#' rfx_group_ids <- sample(1:2, size = n, replace = TRUE)
#' rfx_basis <- rep(1.0, n)
#' mu_x <- X[,1]*2
#' tau_x <- X[,2]*(-1)
#' y <- mu_x + tau_x*Z + (-2*(rfx_group_ids==1)+2*(rfx_group_ids==2)) + rnorm(n)
#' bcf_model <- bcf(X_train=X, Z_train=Z, y_train=y, propensity_train=pi_x,
#'                  rfx_group_ids_train=rfx_group_ids, rfx_basis_train=rfx_basis,
#'                  num_gfr=0, num_mcmc=10)
#' rfx_samples <- getRandomEffectSamples(bcf_model)
getRandomEffectSamples.bcfmodel <- function(object, ...) {
  extractRandomEffectSamples(object, ...)
}

#' @title Extract BCF Forests
#' @description Extract a forest from a BCF model by name.
#' If the requested forest type is not found, an error is thrown.
#' The following conventions are used for forest:
#' - Prognostic (mu) forest: `"prognostic"`, `"prognostic_forest"`, `"mu"`
#' - Treatment effect (tau) forest: `"treatment"`, `"treatment_forest"`, `"tau"`
#' - Variance forest: `"variance"`, `"variance_forest"`
#'
#' The treatment forest is the raw treatment-effect forest tau(x) (without `tau_0` or any
#' adaptive-coding scaling); for the full CATE use `extractParameter(object, "tau_hat_train")`.
#'
#' @param object Object of type `bcfmodel` containing draws of a BCF model and associated sampling outputs.
#' @param term Name of the forest to extract (e.g., `"prognostic"`, `"treatment"`, `"variance"`).
#' @return Object of class ForestSamples containing a deep copy of the requested forest samples.
#' @export
#'
#' @examples
#' n <- 100
#' p <- 5
#' X <- matrix(runif(n*p), ncol = p)
#' pi_x <- 0.25 + 0.5*X[,1]
#' Z <- rbinom(n, 1, pi_x)
#' mu_x <- X[,1]*2
#' tau_x <- X[,2]*(-1)
#' y <- mu_x + tau_x*Z + rnorm(n)
#' bcf_model <- bcf(X_train=X, Z_train=Z, y_train=y, propensity_train=pi_x,
#'                  num_gfr=0, num_mcmc=10)
#' prognostic_forest <- extractForest(bcf_model, "prognostic")
#' treatment_forest <- extractForest(bcf_model, "treatment")
extractForest.bcfmodel <- function(object, term) {
  if (term %in% c("prognostic", "prognostic_forest", "mu")) {
    if (object$samples$has_mu_forest()) {
      return(object$samples$materialize_mu_forest())
    } else {
      stop("This model does not have a prognostic (mu) forest")
    }
  }

  if (term %in% c("treatment", "treatment_forest", "tau")) {
    if (object$samples$has_tau_forest()) {
      return(object$samples$materialize_tau_forest())
    } else {
      stop("This model does not have a treatment effect (tau) forest")
    }
  }

  if (term %in% c("variance", "variance_forest")) {
    if (object$samples$has_variance_forest()) {
      return(object$samples$materialize_variance_forest())
    } else {
      stop("This model does not have a variance forest")
    }
  }

  stop(paste0("term ", term, " is not a valid BCF forest term"))
}

#' @title Continue Sampling a BCF Model
#' @description Continue sampling from a BCF model, appending additional draws to the existing
#' posterior samples. The training data must be re-supplied (it is not retained on the model).
#' The sampler initializes every term (prognostic + treatment forests, variance forest,
#' random effects, tau_0, and adaptive-coding b0/b1) from its last retained sample.
#'
#' @param object Fitted `bcfmodel` to continue sampling.
#' @param X_train Training covariates (re-supplied; same structure used to fit the model).
#' @param Z_train Training treatment assignments (re-supplied).
#' @param y_train Training outcome (re-supplied).
#' @param propensity_train (Optional) Training propensity scores. Required if the model used a propensity
#'   covariate and no internal propensity model is available to re-derive them.
#' @param rfx_group_ids_train (Optional) Training random effects group labels (required if the model has rfx).
#' @param rfx_basis_train (Optional) Training random effects basis (required for a custom rfx model).
#' @param X_test (Optional) Test covariates. When supplied, test-set predictions are recomputed in full
#'   from all retained forests, so the test set need not match any test set used in the original fit (the
#'   model may have been fit with none). When omitted, any cached test predictions are dropped.
#' @param Z_test (Optional) Test treatment assignments (required when `X_test` is provided).
#' @param propensity_test (Optional) Test propensity scores. Required if the model used a propensity covariate
#'   and no internal propensity model is available to re-derive them.
#' @param rfx_group_ids_test (Optional) Test random effects group labels (required when `X_test` is provided and the model has rfx).
#' @param rfx_basis_test (Optional) Test random effects basis (required for a custom rfx model when `X_test` is provided).
#' @param num_burnin Number of additional burn-in iterations to discard. Default `0`.
#' @param num_mcmc Number of additional retained MCMC draws. Default `100`.
#' @param general_params (Optional) List of changeable general parameters (e.g. `random_seed`, `keep_every`,
#'   `keep_burnin`, `cutpoint_grid_size`, `sigma2_global_shape`, `sigma2_global_scale`, `num_threads`, `verbose`).
#'   May also include `variable_weights` (per-covariate split weights); if omitted, the fit-time weights are reused.
#' @param prognostic_forest_params (Optional) Changeable prognostic (mu) forest parameters (`alpha`, `beta`,
#'   `min_samples_leaf`, `max_depth`, `num_features_subsample`, `sigma2_leaf_shape`, `sigma2_leaf_scale`).
#'   May also include `keep_vars` / `drop_vars` to change which covariates this forest may split on; if omitted,
#'   the fit-time split-variable subset is reused.
#' @param treatment_effect_forest_params (Optional) Changeable treatment (tau) forest parameters (same keys as
#'   prognostic, including `keep_vars` / `drop_vars`).
#' @param variance_forest_params (Optional) Changeable variance forest parameters (`alpha`, `beta`,
#'   `min_samples_leaf`, `max_depth`, `num_features_subsample`, `var_forest_prior_shape`, `var_forest_prior_scale`,
#'   and `keep_vars` / `drop_vars`).
#' @param random_effects_params (Optional) Changeable random effects parameters (`variance_prior_shape`,
#'   `variance_prior_scale` — the inverse-gamma prior on the random effects group-parameter variance).
#'   Ignored if the model has no random effects.
#' @param ... Other parameters (ignored).
#' @return The updated `bcfmodel` (mutated in place; the sampled forests are extended).
#' @export
continueSampling.bcfmodel <- function(
  object,
  X_train,
  Z_train,
  y_train,
  propensity_train = NULL,
  rfx_group_ids_train = NULL,
  rfx_basis_train = NULL,
  X_test = NULL,
  Z_test = NULL,
  propensity_test = NULL,
  rfx_group_ids_test = NULL,
  rfx_basis_test = NULL,
  num_burnin = 0,
  num_mcmc = 100,
  general_params = list(),
  prognostic_forest_params = list(),
  treatment_effect_forest_params = list(),
  variance_forest_params = list(),
  random_effects_params = list(),
  ...
) {
  if (!inherits(object, "bcfmodel")) {
    stop("object must be a bcfmodel")
  }
  bcf_config <- object[["bcf_config"]]
  if (is.null(bcf_config)) {
    stop(
      "Cannot continue sampling: the cached sampler configuration is unavailable. ",
      "Continuation is not supported for deserialized models (a model loaded from JSON)."
    )
  }
  if (object$model_params$outcome_model$link == "cloglog") {
    stop("Continued sampling is not yet supported for cloglog link functions")
  }

  # Update any changeable model parameters
  general_map <- list(
    random_seed = "random_seed",
    keep_burnin = "keep_burnin",
    cutpoint_grid_size = "cutpoint_grid_size",
    sigma2_global_shape = "a_sigma2_global",
    sigma2_global_scale = "b_sigma2_global",
    num_threads = "num_threads",
    verbose = "verbose"
  )
  prognostic_map <- list(
    alpha = "alpha_mu",
    beta = "beta_mu",
    min_samples_leaf = "min_samples_leaf_mu",
    max_depth = "max_depth_mu",
    num_features_subsample = "num_features_subsample_mu",
    sigma2_leaf_shape = "a_sigma2_mu",
    sigma2_leaf_scale = "b_sigma2_mu"
  )
  treatment_map <- list(
    alpha = "alpha_tau",
    beta = "beta_tau",
    min_samples_leaf = "min_samples_leaf_tau",
    max_depth = "max_depth_tau",
    num_features_subsample = "num_features_subsample_tau",
    sigma2_leaf_shape = "a_sigma2_tau",
    sigma2_leaf_scale = "b_sigma2_tau"
  )
  variance_map <- list(
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
  # keep_every is passed to the binding as an explicit arg (not a config key); pull it out so the
  # overlay does not flag it as unchangeable.
  user_keep_every <- general_params[["keep_every"]]
  general_params[["keep_every"]] <- NULL
  # Split-variable configuration overrides. These are re-derived below (not overlaid as raw config
  # keys), so pull them out of their params lists before the overlay so they are not flagged as
  # unchangeable. When left NULL, the cached fit-time values are reused.
  user_variable_weights <- general_params[["variable_weights"]]
  general_params[["variable_weights"]] <- NULL
  keep_vars_mu <- prognostic_forest_params[["keep_vars"]]
  drop_vars_mu <- prognostic_forest_params[["drop_vars"]]
  prognostic_forest_params[["keep_vars"]] <- NULL
  prognostic_forest_params[["drop_vars"]] <- NULL
  keep_vars_tau <- treatment_effect_forest_params[["keep_vars"]]
  drop_vars_tau <- treatment_effect_forest_params[["drop_vars"]]
  treatment_effect_forest_params[["keep_vars"]] <- NULL
  treatment_effect_forest_params[["drop_vars"]] <- NULL
  keep_vars_variance <- variance_forest_params[["keep_vars"]]
  drop_vars_variance <- variance_forest_params[["drop_vars"]]
  variance_forest_params[["keep_vars"]] <- NULL
  variance_forest_params[["drop_vars"]] <- NULL
  config <- bcf_config
  config <- overlayContinuationParams(
    config,
    general_params,
    general_map,
    "general_params"
  )
  config <- overlayContinuationParams(
    config,
    prognostic_forest_params,
    prognostic_map,
    "prognostic_forest_params"
  )
  config <- overlayContinuationParams(
    config,
    treatment_effect_forest_params,
    treatment_map,
    "treatment_effect_forest_params"
  )
  config <- overlayContinuationParams(
    config,
    variance_forest_params,
    variance_map,
    "variance_forest_params"
  )
  config <- overlayContinuationParams(
    config,
    random_effects_params,
    random_effects_map,
    "random_effects_params"
  )
  keep_every <- if (!is.null(user_keep_every)) user_keep_every else 1

  # Cached split-variable state is required to re-derive per-forest weights on continuation.
  cstate <- object$continuation_state
  if (is.null(cstate)) {
    stop(
      "Cannot continue sampling: cached continuation state is unavailable for this model."
    )
  }

  # Update training data
  train_set_metadata <- object$train_set_metadata
  if (ncol(X_train) != object$model_params$num_covariates) {
    stop(sprintf(
      "X_train has %d columns; the model was trained on %d covariates",
      ncol(X_train),
      object$model_params$num_covariates
    ))
  }
  # Keep the raw (un-preprocessed) covariates so name-based keep_vars / drop_vars resolve against the original column names
  X_train_raw <- X_train
  X_train <- preprocessPredictionData(X_train, train_set_metadata)
  y_train <- as.numeric(y_train)
  Z_train <- as.matrix(Z_train)
  if (nrow(X_train) != length(y_train) || nrow(X_train) != nrow(Z_train)) {
    stop(
      "X_train, Z_train, and y_train must have the same number of observations"
    )
  }

  # Propensity: re-derive from the internal model if not supplied.
  if (
    object$model_params$propensity_covariate != "none" &&
      is.null(propensity_train)
  ) {
    if (!object$model_params$internal_propensity_model) {
      stop("propensity_train must be provided to continue sampling this model")
    }
    propensity_train <- predict(
      object$bart_propensity_model,
      X_train,
      type = "mean",
      terms = "y_hat"
    )
  }

  # Random effects, encoded with the model's stored factor levels.
  has_rfx <- object$model_params$has_rfx
  rfx_num_groups <- 0L
  rfx_basis_dim <- 0L
  if (has_rfx) {
    if (is.null(rfx_group_ids_train)) {
      stop(
        "This model was fit with random effects; rfx_group_ids_train must be supplied to continue sampling"
      )
    }
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
    spec <- object$model_params$rfx_model_spec
    if (spec == "custom" && is.null(rfx_basis_train)) {
      stop(
        "A user-provided rfx_basis_train must be supplied for a 'custom' random effects model"
      )
    } else if (spec == "intercept_only" && is.null(rfx_basis_train)) {
      rfx_basis_train <- matrix(rep(1, nrow(X_train)), ncol = 1)
    } else if (spec == "intercept_plus_treatment" && is.null(rfx_basis_train)) {
      rfx_basis_train <- cbind(rep(1, nrow(X_train)), Z_train)
    }
  }

  # Append propensity to the covariates exactly as bcf() / predict.bcfmodel do.
  X_combined <- X_train
  if (object$model_params$propensity_covariate != "none") {
    X_combined <- cbind(X_train, propensity_train)
  }

  # If the user overrode variable_weights / keep_vars / drop_vars for any forest, recompute that
  # forest's weights; otherwise reuse the cached fit-time values. The full BCF weight pipeline
  # (expand across preprocessed features, append propensity columns, renormalize) is shared with
  # bcf() via computeBCFForestWeights().
  num_cov_orig <- object$model_params$num_covariates
  variable_weights <- if (!is.null(user_variable_weights)) {
    user_variable_weights
  } else {
    cstate$variable_weights
  }
  if (length(variable_weights) != num_cov_orig) {
    stop(sprintf(
      "variable_weights must have length %d (the number of covariates)",
      num_cov_orig
    ))
  }
  if (any(variable_weights < 0)) {
    stop("variable_weights cannot have any negative weights")
  }
  variable_subset_mu <- if (!is.null(keep_vars_mu) || !is.null(drop_vars_mu)) {
    resolveVariableSubset(keep_vars_mu, drop_vars_mu, X_train_raw, "mu")
  } else {
    cstate$variable_subset_mu
  }
  variable_subset_tau <- if (
    !is.null(keep_vars_tau) || !is.null(drop_vars_tau)
  ) {
    resolveVariableSubset(keep_vars_tau, drop_vars_tau, X_train_raw, "tau")
  } else {
    cstate$variable_subset_tau
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
  include_variance_forest <- object$model_params$include_variance_forest
  variable_weights_list <- computeBCFForestWeights(
    variable_weights = variable_weights,
    original_var_indices = train_set_metadata$original_var_indices,
    variable_subset_mu = variable_subset_mu,
    variable_subset_tau = variable_subset_tau,
    variable_subset_variance = variable_subset_variance,
    propensity_covariate = object$model_params$propensity_covariate,
    ncol_propensity = ncol(X_combined) - ncol(X_train),
    num_cov_orig = num_cov_orig,
    include_variance_forest = include_variance_forest
  )
  config[["var_weights_mu"]] <- variable_weights_list$mu
  config[["var_weights_tau"]] <- variable_weights_list$tau
  if (include_variance_forest) {
    config[["var_weights_variance"]] <- variable_weights_list$variance
  }

  # --- Re-supplied test data (optional) -----------------------------------------------------------
  # When X_test is supplied, the sampler recomputes the full test-prediction trace from all retained
  # forests (assign, not append), so the test set need not match any test set used in the original fit.
  # When it is NOT supplied, any cached test predictions from the original fit are stale and the
  # sampler drops them (postprocess clears test predictions when no test set is present).
  has_test <- !is.null(X_test)
  X_test_combined <- NULL
  rfx_group_ids_test_int <- NULL
  n_test <- 0L
  if (!has_test && isTRUE(object$model_params$has_test)) {
    warning(
      "Continuing without X_test on a model fit with a test set: the existing test-set predictions ",
      "are stale and will be dropped. Re-supply X_test to retain test-set predictions."
    )
  }
  if (has_test) {
    if (ncol(X_test) != object$model_params$num_covariates) {
      stop(sprintf(
        "X_test has %d columns; the model was trained on %d covariates",
        ncol(X_test),
        object$model_params$num_covariates
      ))
    }
    X_test <- preprocessPredictionData(X_test, train_set_metadata)
    if (is.null(Z_test)) {
      stop("Z_test must be supplied when X_test is provided")
    }
    Z_test <- as.matrix(Z_test)
    if (nrow(X_test) != nrow(Z_test)) {
      stop("X_test and Z_test must have the same number of observations")
    }
    # Test propensity: re-derive from the internal model if not supplied.
    if (
      object$model_params$propensity_covariate != "none" &&
        is.null(propensity_test)
    ) {
      if (!object$model_params$internal_propensity_model) {
        stop(
          "propensity_test must be provided to continue sampling this model with a test set"
        )
      }
      propensity_test <- predict(
        object$bart_propensity_model,
        X_test,
        type = "mean",
        terms = "y_hat"
      )
    }
    X_test_combined <- X_test
    if (object$model_params$propensity_covariate != "none") {
      X_test_combined <- cbind(X_test, propensity_test)
    }
    n_test <- nrow(X_test_combined)
    if (has_rfx) {
      if (is.null(rfx_group_ids_test)) {
        stop(
          "This model was fit with random effects; rfx_group_ids_test must be supplied when X_test is provided"
        )
      }
      group_ids_test_factor <- factor(
        rfx_group_ids_test,
        levels = object$rfx_unique_group_ids
      )
      if (any(is.na(group_ids_test_factor))) {
        stop(
          "rfx_group_ids_test contains group labels not present in the fitted model"
        )
      }
      rfx_group_ids_test_int <- as.integer(group_ids_test_factor)
      if (spec == "custom" && is.null(rfx_basis_test)) {
        stop(
          "A user-provided rfx_basis_test must be supplied for a 'custom' random effects model"
        )
      } else if (spec == "intercept_only" && is.null(rfx_basis_test)) {
        rfx_basis_test <- matrix(rep(1, nrow(X_test)), ncol = 1)
      } else if (spec == "intercept_plus_treatment" && is.null(rfx_basis_test)) {
        rfx_basis_test <- cbind(rep(1, nrow(X_test)), Z_test)
      }
    }
  }

  # Update or restore RNG state
  override_seed <- !is.null(general_params$random_seed)
  rng_state_in <- if (
    !override_seed && !is.null(object$model_params$rng_state)
  ) {
    object$model_params$rng_state
  } else {
    ""
  }

  # Continue running the sampler with in-place updates to the samples object
  bcf_samples <- object$samples
  bcf_metadata <- bcf_continue_sample_cpp(
    samples = bcf_samples$samples_ptr,
    X_train = X_combined,
    Z_train = Z_train,
    y_train = y_train,
    X_test = if (has_test) X_test_combined else NULL,
    Z_test = if (has_test) Z_test else NULL,
    n_train = nrow(X_combined),
    n_test = as.integer(n_test),
    p = ncol(X_combined),
    treatment_dim = ncol(Z_train),
    obs_weights_train = NULL,
    rfx_group_ids_train = if (has_rfx) rfx_group_ids_train else NULL,
    rfx_basis_train = if (has_rfx) rfx_basis_train else NULL,
    rfx_group_ids_test = if (has_test && has_rfx) rfx_group_ids_test_int else NULL,
    rfx_basis_test = if (has_test && has_rfx) rfx_basis_test else NULL,
    rfx_num_groups = as.integer(rfx_num_groups),
    rfx_basis_dim = as.integer(rfx_basis_dim),
    num_burnin = as.integer(num_burnin),
    keep_every = as.integer(keep_every),
    num_mcmc = as.integer(num_mcmc),
    rng_state_in = rng_state_in,
    override_seed = override_seed,
    config_input = config
  )

  # Update model metadata
  object$model_params$num_samples <- bcf_samples$num_samples()
  object$model_params$num_mcmc <- object$model_params$num_mcmc + num_mcmc
  object$model_params$rng_state <- bcf_metadata[["rng_state"]]
  object$model_params$has_test <- has_test
  object$model_params$num_test <- if (has_test) n_test else 0L

  return(object)
}

#' @title Convert BCF Model to JSON
#' @param object Object of type `bcfmodel` containing draws of a Bayesian causal forest model and associated sampling outputs.
#' @export
#' @rdname BCFSerialization
saveBCFModelToJson <- function(object) {
  jsonobj <- createCppJson()

  if (!inherits(object, "bcfmodel")) {
    stop("`object` must be a BCF model")
  }

  if (is.null(object$model_params)) {
    stop("This BCF model has not yet been sampled")
  }

  # Add the samples to the JSON object
  bcf_samples <- object$samples
  bcf_samples$append_to_json(jsonobj)

  # Add version stamp and global parameters
  jsonobj$add_string("stochtree_version", getStochtreeVersion())
  jsonobj$add_string("platform", "R")
  jsonobj$add_integer("schema_version", STOCHTREE_SCHEMA_VERSION)
  jsonobj$add_scalar("outcome_scale", object$model_params$outcome_scale)
  jsonobj$add_scalar("outcome_mean", object$model_params$outcome_mean)
  jsonobj$add_boolean("standardize", object$model_params$standardize)
  jsonobj$add_scalar("sigma2_init", object$model_params$initial_sigma2)
  jsonobj$add_boolean(
    "sample_sigma2_global",
    object$model_params$sample_sigma2_global
  )
  jsonobj$add_boolean(
    "sample_sigma2_leaf_mu",
    object$model_params$sample_sigma2_leaf_mu
  )
  jsonobj$add_boolean(
    "sample_sigma2_leaf_tau",
    object$model_params$sample_sigma2_leaf_tau
  )
  jsonobj$add_boolean(
    "include_variance_forest",
    object$model_params$include_variance_forest
  )
  jsonobj$add_string(
    "propensity_covariate",
    object$model_params$propensity_covariate
  )
  jsonobj$add_boolean("has_rfx", object$model_params$has_rfx)
  jsonobj$add_boolean("has_rfx_basis", object$model_params$has_rfx_basis)
  jsonobj$add_scalar("num_rfx_basis", object$model_params$num_rfx_basis)
  jsonobj$add_boolean(
    "multivariate_treatment",
    object$model_params$multivariate_treatment
  )
  jsonobj$add_boolean("adaptive_coding", object$model_params$adaptive_coding)
  jsonobj$add_boolean(
    "binary_treatment",
    object$model_params$binary_treatment
  )
  jsonobj$add_scalar("treatment_dim", object$model_params$treatment_dim)
  jsonobj$add_boolean("sample_tau_0", object$model_params$sample_tau_0)
  jsonobj$add_boolean(
    "internal_propensity_model",
    object$model_params$internal_propensity_model
  )
  jsonobj$add_scalar("num_gfr", object$model_params$num_gfr)
  jsonobj$add_scalar("num_burnin", object$model_params$num_burnin)
  jsonobj$add_scalar("num_mcmc", object$model_params$num_mcmc)
  jsonobj$add_scalar("num_samples", object$model_params$num_samples)
  jsonobj$add_scalar("keep_every", object$model_params$keep_every)
  jsonobj$add_scalar("num_chains", object$model_params$num_chains)
  jsonobj$add_scalar("num_covariates", object$model_params$num_covariates)
  jsonobj$add_boolean(
    "probit_outcome_model",
    object$model_params$probit_outcome_model
  )
  outcome_model_outcome <- object$model_params$outcome_model$outcome
  outcome_model_link <- object$model_params$outcome_model$link
  jsonobj$add_string(
    "outcome",
    outcome_model_outcome,
    "outcome_model"
  )
  jsonobj$add_string(
    "link",
    outcome_model_link,
    "outcome_model"
  )

  # Add random effects group IDs
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
  jsonobj$add_string(
    "rfx_model_spec",
    object$model_params$rfx_model_spec
  )

  # Add propensity model (if it exists)
  if (object$model_params$internal_propensity_model) {
    bart_propensity_string <- saveBARTModelToJsonString(
      object$bart_propensity_model
    )
    jsonobj$add_string("bart_propensity_model", bart_propensity_string)
  }

  # Add covariate preprocessor metadata
  preprocessor_metadata_string <- savePreprocessorToJsonString(
    object$train_set_metadata
  )
  jsonobj$add_string("covariate_preprocessor", preprocessor_metadata_string)

  return(jsonobj)
}

#' @title Save BCF Model to JSON File
#' @param object Object of type `bcfmodel` containing draws of a Bayesian causal forest model and associated sampling outputs.
#' @param filename String of filepath, must end in ".json"
#' @export
#' @rdname BCFSerialization
saveBCFModelToJsonFile <- function(object, filename) {
  # Convert to Json
  jsonobj <- saveBCFModelToJson(object)

  # Save to file
  jsonobj$save_file(filename)
}

#' @title Convert BCF Model to JSON String
#' @param object Object of type `bcfmodel` containing draws of a Bayesian causal forest model and associated sampling outputs.
#' @export
#' @rdname BCFSerialization
saveBCFModelToJsonString <- function(object) {
  # Convert to Json
  jsonobj <- saveBCFModelToJson(object)

  # Dump to string
  return(jsonobj$return_json_string())
}

# Recover `binary_treatment` for legacy BCF JSON written before the field was
# serialized. Two sample-time invariants make this exact in most cases:
#   * multivariate treatment is never binary, and
#   * adaptive_coding is forced FALSE unless the treatment is binary, so
#     adaptive_coding == TRUE implies binary_treatment == TRUE.
# A univariate, default-coded treatment is genuinely ambiguous from JSON alone
# (it would require the original Z_train), so we conservatively return FALSE.
# Reads directly from the JSON object so it does not depend on the order in
# which the calling load path populates model_params.
.inferBinaryTreatmentFromJson <- function(json_obj, has_field_fn) {
  multivariate <- if (has_field_fn("multivariate_treatment")) {
    json_obj$get_boolean("multivariate_treatment")
  } else {
    FALSE
  }
  if (isTRUE(multivariate)) {
    return(FALSE)
  }
  adaptive <- if (has_field_fn("adaptive_coding")) {
    json_obj$get_boolean("adaptive_coding")
  } else {
    FALSE
  }
  isTRUE(adaptive)
}

# In-place v0 -> v1 migration for a BCF model envelope: positional forest keys ->
# named keys (forest_0 -> prognostic_forest, forest_1 -> treatment_forest, and when
# present forest_2 -> variance_forest). The mu/tau forests are always present.
.migrateBcfJsonV0ToV1 <- function(json_object, loaded_version) {
  json_object$add_string("platform", inferPlatformV0(json_object, "R"))
  json_object$rename_field(
    "forest_0",
    "prognostic_forest",
    subfolder_name = "forests"
  )
  json_object$rename_field(
    "forest_1",
    "treatment_forest",
    subfolder_name = "forests"
  )
  if (json_object$get_boolean_or_default("include_variance_forest", FALSE)) {
    json_object$rename_field(
      "forest_2",
      "variance_forest",
      subfolder_name = "forests"
    )
  }
  # R's legacy preprocessor key -> unified v1 key (no-op for Python v0 JSON,
  # which already uses `covariate_preprocessor`).
  json_object$rename_field("preprocessor_metadata", "covariate_preprocessor")
  # Legacy parameter-trace field names -> canonical v1 names. All no-ops when the
  # field is absent (e.g. non-adaptive-coding models have no b0/b1 samples).
  json_object$rename_field("initial_sigma2", "sigma2_init")
  json_object$rename_field(
    "b_0_samples",
    "b0_samples",
    subfolder_name = "parameters"
  )
  json_object$rename_field(
    "b_1_samples",
    "b1_samples",
    subfolder_name = "parameters"
  )
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

#' @description Create a BCFSamples object from JSON
#' @noRd
createBCFSamplesFromJson <- function(json) {
  bcf_samples <- BCFSamples$new()
  bcf_samples$from_json(json)
  bcf_samples
}

#' @title Convert JSON to BCF Model
#' @param json_object Object of type `CppJson` containing Json representation of a BCF model
#' @export
#' @rdname BCFSerialization
createBCFModelFromJson <- function(json_object) {
  # Initialize the BCF model
  output <- list()

  # Version inference and presence-check helpers
  .ver <- inferStochtreeJsonVersion(json_object)
  resolveSchemaVersion(json_object, migrate = .migrateBcfJsonV0ToV1)
  cross_platform <- enforceCrossPlatformGate(json_object, "R")
  has_field <- function(name) {
    json_contains_field_cpp(json_object$json_ptr, name)
  }
  has_subfolder_field <- function(subfolder, name) {
    json_contains_field_subfolder_cpp(json_object$json_ptr, subfolder, name)
  }

  # Unpack model params
  model_params <- list()
  include_variance_forest <- json_object$get_boolean(
    "include_variance_forest"
  )
  model_params[["outcome_scale"]] <- json_object$get_scalar("outcome_scale")
  model_params[["outcome_mean"]] <- json_object$get_scalar("outcome_mean")
  model_params[["standardize"]] <- json_object$get_boolean("standardize")
  # Legacy `initial_sigma2` -> `sigma2_init` is handled by the v0 -> v1 migration
  # (.migrateBcfJsonV0ToV1), so by this point the canonical key is always present.
  model_params[["initial_sigma2"]] <- json_object$get_scalar("sigma2_init")
  model_params[["sample_sigma2_global"]] <- json_object$get_boolean(
    "sample_sigma2_global"
  )
  model_params[["sample_sigma2_leaf_mu"]] <- json_object$get_boolean(
    "sample_sigma2_leaf_mu"
  )
  model_params[["sample_sigma2_leaf_tau"]] <- json_object$get_boolean(
    "sample_sigma2_leaf_tau"
  )
  model_params[["include_variance_forest"]] <- include_variance_forest
  model_params[["propensity_covariate"]] <- json_object$get_string(
    "propensity_covariate"
  )
  model_params[["has_rfx"]] <- json_object$get_boolean("has_rfx")
  if (has_field("has_rfx_basis")) {
    model_params[["has_rfx_basis"]] <- json_object$get_boolean("has_rfx_basis")
    model_params[["num_rfx_basis"]] <- json_object$get_scalar("num_rfx_basis")
  } else {
    model_params[["has_rfx_basis"]] <- FALSE
    model_params[["num_rfx_basis"]] <- 1
    warning(sprintf(
      "Fields 'has_rfx_basis' and 'num_rfx_basis' not found in BCF JSON (inferred version: %s). Defaulting to has_rfx_basis=FALSE, num_rfx_basis=1.",
      .ver
    ))
  }
  model_params[["adaptive_coding"]] <- json_object$get_boolean(
    "adaptive_coding"
  )
  if (has_field("binary_treatment")) {
    model_params[["binary_treatment"]] <- json_object$get_boolean(
      "binary_treatment"
    )
  } else {
    model_params[["binary_treatment"]] <- .inferBinaryTreatmentFromJson(
      json_object,
      has_field
    )
    warning(sprintf(
      "Field 'binary_treatment' not found in BCF JSON (inferred version: %s). Inferred binary_treatment=%s from other JSON fields.",
      .ver,
      model_params[["binary_treatment"]]
    ))
  }
  if (has_field("treatment_dim")) {
    model_params[["treatment_dim"]] <- json_object$get_scalar("treatment_dim")
  } else {
    model_params[["treatment_dim"]] <- 1
    if (
      has_field("multivariate_treatment") &&
        isTRUE(json_object$get_boolean("multivariate_treatment"))
    ) {
      warning(sprintf(
        "Field 'treatment_dim' not found in BCF JSON (inferred version: %s) for a multivariate-treatment model. Defaulting to 1.",
        .ver
      ))
    }
  }
  if (has_field("sample_tau_0")) {
    model_params[["sample_tau_0"]] <- json_object$get_boolean("sample_tau_0")
  } else {
    model_params[["sample_tau_0"]] <- FALSE
    warning(sprintf(
      "Field 'sample_tau_0' not found in BCF JSON (inferred version: %s). Defaulting to FALSE.",
      .ver
    ))
  }
  if (has_field("multivariate_treatment")) {
    model_params[["multivariate_treatment"]] <- json_object$get_boolean(
      "multivariate_treatment"
    )
  } else {
    model_params[["multivariate_treatment"]] <- FALSE
    warning(sprintf(
      "Field 'multivariate_treatment' not found in BCF JSON (inferred version: %s). Defaulting to FALSE.",
      .ver
    ))
  }
  if (has_field("internal_propensity_model")) {
    model_params[["internal_propensity_model"]] <- json_object$get_boolean(
      "internal_propensity_model"
    )
  } else {
    model_params[["internal_propensity_model"]] <- FALSE
    warning(sprintf(
      "Field 'internal_propensity_model' not found in BCF JSON (inferred version: %s). Defaulting to FALSE.",
      .ver
    ))
  }
  model_params[["num_gfr"]] <- json_object$get_scalar("num_gfr")
  model_params[["num_burnin"]] <- json_object$get_scalar("num_burnin")
  model_params[["num_mcmc"]] <- json_object$get_scalar("num_mcmc")
  model_params[["num_samples"]] <- json_object$get_scalar("num_samples")
  model_params[["num_covariates"]] <- json_object$get_scalar("num_covariates")
  if (has_field("num_chains")) {
    model_params[["num_chains"]] <- json_object$get_scalar("num_chains")
  } else {
    model_params[["num_chains"]] <- 1
    warning(sprintf(
      "Field 'num_chains' not found in BCF JSON (inferred version: %s). Defaulting to 1.",
      .ver
    ))
  }
  if (has_field("keep_every")) {
    model_params[["keep_every"]] <- json_object$get_scalar("keep_every")
  } else {
    model_params[["keep_every"]] <- 1
    warning(sprintf(
      "Field 'keep_every' not found in BCF JSON (inferred version: %s). Defaulting to 1.",
      .ver
    ))
  }
  if (has_field("probit_outcome_model")) {
    model_params[["probit_outcome_model"]] <- json_object$get_boolean(
      "probit_outcome_model"
    )
  } else {
    model_params[["probit_outcome_model"]] <- FALSE
    warning(sprintf(
      "Field 'probit_outcome_model' not found in BCF JSON (inferred version: %s). Defaulting to FALSE.",
      .ver
    ))
  }
  if (has_subfolder_field("outcome_model", "outcome")) {
    outcome_model_outcome <- json_object$get_string("outcome", "outcome_model")
    outcome_model_link <- json_object$get_string("link", "outcome_model")
  } else {
    outcome_model_outcome <- "continuous"
    outcome_model_link <- "identity"
    warning(sprintf(
      "Subfolder 'outcome_model' not found in BCF JSON (inferred version: %s). Defaulting to outcome='continuous', link='identity'.",
      .ver
    ))
  }
  model_params[["outcome_model"]] <- OutcomeModel(
    outcome = outcome_model_outcome,
    link = outcome_model_link
  )
  if (has_field("rfx_model_spec")) {
    model_params[["rfx_model_spec"]] <- json_object$get_string(
      "rfx_model_spec"
    )
  } else {
    model_params[["rfx_model_spec"]] <- ""
    if (model_params[["has_rfx"]]) {
      warning(sprintf(
        "Field 'rfx_model_spec' not found in BCF JSON (inferred version: %s) but has_rfx=TRUE.",
        .ver
      ))
    }
  }
  output[["model_params"]] <- model_params

  # Unpack samples
  output[["samples"]] <- createBCFSamplesFromJson(json_object)

  # Unpack random effects group IDs
  if (model_params[["has_rfx"]]) {
    output[["rfx_unique_group_ids"]] <- resolveRfxUniqueGroupIds(
      json_object,
      output[["samples"]]$materialize_rfx()
    )
  }

  # Unpack propensity model (if it exists)
  if (model_params[["internal_propensity_model"]]) {
    bart_propensity_string <- json_object$get_string(
      "bart_propensity_model"
    )
    output[["bart_propensity_model"]] <- createBARTModelFromJsonString(
      bart_propensity_string
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
    warning(sprintf(
      "Field 'covariate_preprocessor' not found in BCF JSON (inferred version: %s). Preprocessor is unavailable; prediction may fail.",
      .ver
    ))
  }

  class(output) <- "bcfmodel"
  return(output)
}

#' @title Convert JSON File to BCF Model
#' @param json_filename String of filepath, must end in ".json"
#' @export
#' @rdname BCFSerialization
createBCFModelFromJsonFile <- function(json_filename) {
  # Load a `CppJson` object from file
  bcf_json <- createCppJsonFile(json_filename)

  # Create and return the BCF object
  bcf_object <- createBCFModelFromJson(bcf_json)

  return(bcf_object)
}

#' @title Convert JSON String to BCF Model
#' @param json_string JSON string dump
#' @export
#' @rdname BCFSerialization
createBCFModelFromJsonString <- function(json_string) {
  # Load a `CppJson` object from string
  bcf_json <- createCppJsonString(json_string)

  # Create and return the BCF object
  bcf_object <- createBCFModelFromJson(bcf_json)

  return(bcf_object)
}

#' @title Convert JSON List to BCF Model
#' @param json_object_list List of objects of type `CppJson` containing Json representation of a BCF model
#' @export
#' @rdname BCFSerialization
createBCFModelFromCombinedJson <- function(json_object_list) {
  # Initialize the BCF model
  output <- list()

  # For scalar / preprocessing details which aren't sample-dependent,
  # defer to the first json
  json_object_default <- json_object_list[[1]]

  # Version inference and presence-check helpers
  .ver <- inferStochtreeJsonVersion(json_object_default)
  for (.jo in json_object_list) {
    resolveSchemaVersion(.jo, migrate = .migrateBcfJsonV0ToV1)
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
  model_params <- list()
  include_variance_forest <- json_object_default$get_boolean(
    "include_variance_forest"
  )
  model_params[["outcome_scale"]] <- json_object_default$get_scalar(
    "outcome_scale"
  )
  model_params[["outcome_mean"]] <- json_object_default$get_scalar(
    "outcome_mean"
  )
  model_params[["standardize"]] <- json_object_default$get_boolean(
    "standardize"
  )
  # Legacy `initial_sigma2` -> `sigma2_init` is handled by the v0 -> v1 migration
  # (.migrateBcfJsonV0ToV1), so by this point the canonical key is always present.
  model_params[["initial_sigma2"]] <- json_object_default$get_scalar(
    "sigma2_init"
  )
  model_params[["sample_sigma2_global"]] <- json_object_default$get_boolean(
    "sample_sigma2_global"
  )
  model_params[["sample_sigma2_leaf_mu"]] <- json_object_default$get_boolean(
    "sample_sigma2_leaf_mu"
  )
  model_params[["sample_sigma2_leaf_tau"]] <- json_object_default$get_boolean(
    "sample_sigma2_leaf_tau"
  )
  model_params[["include_variance_forest"]] <- include_variance_forest
  model_params[["propensity_covariate"]] <- json_object_default$get_string(
    "propensity_covariate"
  )
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
    warning(sprintf(
      "Fields 'has_rfx_basis' and 'num_rfx_basis' not found in BCF JSON (inferred version: %s). Defaulting to has_rfx_basis=FALSE, num_rfx_basis=1.",
      .ver
    ))
  }
  model_params[["num_covariates"]] <- json_object_default$get_scalar(
    "num_covariates"
  )
  if (has_field("num_chains")) {
    model_params[["num_chains"]] <- json_object_default$get_scalar("num_chains")
  } else {
    model_params[["num_chains"]] <- 1
    warning(sprintf(
      "Field 'num_chains' not found in BCF JSON (inferred version: %s). Defaulting to 1.",
      .ver
    ))
  }
  if (has_field("keep_every")) {
    model_params[["keep_every"]] <- json_object_default$get_scalar("keep_every")
  } else {
    model_params[["keep_every"]] <- 1
    warning(sprintf(
      "Field 'keep_every' not found in BCF JSON (inferred version: %s). Defaulting to 1.",
      .ver
    ))
  }
  model_params[["adaptive_coding"]] <- json_object_default$get_boolean(
    "adaptive_coding"
  )
  if (has_field("binary_treatment")) {
    model_params[["binary_treatment"]] <- json_object_default$get_boolean(
      "binary_treatment"
    )
  } else {
    model_params[["binary_treatment"]] <- .inferBinaryTreatmentFromJson(
      json_object_default,
      has_field
    )
    warning(sprintf(
      "Field 'binary_treatment' not found in BCF JSON (inferred version: %s). Inferred binary_treatment=%s from other JSON fields.",
      .ver,
      model_params[["binary_treatment"]]
    ))
  }
  if (has_field("treatment_dim")) {
    model_params[["treatment_dim"]] <- json_object_default$get_scalar(
      "treatment_dim"
    )
  } else {
    model_params[["treatment_dim"]] <- 1
    if (
      has_field("multivariate_treatment") &&
        isTRUE(json_object_default$get_boolean("multivariate_treatment"))
    ) {
      warning(sprintf(
        "Field 'treatment_dim' not found in BCF JSON (inferred version: %s) for a multivariate-treatment model. Defaulting to 1.",
        .ver
      ))
    }
  }
  if (has_field("sample_tau_0")) {
    model_params[["sample_tau_0"]] <- json_object_default$get_boolean(
      "sample_tau_0"
    )
  } else {
    model_params[["sample_tau_0"]] <- FALSE
    warning(sprintf(
      "Field 'sample_tau_0' not found in BCF JSON (inferred version: %s). Defaulting to FALSE.",
      .ver
    ))
  }
  if (has_field("multivariate_treatment")) {
    model_params[["multivariate_treatment"]] <- json_object_default$get_boolean(
      "multivariate_treatment"
    )
  } else {
    model_params[["multivariate_treatment"]] <- FALSE
    warning(sprintf(
      "Field 'multivariate_treatment' not found in BCF JSON (inferred version: %s). Defaulting to FALSE.",
      .ver
    ))
  }
  if (has_field("internal_propensity_model")) {
    model_params[[
      "internal_propensity_model"
    ]] <- json_object_default$get_boolean("internal_propensity_model")
  } else {
    model_params[["internal_propensity_model"]] <- FALSE
    warning(sprintf(
      "Field 'internal_propensity_model' not found in BCF JSON (inferred version: %s). Defaulting to FALSE.",
      .ver
    ))
  }
  if (has_field("probit_outcome_model")) {
    model_params[["probit_outcome_model"]] <- json_object_default$get_boolean(
      "probit_outcome_model"
    )
  } else {
    model_params[["probit_outcome_model"]] <- FALSE
    warning(sprintf(
      "Field 'probit_outcome_model' not found in BCF JSON (inferred version: %s). Defaulting to FALSE.",
      .ver
    ))
  }
  if (has_subfolder_field("outcome_model", "outcome")) {
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
    warning(sprintf(
      "Subfolder 'outcome_model' not found in BCF JSON (inferred version: %s). Defaulting to outcome='continuous', link='identity'.",
      .ver
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
      warning(sprintf(
        "Field 'rfx_model_spec' not found in BCF JSON (inferred version: %s) but has_rfx=TRUE.",
        .ver
      ))
    }
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
      combined_samples <- createBCFSamplesFromJson(json_object)
    } else {
      additional_samples <- createBCFSamplesFromJson(json_object)
      combined_samples$merge(additional_samples)
    }
  }
  output[["samples"]] <- combined_samples

  # Unpack random effects group IDs
  if (model_params[["has_rfx"]]) {
    output[["rfx_unique_group_ids"]] <- resolveRfxUniqueGroupIds(
      json_object_default,
      output[["samples"]]$materialize_rfx()
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
    warning(sprintf(
      "Field 'covariate_preprocessor' not found in BCF JSON (inferred version: %s). Preprocessor is unavailable; prediction may fail.",
      .ver
    ))
  }

  class(output) <- "bcfmodel"
  return(output)
}

#' @title Convert JSON String List to BCF Model
#' @param json_string_list List of JSON strings which can be parsed to objects of type `CppJson` containing Json representation of a BCF model
#' @export
#' @rdname BCFSerialization
createBCFModelFromCombinedJsonString <- function(json_string_list) {
  # Convert JSON strings
  json_object_list <- list()
  for (i in 1:length(json_string_list)) {
    json_string <- json_string_list[[i]]
    json_object_list[[i]] <- createCppJsonString(json_string)
    # Add runtime check for separately serialized propensity models
    # We don't support merging BCF models with independent propensity models
    # this way at the moment
    if (
      json_contains_field_cpp(
        json_object_list[[i]]$json_ptr,
        "internal_propensity_model"
      ) &&
        json_object_list[[i]]$get_boolean("internal_propensity_model")
    ) {
      stop(
        "Combining separate BCF models with cached internal propensity models is currently unsupported. To make this work, please first train a propensity model and then pass the propensities as data to the separate BCF models before sampling."
      )
    }
  }

  # Create BCF model from list of JSON objects
  createBCFModelFromCombinedJson(json_object_list)
}
