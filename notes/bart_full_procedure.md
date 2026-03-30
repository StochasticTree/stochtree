# BART Procedure Documentation

Here, we document the steps taken to validate and preprocess data / hyperparameters and then run the BART algorithm.

## Sampling

### Parameter Acceptance and Validation

We accept optional user-provided parameter lists / dictionaries, which override the following defaults:

**General Model Parameters**:
1. `cutpoint_grid_size`: 100
1. `standardize`: True
1. `sample_sigma2_global`: True
1. `sigma2_global_init`: None
1. `sigma2_global_shape`: 0
1. `sigma2_global_scale`: 0
1. `variable_weights`: None
1. `random_seed`: -1
1. `keep_burnin`: False
1. `keep_gfr`: False
1. `keep_every`: 1
1. `num_chains`: 1
1. `verbose`: False
1. `outcome_model`: `OutcomeModel(outcome = 'continuous', link = 'identity')`
1. `probit_outcome_model`: False
1. `num_threads`: -1

**Mean Forest Parameters**
1. `num_trees`: 200
1. `alpha`: 0.95
1. `beta`: 2.0
1. `min_samples_leaf`: 5
1. `max_depth`: 10
1. `sample_sigma2_leaf`: True
1. `sigma2_leaf_init`: None
1. `sigma2_leaf_shape`: 3
1. `sigma2_leaf_scale`: None
1. `keep_vars`: None
1. `drop_vars`: None
1. `num_features_subsample`: None
1. `cloglog_leaf_prior_shape`: 2.0
1. `cloglog_leaf_prior_scale`: 2.0

**Variance Forest Parameters**
1. `num_trees`: 0
1. `alpha`: 0.95
1. `beta`: 2.0
1. `min_samples_leaf`: 5
1. `max_depth`: 10
1. `leaf_prior_calibration_param`: 1.5
1. `var_forest_leaf_init`: None
1. `var_forest_prior_shape`: None
1. `var_forest_prior_scale`: None
1. `keep_vars`: None
1. `drop_vars`: None
1. `num_features_subsample`: None

**Random Effects Parameters**
1. `model_spec`: "custom"
1. `working_parameter_prior_mean`: None
1. `group_parameter_prior_mean`: None
1. `working_parameter_prior_cov`: None
1. `group_parameter_prior_cov`: None
1. `variance_prior_shape`: 1
1. `variance_prior_scale`: 1

If `random_seed` is specified, we ensure that the internal random number generators use this seed, otherwise we initialize the seed non-deterministically.

If `num_gfr > 0`, then we raise an error if `num_chains > num_gfr` (assuming that chains are initialized from separate GFR samples).

If `num_mcmc == 0`, then we override `keep_gfr` to `True`.

### Previous Model JSON Initialization

**[CLOGLOG UPDATE]**

If a "previous" model was passed as json, check if an initialization index passed. 

If provided, it must be between 1 and the number of draws stored in a JSON object. 

If not provided, assume the final sample of the previous JSON model was desired.

Initialize the following model terms from JSON if they exist:

1. Outcome offset (`previous_y_bar`)
1. Outcome scale (`previous_y_scale`)
1. Container of mean forests (`previous_forest_samples_mean`)
1. Container of variance forests (`previous_forest_samples_variance`)
1. Global error scale samples (`previous_global_var_samples`)
1. Leaf scale samples (`previous_leaf_var_samples`)
1. Random effects samples (`previous_rfx_samples`)

### Unpacking Outcome Model / Link Details

Based on the user-provided `outcome_model` specification, we extract the following details:

1. Outcome type (e.g., 'continuous', 'binary', 'ordinal')
1. Link function (e.g., 'identity', 'probit', 'cloglog')

into the following boolean variables:

1. `outcome_is_continuous`
1. `outcome_is_binary`
1. `outcome_is_ordinal`
1. `link_is_identity`
1. `link_is_probit`
1. `link_is_cloglog`

If a user specifies `probit_outcome_model = True`, we raise a deprecation warning and instruct users to use the `outcome_model` variable instead.

### Forest Parameter Overrides

#### Mean Forest

If `num_trees > 0` for the `mean_forest_params` list / dict, then set `include_mean_forest = True`

Override `sample_sigma2_leaf` to `False` if `include_mean_forest = False`

#### Variance Forest

If `num_trees > 0` for the `variance_forest_params` list / dict, then set `include_variance_forest = True`

### Covariate Transformation

If users pass `variance_weights` through the `general_params` list / dict, then raise an error if any weights are negative.

Check that train and test covariates are either a matrix or dataframe.

Raise warnings about potential problems in the GFR algorithm for the following 4 cases for any covariate treated as continuous:

num_unique_values / num_values < 0.2

num_unique_values < 20 and num_values > 100

number of duplicate values of any covariate > 2 * max_grid_size

num_unique_values = 2

In some cases these are indications that the covariates should be treated as categorical.

Check both the `keep_vars` and `drop_vars` arguments to `mean_forest_params` and `variance_forest_params`. `keep_vars` pre-empts `drop_vars`, and if the argument is provided as a character vector, it is converted to a variable index vector. Invalid variables (either indices or variable names) raise errors. Unpacked into binary `variable_subset_mean` and `variable_subset_variance` vectors.

Preprocess train set covariates, handling ordered categorical (integer conversion) and unordered categoricals (one-hot encoding) and run the same preprocessing on test covariates if provided. If the preprocessing involves expanding the number of covariates (by e.g. one-hot-encoding categorical variables), adjustments are made to `variable_weights`, `variable_subset_mean`, and `variable_subset_variance` vectors.

Set `num_features_subsample_mean` and `num_features_subsample_variance` to the number of columns of `X_train` if not user-specified (this tells the GFR algorithm to sweep all features at each split search).

### Basis Matrix Validation

If any of `leaf_basis_train`, `leaf_basis_test`, `rfx_basis_train`, and `rfx_basis_test` are provided, check if they are matrices, converting numeric vectors to matrices (or 2d numpy arrays) if offered.

### Random Effects Validation

If `rfx_group_ids_train` (and `rfx_group_ids_test`) are provided as factor / character array, map to an array / vector of integers.

### Data Consistency Checks

Check that `X_train` and `X_test` (if both exist) have the same number of columns

Check that `leaf_basis_train` and `leaf_basis_test` (if both exist) have the same number of columns

Check that `leaf_basis_train` and `X_train` (if both exist) have the same number of rows

Check that `X_train` and `y_train` (if both exist) have the same number of rows / observations

Check that `rfx_basis_train` and `rfx_basis_test` (if both exist) have the same number of columns

If `rfx_group_ids_train` and `rfx_group_ids_test` are provided, and `rfx_basis_train` is provided, check that `rfx_basis_test` is provided

### Random Effects Basis Processing

If random effects are specified with a `model_spec == 'custom'` in the `rfx_params` list / dict, then `num_basis_rfx` is set to the number of columns in `rfx_basis_train`.

If random effects are specified with a `model_spec == 'intercept_only'` in the `rfx_params` list / dict, then `rfx_basis_train` is set to a single-column matrix of all ones with as many rows as in `X_train`, and `num_basis_rfx` is set to 1. (`rfx_basis_test` is similarly specified if a test set is provided.)

### Outcome preprocessing

Convert `y_train` to a (single-column) matrix if not already.

### Setting "helper" variables

Set `has_basis` to `True` if `leaf_basis_train` is provided. Set `has_test` to `True` if `X_test` is provided.

### Runtime checks for probit model

Override `link_is_probit = True` specification if no mean forest is requested.

Check that `y_train` has two unique values, 0 and 1, if `link_is_probit = True`, raise an error if not.

Raise an error if a variance forest is requested alongside a mean forest with probit link.

Override `sample_sigma2_global` to `False`, since $\sigma^2$ is fixed at 1 in the probit model.

### Runtime checks for cloglog model

Override `link_is_cloglog = True` specification if no mean forest is requested.

Check that `y_train` is integer-valued, starting at either 0 or 1, if `link_is_cloglog = True`, with at least two unique values, raise an error if not.

If `y_train` starts at 1, subtract 1 from all values to make it start at 0 before passing through to a C++ `Outcome` object.

Raise an error if a variance forest is requested alongside a mean forest with cloglog link.

Raise an error if leaf basis regression is requested alongside a mean forest with cloglog link.

Override `sample_sigma2_global` to `False`.

Override `sample_sigma2_leaf` to `False`.

### Runtime checks for variance model

Override `sample_sigma2_global` to `False`, since $\sigma^2$ is unidentified in a model with heteroskedastic conditional variance function.

### Calibration and initialization for mean forest

#### Probit Model

Set outcome scale (`y_std`) to 1.

Set outcome offset (`y_bar`) to the standard normal percentile of `mean(y_train)`.

Set a "pseudo outcome" by subtracting `mean(y_train)` from `y_train`.

Set the initial values of every root in the mean forest to 0.0 (on probit scale).

Set the initial value of $\sigma^2$ (which is fixed in the sampler) to 1

Set the scale parameter for the leaf scale model to `1 / num_trees_mean` if not provided

Set the initial value for the leaf scale parameter to `2 / num_trees_mean` if not provided

#### Gaussian Model

If standardization requested, set outcome scale (`y_std`) to `sd(y_train)` and outcome offset (`y_bar`) to `mean(y_train)`.

If standardization not requested, set outcome scale (`y_std`) to 1 and outcome offset (`y_bar`) to 0.

Define a "pseudo outcome" as `resid = (y_train - y_bar) / y_std`.

Compute the default value of each root node of every tree in the mean forest as `mean(resid_train) / num_trees_mean`.

If an initial value for $\sigma^2$ not provided, set it as `var(resid_train)`.

If an initial value for $\sigma^2(X)$ (variance forest) not provided, set it as `var(resid_train)`.

If an initial value for the scale parameter of the leaf scale model not provided, set it as `var(resid_train) / 2 * num_trees_mean`.

If an initial value for the leaf scale parameter not provided, set it as `2 * var(resid_train) / num_trees_mean`.

### Model Type Checks

Map mean forest model type to internal integer codes as follows:

* If `has_basis = False`, `leaf_model_mean_forest = 0`
* If `leaf_basis_train` has one column, `leaf_model_mean_forest = 1`
* If `leaf_basis_train` has more than one column, `leaf_model_mean_forest = 2`

There is only one integer code for the variance forest model, so if a variance forest is requested, we set `leaf_model_variance_forest = 3`.

Raise error if `leaf_model_mean_forest` is 1 or 2 but basis is not provided.

Override `sample_sigma2_leaf` if `leaf_model_mean_forest == 2`.

### C++ Data Structure Initialization

Create a `ForestDataset` from `X_train` and `leaf_basis_train`.

Create an `Outcome` from `resid_train`.

Create a `CppRNG`, using `random_seed` if provided.

Create a `GlobalModelConfig` object, storing $\sigma^2$.

If the global error variance parameter is to be sampled:
* Create a container for variance samples

If the leaf scale parameter is to be sampled:
* Create a container for leaf scale samples

If a mean forest is requested:
* Create a `ForestModelConfig` object
* Create a `ForestModel` (`ForestSampler` in Python) object
* Create a `Forest` object as the "active mean forest"
* Create a `ForestSamples` object as the forest container

If a cloglog link is used for the mean forest:
* Create a container for cutpoint samples

If a variance forest is requested:
* Create a `ForestModelConfig` object
* Create a `ForestModel` (`ForestSampler` in Python) object
* Create a `Forest` object as the "active variance forest"
* Create a `ForestSamples` object as the forest container

If random effects are requested:
1. Set model hyperparameters
  1. If working parameter prior mean not specified, initialize it to 0 for every model component
  1. If group parameter prior mean not specified, initialize it to 0 for every group / model component
  1. If working parameter prior covariance matrix not specified, initialize it to identity matrix with as many rows / columns as components
  1. If group parameter prior covariance matrix not specified, initialize it to identity matrix with as many rows / columns as components
  1. If shape / scale parameter for group parameter variance model not specified, initialize both to 1
2. Create a `RandomEffectsDataset` from `rfx_group_ids_train` and `rfx_basis_train`.
3. Create a `RandomEffectsTracker` from `rfx_group_ids_train`.
4. Create a `RandomEffectsModel` from hyperparameters set in step 1 above.
5. Create an empty `RandomEffectSamples` object.

### "Active Forest" Initialization

#### Mean Forest

For models without a leaf basis, we simply set each root to `mean(resid_train) / num_trees_mean`.

**[CLOGLOG UPDATE]**

For models with a leaf basis, we consider two cases:

1. `mean(resid_train) == 0`: we simply set each coefficient of the leaf model to 0
2. `mean(resid_train) != 0`: in this case, we regress a constant vector of `mean(resid_train)` observations on `leaf_basis_train` and set `coef / num_trees_mean`  as the initial values of each root in each tree. If the regression model is overdetermined, we simply replace any degenerate (`NA`) coefficients with 0.

After setting every root to its initial value, we update the `Outcome` object to serve as a partial residual by subtracting the prediction of each tree.

**[CLOGLOG UPDATE]**

#### Variance Forest

Set each root to `log(var(resid_train)) / num_trees_variance`. Update the variance weights in the `ForestDataset` to reflect the exponentiated forest sum of predictions.

### "Grow-From-Root" Sampling Step

The grow-from-root algorithm runs `num_gfr` iterations of the following steps

#### Mean Forest Sampling

**[CLOGLOG UPDATE]**

If a probit outcome model is specified, then a continuous latent outcome is generated by 

1. Predicting $f(X)$ from the mean forest and separating predictions into $f_0$ and $f_1$ for observations with $y = 0$ and $y = 1$
2. Sampling $u_0(X_i) = \text{Uniform}(0, \Phi(- f_0(X_i)))$ for every $i$ with $y_i = 0$
3. Sampling $u_1(X_i) = \text{Uniform}(\Phi(- f_1(X_i)), 1)$ for every $i$ with $y_i = 1$
4. Sampling $\tilde{f}_0(X_i) = f_0(X_i) + \Phi^{-1}(u_0(X_i))$ for every $i$ with $y_i = 0$
5. Sampling $\tilde{f}_1(X_i) = f_1(X_i) + \Phi^{-1}(u_1(X_i))$ for every $i$ with $y_i = 1$
6. Setting the latent outcome to $\tilde{r} = r - \tilde{f}$ where $r$ is the original outcome (potentially scaled / shifted from the 0 / 1 coding to different binary values)

If a probit outcome model is **not** specified, then $\tilde{r}$ is specified as the partial residual, net of previous mean forest predictions and other additive terms (random effects)

The mean forest is then sampled with $\tilde{r}$ as outcome.

After sampling, $\tilde{r}$ will have been updated with new forest predictions and will be available without modification to other model terms (i.e. random effects, variance terms).

If an iteration is to be retained in the sampler state, then the state of the "active forest" after sampling is copied to a container of mean forest samples

#### Variance Forest Sampling

As above $\tilde{r}$ is specified as the partial residual, net of previous mean forest predictions and other additive terms (random effects)

The variance forest is sampled with $\tilde{r}$ as outcome and observation-specific variances (net of the current tree being modified) as weights.

If an iteration is to be retained in the sampler state, then the state of the "active forest" after sampling is copied to a container of variance forest samples

#### Global Error Variance Parameter Sampling

The global error scale parameter is sampled using $\tilde{r}$ as data, where $\tilde{r}$ is specified as the full residual (net of all tree predictions for the mean forest and other additive terms)

#### Mean Forest Leaf Scale Parameter Sampling

The leaf scale parameter for the mean forest is sampled using all leaf parameters from the mean forest as data.

#### Random Effects Sampling

Random effects are sampled using $\tilde{r}$ as data, where $\tilde{r}$ is specified as the partial residual, net of all mean forest predictions but not the random effects estimates.

### MCMC Sampling Step

The MCMC algorithm runs `(num_burnin + num_mcmc) * num_chains` iterations of the following steps

#### Chain State Initialization

**[CLOGLOG UPDATE]**

There are three different cases for chain initialization:
1. Initializing from a GFR iteration run in the sampling loop specified above
  1. This case is run first whenever `num_gfr > 0`. 
  2. Draws from distinct GFR iterations are used to initialize different MCMC chains (thus, requiring that `num_chains <= num_gfr` if both are nonzero).
2. Initializing from a previous model reloaded from JSON
  1. This case is reached when `num_gfr == 0` and a previous model was provided as JSON
  1. Draws from distinct model samples are used to initialize different MCMC chains (thus, requiring that `num_chains <= num_samples` if both are nonzero).
3. Starting each chain from root / default settings

The following model terms are loaded either from root / default or a previous model iteration depending on the case:
1. Mean forest
2. Variance forest
3. Random forest
4. Global variance parameter
5. Leaf scale parameter

#### Mean Forest Sampling

**[CLOGLOG UPDATE]**

If a probit outcome model is specified, then a continuous latent outcome is generated by 

1. Predicting $f(X)$ from the mean forest and separating predictions into $f_0$ and $f_1$ for observations with $y = 0$ and $y = 1$
2. Sampling $u_0(X_i) = \text{Uniform}(0, \Phi(- f_0(X_i)))$ for every $i$ with $y_i = 0$
3. Sampling $u_1(X_i) = \text{Uniform}(\Phi(- f_1(X_i)), 1)$ for every $i$ with $y_i = 1$
4. Sampling $\tilde{f}_0(X_i) = f_0(X_i) + \Phi^{-1}(u_0(X_i))$ for every $i$ with $y_i = 0$
5. Sampling $\tilde{f}_1(X_i) = f_1(X_i) + \Phi^{-1}(u_1(X_i))$ for every $i$ with $y_i = 1$
6. Setting the latent outcome to $\tilde{r} = r - \tilde{f}$ where $r$ is the original outcome (potentially scaled / shifted from the 0 / 1 coding to different binary values)

If a probit outcome model is **not** specified, then $\tilde{r}$ is specified as the partial residual, net of previous mean forest predictions and other additive terms (random effects)

The mean forest is then sampled with $\tilde{r}$ as outcome.

After sampling, $\tilde{r}$ will have been updated with new forest predictions and will be available without modification to other model terms (i.e. random effects, variance terms).

If an iteration is to be retained in the sampler state, then the state of the "active forest" after sampling is copied to a container of mean forest samples

#### Variance Forest Sampling

As above $\tilde{r}$ is specified as the partial residual, net of previous mean forest predictions and other additive terms (random effects)

The variance forest is sampled with $\tilde{r}$ as outcome and observation-specific variances (net of the current tree being modified) as weights.

If an iteration is to be retained in the sampler state, then the state of the "active forest" after sampling is copied to a container of variance forest samples

#### Global Error Variance Parameter Sampling

The global error scale parameter is sampled using $\tilde{r}$ as data, where $\tilde{r}$ is specified as the full residual (net of all tree predictions for the mean forest and other additive terms)

#### Mean Forest Leaf Scale Parameter Sampling

The leaf scale parameter for the mean forest is sampled using all leaf parameters from the mean forest as data.

#### Random Effects Sampling

Random effects are sampled using $\tilde{r}$ as data, where $\tilde{r}$ is specified as the partial residual, net of all mean forest predictions but not the random effects estimates.

### Post-processing

If users did not specify `keep_gfr = True`, then GFR samples are removed from all forest and parameter containers (they are initially retained for multi-chain MCMC initialization purposes).

If a mean forest is included, we unpack cached training set predictions and scale / shift them by `y_std_train` and `y_bar_train`. If test set covariates / bases are provided, we compute test set predictions and similarly scale / shift them.

If random effects are included, we compute train / test set predictions for each observation.

We compute `y_hat_train` / `y_hat_test` as the combination of either or both of mean forest and random effects predictions and include these matrices in the returned model object. We also return separate `rfx_preds_train` / `rfx_preds_test` if random effects terms are present.

If a variance forest is included, we compute train / test set predictions for each observation and include the resulting predictions in the returned model object as `sigma2_x_hat_train` / `sigma2_x_hat_test`.

The resulting model object stores pointers to mean forests, variance forests and random effects samples if they are part of the model.

We construct a `model_params` list that stores model metadata (fields such as `num_gfr`, `num_mcmc`, `has_rfx`, `include_mean_forest`, `include_variance_forest`) and include this in the returned model object.

## Prediction

### Parameter Acceptance and Validation

**[CLOGLOG UPDATE]**

The BART prediction method accepts two types of inputs:

1. Data
  1. Covariates (**required**)
  2. Leaf bases (**optional**, required if original model used them)
  3. Random effects group IDs (**optional**, required if original model had a random effects term)
  4. Random effects bases (**optional**, required if original model had a random effects term and the model spec was `"custom"`)
2. Prediction Parameters
  1. `type`: type of prediction to return (options are `"posterior"` for the entire distribution of predictions or `"mean"` for the posterior mean)
  2. `terms`: model terms from which to predict (options are `"y_hat"`, `"mean_forest"`, `"rfx"`, `"variance_forest"`, or `"all"`. If a model doesn't have mean forest, random effects, or variance forest predictions, but one of those terms is request, the request will simply be ignored. If none of the requested terms are present in a model, this function will return `None`/`NULL` along with a warning.)
  3. `scale`: scale of mean function predictions. Options are `"linear"`, which returns predictions on the original scale of the mean forest / RFX terms, and `"probability"`, which transforms predictions into a probability of observing `y == 1`. `"probability"` is only valid for models fit with a probit outcome model.

We provide the following default parameters:

1. `type`: `"posterior"`
2. `terms`: `"all"`
3. `scale`: `"linear"`

Input validation on data terms consists of:
* Checking types (ensuring either data frames or matrices)
* Ensuring that models which were fit with a given data object (i.e. leaf bases or random effects bases) are provided with those elements

If the original model required data preprocessing (for instance, due to categorical covariates), then valid covariates are processed through the same procedure before prediction.

Input validation on parameters ensures that arguments are strings (or `None` / `NULL`) and taken from one of the pre-determined options.

### Computing Relevant Predictions

Often, a model component that is not directly specified in `terms` must be computed as an intermediate component of a different term. Specifically, requesting `y_hat` means that both random effects and mean forest predictions must be computed if those terms are present in the model.

If random effects must be computed and the model has a random effects `model_spec` of `"intercept_only"`, then the random effects basis is specified manually as a single-column matrix of ones with as many rows as the covariate set.

Covariates and leaf bases (if provided) are used to create a `ForestDataset` object, which is then used to compute (if present):
1. Variance forest predictions
2. Mean forest predictions

Random effects predictions are computed using group IDs and bases (if provided) directly.

### Output Transformation

#### Scaling and Shifting

Mean forest predictions are scaled by the `y_std` value stored by the model and shifted by the `y_bar` value.

Random effects predictions are scaled by the `y_std` value.

Variance forest predictions are scaled by product of $\sigma^2_0$ and `y_std * y_std`.

#### Aggregation

**[CLOGLOG UPDATE]**

If `scale == "linear"`, then `y_hat` is the sum of mean forest predictions and random effects predictions for which ever terms exist

If `scale == "probability"`, then `y_hat` is the CDF of the sum of mean forest predictions and random effects predictions

If `type == "mean"`, then all prediction terms are returned as their "row-wise" average.
