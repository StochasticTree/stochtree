"""
Bayesian Additive Regression Trees (BART) module
"""

import warnings
from math import log
from numbers import Integral
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import ForestModelConfig, GlobalModelConfig
from .data import Dataset, Residual
from .forest import Forest, ForestContainer
from .preprocessing import CovariatePreprocessor, _preprocess_params
from .random_effects import (
    RandomEffectsContainer,
    RandomEffectsDataset,
    RandomEffectsModel,
    RandomEffectsTracker,
)
from .sampler import RNG, ForestSampler, GlobalVarianceModel, LeafVarianceModel
from .serialization import JSONSerializer
from .utils import NotSampledError


class BARTModel:
    r"""
    Class that handles sampling, storage, and serialization of stochastic forest models for supervised learning. 
    The class takes its name from Bayesian Additive Regression Trees, an MCMC sampler originally developed in 
    Chipman, George, McCulloch (2010), but supports several sampling algorithms:

    - MCMC: The "classic" sampler defined in Chipman, George, McCulloch (2010). In order to run the MCMC sampler, set `num_gfr = 0` (explained below) and then define a sampler according to several parameters:
        - `num_burnin`: the number of iterations to run before "retaining" samples for further analysis. These "burned in" samples are helpful for allowing a sampler to converge before retaining samples.
        - `num_chains`: the number of independent sequences of MCMC samples to generate (typically referred to in the literature as "chains")
        - `num_mcmc`: the number of "retained" samples of the posterior distribution
        - `keep_every`: after a sampler has "burned in", we will run the sampler for `keep_every` * `num_mcmc` iterations, retaining one of each `keep_every` iteration in a chain.
    - GFR (Grow-From-Root): A fast, greedy approximation of the BART MCMC sampling algorithm introduced in He and Hahn (2021). GFR sampler iterations are governed by the `num_gfr` parameter, and there are two primary ways to use this sampler:
        - Standalone: setting `num_gfr > 0` and both `num_burnin = 0` and `num_mcmc = 0` will only run and retain GFR samples of the posterior. This is typically referred to as "XBART" (accelerated BART).
        - Initializer for MCMC: setting `num_gfr > 0` and `num_mcmc > 0` will use ensembles from the GFR algorithm to initialize `num_chains` independent MCMC BART samplers, which are run for `num_mcmc` iterations. This is typically referred to as "warm start BART".
    
    In addition to enabling multiple samplers, we support a broad set of models. First, note that the original BART model of Chipman, George, McCulloch (2010) is

    \begin{equation*}
    \begin{aligned}
    y &= f(X) + \epsilon\\
    f(X) &\sim \text{BART}(\cdot)\\
    \epsilon &\sim N(0, \sigma^2)\\
    \sigma^2 &\sim IG(\nu, \nu\lambda)
    \end{aligned}
    \end{equation*}

    In words, there is a nonparametric mean function governed by a tree ensemble with a BART prior and an additive (mean-zero) Gaussian error 
    term, whose variance is parameterized with an inverse gamma prior.

    The `BARTModel` class supports the following extensions of this model:
    
    - Leaf Regression: Rather than letting `f(X)` define a standard decision tree ensemble, in which each tree uses `X` to partition the data and then serve up constant predictions, we allow for models `f(X,Z)` in which `X` and `Z` together define a partitioned linear model (`X` partitions the data and `Z` serves as the basis for regression models). This model can be run by specifying `leaf_basis_train` in the `sample` method.
    - Heteroskedasticity: Rather than define $\epsilon$ parameterically, we can let a forest $\sigma^2(X)$ model a conditional error variance function. This can be done by setting `num_trees_variance > 0` in the `params` dictionary passed to the `sample` method.
    """

    def __init__(self) -> None:
        # Internal flag for whether the sample() method has been run
        self.sampled = False

    def sample(
        self,
        X_train: Union[np.array, pd.DataFrame],
        y_train: np.array,
        leaf_basis_train: np.array = None,
        rfx_group_ids_train: np.array = None,
        rfx_basis_train: np.array = None,
        X_test: Union[np.array, pd.DataFrame] = None,
        leaf_basis_test: np.array = None,
        rfx_group_ids_test: np.array = None,
        rfx_basis_test: np.array = None,
        num_gfr: int = 5,
        num_burnin: int = 0,
        num_mcmc: int = 100,
        general_params: Optional[Dict[str, Any]] = None,
        mean_forest_params: Optional[Dict[str, Any]] = None,
        variance_forest_params: Optional[Dict[str, Any]] = None,
        previous_model_json: Optional[str] = None,
        previous_model_warmstart_sample_num: Optional[int] = None,
    ) -> None:
        """Runs a BART sampler on provided training set. Predictions will be cached for the training set and (if provided) the test set.
        Does not require a leaf regression basis.

        Parameters
        ----------
        X_train : np.array
            Training set covariates on which trees may be partitioned.
        y_train : np.array
            Training set outcome.
        leaf_basis_train : np.array, optional
            Optional training set basis vector used to define a regression to be run in the leaves of each tree.
        rfx_group_ids_train : np.array, optional
            Optional group labels used for an additive random effects model.
        rfx_basis_train : np.array, optional
            Optional basis for "random-slope" regression in an additive random effects model.
        X_test : np.array, optional
            Optional test set covariates.
        leaf_basis_test : np.array, optional
            Optional test set basis vector used to define a regression to be run in the leaves of each tree.
            Must be included / omitted consistently (i.e. if leaf_basis_train is provided, then leaf_basis_test must be provided alongside X_test).
        rfx_group_ids_test : np.array, optional
            Optional test set group labels used for an additive random effects model. We do not currently support (but plan to in the near future),
            test set evaluation for group labels that were not in the training set.
        rfx_basis_test : np.array, optional
            Optional test set basis for "random-slope" regression in additive random effects model.
        num_gfr : int, optional
            Number of "warm-start" iterations run using the grow-from-root algorithm (He and Hahn, 2021). Defaults to `5`.
        num_burnin : int, optional
            Number of "burn-in" iterations of the MCMC sampler. Defaults to `0`. Ignored if `num_gfr > 0`.
        num_mcmc : int, optional
            Number of "retained" iterations of the MCMC sampler. Defaults to `100`. If this is set to 0, GFR (XBART) samples will be retained.
        general_params : dict, optional
            Dictionary of general model parameters, each of which has a default value processed internally, so this argument is optional.

            * `cutpoint_grid_size` (`int`): Maximum number of cutpoints to consider for each feature. Defaults to `100`.
            * `standardize` (`bool`): Whether or not to standardize the outcome (and store the offset / scale in the model object). Defaults to `True`.
            * `sample_sigma2_global` (`bool`): Whether or not to update the `sigma^2` global error variance parameter based on `IG(sigma2_global_shape, sigma2_global_scale)`. Defaults to `True`.
            * `sigma2_init` (`float`): Starting value of global variance parameter. Set internally to the outcome variance (standardized if `standardize = True`) if not set here.
            * `sigma2_global_shape` (`float`): Shape parameter in the `IG(sigma2_global_shape, b_glsigma2_global_scaleobal)` global error variance model. Defaults to `0`.
            * `sigma2_global_scale` (`float`): Scale parameter in the `IG(sigma2_global_shape, b_glsigma2_global_scaleobal)` global error variance model. Defaults to `0`.
            * `variable_weights` (`np.array`): Numeric weights reflecting the relative probability of splitting on each variable. Does not need to sum to 1 but cannot be negative. Defaults to uniform over the columns of `X_train` if not provided.
            * `random_seed` (`int`): Integer parameterizing the C++ random number generator. If not specified, the C++ random number generator is seeded according to `std::random_device`.
            * `keep_burnin` (`bool`): Whether or not "burnin" samples should be included in predictions. Defaults to `False`. Ignored if `num_mcmc == 0`.
            * `keep_gfr` (`bool`): Whether or not "warm-start" / grow-from-root samples should be included in predictions. Defaults to `False`. Ignored if `num_mcmc == 0`.
            * `keep_every` (`int`): How many iterations of the burned-in MCMC sampler should be run before forests and parameters are retained. Defaults to `1`. Setting `keep_every = k` for some `k > 1` will "thin" the MCMC samples by retaining every `k`-th sample, rather than simply every sample. This can reduce the autocorrelation of the MCMC samples.
            * `num_chains` (`int`): How many independent MCMC chains should be sampled. If `num_mcmc = 0`, this is ignored. If `num_gfr = 0`, then each chain is run from root for `num_mcmc * keep_every + num_burnin` iterations, with `num_mcmc` samples retained. If `num_gfr > 0`, each MCMC chain will be initialized from a separate GFR ensemble, with the requirement that `num_gfr >= num_chains`. Defaults to `1`.

        mean_forest_params : dict, optional
            Dictionary of mean forest model parameters, each of which has a default value processed internally, so this argument is optional.

            * `num_trees` (`int`): Number of trees in the conditional mean model. Defaults to `200`. If `num_trees = 0`, the conditional mean will not be modeled using a forest and sampling will only proceed if `num_trees > 0` for the variance forest.
            * `alpha` (`float`): Prior probability of splitting for a tree of depth 0 in the conditional mean model. Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`. Defaults to `0.95`.
            * `beta` (`float`): Exponent that decreases split probabilities for nodes of depth > 0 in the conditional mean model. Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`. Defaults to `2`.
            * `min_samples_leaf` (`int`): Minimum allowable size of a leaf, in terms of training samples, in the conditional mean model. Defaults to `5`.
            * `max_depth` (`int`): Maximum depth of any tree in the ensemble in the conditional mean model. Defaults to `10`. Can be overriden with `-1` which does not enforce any depth limits on trees.
            * `sample_sigma2_leaf` (`bool`): Whether or not to update the `tau` leaf scale variance parameter based on `IG(sigma2_leaf_shape, sigma2_leaf_scale)`. Cannot (currently) be set to true if `leaf_basis_train` has more than one column. Defaults to `False`.
            * `sigma2_leaf_init` (`float`): Starting value of leaf node scale parameter. Calibrated internally as `1/num_trees` if not set here.
            * `sigma2_leaf_shape` (`float`): Shape parameter in the `IG(sigma2_leaf_shape, sigma2_leaf_scale)` leaf node parameter variance model. Defaults to `3`.
            * `sigma2_leaf_scale` (`float`): Scale parameter in the `IG(sigma2_leaf_shape, sigma2_leaf_scale)` leaf node parameter variance model. Calibrated internally as `0.5/num_trees` if not set here.
            * `keep_vars` (`list` or `np.array`): Vector of variable names or column indices denoting variables that should be included in the mean forest. Defaults to `None`.
            * `drop_vars` (`list` or `np.array`): Vector of variable names or column indices denoting variables that should be excluded from the mean forest. Defaults to `None`. If both `drop_vars` and `keep_vars` are set, `drop_vars` will be ignored.
            * `probit_outcome_model` (`bool`): Whether or not the outcome should be modeled as explicitly binary via a probit link. If `True`, `y` must only contain the values `0` and `1`. Default: `False`.

        variance_forest_params : dict, optional
            Dictionary of variance forest model  parameters, each of which has a default value processed internally, so this argument is optional.

            * `num_trees` (`int`): Number of trees in the conditional variance model. Defaults to `0`. Variance is only modeled using a tree / forest if `num_trees > 0`.
            * `alpha` (`float`): Prior probability of splitting for a tree of depth 0 in the conditional variance model. Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`. Defaults to `0.95`.
            * `beta` (`float`): Exponent that decreases split probabilities for nodes of depth > 0 in the conditional variance model. Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`. Defaults to `2`.
            * `min_samples_leaf` (`int`): Minimum allowable size of a leaf, in terms of training samples, in the conditional variance model. Defaults to `5`.
            * `max_depth` (`int`): Maximum depth of any tree in the ensemble in the conditional variance model. Defaults to `10`. Can be overriden with `-1` which does not enforce any depth limits on trees.
            * `leaf_prior_calibration_param` (`float`): Hyperparameter used to calibrate the [optional] `IG(var_forest_prior_shape, var_forest_prior_scale)` conditional error variance model. If `var_forest_prior_shape` and `var_forest_prior_scale` are not set below, this calibration parameter is used to set these values to `num_trees / leaf_prior_calibration_param^2 + 0.5` and `num_trees / leaf_prior_calibration_param^2`, respectively. Defaults to `1.5`.
            * `var_forest_leaf_init` (`float`): Starting value of root forest prediction in conditional (heteroskedastic) error variance model. Calibrated internally as `np.log(0.6*np.var(y_train))/num_trees_variance`, where `y_train` is the possibly standardized outcome, if not set.
            * `var_forest_prior_shape` (`float`): Shape parameter in the [optional] `IG(var_forest_prior_shape, var_forest_prior_scale)` conditional error variance forest (which is only sampled if `num_trees > 0`). Calibrated internally as `num_trees / leaf_prior_calibration_param^2 + 0.5` if not set here.
            * `var_forest_prior_scale` (`float`): Scale parameter in the [optional] `IG(var_forest_prior_shape, var_forest_prior_scale)` conditional error variance forest (which is only sampled if `num_trees > 0`). Calibrated internally as `num_trees / leaf_prior_calibration_param^2` if not set here.
            * `keep_vars` (`list` or `np.array`): Vector of variable names or column indices denoting variables that should be included in the variance forest. Defaults to `None`.
            * `drop_vars` (`list` or `np.array`): Vector of variable names or column indices denoting variables that should be excluded from the variance forest. Defaults to `None`. If both `drop_vars` and `keep_vars` are set, `drop_vars` will be ignored.

        previous_model_json : str, optional
            JSON string containing a previous BART model. This can be used to "continue" a sampler interactively after inspecting the samples or to run parallel chains "warm-started" from existing forest samples. Defaults to `None`.
        previous_model_warmstart_sample_num : int, optional
            Sample number from `previous_model_json` that will be used to warmstart this BART sampler. Zero-indexed (so that the first sample is used for warm-start by setting `previous_model_warmstart_sample_num = 0`). Defaults to `None`.

        Returns
        -------
        self : BARTModel
            Sampled BART Model.
        """
        # Update general BART parameters
        general_params_default = {
            "cutpoint_grid_size": 100,
            "standardize": True,
            "sample_sigma2_global": True,
            "sigma2_init": None,
            "sigma2_global_shape": 0,
            "sigma2_global_scale": 0,
            "variable_weights": None,
            "random_seed": None,
            "keep_burnin": False,
            "keep_gfr": False,
            "keep_every": 1,
            "num_chains": 1,
            "probit_outcome_model": False,
        }
        general_params_updated = _preprocess_params(
            general_params_default, general_params
        )

        # Update mean forest BART parameters
        mean_forest_params_default = {
            "num_trees": 200,
            "alpha": 0.95,
            "beta": 2.0,
            "min_samples_leaf": 5,
            "max_depth": 10,
            "sample_sigma2_leaf": True,
            "sigma2_leaf_init": None,
            "sigma2_leaf_shape": 3,
            "sigma2_leaf_scale": None,
            "keep_vars": None,
            "drop_vars": None,
        }
        mean_forest_params_updated = _preprocess_params(
            mean_forest_params_default, mean_forest_params
        )

        # Update variance forest BART parameters
        variance_forest_params_default = {
            "num_trees": 0,
            "alpha": 0.95,
            "beta": 2.0,
            "min_samples_leaf": 5,
            "max_depth": 10,
            "leaf_prior_calibration_param": 1.5,
            "var_forest_leaf_init": None,
            "var_forest_prior_shape": None,
            "var_forest_prior_scale": None,
            "keep_vars": None,
            "drop_vars": None,
        }
        variance_forest_params_updated = _preprocess_params(
            variance_forest_params_default, variance_forest_params
        )

        ### Unpack all parameter values
        # 1. General parameters
        cutpoint_grid_size = general_params_updated["cutpoint_grid_size"]
        self.standardize = general_params_updated["standardize"]
        sample_sigma2_global = general_params_updated["sample_sigma2_global"]
        sigma2_init = general_params_updated["sigma2_init"]
        a_global = general_params_updated["sigma2_global_shape"]
        b_global = general_params_updated["sigma2_global_scale"]
        variable_weights = general_params_updated["variable_weights"]
        random_seed = general_params_updated["random_seed"]
        keep_burnin = general_params_updated["keep_burnin"]
        keep_gfr = general_params_updated["keep_gfr"]
        keep_every = general_params_updated["keep_every"]
        num_chains = general_params_updated["num_chains"]
        self.probit_outcome_model = general_params_updated["probit_outcome_model"]

        # 2. Mean forest parameters
        num_trees_mean = mean_forest_params_updated["num_trees"]
        alpha_mean = mean_forest_params_updated["alpha"]
        beta_mean = mean_forest_params_updated["beta"]
        min_samples_leaf_mean = mean_forest_params_updated["min_samples_leaf"]
        max_depth_mean = mean_forest_params_updated["max_depth"]
        sample_sigma2_leaf = mean_forest_params_updated["sample_sigma2_leaf"]
        sigma2_leaf = mean_forest_params_updated["sigma2_leaf_init"]
        a_leaf = mean_forest_params_updated["sigma2_leaf_shape"]
        b_leaf = mean_forest_params_updated["sigma2_leaf_scale"]
        keep_vars_mean = mean_forest_params_updated["keep_vars"]
        drop_vars_mean = mean_forest_params_updated["drop_vars"]

        # 3. Variance forest parameters
        num_trees_variance = variance_forest_params_updated["num_trees"]
        alpha_variance = variance_forest_params_updated["alpha"]
        beta_variance = variance_forest_params_updated["beta"]
        min_samples_leaf_variance = variance_forest_params_updated["min_samples_leaf"]
        max_depth_variance = variance_forest_params_updated["max_depth"]
        a_0 = variance_forest_params_updated["leaf_prior_calibration_param"]
        variance_forest_leaf_init = variance_forest_params_updated[
            "var_forest_leaf_init"
        ]
        a_forest = variance_forest_params_updated["var_forest_prior_shape"]
        b_forest = variance_forest_params_updated["var_forest_prior_scale"]
        keep_vars_variance = variance_forest_params_updated["keep_vars"]
        drop_vars_variance = variance_forest_params_updated["drop_vars"]

        # Override keep_gfr if there are no MCMC samples
        if num_mcmc == 0:
            keep_gfr = True

        # Check that num_chains >= 1
        if not isinstance(num_chains, Integral) or num_chains < 1:
            raise ValueError("num_chains must be an integer greater than 0")

        # Check if there are enough GFR samples to seed num_chains samplers
        if num_gfr > 0:
            if num_chains > num_gfr:
                raise ValueError(
                    "num_chains > num_gfr, meaning we do not have enough GFR samples to seed num_chains distinct MCMC chains"
                )

        # Determine which models (conditional mean, conditional variance, or both) we will fit
        self.include_mean_forest = True if num_trees_mean > 0 else False
        self.include_variance_forest = True if num_trees_variance > 0 else False

        # Check data inputs
        if not isinstance(X_train, pd.DataFrame) and not isinstance(
            X_train, np.ndarray
        ):
            raise ValueError("X_train must be a pandas dataframe or numpy array")
        if X_test is not None:
            if not isinstance(X_test, pd.DataFrame) and not isinstance(
                X_test, np.ndarray
            ):
                raise ValueError("X_test must be a pandas dataframe or numpy array")
        if not isinstance(y_train, np.ndarray):
            raise ValueError("y_train must be a numpy array")
        if leaf_basis_train is not None:
            if not isinstance(leaf_basis_train, np.ndarray):
                raise ValueError("leaf_basis_train must be a numpy array")
        if leaf_basis_test is not None:
            if not isinstance(leaf_basis_test, np.ndarray):
                raise ValueError("X_test must be a numpy array")
        if rfx_group_ids_train is not None:
            if not isinstance(rfx_group_ids_train, np.ndarray):
                raise ValueError("rfx_group_ids_train must be a numpy array")
            if not np.issubdtype(rfx_group_ids_train.dtype, np.integer):
                raise ValueError(
                    "rfx_group_ids_train must be a numpy array of integer-valued group IDs"
                )
        if rfx_basis_train is not None:
            if not isinstance(rfx_basis_train, np.ndarray):
                raise ValueError("rfx_basis_train must be a numpy array")
        if rfx_group_ids_test is not None:
            if not isinstance(rfx_group_ids_test, np.ndarray):
                raise ValueError("rfx_group_ids_test must be a numpy array")
            if not np.issubdtype(rfx_group_ids_test.dtype, np.integer):
                raise ValueError(
                    "rfx_group_ids_test must be a numpy array of integer-valued group IDs"
                )
        if rfx_basis_test is not None:
            if not isinstance(rfx_basis_test, np.ndarray):
                raise ValueError("rfx_basis_test must be a numpy array")

        # Convert everything to standard shape (2-dimensional)
        if isinstance(X_train, np.ndarray):
            if X_train.ndim == 1:
                X_train = np.expand_dims(X_train, 1)
        if leaf_basis_train is not None:
            if leaf_basis_train.ndim == 1:
                leaf_basis_train = np.expand_dims(leaf_basis_train, 1)
        if y_train.ndim == 1:
            y_train = np.expand_dims(y_train, 1)
        if X_test is not None:
            if isinstance(X_test, np.ndarray):
                if X_test.ndim == 1:
                    X_test = np.expand_dims(X_test, 1)
        if leaf_basis_test is not None:
            if leaf_basis_test.ndim == 1:
                leaf_basis_test = np.expand_dims(leaf_basis_test, 1)
        if rfx_group_ids_train is not None:
            if rfx_group_ids_train.ndim != 1:
                rfx_group_ids_train = np.squeeze(rfx_group_ids_train)
        if rfx_group_ids_test is not None:
            if rfx_group_ids_test.ndim != 1:
                rfx_group_ids_test = np.squeeze(rfx_group_ids_test)
        if rfx_basis_train is not None:
            if rfx_basis_train.ndim == 1:
                rfx_basis_train = np.expand_dims(rfx_basis_train, 1)
        if rfx_basis_test is not None:
            if rfx_basis_test.ndim == 1:
                rfx_basis_test = np.expand_dims(rfx_basis_test, 1)

        # Data checks
        if X_test is not None:
            if X_test.shape[1] != X_train.shape[1]:
                raise ValueError(
                    "X_train and X_test must have the same number of columns"
                )
        if leaf_basis_test is not None:
            if leaf_basis_train is not None:
                if leaf_basis_test.shape[1] != leaf_basis_train.shape[1]:
                    raise ValueError(
                        "leaf_basis_train and leaf_basis_test must have the same number of columns"
                    )
            else:
                raise ValueError(
                    "leaf_basis_test provided but leaf_basis_train was not"
                )
        if leaf_basis_train is not None:
            if leaf_basis_train.shape[0] != X_train.shape[0]:
                raise ValueError(
                    "leaf_basis_train and Z_train must have the same number of rows"
                )
        if y_train.shape[0] != X_train.shape[0]:
            raise ValueError("X_train and y_train must have the same number of rows")
        if X_test is not None and leaf_basis_test is not None:
            if X_test.shape[0] != leaf_basis_test.shape[0]:
                raise ValueError(
                    "X_test and leaf_basis_test must have the same number of rows"
                )

        # Variable weight preprocessing (and initialization if necessary)
        p = X_train.shape[1]
        if variable_weights is None:
            if X_train.ndim > 1:
                variable_weights = np.repeat(1.0 / p, p)
            else:
                variable_weights = np.repeat(1.0, 1)
        if np.any(variable_weights < 0):
            raise ValueError("variable_weights cannot have any negative weights")
        variable_weights_mean = variable_weights
        variable_weights_variance = variable_weights

        # Covariate preprocessing
        self._covariate_preprocessor = CovariatePreprocessor()
        self._covariate_preprocessor.fit(X_train)
        X_train_processed = self._covariate_preprocessor.transform(X_train)
        if X_test is not None:
            X_test_processed = self._covariate_preprocessor.transform(X_test)
        feature_types = np.asarray(
            self._covariate_preprocessor._processed_feature_types
        )
        original_var_indices = (
            self._covariate_preprocessor.fetch_original_feature_indices()
        )
        num_features = len(feature_types)

        # Determine whether a test set is provided
        self.has_test = X_test is not None

        # Determine whether a basis is provided
        self.has_basis = leaf_basis_train is not None

        # Unpack data dimensions
        self.n_train = y_train.shape[0]
        self.n_test = X_test_processed.shape[0] if self.has_test else 0
        self.num_covariates = X_train_processed.shape[1]
        self.num_basis = leaf_basis_train.shape[1] if self.has_basis else 0

        # Standardize the keep variable lists to numeric indices
        if keep_vars_mean is not None:
            if isinstance(keep_vars_mean, list):
                if all(isinstance(i, str) for i in keep_vars_mean):
                    if not np.all(np.isin(keep_vars_mean, X_train.columns)):
                        raise ValueError(
                            "keep_vars_mean includes some variable names that are not in X_train"
                        )
                    variable_subset_mean = [
                        i
                        for i in X_train.shape[1]
                        if keep_vars_mean.count(X_train.columns.array[i]) > 0
                    ]
                elif all(isinstance(i, int) for i in keep_vars_mean):
                    if any(i >= X_train.shape[1] for i in keep_vars_mean):
                        raise ValueError(
                            "keep_vars_mean includes some variable indices that exceed the number of columns in X_train"
                        )
                    if any(i < 0 for i in keep_vars_mean):
                        raise ValueError(
                            "keep_vars_mean includes some negative variable indices"
                        )
                    variable_subset_mean = keep_vars_mean
                else:
                    raise ValueError(
                        "keep_vars_mean must be a list of variable names (str) or column indices (int)"
                    )
            elif isinstance(keep_vars_mean, np.ndarray):
                if keep_vars_mean.dtype == np.str_:
                    if not np.all(np.isin(keep_vars_mean, X_train.columns)):
                        raise ValueError(
                            "keep_vars_mean includes some variable names that are not in X_train"
                        )
                    variable_subset_mean = [
                        i
                        for i in X_train.shape[1]
                        if keep_vars_mean.count(X_train.columns.array[i]) > 0
                    ]
                else:
                    if np.any(keep_vars_mean >= X_train.shape[1]):
                        raise ValueError(
                            "keep_vars_mean includes some variable indices that exceed the number of columns in X_train"
                        )
                    if np.any(keep_vars_mean < 0):
                        raise ValueError(
                            "keep_vars_mean includes some negative variable indices"
                        )
                    variable_subset_mean = [i for i in keep_vars_mean]
            else:
                raise ValueError("keep_vars_mean must be a list or np.array")
        elif keep_vars_mean is None and drop_vars_mean is not None:
            if isinstance(drop_vars_mean, list):
                if all(isinstance(i, str) for i in drop_vars_mean):
                    if not np.all(np.isin(drop_vars_mean, X_train.columns)):
                        raise ValueError(
                            "drop_vars_mean includes some variable names that are not in X_train"
                        )
                    variable_subset_mean = [
                        i
                        for i in range(X_train.shape[1])
                        if drop_vars_mean.count(X_train.columns.array[i]) == 0
                    ]
                elif all(isinstance(i, int) for i in drop_vars_mean):
                    if any(i >= X_train.shape[1] for i in drop_vars_mean):
                        raise ValueError(
                            "drop_vars_mean includes some variable indices that exceed the number of columns in X_train"
                        )
                    if any(i < 0 for i in drop_vars_mean):
                        raise ValueError(
                            "drop_vars_mean includes some negative variable indices"
                        )
                    variable_subset_mean = [
                        i
                        for i in range(X_train.shape[1])
                        if drop_vars_mean.count(i) == 0
                    ]
                else:
                    raise ValueError(
                        "drop_vars_mean must be a list of variable names (str) or column indices (int)"
                    )
            elif isinstance(drop_vars_mean, np.ndarray):
                if drop_vars_mean.dtype == np.str_:
                    if not np.all(np.isin(drop_vars_mean, X_train.columns)):
                        raise ValueError(
                            "drop_vars_mean includes some variable names that are not in X_train"
                        )
                    keep_inds = ~np.isin(X_train.columns.array, drop_vars_mean)
                    variable_subset_mean = [i for i in keep_inds]
                else:
                    if np.any(drop_vars_mean >= X_train.shape[1]):
                        raise ValueError(
                            "drop_vars_mean includes some variable indices that exceed the number of columns in X_train"
                        )
                    if np.any(drop_vars_mean < 0):
                        raise ValueError(
                            "drop_vars_mean includes some negative variable indices"
                        )
                    keep_inds = ~np.isin(np.arange(X_train.shape[1]), drop_vars_mean)
                    variable_subset_mean = [i for i in keep_inds]
            else:
                raise ValueError("drop_vars_mean must be a list or np.array")
        else:
            variable_subset_mean = [i for i in range(X_train.shape[1])]
        if keep_vars_variance is not None:
            if isinstance(keep_vars_variance, list):
                if all(isinstance(i, str) for i in keep_vars_variance):
                    if not np.all(np.isin(keep_vars_variance, X_train.columns)):
                        raise ValueError(
                            "keep_vars_variance includes some variable names that are not in X_train"
                        )
                    variable_subset_variance = [
                        i
                        for i in X_train.shape[1]
                        if keep_vars_variance.count(X_train.columns.array[i]) > 0
                    ]
                elif all(isinstance(i, int) for i in keep_vars_variance):
                    if any(i >= X_train.shape[1] for i in keep_vars_variance):
                        raise ValueError(
                            "keep_vars_variance includes some variable indices that exceed the number of columns in X_train"
                        )
                    if any(i < 0 for i in keep_vars_variance):
                        raise ValueError(
                            "keep_vars_variance includes some negative variable indices"
                        )
                    variable_subset_variance = keep_vars_variance
                else:
                    raise ValueError(
                        "keep_vars_variance must be a list of variable names (str) or column indices (int)"
                    )
            elif isinstance(keep_vars_variance, np.ndarray):
                if keep_vars_variance.dtype == np.str_:
                    if not np.all(np.isin(keep_vars_variance, X_train.columns)):
                        raise ValueError(
                            "keep_vars_variance includes some variable names that are not in X_train"
                        )
                    variable_subset_variance = [
                        i
                        for i in X_train.shape[1]
                        if keep_vars_variance.count(X_train.columns.array[i]) > 0
                    ]
                else:
                    if np.any(keep_vars_variance >= X_train.shape[1]):
                        raise ValueError(
                            "keep_vars_variance includes some variable indices that exceed the number of columns in X_train"
                        )
                    if np.any(keep_vars_variance < 0):
                        raise ValueError(
                            "keep_vars_variance includes some negative variable indices"
                        )
                    variable_subset_variance = [i for i in keep_vars_variance]
            else:
                raise ValueError("keep_vars_variance must be a list or np.array")
        elif keep_vars_variance is None and drop_vars_variance is not None:
            if isinstance(drop_vars_variance, list):
                if all(isinstance(i, str) for i in drop_vars_variance):
                    if not np.all(np.isin(drop_vars_variance, X_train.columns)):
                        raise ValueError(
                            "drop_vars_variance includes some variable names that are not in X_train"
                        )
                    variable_subset_variance = [
                        i
                        for i in range(X_train.shape[1])
                        if drop_vars_variance.count(X_train.columns.array[i]) == 0
                    ]
                elif all(isinstance(i, int) for i in drop_vars_variance):
                    if any(i >= X_train.shape[1] for i in drop_vars_variance):
                        raise ValueError(
                            "drop_vars_variance includes some variable indices that exceed the number of columns in X_train"
                        )
                    if any(i < 0 for i in drop_vars_variance):
                        raise ValueError(
                            "drop_vars_variance includes some negative variable indices"
                        )
                    variable_subset_variance = [
                        i
                        for i in range(X_train.shape[1])
                        if drop_vars_variance.count(i) == 0
                    ]
                else:
                    raise ValueError(
                        "drop_vars_variance must be a list of variable names (str) or column indices (int)"
                    )
            elif isinstance(drop_vars_variance, np.ndarray):
                if drop_vars_variance.dtype == np.str_:
                    if not np.all(np.isin(drop_vars_variance, X_train.columns)):
                        raise ValueError(
                            "drop_vars_variance includes some variable names that are not in X_train"
                        )
                    keep_inds = ~np.isin(X_train.columns.array, drop_vars_variance)
                    variable_subset_variance = [i for i in keep_inds]
                else:
                    if np.any(drop_vars_variance >= X_train.shape[1]):
                        raise ValueError(
                            "drop_vars_variance includes some variable indices that exceed the number of columns in X_train"
                        )
                    if np.any(drop_vars_variance < 0):
                        raise ValueError(
                            "drop_vars_variance includes some negative variable indices"
                        )
                    keep_inds = ~np.isin(
                        np.arange(X_train.shape[1]), drop_vars_variance
                    )
                    variable_subset_variance = [i for i in keep_inds]
            else:
                raise ValueError("drop_vars_variance must be a list or np.array")
        else:
            variable_subset_variance = [i for i in range(X_train.shape[1])]

        # Check if previous model JSON is provided and parse it if so
        has_prev_model = previous_model_json is not None
        if has_prev_model:
            if num_gfr > 0:
                if num_mcmc == 0:
                    raise ValueError(
                        "A previous model is being used to initialize this sampler, so `num_mcmc` must be greater than zero"
                    )
                else:
                    warnings.warn(
                        "A previous model is being used to initialize this sampler, so num_gfr will be ignored and the MCMC sampler will be run from the previous samples"
                    )
            previous_bart_model = BARTModel()
            previous_bart_model.from_json(previous_model_json)
            previous_y_bar = previous_bart_model.y_bar
            previous_y_scale = previous_bart_model.y_std
            previous_model_num_samples = previous_bart_model.num_samples
            if previous_bart_model.include_mean_forest:
                previous_forest_samples_mean = previous_bart_model.forest_container_mean
            else:
                previous_forest_samples_mean = None
            if previous_bart_model.include_variance_forest:
                previous_forest_samples_variance = (
                    previous_bart_model.forest_container_variance
                )
            else:
                previous_forest_samples_variance = None
            if previous_bart_model.sample_sigma2_global:
                previous_global_var_samples = previous_bart_model.global_var_samples / (
                    previous_y_scale * previous_y_scale
                )
            else:
                previous_global_var_samples = None
            if previous_bart_model.sample_sigma2_leaf:
                previous_leaf_var_samples = previous_bart_model.leaf_scale_samples
            else:
                previous_leaf_var_samples = None
            if previous_bart_model.has_rfx:
                previous_rfx_samples = previous_bart_model.rfx_container
            else:
                previous_rfx_samples = None
            if previous_model_warmstart_sample_num + 1 > previous_model_num_samples:
                raise ValueError(
                    "`previous_model_warmstart_sample_num` exceeds the number of samples in `previous_model_json`"
                )
        else:
            previous_y_bar = None
            previous_y_scale = None
            previous_global_var_samples = None
            previous_leaf_var_samples = None
            previous_rfx_samples = None
            previous_forest_samples_mean = None
            previous_forest_samples_variance = None
            previous_model_num_samples = 0

        # Update variable weights if the covariates have been resized (by e.g. one-hot encoding)
        if X_train_processed.shape[1] != X_train.shape[1]:
            variable_counts = [
                original_var_indices.count(i) for i in original_var_indices
            ]
            variable_weights_adj = np.array([1 / i for i in variable_counts])
            if self.include_mean_forest:
                variable_weights_mean = (
                    variable_weights_mean[original_var_indices] * variable_weights_adj
                )
            if self.include_variance_forest:
                variable_weights_variance = (
                    variable_weights_variance[original_var_indices]
                    * variable_weights_adj
                )

        # Zero out weights for excluded variables
        variable_weights_mean[
            [variable_subset_mean.count(i) == 0 for i in original_var_indices]
        ] = 0
        variable_weights_variance[
            [variable_subset_variance.count(i) == 0 for i in original_var_indices]
        ] = 0

        # Preliminary runtime checks for probit link
        if not self.include_mean_forest:
            self.probit_outcome_model = False
        if self.probit_outcome_model:
            if np.unique(y_train).size != 2:
                raise ValueError(
                    "You specified a probit outcome model, but supplied an outcome with more than 2 unique values"
                )
            unique_outcomes = np.squeeze(np.unique(y_train))
            if not np.array_equal(unique_outcomes, [0, 1]):
                raise ValueError(
                    "You specified a probit outcome model, but supplied an outcome with 2 unique values other than 0 and 1"
                )
            if self.include_variance_forest:
                raise ValueError(
                    "We do not support heteroskedasticity with a probit link"
                )
            if sample_sigma2_global:
                warnings.warn(
                    "Global error variance will not be sampled with a probit link as it is fixed at 1"
                )
                sample_sigma2_global = False

        # Handle standardization, prior calibration, and initialization of forest
        # differently for binary and continuous outcomes
        if self.probit_outcome_model:
            # Compute a probit-scale offset and fix scale to 1
            self.y_bar = norm.ppf(np.squeeze(np.mean(y_train)))
            self.y_std = 1.0

            # Set a pseudo outcome by subtracting mean(y_train) from y_train
            resid_train = y_train - np.squeeze(np.mean(y_train))

            # Set initial values of root nodes to 0.0 (in probit scale)
            init_val_mean = 0.0

            # Calibrate priors for sigma^2 and tau
            # Set sigma2_init to 1, ignoring default provided
            sigma2_init = 1.0
            current_sigma2 = sigma2_init
            self.sigma2_init = sigma2_init
            # Skip variance_forest_init, since variance forests are not supported with probit link
            b_leaf = 1.0 / num_trees_mean if b_leaf is None else b_leaf
            if self.has_basis:
                if sigma2_leaf is None:
                    current_leaf_scale = np.zeros(
                        (self.num_basis, self.num_basis), dtype=float
                    )
                    np.fill_diagonal(
                        current_leaf_scale,
                        2.0 / num_trees_mean,
                    )
                elif isinstance(sigma2_leaf, float):
                    current_leaf_scale = np.zeros(
                        (self.num_basis, self.num_basis), dtype=float
                    )
                    np.fill_diagonal(current_leaf_scale, sigma2_leaf)
                elif isinstance(sigma2_leaf, np.ndarray):
                    if sigma2_leaf.ndim != 2:
                        raise ValueError(
                            "sigma2_leaf must be a 2d symmetric numpy array if provided in matrix form"
                        )
                    if sigma2_leaf.shape[0] != sigma2_leaf.shape[1]:
                        raise ValueError(
                            "sigma2_leaf must be a 2d symmetric numpy array if provided in matrix form"
                        )
                    if sigma2_leaf.shape[0] != self.num_basis:
                        raise ValueError(
                            "sigma2_leaf must be a 2d symmetric numpy array with its dimensionality matching the basis dimension"
                        )
                    current_leaf_scale = sigma2_leaf
                else:
                    raise ValueError(
                        "sigma2_leaf must be either a scalar or a 2d symmetric numpy array"
                    )
            else:
                if sigma2_leaf is None:
                    current_leaf_scale = np.array([[2.0 / num_trees_mean]])
                elif isinstance(sigma2_leaf, float):
                    current_leaf_scale = np.array([[sigma2_leaf]])
                elif isinstance(sigma2_leaf, np.ndarray):
                    if sigma2_leaf.ndim != 2:
                        raise ValueError(
                            "sigma2_leaf must be a 2d symmetric numpy array if provided in matrix form"
                        )
                    if sigma2_leaf.shape[0] != sigma2_leaf.shape[1]:
                        raise ValueError(
                            "sigma2_leaf must be a 2d symmetric numpy array if provided in matrix form"
                        )
                    if sigma2_leaf.shape[0] != 1:
                        raise ValueError(
                            "sigma2_leaf must be a 1x1 numpy array for this leaf model"
                        )
                    current_leaf_scale = sigma2_leaf
                else:
                    raise ValueError(
                        "sigma2_leaf must be either a scalar or a 2d numpy array"
                    )
        else:
            # Standardize if requested
            if self.standardize:
                self.y_bar = np.squeeze(np.mean(y_train))
                self.y_std = np.squeeze(np.std(y_train))
            else:
                self.y_bar = 0
                self.y_std = 1

            # Compute residual value
            resid_train = (y_train - self.y_bar) / self.y_std

            # Compute initial value of root nodes in mean forest
            init_val_mean = np.squeeze(np.mean(resid_train))

            # Calibrate priors for global sigma^2 and sigma2_leaf
            if not sigma2_init:
                sigma2_init = 1.0 * np.var(resid_train)
            if not variance_forest_leaf_init:
                variance_forest_leaf_init = 0.6 * np.var(resid_train)
            current_sigma2 = sigma2_init
            self.sigma2_init = sigma2_init
            if self.include_mean_forest:
                b_leaf = (
                    np.squeeze(np.var(resid_train)) / num_trees_mean
                    if b_leaf is None
                    else b_leaf
                )
                if self.has_basis:
                    if sigma2_leaf is None:
                        current_leaf_scale = np.zeros(
                            (self.num_basis, self.num_basis), dtype=float
                        )
                        np.fill_diagonal(
                            current_leaf_scale,
                            np.squeeze(np.var(resid_train)) / num_trees_mean,
                        )
                    elif isinstance(sigma2_leaf, float):
                        current_leaf_scale = np.zeros(
                            (self.num_basis, self.num_basis), dtype=float
                        )
                        np.fill_diagonal(current_leaf_scale, sigma2_leaf)
                    elif isinstance(sigma2_leaf, np.ndarray):
                        if sigma2_leaf.ndim != 2:
                            raise ValueError(
                                "sigma2_leaf must be a 2d symmetric numpy array if provided in matrix form"
                            )
                        if sigma2_leaf.shape[0] != sigma2_leaf.shape[1]:
                            raise ValueError(
                                "sigma2_leaf must be a 2d symmetric numpy array if provided in matrix form"
                            )
                        if sigma2_leaf.shape[0] != self.num_basis:
                            raise ValueError(
                                "sigma2_leaf must be a 2d symmetric numpy array with its dimensionality matching the basis dimension"
                            )
                        current_leaf_scale = sigma2_leaf
                    else:
                        raise ValueError(
                            "sigma2_leaf must be either a scalar or a 2d symmetric numpy array"
                        )
                else:
                    if sigma2_leaf is None:
                        current_leaf_scale = np.array(
                            [[np.squeeze(np.var(resid_train)) / num_trees_mean]]
                        )
                    elif isinstance(sigma2_leaf, float):
                        current_leaf_scale = np.array([[sigma2_leaf]])
                    elif isinstance(sigma2_leaf, np.ndarray):
                        if sigma2_leaf.ndim != 2:
                            raise ValueError(
                                "sigma2_leaf must be a 2d symmetric numpy array if provided in matrix form"
                            )
                        if sigma2_leaf.shape[0] != sigma2_leaf.shape[1]:
                            raise ValueError(
                                "sigma2_leaf must be a 2d symmetric numpy array if provided in matrix form"
                            )
                        if sigma2_leaf.shape[0] != 1:
                            raise ValueError(
                                "sigma2_leaf must be a 1x1 numpy array for this leaf model"
                            )
                        current_leaf_scale = sigma2_leaf
                    else:
                        raise ValueError(
                            "sigma2_leaf must be either a scalar or a 2d numpy array"
                        )
            else:
                current_leaf_scale = np.array([[1.0]])
            if self.include_variance_forest:
                if not a_forest:
                    a_forest = num_trees_variance / a_0**2 + 0.5
                if not b_forest:
                    b_forest = num_trees_variance / a_0**2
            else:
                if not a_forest:
                    a_forest = 1.0
                if not b_forest:
                    b_forest = 1.0

        # Runtime checks on RFX group ids
        self.has_rfx = False
        has_rfx_test = False
        if rfx_group_ids_train is not None:
            self.has_rfx = True
            if rfx_group_ids_test is not None:
                has_rfx_test = True
                if not np.all(np.isin(rfx_group_ids_test, rfx_group_ids_train)):
                    raise ValueError(
                        "All random effect group labels provided in rfx_group_ids_test must be present in rfx_group_ids_train"
                    )

        # Fill in rfx basis as a vector of 1s (random intercept) if a basis not provided
        has_basis_rfx = False
        if self.has_rfx:
            if rfx_basis_train is None:
                rfx_basis_train = np.ones((rfx_group_ids_train.shape[0], 1))
            else:
                has_basis_rfx = True
            num_rfx_groups = np.unique(rfx_group_ids_train).shape[0]
            num_rfx_components = rfx_basis_train.shape[1]
            # TODO warn if num_rfx_groups is 1
        if has_rfx_test:
            if rfx_basis_test is None:
                if has_basis_rfx:
                    raise ValueError(
                        "Random effects basis provided for training set, must also be provided for the test set"
                    )
                rfx_basis_test = np.ones((rfx_group_ids_test.shape[0], 1))

        # Set up random effects structures
        if self.has_rfx:
            if num_rfx_components == 1:
                alpha_init = np.array([1])
            elif num_rfx_components > 1:
                alpha_init = np.concatenate(
                    (
                        np.ones(1, dtype=float),
                        np.zeros(num_rfx_components - 1, dtype=float),
                    )
                )
            else:
                raise ValueError("There must be at least 1 random effect component")
            xi_init = np.tile(np.expand_dims(alpha_init, 1), (1, num_rfx_groups))
            sigma_alpha_init = np.identity(num_rfx_components)
            sigma_xi_init = np.identity(num_rfx_components)
            sigma_xi_shape = 1.0
            sigma_xi_scale = 1.0
            rfx_dataset_train = RandomEffectsDataset()
            rfx_dataset_train.add_group_labels(rfx_group_ids_train)
            rfx_dataset_train.add_basis(rfx_basis_train)
            rfx_tracker = RandomEffectsTracker(rfx_group_ids_train)
            rfx_model = RandomEffectsModel(num_rfx_components, num_rfx_groups)
            rfx_model.set_working_parameter(alpha_init)
            rfx_model.set_group_parameters(xi_init)
            rfx_model.set_working_parameter_covariance(sigma_alpha_init)
            rfx_model.set_group_parameter_covariance(sigma_xi_init)
            rfx_model.set_variance_prior_shape(sigma_xi_shape)
            rfx_model.set_variance_prior_scale(sigma_xi_scale)
            self.rfx_container = RandomEffectsContainer()
            self.rfx_container.load_new_container(
                num_rfx_components, num_rfx_groups, rfx_tracker
            )

        # Container of variance parameter samples
        self.num_gfr = num_gfr
        self.num_burnin = num_burnin
        self.num_mcmc = num_mcmc
        num_temp_samples = num_gfr + num_burnin + num_mcmc * keep_every
        num_retained_samples = num_mcmc * num_chains
        # Delete GFR samples from these containers after the fact if desired
        # if keep_gfr:
        #     num_retained_samples += num_gfr
        num_retained_samples += num_gfr
        if keep_burnin:
            num_retained_samples += num_burnin * num_chains
        self.num_samples = num_retained_samples
        self.sample_sigma2_global = sample_sigma2_global
        self.sample_sigma2_leaf = sample_sigma2_leaf
        if sample_sigma2_global:
            self.global_var_samples = np.empty(self.num_samples, dtype=np.float64)
        if sample_sigma2_leaf:
            self.leaf_scale_samples = np.empty(self.num_samples, dtype=np.float64)
        sample_counter = -1

        # Forest Dataset (covariates and optional basis)
        forest_dataset_train = Dataset()
        forest_dataset_train.add_covariates(X_train_processed)
        if self.has_basis:
            forest_dataset_train.add_basis(leaf_basis_train)
        if self.has_test:
            forest_dataset_test = Dataset()
            forest_dataset_test.add_covariates(X_test_processed)
            if self.has_basis:
                forest_dataset_test.add_basis(leaf_basis_test)

        # Residual
        residual_train = Residual(resid_train)

        # C++ and Numpy random number generator
        if random_seed is None:
            cpp_rng = RNG(-1)
            self.rng = np.random.default_rng()
        else:
            cpp_rng = RNG(random_seed)
            self.rng = np.random.default_rng(random_seed)

        # Set variance leaf model type (currently only one option)
        leaf_model_variance_forest = 3
        leaf_dimension_variance = 1

        # Determine the mean forest leaf model type
        if not self.has_basis:
            leaf_model_mean_forest = 0
            leaf_dimension_mean = 1
        elif self.num_basis == 1:
            leaf_model_mean_forest = 1
            leaf_dimension_mean = 1
        else:
            leaf_model_mean_forest = 2
            leaf_dimension_mean = self.num_basis

        # Sampling data structures
        global_model_config = GlobalModelConfig(global_error_variance=current_sigma2)
        if self.include_mean_forest:
            forest_model_config_mean = ForestModelConfig(
                num_trees=num_trees_mean,
                num_features=num_features,
                num_observations=self.n_train,
                feature_types=feature_types,
                variable_weights=variable_weights_mean,
                leaf_dimension=leaf_dimension_mean,
                alpha=alpha_mean,
                beta=beta_mean,
                min_samples_leaf=min_samples_leaf_mean,
                max_depth=max_depth_mean,
                leaf_model_type=leaf_model_mean_forest,
                leaf_model_scale=current_leaf_scale,
                cutpoint_grid_size=cutpoint_grid_size,
            )
            forest_sampler_mean = ForestSampler(
                forest_dataset_train,
                global_model_config,
                forest_model_config_mean,
            )
        if self.include_variance_forest:
            forest_model_config_variance = ForestModelConfig(
                num_trees=num_trees_variance,
                num_features=num_features,
                num_observations=self.n_train,
                feature_types=feature_types,
                variable_weights=variable_weights_variance,
                leaf_dimension=leaf_dimension_variance,
                alpha=alpha_variance,
                beta=beta_variance,
                min_samples_leaf=min_samples_leaf_variance,
                max_depth=max_depth_variance,
                leaf_model_type=leaf_model_variance_forest,
                cutpoint_grid_size=cutpoint_grid_size,
                variance_forest_shape=a_forest,
                variance_forest_scale=b_forest,
            )
            forest_sampler_variance = ForestSampler(
                forest_dataset_train,
                global_model_config,
                forest_model_config_variance,
            )

        # Container of forest samples
        if self.include_mean_forest:
            self.forest_container_mean = (
                ForestContainer(num_trees_mean, 1, True, False)
                if not self.has_basis
                else ForestContainer(num_trees_mean, self.num_basis, False, False)
            )
            active_forest_mean = (
                Forest(num_trees_mean, 1, True, False)
                if not self.has_basis
                else Forest(num_trees_mean, self.num_basis, False, False)
            )
        if self.include_variance_forest:
            self.forest_container_variance = ForestContainer(
                num_trees_variance, 1, True, True
            )
            active_forest_variance = Forest(num_trees_variance, 1, True, True)

        # Variance samplers
        if self.sample_sigma2_global:
            global_var_model = GlobalVarianceModel()
        if self.sample_sigma2_leaf:
            leaf_var_model = LeafVarianceModel()

        # Initialize the leaves of each tree in the mean forest
        if self.include_mean_forest:
            if self.has_basis:
                init_val_mean = np.repeat(0.0, leaf_basis_train.shape[1])
            else:
                init_val_mean = np.array([0.0])
            forest_sampler_mean.prepare_for_sampler(
                forest_dataset_train,
                residual_train,
                active_forest_mean,
                leaf_model_mean_forest,
                init_val_mean,
            )

        # Initialize the leaves of each tree in the variance forest
        if self.include_variance_forest:
            init_val_variance = np.array([variance_forest_leaf_init])
            forest_sampler_variance.prepare_for_sampler(
                forest_dataset_train,
                residual_train,
                active_forest_variance,
                leaf_model_variance_forest,
                init_val_variance,
            )

        # Run GFR (warm start) if specified
        if self.num_gfr > 0:
            for i in range(self.num_gfr):
                # Keep all GFR samples at this stage -- remove from ForestSamples after MCMC
                # keep_sample = keep_gfr
                keep_sample = True
                if keep_sample:
                    sample_counter += 1
                if self.include_mean_forest:
                    if self.probit_outcome_model:
                        # Sample latent probit variable z | -
                        forest_pred = active_forest_mean.predict(forest_dataset_train)
                        mu0 = forest_pred[y_train[:, 0] == 0]
                        mu1 = forest_pred[y_train[:, 0] == 1]
                        n0 = np.sum(y_train[:, 0] == 0)
                        n1 = np.sum(y_train[:, 0] == 1)
                        u0 = self.rng.uniform(
                            low=0.0,
                            high=norm.cdf(0 - mu0),
                            size=n0,
                        )
                        u1 = self.rng.uniform(
                            low=norm.cdf(0 - mu1),
                            high=1.0,
                            size=n1,
                        )
                        resid_train[y_train[:, 0] == 0, 0] = mu0 + norm.ppf(u0)
                        resid_train[y_train[:, 0] == 1, 0] = mu1 + norm.ppf(u1)

                        # Update outcome
                        new_outcome = np.squeeze(resid_train) - forest_pred
                        residual_train.update_data(new_outcome)

                    # Sample the mean forest
                    forest_sampler_mean.sample_one_iteration(
                        self.forest_container_mean,
                        active_forest_mean,
                        forest_dataset_train,
                        residual_train,
                        cpp_rng,
                        global_model_config,
                        forest_model_config_mean,
                        keep_sample,
                        True,
                    )

                # Sample the variance forest
                if self.include_variance_forest:
                    forest_sampler_variance.sample_one_iteration(
                        self.forest_container_variance,
                        active_forest_variance,
                        forest_dataset_train,
                        residual_train,
                        cpp_rng,
                        global_model_config,
                        forest_model_config_variance,
                        keep_sample,
                        True,
                    )

                # Sample variance parameters (if requested)
                if self.sample_sigma2_global:
                    current_sigma2 = global_var_model.sample_one_iteration(
                        residual_train, cpp_rng, a_global, b_global
                    )
                    global_model_config.update_global_error_variance(current_sigma2)
                    if keep_sample:
                        self.global_var_samples[sample_counter] = current_sigma2
                if self.sample_sigma2_leaf:
                    current_leaf_scale[0, 0] = leaf_var_model.sample_one_iteration(
                        active_forest_mean, cpp_rng, a_leaf, b_leaf
                    )
                    forest_model_config_mean.update_leaf_model_scale(current_leaf_scale)
                    if keep_sample:
                        self.leaf_scale_samples[sample_counter] = current_leaf_scale[
                            0, 0
                        ]

                # Sample random effects
                if self.has_rfx:
                    rfx_model.sample(
                        rfx_dataset_train,
                        residual_train,
                        rfx_tracker,
                        self.rfx_container,
                        keep_sample,
                        current_sigma2,
                        cpp_rng,
                    )

        # Run MCMC
        if self.num_burnin + self.num_mcmc > 0:
            for chain_num in range(num_chains):
                if num_gfr > 0:
                    forest_ind = num_gfr - chain_num - 1
                    if self.include_mean_forest:
                        active_forest_mean.reset(self.forest_container_mean, forest_ind)
                        forest_sampler_mean.reconstitute_from_forest(
                            active_forest_mean,
                            forest_dataset_train,
                            residual_train,
                            True,
                        )
                    if self.include_variance_forest:
                        active_forest_variance.reset(
                            self.forest_container_variance, forest_ind
                        )
                        forest_sampler_variance.reconstitute_from_forest(
                            active_forest_variance,
                            forest_dataset_train,
                            residual_train,
                            False,
                        )
                    if sample_sigma2_global:
                        current_sigma2 = self.global_var_samples[forest_ind]
                elif has_prev_model:
                    if self.include_mean_forest:
                        active_forest_mean.reset(
                            previous_bart_model.forest_container_mean,
                            previous_model_warmstart_sample_num,
                        )
                        forest_sampler_mean.reconstitute_from_forest(
                            active_forest_mean,
                            forest_dataset_train,
                            residual_train,
                            True,
                        )
                        if sample_sigma2_leaf and previous_leaf_var_samples is not None:
                            leaf_scale_double = previous_leaf_var_samples[
                                previous_model_warmstart_sample_num
                            ]
                            current_leaf_scale[0, 0] = leaf_scale_double
                            forest_model_config_mean.update_leaf_model_scale(
                                leaf_scale_double
                            )
                    if self.include_variance_forest:
                        active_forest_variance.reset(
                            previous_bart_model.forest_container_variance,
                            previous_model_warmstart_sample_num,
                        )
                        forest_sampler_variance.reconstitute_from_forest(
                            active_forest_variance,
                            forest_dataset_train,
                            residual_train,
                            True,
                        )
                    # if self.has_rfx:
                    #     pass
                    if self.sample_sigma2_global:
                        current_sigma2 = previous_global_var_samples[
                            previous_model_warmstart_sample_num
                        ]
                        global_model_config.update_global_error_variance(current_sigma2)
                else:
                    if self.include_mean_forest:
                        active_forest_mean.reset_root()
                        if init_val_mean.shape[0] == 1:
                            active_forest_mean.set_root_leaves(
                                init_val_mean[0] / num_trees_mean
                            )
                        else:
                            active_forest_mean.set_root_leaves(
                                init_val_mean / num_trees_mean
                            )
                        forest_sampler_mean.reconstitute_from_forest(
                            active_forest_mean,
                            forest_dataset_train,
                            residual_train,
                            True,
                        )
                    if self.include_variance_forest:
                        active_forest_variance.reset_root()
                        active_forest_variance.set_root_leaves(
                            log(variance_forest_leaf_init) / num_trees_mean
                        )
                        forest_sampler_variance.reconstitute_from_forest(
                            active_forest_variance,
                            forest_dataset_train,
                            residual_train,
                            False,
                        )

                for i in range(self.num_gfr, num_temp_samples):
                    is_mcmc = i + 1 > num_gfr + num_burnin
                    if is_mcmc:
                        mcmc_counter = i - num_gfr - num_burnin + 1
                        if mcmc_counter % keep_every == 0:
                            keep_sample = True
                        else:
                            keep_sample = False
                    else:
                        if keep_burnin:
                            keep_sample = True
                        else:
                            keep_sample = False
                    if keep_sample:
                        sample_counter += 1

                    if self.include_mean_forest:
                        if self.probit_outcome_model:
                            # Sample latent probit variable z | -
                            forest_pred = active_forest_mean.predict(
                                forest_dataset_train
                            )
                            mu0 = forest_pred[y_train[:, 0] == 0]
                            mu1 = forest_pred[y_train[:, 0] == 1]
                            n0 = np.sum(y_train[:, 0] == 0)
                            n1 = np.sum(y_train[:, 0] == 1)
                            u0 = self.rng.uniform(
                                low=0.0,
                                high=norm.cdf(0 - mu0),
                                size=n0,
                            )
                            u1 = self.rng.uniform(
                                low=norm.cdf(0 - mu1),
                                high=1.0,
                                size=n1,
                            )
                            resid_train[y_train[:, 0] == 0, 0] = mu0 + norm.ppf(u0)
                            resid_train[y_train[:, 0] == 1, 0] = mu1 + norm.ppf(u1)

                            # Update outcome
                            new_outcome = np.squeeze(resid_train) - forest_pred
                            residual_train.update_data(new_outcome)

                        # Sample the mean forest
                        forest_sampler_mean.sample_one_iteration(
                            self.forest_container_mean,
                            active_forest_mean,
                            forest_dataset_train,
                            residual_train,
                            cpp_rng,
                            global_model_config,
                            forest_model_config_mean,
                            keep_sample,
                            False,
                        )

                    # Sample the variance forest
                    if self.include_variance_forest:
                        forest_sampler_variance.sample_one_iteration(
                            self.forest_container_variance,
                            active_forest_variance,
                            forest_dataset_train,
                            residual_train,
                            cpp_rng,
                            global_model_config,
                            forest_model_config_variance,
                            keep_sample,
                            False,
                        )

                    # Sample variance parameters (if requested)
                    if self.sample_sigma2_global:
                        current_sigma2 = global_var_model.sample_one_iteration(
                            residual_train, cpp_rng, a_global, b_global
                        )
                        global_model_config.update_global_error_variance(current_sigma2)
                        if keep_sample:
                            self.global_var_samples[sample_counter] = current_sigma2
                    if self.sample_sigma2_leaf:
                        current_leaf_scale[0, 0] = leaf_var_model.sample_one_iteration(
                            active_forest_mean, cpp_rng, a_leaf, b_leaf
                        )
                        forest_model_config_mean.update_leaf_model_scale(
                            current_leaf_scale
                        )
                        if keep_sample:
                            self.leaf_scale_samples[sample_counter] = (
                                current_leaf_scale[0, 0]
                            )

                    # Sample random effects
                    if self.has_rfx:
                        rfx_model.sample(
                            rfx_dataset_train,
                            residual_train,
                            rfx_tracker,
                            self.rfx_container,
                            keep_sample,
                            current_sigma2,
                            cpp_rng,
                        )

        # Mark the model as sampled
        self.sampled = True

        # Remove GFR samples if they are not to be retained
        if not keep_gfr and num_gfr > 0:
            for i in range(num_gfr):
                if self.include_mean_forest:
                    self.forest_container_mean.delete_sample(0)
                if self.include_variance_forest:
                    self.forest_container_variance.delete_sample(0)
                if self.has_rfx:
                    self.rfx_container.delete_sample(0)
            if self.sample_sigma2_global:
                self.global_var_samples = self.global_var_samples[num_gfr:]
            if self.sample_sigma2_leaf:
                self.leaf_scale_samples = self.leaf_scale_samples[num_gfr:]
            self.num_samples -= num_gfr

        # Store predictions
        if self.sample_sigma2_global:
            self.global_var_samples = self.global_var_samples * self.y_std * self.y_std

        if self.sample_sigma2_leaf:
            self.leaf_scale_samples = self.leaf_scale_samples

        if self.include_mean_forest:
            yhat_train_raw = self.forest_container_mean.forest_container_cpp.Predict(
                forest_dataset_train.dataset_cpp
            )
            self.y_hat_train = yhat_train_raw * self.y_std + self.y_bar
            if self.has_test:
                yhat_test_raw = self.forest_container_mean.forest_container_cpp.Predict(
                    forest_dataset_test.dataset_cpp
                )
                self.y_hat_test = yhat_test_raw * self.y_std + self.y_bar

        # TODO: make rfx_preds_train and rfx_preds_test persistent properties
        if self.has_rfx:
            rfx_preds_train = (
                self.rfx_container.predict(rfx_group_ids_train, rfx_basis_train)
                * self.y_std
            )
            if has_rfx_test:
                rfx_preds_test = (
                    self.rfx_container.predict(rfx_group_ids_test, rfx_basis_test)
                    * self.y_std
                )
            if self.include_mean_forest:
                self.y_hat_train = self.y_hat_train + rfx_preds_train
                if self.has_test:
                    self.y_hat_test = self.y_hat_test + rfx_preds_test
            else:
                self.y_hat_train = rfx_preds_train
                if self.has_test:
                    self.y_hat_test = rfx_preds_test

        if self.include_variance_forest:
            sigma2_x_train_raw = (
                self.forest_container_variance.forest_container_cpp.Predict(
                    forest_dataset_train.dataset_cpp
                )
            )
            if self.sample_sigma2_global:
                self.sigma2_x_train = sigma2_x_train_raw
                for i in range(self.num_samples):
                    self.sigma2_x_train[:, i] = (
                        sigma2_x_train_raw[:, i] * self.global_var_samples[i]
                    )
            else:
                self.sigma2_x_train = (
                    sigma2_x_train_raw * self.sigma2_init * self.y_std * self.y_std
                )
            if self.has_test:
                sigma2_x_test_raw = (
                    self.forest_container_variance.forest_container_cpp.Predict(
                        forest_dataset_test.dataset_cpp
                    )
                )
                if self.sample_sigma2_global:
                    self.sigma2_x_test = sigma2_x_test_raw
                    for i in range(self.num_samples):
                        self.sigma2_x_test[:, i] = (
                            sigma2_x_test_raw[:, i] * self.global_var_samples[i]
                        )
                else:
                    self.sigma2_x_test = (
                        sigma2_x_test_raw * self.sigma2_init * self.y_std * self.y_std
                    )

    def predict(
        self,
        covariates: Union[np.array, pd.DataFrame],
        basis: np.array = None,
        rfx_group_ids: np.array = None,
        rfx_basis: np.array = None,
    ) -> Union[np.array, tuple]:
        """Return predictions from every forest sampled (either / both of mean and variance).
        Return type is either a single array of predictions, if a BART model only includes a
        mean or variance term, or a tuple of prediction arrays, if a BART model includes both.

        Parameters
        ----------
        covariates : np.array
            Test set covariates.
        basis : np.array, optional
            Optional test set basis vector, must be provided if the model was trained with a leaf regression basis.
        rfx_group_ids : np.array, optional
            Optional group labels used for an additive random effects model.
        rfx_basis : np.array, optional
            Optional basis for "random-slope" regression in an additive random effects model.

        Returns
        -------
        mu_x : np.array, optional
            Mean forest and / or random effects predictions.
        sigma2_x : np.array, optional
            Variance forest predictions.
        """
        if not self.is_sampled():
            msg = (
                "This BARTModel instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this model."
            )
            raise NotSampledError(msg)

        # Data checks
        if not isinstance(covariates, pd.DataFrame) and not isinstance(
            covariates, np.ndarray
        ):
            raise ValueError("covariates must be a pandas dataframe or numpy array")
        if basis is not None:
            if not isinstance(basis, np.ndarray):
                raise ValueError("basis must be a numpy array")
            if basis.shape[0] != covariates.shape[0]:
                raise ValueError(
                    "covariates and basis must have the same number of rows"
                )

        # Convert everything to standard shape (2-dimensional)
        if isinstance(covariates, np.ndarray):
            if covariates.ndim == 1:
                covariates = np.expand_dims(covariates, 1)
        if basis is not None:
            if basis.ndim == 1:
                basis = np.expand_dims(basis, 1)

        # Covariate preprocessing
        if not self._covariate_preprocessor._check_is_fitted():
            if not isinstance(covariates, np.ndarray):
                raise ValueError(
                    "Prediction cannot proceed on a pandas dataframe, since the BART model was not fit with a covariate preprocessor. Please refit your model by passing covariate data as a Pandas dataframe."
                )
            else:
                warnings.warn(
                    "This BART model has not run any covariate preprocessing routines. We will attempt to predict on the raw covariate values, but this will trigger an error with non-numeric columns. Please refit your model by passing non-numeric covariate data a a Pandas dataframe.",
                    RuntimeWarning,
                )
                if not np.issubdtype(
                    covariates.dtype, np.floating
                ) and not np.issubdtype(covariates.dtype, np.integer):
                    raise ValueError(
                        "Prediction cannot proceed on a non-numeric numpy array, since the BART model was not fit with a covariate preprocessor. Please refit your model by passing non-numeric covariate data as a Pandas dataframe."
                    )
                covariates_processed = covariates
        else:
            covariates_processed = self._covariate_preprocessor.transform(covariates)

        # Dataset construction
        pred_dataset = Dataset()
        pred_dataset.add_covariates(covariates_processed)
        if basis is not None:
            pred_dataset.add_basis(basis)

        # Forest predictions
        if self.include_mean_forest:
            mean_pred_raw = self.forest_container_mean.forest_container_cpp.Predict(
                pred_dataset.dataset_cpp
            )
            mean_pred = mean_pred_raw * self.y_std + self.y_bar

        if self.has_rfx:
            rfx_preds = (
                self.rfx_container.predict(rfx_group_ids, rfx_basis) * self.y_std
            )
            if self.include_mean_forest:
                mean_pred = mean_pred + rfx_preds
            else:
                mean_pred = rfx_preds + self.y_bar

        if self.include_variance_forest:
            variance_pred_raw = (
                self.forest_container_variance.forest_container_cpp.Predict(
                    pred_dataset.dataset_cpp
                )
            )
            if self.sample_sigma2_global:
                variance_pred = variance_pred_raw
                for i in range(self.num_samples):
                    variance_pred[:, i] = np.sqrt(
                        variance_pred_raw[:, i] * self.global_var_samples[i]
                    )
            else:
                variance_pred = (
                    np.sqrt(variance_pred_raw * self.sigma2_init) * self.y_std
                )

        has_mean_predictions = self.include_mean_forest or self.has_rfx
        if has_mean_predictions and self.include_variance_forest:
            return (mean_pred, variance_pred)
        elif has_mean_predictions and not self.include_variance_forest:
            return mean_pred
        elif not has_mean_predictions and self.include_variance_forest:
            return variance_pred

    def predict_mean(
        self,
        covariates: np.array,
        basis: np.array = None,
        rfx_group_ids: np.array = None,
        rfx_basis: np.array = None,
    ) -> np.array:
        """Predict expected conditional outcome from a BART model.

        Parameters
        ----------
        covariates : np.array
            Test set covariates.
        basis : np.array, optional
            Optional test set basis vector, must be provided if the model was trained with a leaf regression basis.

        Returns
        -------
        np.array
            Mean forest predictions.
        """
        if not self.is_sampled():
            msg = (
                "This BARTModel instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this model."
            )
            raise NotSampledError(msg)

        has_mean_predictions = self.include_mean_forest or self.has_rfx
        if not has_mean_predictions:
            msg = (
                "This BARTModel instance was not sampled with a mean forest or random effects. "
                "Call 'fit' with appropriate arguments before using this model."
            )
            raise NotSampledError(msg)

        # Data checks
        if not isinstance(covariates, pd.DataFrame) and not isinstance(
            covariates, np.ndarray
        ):
            raise ValueError("covariates must be a pandas dataframe or numpy array")
        if basis is not None:
            if not isinstance(basis, np.ndarray):
                raise ValueError("basis must be a numpy array")
            if basis.shape[0] != covariates.shape[0]:
                raise ValueError(
                    "covariates and basis must have the same number of rows"
                )

        # Convert everything to standard shape (2-dimensional)
        if isinstance(covariates, np.ndarray):
            if covariates.ndim == 1:
                covariates = np.expand_dims(covariates, 1)
        if basis is not None:
            if basis.ndim == 1:
                basis = np.expand_dims(basis, 1)

        # Covariate preprocessing
        if not self._covariate_preprocessor._check_is_fitted():
            if not isinstance(covariates, np.ndarray):
                raise ValueError(
                    "Prediction cannot proceed on a pandas dataframe, since the BART model was not fit with a covariate preprocessor. Please refit your model by passing covariate data as a Pandas dataframe."
                )
            else:
                warnings.warn(
                    "This BART model has not run any covariate preprocessing routines. We will attempt to predict on the raw covariate values, but this will trigger an error with non-numeric columns. Please refit your model by passing non-numeric covariate data a a Pandas dataframe.",
                    RuntimeWarning,
                )
                if not np.issubdtype(
                    covariates.dtype, np.floating
                ) and not np.issubdtype(covariates.dtype, np.integer):
                    raise ValueError(
                        "Prediction cannot proceed on a non-numeric numpy array, since the BART model was not fit with a covariate preprocessor. Please refit your model by passing non-numeric covariate data as a Pandas dataframe."
                    )
                covariates_processed = covariates
        else:
            covariates_processed = self._covariate_preprocessor.transform(covariates)

        # Dataset construction
        pred_dataset = Dataset()
        pred_dataset.add_covariates(covariates_processed)
        if basis is not None:
            pred_dataset.add_basis(basis)

        # Mean forest predictions
        if self.include_mean_forest:
            mean_pred_raw = self.forest_container_mean.forest_container_cpp.Predict(
                pred_dataset.dataset_cpp
            )
            mean_pred = mean_pred_raw * self.y_std + self.y_bar

        # RFX predictions
        if self.has_rfx:
            rfx_preds = (
                self.rfx_container.predict(rfx_group_ids, rfx_basis) * self.y_std
            )
            if self.include_mean_forest:
                mean_pred = mean_pred + rfx_preds
            else:
                mean_pred = rfx_preds + self.y_bar

        return mean_pred

    def predict_variance(self, covariates: np.array) -> np.array:
        """Predict expected conditional variance from a BART model.

        Parameters
        ----------
        covariates : np.array
            Test set covariates.

        Returns
        -------
        np.array
            Variance forest predictions.
        """
        if not self.is_sampled():
            msg = (
                "This BARTModel instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this model."
            )
            raise NotSampledError(msg)

        if not self.include_variance_forest:
            msg = (
                "This BARTModel instance was not sampled with a variance forest. "
                "Call 'fit' with appropriate arguments before using this model."
            )
            raise NotSampledError(msg)

        # Data checks
        if not isinstance(covariates, pd.DataFrame) and not isinstance(
            covariates, np.ndarray
        ):
            raise ValueError("covariates must be a pandas dataframe or numpy array")

        # Convert everything to standard shape (2-dimensional)
        if isinstance(covariates, np.ndarray):
            if covariates.ndim == 1:
                covariates = np.expand_dims(covariates, 1)

        # Covariate preprocessing
        if not self._covariate_preprocessor._check_is_fitted():
            if not isinstance(covariates, np.ndarray):
                raise ValueError(
                    "Prediction cannot proceed on a pandas dataframe, since the BART model was not fit with a covariate preprocessor. Please refit your model by passing covariate data as a Pandas dataframe."
                )
            else:
                warnings.warn(
                    "This BART model has not run any covariate preprocessing routines. We will attempt to predict on the raw covariate values, but this will trigger an error with non-numeric columns. Please refit your model by passing non-numeric covariate data a a Pandas dataframe.",
                    RuntimeWarning,
                )
                if not np.issubdtype(
                    covariates.dtype, np.floating
                ) and not np.issubdtype(covariates.dtype, np.integer):
                    raise ValueError(
                        "Prediction cannot proceed on a non-numeric numpy array, since the BART model was not fit with a covariate preprocessor. Please refit your model by passing non-numeric covariate data as a Pandas dataframe."
                    )
                covariates_processed = covariates
        else:
            covariates_processed = self._covariate_preprocessor.transform(covariates)

        # Dataset construction
        pred_dataset = Dataset()
        pred_dataset.add_covariates(covariates_processed)

        # Variance forest predictions
        variance_pred_raw = self.forest_container_variance.forest_container_cpp.Predict(
            pred_dataset.dataset_cpp
        )
        if self.sample_sigma2_global:
            variance_pred = variance_pred_raw
            for i in range(self.num_samples):
                variance_pred[:, i] = (
                    variance_pred_raw[:, i] * self.global_var_samples[i]
                )
        else:
            variance_pred = (
                variance_pred_raw * self.sigma2_init * self.y_std * self.y_std
            )

        return variance_pred

    def to_json(self) -> str:
        """
        Converts a sampled BART model to JSON string representation (which can then be saved to a file or
        processed using the `json` library)

        Returns
        -------
        str
            JSON string representing model metadata (hyperparameters), sampled parameters, and sampled forests
        """
        if not self.is_sampled:
            msg = (
                "This BARTModel instance has not yet been sampled. "
                "Call 'fit' with appropriate arguments before using this model."
            )
            raise NotSampledError(msg)

        # Initialize JSONSerializer object
        bart_json = JSONSerializer()

        # Add the forests
        if self.include_mean_forest:
            bart_json.add_forest(self.forest_container_mean)
        if self.include_variance_forest:
            bart_json.add_forest(self.forest_container_variance)

        # Add the rfx
        if self.has_rfx:
            bart_json.add_random_effects(self.rfx_container)

        # Add global parameters
        bart_json.add_scalar("outcome_scale", self.y_std)
        bart_json.add_scalar("outcome_mean", self.y_bar)
        bart_json.add_boolean("standardize", self.standardize)
        bart_json.add_scalar("sigma2_init", self.sigma2_init)
        bart_json.add_boolean("sample_sigma2_global", self.sample_sigma2_global)
        bart_json.add_boolean("sample_sigma2_leaf", self.sample_sigma2_leaf)
        bart_json.add_boolean("include_mean_forest", self.include_mean_forest)
        bart_json.add_boolean("include_variance_forest", self.include_variance_forest)
        bart_json.add_boolean("has_rfx", self.has_rfx)
        bart_json.add_integer("num_gfr", self.num_gfr)
        bart_json.add_integer("num_burnin", self.num_burnin)
        bart_json.add_integer("num_mcmc", self.num_mcmc)
        bart_json.add_integer("num_samples", self.num_samples)
        bart_json.add_integer("num_basis", self.num_basis)
        bart_json.add_boolean("requires_basis", self.has_basis)
        bart_json.add_boolean("probit_outcome_model", self.probit_outcome_model)

        # Add parameter samples
        if self.sample_sigma2_global:
            bart_json.add_numeric_vector(
                "sigma2_global_samples", self.global_var_samples, "parameters"
            )
        if self.sample_sigma2_leaf:
            bart_json.add_numeric_vector(
                "sigma2_leaf_samples", self.leaf_scale_samples, "parameters"
            )

        # Add covariate preprocessor
        covariate_preprocessor_string = self._covariate_preprocessor.to_json()
        bart_json.add_string("covariate_preprocessor", covariate_preprocessor_string)

        return bart_json.return_json_string()

    def from_json(self, json_string: str) -> None:
        """
        Converts a JSON string to an in-memory BART model.

        Parameters
        ----------
        json_string : str
            JSON string representing model metadata (hyperparameters), sampled parameters, and sampled forests
        """
        # Parse string to a JSON object in C++
        bart_json = JSONSerializer()
        bart_json.load_from_json_string(json_string)

        # Unpack forests
        self.include_mean_forest = bart_json.get_boolean("include_mean_forest")
        self.include_variance_forest = bart_json.get_boolean("include_variance_forest")
        self.has_rfx = bart_json.get_boolean("has_rfx")
        if self.include_mean_forest:
            # TODO: don't just make this a placeholder that we overwrite
            self.forest_container_mean = ForestContainer(0, 0, False, False)
            self.forest_container_mean.forest_container_cpp.LoadFromJson(
                bart_json.json_cpp, "forest_0"
            )
            if self.include_variance_forest:
                # TODO: don't just make this a placeholder that we overwrite
                self.forest_container_variance = ForestContainer(0, 0, False, False)
                self.forest_container_variance.forest_container_cpp.LoadFromJson(
                    bart_json.json_cpp, "forest_1"
                )
        else:
            # TODO: don't just make this a placeholder that we overwrite
            self.forest_container_variance = ForestContainer(0, 0, False, False)
            self.forest_container_variance.forest_container_cpp.LoadFromJson(
                bart_json.json_cpp, "forest_0"
            )

        # Unpack random effects
        if self.has_rfx:
            self.rfx_container = RandomEffectsContainer()
            self.rfx_container.load_from_json(bart_json, 0)

        # Unpack global parameters
        self.y_std = bart_json.get_scalar("outcome_scale")
        self.y_bar = bart_json.get_scalar("outcome_mean")
        self.standardize = bart_json.get_boolean("standardize")
        self.sigma2_init = bart_json.get_scalar("sigma2_init")
        self.sample_sigma2_global = bart_json.get_boolean("sample_sigma2_global")
        self.sample_sigma2_leaf = bart_json.get_boolean("sample_sigma2_leaf")
        self.num_gfr = bart_json.get_integer("num_gfr")
        self.num_burnin = bart_json.get_integer("num_burnin")
        self.num_mcmc = bart_json.get_integer("num_mcmc")
        self.num_samples = bart_json.get_integer("num_samples")
        self.num_basis = bart_json.get_integer("num_basis")
        self.has_basis = bart_json.get_boolean("requires_basis")
        self.probit_outcome_model = bart_json.get_boolean("probit_outcome_model")

        # Unpack parameter samples
        if self.sample_sigma2_global:
            self.global_var_samples = bart_json.get_numeric_vector(
                "sigma2_global_samples", "parameters"
            )
        if self.sample_sigma2_leaf:
            self.leaf_scale_samples = bart_json.get_numeric_vector(
                "sigma2_leaf_samples", "parameters"
            )

        # Unpack covariate preprocessor
        covariate_preprocessor_string = bart_json.get_string("covariate_preprocessor")
        self._covariate_preprocessor = CovariatePreprocessor()
        self._covariate_preprocessor.from_json(covariate_preprocessor_string)

        # Mark the deserialized model as "sampled"
        self.sampled = True

    def from_json_string_list(self, json_string_list: list[str]) -> None:
        """
        Convert a list of (in-memory) JSON strings that represent BART models to a single combined BART model object
        which can be used for prediction, etc...

        Parameters
        -------
        json_string_list : list of str
            List of JSON strings which can be parsed to objects of type `JSONSerializer` containing Json representation of a BART model
        """
        # Convert strings to JSONSerializer
        json_object_list = []
        for i in range(len(json_string_list)):
            json_string = json_string_list[i]
            json_object_list.append(JSONSerializer())
            json_object_list[i].load_from_json_string(json_string)

        # For scalar / preprocessing details which aren't sample-dependent, defer to the first json
        json_object_default = json_object_list[0]

        # Unpack forests
        self.include_mean_forest = json_object_default.get_boolean(
            "include_mean_forest"
        )
        self.include_variance_forest = json_object_default.get_boolean(
            "include_variance_forest"
        )
        if self.include_mean_forest:
            # TODO: don't just make this a placeholder that we overwrite
            self.forest_container_mean = ForestContainer(0, 0, False, False)
            for i in range(len(json_object_list)):
                if i == 0:
                    self.forest_container_mean.forest_container_cpp.LoadFromJson(
                        json_object_list[i].json_cpp, "forest_0"
                    )
                else:
                    self.forest_container_mean.forest_container_cpp.AppendFromJson(
                        json_object_list[i].json_cpp, "forest_0"
                    )
            if self.include_variance_forest:
                # TODO: don't just make this a placeholder that we overwrite
                self.forest_container_variance = ForestContainer(0, 0, False, False)
                for i in range(len(json_object_list)):
                    if i == 0:
                        self.forest_container_variance.forest_container_cpp.LoadFromJson(
                            json_object_list[i].json_cpp, "forest_1"
                        )
                    else:
                        self.forest_container_variance.forest_container_cpp.AppendFromJson(
                            json_object_list[i].json_cpp, "forest_1"
                        )
        else:
            # TODO: don't just make this a placeholder that we overwrite
            self.forest_container_variance = ForestContainer(0, 0, False, False)
            for i in range(len(json_object_list)):
                if i == 0:
                    self.forest_container_variance.forest_container_cpp.LoadFromJson(
                        json_object_list[i].json_cpp, "forest_1"
                    )
                else:
                    self.forest_container_variance.forest_container_cpp.AppendFromJson(
                        json_object_list[i].json_cpp, "forest_1"
                    )

        # Unpack random effects
        self.has_rfx = json_object_default.get_boolean("has_rfx")
        if self.has_rfx:
            self.rfx_container = RandomEffectsContainer()
            for i in range(len(json_object_list)):
                if i == 0:
                    self.rfx_container.load_from_json(json_object_list[i], 0)
                else:
                    self.rfx_container.append_from_json(json_object_list[i], 0)

        # Unpack global parameters
        self.y_std = json_object_default.get_scalar("outcome_scale")
        self.y_bar = json_object_default.get_scalar("outcome_mean")
        self.standardize = json_object_default.get_boolean("standardize")
        self.sigma2_init = json_object_default.get_scalar("sigma2_init")
        self.sample_sigma2_global = json_object_default.get_boolean(
            "sample_sigma2_global"
        )
        self.sample_sigma2_leaf = json_object_default.get_boolean("sample_sigma2_leaf")
        self.num_gfr = json_object_default.get_integer("num_gfr")
        self.num_burnin = json_object_default.get_integer("num_burnin")
        self.num_mcmc = json_object_default.get_integer("num_mcmc")
        self.num_samples = json_object_default.get_integer("num_samples")
        self.num_basis = json_object_default.get_integer("num_basis")
        self.has_basis = json_object_default.get_boolean("requires_basis")
        self.probit_outcome_model = json_object_default.get_boolean(
            "probit_outcome_model"
        )

        # Unpack parameter samples
        if self.sample_sigma2_global:
            for i in range(len(json_object_list)):
                if i == 0:
                    self.global_var_samples = json_object_list[i].get_numeric_vector(
                        "sigma2_global_samples", "parameters"
                    )
                else:
                    global_var_samples = json_object_list[i].get_numeric_vector(
                        "sigma2_global_samples", "parameters"
                    )
                    self.global_var_samples = np.concatenate(
                        (self.global_var_samples, global_var_samples)
                    )

        if self.sample_sigma2_leaf:
            for i in range(len(json_object_list)):
                if i == 0:
                    self.leaf_scale_samples = json_object_list[i].get_numeric_vector(
                        "sigma2_leaf_samples", "parameters"
                    )
                else:
                    leaf_scale_samples = json_object_list[i].get_numeric_vector(
                        "sigma2_leaf_samples", "parameters"
                    )
                    self.leaf_scale_samples = np.concatenate(
                        (self.leaf_scale_samples, leaf_scale_samples)
                    )

        # Unpack covariate preprocessor
        covariate_preprocessor_string = json_object_default.get_string(
            "covariate_preprocessor"
        )
        self._covariate_preprocessor = CovariatePreprocessor()
        self._covariate_preprocessor.from_json(covariate_preprocessor_string)

        # Mark the deserialized model as "sampled"
        self.sampled = True

    def is_sampled(self) -> bool:
        """Whether or not a BART model has been sampled.

        Returns
        -------
        bool
            `True` if a BART model has been sampled, `False` otherwise
        """
        return self.sampled
