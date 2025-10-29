import warnings
from math import log
from numbers import Integral
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import norm

from stochtree_cpp import OrdinalSamplerCpp
from .config import ForestModelConfig, GlobalModelConfig
from .data import Dataset, Residual
from .forest import Forest, ForestContainer
from .preprocessing import CovariatePreprocessor, _preprocess_params
from .sampler import RNG, ForestSampler, GlobalVarianceModel, LeafVarianceModel
from .serialization import JSONSerializer
from .utils import (
    NotSampledError,
    _expand_dims_1d,
    _expand_dims_2d,
    _expand_dims_2d_diag,
)


class CloglogOrdinalBARTModel:
    r"""
    Class that handles sampling, storage, and serialization of BART models with a cloglog link for ordinal outcomes.
    This is an implementation of the model of Alam and Linero (2025), in which y is an ordinal outcome with K categories, ordered from 0 to K-1.
    """

    def __init__(self) -> None:
        # Internal flag for whether the sample() method has been run
        self.sampled = False

    def sample(
        self,
        X_train: Union[np.array, pd.DataFrame],
        y_train: np.array,
        X_test: Union[np.array, pd.DataFrame] = None,
        n_trees: int = 50,
        num_gfr: int = 0,
        num_burnin: int = 1000,
        num_mcmc: int = 500,
        n_thin: int = 1,
        alpha_gamma: float = 2.0,
        beta_gamma: float = 2.0,
        variable_weights: np.array = None,
        feature_types: np.array = None,
        seed: int = None,
        num_threads=1,
    ) -> None:
        """Runs a Cloglog BART sampler on provided training set. Predictions will be cached for the training set and (if provided) the test set.

        Parameters
        ----------
        X_train : np.array
            Training set covariates on which trees may be partitioned.
        y_train : np.array
            Training set outcome (must be integer-valued from 0 to K-1, where K is the number of outcome categories).
        X_test : np.array, optional
            Optional test set covariates.
        n_trees : int, optional
            Number of trees in the BART ensemble. Defaults to `50`.
        num_gfr : int, optional
            Number of "warm-start" iterations run using the grow-from-root algorithm (He and Hahn, 2021). Defaults to `0`.
        num_burnin : int, optional
            Number of "burn-in" iterations of the MCMC sampler. Defaults to `1000`.
        num_mcmc : int, optional
            Number of "retained" iterations of the MCMC sampler. Defaults to `500`.
        n_thin : int, optional
            Thinning interval for MCMC samples. Defaults to `1` (no thinning).
        alpha_gamma : float, optional
            Shape parameter for the log-gamma prior on cutpoints. Defaults to `2.0`.
        beta_gamma : float, optional
            Rate parameter for the log-gamma prior on cutpoints. Defaults to `2.0`.
        variable_weights : np.array, optional
            Variable weights for covariate selection probabilities. If `None`, uniform weights are used.
        seed : int, optional
            Random seed for reproducibility. If `None`, a random seed is used.
        num_threads : int, optional
            Number of threads to use for parallel processing. Defaults to `1`.

        Returns
        -------
        self : BARTModel
            Sampled BART Model.
        """
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
        if y_train.dtype not in [np.int32, np.int64]:
            raise ValueError("y_train must be an integer-valued numpy array")
        if np.any(y_train < 0):
            raise ValueError("y_train must be non-negative integer-valued")

        # Convert everything to standard shape (2-dimensional)
        if isinstance(X_train, np.ndarray):
            if X_train.ndim == 1:
                X_train = np.expand_dims(X_train, 1)
        if y_train.ndim == 1:
            y_train = np.expand_dims(y_train, 1)
        if X_test is not None:
            if isinstance(X_test, np.ndarray):
                if X_test.ndim == 1:
                    X_test = np.expand_dims(X_test, 1)

        # Data checks
        if X_test is not None:
            if X_test.shape[1] != X_train.shape[1]:
                raise ValueError(
                    "X_train and X_test must have the same number of columns"
                )
        if y_train.shape[0] != X_train.shape[0]:
            raise ValueError("X_train and y_train must have the same number of rows")

        # Variable weight preprocessing (and initialization if necessary)
        p = X_train.shape[1]
        if variable_weights is None:
            if X_train.ndim > 1:
                variable_weights = np.repeat(1.0 / p, p)
            else:
                variable_weights = np.repeat(1.0, 1)
        if np.any(variable_weights < 0):
            raise ValueError("variable_weights cannot have any negative weights")

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

        # Update variable weights if the covariates have been resized (by e.g. one-hot encoding)
        if X_train_processed.shape[1] != X_train.shape[1]:
            variable_counts = [
                original_var_indices.count(i) for i in original_var_indices
            ]
            variable_weights_adj = np.array([1 / i for i in variable_counts])
            variable_weights = (
                variable_weights[original_var_indices] * variable_weights_adj
            )

        # Determine whether a test set is provided
        self.has_test = X_test is not None

        # Unpack data dimensions
        self.n_train = y_train.shape[0]
        self.n_test = X_test_processed.shape[0] if self.has_test else 0
        self.num_covariates = X_train_processed.shape[1]

        # Determine number of outcome categories
        self.n_levels = np.max(np.unique(np.squeeze(y_train))) + 1

        # Check that there are at least 2 outcome categories
        if self.n_levels < 2:
            raise ValueError("y_train must have at least 2 outcome categories")

        # BART parameters
        alpha_bart = 0.95
        beta_bart = 2
        min_samples_in_leaf = 5
        max_depth = 10
        scale_leaf = 2 / np.sqrt(n_trees)
        cutpoint_grid_size = 100

        # Fixed for identifiability (can be pass as argument later if desired)
        gamma_0 = 0.0  # First gamma cutpoint fixed at gamma_0 = 0

        # Indices of MCMC samples to keep after GFR, burn-in, and thinning
        keep_idx = np.arange(
            num_gfr + num_burnin, num_gfr + num_burnin + num_mcmc, n_thin
        )
        n_keep = len(keep_idx)

        # Container of parameter samples / model draws
        self.num_gfr = num_gfr
        self.num_burnin = num_burnin
        self.num_mcmc = num_mcmc
        self.forest_pred_train = np.empty((self.n_train, n_keep), dtype=np.float64)
        if self.has_test:
            self.forest_pred_test = np.empty((self.n_test, n_keep), dtype=np.float64)
        self.gamma_samples = np.empty((self.n_levels - 1, n_keep), dtype=np.float64)
        self.latent_samples = np.empty((self.n_train, n_keep), dtype=np.float64)

        # Initialize samplers
        ordinal_sampler_cpp = OrdinalSamplerCpp()
        if seed is None:
            cpp_rng = RNG(-1)
            self.rng = np.random.default_rng()
        else:
            cpp_rng = RNG(seed)
            self.rng = np.random.default_rng(seed)

        # Data structures
        forest_dataset_train = Dataset()
        forest_dataset_train.add_covariates(X_train_processed)
        if self.has_test:
            forest_dataset_test = Dataset()
            forest_dataset_test.add_covariates(X_test_processed)
        outcome_train = Residual(y_train)
        active_forest = Forest(n_trees, 1, True, False)
        active_forest.set_root_leaves(0.0)
        self.forest_samples = ForestContainer(n_trees, 1, True, False)
        global_model_config = GlobalModelConfig(global_error_variance=1.0)
        forest_model_config = ForestModelConfig(
            num_trees=n_trees,
            num_features=self.num_covariates,
            num_observations=self.n_train,
            feature_types=feature_types,
            variable_weights=variable_weights,
            leaf_dimension=1,
            alpha=alpha_bart,
            beta=beta_bart,
            min_samples_leaf=min_samples_in_leaf,
            max_depth=max_depth,
            leaf_model_type=4,
            cutpoint_grid_size=cutpoint_grid_size,
            leaf_model_scale=scale_leaf,
        )
        forest_sampler = ForestSampler(
            forest_dataset_train, global_model_config, forest_model_config
        )

        # Latent variable (Z in Alam et al (2025) notation)
        forest_dataset_train.add_auxiliary_dimension(self.n_train)
        # Forest predictions (eta in Alam et al (2025) notation)
        forest_dataset_train.add_auxiliary_dimension(self.n_train)
        # Log-scale non-cumulative cutpoint (gamma in Alam et al (2025) notation)
        forest_dataset_train.add_auxiliary_dimension(self.n_levels - 1)
        # Exponentiated cumulative cutpoints (exp(c_k) in Alam et al (2025) notation)
        # This auxiliary series is designed so that the element stored at position `i`
        # corresponds to the sum of all exponentiated gamma_j values for j < i.
        # It has n_levels elements instead of n_levels - 1 because even the largest
        # categorical index has a valid value of sum_{j < i} exp(gamma_j)
        forest_dataset_train.add_auxiliary_dimension(self.n_levels)

        # Initialize gamma parameters to zero (3rd auxiliary data series, mapped to `dim_idx = 2` with 0-indexing)
        initial_gamma = np.zeros((self.n_levels - 1,), dtype=np.float64)
        for i in range(self.n_levels - 1):
            forest_dataset_train.set_auxiliary_data_value(2, i - 1, initial_gamma[i])

        # Convert the log-scale parameters into cumulative exponentiated parameters.
        # This is done under the hood in a C++ function for efficiency.
        ordinal_sampler_cpp.UpdateCumulativeExpSums(forest_dataset_train.dataset_cpp)

        # Initialize forest predictions to zero (slot 1)
        for i in range(self.n_train):
            forest_dataset_train.set_auxiliary_data_value(1, i, 0.0)

        # Initialize latent variables to zero (slot 0)
        for i in range(self.n_train):
            forest_dataset_train.set_auxiliary_data_value(0, i, 0.0)

        # Run the algorithm
        sample_counter = -1
        for i in range(num_gfr + num_burnin + num_mcmc):
            keep_sample = i in keep_idx
            if keep_sample:
                sample_counter += 1

            # 1. Sample forest using MCMC
            if i > self.num_gfr - 1:
                forest_sampler.sample_one_iteration(
                    self.forest_samples,
                    active_forest,
                    forest_dataset_train,
                    outcome_train,
                    cpp_rng,
                    global_model_config,
                    forest_model_config,
                    keep_sample,
                    True,
                    num_threads,
                )
            else:
                forest_sampler.sample_one_iteration(
                    self.forest_samples,
                    active_forest,
                    forest_dataset_train,
                    outcome_train,
                    cpp_rng,
                    global_model_config,
                    forest_model_config,
                    keep_sample,
                    False,
                    num_threads,
                )

            # Set auxiliary data slot 1 to current forest predictions = lambda_hat = sum of all the tree predictions
            # This is needed for updating gamma parameters, latent z_i's
            forest_pred_current = active_forest.predict(forest_dataset_train)
            for i in range(self.n_train):
                forest_dataset_train.set_auxiliary_data_value(
                    1, i, forest_pred_current[i]
                )

            # 2. Sample latent z_i's using truncated exponential
            ordinal_sampler_cpp.UpdateLatentVariables(
                forest_dataset_train.dataset_cpp,
                outcome_train.residual_cpp,
                cpp_rng.rng_cpp,
            )

            # 3. Sample gamma cutpoints
            ordinal_sampler_cpp.UpdateGammaParams(
                forest_dataset_train.dataset_cpp,
                outcome_train.residual_cpp,
                alpha_gamma,
                beta_gamma,
                gamma_0,
                cpp_rng.rng_cpp,
            )

            # 4. Update cumulative sum of exp(gamma) values
            ordinal_sampler_cpp.UpdateCumulativeExpSums(
                forest_dataset_train.dataset_cpp
            )

            if keep_sample:
                self.forest_pred_train[:, sample_counter] = active_forest.predict(
                    forest_dataset_train
                )
                if self.has_test:
                    self.forest_pred_test[:, sample_counter] = active_forest.predict(
                        forest_dataset_test
                    )
                gamma_current = forest_dataset_train.get_auxiliary_data_array(2)
                self.gamma_samples[:, sample_counter] = gamma_current
                latent_current = forest_dataset_train.get_auxiliary_data_array(0)
                self.latent_samples[:, sample_counter] = latent_current

        # Mark the model as sampled
        self.sampled = True

    def predict(
        self,
        X: Union[np.array, pd.DataFrame],
    ) -> np.array:
        """Return predictions from the cloglog forest.

        Parameters
        ----------
        covariates : np.array
            Test set covariates.

        Returns
        -------
        lambda_x : np.array, optional
            Cloglog forest predictions
        """
        if not self.is_sampled():
            msg = (
                "This CloglogOrdinalBARTModel instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this model."
            )
            raise NotSampledError(msg)

        # Data checks
        if not isinstance(X, pd.DataFrame) and not isinstance(X, np.ndarray):
            raise ValueError("X must be a pandas dataframe or numpy array")

        # Convert everything to standard shape (2-dimensional)
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = np.expand_dims(X, 1)

        # Covariate preprocessing
        if not self._covariate_preprocessor._check_is_fitted():
            if not isinstance(X, np.ndarray):
                raise ValueError(
                    "Prediction cannot proceed on a pandas dataframe, since the BART model was not fit with a covariate preprocessor. Please refit your model by passing covariate data as a Pandas dataframe."
                )
            else:
                warnings.warn(
                    "This BART model has not run any covariate preprocessing routines. We will attempt to predict on the raw covariate values, but this will trigger an error with non-numeric columns. Please refit your model by passing non-numeric covariate data a a Pandas dataframe.",
                    RuntimeWarning,
                )
                if not np.issubdtype(X.dtype, np.floating) and not np.issubdtype(
                    X.dtype, np.integer
                ):
                    raise ValueError(
                        "Prediction cannot proceed on a non-numeric numpy array, since the BART model was not fit with a covariate preprocessor. Please refit your model by passing non-numeric covariate data as a Pandas dataframe."
                    )
                X_processed = X
        else:
            X_processed = self._covariate_preprocessor.transform(X)

        # Dataset construction
        pred_dataset = Dataset()
        pred_dataset.add_covariates(X_processed)

        # Forest predictions
        forest_pred = self.forest_samples.forest_container_cpp.Predict(
            pred_dataset.dataset_cpp
        )

        return forest_pred

    # def to_json(self) -> str:
    #     """
    #     Converts a sampled BART model to JSON string representation (which can then be saved to a file or
    #     processed using the `json` library)

    #     Returns
    #     -------
    #     str
    #         JSON string representing model metadata (hyperparameters), sampled parameters, and sampled forests
    #     """
    #     if not self.is_sampled:
    #         msg = (
    #             "This BARTModel instance has not yet been sampled. "
    #             "Call 'fit' with appropriate arguments before using this model."
    #         )
    #         raise NotSampledError(msg)

    #     # Initialize JSONSerializer object
    #     bart_json = JSONSerializer()

    #     # Add the forests
    #     if self.include_mean_forest:
    #         bart_json.add_forest(self.forest_container_mean)
    #     if self.include_variance_forest:
    #         bart_json.add_forest(self.forest_container_variance)

    #     # Add the rfx
    #     if self.has_rfx:
    #         bart_json.add_random_effects(self.rfx_container)

    #     # Add global parameters
    #     bart_json.add_scalar("outcome_scale", self.y_std)
    #     bart_json.add_scalar("outcome_mean", self.y_bar)
    #     bart_json.add_boolean("standardize", self.standardize)
    #     bart_json.add_scalar("sigma2_init", self.sigma2_init)
    #     bart_json.add_boolean("sample_sigma2_global", self.sample_sigma2_global)
    #     bart_json.add_boolean("sample_sigma2_leaf", self.sample_sigma2_leaf)
    #     bart_json.add_boolean("include_mean_forest", self.include_mean_forest)
    #     bart_json.add_boolean("include_variance_forest", self.include_variance_forest)
    #     bart_json.add_boolean("has_rfx", self.has_rfx)
    #     bart_json.add_integer("num_gfr", self.num_gfr)
    #     bart_json.add_integer("num_burnin", self.num_burnin)
    #     bart_json.add_integer("num_mcmc", self.num_mcmc)
    #     bart_json.add_integer("num_samples", self.num_samples)
    #     bart_json.add_integer("num_basis", self.num_basis)
    #     bart_json.add_boolean("requires_basis", self.has_basis)
    #     bart_json.add_boolean("probit_outcome_model", self.probit_outcome_model)

    #     # Add parameter samples
    #     if self.sample_sigma2_global:
    #         bart_json.add_numeric_vector(
    #             "sigma2_global_samples", self.global_var_samples, "parameters"
    #         )
    #     if self.sample_sigma2_leaf:
    #         bart_json.add_numeric_vector(
    #             "sigma2_leaf_samples", self.leaf_scale_samples, "parameters"
    #         )

    #     # Add covariate preprocessor
    #     covariate_preprocessor_string = self._covariate_preprocessor.to_json()
    #     bart_json.add_string("covariate_preprocessor", covariate_preprocessor_string)

    #     return bart_json.return_json_string()

    # def from_json(self, json_string: str) -> None:
    #     """
    #     Converts a JSON string to an in-memory BART model.

    #     Parameters
    #     ----------
    #     json_string : str
    #         JSON string representing model metadata (hyperparameters), sampled parameters, and sampled forests
    #     """
    #     # Parse string to a JSON object in C++
    #     bart_json = JSONSerializer()
    #     bart_json.load_from_json_string(json_string)

    #     # Unpack forests
    #     self.include_mean_forest = bart_json.get_boolean("include_mean_forest")
    #     self.include_variance_forest = bart_json.get_boolean("include_variance_forest")
    #     self.has_rfx = bart_json.get_boolean("has_rfx")
    #     if self.include_mean_forest:
    #         # TODO: don't just make this a placeholder that we overwrite
    #         self.forest_container_mean = ForestContainer(0, 0, False, False)
    #         self.forest_container_mean.forest_container_cpp.LoadFromJson(
    #             bart_json.json_cpp, "forest_0"
    #         )
    #         if self.include_variance_forest:
    #             # TODO: don't just make this a placeholder that we overwrite
    #             self.forest_container_variance = ForestContainer(0, 0, False, False)
    #             self.forest_container_variance.forest_container_cpp.LoadFromJson(
    #                 bart_json.json_cpp, "forest_1"
    #             )
    #     else:
    #         # TODO: don't just make this a placeholder that we overwrite
    #         self.forest_container_variance = ForestContainer(0, 0, False, False)
    #         self.forest_container_variance.forest_container_cpp.LoadFromJson(
    #             bart_json.json_cpp, "forest_0"
    #         )

    #     # Unpack random effects
    #     if self.has_rfx:
    #         self.rfx_container = RandomEffectsContainer()
    #         self.rfx_container.load_from_json(bart_json, 0)

    #     # Unpack global parameters
    #     self.y_std = bart_json.get_scalar("outcome_scale")
    #     self.y_bar = bart_json.get_scalar("outcome_mean")
    #     self.standardize = bart_json.get_boolean("standardize")
    #     self.sigma2_init = bart_json.get_scalar("sigma2_init")
    #     self.sample_sigma2_global = bart_json.get_boolean("sample_sigma2_global")
    #     self.sample_sigma2_leaf = bart_json.get_boolean("sample_sigma2_leaf")
    #     self.num_gfr = bart_json.get_integer("num_gfr")
    #     self.num_burnin = bart_json.get_integer("num_burnin")
    #     self.num_mcmc = bart_json.get_integer("num_mcmc")
    #     self.num_samples = bart_json.get_integer("num_samples")
    #     self.num_basis = bart_json.get_integer("num_basis")
    #     self.has_basis = bart_json.get_boolean("requires_basis")
    #     self.probit_outcome_model = bart_json.get_boolean("probit_outcome_model")

    #     # Unpack parameter samples
    #     if self.sample_sigma2_global:
    #         self.global_var_samples = bart_json.get_numeric_vector(
    #             "sigma2_global_samples", "parameters"
    #         )
    #     if self.sample_sigma2_leaf:
    #         self.leaf_scale_samples = bart_json.get_numeric_vector(
    #             "sigma2_leaf_samples", "parameters"
    #         )

    #     # Unpack covariate preprocessor
    #     covariate_preprocessor_string = bart_json.get_string("covariate_preprocessor")
    #     self._covariate_preprocessor = CovariatePreprocessor()
    #     self._covariate_preprocessor.from_json(covariate_preprocessor_string)

    #     # Mark the deserialized model as "sampled"
    #     self.sampled = True

    # def from_json_string_list(self, json_string_list: list[str]) -> None:
    #     """
    #     Convert a list of (in-memory) JSON strings that represent BART models to a single combined BART model object
    #     which can be used for prediction, etc...

    #     Parameters
    #     -------
    #     json_string_list : list of str
    #         List of JSON strings which can be parsed to objects of type `JSONSerializer` containing Json representation of a BART model
    #     """
    #     # Convert strings to JSONSerializer
    #     json_object_list = []
    #     for i in range(len(json_string_list)):
    #         json_string = json_string_list[i]
    #         json_object_list.append(JSONSerializer())
    #         json_object_list[i].load_from_json_string(json_string)

    #     # For scalar / preprocessing details which aren't sample-dependent, defer to the first json
    #     json_object_default = json_object_list[0]

    #     # Unpack forests
    #     self.include_mean_forest = json_object_default.get_boolean(
    #         "include_mean_forest"
    #     )
    #     self.include_variance_forest = json_object_default.get_boolean(
    #         "include_variance_forest"
    #     )
    #     if self.include_mean_forest:
    #         # TODO: don't just make this a placeholder that we overwrite
    #         self.forest_container_mean = ForestContainer(0, 0, False, False)
    #         for i in range(len(json_object_list)):
    #             if i == 0:
    #                 self.forest_container_mean.forest_container_cpp.LoadFromJson(
    #                     json_object_list[i].json_cpp, "forest_0"
    #                 )
    #             else:
    #                 self.forest_container_mean.forest_container_cpp.AppendFromJson(
    #                     json_object_list[i].json_cpp, "forest_0"
    #                 )
    #         if self.include_variance_forest:
    #             # TODO: don't just make this a placeholder that we overwrite
    #             self.forest_container_variance = ForestContainer(0, 0, False, False)
    #             for i in range(len(json_object_list)):
    #                 if i == 0:
    #                     self.forest_container_variance.forest_container_cpp.LoadFromJson(
    #                         json_object_list[i].json_cpp, "forest_1"
    #                     )
    #                 else:
    #                     self.forest_container_variance.forest_container_cpp.AppendFromJson(
    #                         json_object_list[i].json_cpp, "forest_1"
    #                     )
    #     else:
    #         # TODO: don't just make this a placeholder that we overwrite
    #         self.forest_container_variance = ForestContainer(0, 0, False, False)
    #         for i in range(len(json_object_list)):
    #             if i == 0:
    #                 self.forest_container_variance.forest_container_cpp.LoadFromJson(
    #                     json_object_list[i].json_cpp, "forest_0"
    #                 )
    #             else:
    #                 self.forest_container_variance.forest_container_cpp.AppendFromJson(
    #                     json_object_list[i].json_cpp, "forest_0"
    #                 )

    #     # Unpack random effects
    #     self.has_rfx = json_object_default.get_boolean("has_rfx")
    #     if self.has_rfx:
    #         self.rfx_container = RandomEffectsContainer()
    #         for i in range(len(json_object_list)):
    #             if i == 0:
    #                 self.rfx_container.load_from_json(json_object_list[i], 0)
    #             else:
    #                 self.rfx_container.append_from_json(json_object_list[i], 0)

    #     # Unpack global parameters
    #     self.y_std = json_object_default.get_scalar("outcome_scale")
    #     self.y_bar = json_object_default.get_scalar("outcome_mean")
    #     self.standardize = json_object_default.get_boolean("standardize")
    #     self.sigma2_init = json_object_default.get_scalar("sigma2_init")
    #     self.sample_sigma2_global = json_object_default.get_boolean(
    #         "sample_sigma2_global"
    #     )
    #     self.sample_sigma2_leaf = json_object_default.get_boolean("sample_sigma2_leaf")
    #     self.num_gfr = json_object_default.get_integer("num_gfr")
    #     self.num_burnin = json_object_default.get_integer("num_burnin")
    #     self.num_mcmc = json_object_default.get_integer("num_mcmc")
    #     self.num_basis = json_object_default.get_integer("num_basis")
    #     self.has_basis = json_object_default.get_boolean("requires_basis")
    #     self.probit_outcome_model = json_object_default.get_boolean(
    #         "probit_outcome_model"
    #     )

    #     # Unpack number of samples
    #     for i in range(len(json_object_list)):
    #         if i == 0:
    #             self.num_samples = json_object_list[i].get_integer("num_samples")
    #         else:
    #             self.num_samples += json_object_list[i].get_integer("num_samples")

    #     # Unpack parameter samples
    #     if self.sample_sigma2_global:
    #         for i in range(len(json_object_list)):
    #             if i == 0:
    #                 self.global_var_samples = json_object_list[i].get_numeric_vector(
    #                     "sigma2_global_samples", "parameters"
    #                 )
    #             else:
    #                 global_var_samples = json_object_list[i].get_numeric_vector(
    #                     "sigma2_global_samples", "parameters"
    #                 )
    #                 self.global_var_samples = np.concatenate(
    #                     (self.global_var_samples, global_var_samples)
    #                 )

    #     if self.sample_sigma2_leaf:
    #         for i in range(len(json_object_list)):
    #             if i == 0:
    #                 self.leaf_scale_samples = json_object_list[i].get_numeric_vector(
    #                     "sigma2_leaf_samples", "parameters"
    #                 )
    #             else:
    #                 leaf_scale_samples = json_object_list[i].get_numeric_vector(
    #                     "sigma2_leaf_samples", "parameters"
    #                 )
    #                 self.leaf_scale_samples = np.concatenate(
    #                     (self.leaf_scale_samples, leaf_scale_samples)
    #                 )

    #     # Unpack covariate preprocessor
    #     covariate_preprocessor_string = json_object_default.get_string(
    #         "covariate_preprocessor"
    #     )
    #     self._covariate_preprocessor = CovariatePreprocessor()
    #     self._covariate_preprocessor.from_json(covariate_preprocessor_string)

    #     # Mark the deserialized model as "sampled"
    #     self.sampled = True

    def is_sampled(self) -> bool:
        """Whether or not a BART model has been sampled.

        Returns
        -------
        bool
            `True` if a BART model has been sampled, `False` otherwise
        """
        return self.sampled
