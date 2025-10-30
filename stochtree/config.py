import warnings
from typing import Union

import numpy as np

from .utils import (
    _check_is_int,
    _check_is_numeric,
    _check_matrix_square,
    _standardize_array_to_np,
)


class ForestModelConfig:
    """
    Object used to get / set parameters and other model configuration options for a forest model in the "low-level" stochtree interface.

    The "low-level" stochtree interface enables a high degreee of sampler customization, in which users employ R wrappers around
    C++ objects like `ForestDataset`, `Outcome`, `CppRng`, and `ForestModel` to run the Gibbs sampler of a BART model with custom modifications.
    `ForestModelConfig` allows users to specify / query the parameters of a forest model they wish to run.

    Parameters
    ----------
    num_trees : int
        Number of trees in the forest being sampled
    num_features : int
        Number of features in training dataset
    num_observations : int
        Number of observations in training dataset
    feature_types : np.array or list, optional
        Vector of integer-coded feature types (where 0 = numeric, 1 = ordered categorical, 2 = unordered categorical)
    sweep_update_indices : np.array or list, optional
        Vector of (0-indexed) indices of trees to update in a sweep
    variable_weights : np.array or list, optional
        Vector specifying sampling probability for all p covariates in ForestDataset
    leaf_dimension : int, optional
        Dimension of the leaf model (default: `1`)
    alpha : int, optional
        Root node split probability in tree prior (default: `0.95`)
    beta : int, optional
        Depth prior penalty in tree prior (default: `2.0`)
    min_samples_leaf : int, optional
        Minimum number of samples in a tree leaf (default: `5`)
    max_depth : int, optional
        Maximum depth of any tree in the ensemble in the model. Setting to `-1` does not enforce any depth limits on trees. Default: `-1`.
    leaf_model_type : int, optional
        Integer specifying the leaf model type (0 = constant leaf, 1 = univariate leaf regression, 2 = multivariate leaf regression). Default: `0`.
    leaf_model_scale : float or np.ndarray, optional
        Scale parameter used in Gaussian leaf models (can either be a scalar or a q x q matrix, where q is the dimensionality of the basis and is only >1 when `leaf_model_int = 2`). Calibrated internally as `1/num_trees`, propagated along diagonal if needed for multivariate leaf models.
    variance_forest_shape : int, optional
        Shape parameter for IG leaf models (applicable when `leaf_model_type = 3`). Default: `1`.
    variance_forest_scale : int, optional
        Scale parameter for IG leaf models (applicable when `leaf_model_type = 3`). Default: `1`.
    cutpoint_grid_size : int, optional
        Number of unique cutpoints to consider (default: `100`).
    num_features_subsample : int, optional
        Number of features to subsample for the GFR algorithm (default: `None`).
    """

    def __init__(
        self,
        num_trees=None,
        num_features=None,
        num_observations=None,
        feature_types=None,
        sweep_update_indices=None,
        variable_weights=None,
        leaf_dimension=1,
        alpha=0.95,
        beta=2.0,
        min_samples_leaf=5,
        max_depth=-1,
        leaf_model_type=0,
        leaf_model_scale=None,
        variance_forest_shape=1.0,
        variance_forest_scale=1.0,
        cutpoint_grid_size=100,
        num_features_subsample=None,
    ) -> None:
        # Preprocess inputs and run some error checks
        if feature_types is None:
            if num_features is None:
                raise ValueError(
                    "Neither of `num_features` nor `feature_types` (a vector from which `num_features` can be inferred) was provided.",
                    "Please provide at least one of these inputs when creating a `ForestModelConfig` object.",
                )
            warnings.warn("`feature_types` not provided, will be assumed to be numeric")
            self.feature_types = np.repeat(0, num_features)
        else:
            self.feature_types = _standardize_array_to_np(feature_types)
            if num_features is None:
                num_features = len(self.feature_types)
        if variable_weights is None:
            warnings.warn(
                "`variable_weights` not provided, will be assumed to be equal-weighted"
            )
            self.variable_weights = np.repeat(1.0 / num_features, num_features)
        else:
            self.variable_weights = _standardize_array_to_np(variable_weights)
        if num_trees is None:
            raise ValueError("`num_trees` must be provided")
        if num_observations is None:
            raise ValueError("`num_observations` must be provided")
        if num_features != len(self.feature_types):
            raise ValueError("`feature_types` must have `num_features` total elements")
        if num_features != len(self.variable_weights):
            raise ValueError(
                "`variable_weights` must have `num_features` total elements"
            )
        if leaf_model_type is None:
            leaf_model_type = 0
        if not _check_is_int(leaf_model_type):
            raise ValueError("`leaf_model_type` must be an integer between 0 and 3")
        elif leaf_model_type < 0 or leaf_model_type > 3:
            raise ValueError("`leaf_model_type` must be an integer between 0 and 3")
        if not _check_is_int(leaf_dimension):
            raise ValueError("`leaf_dimension` must be an integer greater than 0")
        elif leaf_dimension <= 0:
            raise ValueError("`leaf_dimension` must be an integer greater than 0")
        if leaf_model_scale is None:
            diag_value = 1.0 / num_trees
            leaf_model_scale_array = np.zeros(
                (leaf_dimension, leaf_dimension), dtype=float
            )
            np.fill_diagonal(leaf_model_scale_array, diag_value)
        else:
            if isinstance(leaf_model_scale, np.ndarray):
                if not _check_matrix_square(leaf_model_scale):
                    raise ValueError(
                        "`leaf_model_scale` must be a square matrix if provided as a numpy array"
                    )
                leaf_model_scale_array = leaf_model_scale
            elif isinstance(leaf_model_scale, (int, float)):
                if leaf_model_scale <= 0:
                    raise ValueError(
                        "`leaf_model_scale` must be positive, if provided as scalar"
                    )
                leaf_model_scale_array = np.zeros(
                    (leaf_dimension, leaf_dimension), dtype=float
                )
                np.fill_diagonal(leaf_model_scale_array, leaf_model_scale)
            else:
                raise ValueError(
                    "`leaf_model_scale` must be a scalar value or a 2d numpy array with matching dimensions"
                )
        if sweep_update_indices is not None:
            sweep_update_indices = _standardize_array_to_np(sweep_update_indices)
            if np.min(sweep_update_indices) < 0:
                raise ValueError(
                    "sweep_update_indices must be a list / np.array of indices >= 0 and < num_trees",
                )
            if np.max(sweep_update_indices) >= num_trees:
                raise ValueError(
                    "sweep_update_indices must be a list / np.array of indices >= 0 and < num_trees",
                )
        self.sweep_update_indices = sweep_update_indices

        if sweep_update_indices is not None:
            sweep_update_indices = _standardize_array_to_np(sweep_update_indices)
            if np.min(sweep_update_indices) < 0:
                raise ValueError(
                    "sweep_update_indices must be a list / np.array of indices >= 0 and < num_trees",
                )
            if np.max(sweep_update_indices) >= num_trees:
                raise ValueError(
                    "sweep_update_indices must be a list / np.array of indices >= 0 and < num_trees",
                )

        if num_features_subsample is None:
            num_features_subsample = num_features
        if num_features_subsample > num_features:
            raise ValueError(
                "`num_features_subsample` cannot be larger than `num_features`",
            )
        if num_features_subsample <= 0:
            raise ValueError(
                "`num_features_subsample` must be at least 1",
            )
        self.num_features_subsample = num_features_subsample

        # Set internal config values
        self.num_trees = num_trees
        self.num_features = num_features
        self.num_observations = num_observations
        self.leaf_dimension = leaf_dimension
        self.alpha = alpha
        self.beta = beta
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.variance_forest_shape = variance_forest_shape
        self.variance_forest_scale = variance_forest_scale
        self.cutpoint_grid_size = cutpoint_grid_size
        self.leaf_model_type = leaf_model_type
        self.leaf_model_scale = leaf_model_scale_array

    def update_feature_types(self, feature_types) -> None:
        """
        Update feature types

        Parameters
        ----------
        feature_types : list or np.ndarray
            Vector of integer-coded feature types (where 0 = numeric, 1 = ordered categorical, 2 = unordered categorical)

        Returns
        -------
        self
        """
        feature_types = _standardize_array_to_np(feature_types)
        if self.num_features != len(feature_types):
            raise ValueError("`feature_types` must have `num_features` total elements")
        self.feature_types = feature_types

    def update_sweep_indices(self, sweep_update_indices) -> None:
        """
        Update feature types

        Parameters
        ----------
        sweep_update_indices : list or np.ndarray
            Vector of (0-indexed) indices of trees to update in a sweep

        Returns
        -------
        self
        """
        if sweep_update_indices is not None:
            sweep_update_indices = _standardize_array_to_np(sweep_update_indices)
            if np.min(sweep_update_indices) < 0:
                raise ValueError(
                    "sweep_update_indices must be a list / np.array of indices >= 0 and < num_trees",
                )
            if np.max(sweep_update_indices) >= self.num_trees:
                raise ValueError(
                    "sweep_update_indices must be a list / np.array of indices >= 0 and < num_trees",
                )
        self.sweep_update_indices = sweep_update_indices

    def update_variable_weights(
        self, variable_weights: Union[list, np.ndarray]
    ) -> None:
        """
        Update variable weights

        Parameters
        ----------
        variable_weights : list or np.ndarray
            List or array specifying sampling probability for all p covariates in ForestDataset

        Returns
        -------
        self
        """
        variable_weights = _standardize_array_to_np(variable_weights)
        if self.num_features != len(variable_weights):
            raise ValueError(
                "`variable_weights` must have `num_features` total elements"
            )
        self.variable_weights = variable_weights

    def update_alpha(self, alpha: float) -> None:
        """
        Update root node split probability in tree prior

        Parameters
        ----------
        alpha : float
            Root node split probability in tree prior

        Returns
        -------
        self
        """
        self.alpha = alpha

    def update_beta(self, beta: float) -> None:
        """
        Update depth prior penalty in tree prior

        Parameters
        ----------
        beta : float
            Depth prior penalty in tree prior

        Returns
        -------
        self
        """
        self.beta = beta

    def update_min_samples_leaf(self, min_samples_leaf: int) -> None:
        """
        Update minimum number of samples per leaf node in the tree prior

        Parameters
        ----------
        min_samples_leaf : int
            Minimum number of samples per leaf node in the tree prior

        Returns
        -------
        self
        """
        self.min_samples_leaf = min_samples_leaf

    def update_max_depth(self, max_depth: int) -> None:
        """
        Update max depth in the tree prior

        Parameters
        ----------
        max_depth : int
            Max depth in the tree prior

        Returns
        -------
        self
        """
        self.max_depth = max_depth

    def update_leaf_model_scale(
        self, leaf_model_scale: Union[float, np.ndarray]
    ) -> None:
        """
        Update scale parameter used in Gaussian leaf models

        Parameters
        ----------
        leaf_model_scale : float or np.ndarray
            Scale parameter used in Gaussian leaf models (can either be a scalar or a q x q matrix, where q is the dimensionality of the basis and is only >1 when `leaf_model_int = 2`).

        Returns
        -------
        self
        """
        if isinstance(leaf_model_scale, np.ndarray):
            if not _check_matrix_square(leaf_model_scale):
                raise ValueError(
                    "`leaf_model_scale` must be a square matrix if provided as a numpy array"
                )
            leaf_model_scale_array = leaf_model_scale
        elif isinstance(leaf_model_scale, (int, float)):
            if leaf_model_scale <= 0:
                raise ValueError(
                    "`leaf_model_scale` must be positive, if provided as scalar"
                )
            leaf_model_scale_array = np.zeros(
                (self.leaf_dimension, self.leaf_dimension), dtype=float
            )
            np.fill_diagonal(leaf_model_scale_array, leaf_model_scale)
        else:
            raise ValueError(
                "`leaf_model_scale` must be a scalar value or a 2d numpy array with matching dimensions"
            )

        self.leaf_model_scale = leaf_model_scale_array

    def update_variance_forest_shape(self, variance_forest_shape: float) -> None:
        """
        Update shape parameter for IG leaf models

        Parameters
        ----------
        variance_forest_shape : float
            Shape parameter for IG leaf models

        Returns
        -------
        self
        """
        self.variance_forest_shape = variance_forest_shape

    def update_variance_forest_scale(self, variance_forest_scale: float) -> None:
        """
        Update scale parameter for IG leaf models

        Parameters
        ----------
        variance_forest_scale : float
            Scale parameter for IG leaf models

        Returns
        -------
        self
        """
        self.variance_forest_scale = variance_forest_scale

    def update_cutpoint_grid_size(self, cutpoint_grid_size: int) -> None:
        """
        Update maximum number of unique cutpoints to consider in a grow-from-root split

        Parameters
        ----------
        cutpoint_grid_size : int
            Maximum number of unique cutpoints to consider in a grow-from-root split

        Returns
        -------
        self
        """
        self.cutpoint_grid_size = cutpoint_grid_size

    def update_num_features_subsample(self, num_features_subsample: int) -> None:
        """
        Update number of features to subsample for the GFR algorithm.

        Parameters
        ----------
        num_features_subsample : int
            Number of features to subsample for the GFR algorithm.

        Returns
        -------
        self
        """
        if num_features_subsample > self.num_features:
            raise ValueError(
                "`num_features_subsample` cannot be larger than `num_features`",
            )
        if num_features_subsample <= 0:
            raise ValueError(
                "`num_features_subsample` must be at least 1",
            )
        self.num_features_subsample = num_features_subsample

    def get_feature_types(self) -> np.ndarray:
        """
        Query feature types (integer-coded so that 0 = numeric, 1 = ordered categorical, 2 = unordered categorical)

        Returns
        -------
        feature_types : np.ndarray
            Array of integer-coded feature types
        """
        return self.feature_types

    def get_sweep_update_indices(self) -> Union[np.ndarray, None]:
        """
        Query vector of (0-indexed) indices of trees to update in a sweep

        Returns
        -------
        sweep_update_indices : np.ndarray or None
            Vector of (0-indexed) indices of trees to update in a sweep, or `None`
        """
        return self.sweep_update_indices

    def get_variable_weights(self) -> np.ndarray:
        """
        Query variable weights

        Returns
        -------
        variable_weights : np.ndarray
            Array of variable split probability weights
        """
        return self.variable_weights

    def get_num_trees(self) -> int:
        """
        Query number of trees

        Returns
        -------
        num_trees : int
            Number of trees in a forest
        """
        return self.num_trees

    def get_num_features(self) -> int:
        """
        Query number of features

        Returns
        -------
        num_features : int
            Number of features in a forest model training set
        """
        return self.num_features

    def get_num_observations(self) -> int:
        """
        Query number of observations

        Returns
        -------
        num_observations : int
            Number of observations in a forest
        """
        return self.num_observations

    def get_alpha(self) -> float:
        """
        Query root node split probability in tree prior

        Returns
        -------
        alpha : float
            Root node split probability in tree prior
        """
        return self.alpha

    def get_beta(self) -> float:
        """
        Query depth prior penalty in tree prior

        Returns
        -------
        beta : float
            Depth prior penalty in tree prior
        """
        return self.beta

    def get_min_samples_leaf(self) -> int:
        """
        Query min samples in a leaf node in the tree prior

        Returns
        -------
        min_samples_leaf : int
            Min samples in a leaf node
        """
        return self.min_samples_leaf

    def get_max_depth(self) -> int:
        """
        Query max depth in the tree prior

        Returns
        -------
        max_depth : int
            Max depth in the tree prior
        """
        return self.max_depth

    def get_leaf_model_type(self) -> int:
        """
        Query (integer-coded) type of leaf model

        Returns
        -------
        leaf_model_type : int
            Integer coded leaf model type
        """
        return self.leaf_model_type

    def get_leaf_model_scale(self) -> np.ndarray:
        """
        Query scale parameter used in Gaussian leaf models

        Returns
        -------
        leaf_model_scale : np.ndarray
            Scale parameter (in array form) used in Gaussian leaf models. If the Gaussian leaf model is univariate, the array returned is a 1x1 matrix.
        """
        return self.leaf_model_scale

    def get_variance_forest_shape(self) -> float:
        """
        Query shape parameter for IG leaf models

        Returns
        -------
        variance_forest_shape : float
            Shape parameter for IG leaf models
        """
        return self.variance_forest_shape

    def get_variance_forest_scale(self) -> float:
        """
        Query scale parameter for IG leaf models

        Returns
        -------
        variance_forest_scale : float
            Scale parameter for IG leaf models
        """
        return self.variance_forest_scale

    def get_cutpoint_grid_size(self) -> int:
        """
        Query maximum number of unique cutpoints considered in a grow-from-root split

        Returns
        -------
        cutpoint_grid_size : int
            Maximum number of unique cutpoints considered in a grow-from-root split
        """
        return self.cutpoint_grid_size

    def get_num_features_subsample(self) -> int:
        """
        Query number of features to subsample for the GFR algorithm

        Returns
        -------
        num_features_subsample : int
            Number of features to subsample for the GFR algorithm
        """
        return self.num_features_subsample


class GlobalModelConfig:
    """
    Object used to get / set global parameters and other global model configuration options in the "low-level" stochtree interface

    The "low-level" stochtree interface enables a high degreee of sampler customization, in which users employ R wrappers around C++ objects
    like ForestDataset, Outcome, CppRng, and ForestModel to run the Gibbs sampler of a BART model with custom modifications.
    GlobalModelConfig allows users to specify / query the global parameters of a model they wish to run.

    Parameters
    ----------
    global_error_variance : float, optional
        Global error variance parameter (default: `1.0`)
    """

    def __init__(
        self,
        global_error_variance=1.0,
    ) -> None:
        # Preprocess inputs and run some error checks
        if not _check_is_numeric(global_error_variance):
            raise ValueError("`global_error_variance` must be a positive scalar")
        elif global_error_variance <= 0:
            raise ValueError("`global_error_variance` must be a positive scalar")

        # Set internal config values
        self.global_error_variance = global_error_variance

    def update_global_error_variance(self, global_error_variance) -> None:
        """
        Update global error variance parameter

        Parameters
        ----------
        global_error_variance : float
            Global error variance parameter

        Returns
        -------
        self
        """
        if not _check_is_numeric(global_error_variance):
            raise ValueError("`global_error_variance` must be a positive scalar")
        elif global_error_variance <= 0:
            raise ValueError("`global_error_variance` must be a positive scalar")
        self.global_error_variance = global_error_variance

    def get_global_error_variance(self) -> float:
        """
        Query the global error variance parameter

        Returns
        -------
        global_error_variance : float
            Global error variance parameter
        """
        return self.global_error_variance
