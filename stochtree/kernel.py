from typing import Union

import pandas as pd
import numpy as np
from stochtree_cpp import (
    cppComputeForestContainerLeafIndices,
    cppComputeForestMaxLeafIndex,
)

from .bart import BARTModel
from .bcf import BCFModel
from .forest import ForestContainer


def compute_forest_leaf_indices(
    model_object: Union[BARTModel, BCFModel, ForestContainer],
    covariates: Union[np.array, pd.DataFrame],
    forest_type: str = None,
    propensity: np.array = None,
    forest_inds: Union[int, np.ndarray] = None,
):
    """
    Compute and return a vector representation of a forest's leaf predictions for every observation in a dataset.

    The vector has a "row-major" format that can be easily re-represented as as a CSR sparse matrix: elements are organized so that the first `n` elements
    correspond to leaf predictions for all `n` observations in a dataset for the first tree in an ensemble, the next `n` elements correspond to predictions for
    the second tree and so on. The "data" for each element corresponds to a uniquely mapped column index that corresponds to a single leaf of a single tree (i.e.
    if tree 1 has 3 leaves, its column indices range from 0 to 2, and then tree 2's leaf indices begin at 3, etc...).

    Parameters
    ----------
    model_object : BARTModel, BCFModel, or ForestContainer
        Object corresponding to a BART / BCF model with at least one forest sample, or a low-level `ForestContainer` object.
    covariates : np.array or pd.DataFrame
        Covariates to use for prediction. Must have the same dimensions / column types as the data used to train a forest.
    forest_type : str
        Which forest to use from `model_object`. Valid inputs depend on the model type, and whether or not a given forest was sampled in that model.

            * **BART**
                * `'mean'`: `'mean'`: Extracts leaf indices for the mean forest
                * `'variance'`: Extracts leaf indices for the variance forest
            * **BCF**
                * `'prognostic'`: Extracts leaf indices for the prognostic forest
                * `'treatment'`: Extracts leaf indices for the treatment effect forest
                * `'variance'`: Extracts leaf indices for the variance forest
            * **ForestContainer**
                * `NULL`: It is not necessary to disambiguate when this function is called directly on a `ForestSamples` object. This is the default value of this

    propensity : `np.array`, optional
        Optional test set propensities. Must be provided if propensities were provided when the model was sampled.
    forest_inds : int or np.ndarray
        Indices of the forest sample(s) for which to compute leaf indices. If not provided, this function will return leaf indices for every sample of a forest.
        This function uses 0-indexing, so the first forest sample corresponds to `forest_num = 0`, and so on.

    Returns
    -------
    Numpy array with dimensions `num_obs` by `num_trees`, where `num_obs` is the number of rows in `covaritates` and `num_trees` is the number of trees in the relevant forest of `model_object`.
    """
    # Extract relevant forest container
    if (
        not isinstance(model_object, BARTModel)
        and not isinstance(model_object, BCFModel)
        and not isinstance(model_object, ForestContainer)
    ):
        raise ValueError(
            "model_object must be one of BARTModel, BCFModel, or ForestContainer"
        )
    if isinstance(model_object, BARTModel):
        model_type = "bart"
        if forest_type is None:
            raise ValueError(
                "forest_type must be specified for a BARTModel model_type (either set to 'mean' or 'variance')"
            )
    elif isinstance(model_object, BCFModel):
        model_type = "bcf"
        if forest_type is None:
            raise ValueError(
                "forest_type must be specified for a BCFModel model_type (either set to 'prognostic', 'treatment' or 'variance')"
            )
    else:
        model_type = "forest"
    if model_type == "bart":
        if forest_type == "mean":
            if not model_object.include_mean_forest:
                raise ValueError(
                    "Mean forest was not sampled for model_object, but requested by forest_type"
                )
            forest_container = model_object.forest_container_mean
        else:
            if not model_object.include_variance_forest:
                raise ValueError(
                    "Variance forest was not sampled for model_object, but requested by forest_type"
                )
            forest_container = model_object.forest_container_variance
    elif model_type == "bcf":
        if forest_type == "prognostic":
            forest_container = model_object.forest_container_mu
        elif forest_type == "treatment":
            forest_container = model_object.forest_container_tau
        else:
            if not model_object.include_variance_forest:
                raise ValueError(
                    "Variance forest was not sampled for model_object, but requested by forest_type"
                )
            forest_container = model_object.forest_container_variance
    else:
        forest_container = model_object

    if not isinstance(covariates, pd.DataFrame) and not isinstance(
        covariates, np.ndarray
    ):
        raise ValueError("covariates must be a matrix or dataframe")

    # Preprocess covariates
    if model_type == "bart" or model_type == "bcf":
        covariates_processed = model_object._covariate_preprocessor.transform(
            covariates
        )
    else:
        covariates_processed = covariates
    covariates_processed = np.asfortranarray(covariates_processed)

    # Handle BCF propensity covariate
    if model_type == "bcf":
        if model_object.propensity_covariate != "none":
            if propensity is None:
                if not model_object.internal_propensity_model:
                    raise ValueError(
                        "Propensity scores not provided, but no propensity model was trained during sampling"
                    )
                propensity = np.mean(
                    model_object.bart_propensity_model.predict(covariates),
                    axis=1,
                    keepdims=True,
                )
            covariates_processed = np.c_[covariates_processed, propensity]

    # Preprocess forest indices
    num_forests = forest_container.num_samples()
    if forest_inds is None:
        forest_inds = np.arange(num_forests)
    elif isinstance(forest_inds, int):
        if not forest_inds >= 0 or not forest_inds < num_forests:
            raise ValueError(
                "The index in forest_inds must be >= 0 and < the total number of samples in a forest container"
            )
        forest_inds = np.array([forest_inds])
    elif isinstance(forest_inds, np.ndarray):
        if forest_inds.size > 1:
            forest_inds = np.squeeze(forest_inds)
            if forest_inds.ndim > 1:
                raise ValueError("forest_inds must be a one-dimensional numpy array")
        if not np.all(forest_inds >= 0) or not np.all(forest_inds < num_forests):
            raise ValueError(
                "The indices in forest_inds must be >= 0 and < the total number of samples in a forest container"
            )
    else:
        raise ValueError("forest_inds must be a one-dimensional numpy array")

    return cppComputeForestContainerLeafIndices(
        forest_container.forest_container_cpp, covariates_processed, forest_inds
    )


def compute_forest_max_leaf_index(
    model_object: Union[BARTModel, BCFModel, ForestContainer],
    forest_type: str = None,
    forest_inds: Union[int, np.ndarray] = None,
):
    """
    Compute and return the largest possible leaf index computable by `compute_forest_leaf_indices` for the forests in a designated forest sample container.

    Parameters
    ----------
    model_object : BARTModel, BCFModel, or ForestContainer
        Object corresponding to a BART / BCF model with at least one forest sample, or a low-level `ForestContainer` object.
    forest_type : str
        Which forest to use from `model_object`. Valid inputs depend on the model type, and whether or not a given forest was sampled in that model.

            * **BART**
                * `'mean'`: `'mean'`: Extracts leaf indices for the mean forest
                * `'variance'`: Extracts leaf indices for the variance forest
            * **BCF**
                * `'prognostic'`: Extracts leaf indices for the prognostic forest
                * `'treatment'`: Extracts leaf indices for the treatment effect forest
                * `'variance'`: Extracts leaf indices for the variance forest
            * **ForestContainer**
                * `NULL`: It is not necessary to disambiguate when this function is called directly on a `ForestSamples` object. This is the default value of this

    forest_inds : int or np.ndarray
        Indices of the forest sample(s) for which to compute max leaf indices. If not provided, this function will return max leaf indices for every sample of a forest.
        This function uses 0-indexing, so the first forest sample corresponds to `forest_num = 0`, and so on.

    Returns
    -------
    Numpy array containing the largest possible leaf index computable by `compute_forest_leaf_indices` for the forests in a designated forest sample container.
    """
    # Extract relevant forest container
    if (
        not isinstance(model_object, BARTModel)
        and not isinstance(model_object, BCFModel)
        and not isinstance(model_object, ForestContainer)
    ):
        raise ValueError(
            "model_object must be one of BARTModel, BCFModel, or ForestContainer"
        )
    if isinstance(model_object, BARTModel):
        model_type = "bart"
        if forest_type is None:
            raise ValueError(
                "forest_type must be specified for a BARTModel model_type (either set to 'mean' or 'variance')"
            )
    elif isinstance(model_object, BCFModel):
        model_type = "bcf"
        if forest_type is None:
            raise ValueError(
                "forest_type must be specified for a BCFModel model_type (either set to 'prognostic', 'treatment' or 'variance')"
            )
    else:
        model_type = "forest"
    if model_type == "bart":
        if forest_type == "mean":
            if not model_object.include_mean_forest:
                raise ValueError(
                    "Mean forest was not sampled for model_object, but requested by forest_type"
                )
            forest_container = model_object.forest_container_mean
        else:
            if not model_object.include_variance_forest:
                raise ValueError(
                    "Variance forest was not sampled for model_object, but requested by forest_type"
                )
            forest_container = model_object.forest_container_variance
    elif model_type == "bcf":
        if forest_type == "prognostic":
            forest_container = model_object.forest_container_mu
        elif forest_type == "treatment":
            forest_container = model_object.forest_container_tau
        else:
            if not model_object.include_variance_forest:
                raise ValueError(
                    "Variance forest was not sampled for model_object, but requested by forest_type"
                )
            forest_container = model_object.forest_container_variance
    else:
        forest_container = model_object

    # Preprocess forest indices
    num_forests = forest_container.num_samples()
    if forest_inds is None:
        forest_inds = np.arange(num_forests)
    elif isinstance(forest_inds, int):
        if not forest_inds >= 0 or not forest_inds < num_forests:
            raise ValueError(
                "The index in forest_inds must be >= 0 and < the total number of samples in a forest container"
            )
        forest_inds = np.array([forest_inds])
    elif isinstance(forest_inds, np.ndarray):
        if forest_inds.size > 1:
            forest_inds = np.squeeze(forest_inds)
            if forest_inds.ndim > 1:
                raise ValueError("forest_inds must be a one-dimensional numpy array")
        if not np.all(forest_inds >= 0) or not np.all(forest_inds < num_forests):
            raise ValueError(
                "The indices in forest_inds must be >= 0 and < the total number of samples in a forest container"
            )
    else:
        raise ValueError("forest_inds must be a one-dimensional numpy array")

    # Compute max index
    output_size = len(forest_inds)
    output = np.empty(output_size)
    for i in np.arange(output_size):
        output[i] = cppComputeForestMaxLeafIndex(
            forest_container.forest_container_cpp, forest_inds[i]
        )

    # Return result
    return output
