from typing import Union

import pandas as pd
import numpy as np
from stochtree import BARTModel, BCFModel, ForestContainer

from .data import Residual
from .sampler import RNG


def compute_forest_leaf_indices(model_object: Union[BARTModel, BCFModel, ForestContainer], covariates: Union[np.array, pd.DataFrame], forest_type: str = None, forest_inds: Union[int, np.ndarray] = None):
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
    
    forest_inds : int or np.ndarray
        Indices of the forest sample(s) for which to compute leaf indices. If not provided, this function will return leaf indices for every sample of a forest. 
        This function uses 0-indexing, so the first forest sample corresponds to `forest_num = 0`, and so on.
    
    Returns
    -------
    Numpy array with dimensions `num_obs` by `num_trees`, where `num_obs` is the number of rows in `covaritates` and `num_trees` is the number of trees in the relevant forest of `model_object`. 
    """
    pass
