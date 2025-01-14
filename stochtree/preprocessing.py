"""
Data preprocessing module, drawn largely from the sklearn preprocessing module, released under the BSD-3-Clause license, with the following copyright

Copyright (c) 2007-2024 The scikit-learn developers.
"""
from typing import Union, Optional, Any, Dict
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import numpy as np
import pandas as pd
import warnings

def _preprocess_params(default_params: Dict[str, Any], user_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if user_params:
        for key, value in user_params.items():
            if key in default_params:
                default_params[key] = value
    
    return default_params


def _preprocess_bart_params(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    processed_params = {
        'cutpoint_grid_size' : 100, 
        'sigma_leaf' : None, 
        'alpha_mean' : 0.95, 
        'beta_mean' : 2.0, 
        'min_samples_leaf_mean' : 5, 
        'max_depth_mean' : 10, 
        'alpha_variance' : 0.95, 
        'beta_variance' : 2.0, 
        'min_samples_leaf_variance' : 5, 
        'max_depth_variance' : 10, 
        'a_global' : 0, 
        'b_global' : 0, 
        'a_leaf' : 3, 
        'b_leaf' : None, 
        'a_forest' : None, 
        'b_forest' : None, 
        'sigma2_init' : None, 
        'variance_forest_leaf_init' : None, 
        'pct_var_sigma2_init' : 1, 
        'pct_var_variance_forest_init' : 1, 
        'variance_scale' : 1, 
        'variable_weights_mean' : None, 
        'variable_weights_variance' : None, 
        'num_trees_mean' : 200, 
        'num_trees_variance' : 0, 
        'sample_sigma_global' : True, 
        'sample_sigma_leaf' : True, 
        'random_seed' : -1, 
        'keep_burnin' : False, 
        'keep_gfr' : False, 
        'standardize': True, 
        'num_chains' : 1, 
        'keep_every' : 1
    }
    
    if params:
        for key, value in params.items():
            if key not in processed_params:
                raise ValueError(f'Parameter {key} not a valid BART parameter')
            processed_params[key] = value
    
    return processed_params


def _preprocess_bcf_params(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    processed_params = {
        'cutpoint_grid_size': 100, 
        'sigma_leaf_mu': None, 
        'sigma_leaf_tau': None, 
        'alpha_mu': 0.95, 
        'alpha_tau': 0.25, 
        'alpha_variance': 0.95, 
        'beta_mu': 2.0, 
        'beta_tau': 3.0, 
        'beta_variance': 2.0, 
        'min_samples_leaf_mu': 5, 
        'min_samples_leaf_tau': 5, 
        'min_samples_leaf_variance': 5, 
        'max_depth_mu': 10, 
        'max_depth_tau': 5, 
        'max_depth_variance': 10, 
        'a_global': 0, 
        'b_global': 0, 
        'a_leaf_mu': 3, 
        'a_leaf_tau': 3, 
        'b_leaf_mu': None, 
        'b_leaf_tau': None,
        'a_forest' : None, 
        'b_forest' : None,  
        'sigma2_init': None, 
        'variance_forest_leaf_init' : None, 
        'pct_var_sigma2_init': 1, 
        'pct_var_variance_forest_init' : 1, 
        'variable_weights_mu': None, 
        'variable_weights_tau': None, 
        'variable_weights_variance': None, 
        'keep_vars_mu': None, 
        'drop_vars_mu': None, 
        'keep_vars_tau': None, 
        'drop_vars_tau': None, 
        'keep_vars_variance': None, 
        'drop_vars_variance': None, 
        'num_trees_mu': 200, 
        'num_trees_tau': 50, 
        'num_trees_variance': 0, 
        'sample_sigma_global': True, 
        'sample_sigma_leaf_mu': True, 
        'sample_sigma_leaf_tau': False, 
        'propensity_covariate': "mu", 
        'adaptive_coding': True, 
        'b_0': -0.5, 
        'b_1': 0.5, 
        'random_seed': -1, 
        'keep_burnin': False, 
        'keep_gfr': False, 
        'standardize': True, 
        'num_chains' : 1, 
        'keep_every' : 1
    }
    
    if params:
        for key, value in params.items():
            if key not in processed_params:
                raise ValueError(f'Parameter {key} not a valid BCF parameter')
            processed_params[key] = value
    
    return processed_params


class CovariateTransformer:
    """
    Class that transforms covariates to a format that can be used to define tree splits. 
    Modeled after the [scikit-learn preprocessing classes](https://scikit-learn.org/1.5/modules/preprocessing.html).
    """
    def __init__(self) -> None:
        self._is_fitted = False
        self._ordinal_encoders = []
        self._onehot_encoders = []
        self._ordinal_feature_index = []
        self._onehot_feature_index = []
        self._processed_feature_types = []
        self._original_feature_types = []
        self._original_feature_indices = []
    
    def _check_is_numeric_dtype(self, dtype: np.dtype) -> bool:
        if dtype.kind == "b" or dtype.kind == "i" or dtype.kind == "u" or dtype.kind == "f":
            return True
        else:
            return False
    
    def _process_unordered_categorical(self, covariate: pd.Series) -> int:
        num_onehot = len(self._onehot_encoders)
        category_list = covariate.array.categories.to_list()
        enc = OneHotEncoder(categories=[category_list], sparse_output=False)
        enc.fit(pd.DataFrame(covariate))
        self._onehot_encoders.append(enc)
        return num_onehot
    
    def _process_ordered_categorical(self, covariate: pd.Series) -> int:
        num_ord = len(self._ordinal_encoders)
        category_list = covariate.array.categories.to_list()
        enc = OrdinalEncoder(categories=[category_list])
        enc.fit(pd.DataFrame(covariate))
        self._ordinal_encoders.append(enc)
        return num_ord

    def _fit_pandas(self, covariates: pd.DataFrame) -> None:
        self._num_original_features = covariates.shape[1]
        self._ordinal_feature_index = [-1 for i in range(self._num_original_features)]
        self._onehot_feature_index = [-1 for i in range(self._num_original_features)]
        self._original_feature_types = [-1 for i in range(self._num_original_features)]
        datetime_types = covariates.apply(lambda x: pd.api.types.is_datetime64_any_dtype(x))
        object_types = covariates.apply(lambda x: pd.api.types.is_object_dtype(x))
        interval_types = covariates.apply(lambda x: isinstance(x.dtype, pd.IntervalDtype))
        period_types = covariates.apply(lambda x: isinstance(x.dtype, pd.PeriodDtype))
        timedelta_types = np.logical_or(covariates.apply(lambda x: pd.api.types.is_timedelta64_dtype(x)), 
                                        covariates.apply(lambda x: pd.api.types.is_timedelta64_ns_dtype(x)))
        sparse_types = covariates.apply(lambda x: isinstance(x.dtype, pd.SparseDtype))
        bool_types = covariates.apply(lambda x: pd.api.types.is_bool_dtype(x))
        categorical_types = covariates.apply(lambda x: isinstance(x.dtype, pd.CategoricalDtype))
        float_types = covariates.apply(lambda x: pd.api.types.is_float_dtype(x))
        integer_types = covariates.apply(lambda x: pd.api.types.is_integer_dtype(x))
        string_types = covariates.apply(lambda x: pd.api.types.is_string_dtype(x))
        if np.any(datetime_types):
            # raise ValueError("DateTime columns are currently unsupported")
            datetime_cols = covariates.columns[datetime_types].to_list()
            warn_msg = "The following columns are a type unsupported by stochtree (DateTime) and will be ignored: {}"
            warnings.warn(warn_msg.format(datetime_cols))
        if np.any(interval_types):
            # raise ValueError("Interval columns are currently unsupported")
            interval_cols = covariates.columns[interval_types].to_list()
            warn_msg = "The following columns are a type unsupported by stochtree (Interval) and will be ignored: {}"
            warnings.warn(warn_msg.format(interval_cols))
        if np.any(period_types):
            # raise ValueError("Period columns are currently unsupported")
            period_cols = covariates.columns[period_types].to_list()
            warn_msg = "The following columns are a type unsupported by stochtree (Period) and will be ignored: {}"
            warnings.warn(warn_msg.format(period_cols))
        if np.any(timedelta_types):
            # raise ValueError("TimeDelta columns are currently unsupported")
            timedelta_cols = covariates.columns[timedelta_types].to_list()
            warn_msg = "The following columns are a type unsupported by stochtree (TimeDelta) and will be ignored: {}"
            warnings.warn(warn_msg.format(timedelta_cols))
        if np.any(sparse_types):
            # raise ValueError("Sparse columns are currently unsupported")
            sparse_cols = covariates.columns[sparse_types].to_list()
            warn_msg = "The following columns are a type unsupported by stochtree (Sparse) and will be ignored: {}"
            warnings.warn(warn_msg.format(sparse_cols))
        if np.any(object_types):
            # raise ValueError("Object columns are currently unsupported")
            object_cols = covariates.columns[object_types].to_list()
            warn_msg = "The following columns are a type unsupported by stochtree (object) and will be ignored: {}"
            warnings.warn(warn_msg.format(object_cols))
        
        for i in range(covariates.shape[1]):
            covariate = covariates.iloc[:,i]
            if categorical_types.iloc[i]:
                self._original_feature_types[i] = "category"
                if covariate.array.ordered:
                    ord_index = self._process_ordered_categorical(covariate)
                    self._ordinal_feature_index[i] = ord_index
                    self._processed_feature_types.append(1)
                else:
                    onehot_index = self._process_unordered_categorical(covariate)
                    self._onehot_feature_index[i] = onehot_index
                    feature_ones = np.repeat(1, len(covariate.array.categories)).tolist()
                    self._processed_feature_types.extend(feature_ones)
            elif string_types.iloc[i]:
                self._original_feature_types[i] = "string"
                onehot_index = self._process_unordered_categorical(covariate)
                self._onehot_feature_index[i] = onehot_index
                feature_ones = np.repeat(1, len(self._onehot_encoders[onehot_index].categories_[0])).tolist()
                self._processed_feature_types.extend(feature_ones)
            elif bool_types.iloc[i]:
                self._original_feature_types[i] = "boolean"
                self._processed_feature_types.append(1)
            elif integer_types.iloc[i]:
                self._original_feature_types[i] = "integer"
                self._processed_feature_types.append(0)
            elif float_types.iloc[i]:
                self._original_feature_types[i] = "float"
                self._processed_feature_types.append(0)
            else:
                self._original_feature_types[i] = "unsupported"
    
    def _fit_numpy(self, covariates: np.array) -> None:
        if covariates.ndim == 1:
            covariates = np.expand_dims(covariates, 1)
        elif covariates.ndim > 2:
            raise ValueError("Covariates passed as a numpy array must be 1d or 2d")
        
        self._num_original_features = covariates.shape[1]
        self._ordinal_feature_index = [-1 for i in range(self._num_original_features)]
        self._onehot_feature_index = [-1 for i in range(self._num_original_features)]
        self._original_feature_types = ["float" for i in range(self._num_original_features)]

        # Check whether the array is numeric
        cov_dtype = covariates.dtype
        if len(cov_dtype) == 0:
            array_numeric = True
        else:
            array_numeric = True
            for i in range(len(cov_dtype)):
                if not self._check_is_numeric_dtype(cov_dtype[i]):
                    array_numeric = False
        if not array_numeric:
            raise ValueError("Covariates passed as np.array must all be simple numeric types (bool, integer, unsigned integer, floating point)")
        
        # Scan for binary columns
        for i in range(self._num_original_features):
            num_unique = np.unique(covariates[:,i]).size
            if num_unique == 2:
                self._processed_feature_types.append(1)
            else:
                self._processed_feature_types.append(0)

    def _fit(self, covariates: Union[pd.DataFrame, np.array]) -> None:
        if isinstance(covariates, pd.DataFrame):
            self._fit_pandas(covariates)
        elif isinstance(covariates, np.ndarray):
            self._fit_numpy(covariates)
        else:
            raise ValueError("covariates must be a pd.DataFrame or a np.array")
        self._is_fitted = True

    def _transform_pandas(self, covariates: pd.DataFrame) -> np.array:
        if self._num_original_features != covariates.shape[1]:
            raise ValueError("Attempting to call transform from a CovariateTransformer that was fit on a dataset with different dimensionality")
        
        output_array = np.empty((covariates.shape[0], len(self._processed_feature_types)), dtype=np.float64)
        output_iter = 0
        self._original_feature_indices = []
        for i in range(covariates.shape[1]):
            covariate = covariates.iloc[:,i]
            if self._original_feature_types[i] == "category" or self._original_feature_types[i] == "string":
                if self._ordinal_feature_index[i] != -1:
                    ord_ind = self._ordinal_feature_index[i]
                    covariate_transformed = self._ordinal_encoders[ord_ind].transform(pd.DataFrame(covariate))
                    output_array[:,output_iter] = np.squeeze(covariate_transformed)
                    output_iter += 1
                    self._original_feature_indices.append(i)
                else:
                    onehot_ind = self._onehot_feature_index[i]
                    covariate_transformed = self._onehot_encoders[onehot_ind].transform(pd.DataFrame(covariate))
                    output_dim = covariate_transformed.shape[1]
                    output_array[:,np.arange(output_iter, output_iter + output_dim)] = np.squeeze(covariate_transformed)
                    output_iter += output_dim
                    self._original_feature_indices.extend([i for _ in range(output_dim)])
            
            elif self._original_feature_types[i] == "boolean":
                output_array[:,output_iter] = (covariate*1.0).to_numpy()
                output_iter += 1
                self._original_feature_indices.append(i)
            
            elif self._original_feature_types[i] == "integer" or self._original_feature_types[i] == "float":
                output_array[:,output_iter] = (covariate).to_numpy()
                output_iter += 1
                self._original_feature_indices.append(i)
        
        return output_array

    def _transform_numpy(self, covariates: np.array) -> np.array:
        if covariates.ndim == 1:
            covariates = np.expand_dims(covariates, 1)
        elif covariates.ndim > 2:
            raise ValueError("Covariates passed as a numpy array must be 1d or 2d")
        if self._num_original_features != covariates.shape[1]:
            raise ValueError("Attempting to call transform from a CovariateTransformer that was fit on a dataset with different dimensionality")
        self._original_feature_indices = [i for i in range(covariates.shape[1])]
        return covariates

    def _transform(self, covariates: Union[pd.DataFrame, np.array]) -> np.array:
        if self._check_is_fitted():
            if isinstance(covariates, pd.DataFrame):
                return self._transform_pandas(covariates)
            elif isinstance(covariates, np.ndarray):
                return self._transform_numpy(covariates)
            else:
                raise ValueError("covariates must be a pd.DataFrame or a np.array")
        else:
            raise ValueError("Attempting to call transform() from an CovariateTransformer that has not yet been fit")

    def _check_is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, covariates: Union[pd.DataFrame, np.array]) -> None:
        r"""Fits a `CovariateTransformer` by unpacking (and storing) data type information on the input (raw) covariates
        and then converting to a numpy array which can be passed to a tree ensemble sampler.

        If `covariates` is a `pd.DataFrame`, [column dtypes](https://pandas.pydata.org/docs/user_guide/basics.html#basics-dtypes) 
        will be handled as follows:
        
        * `category`: one-hot encoded if unordered, ordinal encoded if ordered
        * `string`: one-hot encoded
        * `boolean`: passed through as binary integer, treated as ordered categorical by tree samplers
        * integer (i.e. `Int8`, `Int16`, etc...): passed through as double (**note**: if you have categorical data stored as integers, you should explicitly convert it to categorical in pandas, see this [user guide](https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html))
        * float (i.e. `Float32`, `Float64`): passed through as double
        * `object`: currently unsupported, convert object columns to numeric or categorical before passing
        * Datetime (i.e. `datetime64`): currently unsupported, though datetime columns can be converted to numeric features, see [here](https://pandas.pydata.org/docs/reference/api/pandas.Timestamp.html#pandas.Timestamp)
        * Period (i.e. `period[<freq>]`): currently unsupported, though period columns can be converted to numeric features, see [here](https://pandas.pydata.org/docs/reference/api/pandas.Period.html#pandas.Period)
        * Interval (i.e. `interval`, `Interval[datetime64[ns]]`): currently unsupported, though interval columns can be converted to numeric or categorical features, see [here](https://pandas.pydata.org/docs/reference/api/pandas.Interval.html#pandas.Interval)
        * Sparse (i.e. `Sparse`, `Sparse[float]`): currently unsupported, convert sparse columns to dense before passing

        Columns with unsupported types will be ignored, with a warning.
        
        If `covariates` is a `np.array`, columns must be numeric and the only preprocessing done by `CovariateTransformer.fit()` is to 
        auto-detect binary columns. All other integer-valued columns will be passed through to the tree sampler as (continuous) numeric data. 
        If you would like to treat integer-valued data as categorical, you can either convert your numpy array to a pandas dataframe and 
        explicitly tag such columns as ordered / unordered categorical, or preprocess manually using `sklearn.preprocessing.OneHotEncoder` 
        and `sklearn.preprocessing.OrdinalEncoder`.

        Parameters
        ----------
        covariates : np.array or pd.DataFrame
            Covariates to be preprocessed.
        """
        self._fit(covariates)
        return self

    def transform(self, covariates: Union[pd.DataFrame, np.array]) -> np.array:
        r"""Run a fitted a `CovariateTransformer` on a new covariate set, 
        returning a numpy array of covariates preprocessed into a format needed 
        to sample or predict from a `stochtree` ensemble.

        Parameters
        ----------
        covariates : np.array or pd.DataFrame
            Covariates to be preprocessed.
        
        Returns
        -------
        np.array
            Numpy array of preprocessed covariates, with as many rows as in `covariates` 
            and as many columns as were created during pre-processing (including one-hot encoding 
            categorical features).
        """
        return self._transform(covariates)

    def fit_transform(self, covariates: Union[pd.DataFrame, np.array]) -> np.array:
        r"""Runs the `fit()` and `transform()` methods in sequence.

        Parameters
        ----------
        covariates : np.array or pd.DataFrame
            Covariates to be preprocessed.
        
        Returns
        -------
        np.array
            Numpy array of preprocessed covariates, with as many rows as in `covariates` 
            and as many columns as were created during pre-processing (including one-hot encoding 
            categorical features).
        """
        self._fit(covariates)
        return self._transform(covariates)
    
    def fetch_original_feature_indices(self) -> list:
        r"""Map features in a preprocessed covariate set back to the 
        original set of features provided to a `CovariateTransformer`.

        Returns
        -------
        list
            List with as many entries as features in the preprocessed results 
            returned by a fitted `CovariateTransformer`. Each element is a feature 
            index indicating the feature from which a given preprocessed feature was generated.
            If a single categorical feature were one-hot encoded into 5 binary features, 
            this method would return a list `[0,0,0,0,0]`. If the transformer merely passes
            through `k` numeric features, this method would return a list `[0,...,k-1]`.
        """
        return self._original_feature_indices
