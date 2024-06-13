"""
Data preprocessing module, drawn largely from the sklearn preprocessing module, released under the BSD-3-Clause license, with the following copyright

Copyright (c) 2007-2024 The scikit-learn developers.
"""
from typing import Union, Optional, Any
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils.validation import check_array, column_or_1d
import numpy as np
import pandas as pd
import warnings

class CovariateTransformer:
    """Class that transforms covariates to a format that can be used to define tree splits
    """

    def __init__(self) -> None:
        self._is_fitted = False
        self._ordinal_encoders = []
        self._onehot_encoders = []
        self._ordinal_feature_index = []
        self._onehot_feature_index = []
        self._processed_feature_types = []
        self._original_feature_types = []
    
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
        category_list = [covariate.array.categories.to_list()]
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
        string_types = covariates.apply(lambda x: pd.api.types.is_integer_dtype(x))
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
                    self._processed_feature_types.append([1 for j in range(len(self._onehot_encoders[onehot_index].categories_))])
            elif string_types.iloc[i]:
                self._original_feature_types[i] = "string"
                onehot_index = self._process_unordered_categorical(covariate)
                self._onehot_feature_index[i] = onehot_index
                self._processed_feature_types.append([1 for j in range(len(self._onehot_encoders[onehot_index].categories_))])
            elif bool_types.iloc[i]:
                self._original_feature_types[i] = "boolean"
                covariate = covariate*1.0
                self._processed_feature_types.append(1)
            elif integer_types.iloc[i]:
                self._original_feature_types[i] = "integer"
                covariate = covariate*1.0
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
        for i in range(covariates.shape[1]):
            covariate = covariates.iloc[:,i]
            if self._original_feature_types[i] == "category" or self._original_feature_types[i] == "string":
                if self._onehot_feature_index[i] != -1:
                    covariate_transformed = self._ordinal_encoders[i].transform(pd.DataFrame(covariate))
                    output_array[:,output_iter] = covariate_transformed
                    output_iter += 1
                else:
                    covariate_transformed = self._onehot_encoders[i].transform(pd.DataFrame(covariate))
                    output_dim = covariate_transformed.shape[1]
                    output_array[:,output_iter:(output_iter + output_dim)] = covariate_transformed
                    output_iter += output_dim
            
            elif self._original_feature_types[i] == "boolean":
                output_array[:,output_iter] = (covariate*1.0).to_numpy()
                output_iter += 1
            
            elif self._original_feature_types[i] == "integer" or self._original_feature_types[i] == "float":
                output_array[:,output_iter] = (covariate).to_numpy()
                output_iter += 1
        
        return output_array

    def _transform_numpy(self, covariates: np.array) -> np.array:
        if covariates.ndim == 1:
            covariates = np.expand_dims(covariates, 1)
        elif covariates.ndim > 2:
            raise ValueError("Covariates passed as a numpy array must be 1d or 2d")
        if self._num_original_features != covariates.shape[1]:
            raise ValueError("Attempting to call transform from a CovariateTransformer that was fit on a dataset with different dimensionality")
        
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
        """Fits a ``CovariateTransformer`` by unpacking (and storing) data type information on the input (raw) covariates
        and then converting to a numpy array which can be passed to a tree ensemble sampler.

        If ``covariates`` is a ``pd.DataFrame``, `column dtypes <https://pandas.pydata.org/docs/user_guide/basics.html#basics-dtypes>`_ 
        will be handled as follows:
        
        * ``category``: one-hot encoded if unordered, ordinal encoded if ordered
        * ``string``: one-hot encoded
        * ``boolean``: passed through as binary integer, treated as ordered categorical by tree samplers
        * integer (i.e. ``Int8``, ``Int16``, etc...): passed through as double (**note**: if you have categorical data stored as integers, you should explicitly convert it to categorical in pandas, see this `user guide <https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html>`_)
        * float (i.e. ``Float32``, ``Float64``): passed through as double
        * ``object``: currently unsupported, convert object columns to numeric or categorical before passing
        * Datetime (i.e. ``datetime64``): currently unsupported, though datetime columns can be converted to numeric features, see `here <https://pandas.pydata.org/docs/reference/api/pandas.Timestamp.html#pandas.Timestamp>`_
        * Period (i.e. ``period[<freq>]``): currently unsupported, though period columns can be converted to numeric features, see `here <https://pandas.pydata.org/docs/reference/api/pandas.Period.html#pandas.Period>`_
        * Interval (i.e. ``interval``, ``Interval[datetime64[ns]]``): currently unsupported, though interval columns can be converted to numeric or categorical features, see `here <https://pandas.pydata.org/docs/reference/api/pandas.Interval.html#pandas.Interval>`_
        * Sparse (i.e. ``Sparse``, ``Sparse[float]``): currently unsupported, convert sparse columns to dense before passing

        Columns with unsupported types will be ignored, with a warning.
        
        If ``covariates`` is a ``np.array``, columns must be numeric and the only preprocessing done by ``CovariateTransformer.fit()`` is to 
        auto-detect binary columns. All other integer-valued columns will be passed through to the tree sampler as (continuous) numeric data. 
        If you would like to treat integer-valued data as categorical, you can either convert your numpy array to a pandas dataframe and 
        explicitly tag such columns as ordered / unordered categorical, or preprocess manually using ``sklearn.preprocessing.OneHotEncoder`` 
        and ``sklearn.preprocessing.OrdinalEncoder``.

        Parameters
        ----------
        covariates : np.array or pd.DataFrame
            Covariates to be preprocessed.
        
        Returns
        -------
        self : CovariateTransformer
            Fitted CovariateTransformer.
        """
        self._fit(covariates)
        return self

    def transform(self, covariates: Union[pd.DataFrame, np.array]) -> np.array:
        return self._transform(covariates)

    def fit_transform(self, covariates: Union[pd.DataFrame, np.array]) -> np.array:
        self._fit(covariates)
        return self._transform(covariates)
