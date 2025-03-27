from typing import Union

import numpy as np


class NotSampledError(ValueError, AttributeError):
    """Exception class to raise if attempting to predict from a model before it has been sampled.

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.

    Renamed from scikit-learn's "NotFittedError"
    https://github.com/scikit-learn/scikit-learn/blob/8721245511de2f225ff5f9aa5f5fadce663cd4a3/sklearn/exceptions.py#L45C7-L45C21
    """


def _standardize_array_to_list(input: Union[list, np.ndarray]) -> list:
    """
    Standarize an array (either a python list or numpy array) to a python list

    Parameters
    ----------
    input : list or np.array
        Array to be standardized

    Returns
    -------
    list
        Input array, standardized into a simple python list
    """
    if isinstance(input, list):
        return input
    elif isinstance(input, np.ndarray):
        if input.ndim > 1:
            if np.squeeze(input).ndim > 1:
                raise ValueError(
                    "`input` is not a one-dimensional numpy array, cannot be flattened into a python list"
                )
            return np.squeeze(input).tolist()
        else:
            return input.tolist()
    else:
        raise ValueError("`input` must be either a list or numpy array")


def _standardize_array_to_np(input: Union[list, np.ndarray]) -> np.ndarray:
    """
    Standarize an array (either a python list or numpy array) to a 1d numpy array

    Parameters
    ----------
    input : list or np.array
        Array to be standardized

    Returns
    -------
    np.array
        Input array, standardized into a 1d numpy array
    """
    if isinstance(input, list):
        return np.array(input)
    elif isinstance(input, np.ndarray):
        if input.ndim > 1:
            if np.squeeze(input).ndim > 1:
                raise ValueError(
                    "`input` is not a one-dimensional numpy array, cannot be flattened into a 1d numpy array"
                )
            return np.squeeze(input)
        else:
            return input
    else:
        raise ValueError("`input` must be either a list or numpy array")


def _check_is_int(input: Union[int, float]) -> bool:
    """
    Checks whether a scalar input is or is convertible to an integer

    Parameters
    ----------
    input : int or float
        Input to be checked for integer status

    Returns
    -------
    bool
        True if integer, False otherwise
    """
    if not isinstance(input, (int, float)):
        return False
    elif isinstance(input, float):
        return int(input) == input
    else:
        return True


def _check_is_numeric(input: Union[int, float]) -> bool:
    """
    Checks whether a scalar input is numeric

    Parameters
    ----------
    input : int or float
        Input to be checked for numeric status

    Returns
    -------
    bool
        True if integer, False otherwise
    """
    if not isinstance(input, (int, float)):
        return False
    else:
        return True


def _check_array_numeric(input: Union[list, np.ndarray]) -> bool:
    """
    Checks whether an array is populated with numeric values

    Parameters
    ----------
    input : list or np.ndarray
        Input array to be checked for numeric values

    Returns
    -------
    bool
        True if the array is all numeric values, False otherwise
    """
    if isinstance(input, list):
        return all([isinstance(item, (int, float)) for item in input])
    elif isinstance(input, np.ndarray):
        return np.issubdtype(input.dtype, np.number)
    else:
        return False


def _check_array_integer(input: Union[list, np.ndarray]) -> bool:
    """
    Checks whether an array is populated with integer values

    Parameters
    ----------
    input : list or np.ndarray
        Input array to be checked for integer values

    Returns
    -------
    bool
        True if the array is all integer values, False otherwise
    """
    if isinstance(input, list):
        return all([isinstance(item, (int)) for item in input])
    elif isinstance(input, np.ndarray):
        return np.issubdtype(input.dtype, np.integer)
    else:
        return False


def _check_matrix_square(input: np.ndarray) -> bool:
    """
    Checks whether a numpy array is a 2d square matrix

    Parameters
    ----------
    input : np.ndarray
        Input array to be checked

    Returns
    -------
    bool
        True if the array is a square matrix, False otherwise
    """
    if isinstance(input, np.ndarray):
        if input.ndim == 2:
            nrow, ncol = input.shape
            return nrow == ncol
        elif input.ndim > 2:
            if np.squeeze(input).ndim == 2:
                nrow, ncol = np.squeeze(input).shape
                return nrow == ncol
            else:
                return False
        else:
            return False
    else:
        return False
