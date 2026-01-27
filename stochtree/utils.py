from typing import Union
import math

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


def _expand_dims_1d(input: Union[int, float, np.array], output_size: int) -> np.array:
    """
    Convert scalar input to 1D numpy array of dimension `output_size`,
    or check that input array is equivalent to a 1D array of dimension `output_size`.
    Single element numpy arrays (i.e. `np.array([2.5])`) are treated as scalars.

    Parameters
    ----------
    input : int, float, np.array
        Input to be converted to a 1D array (or passed through as-is)
    output_size : int
        Intended size of the output vector

    Returns
    -------
    np.array
        A 1D numpy array of length `output_size`
    """
    if isinstance(input, np.ndarray):
        input = np.squeeze(input)
        if input.ndim > 1:
            raise ValueError(
                "`input` must be convertible to a 1D numpy array or scalar"
            )
        if input.ndim == 0:
            output = np.repeat(input, output_size)
        else:
            if input.shape[0] != output_size:
                raise ValueError(
                    "`input` must be a 1D numpy array with `output_size` elements"
                )
            output = input
    elif isinstance(input, (int, float)):
        output = np.repeat(input, output_size)
    else:
        raise ValueError(
            "`input` must be either a 1D numpy array or a scalar that can be repeated `output_size` times"
        )
    return output


def _expand_dims_2d(
    input: Union[int, float, np.array], output_rows: int, output_cols: int
) -> np.array:
    """
    Ensures that input is propagated appropriately to a 2D numpy array of dimension `output_rows` x `output_cols`.
    Handles the following cases:
        1. `input` is a scalar: output is simply a (`output_rows`, `output_cols`) array with `input` repeated for each element
        2. `input` is a 1D array of length `output_rows`: output is a (`output_rows`, `output_cols`) array with `input` broadcast across each of `output_cols` columns
        3. `input` is a 1D array of length `output_cols`: output is a (`output_rows`, `output_cols`) array with `input` broadcast across each of `output_rows` rows
        4. `input` is a 2D array of dimension (`output_rows`, `output_cols`): input is passed through as-is
    All other cases raise a `ValueError`. Single element numpy arrays (i.e. `np.array([2.5])`) are treated as scalars.

    Parameters
    ----------
    input : int, float, np.array
        Input to be converted to a 2D array (or passed through as-is)
    output_rows : int
        Intended number of rows in the output array
    output_cols : int
        Intended number of columns in the output array

    Returns
    -------
    np.array
        A 2D numpy array of dimension `output_rows` x `output_cols`
    """
    if isinstance(input, np.ndarray):
        input = np.squeeze(input)
        if input.ndim > 2:
            raise ValueError("`input` must be a 1D or 2D numpy array")
        elif input.ndim == 2:
            if input.shape[0] != output_rows:
                raise ValueError(
                    "If `input` is passed as a 2D numpy array, it must contain `output_rows` rows"
                )
            if input.shape[1] != output_cols:
                raise ValueError(
                    "If `input` is passed as a 2D numpy array, it must contain `output_cols` columns"
                )
            output = input
        elif input.ndim == 1:
            if input.shape[0] == output_cols:
                output = np.tile(input, (output_rows, 1))
            elif input.shape[0] == output_rows:
                output = np.tile(input, (output_cols, 1)).T
            else:
                raise ValueError(
                    "If `input` is a 1D numpy array, it must either contain `output_rows` or `output_cols` elements"
                )
        elif input.ndim == 0:
            output = np.tile(input, (output_rows, output_cols))
    elif isinstance(input, (int, float)):
        output = np.tile(input, (output_rows, output_cols))
    else:
        raise ValueError("`input` must be either a 1D or 2D numpy array or a scalar")
    return output


def _expand_dims_2d_diag(
    input: Union[int, float, np.array], output_size: int
) -> np.array:
    """
    Convert scalar input to 2D square numpy array of dimension `output_size` x `output_size` with `input` along the diagonal,
    or check that input array is equivalent to a 2D square array of dimension `output_size` x `output_size`.
    Single element numpy arrays (i.e. `np.array([2.5])`) are treated as scalars.

    Parameters
    ----------
    input : int, float, np.array
        Input to be converted to a 2D square array (or passed through as-is)
    output_size : int
        Intended row and column dimension of the square output matrix

    Returns
    -------
    np.array
        A 2D square numpy array of dimension `output_size` x `output_size`
    """
    if isinstance(input, np.ndarray):
        input = np.squeeze(input)
        if (input.ndim != 2) and (input.ndim != 0):
            raise ValueError(
                "`input` must be convertible to a 2D numpy array or scalar"
            )
        if input.ndim == 0:
            output = np.zeros((output_size, output_size), dtype=float)
            np.fill_diagonal(output, input)
        else:
            if input.shape[0] != input.shape[1]:
                raise ValueError("`input` must be a 2D square numpy array")
            if input.shape[0] != output_size:
                raise ValueError(
                    "`input` must be a 2D square numpy array with exactly `output_size` rows and columns"
                )
            output = input
    elif isinstance(input, (int, float)):
        output = np.zeros((output_size, output_size), dtype=float)
        np.fill_diagonal(output, input)
    else:
        raise ValueError(
            "`input` must be either a 2D square numpy array or a scalar that can be propagated along the diagonal of a square matrix"
        )
    return output


def _posterior_predictive_heuristic_multiplier(
    num_samples: int, num_observations: int
) -> int:
    if num_samples >= 1000:
        return 1
    else:
        return math.ceil(1000 / num_samples)


def _summarize_interval(
    array: np.ndarray, sample_dim: int = 2, level: float = 0.95
) -> dict:
    # Check that the array is numeric and at least 2 dimensional
    if not isinstance(array, np.ndarray):
        raise ValueError("`array` must be a numpy array")
    if not _check_array_numeric(array):
        raise ValueError("`array` must be a numeric numpy array")
    if not len(array.shape) >= 2:
        raise ValueError("`array` must be at least a 2-dimensional numpy array")
    if (
        not _check_is_int(sample_dim)
        or (sample_dim < 0)
        or (sample_dim >= len(array.shape))
    ):
        raise ValueError(
            "`sample_dim` must be an integer between 0 and the number of dimensions of `array` - 1"
        )
    if not isinstance(level, float) or (level <= 0) or (level >= 1):
        raise ValueError("`level` must be a float between 0 and 1")

    # Compute lower and upper quantiles based on the requested interval
    quantile_lb = (1 - level) / 2
    quantile_ub = 1 - quantile_lb

    # Calculate the interval
    result_lb = np.quantile(array, q=quantile_lb, axis=sample_dim)
    result_ub = np.quantile(array, q=quantile_ub, axis=sample_dim)

    # Return results as a dictionary
    return {"lower": result_lb, "upper": result_ub}
