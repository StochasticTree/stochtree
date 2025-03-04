import numpy as np
import pytest

from stochtree.utils import (
    _check_array_integer,
    _check_array_numeric,
    _check_is_int,
    _check_is_numeric,
    _check_matrix_square,
    _standardize_array_to_list,
    _standardize_array_to_np,
)


class TestUtils:
    def test_check_array(self):
        # Test data
        array_list1 = [1, 2, 3, 4, 5]
        array_list2 = [1.5, 2.3, 3.5, 4.1, 5.3]
        array_np1 = np.array([1, 2, 3, 4, 5])
        array_np2 = np.array([1.5, 2.3, 3.5, 4.1, 5.3])
        not_array = dict({"a": 1, "b": 2})

        # Integer checks
        assert _check_array_integer(array_list1)
        assert not _check_array_integer(array_list2)
        assert _check_array_integer(array_np1)
        assert not _check_array_integer(array_np2)
        assert not _check_array_integer(not_array)

        # Numeric checks
        assert _check_array_numeric(array_list1)
        assert _check_array_numeric(array_list2)
        assert _check_array_numeric(array_np1)
        assert _check_array_numeric(array_np2)
        assert not _check_array_numeric(not_array)

    def test_check_scalar(self):
        # Test data
        int_py1 = 1
        int_py2 = 100000000
        float_py1 = 1.5
        float_py2 = 1000000000001.5
        not_scalar = "a"

        # Integer checks
        assert _check_is_int(int_py1)
        assert _check_is_int(int_py2)
        assert not _check_is_int(float_py1)
        assert not _check_is_int(float_py2)
        assert not _check_is_int(not_scalar)

        # Numeric checks
        assert _check_is_numeric(int_py1)
        assert _check_is_numeric(int_py2)
        assert _check_is_numeric(float_py1)
        assert _check_is_numeric(float_py2)
        assert not _check_is_numeric(not_scalar)

    def test_check_matrix(self):
        # Test data
        array_11 = np.array([[1.6]])
        array_22 = np.array([[1.6, 5.6], [2.3, 4.5]])
        array_33 = np.array([[1.6, 5.6, 3.4], [2.3, 4.5, 7.2], [2.7, 6.1, 3.0]])
        array_23 = np.array([[1.6, 5.6, 3.4], [2.3, 4.5, 7.2]])
        array_32 = np.array([[1.6, 5.6], [2.3, 4.5], [2.7, 6.1]])
        non_array_1 = 100000000
        non_array_2 = "a"
        non_array_3 = [[1, 2], [3, 4]]

        # Array checks
        assert _check_matrix_square(array_11)
        assert _check_matrix_square(array_22)
        assert _check_matrix_square(array_33)
        assert not _check_matrix_square(array_23)
        assert not _check_matrix_square(array_32)
        assert not _check_matrix_square(non_array_1)
        assert not _check_matrix_square(non_array_2)
        assert not _check_matrix_square(non_array_3)

    def test_standardize(self):
        # Test data
        array_py1 = [1.6, 3.4, 7.6, 8.7]
        array_py2 = [8.2, 4.5, 3.8]
        array_np1 = np.array([1.6, 3.4, 7.6, 8.7])
        array_np2 = np.array([[1.6, 3.4, 7.6, 8.7]])
        array_np3 = np.array([8.2, 4.5, 3.8])
        array_np4 = np.array([[8.2, 4.5, 3.8]])
        nonconforming_array_np1 = np.array([[8.2, 4.5, 3.8], [1.6, 3.4, 7.6]])
        nonconforming_array_np2 = np.array(
            [[8.2, 4.5, 3.8], [1.6, 3.4, 7.6], [1.6, 3.4, 7.6]]
        )
        non_array_1 = 100000000
        non_array_2 = "a"

        # List standardization checks
        np.testing.assert_array_equal(array_py1, _standardize_array_to_list(array_np1))
        np.testing.assert_array_equal(array_py1, _standardize_array_to_list(array_np2))
        np.testing.assert_array_equal(array_py2, _standardize_array_to_list(array_np3))
        np.testing.assert_array_equal(array_py2, _standardize_array_to_list(array_np4))
        with pytest.raises(ValueError):
            _ = _standardize_array_to_list(non_array_1)
            _ = _standardize_array_to_list(non_array_2)
            _ = _standardize_array_to_list(nonconforming_array_np1)
            _ = _standardize_array_to_list(nonconforming_array_np2)

        # Numpy standardization checks
        np.testing.assert_array_equal(array_np1, _standardize_array_to_np(array_py1))
        np.testing.assert_array_equal(array_np1, _standardize_array_to_np(array_np2))
        np.testing.assert_array_equal(array_np3, _standardize_array_to_np(array_py2))
        np.testing.assert_array_equal(array_np3, _standardize_array_to_np(array_np4))
        with pytest.raises(ValueError):
            _ = _standardize_array_to_np(non_array_1)
            _ = _standardize_array_to_np(non_array_2)
            _ = _standardize_array_to_np(nonconforming_array_np1)
            _ = _standardize_array_to_np(nonconforming_array_np2)
