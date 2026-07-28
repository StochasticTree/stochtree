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
    _expand_dims_1d,
    _expand_dims_2d,
    _expand_dims_2d_diag,
    OutcomeModel
)


class TestUtils:
    def test_outcome_model(self):
        # Valid initializations with both terms specified
        model = OutcomeModel(outcome="continuous", link="identity")
        assert model.outcome == "continuous"
        assert model.link == "identity"
        model = OutcomeModel(outcome="binary", link="probit")
        assert model.outcome == "binary"
        assert model.link == "probit"
        model = OutcomeModel(outcome="ordinal", link="cloglog")
        assert model.outcome == "ordinal"
        assert model.link == "cloglog"

        # Valid initializations with only outcome specified
        model = OutcomeModel(outcome="continuous")
        assert model.outcome == "continuous"
        assert model.link == "identity"
        model = OutcomeModel(outcome="binary")
        assert model.outcome == "binary"
        assert model.link == "probit"
        model = OutcomeModel(outcome="ordinal")
        assert model.outcome == "ordinal"
        assert model.link == "cloglog"

        # Invalid initializations
        with pytest.raises(ValueError):
            _ = OutcomeModel(outcome="continuous", link="other")
            _ = OutcomeModel(outcome="binary", link="other")
            _ = OutcomeModel(outcome="ordinal", link="other")
            _ = OutcomeModel(outcome="other", link="identity")
            _ = OutcomeModel(outcome="other", link="probit")
            _ = OutcomeModel(outcome="other", link="cloglog")

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
        nonconforming_array_np2 = np.array([
            [8.2, 4.5, 3.8],
            [1.6, 3.4, 7.6],
            [1.6, 3.4, 7.6],
        ])
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

    def test_array_conversion(self):
        scalar_1 = 1.5
        scalar_2 = -2.5
        scalar_3 = 4
        array_1d_1 = np.array([1.6, 3.4, 7.6, 8.7])
        array_1d_2 = np.array([2.5, 3.1, 5.6])
        array_1d_3 = np.array([2.5])
        array_2d_1 = np.array([
            [2.5, 1.2, 4.3, 7.4],
            [1.7, 2.9, 3.6, 9.1],
            [7.2, 4.5, 6.7, 1.4],
        ])
        array_2d_2 = np.array([[2.5, 1.2, 4.3, 7.4], [1.7, 2.9, 3.6, 9.1]])
        array_square_1 = np.array([[2.5, 1.2], [1.7, 2.9]])
        array_square_2 = np.array([[2.5, 0.0], [0.0, 2.9]])
        array_square_3 = np.array([[2.5, 0.0, 0.0], [0.0, 2.9, 0.0], [0.0, 0.0, 5.6]])
        with pytest.raises(ValueError):
            _ = _expand_dims_1d(array_1d_1, 5)
            _ = _expand_dims_1d(array_1d_2, 4)
            _ = _expand_dims_1d(array_1d_3, 3)
            _ = _expand_dims_2d(array_2d_1, 2, 4)
            _ = _expand_dims_2d(array_2d_2, 3, 4)
            _ = _expand_dims_2d_diag(array_square_1, 4)
            _ = _expand_dims_2d_diag(array_square_2, 3)
            _ = _expand_dims_2d_diag(array_square_3, 2)

        np.testing.assert_array_equal(
            np.array([scalar_1, scalar_1, scalar_1]), _expand_dims_1d(scalar_1, 3)
        )
        np.testing.assert_array_equal(
            np.array([scalar_2, scalar_2, scalar_2, scalar_2]),
            _expand_dims_1d(scalar_2, 4),
        )
        np.testing.assert_array_equal(
            np.array([scalar_3, scalar_3]), _expand_dims_1d(scalar_3, 2)
        )
        np.testing.assert_array_equal(
            np.array([array_1d_3[0], array_1d_3[0], array_1d_3[0]]),
            _expand_dims_1d(array_1d_3, 3),
        )

        np.testing.assert_array_equal(
            np.array([[scalar_1, scalar_1, scalar_1], [scalar_1, scalar_1, scalar_1]]),
            _expand_dims_2d(scalar_1, 2, 3),
        )
        np.testing.assert_array_equal(
            np.array([
                [scalar_2, scalar_2, scalar_2, scalar_2],
                [scalar_2, scalar_2, scalar_2, scalar_2],
            ]),
            _expand_dims_2d(scalar_2, 2, 4),
        )
        np.testing.assert_array_equal(
            np.array([
                [scalar_3, scalar_3],
                [scalar_3, scalar_3],
                [scalar_3, scalar_3],
            ]),
            _expand_dims_2d(scalar_3, 3, 2),
        )
        np.testing.assert_array_equal(
            np.array([
                [array_1d_3[0], array_1d_3[0]],
                [array_1d_3[0], array_1d_3[0]],
                [array_1d_3[0], array_1d_3[0]],
            ]),
            _expand_dims_2d(array_1d_3, 3, 2),
        )
        np.testing.assert_array_equal(
            np.vstack((array_1d_1, array_1d_1)), _expand_dims_2d(array_1d_1, 2, 4)
        )
        np.testing.assert_array_equal(
            np.vstack((array_1d_2, array_1d_2, array_1d_2)),
            _expand_dims_2d(array_1d_2, 3, 3),
        )
        np.testing.assert_array_equal(
            np.column_stack((array_1d_2, array_1d_2, array_1d_2, array_1d_2)),
            _expand_dims_2d(array_1d_2, 3, 4),
        )
        np.testing.assert_array_equal(
            np.column_stack((array_1d_3, array_1d_3, array_1d_3, array_1d_3)),
            _expand_dims_2d(array_1d_3, 1, 4),
        )
        np.testing.assert_array_equal(
            np.vstack((array_1d_3, array_1d_3, array_1d_3, array_1d_3)),
            _expand_dims_2d(array_1d_3, 4, 1),
        )

        np.testing.assert_array_equal(
            np.array([
                [scalar_1, 0.0, 0.0],
                [0.0, scalar_1, 0.0],
                [0.0, 0.0, scalar_1],
            ]),
            _expand_dims_2d_diag(scalar_1, 3),
        )
        np.testing.assert_array_equal(
            np.array([[scalar_2, 0.0], [0.0, scalar_2]]),
            _expand_dims_2d_diag(scalar_2, 2),
        )
        np.testing.assert_array_equal(
            np.array([
                [scalar_3, 0.0, 0.0, 0.0],
                [0.0, scalar_3, 0.0, 0.0],
                [0.0, 0.0, scalar_3, 0.0],
                [0.0, 0.0, 0.0, scalar_3],
            ]),
            _expand_dims_2d_diag(scalar_3, 4),
        )
        np.testing.assert_array_equal(
            np.array([[array_1d_3[0], 0.0], [0.0, array_1d_3[0]]]),
            _expand_dims_2d_diag(array_1d_3, 2),
        )
