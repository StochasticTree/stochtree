import numpy as np

from stochtree import Dataset, RandomEffectsDataset


class TestDataset:
    def test_dataset_update(self):
        # Generate data
        n = 20
        num_covariates = 10
        num_basis = 5
        rng = np.random.default_rng()
        covariates = rng.uniform(0, 1, size=(n, num_covariates))
        basis = rng.uniform(0, 1, size=(n, num_basis))
        variance_weights = rng.uniform(0, 1, size=n)

        # Construct dataset
        forest_dataset = Dataset()
        forest_dataset.add_covariates(covariates)
        forest_dataset.add_basis(basis)
        forest_dataset.add_variance_weights(variance_weights)
        assert forest_dataset.num_observations() == n
        assert forest_dataset.num_covariates() == num_covariates
        assert forest_dataset.num_basis() == num_basis
        assert forest_dataset.has_variance_weights()

        # Update dataset
        new_basis = rng.uniform(0, 1, size=(n, num_basis))
        new_variance_weights = rng.uniform(0, 1, size=n)
        with np.testing.assert_no_warnings():
            forest_dataset.update_basis(new_basis)
            forest_dataset.update_variance_weights(new_variance_weights)

        # Check that we recover the correct data through get_covariates, get_basis, and get_variance_weights
        np.testing.assert_array_equal(forest_dataset.get_covariates(), covariates)
        np.testing.assert_array_equal(forest_dataset.get_basis(), new_basis)
        np.testing.assert_array_equal(
            forest_dataset.get_variance_weights(), new_variance_weights
        )


class TestRFXDataset:
    def test_rfx_dataset_update(self):
        # Generate data
        n = 20
        num_groups = 4
        num_basis = 5
        rng = np.random.default_rng()
        group_labels = rng.choice(num_groups, size=n)
        basis = np.empty((n, num_basis))
        basis[:, 0] = 1.0
        if num_basis > 1:
            basis[:, 1:] = rng.uniform(-1, 1, (n, num_basis - 1))
        variance_weights = rng.uniform(0, 1, size=n)

        # Construct dataset
        rfx_dataset = RandomEffectsDataset()
        rfx_dataset.add_group_labels(group_labels)
        rfx_dataset.add_basis(basis)
        rfx_dataset.add_variance_weights(variance_weights)
        assert rfx_dataset.num_observations() == n
        assert rfx_dataset.num_basis() == num_basis
        assert rfx_dataset.has_variance_weights()

        # Update dataset
        new_basis = rng.uniform(0, 1, size=(n, num_basis))
        new_variance_weights = rng.uniform(0, 1, size=n)
        with np.testing.assert_no_warnings():
            rfx_dataset.update_basis(new_basis)
            rfx_dataset.update_variance_weights(new_variance_weights)

        # Check that we recover the correct data through get_group_labels, get_basis, and get_variance_weights
        np.testing.assert_array_equal(rfx_dataset.get_group_labels(), group_labels)
        np.testing.assert_array_equal(rfx_dataset.get_basis(), new_basis)
        np.testing.assert_array_equal(
            rfx_dataset.get_variance_weights(), new_variance_weights
        )


class TestReadOnlyInputs:
    """Read-only numpy inputs must be accepted at the C++ boundary.

    Under pandas Copy-on-Write (default from pandas 3.0), `df.loc[:, col].to_numpy()`
    returns a read-only array. The C++ bindings take mutable access to the outcome and
    observation-weight buffers, so these inputs previously raised
    "ValueError: array is not writeable".
    """

    @staticmethod
    def _read_only(arr):
        arr = arr.copy()
        arr.flags.writeable = False
        return arr

    @staticmethod
    def _data(seed=1):
        rng = np.random.default_rng(seed)
        n, p = 150, 3
        X = rng.uniform(size=(n, p))
        y = X[:, 0] + rng.normal(size=n)
        return X, y, n

    def test_pandas_copy_on_write_outcome_is_read_only(self):
        # Guards the premise: if pandas stops returning read-only arrays this test still
        # passes, but the regression tests below would no longer be exercising the path.
        import pandas as pd

        df = pd.DataFrame({"y": np.arange(10.0)})
        arr = df.loc[:, "y"].to_numpy().squeeze()
        assert isinstance(arr, np.ndarray)

    def test_bart_accepts_read_only_outcome(self):
        from stochtree import BARTModel

        X, y, _ = self._data()
        model = BARTModel()
        model.sample(
            X_train=X, y_train=self._read_only(y),
            num_gfr=2, num_burnin=0, num_mcmc=3,
            general_params={"num_threads": 1},
        )
        assert model.y_hat_train.shape == (X.shape[0], 3)

    def test_bart_accepts_read_only_observation_weights(self):
        from stochtree import BARTModel

        X, y, n = self._data()
        model = BARTModel()
        model.sample(
            X_train=X, y_train=y,
            observation_weights_train=self._read_only(np.ones(n)),
            num_gfr=2, num_burnin=0, num_mcmc=3,
            general_params={"num_threads": 1},
        )
        assert model.y_hat_train.shape == (n, 3)

    def test_bcf_accepts_read_only_outcome(self):
        from stochtree import BCFModel

        X, y, n = self._data(2)
        rng = np.random.default_rng(3)
        Z = rng.binomial(1, 0.5, n).astype(float)
        pi = np.full(n, 0.5)
        model = BCFModel()
        model.sample(
            X_train=X, Z_train=Z, y_train=self._read_only(y), propensity_train=pi,
            num_gfr=2, num_burnin=0, num_mcmc=3,
            general_params={"num_threads": 1},
        )
        assert model.y_hat_train.shape == (n, 3)

    def test_read_only_input_is_not_modified_and_writeable_is_not_copied(self):
        from stochtree.utils import _as_cpp_writeable

        arr = np.arange(5.0)
        assert _as_cpp_writeable(arr) is arr  # no needless copy in the common case
        ro = self._read_only(np.arange(5.0))
        out = _as_cpp_writeable(ro)
        assert out is not ro and out.flags.writeable
        np.testing.assert_array_equal(out, ro)
        assert _as_cpp_writeable(None) is None

    def test_dataset_accepts_read_only_covariates_basis_and_weights(self):
        rng = np.random.default_rng(4)
        n, p, k = 40, 3, 2
        X = rng.uniform(size=(n, p))
        B = rng.uniform(size=(n, k))
        w = rng.uniform(1, 2, n)

        writeable = Dataset()
        writeable.add_covariates(X)
        writeable.add_basis(B)
        writeable.add_variance_weights(w)

        read_only = Dataset()
        read_only.add_covariates(self._read_only(X))
        read_only.add_basis(self._read_only(B))
        read_only.add_variance_weights(self._read_only(w))

        np.testing.assert_array_equal(
            writeable.get_covariates(), read_only.get_covariates()
        )
        np.testing.assert_array_equal(writeable.get_basis(), read_only.get_basis())
        np.testing.assert_allclose(
            writeable.get_variance_weights(), read_only.get_variance_weights()
        )

    def test_dataset_update_basis_accepts_read_only(self):
        # The low-level custom sampler loop calls update_basis every iteration, so a
        # read-only basis must be accepted there and must land with the right values.
        rng = np.random.default_rng(5)
        n, p, k = 40, 3, 2
        dataset = Dataset()
        dataset.add_covariates(rng.uniform(size=(n, p)))
        dataset.add_basis(rng.uniform(size=(n, k)))

        new_basis = rng.uniform(size=(n, k))
        dataset.update_basis(self._read_only(new_basis))
        np.testing.assert_allclose(dataset.get_basis(), new_basis)

    def test_dataset_update_variance_weights_accepts_read_only(self):
        rng = np.random.default_rng(6)
        n, p = 40, 3
        dataset = Dataset()
        dataset.add_covariates(rng.uniform(size=(n, p)))
        dataset.add_variance_weights(np.ones(n))

        new_weights = rng.uniform(1, 2, n)
        dataset.update_variance_weights(self._read_only(new_weights))
        np.testing.assert_allclose(
            np.squeeze(dataset.get_variance_weights()), new_weights
        )

    def test_residual_accepts_read_only_inputs(self):
        from stochtree import Residual

        rng = np.random.default_rng(7)
        n = 40
        y = rng.normal(size=n)
        v = rng.normal(size=n)

        residual = Residual(self._read_only(y))
        residual.add_vector(self._read_only(v))
        residual.subtract_vector(self._read_only(v / 2))
        np.testing.assert_allclose(np.squeeze(residual.get_residual()), y + v - v / 2)

        replacement = rng.normal(size=n)
        residual.update_data(self._read_only(replacement))
        np.testing.assert_allclose(np.squeeze(residual.get_residual()), replacement)

    def test_rfx_dataset_accepts_read_only_inputs(self):
        rng = np.random.default_rng(8)
        n, k = 50, 2
        labels = rng.integers(0, 3, n).astype(np.int32)
        basis = rng.uniform(size=(n, k))
        weights = rng.uniform(1, 2, n)

        writeable = RandomEffectsDataset()
        writeable.add_group_labels(labels)
        writeable.add_basis(basis)
        writeable.add_variance_weights(weights)

        read_only = RandomEffectsDataset()
        read_only.add_group_labels(self._read_only(labels))
        read_only.add_basis(self._read_only(basis))
        read_only.add_variance_weights(self._read_only(weights))

        np.testing.assert_array_equal(
            writeable.get_group_labels(), read_only.get_group_labels()
        )
        # Integer group labels must not be silently coerced to float
        assert read_only.get_group_labels().dtype == writeable.get_group_labels().dtype
        np.testing.assert_array_equal(writeable.get_basis(), read_only.get_basis())
        np.testing.assert_allclose(
            writeable.get_variance_weights(), read_only.get_variance_weights()
        )

    def test_rfx_dataset_update_methods_accept_read_only(self):
        rng = np.random.default_rng(9)
        n, k = 50, 2
        labels = rng.integers(0, 3, n).astype(np.int32)
        dataset = RandomEffectsDataset()
        dataset.add_group_labels(labels)
        dataset.add_basis(rng.uniform(size=(n, k)))
        dataset.add_variance_weights(np.ones(n))

        # update_group_labels requires a basis first (NumObservations must be defined);
        # that ordering constraint is unrelated to writeability.
        new_labels = rng.integers(0, 3, n).astype(np.int32)
        dataset.update_group_labels(self._read_only(new_labels))
        np.testing.assert_array_equal(
            np.squeeze(dataset.get_group_labels()), new_labels
        )

        new_basis = rng.uniform(size=(n, k))
        dataset.update_basis(self._read_only(new_basis))
        np.testing.assert_allclose(dataset.get_basis(), new_basis)

        new_weights = rng.uniform(1, 2, n)
        dataset.update_variance_weights(self._read_only(new_weights))
        np.testing.assert_allclose(
            np.squeeze(dataset.get_variance_weights()), new_weights
        )

    def test_rfx_update_variance_weights_accepts_column_vector(self):
        # update_variance_weights sized off the squeezed array but passed the unsqueezed
        # one to C++; an (n, 1) input must now round-trip correctly.
        rng = np.random.default_rng(10)
        n, k = 40, 2
        dataset = RandomEffectsDataset()
        dataset.add_group_labels(rng.integers(0, 3, n).astype(np.int32))
        dataset.add_basis(rng.uniform(size=(n, k)))
        dataset.add_variance_weights(np.ones(n))

        new_weights = rng.uniform(1, 2, (n, 1))
        dataset.update_variance_weights(new_weights)
        np.testing.assert_allclose(
            np.squeeze(dataset.get_variance_weights()), np.squeeze(new_weights)
        )
