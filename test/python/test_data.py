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
        np.testing.assert_array_equal(forest_dataset.get_variance_weights(), new_variance_weights)

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
        np.testing.assert_array_equal(rfx_dataset.get_variance_weights(), new_variance_weights)

