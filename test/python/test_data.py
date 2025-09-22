import numpy as np

from stochtree import Dataset

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
