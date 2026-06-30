import numpy as np
import pandas as pd

from stochtree import (
    BARTModel,
    Dataset,
    Forest,
    ForestContainer,
    compute_forest_leaf_indices,
    compute_forest_max_leaf_index,
)


class TestKernel:
    def test_forest(self):
        # Create dataset
        X = np.array([
            [1.5, 8.7, 1.2],
            [2.7, 3.4, 5.4],
            [3.6, 1.2, 9.3],
            [4.4, 5.4, 10.4],
            [5.3, 9.3, 3.6],
            [6.1, 10.4, 4.4],
        ])
        n, p = X.shape
        num_trees = 2
        output_dim = 1
        forest_dataset = Dataset()
        forest_dataset.add_covariates(X)
        forest_samples = ForestContainer(num_trees, output_dim, True, False)

        # Initialize a forest with constant root predictions
        forest_samples.add_sample(0.0)

        # Split the root of the first tree in the ensemble at X[,1] > 4.0
        forest_samples.add_numeric_split(0, 0, 0, 0, 4.0, -5.0, 5.0)

        # Check that regular and "raw" predictions are the same (since the leaf is constant)
        computed = compute_forest_leaf_indices(forest_samples, X)
        max_leaf_index = compute_forest_max_leaf_index(forest_samples)
        expected = np.array([
            [0],
            [0],
            [0],
            [1],
            [1],
            [1],
            [2],
            [2],
            [2],
            [2],
            [2],
            [2],
        ])

        # Assertion
        np.testing.assert_almost_equal(computed, expected)
        assert max_leaf_index == [2]

        # Split the left leaf of the first tree in the ensemble at X[,2] > 4.0
        forest_samples.add_numeric_split(0, 0, 1, 1, 4.0, -7.5, -2.5)

        # Check that regular and "raw" predictions are the same (since the leaf is constant)
        computed = compute_forest_leaf_indices(forest_samples, X)
        max_leaf_index = compute_forest_max_leaf_index(forest_samples)
        expected = np.array([
            [2],
            [1],
            [1],
            [0],
            [0],
            [0],
            [3],
            [3],
            [3],
            [3],
            [3],
            [3],
        ])

        # Assertion
        np.testing.assert_almost_equal(computed, expected)
        assert max_leaf_index == [3]

    def test_bart_model(self):
        # Regression: the kernel functions on a BART *model* must reach the forests via
        # extract_forest(), not the removed direct-forest properties (which now raise).
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, (100, 3))
        y = X[:, 0] * 2 + rng.normal(0, 0.5, 100)
        model = BARTModel()
        model.sample(
            X_train=X,
            y_train=y,
            num_gfr=0,
            num_burnin=0,
            num_mcmc=5,
            general_params={"random_seed": 42},
        )

        leaf_indices = compute_forest_leaf_indices(model, X, forest_type="mean")
        max_leaf_index = compute_forest_max_leaf_index(model, forest_type="mean")

        assert leaf_indices.size > 0
        assert np.all(leaf_indices >= 0)
        assert np.all(max_leaf_index >= 0)
        assert leaf_indices.max() <= np.max(max_leaf_index)
