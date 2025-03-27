import numpy as np
import pandas as pd

from stochtree import (
    Dataset,
    Forest,
    ForestContainer,
    compute_forest_leaf_indices, 
    compute_forest_max_leaf_index
)


class TestKernel:
    def test_forest(self):
        # Create dataset
        X = np.array(
            [[1.5, 8.7, 1.2],
             [2.7, 3.4, 5.4],
             [3.6, 1.2, 9.3],
             [4.4, 5.4, 10.4],
             [5.3, 9.3, 3.6],
             [6.1, 10.4, 4.4]]
        )
        n, p = X.shape
        num_trees = 2
        output_dim = 1
        forest_dataset = Dataset()
        forest_dataset.add_covariates(X)
        forest_samples = ForestContainer(num_trees, output_dim, True, False)

        # Initialize a forest with constant root predictions
        forest_samples.add_sample(0.)

        # Split the root of the first tree in the ensemble at X[,1] > 4.0
        forest_samples.add_numeric_split(0, 0, 0, 0, 4.0, -5., 5.)

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
            [2]
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
            [3]
        ])
        
        # Assertion
        np.testing.assert_almost_equal(computed, expected)
        assert max_leaf_index == [3]
