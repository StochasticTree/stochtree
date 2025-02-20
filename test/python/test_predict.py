import numpy as np

from stochtree import Dataset, ForestContainer


class TestPredict:
    def test_constant_leaf_prediction(self):
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
        num_trees = 10
        output_dim = 1
        forest_dataset = Dataset()
        forest_dataset.add_covariates(X)
        forest_samples = ForestContainer(num_trees, output_dim, True, False)

        # Initialize a forest with constant root predictions
        forest_samples.add_sample(0.)

        # Check that regular and "raw" predictions are the same (since the leaf is constant)
        pred = forest_samples.predict(forest_dataset)
        pred_raw = forest_samples.predict_raw(forest_dataset)

        # Assertion
        np.testing.assert_almost_equal(pred, pred_raw)

        # Split the root of the first tree in the ensemble at X[,1] > 4.0
        forest_samples.add_numeric_split(0, 0, 0, 0, 4.0, -5., 5.)

        # Check that regular and "raw" predictions are the same (since the leaf is constant)
        pred = forest_samples.predict(forest_dataset)
        pred_raw = forest_samples.predict_raw(forest_dataset)
        
        # Assertion
        np.testing.assert_almost_equal(pred, pred_raw)

        # Split the left leaf of the first tree in the ensemble at X[,2] > 4.0
        forest_samples.add_numeric_split(0, 0, 1, 1, 4.0, -7.5, -2.5)
        
        # Check that regular and "raw" predictions are the same (since the leaf is constant)
        pred = forest_samples.predict(forest_dataset)
        pred_raw = forest_samples.predict_raw(forest_dataset)
        
        # Assertion
        np.testing.assert_almost_equal(pred, pred_raw)
        
        # Check the split count for the first tree in the ensemble
        split_counts = forest_samples.get_tree_split_counts(0, 0, p)
        split_counts_expected = np.array([1,1,0])
        
        # Assertion
        np.testing.assert_almost_equal(split_counts, split_counts_expected)
    
    def test_univariate_regression_leaf_prediction(self):
        # Create dataset
        X = np.array(
            [[1.5, 8.7, 1.2],
             [2.7, 3.4, 5.4],
             [3.6, 1.2, 9.3],
             [4.4, 5.4, 10.4],
             [5.3, 9.3, 3.6],
             [6.1, 10.4, 4.4]]
        )
        W = np.array(
            [[-1],
             [-1],
             [-1],
             [1],
             [1],
             [1]]
        )
        n, p = X.shape
        num_trees = 10
        output_dim = 1
        forest_dataset = Dataset()
        forest_dataset.add_covariates(X)
        forest_dataset.add_basis(W)
        forest_samples = ForestContainer(num_trees, output_dim, False, False)

        # Initialize a forest with constant root predictions
        forest_samples.add_sample(0.)

        # Check that regular and "raw" predictions are the same (since the leaf is constant)
        pred = forest_samples.predict(forest_dataset)
        pred_raw = forest_samples.predict_raw(forest_dataset)

        # Assertion
        np.testing.assert_almost_equal(pred, pred_raw)

        # Split the root of the first tree in the ensemble at X[,1] > 4.0
        forest_samples.add_numeric_split(0, 0, 0, 0, 4.0, -5., 5.)

        # Check that regular and "raw" predictions are the same (since the leaf is constant)
        pred = forest_samples.predict(forest_dataset)
        pred_raw = forest_samples.predict_raw(forest_dataset)
        pred_manual = pred_raw*W
        
        # Assertion
        np.testing.assert_almost_equal(pred, pred_manual)

        # Split the left leaf of the first tree in the ensemble at X[,2] > 4.0
        forest_samples.add_numeric_split(0, 0, 1, 1, 4.0, -7.5, -2.5)
        
        # Check that regular and "raw" predictions are the same (since the leaf is constant)
        pred = forest_samples.predict(forest_dataset)
        pred_raw = forest_samples.predict_raw(forest_dataset)
        pred_manual = pred_raw*W
        
        # Assertion
        np.testing.assert_almost_equal(pred, pred_manual)
        
        # Check the split count for the first tree in the ensemble
        split_counts = forest_samples.get_tree_split_counts(0, 0, p)
        split_counts_expected = np.array([1, 1, 0])
        
        # Assertion
        np.testing.assert_almost_equal(split_counts, split_counts_expected)
    
    def test_multivariate_regression_leaf_prediction(self):
        # Create dataset
        X = np.array(
            [[1.5, 8.7, 1.2],
             [2.7, 3.4, 5.4],
             [3.6, 1.2, 9.3],
             [4.4, 5.4, 10.4],
             [5.3, 9.3, 3.6],
             [6.1, 10.4, 4.4]]
        )
        W = np.array(
            [[1,-1],
             [1,-1],
             [1,-1],
             [1, 1],
             [1, 1],
             [1, 1]]
        )
        n, p = X.shape
        num_trees = 10
        output_dim = 2
        num_samples = 0
        forest_dataset = Dataset()
        forest_dataset.add_covariates(X)
        forest_dataset.add_basis(W)
        forest_samples = ForestContainer(num_trees, output_dim, False, False)

        # Initialize a forest with constant root predictions
        forest_samples.add_sample(np.array([1.,1.]))
        num_samples += 1

        # Check that regular and "raw" predictions are the same (since the leaf is constant)
        pred = forest_samples.predict(forest_dataset)
        pred_raw = forest_samples.predict_raw(forest_dataset)
        pred_intermediate = pred_raw * W
        pred_manual = pred_intermediate.sum(axis=1, keepdims = True)

        # Assertion
        np.testing.assert_almost_equal(pred, pred_manual)

        # Split the root of the first tree in the ensemble at X[,1] > 4.0
        forest_samples.add_numeric_split(0, 0, 0, 0, 4.0, np.array([-5.,-1.]), np.array([5.,1.]))

        # Check that regular and "raw" predictions are the same (since the leaf is constant)
        pred = forest_samples.predict(forest_dataset)
        pred_raw = forest_samples.predict_raw(forest_dataset)
        pred_intermediate = pred_raw * W
        pred_manual = pred_intermediate.sum(axis=1, keepdims = True)
        
        # Assertion
        np.testing.assert_almost_equal(pred, pred_manual)

        # Split the left leaf of the first tree in the ensemble at X[,2] > 4.0
        forest_samples.add_numeric_split(0, 0, 1, 1, 4.0, np.array([-7.5,2.5]), np.array([-2.5,7.5]))
        
        # Check that regular and "raw" predictions are the same (since the leaf is constant)
        pred = forest_samples.predict(forest_dataset)
        pred_raw = forest_samples.predict_raw(forest_dataset)
        pred_intermediate = pred_raw * W
        pred_manual = pred_intermediate.sum(axis=1, keepdims = True)
        
        # Assertion
        np.testing.assert_almost_equal(pred, pred_manual)
        
        # Check the split count for the first tree in the ensemble
        split_counts = forest_samples.get_tree_split_counts(0, 0, p)
        split_counts_expected = np.array([1, 1, 0])
        
        # Assertion
        np.testing.assert_almost_equal(split_counts, split_counts_expected)
