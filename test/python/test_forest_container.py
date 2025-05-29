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

        # Split the root of the first tree in the ensemble at X[,1] > 4.0
        # and then split the left leaf of the first tree in the ensemble at X[,2] > 4.0
        forest_samples.add_numeric_split(0, 0, 0, 0, 4.0, -5., 5.)
        forest_samples.add_numeric_split(0, 0, 1, 1, 4.0, -7.5, -2.5)

        # Store the predictions of the "original" forest before modifications
        pred_orig = forest_samples.predict(forest_dataset)

        # Multiply first forest by 2.0
        forest_samples.multiply_forest(0, 2.0)
        
        # Check that predictions are all double
        pred = forest_samples.predict(forest_dataset)
        pred_expected = pred_orig * 2.0
        
        # Assertion
        np.testing.assert_almost_equal(pred, pred_expected)

        # Add 1.0 to every tree in first forest
        forest_samples.add_to_forest(0, 1.0)
        
        # Check that predictions are += num_trees
        pred_expected = pred + num_trees
        pred = forest_samples.predict(forest_dataset)
        
        # Assertion
        np.testing.assert_almost_equal(pred, pred_expected)

        # Initialize a new forest with constant root predictions
        forest_samples.add_sample(0.)

        # Split the second forest as the first forest was split
        forest_samples.add_numeric_split(1, 0, 0, 0, 4.0, -5., 5.)
        forest_samples.add_numeric_split(1, 0, 1, 1, 4.0, -7.5, -2.5)
        
        # Check that predictions are as expected
        pred_expected_new = np.c_[pred_expected, pred_orig]
        pred = forest_samples.predict(forest_dataset)
        
        # Assertion
        np.testing.assert_almost_equal(pred, pred_expected_new)

        # Combine second forest with the first forest
        forest_samples.combine_forests(np.array([0,1]))

        # Check that predictions are as expected
        pred_expected_new = np.c_[pred_expected + pred_orig, pred_orig]
        pred = forest_samples.predict(forest_dataset)
        
        # Assertion
        np.testing.assert_almost_equal(pred, pred_expected_new)
