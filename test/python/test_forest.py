import numpy as np

from stochtree import Dataset, Forest


class TestPredict:
    def test_constant_forest_construction(self):
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
        num_trees = 10
        output_dim = 1
        forest_dataset = Dataset()
        forest_dataset.add_covariates(X)
        forest = Forest(num_trees, output_dim, True, False)

        # Initialize a forest with constant root predictions
        forest.set_root_leaves(0.0)

        # Check that regular and "raw" predictions are the same (since the leaf is constant)
        pred = forest.predict(forest_dataset)
        pred_exp = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Assertion
        np.testing.assert_almost_equal(pred, pred_exp)

        # Split the root of the first tree in the ensemble at X[,1] > 4.0
        forest.add_numeric_split(0, 0, 0, 4.0, -5.0, 5.0)

        # Check that regular and "raw" predictions are the same (since the leaf is constant)
        pred = forest.predict(forest_dataset)
        pred_exp = np.array([-5.0, -5.0, -5.0, 5.0, 5.0, 5.0])

        # Assertion
        np.testing.assert_almost_equal(pred, pred_exp)

        # Split the left leaf of the first tree in the ensemble at X[,2] > 4.0
        forest.add_numeric_split(0, 1, 1, 4.0, -7.5, -2.5)

        # Check that regular and "raw" predictions are the same (since the leaf is constant)
        pred = forest.predict(forest_dataset)
        pred_exp = np.array([-2.5, -7.5, -7.5, 5.0, 5.0, 5.0])

        # Assertion
        np.testing.assert_almost_equal(pred, pred_exp)

    def test_constant_forest_merge_arithmetic(self):
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
        num_trees = 10
        output_dim = 1
        forest_dataset = Dataset()
        forest_dataset.add_covariates(X)

        # Create two forests
        forest1 = Forest(num_trees, output_dim, True, False)
        forest2 = Forest(num_trees, output_dim, True, False)

        # Initialize forests with constant root predictions
        forest1.set_root_leaves(0.0)
        forest2.set_root_leaves(0.0)

        # Check that predictions are as expected
        pred1 = forest1.predict(forest_dataset)
        pred2 = forest2.predict(forest_dataset)
        pred_exp1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        pred_exp2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Assertion
        np.testing.assert_almost_equal(pred1, pred_exp1)
        np.testing.assert_almost_equal(pred2, pred_exp2)

        # Split the root of the first tree in the first forest at X[,1] > 4.0
        forest1.add_numeric_split(0, 0, 0, 4.0, -5.0, 5.0)

        # Split the root of the first tree in the second forest at X[,1] > 3.0
        forest2.add_numeric_split(0, 0, 0, 3.0, -1.0, 1.0)

        # Check that predictions are as expected
        pred1 = forest1.predict(forest_dataset)
        pred2 = forest2.predict(forest_dataset)
        pred_exp1 = np.array([-5.0, -5.0, -5.0, 5.0, 5.0, 5.0])
        pred_exp2 = np.array([-1.0, -1.0, 1.0, 1.0, 1.0, 1.0])

        # Assertion
        np.testing.assert_almost_equal(pred1, pred_exp1)
        np.testing.assert_almost_equal(pred2, pred_exp2)

        # Split the left leaf of the first tree in the first forest at X[,2] > 4.0
        forest1.add_numeric_split(0, 1, 1, 4.0, -7.5, -2.5)

        # Split the left leaf of the first tree in the first forest at X[,2] > 4.0
        forest2.add_numeric_split(0, 1, 1, 4.0, -1.5, -0.5)

        # Check that predictions are as expected
        pred1 = forest1.predict(forest_dataset)
        pred2 = forest2.predict(forest_dataset)
        pred_exp1 = np.array([-2.5, -7.5, -7.5, 5.0, 5.0, 5.0])
        pred_exp2 = np.array([-0.5, -1.5, 1.0, 1.0, 1.0, 1.0])

        # Assertion
        np.testing.assert_almost_equal(pred1, pred_exp1)
        np.testing.assert_almost_equal(pred2, pred_exp2)

        # Merge forests
        forest1.merge_forest(forest2)

        # Check that predictions are as expected
        pred = forest1.predict(forest_dataset)
        pred_exp = np.array([-3.0, -9.0, -6.5, 6.0, 6.0, 6.0])

        # Assertion
        np.testing.assert_almost_equal(pred, pred_exp)

        # Add constant to every value of the combined forest
        forest1.add_constant(0.5)

        # Check that predictions are as expected
        pred = forest1.predict(forest_dataset)
        pred_exp = np.array([7.0, 1.0, 3.5, 16.0, 16.0, 16.0])

        # Assertion
        np.testing.assert_almost_equal(pred, pred_exp)

        # Check that "old" forest is still intact
        pred = forest2.predict(forest_dataset)
        pred_exp = np.array([-0.5, -1.5, 1.0, 1.0, 1.0, 1.0])

        # Assertion
        np.testing.assert_almost_equal(pred, pred_exp)

        # Subtract constant back off of every value of the combined forest
        forest1.add_constant(-0.5)

        # Check that predictions are as expected
        pred = forest1.predict(forest_dataset)
        pred_exp = np.array([-3.0, -9.0, -6.5, 6.0, 6.0, 6.0])

        # Assertion
        np.testing.assert_almost_equal(pred, pred_exp)

        # Multiply every value of the combined forest by a constant
        forest1.multiply_constant(2.0)

        # Check that predictions are as expected
        pred = forest1.predict(forest_dataset)
        pred_exp = np.array([-6.0, -18.0, -13.0, 12.0, 12.0, 12.0])

        # Assertion
        np.testing.assert_almost_equal(pred, pred_exp)
