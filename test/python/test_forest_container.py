import numpy as np

from stochtree import Dataset, ForestContainer, BARTModel
from sklearn.model_selection import train_test_split


class TestPredict:
    def test_constant_leaf_forest_container(self):
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
    
    def test_collapse_forest_container(self):
        # RNG
        rng = np.random.default_rng()

        # Generate covariates and basis
        n = 100
        p_X = 10
        X = rng.uniform(0, 1, (n, p_X))

        # Define the outcome mean function
        def outcome_mean(X):
            return np.where(
                (X[:, 0] >= 0.0) & (X[:, 0] < 0.25),
                -7.5,
                np.where(
                    (X[:, 0] >= 0.25) & (X[:, 0] < 0.5),
                    -2.5,
                    np.where((X[:, 0] >= 0.5) & (X[:, 0] < 0.75), 2.5, 7.5),
                ),
            )

        # Generate outcome
        epsilon = rng.normal(0, 1, n)
        y = outcome_mean(X) + epsilon

        # Test-train split
        sample_inds = np.arange(n)
        train_inds, test_inds = train_test_split(sample_inds, test_size=0.5)
        X_train = X[train_inds, :]
        X_test = X[test_inds, :]
        y_train = y[train_inds]
        # y_test = y[test_inds]
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        # Create forest dataset
        forest_dataset_test = Dataset()
        forest_dataset_test.add_covariates(X_test)

        # Run BART with 50 MCMC
        num_mcmc = 50
        bart_model = BARTModel()
        bart_model.sample(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            num_gfr=0,
            num_burnin=0,
            num_mcmc=num_mcmc,
        )

        # Extract the mean forest container
        mean_forest_container = bart_model.forest_container_mean

        # Predict from the original container
        pred_orig = mean_forest_container.predict(forest_dataset_test)

        # Collapse the container in batches of 5
        batch_size = 5
        mean_forest_container.collapse(batch_size)

        # Predict from the modified container
        pred_new = mean_forest_container.predict(forest_dataset_test)

        # Check that corresponding (sums of) predictions match
        container_inds = np.linspace(start=1, stop=num_mcmc, num=num_mcmc)
        batch_inds = (container_inds - (num_mcmc - ((num_mcmc // (num_mcmc // batch_size)) * (num_mcmc // batch_size))) - 1) // batch_size
        batch_inds = batch_inds.astype(int)
        num_batches = np.max(batch_inds) + 1
        pred_orig_collapsed = np.empty((n_test, num_batches))
        for i in range(num_batches):
            pred_orig_collapsed[:,i] = np.sum(pred_orig[:,batch_inds == i], axis=1) / np.sum(batch_inds == i)
        
        # Assertion
        np.testing.assert_almost_equal(pred_orig_collapsed, pred_new)

        # Run BART with 52 MCMC
        num_mcmc = 52
        bart_model = BARTModel()
        bart_model.sample(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            num_gfr=0,
            num_burnin=0,
            num_mcmc=num_mcmc,
        )

        # Extract the mean forest container
        mean_forest_container = bart_model.forest_container_mean

        # Predict from the original container
        pred_orig = mean_forest_container.predict(forest_dataset_test)

        # Collapse the container in batches of 5
        batch_size = 5
        mean_forest_container.collapse(batch_size)

        # Predict from the modified container
        pred_new = mean_forest_container.predict(forest_dataset_test)

        # Check that corresponding (sums of) predictions match
        container_inds = np.linspace(start=1, stop=num_mcmc, num=num_mcmc)
        batch_inds = (container_inds - (num_mcmc - ((num_mcmc // (num_mcmc // batch_size)) * (num_mcmc // batch_size))) - 1) // batch_size
        batch_inds = batch_inds.astype(int)
        num_batches = np.max(batch_inds) + 1
        pred_orig_collapsed = np.empty((n_test, num_batches))
        for i in range(num_batches):
            pred_orig_collapsed[:,i] = np.sum(pred_orig[:,batch_inds == i], axis=1) / np.sum(batch_inds == i)
        
        # Assertion
        np.testing.assert_almost_equal(pred_orig_collapsed, pred_new)

        # Run BART with 5 MCMC
        num_mcmc = 5
        bart_model = BARTModel()
        bart_model.sample(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            num_gfr=0,
            num_burnin=0,
            num_mcmc=num_mcmc,
        )

        # Extract the mean forest container
        mean_forest_container = bart_model.forest_container_mean

        # Predict from the original container
        pred_orig = mean_forest_container.predict(forest_dataset_test)

        # Collapse the container in batches of 5
        batch_size = 5
        mean_forest_container.collapse(batch_size)

        # Predict from the modified container
        pred_new = mean_forest_container.predict(forest_dataset_test)

        # Check that corresponding (sums of) predictions match
        num_batches = 1
        pred_orig_collapsed = np.empty((n_test, num_batches))
        pred_orig_collapsed[:,0] = np.sum(pred_orig, axis=1) / batch_size
        
        # Assertion
        np.testing.assert_almost_equal(pred_orig_collapsed, pred_new)

        # Run BART with 4 MCMC
        num_mcmc = 4
        bart_model = BARTModel()
        bart_model.sample(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            num_gfr=0,
            num_burnin=0,
            num_mcmc=num_mcmc,
        )

        # Extract the mean forest container
        mean_forest_container = bart_model.forest_container_mean

        # Predict from the original container
        pred_orig = mean_forest_container.predict(forest_dataset_test)

        # Collapse the container in batches of 5
        batch_size = 5
        mean_forest_container.collapse(batch_size)

        # Predict from the modified container
        pred_new = mean_forest_container.predict(forest_dataset_test)

        # Check that corresponding (sums of) predictions match
        pred_orig_collapsed = pred_orig
        
        # Assertion
        np.testing.assert_almost_equal(pred_orig_collapsed, pred_new)
