import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import norm

from stochtree import Dataset, ForestContainer, BARTModel, BCFModel


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
    
    def test_bart_prediction(self):
        # Generate data and test/train split
        rng = np.random.default_rng(1234)
        n = 100
        p = 5
        X = rng.uniform(size=(n, p))
        f_XW = np.where((0 <= X[:, 0]) & (X[:, 0] < 0.25), -7.5,
               np.where((0.25 <= X[:, 0]) & (X[:, 0] < 0.5), -2.5,
               np.where((0.5 <= X[:, 0]) & (X[:, 0] < 0.75), 2.5, 7.5)))
        noise_sd = 1
        y = f_XW + rng.normal(0, noise_sd, size=n)
        test_set_pct = 0.2
        train_inds, test_inds = train_test_split(np.arange(n), test_size=test_set_pct, random_state=1234)
        X_train = X[train_inds, :]
        X_test = X[test_inds, :]
        y_train = y[train_inds]
        y_test = y[test_inds]

        # Fit a "classic" BART model
        bart_model = BARTModel()
        bart_model.sample(X_train = X_train, y_train = y_train, num_gfr=10, num_burnin=0, num_mcmc=10)

        # Check that the default predict method returns a dictionary
        pred = bart_model.predict(covariates=X_test)
        y_hat_posterior_test = pred['y_hat']
        assert y_hat_posterior_test.shape == (20, 10)

        # Check that the pre-aggregated predictions match with those computed by np.mean
        pred_mean = bart_model.predict(covariates=X_test, type="mean")
        y_hat_mean_test = pred_mean['y_hat']
        np.testing.assert_almost_equal(y_hat_mean_test, np.mean(y_hat_posterior_test, axis=1))

        # Fit a heteroskedastic BART model
        var_params = {'num_trees': 20}
        het_bart_model = BARTModel()
        het_bart_model.sample(
            X_train = X_train, y_train = y_train, 
            num_gfr=10, num_burnin=0, num_mcmc=10, 
            variance_forest_params=var_params
        )

        # Check that the default predict method returns a dictionary
        pred = het_bart_model.predict(covariates=X_test)
        y_hat_posterior_test = pred['y_hat']
        sigma2_hat_posterior_test = pred['variance_forest_predictions']
        assert y_hat_posterior_test.shape == (20, 10)
        assert sigma2_hat_posterior_test.shape == (20, 10)

        # Check that the pre-aggregated predictions match with those computed by np.mean
        pred_mean = het_bart_model.predict(covariates=X_test, type="mean")
        y_hat_mean_test = pred_mean['y_hat']
        sigma2_hat_mean_test = pred_mean['variance_forest_predictions']
        np.testing.assert_almost_equal(y_hat_mean_test, np.mean(y_hat_posterior_test, axis=1))
        np.testing.assert_almost_equal(sigma2_hat_mean_test, np.mean(sigma2_hat_posterior_test, axis=1))

        # Check that the "single-term" pre-aggregated predictions
        # match those computed by pre-aggregated predictions returned in a dictionary
        y_hat_mean_test_single_term = het_bart_model.predict(covariates=X_test, type="mean", terms="y_hat")
        sigma2_hat_mean_test_single_term = het_bart_model.predict(covariates=X_test, type="mean", terms="variance_forest")
        np.testing.assert_almost_equal(y_hat_mean_test, y_hat_mean_test_single_term)
        np.testing.assert_almost_equal(sigma2_hat_mean_test, sigma2_hat_mean_test_single_term)
    
    def test_bcf_prediction(self):
        # Generate data and test/train split
        rng = np.random.default_rng(1234)
        

        # Convert the R code down below to Python
        rng = np.random.default_rng(1234)
        n = 100
        g = lambda x: np.where(x[:, 4] == 1, 2, np.where(x[:, 4] == 2, -1, -4))
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        x3 = rng.normal(size=n)
        x4 = rng.binomial(n=1,p=0.5,size=(n,))
        x5 = rng.choice(a=[0,1,2], size=(n,), replace=True)
        x4_cat = pd.Categorical(x4, categories=[0,1], ordered=True)
        x5_cat = pd.Categorical(x4, categories=[0,1,2], ordered=True)
        p = 5
        X = pd.DataFrame(data={
            "x1": pd.Series(x1),
            "x2": pd.Series(x2),
            "x3": pd.Series(x3),
            "x4": pd.Series(x4_cat),
            "x5": pd.Series(x5_cat)
        })
        def g(x5):
            return np.where(
                x5 == 0, 2.0, 
                np.where(
                    x5 == 1, -1.0, -4.0
                )
            )
        p = X.shape[1]
        mu_x = 1.0 + g(x5) + x1*x3
        tau_x = 1.0 + 2*x2*x4
        pi_x = (
            0.8 * norm.cdf(3.0 * mu_x / np.squeeze(np.std(mu_x)) - 0.5 * x1) + 
            0.05 + rng.uniform(low=0., high=0.1, size=(n,))
        )
        Z = rng.binomial(n=1, p=pi_x, size=(n,))
        E_XZ = mu_x + tau_x*Z
        snr = 2
        y = E_XZ + rng.normal(loc=0., scale=np.std(E_XZ) / snr, size=(n,))
        test_set_pct = 0.2
        train_inds, test_inds = train_test_split(np.arange(n), test_size=test_set_pct, random_state=1234)
        X_train = X.iloc[train_inds,:]
        X_test = X.iloc[test_inds,:]
        Z_train = Z[train_inds]
        Z_test = Z[test_inds]
        pi_x_train = pi_x[train_inds]
        pi_x_test = pi_x[test_inds]
        y_train = y[train_inds]
        y_test = y[test_inds]
        
        # Fit a "classic" BCF model
        bcf_model = BCFModel()
        bcf_model.sample(
            X_train = X_train,
            Z_train = Z_train,
            y_train = y_train,
            pi_train = pi_x_train,
            X_test = X_test,
            Z_test = Z_test,
            pi_test = pi_x_test,
            num_gfr = 10,
            num_burnin = 0,
            num_mcmc = 10
        )

        # Check that the default predict method returns a list
        pred = bcf_model.predict(X=X_test, Z=Z_test, propensity=pi_x_test)
        y_hat_posterior_test = pred['y_hat']
        assert y_hat_posterior_test.shape == (20, 10)

        # Check that the pre-aggregated predictions match with those computed by np.mean
        pred_mean = bcf_model.predict(X=X_test, Z=Z_test, propensity=pi_x_test, type="mean")
        y_hat_mean_test = pred_mean['y_hat']
        np.testing.assert_almost_equal(y_hat_mean_test, np.mean(y_hat_posterior_test, axis=1))

        # Check that we warn and return None when requesting terms that weren't fit
        with pytest.warns(UserWarning):
            pred_mean = bcf_model.predict(
                X=X_test, Z=Z_test, propensity=pi_x_test, 
                type="mean", terms=["rfx", "variance_forest"]
            )
        
        # Fit a heteroskedastic BCF model
        var_params = {'num_trees': 20}
        het_bcf_model = BCFModel()
        het_bcf_model.sample(
            X_train = X_train,
            Z_train = Z_train,
            y_train = y_train,
            pi_train = pi_x_train,
            X_test = X_test,
            Z_test = Z_test,
            pi_test = pi_x_test,
            num_gfr = 10,
            num_burnin = 0,
            num_mcmc = 10,
            variance_forest_params=var_params
        )

        # Check that the default predict method returns a dictionary
        pred = het_bcf_model.predict(X=X_test, Z=Z_test, propensity=pi_x_test)
        y_hat_posterior_test = pred['y_hat']
        sigma2_hat_posterior_test = pred['variance_forest_predictions']
        assert y_hat_posterior_test.shape == (20, 10)
        assert sigma2_hat_posterior_test.shape == (20, 10)

        # Check that the pre-aggregated predictions match with those computed by np.mean
        pred_mean = het_bcf_model.predict(X=X_test, Z=Z_test, propensity=pi_x_test, type="mean")
        y_hat_mean_test = pred_mean['y_hat']
        sigma2_hat_mean_test = pred_mean['variance_forest_predictions']
        np.testing.assert_almost_equal(y_hat_mean_test, np.mean(y_hat_posterior_test, axis=1))
        np.testing.assert_almost_equal(sigma2_hat_mean_test, np.mean(sigma2_hat_posterior_test, axis=1))

        # Check that the "single-term" pre-aggregated predictions
        # match those computed by pre-aggregated predictions returned in a dictionary
        y_hat_mean_test_single_term = het_bcf_model.predict(
            X=X_test, Z=Z_test, propensity=pi_x_test, 
            type="mean", terms="y_hat"
        )
        sigma2_hat_mean_test_single_term = het_bcf_model.predict(
            X=X_test, Z=Z_test, propensity=pi_x_test, 
            type="mean", terms="variance_forest"
        )
        np.testing.assert_almost_equal(y_hat_mean_test, y_hat_mean_test_single_term)
        np.testing.assert_almost_equal(sigma2_hat_mean_test, sigma2_hat_mean_test_single_term)
        

