import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import norm

from stochtree import Dataset, ForestContainer, BARTModel, BCFModel, OutcomeModel


class TestPredict:
    def test_constant_leaf_prediction(self):
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
        forest_samples = ForestContainer(num_trees, output_dim, True, False)

        # Initialize a forest with constant root predictions
        forest_samples.add_sample(0.0)

        # Check that regular and "raw" predictions are the same (since the leaf is constant)
        pred = forest_samples.predict(forest_dataset)
        pred_raw = forest_samples.predict_raw(forest_dataset)

        # Assertion
        np.testing.assert_almost_equal(pred, pred_raw)

        # Split the root of the first tree in the ensemble at X[,1] > 4.0
        forest_samples.add_numeric_split(0, 0, 0, 0, 4.0, -5.0, 5.0)

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
        split_counts_expected = np.array([1, 1, 0])

        # Assertion
        np.testing.assert_almost_equal(split_counts, split_counts_expected)

    def test_univariate_regression_leaf_prediction(self):
        # Create dataset
        X = np.array([
            [1.5, 8.7, 1.2],
            [2.7, 3.4, 5.4],
            [3.6, 1.2, 9.3],
            [4.4, 5.4, 10.4],
            [5.3, 9.3, 3.6],
            [6.1, 10.4, 4.4],
        ])
        W = np.array([[-1], [-1], [-1], [1], [1], [1]])
        n, p = X.shape
        num_trees = 10
        output_dim = 1
        forest_dataset = Dataset()
        forest_dataset.add_covariates(X)
        forest_dataset.add_basis(W)
        forest_samples = ForestContainer(num_trees, output_dim, False, False)

        # Initialize a forest with constant root predictions
        forest_samples.add_sample(0.0)

        # Check that regular and "raw" predictions are the same (since the leaf is constant)
        pred = forest_samples.predict(forest_dataset)
        pred_raw = forest_samples.predict_raw(forest_dataset)

        # Assertion
        np.testing.assert_almost_equal(pred, pred_raw)

        # Split the root of the first tree in the ensemble at X[,1] > 4.0
        forest_samples.add_numeric_split(0, 0, 0, 0, 4.0, -5.0, 5.0)

        # Check that regular and "raw" predictions are the same (since the leaf is constant)
        pred = forest_samples.predict(forest_dataset)
        pred_raw = forest_samples.predict_raw(forest_dataset)
        pred_manual = pred_raw * W

        # Assertion
        np.testing.assert_almost_equal(pred, pred_manual)

        # Split the left leaf of the first tree in the ensemble at X[,2] > 4.0
        forest_samples.add_numeric_split(0, 0, 1, 1, 4.0, -7.5, -2.5)

        # Check that regular and "raw" predictions are the same (since the leaf is constant)
        pred = forest_samples.predict(forest_dataset)
        pred_raw = forest_samples.predict_raw(forest_dataset)
        pred_manual = pred_raw * W

        # Assertion
        np.testing.assert_almost_equal(pred, pred_manual)

        # Check the split count for the first tree in the ensemble
        split_counts = forest_samples.get_tree_split_counts(0, 0, p)
        split_counts_expected = np.array([1, 1, 0])

        # Assertion
        np.testing.assert_almost_equal(split_counts, split_counts_expected)

    def test_multivariate_regression_leaf_prediction(self):
        # Create dataset
        X = np.array([
            [1.5, 8.7, 1.2],
            [2.7, 3.4, 5.4],
            [3.6, 1.2, 9.3],
            [4.4, 5.4, 10.4],
            [5.3, 9.3, 3.6],
            [6.1, 10.4, 4.4],
        ])
        W = np.array([[1, -1], [1, -1], [1, -1], [1, 1], [1, 1], [1, 1]])
        n, p = X.shape
        num_trees = 10
        output_dim = 2
        num_samples = 0
        forest_dataset = Dataset()
        forest_dataset.add_covariates(X)
        forest_dataset.add_basis(W)
        forest_samples = ForestContainer(num_trees, output_dim, False, False)

        # Initialize a forest with constant root predictions
        forest_samples.add_sample(np.array([1.0, 1.0]))
        num_samples += 1

        # Check that regular and "raw" predictions are the same (since the leaf is constant)
        pred = forest_samples.predict(forest_dataset)
        pred_raw = forest_samples.predict_raw(forest_dataset)
        pred_intermediate = pred_raw * W
        pred_manual = pred_intermediate.sum(axis=1, keepdims=True)

        # Assertion
        np.testing.assert_almost_equal(pred, pred_manual)

        # Split the root of the first tree in the ensemble at X[,1] > 4.0
        forest_samples.add_numeric_split(
            0, 0, 0, 0, 4.0, np.array([-5.0, -1.0]), np.array([5.0, 1.0])
        )

        # Check that regular and "raw" predictions are the same (since the leaf is constant)
        pred = forest_samples.predict(forest_dataset)
        pred_raw = forest_samples.predict_raw(forest_dataset)
        pred_intermediate = pred_raw * W
        pred_manual = pred_intermediate.sum(axis=1, keepdims=True)

        # Assertion
        np.testing.assert_almost_equal(pred, pred_manual)

        # Split the left leaf of the first tree in the ensemble at X[,2] > 4.0
        forest_samples.add_numeric_split(
            0, 0, 1, 1, 4.0, np.array([-7.5, 2.5]), np.array([-2.5, 7.5])
        )

        # Check that regular and "raw" predictions are the same (since the leaf is constant)
        pred = forest_samples.predict(forest_dataset)
        pred_raw = forest_samples.predict_raw(forest_dataset)
        pred_intermediate = pred_raw * W
        pred_manual = pred_intermediate.sum(axis=1, keepdims=True)

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
        f_XW = np.where(
            (0 <= X[:, 0]) & (X[:, 0] < 0.25),
            -7.5,
            np.where(
                (0.25 <= X[:, 0]) & (X[:, 0] < 0.5),
                -2.5,
                np.where((0.5 <= X[:, 0]) & (X[:, 0] < 0.75), 2.5, 7.5),
            ),
        )
        noise_sd = 1
        y = f_XW + rng.normal(0, noise_sd, size=n)
        test_set_pct = 0.2
        train_inds, test_inds = train_test_split(
            np.arange(n), test_size=test_set_pct, random_state=1234
        )
        X_train = X[train_inds, :]
        X_test = X[test_inds, :]
        y_train = y[train_inds]
        y_test = y[test_inds]

        # Fit a "classic" BART model
        bart_model = BARTModel()
        bart_model.sample(
            X_train=X_train, y_train=y_train, num_gfr=10, num_burnin=0, num_mcmc=10
        )

        # Check that the default predict method returns a dictionary
        pred = bart_model.predict(X=X_test)
        y_hat_posterior_test = pred["y_hat"]
        assert y_hat_posterior_test.shape == (20, 10)

        # Check that the pre-aggregated predictions match with those computed by np.mean
        pred_mean = bart_model.predict(X=X_test, type="mean")
        y_hat_mean_test = pred_mean["y_hat"]
        np.testing.assert_almost_equal(
            y_hat_mean_test, np.mean(y_hat_posterior_test, axis=1)
        )

        # Fit a heteroskedastic BART model
        with pytest.warns(UserWarning):
            var_params = {"num_trees": 20}
            het_bart_model = BARTModel()
            het_bart_model.sample(
                X_train=X_train,
                y_train=y_train,
                num_gfr=10,
                num_burnin=0,
                num_mcmc=10,
                variance_forest_params=var_params,
            )

        # Check that the default predict method returns a dictionary
        pred = het_bart_model.predict(X=X_test)
        y_hat_posterior_test = pred["y_hat"]
        sigma2_hat_posterior_test = pred["variance_forest_predictions"]
        assert y_hat_posterior_test.shape == (20, 10)
        assert sigma2_hat_posterior_test.shape == (20, 10)

        # Check that the pre-aggregated predictions match with those computed by np.mean
        pred_mean = het_bart_model.predict(X=X_test, type="mean")
        y_hat_mean_test = pred_mean["y_hat"]
        sigma2_hat_mean_test = pred_mean["variance_forest_predictions"]
        np.testing.assert_almost_equal(
            y_hat_mean_test, np.mean(y_hat_posterior_test, axis=1)
        )
        np.testing.assert_almost_equal(
            sigma2_hat_mean_test, np.mean(sigma2_hat_posterior_test, axis=1)
        )

        # Check that the "single-term" pre-aggregated predictions
        # match those computed by pre-aggregated predictions returned in a dictionary
        y_hat_mean_test_single_term = het_bart_model.predict(
            X=X_test, type="mean", terms="y_hat"
        )
        sigma2_hat_mean_test_single_term = het_bart_model.predict(
            X=X_test, type="mean", terms="variance_forest"
        )
        np.testing.assert_almost_equal(y_hat_mean_test, y_hat_mean_test_single_term)
        np.testing.assert_almost_equal(
            sigma2_hat_mean_test, sigma2_hat_mean_test_single_term
        )

        # Generate data with random effects
        rfx_group_ids = rng.choice(3, size=n)
        rfx_basis = np.ones((n, 1))
        rfx_coefs = np.array([-2.0, 0.0, 2.0])
        rfx_term = rfx_coefs[rfx_group_ids]
        noise_sd = 1
        y = f_XW + rfx_term + rng.normal(0, noise_sd, size=n)
        test_set_pct = 0.2
        train_inds, test_inds = train_test_split(
            np.arange(n), test_size=test_set_pct, random_state=1234
        )
        X_train = X[train_inds, :]
        X_test = X[test_inds, :]
        rfx_group_ids_train = rfx_group_ids[train_inds]
        rfx_group_ids_test = rfx_group_ids[test_inds]
        rfx_basis_train = rfx_basis[train_inds,:]
        rfx_basis_test = rfx_basis[test_inds,:]
        y_train = y[train_inds]
        y_test = y[test_inds]

        # Fit a BART model with random intercepts
        rfx_params = {"model_spec": "intercept_only"}
        bart_model = BARTModel()
        bart_model.sample(
            X_train=X_train, y_train=y_train, rfx_group_ids_train=rfx_group_ids_train, random_effects_params=rfx_params, num_gfr=10, num_burnin=0, num_mcmc=10
        )

        # Check that the default predict method returns a dictionary
        pred = bart_model.predict(X=X_test, rfx_group_ids=rfx_group_ids_test)
        y_hat_posterior_test = pred["y_hat"]
        assert y_hat_posterior_test.shape == (20, 10)

        # Check that the pre-aggregated predictions match with those computed by np.mean
        pred_mean = bart_model.predict(X=X_test, rfx_group_ids=rfx_group_ids_test, type="mean")
        y_hat_mean_test = pred_mean["y_hat"]
        np.testing.assert_almost_equal(
            y_hat_mean_test, np.mean(y_hat_posterior_test, axis=1)
        )

    def test_bcf_prediction(self):
        # Generate data and test/train split
        rng = np.random.default_rng(1234)
        n = 100
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        x3 = rng.normal(size=n)
        x4 = rng.binomial(n=1, p=0.5, size=(n,))
        x5 = rng.choice(a=[0, 1, 2], size=(n,), replace=True)
        x4_cat = pd.Categorical(x4, categories=[0, 1], ordered=True)
        x5_cat = pd.Categorical(x4, categories=[0, 1, 2], ordered=True)
        p = 5
        X = pd.DataFrame(
            data={
                "x1": pd.Series(x1),
                "x2": pd.Series(x2),
                "x3": pd.Series(x3),
                "x4": pd.Series(x4_cat),
                "x5": pd.Series(x5_cat),
            }
        )

        def g(x5):
            return np.where(x5 == 0, 2.0, np.where(x5 == 1, -1.0, -4.0))

        p = X.shape[1]
        mu_x = 1.0 + g(x5) + x1 * x3
        tau_x = 1.0 + 2 * x2 * x4
        pi_x = (
            0.8 * norm.cdf(3.0 * mu_x / np.squeeze(np.std(mu_x)) - 0.5 * x1)
            + 0.05
            + rng.uniform(low=0.0, high=0.1, size=(n,))
        )
        Z = rng.binomial(n=1, p=pi_x, size=(n,))
        E_XZ = mu_x + tau_x * Z
        snr = 2
        y = E_XZ + rng.normal(loc=0.0, scale=np.std(E_XZ) / snr, size=(n,))
        test_set_pct = 0.2
        train_inds, test_inds = train_test_split(
            np.arange(n), test_size=test_set_pct, random_state=1234
        )
        X_train = X.iloc[train_inds, :]
        X_test = X.iloc[test_inds, :]
        Z_train = Z[train_inds]
        Z_test = Z[test_inds]
        pi_x_train = pi_x[train_inds]
        pi_x_test = pi_x[test_inds]
        y_train = y[train_inds]
        y_test = y[test_inds]

        # Fit a "classic" BCF model
        bcf_model = BCFModel()
        bcf_model.sample(
            X_train=X_train,
            Z_train=Z_train,
            y_train=y_train,
            propensity_train=pi_x_train,
            X_test=X_test,
            Z_test=Z_test,
            propensity_test=pi_x_test,
            num_gfr=10,
            num_burnin=0,
            num_mcmc=10,
        )

        # Check that the default predict method returns a dictionary
        pred = bcf_model.predict(X=X_test, Z=Z_test, propensity=pi_x_test)
        y_hat_posterior_test = pred["y_hat"]
        assert y_hat_posterior_test.shape == (20, 10)

        # Check that the pre-aggregated predictions match with those computed by np.mean
        pred_mean = bcf_model.predict(
            X=X_test, Z=Z_test, propensity=pi_x_test, type="mean"
        )
        y_hat_mean_test = pred_mean["y_hat"]
        np.testing.assert_almost_equal(
            y_hat_mean_test, np.mean(y_hat_posterior_test, axis=1)
        )

        # Check that we warn and return None when requesting terms that weren't fit
        with pytest.warns(UserWarning):
            pred_mean = bcf_model.predict(
                X=X_test,
                Z=Z_test,
                propensity=pi_x_test,
                type="mean",
                terms=["rfx", "variance_forest"],
            )

        # Fit a heteroskedastic BCF model
        with pytest.warns(UserWarning):
            var_params = {"num_trees": 20}
            het_bcf_model = BCFModel()
            het_bcf_model.sample(
                X_train=X_train,
                Z_train=Z_train,
                y_train=y_train,
                propensity_train=pi_x_train,
                X_test=X_test,
                Z_test=Z_test,
                propensity_test=pi_x_test,
                num_gfr=10,
                num_burnin=0,
                num_mcmc=10,
                variance_forest_params=var_params,
            )

        # Check that the default predict method returns a dictionary
        pred = het_bcf_model.predict(X=X_test, Z=Z_test, propensity=pi_x_test)
        y_hat_posterior_test = pred["y_hat"]
        sigma2_hat_posterior_test = pred["variance_forest_predictions"]
        assert y_hat_posterior_test.shape == (20, 10)
        assert sigma2_hat_posterior_test.shape == (20, 10)

        # Check that the pre-aggregated predictions match with those computed by np.mean
        pred_mean = het_bcf_model.predict(
            X=X_test, Z=Z_test, propensity=pi_x_test, type="mean"
        )
        y_hat_mean_test = pred_mean["y_hat"]
        sigma2_hat_mean_test = pred_mean["variance_forest_predictions"]
        np.testing.assert_almost_equal(
            y_hat_mean_test, np.mean(y_hat_posterior_test, axis=1)
        )
        np.testing.assert_almost_equal(
            sigma2_hat_mean_test, np.mean(sigma2_hat_posterior_test, axis=1)
        )

        # Check that the "single-term" pre-aggregated predictions
        # match those computed by pre-aggregated predictions returned in a dictionary
        y_hat_mean_test_single_term = het_bcf_model.predict(
            X=X_test, Z=Z_test, propensity=pi_x_test, type="mean", terms="y_hat"
        )
        sigma2_hat_mean_test_single_term = het_bcf_model.predict(
            X=X_test,
            Z=Z_test,
            propensity=pi_x_test,
            type="mean",
            terms="variance_forest",
        )
        np.testing.assert_almost_equal(y_hat_mean_test, y_hat_mean_test_single_term)
        np.testing.assert_almost_equal(
            sigma2_hat_mean_test, sigma2_hat_mean_test_single_term
        )
    
        # Generate data with random effects
        rfx_group_ids = rng.choice(3, size=n)
        rfx_basis = np.concatenate((np.ones((n, 1)), np.expand_dims(Z, 1)), axis=1)
        rfx_coefs = np.array([[-2.0, -0.5], [0.0, 0.0], [2.0, 0.5]])
        rfx_term = np.multiply(rfx_coefs[rfx_group_ids,:], rfx_basis).sum(axis=1)
        E_XZ = mu_x + tau_x * Z + rfx_term
        snr = 2
        y = E_XZ + rng.normal(loc=0.0, scale=np.std(E_XZ) / snr, size=(n,))
        test_set_pct = 0.2
        train_inds, test_inds = train_test_split(
            np.arange(n), test_size=test_set_pct, random_state=1234
        )
        X_train = X.iloc[train_inds, :]
        X_test = X.iloc[test_inds, :]
        Z_train = Z[train_inds]
        Z_test = Z[test_inds]
        pi_x_train = pi_x[train_inds]
        pi_x_test = pi_x[test_inds]
        rfx_group_ids_train = rfx_group_ids[train_inds]
        rfx_group_ids_test = rfx_group_ids[test_inds]
        rfx_basis_train = rfx_basis[train_inds,:]
        rfx_basis_test = rfx_basis[test_inds,:]
        y_train = y[train_inds]
        y_test = y[test_inds]

        # Fit a "classic" BCF model
        rfx_params = {"model_spec": "intercept_only"}
        bcf_model = BCFModel()
        bcf_model.sample(
            X_train=X_train,
            Z_train=Z_train,
            y_train=y_train,
            propensity_train=pi_x_train,
            rfx_group_ids_train=rfx_group_ids_train,
            X_test=X_test,
            Z_test=Z_test,
            propensity_test=pi_x_test,
            rfx_group_ids_test=rfx_group_ids_test,
            random_effects_params=rfx_params,
            num_gfr=10,
            num_burnin=0,
            num_mcmc=10,
        )

        # Check that the default predict method returns a dictionary
        pred = bcf_model.predict(X=X_test, Z=Z_test, propensity=pi_x_test, rfx_group_ids=rfx_group_ids_test)
        y_hat_posterior_test = pred["y_hat"]
        assert y_hat_posterior_test.shape == (20, 10)

        # Check that the pre-aggregated predictions match with those computed by np.mean
        pred_mean = bcf_model.predict(
            X=X_test, Z=Z_test, propensity=pi_x_test, rfx_group_ids=rfx_group_ids_test, type="mean"
        )
        y_hat_mean_test = pred_mean["y_hat"]
        np.testing.assert_almost_equal(
            y_hat_mean_test, np.mean(y_hat_posterior_test, axis=1)
        )

        # Check that we warn and return None when requesting terms that weren't fit
        with pytest.warns(UserWarning):
            pred_mean = bcf_model.predict(
                X=X_test,
                Z=Z_test,
                propensity=pi_x_test,
                rfx_group_ids=rfx_group_ids_test,
                type="mean",
                terms=["variance_forest"],
            )

    def test_bart_cloglog_binary_interval_and_contrast(self):
        # Generate binary cloglog data
        rng = np.random.default_rng(42)
        n = 100
        p = 3
        X = rng.uniform(size=(n, p))
        beta = np.full(p, 1 / np.sqrt(p))
        true_lambda = X @ beta
        prob_y1 = 1 - np.exp(-np.exp(true_lambda))
        y = rng.binomial(1, prob_y1).astype(float)

        # Train/test split
        train_inds, test_inds = train_test_split(
            np.arange(n), test_size=0.2, random_state=42
        )
        X_train = X[train_inds, :]
        X_test = X[test_inds, :]
        y_train = y[train_inds]
        n_test = X_test.shape[0]

        # Fit binary cloglog BART model
        bart_model = BARTModel()
        bart_model.sample(
            X_train=X_train,
            y_train=y_train,
            num_gfr=10,
            num_burnin=0,
            num_mcmc=10,
            general_params={
                "sample_sigma2_global": False,
                "outcome_model": OutcomeModel(outcome="binary", link="cloglog"),
            },
        )

        # Test posterior interval on linear scale
        interval_linear = bart_model.compute_posterior_interval(
            terms="mean_forest",
            level=0.95,
            scale="linear",
            X=X_test,
        )
        assert "lower" in interval_linear
        assert "upper" in interval_linear
        assert interval_linear["lower"].shape == (n_test,)
        assert interval_linear["upper"].shape == (n_test,)
        assert np.all(interval_linear["lower"] <= interval_linear["upper"])

        # Test posterior interval on probability scale
        interval_prob = bart_model.compute_posterior_interval(
            terms="mean_forest",
            level=0.95,
            scale="probability",
            X=X_test,
        )
        assert interval_prob["lower"].shape == (n_test,)
        assert np.all(interval_prob["lower"] >= 0)
        assert np.all(interval_prob["upper"] <= 1)
        assert np.all(interval_prob["lower"] <= interval_prob["upper"])

        # Test contrast on linear scale
        X0 = X_test
        X1 = X_test + 0.5
        contrast_linear = bart_model.compute_contrast(
            X_0=X0,
            X_1=X1,
            type="posterior",
            scale="linear",
        )
        assert contrast_linear.shape[0] == n_test

        # Test contrast on probability scale
        contrast_prob = bart_model.compute_contrast(
            X_0=X0,
            X_1=X1,
            type="posterior",
            scale="probability",
        )
        assert contrast_prob.shape[0] == n_test

        # Test contrast with type = "mean"
        contrast_mean_linear = bart_model.compute_contrast(
            X_0=X0,
            X_1=X1,
            type="mean",
            scale="linear",
        )
        assert contrast_mean_linear.shape == (n_test,)

        contrast_mean_prob = bart_model.compute_contrast(
            X_0=X0,
            X_1=X1,
            type="mean",
            scale="probability",
        )
        assert contrast_mean_prob.shape == (n_test,)

    def test_bart_cloglog_ordinal_interval_and_contrast(self):
        # Generate ordinal cloglog data (3 categories)
        rng = np.random.default_rng(42)
        n = 500
        p = 3
        n_categories = 3
        X = rng.uniform(size=(n, p))
        beta = np.full(p, 1 / np.sqrt(p))
        true_lambda = X @ beta
        gamma_true = np.array([-1.5, -0.5])

        # Compute class probabilities
        true_probs = np.zeros((n, n_categories))
        for j in range(n_categories):
            if j == 0:
                true_probs[:, j] = 1 - np.exp(-np.exp(gamma_true[j] + true_lambda))
            elif j == n_categories - 1:
                true_probs[:, j] = 1 - np.sum(true_probs[:, :j], axis=1)
            else:
                true_probs[:, j] = np.exp(-np.exp(gamma_true[j - 1] + true_lambda)) * \
                    (1 - np.exp(-np.exp(gamma_true[j] + true_lambda)))

        # Generate ordinal outcomes (1-indexed to match R convention)
        y = np.array([
            rng.choice(np.arange(1, n_categories + 1), p=true_probs[i, :])
            for i in range(n)
        ]).astype(float)

        # Train/test split
        train_inds, test_inds = train_test_split(
            np.arange(n), test_size=0.2, random_state=42
        )
        X_train = X[train_inds, :]
        X_test = X[test_inds, :]
        y_train = y[train_inds]
        n_test = X_test.shape[0]

        # Fit ordinal cloglog BART model
        bart_model = BARTModel()
        bart_model.sample(
            X_train=X_train,
            y_train=y_train,
            num_gfr=10,
            num_burnin=0,
            num_mcmc=10,
            general_params={
                "sample_sigma2_global": False,
                "outcome_model": OutcomeModel(outcome="ordinal", link="cloglog"),
            },
        )

        # Test posterior interval on linear scale
        interval_linear = bart_model.compute_posterior_interval(
            terms="mean_forest",
            level=0.95,
            scale="linear",
            X=X_test,
        )
        assert "lower" in interval_linear
        assert "upper" in interval_linear
        assert interval_linear["lower"].shape == (n_test,)
        assert np.all(interval_linear["lower"] <= interval_linear["upper"])

        # Test posterior interval on probability scale
        # For ordinal models, probability scale returns survival probabilities P(Y > k)
        # which are (n_test, n_categories - 1) matrices
        interval_prob = bart_model.compute_posterior_interval(
            terms="mean_forest",
            level=0.95,
            scale="probability",
            X=X_test,
        )
        assert interval_prob["lower"].shape == (n_test, n_categories - 1)
        assert interval_prob["upper"].shape == (n_test, n_categories - 1)
        assert np.all(interval_prob["lower"] >= 0)
        assert np.all(interval_prob["upper"] <= 1)
        assert np.all(interval_prob["lower"] <= interval_prob["upper"])

        # Test contrast on linear scale
        X0 = X_test
        X1 = X_test + 0.5
        contrast_linear = bart_model.compute_contrast(
            X_0=X0,
            X_1=X1,
            type="posterior",
            scale="linear",
        )
        assert contrast_linear.shape[0] == n_test

        # Test contrast on probability scale (ordinal returns 3D array)
        contrast_prob = bart_model.compute_contrast(
            X_0=X0,
            X_1=X1,
            type="posterior",
            scale="probability",
        )
        assert contrast_prob.shape[0] == n_test
        assert contrast_prob.shape[1] == n_categories - 1

        # Test contrast with type = "mean" on linear scale
        contrast_mean_linear = bart_model.compute_contrast(
            X_0=X0,
            X_1=X1,
            type="mean",
            scale="linear",
        )
        assert contrast_mean_linear.shape == (n_test,)

        # Test contrast with type = "mean" on probability scale
        # For ordinal, mean contrast should be (n_test, n_categories - 1)
        contrast_mean_prob = bart_model.compute_contrast(
            X_0=X0,
            X_1=X1,
            type="mean",
            scale="probability",
        )
        assert contrast_mean_prob.shape == (n_test, n_categories - 1)

    def test_bart_cloglog_binary_posterior_predictive(self):
        # Generate binary cloglog data
        rng = np.random.default_rng(42)
        n = 100
        p = 3
        X = rng.uniform(size=(n, p))
        beta = np.full(p, 1 / np.sqrt(p))
        true_lambda = X @ beta
        prob_y1 = 1 - np.exp(-np.exp(true_lambda))
        y = rng.binomial(1, prob_y1).astype(float)

        # Train/test split
        train_inds, test_inds = train_test_split(
            np.arange(n), test_size=0.2, random_state=42
        )
        X_train = X[train_inds, :]
        X_test = X[test_inds, :]
        y_train = y[train_inds]
        n_test = X_test.shape[0]
        num_mcmc = 10

        # Fit binary cloglog BART model
        bart_model = BARTModel()
        bart_model.sample(
            X_train=X_train,
            y_train=y_train,
            num_gfr=10,
            num_burnin=0,
            num_mcmc=num_mcmc,
            general_params={
                "sample_sigma2_global": False,
                "outcome_model": OutcomeModel(outcome="binary", link="cloglog"),
            },
        )

        # Test with multiple draws per sample
        ppd = bart_model.sample_posterior_predictive(
            X=X_test, num_draws_per_sample=3
        )
        assert ppd.shape == (n_test, num_mcmc, 3)
        assert set(np.unique(ppd)).issubset({0, 1})

        # Test with single draw per sample
        ppd1 = bart_model.sample_posterior_predictive(
            X=X_test, num_draws_per_sample=1
        )
        assert ppd1.shape == (n_test, num_mcmc)
        assert set(np.unique(ppd1)).issubset({0, 1})

    def test_bart_cloglog_ordinal_posterior_predictive(self):
        # Generate ordinal cloglog data (3 categories)
        rng = np.random.default_rng(42)
        n = 500
        p = 3
        n_categories = 3
        X = rng.uniform(size=(n, p))
        beta = np.full(p, 1 / np.sqrt(p))
        true_lambda = X @ beta
        gamma_true = np.array([-1.5, -0.5])

        # Compute class probabilities
        true_probs = np.zeros((n, n_categories))
        for j in range(n_categories):
            if j == 0:
                true_probs[:, j] = 1 - np.exp(-np.exp(gamma_true[j] + true_lambda))
            elif j == n_categories - 1:
                true_probs[:, j] = 1 - np.sum(true_probs[:, :j], axis=1)
            else:
                true_probs[:, j] = np.exp(-np.exp(gamma_true[j - 1] + true_lambda)) * \
                    (1 - np.exp(-np.exp(gamma_true[j] + true_lambda)))

        # Generate ordinal outcomes (1-indexed)
        y = np.array([
            rng.choice(np.arange(1, n_categories + 1), p=true_probs[i, :])
            for i in range(n)
        ]).astype(float)

        # Train/test split
        train_inds, test_inds = train_test_split(
            np.arange(n), test_size=0.2, random_state=42
        )
        X_train = X[train_inds, :]
        X_test = X[test_inds, :]
        y_train = y[train_inds]
        n_test = X_test.shape[0]
        num_mcmc = 10

        # Fit ordinal cloglog BART model
        bart_model = BARTModel()
        bart_model.sample(
            X_train=X_train,
            y_train=y_train,
            num_gfr=10,
            num_burnin=0,
            num_mcmc=num_mcmc,
            general_params={
                "sample_sigma2_global": False,
                "outcome_model": OutcomeModel(outcome="ordinal", link="cloglog"),
            },
        )

        # Test with multiple draws per sample
        ppd = bart_model.sample_posterior_predictive(
            X=X_test, num_draws_per_sample=3
        )
        assert ppd.shape == (n_test, num_mcmc, 3)
        assert set(np.unique(ppd)).issubset(set(range(1, n_categories + 1)))

        # Test with single draw per sample
        ppd1 = bart_model.sample_posterior_predictive(
            X=X_test, num_draws_per_sample=1
        )
        assert ppd1.shape == (n_test, num_mcmc)
        assert set(np.unique(ppd1)).issubset(set(range(1, n_categories + 1)))

    def test_bart_gaussian_interval_and_contrast(self):
        # Generate gaussian data
        rng = np.random.default_rng(42)
        n = 100
        p = 3
        X = rng.uniform(size=(n, p))
        beta = np.full(p, 1 / np.sqrt(p))
        mu = X @ beta
        y = rng.normal(mu, 1.0)

        # Train/test split
        train_inds, test_inds = train_test_split(
            np.arange(n), test_size=0.2, random_state=42
        )
        X_train = X[train_inds, :]
        X_test = X[test_inds, :]
        y_train = y[train_inds]
        n_test = X_test.shape[0]

        # Fit gaussian BART model (default link)
        bart_model = BARTModel()
        bart_model.sample(
            X_train=X_train,
            y_train=y_train,
            num_gfr=10,
            num_burnin=0,
            num_mcmc=10,
        )

        # Test posterior interval on linear scale (mean_forest term)
        interval_linear = bart_model.compute_posterior_interval(
            terms="mean_forest",
            level=0.95,
            scale="linear",
            X=X_test,
        )
        assert "lower" in interval_linear
        assert "upper" in interval_linear
        assert interval_linear["lower"].shape == (n_test,)
        assert interval_linear["upper"].shape == (n_test,)
        assert np.all(interval_linear["lower"] <= interval_linear["upper"])

        # Test posterior interval for y_hat term
        interval_yhat = bart_model.compute_posterior_interval(
            terms="y_hat",
            level=0.95,
            scale="linear",
            X=X_test,
        )
        assert interval_yhat["lower"].shape == (n_test,)
        assert np.all(interval_yhat["lower"] <= interval_yhat["upper"])

        # Test contrast on linear scale, type = "posterior"
        X0 = X_test
        X1 = X_test + 0.5
        contrast_posterior = bart_model.compute_contrast(
            X_0=X0,
            X_1=X1,
            type="posterior",
            scale="linear",
        )
        assert contrast_posterior.shape[0] == n_test

        # Test contrast with type = "mean"
        contrast_mean = bart_model.compute_contrast(
            X_0=X0,
            X_1=X1,
            type="mean",
            scale="linear",
        )
        assert contrast_mean.shape == (n_test,)

    def test_bart_probit_binary_interval_and_contrast(self):
        # Generate binary probit data
        rng = np.random.default_rng(42)
        n = 100
        p = 3
        X = rng.uniform(size=(n, p))
        beta = np.full(p, 1 / np.sqrt(p))
        mu = X @ beta
        prob_y1 = norm.cdf(mu)
        y = rng.binomial(1, prob_y1).astype(float)

        # Train/test split
        train_inds, test_inds = train_test_split(
            np.arange(n), test_size=0.2, random_state=42
        )
        X_train = X[train_inds, :]
        X_test = X[test_inds, :]
        y_train = y[train_inds]
        n_test = X_test.shape[0]

        # Fit binary probit BART model
        bart_model = BARTModel()
        bart_model.sample(
            X_train=X_train,
            y_train=y_train,
            num_gfr=10,
            num_burnin=0,
            num_mcmc=10,
            general_params={
                "sample_sigma2_global": False,
                "outcome_model": OutcomeModel(outcome="binary", link="probit"),
            },
        )

        # Test posterior interval on linear scale
        interval_linear = bart_model.compute_posterior_interval(
            terms="mean_forest",
            level=0.95,
            scale="linear",
            X=X_test,
        )
        assert "lower" in interval_linear
        assert "upper" in interval_linear
        assert interval_linear["lower"].shape == (n_test,)
        assert interval_linear["upper"].shape == (n_test,)
        assert np.all(interval_linear["lower"] <= interval_linear["upper"])

        # Test posterior interval on probability scale
        interval_prob = bart_model.compute_posterior_interval(
            terms="mean_forest",
            level=0.95,
            scale="probability",
            X=X_test,
        )
        assert interval_prob["lower"].shape == (n_test,)
        assert interval_prob["upper"].shape == (n_test,)
        assert np.all(interval_prob["lower"] >= 0)
        assert np.all(interval_prob["upper"] <= 1)
        assert np.all(interval_prob["lower"] <= interval_prob["upper"])

        # Test contrast on linear scale
        X0 = X_test
        X1 = X_test + 0.5
        contrast_linear = bart_model.compute_contrast(
            X_0=X0,
            X_1=X1,
            type="posterior",
            scale="linear",
        )
        assert contrast_linear.shape[0] == n_test

        # Test contrast on probability scale
        contrast_prob = bart_model.compute_contrast(
            X_0=X0,
            X_1=X1,
            type="posterior",
            scale="probability",
        )
        assert contrast_prob.shape[0] == n_test

        # Test contrast with type = "mean" on linear scale
        contrast_mean_linear = bart_model.compute_contrast(
            X_0=X0,
            X_1=X1,
            type="mean",
            scale="linear",
        )
        assert contrast_mean_linear.shape == (n_test,)

        # Test contrast with type = "mean" on probability scale
        contrast_mean_prob = bart_model.compute_contrast(
            X_0=X0,
            X_1=X1,
            type="mean",
            scale="probability",
        )
        assert contrast_mean_prob.shape == (n_test,)

    def test_bart_gaussian_posterior_predictive(self):
        # Generate gaussian data
        rng = np.random.default_rng(42)
        n = 100
        p = 3
        X = rng.uniform(size=(n, p))
        beta = np.full(p, 1 / np.sqrt(p))
        mu = X @ beta
        y = rng.normal(mu, 1.0)

        # Train/test split
        train_inds, test_inds = train_test_split(
            np.arange(n), test_size=0.2, random_state=42
        )
        X_train = X[train_inds, :]
        X_test = X[test_inds, :]
        y_train = y[train_inds]
        n_test = X_test.shape[0]
        num_mcmc = 10

        # Fit gaussian BART model (default link)
        bart_model = BARTModel()
        bart_model.sample(
            X_train=X_train,
            y_train=y_train,
            num_gfr=10,
            num_burnin=0,
            num_mcmc=num_mcmc,
        )

        # Test with multiple draws per sample
        ppd = bart_model.sample_posterior_predictive(
            X=X_test, num_draws_per_sample=3
        )
        assert ppd.shape == (n_test, num_mcmc, 3)
        assert np.issubdtype(ppd.dtype, np.floating)

        # Test with single draw per sample
        ppd1 = bart_model.sample_posterior_predictive(
            X=X_test, num_draws_per_sample=1
        )
        assert ppd1.shape == (n_test, num_mcmc)
        assert np.issubdtype(ppd1.dtype, np.floating)

    def test_bart_probit_binary_posterior_predictive(self):
        # Generate binary probit data
        rng = np.random.default_rng(42)
        n = 100
        p = 3
        X = rng.uniform(size=(n, p))
        beta = np.full(p, 1 / np.sqrt(p))
        mu = X @ beta
        prob_y1 = norm.cdf(mu)
        y = rng.binomial(1, prob_y1).astype(float)

        # Train/test split
        train_inds, test_inds = train_test_split(
            np.arange(n), test_size=0.2, random_state=42
        )
        X_train = X[train_inds, :]
        X_test = X[test_inds, :]
        y_train = y[train_inds]
        n_test = X_test.shape[0]
        num_mcmc = 10

        # Fit binary probit BART model
        bart_model = BARTModel()
        bart_model.sample(
            X_train=X_train,
            y_train=y_train,
            num_gfr=10,
            num_burnin=0,
            num_mcmc=num_mcmc,
            general_params={
                "sample_sigma2_global": False,
                "outcome_model": OutcomeModel(outcome="binary", link="probit"),
            },
        )

        # Test with multiple draws per sample
        ppd = bart_model.sample_posterior_predictive(
            X=X_test, num_draws_per_sample=3
        )
        assert ppd.shape == (n_test, num_mcmc, 3)
        assert set(np.unique(ppd)).issubset({0, 1})

        # Test with single draw per sample
        ppd1 = bart_model.sample_posterior_predictive(
            X=X_test, num_draws_per_sample=1
        )
        assert ppd1.shape == (n_test, num_mcmc)
        assert set(np.unique(ppd1)).issubset({0, 1})

    def test_bcf_gaussian_interval_and_contrast(self):
        # Generate gaussian causal data
        rng = np.random.default_rng(42)
        n = 100
        p = 3
        X = rng.uniform(size=(n, p))
        pi_x = norm.cdf(X[:, 0])
        Z = rng.binomial(1, pi_x).astype(float)
        mu_x = X[:, 0] + X[:, 1]
        tau_x = np.ones(n)
        y = mu_x + tau_x * Z + rng.normal(size=n)

        # Train/test split
        train_inds, test_inds = train_test_split(
            np.arange(n), test_size=0.2, random_state=42
        )
        X_train = X[train_inds, :]
        X_test = X[test_inds, :]
        Z_train = Z[train_inds]
        Z_test = Z[test_inds]
        pi_train = pi_x[train_inds]
        pi_test = pi_x[test_inds]
        y_train = y[train_inds]
        n_test = X_test.shape[0]
        num_mcmc = 10

        # Fit gaussian BCF model
        bcf_model = BCFModel()
        bcf_model.sample(
            X_train=X_train,
            Z_train=Z_train,
            y_train=y_train,
            propensity_train=pi_train,
            X_test=X_test,
            Z_test=Z_test,
            propensity_test=pi_test,
            num_gfr=10,
            num_burnin=0,
            num_mcmc=num_mcmc,
        )

        # Test posterior interval on linear scale (prognostic_function term)
        interval_prog = bcf_model.compute_posterior_interval(
            terms="prognostic_function",
            level=0.95,
            scale="linear",
            X=X_test,
            Z=Z_test,
            propensity=pi_test,
        )
        assert "lower" in interval_prog
        assert "upper" in interval_prog
        assert interval_prog["lower"].shape == (n_test,)
        assert interval_prog["upper"].shape == (n_test,)
        assert np.all(interval_prog["lower"] <= interval_prog["upper"])

        # Test posterior interval for cate term
        interval_cate = bcf_model.compute_posterior_interval(
            terms="cate",
            level=0.95,
            scale="linear",
            X=X_test,
            Z=Z_test,
            propensity=pi_test,
        )
        assert interval_cate["lower"].shape == (n_test,)
        assert np.all(interval_cate["lower"] <= interval_cate["upper"])

        # Test posterior interval for y_hat term
        interval_yhat = bcf_model.compute_posterior_interval(
            terms="y_hat",
            level=0.95,
            scale="linear",
            X=X_test,
            Z=Z_test,
            propensity=pi_test,
        )
        assert interval_yhat["lower"].shape == (n_test,)
        assert np.all(interval_yhat["lower"] <= interval_yhat["upper"])

        # Test contrast on linear scale (CATE: Z=1 vs Z=0)
        contrast_posterior = bcf_model.compute_contrast(
            X_0=X_test,
            X_1=X_test,
            Z_0=np.zeros(n_test),
            Z_1=np.ones(n_test),
            propensity_0=pi_test,
            propensity_1=pi_test,
            type="posterior",
            scale="linear",
        )
        assert contrast_posterior.shape[0] == n_test
        assert contrast_posterior.shape[1] == num_mcmc

        # Test contrast with type = "mean"
        contrast_mean = bcf_model.compute_contrast(
            X_0=X_test,
            X_1=X_test,
            Z_0=np.zeros(n_test),
            Z_1=np.ones(n_test),
            propensity_0=pi_test,
            propensity_1=pi_test,
            type="mean",
            scale="linear",
        )
        assert contrast_mean.shape == (n_test,)

    def test_bcf_probit_binary_interval_and_contrast(self):
        # Generate binary probit causal data
        rng = np.random.default_rng(42)
        n = 100
        p = 3
        X = rng.uniform(size=(n, p))
        pi_x = norm.cdf(X[:, 0])
        Z = rng.binomial(1, pi_x).astype(float)
        mu_x = X[:, 0] + X[:, 1]
        tau_x = np.full(n, 0.5)
        prob_y1 = norm.cdf(mu_x + tau_x * Z)
        y = rng.binomial(1, prob_y1).astype(float)

        # Train/test split
        train_inds, test_inds = train_test_split(
            np.arange(n), test_size=0.2, random_state=42
        )
        X_train = X[train_inds, :]
        X_test = X[test_inds, :]
        Z_train = Z[train_inds]
        Z_test = Z[test_inds]
        pi_train = pi_x[train_inds]
        pi_test = pi_x[test_inds]
        y_train = y[train_inds]
        n_test = X_test.shape[0]
        num_mcmc = 10

        # Fit binary probit BCF model
        bcf_model = BCFModel()
        bcf_model.sample(
            X_train=X_train,
            Z_train=Z_train,
            y_train=y_train,
            propensity_train=pi_train,
            X_test=X_test,
            Z_test=Z_test,
            propensity_test=pi_test,
            num_gfr=10,
            num_burnin=0,
            num_mcmc=num_mcmc,
            general_params={
                "sample_sigma2_global": False,
                "outcome_model": OutcomeModel(outcome="binary", link="probit"),
            },
        )

        # Test posterior interval on linear scale
        interval_linear = bcf_model.compute_posterior_interval(
            terms="prognostic_function",
            level=0.95,
            scale="linear",
            X=X_test,
            Z=Z_test,
            propensity=pi_test,
        )
        assert "lower" in interval_linear
        assert "upper" in interval_linear
        assert interval_linear["lower"].shape == (n_test,)
        assert interval_linear["upper"].shape == (n_test,)
        assert np.all(interval_linear["lower"] <= interval_linear["upper"])

        # Test posterior interval on probability scale
        interval_prob = bcf_model.compute_posterior_interval(
            terms="prognostic_function",
            level=0.95,
            scale="probability",
            X=X_test,
            Z=Z_test,
            propensity=pi_test,
        )
        assert interval_prob["lower"].shape == (n_test,)
        assert interval_prob["upper"].shape == (n_test,)
        assert np.all(interval_prob["lower"] >= 0)
        assert np.all(interval_prob["upper"] <= 1)
        assert np.all(interval_prob["lower"] <= interval_prob["upper"])

        # Test contrast on linear scale (CATE: Z=1 vs Z=0)
        contrast_linear = bcf_model.compute_contrast(
            X_0=X_test,
            X_1=X_test,
            Z_0=np.zeros(n_test),
            Z_1=np.ones(n_test),
            propensity_0=pi_test,
            propensity_1=pi_test,
            type="posterior",
            scale="linear",
        )
        assert contrast_linear.shape[0] == n_test
        assert contrast_linear.shape[1] == num_mcmc

        # Test contrast on probability scale
        contrast_prob = bcf_model.compute_contrast(
            X_0=X_test,
            X_1=X_test,
            Z_0=np.zeros(n_test),
            Z_1=np.ones(n_test),
            propensity_0=pi_test,
            propensity_1=pi_test,
            type="posterior",
            scale="probability",
        )
        assert contrast_prob.shape[0] == n_test
        assert contrast_prob.shape[1] == num_mcmc

        # Test contrast with type = "mean" on linear scale
        contrast_mean_linear = bcf_model.compute_contrast(
            X_0=X_test,
            X_1=X_test,
            Z_0=np.zeros(n_test),
            Z_1=np.ones(n_test),
            propensity_0=pi_test,
            propensity_1=pi_test,
            type="mean",
            scale="linear",
        )
        assert contrast_mean_linear.shape == (n_test,)

        # Test contrast with type = "mean" on probability scale
        contrast_mean_prob = bcf_model.compute_contrast(
            X_0=X_test,
            X_1=X_test,
            Z_0=np.zeros(n_test),
            Z_1=np.ones(n_test),
            propensity_0=pi_test,
            propensity_1=pi_test,
            type="mean",
            scale="probability",
        )
        assert contrast_mean_prob.shape == (n_test,)

    def test_bcf_gaussian_posterior_predictive(self):
        # Generate gaussian causal data
        rng = np.random.default_rng(42)
        n = 100
        p = 3
        X = rng.uniform(size=(n, p))
        pi_x = norm.cdf(X[:, 0])
        Z = rng.binomial(1, pi_x).astype(float)
        mu_x = X[:, 0] + X[:, 1]
        tau_x = np.ones(n)
        y = mu_x + tau_x * Z + rng.normal(size=n)

        # Train/test split
        train_inds, test_inds = train_test_split(
            np.arange(n), test_size=0.2, random_state=42
        )
        X_train = X[train_inds, :]
        X_test = X[test_inds, :]
        Z_train = Z[train_inds]
        Z_test = Z[test_inds]
        pi_train = pi_x[train_inds]
        pi_test = pi_x[test_inds]
        y_train = y[train_inds]
        n_test = X_test.shape[0]
        num_mcmc = 10

        # Fit gaussian BCF model
        bcf_model = BCFModel()
        bcf_model.sample(
            X_train=X_train,
            Z_train=Z_train,
            y_train=y_train,
            propensity_train=pi_train,
            X_test=X_test,
            Z_test=Z_test,
            propensity_test=pi_test,
            num_gfr=10,
            num_burnin=0,
            num_mcmc=num_mcmc,
        )

        # Test with multiple draws per sample
        ppd = bcf_model.sample_posterior_predictive(
            X=X_test,
            Z=Z_test,
            propensity=pi_test,
            num_draws_per_sample=3,
        )
        assert ppd.shape == (n_test, num_mcmc, 3)
        assert np.issubdtype(ppd.dtype, np.floating)

        # Test with single draw per sample
        ppd1 = bcf_model.sample_posterior_predictive(
            X=X_test,
            Z=Z_test,
            propensity=pi_test,
            num_draws_per_sample=1,
        )
        assert ppd1.shape == (n_test, num_mcmc)
        assert np.issubdtype(ppd1.dtype, np.floating)

    def test_bcf_probit_binary_posterior_predictive(self):
        # Generate binary probit causal data
        rng = np.random.default_rng(42)
        n = 100
        p = 3
        X = rng.uniform(size=(n, p))
        pi_x = norm.cdf(X[:, 0])
        Z = rng.binomial(1, pi_x).astype(float)
        mu_x = X[:, 0] + X[:, 1]
        tau_x = np.full(n, 0.5)
        prob_y1 = norm.cdf(mu_x + tau_x * Z)
        y = rng.binomial(1, prob_y1).astype(float)

        # Train/test split
        train_inds, test_inds = train_test_split(
            np.arange(n), test_size=0.2, random_state=42
        )
        X_train = X[train_inds, :]
        X_test = X[test_inds, :]
        Z_train = Z[train_inds]
        Z_test = Z[test_inds]
        pi_train = pi_x[train_inds]
        pi_test = pi_x[test_inds]
        y_train = y[train_inds]
        n_test = X_test.shape[0]
        num_mcmc = 10

        # Fit binary probit BCF model
        bcf_model = BCFModel()
        bcf_model.sample(
            X_train=X_train,
            Z_train=Z_train,
            y_train=y_train,
            propensity_train=pi_train,
            X_test=X_test,
            Z_test=Z_test,
            propensity_test=pi_test,
            num_gfr=10,
            num_burnin=0,
            num_mcmc=num_mcmc,
            general_params={
                "sample_sigma2_global": False,
                "outcome_model": OutcomeModel(outcome="binary", link="probit"),
            },
        )

        # Test with multiple draws per sample
        ppd = bcf_model.sample_posterior_predictive(
            X=X_test,
            Z=Z_test,
            propensity=pi_test,
            num_draws_per_sample=3,
        )
        assert ppd.shape == (n_test, num_mcmc, 3)
        assert set(np.unique(ppd)).issubset({0, 1})

        # Test with single draw per sample
        ppd1 = bcf_model.sample_posterior_predictive(
            X=X_test,
            Z=Z_test,
            propensity=pi_test,
            num_draws_per_sample=1,
        )
        assert ppd1.shape == (n_test, num_mcmc)
        assert set(np.unique(ppd1)).issubset({0, 1})
