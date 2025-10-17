import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from stochtree import BCFModel


class TestBCF:
    def test_binary_bcf(self):
        # RNG
        random_seed = 101
        rng = np.random.default_rng(random_seed)

        # Generate covariates and basis
        n = 100
        p_X = 5
        X = rng.uniform(0, 1, (n, p_X))
        pi_X = 0.25 + 0.5 * X[:, 0]
        Z = rng.binomial(1, pi_X, n).astype(float)

        # Define the outcome mean functions (prognostic and treatment effects)
        mu_X = pi_X * 5
        tau_X = X[:, 1] * 2

        # Generate outcome
        epsilon = rng.normal(0, 1, n)
        y = mu_X + tau_X * Z + epsilon

        # Test-train split
        sample_inds = np.arange(n)
        train_inds, test_inds = train_test_split(sample_inds, test_size=0.5)
        X_train = X[train_inds, :]
        X_test = X[test_inds, :]
        Z_train = Z[train_inds]
        Z_test = Z[test_inds]
        y_train = y[train_inds]
        pi_train = pi_X[train_inds]
        pi_test = pi_X[test_inds]
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        # BCF settings
        num_gfr = 10
        num_burnin = 0
        num_mcmc = 10

        # Run BCF with test set and propensity score
        bcf_model = BCFModel()
        variance_forest_params = {"num_trees": 0}
        bcf_model.sample(
            X_train=X_train,
            Z_train=Z_train,
            y_train=y_train,
            pi_train=pi_train,
            X_test=X_test,
            Z_test=Z_test,
            pi_test=pi_test,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
            variance_forest_params=variance_forest_params,
        )

        # Assertions
        assert bcf_model.y_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.mu_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.tau_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.y_hat_test.shape == (n_test, num_mcmc)
        assert bcf_model.mu_hat_test.shape == (n_test, num_mcmc)
        assert bcf_model.tau_hat_test.shape == (n_test, num_mcmc)

        # Check overall prediction method
        bcf_preds = bcf_model.predict(X_test, Z_test, pi_test)
        tau_hat, mu_hat, y_hat = bcf_preds['tau_hat'], bcf_preds['mu_hat'], bcf_preds['y_hat']
        assert tau_hat.shape == (n_test, num_mcmc)
        assert mu_hat.shape == (n_test, num_mcmc)
        assert y_hat.shape == (n_test, num_mcmc)

        # Check that we can predict just treatment effects
        tau_hat = bcf_model.predict(X = X_test, Z = Z_test, propensity = pi_test, terms = "cate")
        assert tau_hat.shape == (n_test, num_mcmc)

        # Run BCF without test set and with propensity score
        bcf_model = BCFModel()
        variance_forest_params = {"num_trees": 0}
        bcf_model.sample(
            X_train=X_train,
            Z_train=Z_train,
            y_train=y_train,
            pi_train=pi_train,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
            variance_forest_params=variance_forest_params,
        )

        # Assertions
        assert bcf_model.y_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.mu_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.tau_hat_train.shape == (n_train, num_mcmc)

        # Check overall prediction method
        bcf_preds = bcf_model.predict(X_test, Z_test, pi_test)
        tau_hat, mu_hat, y_hat = bcf_preds['tau_hat'], bcf_preds['mu_hat'], bcf_preds['y_hat']
        assert tau_hat.shape == (n_test, num_mcmc)
        assert mu_hat.shape == (n_test, num_mcmc)
        assert y_hat.shape == (n_test, num_mcmc)
        # Check treatment effect prediction method
        tau_hat = bcf_model.predict(X = X_test, Z = Z_test, propensity = pi_test, terms = "cate")
        assert tau_hat.shape == (n_test, num_mcmc)

        # Run BCF with test set and without propensity score
        bcf_model = BCFModel()
        variance_forest_params = {"num_trees": 0}
        bcf_model.sample(
            X_train=X_train,
            Z_train=Z_train,
            y_train=y_train,
            X_test=X_test,
            Z_test=Z_test,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
            variance_forest_params=variance_forest_params,
        )

        # Assertions
        assert bcf_model.y_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.mu_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.tau_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.bart_propensity_model.y_hat_train.shape == (n_train, 10)
        assert bcf_model.y_hat_test.shape == (n_test, num_mcmc)
        assert bcf_model.mu_hat_test.shape == (n_test, num_mcmc)
        assert bcf_model.tau_hat_test.shape == (n_test, num_mcmc)
        assert bcf_model.bart_propensity_model.y_hat_test.shape == (n_test, 10)

        # Check overall prediction method
        bcf_preds = bcf_model.predict(X_test, Z_test)
        tau_hat, mu_hat, y_hat = bcf_preds['tau_hat'], bcf_preds['mu_hat'], bcf_preds['y_hat']
        assert tau_hat.shape == (n_test, num_mcmc)
        assert mu_hat.shape == (n_test, num_mcmc)
        assert y_hat.shape == (n_test, num_mcmc)

        # Check treatment effect prediction method
        tau_hat = bcf_model.predict(X = X_test, Z = Z_test, terms = "cate")
        assert tau_hat.shape == (n_test, num_mcmc)

        # Run BCF without test set and without propensity score
        bcf_model = BCFModel()
        variance_forest_params = {"num_trees": 0}
        bcf_model.sample(
            X_train=X_train,
            Z_train=Z_train,
            y_train=y_train,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
            variance_forest_params=variance_forest_params,
        )

        # Assertions
        assert bcf_model.y_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.mu_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.tau_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.bart_propensity_model.y_hat_train.shape == (n_train, 10)

        # Check overall prediction method
        bcf_preds = bcf_model.predict(X_test, Z_test)
        tau_hat, mu_hat, y_hat = bcf_preds['tau_hat'], bcf_preds['mu_hat'], bcf_preds['y_hat']
        assert tau_hat.shape == (n_test, num_mcmc)
        assert mu_hat.shape == (n_test, num_mcmc)
        assert y_hat.shape == (n_test, num_mcmc)

        # Check treatment effect prediction method
        tau_hat = bcf_model.predict(X = X_test, Z = Z_test, terms = "cate")

    def test_continuous_univariate_bcf(self):
        # RNG
        random_seed = 101
        rng = np.random.default_rng(random_seed)

        # Generate covariates and basis
        n = 100
        p_X = 5
        X = rng.uniform(0, 1, (n, p_X))
        pi_X = 0.25 + 0.5 * X[:, 0]
        Z = pi_X + rng.normal(0, 1, n)

        # Define the outcome mean functions (prognostic and treatment effects)
        mu_X = pi_X * 5
        tau_X = X[:, 1] * 2

        # Generate outcome
        epsilon = rng.normal(0, 1, n)
        y = mu_X + tau_X * Z + epsilon

        # Test-train split
        sample_inds = np.arange(n)
        train_inds, test_inds = train_test_split(sample_inds, test_size=0.5)
        X_train = X[train_inds, :]
        X_test = X[test_inds, :]
        Z_train = Z[train_inds]
        Z_test = Z[test_inds]
        y_train = y[train_inds]
        pi_train = pi_X[train_inds]
        pi_test = pi_X[test_inds]
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        # BCF settings
        num_gfr = 10
        num_burnin = 0
        num_mcmc = 10

        # Run BCF with test set and propensity score
        bcf_model = BCFModel()
        variance_forest_params = {"num_trees": 0}
        bcf_model.sample(
            X_train=X_train,
            Z_train=Z_train,
            y_train=y_train,
            pi_train=pi_train,
            X_test=X_test,
            Z_test=Z_test,
            pi_test=pi_test,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
            variance_forest_params=variance_forest_params,
        )

        # Assertions
        assert bcf_model.y_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.mu_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.tau_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.y_hat_test.shape == (n_test, num_mcmc)
        assert bcf_model.mu_hat_test.shape == (n_test, num_mcmc)
        assert bcf_model.tau_hat_test.shape == (n_test, num_mcmc)

        # Check overall prediction method
        bcf_preds = bcf_model.predict(X_test, Z_test, pi_test)
        tau_hat, mu_hat, y_hat = bcf_preds['tau_hat'], bcf_preds['mu_hat'], bcf_preds['y_hat']
        assert tau_hat.shape == (n_test, num_mcmc)
        assert mu_hat.shape == (n_test, num_mcmc)
        assert y_hat.shape == (n_test, num_mcmc)

        # Check treatment effect prediction method
        tau_hat = bcf_model.predict(X = X_test, Z = Z_test, propensity = pi_test, terms = "cate")
        assert tau_hat.shape == (n_test, num_mcmc)

        # Run second BCF model with test set and propensity score
        bcf_model_2 = BCFModel()
        variance_forest_params = {"num_trees": 0}
        bcf_model_2.sample(
            X_train=X_train,
            Z_train=Z_train,
            y_train=y_train,
            pi_train=pi_train,
            X_test=X_test,
            Z_test=Z_test,
            pi_test=pi_test,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
            variance_forest_params=variance_forest_params,
        )

        # Assertions
        assert bcf_model_2.y_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model_2.mu_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model_2.tau_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model_2.y_hat_test.shape == (n_test, num_mcmc)
        assert bcf_model_2.mu_hat_test.shape == (n_test, num_mcmc)
        assert bcf_model_2.tau_hat_test.shape == (n_test, num_mcmc)

        # Check overall prediction method
        bcf_preds_2 = bcf_model_2.predict(X_test, Z_test, pi_test)
        tau_hat_2, mu_hat_2, y_hat_2 = bcf_preds_2['tau_hat'], bcf_preds_2['mu_hat'], bcf_preds_2['y_hat']
        assert tau_hat_2.shape == (n_test, num_mcmc)
        assert mu_hat_2.shape == (n_test, num_mcmc)
        assert y_hat_2.shape == (n_test, num_mcmc)

        # Check treatment effect prediction method
        tau_hat_2 = bcf_model_2.predict(X = X_test, Z = Z_test, propensity = pi_test, terms = "cate")
        assert tau_hat_2.shape == (n_test, num_mcmc)

        # Combine into a single model
        bcf_models_json = [bcf_model.to_json(), bcf_model_2.to_json()]
        bcf_model_3 = BCFModel()
        bcf_model_3.from_json_string_list(bcf_models_json)

        # Assertions
        bcf_preds_3 = bcf_model_3.predict(X_test, Z_test, pi_test)
        tau_hat_3, mu_hat_3, y_hat_3 = bcf_preds_3['tau_hat'], bcf_preds_3['mu_hat'], bcf_preds_3['y_hat']
        assert tau_hat_3.shape == (n_train, num_mcmc * 2)
        assert mu_hat_3.shape == (n_train, num_mcmc * 2)
        assert y_hat_3.shape == (n_train, num_mcmc * 2)
        np.testing.assert_allclose(y_hat_3[:, 0:num_mcmc], y_hat)
        np.testing.assert_allclose(y_hat_3[:, num_mcmc : (2 * num_mcmc)], y_hat_2)
        np.testing.assert_allclose(mu_hat_3[:, 0:num_mcmc], mu_hat)
        np.testing.assert_allclose(mu_hat_3[:, num_mcmc : (2 * num_mcmc)], mu_hat_2)
        np.testing.assert_allclose(tau_hat_3[:, 0:num_mcmc], tau_hat)
        np.testing.assert_allclose(tau_hat_3[:, num_mcmc : (2 * num_mcmc)], tau_hat_2)
        np.testing.assert_allclose(
            bcf_model_3.global_var_samples[0:num_mcmc], bcf_model.global_var_samples
        )
        np.testing.assert_allclose(
            bcf_model_3.global_var_samples[num_mcmc : (2 * num_mcmc)],
            bcf_model_2.global_var_samples,
        )

        # Run BCF without test set and with propensity score
        bcf_model = BCFModel()
        variance_forest_params = {"num_trees": 0}
        bcf_model.sample(
            X_train=X_train,
            Z_train=Z_train,
            y_train=y_train,
            pi_train=pi_train,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
            variance_forest_params=variance_forest_params,
        )

        # Assertions
        assert bcf_model.y_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.mu_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.tau_hat_train.shape == (n_train, num_mcmc)

        # Check overall prediction method
        bcf_preds = bcf_model.predict(X_test, Z_test, pi_test)
        tau_hat, mu_hat, y_hat = bcf_preds['tau_hat'], bcf_preds['mu_hat'], bcf_preds['y_hat']
        assert tau_hat.shape == (n_test, num_mcmc)
        assert mu_hat.shape == (n_test, num_mcmc)
        assert y_hat.shape == (n_test, num_mcmc)

        # Check treatment effect prediction method
        tau_hat = bcf_model.predict(X = X_test, Z = Z_test, propensity = pi_test, terms = "cate")
        assert tau_hat.shape == (n_test, num_mcmc)

        # Run BCF with test set and without propensity score
        bcf_model = BCFModel()
        variance_forest_params = {"num_trees": 0}
        bcf_model.sample(
            X_train=X_train,
            Z_train=Z_train,
            y_train=y_train,
            X_test=X_test,
            Z_test=Z_test,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
            variance_forest_params=variance_forest_params,
        )

        # Assertions
        assert bcf_model.y_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.mu_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.tau_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.bart_propensity_model.y_hat_train.shape == (n_train, 10)
        assert bcf_model.y_hat_test.shape == (n_test, num_mcmc)
        assert bcf_model.mu_hat_test.shape == (n_test, num_mcmc)
        assert bcf_model.tau_hat_test.shape == (n_test, num_mcmc)
        assert bcf_model.bart_propensity_model.y_hat_test.shape == (n_test, 10)

        # Check overall prediction method
        bcf_preds = bcf_model.predict(X_test, Z_test, pi_test)
        tau_hat, mu_hat, y_hat = bcf_preds['tau_hat'], bcf_preds['mu_hat'], bcf_preds['y_hat']
        assert tau_hat.shape == (n_test, num_mcmc)
        assert mu_hat.shape == (n_test, num_mcmc)
        assert y_hat.shape == (n_test, num_mcmc)

        # Check treatment effect prediction method
        tau_hat = bcf_model.predict(X = X_test, Z = Z_test, terms = "cate")
        assert tau_hat.shape == (n_test, num_mcmc)

        # Run BCF without test set and without propensity score
        bcf_model = BCFModel()
        variance_forest_params = {"num_trees": 0}
        bcf_model.sample(
            X_train=X_train,
            Z_train=Z_train,
            y_train=y_train,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
            variance_forest_params=variance_forest_params,
        )

        # Assertions
        assert bcf_model.y_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.mu_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.tau_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.bart_propensity_model.y_hat_train.shape == (n_train, 10)

        # Check overall prediction method
        bcf_preds = bcf_model.predict(X_test, Z_test)
        tau_hat, mu_hat, y_hat = bcf_preds['tau_hat'], bcf_preds['mu_hat'], bcf_preds['y_hat']
        assert tau_hat.shape == (n_test, num_mcmc)
        assert mu_hat.shape == (n_test, num_mcmc)
        assert y_hat.shape == (n_test, num_mcmc)

        # Check treatment effect prediction method
        tau_hat = bcf_model.predict(X = X_test, Z = Z_test, terms = "cate")

        # Run second BCF model with test set and propensity score
        bcf_model_2 = BCFModel()
        variance_forest_params = {"num_trees": 0}
        bcf_model_2.sample(
            X_train=X_train,
            Z_train=Z_train,
            y_train=y_train,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
            variance_forest_params=variance_forest_params,
        )

        # Assertions
        assert bcf_model_2.y_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model_2.mu_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model_2.tau_hat_train.shape == (n_train, num_mcmc)

        # Check overall prediction method
        bcf_preds_2 = bcf_model_2.predict(X_test, Z_test)
        tau_hat_2, mu_hat_2, y_hat_2 = bcf_preds_2['tau_hat'], bcf_preds_2['mu_hat'], bcf_preds_2['y_hat']
        assert tau_hat_2.shape == (n_test, num_mcmc)
        assert mu_hat_2.shape == (n_test, num_mcmc)
        assert y_hat_2.shape == (n_test, num_mcmc)

        # Check treatment effect prediction method
        tau_hat_2 = bcf_model_2.predict(X = X_test, Z = Z_test, terms = "cate")
        assert tau_hat_2.shape == (n_test, num_mcmc)

        # Combine into a single model
        bcf_models_json = [bcf_model.to_json(), bcf_model_2.to_json()]
        bcf_model_3 = BCFModel()
        bcf_model_3.from_json_string_list(bcf_models_json)

        # Assertions
        bcf_preds_3 = bcf_model_3.predict(X_test, Z_test)
        tau_hat_3, mu_hat_3, y_hat_3 = bcf_preds_3['tau_hat'], bcf_preds_3['mu_hat'], bcf_preds_3['y_hat']
        assert tau_hat_3.shape == (n_train, num_mcmc * 2)
        assert mu_hat_3.shape == (n_train, num_mcmc * 2)
        assert y_hat_3.shape == (n_train, num_mcmc * 2)
        np.testing.assert_allclose(y_hat_3[:, 0:num_mcmc], y_hat)
        np.testing.assert_allclose(mu_hat_3[:, 0:num_mcmc], mu_hat)
        np.testing.assert_allclose(tau_hat_3[:, 0:num_mcmc], tau_hat)
        np.testing.assert_allclose(
            bcf_model_3.global_var_samples[0:num_mcmc], bcf_model.global_var_samples
        )
        np.testing.assert_allclose(
            bcf_model_3.global_var_samples[num_mcmc : (2 * num_mcmc)],
            bcf_model_2.global_var_samples,
        )

    def test_multivariate_bcf(self):
        # RNG
        random_seed = 101
        rng = np.random.default_rng(random_seed)

        # Generate covariates and basis
        n = 100
        p_X = 5
        X = rng.uniform(0, 1, (n, p_X))
        pi_X = np.c_[0.25 + 0.5 * X[:, 0], 0.5 - 0.25 * X[:, 1]]
        Z = pi_X + rng.normal(0, 1, (n, 2))
        treatment_dim = Z.shape[1]

        # Define the outcome mean functions (prognostic and treatment effects)
        mu_X = pi_X[:, 0] * 5
        tau_X = np.c_[X[:, 1] * 2, -0.5 * X[:, 2]]

        # Generate outcome
        epsilon = rng.normal(0, 1, n)
        treatment_term = (tau_X * Z).sum(axis=1)
        y = mu_X + treatment_term + epsilon

        # Test-train split
        sample_inds = np.arange(n)
        train_inds, test_inds = train_test_split(sample_inds, test_size=0.5)
        X_train = X[train_inds, :]
        X_test = X[test_inds, :]
        Z_train = Z[train_inds]
        Z_test = Z[test_inds]
        y_train = y[train_inds]
        pi_train = pi_X[train_inds]
        pi_test = pi_X[test_inds]
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        # BCF settings
        num_gfr = 10
        num_burnin = 0
        num_mcmc = 10

        # Run BCF with test set and propensity score
        bcf_model = BCFModel()
        variance_forest_params = {"num_trees": 0}
        bcf_model.sample(
            X_train=X_train,
            Z_train=Z_train,
            y_train=y_train,
            pi_train=pi_train,
            X_test=X_test,
            Z_test=Z_test,
            pi_test=pi_test,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
            variance_forest_params=variance_forest_params,
        )

        # Assertions
        assert bcf_model.y_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.mu_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.tau_hat_train.shape == (n_train, num_mcmc, treatment_dim)
        assert bcf_model.y_hat_test.shape == (n_test, num_mcmc)
        assert bcf_model.mu_hat_test.shape == (n_test, num_mcmc)
        assert bcf_model.tau_hat_test.shape == (n_test, num_mcmc, treatment_dim)

        # Check overall prediction method
        bcf_preds = bcf_model.predict(X_test, Z_test, pi_test)
        tau_hat, mu_hat, y_hat = bcf_preds['tau_hat'], bcf_preds['mu_hat'], bcf_preds['y_hat']
        assert tau_hat.shape == (n_test, num_mcmc, treatment_dim)
        assert mu_hat.shape == (n_test, num_mcmc)
        assert y_hat.shape == (n_test, num_mcmc)

        # Check treatment effect prediction method
        tau_hat = bcf_model.predict(X = X_test, Z = Z_test, propensity = pi_test, terms = "cate")
        assert tau_hat.shape == (n_test, num_mcmc, treatment_dim)

        # Run BCF without test set and with propensity score
        bcf_model = BCFModel()
        variance_forest_params = {"num_trees": 0}
        bcf_model.sample(
            X_train=X_train,
            Z_train=Z_train,
            y_train=y_train,
            pi_train=pi_train,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
            variance_forest_params=variance_forest_params,
        )

        # Assertions
        assert bcf_model.y_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.mu_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.tau_hat_train.shape == (n_train, num_mcmc, treatment_dim)

        # Check overall prediction method
        bcf_preds = bcf_model.predict(X_test, Z_test, pi_test)
        tau_hat, mu_hat, y_hat = bcf_preds['tau_hat'], bcf_preds['mu_hat'], bcf_preds['y_hat']
        assert tau_hat.shape == (n_test, num_mcmc, treatment_dim)
        assert mu_hat.shape == (n_test, num_mcmc)
        assert y_hat.shape == (n_test, num_mcmc)

        # Check treatment effect prediction method
        tau_hat = bcf_model.predict(X = X_test, Z = Z_test, propensity = pi_test, terms = "cate")
        assert tau_hat.shape == (n_test, num_mcmc, treatment_dim)

        # Run BCF with test set and without propensity score
        with pytest.raises(ValueError):
            bcf_model = BCFModel()
            variance_forest_params = {"num_trees": 0}
            bcf_model.sample(
                X_train=X_train,
                Z_train=Z_train,
                y_train=y_train,
                X_test=X_test,
                Z_test=Z_test,
                num_gfr=num_gfr,
                num_burnin=num_burnin,
                num_mcmc=num_mcmc,
                variance_forest_params=variance_forest_params,
            )

        # Run BCF without test set and without propensity score
        with pytest.raises(ValueError):
            bcf_model = BCFModel()
            variance_forest_params = {"num_trees": 0}
            bcf_model.sample(
                X_train=X_train,
                Z_train=Z_train,
                y_train=y_train,
                num_gfr=num_gfr,
                num_burnin=num_burnin,
                num_mcmc=num_mcmc,
                variance_forest_params=variance_forest_params,
            )

    def test_binary_bcf_heteroskedastic(self):
        # RNG
        random_seed = 101
        rng = np.random.default_rng(random_seed)

        # Generate covariates and basis
        n = 100
        p_X = 5
        X = rng.uniform(0, 1, (n, p_X))
        pi_X = 0.25 + 0.5 * X[:, 0]
        Z = rng.binomial(1, pi_X, n).astype(float)

        # Define the outcome mean functions (prognostic and treatment effects)
        mu_X = pi_X * 5
        tau_X = X[:, 1] * 2

        # Generate outcome
        epsilon = rng.normal(0, 1, n)
        y = mu_X + tau_X * Z + epsilon

        # Test-train split
        sample_inds = np.arange(n)
        train_inds, test_inds = train_test_split(sample_inds, test_size=0.5)
        X_train = X[train_inds, :]
        X_test = X[test_inds, :]
        Z_train = Z[train_inds]
        Z_test = Z[test_inds]
        y_train = y[train_inds]
        pi_train = pi_X[train_inds]
        pi_test = pi_X[test_inds]
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        # BCF settings
        num_gfr = 10
        num_burnin = 0
        num_mcmc = 10

        # Run BCF with test set and propensity score
        bcf_model = BCFModel()
        variance_forest_params = {"num_trees": 50}
        bcf_model.sample(
            X_train=X_train,
            Z_train=Z_train,
            y_train=y_train,
            pi_train=pi_train,
            X_test=X_test,
            Z_test=Z_test,
            pi_test=pi_test,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
            variance_forest_params=variance_forest_params,
        )

        # Assertions
        assert bcf_model.y_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.mu_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.tau_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.sigma2_x_train.shape == (n_train, num_mcmc)
        assert bcf_model.y_hat_test.shape == (n_test, num_mcmc)
        assert bcf_model.mu_hat_test.shape == (n_test, num_mcmc)
        assert bcf_model.tau_hat_test.shape == (n_test, num_mcmc)
        assert bcf_model.sigma2_x_test.shape == (n_train, num_mcmc)

        # Check overall prediction method
        bcf_preds = bcf_model.predict(X_test, Z_test, pi_test)
        tau_hat, mu_hat, y_hat, sigma2_x_hat = bcf_preds['tau_hat'], bcf_preds['mu_hat'], bcf_preds['y_hat'], bcf_preds['variance_forest_predictions']
        assert tau_hat.shape == (n_test, num_mcmc)
        assert mu_hat.shape == (n_test, num_mcmc)
        assert y_hat.shape == (n_test, num_mcmc)
        assert sigma2_x_hat.shape == (n_test, num_mcmc)

        # Check treatment effect prediction method
        tau_hat = bcf_model.predict(X = X_test, Z = Z_test, propensity = pi_test, terms = "cate")
        assert tau_hat.shape == (n_test, num_mcmc)

        # Run BCF without test set and with propensity score
        bcf_model = BCFModel()
        variance_forest_params = {"num_trees": 50}
        bcf_model.sample(
            X_train=X_train,
            Z_train=Z_train,
            y_train=y_train,
            pi_train=pi_train,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
            variance_forest_params=variance_forest_params,
        )

        # Assertions
        assert bcf_model.y_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.mu_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.tau_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.sigma2_x_train.shape == (n_train, num_mcmc)

        # Check overall prediction method
        bcf_preds = bcf_model.predict(X_test, Z_test, pi_test)
        assert bcf_preds['tau_hat'].shape == (n_test, num_mcmc)
        assert bcf_preds['mu_hat'].shape == (n_test, num_mcmc)
        assert bcf_preds['y_hat'].shape == (n_test, num_mcmc)
        assert bcf_preds['variance_forest_predictions'].shape == (n_test, num_mcmc)

        # Check predictions match
        bcf_preds = bcf_model.predict(X_train, Z_train, pi_train)
        assert bcf_preds['tau_hat'].shape == (n_train, num_mcmc)
        assert bcf_preds['mu_hat'].shape == (n_train, num_mcmc)
        assert bcf_preds['y_hat'].shape == (n_train, num_mcmc)
        assert bcf_preds['variance_forest_predictions'].shape == (n_train, num_mcmc)
        np.testing.assert_allclose(
            bcf_preds['y_hat'], bcf_model.y_hat_train
        )
        np.testing.assert_allclose(
            bcf_preds['mu_hat'], bcf_model.mu_hat_train
        )
        np.testing.assert_allclose(
            bcf_preds['tau_hat'], bcf_model.tau_hat_train
        )
        np.testing.assert_allclose(
            bcf_preds['variance_forest_predictions'], bcf_model.sigma2_x_train
        )

        # Check treatment effect prediction method
        tau_hat = bcf_model.predict(X = X_test, Z = Z_test, propensity = pi_test, terms = "cate")
        assert tau_hat.shape == (n_test, num_mcmc)

        # Run BCF with test set and without propensity score
        bcf_model = BCFModel()
        variance_forest_params = {"num_trees": 50}
        bcf_model.sample(
            X_train=X_train,
            Z_train=Z_train,
            y_train=y_train,
            X_test=X_test,
            Z_test=Z_test,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
            variance_forest_params=variance_forest_params,
        )

        # Assertions
        assert bcf_model.y_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.mu_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.tau_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.tau_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.bart_propensity_model.y_hat_train.shape == (n_train, 10)
        assert bcf_model.y_hat_test.shape == (n_test, num_mcmc)
        assert bcf_model.mu_hat_test.shape == (n_test, num_mcmc)
        assert bcf_model.tau_hat_test.shape == (n_test, num_mcmc)
        assert bcf_model.bart_propensity_model.y_hat_test.shape == (n_test, 10)

        # Check overall prediction method
        bcf_preds = bcf_model.predict(X_test, Z_test)
        assert bcf_preds['tau_hat'].shape == (n_test, num_mcmc)
        assert bcf_preds['mu_hat'].shape == (n_test, num_mcmc)
        assert bcf_preds['y_hat'].shape == (n_test, num_mcmc)
        assert bcf_preds['variance_forest_predictions'].shape == (n_test, num_mcmc)

        # Check treatment effect prediction method
        tau_hat = bcf_model.predict(X = X_test, Z = Z_test, terms = "cate")
        assert tau_hat.shape == (n_test, num_mcmc)

        # Run BCF without test set and without propensity score
        bcf_model = BCFModel()
        variance_forest_params = {"num_trees": 0}
        bcf_model.sample(
            X_train=X_train,
            Z_train=Z_train,
            y_train=y_train,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
            variance_forest_params=variance_forest_params,
        )

        # Assertions
        assert bcf_model.y_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.mu_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.tau_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.bart_propensity_model.y_hat_train.shape == (n_train, 10)

        # Check overall prediction method
        bcf_preds = bcf_model.predict(X_test, Z_test)
        assert bcf_preds['tau_hat'].shape == (n_test, num_mcmc)
        assert bcf_preds['mu_hat'].shape == (n_test, num_mcmc)
        assert bcf_preds['y_hat'].shape == (n_test, num_mcmc)

        # Check treatment effect prediction method
        tau_hat = bcf_model.predict(X = X_test, Z = Z_test, terms = "cate")
    
    def test_bcf_rfx_parameters(self):
        # RNG
        random_seed = 101
        rng = np.random.default_rng(random_seed)

        # Generate covariates and basis
        n = 100
        p_X = 5
        X = rng.uniform(0, 1, (n, p_X))
        pi_X = 0.25 + 0.5 * X[:, 0]
        Z = rng.binomial(1, pi_X, n).astype(float)

        # Define the outcome mean functions (prognostic and treatment effects)
        mu_X = pi_X * 5
        tau_X = X[:, 1] * 2

        # Generate RFX group labels and basis term
        num_rfx_basis = 2
        num_rfx_groups = 4
        group_labels = rng.choice(num_rfx_groups, size=n)
        rfx_basis = np.empty((n, num_rfx_basis))
        rfx_basis[:, 0] = 1.0
        if num_rfx_basis > 1:
            rfx_basis[:, 1:] = rng.uniform(-1, 1, (n, num_rfx_basis - 1))

        # Define the group rfx function
        def rfx_term(group_labels, basis):
            return np.where(
                group_labels == 0, -5 + 1. * basis[:,1], 5 - 1. * basis[:,1]
            )
        
        # Generate outcome
        epsilon = rng.normal(0, 1, n)
        y = mu_X + tau_X * Z + rfx_term(group_labels, rfx_basis) + epsilon

        # Test-train split
        sample_inds = np.arange(n)
        train_inds, test_inds = train_test_split(sample_inds, test_size=0.5)
        X_train = X[train_inds, :]
        X_test = X[test_inds, :]
        Z_train = Z[train_inds]
        Z_test = Z[test_inds]
        y_train = y[train_inds]
        pi_train = pi_X[train_inds]
        pi_test = pi_X[test_inds]
        group_labels_train = group_labels[train_inds]
        group_labels_test = group_labels[test_inds]
        rfx_basis_train = rfx_basis[train_inds, :]
        rfx_basis_test = rfx_basis[test_inds, :]

        # BART settings
        num_gfr = 10
        num_burnin = 0
        num_mcmc = 10

        # Specify no rfx parameters
        general_params = {}
        bcf_model = BCFModel()
        bcf_model.sample(
            X_train=X_train,
            Z_train=Z_train,
            y_train=y_train,
            pi_train=pi_train,
            X_test=X_test,
            Z_test=Z_test,
            pi_test=pi_test,
            rfx_group_ids_train=group_labels_train,
            rfx_basis_train=rfx_basis_train,
            rfx_group_ids_test=group_labels_test,
            rfx_basis_test=rfx_basis_test,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
            general_params=general_params
        )

        # Specify scalar rfx parameters
        general_params = {
            "rfx_working_parameter_prior_mean": 1.,
            "rfx_group_parameter_prior_mean": 1.,
            "rfx_working_parameter_prior_cov": 1.,
            "rfx_group_parameter_prior_cov": 1.,
            "rfx_variance_prior_shape": 1,
            "rfx_variance_prior_scale": 1
        }
        bcf_model_2 = BCFModel()
        bcf_model_2.sample(
            X_train=X_train,
            Z_train=Z_train,
            y_train=y_train,
            pi_train=pi_train,
            X_test=X_test,
            Z_test=Z_test,
            pi_test=pi_test,
            rfx_group_ids_train=group_labels_train,
            rfx_basis_train=rfx_basis_train,
            rfx_group_ids_test=group_labels_test,
            rfx_basis_test=rfx_basis_test,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
            general_params=general_params
        )

        # Specify all relevant rfx parameters as vectors
        general_params = {
            "rfx_working_parameter_prior_mean": np.repeat(1., num_rfx_basis),
            "rfx_group_parameter_prior_mean": np.repeat(1., num_rfx_basis),
            "rfx_working_parameter_prior_cov": np.identity(num_rfx_basis),
            "rfx_group_parameter_prior_cov": np.identity(num_rfx_basis),
            "rfx_variance_prior_shape": 1,
            "rfx_variance_prior_scale": 1
        }
        bcf_model_3 = BCFModel()
        bcf_model_3.sample(
            X_train=X_train,
            Z_train=Z_train,
            y_train=y_train,
            pi_train=pi_train,
            X_test=X_test,
            Z_test=Z_test,
            pi_test=pi_test,
            rfx_group_ids_train=group_labels_train,
            rfx_basis_train=rfx_basis_train,
            rfx_group_ids_test=group_labels_test,
            rfx_basis_test=rfx_basis_test,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
            general_params=general_params
        )
