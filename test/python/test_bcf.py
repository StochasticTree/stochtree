import numpy as np
import pandas as pd
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
            propensity_train=pi_train,
            X_test=X_test,
            Z_test=Z_test,
            propensity_test=pi_test,
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
        tau_hat, mu_hat, y_hat = (
            bcf_preds["tau_hat"],
            bcf_preds["mu_hat"],
            bcf_preds["y_hat"],
        )
        assert tau_hat.shape == (n_test, num_mcmc)
        assert mu_hat.shape == (n_test, num_mcmc)
        assert y_hat.shape == (n_test, num_mcmc)

        # Check that we can predict just treatment effects
        tau_hat = bcf_model.predict(
            X=X_test, Z=Z_test, propensity=pi_test, terms="cate"
        )
        assert tau_hat.shape == (n_test, num_mcmc)

        # Run BCF without test set and with propensity score
        bcf_model = BCFModel()
        variance_forest_params = {"num_trees": 0}
        bcf_model.sample(
            X_train=X_train,
            Z_train=Z_train,
            y_train=y_train,
            propensity_train=pi_train,
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
        tau_hat, mu_hat, y_hat = (
            bcf_preds["tau_hat"],
            bcf_preds["mu_hat"],
            bcf_preds["y_hat"],
        )
        assert tau_hat.shape == (n_test, num_mcmc)
        assert mu_hat.shape == (n_test, num_mcmc)
        assert y_hat.shape == (n_test, num_mcmc)
        # Check treatment effect prediction method
        tau_hat = bcf_model.predict(
            X=X_test, Z=Z_test, propensity=pi_test, terms="cate"
        )
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
        tau_hat, mu_hat, y_hat = (
            bcf_preds["tau_hat"],
            bcf_preds["mu_hat"],
            bcf_preds["y_hat"],
        )
        assert tau_hat.shape == (n_test, num_mcmc)
        assert mu_hat.shape == (n_test, num_mcmc)
        assert y_hat.shape == (n_test, num_mcmc)

        # Check treatment effect prediction method
        tau_hat = bcf_model.predict(X=X_test, Z=Z_test, terms="cate")
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
        tau_hat, mu_hat, y_hat = (
            bcf_preds["tau_hat"],
            bcf_preds["mu_hat"],
            bcf_preds["y_hat"],
        )
        assert tau_hat.shape == (n_test, num_mcmc)
        assert mu_hat.shape == (n_test, num_mcmc)
        assert y_hat.shape == (n_test, num_mcmc)

        # Check treatment effect prediction method
        tau_hat = bcf_model.predict(X=X_test, Z=Z_test, terms="cate")

        # Check that we can run BCF without propensities
        bcf_model = BCFModel()
        general_params = {"propensity_covariate": "none"}
        bcf_model.sample(
            X_train=X_train,
            Z_train=Z_train,
            y_train=y_train,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
            general_params=general_params,
        )

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
        # adaptive_coding=True triggers a UserWarning for non-binary treatment
        with pytest.warns(UserWarning):
            bcf_model = BCFModel()
            variance_forest_params = {"num_trees": 0}
            bcf_model.sample(
                X_train=X_train,
                Z_train=Z_train,
                y_train=y_train,
                propensity_train=pi_train,
                X_test=X_test,
                Z_test=Z_test,
                propensity_test=pi_test,
                num_gfr=num_gfr,
                num_burnin=num_burnin,
                num_mcmc=num_mcmc,
                general_params={"adaptive_coding": True},
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
        tau_hat, mu_hat, y_hat = (
            bcf_preds["tau_hat"],
            bcf_preds["mu_hat"],
            bcf_preds["y_hat"],
        )
        assert tau_hat.shape == (n_test, num_mcmc)
        assert mu_hat.shape == (n_test, num_mcmc)
        assert y_hat.shape == (n_test, num_mcmc)

        # Check treatment effect prediction method
        tau_hat = bcf_model.predict(
            X=X_test, Z=Z_test, propensity=pi_test, terms="cate"
        )
        assert tau_hat.shape == (n_test, num_mcmc)

        # Run second BCF model with test set and propensity score
        with pytest.warns(UserWarning):
            bcf_model_2 = BCFModel()
            variance_forest_params = {"num_trees": 0}
            bcf_model_2.sample(
                X_train=X_train,
                Z_train=Z_train,
                y_train=y_train,
                propensity_train=pi_train,
                X_test=X_test,
                Z_test=Z_test,
                propensity_test=pi_test,
                num_gfr=num_gfr,
                num_burnin=num_burnin,
                num_mcmc=num_mcmc,
                general_params={"adaptive_coding": True},
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
        tau_hat_2, mu_hat_2, y_hat_2 = (
            bcf_preds_2["tau_hat"],
            bcf_preds_2["mu_hat"],
            bcf_preds_2["y_hat"],
        )
        assert tau_hat_2.shape == (n_test, num_mcmc)
        assert mu_hat_2.shape == (n_test, num_mcmc)
        assert y_hat_2.shape == (n_test, num_mcmc)

        # Check treatment effect prediction method
        tau_hat_2 = bcf_model_2.predict(
            X=X_test, Z=Z_test, propensity=pi_test, terms="cate"
        )
        assert tau_hat_2.shape == (n_test, num_mcmc)

        # Combine into a single model
        bcf_models_json = [bcf_model.to_json(), bcf_model_2.to_json()]
        bcf_model_3 = BCFModel()
        bcf_model_3.from_json_string_list(bcf_models_json)

        # Assertions
        bcf_preds_3 = bcf_model_3.predict(X_test, Z_test, pi_test)
        # Use "cate" (tau_0 + tau(X)) for the CATE comparison, consistent with how
        # tau_hat and tau_hat_2 were set above (via predict(terms="cate"))
        cate_hat_3, mu_hat_3, y_hat_3 = (
            bcf_preds_3["cate"],
            bcf_preds_3["mu_hat"],
            bcf_preds_3["y_hat"],
        )
        assert cate_hat_3.shape == (n_train, num_mcmc * 2)
        assert mu_hat_3.shape == (n_train, num_mcmc * 2)
        assert y_hat_3.shape == (n_train, num_mcmc * 2)
        np.testing.assert_allclose(y_hat_3[:, 0:num_mcmc], y_hat)
        np.testing.assert_allclose(y_hat_3[:, num_mcmc : (2 * num_mcmc)], y_hat_2)
        np.testing.assert_allclose(mu_hat_3[:, 0:num_mcmc], mu_hat)
        np.testing.assert_allclose(mu_hat_3[:, num_mcmc : (2 * num_mcmc)], mu_hat_2)
        np.testing.assert_allclose(cate_hat_3[:, 0:num_mcmc], tau_hat)
        np.testing.assert_allclose(cate_hat_3[:, num_mcmc : (2 * num_mcmc)], tau_hat_2)
        np.testing.assert_allclose(
            bcf_model_3.global_var_samples[0:num_mcmc], bcf_model.global_var_samples
        )
        np.testing.assert_allclose(
            bcf_model_3.global_var_samples[num_mcmc : (2 * num_mcmc)],
            bcf_model_2.global_var_samples,
        )

        # Run BCF without test set and with propensity score
        with pytest.warns(UserWarning):
            bcf_model = BCFModel()
            variance_forest_params = {"num_trees": 0}
            bcf_model.sample(
                X_train=X_train,
                Z_train=Z_train,
                y_train=y_train,
                propensity_train=pi_train,
                num_gfr=num_gfr,
                num_burnin=num_burnin,
                num_mcmc=num_mcmc,
                general_params={"adaptive_coding": True},
                variance_forest_params=variance_forest_params,
            )

        # Assertions
        assert bcf_model.y_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.mu_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.tau_hat_train.shape == (n_train, num_mcmc)

        # Check overall prediction method
        bcf_preds = bcf_model.predict(X_test, Z_test, pi_test)
        tau_hat, mu_hat, y_hat = (
            bcf_preds["tau_hat"],
            bcf_preds["mu_hat"],
            bcf_preds["y_hat"],
        )
        assert tau_hat.shape == (n_test, num_mcmc)
        assert mu_hat.shape == (n_test, num_mcmc)
        assert y_hat.shape == (n_test, num_mcmc)

        # Check treatment effect prediction method
        tau_hat = bcf_model.predict(
            X=X_test, Z=Z_test, propensity=pi_test, terms="cate"
        )
        assert tau_hat.shape == (n_test, num_mcmc)

        # Run BCF with test set and without propensity score
        with pytest.warns(UserWarning):
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
                general_params={"adaptive_coding": True},
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
        tau_hat, mu_hat, y_hat = (
            bcf_preds["tau_hat"],
            bcf_preds["mu_hat"],
            bcf_preds["y_hat"],
        )
        assert tau_hat.shape == (n_test, num_mcmc)
        assert mu_hat.shape == (n_test, num_mcmc)
        assert y_hat.shape == (n_test, num_mcmc)

        # Check treatment effect prediction method
        tau_hat = bcf_model.predict(X=X_test, Z=Z_test, terms="cate")
        assert tau_hat.shape == (n_test, num_mcmc)

        # Run BCF without test set and without propensity score
        with pytest.warns(UserWarning):
            bcf_model = BCFModel()
            variance_forest_params = {"num_trees": 0}
            bcf_model.sample(
                X_train=X_train,
                Z_train=Z_train,
                y_train=y_train,
                num_gfr=num_gfr,
                num_burnin=num_burnin,
                num_mcmc=num_mcmc,
                general_params={"adaptive_coding": True},
                variance_forest_params=variance_forest_params,
            )

        # Assertions
        assert bcf_model.y_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.mu_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.tau_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.bart_propensity_model.y_hat_train.shape == (n_train, 10)

        # Check overall prediction method
        bcf_preds = bcf_model.predict(X_test, Z_test)
        tau_hat, mu_hat, y_hat = (
            bcf_preds["tau_hat"],
            bcf_preds["mu_hat"],
            bcf_preds["y_hat"],
        )
        assert tau_hat.shape == (n_test, num_mcmc)
        assert mu_hat.shape == (n_test, num_mcmc)
        assert y_hat.shape == (n_test, num_mcmc)

        # Check treatment effect prediction method
        tau_hat = bcf_model.predict(X=X_test, Z=Z_test, terms="cate")

        # Run second BCF model with test set and propensity score
        with pytest.warns(UserWarning):
            bcf_model_2 = BCFModel()
            variance_forest_params = {"num_trees": 0}
            bcf_model_2.sample(
                X_train=X_train,
                Z_train=Z_train,
                y_train=y_train,
                num_gfr=num_gfr,
                num_burnin=num_burnin,
                num_mcmc=num_mcmc,
                general_params={"adaptive_coding": True},
                variance_forest_params=variance_forest_params,
            )

        # Assertions
        assert bcf_model_2.y_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model_2.mu_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model_2.tau_hat_train.shape == (n_train, num_mcmc)

        # Check overall prediction method
        bcf_preds_2 = bcf_model_2.predict(X_test, Z_test)
        tau_hat_2, mu_hat_2, y_hat_2 = (
            bcf_preds_2["tau_hat"],
            bcf_preds_2["mu_hat"],
            bcf_preds_2["y_hat"],
        )
        assert tau_hat_2.shape == (n_test, num_mcmc)
        assert mu_hat_2.shape == (n_test, num_mcmc)
        assert y_hat_2.shape == (n_test, num_mcmc)

        # Check treatment effect prediction method
        tau_hat_2 = bcf_model_2.predict(X=X_test, Z=Z_test, terms="cate")
        assert tau_hat_2.shape == (n_test, num_mcmc)

        # Combine into a single model
        bcf_models_json = [bcf_model.to_json(), bcf_model_2.to_json()]
        bcf_model_3 = BCFModel()
        bcf_model_3.from_json_string_list(bcf_models_json)

        # Assertions
        bcf_preds_3 = bcf_model_3.predict(X_test, Z_test)
        cate_hat_3, mu_hat_3, y_hat_3 = (
            bcf_preds_3["cate"],
            bcf_preds_3["mu_hat"],
            bcf_preds_3["y_hat"],
        )
        assert cate_hat_3.shape == (n_train, num_mcmc * 2)
        assert mu_hat_3.shape == (n_train, num_mcmc * 2)
        assert y_hat_3.shape == (n_train, num_mcmc * 2)
        np.testing.assert_allclose(y_hat_3[:, 0:num_mcmc], y_hat)
        np.testing.assert_allclose(mu_hat_3[:, 0:num_mcmc], mu_hat)
        np.testing.assert_allclose(cate_hat_3[:, 0:num_mcmc], tau_hat)
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
        # adaptive_coding=True triggers a UserWarning for non-binary treatment
        with pytest.warns(UserWarning):
            bcf_model = BCFModel()
            variance_forest_params = {"num_trees": 0}
            bcf_model.sample(
                X_train=X_train,
                Z_train=Z_train,
                y_train=y_train,
                propensity_train=pi_train,
                X_test=X_test,
                Z_test=Z_test,
                propensity_test=pi_test,
                num_gfr=num_gfr,
                num_burnin=num_burnin,
                num_mcmc=num_mcmc,
                general_params={"adaptive_coding": True},
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
        tau_hat, mu_hat, y_hat = (
            bcf_preds["tau_hat"],
            bcf_preds["mu_hat"],
            bcf_preds["y_hat"],
        )
        assert tau_hat.shape == (n_test, num_mcmc, treatment_dim)
        assert mu_hat.shape == (n_test, num_mcmc)
        assert y_hat.shape == (n_test, num_mcmc)

        # Check treatment effect prediction method
        tau_hat = bcf_model.predict(
            X=X_test, Z=Z_test, propensity=pi_test, terms="cate"
        )
        assert tau_hat.shape == (n_test, num_mcmc, treatment_dim)

        # Run BCF without test set and with propensity score
        with pytest.warns(UserWarning):
            bcf_model = BCFModel()
            variance_forest_params = {"num_trees": 0}
            bcf_model.sample(
                X_train=X_train,
                Z_train=Z_train,
                y_train=y_train,
                propensity_train=pi_train,
                num_gfr=num_gfr,
                num_burnin=num_burnin,
                num_mcmc=num_mcmc,
                general_params={"adaptive_coding": True},
                variance_forest_params=variance_forest_params,
            )

        # Assertions
        assert bcf_model.y_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.mu_hat_train.shape == (n_train, num_mcmc)
        assert bcf_model.tau_hat_train.shape == (n_train, num_mcmc, treatment_dim)

        # Check overall prediction method
        bcf_preds = bcf_model.predict(X_test, Z_test, pi_test)
        tau_hat, mu_hat, y_hat = (
            bcf_preds["tau_hat"],
            bcf_preds["mu_hat"],
            bcf_preds["y_hat"],
        )
        assert tau_hat.shape == (n_test, num_mcmc, treatment_dim)
        assert mu_hat.shape == (n_test, num_mcmc)
        assert y_hat.shape == (n_test, num_mcmc)

        # Check treatment effect prediction method
        tau_hat = bcf_model.predict(
            X=X_test, Z=Z_test, propensity=pi_test, terms="cate"
        )
        assert tau_hat.shape == (n_test, num_mcmc, treatment_dim)

        # Run BCF with test set and without propensity score
        with pytest.warns(UserWarning):
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
                general_params={"adaptive_coding": True},
                variance_forest_params=variance_forest_params,
            )

        # Run BCF without test set and without propensity score
        with pytest.warns(UserWarning):
            bcf_model = BCFModel()
            variance_forest_params = {"num_trees": 0}
            bcf_model.sample(
                X_train=X_train,
                Z_train=Z_train,
                y_train=y_train,
                num_gfr=num_gfr,
                num_burnin=num_burnin,
                num_mcmc=num_mcmc,
                general_params={"adaptive_coding": True},
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
        with pytest.warns(UserWarning):
            bcf_model = BCFModel()
            variance_forest_params = {"num_trees": 50}
            bcf_model.sample(
                X_train=X_train,
                Z_train=Z_train,
                y_train=y_train,
                propensity_train=pi_train,
                X_test=X_test,
                Z_test=Z_test,
                propensity_test=pi_test,
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
        tau_hat, mu_hat, y_hat, sigma2_x_hat = (
            bcf_preds["tau_hat"],
            bcf_preds["mu_hat"],
            bcf_preds["y_hat"],
            bcf_preds["variance_forest_predictions"],
        )
        assert tau_hat.shape == (n_test, num_mcmc)
        assert mu_hat.shape == (n_test, num_mcmc)
        assert y_hat.shape == (n_test, num_mcmc)
        assert sigma2_x_hat.shape == (n_test, num_mcmc)

        # Check treatment effect prediction method
        tau_hat = bcf_model.predict(
            X=X_test, Z=Z_test, propensity=pi_test, terms="cate"
        )
        assert tau_hat.shape == (n_test, num_mcmc)

        # Run BCF without test set and with propensity score
        with pytest.warns(UserWarning):
            bcf_model = BCFModel()
            variance_forest_params = {"num_trees": 50}
            bcf_model.sample(
                X_train=X_train,
                Z_train=Z_train,
                y_train=y_train,
                propensity_train=pi_train,
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
        assert bcf_preds["tau_hat"].shape == (n_test, num_mcmc)
        assert bcf_preds["mu_hat"].shape == (n_test, num_mcmc)
        assert bcf_preds["y_hat"].shape == (n_test, num_mcmc)
        assert bcf_preds["variance_forest_predictions"].shape == (n_test, num_mcmc)

        # Check predictions match
        bcf_preds = bcf_model.predict(X_train, Z_train, pi_train)
        assert bcf_preds["tau_hat"].shape == (n_train, num_mcmc)
        assert bcf_preds["mu_hat"].shape == (n_train, num_mcmc)
        assert bcf_preds["y_hat"].shape == (n_train, num_mcmc)
        assert bcf_preds["variance_forest_predictions"].shape == (n_train, num_mcmc)
        np.testing.assert_allclose(bcf_preds["y_hat"], bcf_model.y_hat_train)
        np.testing.assert_allclose(bcf_preds["mu_hat"], bcf_model.mu_hat_train)
        np.testing.assert_allclose(bcf_preds["tau_hat"], bcf_model.tau_hat_train)
        np.testing.assert_allclose(
            bcf_preds["variance_forest_predictions"], bcf_model.sigma2_x_train
        )

        # Check treatment effect prediction method
        tau_hat = bcf_model.predict(
            X=X_test, Z=Z_test, propensity=pi_test, terms="cate"
        )
        assert tau_hat.shape == (n_test, num_mcmc)

        # Run BCF with test set and without propensity score
        with pytest.warns(UserWarning):
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
        assert bcf_preds["tau_hat"].shape == (n_test, num_mcmc)
        assert bcf_preds["mu_hat"].shape == (n_test, num_mcmc)
        assert bcf_preds["y_hat"].shape == (n_test, num_mcmc)
        assert bcf_preds["variance_forest_predictions"].shape == (n_test, num_mcmc)

        # Check treatment effect prediction method
        tau_hat = bcf_model.predict(X=X_test, Z=Z_test, terms="cate")
        assert tau_hat.shape == (n_test, num_mcmc)

        # Run BCF without test set and without propensity score
        with pytest.warns(UserWarning):
            bcf_model = BCFModel()
            variance_forest_params = {"num_trees": 50}
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
        assert bcf_preds["tau_hat"].shape == (n_test, num_mcmc)
        assert bcf_preds["mu_hat"].shape == (n_test, num_mcmc)
        assert bcf_preds["y_hat"].shape == (n_test, num_mcmc)

        # Check treatment effect prediction method
        tau_hat = bcf_model.predict(X=X_test, Z=Z_test, terms="cate")

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
                group_labels == 0, -5 + 1.0 * basis[:, 1], 5 - 1.0 * basis[:, 1]
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
            propensity_train=pi_train,
            X_test=X_test,
            Z_test=Z_test,
            propensity_test=pi_test,
            rfx_group_ids_train=group_labels_train,
            rfx_basis_train=rfx_basis_train,
            rfx_group_ids_test=group_labels_test,
            rfx_basis_test=rfx_basis_test,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
            general_params=general_params,
        )

        # Specify scalar rfx parameters
        rfx_params = {
            "working_parameter_prior_mean": 1.0,
            "group_parameter_prior_mean": 1.0,
            "working_parameter_prior_cov": 1.0,
            "group_parameter_prior_cov": 1.0,
            "variance_prior_shape": 1,
            "variance_prior_scale": 1,
        }
        bcf_model_2 = BCFModel()
        bcf_model_2.sample(
            X_train=X_train,
            Z_train=Z_train,
            y_train=y_train,
            propensity_train=pi_train,
            X_test=X_test,
            Z_test=Z_test,
            propensity_test=pi_test,
            rfx_group_ids_train=group_labels_train,
            rfx_basis_train=rfx_basis_train,
            rfx_group_ids_test=group_labels_test,
            rfx_basis_test=rfx_basis_test,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
            random_effects_params=rfx_params,
        )

        # Specify all relevant rfx parameters as vectors
        rfx_params = {
            "working_parameter_prior_mean": np.repeat(1.0, num_rfx_basis),
            "group_parameter_prior_mean": np.repeat(1.0, num_rfx_basis),
            "working_parameter_prior_cov": np.identity(num_rfx_basis),
            "group_parameter_prior_cov": np.identity(num_rfx_basis),
            "variance_prior_shape": 1,
            "variance_prior_scale": 1,
        }
        bcf_model_3 = BCFModel()
        bcf_model_3.sample(
            X_train=X_train,
            Z_train=Z_train,
            y_train=y_train,
            propensity_train=pi_train,
            X_test=X_test,
            Z_test=Z_test,
            propensity_test=pi_test,
            rfx_group_ids_train=group_labels_train,
            rfx_basis_train=rfx_basis_train,
            rfx_group_ids_test=group_labels_test,
            rfx_basis_test=rfx_basis_test,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
            random_effects_params=rfx_params,
        )

    def test_internal_propensity_with_categorical_dataframe(self):
        # When X is a DataFrame with categorical columns, preprocessing changes
        # the column structure (one-hot encoding). The internal BART propensity
        # model must receive the preprocessed X_test, not the raw original.
        rng = np.random.default_rng(42)
        n = 100
        p = 4
        X_num = rng.uniform(0, 1, (n, p))
        X_cat = rng.choice(["a", "b", "c"], size=n)
        X = pd.DataFrame(X_num, columns=[f"x{i}" for i in range(p)])
        X["cat"] = pd.Categorical(X_cat)

        pi_X = 0.25 + 0.5 * X_num[:, 0]
        Z = rng.binomial(1, pi_X, n).astype(float)
        y = X_num[:, 0] * 2 + X_num[:, 1] * Z + rng.normal(0, 1, n)

        train_inds, test_inds = train_test_split(np.arange(n), test_size=0.2, random_state=0)
        X_train = X.iloc[train_inds].reset_index(drop=True)
        X_test = X.iloc[test_inds].reset_index(drop=True)
        Z_train = Z[train_inds]
        Z_test = Z[test_inds]
        y_train = y[train_inds]

        # No propensity_train provided — triggers internal BART propensity model.
        # Python uses preprocessed covariates correctly, so this should not raise.
        bcf_model = BCFModel()
        bcf_model.sample(
            X_train=X_train,
            Z_train=Z_train,
            y_train=y_train,
            X_test=X_test,
            Z_test=Z_test,
            num_gfr=5,
            num_burnin=0,
            num_mcmc=5,
        )
        assert bcf_model.y_hat_train is not None
        assert bcf_model.y_hat_test is not None
        assert bcf_model.tau_hat_train is not None
        assert bcf_model.tau_hat_test is not None

    def test_warmstart_reuses_internal_propensity(self):
        # When a BCF model fitted without user-supplied propensities is used to
        # warm-start a new run, the second run should reuse the internal
        # propensity model rather than re-fitting it from scratch.
        rng = np.random.default_rng(7)
        n = 100
        X = rng.uniform(0, 1, (n, 5))
        pi_X = 0.25 + 0.5 * X[:, 0]
        Z = rng.binomial(1, pi_X, n).astype(float)
        y = pi_X * 3 + X[:, 1] * Z + rng.normal(0, 1, n)
        n_train = 80
        X_train, X_test = X[:n_train], X[n_train:]
        Z_train, Z_test = Z[:n_train], Z[n_train:]
        y_train = y[:n_train]

        # Fit first model — no propensity provided, so internal model is fitted
        m1 = BCFModel()
        m1.sample(
            X_train=X_train, Z_train=Z_train, y_train=y_train,
            X_test=X_test, Z_test=Z_test,
            num_gfr=5, num_burnin=0, num_mcmc=10,
            general_params={"random_seed": 1},
        )
        assert m1.internal_propensity_model
        # Propensity predictions from the first model (via predict, which is
        # what the warm-start path uses after JSON round-trip)
        pi_train_m1 = m1.bart_propensity_model.predict(X=X_train, terms="y_hat", type="mean")

        # Warm-start a second model from the first — propensity should be reused
        m2 = BCFModel()
        m2.sample(
            X_train=X_train, Z_train=Z_train, y_train=y_train,
            X_test=X_test, Z_test=Z_test,
            num_gfr=0, num_burnin=0, num_mcmc=10,
            previous_model_json=m1.to_json(),
            previous_model_warmstart_sample_num=9,
            general_params={"random_seed": 2},
        )
        assert m2.internal_propensity_model
        # Propensities used in m2 should match those from the reused propensity model
        pi_train_m2 = m2.bart_propensity_model.predict(X_train, terms="y_hat", type="mean")
        np.testing.assert_array_equal(pi_train_m1, pi_train_m2)
        # Output shapes should be correct
        assert m2.y_hat_train.shape == (n_train, 10)
        assert m2.y_hat_test.shape == (n - n_train, 10)


class TestBCFFloat32:
    """Tests that float32 inputs are accepted for all BCF array parameters (GH #389)."""

    def setup_method(self):
        rng = np.random.default_rng(42)
        n = 200
        p_X = 5
        X = rng.uniform(0, 1, (n, p_X)).astype(np.float32)
        pi_X = (0.25 + 0.5 * X[:, 0]).astype(np.float32)
        Z = rng.binomial(1, pi_X.astype(float), n).astype(np.float32)
        y = (pi_X * 3 + Z * 0.5 + rng.normal(0, 0.3, n)).astype(np.float32)
        idx = np.arange(n)
        train, test = idx[:150], idx[150:]
        self.X_train = X[train]
        self.X_test = X[test]
        self.Z_train = Z[train]
        self.Z_test = Z[test]
        self.y_train = y[train]
        self.pi_train = pi_X[train]
        self.pi_test = pi_X[test]
        self.n_train = 150
        self.n_test = 50
        self.num_mcmc = 10

    def test_bcf_float32_with_propensity(self):
        """float32 Z, y, X, and explicit propensity."""
        bcf_model = BCFModel()
        bcf_model.sample(
            X_train=self.X_train,
            Z_train=self.Z_train,
            y_train=self.y_train,
            propensity_train=self.pi_train,
            X_test=self.X_test,
            Z_test=self.Z_test,
            propensity_test=self.pi_test,
            num_gfr=5,
            num_burnin=0,
            num_mcmc=self.num_mcmc,
        )
        assert bcf_model.y_hat_train.shape == (self.n_train, self.num_mcmc)
        assert bcf_model.y_hat_test.shape == (self.n_test, self.num_mcmc)
        assert bcf_model.tau_hat_train.shape == (self.n_train, self.num_mcmc)
        assert bcf_model.tau_hat_test.shape == (self.n_test, self.num_mcmc)
        preds = bcf_model.predict(X=self.X_test, Z=self.Z_test, propensity=self.pi_test)
        assert preds["y_hat"].shape == (self.n_test, self.num_mcmc)
        assert preds["tau_hat"].shape == (self.n_test, self.num_mcmc)

    def test_bcf_float32_with_propensity_matches_float64(self):
        common = dict(num_gfr=5, num_burnin=0, num_mcmc=self.num_mcmc, general_params={"random_seed": 1})
        bcf32 = BCFModel()
        bcf32.sample(X_train=self.X_train, Z_train=self.Z_train, y_train=self.y_train,
                     propensity_train=self.pi_train, X_test=self.X_test,
                     Z_test=self.Z_test, propensity_test=self.pi_test, **common)
        bcf64 = BCFModel()
        bcf64.sample(X_train=self.X_train.astype(np.float64),
                     Z_train=self.Z_train.astype(np.float64),
                     y_train=self.y_train.astype(np.float64),
                     propensity_train=self.pi_train.astype(np.float64),
                     X_test=self.X_test.astype(np.float64),
                     Z_test=self.Z_test.astype(np.float64),
                     propensity_test=self.pi_test.astype(np.float64), **common)
        np.testing.assert_allclose(bcf32.y_hat_train, bcf64.y_hat_train, rtol=1e-4, atol=1e-6)
        pred32 = bcf32.predict(X=self.X_test, Z=self.Z_test, propensity=self.pi_test)
        pred64 = bcf32.predict(X=self.X_test.astype(np.float64), Z=self.Z_test.astype(np.float64),
                               propensity=self.pi_test.astype(np.float64))
        np.testing.assert_allclose(pred32["y_hat"], pred64["y_hat"], rtol=1e-4, atol=1e-6)

    def test_bcf_float32_no_propensity(self):
        """float32 Z, y, X with internal propensity estimation."""
        bcf_model = BCFModel()
        bcf_model.sample(
            X_train=self.X_train,
            Z_train=self.Z_train,
            y_train=self.y_train,
            X_test=self.X_test,
            Z_test=self.Z_test,
            num_gfr=5,
            num_burnin=0,
            num_mcmc=self.num_mcmc,
        )
        assert bcf_model.y_hat_train.shape == (self.n_train, self.num_mcmc)
        assert bcf_model.y_hat_test.shape == (self.n_test, self.num_mcmc)
        preds = bcf_model.predict(X=self.X_test, Z=self.Z_test)
        assert preds["y_hat"].shape == (self.n_test, self.num_mcmc)

    def test_bcf_float32_no_propensity_matches_float64(self):
        common = dict(num_gfr=5, num_burnin=0, num_mcmc=self.num_mcmc, general_params={"random_seed": 1})
        bcf32 = BCFModel()
        bcf32.sample(X_train=self.X_train, Z_train=self.Z_train, y_train=self.y_train,
                     X_test=self.X_test, Z_test=self.Z_test, **common)
        bcf64 = BCFModel()
        bcf64.sample(X_train=self.X_train.astype(np.float64),
                     Z_train=self.Z_train.astype(np.float64),
                     y_train=self.y_train.astype(np.float64),
                     X_test=self.X_test.astype(np.float64),
                     Z_test=self.Z_test.astype(np.float64), **common)
        np.testing.assert_allclose(bcf32.y_hat_train, bcf64.y_hat_train, rtol=1e-4, atol=1e-6)
        pred32 = bcf32.predict(X=self.X_test, Z=self.Z_test)
        pred64 = bcf32.predict(X=self.X_test.astype(np.float64), Z=self.Z_test.astype(np.float64))
        np.testing.assert_allclose(pred32["y_hat"], pred64["y_hat"], rtol=1e-4, atol=1e-6)

    def test_bcf_float32_rfx(self):
        """float32 rfx_basis_train and rfx_basis_test."""
        rng = np.random.default_rng(7)
        n_groups = 4
        group_ids_train = rng.integers(0, n_groups, self.n_train)
        group_ids_test = rng.integers(0, n_groups, self.n_test)
        rfx_basis_train = np.ones((self.n_train, 1), dtype=np.float32)
        rfx_basis_test = np.ones((self.n_test, 1), dtype=np.float32)
        bcf_model = BCFModel()
        bcf_model.sample(
            X_train=self.X_train,
            Z_train=self.Z_train,
            y_train=self.y_train,
            propensity_train=self.pi_train,
            X_test=self.X_test,
            Z_test=self.Z_test,
            propensity_test=self.pi_test,
            rfx_group_ids_train=group_ids_train,
            rfx_basis_train=rfx_basis_train,
            rfx_group_ids_test=group_ids_test,
            rfx_basis_test=rfx_basis_test,
            num_gfr=5,
            num_burnin=0,
            num_mcmc=self.num_mcmc,
        )
        assert bcf_model.y_hat_train.shape == (self.n_train, self.num_mcmc)
        assert bcf_model.y_hat_test.shape == (self.n_test, self.num_mcmc)
        preds = bcf_model.predict(X=self.X_test, Z=self.Z_test, propensity=self.pi_test,
                                   rfx_group_ids=group_ids_test, rfx_basis=rfx_basis_test)
        assert preds["y_hat"].shape == (self.n_test, self.num_mcmc)

    def test_bcf_float32_rfx_matches_float64(self):
        rng = np.random.default_rng(7)
        n_groups = 4
        group_ids_train = rng.integers(0, n_groups, self.n_train)
        group_ids_test = rng.integers(0, n_groups, self.n_test)
        rfx_basis_train = np.ones((self.n_train, 1), dtype=np.float32)
        rfx_basis_test = np.ones((self.n_test, 1), dtype=np.float32)
        common = dict(propensity_train=self.pi_train, propensity_test=self.pi_test,
                      rfx_group_ids_train=group_ids_train, rfx_group_ids_test=group_ids_test,
                      num_gfr=5, num_burnin=0, num_mcmc=self.num_mcmc, general_params={"random_seed": 1})
        bcf32 = BCFModel()
        bcf32.sample(X_train=self.X_train, Z_train=self.Z_train, y_train=self.y_train,
                     X_test=self.X_test, Z_test=self.Z_test,
                     rfx_basis_train=rfx_basis_train, rfx_basis_test=rfx_basis_test, **common)
        bcf64 = BCFModel()
        bcf64.sample(X_train=self.X_train.astype(np.float64),
                     Z_train=self.Z_train.astype(np.float64),
                     y_train=self.y_train.astype(np.float64),
                     X_test=self.X_test.astype(np.float64),
                     Z_test=self.Z_test.astype(np.float64),
                     rfx_basis_train=rfx_basis_train.astype(np.float64),
                     rfx_basis_test=rfx_basis_test.astype(np.float64), **common)
        np.testing.assert_allclose(bcf32.y_hat_train, bcf64.y_hat_train, rtol=1e-4, atol=1e-6)
        pred32 = bcf32.predict(X=self.X_test, Z=self.Z_test, propensity=self.pi_test,
                               rfx_group_ids=group_ids_test, rfx_basis=rfx_basis_test)
        pred64 = bcf32.predict(X=self.X_test.astype(np.float64), Z=self.Z_test.astype(np.float64),
                               propensity=self.pi_test.astype(np.float64),
                               rfx_group_ids=group_ids_test, rfx_basis=rfx_basis_test.astype(np.float64))
        np.testing.assert_allclose(pred32["y_hat"], pred64["y_hat"], rtol=1e-4, atol=1e-6)


class TestBCFUnstandardized:
    def test_bcf_unstandardized_nonzero_mean(self):
        rng = np.random.default_rng(0)
        n = 150
        X = rng.uniform(size=(n, 2))
        pi = 0.3 + 0.4 * X[:, 0]
        Z = rng.binomial(1, pi).astype(float)
        y = 2 * X[:, 1] + Z + rng.normal(scale=0.5, size=n)
        y_shifted = y + 10.0

        m_std = BCFModel()
        m_std.sample(
            X_train=X,
            Z_train=Z,
            y_train=y_shifted,
            propensity_train=pi,
            num_gfr=5,
            num_burnin=10,
            num_mcmc=20,
            general_params={"standardize": True, "random_seed": 1},
        )
        m_unstd = BCFModel()
        m_unstd.sample(
            X_train=X,
            Z_train=Z,
            y_train=y_shifted,
            propensity_train=pi,
            num_gfr=5,
            num_burnin=10,
            num_mcmc=20,
            general_params={"standardize": False, "random_seed": 1},
        )

        rmse_unstd = np.sqrt(np.mean((m_unstd.y_hat_train.mean(axis=1) - y_shifted) ** 2))
        assert rmse_unstd < 0.75
        np.testing.assert_allclose(
            m_unstd.tau_hat_train.mean(),
            m_std.tau_hat_train.mean(),
            atol=0.25,
        )
        np.testing.assert_allclose(
            m_unstd.global_var_samples.mean(),
            m_std.global_var_samples.mean(),
            rtol=0.35,
        )


class TestBCFBooleanTreatment:
    def test_bcf_bool_treatment_matches_float64(self):
        rng = np.random.default_rng(0)
        n_train, n_test = 150, 30
        n = n_train + n_test
        X = rng.uniform(size=(n, 3))
        pi = 0.3 + 0.4 * X[:, 1]
        Z_bool = rng.uniform(size=n) < pi
        Z_float = Z_bool.astype(np.float64)
        tau = 1.0 + X[:, 2]
        y = X[:, 0] + tau * Z_float + rng.normal(scale=0.2, size=n)

        X_tr, X_te = X[:n_train], X[n_train:]
        pi_tr, pi_te = pi[:n_train], pi[n_train:]
        y_tr = y[:n_train]
        Z_bool_tr, Z_bool_te = Z_bool[:n_train], Z_bool[n_train:]
        Z_float_tr, Z_float_te = Z_float[:n_train], Z_float[n_train:]

        common = dict(
            X_train=X_tr,
            y_train=y_tr,
            propensity_train=pi_tr,
            X_test=X_te,
            propensity_test=pi_te,
            num_gfr=5,
            num_burnin=5,
            num_mcmc=10,
            general_params={"random_seed": 1},
        )
        m_bool = BCFModel()
        m_bool.sample(Z_train=Z_bool_tr, Z_test=Z_bool_te, **common)

        m_float = BCFModel()
        m_float.sample(Z_train=Z_float_tr, Z_test=Z_float_te, **common)

        np.testing.assert_array_equal(m_bool.tau_0_samples, m_float.tau_0_samples)
        np.testing.assert_array_equal(m_bool.tau_hat_train, m_float.tau_hat_train)
        np.testing.assert_array_equal(m_bool.y_hat_train, m_float.y_hat_train)
        np.testing.assert_array_equal(m_bool.y_hat_test, m_float.y_hat_test)

        pred_bool = m_bool.predict(X=X_te, Z=Z_bool_te, propensity=pi_te)
        pred_float = m_float.predict(X=X_te, Z=Z_float_te, propensity=pi_te)
        np.testing.assert_array_equal(pred_bool["y_hat"], pred_float["y_hat"])
        np.testing.assert_array_equal(pred_bool["tau_hat"], pred_float["tau_hat"])

    def test_bcf_bool_treatment_internal_propensity_and_multivariate(self):
        rng = np.random.default_rng(42)
        n = 120
        X = rng.uniform(size=(n, 3))
        pi = 0.3 + 0.4 * X[:, 0]
        Z_bool = rng.uniform(size=n) < pi
        y = X[:, 1] + 1.5 * Z_bool.astype(float) + rng.normal(scale=0.3, size=n)

        # Internal propensity model (propensity_train=None) with boolean Z_train
        m_int_bool = BCFModel()
        m_int_bool.sample(
            X_train=X, Z_train=Z_bool, y_train=y,
            num_gfr=5, num_burnin=0, num_mcmc=10,
            general_params={"random_seed": 7},
        )
        m_int_float = BCFModel()
        m_int_float.sample(
            X_train=X, Z_train=Z_bool.astype(np.float64), y_train=y,
            num_gfr=5, num_burnin=0, num_mcmc=10,
            general_params={"random_seed": 7},
        )
        np.testing.assert_array_equal(m_int_bool.tau_hat_train, m_int_float.tau_hat_train)

        # Multivariate boolean Z_train
        Z2_bool = np.column_stack([Z_bool, rng.uniform(size=n) < 0.5])
        y2 = X[:, 1] + Z2_bool[:, 0].astype(float) - 0.5 * Z2_bool[:, 1].astype(float) + rng.normal(scale=0.3, size=n)
        m_mv_bool = BCFModel()
        m_mv_bool.sample(
            X_train=X, Z_train=Z2_bool, y_train=y2,
            num_gfr=5, num_burnin=0, num_mcmc=10,
            general_params={"random_seed": 11, "propensity_covariate": "none"},
        )
        m_mv_float = BCFModel()
        m_mv_float.sample(
            X_train=X, Z_train=Z2_bool.astype(np.float64), y_train=y2,
            num_gfr=5, num_burnin=0, num_mcmc=10,
            general_params={"random_seed": 11, "propensity_covariate": "none"},
        )
        np.testing.assert_array_equal(m_mv_bool.tau_hat_train, m_mv_float.tau_hat_train)
