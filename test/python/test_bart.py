import numpy as np
from sklearn.model_selection import train_test_split

from stochtree import BARTModel


class TestBART:
    def test_bart_constant_leaf_homoskedastic(self):
        # RNG
        random_seed = 101
        rng = np.random.default_rng(random_seed)

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

        # BART settings
        num_gfr = 10
        num_burnin = 0
        num_mcmc = 10

        # Run BART with test set and propensity score
        bart_model = BARTModel()
        bart_model.sample(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
        )

        # Assertions
        assert bart_model.y_hat_train.shape == (n_train, num_mcmc)
        assert bart_model.y_hat_test.shape == (n_test, num_mcmc)

        # Run second BART model
        bart_model_2 = BARTModel()
        bart_model_2.sample(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
        )

        # Assertions
        assert bart_model_2.y_hat_train.shape == (n_train, num_mcmc)
        assert bart_model_2.y_hat_test.shape == (n_test, num_mcmc)

        # Combine into a single model
        bart_models_json = [bart_model.to_json(), bart_model_2.to_json()]
        bart_model_3 = BARTModel()
        bart_model_3.from_json_string_list(bart_models_json)

        # Assertions
        y_hat_train_combined = bart_model_3.predict(covariates=X_train)
        assert y_hat_train_combined.shape == (n_train, num_mcmc * 2)
        np.testing.assert_allclose(
            y_hat_train_combined[:, 0:num_mcmc], bart_model.y_hat_train
        )
        np.testing.assert_allclose(
            y_hat_train_combined[:, num_mcmc : (2 * num_mcmc)], bart_model_2.y_hat_train
        )
        np.testing.assert_allclose(
            bart_model_3.global_var_samples[0:num_mcmc], bart_model.global_var_samples
        )
        np.testing.assert_allclose(
            bart_model_3.global_var_samples[num_mcmc : (2 * num_mcmc)],
            bart_model_2.global_var_samples,
        )

    def test_bart_univariate_leaf_regression_homoskedastic(self):
        # RNG
        random_seed = 101
        rng = np.random.default_rng(random_seed)

        # Generate covariates and basis
        n = 100
        p_X = 10
        p_W = 1
        X = rng.uniform(0, 1, (n, p_X))
        W = rng.uniform(0, 1, (n, p_W))

        # Define the outcome mean function
        def outcome_mean(X, W):
            return np.where(
                (X[:, 0] >= 0.0) & (X[:, 0] < 0.25),
                -7.5 * W[:, 0],
                np.where(
                    (X[:, 0] >= 0.25) & (X[:, 0] < 0.5),
                    -2.5 * W[:, 0],
                    np.where(
                        (X[:, 0] >= 0.5) & (X[:, 0] < 0.75),
                        2.5 * W[:, 0],
                        7.5 * W[:, 0],
                    ),
                ),
            )

        # Generate outcome
        epsilon = rng.normal(0, 1, n)
        y = outcome_mean(X, W) + epsilon

        # Test-train split
        sample_inds = np.arange(n)
        train_inds, test_inds = train_test_split(sample_inds, test_size=0.5)
        X_train = X[train_inds, :]
        X_test = X[test_inds, :]
        basis_train = W[train_inds, :]
        basis_test = W[test_inds, :]
        y_train = y[train_inds]
        # y_test = y[test_inds]
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        # BART settings
        num_gfr = 10
        num_burnin = 0
        num_mcmc = 10

        # Run BART with test set and propensity score
        bart_model = BARTModel()
        bart_model.sample(
            X_train=X_train,
            y_train=y_train,
            leaf_basis_train=basis_train,
            X_test=X_test,
            leaf_basis_test=basis_test,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
        )

        # Assertions
        assert bart_model.y_hat_train.shape == (n_train, num_mcmc)
        assert bart_model.y_hat_test.shape == (n_test, num_mcmc)

        # Run second BART model
        bart_model_2 = BARTModel()
        bart_model_2.sample(
            X_train=X_train,
            y_train=y_train,
            leaf_basis_train=basis_train,
            X_test=X_test,
            leaf_basis_test=basis_test,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
        )

        # Assertions
        assert bart_model_2.y_hat_train.shape == (n_train, num_mcmc)
        assert bart_model_2.y_hat_test.shape == (n_test, num_mcmc)

        # Combine into a single model
        bart_models_json = [bart_model.to_json(), bart_model_2.to_json()]
        bart_model_3 = BARTModel()
        bart_model_3.from_json_string_list(bart_models_json)

        # Assertions
        y_hat_train_combined = bart_model_3.predict(
            covariates=X_train, basis=basis_train
        )
        assert y_hat_train_combined.shape == (n_train, num_mcmc * 2)
        np.testing.assert_allclose(
            y_hat_train_combined[:, 0:num_mcmc], bart_model.y_hat_train
        )
        np.testing.assert_allclose(
            y_hat_train_combined[:, num_mcmc : (2 * num_mcmc)], bart_model_2.y_hat_train
        )
        np.testing.assert_allclose(
            bart_model_3.global_var_samples[0:num_mcmc], bart_model.global_var_samples
        )
        np.testing.assert_allclose(
            bart_model_3.global_var_samples[num_mcmc : (2 * num_mcmc)],
            bart_model_2.global_var_samples,
        )

    def test_bart_multivariate_leaf_regression_homoskedastic(self):
        # RNG
        random_seed = 101
        rng = np.random.default_rng(random_seed)

        # Generate covariates and basis
        n = 100
        p_X = 10
        p_W = 5
        X = rng.uniform(0, 1, (n, p_X))
        W = rng.uniform(0, 1, (n, p_W))

        # Define the outcome mean function
        def outcome_mean(X, W):
            return np.where(
                (X[:, 0] >= 0.0) & (X[:, 0] < 0.25),
                -7.5 * W[:, 0],
                np.where(
                    (X[:, 0] >= 0.25) & (X[:, 0] < 0.5),
                    -2.5 * W[:, 0],
                    np.where(
                        (X[:, 0] >= 0.5) & (X[:, 0] < 0.75),
                        2.5 * W[:, 0],
                        7.5 * W[:, 0],
                    ),
                ),
            )

        # Generate outcome
        epsilon = rng.normal(0, 1, n)
        y = outcome_mean(X, W) + epsilon

        # Test-train split
        sample_inds = np.arange(n)
        train_inds, test_inds = train_test_split(sample_inds, test_size=0.5)
        X_train = X[train_inds, :]
        X_test = X[test_inds, :]
        basis_train = W[train_inds, :]
        basis_test = W[test_inds, :]
        y_train = y[train_inds]
        # y_test = y[test_inds]
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        # BART settings
        num_gfr = 10
        num_burnin = 0
        num_mcmc = 10

        # Run BART with test set and propensity score
        bart_model = BARTModel()
        bart_model.sample(
            X_train=X_train,
            y_train=y_train,
            leaf_basis_train=basis_train,
            X_test=X_test,
            leaf_basis_test=basis_test,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
        )

        # Assertions
        assert bart_model.y_hat_train.shape == (n_train, num_mcmc)
        assert bart_model.y_hat_test.shape == (n_test, num_mcmc)

        # Run second BART model
        bart_model_2 = BARTModel()
        bart_model_2.sample(
            X_train=X_train,
            y_train=y_train,
            leaf_basis_train=basis_train,
            X_test=X_test,
            leaf_basis_test=basis_test,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
        )

        # Assertions
        assert bart_model_2.y_hat_train.shape == (n_train, num_mcmc)
        assert bart_model_2.y_hat_test.shape == (n_test, num_mcmc)

        # Combine into a single model
        bart_models_json = [bart_model.to_json(), bart_model_2.to_json()]
        bart_model_3 = BARTModel()
        bart_model_3.from_json_string_list(bart_models_json)

        # Assertions
        y_hat_train_combined = bart_model_3.predict(
            covariates=X_train, basis=basis_train
        )
        assert y_hat_train_combined.shape == (n_train, num_mcmc * 2)
        np.testing.assert_allclose(
            y_hat_train_combined[:, 0:num_mcmc], bart_model.y_hat_train
        )
        np.testing.assert_allclose(
            y_hat_train_combined[:, num_mcmc : (2 * num_mcmc)], bart_model_2.y_hat_train
        )
        np.testing.assert_allclose(
            bart_model_3.global_var_samples[0:num_mcmc], bart_model.global_var_samples
        )
        np.testing.assert_allclose(
            bart_model_3.global_var_samples[num_mcmc : (2 * num_mcmc)],
            bart_model_2.global_var_samples,
        )

    def test_bart_constant_leaf_heteroskedastic(self):
        # RNG
        random_seed = 101
        rng = np.random.default_rng(random_seed)

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

        # Define the conditional standard deviation function
        def conditional_stddev(X):
            return np.where(
                (X[:, 0] >= 0.0) & (X[:, 0] < 0.25),
                0.25,
                np.where(
                    (X[:, 0] >= 0.25) & (X[:, 0] < 0.5),
                    0.5,
                    np.where((X[:, 0] >= 0.5) & (X[:, 0] < 0.75), 1, 2),
                ),
            )

        # Generate outcome
        epsilon = rng.normal(0, 1, n)
        y = outcome_mean(X) + epsilon * conditional_stddev(X)

        # Test-train split
        sample_inds = np.arange(n)
        train_inds, test_inds = train_test_split(sample_inds, test_size=0.5)
        X_train = X[train_inds, :]
        X_test = X[test_inds, :]
        y_train = y[train_inds]
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        # BART settings
        num_gfr = 10
        num_burnin = 0
        num_mcmc = 10

        # Run BART with test set and propensity score
        bart_model = BARTModel()
        general_params = {"sample_sigma2_global": True}
        variance_forest_params = {"num_trees": 50}
        bart_model.sample(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            general_params=general_params,
            variance_forest_params=variance_forest_params,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
        )

        # Assertions
        assert bart_model.y_hat_train.shape == (n_train, num_mcmc)
        assert bart_model.y_hat_test.shape == (n_test, num_mcmc)

        # Run second BART model
        bart_model_2 = BARTModel()
        bart_model_2.sample(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            general_params=general_params,
            variance_forest_params=variance_forest_params,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
        )

        # Assertions
        assert bart_model_2.y_hat_train.shape == (n_train, num_mcmc)
        assert bart_model_2.y_hat_test.shape == (n_test, num_mcmc)

        # Combine into a single model
        bart_models_json = [bart_model.to_json(), bart_model_2.to_json()]
        bart_model_3 = BARTModel()
        bart_model_3.from_json_string_list(bart_models_json)

        # Assertions
        y_hat_train_combined, _ = bart_model_3.predict(covariates=X_train)
        assert y_hat_train_combined.shape == (n_train, num_mcmc * 2)
        np.testing.assert_allclose(
            y_hat_train_combined[:, 0:num_mcmc], bart_model.y_hat_train
        )
        np.testing.assert_allclose(
            y_hat_train_combined[:, num_mcmc : (2 * num_mcmc)], bart_model_2.y_hat_train
        )
        np.testing.assert_allclose(
            bart_model_3.global_var_samples[0:num_mcmc], bart_model.global_var_samples
        )
        np.testing.assert_allclose(
            bart_model_3.global_var_samples[num_mcmc : (2 * num_mcmc)],
            bart_model_2.global_var_samples,
        )

    def test_bart_univariate_leaf_regression_heteroskedastic(self):
        # RNG
        random_seed = 101
        rng = np.random.default_rng(random_seed)

        # Generate covariates and basis
        n = 100
        p_X = 10
        p_W = 1
        X = rng.uniform(0, 1, (n, p_X))
        W = rng.uniform(0, 1, (n, p_W))

        # Define the outcome mean function
        def outcome_mean(X, W):
            return np.where(
                (X[:, 0] >= 0.0) & (X[:, 0] < 0.25),
                -7.5 * W[:, 0],
                np.where(
                    (X[:, 0] >= 0.25) & (X[:, 0] < 0.5),
                    -2.5 * W[:, 0],
                    np.where(
                        (X[:, 0] >= 0.5) & (X[:, 0] < 0.75),
                        2.5 * W[:, 0],
                        7.5 * W[:, 0],
                    ),
                ),
            )

        # Define the conditional standard deviation function
        def conditional_stddev(X):
            return np.where(
                (X[:, 0] >= 0.0) & (X[:, 0] < 0.25),
                0.25,
                np.where(
                    (X[:, 0] >= 0.25) & (X[:, 0] < 0.5),
                    0.5,
                    np.where((X[:, 0] >= 0.5) & (X[:, 0] < 0.75), 1, 2),
                ),
            )

        # Generate outcome
        epsilon = rng.normal(0, 1, n)
        y = outcome_mean(X, W) + epsilon * conditional_stddev(X)

        # Test-train split
        sample_inds = np.arange(n)
        train_inds, test_inds = train_test_split(sample_inds, test_size=0.5)
        X_train = X[train_inds, :]
        X_test = X[test_inds, :]
        basis_train = W[train_inds, :]
        basis_test = W[test_inds, :]
        y_train = y[train_inds]
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        # BART settings
        num_gfr = 10
        num_burnin = 0
        num_mcmc = 10

        # Run BART with test set and propensity score
        bart_model = BARTModel()
        general_params = {"sample_sigma2_global": True}
        variance_forest_params = {"num_trees": 50}
        bart_model.sample(
            X_train=X_train,
            y_train=y_train,
            leaf_basis_train=basis_train,
            X_test=X_test,
            leaf_basis_test=basis_test,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
            general_params=general_params,
            variance_forest_params=variance_forest_params,
        )

        # Assertions
        assert bart_model.y_hat_train.shape == (n_train, num_mcmc)
        assert bart_model.y_hat_test.shape == (n_test, num_mcmc)

        # Run second BART model
        bart_model_2 = BARTModel()
        bart_model_2.sample(
            X_train=X_train,
            y_train=y_train,
            leaf_basis_train=basis_train,
            X_test=X_test,
            leaf_basis_test=basis_test,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
            general_params=general_params,
            variance_forest_params=variance_forest_params,
        )

        # Assertions
        assert bart_model_2.y_hat_train.shape == (n_train, num_mcmc)
        assert bart_model_2.y_hat_test.shape == (n_test, num_mcmc)

        # Combine into a single model
        bart_models_json = [bart_model.to_json(), bart_model_2.to_json()]
        bart_model_3 = BARTModel()
        bart_model_3.from_json_string_list(bart_models_json)

        # Assertions
        y_hat_train_combined, _ = bart_model_3.predict(
            covariates=X_train, basis=basis_train
        )
        assert y_hat_train_combined.shape == (n_train, num_mcmc * 2)
        np.testing.assert_allclose(
            y_hat_train_combined[:, 0:num_mcmc], bart_model.y_hat_train
        )
        np.testing.assert_allclose(
            y_hat_train_combined[:, num_mcmc : (2 * num_mcmc)], bart_model_2.y_hat_train
        )
        np.testing.assert_allclose(
            bart_model_3.global_var_samples[0:num_mcmc], bart_model.global_var_samples
        )
        np.testing.assert_allclose(
            bart_model_3.global_var_samples[num_mcmc : (2 * num_mcmc)],
            bart_model_2.global_var_samples,
        )

    def test_bart_multivariate_leaf_regression_heteroskedastic(self):
        # RNG
        random_seed = 101
        rng = np.random.default_rng(random_seed)

        # Generate covariates and basis
        n = 100
        p_X = 10
        p_W = 5
        X = rng.uniform(0, 1, (n, p_X))
        W = rng.uniform(0, 1, (n, p_W))

        # Define the outcome mean function
        def outcome_mean(X, W):
            return np.where(
                (X[:, 0] >= 0.0) & (X[:, 0] < 0.25),
                -7.5 * W[:, 0],
                np.where(
                    (X[:, 0] >= 0.25) & (X[:, 0] < 0.5),
                    -2.5 * W[:, 0],
                    np.where(
                        (X[:, 0] >= 0.5) & (X[:, 0] < 0.75),
                        2.5 * W[:, 0],
                        7.5 * W[:, 0],
                    ),
                ),
            )

        # Define the conditional standard deviation function
        def conditional_stddev(X):
            return np.where(
                (X[:, 0] >= 0.0) & (X[:, 0] < 0.25),
                0.25,
                np.where(
                    (X[:, 0] >= 0.25) & (X[:, 0] < 0.5),
                    0.5,
                    np.where((X[:, 0] >= 0.5) & (X[:, 0] < 0.75), 1, 2),
                ),
            )

        # Generate outcome
        epsilon = rng.normal(0, 1, n)
        y = outcome_mean(X, W) + epsilon * conditional_stddev(X)

        # Test-train split
        sample_inds = np.arange(n)
        train_inds, test_inds = train_test_split(sample_inds, test_size=0.5)
        X_train = X[train_inds, :]
        X_test = X[test_inds, :]
        basis_train = W[train_inds, :]
        basis_test = W[test_inds, :]
        y_train = y[train_inds]
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        # BART settings
        num_gfr = 10
        num_burnin = 0
        num_mcmc = 10

        # Run BART with test set and propensity score
        bart_model = BARTModel()
        general_params = {"sample_sigma2_global": True}
        variance_forest_params = {"num_trees": 50}
        bart_model.sample(
            X_train=X_train,
            y_train=y_train,
            leaf_basis_train=basis_train,
            X_test=X_test,
            leaf_basis_test=basis_test,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
            general_params=general_params,
            variance_forest_params=variance_forest_params,
        )

        # Assertions
        assert bart_model.y_hat_train.shape == (n_train, num_mcmc)
        assert bart_model.y_hat_test.shape == (n_test, num_mcmc)

        # Run second BART model
        bart_model_2 = BARTModel()
        bart_model_2.sample(
            X_train=X_train,
            y_train=y_train,
            leaf_basis_train=basis_train,
            X_test=X_test,
            leaf_basis_test=basis_test,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
            general_params=general_params,
            variance_forest_params=variance_forest_params,
        )

        # Assertions
        assert bart_model_2.y_hat_train.shape == (n_train, num_mcmc)
        assert bart_model_2.y_hat_test.shape == (n_test, num_mcmc)

        # Combine into a single model
        bart_models_json = [bart_model.to_json(), bart_model_2.to_json()]
        bart_model_3 = BARTModel()
        bart_model_3.from_json_string_list(bart_models_json)

        # Assertions
        y_hat_train_combined, _ = bart_model_3.predict(
            covariates=X_train, basis=basis_train
        )
        assert y_hat_train_combined.shape == (n_train, num_mcmc * 2)
        np.testing.assert_allclose(
            y_hat_train_combined[:, 0:num_mcmc], bart_model.y_hat_train
        )
        np.testing.assert_allclose(
            y_hat_train_combined[:, num_mcmc : (2 * num_mcmc)], bart_model_2.y_hat_train
        )
        np.testing.assert_allclose(
            bart_model_3.global_var_samples[0:num_mcmc], bart_model.global_var_samples
        )
        np.testing.assert_allclose(
            bart_model_3.global_var_samples[num_mcmc : (2 * num_mcmc)],
            bart_model_2.global_var_samples,
        )

    def test_bart_constant_leaf_heteroskedastic_rfx(self):
        # RNG
        random_seed = 101
        rng = np.random.default_rng(random_seed)

        # Generate covariates and basis
        n = 100
        p_X = 10
        X = rng.uniform(0, 1, (n, p_X))

        # Generate RFX group labels and basis term
        num_rfx_basis = 1
        num_rfx_groups = 4
        group_labels = rng.choice(num_rfx_groups, size=n)
        rfx_basis = np.empty((n, num_rfx_basis))
        rfx_basis[:, 0] = 1.0
        if num_rfx_basis > 1:
            rfx_basis[:, 1:] = rng.uniform(-1, 1, (n, num_rfx_basis - 1))

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

        # Define the conditional standard deviation function
        def conditional_stddev(X):
            return np.where(
                (X[:, 0] >= 0.0) & (X[:, 0] < 0.25),
                0.25,
                np.where(
                    (X[:, 0] >= 0.25) & (X[:, 0] < 0.5),
                    0.5,
                    np.where((X[:, 0] >= 0.5) & (X[:, 0] < 0.75), 1, 2),
                ),
            )

        # Define the group rfx function
        def rfx_term(group_labels, basis):
            return np.where(
                group_labels == 0,
                0,
                np.where(group_labels == 1, 4, np.where(group_labels == 2, 8, 12)),
            )

        # Generate outcome
        epsilon = rng.normal(0, 1, n)
        y = (
            outcome_mean(X)
            + rfx_term(group_labels, rfx_basis)
            + epsilon * conditional_stddev(X)
        )

        # Test-train split
        sample_inds = np.arange(n)
        train_inds, test_inds = train_test_split(sample_inds, test_size=0.5)
        X_train = X[train_inds, :]
        X_test = X[test_inds, :]
        group_labels_train = group_labels[train_inds]
        group_labels_test = group_labels[test_inds]
        rfx_basis_train = rfx_basis[train_inds, :]
        rfx_basis_test = rfx_basis[test_inds, :]
        y_train = y[train_inds]
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        # BART settings
        num_gfr = 10
        num_burnin = 0
        num_mcmc = 10

        # Run BART with test set and propensity score
        bart_model = BARTModel()
        general_params = {"sample_sigma2_global": True}
        variance_forest_params = {"num_trees": 50}
        bart_model.sample(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            rfx_group_ids_train=group_labels_train,
            rfx_basis_train=rfx_basis_train,
            rfx_group_ids_test=group_labels_test,
            rfx_basis_test=rfx_basis_test,
            general_params=general_params,
            variance_forest_params=variance_forest_params,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
        )
        rfx_preds_train = bart_model.rfx_container.predict(
            group_labels_train, rfx_basis_train
        )

        # Assertions
        assert bart_model.y_hat_train.shape == (n_train, num_mcmc)
        assert bart_model.y_hat_test.shape == (n_test, num_mcmc)

        # Run second BART model
        bart_model_2 = BARTModel()
        bart_model_2.sample(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            rfx_group_ids_train=group_labels_train,
            rfx_basis_train=rfx_basis_train,
            rfx_group_ids_test=group_labels_test,
            rfx_basis_test=rfx_basis_test,
            general_params=general_params,
            variance_forest_params=variance_forest_params,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
        )
        rfx_preds_train_2 = bart_model_2.rfx_container.predict(
            group_labels_train, rfx_basis_train
        )

        # Assertions
        assert bart_model_2.y_hat_train.shape == (n_train, num_mcmc)
        assert bart_model_2.y_hat_test.shape == (n_test, num_mcmc)

        # Combine into a single model
        bart_models_json = [bart_model.to_json(), bart_model_2.to_json()]
        bart_model_3 = BARTModel()
        bart_model_3.from_json_string_list(bart_models_json)
        rfx_preds_train_3 = bart_model_3.rfx_container.predict(
            group_labels_train, rfx_basis_train
        )

        # Assertions
        y_hat_train_combined, _ = bart_model_3.predict(
            covariates=X_train,
            rfx_group_ids=group_labels_train,
            rfx_basis=rfx_basis_train,
        )
        assert y_hat_train_combined.shape == (n_train, num_mcmc * 2)
        np.testing.assert_allclose(
            y_hat_train_combined[:, 0:num_mcmc], bart_model.y_hat_train
        )
        np.testing.assert_allclose(
            y_hat_train_combined[:, num_mcmc : (2 * num_mcmc)], bart_model_2.y_hat_train
        )
        np.testing.assert_allclose(
            bart_model_3.global_var_samples[0:num_mcmc], bart_model.global_var_samples
        )
        np.testing.assert_allclose(
            bart_model_3.global_var_samples[num_mcmc : (2 * num_mcmc)],
            bart_model_2.global_var_samples,
        )
        np.testing.assert_allclose(rfx_preds_train_3[:, 0:num_mcmc], rfx_preds_train)
        np.testing.assert_allclose(
            rfx_preds_train_3[:, num_mcmc : (2 * num_mcmc)], rfx_preds_train_2
        )

    def test_bart_univariate_leaf_regression_heteroskedastic_rfx(self):
        # RNG
        random_seed = 101
        rng = np.random.default_rng(random_seed)

        # Generate covariates and basis
        n = 100
        p_X = 10
        p_W = 1
        X = rng.uniform(0, 1, (n, p_X))
        W = rng.uniform(0, 1, (n, p_W))

        # Generate RFX group labels and basis term
        num_rfx_basis = 1
        num_rfx_groups = 4
        group_labels = rng.choice(num_rfx_groups, size=n)
        rfx_basis = np.empty((n, num_rfx_basis))
        rfx_basis[:, 0] = 1.0
        if num_rfx_basis > 1:
            rfx_basis[:, 1:] = rng.uniform(-1, 1, (n, num_rfx_basis - 1))

        # Define the outcome mean function
        def outcome_mean(X, W):
            return np.where(
                (X[:, 0] >= 0.0) & (X[:, 0] < 0.25),
                -7.5 * W[:, 0],
                np.where(
                    (X[:, 0] >= 0.25) & (X[:, 0] < 0.5),
                    -2.5 * W[:, 0],
                    np.where(
                        (X[:, 0] >= 0.5) & (X[:, 0] < 0.75),
                        2.5 * W[:, 0],
                        7.5 * W[:, 0],
                    ),
                ),
            )

        # Define the group rfx function
        def rfx_term(group_labels, basis):
            return np.where(
                group_labels == 0,
                0,
                np.where(group_labels == 1, 4, np.where(group_labels == 2, 8, 12)),
            )

        # Define the conditional standard deviation function
        def conditional_stddev(X):
            return np.where(
                (X[:, 0] >= 0.0) & (X[:, 0] < 0.25),
                0.25,
                np.where(
                    (X[:, 0] >= 0.25) & (X[:, 0] < 0.5),
                    0.5,
                    np.where((X[:, 0] >= 0.5) & (X[:, 0] < 0.75), 1, 2),
                ),
            )

        # Generate outcome
        epsilon = rng.normal(0, 1, n)
        y = (
            outcome_mean(X, W)
            + rfx_term(group_labels, rfx_basis)
            + epsilon * conditional_stddev(X)
        )

        # Test-train split
        sample_inds = np.arange(n)
        train_inds, test_inds = train_test_split(sample_inds, test_size=0.5)
        X_train = X[train_inds, :]
        X_test = X[test_inds, :]
        basis_train = W[train_inds, :]
        basis_test = W[test_inds, :]
        group_labels_train = group_labels[train_inds]
        group_labels_test = group_labels[test_inds]
        rfx_basis_train = rfx_basis[train_inds, :]
        rfx_basis_test = rfx_basis[test_inds, :]
        y_train = y[train_inds]
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        # BART settings
        num_gfr = 10
        num_burnin = 0
        num_mcmc = 10

        # Run BART with test set and propensity score
        bart_model = BARTModel()
        general_params = {"sample_sigma2_global": True}
        variance_forest_params = {"num_trees": 50}
        bart_model.sample(
            X_train=X_train,
            y_train=y_train,
            leaf_basis_train=basis_train,
            rfx_group_ids_train=group_labels_train,
            rfx_basis_train=rfx_basis_train,
            X_test=X_test,
            leaf_basis_test=basis_test,
            rfx_group_ids_test=group_labels_test,
            rfx_basis_test=rfx_basis_test,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
            general_params=general_params,
            variance_forest_params=variance_forest_params,
        )
        rfx_preds_train = bart_model.rfx_container.predict(
            group_labels_train, rfx_basis_train
        )

        # Assertions
        assert bart_model.y_hat_train.shape == (n_train, num_mcmc)
        assert bart_model.y_hat_test.shape == (n_test, num_mcmc)

        # Run second BART model
        bart_model_2 = BARTModel()
        bart_model_2.sample(
            X_train=X_train,
            y_train=y_train,
            leaf_basis_train=basis_train,
            rfx_group_ids_train=group_labels_train,
            rfx_basis_train=rfx_basis_train,
            X_test=X_test,
            leaf_basis_test=basis_test,
            rfx_group_ids_test=group_labels_test,
            rfx_basis_test=rfx_basis_test,
            num_gfr=num_gfr,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
            general_params=general_params,
            variance_forest_params=variance_forest_params,
        )
        rfx_preds_train_2 = bart_model_2.rfx_container.predict(
            group_labels_train, rfx_basis_train
        )

        # Assertions
        assert bart_model_2.y_hat_train.shape == (n_train, num_mcmc)
        assert bart_model_2.y_hat_test.shape == (n_test, num_mcmc)

        # Combine into a single model
        bart_models_json = [bart_model.to_json(), bart_model_2.to_json()]
        bart_model_3 = BARTModel()
        bart_model_3.from_json_string_list(bart_models_json)
        rfx_preds_train_3 = bart_model_3.rfx_container.predict(
            group_labels_train, rfx_basis_train
        )

        # Assertions
        y_hat_train_combined, _ = bart_model_3.predict(
            covariates=X_train,
            basis=basis_train,
            rfx_group_ids=group_labels_train,
            rfx_basis=rfx_basis_train,
        )
        assert y_hat_train_combined.shape == (n_train, num_mcmc * 2)
        np.testing.assert_allclose(
            y_hat_train_combined[:, 0:num_mcmc], bart_model.y_hat_train
        )
        np.testing.assert_allclose(
            y_hat_train_combined[:, num_mcmc : (2 * num_mcmc)], bart_model_2.y_hat_train
        )
        np.testing.assert_allclose(
            bart_model_3.global_var_samples[0:num_mcmc], bart_model.global_var_samples
        )
        np.testing.assert_allclose(
            bart_model_3.global_var_samples[num_mcmc : (2 * num_mcmc)],
            bart_model_2.global_var_samples,
        )
        np.testing.assert_allclose(rfx_preds_train_3[:, 0:num_mcmc], rfx_preds_train)
        np.testing.assert_allclose(
            rfx_preds_train_3[:, num_mcmc : (2 * num_mcmc)], rfx_preds_train_2
        )
