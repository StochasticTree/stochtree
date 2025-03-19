import numpy as np
import pandas as pd

from stochtree import (
    RNG,
    BARTModel,
    BCFModel,
    CovariatePreprocessor,
    Dataset,
    Forest,
    ForestContainer,
    ForestSampler,
    GlobalVarianceModel,
    JSONSerializer,
    Residual,
    ForestModelConfig,
    GlobalModelConfig
)


class TestJson:
    def test_value(self):
        json_test = JSONSerializer()
        a = 1.5
        b = True
        c = "Example"
        json_test.add_scalar("a", a)
        json_test.add_boolean("b", b)
        json_test.add_string("c", c)
        assert a == json_test.get_scalar("a")
        assert b == json_test.get_boolean("b")
        assert c == json_test.get_string("c")

    def test_array(self):
        json_test = JSONSerializer()
        a = np.array([1.5, 2.4, 3.3])
        b = ["a", "b", "c"]
        json_test.add_numeric_vector("a", a)
        json_test.add_string_vector("b", b)
        np.testing.assert_array_equal(a, json_test.get_numeric_vector("a"))
        assert b == json_test.get_string_vector("b")

    def test_preprocessor(self):
        df = pd.DataFrame(
            {
                "x1": [1.5, 2.7, 3.6, 4.4, 5.3, 6.1],
                "x2": pd.Categorical(
                    ["a", "b", "c", "a", "b", "c"],
                    ordered=False,
                    categories=["c", "b", "a"],
                ),
                "x3": [1.2, 5.4, 9.3, 10.4, 3.6, 4.4],
            }
        )
        cov_transformer = CovariatePreprocessor()
        df_transformed_orig = cov_transformer.fit_transform(df)
        cov_transformer_json = cov_transformer.to_json()
        cov_transformer_reloaded = CovariatePreprocessor()
        cov_transformer_reloaded.from_json(cov_transformer_json)
        df_transformed_reloaded = cov_transformer_reloaded.transform(df)
        np.testing.assert_array_equal(df_transformed_orig, df_transformed_reloaded)

        df_2 = pd.DataFrame(
            {
                "x1": [1.5, 2.7, 3.6, 4.4, 5.3, 6.1],
                "x2": pd.Categorical(
                    ["a", "b", "c", "a", "b", "c"],
                    ordered=False,
                    categories=["c", "b", "a"],
                ),
                "x3": pd.Categorical(
                    ["a", "c", "d", "b", "d", "b"],
                    ordered=False,
                    categories=["c", "b", "a", "d"],
                ),
                "x4": pd.Categorical(
                    ["a", "b", "f", "f", "c", "a"],
                    ordered=True,
                    categories=["c", "b", "a", "f"],
                ),
                "x5": [1.2, 5.4, 9.3, 10.4, 3.6, 4.4],
            }
        )
        cov_transformer_2 = CovariatePreprocessor()
        df_transformed_orig_2 = cov_transformer_2.fit_transform(df_2)
        cov_transformer_json_2 = cov_transformer_2.to_json()
        cov_transformer_reloaded_2 = CovariatePreprocessor()
        cov_transformer_reloaded_2.from_json(cov_transformer_json_2)
        df_transformed_reloaded_2 = cov_transformer_reloaded_2.transform(df_2)
        np.testing.assert_array_equal(df_transformed_orig_2, df_transformed_reloaded_2)

        np_3 = np.array(
            [[1.5, 1.2], [2.7, 5.4], [3.6, 9.3], [4.4, 10.4], [5.3, 3.6], [6.1, 4.4]]
        )
        cov_transformer_3 = CovariatePreprocessor()
        df_transformed_orig_3 = cov_transformer_3.fit_transform(np_3)
        cov_transformer_json_3 = cov_transformer_3.to_json()
        cov_transformer_reloaded_3 = CovariatePreprocessor()
        cov_transformer_reloaded_3.from_json(cov_transformer_json_3)
        df_transformed_reloaded_3 = cov_transformer_reloaded_3.transform(np_3)
        np.testing.assert_array_equal(df_transformed_orig_3, df_transformed_reloaded_3)

    def test_forest(self):
        # Generate sample data
        random_seed = 1234
        rng = np.random.default_rng(random_seed)
        n = 1000
        p_X = 10
        X = rng.uniform(0, 1, (n, p_X))

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

        epsilon = rng.normal(0, 1, n)
        y = outcome_mean(X) + epsilon

        # Train a BART model
        bart_model = BARTModel()
        bart_model.sample(X_train=X, y_train=y, num_gfr=10, num_mcmc=10)

        # Extract original predictions
        forest_preds_y_mcmc_cached = bart_model.y_hat_train

        # Extract original predictions
        forest_preds_y_mcmc_retrieved = bart_model.predict(X)

        # Roundtrip to / from JSON
        json_test = JSONSerializer()
        json_test.add_forest(bart_model.forest_container_mean)
        forest_container = json_test.get_forest_container("forest_0")

        # Predict from the deserialized forest container
        forest_dataset = Dataset()
        forest_dataset.add_covariates(X)
        forest_preds_json_reload = forest_container.predict(forest_dataset)
        forest_preds_json_reload = (
            forest_preds_json_reload * bart_model.y_std + bart_model.y_bar
        )
        # Check the predictions
        np.testing.assert_almost_equal(
            forest_preds_y_mcmc_cached, forest_preds_json_reload
        )
        np.testing.assert_almost_equal(
            forest_preds_y_mcmc_retrieved, forest_preds_json_reload
        )

    def test_forest_string(self):
        # RNG
        random_seed = 1234
        rng = np.random.default_rng(random_seed)

        # Generate covariates and basis
        n = 1000
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

        # Standardize outcome
        y_bar = np.mean(y)
        y_std = np.std(y)
        resid = (y - y_bar) / y_std

        # Sampler parameters
        alpha = 0.9
        beta = 1.25
        min_samples_leaf = 1
        num_trees = 100
        cutpoint_grid_size = 100
        global_variance_init = 1.0
        tau_init = 0.5
        leaf_prior_scale = np.array([[tau_init]], order="C")
        a_global = 4.0
        b_global = 2.0
        feature_types = np.repeat(0, p_X).astype(int)  # 0 = numeric
        var_weights = np.repeat(1 / p_X, p_X)

        # Dataset (covariates and basis)
        dataset = Dataset()
        dataset.add_covariates(X)
        dataset.add_basis(W)

        # Residual
        residual = Residual(resid)

        # Forest samplers and temporary tracking data structures
        leaf_model_type = 0 if p_W == 0 else 1 + 1*(p_W > 1)
        forest_config = ForestModelConfig(
            num_trees=num_trees,
            num_features=p_X,
            num_observations=n,
            feature_types=feature_types,
            variable_weights=var_weights,
            leaf_dimension=p_W,
            alpha=alpha,
            beta=beta,
            min_samples_leaf=min_samples_leaf,
            leaf_model_type=leaf_model_type,
            cutpoint_grid_size=cutpoint_grid_size,
            leaf_model_scale=leaf_prior_scale,
        )
        global_config = GlobalModelConfig(global_error_variance=global_variance_init)
        forest_container = ForestContainer(num_trees, W.shape[1], False, False)
        active_forest = Forest(num_trees, W.shape[1], False, False)
        forest_sampler = ForestSampler(
            dataset, global_config, forest_config
        )
        cpp_rng = RNG(random_seed)
        global_var_model = GlobalVarianceModel()

        # Prepare to run sampler
        num_warmstart = 10
        num_mcmc = 100
        num_samples = num_warmstart + num_mcmc
        global_var_samples = np.concatenate(
            (np.array([global_variance_init]), np.repeat(0, num_samples))
        )
        if p_W > 0:
            init_val = np.repeat(0.0, W.shape[1])
        else:
            init_val = np.array([0.0])
        forest_sampler.prepare_for_sampler(
            dataset,
            residual,
            active_forest,
            leaf_model_type,
            init_val,
        )

        # Run "grow-from-root" sampler
        for i in range(num_warmstart):
            forest_sampler.sample_one_iteration(
                forest_container,
                active_forest,
                dataset,
                residual,
                cpp_rng,
                global_config, 
                forest_config, 
                True,
                True,
            )
            global_var_samples[i + 1] = global_var_model.sample_one_iteration(
                residual, cpp_rng, a_global, b_global
            )

        # Run MCMC sampler
        for i in range(num_warmstart, num_samples):
            forest_sampler.sample_one_iteration(
                forest_container,
                active_forest,
                dataset,
                residual,
                cpp_rng,
                global_config, 
                forest_config, 
                True,
                True,
            )
            global_var_samples[i + 1] = global_var_model.sample_one_iteration(
                residual, cpp_rng, a_global, b_global
            )

        # Extract predictions from the sampler
        y_hat_orig = forest_container.predict(dataset)

        # "Round-trip" the forest to JSON string and back and check that the predictions agree
        forest_json_string = forest_container.dump_json_string()
        forest_container_reloaded = ForestContainer(num_trees, W.shape[1], False, False)
        forest_container_reloaded.load_from_json_string(forest_json_string)
        y_hat_reloaded = forest_container_reloaded.predict(dataset)
        np.testing.assert_almost_equal(y_hat_orig, y_hat_reloaded)

    def test_bart_string(self):
        # RNG
        random_seed = 1234
        rng = np.random.default_rng(random_seed)

        # Generate covariates and basis
        n = 1000
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

        # Run BART
        bart_orig = BARTModel()
        bart_orig.sample(X_train=X, y_train=y, leaf_basis_train=W, num_gfr=10, num_mcmc=10)

        # Extract predictions from the sampler
        y_hat_orig = bart_orig.predict(X, W)

        # "Round-trip" the model to JSON string and back and check that the predictions agree
        bart_json_string = bart_orig.to_json()
        bart_reloaded = BARTModel()
        bart_reloaded.from_json(bart_json_string)
        y_hat_reloaded = bart_reloaded.predict(X, W)
        np.testing.assert_almost_equal(y_hat_orig, y_hat_reloaded)

    def test_bart_rfx_string(self):
        # RNG
        random_seed = 1234
        rng = np.random.default_rng(random_seed)

        # Generate covariates and basis
        n = 1000
        p_X = 10
        p_W = 1
        X = rng.uniform(0, 1, (n, p_X))
        W = rng.uniform(0, 1, (n, p_W))

        # Generate random effects terms
        num_basis = 2
        num_groups = 4
        group_labels = rng.choice(num_groups, size=n)
        basis = np.empty((n, num_basis))
        basis[:, 0] = 1.0
        if num_basis > 1:
            basis[:, 1:] = rng.uniform(-1, 1, (n, num_basis - 1))

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
        def rfx_mean(group_labels, basis):
            return np.where(
                group_labels == 0,
                0 - 1 * basis[:, 1],
                np.where(
                    group_labels == 1,
                    4 + 1 * basis[:, 1],
                    np.where(
                        group_labels == 2, 8 + 3 * basis[:, 1], 12 + 5 * basis[:, 1]
                    ),
                ),
            )

        # Generate outcome
        epsilon = rng.normal(0, 1, n)
        forest_term = outcome_mean(X, W)
        rfx_term = rfx_mean(group_labels, basis)
        y = forest_term + rfx_term + epsilon

        # Run BART
        bart_orig = BARTModel()
        bart_orig.sample(X_train=X, y_train=y, leaf_basis_train=W, rfx_group_ids_train=group_labels, 
                         rfx_basis_train=basis, num_gfr=10, num_mcmc=10)

        # Extract predictions from the sampler
        y_hat_orig = bart_orig.predict(X, W, group_labels, basis)

        # "Round-trip" the model to JSON string and back and check that the predictions agree
        bart_json_string = bart_orig.to_json()
        bart_reloaded = BARTModel()
        bart_reloaded.from_json(bart_json_string)
        y_hat_reloaded = bart_reloaded.predict(X, W, group_labels, basis)
        np.testing.assert_almost_equal(y_hat_orig, y_hat_reloaded)

    def test_bcf_string(self):
        # RNG
        random_seed = 1234
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

        # Run BCF
        bcf_orig = BCFModel()
        bcf_orig.sample(
            X_train=X, Z_train=Z, y_train=y, pi_train=pi_X, num_gfr=10, num_mcmc=10
        )

        # Extract predictions from the sampler
        mu_hat_orig, tau_hat_orig, y_hat_orig = bcf_orig.predict(X, Z, pi_X)

        # "Round-trip" the model to JSON string and back and check that the predictions agree
        bcf_json_string = bcf_orig.to_json()
        bcf_reloaded = BCFModel()
        bcf_reloaded.from_json(bcf_json_string)
        mu_hat_reloaded, tau_hat_reloaded, y_hat_reloaded = bcf_reloaded.predict(
            X, Z, pi_X
        )
        np.testing.assert_almost_equal(y_hat_orig, y_hat_reloaded)
        np.testing.assert_almost_equal(tau_hat_orig, tau_hat_reloaded)
        np.testing.assert_almost_equal(mu_hat_orig, mu_hat_reloaded)

    def test_bcf_rfx_string(self):
        # RNG
        random_seed = 1234
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

        # Generate random effects terms
        num_basis = 2
        num_groups = 4
        group_labels = rng.choice(num_groups, size=n)
        basis = np.empty((n, num_basis))
        basis[:, 0] = 1.0
        if num_basis > 1:
            basis[:, 1:] = rng.uniform(-1, 1, (n, num_basis - 1))

        # Define the group rfx function
        def rfx_mean(group_labels, basis):
            return np.where(
                group_labels == 0,
                0 - 1 * basis[:, 1],
                np.where(
                    group_labels == 1,
                    4 + 1 * basis[:, 1],
                    np.where(
                        group_labels == 2, 8 + 3 * basis[:, 1], 12 + 5 * basis[:, 1]
                    ),
                ),
            )

        # Generate outcome
        epsilon = rng.normal(0, 1, n)
        rfx_term = rfx_mean(group_labels, basis)
        y = mu_X + tau_X * Z + rfx_term + epsilon

        # Run BCF
        bcf_orig = BCFModel()
        bcf_orig.sample(
            X_train=X, Z_train=Z, y_train=y, pi_train=pi_X, rfx_group_ids_train=group_labels, rfx_basis_train=basis, num_gfr=10, num_mcmc=10
        )

        # Extract predictions from the sampler
        mu_hat_orig, tau_hat_orig, rfx_hat_orig, y_hat_orig = bcf_orig.predict(X, Z, pi_X, group_labels, basis)

        # "Round-trip" the model to JSON string and back and check that the predictions agree
        bcf_json_string = bcf_orig.to_json()
        bcf_reloaded = BCFModel()
        bcf_reloaded.from_json(bcf_json_string)
        mu_hat_reloaded, tau_hat_reloaded, rfx_hat_reloaded, y_hat_reloaded = bcf_reloaded.predict(
            X, Z, pi_X, group_labels, basis
        )
        np.testing.assert_almost_equal(y_hat_orig, y_hat_reloaded)
        np.testing.assert_almost_equal(tau_hat_orig, tau_hat_reloaded)
        np.testing.assert_almost_equal(mu_hat_orig, mu_hat_reloaded)
        np.testing.assert_almost_equal(rfx_hat_orig, rfx_hat_reloaded)

    def test_bcf_propensity_string(self):
        # RNG
        random_seed = 1234
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

        # Run BCF without passing propensity scores (so an internal propensity model must be constructed)
        bcf_orig = BCFModel()
        bcf_orig.sample(X_train=X, Z_train=Z, y_train=y, num_gfr=10, num_mcmc=10)

        # Extract predictions from the sampler
        mu_hat_orig, tau_hat_orig, y_hat_orig = bcf_orig.predict(X, Z, pi_X)

        # "Round-trip" the model to JSON string and back and check that the predictions agree
        bcf_json_string = bcf_orig.to_json()
        bcf_reloaded = BCFModel()
        bcf_reloaded.from_json(bcf_json_string)
        mu_hat_reloaded, tau_hat_reloaded, y_hat_reloaded = bcf_reloaded.predict(
            X, Z, pi_X
        )
        np.testing.assert_almost_equal(y_hat_orig, y_hat_reloaded)
        np.testing.assert_almost_equal(tau_hat_orig, tau_hat_reloaded)
        np.testing.assert_almost_equal(mu_hat_orig, mu_hat_reloaded)
