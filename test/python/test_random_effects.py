import numpy as np

from stochtree import (
    BARTModel,
    BCFModel,
    RandomEffectsContainer,
    RandomEffectsDataset,
    RandomEffectsModel,
    RandomEffectsTracker,
    Residual,
    RNG,
)


class TestRandomEffects:
    def test_random_intercept(self):
        # RNG
        rng = np.random.default_rng()

        # Generate group labels and random effects basis
        num_observations = 1000
        num_basis = 1
        num_groups = 4
        group_labels = rng.choice(num_groups, size=num_observations)
        basis = np.empty((num_observations, num_basis))
        basis[:, 0] = 1.0
        if num_basis > 1:
            basis[:, 1:] = rng.uniform(-1, 1, (num_observations, num_basis - 1))

        # Define the group rfx function
        def outcome_mean(group_labels, basis):
            return np.where(
                group_labels == 0,
                0,
                np.where(group_labels == 1, 4, np.where(group_labels == 2, 8, 12)),
            )

        # Generate outcome
        epsilon = rng.normal(0, 1, num_observations)
        rfx_term = outcome_mean(group_labels, basis)
        y = rfx_term + epsilon

        # Standardize outcome
        y_bar = np.mean(y)
        y_std = np.std(y)
        resid = (y - y_bar) / y_std

        # Construct python objects used for rfx sampling
        outcome = Residual(resid)
        rfx_dataset = RandomEffectsDataset()
        rfx_dataset.add_group_labels(group_labels)
        rfx_dataset.add_basis(basis)
        rfx_tracker = RandomEffectsTracker(group_labels)
        rfx_model = RandomEffectsModel(num_basis, num_groups)
        rfx_model.set_working_parameter(np.ones(num_basis))
        rfx_model.set_group_parameters(np.ones((num_basis, num_groups)))
        rfx_model.set_working_parameter_covariance(np.identity(num_basis))
        rfx_model.set_group_parameter_covariance(np.identity(num_basis))
        rfx_model.set_variance_prior_shape(1.0)
        rfx_model.set_variance_prior_scale(1.0)
        rfx_container = RandomEffectsContainer()
        rfx_container.load_new_container(num_basis, num_groups, rfx_tracker)
        cpp_rng = RNG()

        # Sample the model
        num_mcmc = 10
        for _ in range(num_mcmc):
            rfx_model.sample(
                rfx_dataset, outcome, rfx_tracker, rfx_container, True, 1.0, cpp_rng
            )

        # Inspect the samples
        rfx_preds = rfx_container.predict(group_labels, basis) * y_std + y_bar
        assert rfx_preds.shape == (num_observations, num_mcmc)

    def test_random_slope(self):
        # RNG
        rng = np.random.default_rng()

        # Generate group labels and random effects basis
        num_observations = 1000
        num_basis = 2
        num_groups = 4
        group_labels = rng.choice(num_groups, size=num_observations)
        basis = np.empty((num_observations, num_basis))
        basis[:, 0] = 1.0
        if num_basis > 1:
            basis[:, 1:] = rng.uniform(-1, 1, (num_observations, num_basis - 1))

        # Define the group rfx function
        def outcome_mean(group_labels, basis):
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
        epsilon = rng.normal(0, 1, num_observations)
        rfx_term = outcome_mean(group_labels, basis)
        y = rfx_term + epsilon

        # Standardize outcome
        y_bar = np.mean(y)
        y_std = np.std(y)
        resid = (y - y_bar) / y_std

        # Construct python objects used for rfx sampling
        outcome = Residual(resid)
        rfx_dataset = RandomEffectsDataset()
        rfx_dataset.add_group_labels(group_labels)
        rfx_dataset.add_basis(basis)
        rfx_tracker = RandomEffectsTracker(group_labels)
        rfx_model = RandomEffectsModel(num_basis, num_groups)
        rfx_model.set_working_parameter(np.ones(num_basis))
        rfx_model.set_group_parameters(np.ones((num_basis, num_groups)))
        rfx_model.set_working_parameter_covariance(np.identity(num_basis))
        rfx_model.set_group_parameter_covariance(np.identity(num_basis))
        rfx_model.set_variance_prior_shape(1.0)
        rfx_model.set_variance_prior_scale(1.0)
        rfx_container = RandomEffectsContainer()
        rfx_container.load_new_container(num_basis, num_groups, rfx_tracker)
        cpp_rng = RNG()

        # Sample the model
        num_mcmc = 10
        for _ in range(num_mcmc):
            rfx_model.sample(
                rfx_dataset, outcome, rfx_tracker, rfx_container, True, 1.0, cpp_rng
            )

        # Inspect the samples
        rfx_preds = rfx_container.predict(group_labels, basis) * y_std + y_bar
        assert rfx_preds.shape == (num_observations, num_mcmc)

    def test_bart_rfx_default_numbering(self):
        # RNG
        rng = np.random.default_rng()

        # Generate covariates
        n = 100
        p = 10
        X = rng.uniform(0, 1, (n, p))

        # Generate group labels and random effects basis
        num_rfx_basis = 2
        num_rfx_groups = 4
        group_labels = rng.choice(num_rfx_groups, size=n)
        basis = np.empty((n, num_rfx_basis))
        basis[:, 0] = 1.0
        if num_rfx_basis > 1:
            basis[:, 1:] = rng.uniform(-1, 1, (n, num_rfx_basis - 1))

        # Define the covariate-dependent function
        def covariate_fn(X):
            return 5 * np.sin(2 * np.pi * X[:, 0])

        # Define the group rfx function
        def rfx_fn(group_labels, basis):
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
        mean_term = covariate_fn(X)
        rfx_term = rfx_fn(group_labels, basis)
        y = mean_term + rfx_term + epsilon

        # Fit a BART model
        bart_model = BARTModel()
        bart_model.sample(
            X_train=X,
            y_train=y,
            rfx_group_ids_train=group_labels,
            rfx_basis_train=basis
        )

        # Check all of the prediction / summary computation methods
        
        # Predict
        mean_preds = bart_model.predict(
            X=X,
            rfx_group_ids=group_labels,
            rfx_basis=basis,
            type="mean",
            terms="all",
            scale="linear",
        )
        posterior_preds = bart_model.predict(
            X=X,
            rfx_group_ids=group_labels,
            rfx_basis=basis,
            type="posterior",
            terms="all",
            scale="linear",
        )

        # Compute intervals
        posterior_interval = bart_model.compute_posterior_interval(
            terms="all",
            level=0.95,
            scale="linear",
            X=X,
            rfx_group_ids=group_labels,
            rfx_basis=basis,
        )

        # Sample posterior predictive
        posterior_predictive_draws = bart_model.sample_posterior_predictive(
            X=X,
            rfx_group_ids=group_labels,
            rfx_basis=basis,
            num_draws_per_sample=5,
        )

    def test_bart_rfx_offset_numbering(self):
        # RNG
        rng = np.random.default_rng()

        # Generate covariates
        n = 100
        p = 10
        X = rng.uniform(0, 1, (n, p))

        # Generate group labels and random effects basis
        num_rfx_basis = 2
        num_rfx_groups = 4
        group_labels = rng.choice(range(2, 2 + num_rfx_groups), size=n)
        basis = np.empty((n, num_rfx_basis))
        basis[:, 0] = 1.0
        if num_rfx_basis > 1:
            basis[:, 1:] = rng.uniform(-1, 1, (n, num_rfx_basis - 1))

        # Define the covariate-dependent function
        def covariate_fn(X):
            return 5 * np.sin(2 * np.pi * X[:, 0])

        # Define the group rfx function
        def rfx_fn(group_labels, basis):
            return np.where(
                group_labels == 2,
                0 - 1 * basis[:, 1],
                np.where(
                    group_labels == 3,
                    4 + 1 * basis[:, 1],
                    np.where(
                        group_labels == 4, 8 + 3 * basis[:, 1], 12 + 5 * basis[:, 1]
                    ),
                ),
            )

        # Generate outcome
        epsilon = rng.normal(0, 1, n)
        mean_term = covariate_fn(X)
        rfx_term = rfx_fn(group_labels, basis)
        y = mean_term + rfx_term + epsilon

        # Fit a BART model
        bart_model = BARTModel()
        bart_model.sample(
            X_train=X,
            y_train=y,
            rfx_group_ids_train=group_labels,
            rfx_basis_train=basis
        )

        # Check all of the prediction / summary computation methods
        
        # Predict
        mean_preds = bart_model.predict(
            X=X,
            rfx_group_ids=group_labels,
            rfx_basis=basis,
            type="mean",
            terms="all",
            scale="linear",
        )
        posterior_preds = bart_model.predict(
            X=X,
            rfx_group_ids=group_labels,
            rfx_basis=basis,
            type="posterior",
            terms="all",
            scale="linear",
        )

        # Compute intervals
        posterior_interval = bart_model.compute_posterior_interval(
            terms="all",
            level=0.95,
            scale="linear",
            X=X,
            rfx_group_ids=group_labels,
            rfx_basis=basis,
        )

        # Sample posterior predictive
        posterior_predictive_draws = bart_model.sample_posterior_predictive(
            X=X,
            rfx_group_ids=group_labels,
            rfx_basis=basis,
            num_draws_per_sample=5,
        )

    def test_bart_rfx_default_numbering_model_spec(self):
        # RNG
        rng = np.random.default_rng()

        # Generate covariates
        n = 100
        p = 10
        X = rng.uniform(0, 1, (n, p))

        # Generate group labels and random effects basis
        num_rfx_basis = 2
        num_rfx_groups = 4
        group_labels = rng.choice(num_rfx_groups, size=n)
        basis = np.empty((n, num_rfx_basis))
        basis[:, 0] = 1.0
        if num_rfx_basis > 1:
            basis[:, 1:] = rng.uniform(-1, 1, (n, num_rfx_basis - 1))

        # Define the covariate-dependent function
        def covariate_fn(X):
            return 5 * np.sin(2 * np.pi * X[:, 0])

        # Define the group rfx function
        def rfx_fn(group_labels, basis):
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
        mean_term = covariate_fn(X)
        rfx_term = rfx_fn(group_labels, basis)
        y = mean_term + rfx_term + epsilon

        # Fit a BART model
        bart_model = BARTModel()
        bart_model.sample(
            X_train=X,
            y_train=y,
            rfx_group_ids_train=group_labels,
            random_effects_params={"model_spec": "intercept_only"}
        )

        # Check all of the prediction / summary computation methods
        
        # Predict
        mean_preds = bart_model.predict(
            X=X,
            rfx_group_ids=group_labels,
            rfx_basis=basis,
            type="mean",
            terms="all",
            scale="linear",
        )
        posterior_preds = bart_model.predict(
            X=X,
            rfx_group_ids=group_labels,
            rfx_basis=basis,
            type="posterior",
            terms="all",
            scale="linear",
        )

        # Compute intervals
        posterior_interval = bart_model.compute_posterior_interval(
            terms="all",
            level=0.95,
            scale="linear",
            X=X,
            rfx_group_ids=group_labels,
            rfx_basis=basis,
        )

        # Sample posterior predictive
        posterior_predictive_draws = bart_model.sample_posterior_predictive(
            X=X,
            rfx_group_ids=group_labels,
            rfx_basis=basis,
            num_draws_per_sample=5,
        )

    def test_bart_rfx_offset_numbering_model_spec(self):
        # RNG
        rng = np.random.default_rng()

        # Generate covariates
        n = 100
        p = 10
        X = rng.uniform(0, 1, (n, p))

        # Generate group labels and random effects basis
        num_rfx_basis = 2
        num_rfx_groups = 4
        group_labels = rng.choice(range(2, 2 + num_rfx_groups), size=n)
        basis = np.empty((n, num_rfx_basis))
        basis[:, 0] = 1.0
        if num_rfx_basis > 1:
            basis[:, 1:] = rng.uniform(-1, 1, (n, num_rfx_basis - 1))

        # Define the covariate-dependent function
        def covariate_fn(X):
            return 5 * np.sin(2 * np.pi * X[:, 0])

        # Define the group rfx function
        def rfx_fn(group_labels, basis):
            return np.where(
                group_labels == 2,
                0 - 1 * basis[:, 1],
                np.where(
                    group_labels == 3,
                    4 + 1 * basis[:, 1],
                    np.where(
                        group_labels == 4, 8 + 3 * basis[:, 1], 12 + 5 * basis[:, 1]
                    ),
                ),
            )

        # Generate outcome
        epsilon = rng.normal(0, 1, n)
        mean_term = covariate_fn(X)
        rfx_term = rfx_fn(group_labels, basis)
        y = mean_term + rfx_term + epsilon

        # Fit a BART model
        bart_model = BARTModel()
        bart_model.sample(
            X_train=X,
            y_train=y,
            rfx_group_ids_train=group_labels,
            random_effects_params={"model_spec": "intercept_only"}
        )

        # Check all of the prediction / summary computation methods
        
        # Predict
        mean_preds = bart_model.predict(
            X=X,
            rfx_group_ids=group_labels,
            type="mean",
            terms="all",
            scale="linear",
        )
        posterior_preds = bart_model.predict(
            X=X,
            rfx_group_ids=group_labels,
            type="posterior",
            terms="all",
            scale="linear",
        )

        # Compute intervals
        posterior_interval = bart_model.compute_posterior_interval(
            terms="all",
            level=0.95,
            scale="linear",
            X=X,
            rfx_group_ids=group_labels,
        )

        # Sample posterior predictive
        posterior_predictive_draws = bart_model.sample_posterior_predictive(
            X=X,
            rfx_group_ids=group_labels,
            num_draws_per_sample=5,
        )

    def test_bcf_rfx_default_numbering(self):
        # RNG
        rng = np.random.default_rng()

        # Generate covariates
        n = 100
        p = 10
        X = rng.uniform(0, 1, (n, p))

        # Generate group labels and random effects basis
        num_rfx_basis = 2
        num_rfx_groups = 4
        group_labels = rng.choice(num_rfx_groups, size=n)
        basis = np.empty((n, num_rfx_basis))
        basis[:, 0] = 1.0
        if num_rfx_basis > 1:
            basis[:, 1:] = rng.uniform(-1, 1, (n, num_rfx_basis - 1))

        # Define the prognostic function
        def prog_fn(X):
            return 5 * np.sin(2 * np.pi * X[:, 0])

        # Define the treatment effect function
        def cate_fn(X):
            return 1.0 * X[:, 1]

        # Define the group rfx function
        def rfx_fn(group_labels, basis):
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
        propensity = rng.uniform(0.4, 0.6, n)
        Z = rng.binomial(1, propensity, n)
        mean_term = prog_fn(X) + Z * cate_fn(X)
        rfx_term = rfx_fn(group_labels, basis)
        y = mean_term + rfx_term + epsilon

        # Fit a BCF model with intercept plus treatment random effects model specification
        bcf_model = BCFModel()
        bcf_model.sample(
            X_train=X,
            Z_train=Z,
            propensity_train=propensity,
            y_train=y,
            rfx_group_ids_train=group_labels,
            rfx_basis_train=basis,
        )

        # Check all of the prediction / summary computation methods
        
        # Predict
        rfx_preds = bcf_model.predict(
            X=X,
            Z=Z,
            propensity=propensity,
            rfx_group_ids=group_labels,
            rfx_basis=basis,
            type="posterior",
            terms="rfx",
            scale="linear",
        )
        yhat_preds = bcf_model.predict(
            X=X,
            Z=Z,
            propensity=propensity,
            rfx_group_ids=group_labels,
            rfx_basis=basis,
            type="posterior",
            terms="y_hat",
            scale="linear",
        )

        # Compute intervals
        posterior_interval = bcf_model.compute_posterior_interval(
            terms="all",
            level=0.95,
            scale="linear",
            X=X,
            Z=Z,
            propensity=propensity,
            rfx_group_ids=group_labels,
            rfx_basis=basis,
        )

        # Sample posterior predictive
        posterior_predictive_draws = bcf_model.sample_posterior_predictive(
            X=X,
            Z=Z,
            propensity=propensity,
            rfx_group_ids=group_labels,
            rfx_basis=basis,
            num_draws_per_sample=5,
        )

    def test_bcf_rfx_default_numbering_model_spec(self):
        # RNG
        rng = np.random.default_rng()

        # Generate covariates
        n = 100
        p = 10
        X = rng.uniform(0, 1, (n, p))

        # Generate group labels and random effects basis
        num_rfx_basis = 2
        num_rfx_groups = 4
        group_labels = rng.choice(num_rfx_groups, size=n)
        basis = np.empty((n, num_rfx_basis))
        basis[:, 0] = 1.0
        if num_rfx_basis > 1:
            basis[:, 1:] = rng.uniform(-1, 1, (n, num_rfx_basis - 1))

        # Define the prognostic function
        def prog_fn(X):
            return 5 * np.sin(2 * np.pi * X[:, 0])

        # Define the treatment effect function
        def cate_fn(X):
            return 1.0 * X[:, 1]

        # Define the group rfx function
        def rfx_fn(group_labels, basis):
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
        propensity = rng.uniform(0.4, 0.6, n)
        Z = rng.binomial(1, propensity, n)
        mean_term = prog_fn(X) + Z * cate_fn(X)
        rfx_term = rfx_fn(group_labels, basis)
        y = mean_term + rfx_term + epsilon

        # Fit a BCF model with intercept plus treatment random effects model specification
        bcf_model = BCFModel()
        bcf_model.sample(
            X_train=X,
            Z_train=Z,
            propensity_train=propensity,
            y_train=y,
            rfx_group_ids_train=group_labels,
            random_effects_params={"model_spec": "intercept_plus_treatment"}
        )

        # Check all of the prediction / summary computation methods
        
        # Predict
        rfx_preds = bcf_model.predict(
            X=X,
            Z=Z,
            propensity=propensity,
            rfx_group_ids=group_labels,
            type="posterior",
            terms="rfx",
            scale="linear",
        )
        yhat_preds = bcf_model.predict(
            X=X,
            Z=Z,
            propensity=propensity,
            rfx_group_ids=group_labels,
            type="posterior",
            terms="y_hat",
            scale="linear",
        )

        # Compute intervals
        posterior_interval = bcf_model.compute_posterior_interval(
            terms="all",
            level=0.95,
            scale="linear",
            X=X,
            Z=Z,
            propensity=propensity,
            rfx_group_ids=group_labels,
        )

        # Sample posterior predictive
        posterior_predictive_draws = bcf_model.sample_posterior_predictive(
            X=X,
            Z=Z,
            propensity=propensity,
            rfx_group_ids=group_labels,
            num_draws_per_sample=5,
        )

        # Fit a BCF model with intercept-only random effects model specification
        bcf_model = BCFModel()
        bcf_model.sample(
            X_train=X,
            Z_train=Z,
            propensity_train=propensity,
            y_train=y,
            rfx_group_ids_train=group_labels,
            random_effects_params={"model_spec": "intercept_only"}
        )

        # Check all of the prediction / summary computation methods
        
        # Predict
        rfx_preds = bcf_model.predict(
            X=X,
            Z=Z,
            propensity=propensity,
            rfx_group_ids=group_labels,
            type="posterior",
            terms="rfx",
            scale="linear",
        )
        yhat_preds = bcf_model.predict(
            X=X,
            Z=Z,
            propensity=propensity,
            rfx_group_ids=group_labels,
            type="posterior",
            terms="y_hat",
            scale="linear",
        )

        # Compute intervals
        posterior_interval = bcf_model.compute_posterior_interval(
            terms="all",
            level=0.95,
            scale="linear",
            X=X,
            Z=Z,
            propensity=propensity,
            rfx_group_ids=group_labels,
        )

        # Sample posterior predictive
        posterior_predictive_draws = bcf_model.sample_posterior_predictive(
            X=X,
            Z=Z,
            propensity=propensity,
            rfx_group_ids=group_labels,
            num_draws_per_sample=5,
        )
    
    def test_bcf_rfx_offset_numbering(self):
        # RNG
        rng = np.random.default_rng()

        # Generate covariates
        n = 100
        p = 10
        X = rng.uniform(0, 1, (n, p))

        # Generate group labels and random effects basis
        num_rfx_basis = 2
        num_rfx_groups = 4
        group_labels = rng.choice(num_rfx_groups, size=n) + 2
        basis = np.empty((n, num_rfx_basis))
        basis[:, 0] = 1.0
        if num_rfx_basis > 1:
            basis[:, 1:] = rng.uniform(-1, 1, (n, num_rfx_basis - 1))

        # Define the prognostic function
        def prog_fn(X):
            return 5 * np.sin(2 * np.pi * X[:, 0])

        # Define the treatment effect function
        def cate_fn(X):
            return 1.0 * X[:, 1]

        # Define the group rfx function
        def rfx_fn(group_labels, basis):
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
        propensity = rng.uniform(0.4, 0.6, n)
        Z = rng.binomial(1, propensity, n)
        mean_term = prog_fn(X) + Z * cate_fn(X)
        rfx_term = rfx_fn(group_labels - 2, basis)
        y = mean_term + rfx_term + epsilon

        # Fit a BCF model with intercept plus treatment random effects model specification
        bcf_model = BCFModel()
        bcf_model.sample(
            X_train=X,
            Z_train=Z,
            propensity_train=propensity,
            y_train=y,
            rfx_group_ids_train=group_labels,
            rfx_basis_train=basis,
        )

        # Check all of the prediction / summary computation methods
        
        # Predict
        rfx_preds = bcf_model.predict(
            X=X,
            Z=Z,
            propensity=propensity,
            rfx_group_ids=group_labels,
            rfx_basis=basis,
            type="posterior",
            terms="rfx",
            scale="linear",
        )
        yhat_preds = bcf_model.predict(
            X=X,
            Z=Z,
            propensity=propensity,
            rfx_group_ids=group_labels,
            rfx_basis=basis,
            type="posterior",
            terms="y_hat",
            scale="linear",
        )

        # Compute intervals
        posterior_interval = bcf_model.compute_posterior_interval(
            terms="all",
            level=0.95,
            scale="linear",
            X=X,
            Z=Z,
            propensity=propensity,
            rfx_group_ids=group_labels,
            rfx_basis=basis,
        )

        # Sample posterior predictive
        posterior_predictive_draws = bcf_model.sample_posterior_predictive(
            X=X,
            Z=Z,
            propensity=propensity,
            rfx_group_ids=group_labels,
            rfx_basis=basis,
            num_draws_per_sample=5,
        )

    def test_bcf_rfx_offset_numbering_model_spec(self):
        # RNG
        rng = np.random.default_rng()

        # Generate covariates
        n = 100
        p = 10
        X = rng.uniform(0, 1, (n, p))

        # Generate group labels and random effects basis
        num_rfx_basis = 2
        num_rfx_groups = 4
        group_labels = rng.choice(num_rfx_groups, size=n) + 2
        basis = np.empty((n, num_rfx_basis))
        basis[:, 0] = 1.0
        if num_rfx_basis > 1:
            basis[:, 1:] = rng.uniform(-1, 1, (n, num_rfx_basis - 1))

        # Define the prognostic function
        def prog_fn(X):
            return 5 * np.sin(2 * np.pi * X[:, 0])

        # Define the treatment effect function
        def cate_fn(X):
            return 1.0 * X[:, 1]

        # Define the group rfx function
        def rfx_fn(group_labels, basis):
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
        propensity = rng.uniform(0.4, 0.6, n)
        Z = rng.binomial(1, propensity, n)
        mean_term = prog_fn(X) + Z * cate_fn(X)
        rfx_term = rfx_fn(group_labels - 2, basis)
        y = mean_term + rfx_term + epsilon

        # Fit a BCF model with intercept plus treatment random effects model specification
        bcf_model = BCFModel()
        bcf_model.sample(
            X_train=X,
            Z_train=Z,
            propensity_train=propensity,
            y_train=y,
            rfx_group_ids_train=group_labels,
            random_effects_params={"model_spec": "intercept_plus_treatment"}
        )

        # Check all of the prediction / summary computation methods
        
        # Predict
        rfx_preds = bcf_model.predict(
            X=X,
            Z=Z,
            propensity=propensity,
            rfx_group_ids=group_labels,
            type="posterior",
            terms="rfx",
            scale="linear",
        )
        yhat_preds = bcf_model.predict(
            X=X,
            Z=Z,
            propensity=propensity,
            rfx_group_ids=group_labels,
            type="posterior",
            terms="y_hat",
            scale="linear",
        )

        # Compute intervals
        posterior_interval = bcf_model.compute_posterior_interval(
            terms="all",
            level=0.95,
            scale="linear",
            X=X,
            Z=Z,
            propensity=propensity,
            rfx_group_ids=group_labels,
        )

        # Sample posterior predictive
        posterior_predictive_draws = bcf_model.sample_posterior_predictive(
            X=X,
            Z=Z,
            propensity=propensity,
            rfx_group_ids=group_labels,
            num_draws_per_sample=5,
        )

        # Fit a BCF model with intercept-only random effects model specification
        bcf_model = BCFModel()
        bcf_model.sample(
            X_train=X,
            Z_train=Z,
            propensity_train=propensity,
            y_train=y,
            rfx_group_ids_train=group_labels,
            random_effects_params={"model_spec": "intercept_only"}
        )

        # Check all of the prediction / summary computation methods
        
        # Predict
        rfx_preds = bcf_model.predict(
            X=X,
            Z=Z,
            propensity=propensity,
            rfx_group_ids=group_labels,
            type="posterior",
            terms="rfx",
            scale="linear",
        )
        yhat_preds = bcf_model.predict(
            X=X,
            Z=Z,
            propensity=propensity,
            rfx_group_ids=group_labels,
            type="posterior",
            terms="y_hat",
            scale="linear",
        )

        # Compute intervals
        posterior_interval = bcf_model.compute_posterior_interval(
            terms="all",
            level=0.95,
            scale="linear",
            X=X,
            Z=Z,
            propensity=propensity,
            rfx_group_ids=group_labels,
        )

        # Sample posterior predictive
        posterior_predictive_draws = bcf_model.sample_posterior_predictive(
            X=X,
            Z=Z,
            propensity=propensity,
            rfx_group_ids=group_labels,
            num_draws_per_sample=5,
        )



