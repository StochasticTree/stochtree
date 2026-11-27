import numpy as np

from stochtree import (
    RNG,
    Dataset,
    Forest,
    ForestContainer,
    ForestSampler,
    Residual,
    ForestModelConfig,
    GlobalModelConfig,
)


class TestResidual:
    def test_basis_update(self):
        # Create dataset
        X = np.array([
            [1.5, 8.7, 1.2],
            [2.7, 3.4, 5.4],
            [3.6, 1.2, 9.3],
            [4.4, 5.4, 10.4],
            [5.3, 9.3, 3.6],
            [6.1, 10.4, 4.4],
        ])
        W = np.array([[1], [1], [1], [1], [1], [1]])
        n = X.shape[0]
        p = X.shape[1]
        y = np.expand_dims(np.where(X[:, 0] > 4, -5, 5) + np.random.normal(0, 1, n), 1)
        y_bar = np.squeeze(np.mean(y))
        y_std = np.squeeze(np.std(y))
        resid = (y - y_bar) / y_std
        forest_dataset = Dataset()
        forest_dataset.add_covariates(X)
        forest_dataset.add_basis(W)
        residual = Residual(resid)
        variable_weights = np.repeat(1.0 / p, p)
        feature_types = np.repeat(0, p).astype(int)

        # Forest parameters
        num_trees = 50
        alpha = 0.95
        beta = 2.0
        min_samples_leaf = 1
        current_sigma2 = 1.0
        current_leaf_scale = np.array([[1.0 / num_trees]])
        cutpoint_grid_size = 100

        # RNG
        cpp_rng = RNG(-1)

        # Create forest sampler and forest container
        forest_config = ForestModelConfig(
            num_trees=num_trees,
            num_features=p,
            num_observations=n,
            feature_types=feature_types,
            variable_weights=variable_weights,
            leaf_dimension=1,
            alpha=alpha,
            beta=beta,
            min_samples_leaf=min_samples_leaf,
            leaf_model_type=1,
            cutpoint_grid_size=cutpoint_grid_size,
            leaf_model_scale=current_leaf_scale,
        )
        global_config = GlobalModelConfig(global_error_variance=current_sigma2)
        forest_sampler = ForestSampler(forest_dataset, global_config, forest_config)
        forest_container = ForestContainer(num_trees, 1, False, False)
        active_forest = Forest(num_trees, 1, False, False)

        # Initialize the leaves of each tree in the prognostic forest
        init_root = np.squeeze(np.mean(resid)) / num_trees
        active_forest.set_root_leaves(init_root)
        forest_sampler.adjust_residual(
            forest_dataset, residual, active_forest, False, True
        )

        # Run the forest sampling algorithm for a single iteration
        forest_sampler.sample_one_iteration(
            forest_container,
            active_forest,
            forest_dataset,
            residual,
            cpp_rng,
            global_config,
            forest_config,
            True,
            True,
        )

        # Get the current residual after running the sampler
        initial_resid = residual.get_residual()

        # Get initial prediction from the tree ensemble
        initial_yhat = forest_container.predict(forest_dataset)

        # Update the basis vector
        scalar = 2.0
        W_update = W * scalar
        forest_dataset.update_basis(W_update)

        # Update residual to reflect adjusted basis
        forest_sampler.propagate_basis_update(forest_dataset, residual, active_forest)

        # Get updated prediction from the tree ensemble
        updated_yhat = forest_container.predict(forest_dataset)

        # Compute the expected residual
        expected_resid = initial_resid + initial_yhat - updated_yhat

        # Get the current residual after running the sampler
        updated_resid = residual.get_residual()
        np.testing.assert_almost_equal(expected_resid, updated_resid)

    def test_prepare_for_sampler_and_adjust_residual_nonzero(self):
        n = 5
        num_trees = 3
        dataset = Dataset()
        dataset.add_covariates(np.arange(n, dtype=float).reshape(-1, 1))
        residual = Residual(np.zeros(n))
        sampler = ForestSampler(
            dataset,
            GlobalModelConfig(global_error_variance=1.0),
            ForestModelConfig(
                num_trees=num_trees,
                num_features=1,
                num_observations=n,
                feature_types=np.array([0], dtype=int),
                variable_weights=np.array([1.0]),
            ),
        )
        forest = Forest(num_trees, 1, True, False)
        sampler.prepare_for_sampler(dataset, residual, forest, 0, np.array([3.0]))

        np.testing.assert_allclose(forest.predict(dataset), np.full(n, 3.0))
        np.testing.assert_allclose(residual.get_residual().ravel(), np.full(n, -3.0))
        np.testing.assert_allclose(
            sampler.get_cached_forest_predictions(), np.full(n, 3.0)
        )

        sampler.adjust_residual(dataset, residual, forest, False, True)
        np.testing.assert_allclose(residual.get_residual().ravel(), np.zeros(n))
        np.testing.assert_allclose(
            sampler.get_cached_forest_predictions(), np.full(n, 3.0)
        )
