import numpy as np
from stochtree import ForestContainer, Dataset, Residual, ForestSampler, RNG

class TestResidual:
    def test_basis_update(self):
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
            [[1],
             [1],
             [1],
             [1],
             [1],
             [1]]
        )
        n = X.shape[0]
        p = X.shape[1]
        y = np.expand_dims(np.where(X[:,0]>4,-5,5) + np.random.normal(0,1,n), 1)
        y_bar = np.squeeze(np.mean(y))
        y_std = np.squeeze(np.std(y))
        resid = (y-y_bar)/y_std
        forest_dataset = Dataset()
        forest_dataset.add_covariates(X)
        forest_dataset.add_basis(W)
        residual = Residual(resid)
        variable_weights = np.repeat(1.0/p, p)
        feature_types = np.repeat(0, p).astype(int)

        # Forest parameters
        num_trees = 50
        alpha = 0.95
        beta = 2.0
        min_samples_leaf = 1
        current_sigma2 = 1.
        current_leaf_scale = np.array([[1./num_trees]])
        cutpoint_grid_size = 100

        # RNG
        cpp_rng = RNG(-1)

        # Create forest sampler and forest container
        forest_sampler = ForestSampler(forest_dataset, feature_types, num_trees, n, alpha, beta, min_samples_leaf)
        forest_container = ForestContainer(num_trees, 1, False)
        
        # Initialize the leaves of each tree in the prognostic forest
        init_root = np.squeeze(np.mean(resid)) / num_trees
        forest_container.set_root_leaves(0, init_root)
        forest_sampler.adjust_residual(forest_dataset, residual, forest_container, False, 0, True)
        
        # Run the forest sampling algorithm for a single iteration
        forest_sampler.sample_one_iteration(
            forest_container, forest_dataset, residual, cpp_rng, feature_types, 
            cutpoint_grid_size, current_leaf_scale, variable_weights, current_sigma2, 1, True, True
        )

        # Get the current residual after running the sampler
        initial_resid = residual.get_residual()

        # Get initial prediction from the tree ensemble
        initial_yhat = forest_container.predict(forest_dataset)

        # Update the basis vector
        scalar = 2.0
        W_update = W*scalar
        forest_dataset.update_basis(W_update)

        # Update residual to reflect adjusted basis
        forest_sampler.update_residual(forest_dataset, residual, forest_container, 0)

        # Get updated prediction from the tree ensemble
        updated_yhat = forest_container.predict(forest_dataset)

        # Compute the expected residual
        expected_resid = initial_resid + initial_yhat - updated_yhat

        # Get the current residual after running the sampler
        updated_resid = residual.get_residual()
        np.testing.assert_almost_equal(expected_resid, updated_resid)
