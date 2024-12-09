import pytest
import numpy as np
from sklearn.model_selection import train_test_split
from stochtree import BARTModel

class TestBART:
    def test_bart_constant_leaf_homoskedastic(self):
        # RNG
        random_seed = 101
        rng = np.random.default_rng(random_seed)

        # Generate covariates and basis
        n = 1000
        p_X = 10
        X = rng.uniform(0, 1, (n, p_X))

        # Define the outcome mean function
        def outcome_mean(X):
            return np.where(
                (X[:,0] >= 0.0) & (X[:,0] < 0.25), -7.5, 
                np.where(
                    (X[:,0] >= 0.25) & (X[:,0] < 0.5), -2.5, 
                    np.where(
                        (X[:,0] >= 0.5) & (X[:,0] < 0.75), 2.5, 
                        7.5
                    )
                )
            )

        # Generate outcome
        epsilon = rng.normal(0, 1, n)
        y = outcome_mean(X) + epsilon

        # Test-train split
        sample_inds = np.arange(n)
        train_inds, test_inds = train_test_split(sample_inds, test_size=0.5)
        X_train = X[train_inds,:]
        X_test = X[test_inds,:]
        y_train = y[train_inds]
        y_test = y[test_inds]
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        # BCF settings
        num_gfr = 10
        num_burnin = 0
        num_mcmc = 10
        
        # Run BCF with test set and propensity score
        bart_model = BARTModel()
        bart_model.sample(X_train=X_train, y_train=y_train, X_test=X_test, 
                          num_gfr=num_gfr, num_burnin=num_burnin, num_mcmc=num_mcmc)

        # Assertions
        assert (bart_model.y_hat_train.shape == (n_train, num_mcmc))
        assert (bart_model.y_hat_test.shape == (n_test, num_mcmc))
    
    def test_bart_univariate_leaf_regression_homoskedastic(self):
        # RNG
        random_seed = 101
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
                (X[:,0] >= 0.0) & (X[:,0] < 0.25), -7.5 * W[:,0], 
                np.where(
                    (X[:,0] >= 0.25) & (X[:,0] < 0.5), -2.5 * W[:,0], 
                    np.where(
                        (X[:,0] >= 0.5) & (X[:,0] < 0.75), 2.5 * W[:,0], 
                        7.5 * W[:,0]
                    )
                )
            )

        # Generate outcome
        epsilon = rng.normal(0, 1, n)
        y = outcome_mean(X, W) + epsilon

        # Test-train split
        sample_inds = np.arange(n)
        train_inds, test_inds = train_test_split(sample_inds, test_size=0.5)
        X_train = X[train_inds,:]
        X_test = X[test_inds,:]
        basis_train = W[train_inds,:]
        basis_test = W[test_inds,:]
        y_train = y[train_inds]
        y_test = y[test_inds]
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        # BCF settings
        num_gfr = 10
        num_burnin = 0
        num_mcmc = 10
        
        # Run BCF with test set and propensity score
        bart_model = BARTModel()
        bart_model.sample(X_train=X_train, y_train=y_train, basis_train=basis_train, 
                          X_test=X_test, basis_test=basis_test, num_gfr=num_gfr, 
                          num_burnin=num_burnin, num_mcmc=num_mcmc)

        # Assertions
        assert (bart_model.y_hat_train.shape == (n_train, num_mcmc))
        assert (bart_model.y_hat_test.shape == (n_test, num_mcmc))
    
    def test_bart_multivariate_leaf_regression_homoskedastic(self):
        # RNG
        random_seed = 101
        rng = np.random.default_rng(random_seed)

        # Generate covariates and basis
        n = 1000
        p_X = 10
        p_W = 5
        X = rng.uniform(0, 1, (n, p_X))
        W = rng.uniform(0, 1, (n, p_W))

        # Define the outcome mean function
        def outcome_mean(X, W):
            return np.where(
                (X[:,0] >= 0.0) & (X[:,0] < 0.25), -7.5 * W[:,0], 
                np.where(
                    (X[:,0] >= 0.25) & (X[:,0] < 0.5), -2.5 * W[:,0], 
                    np.where(
                        (X[:,0] >= 0.5) & (X[:,0] < 0.75), 2.5 * W[:,0], 
                        7.5 * W[:,0]
                    )
                )
            )

        # Generate outcome
        epsilon = rng.normal(0, 1, n)
        y = outcome_mean(X, W) + epsilon

        # Test-train split
        sample_inds = np.arange(n)
        train_inds, test_inds = train_test_split(sample_inds, test_size=0.5)
        X_train = X[train_inds,:]
        X_test = X[test_inds,:]
        basis_train = W[train_inds,:]
        basis_test = W[test_inds,:]
        y_train = y[train_inds]
        y_test = y[test_inds]
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        # BCF settings
        num_gfr = 10
        num_burnin = 0
        num_mcmc = 10
        
        # Run BCF with test set and propensity score
        bart_model = BARTModel()
        bart_model.sample(X_train=X_train, y_train=y_train, basis_train=basis_train, 
                          X_test=X_test, basis_test=basis_test, num_gfr=num_gfr, 
                          num_burnin=num_burnin, num_mcmc=num_mcmc)

        # Assertions
        assert (bart_model.y_hat_train.shape == (n_train, num_mcmc))
        assert (bart_model.y_hat_test.shape == (n_test, num_mcmc))
    
    def test_bart_constant_leaf_heteroskedastic(self):
        # RNG
        random_seed = 101
        rng = np.random.default_rng(random_seed)

        # Generate covariates and basis
        n = 1000
        p_X = 10
        X = rng.uniform(0, 1, (n, p_X))

        # Define the outcome mean function
        def outcome_mean(X):
            return np.where(
                (X[:,0] >= 0.0) & (X[:,0] < 0.25), -7.5, 
                np.where(
                    (X[:,0] >= 0.25) & (X[:,0] < 0.5), -2.5, 
                    np.where(
                        (X[:,0] >= 0.5) & (X[:,0] < 0.75), 2.5, 
                        7.5
                    )
                )
            )

        # Define the conditional standard deviation function
        def conditional_stddev(X):
            return np.where(
                (X[:,0] >= 0.0) & (X[:,0] < 0.25), 0.25, 
                np.where(
                    (X[:,0] >= 0.25) & (X[:,0] < 0.5), 0.5, 
                    np.where(
                        (X[:,0] >= 0.5) & (X[:,0] < 0.75), 1, 
                        2
                    )
                )
            )

        # Generate outcome
        epsilon = rng.normal(0, 1, n)
        y = outcome_mean(X) + epsilon * conditional_stddev(X)

        # Test-train split
        sample_inds = np.arange(n)
        train_inds, test_inds = train_test_split(sample_inds, test_size=0.5)
        X_train = X[train_inds,:]
        X_test = X[test_inds,:]
        y_train = y[train_inds]
        y_test = y[test_inds]
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        # BCF settings
        num_gfr = 10
        num_burnin = 0
        num_mcmc = 10
        
        # Run BCF with test set and propensity score
        bart_model = BARTModel()
        bart_params = {'num_trees_variance': 50, 'sample_sigma_global': True}
        bart_model.sample(X_train=X_train, y_train=y_train, X_test=X_test, params=bart_params, 
                          num_gfr=num_gfr, num_burnin=num_burnin, num_mcmc=num_mcmc)

        # Assertions
        assert (bart_model.y_hat_train.shape == (n_train, num_mcmc))
        assert (bart_model.y_hat_test.shape == (n_test, num_mcmc))
    
    def test_bart_univariate_leaf_regression_heteroskedastic(self):
        # RNG
        random_seed = 101
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
                (X[:,0] >= 0.0) & (X[:,0] < 0.25), -7.5 * W[:,0], 
                np.where(
                    (X[:,0] >= 0.25) & (X[:,0] < 0.5), -2.5 * W[:,0], 
                    np.where(
                        (X[:,0] >= 0.5) & (X[:,0] < 0.75), 2.5 * W[:,0], 
                        7.5 * W[:,0]
                    )
                )
            )

        # Define the conditional standard deviation function
        def conditional_stddev(X):
            return np.where(
                (X[:,0] >= 0.0) & (X[:,0] < 0.25), 0.25, 
                np.where(
                    (X[:,0] >= 0.25) & (X[:,0] < 0.5), 0.5, 
                    np.where(
                        (X[:,0] >= 0.5) & (X[:,0] < 0.75), 1, 
                        2
                    )
                )
            )

        # Generate outcome
        epsilon = rng.normal(0, 1, n)
        y = outcome_mean(X, W) + epsilon * conditional_stddev(X)

        # Test-train split
        sample_inds = np.arange(n)
        train_inds, test_inds = train_test_split(sample_inds, test_size=0.5)
        X_train = X[train_inds,:]
        X_test = X[test_inds,:]
        basis_train = W[train_inds,:]
        basis_test = W[test_inds,:]
        y_train = y[train_inds]
        y_test = y[test_inds]
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        # BCF settings
        num_gfr = 10
        num_burnin = 0
        num_mcmc = 10
        
        # Run BCF with test set and propensity score
        bart_model = BARTModel()
        bart_params = {'num_trees_variance': 50, 'sample_sigma_global': True}
        bart_model.sample(X_train=X_train, y_train=y_train, basis_train=basis_train, 
                          X_test=X_test, basis_test=basis_test, num_gfr=num_gfr, 
                          num_burnin=num_burnin, num_mcmc=num_mcmc, params=bart_params)

        # Assertions
        assert (bart_model.y_hat_train.shape == (n_train, num_mcmc))
        assert (bart_model.y_hat_test.shape == (n_test, num_mcmc))
    
    def test_bart_multivariate_leaf_regression_heteroskedastic(self):
        # RNG
        random_seed = 101
        rng = np.random.default_rng(random_seed)

        # Generate covariates and basis
        n = 1000
        p_X = 10
        p_W = 5
        X = rng.uniform(0, 1, (n, p_X))
        W = rng.uniform(0, 1, (n, p_W))

        # Define the outcome mean function
        def outcome_mean(X, W):
            return np.where(
                (X[:,0] >= 0.0) & (X[:,0] < 0.25), -7.5 * W[:,0], 
                np.where(
                    (X[:,0] >= 0.25) & (X[:,0] < 0.5), -2.5 * W[:,0], 
                    np.where(
                        (X[:,0] >= 0.5) & (X[:,0] < 0.75), 2.5 * W[:,0], 
                        7.5 * W[:,0]
                    )
                )
            )

        # Define the conditional standard deviation function
        def conditional_stddev(X):
            return np.where(
                (X[:,0] >= 0.0) & (X[:,0] < 0.25), 0.25, 
                np.where(
                    (X[:,0] >= 0.25) & (X[:,0] < 0.5), 0.5, 
                    np.where(
                        (X[:,0] >= 0.5) & (X[:,0] < 0.75), 1, 
                        2
                    )
                )
            )

        # Generate outcome
        epsilon = rng.normal(0, 1, n)
        y = outcome_mean(X, W) + epsilon * conditional_stddev(X)

        # Test-train split
        sample_inds = np.arange(n)
        train_inds, test_inds = train_test_split(sample_inds, test_size=0.5)
        X_train = X[train_inds,:]
        X_test = X[test_inds,:]
        basis_train = W[train_inds,:]
        basis_test = W[test_inds,:]
        y_train = y[train_inds]
        y_test = y[test_inds]
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        # BCF settings
        num_gfr = 10
        num_burnin = 0
        num_mcmc = 10
        
        # Run BCF with test set and propensity score
        bart_model = BARTModel()
        bart_params = {'num_trees_variance': 50, 'sample_sigma_global': True}
        bart_model.sample(X_train=X_train, y_train=y_train, basis_train=basis_train, 
                          X_test=X_test, basis_test=basis_test, num_gfr=num_gfr, 
                          num_burnin=num_burnin, num_mcmc=num_mcmc, params=bart_params)

        # Assertions
        assert (bart_model.y_hat_train.shape == (n_train, num_mcmc))
        assert (bart_model.y_hat_test.shape == (n_test, num_mcmc))
