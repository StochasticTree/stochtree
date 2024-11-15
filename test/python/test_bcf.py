import pytest
import numpy as np
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
        pi_X = 0.25 + 0.5*X[:,0]
        Z = rng.binomial(1, pi_X, n).astype(float)

        # Define the outcome mean functions (prognostic and treatment effects)
        mu_X = pi_X*5
        tau_X = X[:,1]*2

        # Generate outcome
        epsilon = rng.normal(0, 1, n)
        y = mu_X + tau_X*Z + epsilon

        # Test-train split
        sample_inds = np.arange(n)
        train_inds, test_inds = train_test_split(sample_inds, test_size=0.5)
        X_train = X[train_inds,:]
        X_test = X[test_inds,:]
        Z_train = Z[train_inds]
        Z_test = Z[test_inds]
        y_train = y[train_inds]
        y_test = y[test_inds]
        pi_train = pi_X[train_inds]
        pi_test = pi_X[test_inds]
        mu_train = mu_X[train_inds]
        mu_test = mu_X[test_inds]
        tau_train = tau_X[train_inds]
        tau_test = tau_X[test_inds]
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        # BCF settings
        num_gfr = 10
        num_burnin = 0
        num_mcmc = 10
        
        # Run BCF with test set and propensity score
        bcf_model = BCFModel()
        bcf_model.sample(X_train=X_train, Z_train=Z_train, y_train=y_train, pi_train=pi_train, 
                         X_test=X_test, Z_test=Z_test, pi_test=pi_test, num_gfr=num_gfr, 
                         num_burnin=num_burnin, num_mcmc=num_mcmc)

        # Assertions
        assert (bcf_model.y_hat_train.shape == (n_train, num_mcmc))
        assert (bcf_model.mu_hat_train.shape == (n_train, num_mcmc))
        assert (bcf_model.tau_hat_train.shape == (n_train, num_mcmc))
        assert (bcf_model.y_hat_test.shape == (n_test, num_mcmc))
        assert (bcf_model.mu_hat_test.shape == (n_test, num_mcmc))
        assert (bcf_model.tau_hat_test.shape == (n_test, num_mcmc))

        # Check overall prediction method
        tau_hat, mu_hat, y_hat = bcf_model.predict(X_test, Z_test, pi_test)
        assert (tau_hat.shape == (n_test, num_mcmc))
        assert (mu_hat.shape == (n_test, num_mcmc))
        assert (y_hat.shape == (n_test, num_mcmc))
        
        # Check treatment effect prediction method
        tau_hat = bcf_model.predict_tau(X_test, Z_test, pi_test)
        assert (tau_hat.shape == (n_test, num_mcmc))

        # Run BCF without test set and with propensity score
        bcf_model = BCFModel()
        bcf_model.sample(X_train=X_train, Z_train=Z_train, y_train=y_train, pi_train=pi_train, 
                         num_gfr=num_gfr, num_burnin=num_burnin, num_mcmc=num_mcmc)

        # Assertions
        assert (bcf_model.y_hat_train.shape == (n_train, num_mcmc))
        assert (bcf_model.mu_hat_train.shape == (n_train, num_mcmc))
        assert (bcf_model.tau_hat_train.shape == (n_train, num_mcmc))

        # Check overall prediction method
        tau_hat, mu_hat, y_hat = bcf_model.predict(X_test, Z_test, pi_test)
        assert (tau_hat.shape == (n_test, num_mcmc))
        assert (mu_hat.shape == (n_test, num_mcmc))
        assert (y_hat.shape == (n_test, num_mcmc))
        
        # Check treatment effect prediction method
        tau_hat = bcf_model.predict_tau(X_test, Z_test, pi_test)
        assert (tau_hat.shape == (n_test, num_mcmc))

        # Run BCF with test set and without propensity score
        bcf_model = BCFModel()
        bcf_model.sample(X_train=X_train, Z_train=Z_train, y_train=y_train, 
                         X_test=X_test, Z_test=Z_test, num_gfr=num_gfr, 
                         num_burnin=num_burnin, num_mcmc=num_mcmc)

        # Assertions
        assert (bcf_model.y_hat_train.shape == (n_train, num_mcmc))
        assert (bcf_model.mu_hat_train.shape == (n_train, num_mcmc))
        assert (bcf_model.tau_hat_train.shape == (n_train, num_mcmc))
        assert (bcf_model.bart_propensity_model.y_hat_train.shape == (n_train, 10))
        assert (bcf_model.y_hat_test.shape == (n_test, num_mcmc))
        assert (bcf_model.mu_hat_test.shape == (n_test, num_mcmc))
        assert (bcf_model.tau_hat_test.shape == (n_test, num_mcmc))
        assert (bcf_model.bart_propensity_model.y_hat_test.shape == (n_test, 10))

        # Check overall prediction method
        tau_hat, mu_hat, y_hat = bcf_model.predict(X_test, Z_test)
        assert (tau_hat.shape == (n_test, num_mcmc))
        assert (mu_hat.shape == (n_test, num_mcmc))
        assert (y_hat.shape == (n_test, num_mcmc))
        
        # Check treatment effect prediction method
        tau_hat = bcf_model.predict_tau(X_test, Z_test)
        assert (tau_hat.shape == (n_test, num_mcmc))

        # Run BCF without test set and without propensity score
        bcf_model = BCFModel()
        bcf_model.sample(X_train=X_train, Z_train=Z_train, y_train=y_train, 
                         num_gfr=num_gfr, num_burnin=num_burnin, num_mcmc=num_mcmc)

        # Assertions
        assert (bcf_model.y_hat_train.shape == (n_train, num_mcmc))
        assert (bcf_model.mu_hat_train.shape == (n_train, num_mcmc))
        assert (bcf_model.tau_hat_train.shape == (n_train, num_mcmc))
        assert (bcf_model.bart_propensity_model.y_hat_train.shape == (n_train, 10))

        # Check overall prediction method
        tau_hat, mu_hat, y_hat = bcf_model.predict(X_test, Z_test)
        assert (tau_hat.shape == (n_test, num_mcmc))
        assert (mu_hat.shape == (n_test, num_mcmc))
        assert (y_hat.shape == (n_test, num_mcmc))
        
        # Check treatment effect prediction method
        tau_hat = bcf_model.predict_tau(X_test, Z_test)
    
    def test_continuous_univariate_bcf(self):
        # RNG
        random_seed = 101
        rng = np.random.default_rng(random_seed)

        # Generate covariates and basis
        n = 100
        p_X = 5
        X = rng.uniform(0, 1, (n, p_X))
        pi_X = 0.25 + 0.5*X[:,0]
        Z = pi_X + rng.normal(0, 1, n)

        # Define the outcome mean functions (prognostic and treatment effects)
        mu_X = pi_X*5
        tau_X = X[:,1]*2

        # Generate outcome
        epsilon = rng.normal(0, 1, n)
        y = mu_X + tau_X*Z + epsilon

        # Test-train split
        sample_inds = np.arange(n)
        train_inds, test_inds = train_test_split(sample_inds, test_size=0.5)
        X_train = X[train_inds,:]
        X_test = X[test_inds,:]
        Z_train = Z[train_inds]
        Z_test = Z[test_inds]
        y_train = y[train_inds]
        y_test = y[test_inds]
        pi_train = pi_X[train_inds]
        pi_test = pi_X[test_inds]
        mu_train = mu_X[train_inds]
        mu_test = mu_X[test_inds]
        tau_train = tau_X[train_inds]
        tau_test = tau_X[test_inds]
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        # BCF settings
        num_gfr = 10
        num_burnin = 0
        num_mcmc = 10
        
        # Run BCF with test set and propensity score
        bcf_model = BCFModel()
        bcf_model.sample(X_train=X_train, Z_train=Z_train, y_train=y_train, pi_train=pi_train, 
                         X_test=X_test, Z_test=Z_test, pi_test=pi_test, num_gfr=num_gfr, 
                         num_burnin=num_burnin, num_mcmc=num_mcmc)

        # Assertions
        assert (bcf_model.y_hat_train.shape == (n_train, num_mcmc))
        assert (bcf_model.mu_hat_train.shape == (n_train, num_mcmc))
        assert (bcf_model.tau_hat_train.shape == (n_train, num_mcmc))
        assert (bcf_model.y_hat_test.shape == (n_test, num_mcmc))
        assert (bcf_model.mu_hat_test.shape == (n_test, num_mcmc))
        assert (bcf_model.tau_hat_test.shape == (n_test, num_mcmc))

        # Check overall prediction method
        tau_hat, mu_hat, y_hat = bcf_model.predict(X_test, Z_test, pi_test)
        assert (tau_hat.shape == (n_test, num_mcmc))
        assert (mu_hat.shape == (n_test, num_mcmc))
        assert (y_hat.shape == (n_test, num_mcmc))
        
        # Check treatment effect prediction method
        tau_hat = bcf_model.predict_tau(X_test, Z_test, pi_test)
        assert (tau_hat.shape == (n_test, num_mcmc))

        # Run BCF without test set and with propensity score
        bcf_model = BCFModel()
        bcf_model.sample(X_train=X_train, Z_train=Z_train, y_train=y_train, pi_train=pi_train, 
                         num_gfr=num_gfr, num_burnin=num_burnin, num_mcmc=num_mcmc)

        # Assertions
        assert (bcf_model.y_hat_train.shape == (n_train, num_mcmc))
        assert (bcf_model.mu_hat_train.shape == (n_train, num_mcmc))
        assert (bcf_model.tau_hat_train.shape == (n_train, num_mcmc))

        # Check overall prediction method
        tau_hat, mu_hat, y_hat = bcf_model.predict(X_test, Z_test, pi_test)
        assert (tau_hat.shape == (n_test, num_mcmc))
        assert (mu_hat.shape == (n_test, num_mcmc))
        assert (y_hat.shape == (n_test, num_mcmc))
        
        # Check treatment effect prediction method
        tau_hat = bcf_model.predict_tau(X_test, Z_test, pi_test)
        assert (tau_hat.shape == (n_test, num_mcmc))

        # Run BCF with test set and without propensity score
        bcf_model = BCFModel()
        bcf_model.sample(X_train=X_train, Z_train=Z_train, y_train=y_train, 
                         X_test=X_test, Z_test=Z_test, num_gfr=num_gfr, 
                         num_burnin=num_burnin, num_mcmc=num_mcmc)

        # Assertions
        assert (bcf_model.y_hat_train.shape == (n_train, num_mcmc))
        assert (bcf_model.mu_hat_train.shape == (n_train, num_mcmc))
        assert (bcf_model.tau_hat_train.shape == (n_train, num_mcmc))
        assert (bcf_model.bart_propensity_model.y_hat_train.shape == (n_train, 10))
        assert (bcf_model.y_hat_test.shape == (n_test, num_mcmc))
        assert (bcf_model.mu_hat_test.shape == (n_test, num_mcmc))
        assert (bcf_model.tau_hat_test.shape == (n_test, num_mcmc))
        assert (bcf_model.bart_propensity_model.y_hat_test.shape == (n_test, 10))

        # Check overall prediction method
        tau_hat, mu_hat, y_hat = bcf_model.predict(X_test, Z_test)
        assert (tau_hat.shape == (n_test, num_mcmc))
        assert (mu_hat.shape == (n_test, num_mcmc))
        assert (y_hat.shape == (n_test, num_mcmc))
        
        # Check treatment effect prediction method
        tau_hat = bcf_model.predict_tau(X_test, Z_test)
        assert (tau_hat.shape == (n_test, num_mcmc))

        # Run BCF without test set and without propensity score
        bcf_model = BCFModel()
        bcf_model.sample(X_train=X_train, Z_train=Z_train, y_train=y_train, 
                         num_gfr=num_gfr, num_burnin=num_burnin, num_mcmc=num_mcmc)

        # Assertions
        assert (bcf_model.y_hat_train.shape == (n_train, num_mcmc))
        assert (bcf_model.mu_hat_train.shape == (n_train, num_mcmc))
        assert (bcf_model.tau_hat_train.shape == (n_train, num_mcmc))
        assert (bcf_model.bart_propensity_model.y_hat_train.shape == (n_train, 10))

        # Check overall prediction method
        tau_hat, mu_hat, y_hat = bcf_model.predict(X_test, Z_test)
        assert (tau_hat.shape == (n_test, num_mcmc))
        assert (mu_hat.shape == (n_test, num_mcmc))
        assert (y_hat.shape == (n_test, num_mcmc))
        
        # Check treatment effect prediction method
        tau_hat = bcf_model.predict_tau(X_test, Z_test)
    
    def test_multivariate_bcf(self):
        # RNG
        random_seed = 101
        rng = np.random.default_rng(random_seed)

        # Generate covariates and basis
        n = 100
        p_X = 5
        X = rng.uniform(0, 1, (n, p_X))
        pi_X = np.c_[0.25 + 0.5*X[:,0], 0.5 - 0.25*X[:,1]]
        Z = pi_X + rng.normal(0, 1, (n,2))
        treatment_dim = Z.shape[1]

        # Define the outcome mean functions (prognostic and treatment effects)
        mu_X = pi_X[:,0]*5
        tau_X = np.c_[X[:,1]*2,-0.5*X[:,2]]

        # Generate outcome
        epsilon = rng.normal(0, 1, n)
        treatment_term = (tau_X*Z).sum(axis=1)
        y = mu_X + treatment_term + epsilon

        # Test-train split
        sample_inds = np.arange(n)
        train_inds, test_inds = train_test_split(sample_inds, test_size=0.5)
        X_train = X[train_inds,:]
        X_test = X[test_inds,:]
        Z_train = Z[train_inds]
        Z_test = Z[test_inds]
        y_train = y[train_inds]
        y_test = y[test_inds]
        pi_train = pi_X[train_inds]
        pi_test = pi_X[test_inds]
        mu_train = mu_X[train_inds]
        mu_test = mu_X[test_inds]
        tau_train = tau_X[train_inds]
        tau_test = tau_X[test_inds]
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        # BCF settings
        num_gfr = 10
        num_burnin = 0
        num_mcmc = 10
        
        # Run BCF with test set and propensity score
        bcf_model = BCFModel()
        bcf_model.sample(X_train=X_train, Z_train=Z_train, y_train=y_train, pi_train=pi_train, 
                         X_test=X_test, Z_test=Z_test, pi_test=pi_test, num_gfr=num_gfr, 
                         num_burnin=num_burnin, num_mcmc=num_mcmc)

        # Assertions
        assert (bcf_model.y_hat_train.shape == (n_train, num_mcmc))
        assert (bcf_model.mu_hat_train.shape == (n_train, num_mcmc))
        assert (bcf_model.tau_hat_train.shape == (n_train, num_mcmc, treatment_dim))
        assert (bcf_model.y_hat_test.shape == (n_test, num_mcmc))
        assert (bcf_model.mu_hat_test.shape == (n_test, num_mcmc))
        assert (bcf_model.tau_hat_test.shape == (n_test, num_mcmc, treatment_dim))

        # Check overall prediction method
        tau_hat, mu_hat, y_hat = bcf_model.predict(X_test, Z_test, pi_test)
        assert (tau_hat.shape == (n_test, num_mcmc, treatment_dim))
        assert (mu_hat.shape == (n_test, num_mcmc))
        assert (y_hat.shape == (n_test, num_mcmc))
        
        # Check treatment effect prediction method
        tau_hat = bcf_model.predict_tau(X_test, Z_test, pi_test)
        assert (tau_hat.shape == (n_test, num_mcmc, treatment_dim))

        # Run BCF without test set and with propensity score
        bcf_model = BCFModel()
        bcf_model.sample(X_train=X_train, Z_train=Z_train, y_train=y_train, pi_train=pi_train, 
                         num_gfr=num_gfr, num_burnin=num_burnin, num_mcmc=num_mcmc)

        # Assertions
        assert (bcf_model.y_hat_train.shape == (n_train, num_mcmc))
        assert (bcf_model.mu_hat_train.shape == (n_train, num_mcmc))
        assert (bcf_model.tau_hat_train.shape == (n_train, num_mcmc, treatment_dim))

        # Check overall prediction method
        tau_hat, mu_hat, y_hat = bcf_model.predict(X_test, Z_test, pi_test)
        assert (tau_hat.shape == (n_test, num_mcmc, treatment_dim))
        assert (mu_hat.shape == (n_test, num_mcmc))
        assert (y_hat.shape == (n_test, num_mcmc))
        
        # Check treatment effect prediction method
        tau_hat = bcf_model.predict_tau(X_test, Z_test, pi_test)
        assert (tau_hat.shape == (n_test, num_mcmc, treatment_dim))

        # Run BCF with test set and without propensity score
        with pytest.raises(ValueError):
          bcf_model = BCFModel()
          bcf_model.sample(X_train=X_train, Z_train=Z_train, y_train=y_train, 
                           X_test=X_test, Z_test=Z_test, num_gfr=num_gfr, 
                           num_burnin=num_burnin, num_mcmc=num_mcmc)

        # Run BCF without test set and without propensity score
        with pytest.raises(ValueError):
          bcf_model = BCFModel()
          bcf_model.sample(X_train=X_train, Z_train=Z_train, y_train=y_train, 
                          num_gfr=num_gfr, num_burnin=num_burnin, num_mcmc=num_mcmc)