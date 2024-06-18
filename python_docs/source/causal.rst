Causal Inference
================

This vignette provides a quick overview (using simulated data) of how to use ``stochtree`` for causal inference.
Start by loading stochtree's ``BCFModel`` class and a number of other packages.

.. code-block:: python

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from stochtree import BCFModel
    from sklearn.model_selection import train_test_split

Now, we generate a simulated causal inference problem

.. code-block:: python

    # RNG
    random_seed = 101
    rng = np.random.default_rng(random_seed)

    # Generate covariates and basis
    n = 1000
    p_X = 5
    X = rng.uniform(0, 1, (n, p_X))
    pi_X = 0.25 + 0.5*X[:,0]
    Z = rng.binomial(1, pi_X, n).astype(float)

    # Define the outcome mean functions (prognostic and treatment effects)
    mu_X = pi_X*5
    # tau_X = np.sin(X[:,1]*2*np.pi)
    tau_X = X[:,1]*2

    # Generate outcome
    epsilon = rng.normal(0, 1, n)
    y = mu_X + tau_X*Z + epsilon

Split the dataset into train and test sets

.. code-block:: python

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

Initialize and run a BCF sampler for 1000 iterations (after 10 "warm-start" draws)

.. code-block:: python

    bcf_model = BCFModel()
    bcf_model.sample(X_train, Z_train, y_train, pi_train, X_test, Z_test, pi_test, num_gfr=10, num_mcmc=1000)
