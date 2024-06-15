Supervised Learning
===================

This vignette provides a quick overview (using simulated data) of how to use ``stochtree`` for supervised learning.
Start by loading stochtree's ``BARTModel`` class and a number of other packages.

.. code-block:: python

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from stochtree import BARTModel
    from sklearn.model_selection import train_test_split

Now, we generate a simulated prediction problem

.. code-block:: python

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

    # Standardize outcome
    y_bar = np.mean(y)
    y_std = np.std(y)
    resid = (y-y_bar)/y_std

Split the dataset into train and test sets

.. code-block:: python

    sample_inds = np.arange(n)
    train_inds, test_inds = train_test_split(sample_inds, test_size=0.5)
    X_train = X[train_inds,:]
    X_test = X[test_inds,:]
    basis_train = W[train_inds,:]
    basis_test = W[test_inds,:]
    y_train = y[train_inds]
    y_test = y[test_inds]

Initialize and run a BART sampler for 100 iterations (after 10 "warm-start" draws)

.. code-block:: python

    bart_model = BARTModel()
    bart_model.sample(X_train=X_train, y_train=y_train, basis_train=basis_train, X_test=X_test, basis_test=basis_test, num_gfr=10, num_mcmc=100)
