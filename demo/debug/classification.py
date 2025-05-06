import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

from stochtree import BARTModel

# RNG
rng = np.random.default_rng()

# Generate covariates
n = 1000
p_X = 10
X = rng.uniform(0, 1, (n, p_X))


# Define the outcome mean function
def outcome_mean(X):
    return np.where(
        (X[:, 0] >= 0.0) & (X[:, 0] < 0.25),
        -7.5 * X[:, 1],
        np.where(
            (X[:, 0] >= 0.25) & (X[:, 0] < 0.5),
            -2.5 * X[:, 1],
            np.where((X[:, 0] >= 0.5) & (X[:, 0] < 0.75), 2.5 * X[:, 1], 7.5 * X[:, 1]),
        ),
    )


# Generate outcome
epsilon = rng.normal(0, 1, n)
z = outcome_mean(X) + epsilon
y = np.where(z >= 0, 1, 0)

# Test-train split
sample_inds = np.arange(n)
train_inds, test_inds = train_test_split(sample_inds, test_size=0.5)
X_train = X[train_inds, :]
X_test = X[test_inds, :]
z_train = z[train_inds]
z_test = z[test_inds]
y_train = y[train_inds]
y_test = y[test_inds]

# Fit Probit BART
bart_model = BARTModel()
general_params = {"num_chains": 1}
mean_forest_params = {"probit_outcome_model": True}
bart_model.sample(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    num_gfr=10,
    num_mcmc=100,
    general_params=general_params,
    mean_forest_params=mean_forest_params
)
