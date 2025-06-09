# Supervised Learning Demo Script

# Load necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from stochtree import BARTModel
from sklearn.model_selection import train_test_split

# Generate sample data
# RNG
random_seed = 1234
rng = np.random.default_rng(random_seed)

# Generate covariates and basis
n = 1000
p_X = 20
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
train_inds, test_inds = train_test_split(sample_inds, test_size=0.2)
X_train = X[train_inds,:]
X_test = X[test_inds,:]
y_train = y[train_inds]
y_test = y[test_inds]

# Run XBART with the full feature set
bart_model_a = BARTModel()
forest_config_a = {"num_trees": 100}
bart_model_a.sample(X_train=X_train, y_train=y_train, X_test=X_test, num_gfr=100, num_mcmc=0, mean_forest_params=forest_config_a)

# Run XBART with each tree considering random subsets of 5 features
bart_model_b = BARTModel()
forest_config_b = {"num_trees": 100, "num_features_subsample": 5}
bart_model_b.sample(X_train=X_train, y_train=y_train, X_test=X_test, num_gfr=100, num_mcmc=0, mean_forest_params=forest_config_b)
