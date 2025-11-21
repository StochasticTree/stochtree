# Supervised Learning Demo Script

# Load necessary libraries
import numpy as np
from stochtree import BARTModel
from sklearn.model_selection import train_test_split
import timeit

# Generate sample data
# RNG
random_seed = 1234
rng = np.random.default_rng(random_seed)

# Generate covariates and basis
n = 1000
p = 100
X = rng.uniform(0, 1, (n, p))

# Define the outcome mean function
def outcome_mean(X):
    return (
        np.sin(4*np.pi*X[:,0]) + np.cos(4*np.pi*X[:,1]) + np.sin(4*np.pi*X[:,2]) + np.cos(4*np.pi*X[:,3])
    )

# Generate outcome
snr = 2
f_X = outcome_mean(X)
noise_sd = np.std(f_X) / snr
epsilon = rng.normal(0, 1, n) * noise_sd
y = f_X + epsilon

# Test-train split
sample_inds = np.arange(n)
train_inds, test_inds = train_test_split(sample_inds, test_size=0.2)
X_train = X[train_inds,:]
X_test = X[test_inds,:]
y_train = y[train_inds]
y_test = y[test_inds]

# Run XBART with the full feature set
s = """\
bart_model_a = BARTModel()
forest_config_a = {"num_trees": 100}
bart_model_a.sample(X_train=X_train, y_train=y_train, X_test=X_test, num_gfr=100, num_mcmc=0, mean_forest_params=forest_config_a)
"""
timing_no_subsampling = timeit.timeit(stmt=s, number=5, globals=globals())
print(f"Average runtime, without feature subsampling (p = {p:d}): {timing_no_subsampling:.2f}")

# Run XBART with each tree considering random subsets of 5 features
s = """\
bart_model_b = BARTModel()
forest_config_b = {"num_trees": 100, "num_features_subsample": 5}
bart_model_b.sample(X_train=X_train, y_train=y_train, X_test=X_test, num_gfr=100, num_mcmc=0, mean_forest_params=forest_config_b)
"""
timing_subsampling = timeit.timeit(stmt=s, number=5, globals=globals())
print(f"Average runtime, subsampling 5 out of {p:d} features: {timing_subsampling:.2f}")

# Compare RMSEs of each model
bart_model_a = BARTModel()
forest_config_a = {"num_trees": 100}
bart_model_a.sample(X_train=X_train, y_train=y_train, X_test=X_test, num_gfr=100, num_mcmc=0, mean_forest_params=forest_config_a)
bart_model_b = BARTModel()
forest_config_b = {"num_trees": 100, "num_features_subsample": 5}
bart_model_b.sample(X_train=X_train, y_train=y_train, X_test=X_test, num_gfr=100, num_mcmc=0, mean_forest_params=forest_config_b)
y_hat_test_a = np.squeeze(bart_model_a.y_hat_test).mean(axis = 1)
rmse_no_subsampling = np.sqrt(np.mean(np.power(y_test - y_hat_test_a,2)))
print(f"Test set RMSE, no subsampling (p = {p:d}): {rmse_no_subsampling:.2f}")
y_hat_test_b = np.squeeze(bart_model_b.y_hat_test).mean(axis = 1)
rmse_subsampling = np.sqrt(np.mean(np.power(y_test - y_hat_test_b,2)))
print(f"Test set RMSE, subsampling 5 out of {p:d} features: {rmse_subsampling:.2f}")
