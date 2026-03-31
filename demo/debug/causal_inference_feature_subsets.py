# Supervised Learning Demo Script

# Load necessary libraries
import numpy as np
from stochtree import BCFModel
from sklearn.model_selection import train_test_split
import timeit

# Generate sample data
# RNG
random_seed = 1234
rng = np.random.default_rng(random_seed)

# Generate covariates and basis
n = 1000
p = 100
X = rng.normal(0, 1, (n, p))

# Generate outcome
snr = 2
mu_x = 1 + 2*X[:,0] + np.where(X[:,1] < 0, -4, 4) + 3*(np.abs(X[:,2]) - np.sqrt(2/np.pi))
tau_x = 1 + 2*X[:,3]
u = rng.uniform(0, 1, n)
pi_x = ((mu_x-1.)/4.) + 4*(u-0.5)
Z = pi_x + rng.normal(0, 1, n)
E_XZ = mu_x + Z*tau_x
noise_sd = np.std(E_XZ) / snr
y = E_XZ + rng.normal(0, 1, n)*noise_sd

# Test-train split
sample_inds = np.arange(n)
train_inds, test_inds = train_test_split(sample_inds, test_size=0.2)
X_train = X[train_inds,:]
X_test = X[test_inds,:]
pi_x_train = pi_x[train_inds]
pi_x_test = pi_x[test_inds]
Z_train = Z[train_inds]
Z_test = Z[test_inds]
y_train = y[train_inds]
y_test = y[test_inds]

# Run XBART with the full feature set
s = """\
bcf_model_a = BCFModel()
prog_forest_config_a = {"num_trees": 100}
trt_forest_config_a = {"num_trees": 50}
bcf_model_a.sample(X_train=X_train, Z_train=Z_train, propensity_train=pi_x_train, y_train=y_train, X_test=X_test, Z_test=Z_test, propensity_test=pi_x_test, num_gfr=100, num_mcmc=0, prognostic_forest_params=prog_forest_config_a, treatment_effect_forest_params=trt_forest_config_a)
"""
timing_no_subsampling = timeit.timeit(stmt=s, number=5, globals=globals())
print(f"Average runtime, without feature subsampling (p = {p:d}): {timing_no_subsampling:.2f}")

# Run XBART with each tree considering random subsets of 5 features
s = """\
bcf_model_b = BCFModel()
prog_forest_config_b = {"num_trees": 100, "num_features_subsample": 5}
trt_forest_config_b = {"num_trees": 50, "num_features_subsample": 5}
bcf_model_b.sample(X_train=X_train, Z_train=Z_train, propensity_train=pi_x_train, y_train=y_train, X_test=X_test, Z_test=Z_test, propensity_test=pi_x_test, num_gfr=100, num_mcmc=0, prognostic_forest_params=prog_forest_config_b, treatment_effect_forest_params=trt_forest_config_b)
"""
timing_subsampling = timeit.timeit(stmt=s, number=5, globals=globals())
print(f"Average runtime, subsampling 5 out of {p:d} features: {timing_subsampling:.2f}")

# Compare RMSEs of each model
bcf_model_a = BCFModel()
prog_forest_config_a = {"num_trees": 100}
trt_forest_config_a = {"num_trees": 50}
bcf_model_a.sample(X_train=X_train, Z_train=Z_train, propensity_train=pi_x_train, y_train=y_train, X_test=X_test, Z_test=Z_test, propensity_test=pi_x_test, num_gfr=100, num_mcmc=0, prognostic_forest_params=prog_forest_config_a, treatment_effect_forest_params=trt_forest_config_a)
bcf_model_b = BCFModel()
prog_forest_config_b = {"num_trees": 100, "num_features_subsample": 5}
trt_forest_config_b = {"num_trees": 50, "num_features_subsample": 5}
bcf_model_b.sample(X_train=X_train, Z_train=Z_train, propensity_train=pi_x_train, y_train=y_train, X_test=X_test, Z_test=Z_test, propensity_test=pi_x_test, num_gfr=100, num_mcmc=0, prognostic_forest_params=prog_forest_config_b, treatment_effect_forest_params=trt_forest_config_b)
y_hat_test_a = np.squeeze(bcf_model_a.y_hat_test).mean(axis = 1)
rmse_no_subsampling = np.sqrt(np.mean(np.power(y_test - y_hat_test_a,2)))
print(f"Test set RMSE, no subsampling (p = {p:d}): {rmse_no_subsampling:.2f}")
y_hat_test_b = np.squeeze(bcf_model_b.y_hat_test).mean(axis = 1)
rmse_subsampling = np.sqrt(np.mean(np.power(y_test - y_hat_test_b,2)))
print(f"Test set RMSE, subsampling 5 out of {p:d} features: {rmse_subsampling:.2f}")
