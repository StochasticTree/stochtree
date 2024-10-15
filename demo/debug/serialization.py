import numpy as np
from stochtree import (
    BARTModel, JSONSerializer, ForestContainer, Dataset, Residual, 
    RNG, ForestSampler, ForestContainer, GlobalVarianceModel
)

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

# Sampler parameters
alpha = 0.9
beta = 1.25
min_samples_leaf = 1
num_trees = 100
cutpoint_grid_size = 100
global_variance_init = 1.
tau_init = 0.5
leaf_prior_scale = np.array([[tau_init]], order='C')
a_global = 4.
b_global = 2.
a_leaf = 2.
b_leaf = 0.5
leaf_regression = True
feature_types = np.repeat(0, p_X).astype(int) # 0 = numeric
var_weights = np.repeat(1/p_X, p_X)

# Dataset (covariates and basis)
dataset = Dataset()
dataset.add_covariates(X)
dataset.add_basis(W)

# Residual
residual = Residual(resid)

# Forest samplers and temporary tracking data structures
forest_container = ForestContainer(num_trees, W.shape[1], False, False)
forest_sampler = ForestSampler(dataset, feature_types, num_trees, n, alpha, beta, min_samples_leaf)
cpp_rng = RNG(random_seed)
global_var_model = GlobalVarianceModel()

# Prepare to run sampler
num_warmstart = 10
num_mcmc = 100
num_samples = num_warmstart + num_mcmc
global_var_samples = np.concatenate((np.array([global_variance_init]), np.repeat(0, num_samples)))

# Run "grow-from-root" sampler
for i in range(num_warmstart):
    forest_sampler.sample_one_iteration(forest_container, dataset, residual, cpp_rng, feature_types, cutpoint_grid_size, leaf_prior_scale, var_weights, 1., 1., global_var_samples[i], 1, True, False)
    global_var_samples[i+1] = global_var_model.sample_one_iteration(residual, cpp_rng, a_global, b_global)

# Run MCMC sampler
for i in range(num_warmstart, num_samples):
    forest_sampler.sample_one_iteration(forest_container, dataset, residual, cpp_rng, feature_types, cutpoint_grid_size, leaf_prior_scale, var_weights, 1., 1., global_var_samples[i], 1, False, False)
    global_var_samples[i+1] = global_var_model.sample_one_iteration(residual, cpp_rng, a_global, b_global)

# Extract predictions from the sampler
y_hat_orig = forest_container.predict(dataset)

# "Round-trip" the forest to JSON string and back and check that the predictions agree
forest_json_string = forest_container.dump_json_string()
forest_container_reloaded = ForestContainer(num_trees, W.shape[1], False, False)
forest_container_reloaded.load_from_json_string(forest_json_string)
y_hat_reloaded = forest_container_reloaded.predict(dataset)
np.testing.assert_almost_equal(y_hat_orig, y_hat_reloaded)