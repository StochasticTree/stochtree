# Multiple Initializations Demo Script

# Load necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import arviz as az
from sklearn.model_selection import train_test_split

from stochtree import BARTModel

# Generate sample data
# RNG
random_seed = 1234
rng = np.random.default_rng(random_seed)

# Generate covariates and basis
n = 500
p_X = 10
p_W = 1
X = rng.uniform(0, 1, (n, p_X))
W = rng.uniform(0, 1, (n, p_W))


# Define the outcome mean function
def outcome_mean(X, W):
    return np.where(
        (X[:, 0] >= 0.0) & (X[:, 0] < 0.25),
        -7.5 * W[:, 0],
        np.where(
            (X[:, 0] >= 0.25) & (X[:, 0] < 0.5),
            -2.5 * W[:, 0],
            np.where((X[:, 0] >= 0.5) & (X[:, 0] < 0.75), 2.5 * W[:, 0], 7.5 * W[:, 0]),
        ),
    )


# Generate outcome
f_XW = outcome_mean(X, W)
snr = 3
noise_sd = np.std(f_XW) / snr
epsilon = rng.normal(0, noise_sd, n)
y = f_XW + epsilon

# Test-train split
sample_inds = np.arange(n)
train_inds, test_inds = train_test_split(
    sample_inds, test_size=0.5, random_state=random_seed
)
X_train = X[train_inds, :]
X_test = X[test_inds, :]
basis_train = W[train_inds, :]
basis_test = W[test_inds, :]
y_train = y[train_inds]
y_test = y[test_inds]

# Run the GFR algorithm for a small number of iterations
num_warmstart = 10
xbart_model = BARTModel()
xbart_model.sample(
    X_train=X_train,
    y_train=y_train,
    leaf_basis_train=basis_train,
    X_test=X_test,
    leaf_basis_test=basis_test,
    num_gfr=num_warmstart,
    num_mcmc=0,
)
xbart_model_json = xbart_model.to_json()

# Run several BART MCMC chains from the last GFR forest
num_mcmc = 5000
num_burnin = 2000
num_chains = 4
bart_model = BARTModel()
bart_model.sample(
    X_train=X_train,
    y_train=y_train,
    leaf_basis_train=basis_train,
    X_test=X_test,
    leaf_basis_test=basis_test,
    num_gfr=0,
    num_burnin=num_burnin,
    num_mcmc=num_mcmc,
    previous_model_json=xbart_model_json,
    previous_model_warmstart_sample_num=num_warmstart - 1,
    general_params={"num_chains": num_chains}
)

# Analyze model predictions collectively across all chains
y_hat_test = bart_model.predict(
  covariates = X_test,
  basis = basis_test,
  type = "mean", 
  terms = "y_hat"
)
plt.scatter(y_hat_test, y_test)
plt.xlabel("Estimated conditional mean")
plt.ylabel("Actual outcome")
plt.axline((0, 0), slope=1, color="black", linestyle=(0, (3, 3)))

# Analyze each chain's parameter samples
sigma2_samples = bart_model.global_var_samples
sigma2_samples_by_chain = {"sigma2": np.reshape(sigma2_samples, (num_chains, num_mcmc))}
az.plot_trace(sigma2_samples_by_chain)
