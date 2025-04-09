# Multi Chain Demo Script

# Load necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
epsilon = rng.normal(0, 1, n)
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
general_model_params = {"random_seed": -1}
mean_forest_model_params = {"num_trees": 20}
num_warmstart = 10
num_mcmc = 10
bart_model = BARTModel()
bart_model.sample(
    X_train=X_train,
    y_train=y_train,
    leaf_basis_train=basis_train,
    X_test=X_test,
    leaf_basis_test=basis_test,
    num_gfr=num_warmstart,
    num_mcmc=0,
    general_params=general_model_params,
    mean_forest_params=mean_forest_model_params,
)
bart_model_json = bart_model.to_json()

# Run several BART MCMC samples from the last GFR forest
bart_model_2 = BARTModel()
bart_model_2.sample(
    X_train=X_train,
    y_train=y_train,
    leaf_basis_train=basis_train,
    X_test=X_test,
    leaf_basis_test=basis_test,
    num_gfr=0,
    num_mcmc=num_mcmc,
    previous_model_json=bart_model_json,
    previous_model_warmstart_sample_num=num_warmstart - 1,
    general_params=general_model_params,
    mean_forest_params=mean_forest_model_params,
)

# Run several BART MCMC samples from the second-to-last GFR forest
bart_model_3 = BARTModel()
bart_model_3.sample(
    X_train=X_train,
    y_train=y_train,
    leaf_basis_train=basis_train,
    X_test=X_test,
    leaf_basis_test=basis_test,
    num_gfr=0,
    num_mcmc=num_mcmc,
    previous_model_json=bart_model_json,
    previous_model_warmstart_sample_num=num_warmstart - 2,
    general_params=general_model_params,
    mean_forest_params=mean_forest_model_params,
)

# Run several BART MCMC samples from root
bart_model_4 = BARTModel()
bart_model_4.sample(
    X_train=X_train,
    y_train=y_train,
    leaf_basis_train=basis_train,
    X_test=X_test,
    leaf_basis_test=basis_test,
    num_gfr=0,
    num_mcmc=num_mcmc,
    general_params=general_model_params,
    mean_forest_params=mean_forest_model_params,
)

# Inspect the model outputs
y_hat_mcmc_2 = bart_model_2.predict(X_test, basis_test)
y_avg_mcmc_2 = np.squeeze(y_hat_mcmc_2).mean(axis=1, keepdims=True)
y_hat_mcmc_3 = bart_model_3.predict(X_test, basis_test)
y_avg_mcmc_3 = np.squeeze(y_hat_mcmc_3).mean(axis=1, keepdims=True)
y_hat_mcmc_4 = bart_model_4.predict(X_test, basis_test)
y_avg_mcmc_4 = np.squeeze(y_hat_mcmc_4).mean(axis=1, keepdims=True)
y_df = pd.DataFrame(
    np.concatenate(
        (y_avg_mcmc_2, y_avg_mcmc_3, y_avg_mcmc_4, np.expand_dims(y_test, axis=1)),
        axis=1,
    ),
    columns=["First Chain", "Second Chain", "Third Chain", "Outcome"],
)

# Compare first warm-start chain to root chain with equal number of MCMC draws
sns.scatterplot(data=y_df, x="First Chain", y="Third Chain")
plt.axline((0, 0), slope=1, color="black", linestyle=(0, (3, 3)))
plt.show()

# Compare first warm-start chain to outcome
sns.scatterplot(data=y_df, x="First Chain", y="Outcome")
plt.axline((0, 0), slope=1, color="black", linestyle=(0, (3, 3)))
plt.show()

# Compare root chain to outcome
sns.scatterplot(data=y_df, x="Third Chain", y="Outcome")
plt.axline((0, 0), slope=1, color="black", linestyle=(0, (3, 3)))
plt.show()

# Compute RMSEs
rmse_1 = np.sqrt(
    np.mean((np.squeeze(y_avg_mcmc_2) - y_test) * (np.squeeze(y_avg_mcmc_2) - y_test))
)
rmse_2 = np.sqrt(
    np.mean((np.squeeze(y_avg_mcmc_3) - y_test) * (np.squeeze(y_avg_mcmc_3) - y_test))
)
rmse_3 = np.sqrt(
    np.mean((np.squeeze(y_avg_mcmc_4) - y_test) * (np.squeeze(y_avg_mcmc_4) - y_test))
)
print(
    "Chain 1 rmse: {:0.3f}; Chain 2 rmse: {:0.3f}; Chain 3 rmse: {:0.3f}".format(
        rmse_1, rmse_2, rmse_3
    )
)
