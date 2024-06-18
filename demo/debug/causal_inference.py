# Load necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from stochtree import BCFModel
from sklearn.model_selection import train_test_split

# # Generate sample data
# # RNG
# random_seed = 101
# rng = np.random.default_rng(random_seed)

# # Generate covariates and basis
# n = 1000
# p_X = 5
# X = rng.uniform(0, 1, (n, p_X))
# pi_X = 0.25 + 0.5*X[:,0]
# Z = rng.binomial(1, pi_X, n).astype(float)

# # Define the outcome mean functions (prognostic and treatment effects)
# mu_X = pi_X*5
# # tau_X = np.sin(X[:,1]*2*np.pi)
# tau_X = X[:,1]*2

# # Generate outcome
# epsilon = rng.normal(0, 1, n)
# y = mu_X + tau_X*Z + epsilon

# # Test-train split
# sample_inds = np.arange(n)
# train_inds, test_inds = train_test_split(sample_inds, test_size=0.5)
# X_train = X[train_inds,:]
# X_test = X[test_inds,:]
# Z_train = Z[train_inds]
# Z_test = Z[test_inds]
# y_train = y[train_inds]
# y_test = y[test_inds]
# pi_train = pi_X[train_inds]
# pi_test = pi_X[test_inds]
# mu_train = mu_X[train_inds]
# mu_test = mu_X[test_inds]
# tau_train = tau_X[train_inds]
# tau_test = tau_X[test_inds]

# Load data from CSV
train_df = pd.read_csv("demo/data/python_r_debug_train.csv")
test_df = pd.read_csv("demo/data/python_r_debug_test.csv")

# Unpack into numpy
X_train = train_df.loc[:,["X1","X2","X3","X4","X5"]].to_numpy()
X_test = test_df.loc[:,["X1","X2","X3","X4","X5"]].to_numpy()
Z_train = np.squeeze(train_df.loc[:,["Z"]].to_numpy())
Z_test = np.squeeze(test_df.loc[:,["Z"]].to_numpy())
y_train = np.squeeze(train_df.loc[:,["y"]].to_numpy())
y_test = np.squeeze(test_df.loc[:,["y"]].to_numpy())
pi_train = np.squeeze(train_df.loc[:,["pi"]].to_numpy())
pi_test = np.squeeze(test_df.loc[:,["pi"]].to_numpy())
mu_train = np.squeeze(train_df.loc[:,["mu"]].to_numpy())
mu_test = np.squeeze(test_df.loc[:,["mu"]].to_numpy())
tau_train = np.squeeze(train_df.loc[:,["tau"]].to_numpy())
tau_test = np.squeeze(test_df.loc[:,["tau"]].to_numpy())

# Run BCF
bcf_model = BCFModel()
bcf_model.sample(X_train, Z_train, y_train, pi_train, X_test, Z_test, pi_test, num_gfr=10, num_mcmc=100)

# Inspect the MCMC (BART) samples
forest_preds_y_mcmc = bcf_model.y_hat_test[:,bcf_model.num_gfr:]
y_avg_mcmc = np.squeeze(forest_preds_y_mcmc).mean(axis = 1, keepdims = True)
y_df_mcmc = pd.DataFrame(np.concatenate((np.expand_dims(y_test,1), y_avg_mcmc), axis = 1), columns=["True outcome", "Average estimated outcome"])
sns.scatterplot(data=y_df_mcmc, x="Average estimated outcome", y="True outcome")
plt.axline((0, 0), slope=1, color="black", linestyle=(0, (3,3)))
plt.show()

forest_preds_tau_mcmc = bcf_model.tau_hat_test[:,bcf_model.num_gfr:]
tau_avg_mcmc = np.squeeze(forest_preds_tau_mcmc).mean(axis = 1, keepdims = True)
tau_df_mcmc = pd.DataFrame(np.concatenate((np.expand_dims(tau_test,1), tau_avg_mcmc), axis = 1), columns=["True tau", "Average estimated tau"])
sns.scatterplot(data=tau_df_mcmc, x="Average estimated tau", y="True tau")
plt.axline((0, 0), slope=1, color="black", linestyle=(0, (3,3)))
plt.show()

forest_preds_mu_mcmc = bcf_model.mu_hat_test[:,bcf_model.num_gfr:]
mu_avg_mcmc = np.squeeze(forest_preds_mu_mcmc).mean(axis = 1, keepdims = True)
mu_df_mcmc = pd.DataFrame(np.concatenate((np.expand_dims(mu_test,1), mu_avg_mcmc), axis = 1), columns=["True mu", "Average estimated mu"])
sns.scatterplot(data=mu_df_mcmc, x="Average estimated mu", y="True mu")
plt.axline((0, 0), slope=1, color="black", linestyle=(0, (3,3)))
plt.show()

# sigma_df_mcmc = pd.DataFrame(np.concatenate((np.expand_dims(np.arange(bcf_model.num_samples - bcf_model.num_gfr),axis=1), np.expand_dims(bcf_model.global_var_samples[bcf_model.num_gfr:],axis=1)), axis = 1), columns=["Sample", "Sigma"])
# sns.scatterplot(data=sigma_df_mcmc, x="Sample", y="Sigma")
# plt.show()

# b_df_mcmc = pd.DataFrame(np.concatenate((np.expand_dims(np.arange(bcf_model.num_samples - bcf_model.num_gfr),axis=1), np.expand_dims(bcf_model.b0_samples[bcf_model.num_gfr:],axis=1), np.expand_dims(bcf_model.b1_samples[bcf_model.num_gfr:],axis=1)), axis = 1), columns=["Sample", "Beta_0", "Beta_1"])
# sns.scatterplot(data=b_df_mcmc, x="Sample", y="Beta_0")
# sns.scatterplot(data=b_df_mcmc, x="Sample", y="Beta_1")
# plt.show()

# Compute RMSEs
y_rmse = np.sqrt(np.mean(np.power(np.expand_dims(y_test,1) - y_avg_mcmc, 2)))
tau_rmse = np.sqrt(np.mean(np.power(np.expand_dims(tau_test,1) - tau_avg_mcmc, 2)))
mu_rmse = np.sqrt(np.mean(np.power(np.expand_dims(mu_test,1) - mu_avg_mcmc, 2)))
print("y hat RMSE: {:.2f}".format(y_rmse))
print("tau hat RMSE: {:.2f}".format(tau_rmse))
print("mu hat RMSE: {:.2f}".format(mu_rmse))
