# Load necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from stochtree import BCFModel
from sklearn.model_selection import train_test_split
from scipy.stats import norm

# Generate sample data
# RNG
random_seed = 101
rng = np.random.default_rng(random_seed)

# Generate covariates and basis
n = 4000
x1 = rng.normal(loc=0., scale=1., size=(n,))
x2 = rng.normal(loc=0., scale=1., size=(n,))
x3 = rng.normal(loc=0., scale=1., size=(n,))
x4 = rng.binomial(n=1,p=0.5,size=(n,))
x5 = rng.choice(a=[0,1,2], size=(n,), replace=True)
x4_cat = pd.Categorical(x4, categories=[0,1], ordered=True)
x5_cat = pd.Categorical(x4, categories=[0,1,2], ordered=True)
p = 5
X = pd.DataFrame(data={
    "x1": pd.Series(x1),
    "x2": pd.Series(x2),
    "x3": pd.Series(x3),
    "x4": pd.Series(x4_cat),
    "x5": pd.Series(x5_cat)
})
def g(x5):
    return np.where(
        x5 == 0, 2.0, 
        np.where(
            x5 == 1, -1.0, -4.0
        )
    )
mu_x = (1.0 + g(x5) + x1*x3)*0.25
tau_x = (1.0 + 2*x2*x4)*0.5
pi_x = (
    0.8*norm.cdf(3.0*mu_x / np.squeeze(np.std(mu_x)) - 0.5*x1) + 
    0.05 + rng.uniform(low=0., high=0.1, size=(n,))
)
Z = rng.binomial(n=1, p=pi_x, size=(n,))
E_XZ = mu_x + tau_x*Z
w = E_XZ + rng.normal(loc=0., scale=1., size=(n,))
y = np.where(w > 0, 1, 0)
delta_x = norm.cdf(mu_x + tau_x) - norm.cdf(mu_x)

# Test-train split
sample_inds = np.arange(n)
train_inds, test_inds = train_test_split(sample_inds, test_size=0.2)
X_train = X.iloc[train_inds,:]
X_test = X.iloc[test_inds,:]
Z_train = Z[train_inds]
Z_test = Z[test_inds]
y_train = y[train_inds]
y_test = y[test_inds]
pi_train = pi_x[train_inds]
pi_test = pi_x[test_inds]
mu_train = mu_x[train_inds]
mu_test = mu_x[test_inds]
tau_train = tau_x[train_inds]
tau_test = tau_x[test_inds]
w_train = w[train_inds]
w_test = w[test_inds]
delta_train = delta_x[train_inds]
delta_test = delta_x[test_inds]

# Number of iterations
num_gfr = 10
num_burnin = 0
num_mcmc = 100

# Tau prior calibration (this is also done internally in the BCF sampler)
num_trees_tau = 50
p = 0.6827
q_quantile = norm.ppf((p+1)/2)
delta_max = 0.9
sigma2_leaf_tau = ((delta_max/(q_quantile*norm.pdf(0.)))**2) / num_trees_tau

# Mu prior calibration (this is also done internally in the BCF sampler)
num_trees_mu = 200
sigma2_leaf_mu = 2/num_trees_mu

# Construct parameter lists
general_params = {
    'keep_every': 5, 
    'probit_outcome_model': True, 
    'sample_sigma2_global': False, 
    'adaptive_coding': False}
prognostic_forest_params = {
    'sample_sigma2_leaf': False, 
    'sigma2_leaf_init': sigma2_leaf_mu, 
    'num_trees': num_trees_mu}
treatment_effect_forest_params = {
    'sample_sigma2_leaf': False, 
    'sigma2_leaf_init': sigma2_leaf_tau, 
    'num_trees': num_trees_tau}

# Run the sampler
bcf_model = BCFModel()
bcf_model.sample(X_train=X_train, Z_train=Z_train, y_train=y_train, pi_train=pi_train, 
                 X_test=X_test, Z_test=Z_test, pi_test=pi_test, num_gfr=num_gfr, 
                 num_burnin=num_burnin, num_mcmc=num_mcmc, general_params=general_params, 
                 prognostic_forest_params=prognostic_forest_params, 
                 treatment_effect_forest_params=treatment_effect_forest_params)

# Inspect the MCMC (BART) samples
mu_hat_test = np.squeeze(bcf_model.mu_hat_test).mean(axis = 1)
plt.scatter(mu_hat_test, mu_test, color="black")
plt.axline((0, 0), slope=1, color="red", linestyle=(0, (3,3)))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Prognostic function")
plt.show()

tau_hat_test = np.squeeze(bcf_model.tau_hat_test).mean(axis = 1)
plt.scatter(tau_hat_test, tau_test, color="black")
plt.axline((0, 0), slope=1, color="red", linestyle=(0, (3,3)))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Probit-scale treatment effect function")
plt.show()

delta_hat_test = norm.cdf(mu_hat_test + tau_hat_test) - norm.cdf(mu_hat_test)
plt.scatter(delta_hat_test, delta_test, color="black")
plt.axline((0, 0), slope=1, color="red", linestyle=(0, (3,3)))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Distributional treatment effect function")
plt.show()

w_hat_test = np.squeeze(bcf_model.y_hat_test).mean(axis = 1)
plt.scatter(w_hat_test, w_test, color="black")
plt.axline((0, 0), slope=1, color="red", linestyle=(0, (3,3)))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Probit scale latent outcome")
plt.show()

# Compute prediction accuracy
preds_test = w_hat_test > 0
print(f"Test set accuracy: {np.mean(y_test == preds_test):.3f}")

# Compute RMSEs
w_rmse = np.sqrt(np.mean(np.power(w_test - w_hat_test, 2)))
tau_rmse = np.sqrt(np.mean(np.power(tau_test - tau_hat_test, 2)))
mu_rmse = np.sqrt(np.mean(np.power(mu_test - mu_hat_test, 2)))
print("w hat RMSE: {:.2f}".format(w_rmse))
print("tau hat RMSE: {:.2f}".format(tau_rmse))
print("mu hat RMSE: {:.2f}".format(mu_rmse))
