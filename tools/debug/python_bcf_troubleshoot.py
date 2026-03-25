# Load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from stochtree import BCFModel

# Helper functions
def g(x):
    return np.where(x[:, 4] == 1, 2, np.where(x[:, 4] == 2, -1, -4))

def mu1(x):
    return 1 + g(x) + x[:, 0] * x[:, 2]

def mu2(x):
    return 1 + g(x) + 6 * np.abs(x[:, 2] - 1)

def tau1(x):
    return np.full(x.shape[0], 3.0)

def tau2(x):
    return 1 + 2 * x[:, 1] * x[:, 3]

rng = np.random.default_rng(101)

# Generate data
n = 500
snr = 3
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
x3 = rng.normal(size=n)
x4 = rng.binomial(1, 0.5, n).astype(float)
x5 = rng.choice([1, 2, 3], size=n).astype(float)
X = np.column_stack([x1, x2, x3, x4, x5])
mu_x = mu1(X)
tau_x = tau2(X)
pi_x = (0.8 * norm.cdf((3 * mu_x / np.std(mu_x)) - 0.5 * X[:, 0])
        + 0.05 + rng.uniform(size=n) / 10)
Z = rng.binomial(1, pi_x, n).astype(float)
E_XZ = mu_x + Z * tau_x
y = E_XZ + rng.normal(size=n) * (np.std(E_XZ) / snr)

# Convert to DataFrame with ordered categoricals (matching R's factor(..., ordered=TRUE))
X_df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5})
X_df["x4"] = pd.Categorical(X_df["x4"].astype(int), categories=[0, 1], ordered=True)
X_df["x5"] = pd.Categorical(X_df["x5"].astype(int), categories=[1, 2, 3], ordered=True)

# Split data into test and train sets
test_set_pct = 0.2
n_test = round(test_set_pct * n)
n_train = n - n_test
test_inds = rng.choice(n, n_test, replace=False)
train_inds = np.setdiff1d(np.arange(n), test_inds)
X_test  = X_df.iloc[test_inds]
X_train = X_df.iloc[train_inds]
pi_test, pi_train   = pi_x[test_inds], pi_x[train_inds]
Z_test,  Z_train    = Z[test_inds],    Z[train_inds]
y_test,  y_train    = y[test_inds],    y[train_inds]
mu_test, mu_train   = mu_x[test_inds], mu_x[train_inds]
tau_test, tau_train = tau_x[test_inds], tau_x[train_inds]

# Sample the model
bcf_model = BCFModel()
bcf_model.sample(
    X_train=X_train, 
    Z_train=Z_train, 
    y_train=y_train, 
    propensity_train=pi_train,
    X_test=X_test, 
    Z_test=Z_test, 
    num_gfr=10,
    num_burnin=1000,
    num_mcmc=100,
    propensity_test=pi_test, 
    general_params={"num_threads": 1, "num_chains": 4},
)

# Plot true versus estimated prognostic function
mu_hat_test = bcf_model.predict(X=X_test, Z=Z_test, propensity=pi_test, terms="prognostic_function")
sigma_observed = np.var(y - E_XZ)
mu_pred = mu_hat_test.mean(axis=1)
lo, hi = min(mu_pred.min(), mu_test.min()), max(mu_pred.max(), mu_test.max())
plt.close()
plt.scatter(mu_pred, mu_test, alpha=0.5)
plt.plot([lo, hi], [lo, hi], color="red", linestyle="dashed", linewidth=2)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Prognostic function")
plt.show()

# Plot true versus estimated CATE function
tau_hat_test = bcf_model.predict(X=X_test, Z=Z_test, propensity=pi_test, terms="cate")
tau_pred = tau_hat_test.mean(axis=1)
lo, hi = min(tau_pred.min(), tau_test.min()), max(tau_pred.max(), tau_test.max())
plt.close()
plt.scatter(tau_pred, tau_test, alpha=0.5)
plt.plot([lo, hi], [lo, hi], color="red", linestyle="dashed", linewidth=2)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Treatment effect")
plt.show()
rmse = np.sqrt(np.mean((tau_pred - tau_test) ** 2))
corr = np.corrcoef(tau_pred, tau_test)[0, 1]
print(f"RMSE between predicted and actual treatment effects: {rmse:.2f}")
print(f"Correlation between predicted and actual treatment effects: {corr:.2f}")

# Inspect sigma^2 traceplot
global_var_samples = bcf_model.extract_parameter("sigma2_global")
plt.close()
plt.plot(global_var_samples)
plt.axhline(sigma_observed, color="blue", linestyle="dashed", linewidth=2)
plt.xlabel("Sample")
plt.ylabel(r"$\sigma^2$")
plt.title("Global variance parameter")
plt.show()

# Assess CATE function coverage
test_lb = np.quantile(tau_hat_test, 0.025, axis=1)
test_ub = np.quantile(tau_hat_test, 0.975, axis=1)
cover = ((test_lb <= tau_x[test_inds]) &
  (test_ub >= tau_x[test_inds]))
coverage = np.mean(cover)
print(f"Coverage of 95% credible intervals of CATE function: {coverage*100:.2f}%")
