# Reproducibility check script:
# Run BCF on a simple dataset and compare results across platforms

# Load libraries
import numpy as np
import pandas as pd
from scipy.stats import norm
from stochtree import BCFModel

# Set seed for reproducibility
random_seed = 1234
np.random.seed(random_seed)

# Generate data
n = 200
p = 10
X = np.random.uniform(size=(n, p))
mu_x = np.sin(2 * np.pi * X[:, 0]) + np.cos(2 * np.pi * X[:, 1])
pi_x = (
    0.8 * norm.cdf(3 * mu_x / np.std(mu_x) - 0.5 * X[:, 2])
    + 0.05
    + np.random.uniform(size=n) / 10
)
Z = np.random.binomial(1, pi_x)
tau_x = 1 + 2 * X[:, 3] * X[:, 4]
f_XZ = mu_x + tau_x * Z
eps = np.random.normal(0, 1, n)
y = f_XZ + eps

# Fit BCF with default settings
bcf_model = BCFModel()
bcf_model.sample(
    X_train=X,
    Z_train=Z,
    y_train=y,
    propensity_train=pi_x,
    general_params={"random_seed": random_seed},
)

# Obtain traceplot of global error scale
global_error_scale_trace = bcf_model.global_var_samples

# Obtain in-sample posterior mean estimate of E[y | X, Z]
y_hat_in_sample = bcf_model.predict(
    X=X, Z=Z, propensity=pi_x, type="mean", terms="y_hat"
)

# Compare to stored results
global_error_scale_trace_comparison = pd.read_csv(
    "tools/reproducibility/python/bcf_global_error_scale_trace.csv"
).iloc[:, 0].values
y_hat_in_sample_comparison = pd.read_csv(
    "tools/reproducibility/python/bcf_y_hat_in_sample.csv"
).iloc[:, 0].values
TOL = 0.000001
y_hat_mismatch_loc = np.abs(y_hat_in_sample_comparison - y_hat_in_sample) >= TOL
global_error_scale_mismatch_loc = (
    np.abs(global_error_scale_trace_comparison - global_error_scale_trace) >= TOL
)
if np.any(y_hat_mismatch_loc):
    indices = np.where(y_hat_mismatch_loc)[0]
    print("Differences in posterior mean:")
    for idx in indices:
        print(
            f"  {idx + 1}: {y_hat_in_sample_comparison[idx]} vs "
            f"{y_hat_in_sample[idx]}"
        )
else:
    print("No mismatches found in the posterior mean")
if np.any(global_error_scale_mismatch_loc):
    indices = np.where(global_error_scale_mismatch_loc)[0]
    print("Differences in global error scale trace:")
    for idx in indices:
        print(
            f"  {idx + 1}: {global_error_scale_trace_comparison[idx]} vs "
            f"{global_error_scale_trace[idx]}"
        )
else:
    print("No mismatches found in the global error scale trace")
