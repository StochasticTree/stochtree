# Reproducibility check script:
# Run BART on a simple dataset and compare results across platforms

# Load libraries
import numpy as np
import pandas as pd
from stochtree import BARTModel

# Set seed for reproducibility
random_seed = 1234
np.random.seed(random_seed)

# Generate data
n = 200
p = 10
X = np.random.uniform(size=(n, p))
f_X = np.sin(2 * np.pi * X[:, 0]) + np.cos(2 * np.pi * X[:, 1])
eps = np.random.normal(0, 1, n)
y = f_X + eps

# Fit BART with default settings
bart_model = BARTModel()
bart_model.sample(
    X_train=X,
    y_train=y,
    general_params={"random_seed": random_seed},
)

# Obtain traceplot of global error scale
global_error_scale_trace = bart_model.global_var_samples

# Obtain in-sample posterior mean estimate of f(X)
y_hat_in_sample = bart_model.predict(X=X, type="mean", terms="y_hat")

# Compare to stored results
global_error_scale_trace_comparison = pd.read_csv(
    "tools/reproducibility/python/bart_global_error_scale_trace.csv"
).iloc[:, 0].values
y_hat_in_sample_comparison = pd.read_csv(
    "tools/reproducibility/python/bart_y_hat_in_sample.csv"
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
