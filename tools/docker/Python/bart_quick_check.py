# Minimal BART divergence check
# Run on two platforms and compare the printed output

# Load libraries
import numpy as np
from stochtree import BARTModel

# Generate synthetic data
rng = np.random.default_rng(1234)
n = 10000
p = 5
X = rng.uniform(size=(n, p))
f_x = 1 + 2 * X[:, 0] + X[:, 1]
y = f_x + rng.normal(0, 10, size=n)

# Run BART
bart_model = BARTModel()
bart_model.sample(
    X_train=X,
    y_train=y,
    num_gfr=5,
    num_burnin=0,
    num_mcmc=100,
    general_params={"random_seed": 1234},
)

# Extract parameters and estimates
sigma2 = bart_model.global_var_samples
y_hat = bart_model.predict(X, type="mean", terms="y_hat")

# Print first 10 samples of each
print("sigma2[0:10]:", np.round(sigma2[:10], 8))
print("y_hat[0:10]: ", np.round(y_hat[:10], 8))
