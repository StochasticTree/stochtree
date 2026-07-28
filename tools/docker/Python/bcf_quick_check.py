# Minimal BCF divergence check
# Run on two platforms and compare the printed output

# Load libraries
import numpy as np
from scipy.stats import norm
from stochtree import BCFModel

# Generate synthetic data
rng = np.random.default_rng(1234)
n = 10000
p = 5
X = rng.uniform(size=(n, p))
mu_x = 1 + 2 * X[:, 0] + X[:, 1]
pi_x = norm.cdf(0.5 * X[:, 0] - 0.5 * X[:, 2])
Z = rng.binomial(1, pi_x)
tau_x = 1 + 0.5 * X[:, 3]
y = mu_x + tau_x * Z + rng.normal(0, 10, size=n)

# Run BCF
bcf_model = BCFModel()
bcf_model.sample(
    X_train=X,
    Z_train=Z,
    y_train=y,
    propensity_train=pi_x,
    num_gfr=5,
    num_burnin=0,
    num_mcmc=100,
    general_params={"random_seed": 1234},
)

# Extract parameters and estimates
sigma2 = bcf_model.global_var_samples
y_hat = bcf_model.predict(X, Z, propensity=pi_x, type="mean", terms="y_hat")
tau_hat = bcf_model.predict(X, Z, propensity=pi_x, type="mean", terms="tau")

# Print first 10 samples of each
print("sigma2[0:10]:", np.round(sigma2[:10], 8))
print("y_hat[0:10]: ", np.round(y_hat[:10], 8))
print("tau_hat[0:10]:", np.round(tau_hat[:10], 8))
