# Load libraries
from stochtree import BCFModel
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import norm

# Simulation parameters
n = 250
p = 50
n_sim = 100
test_set_pct = 0.2
rng = np.random.default_rng()

# Simulation containers
rmses_cached = np.empty(n_sim)
rmses_pred = np.empty(n_sim)

# Run the simulation
for i in range(n_sim):
    # Generate data
    X = rng.normal(loc=0.0, scale=1.0, size=(n, p))
    mu_X = X[:, 0]
    tau_X = 0.25 * X[:, 1]
    pi_X = norm.cdf(0.5 * X[:, 1])
    Z = rng.binomial(n=1, p=pi_X, size=(n,))
    E_XZ = mu_X + tau_X * Z
    snr = 2.0
    noise_sd = np.std(E_XZ) / snr
    y = E_XZ + rng.normal(loc=0.0, scale=noise_sd, size=(n,))

    # Train-test split
    sample_inds = np.arange(n)
    train_inds, test_inds = train_test_split(sample_inds, test_size=test_set_pct)
    X_train = X[train_inds, :]
    X_test = X[test_inds, :]
    Z_train = Z[train_inds]
    Z_test = Z[test_inds]
    pi_train = pi_X[train_inds]
    pi_test = pi_X[test_inds]
    tau_train = tau_X[train_inds]
    tau_test = tau_X[test_inds]
    mu_train = mu_X[train_inds]
    mu_test = mu_X[test_inds]
    y_train = y[train_inds]
    y_test = y[test_inds]
    E_XZ_train = E_XZ[train_inds]
    E_XZ_test = E_XZ[test_inds]

    # Fit simple BCF model
    bcf_model = BCFModel()
    bcf_model.sample(
        X_train=X_train,
        Z_train=Z_train,
        propensity_train=pi_train,
        y_train=y_train,
        X_test=X_test,
        Z_test=Z_test,
        propensity_test=pi_test,
    )

    # Predict out of sample
    y_hat_test = bcf_model.predict(X=X_test, Z=Z_test, propensity=pi_test, type="mean", terms = "y_hat")

    # Compute RMSE using both cached predictions and those returned by predict()
    rmses_cached[i] = np.sqrt(np.mean(np.power(np.mean(bcf_model.y_hat_test, axis = 1) - E_XZ_test, 2.0)))
    rmses_pred[i] = np.sqrt(np.mean(np.power(y_hat_test - E_XZ_test, 2.0)))

print(f"Average RMSE, cached: {np.mean(rmses_cached):.4f}, out-of-sample pred: {np.mean(rmses_pred):.4f}")
