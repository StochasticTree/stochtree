# Reproducing issues in GFR when the covariates have a large number of ties
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from stochtree import BARTModel, BCFModel

# Generate covariates that are essentially categorical, but real-valued
rng = np.random.default_rng(1234)
n = 1000
p = 10
X = np.empty((n, p))
for i in range(p):
    coef_vector = rng.uniform(-5, 5, 15)
    X[:, i] = rng.choice(coef_vector, n, replace=True)
f_X = 5 * X[:, 0] - 2 * X[:, 1] + X[:, 2]
eps = rng.normal(0, 1, n)
y = f_X + eps

# Train test split
test_set_pct = 0.2
train_inds, test_inds = train_test_split(
    np.arange(n), test_size=test_set_pct, random_state=1234
)
n_test = len(test_inds)
n_train = len(train_inds)
X_test = X[test_inds, :]
X_train = X[train_inds, :]
y_test = y[test_inds]
y_train = y[train_inds]
E_y_test = f_X[test_inds]
E_y_train = f_X[train_inds]

# Attempt to fit a GFR-only predictive model
xbart_model = BARTModel()
xbart_model.sample(
    X_train=X_train, y_train=y_train, num_gfr=10, num_burnin=0, num_mcmc=0
)

# Inspect the model fit
y_hat_test = xbart_model.predict(X=X_test, type="mean", terms="y_hat")
plt.scatter(y_hat_test, y_test)
plt.axline((0, 0), slope=1, color="red", linestyle="dashed", linewidth=1.5)
plt.xlabel("Predicted Outcome Mean")
plt.ylabel("True Outcome")
plt.xlim(np.min(y_hat_test) * 0.9, np.max(y_hat_test) * 1.1)
plt.ylim(np.min(y), np.max(y))
plt.show()

# Not great! Let's add a few MCMC samples and see how it shakes out
bart_model = BARTModel()
bart_model.sample(
    X_train=X_train, y_train=y_train, num_gfr=10, num_burnin=0, num_mcmc=10
)

# Inspect the model fit
y_hat_test = bart_model.predict(X=X_test, type="mean", terms="y_hat")
plt.clf()
plt.scatter(y_hat_test, y_test)
plt.axline((0, 0), slope=1, color="red", linestyle="dashed", linewidth=1.5)
plt.xlabel("Predicted Outcome Mean")
plt.ylabel("True Outcome")
plt.xlim(np.min(y_hat_test) * 0.9, np.max(y_hat_test) * 1.1)
plt.ylim(np.min(y), np.max(y))
plt.show()

# Generate covariates with zero-inflation
X = np.empty((n, p))
for i in range(p):
    p_zero = rng.uniform(0.1, 0.9)
    zero_vector = rng.binomial(1, p_zero, n)
    runif_vector = rng.uniform(0, 1, n)
    X[:, i] = zero_vector * 0 + (1 - zero_vector) * runif_vector
f_X = 5 * X[:, 0] - 2 * X[:, 1] + X[:, 2]
eps = rng.normal(0, 1, n)
y = f_X + eps

# Train test split
train_inds, test_inds = train_test_split(
    np.arange(n), test_size=test_set_pct, random_state=1234
)
n_test = len(test_inds)
n_train = len(train_inds)
X_test = X[test_inds, :]
X_train = X[train_inds, :]
y_test = y[test_inds]
y_train = y[train_inds]
E_y_test = f_X[test_inds]
E_y_train = f_X[train_inds]

# Attempt to fit a GFR-only predictive model
xbart_model = BARTModel()
xbart_model.sample(
    X_train=X_train, y_train=y_train, num_gfr=10, num_burnin=0, num_mcmc=0
)

# Inspect the model fit
y_hat_test = xbart_model.predict(X=X_test, type="mean", terms="y_hat")
plt.clf()
plt.scatter(y_hat_test, y_test)
plt.axline((0, 0), slope=1, color="red", linestyle="dashed", linewidth=1.5)
plt.xlabel("Predicted Outcome Mean")
plt.ylabel("True Outcome")
plt.xlim(np.min(y_hat_test) * 0.9, np.max(y_hat_test) * 1.1)
plt.ylim(np.min(y), np.max(y))
plt.show()

# Also not great! Let's add a few MCMC samples and see how it shakes out
bart_model = BARTModel()
bart_model.sample(
    X_train=X_train, y_train=y_train, num_gfr=10, num_burnin=0, num_mcmc=10
)

# Inspect the model fit
y_hat_test = bart_model.predict(X=X_test, type="mean", terms="y_hat")
plt.clf()
plt.scatter(y_hat_test, y_test)
plt.axline((0, 0), slope=1, color="red", linestyle="dashed", linewidth=1.5)
plt.xlabel("Predicted Outcome Mean")
plt.ylabel("True Outcome")
plt.xlim(np.min(y_hat_test) * 0.9, np.max(y_hat_test) * 1.1)
plt.ylim(np.min(y), np.max(y))
plt.show()

# Perform the same checks as above for BCF

# Generate covariates that are essentially categorical, but real-valued
X = np.empty((n, p))
for i in range(p):
    coef_vector = rng.uniform(-5, 5, 15)
    X[:, i] = rng.choice(coef_vector, n, replace=True)
mu_X = 5 * X[:, 0] - 2 * X[:, 1] + X[:, 2]
pi_X = norm.cdf(0.1 * mu_X)
tau_X = X[:, 3]
Z = rng.binomial(1, pi_X, n)
E_Y_XZ = mu_X + tau_X * Z
eps = rng.normal(0, 1, n)
y = E_Y_XZ + eps

# Train test split
train_inds, test_inds = train_test_split(
    np.arange(n), test_size=test_set_pct, random_state=1234
)
n_test = len(test_inds)
n_train = len(train_inds)
X_test = X[test_inds, :]
X_train = X[train_inds, :]
Z_test = Z[test_inds]
Z_train = Z[train_inds]
y_test = y[test_inds]
y_train = y[train_inds]
propensity_test = pi_X[test_inds]
propensity_train = pi_X[train_inds]

# Attempt to fit a GFR-only BCF model
xbcf_model = BCFModel()
xbcf_model.sample(
    X_train=X_train,
    Z_train=Z_train,
    propensity_train=propensity_train,
    y_train=y_train,
    num_gfr=10,
    num_burnin=0,
    num_mcmc=0,
)

# Inspect the model fit
y_hat_test = xbcf_model.predict(
    X=X_test, Z=Z_test, propensity=propensity_test, type="mean", terms="y_hat"
)
plt.clf()
plt.scatter(y_hat_test, y_test)
plt.axline((0, 0), slope=1, color="red", linestyle="dashed", linewidth=1.5)
plt.xlabel("Predicted Outcome Mean")
plt.ylabel("True Outcome")
plt.xlim(np.min(y_hat_test) * 0.9, np.max(y_hat_test) * 1.1)
plt.ylim(np.min(y), np.max(y))
plt.show()

# Not great! Let's add a few MCMC samples and see how it shakes out
bcf_model = BCFModel()
bcf_model.sample(
    X_train=X_train,
    Z_train=Z_train,
    propensity_train=propensity_train,
    y_train=y_train,
    num_gfr=10,
    num_burnin=0,
    num_mcmc=10,
)

# Inspect the model fit
y_hat_test = bcf_model.predict(
    X=X_test, Z=Z_test, propensity=propensity_test, type="mean", terms="y_hat"
)
plt.clf()
plt.scatter(y_hat_test, y_test)
plt.axline((0, 0), slope=1, color="red", linestyle="dashed", linewidth=1.5)
plt.xlabel("Predicted Outcome Mean")
plt.ylabel("True Outcome")
plt.xlim(np.min(y_hat_test) * 0.9, np.max(y_hat_test) * 1.1)
plt.ylim(np.min(y), np.max(y))
plt.show()

# Generate covariates that are zero-inflated
X = np.empty((n, p))
for i in range(p):
    p_zero = rng.uniform(0.1, 0.9)
    zero_vector = rng.binomial(1, p_zero, n)
    runif_vector = rng.uniform(0, 1, n)
    X[:, i] = zero_vector * 0 + (1 - zero_vector) * runif_vector
mu_X = 5 * X[:, 0] - 2 * X[:, 1] + X[:, 2]
pi_X = norm.cdf(0.1 * mu_X)
tau_X = X[:, 3]
Z = rng.binomial(1, pi_X, n)
E_Y_XZ = mu_X + tau_X * Z
eps = rng.normal(0, 1, n)
y = E_Y_XZ + eps

# Train test split
train_inds, test_inds = train_test_split(
    np.arange(n), test_size=test_set_pct, random_state=1234
)
n_test = len(test_inds)
n_train = len(train_inds)
X_test = X[test_inds, :]
X_train = X[train_inds, :]
Z_test = Z[test_inds]
Z_train = Z[train_inds]
y_test = y[test_inds]
y_train = y[train_inds]
propensity_test = pi_X[test_inds]
propensity_train = pi_X[train_inds]

# Attempt to fit a GFR-only predictive model
xbcf_model = BCFModel()
xbcf_model.sample(
    X_train=X_train,
    Z_train=Z_train,
    propensity_train=propensity_train,
    y_train=y_train,
    num_gfr=10,
    num_burnin=0,
    num_mcmc=0,
)

# Inspect the model fit
y_hat_test = xbcf_model.predict(
    X=X_test, Z=Z_test, propensity=propensity_test, type="mean", terms="y_hat"
)
plt.clf()
plt.scatter(y_hat_test, y_test)
plt.axline((0, 0), slope=1, color="red", linestyle="dashed", linewidth=1.5)
plt.xlabel("Predicted Outcome Mean")
plt.ylabel("True Outcome")
plt.xlim(np.min(y_hat_test) * 0.9, np.max(y_hat_test) * 1.1)
plt.ylim(np.min(y), np.max(y))
plt.show()

# Not great! Let's add a few MCMC samples and see how it shakes out
bcf_model = BCFModel()
bcf_model.sample(
    X_train=X_train,
    Z_train=Z_train,
    propensity_train=propensity_train,
    y_train=y_train,
    num_gfr=10,
    num_burnin=0,
    num_mcmc=10,
)

# Inspect the model fit
y_hat_test = bcf_model.predict(
    X=X_test, Z=Z_test, propensity=propensity_test, type="mean", terms="y_hat"
)
plt.clf()
plt.scatter(y_hat_test, y_test)
plt.axline((0, 0), slope=1, color="red", linestyle="dashed", linewidth=1.5)
plt.xlabel("Predicted Outcome Mean")
plt.ylabel("True Outcome")
plt.xlim(np.min(y_hat_test) * 0.9, np.max(y_hat_test) * 1.1)
plt.ylim(np.min(y), np.max(y))
plt.show()
