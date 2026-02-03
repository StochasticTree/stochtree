# Demo of contrast computation function for BCF

# Load libraries
from stochtree import BCFModel, OutcomeModel
from sklearn.model_selection import train_test_split
from scipy.stats import norm
import numpy as np

# Generate data
n = 500
p = 5
rng = np.random.default_rng(1234)
X = rng.uniform(low=0.0, high=1.0, size=(n, p))
mu_x = X[:, 0]
tau_x = 0.25 * X[:, 1]
pi_x = norm.cdf(0.5 * X[:, 0])
Z = rng.binomial(n=1, p=pi_x, size=(n,))
E_XZ = mu_x + Z * tau_x
snr = 2
y = E_XZ + rng.normal(loc=0.0, scale=1.0, size=(n,)) * (np.std(E_XZ) / snr)

# Train-test split
test_set_pct = 0.2
train_inds, test_inds = train_test_split(
    np.arange(n), test_size=test_set_pct, random_state=1234
)
X_train = X[train_inds, :]
X_test = X[test_inds, :]
Z_train = Z[train_inds]
Z_test = Z[test_inds]
y_train = y[train_inds]
y_test = y[test_inds]
n_test = len(test_inds)
n_train = len(train_inds)

# Fit BCF model
bcf_model = BCFModel()
bcf_model.sample(
    X_train=X_train,
    Z_train=Z_train,
    y_train=y_train,
    num_gfr=10,
    num_burnin=0,
    num_mcmc=1000,
)

# Compute contrast posterior
contrast_posterior_test = bcf_model.compute_contrast(
    X_0=X_test,
    X_1=X_test,
    Z_0=np.zeros((n_test, 1)),
    Z_1=np.ones((n_test, 1)),
    type="posterior",
    scale="linear",
)

# Compute the same quantity via two predict calls
y_hat_posterior_test_0 = bcf_model.predict(
    X=X_test, Z=np.zeros((n_test, 1)), type="posterior", terms="y_hat", scale="linear"
)
y_hat_posterior_test_1 = bcf_model.predict(
    X=X_test, Z=np.ones((n_test, 1)), type="posterior", terms="y_hat", scale="linear"
)
contrast_posterior_test_comparison = y_hat_posterior_test_1 - y_hat_posterior_test_0

# Compare results
contrast_diff = contrast_posterior_test_comparison - contrast_posterior_test
np.allclose(contrast_diff, 0, atol=0.001)

# Generate data for a BCF model with random effects
X = rng.uniform(low=0.0, high=1.0, size=(n, p))
mu_x = X[:, 0]
tau_x = 0.25 * X[:, 1]
pi_x = norm.cdf(0.5 * X[:, 0])
Z = rng.binomial(n=1, p=pi_x, size=(n,))
num_rfx_groups = 3
group_labels = rng.choice(num_rfx_groups, size=n)
basis = np.empty((n, 2))
basis[:, 0] = 1.0
basis[:, 1] = rng.uniform(0, 1, (n,))
rfx_coefs = np.array([[-2, -2], [0, 0], [2, 2]])
rfx_term = np.sum(rfx_coefs[group_labels, :] * basis, axis=1)
E_XZ = mu_x + Z * tau_x + rfx_term
snr = 2
y = E_XZ + rng.normal(loc=0.0, scale=1.0, size=(n,)) * (np.std(E_XZ) / snr)

# Train-test split
train_inds, test_inds = train_test_split(
    np.arange(n), test_size=test_set_pct, random_state=1234
)
X_train = X[train_inds, :]
X_test = X[test_inds, :]
Z_train = Z[train_inds]
Z_test = Z[test_inds]
y_train = y[train_inds]
y_test = y[test_inds]
group_ids_train = group_labels[train_inds]
group_ids_test = group_labels[test_inds]
rfx_basis_train = basis[train_inds, :]
rfx_basis_test = basis[test_inds, :]
n_test = len(test_inds)
n_train = len(train_inds)

# Fit BCF model
bcf_model = BCFModel()
bcf_model.sample(
    X_train=X_train,
    Z_train=Z_train,
    y_train=y_train,
    rfx_group_ids_train=group_ids_train,
    rfx_basis_train=rfx_basis_train,
    num_gfr=10,
    num_burnin=0,
    num_mcmc=1000,
)

# Compute contrast posterior
contrast_posterior_test = bcf_model.compute_contrast(
    X_0=X_test,
    X_1=X_test,
    Z_0=np.zeros((n_test, 1)),
    Z_1=np.ones((n_test, 1)),
    rfx_group_ids_0=group_ids_test,
    rfx_group_ids_1=group_ids_test,
    rfx_basis_0=rfx_basis_test,
    rfx_basis_1=rfx_basis_test,
    type="posterior",
    scale="linear",
)

# Compute the same quantity via two predict calls
y_hat_posterior_test_0 = bcf_model.predict(
    X=X_test,
    Z=np.zeros((n_test, 1)),
    rfx_group_ids=group_ids_test,
    rfx_basis=rfx_basis_test,
    type="posterior",
    terms="y_hat",
    scale="linear",
)
y_hat_posterior_test_1 = bcf_model.predict(
    X=X_test,
    Z=np.ones((n_test, 1)),
    rfx_group_ids=group_ids_test,
    rfx_basis=rfx_basis_test,
    type="posterior",
    terms="y_hat",
    scale="linear",
)
contrast_posterior_test_comparison = y_hat_posterior_test_1 - y_hat_posterior_test_0

# Compare results
contrast_diff = contrast_posterior_test_comparison - contrast_posterior_test
np.allclose(contrast_diff, 0, atol=0.001)

# Now repeat the same process but via random effects model spec
bcf_model = BCFModel()
bcf_model.sample(
    X_train=X_train,
    Z_train=Z_train,
    y_train=y_train,
    rfx_group_ids_train=group_ids_train,
    num_gfr=10,
    num_burnin=0,
    num_mcmc=1000,
    random_effects_params={"model_spec": "intercept_plus_treatment"},
)

# Compute CATE posterior
tau_hat_posterior_test = bcf_model.compute_contrast(
    X_0=X_test,
    X_1=X_test,
    Z_0=np.zeros((n_test, 1)),
    Z_1=np.ones((n_test, 1)),
    rfx_group_ids_0=group_ids_test,
    rfx_group_ids_1=group_ids_test,
    rfx_basis_0=np.concatenate((np.ones((n_test, 1)), np.zeros((n_test, 1))), axis=1),
    rfx_basis_1=np.ones((n_test, 2)),
    type="posterior",
    scale="linear",
)

# Compute the same quantity via predict
tau_hat_posterior_test_comparison = bcf_model.predict(
    X=X_test,
    Z=Z_test,
    rfx_group_ids=group_ids_test,
    type="posterior",
    terms="cate",
    scale="linear",
)

# Compare results
contrast_diff = tau_hat_posterior_test_comparison - tau_hat_posterior_test
np.allclose(contrast_diff, 0, atol=0.001)

# Generate data for a probit BCF model with random effects
X = rng.uniform(low=0.0, high=1.0, size=(n, p))
mu_x = X[:, 0]
tau_x = 0.25 * X[:, 1]
pi_x = norm.cdf(0.5 * X[:, 0])
Z = rng.binomial(n=1, p=pi_x, size=(n,))
num_rfx_groups = 3
group_labels = rng.choice(num_rfx_groups, size=n)
basis = np.empty((n, 2))
basis[:, 0] = 1.0
basis[:, 1] = rng.uniform(0, 1, (n,))
rfx_coefs = np.array([[-2, -2], [0, 0], [2, 2]])
rfx_term = np.sum(rfx_coefs[group_labels, :] * basis, axis=1)
E_XZ = mu_x + Z * tau_x + rfx_term
W = E_XZ + rng.normal(loc=0.0, scale=1.0, size=(n,))
y = (W > 0) * 1.0

# Train-test split
train_inds, test_inds = train_test_split(
    np.arange(n), test_size=test_set_pct, random_state=1234
)
X_train = X[train_inds, :]
X_test = X[test_inds, :]
Z_train = Z[train_inds]
Z_test = Z[test_inds]
W_train = W[train_inds]
W_test = W[test_inds]
y_train = y[train_inds]
y_test = y[test_inds]
group_ids_train = group_labels[train_inds]
group_ids_test = group_labels[test_inds]
rfx_basis_train = basis[train_inds, :]
rfx_basis_test = basis[test_inds, :]
n_test = len(test_inds)
n_train = len(train_inds)

# Fit BCF model
bcf_model = BCFModel()
bcf_model.sample(
    X_train=X_train,
    Z_train=Z_train,
    y_train=y_train,
    rfx_group_ids_train=group_ids_train,
    rfx_basis_train=rfx_basis_train,
    num_gfr=10,
    num_burnin=0,
    num_mcmc=1000,
    general_params={"outcome_model": OutcomeModel(outcome="binary", link="probit")},
)

# Compute contrast posterior
contrast_posterior_test = bcf_model.compute_contrast(
    X_0=X_test,
    X_1=X_test,
    Z_0=np.zeros((n_test, 1)),
    Z_1=np.ones((n_test, 1)),
    rfx_group_ids_0=group_ids_test,
    rfx_group_ids_1=group_ids_test,
    rfx_basis_0=rfx_basis_test,
    rfx_basis_1=rfx_basis_test,
    type="posterior",
    scale="probability",
)

# Compute the same quantity via two predict calls
y_hat_posterior_test_0 = bcf_model.predict(
    X=X_test,
    Z=np.zeros((n_test, 1)),
    rfx_group_ids=group_ids_test,
    rfx_basis=rfx_basis_test,
    type="posterior",
    terms="y_hat",
    scale="probability",
)
y_hat_posterior_test_1 = bcf_model.predict(
    X=X_test,
    Z=np.ones((n_test, 1)),
    rfx_group_ids=group_ids_test,
    rfx_basis=rfx_basis_test,
    type="posterior",
    terms="y_hat",
    scale="probability",
)
contrast_posterior_test_comparison = y_hat_posterior_test_1 - y_hat_posterior_test_0

# Compare results
contrast_diff = contrast_posterior_test_comparison - contrast_posterior_test
np.allclose(contrast_diff, 0, atol=0.001)
