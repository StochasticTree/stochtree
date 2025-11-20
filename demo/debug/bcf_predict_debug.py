# Demo of updated predict method for BART

# Load library
from stochtree import BCFModel
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import norm
import matplotlib.pyplot as plt

# Generate data
rng = np.random.default_rng()
n = 1000
p = 5
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
test_set_pct = 0.2
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
    num_gfr=10,
    num_burnin=0,
    num_mcmc=1000,
)

# Check several predict approaches
bcf_preds = bcf_model.predict(X=X_test, Z=Z_test, propensity=pi_test)
y_hat_posterior_test = bcf_model.predict(X=X_test, Z=Z_test, propensity=pi_test)[
    "y_hat"
]
y_hat_mean_test = bcf_model.predict(
    X=X_test, Z=Z_test, propensity=pi_test, type="mean", terms=["y_hat"]
)
tau_hat_mean_test = bcf_model.predict(
    X=X_test, Z=Z_test, propensity=pi_test, type="mean", terms=["cate"]
)
# Check that this raises a warning
y_hat_test = bcf_model.predict(
    X=X_test, Z=Z_test, propensity=pi_test, type="mean", terms=["rfx", "variance"]
)

# Plot predicted versus actual
plt.scatter(y_hat_mean_test, y_test, color="black")
plt.axline((0, 0), slope=1, color="red", linestyle=(0, (3, 3)))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Y hat")
plt.show()

# Plot predicted versus actual
plt.clf()
plt.scatter(tau_hat_mean_test, tau_test, color="black")
plt.axline((0, 0), slope=1, color="red", linestyle=(0, (3, 3)))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("CATE function")
plt.show()

# Compute posterior interval
intervals = bcf_model.compute_posterior_interval(
    terms="all",
    scale="linear",
    level=0.95,
    covariates=X_test,
    treatment=Z_test,
    propensity=pi_test,
)

# Check coverage of E[Y | X, Z]
mean_coverage = np.mean(
    (intervals["y_hat"]["lower"] <= E_XZ_test)
    & (E_XZ_test <= intervals["y_hat"]["upper"])
)
print(f"Coverage of 95% posterior interval for E[Y|X,Z]: {mean_coverage:.3f}")

# Check coverage of tau(X)
tau_coverage = np.mean(
    (intervals["tau_hat"]["lower"] <= tau_test)
    & (tau_test <= intervals["tau_hat"]["upper"])
)
print(f"Coverage of 95% posterior interval for tau(X): {tau_coverage:.3f}")

# Check coverage of mu(X)
mu_coverage = np.mean(
    (intervals["mu_hat"]["lower"] <= mu_test)
    & (mu_test <= intervals["mu_hat"]["upper"])
)
print(f"Coverage of 95% posterior interval for mu(X): {mu_coverage:.3f}")

# Sample from the posterior predictive distribution
bcf_ppd_samples = bcf_model.sample_posterior_predictive(
    covariates=X_test, treatment=Z_test, propensity=pi_test, num_draws_per_sample=10
)

# Plot PPD mean vs actual
ppd_mean = np.mean(bcf_ppd_samples, axis=(0, 2))
plt.clf()
plt.scatter(ppd_mean, y_test, color="blue")
plt.axline((0, 0), slope=1, color="red", linestyle=(0, (3, 3)))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Posterior Predictive Mean Comparison")
plt.show()

# Check coverage of posterior predictive distribution
ppd_intervals = np.percentile(bcf_ppd_samples, [2.5, 97.5], axis=(0, 2))
ppd_coverage = np.mean(
    (ppd_intervals[0, :] <= y_test) & (y_test <= ppd_intervals[1, :])
)
print(f"Coverage of 95% posterior predictive interval for Y: {ppd_coverage:.3f}")

# Generate data with random effects
X = rng.normal(loc=0.0, scale=1.0, size=(n, p))
mu_X = X[:, 0]
tau_X = 0.25 * X[:, 1]
pi_X = norm.cdf(0.5 * X[:, 1])
Z = rng.binomial(n=1, p=pi_X, size=(n,))
rfx_group_ids = rng.choice(a=3, size=(n,))
rfx_basis = np.concatenate((np.ones((n, 1)), np.expand_dims(Z, 1)), axis=1)
rfx_coefs = np.array([[-2.0, -0.5], [0.0, 0.0], [2.0, 0.5]])
rfx_term = np.sum(rfx_coefs[rfx_group_ids, :] * rfx_basis, axis=1)
E_XZ = mu_X + tau_X * Z + rfx_term
snr = 2.0
noise_sd = np.std(E_XZ) / snr
y = E_XZ + rng.normal(loc=0.0, scale=noise_sd, size=(n,))

# Train-test split
sample_inds = np.arange(n)
test_set_pct = 0.2
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
rfx_group_ids_train = rfx_group_ids[train_inds]
rfx_group_ids_test = rfx_group_ids[test_inds]
rfx_basis_train = rfx_basis[train_inds, :]
rfx_basis_test = rfx_basis[test_inds, :]

# Fit simple BCF model
rfx_params = {"model_spec": "intercept_plus_treatment"}
bcf_model = BCFModel()
bcf_model.sample(
    X_train=X_train,
    Z_train=Z_train,
    propensity_train=pi_train,
    y_train=y_train,
    rfx_group_ids_train=rfx_group_ids_train,
    num_gfr=10,
    num_burnin=0,
    num_mcmc=1000,
    random_effects_params=rfx_params
)

# Check several predict approaches
bcf_preds = bcf_model.predict(X=X_test, Z=Z_test, propensity=pi_test, rfx_group_ids=rfx_group_ids_test)

# Check that mu + tau + rfx = prognostic + cate
np.allclose(
  (bcf_preds["mu_hat"] +
    np.multiply(bcf_preds["tau_hat"], np.expand_dims(Z_test, 1)) +
    bcf_preds["rfx_predictions"]),
  (bcf_preds["prognostic_function"] +
    np.multiply(bcf_preds["cate"], np.expand_dims(Z_test, 1))),
  atol=1e-4
)

# Retrieve just prognostic predictions
prog_fn_test = bcf_model.predict(
  X=X_test, Z=Z_test, propensity=pi_test, 
  rfx_group_ids=rfx_group_ids_test,
  terms = "prognostic_function"
)

# Compare to prognostic function returned from the larger prediction
np.allclose(prog_fn_test, bcf_preds["prognostic_function"], atol=1e-4)

# Retrieve just prognostic predictions
mu_hat_test = bcf_model.predict(
  X=X_test, Z=Z_test, propensity=pi_test, 
  rfx_group_ids=rfx_group_ids_test,
  terms = "mu"
)

# Compare to prognostic function returned from the larger prediction
np.allclose(mu_hat_test, bcf_preds["mu_hat"], atol=1e-4)

# Compute intervals for all of the model terms
posterior_intervals_test = bcf_model.compute_posterior_interval(
  terms="all",
  scale="linear",
  level=0.95,
  covariates=X_test,
  treatment=Z_test,
  propensity=pi_test,
  rfx_group_ids=rfx_group_ids_test,
)

# Compute intervals for just the prognostic term
prog_intervals_test = bcf_model.compute_posterior_interval(
  terms="prognostic_function",
  scale="linear",
  level=0.95,
  covariates=X_test,
  treatment=Z_test,
  propensity=pi_test,
  rfx_group_ids=rfx_group_ids_test
)

# Compute intervals for just the CATE term
cate_intervals_test = bcf_model.compute_posterior_interval(
  terms="cate",
  scale="linear",
  level=0.95,
  covariates=X_test,
  treatment=Z_test,
  propensity=pi_test,
  rfx_group_ids=rfx_group_ids_test
)

# Check that they match the corresponding terms from the full interval list
(np.allclose(
  posterior_intervals_test['prognostic_function']['lower'],
  prog_intervals_test['lower'], 
  atol=1e-4
) and 
np.allclose(
  posterior_intervals_test['prognostic_function']['upper'],
  prog_intervals_test['upper'], 
  atol=1e-4
) and 
np.allclose(
  posterior_intervals_test['cate']['lower'],
  cate_intervals_test['lower'], 
  atol=1e-4
) and 
np.allclose(
  posterior_intervals_test['cate']['upper'],
  cate_intervals_test['upper'], 
  atol=1e-4
))

# Check that the prog and CATE intervals are different from the mu and tau intervals
mu_intervals_test = bcf_model.compute_posterior_interval(
  terms="mu",
  scale="linear",
  level=0.95,
  covariates=X_test,
  treatment=Z_test,
  propensity=pi_test,
  rfx_group_ids=rfx_group_ids_test
)
tau_intervals_test = bcf_model.compute_posterior_interval(
  terms="tau",
  scale="linear",
  level=0.95,
  covariates=X_test,
  treatment=Z_test,
  propensity=pi_test,
  rfx_group_ids=rfx_group_ids_test
)

(not (np.allclose(
  mu_intervals_test['lower'],
  prog_intervals_test['lower'], 
  atol=1e-4
)) and 
not (np.allclose(
  mu_intervals_test['upper'],
  prog_intervals_test['upper'], 
  atol=1e-4
)) and 
not (np.allclose(
  tau_intervals_test['lower'],
  cate_intervals_test['lower'], 
  atol=1e-4
)) and 
not (np.allclose(
  tau_intervals_test['upper'],
  cate_intervals_test['upper'], 
  atol=1e-4
))
)