# Load library
import time

import matplotlib.pyplot as plt
import numpy as np

from stochtree import BARTModel, OutcomeModel

# Set seed
np.random.seed(2026)

# Sample size and number of predictors
n = 2000
p = 5

# Design matrix and true lambda function
X = np.random.uniform(size=(n, p))
beta = np.full(p, 1 / np.sqrt(p))
true_lambda_function = X @ beta

# Set cutpoints for ordinal categories (2 categories: 1, 2)
n_categories = 2
gamma_true = np.array([-2.0])
ordinal_cutpoints = np.log(np.cumsum(np.exp(gamma_true)))
print("Ordinal cutpoints:", ordinal_cutpoints)

# True ordinal class probabilities
true_probs = np.zeros((n, n_categories))
for j in range(n_categories):
    if j == 0:
        true_probs[:, j] = 1 - np.exp(-np.exp(gamma_true[j] + true_lambda_function))
    elif j == n_categories - 1:
        true_probs[:, j] = 1 - np.sum(true_probs[:, :j], axis=1)
    else:
        true_probs[:, j] = (
            np.exp(-np.exp(gamma_true[j - 1] + true_lambda_function))
            * (1 - np.exp(-np.exp(gamma_true[j] + true_lambda_function)))
        )

# Generate ordinal outcomes (1-indexed like the R version)
y = np.array([
    np.random.choice(np.arange(1, n_categories + 1), p=true_probs[i, :])
    for i in range(n)
]).astype(float)
unique, counts = np.unique(y, return_counts=True)
print("Outcome distribution:", dict(zip(unique.astype(int), counts)))

# Train-test split
train_idx = np.random.choice(n, size=int(0.8 * n), replace=False)
test_idx = np.setdiff1d(np.arange(n), train_idx)
X_train = X[train_idx, :]
y_train = y[train_idx]
X_test = X[test_idx, :]
y_test = y[test_idx]

# Sample the cloglog ordinal BART model
start_time = time.time()
bart_model = BARTModel()
bart_model.sample(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    num_gfr=0,
    num_burnin=1000,
    num_mcmc=1000,
    general_params=dict(
        cutpoint_grid_size=100,
        sample_sigma2_global=False,
        keep_every=1,
        num_chains=1,
        verbose=False,
        outcome_model=OutcomeModel(outcome="binary", link="cloglog"),
    ),
    mean_forest_params=dict(num_trees=50),
)
runtime = time.time() - start_time
print(f"Runtime: {runtime:.1f}s")

# Compute category probabilities for train and test set using predict
est_probs_train = bart_model.predict(
    X=X_train, scale="probability", terms="y_hat"
)
est_probs_test = bart_model.predict(
    X=X_test, scale="probability", terms="y_hat"
)

# Compare forest predictions with the truth (for training and test sets)
fig, axes = plt.subplots(2, 1, figsize=(8, 8))

# Train set
lambda_pred_train = np.mean(bart_model.y_hat_train, axis=1)
axes[0].scatter(
    lambda_pred_train,
    gamma_true[0] + true_lambda_function[train_idx],
    s=5, alpha=0.5,
)
lims = [
    min(lambda_pred_train.min(), (gamma_true[0] + true_lambda_function[train_idx]).min()),
    max(lambda_pred_train.max(), (gamma_true[0] + true_lambda_function[train_idx]).max()),
]
axes[0].plot(lims, lims, "b-", lw=2)
cor_train = np.corrcoef(
    gamma_true[0] + true_lambda_function[train_idx],
    lambda_pred_train,
)[0, 1]
axes[0].set_title(f"Train: Correlation = {cor_train:.3f}")
axes[0].set_xlabel("Predicted lambda")
axes[0].set_ylabel("True gamma + lambda")

# Test set
lambda_pred_test = np.mean(bart_model.y_hat_test, axis=1)
axes[1].scatter(
    lambda_pred_test,
    gamma_true[0] + true_lambda_function[test_idx],
    s=5, alpha=0.5,
)
lims = [
    min(lambda_pred_test.min(), (gamma_true[0] + true_lambda_function[test_idx]).min()),
    max(lambda_pred_test.max(), (gamma_true[0] + true_lambda_function[test_idx]).max()),
]
axes[1].plot(lims, lims, "b-", lw=2)
cor_test = np.corrcoef(gamma_true[0] + true_lambda_function[test_idx], lambda_pred_test)[0, 1]
axes[1].set_title(f"Test: Correlation = {cor_test:.3f}")
axes[1].set_xlabel("Predicted lambda")
axes[1].set_ylabel("True gamma + lambda")
plt.tight_layout()
plt.show()

# Compare estimated vs true class probabilities for train and test sets
fig, axes = plt.subplots(2, 1, figsize=(10, 5))
mean_probs_train = np.mean(est_probs_train, axis=1)
axes[0].scatter(true_probs[train_idx, 1], mean_probs_train, s=5, alpha=0.5)
lims = [0, max(true_probs[train_idx, 1].max(), mean_probs_train.max())]
axes[0].plot(lims, lims, "b-", lw=2)
cor_val = np.corrcoef(true_probs[train_idx, 1], mean_probs_train)[0, 1]
axes[0].set_xlabel(f"True Prob (Train Set)")
axes[0].set_ylabel(f"Estimated Prob (Train Set)")
axes[0].set_title(f"Correlation: {cor_val:.3f}")
mean_probs_test = np.mean(est_probs_test, axis=1)
axes[1].scatter(true_probs[test_idx, 1], mean_probs_test, s=5, alpha=0.5)
lims = [0, max(true_probs[test_idx, 1].max(), mean_probs_test.max())]
axes[1].plot(lims, lims, "b-", lw=2)
cor_val = np.corrcoef(true_probs[test_idx, 1], mean_probs_test)[0, 1]
axes[1].set_xlabel(f"True Prob (Test Set)")
axes[1].set_ylabel(f"Estimated Prob (Test Set)")
axes[1].set_title(f"Correlation: {cor_val:.3f}")
plt.tight_layout()
plt.show()

# Evaluate test set posterior interval coverage of lambda(x)
preds = bart_model.predict(X=X_test, terms="y_hat", scale="linear")
linear_interval = bart_model.compute_posterior_interval(
    X=X_test,
    terms="y_hat",
    scale="linear",
)
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(np.mean(preds, axis=1), gamma_true[0] + true_lambda_function[test_idx], s=5, alpha=0.5)
ax.scatter(linear_interval["lower"], gamma_true[0] + true_lambda_function[test_idx], s=5, alpha=0.3, color="blue")
ax.scatter(linear_interval["upper"], gamma_true[0] + true_lambda_function[test_idx], s=5, alpha=0.3, color="blue")
lims = [
    min((gamma_true[0] + true_lambda_function[test_idx]).min(), linear_interval["lower"].min()),
    max((gamma_true[0] + true_lambda_function[test_idx]).max(), linear_interval["upper"].max()),
]
ax.plot(lims, lims, "k-")
ax.set_xlabel("Predicted lambda")
ax.set_ylabel("True lambda")
ax.set_title("Linear scale posterior interval")
plt.tight_layout()
plt.show()
linear_coverage = np.mean(
    ((linear_interval["lower"]) <= gamma_true[0] + true_lambda_function[test_idx])
    & ((linear_interval["upper"]) >= gamma_true[0] + true_lambda_function[test_idx])
)
print(f"Linear scale posterior interval coverage: {linear_coverage:.3f}")

# Evaluate test set posterior interval coverage of survival function
probability_interval = bart_model.compute_posterior_interval(
    X=X_test,
    terms="y_hat",
    scale="probability",
)
probability_coverage = np.mean(
    (probability_interval["lower"] <= true_probs[test_idx, 1])
    & (probability_interval["upper"] >= true_probs[test_idx, 1])
)
print(f"Probability interval coverage P(Y > 1): {probability_coverage:.3f}")

# Compute test set prediction contrast on linear and probability scale
X0 = X_test
X1 = X_test + 1
linear_contrast = bart_model.compute_contrast(
    X_0=X0,
    X_1=X1,
    type="posterior",
    scale="linear",
)
print(f"Linear contrast shape: {linear_contrast.shape}")

probability_contrast = bart_model.compute_contrast(
    X_0=X0,
    X_1=X1,
    type="posterior",
    scale="probability",
)
print(f"Probability contrast shape: {probability_contrast.shape}")

# Sample from posterior predictive distribution
y_ppd = bart_model.sample_posterior_predictive(
    X=X_test,
    num_draws_per_sample=100
)

# Inspect results
true_probs_test = true_probs[test_idx, :]
max_ind = np.argmax(true_probs_test[:, 0])
true_probs_test[max_ind, :]
np.histogram(y_ppd[max_ind, :, :])
np.histogram(est_probs_test[max_ind, :])

print(f"\nRuntime: {runtime:.1f}s")
