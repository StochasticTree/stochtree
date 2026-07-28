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
X = np.random.randn(n, p)
beta = np.full(p, 1 / np.sqrt(p))
true_lambda_function = X @ beta

# Set cutpoints for ordinal categories (3 categories: 1, 2, 3)
n_categories = 3
gamma_true = np.array([-2.0, 1.0])
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
        true_probs[:, j] = np.exp(-np.exp(gamma_true[j - 1] + true_lambda_function)) * (
            1 - np.exp(-np.exp(gamma_true[j] + true_lambda_function))
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
        random_seed=1,
        outcome_model=OutcomeModel(outcome="ordinal", link="cloglog"),
    ),
    mean_forest_params=dict(num_trees=50),
)
runtime = time.time() - start_time
print(f"Runtime: {runtime:.1f}s")

# Compute category probabilities for train and test set using predict
est_probs_train = bart_model.predict(X=X_train, scale="probability", terms="y_hat")
est_probs_test = bart_model.predict(X=X_test, scale="probability", terms="y_hat")

# Traceplots of cutoff parameters
fig, axes = plt.subplots(2, 1, figsize=(10, 6))
for k in range(2):
    axes[k].plot(bart_model.cloglog_cutpoint_samples[k, :])
    axes[k].axhline(y=gamma_true[k], color="red", linestyle="--")
    axes[k].set_title(f"gamma_{k + 1}")
    axes[k].set_ylabel("Value")
    axes[k].set_xlabel("MCMC Sample")
plt.tight_layout()
plt.show()

# Histograms of cutoff parameters combined with average forest predictions
gamma1 = bart_model.cloglog_cutpoint_samples[0, :] + np.mean(
    bart_model.y_hat_train, axis=0
)
gamma2 = bart_model.cloglog_cutpoint_samples[1, :] + np.mean(
    bart_model.y_hat_train, axis=0
)
print(
    f"gamma1 + mean(yhat): min={gamma1.min():.3f}, mean={gamma1.mean():.3f}, max={gamma1.max():.3f}"
)
print(
    f"gamma2 + mean(yhat): min={gamma2.min():.3f}, mean={gamma2.mean():.3f}, max={gamma2.max():.3f}"
)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(gamma1, bins=30)
axes[0].set_title("gamma_1 + mean(y_hat_train)")
axes[1].hist(gamma2, bins=30)
axes[1].set_title("gamma_2 + mean(y_hat_train)")
plt.tight_layout()
plt.show()

# Traceplots of cutoff parameters combined with average forest predictions
moo = bart_model.cloglog_cutpoint_samples.T + np.mean(
    bart_model.y_hat_train, axis=0
).reshape(-1, 1)
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
axes[0, 0].plot(moo[:, 0])
axes[0, 0].axhline(
    y=gamma_true[0] + np.mean(true_lambda_function[train_idx]), color="red"
)
axes[0, 0].set_title("gamma_1 + mean(y_hat)")
axes[0, 1].plot(moo[:, 1])
axes[0, 1].axhline(
    y=gamma_true[1] + np.mean(true_lambda_function[train_idx]), color="red"
)
axes[0, 1].set_title("gamma_2 + mean(y_hat)")
axes[1, 0].plot(bart_model.cloglog_cutpoint_samples[0, :])
axes[1, 0].set_title("gamma_1 (raw)")
axes[1, 1].plot(bart_model.cloglog_cutpoint_samples[1, :])
axes[1, 1].set_title("gamma_2 (raw)")
for ax in axes[2, :]:
    ax.axis("off")
plt.tight_layout()
plt.show()

# Compare forest predictions with the truth (for training and test sets)
fig, axes = plt.subplots(2, 1, figsize=(8, 8))

# Train set
lambda_pred_train = np.mean(bart_model.y_hat_train, axis=1) - np.mean(
    bart_model.y_hat_train
)
axes[0].scatter(lambda_pred_train, true_lambda_function[train_idx], s=5, alpha=0.5)
lims = [
    min(lambda_pred_train.min(), true_lambda_function[train_idx].min()),
    max(lambda_pred_train.max(), true_lambda_function[train_idx].max()),
]
axes[0].plot(lims, lims, "b-", lw=2)
cor_train = np.corrcoef(true_lambda_function[train_idx], lambda_pred_train)[0, 1]
axes[0].set_title(f"Train: Correlation = {cor_train:.3f}")
axes[0].set_xlabel("Predicted lambda")
axes[0].set_ylabel("True lambda")

# Test set
lambda_pred_test = np.mean(bart_model.y_hat_test, axis=1) - np.mean(
    bart_model.y_hat_test
)
axes[1].scatter(lambda_pred_test, true_lambda_function[test_idx], s=5, alpha=0.5)
lims = [
    min(lambda_pred_test.min(), true_lambda_function[test_idx].min()),
    max(lambda_pred_test.max(), true_lambda_function[test_idx].max()),
]
axes[1].plot(lims, lims, "b-", lw=2)
cor_test = np.corrcoef(true_lambda_function[test_idx], lambda_pred_test)[0, 1]
axes[1].set_title(f"Test: Correlation = {cor_test:.3f}")
axes[1].set_xlabel("Predicted lambda")
axes[1].set_ylabel("True lambda")
plt.tight_layout()
plt.show()

# Plot estimated vs true class probabilities for training set
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for j in range(n_categories):
    mean_probs = np.mean(est_probs_train[:, j, :], axis=1)
    ax = axes[j // 2, j % 2]
    ax.scatter(true_probs[train_idx, j], mean_probs, s=5, alpha=0.5)
    lims = [0, max(true_probs[train_idx, j].max(), mean_probs.max())]
    ax.plot(lims, lims, "b-", lw=2)
    cor_val = np.corrcoef(true_probs[train_idx, j], mean_probs)[0, 1]
    ax.set_xlabel(f"True Prob Category {j + 1}")
    ax.set_ylabel(f"Estimated Prob Category {j + 1}")
    ax.set_title(f"Correlation: {cor_val:.3f}")
if n_categories < 4:
    axes[1, 1].axis("off")
plt.tight_layout()
plt.show()

# Compare estimated vs true class probabilities for test set
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for j in range(n_categories):
    mean_probs = np.mean(est_probs_test[:, j, :], axis=1)
    ax = axes[j // 2, j % 2]
    ax.scatter(true_probs[test_idx, j], mean_probs, s=5, alpha=0.5)
    lims = [0, max(true_probs[test_idx, j].max(), mean_probs.max())]
    ax.plot(lims, lims, "b-", lw=2)
    cor_val = np.corrcoef(true_probs[test_idx, j], mean_probs)[0, 1]
    ax.set_xlabel(f"True Prob Category {j + 1}")
    ax.set_ylabel(f"Estimated Prob Category {j + 1}")
    ax.set_title(f"Correlation: {cor_val:.3f}")
if n_categories < 4:
    axes[1, 1].axis("off")
plt.tight_layout()
plt.show()

# Evaluate test set posterior interval coverage of lambda(x)
preds = bart_model.predict(X=X_test, terms="y_hat", scale="linear")
adj = -1 * np.mean(np.mean(preds, axis=1))
linear_interval = bart_model.compute_posterior_interval(
    X=X_test,
    terms="y_hat",
    scale="linear",
)
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(np.mean(preds, axis=1) + adj, true_lambda_function[test_idx], s=5, alpha=0.5)
ax.scatter(linear_interval["lower"] + adj, true_lambda_function[test_idx], s=5, alpha=0.3, color="blue")
ax.scatter(linear_interval["upper"] + adj, true_lambda_function[test_idx], s=5, alpha=0.3, color="blue")
lims = [
    min(true_lambda_function[test_idx].min(), (linear_interval["lower"] + adj).min()),
    max(true_lambda_function[test_idx].max(), (linear_interval["upper"] + adj).max()),
]
ax.plot(lims, lims, "k-")
ax.set_xlabel("Predicted lambda")
ax.set_ylabel("True lambda")
ax.set_title("Linear scale posterior interval")
plt.tight_layout()
plt.show()
linear_coverage = np.mean(
    ((linear_interval["lower"] + adj) <= true_lambda_function[test_idx])
    & ((linear_interval["upper"] + adj) >= true_lambda_function[test_idx])
)
print(f"Linear scale posterior interval coverage: {linear_coverage:.3f}")

# Evaluate test set posterior interval coverage of survival function
probability_interval = bart_model.compute_posterior_interval(
    X=X_test,
    terms="y_hat",
    scale="probability",
)
# P(Y > 1) coverage
true_survival_gt1 = np.sum(true_probs[test_idx, 1:n_categories], axis=1)
probability_coverage_gt1 = np.mean(
    (probability_interval["lower"][:, 0] <= true_survival_gt1)
    & (probability_interval["upper"][:, 0] >= true_survival_gt1)
)
print(f"Probability interval coverage P(Y > 1): {probability_coverage_gt1:.3f}")

# P(Y > 2) coverage
true_survival_gt2 = np.sum(true_probs[test_idx, 2:n_categories], axis=1)
probability_coverage_gt2 = np.mean(
    (probability_interval["lower"][:, 1] <= true_survival_gt2)
    & (probability_interval["upper"][:, 1] >= true_survival_gt2)
)
print(f"Probability interval coverage P(Y > 2): {probability_coverage_gt2:.3f}")

# P(Y <= 1) = 1 - P(Y > 1) coverage
probability_coverage_le1 = np.mean(
    ((1 - probability_interval["upper"][:, 0]) <= true_probs[test_idx, 0])
    & ((1 - probability_interval["lower"][:, 0]) >= true_probs[test_idx, 0])
)
print(f"Probability interval coverage P(Y <= 1): {probability_coverage_le1:.3f}")

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
np.histogram(est_probs_test[max_ind, 0, :])
np.histogram(est_probs_test[max_ind, 1, :])
np.histogram(est_probs_test[max_ind, 2, :])

print(f"\nRuntime: {runtime:.1f}s")
