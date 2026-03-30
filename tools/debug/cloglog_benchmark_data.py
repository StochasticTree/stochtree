# Generate benchmark data for comparing R and Python cloglog BART
import numpy as np
import os

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

# Generate ordinal outcomes (1-indexed)
y = np.array([
    np.random.choice(np.arange(1, n_categories + 1), p=true_probs[i, :])
    for i in range(n)
]).astype(float)

# Train-test split
train_idx = np.random.choice(n, size=int(0.8 * n), replace=False)
test_idx = np.setdiff1d(np.arange(n), train_idx)

# Save to CSVs
out_dir = "tools/debug"

# Training data: X columns, y column
train_data = np.column_stack([X[train_idx, :], y[train_idx]])
header_train = ",".join([f"x{i+1}" for i in range(p)] + ["y"])
np.savetxt(
    os.path.join(out_dir, "cloglog_benchmark_train.csv"),
    train_data, delimiter=",", header=header_train, comments="",
)

# Test data: X columns, y column
test_data = np.column_stack([X[test_idx, :], y[test_idx]])
header_test = ",".join([f"x{i+1}" for i in range(p)] + ["y"])
np.savetxt(
    os.path.join(out_dir, "cloglog_benchmark_test.csv"),
    test_data, delimiter=",", header=header_test, comments="",
)

# True probabilities for train and test
header_probs = ",".join([f"prob_cat{j+1}" for j in range(n_categories)])
np.savetxt(
    os.path.join(out_dir, "cloglog_benchmark_true_probs_train.csv"),
    true_probs[train_idx, :], delimiter=",", header=header_probs, comments="",
)
np.savetxt(
    os.path.join(out_dir, "cloglog_benchmark_true_probs_test.csv"),
    true_probs[test_idx, :], delimiter=",", header=header_probs, comments="",
)

print(f"Data saved to {out_dir}")
print(f"  Train: {train_data.shape[0]} obs, Test: {test_data.shape[0]} obs")
unique, counts = np.unique(y, return_counts=True)
print(f"  Outcome distribution: {dict(zip(unique.astype(int), counts))}")
