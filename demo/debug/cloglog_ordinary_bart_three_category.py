# Load libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from stochtree import CloglogOrdinalBARTModel

# Set seed
seed = 2025
rng = np.random.default_rng(seed)

# Sample size and number of predictors
n = 2000
p = 5

# Design matrix and true lambda function
X = rng.normal(0, 1, size=(n, p))
beta = np.repeat(1 / np.sqrt(p), p)
true_lambda_function = X @ beta

# Set cutpoints for ordinal categories (3 categories: 1, 2, 3)
n_categories = 3
gamma_true = np.array([-2, 1])
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
print(f"Probability distribution: {np.mean(true_probs, axis=0)}")

# Generate ordinal outcomes
y = np.zeros(n, dtype=int)
for i in range(n):
    y[i] = rng.choice(np.arange(n_categories), p=true_probs[i, :])
print(f"Outcome distribution: {np.bincount(y)}")

# Train-test split
sample_inds = np.arange(n)
train_inds, test_inds = train_test_split(sample_inds, test_size=0.2)
X_train = X[train_inds, :]
X_test = X[test_inds, :]
y_train = y[train_inds]
y_test = y[test_inds]

# Run cloglog ordinal BART model
bart_model = CloglogOrdinalBARTModel()
bart_model.sample(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    n_trees=50,
    num_gfr=0,
    num_burnin=1000,
    num_mcmc=500,
    n_thin=1,
)

# Traceplots of cutoff parameters
plt.subplot(1, 2, 1)
plt.plot(bart_model.gamma_samples[0, :], linestyle="-", label=r"$\gamma_1$")
plt.subplot(1, 2, 2)
plt.plot(bart_model.gamma_samples[1, :], linestyle="-", label=r"$\gamma_2$")
plt.show()

# Histograms of cutoff parameters
plt.clf()
gamma1 = bart_model.gamma_samples[0, :] + np.mean(bart_model.forest_pred_train, axis=0)
plt.subplot(1, 2, 1)
plt.hist(gamma1, bins=30, edgecolor="black")
gamma2 = bart_model.gamma_samples[1, :] + np.mean(bart_model.forest_pred_train, axis=0)
plt.subplot(1, 2, 2)
plt.hist(gamma2, bins=30, edgecolor="black")
plt.show()

# Traceplots of cutoff parameters combined with average forest predictions
plt.clf()
plt.subplot(1, 2, 1)
plt.plot(gamma1, linestyle="-", label=r"$\gamma_1$")
plt.axhline(
    y=gamma_true[0] + np.mean(true_lambda_function[train_inds]),
    color="red",
    linestyle="--",
)
plt.subplot(1, 2, 2)
plt.plot(gamma2, linestyle="-", label=r"$\gamma_2$")
plt.axhline(
    y=gamma_true[1] + np.mean(true_lambda_function[train_inds]),
    color="red",
    linestyle="--",
)
plt.show()

# Compare forest predictions with the truth (for training and test sets)

# Train set
plt.clf()
lambda_pred_train = np.mean(bart_model.forest_pred_train, axis=1) - np.mean(
    bart_model.forest_pred_train
)
plt.subplot(1, 2, 1)
plt.plot(lambda_pred_train, true_lambda_function[train_inds], "o")
plt.xlabel("Predicted lambda (train)")
plt.ylabel("True lambda (train)")
plt.axline((0, 0), slope=1, color="blue", linestyle=(0, (3, 3)))
cor_train = np.corrcoef(true_lambda_function[train_inds], lambda_pred_train)[0, 1]
plt.text(
    min(true_lambda_function[train_inds]),
    max(true_lambda_function[train_inds]),
    f"Correlation: {round(cor_train, 3)}",
)

# Test set
lambda_pred_test = np.mean(bart_model.forest_pred_test, axis=1) - np.mean(
    bart_model.forest_pred_test
)
plt.subplot(1, 2, 2)
plt.plot(lambda_pred_test, true_lambda_function[test_inds], "o")
plt.xlabel("Predicted lambda (test)")
plt.ylabel("True lambda (test)")
plt.axline((0, 0), slope=1, color="blue", linestyle=(0, (3, 3)))
cor_test = np.corrcoef(true_lambda_function[test_inds], lambda_pred_test)[0, 1]
plt.text(
    min(true_lambda_function[test_inds]),
    max(true_lambda_function[test_inds]),
    f"Correlation: {round(cor_test, 3)}",
)
plt.show()

# Estimated ordinal class probabilities for the training set
est_probs_train = np.zeros((len(train_inds), n_categories))
for j in range(n_categories):
    if j == 0:
        est_probs_train[:, j] = np.mean(
            1
            - np.exp(
                -np.exp(bart_model.forest_pred_train + bart_model.gamma_samples[j, :])
            ),
            axis=1,
        )
    elif j == n_categories - 1:
        est_probs_train[:, j] = 1 - np.sum(est_probs_train[:, :j], axis=1)
    else:
        est_probs_train[:, j] = np.mean(
            np.exp(
                -np.exp(
                    bart_model.forest_pred_train + bart_model.gamma_samples[j - 1, :]
                )
            )
            * (
                1
                - np.exp(
                    -np.exp(
                        bart_model.forest_pred_train + bart_model.gamma_samples[j, :]
                    )
                )
            ),
            axis=1,
        )

# Plot estimated vs true class probabilities for training set
plt.clf()
for j in range(n_categories):
    plt.subplot(1, n_categories, j + 1)
    plt.plot(est_probs_train[:, j], true_probs[train_inds, j], "o")
    plt.xlabel("Predicted prob (train)")
    plt.ylabel("True prob (train)")
    plt.axline((0, 0), slope=1, color="blue", linestyle=(0, (3, 3)))
    cor_train = np.corrcoef(est_probs_train[:, j], true_probs[train_inds, j])[0, 1]
    plt.text(
        min(est_probs_train[:, j]),
        max(true_probs[train_inds, j]),
        f"Correlation: {round(cor_train, 3)}",
    )
    plt.show()

# Estimated ordinal class probabilities for the training set
est_probs_test = np.zeros((len(test_inds), n_categories))
for j in range(n_categories):
    if j == 0:
        est_probs_test[:, j] = np.mean(
            1
            - np.exp(
                -np.exp(bart_model.forest_pred_test + bart_model.gamma_samples[j, :])
            ),
            axis=1,
        )
    elif j == n_categories - 1:
        est_probs_test[:, j] = 1 - np.sum(est_probs_test[:, :j], axis=1)
    else:
        est_probs_test[:, j] = np.mean(
            np.exp(
                -np.exp(
                    bart_model.forest_pred_test + bart_model.gamma_samples[j - 1, :]
                )
            )
            * (
                1
                - np.exp(
                    -np.exp(
                        bart_model.forest_pred_test + bart_model.gamma_samples[j, :]
                    )
                )
            ),
            axis=1,
        )

# Plot estimated vs true class probabilities for test set
plt.clf()
for j in range(n_categories):
    plt.subplot(1, n_categories, j + 1)
    plt.plot(est_probs_test[:, j], true_probs[test_inds, j], "o")
    plt.xlabel("Predicted prob (test)")
    plt.ylabel("True prob (test)")
    plt.axline((0, 0), slope=1, color="blue", linestyle=(0, (3, 3)))
    cor_test = np.corrcoef(est_probs_test[:, j], true_probs[test_inds, j])[0, 1]
    plt.text(
        min(est_probs_test[:, j]),
        max(true_probs[test_inds, j]),
        f"Correlation: {round(cor_test, 3)}",
    )
    plt.show()
