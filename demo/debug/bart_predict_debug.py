# Demo of updated predict method for BART

# Load library
from stochtree import BARTModel
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate data
rng = np.random.default_rng()
n = 100
p = 5
X = rng.uniform(low=0.0, high=1.0, size=(n, p))
f_X = np.where(
    ((0 <= X[:, 0]) & (X[:, 0] <= 0.25)),
    -7.5,
    np.where(
        ((0.25 <= X[:, 0]) & (X[:, 0] <= 0.5)),
        -2.5,
        np.where(((0.5 <= X[:, 0]) & (X[:, 0] <= 0.75)), 2.5, 7.5),
    ),
)
noise_sd = 1.0
y = f_X + rng.normal(loc=0.0, scale=1.0, size=(n,))

# Train-test split
sample_inds = np.arange(n)
test_set_pct = 0.2
train_inds, test_inds = train_test_split(sample_inds, test_size=test_set_pct)
X_train = X[train_inds, :]
X_test = X[test_inds, :]
y_train = y[train_inds]
y_test = y[test_inds]
f_X_train = f_X[train_inds]
f_X_test = f_X[test_inds]

# Fit simple BART model
bart_model = BARTModel()
bart_model.sample(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    num_gfr=10,
    num_burnin=0,
    num_mcmc=10,
)

# # Check several predict approaches
bart_preds = bart_model.predict(covariates=X_test)
y_hat_posterior_test = bart_model.predict(covariates=X_test)['y_hat']
y_hat_mean_test = bart_model.predict(
    covariates=X_test, 
    type = "mean",
    terms = ["y_hat"]
)
y_hat_test = bart_model.predict(
    covariates=X_test, 
    type = "mean",
    terms = ["rfx", "variance"]
)

# Plot predicted versus actual
plt.scatter(y_hat_mean_test, y_test, color="black")
plt.axline((0, 0), slope=1, color="red", linestyle=(0, (3,3)))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Y hat")
plt.show()
