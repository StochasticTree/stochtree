# Supervised Learning Demo Script

# Load necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from stochtree import BARTModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

# Generate sample data
# RNG
rng = np.random.default_rng()

# Generate covariates and basis
n = 1000
p_X = 10
p_basis = 1
X = rng.uniform(0, 1, (n, p_X))
basis = rng.uniform(0, 1, (n, p_basis))

# Define the outcome mean function
def outcome_mean(X, basis = None):
    if basis is not None:
        return np.where(
            (X[:,0] >= 0.0) & (X[:,0] < 0.25), -7.5 * basis[:,0], 
            np.where(
                (X[:,0] >= 0.25) & (X[:,0] < 0.5), -2.5 * basis[:,0], 
                np.where(
                    (X[:,0] >= 0.5) & (X[:,0] < 0.75), 2.5 * basis[:,0], 
                    7.5 * basis[:,0]
                )
            )
        )
    else:
        return np.where(
            (X[:,0] >= 0.0) & (X[:,0] < 0.25), -7.5 * X[:,1], 
            np.where(
                (X[:,0] >= 0.25) & (X[:,0] < 0.5), -2.5 * X[:,1], 
                np.where(
                    (X[:,0] >= 0.5) & (X[:,0] < 0.75), 2.5 * X[:,1], 
                    7.5 * X[:,1]
                )
            )
        )


# Generate outcome
epsilon = rng.normal(0, 1, n)
w = outcome_mean(X, basis) + epsilon
# w = outcome_mean(X) + epsilon
y = np.where(w > 0, 1, 0)

# Test-train split
sample_inds = np.arange(n)
train_inds, test_inds = train_test_split(sample_inds, test_size=0.2)
X_train = X[train_inds,:]
X_test = X[test_inds,:]
basis_train = basis[train_inds,:]
basis_test = basis[test_inds,:]
w_train = w[train_inds]
w_test = w[test_inds]
y_train = y[train_inds]
y_test = y[test_inds]

# Construct parameter lists
general_params = {
    'probit_outcome_model': True, 
    'sample_sigma2_global': False
}

# Run BART
num_gfr = 10
num_mcmc = 100
bart_model = BARTModel()
bart_model.sample(X_train=X_train, y_train=y_train, leaf_basis_train=basis_train, 
                  X_test=X_test, leaf_basis_test=basis_test, num_gfr=num_gfr, 
                  num_burnin=0, num_mcmc=num_mcmc, general_params=general_params)
# bart_model.sample(X_train=X_train, y_train=y_train, X_test=X_test, num_gfr=num_gfr, 
#                   num_burnin=0, num_mcmc=num_mcmc, general_params=general_params)

# Inspect the MCMC (BART) samples
w_hat_test = np.squeeze(bart_model.y_hat_test).mean(axis = 1)
plt.scatter(w_hat_test, w_test, color="black")
plt.axline((0, 0), slope=1, color="red", linestyle=(0, (3,3)))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Probit scale latent outcome")
plt.show()

# Compute prediction accuracy
preds_test = w_hat_test > 0
print(f"Test set accuracy: {np.mean(y_test == preds_test):.3f}")

# Present a ROC curve
fpr_list = list()
tpr_list = list()
threshold_list = list()
for i in range(num_mcmc):
    fpr, tpr, thresholds = roc_curve(y_test, bart_model.y_hat_test[:,i], pos_label=1)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    threshold_list.append(thresholds)
fpr_mean, tpr_mean, thresholds_mean = roc_curve(y_test, w_hat_test, pos_label=1)
for i in range(num_mcmc):
    plt.plot(fpr_list[i], tpr_list[i], color = 'blue', linestyle='solid', linewidth = 1.25)
plt.plot(fpr_mean, tpr_mean, color = 'black', linestyle='dashed', linewidth = 2.0)
plt.axline((0, 0), slope=1, color="red", linestyle='dashed', linewidth=1.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()
