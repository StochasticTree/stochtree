import numpy as np
from stochtree import BARTModel

# RNG
random_seed = 1234
rng = np.random.default_rng(random_seed)

# Generate covariates and basis
n = 1000
p_X = 10
p_W = 1
X = rng.uniform(0, 1, (n, p_X))
W = rng.uniform(0, 1, (n, p_W))

# Generate random effects terms
num_basis = 2
num_groups = 4
group_labels = rng.choice(num_groups, size=n)
basis = np.empty((n, num_basis))
basis[:, 0] = 1.0
if num_basis > 1:
    basis[:, 1:] = rng.uniform(-1, 1, (n, num_basis - 1))

# Define the outcome mean function
def outcome_mean(X, W):
    return np.where(
        (X[:,0] >= 0.0) & (X[:,0] < 0.25), -7.5 * W[:,0], 
        np.where(
            (X[:,0] >= 0.25) & (X[:,0] < 0.5), -2.5 * W[:,0], 
            np.where(
                (X[:,0] >= 0.5) & (X[:,0] < 0.75), 2.5 * W[:,0], 
                7.5 * W[:,0]
            )
        )
    )

# Define the group rfx function
def rfx_mean(group_labels, basis):
    return np.where(
        group_labels == 0,
        0 - 1 * basis[:, 1],
        np.where(
            group_labels == 1,
            4 + 1 * basis[:, 1],
            np.where(
                group_labels == 2, 8 + 3 * basis[:, 1], 12 + 5 * basis[:, 1]
            ),
        ),
    )

# Generate outcome
epsilon = rng.normal(0, 1, n)
forest_term = outcome_mean(X, W)
rfx_term = rfx_mean(group_labels, basis)
y = forest_term + rfx_term + epsilon

# Run BART
bart_orig = BARTModel()
bart_orig.sample(X_train=X, y_train=y, leaf_basis_train=W, rfx_group_ids_train=group_labels, 
                  rfx_basis_train=basis, num_gfr=10, num_mcmc=10)

# Extract predictions from the sampler
y_hat_orig = bart_orig.predict(X, W, group_labels, basis)

# "Round-trip" the model to JSON string and back and check that the predictions agree
bart_json_string = bart_orig.to_json()
bart_reloaded = BARTModel()
bart_reloaded.from_json(bart_json_string)
y_hat_reloaded = bart_reloaded.predict(X, W, group_labels, basis)
np.testing.assert_almost_equal(y_hat_orig, y_hat_reloaded)