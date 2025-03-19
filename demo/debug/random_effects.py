# Random Effects Demo Script

# Load necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from stochtree import (
    RandomEffectsContainer,
    RandomEffectsDataset,
    RandomEffectsModel,
    RandomEffectsTracker,
    Residual,
    RNG,
)
# from sklearn.model_selection import train_test_split

# Generate sample data
# RNG
random_seed = 1234
rng = np.random.default_rng(random_seed)

# Generate group labels and random effects basis
num_observations = 1000
num_basis = 2
num_groups = 4
group_labels = rng.choice(num_groups, size=num_observations)
basis = np.empty((num_observations, num_basis))
basis[:, 0] = 1.0
if num_basis > 1:
    basis[:, 1:] = rng.uniform(-1, 1, (num_observations, num_basis - 1))


# Define the group rfx function
def outcome_mean(group_labels, basis):
    return np.where(
        group_labels == 0,
        0 - 1 * basis[:, 1],
        np.where(
            group_labels == 1,
            4 + 1 * basis[:, 1],
            np.where(group_labels == 2, 8 + 3 * basis[:, 1], 12 + 5 * basis[:, 1]),
        ),
    )


# Generate outcome
epsilon = rng.normal(0, 1, num_observations)
rfx_term = outcome_mean(group_labels, basis)
y = rfx_term + epsilon

# Standardize outcome
y_bar = np.mean(y)
y_std = np.std(y)
resid = (y - y_bar) / y_std

# Construct python objects used for rfx sampling
outcome = Residual(resid)
rfx_dataset = RandomEffectsDataset()
rfx_dataset.add_group_labels(group_labels)
rfx_dataset.add_basis(basis)
rfx_tracker = RandomEffectsTracker(group_labels)
rfx_model = RandomEffectsModel(num_basis, num_groups)
rfx_model.set_working_parameter(np.ones(num_basis))
rfx_model.set_group_parameters(np.ones((num_basis, num_groups)))
rfx_model.set_working_parameter_covariance(np.identity(num_basis))
rfx_model.set_group_parameter_covariance(np.identity(num_basis))
rfx_model.set_variance_prior_shape(1.0)
rfx_model.set_variance_prior_scale(1.0)
rfx_container = RandomEffectsContainer(num_basis, num_groups, rfx_tracker)
# cpp_rng = RNG(random_seed)
cpp_rng = RNG()

# Sample the model
rfx_model.sample(rfx_dataset, outcome, rfx_tracker, rfx_container, True, 1.0, cpp_rng)

# Inspect the samples
rfx_preds = rfx_container.predict(group_labels, basis) * y_std + y_bar
rfx_comparison_df = pd.DataFrame(
    np.concatenate((rfx_preds, np.expand_dims(rfx_term, axis=1)), axis=1),
    columns=["Predicted", "Actual"],
)
sns.scatterplot(data=rfx_comparison_df, x="Predicted", y="Actual")
plt.axline((0, 0), slope=1, color="black", linestyle=(0, (3, 3)))
plt.show()
