# Load libraries
library(stochtree)

# Generate "forest" data
n <- 20
num_covariates <- 10
num_basis <- 5
covariates <- matrix(runif(n * num_covariates), ncol = num_covariates)
basis <- matrix(runif(n * num_basis), ncol = num_basis)
variance_weights <- runif(n)

# Create a ForestDataset object
forest_dataset <- createForestDataset(covariates, basis, variance_weights)

# Update forest_dataset's basis
new_basis <- matrix(runif(n * num_basis), ncol = num_basis)
forest_dataset$update_basis(new_basis)

# Update forest_dataset's variance_weights
new_variance_weights <- runif(n)
forest_dataset$update_variance_weights(new_variance_weights)

# Generate RFX data
group_ids <- sample(as.integer(c(1,2)), size = n, replace = T)
rfx_basis <- cbind(1, runif(n))

# Create a RandomEffectsDataset object
rfx_dataset <- createRandomEffectsDataset(
    group_labels = group_ids, basis = rfx_basis, 
    variance_weights = variance_weights
)

# Update rfx_dataset's basis
new_rfx_basis <- cbind(1, runif(n))
rfx_dataset$update_basis(new_rfx_basis)

# Update rfx_dataset's variance weights
rfx_dataset$update_variance_weights(new_variance_weights)

