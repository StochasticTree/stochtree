test_that("ForestDataset can be constructed and updated", {
  # Generate data
  n <- 20
  num_covariates <- 10
  num_basis <- 5
  covariates <- matrix(runif(n * num_covariates), ncol = num_covariates)
  basis <- matrix(runif(n * num_basis), ncol = num_basis)
  variance_weights <- runif(n)

  # Copy data to a ForestDataset object
  forest_dataset <- createForestDataset(covariates, basis, variance_weights)

  # Run first round of expectations
  expect_equal(forest_dataset$num_observations(), n)
  expect_equal(forest_dataset$num_covariates(), num_covariates)
  expect_equal(forest_dataset$num_basis(), num_basis)
  expect_equal(forest_dataset$has_variance_weights(), T)

  # Update data
  new_basis <- matrix(runif(n * num_basis), ncol = num_basis)
  new_variance_weights <- runif(n)
  expect_no_error(
    forest_dataset$update_basis(new_basis)
  )
  expect_no_error(
    forest_dataset$update_variance_weights(new_variance_weights)
  )

  # Check that we recover the correct data through get_covariates, get_basis, and get_variance_weights
  expect_equal(covariates, forest_dataset$get_covariates())
  expect_equal(new_basis, forest_dataset$get_basis())
  expect_equal(new_variance_weights, forest_dataset$get_variance_weights())
})

test_that("RandomEffectsDataset can be constructed and updated", {
  # Generate data
  n <- 20
  num_groups <- 4
  num_basis <- 5
  group_ids <- sample(as.integer(1:num_groups), size = n, replace = T)
  rfx_basis <- cbind(
    1,
    matrix(runif(n * (num_basis - 1)), ncol = (num_basis - 1))
  )
  variance_weights <- runif(n)

  # Copy data to a RandomEffectsDataset object
  rfx_dataset <- createRandomEffectsDataset(
    group_ids,
    rfx_basis,
    variance_weights
  )

  # Run first round of expectations
  expect_equal(rfx_dataset$num_observations(), n)
  expect_equal(rfx_dataset$num_basis(), num_basis)
  expect_equal(rfx_dataset$has_variance_weights(), T)

  # Update data
  new_rfx_basis <- matrix(runif(n * num_basis), ncol = num_basis)
  new_variance_weights <- runif(n)
  expect_no_error(
    rfx_dataset$update_basis(new_rfx_basis)
  )
  expect_no_error(
    rfx_dataset$update_variance_weights(new_variance_weights)
  )

  # Check that we recover the correct data through get_group_labels, get_basis, and get_variance_weights
  expect_equal(group_ids, rfx_dataset$get_group_labels())
  expect_equal(new_rfx_basis, rfx_dataset$get_basis())
  expect_equal(new_variance_weights, rfx_dataset$get_variance_weights())
})
