test_that("Residual updates correctly propagated after forest sampling step", {
  # Setup
  # Create dataset
  # fmt: skip
  X = matrix(c(1.5, 8.7, 1.2, 
                 2.7, 3.4, 5.4, 
                 3.6, 1.2, 9.3, 
                 4.4, 5.4, 10.4, 
                 5.3, 9.3, 3.6, 
                 6.1, 10.4, 4.4), 
               byrow = TRUE, nrow = 6)
  W = matrix(c(1, 1, 1, 1, 1, 1), nrow = 6)
  n = nrow(X)
  p = ncol(X)
  y = as.matrix(ifelse(X[, 1] > 4, -5, 5) + rnorm(n, 0, 1))
  y_bar = mean(y)
  y_std = sd(y)
  resid = (y - y_bar) / y_std
  forest_dataset = createForestDataset(X, W)
  residual = createOutcome(resid)
  variable_weights = rep(1.0 / p, p)
  feature_types = as.integer(rep(0, p))

  # Forest parameters
  num_trees = 50
  alpha = 0.95
  beta = 2.0
  min_samples_leaf = 1
  current_sigma2 = 1.
  current_leaf_scale = as.matrix(1. / num_trees, nrow = 1, ncol = 1)
  cutpoint_grid_size = 100
  max_depth = 10
  a_forest = 0
  b_forest = 0

  # RNG
  cpp_rng = createCppRNG(-1)

  # Create forest sampler and forest container
  global_model_config = createGlobalModelConfig(
    global_error_variance = current_sigma2
  )
  forest_model_config = createForestModelConfig(
    feature_types = feature_types,
    num_trees = num_trees,
    num_observations = n,
    alpha = alpha,
    beta = beta,
    min_samples_leaf = min_samples_leaf,
    max_depth = max_depth,
    leaf_model_type = 0,
    leaf_model_scale = current_leaf_scale,
    variable_weights = variable_weights,
    variance_forest_shape = a_forest,
    variance_forest_scale = b_forest,
    cutpoint_grid_size = cutpoint_grid_size
  )
  forest_model = createForestModel(
    forest_dataset,
    forest_model_config,
    global_model_config
  )
  forest_samples = createForestSamples(num_trees, 1, FALSE)
  active_forest = createForest(num_trees, 1, FALSE)

  # Initialize the leaves of each tree in the prognostic forest
  active_forest$prepare_for_sampler(
    forest_dataset,
    residual,
    forest_model,
    0,
    mean(resid)
  )
  active_forest$adjust_residual(
    forest_dataset,
    residual,
    forest_model,
    FALSE,
    FALSE
  )

  # Run the forest sampling algorithm for a single iteration
  forest_model$sample_one_iteration(
    forest_dataset,
    residual,
    forest_samples,
    active_forest,
    cpp_rng,
    forest_model_config,
    global_model_config,
    keep_forest = TRUE,
    gfr = TRUE
  )

  # Get the current residual after running the sampler
  initial_resid = residual$get_data()

  # Get initial prediction from the tree ensemble
  initial_yhat = as.numeric(forest_samples$predict(forest_dataset))

  # Update the basis vector
  scalar = 2.0
  W_update = W * scalar
  forest_dataset$update_basis(W_update)

  # Update residual to reflect adjusted basis
  forest_model$propagate_basis_update(forest_dataset, residual, active_forest)

  # Get updated prediction from the tree ensemble
  updated_yhat = as.numeric(forest_samples$predict(forest_dataset))

  # Compute the expected residual
  expected_resid = initial_resid + initial_yhat - updated_yhat

  # Get the current residual after running the sampler
  updated_resid = residual$get_data()

  # Assertion
  expect_equal(expected_resid, updated_resid)
})
