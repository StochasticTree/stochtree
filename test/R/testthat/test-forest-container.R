test_that("Univariate constant forest container", {
  # Create dataset and forest container
  num_trees <- 10
  # fmt: skip
  X = matrix(c(1.5, 8.7, 1.2, 
                 2.7, 3.4, 5.4, 
                 3.6, 1.2, 9.3, 
                 4.4, 5.4, 10.4, 
                 5.3, 9.3, 3.6, 
                 6.1, 10.4, 4.4), 
               byrow = TRUE, nrow = 6)
  n <- nrow(X)
  p <- ncol(X)
  forest_dataset = createForestDataset(X)
  forest_samples <- createForestSamples(num_trees, 1, TRUE)

  # Initialize a forest with constant root predictions
  forest_samples$add_forest_with_constant_leaves(0.)

  # Check that regular and "raw" predictions are the same (since the leaf is constant)
  pred <- forest_samples$predict(forest_dataset)
  pred_raw <- forest_samples$predict_raw(forest_dataset)

  # Assertion
  expect_equal(pred, pred_raw)

  # Split the root of the first tree in the ensemble at X[,1] > 4.0
  forest_samples$add_numeric_split_tree(0, 0, 0, 0, 4.0, -5., 5.)

  # Check that regular and "raw" predictions are the same (since the leaf is constant)
  pred <- forest_samples$predict(forest_dataset)
  pred_raw <- forest_samples$predict_raw(forest_dataset)

  # Assertion
  expect_equal(pred, pred_raw)

  # Split the left leaf of the first tree in the ensemble at X[,2] > 4.0
  forest_samples$add_numeric_split_tree(0, 0, 1, 1, 4.0, -7.5, -2.5)

  # Check that regular and "raw" predictions are the same (since the leaf is constant)
  pred <- forest_samples$predict(forest_dataset)
  pred_raw <- forest_samples$predict_raw(forest_dataset)

  # Assertion
  expect_equal(pred, pred_raw)

  # Multiply first forest by 2.0
  forest_samples$multiply_forest(0, 2.0)

  # Check that predictions are all double
  pred_orig <- pred
  pred_expected <- pred * 2.0
  pred <- forest_samples$predict(forest_dataset)

  # Assertion
  expect_equal(pred, pred_expected)

  # Add 1.0 to every tree in first forest
  forest_samples$add_to_forest(0, 1.0)

  # Check that predictions are += num_trees
  pred_expected <- pred + num_trees
  pred <- forest_samples$predict(forest_dataset)

  # Assertion
  expect_equal(pred, pred_expected)

  # Initialize a new forest with constant root predictions
  forest_samples$add_forest_with_constant_leaves(0.)

  # Split the second forest as the first forest was split
  forest_samples$add_numeric_split_tree(1, 0, 0, 0, 4.0, -5., 5.)
  forest_samples$add_numeric_split_tree(1, 0, 1, 1, 4.0, -7.5, -2.5)

  # Check that predictions are as expected
  pred <- forest_samples$predict(forest_dataset)
  pred_expected_new <- cbind(pred_expected, pred_orig)

  # Assertion
  expect_equal(pred, pred_expected_new)

  # Combine second forest with the first forest
  forest_samples$combine_forests(c(0, 1))

  # Check that predictions are as expected
  pred <- forest_samples$predict(forest_dataset)
  pred_expected_new <- cbind(pred_expected + pred_orig, pred_orig)

  # Assertion
  expect_equal(pred, pred_expected_new)

  # Divide first forest predictions by 2
  forest_samples$multiply_forest(0, 0.5)

  # Check that predictions are as expected
  pred <- forest_samples$predict(forest_dataset)
  pred_expected_new <- cbind((pred_expected + pred_orig) / 2.0, pred_orig)

  # Assertion
  expect_equal(pred, pred_expected_new)

  # Delete second forest
  forest_samples$delete_sample(1)

  # Check that predictions are as expected
  pred <- forest_samples$predict(forest_dataset)
  pred_expected_new <- (pred_expected + pred_orig) / 2.0

  # Assertion
  expect_equal(pred, pred_expected_new)
})

test_that("Collapse forests", {
  skip_on_cran()

  # Generate simulated data
  n <- 100
  p <- 5
  X <- matrix(runif(n * p), ncol = p)
  # fmt: skip
  f_XW <- (
        ((0 <= X[,1]) & (0.25 > X[,1])) * (-7.5) + 
        ((0.25 <= X[,1]) & (0.5 > X[,1])) * (-2.5) + 
        ((0.5 <= X[,1]) & (0.75 > X[,1])) * (2.5) + 
        ((0.75 <= X[,1]) & (1 > X[,1])) * (7.5)
    )
  noise_sd <- 1
  y <- f_XW + rnorm(n, 0, noise_sd)
  test_set_pct <- 0.2
  n_test <- round(test_set_pct * n)
  n_train <- n - n_test
  test_inds <- sort(sample(1:n, n_test, replace = FALSE))
  train_inds <- (1:n)[!((1:n) %in% test_inds)]
  X_test <- X[test_inds, ]
  X_train <- X[train_inds, ]
  y_test <- y[test_inds]
  y_train <- y[train_inds]

  # Create forest dataset
  forest_dataset_test <- createForestDataset(covariates = X_test)

  # Run BART for 50 iterations
  num_mcmc <- 50
  general_param_list <- list(num_chains = 1, keep_every = 1)
  bart_model <- bart(
    X_train = X_train,
    y_train = y_train,
    X_test = X_test,
    num_gfr = 0,
    num_burnin = 0,
    num_mcmc = num_mcmc,
    general_params = general_param_list
  )

  # Extract the mean forest container
  mean_forest_container <- bart_model$mean_forests

  # Predict from the original container
  pred_orig <- mean_forest_container$predict(forest_dataset_test)

  # Collapse the container in batches of 5
  batch_size <- 5
  mean_forest_container$collapse(batch_size)

  # Predict from the modified container
  pred_new <- mean_forest_container$predict(forest_dataset_test)

  # Check that corresponding (sums of) predictions match
  batch_inds <- (seq(1, num_mcmc, 1) -
    (num_mcmc -
      (num_mcmc %/% (num_mcmc %/% batch_size)) * (num_mcmc %/% batch_size)) -
    1) %/%
    batch_size +
    1
  pred_orig_collapsed <- matrix(
    NA,
    nrow = nrow(pred_orig),
    ncol = max(batch_inds)
  )
  for (i in 1:max(batch_inds)) {
    pred_orig_collapsed[, i] <- rowSums(pred_orig[, batch_inds == i]) /
      sum(batch_inds == i)
  }

  # Assertion
  expect_equal(pred_orig_collapsed, pred_new)

  # Now run BART for 52 iterations
  num_mcmc <- 52
  general_param_list <- list(num_chains = 1, keep_every = 1)
  bart_model <- bart(
    X_train = X_train,
    y_train = y_train,
    X_test = X_test,
    num_gfr = 0,
    num_burnin = 0,
    num_mcmc = num_mcmc,
    general_params = general_param_list
  )

  # Extract the mean forest container
  mean_forest_container <- bart_model$mean_forests

  # Predict from the original container
  pred_orig <- mean_forest_container$predict(forest_dataset_test)

  # Collapse the container in batches of 5
  batch_size <- 5
  mean_forest_container$collapse(batch_size)

  # Predict from the modified container
  pred_new <- mean_forest_container$predict(forest_dataset_test)

  # Check that corresponding (sums of) predictions match
  batch_inds <- (seq(1, num_mcmc, 1) -
    (num_mcmc -
      (num_mcmc %/% (num_mcmc %/% batch_size)) * (num_mcmc %/% batch_size)) -
    1) %/%
    batch_size +
    1
  pred_orig_collapsed <- matrix(
    NA,
    nrow = nrow(pred_orig),
    ncol = max(batch_inds)
  )
  for (i in 1:max(batch_inds)) {
    pred_orig_collapsed[, i] <- rowSums(pred_orig[, batch_inds == i]) /
      sum(batch_inds == i)
  }

  # Assertion
  expect_equal(pred_orig_collapsed, pred_new)

  # Now run BART for 5 iterations
  num_mcmc <- 5
  general_param_list <- list(num_chains = 1, keep_every = 1)
  bart_model <- bart(
    X_train = X_train,
    y_train = y_train,
    X_test = X_test,
    num_gfr = 0,
    num_burnin = 0,
    num_mcmc = num_mcmc,
    general_params = general_param_list
  )

  # Extract the mean forest container
  mean_forest_container <- bart_model$mean_forests

  # Predict from the original container
  pred_orig <- mean_forest_container$predict(forest_dataset_test)

  # Collapse the container in batches of 5
  batch_size <- 5
  mean_forest_container$collapse(batch_size)

  # Predict from the modified container
  pred_new <- mean_forest_container$predict(forest_dataset_test)

  # Check that corresponding (sums of) predictions match
  pred_orig_collapsed <- as.matrix(rowSums(pred_orig) / batch_size)

  # Assertion
  expect_equal(pred_orig_collapsed, pred_new)

  # Now run BART for 4 iterations
  num_mcmc <- 4
  general_param_list <- list(num_chains = 1, keep_every = 1)
  bart_model <- bart(
    X_train = X_train,
    y_train = y_train,
    X_test = X_test,
    num_gfr = 0,
    num_burnin = 0,
    num_mcmc = num_mcmc,
    general_params = general_param_list
  )

  # Extract the mean forest container
  mean_forest_container <- bart_model$mean_forests

  # Predict from the original container
  pred_orig <- mean_forest_container$predict(forest_dataset_test)

  # Collapse the container in batches of 5
  batch_size <- 5
  mean_forest_container$collapse(batch_size)

  # Predict from the modified container
  pred_new <- mean_forest_container$predict(forest_dataset_test)

  # Check that corresponding (sums of) predictions match
  pred_orig_collapsed <- pred_orig

  # Assertion
  expect_equal(pred_orig_collapsed, pred_new)
})
