test_that("Prediction from trees with constant leaf", {
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

  # Check the split count for the first tree in the ensemble
  split_counts <- forest_samples$get_tree_split_counts(0, 0, p)
  split_counts_expected <- c(1, 1, 0)

  # Assertion
  expect_equal(split_counts, split_counts_expected)
})

test_that("Prediction from trees with univariate leaf basis", {
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
  W = as.matrix(c(-1, -1, -1, 1, 1, 1))
  n <- nrow(X)
  p <- ncol(X)
  forest_dataset = createForestDataset(X, W)
  forest_samples <- createForestSamples(num_trees, 1, FALSE)

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
  pred_manual <- pred_raw * W

  # Assertion
  expect_equal(pred, pred_manual)

  # Split the left leaf of the first tree in the ensemble at X[,2] > 4.0
  forest_samples$add_numeric_split_tree(0, 0, 1, 1, 4.0, -7.5, -2.5)

  # Check that regular and "raw" predictions are the same (since the leaf is constant)
  pred <- forest_samples$predict(forest_dataset)
  pred_raw <- forest_samples$predict_raw(forest_dataset)
  pred_manual <- pred_raw * W

  # Assertion
  expect_equal(pred, pred_manual)

  # Check the split count for the first tree in the ensemble
  split_counts <- forest_samples$get_tree_split_counts(0, 0, p)
  split_counts_expected <- c(1, 1, 0)

  # Assertion
  expect_equal(split_counts, split_counts_expected)
})

test_that("Prediction from trees with multivariate leaf basis", {
  # Create dataset and forest container
  num_trees <- 10
  output_dim <- 2
  num_samples <- 0
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
  W = matrix(c(1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1), byrow = FALSE, nrow = 6)
  forest_dataset = createForestDataset(X, W)
  forest_samples <- createForestSamples(num_trees, output_dim, FALSE)

  # Initialize a forest with constant root predictions
  forest_samples$add_forest_with_constant_leaves(c(1., 1.))
  num_samples <- num_samples + 1

  # Check that regular and "raw" predictions are the same (since the leaf is constant)
  pred <- forest_samples$predict(forest_dataset)
  pred_raw <- forest_samples$predict_raw(forest_dataset)
  pred_intermediate <- as.numeric(pred_raw) * as.numeric(W)
  dim(pred_intermediate) <- c(n, output_dim, num_samples)
  pred_manual <- apply(pred_intermediate, 3, function(x) rowSums(x))

  # Assertion
  expect_equal(pred, pred_manual)

  # Split the root of the first tree in the ensemble at X[,1] > 4.0
  forest_samples$add_numeric_split_tree(0, 0, 0, 0, 4.0, c(-5., -1.), c(5., 1.))

  # Check that regular and "raw" predictions are the same (since the leaf is constant)
  pred <- forest_samples$predict(forest_dataset)
  pred_raw <- forest_samples$predict_raw(forest_dataset)
  pred_intermediate <- as.numeric(pred_raw) * as.numeric(W)
  dim(pred_intermediate) <- c(n, output_dim, num_samples)
  pred_manual <- apply(pred_intermediate, 3, function(x) rowSums(x))

  # Assertion
  expect_equal(pred, pred_manual)

  # Split the left leaf of the first tree in the ensemble at X[,2] > 4.0
  forest_samples$add_numeric_split_tree(
    0,
    0,
    1,
    1,
    4.0,
    c(-7.5, 2.5),
    c(-2.5, 7.5)
  )

  # Check that regular and "raw" predictions are the same (since the leaf is constant)
  pred <- forest_samples$predict(forest_dataset)
  pred_raw <- forest_samples$predict_raw(forest_dataset)
  pred_intermediate <- as.numeric(pred_raw) * as.numeric(W)
  dim(pred_intermediate) <- c(n, output_dim, num_samples)
  pred_manual <- apply(pred_intermediate, 3, function(x) rowSums(x))

  # Assertion
  expect_equal(pred, pred_manual)

  # Check the split count for the first tree in the ensemble
  split_counts <- forest_samples$get_tree_split_counts(0, 0, 3)
  split_counts_expected <- c(1, 1, 0)

  # Assertion
  expect_equal(split_counts, split_counts_expected)
})

test_that("Predictions with pre-summarization", {
  # Generate data and test-train split
  n <- 100
  p <- 5
  X <- matrix(runif(n * p), ncol = p)
  # fmt: skip
  f_XW <- (((0 <= X[, 1]) & (0.25 > X[, 1])) * (-7.5) +
                 ((0.25 <= X[, 1]) & (0.5 > X[, 1])) * (-2.5) +
                 ((0.5 <= X[, 1]) & (0.75 > X[, 1])) * (2.5) +
                 ((0.75 <= X[, 1]) & (1 > X[, 1])) * (7.5))
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

  # Fit a "classic" BART model
  bart_model <- bart(
    X_train = X_train,
    y_train = y_train,
    num_gfr = 10,
    num_burnin = 0,
    num_mcmc = 10
  )

  # Check that the default predict method returns a list
  pred <- predict(bart_model, X_test)
  y_hat_posterior_test <- pred$y_hat
  expect_equal(dim(y_hat_posterior_test), c(20, 10))

  # Check that the pre-aggregated predictions match with those computed by rowMeans
  pred_mean <- predict(bart_model, X_test, type = "mean")
  y_hat_mean_test <- pred_mean$y_hat
  expect_equal(y_hat_mean_test, rowMeans(y_hat_posterior_test))

  # Check that we warn and return a NULL when requesting terms that weren't fit
  expect_warning({
    pred_mean <- predict(
      bart_model,
      X_test,
      type = "mean",
      terms = c("rfx", "variance_forest")
    )
  })
  expect_equal(NULL, pred_mean)

  # Fit a heteroskedastic BART model
  var_params <- list(num_trees = 20)
  het_bart_model <- bart(
    X_train = X_train,
    y_train = y_train,
    num_gfr = 10,
    num_burnin = 0,
    num_mcmc = 10,
    variance_forest_params = var_params
  )

  # Check that the default predict method returns a list
  pred <- predict(het_bart_model, X_test)
  y_hat_posterior_test <- pred$y_hat
  sigma2_hat_posterior_test <- pred$variance_forest

  # Assertion
  expect_equal(dim(y_hat_posterior_test), c(20, 10))
  expect_equal(dim(sigma2_hat_posterior_test), c(20, 10))

  # Check that the pre-aggregated predictions match with those computed by rowMeans
  pred_mean <- predict(het_bart_model, X_test, type = "mean")
  y_hat_mean_test <- pred_mean$y_hat
  sigma2_hat_mean_test <- pred_mean$variance_forest

  # Assertion
  expect_equal(y_hat_mean_test, rowMeans(y_hat_posterior_test))
  expect_equal(sigma2_hat_mean_test, rowMeans(sigma2_hat_posterior_test))

  # Check that the "single-term" pre-aggregated predictions
  # match those computed by pre-aggregated predictions returned in a list
  y_hat_mean_test_single_term <- predict(
    het_bart_model,
    X_test,
    type = "mean",
    terms = "y_hat"
  )
  sigma2_hat_mean_test_single_term <- predict(
    het_bart_model,
    X_test,
    type = "mean",
    terms = "variance_forest"
  )

  # Assertion
  expect_equal(y_hat_mean_test, y_hat_mean_test_single_term)
  expect_equal(sigma2_hat_mean_test, sigma2_hat_mean_test_single_term)
})
