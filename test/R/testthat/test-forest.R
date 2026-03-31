test_that("Univariate forest construction", {
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
  forest <- createForest(num_trees, 1, TRUE)

  # Initialize forest with 0.0 root predictions
  forest$set_root_leaves(0.)

  # Check that regular and "raw" predictions are the same (since the leaf is constant)
  pred <- forest$predict(forest_dataset)
  pred_raw <- forest$predict_raw(forest_dataset)

  # Assertion
  expect_equal(pred, pred_raw)

  # Split the root of the first tree in the ensemble at X[,1] > 4.0
  forest$add_numeric_split_tree(0, 0, 0, 4.0, -5., 5.)

  # Check that predictions are the same (since the leaf is constant)
  pred <- forest$predict(forest_dataset)
  pred_raw <- forest$predict_raw(forest_dataset)

  # Assertion
  expect_equal(pred, pred_raw)

  # Split the left leaf of the first tree in the ensemble at X[,2] > 4.0
  forest$add_numeric_split_tree(0, 1, 1, 4.0, -7.5, -2.5)

  # Check that regular and "raw" predictions are the same (since the leaf is constant)
  pred <- forest$predict(forest_dataset)
  pred_raw <- forest$predict_raw(forest_dataset)

  # Assertion
  expect_equal(pred, pred_raw)

  # Check the split count for the first tree in the ensemble
  split_counts <- forest$get_tree_split_counts(0, p)
  split_counts_expected <- c(1, 1, 0)

  # Assertion
  expect_equal(split_counts, split_counts_expected)
})

test_that("Univariate forest construction and low-level merge / arithmetic ops", {
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
  forest1 <- createForest(num_trees, 1, TRUE)
  forest2 <- createForest(num_trees, 1, TRUE)

  # Initialize forests with 0.0 root predictions
  forest1$set_root_leaves(0.)
  forest2$set_root_leaves(0.)

  # Check that predictions are as expected
  pred1 <- forest1$predict(forest_dataset)
  pred2 <- forest2$predict(forest_dataset)
  pred_exp1 <- c(0, 0, 0, 0, 0, 0)
  pred_exp2 <- c(0, 0, 0, 0, 0, 0)

  # Assertion
  expect_equal(pred1, pred_exp1)
  expect_equal(pred2, pred_exp2)

  # Split the root of the first tree of the first forest in the ensemble at X[,1] > 4.0
  forest1$add_numeric_split_tree(0, 0, 0, 4.0, -5., 5.)

  # Split the root of the first tree of the second forest in the ensemble at X[,1] > 3.0
  forest2$add_numeric_split_tree(0, 0, 0, 3.0, -1., 1.)

  # Check that predictions are as expected
  pred1 <- forest1$predict(forest_dataset)
  pred2 <- forest2$predict(forest_dataset)
  pred_exp1 <- c(-5, -5, -5, 5, 5, 5)
  pred_exp2 <- c(-1, -1, 1, 1, 1, 1)

  # Assertion
  expect_equal(pred1, pred_exp1)
  expect_equal(pred2, pred_exp2)

  # Split the left leaf of the first tree of the first forest in the ensemble at X[,2] > 4.0
  forest1$add_numeric_split_tree(0, 1, 1, 4.0, -7.5, -2.5)

  # Split the left leaf of the first tree of the first forest in the ensemble at X[,2] > 4.0
  forest2$add_numeric_split_tree(0, 1, 1, 4.0, -1.5, -0.5)

  # Check that predictions are as expected
  pred1 <- forest1$predict(forest_dataset)
  pred2 <- forest2$predict(forest_dataset)
  pred_exp1 <- c(-2.5, -7.5, -7.5, 5, 5, 5)
  pred_exp2 <- c(-0.5, -1.5, 1, 1, 1, 1)

  # Assertion
  expect_equal(pred1, pred_exp1)
  expect_equal(pred2, pred_exp2)

  # Merge forests
  forest1$merge_forest(forest2)

  # Check that predictions are as expected
  pred <- forest1$predict(forest_dataset)
  pred_exp <- c(-3.0, -9.0, -6.5, 6.0, 6.0, 6.0)

  # Assertion
  expect_equal(pred, pred_exp)

  # Add constant to every value of the combined forest
  forest1$add_constant(0.5)

  # Check that predictions are as expected
  pred <- forest1$predict(forest_dataset)
  pred_exp <- c(7.0, 1.0, 3.5, 16.0, 16.0, 16.0)

  # Assertion
  expect_equal(pred, pred_exp)

  # Check that "old" forest is still intact
  pred <- forest2$predict(forest_dataset)
  pred_exp <- c(-0.5, -1.5, 1, 1, 1, 1)

  # Assertion
  expect_equal(pred, pred_exp)

  # Subtract constant back off of every value of the combined forest
  forest1$add_constant(-0.5)

  # Check that predictions are as expected
  pred <- forest1$predict(forest_dataset)
  pred_exp <- c(-3.0, -9.0, -6.5, 6.0, 6.0, 6.0)

  # Assertion
  expect_equal(pred, pred_exp)

  # Multiply every value of the combined forest by a constant
  forest1$multiply_constant(2.0)

  # Check that predictions are as expected
  pred <- forest1$predict(forest_dataset)
  pred_exp <- c(-6.0, -18.0, -13.0, 12.0, 12.0, 12.0)

  # Assertion
  expect_equal(pred, pred_exp)
})
