test_that("Univariate constant forest container", {
    # Create dataset and forest container
    num_trees <- 10
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
    forest_samples$combine_forests(c(0,1))
    
    # Check that predictions are as expected
    pred <- forest_samples$predict(forest_dataset)
    pred_expected_new <- cbind(pred_expected + pred_orig, pred_orig)

    # Assertion
    expect_equal(pred, pred_expected_new)
    
    # Divide first forest predictions by 2
    forest_samples$multiply_forest(0, 0.5)
    
    # Check that predictions are as expected
    pred <- forest_samples$predict(forest_dataset)
    pred_expected_new <- cbind((pred_expected + pred_orig)/2.0, pred_orig)
    
    # Assertion
    expect_equal(pred, pred_expected_new)
    
    # Delete second forest
    forest_samples$delete_sample(1)
    
    # Check that predictions are as expected
    pred <- forest_samples$predict(forest_dataset)
    pred_expected_new <- (pred_expected + pred_orig)/2.0
    
    # Assertion
    expect_equal(pred, pred_expected_new)
})
