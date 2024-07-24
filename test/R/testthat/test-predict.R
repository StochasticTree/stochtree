test_that("Prediction from trees with constant leaf", {
    # Create dataset and forest container
    num_trees <- 10
    X = matrix(c(1.5, 8.7, 1.2, 
                 2.7, 3.4, 5.4, 
                 3.6, 1.2, 9.3, 
                 4.4, 5.4, 10.4, 
                 5.3, 9.3, 3.6, 
                 6.1, 10.4, 4.4), 
               byrow = T, nrow = 6)
    n <- nrow(X)
    p <- ncol(X)
    forest_dataset = createForestDataset(X)
    forest_samples <- createForestContainer(num_trees, 1, T)
    
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
    split_counts <- forest_samples$get_tree_split_counts(0,0,p)
    split_counts_expected <- c(1,1,0)
    
    # Assertion
    expect_equal(split_counts, split_counts_expected)
})

test_that("Prediction from trees with univariate leaf basis", {
    # Create dataset and forest container
    num_trees <- 10
    X = matrix(c(1.5, 8.7, 1.2, 
                 2.7, 3.4, 5.4, 
                 3.6, 1.2, 9.3, 
                 4.4, 5.4, 10.4, 
                 5.3, 9.3, 3.6, 
                 6.1, 10.4, 4.4), 
               byrow = T, nrow = 6)
    W = as.matrix(c(-1,-1,-1,1,1,1))
    n <- nrow(X)
    p <- ncol(X)
    forest_dataset = createForestDataset(X,W)
    forest_samples <- createForestContainer(num_trees, 1, F)
    
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
    pred_manual <- pred_raw*W
    
    # Assertion
    expect_equal(pred, pred_manual)
    
    # Split the left leaf of the first tree in the ensemble at X[,2] > 4.0
    forest_samples$add_numeric_split_tree(0, 0, 1, 1, 4.0, -7.5, -2.5)
    
    # Check that regular and "raw" predictions are the same (since the leaf is constant)
    pred <- forest_samples$predict(forest_dataset)
    pred_raw <- forest_samples$predict_raw(forest_dataset)
    pred_manual <- pred_raw*W
    
    # Assertion
    expect_equal(pred, pred_manual)
    
    # Check the split count for the first tree in the ensemble
    split_counts <- forest_samples$get_tree_split_counts(0,0,p)
    split_counts_expected <- c(1,1,0)
    
    # Assertion
    expect_equal(split_counts, split_counts_expected)
})

test_that("Prediction from trees with multivariate leaf basis", {
    # Create dataset and forest container
    num_trees <- 10
    output_dim <- 2
    num_samples <- 0
    X = matrix(c(1.5, 8.7, 1.2, 
                 2.7, 3.4, 5.4, 
                 3.6, 1.2, 9.3, 
                 4.4, 5.4, 10.4, 
                 5.3, 9.3, 3.6, 
                 6.1, 10.4, 4.4), 
               byrow = T, nrow = 6)
    n <- nrow(X)
    p <- ncol(X)
    W = matrix(c(1,1,1,1,1,1,-1,-1,-1,1,1,1), byrow=F, nrow=6)
    forest_dataset = createForestDataset(X,W)
    forest_samples <- createForestContainer(num_trees, output_dim, F)
    
    # Initialize a forest with constant root predictions
    forest_samples$add_forest_with_constant_leaves(c(1.,1.))
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
    forest_samples$add_numeric_split_tree(0, 0, 0, 0, 4.0, c(-5.,-1.), c(5.,1.))
    
    # Check that regular and "raw" predictions are the same (since the leaf is constant)
    pred <- forest_samples$predict(forest_dataset)
    pred_raw <- forest_samples$predict_raw(forest_dataset)
    pred_intermediate <- as.numeric(pred_raw) * as.numeric(W)
    dim(pred_intermediate) <- c(n, output_dim, num_samples)
    pred_manual <- apply(pred_intermediate, 3, function(x) rowSums(x))
    
    # Assertion
    expect_equal(pred, pred_manual)
    
    # Split the left leaf of the first tree in the ensemble at X[,2] > 4.0
    forest_samples$add_numeric_split_tree(0, 0, 1, 1, 4.0, c(-7.5,2.5), c(-2.5,7.5))
    
    # Check that regular and "raw" predictions are the same (since the leaf is constant)
    pred <- forest_samples$predict(forest_dataset)
    pred_raw <- forest_samples$predict_raw(forest_dataset)
    pred_intermediate <- as.numeric(pred_raw) * as.numeric(W)
    dim(pred_intermediate) <- c(n, output_dim, num_samples)
    pred_manual <- apply(pred_intermediate, 3, function(x) rowSums(x))
    
    # Assertion
    expect_equal(pred, pred_manual)
    
    # Check the split count for the first tree in the ensemble
    split_counts <- forest_samples$get_tree_split_counts(0,0,3)
    split_counts_expected <- c(1,1,0)
    
    # Assertion
    expect_equal(split_counts, split_counts_expected)
})
