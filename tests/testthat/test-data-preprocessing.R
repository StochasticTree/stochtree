test_that("Preprocessing of all-numeric covariate dataset works", {
    cov_df <- data.frame(x1 = 1:5, x2 = 5:1, x3 = 6:10)
    cov_mat <- matrix(c(
        1,2,3,4,5,
        5,4,3,2,1,
        6,7,8,9,10
    ), ncol = 3, byrow = F)
    preprocess_list <- createForestCovariates(cov_df)
    expect_equal(preprocess_list$data, cov_mat)
    expect_equal(preprocess_list$metadata$feature_types, rep(0,3))
    expect_equal(preprocess_list$metadata$num_numeric_vars, 3)
    expect_equal(preprocess_list$metadata$num_ordered_cat_vars, 0)
    expect_equal(preprocess_list$metadata$num_unordered_cat_vars, 0)
    expect_equal(preprocess_list$metadata$numeric_vars, c("x1","x2","x3"))
})

test_that("Preprocessing of all-unordered-categorical covariate dataset works", {
    cov_df <- data.frame(x1 = 1:5, x2 = 5:1, x3 = 6:10)
    cov_mat <- matrix(c(
        1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,
        0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,
        0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,
        0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,
        0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0
    ), nrow = 5, byrow = TRUE)
    preprocess_list <- createForestCovariates(cov_df, unordered_cat_vars = c("x1","x2","x3"))
    expect_equal(preprocess_list$data, cov_mat)
    expect_equal(preprocess_list$metadata$feature_types, rep(1,18))
    expect_equal(preprocess_list$metadata$num_numeric_vars, 0)
    expect_equal(preprocess_list$metadata$num_ordered_cat_vars, 0)
    expect_equal(preprocess_list$metadata$num_unordered_cat_vars, 3)
    expect_equal(preprocess_list$metadata$unordered_cat_vars, c("x1","x2","x3"))
    expect_equal(preprocess_list$metadata$unordered_unique_levels, 
                 list(x1=c("1","2","3","4","5"), 
                      x2=c("1","2","3","4","5"), 
                      x3=c("6","7","8","9","10"))
    )
})

test_that("Preprocessing of all-ordered-categorical covariate dataset works", {
    cov_df <- data.frame(x1 = 1:5, x2 = 5:1, x3 = 6:10)
    cov_mat <- matrix(c(
        1,2,3,4,5,
        5,4,3,2,1,
        1,2,3,4,5
    ), ncol = 3, byrow = F)
    preprocess_list <- createForestCovariates(cov_df, ordered_cat_vars = c("x1","x2","x3"))
    expect_equal(preprocess_list$data, cov_mat)
    expect_equal(preprocess_list$metadata$feature_types, rep(1,3))
    expect_equal(preprocess_list$metadata$num_numeric_vars, 0)
    expect_equal(preprocess_list$metadata$num_ordered_cat_vars, 3)
    expect_equal(preprocess_list$metadata$num_unordered_cat_vars, 0)
    expect_equal(preprocess_list$metadata$ordered_cat_vars, c("x1","x2","x3"))
    expect_equal(preprocess_list$metadata$ordered_unique_levels, 
                 list(x1=c("1","2","3","4","5"), 
                      x2=c("1","2","3","4","5"), 
                      x3=c("6","7","8","9","10"))
    )
})

test_that("Preprocessing of mixed-covariate dataset works", {
    cov_df <- data.frame(x1 = 1:5, x2 = 5:1, x3 = 6:10)
    cov_mat <- matrix(c(
        1,5,1,0,0,0,0,0,
        2,4,0,1,0,0,0,0,
        3,3,0,0,1,0,0,0,
        4,2,0,0,0,1,0,0,
        5,1,0,0,0,0,1,0
    ), nrow = 5, byrow = TRUE)
    preprocess_list <- createForestCovariates(cov_df, ordered_cat_vars = c("x2"), unordered_cat_vars = "x3")
    expect_equal(preprocess_list$data, cov_mat)
    expect_equal(preprocess_list$metadata$feature_types, c(0, rep(1,7)))
    expect_equal(preprocess_list$metadata$num_numeric_vars, 1)
    expect_equal(preprocess_list$metadata$num_ordered_cat_vars, 1)
    expect_equal(preprocess_list$metadata$num_unordered_cat_vars, 1)
    expect_equal(preprocess_list$metadata$ordered_cat_vars, c("x2"))
    expect_equal(preprocess_list$metadata$unordered_cat_vars, c("x3"))
    expect_equal(preprocess_list$metadata$ordered_unique_levels, list(x2=c("1","2","3","4","5")))
    expect_equal(preprocess_list$metadata$unordered_unique_levels, list(x3=c("6","7","8","9","10")))
})

test_that("Preprocessing of mixed-covariate matrix works", {
    cov_input <- matrix(c(1:5,5:1,6:10),ncol=3,byrow=F)
    cov_mat <- matrix(c(
        1,5,1,0,0,0,0,0,
        2,4,0,1,0,0,0,0,
        3,3,0,0,1,0,0,0,
        4,2,0,0,0,1,0,0,
        5,1,0,0,0,0,1,0
    ), nrow = 5, byrow = TRUE)
    preprocess_list <- createForestCovariates(cov_input, ordered_cat_vars = 2, unordered_cat_vars = 3)
    expect_equal(preprocess_list$data, cov_mat)
    expect_equal(preprocess_list$metadata$feature_types, c(0, rep(1,7)))
    expect_equal(preprocess_list$metadata$num_numeric_vars, 1)
    expect_equal(preprocess_list$metadata$num_ordered_cat_vars, 1)
    expect_equal(preprocess_list$metadata$num_unordered_cat_vars, 1)
    expect_equal(preprocess_list$metadata$ordered_cat_vars, c("x2"))
    expect_equal(preprocess_list$metadata$unordered_cat_vars, c("x3"))
    expect_equal(preprocess_list$metadata$ordered_unique_levels, list(x2=c("1","2","3","4","5")))
    expect_equal(preprocess_list$metadata$unordered_unique_levels, list(x3=c("6","7","8","9","10")))
    
    alt_preprocess_list <- createForestCovariates(cov_input, ordered_cat_vars = "x2", unordered_cat_vars = "x3")
    expect_equal(alt_preprocess_list$data, cov_mat)
    expect_equal(alt_preprocess_list$metadata$feature_types, c(0, rep(1,7)))
    expect_equal(alt_preprocess_list$metadata$num_numeric_vars, 1)
    expect_equal(alt_preprocess_list$metadata$num_ordered_cat_vars, 1)
    expect_equal(alt_preprocess_list$metadata$num_unordered_cat_vars, 1)
    expect_equal(alt_preprocess_list$metadata$ordered_cat_vars, c("x2"))
    expect_equal(alt_preprocess_list$metadata$unordered_cat_vars, c("x3"))
    expect_equal(alt_preprocess_list$metadata$ordered_unique_levels, list(x2=c("1","2","3","4","5")))
    expect_equal(alt_preprocess_list$metadata$unordered_unique_levels, list(x3=c("6","7","8","9","10")))
})

test_that("Preprocessing of out-of-sample mixed-covariate dataset works", {
    metadata <- list(
        num_numeric_vars = 1, 
        num_ordered_cat_vars = 1, 
        num_unordered_cat_vars = 1, 
        numeric_vars = c("x1"), 
        ordered_cat_vars = c("x2"), 
        unordered_cat_vars = c("x3"), 
        ordered_unique_levels = list(x2=c("1","2","3","4","5")), 
        unordered_unique_levels = list(x3=c("6","7","8","9","10"))
    )
    cov_df <- data.frame(x1 = c(1:5,1), x2 = c(5:1,5), x3 = 6:11)
    cov_mat <- matrix(c(
        1,5,1,0,0,0,0,0,
        2,4,0,1,0,0,0,0,
        3,3,0,0,1,0,0,0,
        4,2,0,0,0,1,0,0,
        5,1,0,0,0,0,1,0,
        1,5,0,0,0,0,0,1
    ), nrow = 6, byrow = TRUE)
    X_preprocessed <- createForestCovariatesFromMetadata(cov_df, metadata)
    expect_equal(X_preprocessed, cov_mat)
})

test_that("Preprocessing of all-numeric covariate dataset works", {
    cov_df <- data.frame(x1 = 1:5, x2 = 5:1, x3 = 6:10)
    cov_mat <- matrix(c(
        1,2,3,4,5,
        5,4,3,2,1,
        6,7,8,9,10
    ), ncol = 3, byrow = F)
    preprocess_list <- preprocessTrainDataFrame(cov_df)
    expect_equal(preprocess_list$data, cov_mat)
    expect_equal(preprocess_list$metadata$feature_types, rep(0,3))
    expect_equal(preprocess_list$metadata$num_numeric_vars, 3)
    expect_equal(preprocess_list$metadata$num_ordered_cat_vars, 0)
    expect_equal(preprocess_list$metadata$num_unordered_cat_vars, 0)
    expect_equal(preprocess_list$metadata$original_var_indices, 1:3)
    expect_equal(preprocess_list$metadata$numeric_vars, c("x1","x2","x3"))
})

test_that("Preprocessing of all-unordered-categorical covariate dataset works", {
    cov_df <- data.frame(x1 = 1:5, x2 = 5:1, x3 = 6:10)
    cov_df$x1 <- factor(cov_df$x1)
    cov_df$x2 <- factor(cov_df$x2)
    cov_df$x3 <- factor(cov_df$x3)
    cov_mat <- matrix(c(
        1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,
        0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,
        0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,
        0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,
        0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0
    ), nrow = 5, byrow = TRUE)
    preprocess_list <- preprocessTrainDataFrame(cov_df)
    expect_equal(preprocess_list$data, cov_mat)
    expect_equal(preprocess_list$metadata$feature_types, rep(1,18))
    expect_equal(preprocess_list$metadata$num_numeric_vars, 0)
    expect_equal(preprocess_list$metadata$num_ordered_cat_vars, 0)
    expect_equal(preprocess_list$metadata$num_unordered_cat_vars, 3)
    expect_equal(preprocess_list$metadata$unordered_cat_vars, c("x1","x2","x3"))
    expected_var_indices <- c(rep(1,6),rep(2,6),rep(3,6))
    expect_equal(preprocess_list$metadata$original_var_indices, expected_var_indices)
    expect_equal(preprocess_list$metadata$unordered_unique_levels, 
                 list(x1=c("1","2","3","4","5"), 
                      x2=c("1","2","3","4","5"), 
                      x3=c("6","7","8","9","10"))
    )
})

test_that("Preprocessing of all-ordered-categorical covariate dataset works", {
    cov_df <- data.frame(x1 = 1:5, x2 = 5:1, x3 = 6:10)
    cov_df$x1 <- factor(cov_df$x1, ordered = TRUE)
    cov_df$x2 <- factor(cov_df$x2, ordered = TRUE)
    cov_df$x3 <- factor(cov_df$x3, ordered = TRUE)
    cov_mat <- matrix(c(
        1,2,3,4,5,
        5,4,3,2,1,
        1,2,3,4,5
    ), ncol = 3, byrow = F)
    preprocess_list <- preprocessTrainDataFrame(cov_df)
    expect_equal(preprocess_list$data, cov_mat)
    expect_equal(preprocess_list$metadata$feature_types, rep(1,3))
    expect_equal(preprocess_list$metadata$num_numeric_vars, 0)
    expect_equal(preprocess_list$metadata$num_ordered_cat_vars, 3)
    expect_equal(preprocess_list$metadata$num_unordered_cat_vars, 0)
    expect_equal(preprocess_list$metadata$ordered_cat_vars, c("x1","x2","x3"))
    expect_equal(preprocess_list$metadata$original_var_indices, 1:3)
    expect_equal(preprocess_list$metadata$ordered_unique_levels, 
                 list(x1=c("1","2","3","4","5"), 
                      x2=c("1","2","3","4","5"), 
                      x3=c("6","7","8","9","10"))
    )
})

test_that("Preprocessing of mixed-covariate dataset works", {
    cov_df <- data.frame(x1 = 1:5, x2 = 5:1, x3 = 6:10)
    cov_df$x2 <- factor(cov_df$x2, ordered = TRUE)
    cov_df$x3 <- factor(cov_df$x3)
    cov_mat <- matrix(c(
        1,5,1,0,0,0,0,0,
        2,4,0,1,0,0,0,0,
        3,3,0,0,1,0,0,0,
        4,2,0,0,0,1,0,0,
        5,1,0,0,0,0,1,0
    ), nrow = 5, byrow = TRUE)
    preprocess_list <- preprocessTrainDataFrame(cov_df)
    expect_equal(preprocess_list$data, cov_mat)
    expect_equal(preprocess_list$metadata$feature_types, c(0, rep(1,7)))
    expect_equal(preprocess_list$metadata$num_numeric_vars, 1)
    expect_equal(preprocess_list$metadata$num_ordered_cat_vars, 1)
    expect_equal(preprocess_list$metadata$num_unordered_cat_vars, 1)
    expect_equal(preprocess_list$metadata$ordered_cat_vars, c("x2"))
    expect_equal(preprocess_list$metadata$unordered_cat_vars, c("x3"))
    expected_var_indices <- c(1,2,rep(3,6))
    expect_equal(preprocess_list$metadata$original_var_indices, expected_var_indices)
    expect_equal(preprocess_list$metadata$ordered_unique_levels, list(x2=c("1","2","3","4","5")))
    expect_equal(preprocess_list$metadata$unordered_unique_levels, list(x3=c("6","7","8","9","10")))
})

test_that("Preprocessing of out-of-sample mixed-covariate dataset works", {
    metadata <- list(
        num_numeric_vars = 1, 
        num_ordered_cat_vars = 1, 
        num_unordered_cat_vars = 1, 
        original_var_indices = c(1, 2, 3, 3, 3, 3, 3, 3),
        numeric_vars = c("x1"), 
        ordered_cat_vars = c("x2"), 
        unordered_cat_vars = c("x3"), 
        ordered_unique_levels = list(x2=c("1","2","3","4","5")), 
        unordered_unique_levels = list(x3=c("6","7","8","9","10"))
    )
    cov_df <- data.frame(x1 = c(1:5,1), x2 = c(5:1,5), x3 = 6:11)
    var_weights <- rep(1./3., 3)
    cov_mat <- matrix(c(
        1,5,1,0,0,0,0,0,
        2,4,0,1,0,0,0,0,
        3,3,0,0,1,0,0,0,
        4,2,0,0,0,1,0,0,
        5,1,0,0,0,0,1,0,
        1,5,0,0,0,0,0,1
    ), nrow = 6, byrow = TRUE)
    X_preprocessed <- preprocessPredictionDataFrame(cov_df, metadata)
    expect_equal(X_preprocessed, cov_mat)
})
