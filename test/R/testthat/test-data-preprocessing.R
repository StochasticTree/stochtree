# test_that("Preprocessing of all-numeric covariate dataset works", {
#     cov_df <- data.frame(x1 = 1:5, x2 = 5:1, x3 = 6:10)
#     cov_mat <- matrix(c(
#         1,2,3,4,5,
#         5,4,3,2,1,
#         6,7,8,9,10
#     ), ncol = 3, byrow = FALSE)
#     preprocess_list <- createForestCovariates(cov_df)
#     expect_equal(preprocess_list$data, cov_mat)
#     expect_equal(preprocess_list$metadata$feature_types, rep(0,3))
#     expect_equal(preprocess_list$metadata$num_numeric_vars, 3)
#     expect_equal(preprocess_list$metadata$num_ordered_cat_vars, 0)
#     expect_equal(preprocess_list$metadata$num_unordered_cat_vars, 0)
#     expect_equal(preprocess_list$metadata$numeric_vars, c("x1","x2","x3"))
# })
#
# test_that("Preprocessing of all-unordered-categorical covariate dataset works", {
#     cov_df <- data.frame(x1 = 1:5, x2 = 5:1, x3 = 6:10)
#     cov_mat <- matrix(c(
#         1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,
#         0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,
#         0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,
#         0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,
#         0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0
#     ), nrow = 5, byrow = TRUE)
#     preprocess_list <- createForestCovariates(cov_df, unordered_cat_vars = c("x1","x2","x3"))
#     expect_equal(preprocess_list$data, cov_mat)
#     expect_equal(preprocess_list$metadata$feature_types, rep(1,18))
#     expect_equal(preprocess_list$metadata$num_numeric_vars, 0)
#     expect_equal(preprocess_list$metadata$num_ordered_cat_vars, 0)
#     expect_equal(preprocess_list$metadata$num_unordered_cat_vars, 3)
#     expect_equal(preprocess_list$metadata$unordered_cat_vars, c("x1","x2","x3"))
#     expect_equal(preprocess_list$metadata$unordered_unique_levels,
#                  list(x1=c("1","2","3","4","5"),
#                       x2=c("1","2","3","4","5"),
#                       x3=c("6","7","8","9","10"))
#     )
# })
#
# test_that("Preprocessing of all-ordered-categorical covariate dataset works", {
#     cov_df <- data.frame(x1 = 1:5, x2 = 5:1, x3 = 6:10)
#     cov_mat <- matrix(c(
#         1,2,3,4,5,
#         5,4,3,2,1,
#         1,2,3,4,5
#     ), ncol = 3, byrow = FALSE)
#     preprocess_list <- createForestCovariates(cov_df, ordered_cat_vars = c("x1","x2","x3"))
#     expect_equal(preprocess_list$data, cov_mat)
#     expect_equal(preprocess_list$metadata$feature_types, rep(1,3))
#     expect_equal(preprocess_list$metadata$num_numeric_vars, 0)
#     expect_equal(preprocess_list$metadata$num_ordered_cat_vars, 3)
#     expect_equal(preprocess_list$metadata$num_unordered_cat_vars, 0)
#     expect_equal(preprocess_list$metadata$ordered_cat_vars, c("x1","x2","x3"))
#     expect_equal(preprocess_list$metadata$ordered_unique_levels,
#                  list(x1=c("1","2","3","4","5"),
#                       x2=c("1","2","3","4","5"),
#                       x3=c("6","7","8","9","10"))
#     )
# })
#
# test_that("Preprocessing of mixed-covariate dataset works", {
#     cov_df <- data.frame(x1 = 1:5, x2 = 5:1, x3 = 6:10)
#     cov_mat <- matrix(c(
#         1,5,1,0,0,0,0,0,
#         2,4,0,1,0,0,0,0,
#         3,3,0,0,1,0,0,0,
#         4,2,0,0,0,1,0,0,
#         5,1,0,0,0,0,1,0
#     ), nrow = 5, byrow = TRUE)
#     preprocess_list <- createForestCovariates(cov_df, ordered_cat_vars = c("x2"), unordered_cat_vars = "x3")
#     expect_equal(preprocess_list$data, cov_mat)
#     expect_equal(preprocess_list$metadata$feature_types, c(0, rep(1,7)))
#     expect_equal(preprocess_list$metadata$num_numeric_vars, 1)
#     expect_equal(preprocess_list$metadata$num_ordered_cat_vars, 1)
#     expect_equal(preprocess_list$metadata$num_unordered_cat_vars, 1)
#     expect_equal(preprocess_list$metadata$ordered_cat_vars, c("x2"))
#     expect_equal(preprocess_list$metadata$unordered_cat_vars, c("x3"))
#     expect_equal(preprocess_list$metadata$ordered_unique_levels, list(x2=c("1","2","3","4","5")))
#     expect_equal(preprocess_list$metadata$unordered_unique_levels, list(x3=c("6","7","8","9","10")))
# })
#
# test_that("Preprocessing of mixed-covariate matrix works", {
#     cov_input <- matrix(c(1:5,5:1,6:10),ncol=3,byrow=FALSE)
#     cov_mat <- matrix(c(
#         1,5,1,0,0,0,0,0,
#         2,4,0,1,0,0,0,0,
#         3,3,0,0,1,0,0,0,
#         4,2,0,0,0,1,0,0,
#         5,1,0,0,0,0,1,0
#     ), nrow = 5, byrow = TRUE)
#     preprocess_list <- createForestCovariates(cov_input, ordered_cat_vars = 2, unordered_cat_vars = 3)
#     expect_equal(preprocess_list$data, cov_mat)
#     expect_equal(preprocess_list$metadata$feature_types, c(0, rep(1,7)))
#     expect_equal(preprocess_list$metadata$num_numeric_vars, 1)
#     expect_equal(preprocess_list$metadata$num_ordered_cat_vars, 1)
#     expect_equal(preprocess_list$metadata$num_unordered_cat_vars, 1)
#     expect_equal(preprocess_list$metadata$ordered_cat_vars, c("x2"))
#     expect_equal(preprocess_list$metadata$unordered_cat_vars, c("x3"))
#     expect_equal(preprocess_list$metadata$ordered_unique_levels, list(x2=c("1","2","3","4","5")))
#     expect_equal(preprocess_list$metadata$unordered_unique_levels, list(x3=c("6","7","8","9","10")))
#
#     alt_preprocess_list <- createForestCovariates(cov_input, ordered_cat_vars = "x2", unordered_cat_vars = "x3")
#     expect_equal(alt_preprocess_list$data, cov_mat)
#     expect_equal(alt_preprocess_list$metadata$feature_types, c(0, rep(1,7)))
#     expect_equal(alt_preprocess_list$metadata$num_numeric_vars, 1)
#     expect_equal(alt_preprocess_list$metadata$num_ordered_cat_vars, 1)
#     expect_equal(alt_preprocess_list$metadata$num_unordered_cat_vars, 1)
#     expect_equal(alt_preprocess_list$metadata$ordered_cat_vars, c("x2"))
#     expect_equal(alt_preprocess_list$metadata$unordered_cat_vars, c("x3"))
#     expect_equal(alt_preprocess_list$metadata$ordered_unique_levels, list(x2=c("1","2","3","4","5")))
#     expect_equal(alt_preprocess_list$metadata$unordered_unique_levels, list(x3=c("6","7","8","9","10")))
# })
#
# test_that("Preprocessing of out-of-sample mixed-covariate dataset works", {
#     metadata <- list(
#         num_numeric_vars = 1,
#         num_ordered_cat_vars = 1,
#         num_unordered_cat_vars = 1,
#         numeric_vars = c("x1"),
#         ordered_cat_vars = c("x2"),
#         unordered_cat_vars = c("x3"),
#         ordered_unique_levels = list(x2=c("1","2","3","4","5")),
#         unordered_unique_levels = list(x3=c("6","7","8","9","10"))
#     )
#     cov_df <- data.frame(x1 = c(1:5,1), x2 = c(5:1,5), x3 = 6:11)
#     cov_mat <- matrix(c(
#         1,5,1,0,0,0,0,0,
#         2,4,0,1,0,0,0,0,
#         3,3,0,0,1,0,0,0,
#         4,2,0,0,0,1,0,0,
#         5,1,0,0,0,0,1,0,
#         1,5,0,0,0,0,0,1
#     ), nrow = 6, byrow = TRUE)
#     X_preprocessed <- createForestCovariatesFromMetadata(cov_df, metadata)
#     expect_equal(X_preprocessed, cov_mat)
# })
#
# test_that("Preprocessing of all-numeric covariate dataset works", {
#     cov_df <- data.frame(x1 = 1:5, x2 = 5:1, x3 = 6:10)
#     cov_mat <- matrix(c(
#         1,2,3,4,5,
#         5,4,3,2,1,
#         6,7,8,9,10
#     ), ncol = 3, byrow = FALSE)
#     preprocess_list <- preprocessTrainDataFrame(cov_df)
#     expect_equal(preprocess_list$data, cov_mat)
#     expect_equal(preprocess_list$metadata$feature_types, rep(0,3))
#     expect_equal(preprocess_list$metadata$num_numeric_vars, 3)
#     expect_equal(preprocess_list$metadata$num_ordered_cat_vars, 0)
#     expect_equal(preprocess_list$metadata$num_unordered_cat_vars, 0)
#     expect_equal(preprocess_list$metadata$original_var_indices, 1:3)
#     expect_equal(preprocess_list$metadata$numeric_vars, c("x1","x2","x3"))
# })
#
# test_that("Preprocessing of all-unordered-categorical covariate dataset works", {
#     cov_df <- data.frame(x1 = 1:5, x2 = 5:1, x3 = 6:10)
#     cov_df$x1 <- factor(cov_df$x1)
#     cov_df$x2 <- factor(cov_df$x2)
#     cov_df$x3 <- factor(cov_df$x3)
#     cov_mat <- matrix(c(
#         1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,
#         0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,
#         0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,
#         0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,
#         0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0
#     ), nrow = 5, byrow = TRUE)
#     preprocess_list <- preprocessTrainDataFrame(cov_df)
#     expect_equal(preprocess_list$data, cov_mat)
#     expect_equal(preprocess_list$metadata$feature_types, rep(1,18))
#     expect_equal(preprocess_list$metadata$num_numeric_vars, 0)
#     expect_equal(preprocess_list$metadata$num_ordered_cat_vars, 0)
#     expect_equal(preprocess_list$metadata$num_unordered_cat_vars, 3)
#     expect_equal(preprocess_list$metadata$unordered_cat_vars, c("x1","x2","x3"))
#     expected_var_indices <- c(rep(1,6),rep(2,6),rep(3,6))
#     expect_equal(preprocess_list$metadata$original_var_indices, expected_var_indices)
#     expect_equal(preprocess_list$metadata$unordered_unique_levels,
#                  list(x1=c("1","2","3","4","5"),
#                       x2=c("1","2","3","4","5"),
#                       x3=c("6","7","8","9","10"))
#     )
# })
#
# test_that("Preprocessing of all-ordered-categorical covariate dataset works", {
#     cov_df <- data.frame(x1 = 1:5, x2 = 5:1, x3 = 6:10)
#     cov_df$x1 <- factor(cov_df$x1, ordered = TRUE)
#     cov_df$x2 <- factor(cov_df$x2, ordered = TRUE)
#     cov_df$x3 <- factor(cov_df$x3, ordered = TRUE)
#     cov_mat <- matrix(c(
#         1,2,3,4,5,
#         5,4,3,2,1,
#         1,2,3,4,5
#     ), ncol = 3, byrow = FALSE)
#     preprocess_list <- preprocessTrainDataFrame(cov_df)
#     expect_equal(preprocess_list$data, cov_mat)
#     expect_equal(preprocess_list$metadata$feature_types, rep(1,3))
#     expect_equal(preprocess_list$metadata$num_numeric_vars, 0)
#     expect_equal(preprocess_list$metadata$num_ordered_cat_vars, 3)
#     expect_equal(preprocess_list$metadata$num_unordered_cat_vars, 0)
#     expect_equal(preprocess_list$metadata$ordered_cat_vars, c("x1","x2","x3"))
#     expect_equal(preprocess_list$metadata$original_var_indices, 1:3)
#     expect_equal(preprocess_list$metadata$ordered_unique_levels,
#                  list(x1=c("1","2","3","4","5"),
#                       x2=c("1","2","3","4","5"),
#                       x3=c("6","7","8","9","10"))
#     )
# })
#
# test_that("Preprocessing of mixed-covariate dataset works", {
#     cov_df <- data.frame(x1 = 1:5, x2 = 5:1, x3 = 6:10)
#     cov_df$x2 <- factor(cov_df$x2, ordered = TRUE)
#     cov_df$x3 <- factor(cov_df$x3)
#     cov_mat <- matrix(c(
#         1,5,1,0,0,0,0,0,
#         2,4,0,1,0,0,0,0,
#         3,3,0,0,1,0,0,0,
#         4,2,0,0,0,1,0,0,
#         5,1,0,0,0,0,1,0
#     ), nrow = 5, byrow = TRUE)
#     preprocess_list <- preprocessTrainDataFrame(cov_df)
#     expect_equal(preprocess_list$data, cov_mat)
#     expect_equal(preprocess_list$metadata$feature_types, c(0, rep(1,7)))
#     expect_equal(preprocess_list$metadata$num_numeric_vars, 1)
#     expect_equal(preprocess_list$metadata$num_ordered_cat_vars, 1)
#     expect_equal(preprocess_list$metadata$num_unordered_cat_vars, 1)
#     expect_equal(preprocess_list$metadata$ordered_cat_vars, c("x2"))
#     expect_equal(preprocess_list$metadata$unordered_cat_vars, c("x3"))
#     expected_var_indices <- c(1,2,rep(3,6))
#     expect_equal(preprocess_list$metadata$original_var_indices, expected_var_indices)
#     expect_equal(preprocess_list$metadata$ordered_unique_levels, list(x2=c("1","2","3","4","5")))
#     expect_equal(preprocess_list$metadata$unordered_unique_levels, list(x3=c("6","7","8","9","10")))
# })
#
# test_that("Preprocessing of out-of-sample mixed-covariate dataset works", {
#     metadata <- list(
#         num_numeric_vars = 1,
#         num_ordered_cat_vars = 1,
#         num_unordered_cat_vars = 1,
#         original_var_indices = c(1, 2, 3, 3, 3, 3, 3, 3),
#         numeric_vars = c("x1"),
#         ordered_cat_vars = c("x2"),
#         unordered_cat_vars = c("x3"),
#         ordered_unique_levels = list(x2=c("1","2","3","4","5")),
#         unordered_unique_levels = list(x3=c("6","7","8","9","10"))
#     )
#     cov_df <- data.frame(x1 = c(1:5,1), x2 = c(5:1,5), x3 = 6:11)
#     var_weights <- rep(1./3., 3)
#     cov_mat <- matrix(c(
#         1,5,1,0,0,0,0,0,
#         2,4,0,1,0,0,0,0,
#         3,3,0,0,1,0,0,0,
#         4,2,0,0,0,1,0,0,
#         5,1,0,0,0,0,1,0,
#         1,5,0,0,0,0,0,1
#     ), nrow = 6, byrow = TRUE)
#     X_preprocessed <- preprocessPredictionDataFrame(cov_df, metadata)
#     expect_equal(X_preprocessed, cov_mat)
# })

test_that("Matrix preprocessor produces clean numeric_vars (no NA tail)", {
  skip_on_cran()

  set.seed(1)
  cov_mat <- matrix(rnorm(40 * 4), ncol = 4)
  md <- preprocessTrainData(cov_mat)$metadata

  expect_equal(md$num_numeric_vars, 4)
  # numeric_vars must be exactly the p column names, not a names() vector
  # padded out to n * ncol with NA (the bug this guards against).
  expect_equal(md$numeric_vars, paste0("x", 1:4))
  expect_length(md$numeric_vars, 4)
  expect_false(anyNA(md$numeric_vars))
})

test_that("Matrix preprocessor JSON round-trip preserves clean numeric_vars", {
  skip_on_cran()

  set.seed(2)
  cov_mat <- matrix(rnorm(60 * 3), ncol = 3)
  md <- preprocessTrainData(cov_mat)$metadata

  md_roundtrip <- createPreprocessorFromJsonString(
    savePreprocessorToJsonString(md)
  )
  expect_equal(md_roundtrip$numeric_vars, paste0("x", 1:3))
  expect_length(md_roundtrip$numeric_vars, 3)
  expect_false(anyNA(md_roundtrip$numeric_vars))
})

test_that("Matrix-trained BART preprocessor has clean numeric_vars after reload", {
  skip_on_cran()

  set.seed(3)
  n <- 200
  X <- matrix(rnorm(n * 4), ncol = 4)
  y <- X[, 1] + 0.5 * X[, 2] + rnorm(n)
  bart_model <- bart(
    X_train = X, y_train = y,
    num_gfr = 0, num_burnin = 0, num_mcmc = 10
  )
  reloaded <- createBARTModelFromJsonString(saveBARTModelToJsonString(bart_model))

  nv <- reloaded$train_set_metadata$numeric_vars
  expect_equal(nv, paste0("x", 1:4))
  expect_false(anyNA(nv))
})
