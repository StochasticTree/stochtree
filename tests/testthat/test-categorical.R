test_that("In-sample one-hot encoding works for unordered categorical variables", {
    x1 <- c(3,2,1,4,3,2,3,2,4,2)
    x1_onehot <- oneHotInitializeAndEncode(x1)
    x1_expected <- matrix(
        c(0,0,1,0,0,
          0,1,0,0,0,
          1,0,0,0,0,
          0,0,0,1,0,
          0,0,1,0,0,
          0,1,0,0,0,
          0,0,1,0,0,
          0,1,0,0,0,
          0,0,0,1,0,
          0,1,0,0,0),
    byrow = T, ncol = 5)
    x1_levels_expected <- c("1","2","3","4")
    
    x2 <- c("a","c","b","c","d","a","c","a","b","d")
    x2_onehot <- oneHotInitializeAndEncode(x2)
    x2_expected <- matrix(
        c(1,0,0,0,0,
          0,0,1,0,0,
          0,1,0,0,0,
          0,0,1,0,0,
          0,0,0,1,0,
          1,0,0,0,0,
          0,0,1,0,0,
          1,0,0,0,0,
          0,1,0,0,0,
          0,0,0,1,0),
        byrow = T, ncol = 5)
    x2_levels_expected <- c("a","b","c","d")
    
    x3 <- c(3.2,2.4,1.5,4.6,3.2,2.4,3.2,2.4,4.6,2.4)
    x3_onehot <- oneHotInitializeAndEncode(x3)
    x3_expected <- matrix(
        c(0,0,1,0,0,
          0,1,0,0,0,
          1,0,0,0,0,
          0,0,0,1,0,
          0,0,1,0,0,
          0,1,0,0,0,
          0,0,1,0,0,
          0,1,0,0,0,
          0,0,0,1,0,
          0,1,0,0,0),
        byrow = T, ncol = 5)
    x3_levels_expected <- c("1.5","2.4","3.2","4.6")
    
    expect_equal(x1_onehot$Xtilde, x1_expected)
    expect_equal(x2_onehot$Xtilde, x2_expected)
    expect_equal(x3_onehot$Xtilde, x3_expected)
    expect_equal(x1_onehot$unique_levels, x1_levels_expected)
    expect_equal(x2_onehot$unique_levels, x2_levels_expected)
    expect_equal(x3_onehot$unique_levels, x3_levels_expected)
})

test_that("Out-of-sample one-hot encoding works for unordered categorical variables", {
    x1 <- c(3,2,1,4,3,2,3,2,4,2)
    x1_test <- c(1,2,4,3,5)
    x1_test_onehot <- oneHotEncode(x1_test, levels(factor(x1)))
    x1_test_expected <- matrix(
        c(1,0,0,0,0,
          0,1,0,0,0,
          0,0,0,1,0,
          0,0,1,0,0,
          0,0,0,0,1),
        byrow = T, ncol = 5)
    
    x2 <- c("a","c","b","c","d","a","c","a","b","d")
    x2_test <- c("a","c","g","b","f")
    x2_test_onehot <- oneHotEncode(x2_test, levels(factor(x2)))
    x2_test_expected <- matrix(
        c(1,0,0,0,0,
          0,0,1,0,0,
          0,0,0,0,1,
          0,1,0,0,0,
          0,0,0,0,1),
        byrow = T, ncol = 5)
    
    x3 <- c(3.2,2.4,1.5,4.6,3.2,2.4,3.2,2.4,4.6,2.4)
    x3_test <- c(10.3,-0.5,4.6,3.2,1.8)
    x3_test_onehot <- oneHotEncode(x3_test, levels(factor(x3)))
    x3_test_expected <- matrix(
        c(0,0,0,0,1,
          0,0,0,0,1,
          0,0,0,1,0,
          0,0,1,0,0,
          0,0,0,0,1),
        byrow = T, ncol = 5)
    
    expect_equal(x1_test_onehot, x1_test_expected)
    expect_equal(x2_test_onehot, x2_test_expected)
    expect_equal(x3_test_onehot, x3_test_expected)
})

test_that("In-sample preprocessing for ordered categorical variables", {
    string_var_response_levels <- c("1. Strongly disagree", "2. Disagree", "3. Neither agree nor disagree", "4. Agree", "5. Strongly agree")
    
    x1 <- c("1. Strongly disagree", "3. Neither agree nor disagree", "2. Disagree", "4. Agree", "3. Neither agree nor disagree", "5. Strongly agree", "4. Agree")
    x1_preprocessing <- orderedCatInitializeAndPreprocess(x1)
    x1_vector_expected <- c(1,3,2,4,3,5,4)
    x1_levels_expected <- string_var_response_levels
    
    x2 <- factor(x1, levels = string_var_response_levels, ordered = T)
    x2_preprocessing <- orderedCatInitializeAndPreprocess(x2)
    x2_vector_expected <- c(1,3,2,4,3,5,4)
    x2_levels_expected <- string_var_response_levels
    
    string_var_levels_reordered <- c("5. Strongly agree", "4. Agree", "3. Neither agree nor disagree", "2. Disagree", "1. Strongly disagree")
    x3 <- factor(x1, levels = string_var_levels_reordered, ordered = T)
    x3_preprocessing <- orderedCatInitializeAndPreprocess(x3)
    x3_vector_expected <- c(5,3,4,2,3,1,2)
    x3_levels_expected <- string_var_levels_reordered
    
    x4 <- c(3,2,4,6,5,2,3,1,3,4,6)
    x4_preprocessing <- orderedCatInitializeAndPreprocess(x4)
    x4_vector_expected <- c(3,2,4,6,5,2,3,1,3,4,6)
    x4_levels_expected <- c("1","2","3","4","5","6")

    expect_equal(x1_preprocessing$x_preprocessed, x1_vector_expected)
    expect_equal(x2_preprocessing$x_preprocessed, x2_vector_expected)
    expect_equal(x3_preprocessing$x_preprocessed, x3_vector_expected)
    expect_equal(x4_preprocessing$x_preprocessed, x4_vector_expected)
    expect_equal(x1_preprocessing$unique_levels, x1_levels_expected)
    expect_equal(x2_preprocessing$unique_levels, x2_levels_expected)
    expect_equal(x3_preprocessing$unique_levels, x3_levels_expected)
    expect_equal(x4_preprocessing$unique_levels, x4_levels_expected)
})

test_that("Out-of-sample preprocessing for ordered categorical variables", {
    string_var_response_levels <- c("1. Strongly disagree", "2. Disagree", "3. Neither agree nor disagree", "4. Agree", "5. Strongly agree")
    
    x1 <- c("1. Strongly disagree", "3. Neither agree nor disagree", "2. Disagree", "4. Agree", "3. Neither agree nor disagree", "5. Strongly agree", "4. Agree")
    x1_preprocessing <- orderedCatPreprocess(x1, string_var_response_levels)
    x1_vector_expected <- c(1,3,2,4,3,5,4)
    
    x2 <- factor(x1, levels = string_var_response_levels, ordered = T)
    x2_preprocessing <- orderedCatPreprocess(x2, string_var_response_levels)
    x2_vector_expected <- c(1,3,2,4,3,5,4)

    x3 <- c("1. Strongly disagree", "6. Other", "7. Also other", "4. Agree", "3. Neither agree nor disagree", "5. Strongly agree", "4. Agree")
    expected_warning_message <- "Variable includes ordered categorical levels not included in the original training set"
    expect_warning(x3_preprocessing <- orderedCatPreprocess(x3, string_var_response_levels), expected_warning_message)
    x3_vector_expected <- c(1,6,6,4,3,5,4)
    
    x4 <- c(3,2,4,6,5,2,3,1,3,4,6)
    x4_preprocessing <- orderedCatPreprocess(x4, c("1","2","3","4","5","6"))
    x4_vector_expected <- c(3,2,4,6,5,2,3,1,3,4,6)
    
    expect_equal(x1_preprocessing, x1_vector_expected)
    expect_equal(x2_preprocessing, x2_vector_expected)
    expect_equal(x3_preprocessing, x3_vector_expected)
    expect_equal(x4_preprocessing, x4_vector_expected)
})
