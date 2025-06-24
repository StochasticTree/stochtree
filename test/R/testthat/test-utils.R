test_that("Array conversion", {
    skip_on_cran()
    
    # Test data
    scalar_1 <- 1.5
    scalar_2 <- -2.5
    scalar_3 <- 4
    array_1d_1 <- c(1.6, 3.4, 7.6, 8.7)
    array_1d_2 <- c(2.5, 3.1, 5.6)
    array_1d_3 <- c(2.5)
    array_2d_1 <- matrix(
        c(2.5,1.2,4.3,7.4,1.7,2.9,3.6,9.1,7.2,4.5,6.7,1.4), 
        nrow = 3, ncol = 4, byrow = T
    )
    array_2d_2 <- matrix(
        c(2.5,1.2,4.3,7.4,1.7,2.9,3.6,9.1), 
        nrow = 2, ncol = 4, byrow = T
    )
    array_square_1 <- matrix(
        c(2.5,1.2,1.7,2.9), 
        nrow = 2, ncol = 2, byrow = T
    )
    array_square_2 <- matrix(
        c(2.5,0.0,0.0,2.9), 
        nrow = 2, ncol = 2, byrow = T
    )
    array_square_3 <- matrix(
        c(2.5,0.0,0.0,0.0,2.9,0.0,0.0,0.0,5.6), 
        nrow = 2, ncol = 2, byrow = T
    )
    
    # Error cases
    expect_error(expand_dims_1d(array_1d_1, 5))
    expect_error(expand_dims_1d(array_1d_2, 4))
    expect_error(expand_dims_1d(array_1d_3, 3))
    expect_error(expand_dims_2d(array_2d_1, 2, 4))
    expect_error(expand_dims_2d(array_2d_2, 3, 4))
    expect_error(expand_dims_2d_diag(array_square_1, 4))
    expect_error(expand_dims_2d_diag(array_square_2, 3))
    expect_error(expand_dims_2d_diag(array_square_3, 2))
    
    # # Assertion
    # expect_equal(y_hat_orig, y_hat_reloaded)
})