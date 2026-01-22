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
    c(2.5, 1.2, 4.3, 7.4, 1.7, 2.9, 3.6, 9.1, 7.2, 4.5, 6.7, 1.4),
    nrow = 3,
    ncol = 4,
    byrow = T
  )
  array_2d_2 <- matrix(
    c(2.5, 1.2, 4.3, 7.4, 1.7, 2.9, 3.6, 9.1),
    nrow = 2,
    ncol = 4,
    byrow = T
  )
  array_square_1 <- matrix(
    c(2.5, 1.2, 1.7, 2.9),
    nrow = 2,
    ncol = 2,
    byrow = T
  )
  array_square_2 <- matrix(
    c(2.5, 0.0, 0.0, 2.9),
    nrow = 2,
    ncol = 2,
    byrow = T
  )
  array_square_3 <- matrix(
    c(2.5, 0.0, 0.0, 0.0, 2.9, 0.0, 0.0, 0.0, 5.6),
    nrow = 3,
    ncol = 3,
    byrow = T
  )

  # Error cases
  expect_error(stochtree:::expand_dims_1d(array_1d_1, 5))
  expect_error(stochtree:::expand_dims_1d(array_1d_2, 4))
  expect_error(stochtree:::expand_dims_2d(array_2d_1, 2, 4))
  expect_error(stochtree:::expand_dims_2d(array_2d_2, 3, 4))
  expect_error(stochtree:::expand_dims_2d_diag(array_square_1, 4))
  expect_error(stochtree:::expand_dims_2d_diag(array_square_2, 3))
  expect_error(stochtree:::expand_dims_2d_diag(array_square_3, 2))

  # Working cases
  expect_equal(
    c(scalar_1, scalar_1, scalar_1),
    stochtree:::expand_dims_1d(scalar_1, 3)
  )
  expect_equal(
    c(scalar_2, scalar_2, scalar_2, scalar_2),
    stochtree:::expand_dims_1d(scalar_2, 4)
  )
  expect_equal(c(scalar_3, scalar_3), stochtree:::expand_dims_1d(scalar_3, 2))
  expect_equal(
    c(array_1d_3, array_1d_3, array_1d_3),
    stochtree:::expand_dims_1d(array_1d_3, 3)
  )

  output_exp <- matrix(rep(scalar_1, 6), nrow = 2, byrow = T)
  expect_equal(output_exp, stochtree:::expand_dims_2d(scalar_1, 2, 3))
  output_exp <- matrix(rep(scalar_2, 8), nrow = 2, byrow = T)
  expect_equal(output_exp, stochtree:::expand_dims_2d(scalar_2, 2, 4))
  output_exp <- matrix(rep(scalar_3, 6), nrow = 3, byrow = T)
  expect_equal(output_exp, stochtree:::expand_dims_2d(scalar_3, 3, 2))
  output_exp <- matrix(rep(array_1d_3, 6), nrow = 3, byrow = T)
  expect_equal(output_exp, stochtree:::expand_dims_2d(array_1d_3, 3, 2))
  output_exp <- unname(rbind(array_1d_1, array_1d_1))
  expect_equal(output_exp, stochtree:::expand_dims_2d(array_1d_1, 2, 4))
  output_exp <- unname(rbind(array_1d_2, array_1d_2, array_1d_2))
  expect_equal(output_exp, stochtree:::expand_dims_2d(array_1d_2, 3, 3))
  output_exp <- unname(cbind(array_1d_2, array_1d_2, array_1d_2, array_1d_2))
  expect_equal(output_exp, stochtree:::expand_dims_2d(array_1d_2, 3, 4))
  output_exp <- unname(cbind(array_1d_3, array_1d_3, array_1d_3, array_1d_3))
  expect_equal(output_exp, stochtree:::expand_dims_2d(array_1d_3, 1, 4))
  output_exp <- unname(rbind(array_1d_3, array_1d_3, array_1d_3, array_1d_3))
  expect_equal(output_exp, stochtree:::expand_dims_2d(array_1d_3, 4, 1))

  expect_equal(diag(scalar_1, 3), stochtree:::expand_dims_2d_diag(scalar_1, 3))
  expect_equal(diag(scalar_2, 2), stochtree:::expand_dims_2d_diag(scalar_2, 2))
  expect_equal(diag(scalar_3, 4), stochtree:::expand_dims_2d_diag(scalar_3, 4))
  expect_equal(
    diag(array_1d_3, 2),
    stochtree:::expand_dims_2d_diag(array_1d_3, 2)
  )
})
