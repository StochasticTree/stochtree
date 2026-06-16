# Unit tests for the JSON key primitives (rename/erase) used by schema migrations.

test_that("CppJson rename_field / erase_field (top-level)", {
  skip_on_cran()
  j <- createCppJson()
  j$add_scalar("old", 3.5)

  j$rename_field("old", "new")
  expect_false(j$contains("old"))
  expect_true(j$contains("new"))
  expect_equal(j$get_scalar("new"), 3.5)

  j$erase_field("new")
  expect_false(j$contains("new"))
})

test_that("CppJson rename_field / erase_field (subfolder)", {
  skip_on_cran()
  j <- createCppJson()
  j$add_scalar("old", 7.0, subfolder_name = "sub")

  j$rename_field("old", "new", subfolder_name = "sub")
  expect_false(j$contains("old", subfolder_name = "sub"))
  expect_true(j$contains("new", subfolder_name = "sub"))
  expect_equal(j$get_scalar("new", subfolder_name = "sub"), 7.0)

  j$erase_field("new", subfolder_name = "sub")
  expect_false(j$contains("new", subfolder_name = "sub"))
})

test_that("CppJson rename/erase of a missing field is a no-op", {
  skip_on_cran()
  j <- createCppJson()
  expect_silent(j$rename_field("nope", "whatever"))
  expect_silent(j$erase_field("nope"))
  expect_false(j$contains("whatever"))
})
