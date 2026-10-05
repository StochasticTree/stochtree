# Check that the MCMC tree sampler targets the exact posterior on a tiny problem.
#
# One tree, one covariate taking values 1, 2, 3, 4, outcomes all zero, alpha = 0.6,
# beta = 2, min_samples_leaf = 1, and global and leaf variances fixed at 1. With so
# few observations every tree can be enumerated, so the posterior over the number
# of leaves is known exactly, and a long chain must reproduce it.
#
# The sampler's target over trees is
#
#   p(T | y)  ∝  prod_internal [ pg(d) / (width of the node's covariate range) ]
#              * prod_leaves  [ (1 - pg(d)) * (1 + n_leaf)^(-1/2) ],  pg(d) = alpha (1+d)^-beta
#
# where 1/width is the uniform cutpoint density integrated over a unit gap between
# adjacent integers, and (1 + n)^(-1/2) is a leaf's marginal likelihood for zero
# outcomes, up to a constant shared by every tree.
#
# The covariate is fed in ascending and descending order: the posterior is the same,
# but the order changes which observation each node scans first when computing its
# split range, which is how the GH #425 extrema bug shows up.

exact_x_values <- c(1, 2, 3, 4)
exact_alpha <- 0.6
exact_beta <- 2

# Posterior probability of 1..4 leaves, by enumerating every tree
exact_leaf_distribution <- function() {
  pg <- function(depth) exact_alpha * (1 + depth)^(-exact_beta)
  n <- length(exact_x_values)
  # Unnormalized posterior mass by number of leaves (element k = k leaves)
  # for the subtree on points lo..hi
  node <- function(lo, hi, depth) {
    out <- numeric(n)
    out[1] <- (1 - pg(depth)) * (1 + hi - lo + 1)^(-0.5)
    width <- exact_x_values[hi] - exact_x_values[lo]
    if (hi > lo) {
      for (cut in lo:(hi - 1)) {
        weight <- pg(depth) *
          (exact_x_values[cut + 1] - exact_x_values[cut]) /
          width
        left <- node(lo, cut, depth + 1)
        right <- node(cut + 1, hi, depth + 1)
        for (kl in which(left > 0)) {
          for (kr in which(right > 0)) {
            out[kl + kr] <- out[kl + kr] + weight * left[kl] * right[kr]
          }
        }
      }
    }
    out
  }
  mass <- node(1, n, 0)
  mass / sum(mass)
}

test_that("Exact leaf-count posterior enumeration", {
  # Hand-checkable anchor for the enumeration itself
  expect_equal(
    exact_leaf_distribution(),
    c(0.50183, 0.42176, 0.07196, 0.00446),
    tolerance = 1e-5
  )
})

test_that("MCMC sampler matches the exact posterior", {
  skip_on_cran()

  exact <- exact_leaf_distribution()
  for (descending in c(FALSE, TRUE)) {
    x <- if (descending) rev(exact_x_values) else exact_x_values
    model <- bart(
      X_train = matrix(x, ncol = 1),
      y_train = rep(0, length(x)),
      num_gfr = 0,
      num_burnin = 5000,
      num_mcmc = 100000,
      general_params = list(
        standardize = FALSE,
        sample_sigma2_global = FALSE,
        sigma2_global_init = 1,
        random_seed = 20260907
      ),
      mean_forest_params = list(
        num_trees = 1,
        alpha = exact_alpha,
        beta = exact_beta,
        min_samples_leaf = 1,
        sample_sigma2_leaf = FALSE,
        sigma2_leaf_init = 1
      )
    )
    forests <- model$mean_forests
    leaves <- sapply(
      seq_len(forests$num_samples()) - 1,
      forests$num_forest_leaves
    )
    observed <- sapply(seq_along(exact), function(k) mean(leaves == k))

    # With 100k draws the Monte Carlo standard error of each share is under 0.2
    # percentage points; the pre-fix sampler missed by 8-10 points (or never split).
    expect_lt(
      max(abs(observed - exact)),
      0.01,
      label = sprintf(
        "max leaf-share error (%s order)",
        if (descending) "descending" else "ascending"
      )
    )
  }
})
