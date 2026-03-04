test_that("BART print method", {
  skip_on_cran()

  # Generate simulated data
  set.seed(42)
  n <- 100
  p <- 5
  X <- matrix(runif(n * p), ncol = p)
  f_X <- 5 * X[, 1]
  rfx_group_ids <- sample(1:4, size = n, replace = TRUE)
  rfx_basis <- cbind(rep(1, n), runif(n))
  y <- f_X + rnorm(n)
  test_inds <- sort(sample(1:n, 20))
  train_inds <- setdiff(1:n, test_inds)
  X_train <- X[train_inds, ]
  X_test <- X[test_inds, ]
  y_train <- y[train_inds]
  y_test <- y[test_inds]
  rfx_group_ids_train <- rfx_group_ids[train_inds]
  rfx_group_ids_test <- rfx_group_ids[test_inds]
  rfx_basis_train <- rfx_basis[train_inds, ]
  rfx_basis_test <- rfx_basis[test_inds, ]

  # --- 1 model term: mean forest only (no global variance, no leaf scale) ---
  bart_model_1 <- bart(
    X_train = X_train, y_train = y_train, X_test = X_test,
    num_gfr = 0, num_burnin = 10, num_mcmc = 10,
    general_params = list(sample_sigma2_global = FALSE),
    mean_forest_params = list(sample_sigma2_leaf = FALSE)
  )
  out_1 <- capture.output(result_1 <- print(bart_model_1))
  # Return value is the model object
  expect_identical(result_1, bart_model_1)
  # Key strings in output
  expect_true(any(grepl("stochtree::bart()", out_1, fixed = TRUE)))
  expect_true(any(grepl("mean forest", out_1, fixed = TRUE)))
  expect_true(any(grepl("constant leaf prior", out_1, fixed = TRUE)))
  expect_true(any(grepl("Outcome was standardized", out_1, fixed = TRUE)))
  expect_true(any(grepl("1 chain of", out_1, fixed = TRUE)))
  expect_true(any(grepl("retaining every iteration", out_1, fixed = TRUE)))

  # --- 2 model terms: mean forest + global error variance ---
  bart_model_2 <- bart(
    X_train = X_train, y_train = y_train, X_test = X_test,
    num_gfr = 0, num_burnin = 10, num_mcmc = 10,
    general_params = list(sample_sigma2_global = TRUE),
    mean_forest_params = list(sample_sigma2_leaf = FALSE)
  )
  out_2 <- capture.output(print(bart_model_2))
  expect_true(any(grepl("mean forest and global error variance model", out_2, fixed = TRUE)))

  # --- >2 model terms: mean forest + global error variance + leaf scale (Oxford comma) ---
  bart_model_3 <- bart(
    X_train = X_train, y_train = y_train, X_test = X_test,
    num_gfr = 0, num_burnin = 10, num_mcmc = 10,
    general_params = list(sample_sigma2_global = TRUE),
    mean_forest_params = list(sample_sigma2_leaf = TRUE)
  )
  out_3 <- capture.output(print(bart_model_3))
  expect_true(any(grepl(", and mean forest leaf scale model", out_3, fixed = TRUE)))

  # --- Leaf regression: basis supplied, exercises leaf_regression branch ---
  basis_train <- matrix(runif(length(y_train)), ncol = 1)
  basis_test <- matrix(runif(length(y_test)), ncol = 1)
  bart_model_lr <- bart(
    X_train = X_train, y_train = y_train, leaf_basis_train = basis_train,
    X_test = X_test, leaf_basis_test = basis_test,
    num_gfr = 0, num_burnin = 10, num_mcmc = 10,
    general_params = list(sample_sigma2_global = FALSE),
    mean_forest_params = list(sample_sigma2_leaf = FALSE)
  )
  out_lr <- capture.output(print(bart_model_lr))
  expect_true(any(grepl("leaf regression prior", out_lr, fixed = TRUE)))

  # --- Intercept-only random effects ---
  bart_model_rfx_1 <- bart(
    X_train = X_train, y_train = y_train, X_test = X_test,
    rfx_group_ids_train = rfx_group_ids_train,
    rfx_group_ids_test = rfx_group_ids_test,
    num_gfr = 0, num_burnin = 10, num_mcmc = 10,
    general_params = list(sample_sigma2_global = FALSE),
    mean_forest_params = list(sample_sigma2_leaf = FALSE),
    random_effects_params = list(model_spec = "intercept_only")
  )
  out_rfx_1 <- capture.output(print(bart_model_rfx_1))
  expect_true(any(grepl("additive random effects", out_rfx_1, fixed = TRUE)))
  expect_true(any(grepl("intercept-only", out_rfx_1, fixed = TRUE)))

  # --- Custom (multi-component) random effects ---
  bart_model_rfx_2 <- bart(
    X_train = X_train, y_train = y_train, X_test = X_test,
    rfx_group_ids_train = rfx_group_ids_train,
    rfx_group_ids_test = rfx_group_ids_test,
    rfx_basis_train = rfx_basis_train,
    rfx_basis_test = rfx_basis_test,
    num_gfr = 0, num_burnin = 10, num_mcmc = 10,
    general_params = list(sample_sigma2_global = FALSE),
    mean_forest_params = list(sample_sigma2_leaf = FALSE),
    random_effects_params = list(model_spec = "custom")
  )
  out_rfx_2 <- capture.output(print(bart_model_rfx_2))
  expect_true(any(grepl("additive random effects", out_rfx_2, fixed = TRUE)))
  expect_true(any(grepl("user-supplied basis", out_rfx_2, fixed = TRUE)))

  # --- GFR count is printed correctly ---
  bart_model_gfr <- bart(
    X_train = X_train, y_train = y_train, X_test = X_test,
    num_gfr = 10, num_burnin = 0, num_mcmc = 10,
    general_params = list(sample_sigma2_global = FALSE),
    mean_forest_params = list(sample_sigma2_leaf = FALSE)
  )
  out_gfr <- capture.output(print(bart_model_gfr))
  expect_true(any(grepl("10 GFR iterations", out_gfr, fixed = TRUE)))

  # --- standardize = FALSE: no "standardized" line ---
  bart_model_nostd <- bart(
    X_train = X_train, y_train = y_train, X_test = X_test,
    num_gfr = 0, num_burnin = 10, num_mcmc = 10,
    general_params = list(standardize = FALSE, sample_sigma2_global = FALSE),
    mean_forest_params = list(sample_sigma2_leaf = FALSE)
  )
  out_nostd <- capture.output(print(bart_model_nostd))
  expect_false(any(grepl("standardized", out_nostd, fixed = TRUE)))

  # --- Multiple chains + thinning ---
  bart_model_mc <- bart(
    X_train = X_train, y_train = y_train, X_test = X_test,
    num_gfr = 0, num_burnin = 10, num_mcmc = 10,
    general_params = list(num_chains = 2, keep_every = 2, sample_sigma2_global = FALSE),
    mean_forest_params = list(sample_sigma2_leaf = FALSE)
  )
  out_mc <- capture.output(print(bart_model_mc))
  expect_true(any(grepl("2 chains of", out_mc, fixed = TRUE)))
  expect_true(any(grepl("thinning", out_mc, fixed = TRUE)))
})

test_that("BART summary method", {
  skip_on_cran()

  # Generate simulated data
  set.seed(42)
  n <- 100
  p <- 5
  X <- matrix(runif(n * p), ncol = p)
  y <- 5 * X[, 1] + rnorm(n)
  rfx_group_ids <- sample(1:4, size = n, replace = TRUE)
  rfx_basis <- cbind(rep(1, n), runif(n))
  test_inds <- sort(sample(1:n, 20))
  train_inds <- setdiff(1:n, test_inds)
  X_train <- X[train_inds, ]
  X_test <- X[test_inds, ]
  y_train <- y[train_inds]
  y_test <- y[test_inds]
  rfx_group_ids_train <- rfx_group_ids[train_inds]
  rfx_group_ids_test <- rfx_group_ids[test_inds]
  rfx_basis_train <- rfx_basis[train_inds, ]
  rfx_basis_test <- rfx_basis[test_inds, ]

  # With sigma2_global, sigma2_leaf, and a test set
  bart_model <- bart(
    X_train = X_train, y_train = y_train, X_test = X_test,
    num_gfr = 0, num_burnin = 10, num_mcmc = 10,
    general_params = list(sample_sigma2_global = TRUE),
    mean_forest_params = list(sample_sigma2_leaf = TRUE)
  )
  out <- capture.output(result <- summary(bart_model))
  # Return value is the model object
  expect_identical(result, bart_model)
  # Sampled quantities are summarized
  expect_true(any(grepl("sigma^2", out, fixed = TRUE)))
  expect_true(any(grepl("leaf scale", out, fixed = TRUE)))
  # Both in-sample and test-set prediction summaries appear
  expect_true(any(grepl("in-sample", out, fixed = TRUE)))
  expect_true(any(grepl("test-set", out, fixed = TRUE)))

  # Without a test set: no test-set summary line
  bart_model_notestset <- bart(
    X_train = X_train, y_train = y_train,
    num_gfr = 0, num_burnin = 10, num_mcmc = 10,
    general_params = list(sample_sigma2_global = FALSE),
    mean_forest_params = list(sample_sigma2_leaf = FALSE)
  )
  out_notestset <- capture.output(summary(bart_model_notestset))
  expect_false(any(grepl("test-set", out_notestset, fixed = TRUE)))

  # Intercept-only RFX: single-component random effects summary
  bart_model_rfx_1 <- bart(
    X_train = X_train, y_train = y_train, X_test = X_test,
    rfx_group_ids_train = rfx_group_ids_train,
    rfx_group_ids_test = rfx_group_ids_test,
    num_gfr = 0, num_burnin = 10, num_mcmc = 10,
    general_params = list(sample_sigma2_global = FALSE),
    mean_forest_params = list(sample_sigma2_leaf = FALSE),
    random_effects_params = list(model_spec = "intercept_only")
  )
  out_rfx_1 <- capture.output(summary(bart_model_rfx_1))
  expect_true(any(grepl("Random effects", out_rfx_1, fixed = TRUE)))
  expect_true(any(grepl("Random effects overall mean", out_rfx_1, fixed = TRUE)))

  # Custom (multi-component) RFX: multi-component random effects summary
  bart_model_rfx_2 <- bart(
    X_train = X_train, y_train = y_train, X_test = X_test,
    rfx_group_ids_train = rfx_group_ids_train,
    rfx_group_ids_test = rfx_group_ids_test,
    rfx_basis_train = rfx_basis_train,
    rfx_basis_test = rfx_basis_test,
    num_gfr = 0, num_burnin = 10, num_mcmc = 10,
    general_params = list(sample_sigma2_global = FALSE),
    mean_forest_params = list(sample_sigma2_leaf = FALSE),
    random_effects_params = list(model_spec = "custom")
  )
  out_rfx_2 <- capture.output(summary(bart_model_rfx_2))
  expect_true(any(grepl("Random effects", out_rfx_2, fixed = TRUE)))
  expect_true(any(grepl("Variance component means", out_rfx_2, fixed = TRUE)))
})

test_that("BCF print method", {
  skip_on_cran()

  # Generate simulated data
  set.seed(42)
  n <- 100
  p <- 5
  X <- matrix(runif(n * p), ncol = p)
  pi_X <- 0.2 + 0.6 * X[, 1]
  Z <- rbinom(n, 1, pi_X)
  mu_X <- 5 * X[, 1]
  tau_X <- 2 * X[, 2]
  y <- mu_X + tau_X * Z + rnorm(n)
  test_inds <- sort(sample(1:n, 20))
  train_inds <- setdiff(1:n, test_inds)
  X_train <- X[train_inds, ]; X_test <- X[test_inds, ]
  Z_train <- Z[train_inds]; Z_test <- Z[test_inds]
  pi_train <- pi_X[train_inds]; pi_test <- pi_X[test_inds]
  y_train <- y[train_inds]; y_test <- y[test_inds]

  # --- User-provided propensity, binary treatment, adaptive coding (defaults) ---
  bcf_model <- bcf(
    X_train = X_train, y_train = y_train, Z_train = Z_train,
    propensity_train = pi_train,
    X_test = X_test, Z_test = Z_test, propensity_test = pi_test,
    num_gfr = 0, num_burnin = 10, num_mcmc = 10
  )
  out <- capture.output(result <- print(bcf_model))
  # Return value is the model object
  expect_identical(result, bcf_model)
  # Key strings in output
  expect_true(any(grepl("stochtree::bcf()", out, fixed = TRUE)))
  expect_true(any(grepl("prognostic forest", out, fixed = TRUE)))
  expect_true(any(grepl("treatment effect forest", out, fixed = TRUE)))
  expect_true(any(grepl("User-provided propensity scores", out, fixed = TRUE)))
  expect_true(any(grepl("adaptive coding", out, fixed = TRUE)))
  expect_true(any(grepl("1 chain of", out, fixed = TRUE)))
  expect_true(any(grepl("retaining every iteration", out, fixed = TRUE)))

  # --- Binary treatment, adaptive coding disabled ---
  bcf_model_noac <- bcf(
    X_train = X_train, y_train = y_train, Z_train = Z_train,
    propensity_train = pi_train,
    X_test = X_test, Z_test = Z_test, propensity_test = pi_test,
    num_gfr = 0, num_burnin = 10, num_mcmc = 10,
    general_params = list(adaptive_coding = FALSE)
  )
  out_noac <- capture.output(print(bcf_model_noac))
  expect_true(any(grepl("default coding", out_noac, fixed = TRUE)))

  # --- Propensity excluded from both forests ---
  bcf_model_noprop <- bcf(
    X_train = X_train, y_train = y_train, Z_train = Z_train,
    propensity_train = pi_train,
    X_test = X_test, Z_test = Z_test, propensity_test = pi_test,
    num_gfr = 0, num_burnin = 10, num_mcmc = 10,
    general_params = list(propensity_covariate = "none")
  )
  out_noprop <- capture.output(print(bcf_model_noprop))
  expect_true(any(grepl("not used in either forest", out_noprop, fixed = TRUE)))

  # --- Continuous (non-binary) treatment ---
  Z_cont_train <- rnorm(length(train_inds))
  Z_cont_test <- rnorm(length(test_inds))
  y_cont_train <- mu_X[train_inds] + 2 * Z_cont_train + rnorm(length(train_inds))
  bcf_model_cont <- bcf(
    X_train = X_train, y_train = y_cont_train, Z_train = Z_cont_train,
    X_test = X_test, Z_test = Z_cont_test,
    num_gfr = 0, num_burnin = 10, num_mcmc = 10,
    general_params = list(propensity_covariate = "none")
  )
  out_cont <- capture.output(print(bcf_model_cont))
  expect_true(any(grepl("univariate but not binary", out_cont, fixed = TRUE)))

  # --- Multiple chains + thinning ---
  bcf_model_mc <- bcf(
    X_train = X_train, y_train = y_train, Z_train = Z_train,
    propensity_train = pi_train,
    X_test = X_test, Z_test = Z_test, propensity_test = pi_test,
    num_gfr = 0, num_burnin = 10, num_mcmc = 10,
    general_params = list(num_chains = 2, keep_every = 2)
  )
  out_mc <- capture.output(print(bcf_model_mc))
  expect_true(any(grepl("2 chains of", out_mc, fixed = TRUE)))
  expect_true(any(grepl("thinning", out_mc, fixed = TRUE)))
})

test_that("BCF summary method", {
  skip_on_cran()

  # Generate simulated data
  set.seed(42)
  n <- 100
  p <- 5
  X <- matrix(runif(n * p), ncol = p)
  pi_X <- 0.2 + 0.6 * X[, 1]
  Z <- rbinom(n, 1, pi_X)
  mu_X <- 5 * X[, 1]
  tau_X <- 2 * X[, 2]
  y <- mu_X + tau_X * Z + rnorm(n)
  test_inds <- sort(sample(1:n, 20))
  train_inds <- setdiff(1:n, test_inds)
  X_train <- X[train_inds, ]; X_test <- X[test_inds, ]
  Z_train <- Z[train_inds]; Z_test <- Z[test_inds]
  pi_train <- pi_X[train_inds]; pi_test <- pi_X[test_inds]
  y_train <- y[train_inds]; y_test <- y[test_inds]

  # With sigma2_global, both leaf scales, adaptive coding, and a test set
  bcf_model <- bcf(
    X_train = X_train, y_train = y_train, Z_train = Z_train,
    propensity_train = pi_train,
    X_test = X_test, Z_test = Z_test, propensity_test = pi_test,
    num_gfr = 0, num_burnin = 10, num_mcmc = 10,
    general_params = list(sample_sigma2_global = TRUE),
    prognostic_forest_params = list(sample_sigma2_leaf = TRUE),
    treatment_effect_forest_params = list(sample_sigma2_leaf = TRUE)
  )
  out <- capture.output(result <- summary(bcf_model))
  # Return value is the model object
  expect_identical(result, bcf_model)
  # Sampled quantities are summarized
  expect_true(any(grepl("sigma^2", out, fixed = TRUE)))
  expect_true(any(grepl("prognostic forest leaf scale", out, fixed = TRUE)))
  expect_true(any(grepl("treatment effect forest leaf scale", out, fixed = TRUE)))
  expect_true(any(grepl("adaptive coding parameters", out, fixed = TRUE)))
  # Both in-sample and test-set prediction summaries appear
  expect_true(any(grepl("in-sample", out, fixed = TRUE)))
  expect_true(any(grepl("test-set", out, fixed = TRUE)))
  # CATE summaries appear
  expect_true(any(grepl("CATEs", out, fixed = TRUE)))

  # Without a test set: no test-set summary lines
  bcf_model_notestset <- bcf(
    X_train = X_train, y_train = y_train, Z_train = Z_train,
    propensity_train = pi_train,
    num_gfr = 0, num_burnin = 10, num_mcmc = 10
  )
  out_notestset <- capture.output(summary(bcf_model_notestset))
  expect_false(any(grepl("test-set", out_notestset, fixed = TRUE)))
})
