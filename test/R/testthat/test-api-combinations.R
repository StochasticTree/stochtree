run_bart_factorial <- function(
  bart_data_train,
  bart_data_test,
  leaf_reg = "none",
  variance_forest = FALSE,
  random_effects = "none",
  sampling_global_error_scale = FALSE,
  sampling_leaf_scale = FALSE,
  outcome_type = "continuous",
  num_chains = 1
) {
  # Unpack BART training data
  y <- bart_data_train[["y"]]
  X <- bart_data_train[["X"]]
  if (leaf_reg != "none") {
    leaf_basis <- bart_data_train[["leaf_basis"]]
  } else {
    leaf_basis <- NULL
  }
  if (random_effects != "none") {
    rfx_group_ids <- bart_data_train[["rfx_group_ids"]]
  } else {
    rfx_group_ids <- NULL
  }
  if (random_effects == "custom") {
    rfx_basis <- bart_data_train[["rfx_basis"]]
  } else {
    rfx_basis <- NULL
  }

  # Set BART model parameters
  general_params <- list(
    num_chains = num_chains,
    sample_sigma2_global = sampling_global_error_scale,
    probit_outcome_model = outcome_type == "binary"
  )
  mean_forest_params <- list(
    sample_sigma2_leaf = sampling_leaf_scale
  )
  variance_forest_params <- list(
    num_trees = ifelse(variance_forest, 20, 0)
  )
  rfx_params <- list(
    model_spec = ifelse(random_effects == "custom", "custom", random_effects)
  )

  # Sample BART model
  bart_model <- stochtree::bart(
    X_train = X,
    y_train = y,
    leaf_basis_train = leaf_basis,
    rfx_group_ids_train = rfx_group_ids,
    rfx_basis_train = rfx_basis,
    general_params = general_params,
    mean_forest_params = mean_forest_params,
    variance_forest_params = variance_forest_params,
    random_effects_params = rfx_params
  )

  # Unpack test set data
  y_test <- bart_data_test[["y"]]
  X_test <- bart_data_test[["X"]]
  if (leaf_reg != "none") {
    leaf_basis_test <- bart_data_test[["leaf_basis"]]
  } else {
    leaf_basis_test <- NULL
  }
  if (random_effects != "none") {
    rfx_group_ids_test <- bart_data_test[["rfx_group_ids"]]
  } else {
    rfx_group_ids_test <- NULL
  }
  if (random_effects == "custom") {
    rfx_basis_test <- bart_data_test[["rfx_basis"]]
  } else {
    rfx_basis_test <- NULL
  }

  # Predict on test set
  mean_preds <- predict(
    bart_model,
    X = X_test,
    leaf_basis = leaf_basis_test,
    rfx_group_ids = rfx_group_ids_test,
    rfx_basis = rfx_basis_test,
    type = "mean",
    terms = "all",
    scale = ifelse(outcome_type == "binary", "probability", "linear")
  )
  posterior_preds <- predict(
    bart_model,
    X = X_test,
    leaf_basis = leaf_basis_test,
    rfx_group_ids = rfx_group_ids_test,
    rfx_basis = rfx_basis_test,
    type = "posterior",
    terms = "all",
    scale = ifelse(outcome_type == "binary", "probability", "linear")
  )

  # Compute intervals
  posterior_interval <- compute_bart_posterior_interval(
    bart_model,
    terms = "all",
    level = 0.95,
    scale = ifelse(outcome_type == "binary", "probability", "linear"),
    X = X_test,
    leaf_basis = leaf_basis_test,
    rfx_group_ids = rfx_group_ids_test,
    rfx_basis = rfx_basis_test
  )

  # Sample posterior predictive
  posterior_predictive_draws <- sample_bart_posterior_predictive(
    bart_model,
    X = X_test,
    leaf_basis = leaf_basis_test,
    rfx_group_ids = rfx_group_ids_test,
    rfx_basis = rfx_basis_test,
    num_draws_per_sample = 5
  )
}

test_that("Quick check of interactions between components of BART functionality", {
  skip_on_cran()

  # Overall, we have seven components of a BART sampler which can be on / off or set to different levels:
  # 1. Leaf regression: none, univariate, multivariate
  # 2. Variance forest: no, yes
  # 3. Random effects: no, custom basis, `intercept_only`
  # 4. Sampling global error scale: no, yes
  # 5. Sampling leaf scale on mean forest: no, yes (only available for constant leaf or univariate leaf regression)
  # 6. Outcome type: continuous (identity link), binary (probit link)
  # 7. Number of chains: 1, >1
  #
  # For each of the possible models this implies,
  # we'd like to be sure that stochtree functions that operate on BART models
  # will run without error. Since there are so many possible models implied by the
  # options above, this test is designed to be quick (small sample size, low dimensional data)
  # and we are only interested in ensuring no errors are triggered.

  # Generate data with random effects
  n <- 50
  p <- 3
  num_basis <- 2
  num_rfx_groups <- 3
  num_rfx_basis <- 2
  X <- matrix(runif(n * p), ncol = p)
  leaf_basis <- matrix(runif(n * num_basis), ncol = num_basis)
  leaf_coefs <- runif(num_basis)
  group_ids <- sample(1:num_rfx_groups, n, replace = T)
  rfx_basis <- matrix(runif(n * num_rfx_basis), ncol = num_rfx_basis)
  rfx_coefs <- matrix(
    runif(num_rfx_groups * num_rfx_basis),
    ncol = num_rfx_basis
  )
  mean_term <- sin(X[, 1]) * rowSums(leaf_basis * leaf_coefs)
  rfx_term <- rowSums(rfx_coefs[group_ids, ] * rfx_basis)
  E_y <- mean_term + rfx_term
  E_y <- E_y - mean(E_y)
  epsilon <- rnorm(n, 0, 1)
  y_continuous <- E_y + epsilon
  y_binary <- 1 * (y_continuous > 0)

  # Split into test and train sets
  test_set_pct <- 0.5
  n_test <- round(test_set_pct * n)
  n_train <- n - n_test
  test_inds <- sort(sample(1:n, n_test, replace = FALSE))
  train_inds <- (1:n)[!((1:n) %in% test_inds)]
  X_test <- X[test_inds, ]
  X_train <- X[train_inds, ]
  leaf_basis_test <- leaf_basis[test_inds, ]
  leaf_basis_train <- leaf_basis[train_inds, ]
  rfx_basis_test <- rfx_basis[test_inds, ]
  rfx_basis_train <- rfx_basis[train_inds, ]
  group_ids_test <- group_ids[test_inds]
  group_ids_train <- group_ids[train_inds]
  y_continuous_test <- y_continuous[test_inds]
  y_continuous_train <- y_continuous[train_inds]
  y_binary_test <- y_binary[test_inds]
  y_binary_train <- y_binary[train_inds]

  # Run the power set of models
  leaf_reg_options <- c("none", "univariate", "multivariate")
  variance_forest_options <- c(FALSE, TRUE)
  random_effects_options <- c("none", "custom", "intercept_only")
  sampling_global_error_scale_options <- c(FALSE, TRUE)
  sampling_leaf_scale_options <- c(FALSE, TRUE)
  outcome_type_options <- c("continuous", "binary")
  num_chains_options <- c(1, 3)
  model_options_df <- expand.grid(
    leaf_reg = leaf_reg_options,
    variance_forest = variance_forest_options,
    random_effects = random_effects_options,
    sampling_global_error_scale = sampling_global_error_scale_options,
    sampling_leaf_scale = sampling_leaf_scale_options,
    outcome_type = outcome_type_options,
    num_chains = num_chains_options,
    stringsAsFactors = FALSE
  )
  for (i in 1:nrow(model_options_df)) {
    error_cond <- (model_options_df$variance_forest[i]) &&
      (model_options_df$outcome_type[i] == "binary")
    warning_cond_1 <- (model_options_df$sampling_leaf_scale[i]) &&
      (model_options_df$leaf_reg[i] == "multivariate")
    warning_message_1 <- "Sampling leaf scale not yet supported for multivariate leaf models, so the leaf scale parameter will not be sampled in this model."
    warning_cond_2 <- (model_options_df$sampling_global_error_scale[i]) &&
      (model_options_df$outcome_type[i] == "binary")
    warning_message_2 <- "Global error variance will not be sampled with a probit link as it is fixed at 1"
    warning_cond <- warning_cond_1 || warning_cond_2
    if (error_cond && warning_cond) {
      if (warning_cond_1 && warning_cond_2) {
        test_fun <- function(x) {
          expect_error(
            expect_warning(
              expect_warning(x, warning_message_1),
              warning_message_2
            )
          )
        }
      } else if (warning_cond_1) {
        test_fun <- function(x) {
          expect_error(
            expect_warning(x, warning_message_1)
          )
        }
      } else {
        test_fun <- function(x) {
          expect_error(
            expect_warning(x, warning_message_2)
          )
        }
      }
    } else if (error_cond && !warning_cond) {
      test_fun <- expect_error
    } else if (!error_cond && warning_cond) {
      if (warning_cond_1 && warning_cond_2) {
        test_fun <- function(x) {
          expect_warning(
            expect_warning(x, warning_message_1),
            warning_message_2
          )
        }
      } else if (warning_cond_1) {
        test_fun <- function(x) {
          expect_warning(x, warning_message_1)
        }
      } else {
        test_fun <- function(x) {
          expect_warning(x, warning_message_2)
        }
      }
    } else {
      test_fun <- expect_no_error
    }
    test_fun({
      bart_data_train <- list(X = X_train)
      bart_data_test <- list(X = X_test)
      if (model_options_df$outcome_type[i] == "continuous") {
        bart_data_train[["y"]] <- y_continuous_train
        bart_data_test[["y"]] <- y_continuous_test
      } else {
        bart_data_train[["y"]] <- y_binary_train
        bart_data_test[["y"]] <- y_binary_test
      }
      if (model_options_df$leaf_reg[i] != "none") {
        if (model_options_df$leaf_reg[i] == "univariate") {
          bart_data_train[["leaf_basis"]] <- leaf_basis_train[, 1, drop = FALSE]
          bart_data_test[["leaf_basis"]] <- leaf_basis_test[, 1, drop = FALSE]
        } else {
          bart_data_train[["leaf_basis"]] <- leaf_basis_train
          bart_data_test[["leaf_basis"]] <- leaf_basis_test
        }
      } else {
        bart_data_train[["leaf_basis"]] <- NULL
        bart_data_test[["leaf_basis"]] <- NULL
      }
      if (model_options_df$random_effects[i] != "none") {
        bart_data_train[["rfx_group_ids"]] <- group_ids_train
        bart_data_test[["rfx_group_ids"]] <- group_ids_test
      } else {
        bart_data_train[["rfx_group_ids"]] <- NULL
        bart_data_test[["rfx_group_ids"]] <- NULL
      }
      if (model_options_df$random_effects[i] == "custom") {
        bart_data_train[["rfx_basis"]] <- rfx_basis_train
        bart_data_test[["rfx_basis"]] <- rfx_basis_test
      } else {
        bart_data_train[["rfx_basis"]] <- NULL
        bart_data_test[["rfx_basis"]] <- NULL
      }
      run_bart_factorial(
        bart_data_train = bart_data_train,
        bart_data_test = bart_data_test,
        leaf_reg = model_options_df$leaf_reg[i],
        variance_forest = model_options_df$variance_forest[i],
        random_effects = model_options_df$random_effects[i],
        sampling_global_error_scale = model_options_df$sampling_global_error_scale[
          i
        ],
        sampling_leaf_scale = model_options_df$sampling_leaf_scale[
          i
        ],
        outcome_type = model_options_df$outcome_type[i],
        num_chains = model_options_df$num_chains[i]
      )
    })
  }
})
