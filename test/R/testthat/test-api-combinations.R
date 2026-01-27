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
    model_spec = ifelse(random_effects == "none", "custom", random_effects)
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

run_bcf_factorial <- function(
  bcf_data_train,
  bcf_data_test,
  treatment_type = "binary",
  variance_forest = FALSE,
  random_effects = "none",
  sampling_global_error_scale = FALSE,
  sampling_mu_leaf_scale = FALSE,
  sampling_tau_leaf_scale = FALSE,
  outcome_type = "continuous",
  num_chains = 1,
  adaptive_coding = TRUE,
  include_propensity = TRUE
) {
  # Unpack BART training data
  y <- bcf_data_train[["y"]]
  X <- bcf_data_train[["X"]]
  Z <- bcf_data_train[["Z"]]
  if (include_propensity) {
    propensity_train <- bcf_data_train[["propensity"]]
  } else {
    propensity_train <- NULL
  }
  if (random_effects != "none") {
    rfx_group_ids <- bcf_data_train[["rfx_group_ids"]]
  } else {
    rfx_group_ids <- NULL
  }
  if (random_effects == "custom") {
    rfx_basis <- bcf_data_train[["rfx_basis"]]
  } else {
    rfx_basis <- NULL
  }

  # Set BART model parameters
  general_params <- list(
    num_chains = num_chains,
    sample_sigma2_global = sampling_global_error_scale,
    probit_outcome_model = outcome_type == "binary",
    adaptive_coding = adaptive_coding
  )
  mu_forest_params <- list(
    sample_sigma2_leaf = sampling_mu_leaf_scale
  )
  tau_forest_params <- list(
    sample_sigma2_leaf = sampling_tau_leaf_scale
  )
  variance_forest_params <- list(
    num_trees = ifelse(variance_forest, 20, 0)
  )
  rfx_params <- list(
    model_spec = ifelse(random_effects == "none", "custom", random_effects)
  )

  # Sample BART model
  bcf_model <- stochtree::bcf(
    X_train = X,
    y_train = y,
    Z_train = Z,
    propensity_train = propensity_train,
    rfx_group_ids_train = rfx_group_ids,
    rfx_basis_train = rfx_basis,
    general_params = general_params,
    prognostic_forest_params = mu_forest_params,
    treatment_effect_forest_params = tau_forest_params,
    variance_forest_params = variance_forest_params,
    random_effects_params = rfx_params
  )

  # Unpack test set data
  y_test <- bcf_data_test[["y"]]
  X_test <- bcf_data_test[["X"]]
  Z_test <- bcf_data_test[["Z"]]
  if (include_propensity) {
    propensity_test <- bcf_data_test[["propensity"]]
  } else {
    propensity_test <- NULL
  }
  if (random_effects != "none") {
    rfx_group_ids_test <- bcf_data_test[["rfx_group_ids"]]
  } else {
    rfx_group_ids_test <- NULL
  }
  if (random_effects == "custom") {
    rfx_basis_test <- bcf_data_test[["rfx_basis"]]
  } else {
    rfx_basis_test <- NULL
  }

  # Predict on test set
  mean_preds <- predict(
    bcf_model,
    X = X_test,
    Z = Z_test,
    propensity = propensity_test,
    rfx_group_ids = rfx_group_ids_test,
    rfx_basis = rfx_basis_test,
    type = "mean",
    terms = "all",
    scale = ifelse(outcome_type == "binary", "probability", "linear")
  )
  posterior_preds <- predict(
    bcf_model,
    X = X_test,
    Z = Z_test,
    propensity = propensity_test,
    rfx_group_ids = rfx_group_ids_test,
    rfx_basis = rfx_basis_test,
    type = "posterior",
    terms = "all",
    scale = ifelse(outcome_type == "binary", "probability", "linear")
  )

  # Compute intervals
  posterior_interval <- compute_bcf_posterior_interval(
    bcf_model,
    terms = "all",
    level = 0.95,
    scale = ifelse(outcome_type == "binary", "probability", "linear"),
    X = X_test,
    Z = Z_test,
    propensity = propensity_test,
    rfx_group_ids = rfx_group_ids_test,
    rfx_basis = rfx_basis_test
  )

  # Sample posterior predictive
  posterior_predictive_draws <- sample_bcf_posterior_predictive(
    bcf_model,
    X = X_test,
    Z = Z_test,
    propensity = propensity_test,
    rfx_group_ids = rfx_group_ids_test,
    rfx_basis = rfx_basis_test,
    num_draws_per_sample = 5
  )
}

# Construct chained expectations without writing out every combination of function calls
construct_chained_expectation_bart <- function(
  error_cond,
  warning_cond_1,
  warning_cond_2,
  warning_cond_3
) {
  # Build the chain from innermost to outermost
  function_text <- "x"
  if (warning_cond_1) {
    function_text <- paste0(
      "warning_fun_1(",
      function_text,
      ")"
    )
  }
  if (warning_cond_2) {
    function_text <- paste0(
      "warning_fun_2(",
      function_text,
      ")"
    )
  }
  if (warning_cond_3) {
    function_text <- paste0(
      "warning_fun_3(",
      function_text,
      ")"
    )
  }
  if (error_cond) {
    function_text <- paste0(
      "expect_error(",
      function_text,
      ")"
    )
  }
  return(as.function(
    c(alist(x = ), parse(text = function_text)[[1]]),
    envir = parent.frame()
  ))
}

construct_chained_expectation_bcf <- function(
  error_cond,
  warning_cond_1,
  warning_cond_2,
  warning_cond_3,
  warning_cond_4,
  warning_cond_5,
  warning_cond_6
) {
  # Build the chain from innermost to outermost
  function_text <- "x"
  if (warning_cond_1) {
    function_text <- paste0(
      "warning_fun_1(",
      function_text,
      ")"
    )
  }
  if (warning_cond_2) {
    function_text <- paste0(
      "warning_fun_2(",
      function_text,
      ")"
    )
  }
  if (warning_cond_3) {
    function_text <- paste0(
      "warning_fun_3(",
      function_text,
      ")"
    )
  }
  if (warning_cond_4) {
    function_text <- paste0(
      "warning_fun_4(",
      function_text,
      ")"
    )
  }
  if (warning_cond_5) {
    function_text <- paste0(
      "warning_fun_5(",
      function_text,
      ")"
    )
  }
  if (warning_cond_6) {
    function_text <- paste0(
      "warning_fun_6(",
      function_text,
      ")"
    )
  }
  if (error_cond) {
    function_text <- paste0(
      "expect_error(",
      function_text,
      ")"
    )
  }
  return(as.function(
    c(alist(x = ), parse(text = function_text)[[1]]),
    envir = parent.frame()
  ))
}

test_that("Quick check of interactions between components of BART functionality", {
  skip_on_cran()
  # Code from: https://github.com/r-lib/testthat/blob/main/R/skip.R
  skip_if(
    isFALSE(as.logical(Sys.getenv("RUN_SLOW_TESTS", "false"))),
    "skipping slow tests"
  )

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
    # Determine which errors and warnings should be triggered
    error_cond <- (model_options_df$variance_forest[i]) &&
      (model_options_df$outcome_type[i] == "binary")
    warning_cond_1 <- (model_options_df$sampling_leaf_scale[i]) &&
      (model_options_df$leaf_reg[i] == "multivariate")
    warning_fun_1 <- function(x) {
      expect_warning(
        x,
        "Sampling leaf scale not yet supported for multivariate leaf models, so the leaf scale parameter will not be sampled in this model."
      )
    }
    warning_cond_2 <- (model_options_df$sampling_global_error_scale[i]) &&
      (model_options_df$outcome_type[i] == "binary")
    warning_fun_2 <- function(x) {
      expect_warning(
        x,
        "Global error variance will not be sampled with a probit link as it is fixed at 1"
      )
    }
    warning_cond_3 <- (model_options_df$sampling_global_error_scale[i]) &&
      (model_options_df$variance_forest[i])
    warning_fun_3 <- function(x) {
      expect_warning(
        x,
        "Global error variance will not be sampled with a heteroskedasticity"
      )
    }
    warning_cond <- warning_cond_1 || warning_cond_2 || warning_cond_3

    if (error_cond || warning_cond) {
      test_fun <- construct_chained_expectation_bart(
        error_cond = error_cond,
        warning_cond_1 = warning_cond_1,
        warning_cond_2 = warning_cond_2,
        warning_cond_3 = warning_cond_3
      )
    } else {
      test_fun <- expect_no_error
    }

    # Prepare test function arguments
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

    # Apply testthat expectation(s)
    test_fun({
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

test_that("Quick check of interactions between components of BCF functionality", {
  skip_on_cran()
  # Code from: https://github.com/r-lib/testthat/blob/main/R/skip.R
  skip_if(
    isFALSE(as.logical(Sys.getenv("RUN_SLOW_TESTS", "false"))),
    "skipping slow tests"
  )

  # Overall, we have nine components of a BCF sampler which can be on / off or set to different levels:
  # 1. treatment: binary, univariate continuous, multivariate
  # 2. Variance forest: no, yes
  # 3. Random effects: no, custom basis, `intercept_only`, `intercept_plus_treatment`
  # 4. Sampling global error scale: no, yes
  # 5. Sampling leaf scale on prognostic forest: no, yes
  # 6. Sampling leaf scale on treatment forest: no, yes (only available for univariate treatment)
  # 7. Outcome type: continuous (identity link), binary (probit link)
  # 8. Number of chains: 1, >1
  # 9. Adaptive coding: no, yes
  #
  # For each of the possible models this implies,
  # we'd like to be sure that stochtree functions that operate on BCF models
  # will run without error. Since there are so many possible models implied by the
  # options above, this test is designed to be quick (small sample size, low dimensional data)
  # and we are only interested in ensuring no errors are triggered.

  # Generate data with random effects
  n <- 50
  p <- 3
  num_rfx_groups <- 3
  num_rfx_basis <- 2
  X <- matrix(runif(n * p), ncol = p)
  binary_treatment <- rbinom(n, 1, 0.5)
  continuous_treatment <- runif(n, 0, 1)
  multivariate_treatment <- cbind(
    binary_treatment,
    continuous_treatment
  )
  group_ids <- sample(1:num_rfx_groups, n, replace = T)
  rfx_basis <- matrix(runif(n * num_rfx_basis), ncol = num_rfx_basis)
  rfx_coefs <- matrix(
    runif(num_rfx_groups * num_rfx_basis),
    ncol = num_rfx_basis
  )
  propensity <- runif(n)
  prognostic_term <- sin(X[, 1])
  binary_treatment_effect <- X[, 2]
  continuous_treatment_effect <- X[, 3]
  rfx_term <- rowSums(rfx_coefs[group_ids, ] * rfx_basis)
  E_y <- prognostic_term +
    binary_treatment_effect * binary_treatment +
    continuous_treatment_effect * continuous_treatment +
    rfx_term
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
  binary_treatment_test <- binary_treatment[test_inds]
  binary_treatment_train <- binary_treatment[train_inds]
  propensity_test <- propensity[test_inds]
  propensity_train <- propensity[train_inds]
  continuous_treatment_test <- continuous_treatment[test_inds]
  continuous_treatment_train <- continuous_treatment[train_inds]
  multivariate_treatment_test <- multivariate_treatment[test_inds, ]
  multivariate_treatment_train <- multivariate_treatment[train_inds, ]
  rfx_basis_test <- rfx_basis[test_inds, ]
  rfx_basis_train <- rfx_basis[train_inds, ]
  group_ids_test <- group_ids[test_inds]
  group_ids_train <- group_ids[train_inds]
  y_continuous_test <- y_continuous[test_inds]
  y_continuous_train <- y_continuous[train_inds]
  y_binary_test <- y_binary[test_inds]
  y_binary_train <- y_binary[train_inds]

  # Run the power set of models
  treatment_options <- c("binary", "univariate_continuous", "multivariate")
  variance_forest_options <- c(FALSE, TRUE)
  random_effects_options <- c(
    "none",
    "custom",
    "intercept_only",
    "intercept_plus_treatment"
  )
  sampling_global_error_scale_options <- c(FALSE, TRUE)
  sampling_mu_leaf_scale_options <- c(FALSE, TRUE)
  sampling_tau_leaf_scale_options <- c(FALSE, TRUE)
  outcome_type_options <- c("continuous", "binary")
  num_chains_options <- c(1, 3)
  adaptive_coding_options <- c(FALSE, TRUE)
  include_propensity_options <- c(FALSE, TRUE)
  model_options_df <- expand.grid(
    treatment_type = treatment_options,
    variance_forest = variance_forest_options,
    random_effects = random_effects_options,
    sampling_global_error_scale = sampling_global_error_scale_options,
    sampling_mu_leaf_scale = sampling_mu_leaf_scale_options,
    sampling_tau_leaf_scale = sampling_tau_leaf_scale_options,
    outcome_type = outcome_type_options,
    num_chains = num_chains_options,
    adaptive_coding = adaptive_coding_options,
    include_propensity = include_propensity_options,
    stringsAsFactors = FALSE
  )
  for (i in 1:nrow(model_options_df)) {
    # Determine which errors and warnings should be triggered
    error_cond <- (model_options_df$variance_forest[i]) &&
      (model_options_df$outcome_type[i] == "binary")
    warning_cond_1 <- (model_options_df$sampling_tau_leaf_scale[i]) &&
      (model_options_df$treatment_type[i] == "multivariate")
    warning_fun_1 <- function(x) {
      expect_warning(
        x,
        "Sampling leaf scale not yet supported for multivariate leaf models, so the leaf scale parameter will not be sampled for the treatment forest in this model.",
        fixed = TRUE
      )
    }
    warning_cond_2 <- (!model_options_df$include_propensity[i]) &&
      (model_options_df$treatment_type[i] == "multivariate")
    warning_fun_2 <- function(x) {
      expect_warning(
        x,
        "No propensities were provided for the multivariate treatment; an internal propensity model will not be fitted to the multivariate treatment and propensity_covariate will be set to 'none'",
        fixed = TRUE
      )
    }
    warning_cond_3 <- (model_options_df$adaptive_coding[i]) &&
      (model_options_df$treatment_type[i] != "binary")
    warning_fun_3 <- function(x) {
      expect_warning(
        x,
        "Adaptive coding is only compatible with binary (univariate) treatment and, as a result, will be ignored in sampling this model",
        fixed = TRUE
      )
    }
    warning_cond_4 <- (model_options_df$sampling_global_error_scale[i]) &&
      (model_options_df$outcome_type[i] == "binary")
    warning_fun_4 <- function(x) {
      expect_warning(
        x,
        "Global error variance will not be sampled with a probit link as it is fixed at 1",
        fixed = TRUE
      )
    }
    warning_cond_5 <- (model_options_df$sampling_global_error_scale[i]) &&
      (model_options_df$variance_forest[i])
    warning_fun_5 <- function(x) {
      expect_warning(
        x,
        "Global error variance will not be sampled with a heteroskedasticity",
        fixed = TRUE
      )
    }
    warning_cond_6 <- (model_options_df$treatment_type[i] == "multivariate") &&
      (model_options_df$random_effects[i] == "intercept_plus_treatment")
    warning_fun_6 <- function(x) {
      expect_warning(
        x,
        "Random effects `intercept_plus_treatment` specification is not currently implemented for multivariate treatments. This model will be fit under the `intercept_only` specification instead. Please provide a custom `rfx_basis_train` if you wish to have random slopes on multivariate treatment variables.",
        fixed = TRUE
      )
    }
    warning_cond <- (warning_cond_1 ||
      warning_cond_2 ||
      warning_cond_3 ||
      warning_cond_4 ||
      warning_cond_5 ||
      warning_cond_6)

    # Generate something like the below code but for all five warnings
    if (error_cond || warning_cond) {
      test_fun <- construct_chained_expectation_bcf(
        error_cond = error_cond,
        warning_cond_1 = warning_cond_1,
        warning_cond_2 = warning_cond_2,
        warning_cond_3 = warning_cond_3,
        warning_cond_4 = warning_cond_4,
        warning_cond_5 = warning_cond_5,
        warning_cond_6 = warning_cond_6
      )
    } else {
      test_fun <- expect_no_error
    }

    # Prepare test function arguments
    bcf_data_train <- list(X = X_train)
    bcf_data_test <- list(X = X_test)
    if (model_options_df$outcome_type[i] == "continuous") {
      bcf_data_train[["y"]] <- y_continuous_train
      bcf_data_test[["y"]] <- y_continuous_test
    } else {
      bcf_data_train[["y"]] <- y_binary_train
      bcf_data_test[["y"]] <- y_binary_test
    }
    if (model_options_df$include_propensity[i]) {
      bcf_data_train[["propensity"]] <- propensity_train
      bcf_data_test[["propensity"]] <- propensity_test
    } else {
      bcf_data_train[["propensity"]] <- NULL
      bcf_data_test[["propensity"]] <- NULL
    }
    if (model_options_df$treatment_type[i] == "binary") {
      bcf_data_train[["Z"]] <- binary_treatment_train
      bcf_data_test[["Z"]] <- binary_treatment_test
    } else if (model_options_df$treatment_type[i] == "univariate_continuous") {
      bcf_data_train[["Z"]] <- continuous_treatment_train
      bcf_data_test[["Z"]] <- continuous_treatment_test
    } else {
      bcf_data_train[["Z"]] <- multivariate_treatment_train
      bcf_data_test[["Z"]] <- multivariate_treatment_test
    }
    if (model_options_df$random_effects[i] != "none") {
      bcf_data_train[["rfx_group_ids"]] <- group_ids_train
      bcf_data_test[["rfx_group_ids"]] <- group_ids_test
    } else {
      bcf_data_train[["rfx_group_ids"]] <- NULL
      bcf_data_test[["rfx_group_ids"]] <- NULL
    }
    if (model_options_df$random_effects[i] == "custom") {
      bcf_data_train[["rfx_basis"]] <- rfx_basis_train
      bcf_data_test[["rfx_basis"]] <- rfx_basis_test
    } else {
      bcf_data_train[["rfx_basis"]] <- NULL
      bcf_data_test[["rfx_basis"]] <- NULL
    }

    # Apply testthat expectation(s)
    test_fun({
      run_bcf_factorial(
        bcf_data_train = bcf_data_train,
        bcf_data_test = bcf_data_test,
        treatment_type = model_options_df$treatment_type[i],
        variance_forest = model_options_df$variance_forest[i],
        random_effects = model_options_df$random_effects[i],
        sampling_global_error_scale = model_options_df$sampling_global_error_scale[
          i
        ],
        sampling_mu_leaf_scale = model_options_df$sampling_mu_leaf_scale[
          i
        ],
        sampling_tau_leaf_scale = model_options_df$sampling_tau_leaf_scale[
          i
        ],
        outcome_type = model_options_df$outcome_type[i],
        num_chains = model_options_df$num_chains[i],
        adaptive_coding = model_options_df$adaptive_coding[i],
        include_propensity = model_options_df$include_propensity[i]
      )
    })
  }
})
