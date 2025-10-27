#' Run the BART algorithm for ordinal outcomes using a complementary log-log link
#'  
#' @param X A numeric matrix of predictors (training data).
#' @param y A numeric vector of ordinal outcomes (positive integers starting from 1).
#' @param X_test An optional numeric matrix of predictors (test data).
#' @param n_trees Number of trees in the BART ensemble. Default: `50`.
#' @param num_gfr Number of GFR samples to draw at the beginning of the sampler. Default: `10`.
#' @param num_burnin Number of burn-in MCMC samples to discard. Default: `0`.
#' @param num_mcmc Total number of MCMC samples to draw. Default: `500`.
#' @param n_thin Thinning interval for MCMC samples. Default: `1`.
#' @param alpha_gamma Shape parameter for the log-gamma prior on cutpoints. Default: `2.0`.
#' @param beta_gamma Rate parameter for the log-gamma prior on cutpoints. Default: `2.0`.
#' @param variable_weights (Optional) vector of variable weights for splitting (default: equal weights).
#' @param feature_types (Optional) vector indicating feature types (0 for continuous, 1 for categorical; default: all continuous).
#' @param seed (Optional) random seed for reproducibility.
#' @param num_threads (Optional) Number of threads to use in split evaluations and other compute-intensive operations. Default: 1.
#' @export
cloglog_ordinal_bart <- function(X, y, X_test = NULL,
                                 n_trees = 50,
                                 num_gfr = 10, 
                                 num_burnin = 0, 
                                 num_mcmc = 500,
                                 n_thin = 1,
                                 alpha_gamma = 2.0,
                                 beta_gamma = 2.0,
                                 variable_weights = NULL,
                                 feature_types = NULL,
                                 seed = NULL, 
                                 num_threads = 1) {
  # BART parameters
  alpha_bart <- 0.95
  beta_bart <- 2
  min_samples_in_leaf <- 5
  max_depth <- 10
  scale_leaf <- 2 / sqrt(n_trees)
  cutpoint_grid_size <- 100       # Needed for stochtree:::sample_mcmc_one_iteration_cpp (for GFR), not used in MCMC BART
  
  # Fixed for identifiability (can be pass as argument later if desired)
  gamma_0 = 0.0  # First gamma cutpoint fixed at gamma_0 = 0
  
  # Determine whether a test dataset is provided
  has_test <- !is.null(X_test)
  
  # Data checks
  if (!is.matrix(X)) X <- as.matrix(X)
  if (!is.numeric(y)) y <- as.numeric(y)
  if (has_test && !is.matrix(X_test)) X_test <- as.matrix(X_test)
  
  n_samples <- nrow(X)
  n_features <- ncol(X)
  
  if (any(y < 1) || any(y != round(y))) {
    stop("Ordinal outcome y must contain positive integers starting from 1")
  }
  
  # Convert from 1-based (R) to 0-based (C++) indexing
  ordinal_outcome <- as.integer(y - 1)
  n_levels <- max(y)  # Number of ordinal categories
  
  if (n_levels < 2) {
    stop("Ordinal outcome must have at least 2 categories")
  }
  
  if (is.null(variable_weights)) {
    variable_weights <- rep(1.0, n_features)
  }
  
  if (is.null(feature_types)) {
    feature_types <- rep(0L, n_features)
  }
  
  if (!is.null(seed)) {
    set.seed(seed)
  }
  
  # Indices of MCMC samples to keep after GFR, burn-in, and thinning
  keep_idx <- seq(num_gfr + num_burnin + 1, num_gfr + num_burnin + num_mcmc, by = n_thin)
  n_keep <- length(keep_idx)
  
  # Storage for MCMC samples
  forest_pred_train <- matrix(0, n_samples, n_keep)
  if (has_test) {
    n_samples_test <- nrow(X_test)
    forest_pred_test <- matrix(0, n_samples_test, n_keep)
  }
  gamma_samples <- matrix(0, n_levels - 1, n_keep)
  latent_samples <- matrix(0, n_samples, n_keep)

  # Initialize samplers
  ordinal_sampler <- stochtree:::ordinal_sampler_cpp()
  rng <- stochtree::createCppRNG(if (is.null(seed)) sample.int(.Machine$integer.max, 1) else seed)
  
  # Initialize other model structures as before
  dataX <- stochtree::createForestDataset(X)
  if (has_test) {
    dataXtest <- stochtree::createForestDataset(X_test)
  }
  outcome_data <- stochtree::createOutcome(as.numeric(ordinal_outcome))
  active_forest <- stochtree::createForest(as.integer(n_trees), 1L, TRUE, FALSE)  # Use constant leaves
  active_forest$set_root_leaves(0.0)
  split_prior <- stochtree:::tree_prior_cpp(alpha_bart, beta_bart, min_samples_in_leaf, max_depth)
  forest_samples <- stochtree::createForestSamples(as.integer(n_trees), 1L, TRUE, FALSE)  # Use constant leaves
  forest_tracker <- stochtree:::forest_tracker_cpp(
    dataX$data_ptr,
    as.integer(feature_types),
    as.integer(n_trees),
    as.integer(n_samples)
  )

  # Latent variable (Z in Alam et al (2025) notation)
  dataX$add_auxiliary_dimension(nrow(X))
  # Forest predictions (eta in Alam et al (2025) notation)
  dataX$add_auxiliary_dimension(nrow(X))
  # Log-scale non-cumulative cutpoint (gamma in Alam et al (2025) notation)
  dataX$add_auxiliary_dimension(n_levels - 1)
  # Exponentiated cumulative cutpoints (exp(c_k) in Alam et al (2025) notation)
  # This auxiliary series is designed so that the element stored at position `i`
  # corresponds to the sum of all exponentiated gamma_j values for j < i.
  # It has n_levels elements instead of n_levels - 1 because even the largest 
  # categorical index has a valid value of sum_{j < i} exp(gamma_j)
  dataX$add_auxiliary_dimension(n_levels)
  
  # Initialize gamma parameters to zero (3rd auxiliary data series, mapped to `dim_idx = 2` with 0-indexing)
  initial_gamma <- rep(0.0, n_levels - 1)
  for (i in seq_along(initial_gamma)) {
    dataX$set_auxiliary_data_value(2, i - 1, initial_gamma[i])
  }

  # Convert the log-scale parameters into cumulative exponentiated parameters.
  # This is done under the hood in a C++ function for efficiency.
  ordinal_sampler_update_cumsum_exp_cpp(ordinal_sampler, dataX$data_ptr)
  
  # Initialize forest predictions to zero (slot 1)
  for (i in 1:n_samples) {
    dataX$set_auxiliary_data_value(1, i - 1, 0.0)
  }
  
  # Initialize latent variables to zero (slot 0)
  for (i in 1:n_samples) {
    dataX$set_auxiliary_data_value(0, i - 1, 0.0)
  }
  
  # Set up sweep indices for tree updates (sample all trees each iteration)
  sweep_indices <- as.integer(seq(0, n_trees - 1))
  
  sample_counter <- 0
  for (i in 1:(num_mcmc + num_burnin + num_gfr)) {
    keep_sample <- i %in% keep_idx
    if (keep_sample) {
      sample_counter <- sample_counter + 1
    }

    # 1. Sample forest using MCMC
    if (i > num_gfr) {
      stochtree:::sample_mcmc_one_iteration_cpp(
        dataX$data_ptr, outcome_data$data_ptr, forest_samples$forest_container_ptr,
        active_forest$forest_ptr, forest_tracker, split_prior, rng$rng_ptr,
        sweep_indices, as.integer(feature_types), as.integer(cutpoint_grid_size),
        scale_leaf, variable_weights, alpha_gamma, beta_gamma, 1.0, 4L, keep_sample, 
        num_threads
      )
    } else {
      stochtree:::sample_gfr_one_iteration_cpp(
        dataX$data_ptr, outcome_data$data_ptr, forest_samples$forest_container_ptr,
        active_forest$forest_ptr, forest_tracker, split_prior, rng$rng_ptr,
        sweep_indices, as.integer(feature_types), as.integer(cutpoint_grid_size),
        scale_leaf, variable_weights, alpha_gamma, beta_gamma, 1.0, 4L, keep_sample, 
        ncol(X), num_threads
      )
    }

    # Set auxiliary data slot 1 to current forest predictions = lambda_hat = sum of all the tree predictions
    # This is needed for updating gamma parameters, latent z_i's
    forest_pred_current <- active_forest$predict(dataX)
    for (i in 1:n_samples) {
      dataX$set_auxiliary_data_value(1, i - 1, forest_pred_current[i]);
    }

    # 2. Sample latent z_i's using truncated exponential
    stochtree:::ordinal_sampler_update_latent_variables_cpp(
      ordinal_sampler, dataX$data_ptr, outcome_data$data_ptr, rng$rng_ptr
    )

    # 3. Sample gamma parameters
    stochtree:::ordinal_sampler_update_gamma_params_cpp(
      ordinal_sampler, dataX$data_ptr, outcome_data$data_ptr,
      alpha_gamma, beta_gamma, gamma_0, rng$rng_ptr
    )
    
    # 4. Update cumulative sum of exp(gamma) values
    ordinal_sampler_update_cumsum_exp_cpp(ordinal_sampler, dataX$data_ptr)
    
    if (keep_sample) {
      forest_pred_train[, sample_counter] <- active_forest$predict(dataX)
      if (has_test) {
        forest_pred_test[, sample_counter] <- active_forest$predict(dataXtest)
      }
      gamma_current <- dataX$get_auxiliary_data_vector(2)
      gamma_samples[, sample_counter] <- gamma_current
      latent_current <- dataX$get_auxiliary_data_vector(0)
      latent_samples[, sample_counter] <- latent_current
    }
  }
  
  result <- list(
    forest_predictions_train = forest_pred_train,
    forest_predictions_test = if (has_test) forest_pred_test else NULL,
    gamma_samples = gamma_samples,
    latent_samples = latent_samples,
    scale_leaf = scale_leaf,
    ordinal_outcome = ordinal_outcome,
    n_trees = n_trees,
    n_keep = n_keep
  )
  
  class(result) <- "cloglog_ordinal_bart"
  return(result)
}
