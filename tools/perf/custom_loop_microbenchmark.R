# Load libraries
library(microbenchmark)
library(stochtree)

# Simulate a simple partitioned linear model
n <- 10000
p_X <- 20
p_W <- 1
X <- matrix(runif(n * p_X), ncol = p_X)
W <- matrix(runif(n * p_W), ncol = p_W)
f_XW <- (((0 <= X[, 1]) & (0.25 > X[, 1])) *
  (-3 * W[, 1]) +
  ((0.25 <= X[, 1]) & (0.5 > X[, 1])) * (-1 * W[, 1]) +
  ((0.5 <= X[, 1]) & (0.75 > X[, 1])) * (1 * W[, 1]) +
  ((0.75 <= X[, 1]) & (1 > X[, 1])) * (3 * W[, 1]))
y <- f_XW + rnorm(n, 0, 1)

# Standardize outcome
y_bar <- mean(y)
y_std <- sd(y)
resid <- (y - y_bar) / y_std

## Sampling

# Set some parameters that inform the forest and variance parameter samplers
alpha <- 0.9
beta <- 1.25
min_samples_leaf <- 1
max_depth <- 10
num_trees <- 100
cutpoint_grid_size <- 100
global_variance_init <- 1.
current_sigma2 <- global_variance_init
tau_init <- 1 / num_trees
leaf_prior_scale <- as.matrix(ifelse(
  p_W >= 1,
  diag(tau_init, p_W),
  diag(tau_init, 1)
))
nu <- 4
lambda <- 0.5
a_leaf <- 2.
b_leaf <- 0.5
leaf_regression <- T
feature_types <- as.integer(rep(0, p_X)) # 0 = numeric
var_weights <- rep(1 / p_X, p_X)

# Initialize R-level access to the C++ classes needed to sample our model

# Data
if (leaf_regression) {
  forest_dataset <- createForestDataset(X, W)
  outcome_model_type <- 1
  leaf_dimension <- p_W
} else {
  forest_dataset <- createForestDataset(X)
  outcome_model_type <- 0
  leaf_dimension <- 1
}
outcome <- createOutcome(resid)

# Random number generator (std::mt19937)
rng <- createCppRNG()

# Sampling data structures
forest_model_config <- createForestModelConfig(
  feature_types = feature_types,
  sweep_update_indices = 0:9,
  num_trees = num_trees,
  num_features = p_X,
  num_observations = n,
  variable_weights = var_weights,
  leaf_dimension = leaf_dimension,
  alpha = alpha,
  beta = beta,
  min_samples_leaf = min_samples_leaf,
  max_depth = max_depth,
  leaf_model_type = outcome_model_type,
  leaf_model_scale = leaf_prior_scale,
  cutpoint_grid_size = cutpoint_grid_size
)
global_model_config <- createGlobalModelConfig(
  global_error_variance = global_variance_init
)
forest_model <- createForestModel(
  forest_dataset,
  forest_model_config,
  global_model_config
)

# "Active forest" (which gets updated by the sample) and
# container of forest samples (which is written to when
# a sample is not discarded due to burn-in / thinning)
if (leaf_regression) {
  forest_samples <- createForestSamples(num_trees, 1, F)
  active_forest <- createForest(num_trees, 1, F)
} else {
  forest_samples <- createForestSamples(num_trees, 1, T)
  active_forest <- createForest(num_trees, 1, T)
}

# Initialize the leaves of each tree in the forest
active_forest$prepare_for_sampler(
  forest_dataset,
  outcome,
  forest_model,
  outcome_model_type,
  mean(resid)
)
active_forest$adjust_residual(
  forest_dataset,
  outcome,
  forest_model,
  ifelse(outcome_model_type == 1, T, F),
  F
)

#Prepare to run the sampler
num_warmstart <- 10
num_mcmc <- 100
num_samples <- num_warmstart + num_mcmc
global_var_samples <- c(global_variance_init, rep(0, num_samples))
leaf_scale_samples <- c(tau_init, rep(0, num_samples))

bench_results <- microbenchmark(
  {
    # Run the grow-from-root sampler to "warm-start" BART
    for (i in 1:num_warmstart) {
      # Sample forest
      forest_model$sample_one_iteration(
        forest_dataset,
        outcome,
        forest_samples,
        active_forest,
        rng,
        forest_model_config,
        global_model_config,
        keep_forest = T,
        gfr = T
      )

      # Sample global variance parameter
      current_sigma2 <- sampleGlobalErrorVarianceOneIteration(
        outcome,
        forest_dataset,
        rng,
        nu,
        lambda
      )
      global_var_samples[i + 1] <- current_sigma2
      global_model_config$update_global_error_variance(current_sigma2)

      # Sample leaf node variance parameter and update `leaf_prior_scale`
      leaf_scale_samples[i + 1] <- sampleLeafVarianceOneIteration(
        active_forest,
        rng,
        a_leaf,
        b_leaf
      )
      leaf_prior_scale[1, 1] <- leaf_scale_samples[i + 1]
      forest_model_config$update_leaf_model_scale(leaf_prior_scale)
    }

    # Update sweep indices
    # forest_model_config$update_sweep_indices(0:(num_trees-1))
    forest_model_config$update_sweep_indices(NULL)

    # Pick up from the last GFR forest (and associated global variance / leaf
    # scale parameters) with an MCMC sampler
    for (i in (num_warmstart + 1):num_samples) {
      # Sample forest
      forest_model$sample_one_iteration(
        forest_dataset,
        outcome,
        forest_samples,
        active_forest,
        rng,
        forest_model_config,
        global_model_config,
        keep_forest = T,
        gfr = F
      )

      # Sample global variance parameter
      current_sigma2 <- sampleGlobalErrorVarianceOneIteration(
        outcome,
        forest_dataset,
        rng,
        nu,
        lambda
      )
      global_var_samples[i + 1] <- current_sigma2
      global_model_config$update_global_error_variance(current_sigma2)

      # Sample leaf node variance parameter and update `leaf_prior_scale`
      leaf_scale_samples[i + 1] <- sampleLeafVarianceOneIteration(
        active_forest,
        rng,
        a_leaf,
        b_leaf
      )
      leaf_prior_scale[1, 1] <- leaf_scale_samples[i + 1]
      forest_model_config$update_leaf_model_scale(leaf_prior_scale)
    }
  },
  times = 5
)

# Forest predictions
preds <- forest_samples$predict(forest_dataset) * y_std + y_bar

# Global error variance
sigma_samples <- sqrt(global_var_samples) * y_std

# Plot samples
# plot(rowMeans(preds), y); abline(0,1,col="red",lty=3,lwd=3)
# plot(sigma_samples); abline(h = 1,col="blue",lty=3,lwd=3)
