library(stochtree)

# Generate the data
n <- 500
p_X <- 10
p_W <- 1
X <- matrix(runif(n * p_X), ncol = p_X)
W <- matrix(runif(n * p_W), ncol = p_W)
f_XW <- (((0 <= X[, 1]) & (0.25 > X[, 1])) *
  (-3 * W[, 1]) +
  ((0.25 <= X[, 1]) & (0.5 > X[, 1])) * (-1 * W[, 1]) +
  ((0.5 <= X[, 1]) & (0.75 > X[, 1])) * (1 * W[, 1]) +
  ((0.75 <= X[, 1]) & (1 > X[, 1])) * (3 * W[, 1]))
# y <- f_XW + rnorm(n, 0, 1)
y <- rep(0, n)
# wgt <- rep(0, n)

# Standardize outcome
# y_bar <- mean(y)
# y_std <- sd(y)
# resid <- (y - y_bar) / y_std
resid <- y

# Sampling parameters
alpha <- 0.99
beta <- 1
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
lambda <- 10
a_leaf <- 2.
b_leaf <- 0.5
leaf_regression <- T
feature_types <- as.integer(rep(0, p_X)) # 0 = numeric
var_weights <- rep(1 / p_X, p_X)

# Sampling data structures
# Data
# forest_dataset <- createForestDataset(X, W, wgt)
forest_dataset <- createForestDataset(X, W)
outcome_model_type <- 1
leaf_dimension <- p_W
outcome <- createOutcome(resid)

# Random number generator (boost::random::mt19937)
rng <- createCppRNG()

# Sampling data structures
forest_model_config <- createForestModelConfig(
  feature_types = feature_types,
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

# Prepare to run sampler
num_mcmc <- 2000
global_var_samples <- rep(0, num_mcmc)

# Run MCMC
for (i in 1:num_mcmc) {
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
    gfr = F,
    num_threads = 1
  )

  # Sample global variance parameter
  # current_sigma2 <- sampleGlobalErrorVarianceOneIteration(
  #   outcome,
  #   forest_dataset,
  #   rng,
  #   nu,
  #   lambda
  # )
  current_sigma2 <- 1 / rgamma(1, shape = nu / 2, rate = lambda / 2)
  global_var_samples[i] <- current_sigma2
  global_model_config$update_global_error_variance(current_sigma2)
}

plot(global_var_samples, type = "l")
hist(global_var_samples, breaks = 30)
hist(
  1 / rgamma(num_mcmc, shape = nu / 2, rate = lambda / 2),
  breaks = 30
)
mean(global_var_samples)
mean(1 / rgamma(num_mcmc, shape = nu / 2, rate = lambda / 2))
sd(global_var_samples)
sd(1 / rgamma(num_mcmc, shape = nu / 2, rate = lambda / 2))

# Extract forest predictions
forest_preds <- forest_samples$predict(
  forest_dataset
)

y_hat_prior <- rowMeans(forest_preds)
y_hat_prior_lb <- apply(forest_preds, 1, quantile, probs = 0.025)
y_hat_prior_ub <- apply(forest_preds, 1, quantile, probs = 0.975)
plot(
  1:length(y_hat_prior),
  y_hat_prior,
  type = "l",
  ylim = range(c(y_hat_prior_lb, y_hat_prior_ub))
)
lines(1:length(y_hat_prior), y_hat_prior_lb, col = "blue")
lines(1:length(y_hat_prior), y_hat_prior_ub, col = "blue")
