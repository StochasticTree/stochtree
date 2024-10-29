library(stochtree)

# Generate the data
n <- 500
p_X <- 10
p_W <- 1
X <- matrix(runif(n*p_X), ncol = p_X)
W <- matrix(runif(n*p_W), ncol = p_W)
f_XW <- (
    ((0 <= X[,1]) & (0.25 > X[,1])) * (-3*W[,1]) + 
        ((0.25 <= X[,1]) & (0.5 > X[,1])) * (-1*W[,1]) + 
        ((0.5 <= X[,1]) & (0.75 > X[,1])) * (1*W[,1]) + 
        ((0.75 <= X[,1]) & (1 > X[,1])) * (3*W[,1])
)
y <- f_XW + rnorm(n, 0, 1)

# Standardize outcome
y_bar <- mean(y)
y_std <- sd(y)
resid <- (y-y_bar)/y_std

alpha <- 0.9
beta <- 1.25
min_samples_leaf <- 1
max_depth <- 10
num_trees <- 100
cutpoint_grid_size = 100
global_variance_init = 1.
tau_init = 0.5
leaf_prior_scale = matrix(c(tau_init), ncol = 1)
nu <- 4
lambda <- 0.5
a_leaf <- 2.
b_leaf <- 0.5
leaf_regression <- T
feature_types <- as.integer(rep(0, p_X)) # 0 = numeric
var_weights <- rep(1/p_X, p_X)

# Data
if (leaf_regression) {
    forest_dataset <- createForestDataset(X, W)
    outcome_model_type <- 1
} else {
    forest_dataset <- createForestDataset(X)
    outcome_model_type <- 0
}
outcome <- createOutcome(resid)

# Random number generator (std::mt19937)
rng <- createRNG()

# Sampling data structures
forest_model <- createForestModel(forest_dataset, feature_types, 
                                  num_trees, n, alpha, beta, 
                                  min_samples_leaf, max_depth)

# "Active forest" (which gets updated by the sample) and 
# container of forest samples (which is written to when 
# a sample is not discarded due to burn-in / thinning)
if (leaf_regression) {
    forest_samples <- createForestContainer(num_trees, 1, F)
    active_forest <- createForest(num_trees, 1, F)
} else {
    forest_samples <- createForestContainer(num_trees, 1, T)
    active_forest <- createForest(num_trees, 1, T)
}

num_warmstart <- 10
num_mcmc <- 100
num_samples <- num_warmstart + num_mcmc
global_var_samples <- c(global_variance_init, rep(0, num_samples))
leaf_scale_samples <- c(tau_init, rep(0, num_samples))

for (i in 1:num_warmstart) {
    # Sample forest
    forest_model$sample_one_iteration(
        forest_dataset, outcome, forest_samples, active_forest, rng, feature_types, 
        outcome_model_type, leaf_prior_scale, var_weights, 
        1, 1, global_var_samples[i], cutpoint_grid_size, keep_forest = T, gfr = T
    )
    
    # Sample global variance parameter
    global_var_samples[i+1] <- sample_sigma2_one_iteration(
        outcome, forest_dataset, rng, nu, lambda
    )
    
    # Sample leaf node variance parameter and update `leaf_prior_scale`
    leaf_scale_samples[i+1] <- sample_tau_one_iteration(
        active_forest, rng, a_leaf, b_leaf, i-1
    )
    leaf_prior_scale[1,1] <- leaf_scale_samples[i+1]
}

preds <- forest_samples$predict(forest_dataset)*y_std + y_bar
