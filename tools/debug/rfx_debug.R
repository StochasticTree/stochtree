library(stochtree)

# Generate the data
n <- 500
p_X <- 10
p_W <- 1
X <- matrix(runif(n*p_X), ncol = p_X)
W <- matrix(runif(n*p_W), ncol = p_W)
group_ids <- rep(c(1,2), n %/% 2)
rfx_coefs <- c(-5, 5)
rfx_basis <- rep(1, n)
f_XW <- (
    ((0 <= X[,1]) & (0.25 > X[,1])) * (-3*W[,1]) + 
        ((0.25 <= X[,1]) & (0.5 > X[,1])) * (-1*W[,1]) + 
        ((0.5 <= X[,1]) & (0.75 > X[,1])) * (1*W[,1]) + 
        ((0.75 <= X[,1]) & (1 > X[,1])) * (3*W[,1])
)
rfx_term <- rfx_coefs[group_ids] * rfx_basis
y <- f_XW + rfx_term + rnorm(n, 0, 1)

# Standardize outcome
y_bar <- mean(y)
y_std <- sd(y)
resid <- (y-y_bar)/y_std

alpha <- 0.9
beta <- 1.25
min_samples_leaf <- 1
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

alpha_init <- c(1)
xi_init <- matrix(c(1,1),1,2)
sigma_alpha_init <- matrix(c(1),1,1)
sigma_xi_init <- matrix(c(1),1,1)
sigma_xi_shape <- 1
sigma_xi_scale <- 1

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
                                  num_trees, n, alpha, beta, min_samples_leaf)

# Container of forest samples
if (leaf_regression) {
    forest_samples <- createForestContainer(num_trees, 1, F)
} else {
    forest_samples <- createForestContainer(num_trees, 1, T)
}

# Random effects dataset
rfx_basis <- as.matrix(rfx_basis)
group_ids <- as.integer(group_ids)
rfx_dataset <- createRandomEffectsDataset(group_ids, rfx_basis)

# Random effects details
num_groups <- length(unique(group_ids))
num_components <- ncol(rfx_basis)

# Random effects tracker
rfx_tracker <- createRandomEffectsTracker(group_ids)

# Random effects model
rfx_model <- createRandomEffectsModel(num_components, num_groups)
rfx_model$set_working_parameter(alpha_init)
rfx_model$set_group_parameters(xi_init)
rfx_model$set_working_parameter_cov(sigma_alpha_init)
rfx_model$set_group_parameter_cov(sigma_xi_init)
rfx_model$set_variance_prior_shape(sigma_xi_shape)
rfx_model$set_variance_prior_scale(sigma_xi_scale)

# Random effect samples
rfx_samples <- createRandomEffectSamples(num_components, num_groups, rfx_tracker)

num_warmstart <- 10
num_mcmc <- 100
num_samples <- num_warmstart + num_mcmc
global_var_samples <- c(global_variance_init, rep(0, num_samples))
leaf_scale_samples <- c(tau_init, rep(0, num_samples))

for (i in 1:num_warmstart) {
    # Sample forest
    forest_model$sample_one_iteration(
        forest_dataset, outcome, forest_samples, rng, feature_types, 
        outcome_model_type, leaf_prior_scale, var_weights, 
        global_var_samples[i], cutpoint_grid_size, gfr = T
    )
    
    # Sample global variance parameter
    global_var_samples[i+1] <- sample_sigma2_one_iteration(
        outcome, forest_dataset, rng, nu, lambda
    )
    
    # Sample leaf node variance parameter and update `leaf_prior_scale`
    leaf_scale_samples[i+1] <- sample_tau_one_iteration(
        forest_samples, rng, a_leaf, b_leaf, i-1
    )
    leaf_prior_scale[1,1] <- leaf_scale_samples[i+1]
    
    # Sample random effects model
    rfx_model$sample_random_effect(rfx_dataset, outcome, rfx_tracker, rfx_samples, global_var_samples[i+1], rng)
}

for (i in (num_warmstart+1):num_samples) {
    # Sample forest
    forest_model$sample_one_iteration(
        forest_dataset, outcome, forest_samples, rng, feature_types, 
        outcome_model_type, leaf_prior_scale, var_weights, 
        global_var_samples[i], cutpoint_grid_size, gfr = F
    )
    
    # Sample global variance parameter
    global_var_samples[i+1] <- sample_sigma2_one_iteration(
        outcome, forest_dataset, rng, nu, lambda
    )
    
    # Sample leaf node variance parameter and update `leaf_prior_scale`
    leaf_scale_samples[i+1] <- sample_tau_one_iteration(
        forest_samples, rng, a_leaf, b_leaf, i-1
    )
    leaf_prior_scale[1,1] <- leaf_scale_samples[i+1]
    
    # Sample random effects model
    rfx_model$sample_random_effect(rfx_dataset, outcome, rfx_tracker, rfx_samples, global_var_samples[i+1], rng)
}

# Forest predictions
preds <- forest_samples$predict(forest_dataset)*y_std + y_bar

# Random effects predictions
rfx_preds <- rfx_samples$predict(group_ids, rfx_basis)*y_std

# Global error variance
sigma_samples <- sqrt(global_var_samples)*y_std
