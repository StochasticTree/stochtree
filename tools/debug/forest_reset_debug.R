# Load libraries
library(stochtree)

# Generate some data
seed <- 1234
set.seed(seed)
n <- 1000
p <- 5
X <- matrix(runif(n*p), ncol = p)
E_y <- 10*X[,1]
eps <- rnorm(n,0,1)
y <- E_y + eps
y_std <- (y-mean(y))/sd(y)

# Prepare to run sampler
num_mcmc <- 100
num_gfr <- 10
num_burnin <- 0
num_chains <- 4
keep_every <- 5
num_trees <- 100
alpha <- 0.95
beta <- 0.95
min_samples_leaf <- 5
max_depth <- 10
cutpoint_grid_size <- 100
variable_weights = rep(1/ncol(X), ncol(X))
output_dimension <- 1
is_leaf_constant <- T
leaf_model <- 0
current_sigma2 <- 1.
current_leaf_scale <- as.matrix(1/num_trees)
a_forest <- 1.
b_forest <- 1.
a_global <- 0.
b_global <- 0.
a_leaf <- 3.
b_leaf <- 1./num_trees
forest_dataset <- createForestDataset(X)
outcome <- createOutcome(y_std)
rng <- createRNG(seed)
feature_types <- as.integer(rep(0,p))
forest_model <- createForestModel(forest_dataset, feature_types, num_trees, nrow(X), alpha, beta, min_samples_leaf, max_depth)
forest_samples <- createForestContainer(num_trees, output_dimension, is_leaf_constant, FALSE)
active_forest <- createForest(num_trees, output_dimension, is_leaf_constant, FALSE)

# Container of parameter samples
sample_sigma_global <- T
sample_sigma_leaf <- F
keep_burnin <- F
keep_gfr <- T
num_actual_mcmc_iter <- num_mcmc * keep_every * num_chains
num_samples <- num_gfr + num_burnin + num_actual_mcmc_iter
num_mcmc_samples <- num_burnin + num_actual_mcmc_iter
num_retained_samples <- ifelse(keep_gfr, num_gfr, 0) + ifelse(keep_burnin, num_burnin * num_chains, 0) + num_mcmc * num_chains
if (sample_sigma_global) global_var_samples <- rep(NA, num_retained_samples)
if (sample_sigma_leaf) leaf_scale_samples <- rep(NA, num_retained_samples)
sample_counter <- 0

# Initialize the forest model (ensemble of root-only trees)
init_root_value <- 0.
active_forest$prepare_for_sampler(forest_dataset, outcome, forest_model, leaf_model, init_root_value)
active_forest$adjust_residual(forest_dataset, outcome, forest_model, FALSE, FALSE)

# Run GFR (warm start) if specified
if (num_gfr > 0){
    gfr_indices = 1:num_gfr
    for (i in 1:num_gfr) {
        keep_sample <- ifelse(keep_gfr, T, F)
        if (keep_sample) sample_counter <- sample_counter + 1

        forest_model$sample_one_iteration(
            forest_dataset, outcome, forest_samples, active_forest, 
            rng, feature_types, leaf_model, current_leaf_scale, variable_weights, 
            a_forest, b_forest, current_sigma2, cutpoint_grid_size, keep_forest = keep_sample, gfr = T, pre_initialized = T
        )
        
        if (sample_sigma_global) {
            current_sigma2 <- sample_sigma2_one_iteration(outcome, forest_dataset, rng, a_global, b_global)
            if (keep_sample) global_var_samples[sample_counter] <- current_sigma2
        }
        
        if (sample_sigma_leaf) {
            leaf_scale_double <- sample_tau_one_iteration(active_forest, rng, a_leaf, b_leaf)
            current_leaf_scale <- as.matrix(leaf_scale_double)
            if (keep_sample) leaf_scale_samples[sample_counter] <- leaf_scale_double
        }
    }
}

# Run MCMC
if (num_burnin + num_mcmc > 0) {
    for (chain_num in 1:num_chains) {
        # Reset state of active_forest and forest_model based on a previous GFR sample
        forest_ind <- num_gfr - chain_num
        resetActiveForest(active_forest, forest_samples, forest_ind)
        resetForestModel(forest_model, forest_dataset, forest_samples, forest_ind)

        # Run the MCMC sampler starting from the current active forest
        for (i in 1:num_mcmc_samples) {
            is_mcmc <- i > num_burnin
            if (is_mcmc) {
                mcmc_counter <- i - num_burnin
                if (mcmc_counter %% keep_every == 0) keep_sample <- T
                else keep_sample <- F
            } else {
                if (keep_burnin) keep_sample <- T
                else keep_sample <- F
            }
            if (keep_sample) sample_counter <- sample_counter + 1
            
            forest_model$sample_one_iteration(
                forest_dataset, outcome, forest_samples, active_forest, 
                rng, feature_types, leaf_model, current_leaf_scale, variable_weights, 
                a_forest, b_forest, current_sigma2, cutpoint_grid_size, keep_forest = keep_sample, gfr = F, pre_initialized = T
            )
            
            if (sample_sigma_global) {
                current_sigma2 <- sample_sigma2_one_iteration(outcome, forest_dataset, rng, a_global, b_global)
                if (keep_sample) global_var_samples[sample_counter] <- current_sigma2
            }
            if (sample_sigma_leaf) {
                leaf_scale_double <- sample_tau_one_iteration(active_forest, rng, a_leaf, b_leaf)
                current_leaf_scale <- as.matrix(leaf_scale_double)
                if (keep_sample) leaf_scale_samples[sample_counter] <- leaf_scale_double
            }
        }
    }
}

# Obtain predictions for all of the warmstarted MCMC samples
yhat <- forest_samples$predict(forest_dataset)*sd(y) + mean(y)

# Plot results for each chain
plot_chain <- function(y, yhat, i) {
    yhat_mean_chain <- rowMeans(yhat[,(num_gfr + (i-1)*num_mcmc):(num_gfr + (i)*num_mcmc)])
    plot(yhat_mean_chain, y, main = paste0("Chain ", i, " MCMC Samples"), xlab = "yhat")
    abline(0,1,col="red",lty=3,lwd=3)
}
yhat_mean_gfr <- rowMeans(yhat[,1:num_gfr])
plot(yhat_mean_gfr, y, main = "GFR Samples", xlab = "yhat")
abline(0,1,col="red",lty=3,lwd=3)
for (i in 1:num_chains) {
    plot_chain(y, yhat, i)
}
