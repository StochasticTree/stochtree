################################################################################
## Demo of the R interface to the low-level C++ structures
## 
## This interface is not necessarily designed for performance or 
## user interface --- rather the intent is to provide a "research" 
## interface to the C++ code that doesn't require modifying the C++
## code. For example, rather than running `sample_sigma2_one_iteration`,
## a researcher might prototype an alternative global variance sampler 
## in R and pass the updated global variance parameter back to the 
## forest sampler for another Gibbs iteration.
################################################################################

# Load library
library(stochtree)

# Set seed
random_seed = 1234
set.seed(random_seed)

# Simulate some data
n <- 500
p_X <- 10
p_W <- 1
X <- matrix(runif(n*p_X), ncol = p_X)
W <- matrix(runif(n*p_W), ncol = p_W)
f_XW <- (
    ((0 <= X[,1]) & (0.25 > X[,1])) * (-7.5*W[,1]) + 
    ((0.25 <= X[,1]) & (0.5 > X[,1])) * (-2.5*W[,1]) + 
    ((0.5 <= X[,1]) & (0.75 > X[,1])) * (2.5*W[,1]) + 
    ((0.75 <= X[,1]) & (1 > X[,1])) * (7.5*W[,1])
)
y <- f_XW + rnorm(n, 0, 1)
feature_types <- as.integer(rep(0, p_X)) # 0 = numeric
var_1_prob <- 0.9
var_weights <- c(var_1_prob, rep((1-var_1_prob)/(p_X - 1), p_X - 1))
# var_weights <- rep(1/p_X, p_X)

# Scale the data
y_bar <- mean(y)
y_std <- sd(y)
resid <- (y-y_bar)/y_std

# Initialize rng
rng_ptr <- random_number_generator(random_seed)

# Create stochtree data objects
data_ptr <- create_forest_dataset(X, W)
outcome_ptr <- create_column_vector(resid)

# Set sampling parameters
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

# Create sampling objects
tree_prior_ptr <- tree_prior(alpha, beta, min_samples_leaf)
forest_samples_ptr <- forest_container(num_trees, 1, F)
tracker_ptr <- forest_tracker(data_ptr, feature_types, num_trees, n)

# Sampling parameters
num_warmstart <- 10
num_mcmc <- 100
num_samples <- num_warmstart + num_mcmc
global_var_samples <- c(global_variance_init, rep(0, num_samples))
leaf_scale_samples <- c(tau_init, rep(0, num_samples))

# Draw warm start (GFR) samples
for (i in 1:num_warmstart) {
    sample_model_one_iteration(data_ptr, outcome_ptr, forest_samples_ptr, tracker_ptr, tree_prior_ptr,
                               rng_ptr, feature_types, 1, leaf_prior_scale, var_weights,
                               global_var_samples[i], cutpoint_grid_size, gfr = T)
    global_var_samples[i+1] <- sample_sigma2_one_iteration(outcome_ptr, rng_ptr, nu, lambda)
    leaf_scale_samples[i+1] <- sample_tau_one_iteration(forest_samples_ptr, rng_ptr, a_leaf, a_leaf, i-1)
    leaf_prior_scale[1,1] <- leaf_scale_samples[i+1]
}

# Draw MCMC samples
for (i in (num_warmstart+1):num_samples) {
    sample_model_one_iteration(data_ptr, outcome_ptr, forest_samples_ptr, tracker_ptr, tree_prior_ptr, 
                               rng_ptr, feature_types, 1, leaf_prior_scale, var_weights, 
                               global_var_samples[i], cutpoint_grid_size, gfr = F)
    global_var_samples[i+1] <- sample_sigma2_one_iteration(outcome_ptr, rng_ptr, nu, lambda)
    leaf_scale_samples[i+1] <- sample_tau_one_iteration(forest_samples_ptr, rng_ptr, a_leaf, a_leaf, i-1)
    leaf_prior_scale[1,1] <- leaf_scale_samples[i+1]
}

# Compute predictions
preds <- predict_forest(forest_samples_ptr, data_ptr)*y_std + y_bar
avg_pred <- rowMeans(preds)

# Plot warm start (GFR) results
plot(leaf_scale_samples[1:num_warmstart], ylab="tau")
plot(sqrt(global_var_samples[1:num_warmstart])*y_std, ylab="sigma^2")
plot(rowMeans(preds[,1:num_warmstart]), y, pch=16, cex=0.75, xlab = "pred", ylab = "actual"); abline(0,1,col="red",lty=2,lwd=2.5)
mean((rowMeans(preds[,1:num_warmstart]) - y)^2)

# Plot MCMC results
plot(leaf_scale_samples[(num_warmstart+1):num_samples], ylab="tau")
plot(sqrt(global_var_samples[(num_warmstart+1):num_samples])*y_std, ylab="sigma^2")
plot(rowMeans(preds[,(num_warmstart+1):num_samples]), y, pch=16, cex=0.75, xlab = "pred", ylab = "actual"); abline(0,1,col="red",lty=2,lwd=2.5)
mean((rowMeans(preds[,(num_warmstart+1):num_samples]) - y)^2)
