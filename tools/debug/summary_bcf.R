# Load library
library(stochtree)

# Generate data
n <- 500
p <- 5
X <- matrix(runif(n * p), ncol = p)
mu_x <- 5 * X[, 1]
pi_x <- 0.2 + 0.6 * X[, 1]
tau_x <- X[, 2]
Z <- rbinom(n, 1, pi_x)
snr <- 3
rfx_group_ids <- sample(c(1, 2, 3), n, replace = TRUE)
rfx_coefs <- matrix(c(-2, -2, 0, 0, 2, 2), nrow = 3, byrow = TRUE)
rfx_basis <- cbind(1, runif(n, -1, 1))
rfx_term <- rowSums(rfx_coefs[rfx_group_ids, ] * rfx_basis)
E_XZ <- mu_x + Z * tau_x + rfx_term
y <- E_XZ + rnorm(n, 0, 1) * (sd(E_XZ) / snr)

# Fit BCF model
mu_params <- list(sample_sigma2_leaf = TRUE)
tau_params <- list(sample_sigma2_leaf = FALSE)
bcf_model <- bcf(
  X_train = X,
  Z_train = Z,
  y_train = y,
  propensity_train = pi_x,
  rfx_group_ids_train = rfx_group_ids,
  rfx_basis_train = rfx_basis,
  num_gfr = 10,
  num_burnin = 0,
  num_mcmc = 100,
  prognostic_forest_params = mu_params,
  treatment_effect_forest_params = tau_params
)

# Print the BCF model
print(bcf_model)

# Summarize the BCF model
summary(bcf_model)

# Plot the BCF model
plot(bcf_model)

# Extract parameters
extract_parameter(bcf_model, "sigma2")
extract_parameter(bcf_model, "sigma2_leaf_mu")
extract_parameter(bcf_model, "sigma2_leaf_tau")
extract_parameter(bcf_model, "tau_hat_train")
extract_parameter(bcf_model, "adaptive_coding")
