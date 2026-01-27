# Load library
library(stochtree)

# Generate data
n <- 500
p <- 5
X <- matrix(runif(n * p), ncol = p)
snr <- 3
rfx_group_ids <- sample(c(1, 2, 3), n, replace = TRUE)
rfx_coefs <- matrix(c(-2, -2, 0, 0, 2, 2), nrow = 3, byrow = TRUE)
rfx_basis <- cbind(1, runif(n, -1, 1))
rfx_term <- rowSums(rfx_coefs[rfx_group_ids, ] * rfx_basis)
E_X <- 5 * X[, 1] + rfx_term
y <- E_X + rnorm(n, 0, 1) * (sd(E_X) / snr)

# Fit BART model
bart_model <- bart(
  X_train = X,
  y_train = y,
  rfx_group_ids_train = rfx_group_ids,
  rfx_basis_train = rfx_basis,
  num_gfr = 10,
  num_burnin = 0,
  num_mcmc = 100
)

# Print the BART model
print(bart_model)

# Summarize the BART model
summary(bart_model)

# Plot the BART model
plot(bart_model)

# Extract parameters
sigma2_samples <- extract_parameter(bart_model, "sigma2")
var_x_samples <- extract_parameter(bart_model, "var_x_train")
