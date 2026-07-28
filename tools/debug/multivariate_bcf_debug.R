# Load libraries
library(stochtree)

# Generate data
n <- 500
p <- 5
snr <- 2.0
X <- matrix(runif(n*p), ncol = p)
pi_x <- cbind(0.25 + 0.5 * X[, 1], 0.75 - 0.5 * X[, 2])
mu_x <- pi_x[, 1] * 5 + pi_x[, 2] * 2 + 2 * X[, 3]
tau_x <- cbind(X[, 2], X[, 3])
Z <- matrix(NA_integer_, nrow = n, ncol = ncol(pi_x))
for (i in 1:ncol(pi_x)) {
    Z[, i] <- rbinom(n, 1, pi_x[, i])
}
E_XZ <- mu_x + rowSums(Z * tau_x)
y <- E_XZ + rnorm(n, 0, 1)*(sd(E_XZ) / snr)

# Split data into test and train sets
test_set_pct <- 0.2
n_test <- round(test_set_pct*n)
n_train <- n - n_test
test_inds <- sort(sample(1:n, n_test, replace = FALSE))
train_inds <- (1:n)[!((1:n) %in% test_inds)]
X_test <- X[test_inds,]
X_train <- X[train_inds,]
pi_test <- pi_x[test_inds,]
pi_train <- pi_x[train_inds,]
Z_test <- Z[test_inds,]
Z_train <- Z[train_inds,]
y_test <- y[test_inds]
y_train <- y[train_inds]
mu_test <- mu_x[test_inds]
mu_train <- mu_x[train_inds]
tau_test <- tau_x[test_inds,]
tau_train <- tau_x[train_inds,]

# Run BCF
num_gfr <- 10
num_burnin <- 0
num_mcmc <- 100
general_params <- list(adaptive_coding = F)
prognostic_forest_params <- list(sample_sigma2_leaf = F)
treatment_effect_forest_params <- list(sample_sigma2_leaf = F)
bcf_model <- bcf(
    X_train = X_train, Z_train = Z_train, y_train = y_train, propensity_train = pi_train, 
    X_test = X_test, Z_test = Z_test, propensity_test = pi_test, 
    num_gfr = num_gfr, num_burnin = num_burnin, num_mcmc = num_mcmc, 
    general_params = general_params, prognostic_forest_params = prognostic_forest_params, 
    treatment_effect_forest_params = treatment_effect_forest_params
)

# Check results
y_hat_test_mean <- rowMeans(bcf_model$y_hat_test)
plot(y_hat_test_mean, y_test); abline(0,1,col="red")
sqrt(mean((y_hat_test_mean - y_test)^2))
