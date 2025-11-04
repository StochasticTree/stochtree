# Debugging XBART surrogate model used to "fit the fit" of a BCF CATE function on ACIC data

# Load library
library(stochtree)

# True tau function
true_tau_fn <- function(df) {
  0.228 +
    0.05 * (df$X1 < 0.07) -
    0.05 * (df$X2 < -0.69) -
    0.08 * (df$C1 %in% c(1, 13, 14))
}

# Load and preprocess ACIC data
df <- read.csv(
  "https://raw.githubusercontent.com/andrewherren/acic2024/refs/heads/main/data/acic2018/synthetic_data.csv"
)

# Extract treatment and outcome
y <- df$Y
Z <- df$Z

# Extract covariates
covariate_df <- df[, !(colnames(df) %in% c("schoolid", "Z", "Y"))]

# Compute the true CATE function (defined in the paper)
tau_x <- true_tau_fn(df)

# Encode categorical data to enable proper stochtree preprocessing
unordered_categorical_cols <- c("C1", "XC")
ordered_categorical_cols <- c("S3", "C2", "C3")
for (col in unordered_categorical_cols) {
  covariate_df[, col] <- factor(covariate_df[, col], ordered = F)
}
for (col in ordered_categorical_cols) {
  covariate_df[, col] <- factor(covariate_df[, col], ordered = T)
}

# Extract data dimensions
n <- nrow(df)
p <- ncol(df)

# Fit a propensity model
general_params_propensity = list(
  probit_outcome_model = T,
  sample_sigma2_global = F
)
propensity_model <- stochtree::bart(
  X_train = covariate_df,
  y_train = Z,
  general_params = general_params_propensity
)
propensity <- predict(
  propensity_model,
  covariates = covariate_df,
  type = "mean",
  terms = "y_hat"
)

# Fit causal model
treatment_forest_params <- list(
  alpha = 0.95,
  beta = 2,
  keep_vars = c("X1", "X2")
)
updated_bcf_model <- stochtree::bcf(
  X_train = covariate_df,
  Z_train = Z,
  y_train = y,
  propensity_train = propensity,
  num_gfr = 10,
  num_burnin = 1000,
  num_mcmc = 100,
  treatment_effect_forest_params = treatment_forest_params
)

# Extract posterior mean of tau(x)
tau_hat_posterior <- predict(
  updated_bcf_model,
  X = covariate_df,
  Z = Z,
  propensity = propensity,
  type = "posterior",
  terms = "cate"
)
tau_hat_mean <- rowMeans(tau_hat_posterior)
tau_hat_mean <- tau_hat_posterior[, 99]
plot(tau_hat_mean, tau_x)
abline(0, 1)

# Fit another XBART model to the predictions of this model
yhat_surrogate <- tau_hat_mean
xbart_surrogate_model <- stochtree::bart(
  X_train = covariate_df,
  y_train = yhat_surrogate,
  num_gfr = 10,
  num_burnin = 0,
  num_mcmc = 0,
  # general_params = list(
  #   sample_sigma2_global = F,
  #   sigma2_global = 0.00000001
  # ),
  mean_forest_params = list(
    # sample_sigma2_leaf = F,
    keep_vars = c("X1", "X2", "X3", "X4", "C1")
  )
)
print(xbart_surrogate_model)
plot(rowMeans(xbart_surrogate_model$y_hat_train), yhat_surrogate)
abline(0, 1)
