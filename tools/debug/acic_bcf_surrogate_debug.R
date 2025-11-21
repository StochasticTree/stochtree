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

# Convert X1 and X2 to ordered categorical, mapping to integers
x1_unique_key <- sort(unique(covariate_df$X1))
x2_unique_key <- sort(unique(covariate_df$X2))
x1_raw <- covariate_df$X1
x2_raw <- covariate_df$X2
x1_int <- match(covariate_df$X1, x1_unique_key)
x2_int <- match(covariate_df$X2, x2_unique_key)
covariate_df$X1 <- as.integer(x1_int)
covariate_df$X2 <- as.integer(x2_int)
raw_columns <- list(
  X1 = x1_raw,
  X2 = x2_raw
)

# Encode categorical data to enable proper stochtree preprocessing
unordered_categorical_cols <- c("C1", "XC")
ordered_categorical_cols <- c("S3", "C2", "C3", "X1", "X2")
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
  X = covariate_df,
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
    # keep_vars = c("X1", "X2")
    # keep_vars = c("X1")
    keep_vars = c("X1")
  )
)
print(xbart_surrogate_model)
plot(rowMeans(xbart_surrogate_model$y_hat_train), yhat_surrogate)
abline(0, 1)

# Build flexible additive surrogate model in each of the features we wish to inspeect

# Additive XBART model for each covariate in the summary
additive_summary_columns <- c("X1", "X2")
xbart_list <- list()
for (col in additive_summary_columns) {
  xbart_list[[col]] <- stochtree::bart(
    X_train = covariate_df[, col, drop = FALSE],
    y_train = yhat_surrogate,
    num_gfr = 5,
    num_burnin = 0,
    num_mcmc = 0,
    mean_forest_params = list(
      num_trees = 1,
      alpha = 0.1,
      beta = 3,
      min_samples_leaf = 1000,
      keep_vars = c(col)
    )
  )
}

# Data frame used for summary evaluation
eval_df <- covariate_df

# Extract bases for each feature in the XBART model
n_train <- nrow(covariate_df)
n_test <- nrow(eval_df)
default_model <- xbart_list[[additive_summary_columns[1]]]
num_trees <- default_model$mean_forests$num_trees()
forest_ind <- default_model$model_params$num_samples - 1
basis_list <- list()
for (col in additive_summary_columns) {
  leaf_mat_train <- computeForestLeafIndices(
    xbart_list[[col]],
    covariates = covariate_df[, col, drop = FALSE],
    forest_type = "mean",
    forest_inds = forest_ind
  )
  leaf_mat_test <- computeForestLeafIndices(
    xbart_list[[col]],
    covariates = eval_df[, col, drop = FALSE],
    forest_type = "mean",
    forest_inds = forest_ind
  )
  xbart_basis_train <- Matrix::sparseMatrix(
    i = rep(1:n_train, num_trees),
    j = leaf_mat_train + 1,
    x = 1
  )
  xbart_basis_test <- Matrix::sparseMatrix(
    i = rep(1:n_train, num_trees),
    j = leaf_mat_test + 1,
    x = 1
  )
  basis_list[[col]] <- list(
    train_basis = xbart_basis_train,
    test_basis = xbart_basis_test
  )
}

# Use the bases for a projection of each posterior sample
alpha_reg <- 1.0 / 10000.0
posterior_projection_list = list()
for (col in additive_summary_columns) {
  X_train <- basis_list[[col]]$train_basis
  X_test <- basis_list[[col]]$test_basis
  XtX_train <- crossprod(X_train)
  regularizer <- diag(alpha_reg, ncol(X_train))
  XtX_train_inv <- solve(XtX_train + regularizer)
  contrast_est <- X_test %*% tcrossprod(XtX_train_inv, X_train)
  posterior_projection_list[[col]] <- matrix(
    0,
    nrow = n_test,
    ncol = ncol(tau_hat_posterior)
  )
  for (i in 1:ncol(tau_hat_posterior)) {
    if (i %% 100 == 0) {
      cat(
        "Projecting model",
        i,
        "of",
        ncol(tau_hat_posterior),
        "for column",
        col,
        "\n"
      )
    }
    posterior_projection_list[[col]][, i] <- as.numeric(
      contrast_est %*% tau_hat_posterior[, i]
    )
  }
}

# Compute median fit with an interval
interval_lb_fits <- list()
interval_ub_fits <- list()
median_fits <- list()
for (col in additive_summary_columns) {
  interval_lb_fits[[col]] <- apply(
    posterior_projection_list[[col]],
    1,
    function(x) quantile(x, probs = 0.025)
  )
  interval_ub_fits[[col]] <- apply(
    posterior_projection_list[[col]],
    1,
    function(x) quantile(x, probs = 0.975)
  )
  median_fits[[col]] <- apply(posterior_projection_list[[col]], 1, function(x) {
    median(x)
  })
}

# Plot results
par(mfrow = c(1, 2))
for (col in additive_summary_columns) {
  x_raw <- raw_columns[[col]]
  plot_bounds <- c(min(interval_lb_fits[[col]]), max(interval_ub_fits[[col]]))
  sort_inds <- order(x_raw)
  plot(
    x_raw[sort_inds],
    median_fits[[col]][sort_inds],
    ylim = plot_bounds,
    type = "l",
    xlab = col,
    ylab = "CATE Projection"
  )
  lines(
    x_raw[sort_inds],
    interval_lb_fits[[col]][sort_inds],
    ylim = plot_bounds
  )
  lines(
    x_raw[sort_inds],
    interval_ub_fits[[col]][sort_inds],
    ylim = plot_bounds
  )
}
par(mfrow = c(1, 1))
