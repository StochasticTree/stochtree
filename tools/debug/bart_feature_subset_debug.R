# Debugging XBART surrogate model used to "fit the fit" of a BCF CATE function on a simulated and simplified version of the ACIC data

# Load library
library(stochtree)

# Generate data
set.seed(12345)
n <- 100

# 5 continuous features
p_continuous <- 5
X_continuous <- matrix(runif(n * p_continuous), nrow = n, ncol = p_continuous)

# 5 categorical features: 2 unordered, 2 binary, 1 ordered
xc_1 <- sample(1:15, n, replace = TRUE) # unordered categorical with 15 levels
xc_2 <- sample(1:5, n, replace = TRUE) # unordered categorical with 5 levels
xc_3 <- sample(0:1, n, replace = TRUE) # binary categorical
xc_4 <- sample(0:1, n, replace = TRUE) # binary categorical
xc_5 <- sample(1:7, n, replace = TRUE) # ordered categorical with 7 levels
X_categorical <- cbind(xc_1, xc_2, xc_3, xc_4, xc_5)

# Combine continuous and categorical features into a dataframe
covariate_df <- data.frame(X_continuous, X_categorical)
colnames(covariate_df) <- c(
  paste0("X", 1:ncol(covariate_df))
)

# Tag categorical features as such
unordered_categorical_cols <- c("X6", "X7")
ordered_categorical_cols <- c("X8", "X9", "X10")
for (col in unordered_categorical_cols) {
  covariate_df[, col] <- factor(covariate_df[, col], ordered = F)
}
for (col in ordered_categorical_cols) {
  covariate_df[, col] <- factor(covariate_df[, col], ordered = T)
}

# Generate simple outcome data
y <- 0.228 +
  0.05 * (covariate_df$X1 < 0.07) -
  0.05 * (covariate_df$X2 < -0.69) -
  0.08 * (covariate_df$X6 %in% c(1, 13, 14)) +
  rnorm(n, 0, 0.01)

# Fit an XBART model that only uses the first two features
xbart_model <- stochtree::bart(
  X_train = covariate_df,
  y_train = y,
  num_gfr = 10,
  num_burnin = 0,
  num_mcmc = 0,
  mean_forest_params = list(
    num_trees = 200,
    keep_vars = c("X1", "X2")
  )
)
print(xbart_model)
plot(rowMeans(xbart_model$y_hat_train), y)
abline(0, 1)

# Fit another XBART model to the predictions of this model
yhat_surrogate <- rowMeans(xbart_model$y_hat_train)
xbart_surrogate_model <- stochtree::bart(
  X_train = covariate_df,
  y_train = yhat_surrogate,
  num_gfr = 10,
  num_burnin = 0,
  num_mcmc = 0,
  mean_forest_params = list(
    num_trees = 200,
    keep_vars = c("X1")
  )
)
print(xbart_surrogate_model)
plot(rowMeans(xbart_surrogate_model$y_hat_train), yhat_surrogate)
abline(0, 1)
