# Demo of updated predict method for BART

# Load library
library(stochtree)

# Generate data
n <- 1000
p <- 5
X <- matrix(runif(n * p), ncol = p)
# fmt: skip
f_XW <- (((0 <= X[, 1]) & (0.25 > X[, 1])) * (-7.5) +
  ((0.25 <= X[, 1]) & (0.5 > X[, 1])) * (-2.5) +
  ((0.5 <= X[, 1]) & (0.75 > X[, 1])) * (2.5) +
  ((0.75 <= X[, 1]) & (1 > X[, 1])) * (7.5))
noise_sd <- 1
y <- f_XW + rnorm(n, 0, noise_sd)

# Train-test split
test_set_pct <- 0.2
n_test <- round(test_set_pct * n)
n_train <- n - n_test
test_inds <- sort(sample(1:n, n_test, replace = FALSE))
train_inds <- (1:n)[!((1:n) %in% test_inds)]
X_test <- X[test_inds, ]
X_train <- X[train_inds, ]
y_test <- y[test_inds]
y_train <- y[train_inds]
E_y_test <- f_XW[test_inds]
E_y_train <- f_XW[train_inds]

# Fit simple BART model
bart_model <- bart(
  X_train = X_train,
  y_train = y_train,
  num_gfr = 10,
  num_burnin = 0,
  num_mcmc = 1000
)

# Check several predict approaches
y_hat_posterior_test <- predict(bart_model, X = X_test)$y_hat
y_hat_mean_test <- predict(
  bart_model,
  X = X_test,
  type = "mean",
  terms = c("y_hat")
)
y_hat_test <- predict(
  bart_model,
  X = X_test,
  type = "mean",
  terms = c("rfx", "variance")
)

y_hat_intervals <- compute_bart_posterior_interval(
  model_object = bart_model,
  transform = function(x) x,
  terms = c("y_hat", "mean_forest"),
  covariates = X_test,
  level = 0.95
)

(coverage <- mean(
  (y_hat_intervals$mean_forest_predictions$lower <= E_y_test) &
    (y_hat_intervals$mean_forest_predictions$upper >= E_y_test)
))

pred_intervals <- sample_bart_posterior_predictive(
  model_object = bart_model,
  covariates = X_test,
  level = 0.95
)

(coverage_pred <- mean(
  (pred_intervals$lower <= y_test) &
    (pred_intervals$upper >= y_test)
))

# Generate probit data
n <- 1000
p <- 5
X <- matrix(runif(n * p), ncol = p)
# fmt: skip
f_X <- (((0 <= X[, 1]) & (0.25 > X[, 1])) * (-2.5) +
  ((0.25 <= X[, 1]) & (0.5 > X[, 1])) * (-1.25) +
  ((0.5 <= X[, 1]) & (0.75 > X[, 1])) * (1.25) +
  ((0.75 <= X[, 1]) & (1 > X[, 1])) * (2.5))
noise_sd <- 1
W <- f_X + rnorm(n, 0, noise_sd)
y <- as.numeric(W > 0)

# Train-test split
test_set_pct <- 0.2
n_test <- round(test_set_pct * n)
n_train <- n - n_test
test_inds <- sort(sample(1:n, n_test, replace = FALSE))
train_inds <- (1:n)[!((1:n) %in% test_inds)]
X_test <- X[test_inds, ]
X_train <- X[train_inds, ]
W_test <- W[test_inds]
W_train <- W[train_inds]
y_test <- y[test_inds]
y_train <- y[train_inds]
E_y_test <- f_X[test_inds]
E_y_train <- f_X[train_inds]

# Fit simple BART model
bart_model <- bart(
  X_train = X_train,
  y_train = y_train,
  num_gfr = 10,
  num_burnin = 0,
  num_mcmc = 1000,
  general_params = list(probit_outcome_model = TRUE)
)

# Predict on latent scale
y_hat_post <- predict(
  object = bart_model,
  X = X_test,
  type = "posterior",
  terms = c("y_hat"),
  scale = "linear"
)

# Predict on probability scale
y_hat_post_prob <- predict(
  object = bart_model,
  X = X_test,
  type = "posterior",
  terms = c("y_hat"),
  scale = "probability"
)

# Compute intervals on latent scale
y_hat_intervals <- compute_bart_posterior_interval(
  model_object = bart_model,
  scale = "linear",
  terms = c("y_hat"),
  covariates = X_test,
  level = 0.95
)

# Compute intervals on probability scale
y_hat_prob_intervals <- compute_bart_posterior_interval(
  model_object = bart_model,
  scale = "probability",
  terms = c("y_hat"),
  covariates = X_test,
  level = 0.95
)

# Compute posterior means
y_hat_mean_latent <- rowMeans(y_hat_post)
y_hat_mean_prob <- rowMeans(y_hat_post_prob)

# Plot on latent scale
sort_inds <- order(y_hat_mean_latent)
plot(y_hat_mean_latent[sort_inds])
lines(y_hat_intervals$lower[sort_inds])
lines(y_hat_intervals$upper[sort_inds])

# Plot on probability scale
sort_inds <- order(y_hat_mean_prob)
plot(y_hat_mean_prob[sort_inds])
lines(y_hat_prob_intervals$lower[sort_inds])
lines(y_hat_prob_intervals$upper[sort_inds])

# Draw from posterior predictive for covariates in the test set
ppd_samples <- sample_bart_posterior_predictive(
  model_object = bart_model,
  covariates = X_test,
  num_draws = 10
)

# Compute histogram of PPD probabilities for both outcome classes
ppd_samples_prob <- apply(ppd_samples, 1, mean)
ppd_outcome_0 <- ppd_samples_prob[y_test == 0]
ppd_outcome_1 <- ppd_samples_prob[y_test == 1]
hist(ppd_outcome_0, breaks = 50, xlim = c(0, 1))
hist(ppd_outcome_1, breaks = 50, xlim = c(0, 1))

# Compute posterior ROC
num_mcmc <- 1000
num_thresholds <- 1000
thresholds <- seq(0.001, 0.999, length.out = num_thresholds)
tpr_mean <- rep(NA, num_thresholds)
fpr_mean <- rep(NA, num_thresholds)
tpr_samples <- matrix(NA, num_thresholds, num_mcmc)
fpr_samples <- matrix(NA, num_thresholds, num_mcmc)
for (i in 1:num_thresholds) {
  is_above_threshold_samples <- y_hat_post > qnorm(thresholds[i])
  is_above_threshold_mean <- y_hat_mean_latent > qnorm(thresholds[i])
  n_positive <- sum(y_test)
  n_negative <- sum(y_test == 0)
  y_above_threshold_mean <- y_test[is_above_threshold_mean]
  for (j in 1:num_mcmc) {
    y_above_threshold <- y_test[is_above_threshold_samples[, j]]
    tpr_samples[i, j] <- sum(y_above_threshold) / n_positive
    fpr_samples[i, j] <- sum(y_above_threshold == 0) / n_negative
  }
  # tpr_mean[i] <- sum(y_above_threshold_mean) / n_positive
  # fpr_mean[i] <- sum(y_above_threshold_mean == 0) / n_negative
  tpr_mean[i] <- mean(tpr_samples[i, ])
  fpr_mean[i] <- mean(fpr_samples[i, ])
}

for (i in 1:num_mcmc) {
  if (i == 1) {
    plot(
      fpr_samples[, i],
      tpr_samples[, i],
      type = "line",
      col = "blue",
      lwd = 1,
      lty = 1,
      xlab = "False positive rate",
      ylab = "True positive rate"
    )
  } else {
    lines(fpr_samples[, i], tpr_samples[, i], col = "blue", lwd = 1, lty = 1)
  }
}
lines(fpr_mean, tpr_mean, col = "black", lwd = 3, lty = 3)
