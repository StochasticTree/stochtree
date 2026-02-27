# Load library
library(stochtree)

# Set seed
set.seed(2026)

# Sample size and number of predictors
n <- 2000
p <- 5

# Design matrix and true lambda function
X <- matrix(rnorm(n * p), n, p)
beta <- rep(1 / sqrt(p), p)
true_lambda_function <- X %*% beta

# Set cutpoints for ordinal categories (3 categories: 1, 2, 3)
n_categories <- 3
gamma_true <- c(-2, 1)
ordinal_cutpoints <- log(cumsum(exp(gamma_true)))
ordinal_cutpoints

# True ordinal class probabilities
true_probs <- matrix(0, nrow = n, ncol = n_categories)
for (j in 1:n_categories) {
  if (j == 1) {
    true_probs[, j] <- 1 - exp(-exp(gamma_true[j] + true_lambda_function))
  } else if (j == n_categories) {
    true_probs[, j] <- 1 - rowSums(true_probs[, 1:(j - 1), drop = FALSE])
  } else {
    true_probs[, j] <- exp(-exp(gamma_true[j - 1] + true_lambda_function)) *
      (1 - exp(-exp(gamma_true[j] + true_lambda_function)))
  }
}

# Generate ordinal outcomes
y <- sapply(1:nrow(X), function(i) {
  sample(1:n_categories, 1, prob = true_probs[i, ])
})
cat("Outcome distribution:", table(y), "\n")

# Train test split
train_idx <- sample(1:n, size = floor(0.8 * n))
test_idx <- setdiff(1:n, train_idx)
X_train <- X[train_idx, ]
y_train <- y[train_idx]
X_test <- X[test_idx, ]
y_test <- y[test_idx]

# Sample the cloglog ordinal BART model
runtime <- system.time({
  bart_model <- bart(
    X_train = X_train,
    y_train = y_train,
    X_test = X_test,
    num_gfr = 0,
    num_burnin = 1000,
    num_mcmc = 1000,
    general_params = list(
      cutpoint_grid_size = 100,
      sample_sigma2_global = FALSE,
      keep_every = 1,
      num_chains = 1,
      verbose = FALSE,
      random_seed = 3,
      outcome_model = outcome_model(outcome = 'ordinal', link = 'cloglog')
    ),
    mean_forest_params = list(num_trees = 50)
  )
})

# Compute category probabilities for train and test set
est_probs_train <- predict(
  bart_model,
  X = X_train,
  scale = "probability",
  terms = "y_hat"
)
est_probs_test <- predict(
  bart_model,
  X = X_test,
  scale = "probability",
  terms = "y_hat"
)

# Traceplots of cutoff parameters
par(mfrow = c(2, 1))
plot(
  bart_model$cloglog_cutpoint_samples[1, ],
  type = 'l',
  main = expression(gamma[1]),
  ylab = "Value",
  xlab = "MCMC Sample"
)
abline(h = gamma_true[1], col = 'red', lty = 2)
plot(
  bart_model$cloglog_cutpoint_samples[2, ],
  type = 'l',
  main = expression(gamma[2]),
  ylab = "Value",
  xlab = "MCMC Sample"
)
abline(h = gamma_true[2], col = 'red', lty = 2)

# Histograms of cutoff parameters
gamma1 <- bart_model$cloglog_cutpoint_samples[1, ] +
  colMeans(bart_model$y_hat_train)
summary(gamma1)
hist(gamma1)
gamma2 <- bart_model$cloglog_cutpoint_samples[2, ] +
  colMeans(bart_model$y_hat_train)
summary(gamma2)
hist(gamma2)

# Traceplots of cutoff parameters combined with average forest predictions
par(mfrow = c(3, 2))
rowMeans(bart_model$cloglog_cutpoint_samples)
moo <- t(bart_model$cloglog_cutpoint_samples) + colMeans(bart_model$y_hat_train)
plot(moo[, 1])
abline(h = gamma_true[1] + mean(true_lambda_function[train_idx]))
plot(moo[, 2])
abline(h = gamma_true[2] + mean(true_lambda_function[train_idx]))
plot(bart_model$cloglog_cutpoint_samples[1, ])
plot(bart_model$cloglog_cutpoint_samples[2, ])

# Compare forest predictions with the truth (for training and test sets)
par(mfrow = c(2, 1))

# Train set
lambda_pred_train <- rowMeans(bart_model$y_hat_train) -
  mean(bart_model$y_hat_train)
plot(lambda_pred_train, true_lambda_function[train_idx])
abline(a = 0, b = 1, col = 'blue', lwd = 2)
cor_train <- cor(true_lambda_function[train_idx], lambda_pred_train)
text(
  min(true_lambda_function[train_idx]),
  max(true_lambda_function[train_idx]),
  paste('Correlation:', round(cor_train, 3)),
  adj = 0,
  col = 'red'
)

# Test set
lambda_pred_test <- rowMeans(bart_model$y_hat_test) -
  mean(bart_model$y_hat_test)
plot(lambda_pred_test, true_lambda_function[test_idx])
abline(a = 0, b = 1, col = 'blue', lwd = 2)
cor_test <- cor(true_lambda_function[test_idx], lambda_pred_test)
text(
  min(true_lambda_function[test_idx]),
  max(true_lambda_function[test_idx]),
  paste('Correlation:', round(cor_test, 3)),
  adj = 0,
  col = 'red'
)

# Compare estimated vs true class probabilities for training set
par(mfrow = c(2, 2))
for (j in 1:n_categories) {
  mean_probs <- rowMeans(est_probs_train[, j, ])
  plot(
    true_probs[train_idx, j],
    mean_probs,
    xlab = paste("True Prob Category", j),
    ylab = paste("Estimated Prob Category", j)
  )
  abline(a = 0, b = 1, col = 'blue', lwd = 2)
  cor_train_prob <- cor(true_probs[train_idx, j], mean_probs)
  text(
    min(true_probs[train_idx, j]),
    max(mean_probs),
    paste('Correlation:', round(cor_train_prob, 3)),
    adj = 0,
    col = 'red'
  )
}

# Compare estimated vs true class probabilities for test set
par(mfrow = c(2, 2))
for (j in 1:n_categories) {
  mean_probs <- rowMeans(est_probs_test[, j, ])
  plot(
    true_probs[test_idx, j],
    mean_probs,
    xlab = paste("True Prob Category", j),
    ylab = paste("Estimated Prob Category", j)
  )
  abline(a = 0, b = 1, col = 'blue', lwd = 2)
  cor_test_prob <- cor(true_probs[test_idx, j], mean_probs)
  text(
    min(true_probs[test_idx, j]),
    max(mean_probs),
    paste('Correlation:', round(cor_test_prob, 3)),
    adj = 0,
    col = 'red'
  )
}

# Evaluate test set posterior interval coverage of lambda(x)
preds <- predict(bart_model, X = X_test, terms = "y_hat", scale = "linear")
adj <- -1 * mean(rowMeans(preds))
linear_interval <- compute_bart_posterior_interval(
  bart_model,
  X = X_test,
  terms = "y_hat",
  scale = "linear"
)
plot(rowMeans(preds) + adj, true_lambda_function[test_idx])
points(
  linear_interval$lower + adj,
  true_lambda_function[test_idx],
  col = 'blue'
)
points(
  linear_interval$upper + adj,
  true_lambda_function[test_idx],
  col = 'blue'
)
abline(0, 1)
linear_coverage <- (((linear_interval$lower + adj) <=
  true_lambda_function[test_idx]) &
  ((linear_interval$upper + adj) >= true_lambda_function[test_idx]))
mean(linear_coverage)

# Evaluate test set posterior interval coverage of survival function
probability_interval <- compute_bart_posterior_interval(
  bart_model,
  X = X_test,
  terms = "y_hat",
  scale = "probability"
)
probability_coverage_gt1 <- (((probability_interval$lower[, 1]) <=
  rowSums(true_probs[test_idx, 2:n_categories, drop = FALSE])) &
  ((probability_interval$upper[, 1]) >=
    rowSums(true_probs[test_idx, 2:n_categories, drop = FALSE])))
mean(probability_coverage_gt1)
probability_coverage_gt2 <- (((probability_interval$lower[, 2]) <=
  rowSums(true_probs[test_idx, 3:n_categories, drop = FALSE])) &
  ((probability_interval$upper[, 2]) >=
    rowSums(true_probs[test_idx, 3:n_categories, drop = FALSE])))
mean(probability_coverage_gt2)
probability_coverage_le1 <- (((1 - probability_interval$upper[, 1]) <=
  true_probs[test_idx, 1, drop = FALSE]) &
  ((1 - probability_interval$lower[, 1]) >=
    true_probs[test_idx, 1, drop = FALSE]))
mean(probability_coverage_le1)

# Compute test set prediction contrast on probability scale
X0 <- X_test
X1 <- X_test + 1
linear_contrast <- compute_contrast_bart_model(
  bart_model,
  X_0 = X0,
  X_1 = X1,
  type = "posterior",
  scale = "linear"
)
probability_contrast <- compute_contrast_bart_model(
  bart_model,
  X_0 = X0,
  X_1 = X1,
  type = "posterior",
  scale = "probability"
)

# Sample from posterior predictive distribution
y_ppd <- sample_bart_posterior_predictive(
  bart_model,
  X = X_test,
  num_draws_per_sample = 100
)
# Inspect results
true_probs_test <- true_probs[test_idx, ]
max_ind <- which.max(true_probs_test[, 1])
true_probs_test[max_ind, ]
hist(y_ppd[max_ind, , ])
hist(est_probs_test[max_ind, 1, ])
hist(est_probs_test[max_ind, 2, ])
hist(est_probs_test[max_ind, 3, ])

runtime
