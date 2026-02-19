# Load library
library(stochtree)

# Set seed
set.seed(2026)

# Sample size and number of predictors
n <- 2000
p <- 5

# Design matrix and true lambda function
X <- matrix(runif(n * p), ncol = p)
beta <- rep(1 / sqrt(p), p)
true_lambda_function <- X %*% beta

# Set cutpoints for ordinal categories (2 categories: 1, 2)
n_categories <- 2
gamma_true <- c(-2)
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
      outcome_model = outcome_model(outcome = 'binary', link = 'cloglog')
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

# Compare forest predictions with the truth (for training and test sets)
par(mfrow = c(2, 1))

# Train set
lambda_pred_train <- rowMeans(bart_model$y_hat_train)
plot(lambda_pred_train, gamma_true[1] + true_lambda_function[train_idx])
abline(a = 0, b = 1, col = 'blue', lwd = 2)
cor_train <- cor(
  gamma_true[1] + true_lambda_function[train_idx],
  lambda_pred_train
)
text(
  min(true_lambda_function[train_idx]),
  max(true_lambda_function[train_idx]),
  paste('Correlation:', round(cor_train, 3)),
  adj = 0,
  col = 'red'
)

# Test set
lambda_pred_test <- rowMeans(bart_model$y_hat_test)
plot(lambda_pred_test, gamma_true[1] + true_lambda_function[test_idx])
abline(a = 0, b = 1, col = 'blue', lwd = 2)
cor_test <- cor(
  gamma_true[1] + true_lambda_function[test_idx],
  lambda_pred_test
)
text(
  min(true_lambda_function[test_idx]),
  max(true_lambda_function[test_idx]),
  paste('Correlation:', round(cor_test, 3)),
  adj = 0,
  col = 'red'
)

# Compare estimated vs true class probabilities for train and test sets
mean_probs_train <- rowMeans(est_probs_train)
plot(
  true_probs[train_idx, 2],
  mean_probs_train,
  xlab = "True Prob (Train Set)",
  ylab = "Estimated Prob (Train Set)"
)
abline(a = 0, b = 1, col = 'blue', lwd = 2)
cor_train_prob <- cor(true_probs[train_idx, 2], mean_probs_train)
text(
  min(true_probs[train_idx, 2]),
  max(mean_probs_train),
  paste('Correlation:', round(cor_train_prob, 3)),
  adj = 0,
  col = 'red'
)
plot(
  true_probs[test_idx, 2],
  mean_probs_test <- rowMeans(est_probs_test),
  xlab = "True Prob (Test Set)",
  ylab = "Estimated Prob (Test Set)"
)
abline(a = 0, b = 1, col = 'blue', lwd = 2)
cor_test_prob <- cor(true_probs[test_idx, 2], mean_probs_test)
text(
  min(true_probs[test_idx, 2]),
  max(mean_probs_test),
  paste('Correlation:', round(cor_test_prob, 3)),
  adj = 0,
  col = 'red'
)

# Evaluate test set posterior interval coverage of lambda(x)
par(mfrow = c(1, 1))
preds <- predict(bart_model, X = X_test, terms = "y_hat", scale = "linear")
linear_interval <- compute_bart_posterior_interval(
  bart_model,
  X = X_test,
  terms = "y_hat",
  scale = "linear"
)
plot(rowMeans(preds), gamma_true[1] + true_lambda_function[test_idx])
points(
  linear_interval$lower,
  gamma_true[1] + true_lambda_function[test_idx],
  col = 'blue'
)
points(
  linear_interval$upper,
  gamma_true[1] + true_lambda_function[test_idx],
  col = 'blue'
)
abline(0, 1)
linear_coverage <- (((linear_interval$lower) <=
  gamma_true[1] + true_lambda_function[test_idx]) &
  ((linear_interval$upper) >= gamma_true[1] + true_lambda_function[test_idx]))
mean(linear_coverage)

# Evaluate test set posterior interval coverage of survival function
probability_interval <- compute_bart_posterior_interval(
  bart_model,
  X = X_test,
  terms = "y_hat",
  scale = "probability"
)
probability_coverage <- (((probability_interval$lower) <=
  true_probs[test_idx, 2]) &
  ((probability_interval$upper) >= true_probs[test_idx, 2]))
mean(probability_coverage)

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
hist(est_probs_test[max_ind, ])
