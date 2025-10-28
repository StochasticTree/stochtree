# Simulate ordinal data and run Cloglog Ordinal BART

# Load
library(stochtree)

set.seed(2025)

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
y <- sapply(1:nrow(X), function(i) sample(1:n_categories, 1, prob = true_probs[i, ]))
cat("Outcome distribution:", table(y), "\n")

# Train test split
train_idx <- sample(1:n, size = floor(0.8 * n))
test_idx <- setdiff(1:n, train_idx)
X_train <- X[train_idx, ]
y_train <- y[train_idx]
X_test <- X[test_idx, ]
y_test <- y[test_idx]

start <- Sys.time()

# Sample the cloglog ordinal BART model
out <- cloglog_ordinal_bart(
  X = X_train,
  y = y_train,
  X_test = X_test,
  num_gfr = 10,
  num_burnin = 0,
  num_mcmc = 1000,
  n_thin = 1
)

end <- Sys.time()
print(end - start)

# Inference and diagnostics
par(mfrow = c(2, 1))
plot(out$gamma_samples[1, ], type = 'l', main = expression(gamma[1]), ylab = "Value", xlab = "MCMC Sample")
abline(h = gamma_true[1], col = 'red', lty = 2)
plot(out$gamma_samples[2, ], type = 'l', main = expression(gamma[2]), ylab = "Value", xlab = "MCMC Sample")
abline(h = gamma_true[2], col = 'red', lty = 2)

gamma1 <- out$gamma_samples[1,] + colMeans(out$forest_predictions_train)
summary(gamma1)
hist(gamma1)

gamma2 <- out$gamma_samples[2,] + colMeans(out$forest_predictions_train)
summary(gamma2)
hist(gamma2)

par(mfrow = c(3,2), mar = c(5,4,1,1))
rowMeans(out$gamma_samples)
moo <- t(out$gamma_samples) + colMeans(out$forest_predictions_train)
plot(moo[,1])
abline(h = gamma_true[1] + mean(true_lambda_function[train_idx]))
plot(moo[,2])
abline(h = gamma_true[2] + mean(true_lambda_function[train_idx]))
plot(out$gamma_samples[1,])
plot(out$gamma_samples[2,])

# Compare forest predictions with the truth function (for training and test sets)
par(mfrow = c(2,1))
lambda_pred_train <- rowMeans(out$forest_predictions_train) - mean(out$forest_predictions_train)
plot(lambda_pred_train, true_lambda_function[train_idx])
abline(a=0,b=1,col='blue', lwd=2)
cor_train <- cor(true_lambda_function[train_idx], lambda_pred_train)
text(min(true_lambda_function[train_idx]), max(true_lambda_function[train_idx]), paste('Correlation:', round(cor_train, 3)), adj = 0, col = 'red')

lambda_pred_test <- rowMeans(out$forest_predictions_test) - mean(out$forest_predictions_test)
plot(lambda_pred_test, true_lambda_function[test_idx])
abline(a=0,b=1,col='blue', lwd=2)
cor_test <- cor(true_lambda_function[test_idx], lambda_pred_test)
text(min(true_lambda_function[test_idx]), max(true_lambda_function[test_idx]), paste('Correlation:', round(cor_test, 3)), adj = 0, col = 'red')

# Estimated ordinal class probabilities for the training set
est_probs_train <- matrix(0, nrow=length(train_idx), ncol=n_categories)
for (j in 1:n_categories) {
  if (j == 1) {
    est_probs_train[, j] <- rowMeans(1 - exp(-exp(out$forest_predictions_train + out$gamma_samples[j, ])))
  } else if (j == n_categories) {
    est_probs_train[, j] <- 1 - rowSums(est_probs_train[, 1:(j - 1), drop = FALSE])
  } else {
    est_probs_train[, j] <- rowMeans(exp(-exp(out$forest_predictions_train + out$gamma_samples[j-1,])) *
                                       (1 - exp(-exp(out$forest_predictions_train + out$gamma_samples[j,]))))
  }
}

mean(log(-log(1 - est_probs_train[, 1])) - rowMeans(out$forest_predictions_train))

# Compare estimated vs true class probabilities for training set
for (j in 1:n_categories) {
  plot(true_probs[train_idx, j], est_probs_train[, j], xlab = paste("True Prob Category", j), ylab = paste("Estimated Prob Category", j))
  abline(a = 0, b = 1, col = 'blue', lwd = 2)
  cor_train_prob <- cor(true_probs[train_idx, j], est_probs_train[, j])
  text(min(true_probs[train_idx, j]), max(est_probs_train[, j]), paste('Correlation:', round(cor_train_prob, 3)), adj = 0, col = 'red')
}

# Estimated ordinal class probabilities for the test set
est_probs_test <- matrix(0, nrow=length(test_idx), ncol=n_categories)
for (j in 1:n_categories) {
  if (j == 1) {
    est_probs_test[, j] <- rowMeans(1 - exp(-exp(out$forest_predictions_test + out$gamma_samples[j, ])))
  } else if (j == n_categories) {
    est_probs_test[, j] <- 1 - rowSums(est_probs_test[, 1:(j - 1), drop = FALSE])
  } else {
    est_probs_test[, j] <- rowMeans(exp(-exp(out$forest_predictions_test + out$gamma_samples[j-1,])) *
                                       (1 - exp(-exp(out$forest_predictions_test + out$gamma_samples[j,]))))
  }
}

mean(log(-log(1 - est_probs_test[, 1])) - rowMeans(out$forest_predictions_test))

# Compare estimated vs true class probabilities for test set
for (j in 1:n_categories) {
  plot(true_probs[test_idx, j], est_probs_test[, j], xlab = paste("True Prob Category", j), ylab = paste("Estimated Prob Category", j))
  abline(a = 0, b = 1, col = 'blue', lwd = 2)
  cor_test_prob <- cor(true_probs[test_idx, j], est_probs_test[, j])
  text(min(true_probs[test_idx, j]), max(est_probs_test[, j]), paste('Correlation:', round(cor_test_prob, 3)), adj = 0, col = 'red')
}
