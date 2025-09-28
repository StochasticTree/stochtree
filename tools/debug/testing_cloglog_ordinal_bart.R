# Simulate ordinal data and run Cloglog Ordinal BART

# Load
library(stochtree)

set.seed(2025)

# Simulation
n_samples <- 2000
n_features <- 5
n_categories <- 3

X <- matrix(rnorm(n_samples * n_features), n_samples, n_features)

beta <- rep(1 / sqrt(n_features), n_features)
gamma_true <- c(-2, 1)

linear_predictor <- X %*% beta

# Transform linear predictor using the complementary log-log link function
p_0 <- 1 - exp(-exp(gamma_true[1] + linear_predictor))
p_1 <- exp(-exp(gamma_true[1] + linear_predictor)) *
  (1 - exp(-exp(gamma_true[2] + linear_predictor)))
p_2 <- exp(-exp(gamma_true[1] + linear_predictor)) *
  exp(-exp(gamma_true[2] + linear_predictor))

true_probs <- cbind(p_0, p_1, p_2)

# Get Outcomes
ordinal_outcome <- sapply(1:nrow(X), function(i) sample(1:n_categories, 1, prob = true_probs[i, ]))
cat("Outcome distribution:", table(ordinal_outcome), "\n")

train_index <- 1:(n_samples/2)
test_index <- (1:n_samples)[- train_index]

X_train <- X[train_index, ]
y_train <- ordinal_outcome[train_index]
X_test <- X[-train_index, ]
y_test <- ordinal_outcome[-train_index]

out <- cloglog_ordinal_bart(
  X = X_train,
  y = y_train,
  X_test = X_test,
  n_samples_mcmc = 1000,
  n_burnin = 500,
  n_thin = 1
)


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
abline(h = gamma_true[1] + mean(linear_predictor[train_index]))
plot(moo[,2])
abline(h = gamma_true[2] + mean(linear_predictor[train_index]))
plot(out$gamma_samples[1,])
plot(out$gamma_samples[2,])

# Compare forest predictions with the truth

plot(rowMeans(out$forest_predictions_train) - mean(out$forest_predictions_train), linear_predictor[train_index])
abline(a=0,b=1,col='blue', lwd=2)

plot(rowMeans(out$forest_predictions_test) - mean(out$forest_predictions_test), linear_predictor[test_index])
abline(a=0,b=1,col='blue', lwd=2)

# Train set ordinal class probabilities

p_hat_0 <- rowMeans(1 - exp(-exp(out$forest_predictions_train + out$gamma_samples[1, ])))
p_hat_1 <- rowMeans((1 - exp(-exp(out$forest_predictions_train + out$gamma_samples[2,]))) * exp(-exp(out$forest_predictions_train + out$gamma_samples[1,])))
p_hat_2 <- 1 - p_hat_1 - p_hat_0

mean(log(-log(1 - p_hat_0)) - rowMeans(out$forest_predictions_train))

plot(p_hat_0, p_0[train_index])
abline(a=0,b=1,col='blue', lwd=2)
plot(p_hat_1, p_1[train_index])
abline(a=0,b=1,col='blue', lwd=2)
plot(p_hat_2, p_2[train_index])
abline(a=0,b=1,col='blue', lwd=2)

# Test set ordinal class probabilities

p_hat_0 <- rowMeans(1 - exp(-exp(out$forest_predictions_test + out$gamma_samples[1, ])))
p_hat_1 <- rowMeans((1 - exp(-exp(out$forest_predictions_test + out$gamma_samples[2,]))) * exp(-exp(out$forest_predictions_test + out$gamma_samples[1,])))
p_hat_2 <- 1 - p_hat_1 - p_hat_0

plot(p_hat_0, p_0[test_index])
abline(a=0,b=1,col='blue', lwd=2)
plot(p_hat_1, p_1[test_index])
abline(a=0,b=1,col='blue', lwd=2)
plot(p_hat_2, p_2[test_index])
abline(a=0,b=1,col='blue', lwd=2)

