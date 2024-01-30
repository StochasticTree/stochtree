################################################################################
## Brief BART demo script
################################################################################

# Load stochtree
library(stoch.tree)

# Set seed
random_seed = 12
set.seed(random_seed)

# Run a single iteration of the comparison and plot the results

# Generate simulated data and split into training and prediction sets
n <- 500
p <- 20
num_samples <- 50
num_burnin <- 10000
num_trees <- 50
train_inds <- sample(1:n, round(n*0.8), replace = F)
test_inds <- (1:n)[!((1:n) %in% train_inds)]
X <- matrix(runif(n*p), ncol=p)
omega <- runif(n)
betas <- c(-5, -10, 10, 5)
y <- ifelse(
    (X[,1] >= 0) & (X[,1] < 0.25), betas[1] * omega,
    ifelse(
        (X[,1] >= 0.25) & (X[,1] < 0.5), betas[2] * omega,
        ifelse(
            (X[,1] >= 0.5) & (X[,1] < 0.75), betas[3] * omega, betas[4] * omega
        )
    )
) + rnorm(n, 0, 1)
Xtrain <- X[train_inds,]
Xtest <- X[test_inds,]
ytrain <- y[train_inds]
ytest <- y[test_inds]
omegatrain <- omega[train_inds]
omegatest <- omega[test_inds]

# Sample from BART and evaluate its predictions on out of sample data
stochtree_samples <- bart(ytrain, Xtrain, omegatrain, num_samples, num_burnin, num_trees, 0.5, 2, random_seed = random_seed)
stochtree_predictions <- predict(stochtree_samples, Xtest, omegatest, num_samples)
stochtree_avg_prediction <- rowMeans(stochtree_predictions)
(stochtree_rmse <- sqrt(mean((stochtree_avg_prediction - ytest)^2)))
for (j in 1:ncol(stochtree_predictions)) {plot(ytest, stochtree_predictions[,j], ylab = paste0("yhat_",j)); abline(0,1); Sys.sleep(0.25)}
plot(ytest, rowMeans(stochtree_predictions), ylab = "yhat_mean"); abline(0,1)
