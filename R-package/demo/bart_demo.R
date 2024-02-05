################################################################################
## Brief BART demo script
################################################################################

# Load stochtree
library(stochtree)

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
rfx_labels <- as.integer(rep(c(1,2), times = n/2))
rfx_basis <- rep(1, times = n)
omega <- runif(n)
betas <- c(-5, -10, 10, 5)
f_x_omega <- ifelse(
    (X[,1] < 0.5) & (X[,2] < 0.5), betas[1] * omega,
    ifelse(
        (X[,1] >= 0.5) & (X[,2] < 0.5), betas[2] * omega,
        ifelse(
            (X[,1] < 0.5) & (X[,1] >= 0.5), betas[3] * omega, betas[4] * omega
        )
    )
)
rfx <- ifelse(rfx_labels == 1, -1, 5)
y <- f_x_omega + rfx + rnorm(n, 0, 1)
Xtrain <- X[train_inds,]
Xtest <- X[test_inds,]
ytrain <- y[train_inds]
ytest <- y[test_inds]
omegatrain <- omega[train_inds]
omegatest <- omega[test_inds]
rfx_labels_train <- rfx_labels[train_inds]
rfx_labels_test <- rfx_labels[test_inds]
rfx_basis_train <- rfx_basis[train_inds]
rfx_basis_test <- rfx_basis[test_inds]

# Sample from BART and evaluate its predictions on out of sample data
nu <- 0.5
lambda <- 2.
a_rfx <- 1.
b_rfx <- 1.
stochtree_samples <- bart(ytrain, Xtrain, omegatrain, rfx_basis_train, rfx_labels_train, num_samples, num_burnin, num_trees, nu, lambda, a_rfx, b_rfx, random_seed = random_seed)
stochtree_predictions <- predict(stochtree_samples, Xtest, omegatest, rfx_basis_test, rfx_labels_test, num_samples)
stochtree_avg_prediction <- rowMeans(stochtree_predictions)
(stochtree_rmse <- sqrt(mean((stochtree_avg_prediction - ytest)^2)))
for (j in 1:ncol(stochtree_predictions)) {plot(ytest, stochtree_predictions[,j], ylab = paste0("yhat_",j)); abline(0,1); Sys.sleep(0.25)}
plot(ytest, rowMeans(stochtree_predictions), ylab = "yhat_mean"); abline(0,1)
