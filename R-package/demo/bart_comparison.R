################################################################################
## Brief demo script for comparing stochtree BART implementation with
## the implementation in the BART package
################################################################################

# Load BART
library(BART)

# Run a single iteration of the comparison and plot the results

# Generate simulated data and split into training and prediction sets
n <- 500
p <- 3
num_samples <- 50
num_burnin <- 100
num_trees <- 200
train_inds <- sample(1:n, round(n*0.8), replace = F)
test_inds <- (1:n)[!((1:n) %in% train_inds)]
X <- matrix(runif(n*p), ncol=p)
y <- X[,1] * 5 + X[,2] * 500 + rnorm(n, 0, 1)
Xtrain <- X[train_inds,]
Xtest <- X[test_inds,]
ytrain <- y[train_inds]
ytest <- y[test_inds]
model_matrix_train <- cbind(ytrain, Xtrain)
model_matrix_test <- cbind(ytest, Xtest)

# Sample from BART and evaluate its predictions on out of sample data
bartFit = wbart(Xtrain,ytrain,nskip=num_burnin,ndpost=num_samples,ntree=num_trees)
bart_predictions = t(predict(bartFit, Xtest))
bart_avg_prediction <- rowMeans(bart_predictions)
(bart_rmse <- sqrt(mean((bart_avg_prediction - ytest)^2)))
# for (j in 1:ncol(bart_predictions)) {plot(ytest, bart_predictions[,j], ylab = paste0("yhat_",j)); abline(0,1); Sys.sleep(0.25)}
# plot(ytest, bart_avg_prediction, ylab = "yhat_mean"); abline(0,1)

