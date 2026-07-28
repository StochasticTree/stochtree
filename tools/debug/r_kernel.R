library(stochtree)
library(tgp)

# Generate the data, add many "noise variables"
n <- 500
p_extra <- 10
friedman.df <- friedman.1.data(n=n)
train_inds <- sort(sample(1:n, floor(0.8*n), replace = FALSE))
test_inds <- (1:n)[!((1:n) %in% train_inds)]
X <- as.matrix(friedman.df)[,1:10]
X <- cbind(X, matrix(runif(n*p_extra), ncol = p_extra))
y <- as.matrix(friedman.df)[,12] + rnorm(n,0,1)*(sd(as.matrix(friedman.df)[,11])/2)
X_train <- X[train_inds,]
X_test <- X[test_inds,]
y_train <- y[train_inds]
y_test <- y[test_inds]

# Run BART on the data
X_train <- as.data.frame(X_train)
X_test <- as.data.frame(X_test)
bart_params <- list(num_trees_mean=200, num_trees_variance=50)
bart_model <- bart(X_train=X_train, y_train=y_train, X_test=X_test, params = bart_params, num_mcmc=1000)

# Compute leaf indices for selected samples from the mean forest
leaf_mat <- computeForestLeafIndices(bart_model, X_test, forest_type = "mean", 
                                     forest_inds = c(99,100))

# Compute leaf indices for all samples from the mean forest
leaf_mat <- computeForestLeafIndices(bart_model, X_test, forest_type = "mean")

# Construct sparse matrix of leaf membership
W <- Matrix::sparseMatrix(i=rep(1:length(y_test),200), j=leaf_mat[,forest_num] + 1, x=1)
tcrossprod(W)

# Compute leaf indices for selected samples from the variance forest
leaf_mat <- computeForestLeafIndices(bart_model, X_test, forest_type = "variance", 
                                     forest_inds = c(99,100))

# Compute leaf indices for all samples from the variance forest
leaf_mat <- computeForestLeafIndices(bart_model, X_test, forest_type = "variance")