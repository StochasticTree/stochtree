# Load the library
library(stochtree)

# Generate the data
n <- 500
p_x <- 10
p_w <- 1
snr <- 3
X <- matrix(runif(n*p_x), ncol = p_x)
W <- matrix(runif(n*p_w), ncol = p_w)
group_ids <- rep(c(1,2), n %/% 2)
rfx_coefs <- matrix(c(-5, -3, 5, 3),nrow=2,byrow=T)
rfx_basis <- cbind(1, runif(n, -1, 1))
f_XW <- (
    ((0 <= X[,1]) & (0.25 > X[,1])) * (-7.5*W[,1]) + 
        ((0.25 <= X[,1]) & (0.5 > X[,1])) * (-2.5*W[,1]) + 
        ((0.5 <= X[,1]) & (0.75 > X[,1])) * (2.5*W[,1]) + 
        ((0.75 <= X[,1]) & (1 > X[,1])) * (7.5*W[,1])
)
rfx_term <- rowSums(rfx_coefs[group_ids,] * rfx_basis)
noise_sd <- sd(f_XW) / snr
y <- f_XW + rfx_term + rnorm(n, 0, 1)*noise_sd

# Split data into test and train sets
test_set_pct <- 0.2
n_test <- round(test_set_pct*n)
n_train <- n - n_test
test_inds <- sort(sample(1:n, n_test, replace = F))
train_inds <- (1:n)[!((1:n) %in% test_inds)]
X_test <- X[test_inds,]
X_train <- X[train_inds,]
W_test <- W[test_inds,]
W_train <- W[train_inds,]
y_test <- y[test_inds]
y_train <- y[train_inds]
group_ids_test <- group_ids[test_inds]
group_ids_train <- group_ids[train_inds]
rfx_basis_test <- rfx_basis[test_inds,]
rfx_basis_train <- rfx_basis[train_inds,]

# Sample a BART model
num_gfr <- 10
num_burnin <- 0
num_mcmc <- 10
num_samples <- num_gfr + num_burnin + num_mcmc
bart_model <- stochtree::bart(
    X_train = X_train, W_train = W_train, y_train = y_train, 
    group_ids_train = group_ids_train, rfx_basis_train = rfx_basis_train, 
    X_test = X_test, W_test = W_test, group_ids_test = group_ids_test,
    rfx_basis_test = rfx_basis_test, num_trees = 100, 
    num_gfr = num_gfr, num_burnin = num_burnin, num_mcmc = num_mcmc, 
    sample_sigma = T, sample_tau = T
)
rfx_preds_test_orig <- bart_model$rfx_preds_test
forest_preds_test_orig <- bart_model$y_hat_test - rfx_preds_test_orig
y_scale <- bart_model$model_params$outcome_scale
y_bar <- bart_model$model_params$outcome_mean
sigma2_samples <- bart_model$sigma2_samples

# Convert to json
jsonobj <- createCppJson()
jsonobj$add_forest(bart_model$forests)
jsonobj$add_random_effects(bart_model$rfx_samples)
jsonobj$add_scalar("y_scale", bart_model$model_params$outcome_scale)
jsonobj$add_scalar("y_bar", bart_model$model_params$outcome_mean)
jsonobj$add_vector("sigma2_samples", bart_model$sigma2_samples, "parameters")
jsonobj$add_random_effects(bart_model$rfx_samples)
jsonobj$save_file("test.json")

# Check scalars
jsonobj$get_scalar("y_scale") - y_scale
jsonobj$get_scalar("y_bar") - y_bar

# Check vectors
jsonobj$get_vector("sigma2_samples", "parameters") - bart_model$sigma2_samples

# Check ForestSamples
forest_samples_roundtrip <- loadForestContainerJson(jsonobj, "forest_0")
forest_dataset_test_roundtrip <- createForestDataset(X_test, W_test)
forest_preds_test_roundtrip <- forest_samples_roundtrip$predict(forest_dataset_test_roundtrip)*y_scale + y_bar
plot(rowMeans(forest_preds_test_orig), rowMeans(forest_preds_test_roundtrip)); abline(0,1,col="red",lwd=3,lty=3)

# Check RandomEffectsSamples
rfx_samples_roundtrip <- loadRandomEffectSamplesJson(jsonobj, 0)
rfx_preds_test_roundtrip <- rfx_samples_roundtrip$predict(group_ids_test, rfx_basis_test)*y_scale
plot(rowMeans(rfx_preds_test_orig), rowMeans(rfx_preds_test_roundtrip)); abline(0,1,col="red",lwd=3,lty=3)

# Now, test the full file serialization roundtrip
json_filename <- "test.json"
jsonobj$save_file(json_filename)
jsonobj_reload <- createCppJsonFile(json_filename)

# Check ForestSamples
forest_samples_roundtrip <- loadForestContainerJson(jsonobj_reload, "forest_0")
forest_preds_test_roundtrip <- forest_samples_roundtrip$predict(forest_dataset_test_roundtrip)*y_scale + y_bar
plot(rowMeans(forest_preds_test_orig), rowMeans(forest_preds_test_roundtrip)); abline(0,1,col="red",lwd=3,lty=3)

# Check RandomEffectsSamples
rfx_samples_roundtrip <- loadRandomEffectSamplesJson(jsonobj_reload, 0)
rfx_preds_test_roundtrip <- rfx_samples_roundtrip$predict(group_ids_test, rfx_basis_test)*y_scale
plot(rowMeans(rfx_preds_test_orig), rowMeans(rfx_preds_test_roundtrip)); abline(0,1,col="red",lwd=3,lty=3)
