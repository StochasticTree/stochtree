# Load libraries
library(stochtree)
library(here)

# Load train and test data
project_dir <- here()
train_set_path <- file.path(project_dir, "debug", "data", "heterosked_train.csv")
test_set_path <- file.path(project_dir, "debug", "data", "heterosked_test.csv")
train_df <- read.csv(train_set_path)
test_df <- read.csv(test_set_path)
y_train <- train_df[,1]
y_test <- test_df[,1]
X_train <- train_df[,2:11]
X_test <- test_df[,2:11]
f_x_train <- train_df[,12]
f_x_test <- test_df[,12]
s_x_train <- train_df[,13]
s_x_test <- test_df[,13]

# Run BART
num_gfr <- 10
num_burnin <- 0
num_mcmc <- 200
num_samples <- num_gfr + num_burnin + num_mcmc
bart_model <- stochtree::bart(
    X_train = X_train, y_train = y_train, X_test = X_test,
    num_gfr = num_gfr, num_burnin = num_burnin, num_mcmc = num_mcmc,
    num_trees_mean = 0, num_trees_variance = 20,
    alpha_mean = 0.8, beta_mean = 3, min_samples_leaf_mean = 5,
    max_depth_mean = 3, alpha_variance = 0.95, beta_variance = 0.1,
    min_samples_leaf_variance = 1, max_depth_variance = 10,
    sample_sigma = T, sample_tau = F, keep_gfr = F
)

s_x_hat_train <- rowMeans(bart_model$sigma_x_hat_train)
# plot(s_x_hat_train, s_x_train, main = "Conditional std dev as a function of x", xlab = "Predicted", ylab = "Actual"); abline(0,1,col="red",lty=3,lwd=3)
# sqrt(mean((s_x_hat_train - s_x_train)^2))

s_x_hat_test <- rowMeans(bart_model$sigma_x_hat_test)
plot(s_x_hat_test, s_x_test, main = "Conditional std dev as a function of x", xlab = "Predicted", ylab = "Actual"); abline(0,1,col="red",lty=3,lwd=3)
sqrt(mean((s_x_hat_test - s_x_test)^2))
