# Comparison script for R and Python BCF
library(stochtree)

# # Generate data
# n <- 1000
# p <- 5
# X <- matrix(runif(n*p),ncol=p)
# pi_x <- 0.25 + 0.5*X[,1]
# mu_x <- 5*pi_x
# tau_x <- X[,2]*2
# Z <- rbinom(n,1,pi_x)
# E_XZ <- mu_x + Z*tau_x
# y <- E_XZ + rnorm(n, 0, 1)
# 
# # Split data into test and train sets
# test_set_pct <- 0.5
# n_test <- round(test_set_pct*n)
# n_train <- n - n_test
# test_inds <- sort(sample(1:n, n_test, replace = FALSE))
# train_inds <- (1:n)[!((1:n) %in% test_inds)]
# X_test <- X[test_inds,]
# X_train <- X[train_inds,]
# pi_test <- pi_x[test_inds]
# pi_train <- pi_x[train_inds]
# Z_test <- Z[test_inds]
# Z_train <- Z[train_inds]
# y_test <- y[test_inds]
# y_train <- y[train_inds]
# mu_test <- mu_x[test_inds]
# mu_train <- mu_x[train_inds]
# tau_test <- tau_x[test_inds]
# tau_train <- tau_x[train_inds]
# 
# # Save to CSV
# column_names <- c(paste0("X",1:ncol(X_test)), "Z", "y", "pi", "mu", "tau")
# test_df <- data.frame(
#     X = X_test, 
#     Z = Z_test, 
#     y = y_test, 
#     pi = pi_test, 
#     mu = mu_test, 
#     tau = tau_test
# )
# train_df <- data.frame(
#     X = X_train, 
#     Z = Z_train, 
#     y = y_train, 
#     pi = pi_train, 
#     mu = mu_train, 
#     tau = tau_train
# )
# colnames(test_df) <- column_names
# colnames(train_df) <- column_names
# write.csv(test_df, file = "tools/data/python_r_debug_test.csv")
# write.csv(train_df, file = "tools/data/python_r_debug_train.csv")

# Load data from CSV
test_df <- read.csv("tools/data/python_r_debug_test.csv")
train_df <- read.csv("tools/data/python_r_debug_train.csv")
X_test <- as.matrix(test_df[,c("X1","X2","X3","X4","X5")])
X_train <- as.matrix(test_df[,c("X1","X2","X3","X4","X5")])
Z_test <- test_df[,c("Z")]
Z_train <- test_df[,c("Z")]
y_test <- test_df[,c("y")]
y_train <- test_df[,c("y")]
pi_test <- test_df[,c("pi")]
pi_train <- test_df[,c("pi")]
mu_test <- test_df[,c("mu")]
mu_train <- test_df[,c("mu")]
tau_test <- test_df[,c("tau")]
tau_train <- test_df[,c("tau")]

# Run BCF
num_gfr <- 10
num_burnin <- 0
num_mcmc <- 100
num_samples <- num_gfr + num_burnin + num_mcmc
bcf_model_warmstart <- bcf(
    X_train = X_train, Z_train = Z_train, y_train = y_train, pi_train = pi_train, 
    X_test = X_test, Z_test = Z_test, pi_test = pi_test,  
    num_gfr = num_gfr, num_burnin = num_burnin, num_mcmc = num_mcmc, 
    sample_sigma_leaf_mu = T, sample_sigma_leaf_tau = F
)

# Plot results
plot(rowMeans(bcf_model_warmstart$mu_hat_test), mu_test,
     xlab = "predicted", ylab = "actual", main = "Prognostic function")
abline(0,1,col="red",lty=3,lwd=3)
plot(rowMeans(bcf_model_warmstart$tau_hat_test), tau_test,
     xlab = "predicted", ylab = "actual", main = "Treatment effect")
abline(0,1,col="red",lty=3,lwd=3)
plot(rowMeans(bcf_model_warmstart$y_hat_test), y_test,
     xlab = "predicted", ylab = "actual", main = "Outcome")
abline(0,1,col="red",lty=3,lwd=3)
sigma_observed <- 1
plot_bounds <- c(min(c(bcf_model_warmstart$sigma2_samples, sigma_observed)),
                 max(c(bcf_model_warmstart$sigma2_samples, sigma_observed)))
plot(bcf_model_warmstart$sigma2_samples, ylim = plot_bounds,
     ylab = "sigma^2", xlab = "Sample", main = "Global variance parameter")
abline(h = sigma_observed, lty=3, lwd = 3, col = "blue")
ymin <- min(c(min(bcf_model_warmstart$b_0_samples), min(bcf_model_warmstart$b_1_samples)))
ymax <- max(c(max(bcf_model_warmstart$b_0_samples), max(bcf_model_warmstart$b_1_samples)))
plot(bcf_model_warmstart$b_0_samples, ylim = c(ymin, ymax), col = "blue",
     ylab = "Coding parameters", xlab = "Sample", main = "Coding parameters")
points(bcf_model_warmstart$b_1_samples, ylim = plot_bounds, col = "orange")

# Evaluate RMSEs
sqrt(mean((rowMeans(bcf_model_warmstart$y_hat_test) - y_test)^2))
sqrt(mean((rowMeans(bcf_model_warmstart$mu_hat_test) - mu_test)^2))
sqrt(mean((rowMeans(bcf_model_warmstart$tau_hat_test) - tau_test)^2))
