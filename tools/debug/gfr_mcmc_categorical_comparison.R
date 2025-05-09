################################################################################
## Comparison of GFR / warm start with pure MCMC on datasets with a 
## mix of numeric features and low-cardinality categorical features.
################################################################################

# Load libraries
library(stochtree)

# Generate data
n <- 500
p_continuous <- 5
p_binary <- 2
p_ordered_cat <- 2
p <- p_continuous + p_binary + p_ordered_cat
stopifnot(p_continuous >= 3)
stopifnot(p_binary >= 2)
stopifnot(p_ordered_cat >= 1)
x_continuous <- matrix(
    runif(n*p_continuous), 
    ncol = p_continuous
)
x_binary <- matrix(
    rbinom(n*p_binary, size = 1, prob = 0.5), 
    ncol = p_binary
)
x_ordered_cat <- matrix(
    sample(1:5, size = n*p_ordered_cat, replace = T), 
    ncol = p_ordered_cat
)
X_matrix <- cbind(x_continuous, x_binary, x_ordered_cat)
X_df <- as.data.frame(X_matrix)
colnames(X_df) <- paste0("x", 1:p)
for (i in (p_continuous+1):(p_continuous+p_binary+p_ordered_cat)) {
    X_df[,i] <- factor(X_df[,i], ordered = T)
}
f_x_cont <- (2 + 4*x_continuous[,1] - 6*(x_continuous[,2] < 0) + 
             6*(x_continuous[,2] >= 0) + 5*(abs(x_continuous[,3]) - sqrt(2/pi)))
f_x_binary <- -1.5 + 1*x_binary[,1] + 2*x_binary[,2]
f_x_ordered_cat <- 3 - 1*x_ordered_cat[,1]
pct_var_cont <- 1/3
pct_var_binary <- 1/3
pct_var_ordered_cat <- 1/3
stopifnot(pct_var_cont + pct_var_binary + pct_var_ordered_cat == 1.0)
total_var <- var(f_x_cont+f_x_binary+f_x_ordered_cat)
f_x_cont_rescaled <- f_x_cont * sqrt(
    pct_var_cont / (var(f_x_cont) / total_var)
)
f_x_binary_rescaled <- f_x_binary * sqrt(
    pct_var_binary / (var(f_x_binary) / total_var)
)
f_x_ordered_cat_rescaled <- f_x_ordered_cat * sqrt(
    pct_var_ordered_cat / (var(f_x_ordered_cat) / total_var)
)
E_y <- f_x_cont_rescaled + f_x_binary_rescaled + f_x_ordered_cat_rescaled
# var(f_x_cont_rescaled) / var(E_y)
# var(f_x_binary_rescaled) / var(E_y)
# var(f_x_ordered_cat_rescaled) / var(E_y)
snr <- 3
epsilon <- rnorm(n, 0, 1) * sd(E_y) / snr
y <- E_y + epsilon
jitter_eps <- 0.1
x_binary_jitter <- x_binary + matrix(
    runif(n*p_binary, -jitter_eps, jitter_eps), ncol = p_binary
)
x_ordered_cat_jitter <- x_ordered_cat + matrix(
    runif(n*p_ordered_cat, -jitter_eps, jitter_eps), ncol = p_ordered_cat
)
X_matrix_jitter <- cbind(x_continuous, x_binary_jitter, x_ordered_cat_jitter)
X_df_jitter <- as.data.frame(X_matrix_jitter)
colnames(X_df_jitter) <- paste0("x", 1:p)

# Test-train split
test_set_pct <- 0.2
n_test <- round(test_set_pct*n)
n_train <- n - n_test
test_inds <- sort(sample(1:n, n_test, replace = FALSE))
train_inds <- (1:n)[!((1:n) %in% test_inds)]
X_df_test <- X_df[test_inds,]
X_df_train <- X_df[train_inds,]
X_df_jitter_test <- X_df_jitter[test_inds,]
X_df_jitter_train <- X_df_jitter[train_inds,]
y_test <- y[test_inds]
y_train <- y[train_inds]

# Fit BART with warmstart on the original data
ws_bart_fit <- bart(X_train = X_df_train, y_train = y_train, 
                    X_test = X_df_test, num_gfr = 15, 
                    num_burnin = 0, num_mcmc = 100)

# Fit BART with MCMC only on the original data
bart_fit <- bart(X_train = X_df_train, y_train = y_train, 
                 X_test = X_df_test, num_gfr = 0, 
                 num_burnin = 2000, num_mcmc = 100)

# Fit BART with warmstart on the jittered data
ws_bart_jitter_fit <- bart(X_train = X_df_jitter_train, y_train = y_train, 
                           X_test = X_df_jitter_test, num_gfr = 15, 
                           num_burnin = 0, num_mcmc = 100)

# Fit BART with MCMC only on the jittered data
bart_jitter_fit <- bart(X_train = X_df_jitter_train, y_train = y_train, 
                        X_test = X_df_jitter_test, num_gfr = 0, 
                        num_burnin = 2000, num_mcmc = 100)

# Compare the variable split counds
ws_bart_fit$mean_forests$get_aggregate_split_counts(p)
bart_fit$mean_forests$get_aggregate_split_counts(p)
ws_bart_jitter_fit$mean_forests$get_aggregate_split_counts(p)
bart_jitter_fit$mean_forests$get_aggregate_split_counts(p)

# Compute out-of-sample RMSE
sqrt(mean((rowMeans(ws_bart_fit$y_hat_test) - y_test)^2))
sqrt(mean((rowMeans(bart_fit$y_hat_test) - y_test)^2))
sqrt(mean((rowMeans(ws_bart_jitter_fit$y_hat_test) - y_test)^2))
sqrt(mean((rowMeans(bart_jitter_fit$y_hat_test) - y_test)^2))

# Compare sigma traceplots
sigma_min <- min(c(ws_bart_fit$sigma2_global_samples, 
                   bart_fit$sigma2_global_samples, 
                   ws_bart_jitter_fit$sigma2_global_samples, 
                   bart_jitter_fit$sigma2_global_samples))
sigma_max <- max(c(ws_bart_fit$sigma2_global_samples, 
                   bart_fit$sigma2_global_samples, 
                   ws_bart_jitter_fit$sigma2_global_samples, 
                   bart_jitter_fit$sigma2_global_samples))
plot(ws_bart_fit$sigma2_global_samples, 
     ylim = c(sigma_min - 0.1, sigma_max + 0.1), 
     type = "line", col = "black")
lines(bart_fit$sigma2_global_samples, col = "blue")
lines(ws_bart_jitter_fit$sigma2_global_samples, col = "green")
lines(bart_jitter_fit$sigma2_global_samples, col = "red")
