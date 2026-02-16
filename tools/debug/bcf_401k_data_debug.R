################################################################################
## Investigation of GFR vs MCMC fit issues on the 401k dataset
################################################################################

# Load libraries and set seed
library(stochtree)
library(DoubleML)
library(tidyverse)
# seed = 102
# set.seed(seed)

# Load 401k data
dat = DoubleML::fetch_401k(return_type = "data.frame")
dat_orig = dat

# Trim outliers
dat = dat %>% filter(abs(inc) < quantile(abs(inc), 0.9))

# Isolate covariates and convert to df
x = dat %>% dplyr::select(-c(e401, net_tfa))

# Convert to df and define categorical data types
xdf = data.frame(x)
xdf_st = xdf %>%
  mutate(
    age = factor(age, ordered = TRUE),
    inc = factor(inc, ordered = TRUE),
    educ = factor(educ, ordered = TRUE),
    fsize = factor(fsize, ordered = TRUE),
    marr = factor(marr, ordered = TRUE),
    twoearn = factor(twoearn, ordered = TRUE),
    db = factor(db, ordered = TRUE),
    pira = factor(pira, ordered = TRUE),
    hown = factor(hown, ordered = TRUE)
  )

# Isolate treatment and outcome
z = dat %>% dplyr::select(e401) %>% as.matrix()
y = dat %>% dplyr::select(net_tfa) %>% as.matrix()

# Define a "jittered" version of the original (integer-valued) x columns
# in which all categories are "upper-jittered" with uniform [0, eps] noise
# except for the largest category which is "lower-jittered" with [-eps, 0] noise
x_jitter = x
for (j in 1:ncol(x)) {
  min_diff <- min(diff(sort(x[, j]))[diff(sort(x[, j])) > 0])
  jitter_param <- min_diff / 3.0
  has_max_category <- x[, j] == max(x[, j])
  x_jitter[has_max_category, j] <- x[has_max_category, j] +
    runif(sum(has_max_category), -jitter_param, 0.0)
  x_jitter[!has_max_category, j] <- x[!has_max_category, j] +
    runif(sum(!has_max_category), 0.0, jitter_param)
}
# Visualize jitters
# for (j in 1:ncol(x)) {
#     plot(x[,j], x_jitter[,j], ylab = "jittered", xlab = "original")
#     unique_xs <- unique(x[,j])
#     for (i in unique_xs) {
#         abline(h = unique_xs[i], col = "red", lty = 3)
#     }
# }

# Fit a p(z = 1 | x) model for propensity features
general_params <- list(
  probit_outcome_model = TRUE,
  sample_sigma2_global = FALSE
)
mean_forest_params <- list(
  num_trees = 200
)
propensity_model <- bart(
  X_train = xdf,
  y_train = z,
  general_params = general_params,
  mean_forest_params = mean_forest_params
)
propensity = predict(
  propensity_model,
  X = xdf,
  type = "mean",
  terms = "y_hat",
  scale = "probability"
)

# Test-train split
n <- nrow(x)
test_set_pct <- 0.2
n_test <- round(test_set_pct * n)
n_train <- n - n_test
test_inds <- sort(sample(1:n, n_test, replace = FALSE))
train_inds <- (1:n)[!((1:n) %in% test_inds)]
xdf_st_test <- xdf_st[test_inds, ]
xdf_st_train <- xdf_st[train_inds, ]
x_test <- x[test_inds, ]
x_train <- x[train_inds, ]
x_jitter_test <- x_jitter[test_inds, ]
x_jitter_train <- x_jitter[train_inds, ]
pi_test <- propensity[test_inds]
pi_train <- propensity[train_inds]
z_test <- z[test_inds, ]
z_train <- z[train_inds, ]
y_test <- y[test_inds, ]
y_train <- y[train_inds, ]
y_train_scale <- scale(y_train)
y_train_sd <- attr(y_train_scale, "scaled:scale")
y_train_mean <- attr(y_train_scale, "scaled:center")
y_test_scale <- (y_test - y_train_mean) / y_train_sd

# Fit BCF with GFR algorithm on the jittered covariates
# and save model to JSON
num_gfr <- 1000
general_params <- list(
  adaptive_coding = FALSE,
  propensity_covariate = "none",
  keep_every = 1,
  verbose = TRUE,
  keep_gfr = TRUE
)
bcf_model_gfr <- stochtree::bcf(
  X_train = xdf_st_train,
  Z_train = z_train,
  y_train = y_train_scale,
  propensity_train = pi_train,
  X_test = xdf_st_test,
  Z_test = z_test,
  propensity_test = pi_test,
  num_gfr = num_gfr,
  num_burnin = 0,
  num_mcmc = 0,
  general_params = general_params
)
fit_json_gfr = saveBCFModelToJsonString(bcf_model_gfr)

# Run MCMC chain from the last GFR sample, setting covariate
# equal to an interpolation between the original x and x_jitter
# (alpha = 0 is 100% x_jitter and alpha = 1 is 100% x)
# alpha <- 1.0
# x_jitter_new_train <- (alpha) * x_train + (1-alpha) * x_jitter_train
# x_jitter_new_test <- (alpha) * x_test + (1-alpha) * x_jitter_test
x_jitter_new_train <- xdf_st_train
x_jitter_new_test <- xdf_st_test
num_mcmc <- 10000
bcf_model_mcmc <- stochtree::bcf(
  X_train = x_jitter_new_train,
  Z_train = z_train,
  y_train = y_train_scale,
  propensity_train = pi_train,
  X_test = x_jitter_new_test,
  Z_test = z_test,
  propensity_test = pi_test,
  num_gfr = 0,
  num_burnin = 0,
  num_mcmc = num_mcmc,
  previous_model_json = fit_json_gfr,
  previous_model_warmstart_sample_num = num_gfr,
  general_params = general_params
)

# Inspect the "in-sample sigma" via the traceplot
# of the global error variance parameter
combined_sigma <- c(
  bcf_model_gfr$sigma2_global_samples,
  bcf_model_mcmc$sigma2_global_samples
)
plot(
  combined_sigma,
  ylab = "sigma2",
  xlab = "sample num",
  main = "Global error var traceplot"
)

# Inspect the "out-of-sample sigma" by compute the MSE
# of the yhat on the test set
yhat_combined_train <- cbind(
  bcf_model_gfr$y_hat_train,
  bcf_model_mcmc$y_hat_train
)
yhat_combined_test <- cbind(
  bcf_model_gfr$y_hat_test,
  bcf_model_mcmc$y_hat_test
)
num_samples <- ncol(yhat_combined_train)
train_mses <- rep(NA, num_samples)
for (i in 1:num_samples) {
  train_mses[i] <- mean((yhat_combined_train[, i] - y_train_scale)^2)
}
test_mses <- rep(NA, num_samples)
for (i in 1:num_samples) {
  test_mses[i] <- mean((yhat_combined_test[, i] - y_test_scale)^2)
}
max_y <- max(c(max(train_mses, test_mses)))
min_y <- min(c(min(train_mses, test_mses)))
plot(
  test_mses,
  ylab = "outcome MSE",
  xlab = "sample num",
  main = "Outcome MSE Traceplot",
  ylim = c(min_y, max_y)
)
points(train_mses, col = "blue")
legend(
  "right",
  legend = c("Out-of-Sample", "In-Sample"),
  col = c("black", "blue"),
  pch = c(1, 1)
)

# Run some one-off pred vs actual plots
plot(yhat_combined_test[, 11000], y_test_scale)
abline(0, 1, col = "red", lty = 3)
plot(bcf_model_mcmc$y_hat_train[, 10000], y_train_scale)
abline(0, 1, col = "red", lty = 3)
plot(bcf_model_mcmc$y_hat_test[, 10000], y_test_scale)
abline(0, 1, col = "red", lty = 3)
plot(bcf_model_gfr$y_hat_train[, 1000], y_train_scale)
abline(0, 1, col = "red", lty = 3)
plot(bcf_model_gfr$y_hat_test[, 1000], y_test_scale)
abline(0, 1, col = "red", lty = 3)
plot(bcf_model_gfr$y_hat_train[, 10], y_train_scale)
abline(0, 1, col = "red", lty = 3)
plot(bcf_model_gfr$y_hat_test[, 10], y_test_scale)
abline(0, 1, col = "red", lty = 3)

# Run MCMC chain from root
num_mcmc <- 10000
bcf_model_mcmc_root <- stochtree::bcf(
  X_train = xdf_st_train,
  Z_train = z_train,
  y_train = y_train_scale,
  propensity_train = pi_train,
  X_test = xdf_st_test,
  Z_test = z_test,
  propensity_test = pi_test,
  num_gfr = 0,
  num_burnin = 0,
  num_mcmc = num_mcmc,
  general_params = general_params
)

# Inspect the "in-sample sigma" via the traceplot
# of the global error variance parameter
sigma_trace <- bcf_model_mcmc_root$sigma2_global_samples
plot(
  sigma_trace,
  ylab = "sigma2",
  xlab = "sample num",
  main = "Global error var traceplot"
)

# Inspect the "out-of-sample sigma" by compute the MSE
# of the yhat on the test set
yhat_combined_train <- cbind(
  bcf_model_mcmc_root$y_hat_train
)
yhat_combined_test <- cbind(
  bcf_model_mcmc_root$y_hat_test
)
num_samples <- ncol(yhat_combined_train)
train_mses <- rep(NA, num_samples)
for (i in 1:num_samples) {
  train_mses[i] <- mean((yhat_combined_train[, i] - y_train_scale)^2)
}
test_mses <- rep(NA, num_samples)
for (i in 1:num_samples) {
  test_mses[i] <- mean((yhat_combined_test[, i] - y_test_scale)^2)
}
max_y <- max(c(max(train_mses, test_mses)))
min_y <- min(c(min(train_mses, test_mses)))
plot(
  test_mses,
  ylab = "outcome MSE",
  xlab = "sample num",
  main = "Test set outcome MSEs",
  ylim = c(min_y, max_y)
)
points(train_mses, col = "blue")
