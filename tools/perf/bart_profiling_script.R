# Load libraries
library(stochtree)

# Capture command line arguments
args <- commandArgs(trailingOnly = T)
if (length(args) > 0) {
  n <- as.integer(args[1])
  p <- as.integer(args[2])
  num_gfr <- as.integer(args[3])
  num_mcmc <- as.integer(args[4])
  snr <- as.numeric(args[5])
  num_threads <- as.numeric(args[6])
} else {
  # Default arguments
  n <- 1000
  p <- 5
  num_gfr <- 10
  num_mcmc <- 100
  snr <- 3.0
  num_threads <- -1
}
cat(
  "n = ",
  n,
  "\np = ",
  p,
  "\nnum_gfr = ",
  num_gfr,
  "\nnum_mcmc = ",
  num_mcmc,
  "\nsnr = ",
  snr,
  "\nnum_threads = ",
  num_threads,
  "\n",
  sep = ""
)

# Generate data needed to train BART model
X <- matrix(runif(n * p), ncol = p)
plm_term <- (((0 <= X[, 1]) & (0.25 > X[, 1])) *
  (-7.5 * X[, 2]) +
  ((0.25 <= X[, 1]) & (0.5 > X[, 1])) * (-2.5 * X[, 2]) +
  ((0.5 <= X[, 1]) & (0.75 > X[, 1])) * (2.5 * X[, 2]) +
  ((0.75 <= X[, 1]) & (1 > X[, 1])) * (7.5 * X[, 2]))
trig_term <- (2 * sin(X[, 3] * 2 * pi) - 1.5 * cos(X[, 4] * 2 * pi))
f_XW <- plm_term + trig_term
noise_sd <- sd(f_XW) / snr
y <- f_XW + rnorm(n, 0, noise_sd)

# Split into train and test sets
test_set_pct <- 0.2
n_test <- round(test_set_pct * n)
n_train <- n - n_test
test_inds <- sort(sample(1:n, n_test, replace = FALSE))
train_inds <- (1:n)[!((1:n) %in% test_inds)]
X_test <- X[test_inds, ]
X_train <- X[train_inds, ]
y_test <- y[test_inds]
y_train <- y[train_inds]

system.time({
  # Sample BART model
  general_params <- list(num_threads = num_threads)
  bart_model <- bart(
    X_train = X_train,
    y_train = y_train,
    X_test = X_test,
    num_gfr = num_gfr,
    num_mcmc = num_mcmc,
    general_params = general_params
  )

  # Predict on the test set
  test_preds <- predict(bart_model, X = X_test)
})
