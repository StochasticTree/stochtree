# Benchmark cloglog BART in R using shared data (multiple replications)
library(stochtree)

# Load benchmark data
script_dir <- "tools/debug"
train <- read.csv(file.path(script_dir, "cloglog_benchmark_train.csv"))
test <- read.csv(file.path(script_dir, "cloglog_benchmark_test.csv"))
true_probs_train <- as.matrix(read.csv(
  file.path(script_dir, "cloglog_benchmark_true_probs_train.csv")
))
true_probs_test <- as.matrix(read.csv(
  file.path(script_dir, "cloglog_benchmark_true_probs_test.csv")
))

# Extract X and y
p <- ncol(train) - 1
X_train <- as.matrix(train[, 1:p])
y_train <- train$y
X_test <- as.matrix(test[, 1:p])
y_test <- test$y
n_categories <- length(unique(y_train))

cat("Train:", nrow(X_train), "obs, Test:", nrow(X_test), "obs\n")
cat("Outcome distribution:", table(y_train), "\n")

# Run multiple replications with explicit C++ seeds
n_reps <- 10
seeds <- 1:n_reps
train_cors <- matrix(NA, n_reps, n_categories)
test_cors <- matrix(NA, n_reps, n_categories)

for (rep in 1:n_reps) {
  seed <- seeds[rep]
  runtime <- system.time({
    bart_model <- bart(
      X_train = X_train,
      y_train = y_train,
      X_test = X_test,
      num_gfr = 0,
      num_burnin = 1000,
      num_mcmc = 1000,
      general_params = list(
        cutpoint_grid_size = 100,
        sample_sigma2_global = FALSE,
        keep_every = 1,
        num_chains = 1,
        verbose = FALSE,
        random_seed = seed,
        outcome_model = outcome_model(outcome = "ordinal", link = "cloglog")
      ),
      mean_forest_params = list(num_trees = 50)
    )
  })

  est_probs_train <- predict(
    bart_model,
    X = X_train,
    scale = "probability",
    terms = "y_hat"
  )
  est_probs_test <- predict(
    bart_model,
    X = X_test,
    scale = "probability",
    terms = "y_hat"
  )

  for (j in 1:n_categories) {
    train_cors[rep, j] <- cor(
      true_probs_train[, j],
      rowMeans(est_probs_train[, j, ])
    )
    test_cors[rep, j] <- cor(
      true_probs_test[, j],
      rowMeans(est_probs_test[, j, ])
    )
  }

  cat(sprintf(
    "Rep %d (seed=%d, %.1fs): train=[%s] test=[%s]\n",
    rep,
    seed,
    runtime["elapsed"],
    paste(sprintf("%.4f", train_cors[rep, ]), collapse = ", "),
    paste(sprintf("%.4f", test_cors[rep, ]), collapse = ", ")
  ))
}

# Summary
cat("\n--- Summary across replications ---\n")
cat(sprintf("%20s  %12s  %12s  %12s\n", "", "Cat 1", "Cat 2", "Cat 3"))
cat(sprintf(
  "%20s  %12.4f  %12.4f  %12.4f\n",
  "Train mean",
  mean(train_cors[, 1]),
  mean(train_cors[, 2]),
  mean(train_cors[, 3])
))
cat(sprintf(
  "%20s  %12.4f  %12.4f  %12.4f\n",
  "Train std",
  sd(train_cors[, 1]),
  sd(train_cors[, 2]),
  sd(train_cors[, 3])
))
cat(sprintf(
  "%20s  %12.4f  %12.4f  %12.4f\n",
  "Test mean",
  mean(test_cors[, 1]),
  mean(test_cors[, 2]),
  mean(test_cors[, 3])
))
cat(sprintf(
  "%20s  %12.4f  %12.4f  %12.4f\n",
  "Test std",
  sd(test_cors[, 1]),
  sd(test_cors[, 2]),
  sd(test_cors[, 3])
))
