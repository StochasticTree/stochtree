################################################################################
# Minimal script for debugging the C++ sampler under lldb.
#
# Usage (from the repo root):
#   lldb -- R --vanilla -f tools/debug/debug_cpp_sampler.R
#   # then at the (lldb) prompt:
#   #   run
#   #   bt          (after the crash, to get a backtrace)
#   #   frame info  (to see the crashing frame)
#
# Alternatively, attach to an already-running R process:
#   lldb -p $(pgrep -n R)
################################################################################

suppressPackageStartupMessages(devtools::load_all("."))

# --- Data generation (mirrors debug_cpp_sampler.py) --------------------------
seed <- 1001
n <- 10000
p <- 10
set.seed(1234)

X <- matrix(runif(n * p), nrow = n, ncol = p)
f_X <- ifelse(
  X[, 1] < 0.25,
  -7.5,
  ifelse(X[, 1] < 0.50, -2.5, ifelse(X[, 1] < 0.75, 2.5, 7.5))
)
y <- f_X + rnorm(n, sd = 1.0)

n_test <- round(0.2 * n)
test_inds <- sort(sample(seq_len(n), n_test, replace = FALSE))
train_inds <- setdiff(seq_len(n), test_inds)

X_train <- X[train_inds, ]
X_test <- X[test_inds, ]
y_train <- y[train_inds]
y_test <- y[test_inds]

cat(sprintf(
  "n_train=%d  n_test=%d  p=%d  seed=%d\n",
  length(train_inds),
  n_test,
  p,
  seed
))
cat("Calling bart() with run_cpp=TRUE ...\n")

# --- Run C++ sampler ----------------------------------------------------------
m <- bart(
  X_train = X_train,
  y_train = y_train,
  X_test = X_test,
  num_gfr = 10,
  num_burnin = 0,
  num_mcmc = 100,
  general_params = list(random_seed = seed),
  mean_forest_params = list(num_trees = 200),
  run_cpp = TRUE
)

cat("Completed successfully.\n")
cat(sprintf(
  "dim(y_hat_test): %d x %d\n",
  nrow(m$y_hat_test),
  ncol(m$y_hat_test)
))
