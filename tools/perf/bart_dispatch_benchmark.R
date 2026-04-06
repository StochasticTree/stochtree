#!/usr/bin/env Rscript
#
# bart_dispatch_benchmark.R
#
# Compare C++ fast-path vs R slow-path dispatch times for every supported
# BART model variant.
#
# Usage:
#   Rscript tools/perf/bart_dispatch_benchmark.R
#   Rscript tools/perf/bart_dispatch_benchmark.R identity probit rfx
#
# If scenario names are given as arguments only those are run, otherwise all
# scenarios are run.  Available scenario names (print with --list):
#
#   identity          Continuous outcome, identity link
#   identity_rfx      Continuous + intercept-only random effects
#   identity_basis    Continuous + univariate leaf basis (leaf regression)
#   identity_basis_rfx  Continuous + leaf basis + RFX
#   variance_forest   Continuous + heteroskedastic variance forest
#   probit            Binary outcome, probit link
#   cloglog_binary    Binary outcome, cloglog link
#   cloglog_ordinal   Ordinal outcome (K=3), cloglog link
#   cloglog_rfx       Ordinal outcome (K=3), cloglog + RFX
#
# Parameters
N_REPS  <- 10L   # timing repetitions per (scenario × path) cell
N_TRAIN <- 2000L
N_TEST  <- 500L
P       <- 10L
NUM_GFR    <- 10L
NUM_BURNIN <- 0L
NUM_MCMC   <- 100L

library(stochtree)

# ── helpers ───────────────────────────────────────────────────────────────────

# Step function used as the mean signal
step_f <- function(x) {
  ((0.00 <= x & x < 0.25) * (-7.5) +
   (0.25 <= x & x < 0.50) * (-2.5) +
   (0.50 <= x & x < 0.75) * ( 2.5) +
   (0.75 <= x & x <= 1.00) * ( 7.5))
}

# Split a matrix into train/test rows
split_data <- function(X, y, n_test = N_TEST) {
  n    <- nrow(X)
  test_inds  <- sort(sample(n, n_test))
  train_inds <- setdiff(seq_len(n), test_inds)
  list(
    X_train = X[train_inds, , drop = FALSE],
    X_test  = X[test_inds,  , drop = FALSE],
    y_train = y[train_inds],
    y_test  = y[test_inds],
    train_inds = train_inds,
    test_inds  = test_inds
  )
}

# Time `fn` for `reps` repetitions; return numeric vector of elapsed seconds
time_reps <- function(fn, reps = N_REPS) {
  vapply(seq_len(reps), function(i) {
    gc(verbose = FALSE)
    as.numeric(system.time(fn())["elapsed"])
  }, numeric(1))
}

# ── scenario definitions ───────────────────────────────────────────────────────
# Each scenario is a list with:
#   setup() -> data_list    generate data (called once)
#   run(data)               call bart() and return invisibly
# The same run() function is used for both paths; the dispatch option is set
# externally before each run.

set.seed(42)
X_base <- matrix(runif((N_TRAIN + N_TEST) * P), ncol = P)
f_base <- step_f(X_base[, 1])
splits  <- split_data(X_base, f_base + rnorm(N_TRAIN + N_TEST))

scenarios <- list(

  identity = list(
    setup = function() {
      list(
        X_train = splits$X_train, y_train = splits$y_train,
        X_test  = splits$X_test
      )
    },
    run = function(d) {
      invisible(bart(
        X_train = d$X_train, y_train = d$y_train, X_test = d$X_test,
        num_gfr = NUM_GFR, num_burnin = NUM_BURNIN, num_mcmc = NUM_MCMC
      ))
    }
  ),

  identity_rfx = list(
    setup = function() {
      n_tr <- nrow(splits$X_train)
      n_te <- nrow(splits$X_test)
      rfx_ids  <- sample(1:4, N_TRAIN + N_TEST, replace = TRUE)
      rfx_coef <- c(-6, -2, 2, 6)
      y_full   <- f_base + rfx_coef[rfx_ids] + rnorm(N_TRAIN + N_TEST)
      sp <- split_data(X_base, y_full)
      list(
        X_train            = sp$X_train, y_train = sp$y_train,
        X_test             = sp$X_test,
        rfx_group_ids_train = rfx_ids[sp$train_inds],
        rfx_group_ids_test  = rfx_ids[sp$test_inds]
      )
    },
    run = function(d) {
      invisible(bart(
        X_train = d$X_train, y_train = d$y_train, X_test = d$X_test,
        rfx_group_ids_train = d$rfx_group_ids_train,
        rfx_group_ids_test  = d$rfx_group_ids_test,
        num_gfr = NUM_GFR, num_burnin = NUM_BURNIN, num_mcmc = NUM_MCMC,
        random_effects_params = list(model_spec = "intercept_only")
      ))
    }
  ),

  identity_basis = list(
    setup = function() {
      n_all <- N_TRAIN + N_TEST
      W  <- matrix(runif(n_all * 2), ncol = 2)
      y_full <- (f_base * W[, 1]) + rnorm(n_all)
      sp <- split_data(X_base, y_full)
      list(
        X_train = sp$X_train, y_train = sp$y_train, X_test = sp$X_test,
        W_train = W[sp$train_inds, ], W_test = W[sp$test_inds, ]
      )
    },
    run = function(d) {
      invisible(bart(
        X_train = d$X_train, y_train = d$y_train, X_test = d$X_test,
        leaf_basis_train = d$W_train, leaf_basis_test = d$W_test,
        num_gfr = NUM_GFR, num_burnin = NUM_BURNIN, num_mcmc = NUM_MCMC,
        mean_forest_params = list(sample_sigma2_leaf = FALSE)
      ))
    }
  ),

  identity_basis_rfx = list(
    setup = function() {
      n_all <- N_TRAIN + N_TEST
      W        <- matrix(runif(n_all * 2), ncol = 2)
      rfx_ids  <- sample(1:3, n_all, replace = TRUE)
      rfx_coef <- c(-4, 0, 4)
      y_full   <- (f_base * W[, 1]) + rfx_coef[rfx_ids] + rnorm(n_all)
      sp <- split_data(X_base, y_full)
      list(
        X_train = sp$X_train, y_train = sp$y_train, X_test = sp$X_test,
        W_train = W[sp$train_inds, ], W_test = W[sp$test_inds, ],
        rfx_group_ids_train = rfx_ids[sp$train_inds],
        rfx_group_ids_test  = rfx_ids[sp$test_inds]
      )
    },
    run = function(d) {
      invisible(bart(
        X_train = d$X_train, y_train = d$y_train, X_test = d$X_test,
        leaf_basis_train = d$W_train, leaf_basis_test = d$W_test,
        rfx_group_ids_train = d$rfx_group_ids_train,
        rfx_group_ids_test  = d$rfx_group_ids_test,
        num_gfr = NUM_GFR, num_burnin = NUM_BURNIN, num_mcmc = NUM_MCMC,
        mean_forest_params    = list(sample_sigma2_leaf = FALSE),
        random_effects_params = list(model_spec = "intercept_only")
      ))
    }
  ),

  variance_forest = list(
    setup = function() {
      list(
        X_train = splits$X_train, y_train = splits$y_train,
        X_test  = splits$X_test
      )
    },
    run = function(d) {
      invisible(bart(
        X_train = d$X_train, y_train = d$y_train, X_test = d$X_test,
        num_gfr = NUM_GFR, num_burnin = NUM_BURNIN, num_mcmc = NUM_MCMC,
        general_params = list(include_variance_forest = TRUE)
      ))
    }
  ),

  probit = list(
    setup = function() {
      z_full <- f_base / 5 + rnorm(N_TRAIN + N_TEST)
      y_full <- as.integer(z_full > 0)
      sp <- split_data(X_base, y_full)
      list(X_train = sp$X_train, y_train = sp$y_train, X_test = sp$X_test)
    },
    run = function(d) {
      invisible(bart(
        X_train = d$X_train, y_train = d$y_train, X_test = d$X_test,
        num_gfr = NUM_GFR, num_burnin = NUM_BURNIN, num_mcmc = NUM_MCMC,
        general_params = list(
          outcome_model      = OutcomeModel(outcome = "binary", link = "probit"),
          sample_sigma2_global = FALSE
        )
      ))
    }
  ),

  cloglog_binary = list(
    setup = function() {
      p_pos  <- plogis(f_base / 5)
      y_full <- rbinom(N_TRAIN + N_TEST, 1, p_pos)
      sp <- split_data(X_base, y_full)
      list(X_train = sp$X_train, y_train = sp$y_train, X_test = sp$X_test)
    },
    run = function(d) {
      invisible(suppressWarnings(bart(
        X_train = d$X_train, y_train = d$y_train, X_test = d$X_test,
        num_gfr = NUM_GFR, num_burnin = NUM_BURNIN, num_mcmc = NUM_MCMC,
        general_params = list(
          outcome_model        = OutcomeModel(outcome = "binary", link = "cloglog"),
          sample_sigma2_global = FALSE
        )
      )))
    }
  ),

  cloglog_ordinal = list(
    setup = function() {
      z_full <- f_base / 5
      cuts   <- c(-0.5, 0.5)
      y_full <- as.integer(cut(z_full, c(-Inf, cuts, Inf), labels = FALSE)) - 1L
      sp <- split_data(X_base, y_full)
      list(X_train = sp$X_train, y_train = sp$y_train, X_test = sp$X_test)
    },
    run = function(d) {
      invisible(bart(
        X_train = d$X_train, y_train = d$y_train, X_test = d$X_test,
        num_gfr = NUM_GFR, num_burnin = NUM_BURNIN, num_mcmc = NUM_MCMC,
        general_params = list(
          outcome_model        = OutcomeModel(outcome = "ordinal", link = "cloglog"),
          sample_sigma2_global = FALSE
        )
      ))
    }
  ),

  cloglog_rfx = list(
    setup = function() {
      rfx_ids  <- sample(1:3, N_TRAIN + N_TEST, replace = TRUE)
      rfx_coef <- c(-1, 0, 1)
      z_full   <- f_base / 5 + rfx_coef[rfx_ids]
      cuts     <- c(-0.5, 0.5)
      y_full   <- as.integer(cut(z_full, c(-Inf, cuts, Inf), labels = FALSE)) - 1L
      sp <- split_data(X_base, y_full)
      list(
        X_train = sp$X_train, y_train = sp$y_train, X_test = sp$X_test,
        rfx_group_ids_train = rfx_ids[sp$train_inds],
        rfx_group_ids_test  = rfx_ids[sp$test_inds]
      )
    },
    run = function(d) {
      invisible(bart(
        X_train = d$X_train, y_train = d$y_train, X_test = d$X_test,
        rfx_group_ids_train = d$rfx_group_ids_train,
        rfx_group_ids_test  = d$rfx_group_ids_test,
        num_gfr = NUM_GFR, num_burnin = NUM_BURNIN, num_mcmc = NUM_MCMC,
        general_params = list(
          outcome_model        = OutcomeModel(outcome = "ordinal", link = "cloglog"),
          sample_sigma2_global = FALSE
        ),
        random_effects_params = list(model_spec = "intercept_only")
      ))
    }
  )
)

# ── CLI argument parsing ──────────────────────────────────────────────────────

args <- commandArgs(trailingOnly = TRUE)
if ("--list" %in% args) {
  cat("Available scenarios:\n")
  cat(paste0("  ", names(scenarios), "\n"), sep = "")
  quit(save = "no")
}
selected <- if (length(args) > 0) intersect(args, names(scenarios)) else names(scenarios)
if (length(selected) == 0) {
  cat("No matching scenarios found. Use --list to see available names.\n")
  quit(save = "no")
}

# ── run ────────────────────────────────────────────────────────────────────────

cat(sprintf(
  "\nBART dispatch benchmark  |  n_train=%d  n_test=%d  p=%d  reps=%d\n",
  N_TRAIN, N_TEST, P, N_REPS
))
cat(sprintf(
  "  GFR=%d  burnin=%d  MCMC=%d\n\n",
  NUM_GFR, NUM_BURNIN, NUM_MCMC
))
cat(sprintf("%-22s  %10s  %10s  %8s\n",
            "scenario", "fast (s)", "slow (s)", "speedup"))
cat(strrep("-", 57), "\n")

results <- vector("list", length(selected))
names(results) <- selected

for (nm in selected) {
  sc   <- scenarios[[nm]]
  data <- sc$setup()

  options(stochtree.use_cpp_dispatch = TRUE)
  fast_t <- time_reps(function() sc$run(data))

  options(stochtree.use_cpp_dispatch = FALSE)
  slow_t <- time_reps(function() sc$run(data))

  options(stochtree.use_cpp_dispatch = TRUE)   # restore

  results[[nm]] <- list(fast = fast_t, slow = slow_t)

  cat(sprintf("%-22s  %10.3f  %10.3f  %7.2fx\n",
              nm,
              mean(fast_t), mean(slow_t),
              mean(slow_t) / mean(fast_t)))
}

cat("\n(times are mean elapsed seconds across", N_REPS, "repetitions)\n")
