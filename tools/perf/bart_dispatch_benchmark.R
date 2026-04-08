##
## Dispatch benchmark: slow path vs C++ BARTSampler (fast_path=TRUE).
##
## Times the existing R bart() function (slow path) against the new
## BARTSamplerFit fast path (fast_path=TRUE) across the same scenario grid
## used in tools/perf/bart_dispatch_benchmark.py.
##
## Usage (from repo root):
##   NOT_CRAN=true Rscript tools/perf/bart_dispatch_benchmark.R
##   NOT_CRAN=true Rscript tools/perf/bart_dispatch_benchmark.R --reps 5
##
## Requirements: devtools::load_all('.') or the stochtree package installed.
##

suppressPackageStartupMessages({
  if (requireNamespace("devtools", quietly = TRUE)) {
    devtools::load_all(".", quiet = TRUE)
  } else {
    library(stochtree)
  }
})

# ── Argument parsing ──────────────────────────────────────────────────────────

args <- commandArgs(trailingOnly = TRUE)
reps <- 3L
if ("--reps" %in% args) {
  idx  <- which(args == "--reps")
  reps <- as.integer(args[idx + 1])
}

# ── Data generators ───────────────────────────────────────────────────────────

make_continuous <- function(n, p, seed = 42L) {
  set.seed(seed)
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  y <- 2.0 * X[, 1] - X[, 2] + 0.5 * rnorm(n)
  list(X = X, y = y)
}

make_binary <- function(n, p, seed = 123L) {
  set.seed(seed)
  X   <- matrix(rnorm(n * p), nrow = n, ncol = p)
  eta <- X[, 1] - 0.5 * X[, 2]
  y   <- as.double(runif(n) < pnorm(eta))
  list(X = X, y = y)
}

make_ordinal <- function(n, p, K, seed = 99L) {
  set.seed(seed)
  X    <- matrix(rnorm(n * p), nrow = n, ncol = p)
  eta  <- X[, 1] - 0.5 * X[, 2]
  cuts <- seq(-2, 2, length.out = K + 1)[-c(1, K + 1)]  # K-1 interior cuts
  u    <- runif(n)
  y    <- rep(K - 1L, n)
  for (k in seq_along(cuts)) {
    cdf_k <- 1.0 - exp(-exp(cuts[k] + eta))
    y[u < cdf_k & y > k - 1L] <- k - 1L
  }
  list(X = X, y = as.double(y))
}

make_heterosked <- function(n, p, seed = 789L) {
  set.seed(seed)
  X  <- matrix(runif(n * p), nrow = n, ncol = p)
  x0 <- X[, 1]
  s  <- ifelse(x0 < 0.25, 0.5, ifelse(x0 < 0.50, 1.0, ifelse(x0 < 0.75, 2.0, 3.0)))
  y  <- s * rnorm(n)
  list(X = X, y = y)
}

make_rfx <- function(n, p, num_groups, seed = 42L) {
  set.seed(seed)
  X      <- matrix(rnorm(n * p), nrow = n, ncol = p)
  alpha  <- rnorm(num_groups, 0, 0.5)
  groups <- (seq_len(n) - 1L) %% num_groups + 1L  # 1-indexed
  y      <- 2.0 * X[, 1] - X[, 2] + alpha[groups] + 0.3 * rnorm(n)
  list(X = X, y = y, groups = groups)
}

# ── Timer ─────────────────────────────────────────────────────────────────────

time_ms <- function(expr_fn, reps) {
  t0 <- proc.time()["elapsed"]
  for (i in seq_len(reps)) expr_fn()
  (proc.time()["elapsed"] - t0) / reps * 1000.0
}

# ── Scenario runner ───────────────────────────────────────────────────────────

run_scenario <- function(label, n, p, num_trees, num_gfr, num_mcmc, num_chains = 1L,
                          link = "identity", cloglog_K = 2L,
                          variance_forest = FALSE, num_trees_variance = 0L,
                          rfx = FALSE, rfx_num_groups = 0L,
                          reps = 3L)
{
  # Prepare data (not counted in timing)
  if (link == "cloglog") {
    d <- make_ordinal(n, p, cloglog_K)
  } else if (link == "probit") {
    d <- make_binary(n, p)
  } else if (variance_forest && num_trees == 0L) {
    d <- make_heterosked(n, p)
  } else if (rfx) {
    d <- make_rfx(n, p, rfx_num_groups)
  } else {
    d <- make_continuous(n, p)
  }

  # Shared bart() arguments
  general_params <- list(num_chains = num_chains)
  if (link == "probit") {
    general_params$outcome_model <- OutcomeModel("binary", "probit")
  } else if (link == "cloglog") {
    general_params$outcome_model <- OutcomeModel("binary", "cloglog")
  }

  variance_forest_params <- list(num_trees = num_trees_variance)

  rfx_params <- list(model_spec = "intercept_only")
  rfx_group_ids_train <- NULL
  if (rfx) rfx_group_ids_train <- d$groups

  num_samples <- num_gfr + num_mcmc * num_chains

  # ── Slow path ────────────────────────────────────────────────────────────────
  ms_slow <- time_ms(function() {
    suppressWarnings(
      bart(
        X_train                = d$X,
        y_train                = d$y,
        rfx_group_ids_train    = rfx_group_ids_train,
        num_gfr                = num_gfr,
        num_mcmc               = num_mcmc,
        general_params         = general_params,
        mean_forest_params     = list(num_trees = num_trees),
        variance_forest_params = variance_forest_params,
        random_effects_params  = if (rfx) rfx_params else list()
      )
    )
  }, reps)

  # ── Fast path (BARTSamplerFit) ───────────────────────────────────────────────
  if (num_gfr > 0L || num_mcmc > 0L) {
    ms_smp <- time_ms(function() {
      suppressWarnings(
        bart(
          X_train                = d$X,
          y_train                = d$y,
          rfx_group_ids_train    = rfx_group_ids_train,
          num_gfr                = num_gfr,
          num_mcmc               = num_mcmc,
          general_params         = general_params,
          mean_forest_params     = list(num_trees = num_trees),
          variance_forest_params = variance_forest_params,
          random_effects_params  = if (rfx) rfx_params else list(),
          fast_path              = TRUE
        )
      )
    }, reps)
  } else {
    ms_smp <- NA_real_
  }

  speedup_str <- if (!is.na(ms_smp) && ms_smp > 0) {
    sprintf("%.2fx", ms_slow / ms_smp)
  } else {
    "     —"
  }

  list(
    label       = label,
    n           = n,
    num_trees   = num_trees,
    num_samples = num_samples,
    ms_slow     = ms_slow,
    ms_smp      = ms_smp,
    speedup     = speedup_str
  )
}

# ── Scenarios ─────────────────────────────────────────────────────────────────

scenarios <- list(
  list(label="GFR-only (id)",          n=500L,  p=10L, num_trees=200L, num_gfr=10L, num_mcmc=0L,   num_chains=1L),
  list(label="MCMC-only (id)",         n=500L,  p=10L, num_trees=200L, num_gfr=0L,  num_mcmc=100L, num_chains=1L),
  list(label="Warm-start (id)",        n=500L,  p=10L, num_trees=200L, num_gfr=5L,  num_mcmc=100L, num_chains=1L),
  list(label="Multi-chain (id)",       n=500L,  p=10L, num_trees=200L, num_gfr=3L,  num_mcmc=50L,  num_chains=3L),
  list(label="Large-n (id)",           n=2000L, p=10L, num_trees=200L, num_gfr=5L,  num_mcmc=100L, num_chains=1L),
  list(label="Warm-start (probit)",    n=500L,  p=10L, num_trees=200L, num_gfr=5L,  num_mcmc=100L, num_chains=1L,
       link="probit"),
  list(label="Warm-start (cloglog2)",  n=500L,  p=10L, num_trees=200L, num_gfr=5L,  num_mcmc=100L, num_chains=1L,
       link="cloglog", cloglog_K=2L),
  list(label="Warm-start (cloglog3)",  n=500L,  p=10L, num_trees=200L, num_gfr=5L,  num_mcmc=100L, num_chains=1L,
       link="cloglog", cloglog_K=3L),
  list(label="Warm-start (varforest)", n=500L,  p=10L, num_trees=0L,   num_gfr=5L,  num_mcmc=100L, num_chains=1L,
       variance_forest=TRUE, num_trees_variance=50L),
  list(label="Warm-start (rfx-10g)",   n=500L,  p=10L, num_trees=200L, num_gfr=5L,  num_mcmc=100L, num_chains=1L,
       rfx=TRUE, rfx_num_groups=10L)
)

# ── Run and print ─────────────────────────────────────────────────────────────

W <- c(label=28L, n=6L, T=5L, S=5L, slow=14L, smp=16L, spdup=10L)
header <- sprintf("%-*s %*s %*s %*s %*s %*s %*s",
  W["label"], "Scenario",
  W["n"],     "n",
  W["T"],     "T",
  W["S"],     "S",
  W["slow"],  "slow_path(ms)",
  W["smp"],   "bartsampler(ms)",
  W["spdup"], "speedup")
sep <- paste(rep("-", sum(W) + length(W) - 1L), collapse = "")

cat(sprintf("\nDispatch benchmark  (reps=%d)\n", reps))
cat(sep, "\n")
cat(header, "\n")
cat(sep, "\n")

for (s in scenarios) {
  res <- do.call(run_scenario, c(s, list(reps = reps)))

  smp_str <- if (!is.na(res$ms_smp)) sprintf("%*.1f", W["smp"], res$ms_smp) else
             sprintf("%*s", W["smp"], "—")

  cat(sprintf("%-*s %*d %*d %*d %*.1f %s %*s\n",
    W["label"], res$label,
    W["n"],     res$n,
    W["T"],     res$num_trees,
    W["S"],     res$num_samples,
    W["slow"],  res$ms_slow,
    smp_str,
    W["spdup"], res$speedup
  ))
}

cat(sep, "\n")
cat("  speedup > 1.0x: BARTSampler (fast_path=TRUE) is faster\n")
cat("  speedup < 1.0x: slow path is faster (regression — investigate)\n")
cat("  '—' in bartsampler column: fast_path not applicable (num_gfr=0 and num_mcmc=0)\n\n")
