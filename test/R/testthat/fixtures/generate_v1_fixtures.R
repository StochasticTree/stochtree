# Generate the v1 (schema_version=1) golden-fixture matrix for the unified envelope.
#
# Run once to (re)mint the checked-in v1 fixtures:
#   NOT_CRAN=true Rscript -e "devtools::load_all('.'); \
#     source('test/R/testthat/fixtures/generate_v1_fixtures.R')"
#
# The matrix is {bart, bcf} x {numeric, categorical} x {no-rfx, rfx} = 8 models,
# covering the format-relevant axes: forest naming (BART vs BCF), the covariate
# preprocessor (identity vs categorical encoding -> the cross-platform-portable
# axis), and random effects (which add the random_effects subfolder, including
# the relocated rfx_unique_group_ids).
#
# Fixtures only need to be structurally complete and loadable, not statistically
# meaningful, so the ensembles are tiny (5 trees, 2 MCMC) to keep the checked-in
# files small. The legacy v0 fixtures (bart_mcmc.json / bcf_mcmc.json) are kept
# separately and exercise the v0 -> v1 migration path.

out_dir <- "test/R/testthat/fixtures"

NUM_MCMC <- 2
SMALL <- list(num_trees = 5)
N <- 200
P <- 4 # numeric covariate columns (categorical models add one "cat" column)

# Superseded ad-hoc fixtures from the first cut (replaced by the matrix below).
SUPERSEDED <- c("bart_mcmc_v1.json", "bart_rfx_v1.json", "bcf_mcmc_v1.json")

make_covariates <- function(categorical) {
  X_num <- matrix(runif(N * P), ncol = P)
  if (!categorical) {
    return(list(X = X_num, x0 = X_num[, 1]))
  }
  X <- data.frame(X_num)
  X$cat <- factor(sample(c("a", "b", "c"), N, replace = TRUE))
  list(X = X, x0 = X_num[, 1])
}

bart_fixture <- function(seed, rfx, categorical) {
  set.seed(seed)
  cv <- make_covariates(categorical)
  y <- cv$x0 + rnorm(N, 0, 0.5)
  args <- list(
    X_train = cv$X,
    num_gfr = 0,
    num_burnin = 0,
    num_mcmc = NUM_MCMC,
    mean_forest_params = SMALL
  )
  if (rfx) {
    g <- sample(0:2, N, replace = TRUE)
    y <- y + g
    args$rfx_group_ids_train <- g
    args$rfx_basis_train <- matrix(1, nrow = N, ncol = 1)
  }
  args$y_train <- y
  saveBARTModelToJsonString(do.call(bart, args))
}

bcf_fixture <- function(seed, rfx, categorical) {
  set.seed(seed)
  cv <- make_covariates(categorical)
  pi_x <- 0.25 + 0.5 * cv$x0
  Z <- rbinom(N, 1, pi_x)
  y <- pi_x * 5 + Z * cv$x0 * 2 + rnorm(N)
  args <- list(
    X_train = cv$X,
    Z_train = Z,
    propensity_train = pi_x,
    num_gfr = 0,
    num_burnin = 0,
    num_mcmc = NUM_MCMC,
    prognostic_forest_params = SMALL,
    treatment_effect_forest_params = SMALL
  )
  if (rfx) {
    g <- sample(0:2, N, replace = TRUE)
    y <- y + g
    args$rfx_group_ids_train <- g
    args$rfx_basis_train <- matrix(1, nrow = N, ncol = 1)
  }
  args$y_train <- y
  saveBCFModelToJsonString(do.call(bcf, args))
}

seed <- 110
for (model in c("bart", "bcf")) {
  fn <- if (model == "bart") bart_fixture else bcf_fixture
  for (categorical in c(FALSE, TRUE)) {
    for (rfx in c(FALSE, TRUE)) {
      kind <- if (categorical) "categorical" else "numeric"
      suffix <- if (rfx) "_rfx" else ""
      name <- paste0(model, "_", kind, suffix, "_v1.json")
      js <- fn(seed, rfx, categorical)
      seed <- seed + 1
      ver <- jsonlite::fromJSON(js, simplifyVector = FALSE)$schema_version
      stopifnot(ver == 1)
      writeLines(js, file.path(out_dir, name))
      cat("wrote", name, "(schema_version =", ver, ")\n")
    }
  }
}

for (old in SUPERSEDED) {
  p <- file.path(out_dir, old)
  if (file.exists(p)) {
    file.remove(p)
    cat("removed superseded", old, "\n")
  }
}
