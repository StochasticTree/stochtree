#' Compute a contrast between two outcome prediction specifications for a BCF model
#'
#' Compute a contrast using a BCF model by making two sets of outcome predictions and taking their difference.
#' For simple BCF models with binary treatment, this will yield the same prediction as requesting `terms = "cate"`
#' in the `predict.bcfmodel` function. For more general models, such as models with continuous / multivariate treatments or
#' an additive random effects term with a coefficient on the treatment, this function provides the flexibility to compute a
#' any contrast of interest by specifying covariates, treatment, and random effects bases and IDs for both sides of a two term
#' contrast. For simplicity, we refer to the subtrahend of the contrast as the "control" or `Y0` term and the minuend of the
#' contrast as the `Y1` term, though the requested contrast need not match the "control vs treatment" terminology of a classic
#' two-arm experiment. We mirror the function calls and terminology of the `predict.bcfmodel` function, labeling each prediction
#' data term with a `1` to denote its contribution to the treatment prediction of a contrast and `0` to denote inclusion in the
#' control prediction.
#'
#' @param object Object of type `bcfmodel` containing draws of a Bayesian causal forest model and associated sampling outputs.
#' @param X_0 Covariates used for prediction in the "control" case. Must be a matrix or dataframe.
#' @param X_1 Covariates used for prediction in the "treatment" case. Must be a matrix or dataframe.
#' @param Z_0 Treatments used for prediction in the "control" case. Must be a matrix or vector.
#' @param Z_1 Treatments used for prediction in the "treatment" case. Must be a matrix or vector.
#' @param propensity_0 (Optional) Propensities used for prediction in the "control" case. Must be a matrix or vector.
#' @param propensity_1 (Optional) Propensities used for prediction in the "treatment" case. Must be a matrix or vector.
#' @param rfx_group_ids_0 (Optional) Test set group labels used for prediction from an additive random effects
#' model in the "control" case. We do not currently support (but plan to in the near future), test set evaluation
#' for group labels that were not in the training set. Must be a vector.
#' @param rfx_group_ids_1 (Optional) Test set group labels used for prediction from an additive random effects
#' model in the "treatment" case. We do not currently support (but plan to in the near future), test set evaluation
#' for group labels that were not in the training set. Must be a vector.
#' @param rfx_basis_0 (Optional) Test set basis for used for prediction from an additive random effects model in the "control" case.  Must be a matrix or vector.
#' @param rfx_basis_1 (Optional) Test set basis for used for prediction from an additive random effects model in the "treatment" case. Must be a matrix or vector.
#' @param type (Optional) Aggregation level of the contrast. Options are "mean", which averages the contrast evaluations over every draw of a BCF model, and "posterior", which returns the entire matrix of posterior contrast estimates. Default: "posterior".
#' @param scale (Optional) Scale of the contrast. Options are "linear", which returns a contrast on the original scale of the mean forest / RFX terms, and "probability", which transforms each contrast term into a probability of observing `y == 1` before taking their difference. "probability" is only valid for models fit with a probit outcome model. Default: "linear".
#'
#' @return List of prediction matrices or single prediction matrix / vector, depending on the terms requested.
#' @export
#'
#' @examples
#' n <- 500
#' p <- 5
#' X <- matrix(runif(n*p), ncol = p)
#' mu_x <- (
#'     ((0 <= X[,1]) & (0.25 > X[,1])) * (-7.5) +
#'     ((0.25 <= X[,1]) & (0.5 > X[,1])) * (-2.5) +
#'     ((0.5 <= X[,1]) & (0.75 > X[,1])) * (2.5) +
#'     ((0.75 <= X[,1]) & (1 > X[,1])) * (7.5)
#' )
#' pi_x <- (
#'     ((0 <= X[,1]) & (0.25 > X[,1])) * (0.2) +
#'     ((0.25 <= X[,1]) & (0.5 > X[,1])) * (0.4) +
#'     ((0.5 <= X[,1]) & (0.75 > X[,1])) * (0.6) +
#'     ((0.75 <= X[,1]) & (1 > X[,1])) * (0.8)
#' )
#' tau_x <- (
#'     ((0 <= X[,2]) & (0.25 > X[,2])) * (0.5) +
#'     ((0.25 <= X[,2]) & (0.5 > X[,2])) * (1.0) +
#'     ((0.5 <= X[,2]) & (0.75 > X[,2])) * (1.5) +
#'     ((0.75 <= X[,2]) & (1 > X[,2])) * (2.0)
#' )
#' Z <- rbinom(n, 1, pi_x)
#' noise_sd <- 1
#' y <- mu_x + tau_x*Z + rnorm(n, 0, noise_sd)
#' test_set_pct <- 0.2
#' n_test <- round(test_set_pct*n)
#' n_train <- n - n_test
#' test_inds <- sort(sample(1:n, n_test, replace = FALSE))
#' train_inds <- (1:n)[!((1:n) %in% test_inds)]
#' X_test <- X[test_inds,]
#' X_train <- X[train_inds,]
#' pi_test <- pi_x[test_inds]
#' pi_train <- pi_x[train_inds]
#' Z_test <- Z[test_inds]
#' Z_train <- Z[train_inds]
#' y_test <- y[test_inds]
#' y_train <- y[train_inds]
#' mu_test <- mu_x[test_inds]
#' mu_train <- mu_x[train_inds]
#' tau_test <- tau_x[test_inds]
#' tau_train <- tau_x[train_inds]
#' bcf_model <- bcf(X_train = X_train, Z_train = Z_train, y_train = y_train,
#'                  propensity_train = pi_train, num_gfr = 10,
#'                  num_burnin = 0, num_mcmc = 10)
#' tau_hat_test <- compute_contrast_bcf_model(
#'     bcf_model, X_0=X_test, X_1=X_test, Z_0=rep(0, n_test), Z_1=rep(1, n_test),
#'     propensity_0 = pi_test, propensity_1 = pi_test
#' )
compute_contrast_bcf_model <- function(
  object,
  X_0,
  X_1,
  Z_0,
  Z_1,
  propensity_0 = NULL,
  propensity_1 = NULL,
  rfx_group_ids_0 = NULL,
  rfx_group_ids_1 = NULL,
  rfx_basis_0 = NULL,
  rfx_basis_1 = NULL,
  type = "posterior",
  scale = "linear"
) {
  # Handle mean function scale
  if (!is.character(scale)) {
    stop("scale must be a string or character vector")
  }
  if (!(scale %in% c("linear", "probability"))) {
    stop("scale must either be 'linear' or 'probability'")
  }
  is_probit <- object$model_params$probit_outcome_model
  if ((scale == "probability") && (!is_probit)) {
    stop(
      "scale cannot be 'probability' for models not fit with a probit outcome model"
    )
  }
  probability_scale <- scale == "probability"

  # Handle prediction type
  if (!is.character(type)) {
    stop("type must be a string or character vector")
  }
  if (!(type %in% c("mean", "posterior"))) {
    stop("type must either be 'mean' or 'posterior")
  }
  predict_mean <- type == "mean"

  # Make sure covariates are matrix or data frame
  if ((!is.data.frame(X_0)) && (!is.matrix(X_0))) {
    stop("X_0 must be a matrix or dataframe")
  }
  if ((!is.data.frame(X_1)) && (!is.matrix(X_1))) {
    stop("X_1 must be a matrix or dataframe")
  }

  # Convert all input data to matrices if not already converted
  if ((is.null(dim(Z_0))) && (!is.null(Z_0))) {
    Z_0 <- as.matrix(as.numeric(Z_0))
  }
  if ((is.null(dim(Z_1))) && (!is.null(Z_1))) {
    Z_1 <- as.matrix(as.numeric(Z_1))
  }
  if ((is.null(dim(propensity_0))) && (!is.null(propensity_0))) {
    propensity_0 <- as.matrix(propensity_0)
  }
  if ((is.null(dim(propensity_1))) && (!is.null(propensity_1))) {
    propensity_1 <- as.matrix(propensity_1)
  }
  if ((is.null(dim(rfx_basis_0))) && (!is.null(rfx_basis_0))) {
    rfx_basis_0 <- as.matrix(rfx_basis_0)
  }
  if ((is.null(dim(rfx_basis_1))) && (!is.null(rfx_basis_1))) {
    rfx_basis_1 <- as.matrix(rfx_basis_1)
  }

  # Data checks
  if (
    (object$model_params$propensity_covariate != "none") &&
      ((is.null(propensity_0)) ||
        (is.null(propensity_1)))
  ) {
    if (!object$model_params$internal_propensity_model) {
      stop("propensity_0 and propensity_1 must be provided for this model")
    }
  }
  if (nrow(X_0) != nrow(Z_0)) {
    stop("X_0 and Z_0 must have the same number of rows")
  }
  if (nrow(X_1) != nrow(Z_1)) {
    stop("X_1 and Z_1 must have the same number of rows")
  }
  if (object$model_params$num_covariates != ncol(X_0)) {
    stop(
      "X_0 and must have the same number of columns as the covariates used to train the model"
    )
  }
  if (object$model_params$num_covariates != ncol(X_1)) {
    stop(
      "X_1 and must have the same number of columns as the covariates used to train the model"
    )
  }
  if ((object$model_params$has_rfx) && (is.null(rfx_group_ids_0))) {
    stop(
      "Random effect group labels (rfx_group_ids_0) must be provided for this model"
    )
  }
  if ((object$model_params$has_rfx) && (is.null(rfx_group_ids_1))) {
    stop(
      "Random effect group labels (rfx_group_ids_1) must be provided for this model"
    )
  }
  if ((object$model_params$has_rfx_basis) && (is.null(rfx_basis_0))) {
    stop("Random effects basis (rfx_basis_0) must be provided for this model")
  }
  if ((object$model_params$has_rfx_basis) && (is.null(rfx_basis_1))) {
    stop("Random effects basis (rfx_basis_1) must be provided for this model")
  }
  if (
    (object$model_params$num_rfx_basis > 0) &&
      (ncol(rfx_basis_0) != object$model_params$num_rfx_basis)
  ) {
    stop(
      "Random effects basis has a different dimension than the basis used to train this model"
    )
  }
  if (
    (object$model_params$num_rfx_basis > 0) &&
      (ncol(rfx_basis_1) != object$model_params$num_rfx_basis)
  ) {
    stop(
      "Random effects basis has a different dimension than the basis used to train this model"
    )
  }

  # Predict for the control arm
  control_preds <- predict(
    object = object,
    X = X_0,
    Z = Z_0,
    propensity = propensity_0,
    rfx_group_ids = rfx_group_ids_0,
    rfx_basis = rfx_basis_0,
    type = "posterior",
    term = "y_hat",
    scale = "linear"
  )

  # Predict for the treatment arm
  treatment_preds <- predict(
    object = object,
    X = X_1,
    Z = Z_1,
    propensity = propensity_1,
    rfx_group_ids = rfx_group_ids_1,
    rfx_basis = rfx_basis_1,
    type = "posterior",
    term = "y_hat",
    scale = "linear"
  )

  # Transform to probability scale if requested
  if (probability_scale) {
    treatment_preds <- pnorm(treatment_preds)
    control_preds <- pnorm(control_preds)
  }

  # Compute and return contrast
  if (predict_mean) {
    return(rowMeans(treatment_preds - control_preds))
  } else {
    return(treatment_preds - control_preds)
  }
}

#' Compute a contrast between two outcome prediction specifications for a BART model
#'
#' Compute a contrast using a BART model by making two sets of outcome predictions and taking their difference.
#' This function provides the flexibility to compute any contrast of interest by specifying covariates, leaf basis, and random effects
#' bases / IDs for both sides of a two term contrast. For simplicity, we refer to the subtrahend of the contrast as the "control" or
#' `Y0` term and the minuend of the contrast as the `Y1` term, though the requested contrast need not match the "control vs treatment"
#' terminology of a classic two-treatment causal inference problem. We mirror the function calls and terminology of the `predict.bartmodel`
#' function, labeling each prediction data term with a `1` to denote its contribution to the treatment prediction of a contrast and
#' `0` to denote inclusion in the control prediction.
#'
#' Only valid when there is either a mean forest or a random effects term in the BART model.
#'
#' @param object Object of type `bart` containing draws of a regression forest and associated sampling outputs.
#' @param X_0 Covariates used for prediction in the "control" case. Must be a matrix or dataframe.
#' @param X_1 Covariates used for prediction in the "treatment" case. Must be a matrix or dataframe.
#' @param leaf_basis_0 (Optional) Bases used for prediction in the "control" case (by e.g. dot product with leaf values). Default: `NULL`.
#' @param leaf_basis_1 (Optional) Bases used for prediction in the "treatment" case (by e.g. dot product with leaf values). Default: `NULL`.
#' @param rfx_group_ids_0 (Optional) Test set group labels used for prediction from an additive random effects
#' model in the "control" case. We do not currently support (but plan to in the near future), test set evaluation
#' for group labels that were not in the training set. Must be a vector.
#' @param rfx_group_ids_1 (Optional) Test set group labels used for prediction from an additive random effects
#' model in the "treatment" case. We do not currently support (but plan to in the near future), test set evaluation
#' for group labels that were not in the training set. Must be a vector.
#' @param rfx_basis_0 (Optional) Test set basis for used for prediction from an additive random effects model in the "control" case.  Must be a matrix or vector.
#' @param rfx_basis_1 (Optional) Test set basis for used for prediction from an additive random effects model in the "treatment" case. Must be a matrix or vector.
#' @param type (Optional) Aggregation level of the contrast. Options are "mean", which averages the contrast evaluations over every draw of a BART model, and "posterior", which returns the entire matrix of posterior contrast estimates. Default: "posterior".
#' @param scale (Optional) Scale of the contrast. Options are "linear", which returns a contrast on the original scale of the mean forest / RFX terms, and "probability", which transforms each contrast term into a probability of observing `y == 1` before taking their difference. "probability" is only valid for models fit with a probit outcome model. Default: "linear".
#'
#' @return Contrast matrix or vector, depending on whether type = "mean" or "posterior".
#' @export
#'
#' @examples
#' n <- 100
#' p <- 5
#' X <- matrix(runif(n*p), ncol = p)
#' W <- matrix(runif(n*1), ncol = 1)
#' f_XW <- (
#'     ((0 <= X[,1]) & (0.25 > X[,1])) * (-7.5*W[,1]) +
#'     ((0.25 <= X[,1]) & (0.5 > X[,1])) * (-2.5*W[,1]) +
#'     ((0.5 <= X[,1]) & (0.75 > X[,1])) * (2.5*W[,1]) +
#'     ((0.75 <= X[,1]) & (1 > X[,1])) * (7.5*W[,1])
#' )
#' noise_sd <- 1
#' y <- f_XW + rnorm(n, 0, noise_sd)
#' test_set_pct <- 0.2
#' n_test <- round(test_set_pct*n)
#' n_train <- n - n_test
#' test_inds <- sort(sample(1:n, n_test, replace = FALSE))
#' train_inds <- (1:n)[!((1:n) %in% test_inds)]
#' X_test <- X[test_inds,]
#' X_train <- X[train_inds,]
#' W_test <- W[test_inds,]
#' W_train <- W[train_inds,]
#' y_test <- y[test_inds]
#' y_train <- y[train_inds]
#' bart_model <- bart(X_train = X_train, leaf_basis_train = W_train, y_train = y_train,
#'                    num_gfr = 10, num_burnin = 0, num_mcmc = 10)
#' contrast_test <- compute_contrast_bart_model(
#'     bart_model,
#'     X_0 = X_test,
#'     X_1 = X_test,
#'     leaf_basis_0 = matrix(0, nrow = n_test, ncol = 1),
#'     leaf_basis_1 = matrix(1, nrow = n_test, ncol = 1),
#'     type = "posterior",
#'     scale = "linear"
#' )
compute_contrast_bart_model <- function(
  object,
  X_0,
  X_1,
  leaf_basis_0 = NULL,
  leaf_basis_1 = NULL,
  rfx_group_ids_0 = NULL,
  rfx_group_ids_1 = NULL,
  rfx_basis_0 = NULL,
  rfx_basis_1 = NULL,
  type = "posterior",
  scale = "linear"
) {
  # Handle mean function scale
  if (!is.character(scale)) {
    stop("scale must be a string or character vector")
  }
  if (!(scale %in% c("linear", "probability"))) {
    stop("scale must either be 'linear' or 'probability'")
  }
  is_probit <- object$model_params$probit_outcome_model
  if ((scale == "probability") && (!is_probit)) {
    stop(
      "scale cannot be 'probability' for models not fit with a probit outcome model"
    )
  }
  probability_scale <- scale == "probability"

  # Handle prediction type
  if (!is.character(type)) {
    stop("type must be a string or character vector")
  }
  if (!(type %in% c("mean", "posterior"))) {
    stop("type must either be 'mean' or 'posterior'")
  }
  predict_mean <- type == "mean"

  # Handle prediction terms
  has_mean_forest <- object$model_params$include_mean_forest
  has_rfx <- object$model_params$has_rfx
  if ((!has_mean_forest) && (!has_rfx)) {
    stop(
      "Model must have either or both of mean forest or random effects terms to compute the requested contrast."
    )
  }

  # Check that covariates are matrix or data frame
  if ((!is.data.frame(X_0)) && (!is.matrix(X_0))) {
    stop("X_0 must be a matrix or dataframe")
  }
  if ((!is.data.frame(X_1)) && (!is.matrix(X_1))) {
    stop("X_1 must be a matrix or dataframe")
  }

  # Convert all input data to matrices if not already converted
  if ((is.null(dim(leaf_basis_0))) && (!is.null(leaf_basis_0))) {
    leaf_basis_0 <- as.matrix(leaf_basis_0)
  }
  if ((is.null(dim(leaf_basis_1))) && (!is.null(leaf_basis_1))) {
    leaf_basis_1 <- as.matrix(leaf_basis_1)
  }
  if ((is.null(dim(rfx_basis_0))) && (!is.null(rfx_basis_0))) {
    rfx_basis_0 <- as.matrix(rfx_basis_0)
  }
  if ((is.null(dim(rfx_basis_1))) && (!is.null(rfx_basis_1))) {
    rfx_basis_1 <- as.matrix(rfx_basis_1)
  }

  # Data checks
  if (
    (object$model_params$requires_basis) &&
      (is.null(leaf_basis_0) || is.null(leaf_basis_1))
  ) {
    stop("leaf_basis_0 and leaf_basis_1 must be provided for this model")
  }
  if ((!is.null(leaf_basis_0)) && (nrow(X_0) != nrow(leaf_basis_0))) {
    stop("X_0 and leaf_basis_0 must have the same number of rows")
  }
  if ((!is.null(leaf_basis_1)) && (nrow(X_1) != nrow(leaf_basis_1))) {
    stop("X_1 and leaf_basis_1 must have the same number of rows")
  }
  if (object$model_params$num_covariates != ncol(X_0)) {
    stop(
      "X_0 must contain the same number of columns as the BART model's training dataset"
    )
  }
  if (object$model_params$num_covariates != ncol(X_1)) {
    stop(
      "X_1 must contain the same number of columns as the BART model's training dataset"
    )
  }
  if ((has_rfx) && (is.null(rfx_group_ids_0) || is.null(rfx_group_ids_1))) {
    stop(
      "rfx_group_ids_0 and rfx_group_ids_1 must be provided for this model"
    )
  }
  if (has_rfx) {
    if (object$model_params$rfx_model_spec == "custom") {
      if ((is.null(rfx_basis_0) || is.null(rfx_basis_1))) {
        stop(
          "A user-provided basis (`rfx_basis_0` and `rfx_basis_1`) must be provided when the model was sampled with a random effects model spec set to 'custom'"
        )
      }
      if (!is.matrix(rfx_basis_0) || !is.matrix(rfx_basis_1)) {
        stop("'rfx_basis_0' and 'rfx_basis_1' must be matrices")
      }
      if (
        (nrow(rfx_basis_0) != nrow(X_0)) || (nrow(rfx_basis_1) != nrow(X_1))
      ) {
        stop(
          "'rfx_basis_0' and 'rfx_basis_1' must have the same number of rows as 'X_0' and 'X_1'"
        )
      }
      if (
        (object$model_params$num_rfx_basis > 0) &&
          ((ncol(rfx_basis_0) != object$model_params$num_rfx_basis) ||
            (ncol(rfx_basis_1) != object$model_params$num_rfx_basis))
      ) {
        stop(
          "rfx_basis_0 and / or rfx_basis_1 have a different dimension than the basis used to train this model"
        )
      }
    }
  }

  # Predict for the control arm
  control_preds <- predict(
    object = object,
    X = X_0,
    leaf_basis = leaf_basis_0,
    rfx_group_ids = rfx_group_ids_0,
    rfx_basis = rfx_basis_0,
    type = "posterior",
    term = "y_hat",
    scale = "linear"
  )

  # Predict for the treatment arm
  treatment_preds <- predict(
    object = object,
    X = X_1,
    leaf_basis = leaf_basis_1,
    rfx_group_ids = rfx_group_ids_1,
    rfx_basis = rfx_basis_1,
    type = "posterior",
    term = "y_hat",
    scale = "linear"
  )

  # Transform to probability scale if requested
  if (probability_scale) {
    treatment_preds <- pnorm(treatment_preds)
    control_preds <- pnorm(control_preds)
  }

  # Compute and return contrast
  if (predict_mean) {
    return(rowMeans(treatment_preds - control_preds))
  } else {
    return(treatment_preds - control_preds)
  }
}

#' Sample from the posterior predictive distribution for outcomes modeled by BCF
#'
#' @param model_object A fitted BCF model object of class `bcfmodel`.
#' @param X A matrix or data frame of covariates.
#' @param Z A vector or matrix of treatment assignments.
#' @param propensity (Optional) A vector or matrix of propensity scores. Required if the underlying model depends on user-provided propensities.
#' @param rfx_group_ids (Optional) A vector of group IDs for random effects model. Required if the BCF model includes random effects.
#' @param rfx_basis (Optional) A matrix of bases for random effects model. Required if the BCF model includes random effects.
#' @param num_draws_per_sample (Optional) The number of samples to draw from the likelihood for each draw of the posterior. Defaults to a heuristic based on the number of samples in a BCF model (i.e. if the BCF model has >1000 draws, we use 1 draw from the likelihood per sample, otherwise we upsample to ensure at least 1000 posterior predictive draws).
#'
#' @returns Array of posterior predictive samples with dimensions (num_observations, num_posterior_samples, num_draws_per_sample) if num_draws_per_sample > 1, otherwise (num_observations, num_posterior_samples).
#'
#' @export
#' @examples
#' n <- 100
#' p <- 5
#' X <- matrix(rnorm(n * p), nrow = n, ncol = p)
#' pi_X <- pnorm(X[,1] / 2)
#' Z <- rbinom(n, 1, pi_X)
#' y <- 2 * X[,2] + 0.5 * X[,2] * Z + rnorm(n)
#' bcf_model <- bcf(X_train = X, Z_train = Z, y_train = y, propensity_train = pi_X)
#' ppd_samples <- sample_bcf_posterior_predictive(
#'   model_object = bcf_model, X = X,
#'   Z = Z, propensity = pi_X
#' )
sample_bcf_posterior_predictive <- function(
  model_object,
  X = NULL,
  Z = NULL,
  propensity = NULL,
  rfx_group_ids = NULL,
  rfx_basis = NULL,
  num_draws_per_sample = NULL
) {
  # Check the provided model object
  check_model_is_valid(model_object)

  # Determine whether the outcome is continuous (Gaussian) or binary (probit-link)
  is_probit <- model_object$model_params$probit_outcome_model

  # Check that all the necessary inputs were provided for interval computation
  needs_covariates <- TRUE
  if (needs_covariates) {
    if (is.null(X)) {
      stop(
        "'X' must be provided in order to compute the requested intervals"
      )
    }
    if (!is.matrix(X) && !is.data.frame(X)) {
      stop("'X' must be a matrix or data frame")
    }
  }
  needs_treatment <- needs_covariates
  if (needs_treatment) {
    if (is.null(Z)) {
      stop(
        "'Z' must be provided in order to compute the requested intervals"
      )
    }
    if (!is.matrix(Z) && !is.numeric(Z)) {
      stop("'Z' must be a numeric vector or matrix")
    }
    if (is.matrix(Z)) {
      if (nrow(Z) != nrow(X)) {
        stop("'Z' must have the same number of rows as 'X'")
      }
    } else {
      if (length(Z) != nrow(X)) {
        stop(
          "'Z' must have the same number of elements as 'X'"
        )
      }
    }
  }
  uses_propensity <- model_object$model_params$propensity_covariate != "none"
  internal_propensity_model <- model_object$model_params$internal_propensity_model
  needs_propensity <- (needs_covariates &&
    uses_propensity &&
    (!internal_propensity_model))
  if (needs_propensity) {
    if (is.null(propensity)) {
      stop(
        "'propensity' must be provided in order to compute the requested intervals"
      )
    }
    if (!is.matrix(propensity) && !is.numeric(propensity)) {
      stop("'propensity' must be a numeric vector or matrix")
    }
    if (is.matrix(propensity)) {
      if (nrow(propensity) != nrow(X)) {
        stop("'propensity' must have the same number of rows as 'X'")
      }
    } else {
      if (length(propensity) != nrow(X)) {
        stop(
          "'propensity' must have the same number of elements as 'X'"
        )
      }
    }
  }
  needs_rfx_data <- model_object$model_params$has_rfx
  if (needs_rfx_data) {
    if (is.null(rfx_group_ids)) {
      stop(
        "'rfx_group_ids' must be provided in order to compute the requested intervals"
      )
    }
    if (length(rfx_group_ids) != nrow(X)) {
      stop(
        "'rfx_group_ids' must have the same length as the number of rows in 'X'"
      )
    }

    if (model_object$model_params$rfx_model_spec == "custom") {
      if (is.null(rfx_basis)) {
        stop(
          "A user-provided basis (`rfx_basis`) must be provided when the model was sampled with a random effects model spec set to 'custom'"
        )
      }
    }

    if (!is.null(rfx_basis)) {
      if (!is.matrix(rfx_basis)) {
        stop("'rfx_basis' must be a matrix")
      }
      if (nrow(rfx_basis) != nrow(X)) {
        stop("'rfx_basis' must have the same number of rows as 'X'")
      }
    }
  }

  # Compute posterior samples
  bcf_preds <- predict(
    model_object,
    X = X,
    Z = Z,
    propensity = propensity,
    rfx_group_ids = rfx_group_ids,
    rfx_basis = rfx_basis,
    type = "posterior",
    terms = c("all"),
    scale = "linear"
  )

  # Compute outcome mean and variance for every posterior draw
  has_rfx <- model_object$model_params$has_rfx
  has_variance_forest <- model_object$model_params$include_variance_forest
  samples_global_variance <- model_object$model_params$sample_sigma2_global
  num_posterior_draws <- model_object$model_params$num_samples
  num_observations <- nrow(X)
  ppd_mean <- bcf_preds$y_hat
  if (has_variance_forest) {
    ppd_variance <- bcf_preds$variance_forest_predictions
  } else {
    if (samples_global_variance) {
      ppd_variance <- matrix(
        rep(
          model_object$sigma2_global_samples,
          each = num_observations
        ),
        nrow = num_observations
      )
    } else {
      ppd_variance <- model_object$model_params$initial_sigma2
    }
  }

  # Sample from the posterior predictive distribution
  if (is.null(num_draws_per_sample)) {
    ppd_draw_multiplier <- posterior_predictive_heuristic_multiplier(
      num_posterior_draws,
      num_observations
    )
  } else {
    ppd_draw_multiplier <- num_draws_per_sample
  }
  num_ppd_draws <- ppd_draw_multiplier * num_posterior_draws * num_observations
  ppd_vector <- rnorm(num_ppd_draws, ppd_mean, sqrt(ppd_variance))

  # Reshape data
  if (ppd_draw_multiplier > 1) {
    ppd_array <- array(
      ppd_vector,
      dim = c(num_observations, num_posterior_draws, ppd_draw_multiplier)
    )
  } else {
    ppd_array <- array(
      ppd_vector,
      dim = c(num_observations, num_posterior_draws)
    )
  }

  # Binarize outcomes for probit models
  if (is_probit) {
    ppd_array <- (ppd_array > 0.0) * 1
  }

  return(ppd_array)
}

#' Sample from the posterior predictive distribution for outcomes modeled by BART
#'
#' @param model_object A fitted BART model object of class `bartmodel`.
#' @param X A matrix or data frame of covariates. Required if the BART model depends on covariates (e.g., contains a mean or variance forest).
#' @param leaf_basis A matrix of bases for mean forest models with regression defined in the leaves. Required for "leaf regression" models.
#' @param rfx_group_ids A vector of group IDs for random effects model. Required if the BART model includes random effects.
#' @param rfx_basis A matrix of bases for random effects model. Required if the BART model includes random effects.
#' @param num_draws_per_sample The number of posterior predictive samples to draw for each posterior sample. Defaults to a heuristic based on the number of samples in a BART model (i.e. if the BART model has >1000 draws, we use 1 draw from the likelihood per sample, otherwise we upsample to ensure intervals are based on at least 1000 posterior predictive draws).
#'
#' @returns Array of posterior predictive samples with dimensions (num_observations, num_posterior_samples, num_draws_per_sample) if num_draws_per_sample > 1, otherwise (num_observations, num_posterior_samples).
#'
#' @export
#' @examples
#' n <- 100
#' p <- 5
#' X <- matrix(rnorm(n * p), nrow = n, ncol = p)
#' y <- 2 * X[,1] + rnorm(n)
#' bart_model <- bart(y_train = y, X_train = X)
#' ppd_samples <- sample_bart_posterior_predictive(
#'   model_object = bart_model, X = X
#' )
sample_bart_posterior_predictive <- function(
  model_object,
  X = NULL,
  leaf_basis = NULL,
  rfx_group_ids = NULL,
  rfx_basis = NULL,
  num_draws_per_sample = NULL
) {
  # Check the provided model object
  check_model_is_valid(model_object)

  # Determine whether the outcome is continuous (Gaussian) or binary (probit-link)
  is_probit <- model_object$model_params$probit_outcome_model

  # Check that all the necessary inputs were provided for interval computation
  needs_covariates <- model_object$model_params$include_mean_forest
  if (needs_covariates) {
    if (is.null(X)) {
      stop(
        "'X' must be provided in order to compute the requested intervals"
      )
    }
    if (!is.matrix(X) && !is.data.frame(X)) {
      stop("'X' must be a matrix or data frame")
    }
  }
  needs_basis <- needs_covariates && model_object$model_params$has_basis
  if (needs_basis) {
    if (is.null(leaf_basis)) {
      stop(
        "'leaf_basis' must be provided in order to compute the requested intervals"
      )
    }
    if (!is.matrix(leaf_basis)) {
      stop("'leaf_basis' must be a matrix")
    }
    if (is.matrix(leaf_basis)) {
      if (nrow(leaf_basis) != nrow(X)) {
        stop("'leaf_basis' must have the same number of rows as 'X'")
      }
    } else {
      if (length(leaf_basis) != nrow(X)) {
        stop("'leaf_basis' must have the same number of elements as 'X'")
      }
    }
  }
  needs_rfx_data <- model_object$model_params$has_rfx
  if (needs_rfx_data) {
    if (is.null(rfx_group_ids)) {
      stop(
        "'rfx_group_ids' must be provided in order to compute the requested intervals"
      )
    }
    if (length(rfx_group_ids) != nrow(X)) {
      stop(
        "'rfx_group_ids' must have the same length as the number of rows in 'X'"
      )
    }
    if (model_object$model_params$rfx_model_spec == "custom") {
      if (is.null(rfx_basis)) {
        stop(
          "A user-provided basis (`rfx_basis`) must be provided when the model was sampled with a random effects model spec set to 'custom'"
        )
      }
      if (!is.matrix(rfx_basis)) {
        stop("'rfx_basis' must be a matrix")
      }
      if (nrow(rfx_basis) != nrow(X)) {
        stop("'rfx_basis' must have the same number of rows as 'X'")
      }
    }
  }

  # Compute posterior samples
  bart_preds <- predict(
    model_object,
    X = X,
    leaf_basis = leaf_basis,
    rfx_group_ids = rfx_group_ids,
    rfx_basis = rfx_basis,
    type = "posterior",
    terms = c("all"),
    scale = "linear"
  )

  # Compute outcome mean and variance for every posterior draw
  has_mean_term <- (model_object$model_params$include_mean_forest ||
    model_object$model_params$has_rfx)
  has_variance_forest <- model_object$model_params$include_variance_forest
  samples_global_variance <- model_object$model_params$sample_sigma2_global
  num_posterior_draws <- model_object$model_params$num_samples
  num_observations <- nrow(X)
  if (has_mean_term) {
    ppd_mean <- bart_preds$y_hat
  } else {
    ppd_mean <- 0
  }
  if (has_variance_forest) {
    ppd_variance <- bart_preds$variance_forest_predictions
  } else {
    if (samples_global_variance) {
      ppd_variance <- matrix(
        rep(
          model_object$sigma2_global_samples,
          each = num_observations
        ),
        nrow = num_observations
      )
    } else {
      ppd_variance <- model_object$model_params$sigma2_init
    }
  }

  # Sample from the posterior predictive distribution
  if (is.null(num_draws_per_sample)) {
    ppd_draw_multiplier <- posterior_predictive_heuristic_multiplier(
      num_posterior_draws,
      num_observations
    )
  } else {
    ppd_draw_multiplier <- num_draws_per_sample
  }
  num_ppd_draws <- ppd_draw_multiplier * num_posterior_draws * num_observations
  ppd_vector <- rnorm(num_ppd_draws, ppd_mean, sqrt(ppd_variance))

  # Reshape data
  if (ppd_draw_multiplier > 1) {
    ppd_array <- array(
      ppd_vector,
      dim = c(num_observations, num_posterior_draws, ppd_draw_multiplier)
    )
  } else {
    ppd_array <- array(
      ppd_vector,
      dim = c(num_observations, num_posterior_draws)
    )
  }

  # Binarize outcomes for probit models
  if (is_probit) {
    ppd_array <- (ppd_array > 0.0) * 1
  }

  return(ppd_array)
}

posterior_predictive_heuristic_multiplier <- function(
  num_samples,
  num_observations
) {
  if (num_samples >= 1000) {
    return(1)
  } else {
    return(ceiling(1000 / num_samples))
  }
}

#' Compute posterior credible intervals for BCF model terms
#'
#' Compute posterior credible intervals for specified terms from a fitted BCF model. Supports intervals for prognostic forests, CATE forests, variance forests, random effects, and overall mean outcome predictions.
#'
#' @param model_object A fitted BCF model object of class `bcfmodel`.
#' @param terms A character string specifying the model term(s) for which to compute intervals. Options for BCF models are `"prognostic_function"`, `"mu"`, `"cate"`, `"tau"`, `"variance_forest"`, `"rfx"`, or `"y_hat"`. Note that `"mu"` is only different from `"prognostic_function"` if random effects are included with a model spec of `"intercept_only"` or `"intercept_plus_treatment"` and `"tau"` is only different from `"cate"` if random effects are included with a model spec of `"intercept_plus_treatment"`.
#' @param level A numeric value between 0 and 1 specifying the credible interval level (default is 0.95 for a 95% credible interval).
#' @param scale (Optional) Scale of mean function predictions. Options are "linear", which returns predictions on the original scale of the mean forest / RFX terms, and "probability", which transforms predictions into a probability of observing `y == 1`. "probability" is only valid for models fit with a probit outcome model. Default: "linear".
#' @param X (Optional) A matrix or data frame of covariates at which to compute the intervals. Required if the requested term depends on covariates (e.g., prognostic forest, CATE forest, variance forest, or overall predictions).
#' @param Z (Optional) A vector or matrix of treatment assignments. Required if the requested term is `"y_hat"` (overall predictions).
#' @param propensity (Optional) A vector or matrix of propensity scores. Required if the underlying model depends on user-provided propensities.
#' @param rfx_group_ids An optional vector of group IDs for random effects. Required if the requested term includes random effects.
#' @param rfx_basis An optional matrix of basis function evaluations for random effects. Required if the requested term includes random effects.
#'
#' @returns A list containing the lower and upper bounds of the credible interval for the specified term. If multiple terms are requested, a named list with intervals for each term is returned.
#'
#' @export
#' @examples
#' n <- 100
#' p <- 5
#' X <- matrix(rnorm(n * p), nrow = n, ncol = p)
#' pi_X <- pnorm(0.5 * X[,1])
#' Z <- rbinom(n, 1, pi_X)
#' mu_X <- X[,1]
#' tau_X <- 0.25 * X[,2]
#' y <- mu_X + tau_X * Z + rnorm(n)
#' bcf_model <- bcf(X_train = X, Z_train = Z, y_train = y,
#'                  propensity_train = pi_X)
#' intervals <- compute_bcf_posterior_interval(
#'  model_object = bcf_model,
#'  terms = c("prognostic_function", "cate"),
#'  X = X,
#'  Z = Z,
#'  propensity = pi_X,
#'  level = 0.90
#' )
compute_bcf_posterior_interval <- function(
  model_object,
  terms,
  level = 0.95,
  scale = "linear",
  X = NULL,
  Z = NULL,
  propensity = NULL,
  rfx_group_ids = NULL,
  rfx_basis = NULL
) {
  # Check the provided model object and requested term
  check_model_is_valid(model_object)
  for (term in terms) {
    check_model_has_term(model_object, term)
  }

  # Handle mean function scale
  if (!is.character(scale)) {
    stop("scale must be a string or character vector")
  }
  if (!(scale %in% c("linear", "probability"))) {
    stop("scale must either be 'linear' or 'probability'")
  }
  is_probit <- model_object$model_params$probit_outcome_model
  if ((scale == "probability") && (!is_probit)) {
    stop(
      "scale cannot be 'probability' for models not fit with a probit outcome model"
    )
  }

  # Check that all the necessary inputs were provided for interval computation
  for (term in terms) {
    if (
      !(term %in%
        c(
          "prognostic_function",
          "mu",
          "cate",
          "tau",
          "variance_forest",
          "rfx",
          "y_hat",
          "all"
        ))
    ) {
      stop(
        paste0(
          "Term '",
          term,
          "' was requested. Valid terms are 'prognostic_function', 'mu', 'cate', 'tau', 'variance_forest', 'rfx', 'y_hat', and 'all'."
        )
      )
    }
  }
  needs_covariates_intermediate <- ((("y_hat" %in% terms) ||
    ("all" %in% terms)))
  needs_covariates <- (("prognostic_function" %in% terms) ||
    ("cate" %in% terms) ||
    ("variance_forest" %in% terms) ||
    (needs_covariates_intermediate))
  if (needs_covariates) {
    if (is.null(X)) {
      stop(
        "'X' must be provided in order to compute the requested intervals"
      )
    }
    if (!is.matrix(X) && !is.data.frame(X)) {
      stop("'X' must be a matrix or data frame")
    }
  }
  needs_treatment <- needs_covariates
  if (needs_treatment) {
    if (is.null(Z)) {
      stop(
        "'Z' must be provided in order to compute the requested intervals"
      )
    }
    if (!is.matrix(Z) && !is.numeric(Z)) {
      stop("'Z' must be a numeric vector or matrix")
    }
    if (is.matrix(Z)) {
      if (nrow(Z) != nrow(X)) {
        stop("'Z' must have the same number of rows as 'X'")
      }
    } else {
      if (length(Z) != nrow(X)) {
        stop(
          "'Z' must have the same number of elements as 'X'"
        )
      }
    }
  }
  uses_propensity <- model_object$model_params$propensity_covariate != "none"
  internal_propensity_model <- model_object$model_params$internal_propensity_model
  needs_propensity <- (needs_covariates &&
    uses_propensity &&
    (!internal_propensity_model))
  if (needs_propensity) {
    if (is.null(propensity)) {
      stop(
        "'propensity' must be provided in order to compute the requested intervals"
      )
    }
    if (!is.matrix(propensity) && !is.numeric(propensity)) {
      stop("'propensity' must be a numeric vector or matrix")
    }
    if (is.matrix(propensity)) {
      if (nrow(propensity) != nrow(X)) {
        stop("'propensity' must have the same number of rows as 'X'")
      }
    } else {
      if (length(propensity) != nrow(X)) {
        stop(
          "'propensity' must have the same number of elements as 'X'"
        )
      }
    }
  }
  needs_rfx_data_intermediate <- ((("y_hat" %in% terms) ||
    ("all" %in% terms)) &&
    model_object$model_params$has_rfx)
  needs_rfx_data <- (("rfx" %in% terms) ||
    (needs_rfx_data_intermediate))
  if (needs_rfx_data) {
    if (is.null(rfx_group_ids)) {
      stop(
        "'rfx_group_ids' must be provided in order to compute the requested intervals"
      )
    }
    if (length(rfx_group_ids) != nrow(X)) {
      stop(
        "'rfx_group_ids' must have the same length as the number of rows in 'X'"
      )
    }

    if (model_object$model_params$rfx_model_spec == "custom") {
      if (is.null(rfx_basis)) {
        stop(
          "A user-provided basis (`rfx_basis`) must be provided when the model was sampled with a random effects model spec set to 'custom'"
        )
      }
    }

    if (!is.null(rfx_basis)) {
      if (!is.matrix(rfx_basis)) {
        stop("'rfx_basis' must be a matrix")
      }
      if (nrow(rfx_basis) != nrow(X)) {
        stop("'rfx_basis' must have the same number of rows as 'X'")
      }
    }
  }

  # Compute posterior matrices for the requested model terms
  predictions <- predict(
    model_object,
    X = X,
    Z = Z,
    propensity = propensity,
    rfx_group_ids = rfx_group_ids,
    rfx_basis = rfx_basis,
    type = "posterior",
    terms = terms,
    scale = scale
  )
  has_multiple_terms <- ifelse(is.list(predictions), TRUE, FALSE)

  # Compute the interval
  if (has_multiple_terms) {
    result <- list()
    for (term_name in names(predictions)) {
      if (!is.null(predictions[[term_name]])) {
        result[[term_name]] <- summarize_interval(
          predictions[[term_name]],
          sample_dim = 2,
          level = level
        )
      } else {
        result[[term_name]] <- NULL
      }
    }
    return(result)
  } else {
    return(summarize_interval(
      predictions,
      sample_dim = 2,
      level = level
    ))
  }
}

#' Compute posterior credible intervals for specified terms from a fitted BART model.
#'
#' Compute posterior credible intervals for specified terms from a fitted BART model. Supports intervals for mean functions, variance functions, random effects, and overall outcome predictions.
#'
#' @param model_object A fitted BART or BCF model object of class `bartmodel`.
#' @param terms A character string specifying the model term(s) for which to compute intervals. Options for BART models are `"mean_forest"`, `"variance_forest"`, `"rfx"`, or `"y_hat"`.
#' @param level A numeric value between 0 and 1 specifying the credible interval level (default is 0.95 for a 95% credible interval).
#' @param scale (Optional) Scale of mean function predictions. Options are "linear", which returns predictions on the original scale of the mean forest / RFX terms, and "probability", which transforms predictions into a probability of observing `y == 1`. "probability" is only valid for models fit with a probit outcome model. Default: "linear".
#' @param X A matrix or data frame of covariates at which to compute the intervals. Required if the requested term depends on covariates (e.g., mean forest, variance forest, or overall predictions).
#' @param leaf_basis An optional matrix of basis function evaluations for mean forest models with regression defined in the leaves. Required for "leaf regression" models.
#' @param rfx_group_ids An optional vector of group IDs for random effects. Required if the requested term includes random effects.
#' @param rfx_basis An optional matrix of basis function evaluations for random effects. Required if the requested term includes random effects.
#'
#' @returns A list containing the lower and upper bounds of the credible interval for the specified term. If multiple terms are requested, a named list with intervals for each term is returned.
#'
#' @export
#' @examples
#' n <- 100
#' p <- 5
#' X <- matrix(rnorm(n * p), nrow = n, ncol = p)
#' y <- 2 * X[,1] + rnorm(n)
#' bart_model <- bart(y_train = y, X_train = X)
#' intervals <- compute_bart_posterior_interval(
#'  model_object = bart_model,
#'  terms = c("mean_forest", "y_hat"),
#'  X = X,
#'  level = 0.90
#' )
#' @export
compute_bart_posterior_interval <- function(
  model_object,
  terms,
  level = 0.95,
  scale = "linear",
  X = NULL,
  leaf_basis = NULL,
  rfx_group_ids = NULL,
  rfx_basis = NULL
) {
  # Check the provided model object and requested term
  check_model_is_valid(model_object)
  for (term in terms) {
    check_model_has_term(model_object, term)
  }

  # Handle mean function scale
  if (!is.character(scale)) {
    stop("scale must be a string or character vector")
  }
  if (!(scale %in% c("linear", "probability"))) {
    stop("scale must either be 'linear' or 'probability'")
  }
  is_probit <- model_object$model_params$probit_outcome_model
  if ((scale == "probability") && (!is_probit)) {
    stop(
      "scale cannot be 'probability' for models not fit with a probit outcome model"
    )
  }

  # Check that all the necessary inputs were provided for interval computation
  needs_covariates_intermediate <- ((("y_hat" %in% terms) ||
    ("all" %in% terms)) &&
    model_object$model_params$include_mean_forest)
  needs_covariates <- (("mean_forest" %in% terms) ||
    ("variance_forest" %in% terms) ||
    (needs_covariates_intermediate))
  if (needs_covariates) {
    if (is.null(X)) {
      stop(
        "'X' must be provided in order to compute the requested intervals"
      )
    }
    if (!is.matrix(X) && !is.data.frame(X)) {
      stop("'X' must be a matrix or data frame")
    }
  }
  needs_basis <- needs_covariates && model_object$model_params$has_basis
  if (needs_basis) {
    if (is.null(leaf_basis)) {
      stop(
        "'leaf_basis' must be provided in order to compute the requested intervals"
      )
    }
    if (!is.matrix(leaf_basis)) {
      stop("'leaf_basis' must be a matrix")
    }
    if (is.matrix(leaf_basis)) {
      if (nrow(leaf_basis) != nrow(X)) {
        stop("'leaf_basis' must have the same number of rows as 'X'")
      }
    } else {
      if (length(leaf_basis) != nrow(X)) {
        stop("'leaf_basis' must have the same number of elements as 'X'")
      }
    }
  }
  needs_rfx_data_intermediate <- ((("y_hat" %in% terms) ||
    ("all" %in% terms)) &&
    model_object$model_params$has_rfx)
  needs_rfx_data <- (("rfx" %in% terms) ||
    (needs_rfx_data_intermediate))
  if (needs_rfx_data) {
    if (is.null(rfx_group_ids)) {
      stop(
        "'rfx_group_ids' must be provided in order to compute the requested intervals"
      )
    }
    if (length(rfx_group_ids) != nrow(X)) {
      stop(
        "'rfx_group_ids' must have the same length as the number of rows in 'X'"
      )
    }
    if (model_object$model_params$rfx_model_spec == "custom") {
      if (is.null(rfx_basis)) {
        stop(
          "A user-provided basis (`rfx_basis`) must be provided when the model was sampled with a random effects model spec set to 'custom'"
        )
      }
      if (!is.matrix(rfx_basis)) {
        stop("'rfx_basis' must be a matrix")
      }
      if (nrow(rfx_basis) != nrow(X)) {
        stop("'rfx_basis' must have the same number of rows as 'X'")
      }
    }
  }

  # Compute posterior matrices for the requested model terms
  predictions <- predict(
    model_object,
    X = X,
    leaf_basis = leaf_basis,
    rfx_group_ids = rfx_group_ids,
    rfx_basis = rfx_basis,
    type = "posterior",
    terms = terms,
    scale = scale
  )
  has_multiple_terms <- ifelse(is.list(predictions), TRUE, FALSE)

  # Compute the interval
  if (has_multiple_terms) {
    result <- list()
    for (term_name in names(predictions)) {
      if (!is.null(predictions[[term_name]])) {
        result[[term_name]] <- summarize_interval(
          predictions[[term_name]],
          sample_dim = 2,
          level = level
        )
      } else {
        result[[term_name]] <- NULL
      }
    }
    return(result)
  } else {
    return(summarize_interval(
      predictions,
      sample_dim = 2,
      level = level
    ))
  }
}

summarize_interval <- function(array, sample_dim = 2, level = 0.95) {
  # Check that the array is numeric and at least 2 dimensional
  stopifnot(is.numeric(array) && length(dim(array)) >= 2)

  # Compute lower and upper quantiles based on the requested interval
  quantile_lb <- (1 - level) / 2
  quantile_ub <- 1 - quantile_lb

  # Determine the dimensions over which interval is computed
  apply_dim <- setdiff(1:length(dim(array)), sample_dim)

  # Calculate the interval
  result_lb <- apply(array, apply_dim, function(x) {
    quantile(x, probs = quantile_lb, names = FALSE)
  })
  result_ub <- apply(array, apply_dim, function(x) {
    quantile(x, probs = quantile_ub, names = FALSE)
  })

  return(list(lower = result_lb, upper = result_ub))
}

check_model_is_valid <- function(model_object) {
  if (
    (!inherits(model_object, "bartmodel")) &&
      (!inherits(model_object, "bcfmodel"))
  ) {
    stop("'model_object' must be a bartmodel or bcfmodel")
  }
}

check_model_has_term <- function(model_object, term) {
  # Parse inputs
  if (!is.character(term) || length(term) != 1) {
    stop("'term' must be a single character string")
  }
  if (
    (!inherits(model_object, "bartmodel")) &&
      (!inherits(model_object, "bcfmodel"))
  ) {
    stop("'model_object' must be a bartmodel or bcfmodel")
  }
  model_type <- ifelse(inherits(model_object, "bartmodel"), "bart", "bcf")

  # Check if the term was fitted as part of the provided model
  if (model_type == "bart") {
    validate_bart_term(term)
    return(bart_model_has_term(model_object, term))
  } else {
    validate_bcf_term(term)
    return(bcf_model_has_term(model_object, term))
  }
}

bart_model_has_term <- function(model_object, term) {
  if (term == "mean_forest") {
    return(model_object$model_params$include_mean_forest)
  } else if (term == "variance_forest") {
    return(model_object$model_params$include_variance_forest)
  } else if (term == "rfx") {
    return(model_object$model_params$has_rfx)
  } else if (term == "y_hat") {
    return(
      model_object$model_params$include_mean_forest ||
        model_object$model_params$has_rfx
    )
  } else if (term == "all") {
    return(TRUE)
  } else {
    return(FALSE)
  }
}

bcf_model_has_term <- function(model_object, term) {
  if (term == "prognostic_function") {
    return(TRUE)
  } else if (term == "mu") {
    return(TRUE)
  } else if (term == "cate") {
    return(TRUE)
  } else if (term == "tau") {
    return(TRUE)
  } else if (term == "variance_forest") {
    return(model_object$model_params$include_variance_forest)
  } else if (term == "rfx") {
    return(model_object$model_params$has_rfx)
  } else if (term == "y_hat") {
    return(TRUE)
  } else if (term == "all") {
    return(TRUE)
  } else {
    return(FALSE)
  }
}

validate_bart_term <- function(term) {
  model_terms <- c("mean_forest", "variance_forest", "rfx", "y_hat", "all")
  if (!(term %in% model_terms)) {
    stop(
      "'term' must be one of 'mean_forest', 'variance_forest', 'rfx', 'y_hat', or 'all' for bartmodel objects"
    )
  }
}

validate_bcf_term <- function(term) {
  model_terms <- c(
    "prognostic_function",
    "mu",
    "cate",
    "tau",
    "variance_forest",
    "rfx",
    "y_hat",
    "all"
  )
  if (!(term %in% model_terms)) {
    stop(
      "'term' must be one of 'prognostic_function', 'mu', 'cate', 'tau', 'variance_forest', 'rfx', 'y_hat', or 'all' for bcfmodel objects"
    )
  }
}
