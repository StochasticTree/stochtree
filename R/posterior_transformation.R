#' Sample from the posterior predictive distribution for outcomes modeled by BCF
#'
#' @param model_object A fitted BCF model object of class `bcfmodel`.
#' @param covariates (Optional) A matrix or data frame of covariates at which to compute the intervals. Required if the requested term depends on covariates (e.g., prognostic forest, CATE forest, variance forest, or overall predictions).
#' @param treatment (Optional) A vector or matrix of treatment assignments. Required if the requested term is `"y_hat"` (overall predictions).
#' @param propensity (Optional) A vector or matrix of propensity scores. Required if the requested term is `"y_hat"` (overall predictions) and the underlying model depends on user-provided propensities.
#' @param rfx_group_ids (Optional) A vector of group IDs for random effects model. Required if the BCF model includes random effects.
#' @param rfx_basis (Optional) A matrix of bases for random effects model. Required if the BCF model includes random effects.
#' @param num_draws (Optional) The number of samples to draw from the likelihood, for each draw of the posterior, in computing intervals. Defaults to a heuristic based on the number of samples in a BCF model (i.e. if the BCF model has >1000 draws, we use 1 draw from the likelihood per sample, otherwise we upsample to ensure at least 1000 posterior predictive draws).
#'
#' @returns Array of posterior predictive samples with dimensions (num_observations, num_posterior_samples, num_draws) if num_draws > 1, otherwise (num_observations, num_posterior_samples).
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
#'   model_object = bcf_model, covariates = X,
#'   treatment = Z, propensity = pi_X
#' )
sample_bcf_posterior_predictive <- function(
  model_object,
  covariates = NULL,
  treatment = NULL,
  propensity = NULL,
  rfx_group_ids = NULL,
  rfx_basis = NULL,
  num_draws = NULL
) {
  # Check the provided model object and requested term
  check_model_is_valid(model_object)

  # Determine whether the outcome is continuous (Gaussian) or binary (probit-link)
  is_probit <- model_object$model_params$probit_outcome_model

  # Check that all the necessary inputs were provided for interval computation
  needs_covariates <- TRUE
  if (needs_covariates) {
    if (is.null(covariates)) {
      stop(
        "'covariates' must be provided in order to compute the requested intervals"
      )
    }
    if (!is.matrix(covariates) && !is.data.frame(covariates)) {
      stop("'covariates' must be a matrix or data frame")
    }
  }
  needs_treatment <- needs_covariates
  if (needs_treatment) {
    if (is.null(treatment)) {
      stop(
        "'treatment' must be provided in order to compute the requested intervals"
      )
    }
    if (!is.matrix(treatment) && !is.numeric(treatment)) {
      stop("'treatment' must be a numeric vector or matrix")
    }
    if (is.matrix(treatment)) {
      if (nrow(treatment) != nrow(covariates)) {
        stop("'treatment' must have the same number of rows as 'covariates'")
      }
    } else {
      if (length(treatment) != nrow(covariates)) {
        stop(
          "'treatment' must have the same number of elements as 'covariates'"
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
      if (nrow(propensity) != nrow(covariates)) {
        stop("'propensity' must have the same number of rows as 'covariates'")
      }
    } else {
      if (length(propensity) != nrow(covariates)) {
        stop(
          "'propensity' must have the same number of elements as 'covariates'"
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
    if (length(rfx_group_ids) != nrow(covariates)) {
      stop(
        "'rfx_group_ids' must have the same length as the number of rows in 'covariates'"
      )
    }
    if (is.null(rfx_basis)) {
      stop(
        "'rfx_basis' must be provided in order to compute the requested intervals"
      )
    }
    if (!is.matrix(rfx_basis)) {
      stop("'rfx_basis' must be a matrix")
    }
    if (nrow(rfx_basis) != nrow(covariates)) {
      stop("'rfx_basis' must have the same number of rows as 'covariates'")
    }
  }

  # Compute posterior predictive samples
  bcf_preds <- predict(
    model_object,
    X = covariates,
    Z = treatment,
    propensity = propensity,
    rfx_group_ids = rfx_group_ids,
    rfx_basis = rfx_basis,
    type = "posterior",
    terms = c("all")
  )
  has_rfx <- model_object$model_params$has_rfx
  has_variance_forest <- model_object$model_params$include_variance_forest
  samples_global_variance <- model_object$model_params$sample_sigma2_global
  num_posterior_draws <- model_object$model_params$num_samples
  num_observations <- nrow(covariates)
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
  if (is.null(num_draws)) {
    ppd_draw_multiplier <- posterior_predictive_heuristic_multiplier(
      num_posterior_draws,
      num_observations
    )
  } else {
    ppd_draw_multiplier <- num_draws
  }
  num_ppd_draws <- ppd_draw_multiplier * num_posterior_draws * num_observations
  ppd_vector <- rnorm(num_ppd_draws, ppd_mean, sqrt(ppd_variance))
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

  if (is_probit) {
    ppd_array <- (ppd_array > 0.0) * 1
  }

  return(ppd_array)
}

#' Sample from the posterior predictive distribution for outcomes modeled by BART
#'
#' @param model_object A fitted BART model object of class `bartmodel`.
#' @param covariates A matrix or data frame of covariates at which to compute the intervals. Required if the BART model depends on covariates (e.g., contains a mean or variance forest).
#' @param basis A matrix of bases for mean forest models with regression defined in the leaves. Required for "leaf regression" models.
#' @param rfx_group_ids A vector of group IDs for random effects model. Required if the BART model includes random effects.
#' @param rfx_basis A matrix of bases for random effects model. Required if the BART model includes random effects.
#' @param num_draws The number of posterior predictive samples to draw in computing intervals. Defaults to a heuristic based on the number of samples in a BART model (i.e. if the BART model has >1000 draws, we use 1 draw from the likelihood per sample, otherwise we upsample to ensure intervals are based on at least 1000 posterior predictive draws).
#'
#' @returns Array of posterior predictive samples with dimensions (num_observations, num_posterior_samples, num_draws) if num_draws > 1, otherwise (num_observations, num_posterior_samples).
#'
#' @export
#' @examples
#' n <- 100
#' p <- 5
#' X <- matrix(rnorm(n * p), nrow = n, ncol = p)
#' y <- 2 * X[,1] + rnorm(n)
#' bart_model <- bart(y_train = y, X_train = X)
#' ppd_samples <- sample_bart_posterior_predictive(
#'   model_object = bart_model, covariates = X
#' )
sample_bart_posterior_predictive <- function(
  model_object,
  covariates = NULL,
  basis = NULL,
  rfx_group_ids = NULL,
  rfx_basis = NULL,
  num_draws = NULL
) {
  # Check the provided model object and requested term
  check_model_is_valid(model_object)

  # Determine whether the outcome is continuous (Gaussian) or binary (probit-link)
  is_probit <- model_object$model_params$probit_outcome_model

  # Check that all the necessary inputs were provided for interval computation
  needs_covariates <- model_object$model_params$include_mean_forest
  if (needs_covariates) {
    if (is.null(covariates)) {
      stop(
        "'covariates' must be provided in order to compute the requested intervals"
      )
    }
    if (!is.matrix(covariates) && !is.data.frame(covariates)) {
      stop("'covariates' must be a matrix or data frame")
    }
  }
  needs_basis <- needs_covariates && model_object$model_params$has_basis
  if (needs_basis) {
    if (is.null(basis)) {
      stop(
        "'basis' must be provided in order to compute the requested intervals"
      )
    }
    if (!is.matrix(basis)) {
      stop("'basis' must be a matrix")
    }
    if (is.matrix(basis)) {
      if (nrow(basis) != nrow(covariates)) {
        stop("'basis' must have the same number of rows as 'covariates'")
      }
    } else {
      if (length(basis) != nrow(covariates)) {
        stop("'basis' must have the same number of elements as 'covariates'")
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
    if (length(rfx_group_ids) != nrow(covariates)) {
      stop(
        "'rfx_group_ids' must have the same length as the number of rows in 'covariates'"
      )
    }
    if (is.null(rfx_basis)) {
      stop(
        "'rfx_basis' must be provided in order to compute the requested intervals"
      )
    }
    if (!is.matrix(rfx_basis)) {
      stop("'rfx_basis' must be a matrix")
    }
    if (nrow(rfx_basis) != nrow(covariates)) {
      stop("'rfx_basis' must have the same number of rows as 'covariates'")
    }
  }

  # Compute posterior predictive samples
  bart_preds <- predict(
    model_object,
    covariates = covariates,
    leaf_basis = basis,
    rfx_group_ids = rfx_group_ids,
    rfx_basis = rfx_basis,
    type = "posterior",
    terms = c("all")
  )
  has_mean_term <- (model_object$model_params$include_mean_forest ||
    model_object$model_params$has_rfx)
  has_variance_forest <- model_object$model_params$include_variance_forest
  samples_global_variance <- model_object$model_params$sample_sigma2_global
  num_posterior_draws <- model_object$model_params$num_samples
  num_observations <- nrow(covariates)
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
  if (is.null(num_draws)) {
    ppd_draw_multiplier <- posterior_predictive_heuristic_multiplier(
      num_posterior_draws,
      num_observations
    )
  } else {
    ppd_draw_multiplier <- num_draws
  }
  num_ppd_draws <- ppd_draw_multiplier * num_posterior_draws * num_observations
  ppd_vector <- rnorm(num_ppd_draws, ppd_mean, sqrt(ppd_variance))
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
#' This function computes posterior credible intervals for specified terms from a fitted BCF model. It supports intervals for prognostic forests, CATE forests, variance forests, random effects, and overall mean outcome predictions.
#' @param model_object A fitted BCF model object of class `bcfmodel`.
#' @param term A character string specifying the model term for which to compute intervals. Options for BCF models are `"prognostic_function"`, `"cate"`, `"variance_forest"`, `"rfx"`, or `"y_hat"`.
#' @param level A numeric value between 0 and 1 specifying the credible interval level (default is 0.95 for a 95% credible interval).
#' @param scale (Optional) Scale of mean function predictions. Options are "linear", which returns predictions on the original scale of the mean forest / RFX terms, and "probability", which transforms predictions into a probability of observing `y == 1`. "probability" is only valid for models fit with a probit outcome model. Default: "linear".
#' @param covariates (Optional) A matrix or data frame of covariates at which to compute the intervals. Required if the requested term depends on covariates (e.g., prognostic forest, CATE forest, variance forest, or overall predictions).
#' @param treatment (Optional) A vector or matrix of treatment assignments. Required if the requested term is `"y_hat"` (overall predictions).
#' @param propensity (Optional) A vector or matrix of propensity scores. Required if the requested term is `"y_hat"` (overall predictions) and the underlying model depends on user-provided propensities.
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
#'  covariates = X,
#'  treatment = Z,
#'  propensity = pi_X,
#'  level = 0.90
#' )
compute_bcf_posterior_interval <- function(
  model_object,
  terms,
  level = 0.95,
  scale = "linear",
  covariates = NULL,
  treatment = NULL,
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
  probability_scale <- scale == "probability"

  # Check that all the necessary inputs were provided for interval computation
  needs_covariates_intermediate <- ((("y_hat" %in% terms) ||
    ("all" %in% terms)))
  needs_covariates <- (("prognostic_function" %in% terms) ||
    ("cate" %in% terms) ||
    ("variance_forest" %in% terms) ||
    (needs_covariates_intermediate))
  if (needs_covariates) {
    if (is.null(covariates)) {
      stop(
        "'covariates' must be provided in order to compute the requested intervals"
      )
    }
    if (!is.matrix(covariates) && !is.data.frame(covariates)) {
      stop("'covariates' must be a matrix or data frame")
    }
  }
  needs_treatment <- needs_covariates
  if (needs_treatment) {
    if (is.null(treatment)) {
      stop(
        "'treatment' must be provided in order to compute the requested intervals"
      )
    }
    if (!is.matrix(treatment) && !is.numeric(treatment)) {
      stop("'treatment' must be a numeric vector or matrix")
    }
    if (is.matrix(treatment)) {
      if (nrow(treatment) != nrow(covariates)) {
        stop("'treatment' must have the same number of rows as 'covariates'")
      }
    } else {
      if (length(treatment) != nrow(covariates)) {
        stop(
          "'treatment' must have the same number of elements as 'covariates'"
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
      if (nrow(propensity) != nrow(covariates)) {
        stop("'propensity' must have the same number of rows as 'covariates'")
      }
    } else {
      if (length(propensity) != nrow(covariates)) {
        stop(
          "'propensity' must have the same number of elements as 'covariates'"
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
    if (length(rfx_group_ids) != nrow(covariates)) {
      stop(
        "'rfx_group_ids' must have the same length as the number of rows in 'covariates'"
      )
    }
    if (is.null(rfx_basis)) {
      stop(
        "'rfx_basis' must be provided in order to compute the requested intervals"
      )
    }
    if (!is.matrix(rfx_basis)) {
      stop("'rfx_basis' must be a matrix")
    }
    if (nrow(rfx_basis) != nrow(covariates)) {
      stop("'rfx_basis' must have the same number of rows as 'covariates'")
    }
  }

  # Compute posterior matrices for the requested model terms
  predictions <- predict(
    model_object,
    X = covariates,
    Z = treatment,
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
      result[[term_name]] <- summarize_interval(
        predictions[[term_name]],
        sample_dim = 2,
        level = level
      )
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

#' Compute posterior credible intervals for BART model terms
#'
#' This function computes posterior credible intervals for specified terms from a fitted BART model. It supports intervals for mean functions, variance functions, random effects, and overall predictions.
#' @param model_object A fitted BART or BCF model object of class `bartmodel`.
#' @param term A character string specifying the model term for which to compute intervals. Options for BART models are `"mean_forest"`, `"variance_forest"`, `"rfx"`, or `"y_hat"`.
#' @param level A numeric value between 0 and 1 specifying the credible interval level (default is 0.95 for a 95% credible interval).
#' @param scale (Optional) Scale of mean function predictions. Options are "linear", which returns predictions on the original scale of the mean forest / RFX terms, and "probability", which transforms predictions into a probability of observing `y == 1`. "probability" is only valid for models fit with a probit outcome model. Default: "linear".
#' @param covariates A matrix or data frame of covariates at which to compute the intervals. Required if the requested term depends on covariates (e.g., mean forest, variance forest, or overall predictions).
#' @param basis An optional matrix of basis function evaluations for mean forest models with regression defined in the leaves. Required for "leaf regression" models.
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
#'  covariates = X,
#'  level = 0.90
#' )
#' @export
compute_bart_posterior_interval <- function(
  model_object,
  terms,
  level = 0.95,
  scale = "linear",
  covariates = NULL,
  basis = NULL,
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
  probability_scale <- scale == "probability"

  # Check that all the necessary inputs were provided for interval computation
  needs_covariates_intermediate <- ((("y_hat" %in% terms) ||
    ("all" %in% terms)) &&
    model_object$model_params$include_mean_forest)
  needs_covariates <- (("mean_forest" %in% terms) ||
    ("variance_forest" %in% terms) ||
    (needs_covariates_intermediate))
  if (needs_covariates) {
    if (is.null(covariates)) {
      stop(
        "'covariates' must be provided in order to compute the requested intervals"
      )
    }
    if (!is.matrix(covariates) && !is.data.frame(covariates)) {
      stop("'covariates' must be a matrix or data frame")
    }
  }
  needs_basis <- needs_covariates && model_object$model_params$has_basis
  if (needs_basis) {
    if (is.null(basis)) {
      stop(
        "'basis' must be provided in order to compute the requested intervals"
      )
    }
    if (!is.matrix(basis)) {
      stop("'basis' must be a matrix")
    }
    if (is.matrix(basis)) {
      if (nrow(basis) != nrow(covariates)) {
        stop("'basis' must have the same number of rows as 'covariates'")
      }
    } else {
      if (length(basis) != nrow(covariates)) {
        stop("'basis' must have the same number of elements as 'covariates'")
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
    if (length(rfx_group_ids) != nrow(covariates)) {
      stop(
        "'rfx_group_ids' must have the same length as the number of rows in 'covariates'"
      )
    }
    if (is.null(rfx_basis)) {
      stop(
        "'rfx_basis' must be provided in order to compute the requested intervals"
      )
    }
    if (!is.matrix(rfx_basis)) {
      stop("'rfx_basis' must be a matrix")
    }
    if (nrow(rfx_basis) != nrow(covariates)) {
      stop("'rfx_basis' must have the same number of rows as 'covariates'")
    }
  }

  # Compute posterior matrices for the requested model terms
  predictions <- predict(
    model_object,
    covariates = covariates,
    leaf_basis = basis,
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
      result[[term_name]] <- summarize_interval(
        predictions[[term_name]],
        sample_dim = 2,
        level = level
      )
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
  } else if (term == "cate") {
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
    "cate",
    "variance_forest",
    "rfx",
    "y_hat",
    "all"
  )
  if (!(term %in% model_terms)) {
    stop(
      "'term' must be one of 'prognostic_function', 'cate', 'variance_forest', 'rfx', 'y_hat', or 'all' for bcfmodel objects"
    )
  }
}
