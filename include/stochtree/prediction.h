/*!
 * Copyright (c) 2026 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_PREDICTION_H_
#define STOCHTREE_PREDICTION_H_

#include <vector>
#include <stochtree/bart.h>
#include <stochtree/bcf.h>
#include <stochtree/container.h>
#include <stochtree/meta.h>
#include <stochtree/random_effects.h>

namespace StochTree {

/*! \brief Determines whether posterior predictions are returned as-is or pre-aggregated. */
enum class PredType {
  kPosterior,
  kMean
};

/*! \brief Determines the scale of predictions (i.e. whether a probability / class transformation is applied)
 *
 * \details Options are
 *
 *    - linear (i.e. raw) scale,
 *    - probability scale,
 *    - class predictions.
 *
 * Only valid for binary / ordinal outcome models.
 */
enum class PredScale {
  kLinear,
  kProbability,
  kClass
};

/*! \brief Selector for model terms that should be predicted. */
struct BARTPredTerms {
  bool y_hat = true;
  bool mean_forest = false;
  bool variance_forest = false;
  bool random_effects = false;
};

/*! \brief Struct returning BART model predictions
 *
 * \details The BART prediction routine can return predictions of one or more model terms,
 * and this struct serves as a "container" for model predictions. All observation-specific
 * prediction terms can be:
 *   1. Pre-aggregated (type = mean) or contain the entire posterior (type = posterior)
 *   2. Linear scale, probability scale or class predictions (for binary / ordinal outcomes)
 */
struct BARTPredictionResult {
  // Outcome conditional mean
  std::vector<double> y_hat;

  // Covariate-dependent prognostic term (mu(x))
  std::vector<double> mean_forest_predictions;

  // Conditional variance term
  std::vector<double> variance_forest_predictions;

  // Random effects predictions
  std::vector<double> rfx_predictions;
};

/*! \brief Metadata for the BART prediction routine
 *
 * \details Stores details about the underlying model as well as prediction specifications needed for the prediction routine.
 */
struct BARTPredictionMetadata {
  // Metadata about the samples / model (e.g., number of samples, burn-in, etc.)
  int num_samples = 0;
  int num_obs = 0;
  int num_basis = 0;
  double y_bar = 0.0;
  double y_std = 0.0;
  bool has_variance_forest = false;
  bool has_rfx = false;
  BARTRFXModelSpec rfx_model_spec;
  PredType pred_type = PredType::kPosterior;
  BARTPredTerms pred_terms;
  PredScale pred_scale = PredScale::kLinear;
  LinkFunction link_function = LinkFunction::Identity;
  OutcomeType outcome_type = OutcomeType::Continuous;
  int cloglog_num_classes = 0;
};

/*! \brief BART prediction function
 *
 * \details Accepts BARTData, BARTSamples, and a struct of metadata, which dictates the model terms for which
 * predictions are computed / returned and any transformations done before returning (i.e. pre-aggregation,
 * probit function transformation).
 *
 * \param data Struct wrapping pointers to prediction data from R / Python
 * \param samples Object storing BART samples
 * \param metadata Struct containing prediction metadata
 * \return BARTPredictionResult struct containing prediction vectors
 */
BARTPredictionResult predict_bart_model(BARTData& data, BARTSamples& samples, BARTPredictionMetadata& metadata);

/*! \brief Selector for model terms that should be predicted. */
struct BCFPredTerms {
  bool y_hat = true;
  bool mu_x = false;
  bool tau_x = false;
  bool prognostic_function = false;
  bool cate = false;
  bool conditional_variance = false;
  bool random_effects = false;
};

/*! \brief Struct returning BCF model predictions
 *
 * \details The BCF prediction routine can return predictions of one or more model terms,
 * and this struct serves as a "container" for model predictions. All observation-specific
 * prediction terms can be:
 *   1. Pre-aggregated (type = mean) or contain the entire posterior (type = posterior)
 *   2. Linear scale, probability scale or class predictions (for binary / ordinal outcomes)
 */
struct BCFPredictionResult {
  // Outcome conditional mean
  std::vector<double> y_hat;

  // Covariate-dependent prognostic term (mu(x))
  std::vector<double> mu_x;

  // Covariate-dependent treatment effect term (tau(x))
  std::vector<double> tau_x;

  // Prognostic function (includes mu(x) and any random intercepts, provided random effects
  // were estimated with `intercept_only` or `intercept_plus_treatment` specification)
  std::vector<double> prognostic_function;

  // CATE function (includes tau(x) and any random slopes on treatment, provided
  // random effects were estimated with `intercept_plus_treatment` specification)
  std::vector<double> cate;

  // Conditional variance term
  std::vector<double> conditional_variance;

  // Random effects predictions
  std::vector<double> random_effects;
};

/*! \brief Metadata for the BCF prediction routine
 *
 * \details Stores details about the underlying model as well as prediction specifications needed for the prediction routine.
 */
struct BCFPredictionMetadata {
  // Metadata about the samples / model (e.g., number of samples, burn-in, etc.) could be added here as needed
  int num_samples = 0;
  int num_obs = 0;
  int treatment_dim = 0;
  double y_bar = 0.0;
  double y_std = 0.0;
  bool has_variance_forest = false;
  bool has_rfx = false;
  BCFRFXModelSpec rfx_model_spec;
  bool adaptive_coding = false;
  bool sample_tau_0 = false;
  PredType pred_type = PredType::kPosterior;
  BCFPredTerms pred_terms;
  PredScale pred_scale = PredScale::kLinear;
};

/*! \brief BCF prediction function
 *
 * \details Accepts BCFData, BCFSamples, and a struct of metadata, which dictates the model terms for which
 * predictions are computed / returned and any transformations done before returning (i.e. pre-aggregation,
 * probit function transformation).
 *
 * \param data Struct wrapping pointers to prediction data from R / Python
 * \param samples Object storing BCF samples
 * \param metadata Struct containing prediction metadata
 * \return BCFPredictionResult struct containing prediction vectors
 */
BCFPredictionResult predict_bcf_model(BCFData& data, BCFSamples& samples, BCFPredictionMetadata& metadata);

}  // namespace StochTree

#endif  // STOCHTREE_PREDICTION_H_
