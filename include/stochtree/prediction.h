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
enum class BCFPredType {
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
enum class BCFPredScale {
  kLinear,
  kProbability,
  kClass
};

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

/*! \brief Inputs to the BCF prediction routine
 *
 * \details Model outputs from a sampled BCF model are unpacked into arrays / external pointers
 * in both R and Python, rather than retained as a reference to a pure-C++ object. In order to
 * provide those terms back to C++ for prediction, there are three options:
 *  1. Refactor the codebase so that the R and Python interfaces retain an external pointer to a
 *     `BCFSamples` object. This might be the best long-term approach, but in the near term would
 *     require a lot of changes.
 *  2. Copy all of the model outputs back into a `BCFSamples` format and then write the BCF prediction
 *     routine to operate directly on BCFSamples. This requires copying arrays of parameter samples
 *     at minimum. None of these samples scale with the size of the training or test data, so this
 *     might not be prohibitive, but nonetheless, we can likely get by with approach 3:
 *  3. Pass a struct containing raw pointers / references to all model terms, along with requisite dimension information,
 *     write the BCF prediction routine to operate on this reference-based struct.
 */
struct BCFPredictionInput {
  // Posterior samples of global error variance (num_samples)
  double* global_error_variance_samples = nullptr;

  // Posterior samples of leaf scale (num_samples)
  double* leaf_scale_mu_samples = nullptr;
  double* leaf_scale_tau_samples = nullptr;

  // Pointer to sampled prognostic forests
  ForestContainer* mu_forests = nullptr;

  // Pointer to sampled treatment effect forests
  ForestContainer* tau_forests = nullptr;

  // Pointer to sampled variance forests
  ForestContainer* variance_forests = nullptr;

  // Treatment intercept samples (treatment_dim x num_samples, stored column-major; only populated when sample_tau_0=true)
  double* tau_0_samples = nullptr;

  // Adaptive coding parameter samples
  double* b0_samples = nullptr;
  double* b1_samples = nullptr;

  // Pointer to random effects sample container and label mapping
  RandomEffectsContainer* rfx_container = nullptr;
  LabelMapper* rfx_label_mapper = nullptr;

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
  BCFPredType pred_type = BCFPredType::kPosterior;
  BCFPredTerms pred_terms;
  BCFPredScale pred_scale = BCFPredScale::kLinear;
};

/*! \brief BCF prediction function
 *
 * \details Accepts BCFData and a struct of references to BCF model terms (BCFPredictionInput)
 *
 * BCFPredictionInput dictates the model terms for which predictions are computed / returned
 * and any transformations done before returning (i.e. pre-aggregation, probit function transformation).
 *
 * \param data Struct wrapping pointers to prediction data from R / Python
 * \param model_refs Struct wrapping pointers to model terms / parameters and metadata
 * \return BCFPRedictionResult struct containing prediction vectors
 */
BCFPredictionResult predict_bcf_model(BCFData& data, BCFPredictionInput& model_refs);

}  // namespace StochTree

#endif  // STOCHTREE_PREDICTION_H_
