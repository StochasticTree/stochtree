/*!
 * Copyright (c) 2026 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_BART_H_
#define STOCHTREE_BART_H_

#include <memory>
#include <vector>
#include <stochtree/container.h>
#include <stochtree/log.h>
#include <stochtree/meta.h>
#include <stochtree/random_effects.h>

namespace StochTree {

enum class BARTRFXModelSpec {
  Custom,
  InterceptOnly
};

struct BARTData {
  // Train set covariates
  double* X_train = nullptr;
  int n_train = 0;
  int p = 0;

  // Test set covariates
  double* X_test = nullptr;
  int n_test = 0;

  // Train set outcome
  double* y_train = nullptr;

  // Basis for leaf regression
  double* basis_train = nullptr;
  double* basis_test = nullptr;
  int basis_dim = 0;

  // Observation weights
  double* obs_weights_train = nullptr;
  double* obs_weights_test = nullptr;

  // Random effects
  int* rfx_group_ids_train = nullptr;
  int* rfx_group_ids_test = nullptr;
  double* rfx_basis_train = nullptr;
  double* rfx_basis_test = nullptr;
  int rfx_num_groups = 0;
  int rfx_basis_dim = 0;
};

struct BARTConfig {
  // High level parameters
  bool standardize_outcome = true;                      // whether to standardize the outcome before fitting and unstandardize predictions after
  int num_threads = 1;                                  // number of threads to use for sampling
  bool verbose = false;                                 // whether to print sampler progress to the console
  int cutpoint_grid_size = 100;                         // number of cutpoints to consider for each covariate when sampling splits
  std::vector<FeatureType> feature_types;               // feature types for each covariate (should be same length as number of covariates in the dataset), where 0 = continuous, 1 = categorical
  LinkFunction link_function = LinkFunction::Identity;  // link function to use (Identity, Probit, Cloglog)
  OutcomeType outcome_type = OutcomeType::Continuous;   // type of the outcome variable (Continuous, Binary, Ordinal)
  int random_seed = -1;                                 // random seed for reproducibility (if negative, a random seed will be generated)
  bool keep_gfr = true;                                 // whether or not to keep GFR samples or simply use them to warm-start an MCMC chain
  bool keep_burnin = false;                             // whether or not to keep "burn-in" MCMC samples (largely a debugging flag)

  // Global error variance parameters
  double a_sigma2_global = 0.0;      // shape parameter for inverse gamma prior on global error variance
  double b_sigma2_global = 0.0;      // scale parameter for inverse gamma prior on global error variance
  double sigma2_global_init = 1.0;   // initial value for global error variance
  bool sample_sigma2_global = true;  // whether to sample global error variance (if false, it will be fixed at sigma2_global_init)

  // Mean forest parameters
  int num_trees_mean = 200;                                                      // number of trees in the mean forest
  double alpha_mean = 0.95;                                                      // alpha parameter for mean forest tree prior
  double beta_mean = 2.0;                                                        // beta parameter for mean forest tree prior
  int min_samples_leaf_mean = 5;                                                 // minimum number of samples per leaf for mean forest
  int max_depth_mean = -1;                                                       // maximum depth for mean forest trees (-1 means no maximum)
  bool leaf_constant_mean = true;                                                // whether to use constant leaf model for mean forest
  int leaf_dim_mean = 1;                                                         // dimension of the leaf for mean forest
  bool exponentiated_leaf_mean = false;                                          // whether to exponentiate leaf predictions for mean forest
  int num_features_subsample_mean = 0;                                           // number of features to subsample for each mean forest split (0 means no subsampling)
  double a_sigma2_mean = 3.0;                                                    // shape parameter for inverse gamma prior on mean forest leaf scale
  double b_sigma2_mean = -1.0;                                                   // scale parameter for inverse gamma prior on mean forest leaf scale (-1 is a sentinel value that triggers a data-informed calibration based on the variance of the outcome and the number of trees)
  double sigma2_mean_init = -1.0;                                                // initial value of mean forest leaf scale (-1 is a sentinel value that triggers a data-informed calibration based on the variance of the outcome and the number of trees)
  std::vector<double> var_weights_mean;                                          // variable weights for mean forest splits (should be same length as number of covariates in the dataset)
  std::vector<double> sigma2_leaf_mean_matrix;                                   // prior covariance matrix Sigma_0 for multivariate leaf regression, stored column-major (size leaf_dim_mean^2); empty = use sigma2_mean_init * I
  bool sample_sigma2_leaf_mean = false;                                          // whether to sample mean forest leaf scale (if false, it will be fixed at sigma2_mean_init)
  std::vector<int> sweep_update_indices_mean;                                    // indices of trees to update in a given sweep (should be subset of [0, num_trees - 1])
  MeanLeafModelType mean_leaf_model_type = MeanLeafModelType::GaussianConstant;  // leaf model type for mean forest
  int num_classes_cloglog = 0;                                                   // number of classes for cloglog ordinal leaf model (should be set if mean_leaf_model_type = CloglogOrdinal)
  double cloglog_leaf_prior_shape = 2.0;                                         // shape parameter for cloglog ordinal leaf model prior
  double cloglog_leaf_prior_scale = 2.0;                                         // scale parameter for cloglog ordinal leaf model prior
  double cloglog_cutpoint_0 = 0.0;                                               // Fixed value of the first log-scale cutpoint for the cloglog model (defaults to 0 for identifiability)

  // Variance forest parameters
  int num_trees_variance = 0;                      // number of trees in the variance forest
  double leaf_prior_calibration_param = 1.5;       // calibration parameter for variance forest leaf prior
  double shape_variance_forest = -1.0;             // shape parameter for variance forest leaf model (calibrated internally based on leaf_prior_calibration_param if set to sentinel value of -1)
  double scale_variance_forest = -1.0;             // scale parameter for variance forest leaf model (calibrated internally based on leaf_prior_calibration_param if set to sentinel value of -1)
  double variance_forest_leaf_init = -1.0;         // initial (raw-scale) root value for the variance forest; each leaf starts at log(value)/num_trees_variance. Sentinel <= 0 = calibrate internally
  double alpha_variance = 0.5;                     // alpha parameter for variance forest tree prior
  double beta_variance = 2.0;                      // beta parameter for variance forest tree prior
  int min_samples_leaf_variance = 5;               // minimum number of samples per leaf for variance forest
  int max_depth_variance = -1;                     // maximum depth for variance forest trees (-1 means no maximum)
  bool leaf_constant_variance = true;              // whether to use constant leaf model for variance forest
  int leaf_dim_variance = 1;                       // dimension of the leaf for variance forest (should be 1 if leaf_constant_variance=true)
  bool exponentiated_leaf_variance = true;         // whether to exponentiate leaf predictions for variance forest
  int num_features_subsample_variance = 0;         // number of features to subsample for each variance forest split (0 means no subsampling)
  std::vector<double> var_weights_variance;        // variable weights for variance forest splits (should be same length as number of covariates in the dataset)
  std::vector<int> sweep_update_indices_variance;  // indices of trees to update in a given sweep (should be subset of [0, num_trees - 1])

  // Random effects parameters
  bool has_random_effects = false;                             // whether or not a model includes a random effects term
  BARTRFXModelSpec rfx_model_spec = BARTRFXModelSpec::Custom;  // specification for the random effects model; custom relies on a user-provided basis while intercept-only constructs a varying intercept model without needing a user-provided basis
  std::vector<double> rfx_working_parameter_mean_prior;        // vector of dimension num_basis; empty = use zeros
  std::vector<double> rfx_group_parameter_mean_prior;          // matrix of dimension num_basis x num_groups, stored column-major; empty = use zeros
  std::vector<double> rfx_working_parameter_cov_prior;         // matrix of dimension num_basis x num_basis, stored column-major; empty = use identity matrix
  std::vector<double> rfx_group_parameter_cov_prior;           // matrix of dimension num_basis x num_basis, stored column-major; empty = use identity matrix
  double rfx_variance_prior_shape = 1.0;                       // shape parameter for variance prior in random effects model
  double rfx_variance_prior_scale = 1.0;                       // scale parameter for variance prior in random effects model

  // TODO: Other parameters ...
};

struct BARTSamples {
  // Posterior samples of training set mean forest predictions (num_samples x n_train, stored column-major)
  std::vector<double> mean_forest_predictions_train;

  // Posterior samples of training set variance forest predictions (num_samples x n_train, stored column-major)
  std::vector<double> variance_forest_predictions_train;

  // Posterior samples of test set mean forest predictions (num_samples x n_test, stored column-major)
  std::vector<double> mean_forest_predictions_test;

  // Posterior samples of test set variance forest predictions (num_samples x n_test, stored column-major)
  std::vector<double> variance_forest_predictions_test;

  // Posterior samples of global error variance (num_samples)
  std::vector<double> global_error_variance_samples;

  // Posterior samples of leaf scale (num_samples)
  std::vector<double> leaf_scale_samples;

  // Pointer to sampled mean forests
  std::unique_ptr<ForestContainer> mean_forests;

  // Pointer to sampled variance forests
  std::unique_ptr<ForestContainer> variance_forests;

  // Posterior samples of cloglog cutpoint parameters (num_samples x num_classes - 1, stored column-major)
  std::vector<double> cloglog_cutpoint_samples;

  // Posterior samples of training set RFX predictions (num_samples x n_train, stored column-major)
  std::vector<double> rfx_predictions_train;

  // Posterior samples of test set RFX predictions (num_samples x n_test, stored column-major)
  std::vector<double> rfx_predictions_test;

  // Pointer to random effects sample container and label mapping
  std::unique_ptr<RandomEffectsContainer> rfx_container;
  std::unique_ptr<LabelMapper> rfx_label_mapper;

  // Metadata about the samples (e.g., number of samples, burn-in, etc.) could be added here as needed
  int num_samples = 0;
  int num_train = 0;
  int num_test = 0;
  double y_bar = 0.0;
  double y_std = 0.0;

  // Serialize the samples-owned subtree (forests + parameter traces + intrinsic scalars) into a
  // JSON object. This is the shared C++ source of truth for BART (de)serialization; the per-language
  // layer writes the surrounding envelope (model_params, covariate preprocessor, schema_version) into
  // the same object. Key layout matches the existing R/Python output exactly so the wire format is
  // unchanged (forests under named keys, parameter traces under a "parameters" subfolder, intrinsic
  // scalars top-level). nlohmann dumps keys sorted, so insertion order is irrelevant to the bytes.
  // NOTE: random effects and cloglog cutpoint samples are not yet routed through this path; callers
  // with those still use the per-language serializer. Guarded to avoid silently dropping them.
  nlohmann::json ToJson() const {
    if (rfx_container != nullptr || rfx_label_mapper != nullptr) {
      Log::Fatal("BARTSamples::ToJson does not yet support random effects");
    }
    if (!cloglog_cutpoint_samples.empty()) {
      Log::Fatal("BARTSamples::ToJson does not yet support cloglog cutpoint samples");
    }
    nlohmann::json obj;
    // Forests, under self-describing named keys, with the num_forests counter
    nlohmann::json forests = nlohmann::json::object();
    int num_forests = 0;
    if (mean_forests != nullptr) {
      forests.emplace("mean_forest", mean_forests->to_json());
      num_forests++;
    }
    if (variance_forests != nullptr) {
      forests.emplace("variance_forest", variance_forests->to_json());
      num_forests++;
    }
    obj.emplace("forests", forests);
    obj.emplace("num_forests", num_forests);
    // Parameter traces, under the "parameters" subfolder (presence inferred from non-empty vectors)
    nlohmann::json parameters = nlohmann::json::object();
    if (!global_error_variance_samples.empty()) {
      parameters.emplace("sigma2_global_samples", global_error_variance_samples);
    }
    if (!leaf_scale_samples.empty()) {
      parameters.emplace("sigma2_leaf_samples", leaf_scale_samples);
    }
    if (!parameters.empty()) {
      obj.emplace("parameters", parameters);
    }
    // Intrinsic scalars (stored in user-facing scale, matching the existing wire format)
    obj.emplace("outcome_mean", y_bar);
    obj.emplace("outcome_scale", y_std);
    obj.emplace("num_samples", num_samples);
    return obj;
  }

  // Populate this BARTSamples from the samples-owned subtree of a parsed JSON object. Presence is
  // inferred from the JSON structure (does "forests" contain "mean_forest"? does "parameters"
  // contain "sigma2_global_samples"?) rather than from the envelope's boolean flags, so the samples
  // (de)serialization is self-contained. Inverse of ToJson(); see its note re: rfx/cloglog scope.
  void FromJson(const nlohmann::json& obj) {
    if (obj.contains("num_random_effects") && obj.at("num_random_effects").get<int>() > 0) {
      Log::Fatal("BARTSamples::FromJson does not yet support random effects");
    }
    if (obj.contains("forests")) {
      const nlohmann::json& forests = obj.at("forests");
      if (forests.contains("mean_forest")) {
        mean_forests = std::make_unique<ForestContainer>(0, 0, false, false);
        mean_forests->from_json(forests.at("mean_forest"));
      }
      if (forests.contains("variance_forest")) {
        variance_forests = std::make_unique<ForestContainer>(0, 0, false, false);
        variance_forests->from_json(forests.at("variance_forest"));
      }
    }
    if (obj.contains("parameters")) {
      const nlohmann::json& parameters = obj.at("parameters");
      if (parameters.contains("sigma2_global_samples")) {
        global_error_variance_samples = parameters.at("sigma2_global_samples").get<std::vector<double>>();
      }
      if (parameters.contains("sigma2_leaf_samples")) {
        leaf_scale_samples = parameters.at("sigma2_leaf_samples").get<std::vector<double>>();
      }
    }
    if (obj.contains("outcome_mean")) y_bar = obj.at("outcome_mean").get<double>();
    if (obj.contains("outcome_scale")) y_std = obj.at("outcome_scale").get<double>();
    if (obj.contains("num_samples")) num_samples = obj.at("num_samples").get<int>();
  }
};

}  // namespace StochTree

#endif  // STOCHTREE_BART_H_
