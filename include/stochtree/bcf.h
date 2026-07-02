/*!
 * Copyright (c) 2026 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_BCF_H_
#define STOCHTREE_BCF_H_

#include <memory>
#include <vector>
#include <stochtree/container.h>
#include <stochtree/log.h>
#include <stochtree/meta.h>
#include <stochtree/random_effects.h>

namespace StochTree {

enum class BCFRFXModelSpec {
  Custom,
  InterceptOnly,
  InterceptPlusTreatment
};

struct BCFData {
  // Train set covariates
  double* X_train = nullptr;
  int n_train = 0;
  int p = 0;

  // Test set covariates
  double* X_test = nullptr;
  int n_test = 0;

  // Treatment
  double* treatment_train = nullptr;
  double* treatment_test = nullptr;
  int treatment_dim = 0;

  // Train set outcome
  double* y_train = nullptr;

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

struct BCFConfig {
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
  bool adaptive_coding = false;                         // whether or not to use adaptive coding for the BCF model
  double b_0_init = 0.0;                                // initial value for the b_0 parameter in the adaptive coding scheme (only relevant if adaptive_coding=true)
  double b_1_init = 1.0;                                // initial value for the b_1 parameter in the adaptive coding scheme (only relevant if adaptive_coding=true)

  // Global error variance parameters
  double a_sigma2_global = 0.0;      // shape parameter for inverse gamma prior on global error variance
  double b_sigma2_global = 0.0;      // scale parameter for inverse gamma prior on global error variance
  double sigma2_global_init = 1.0;   // initial value for global error variance
  bool sample_sigma2_global = true;  // whether to sample global error variance (if false, it will be fixed at sigma2_global_init)

  // Prognostic forest parameters
  int num_trees_mu = 200;                    // number of trees in the prognostic forest
  double alpha_mu = 0.95;                    // alpha parameter for prognostic forest tree prior
  double beta_mu = 2.0;                      // beta parameter for prognostic forest tree prior
  int min_samples_leaf_mu = 5;               // minimum number of samples per leaf for prognostic forest
  int max_depth_mu = -1;                     // maximum depth for prognostic forest trees (-1 means no maximum)
  bool leaf_constant_mu = true;              // whether to use constant leaf model for prognostic forest
  int leaf_dim_mu = 1;                       // dimension of the leaf for prognostic forest
  bool exponentiated_leaf_mu = false;        // whether to exponentiate leaf predictions for prognostic forest
  int num_features_subsample_mu = 0;         // number of features to subsample for each prognostic forest split (0 means no subsampling)
  double a_sigma2_mu = 3.0;                  // shape parameter for inverse gamma prior on prognostic forest leaf scale
  double b_sigma2_mu = -1.0;                 // scale parameter for inverse gamma prior on prognostic forest leaf scale (-1 is a sentinel value that triggers a data-informed calibration based on the variance of the outcome and the number of trees)
  double sigma2_mu_init = -1.0;              // initial value of prognostic forest leaf scale (-1 is a sentinel value that triggers a data-informed calibration based on the variance of the outcome and the number of trees)
  std::vector<double> var_weights_mu;        // variable weights for prognostic forest splits (should be same length as number of covariates in the dataset)
  bool sample_sigma2_leaf_mu = false;        // whether to sample prognostic forest leaf scale (if false, it will be fixed at sigma2_mu_init)
  std::vector<int> sweep_update_indices_mu;  // indices of trees to update in a given sweep (should be subset of [0, num_trees - 1])

  // Treatment effect forest parameters
  int num_trees_tau = 50;                                                                   // number of trees in the treatment effect forest
  double alpha_tau = 0.95;                                                                  // alpha parameter for treatment effect forest tree prior
  double beta_tau = 2.0;                                                                    // beta parameter for treatment effect forest tree prior
  int min_samples_leaf_tau = 5;                                                             // minimum number of samples per leaf for treatment effect forest
  int max_depth_tau = -1;                                                                   // maximum depth for treatment effect forest trees (-1 means no maximum)
  bool leaf_constant_tau = false;                                                           // whether to use constant leaf model for treatment effect forest (false for univariate/multivariate regression leaf, true for constant leaf)
  int leaf_dim_tau = 1;                                                                     // dimension of the leaf for treatment effect forest
  bool exponentiated_leaf_tau = false;                                                      // whether to exponentiate leaf predictions for treatment effect forest
  int num_features_subsample_tau = 0;                                                       // number of features to subsample for each treatment effect forest split (0 means no subsampling)
  double a_sigma2_tau = 3.0;                                                                // shape parameter for inverse gamma prior on treatment effect forest leaf scale
  double b_sigma2_tau = -1.0;                                                               // scale parameter for inverse gamma prior on treatment effect forest leaf scale (-1 is a sentinel value that triggers a data-informed calibration based on the variance of the outcome and the number of trees)
  double sigma2_tau_init = -1.0;                                                            // initial value of treatment effect forest leaf scale (-1 is a sentinel value that triggers a data-informed calibration based on the variance of the outcome and the number of trees)
  std::vector<double> var_weights_tau;                                                      // variable weights for treatment effect forest splits (should be same length as number of covariates in the dataset)
  std::vector<double> sigma2_leaf_tau_matrix;                                               // prior covariance matrix Sigma_0 for multivariate leaf regression, stored column-major (size leaf_dim_tau^2); empty = use sigma2_tau_init * I
  bool sample_sigma2_leaf_tau = false;                                                      // whether to sample treatment effect forest leaf scale (if false, it will be fixed at sigma2_tau_init)
  std::vector<int> sweep_update_indices_tau;                                                // indices of trees to update in a given sweep (should be subset of [0, num_trees - 1])
  MeanLeafModelType tau_leaf_model_type = MeanLeafModelType::GaussianUnivariateRegression;  // leaf model type for treatment effect forest
  bool sample_tau_0 = true;                                                                 // whether or not to sample an intercept term on the treatment, additive to the covariate-dependent treatment effect forest
  double tau_0_prior_var_scalar = -1.0;                                                     // scalar-valued prior variance for treatment intercept (only relevant when sample_tau_0=true; -1 is a sentinel value that triggers a data-informed calibration)
  std::vector<double> tau_0_prior_var_multivariate;                                         // vector-valued prior variance for treatment intercept in multivariate treatment case (only relevant when sample_tau_0=true; should be of length treatment_dim; empty = use data-informed calibration)

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
  bool has_random_effects = false;                           // whether or not a model includes a random effects term
  BCFRFXModelSpec rfx_model_spec = BCFRFXModelSpec::Custom;  // specification for the random effects model; custom relies on a user-provided basis while intercept-only constructs a varying intercept model without needing a user-provided basis
  std::vector<double> rfx_working_parameter_mean_prior;      // vector of dimension num_basis; empty = use zeros
  std::vector<double> rfx_group_parameter_mean_prior;        // matrix of dimension num_basis x num_groups, stored column-major; empty = use zeros
  std::vector<double> rfx_working_parameter_cov_prior;       // matrix of dimension num_basis x num_basis, stored column-major; empty = use identity matrix
  std::vector<double> rfx_group_parameter_cov_prior;         // matrix of dimension num_basis x num_basis, stored column-major; empty = use identity matrix
  double rfx_variance_prior_shape = 1.0;                     // shape parameter for variance prior in random effects model
  double rfx_variance_prior_scale = 1.0;                     // scale parameter for variance prior in random effects model

  // TODO: Other parameters ...
};

struct BCFSamples {
  // Posterior samples of training set outcome predictions (n_train x num_samples, stored column-major)
  std::vector<double> y_hat_train;

  // Posterior samples of training set prognostic forest predictions (n_train x num_samples, stored column-major)
  std::vector<double> mu_forest_predictions_train;

  // Posterior samples of training set treatment effect forest predictions (n_train x num_samples, stored column-major)
  std::vector<double> tau_forest_predictions_train;

  // Posterior samples of training set variance forest predictions (n_train x num_samples, stored column-major)
  std::vector<double> variance_forest_predictions_train;

  // Posterior samples of test set outcome predictions (n_test x num_samples, stored column-major)
  std::vector<double> y_hat_test;

  // Posterior samples of test set prognostic forest predictions (n_test x num_samples, stored column-major)
  std::vector<double> mu_forest_predictions_test;

  // Posterior samples of test set treatment effect forest predictions (n_test x num_samples, stored column-major)
  std::vector<double> tau_forest_predictions_test;

  // Posterior samples of test set variance forest predictions (n_test x num_samples, stored column-major)
  std::vector<double> variance_forest_predictions_test;

  // Posterior samples of global error variance (num_samples)
  std::vector<double> global_error_variance_samples;

  // Posterior samples of leaf scale (num_samples)
  std::vector<double> leaf_scale_mu_samples;
  std::vector<double> leaf_scale_tau_samples;

  // Pointer to sampled prognostic forests
  std::unique_ptr<ForestContainer> mu_forests;

  // Pointer to sampled treatment effect forests
  std::unique_ptr<ForestContainer> tau_forests;

  // Pointer to sampled variance forests
  std::unique_ptr<ForestContainer> variance_forests;

  // Posterior samples of training set RFX predictions (n_train x num_samples, stored column-major)
  std::vector<double> rfx_predictions_train;

  // Posterior samples of test set RFX predictions (n_test x num_samples, stored column-major)
  std::vector<double> rfx_predictions_test;

  // Treatment intercept samples (treatment_dim x num_samples, stored column-major; only populated when sample_tau_0=true)
  std::vector<double> tau_0_samples;

  // Adaptive coding parameter samples
  std::vector<double> b0_samples;
  std::vector<double> b1_samples;

  // Pointer to random effects sample container and label mapping
  std::unique_ptr<RandomEffectsContainer> rfx_container;
  std::unique_ptr<LabelMapper> rfx_label_mapper;

  // Metadata about the samples (e.g., number of samples, burn-in, etc.) could be added here as needed
  int num_samples = 0;
  int num_train = 0;
  int num_test = 0;
  int treatment_dim = 0;
  double y_bar = 0.0;
  double y_std = 0.0;

  // Serialize the samples-owned subtree (forests + parameter traces + intrinsic scalars) into a JSON
  // object. BCF mirror of BARTSamples::ToJson -- see that method for the design notes (shared C++
  // source of truth, byte-identical key layout, presence inferred from structure, nlohmann sorts
  // keys so insertion order is irrelevant). Forests use the BCF named keys; the extra BCF parameter
  // traces (leaf_scale_mu/tau, tau_0, b0/b1) go under the same "parameters" subfolder.
  // NOTE: parameter traces are serialized verbatim (identity). The tau_0 user-facing scaling
  // (x y_std) and its multivariate (treatment_dim>1) ravel-order are reconciled at the postprocess /
  // wiring boundary per the locked scale decision, not here. Random effects are not yet routed
  // through this path (guarded to avoid silent drops).
  void AppendToJson(nlohmann::json& obj) const {
    if (rfx_container != nullptr || rfx_label_mapper != nullptr) {
      Log::Fatal("BCFSamples::ToJson does not yet support random effects");
    }
    // Forests, under the BCF self-describing named keys, with the num_forests counter
    nlohmann::json forests = nlohmann::json::object();
    int num_forests = 0;
    if (mu_forests != nullptr) {
      forests.emplace("prognostic_forest", mu_forests->to_json());
      num_forests++;
    }
    if (tau_forests != nullptr) {
      forests.emplace("treatment_forest", tau_forests->to_json());
      num_forests++;
    }
    if (variance_forests != nullptr) {
      forests.emplace("variance_forest", variance_forests->to_json());
      num_forests++;
    }
    obj["forests"] = forests;
    obj["num_forests"] = num_forests;
    // Parameter traces, under the "parameters" subfolder (presence inferred from non-empty vectors)
    nlohmann::json parameters = nlohmann::json::object();
    if (!global_error_variance_samples.empty()) {
      parameters.emplace("sigma2_global_samples", global_error_variance_samples);
    }
    if (!leaf_scale_mu_samples.empty()) {
      parameters.emplace("sigma2_leaf_mu_samples", leaf_scale_mu_samples);
    }
    if (!leaf_scale_tau_samples.empty()) {
      parameters.emplace("sigma2_leaf_tau_samples", leaf_scale_tau_samples);
    }
    if (!b0_samples.empty()) {
      parameters.emplace("b0_samples", b0_samples);
    }
    if (!b1_samples.empty()) {
      parameters.emplace("b1_samples", b1_samples);
    }
    if (!tau_0_samples.empty()) {
      parameters.emplace("tau_0_samples", tau_0_samples);
      obj.emplace("tau_0_dim", treatment_dim);
    }
    if (!parameters.empty()) {
      obj["parameters"] = parameters;
    }
    // Intrinsic scalars (stored in user-facing scale, matching the existing wire format)
    obj.emplace("outcome_mean", y_bar);
    obj.emplace("outcome_scale", y_std);
    obj.emplace("num_samples", num_samples);
    obj.emplace("treatment_dim", treatment_dim);
    // Random effects
    int num_random_effects = 0;
    nlohmann::json rfx = nlohmann::json::object();
    if (rfx_container != nullptr && rfx_label_mapper != nullptr) {
      rfx.emplace("random_effect_container_0", rfx_container->to_json());
      rfx.emplace("random_effect_label_mapper_0", rfx_label_mapper->to_json());
      rfx.emplace("random_effect_groupids_0", rfx_label_mapper->Keys());
      num_random_effects = 1;
    }
    obj["random_effects"] = rfx;
    obj["num_random_effects"] = num_random_effects;
  }

  // Populate this BCFSamples from the samples-owned subtree of a parsed JSON object. Inverse of
  // ToJson(); presence inferred from structure rather than envelope booleans. See ToJson() re: scope.
  void FromJson(const nlohmann::json& obj) {
    // Unpack forests if present, checking for the expected keys
    if (obj.contains("forests")) {
      const nlohmann::json& forests = obj.at("forests");
      if (forests.contains("prognostic_forest")) {
        mu_forests = std::make_unique<ForestContainer>(0, 0, false, false);
        mu_forests->from_json(forests.at("prognostic_forest"));
      }
      if (forests.contains("treatment_forest")) {
        tau_forests = std::make_unique<ForestContainer>(0, 0, false, false);
        tau_forests->from_json(forests.at("treatment_forest"));
      }
      if (forests.contains("variance_forest")) {
        variance_forests = std::make_unique<ForestContainer>(0, 0, false, false);
        variance_forests->from_json(forests.at("variance_forest"));
      }
    }
    // Unpack parameters if present, checking for expected keys
    if (obj.contains("parameters")) {
      const nlohmann::json& parameters = obj.at("parameters");
      if (parameters.contains("sigma2_global_samples")) {
        global_error_variance_samples = parameters.at("sigma2_global_samples").get<std::vector<double>>();
      }
      if (parameters.contains("sigma2_leaf_mu_samples")) {
        leaf_scale_mu_samples = parameters.at("sigma2_leaf_mu_samples").get<std::vector<double>>();
      }
      if (parameters.contains("sigma2_leaf_tau_samples")) {
        leaf_scale_tau_samples = parameters.at("sigma2_leaf_tau_samples").get<std::vector<double>>();
      }
      if (parameters.contains("b0_samples")) {
        b0_samples = parameters.at("b0_samples").get<std::vector<double>>();
      }
      if (parameters.contains("b1_samples")) {
        b1_samples = parameters.at("b1_samples").get<std::vector<double>>();
      }
      if (parameters.contains("tau_0_samples")) {
        tau_0_samples = parameters.at("tau_0_samples").get<std::vector<double>>();
      }
    }
    // Unpack random effects if present, checking for expected keys
    if (obj.contains("num_random_effects") && obj.at("num_random_effects").get<int>() > 0) {
      rfx_container = std::make_unique<RandomEffectsContainer>();
      rfx_label_mapper = std::make_unique<LabelMapper>();
      rfx_container->from_json(obj.at("random_effects").at("random_effect_container_0"));
      rfx_label_mapper->from_json(obj.at("random_effects").at("random_effect_label_mapper_0"));
    }
    // Unpack outcome statistics
    if (obj.contains("outcome_mean")) y_bar = obj.at("outcome_mean").get<double>();
    if (obj.contains("outcome_scale")) y_std = obj.at("outcome_scale").get<double>();
    if (obj.contains("num_samples")) num_samples = obj.at("num_samples").get<int>();
    if (obj.contains("treatment_dim")) treatment_dim = obj.at("treatment_dim").get<int>();
  }

  // Append another chain's draws onto this one (multi-chain combine). BCF mirror of
  // BARTSamples::Merge -- `this` must already be populated, `other` must match model structure
  // (same forests present, same standardization, same treatment_dim). Forests are deep-copied
  // sample-by-sample and parameter traces concatenated, preserving draw order.
  void Merge(const BCFSamples& other) {
    // Runtime checks for samples objects to be combined
    if (y_bar != other.y_bar || y_std != other.y_std) {
      Log::Fatal("Cannot merge BCFSamples with different outcome standardization");
    }
    if (rfx_container != nullptr && other.rfx_container != nullptr) {
      if (rfx_container->NumComponents() != other.rfx_container->NumComponents() ||
          rfx_container->NumGroups() != other.rfx_container->NumGroups()) {
        Log::Fatal("Cannot merge BARTSamples with different random effects structure");
      }
      if (rfx_label_mapper->Keys() != other.rfx_label_mapper->Keys()) {
        Log::Fatal("Cannot merge BARTSamples with different random effects label mapping");
      }
      if (rfx_label_mapper->Map() != other.rfx_label_mapper->Map()) {
        Log::Fatal("Cannot merge BARTSamples with different random effects label mapping");
      }
    }
    if (treatment_dim != other.treatment_dim) {
      Log::Fatal("Cannot merge BCFSamples with different treatment_dim");
    }
    // Append forests if they exist in the samples object
    AppendForestContainerSamples(mu_forests, other.mu_forests, "prognostic");
    AppendForestContainerSamples(tau_forests, other.tau_forests, "treatment");
    AppendForestContainerSamples(variance_forests, other.variance_forests, "variance");
    // Append random effects if they exist in the samples object
    AppendRandomEffectsContainerSamples(rfx_container, other.rfx_container);
    // Append parameters samples
    auto append = [](std::vector<double>& dst, const std::vector<double>& src, const std::string& name = "") {
      if ((!dst.empty() && src.empty()) || (dst.empty() && !src.empty())) {
        Log::Fatal("Cannot merge BARTSamples objects: %s samples present in one chain but not the other", name.c_str());
      }
      dst.insert(dst.end(), src.begin(), src.end());
    };
    append(global_error_variance_samples, other.global_error_variance_samples, "global error variance");
    append(leaf_scale_mu_samples, other.leaf_scale_mu_samples, "leaf scale mu");
    append(leaf_scale_tau_samples, other.leaf_scale_tau_samples, "leaf scale tau");
    append(tau_0_samples, other.tau_0_samples, "tau_0");
    append(b0_samples, other.b0_samples, "b0");
    append(b1_samples, other.b1_samples, "b1");
    num_samples += other.num_samples;
  }
};

}  // namespace StochTree

#endif  // STOCHTREE_BCF_H_
