/*!
 * Copyright (c) 2026 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_BCF_H_
#define STOCHTREE_BCF_H_

#include <memory>
#include <vector>
#include "stochtree/random_effects.h"
#include <stochtree/container.h>
#include <stochtree/meta.h>

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
  int cutpoint_grid_size = 100;                         // number of cutpoints to consider for each covariate when sampling splits
  std::vector<FeatureType> feature_types;               // feature types for each covariate (should be same length as number of covariates in the dataset), where 0 = continuous, 1 = categorical
  LinkFunction link_function = LinkFunction::Identity;  // link function to use (Identity, Probit, Cloglog)
  OutcomeType outcome_type = OutcomeType::Continuous;   // type of the outcome variable (Continuous, Binary, Ordinal)
  int random_seed = -1;                                 // random seed for reproducibility (if negative, a random seed will be generated)
  bool keep_gfr = true;                                 // whether or not to keep GFR samples or simply use them to warm-start an MCMC chain
  bool keep_burnin = false;                             // whether or not to keep "burn-in" MCMC samples (largely a debugging flag)
  bool adaptive_coding = false;                         // whether or not to use adaptive coding for the BCF model

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

  // Variance forest parameters
  int num_trees_variance = 0;                      // number of trees in the variance forest
  double leaf_prior_calibration_param = 1.5;       // calibration parameter for variance forest leaf prior
  double shape_variance_forest = -1.0;             // shape parameter for variance forest leaf model (calibrated internally based on leaf_prior_calibration_param if set to sentinel value of -1)
  double scale_variance_forest = -1.0;             // scale parameter for variance forest leaf model (calibrated internally based on leaf_prior_calibration_param if set to sentinel value of -1)
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
  // Posterior samples of training set outcome predictions (num_samples x n_train, stored column-major)
  std::vector<double> y_hat_train;

  // Posterior samples of training set prognostic forest predictions (num_samples x n_train, stored column-major)
  std::vector<double> mu_forest_predictions_train;

  // Posterior samples of training set treatment effect forest predictions (num_samples x n_train, stored column-major)
  std::vector<double> tau_forest_predictions_train;

  // Posterior samples of training set variance forest predictions (num_samples x n_train, stored column-major)
  std::vector<double> variance_forest_predictions_train;

  // Posterior samples of test set outcome predictions (num_samples x n_train, stored column-major)
  std::vector<double> y_hat_test;

  // Posterior samples of test set prognostic forest predictions (num_samples x n_test, stored column-major)
  std::vector<double> mu_forest_predictions_test;

  // Posterior samples of test set treatment effect forest predictions (num_samples x n_test, stored column-major)
  std::vector<double> tau_forest_predictions_test;

  // Posterior samples of test set variance forest predictions (num_samples x n_test, stored column-major)
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

  // Posterior samples of training set RFX predictions (num_samples x n_train, stored column-major)
  std::vector<double> rfx_predictions_train;

  // Posterior samples of test set RFX predictions (num_samples x n_test, stored column-major)
  std::vector<double> rfx_predictions_test;

  // Adaptive coding parameter samples (num_samples x 2, stored column-major, with b0 / control parameter in the first column and b1 / treatment parameter in the second column)
  std::vector<double> adaptive_coding_samples;

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
};

}  // namespace StochTree

#endif  // STOCHTREE_BCF_H_
