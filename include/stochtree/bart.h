/*!
 * Copyright (c) 2026 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_BART_H_
#define STOCHTREE_BART_H_

#include <memory>
#include <vector>
#include "stochtree/random_effects.h"
#include <stochtree/container.h>
#include <stochtree/meta.h>

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
  int num_trees_mean = 200;                     // number of trees in the mean forest
  double alpha_mean = 0.95;                     // alpha parameter for mean forest tree prior
  double beta_mean = 2.0;                       // beta parameter for mean forest tree prior
  int min_samples_leaf_mean = 5;                // minimum number of samples per leaf for mean forest
  int max_depth_mean = -1;                      // maximum depth for mean forest trees (-1 means no maximum)
  bool leaf_constant_mean = true;               // whether to use constant leaf model for mean forest
  int leaf_dim_mean = 1;                        // dimension of the leaf for mean forest
  bool exponentiated_leaf_mean = false;         // whether to exponentiate leaf predictions for mean forest
  int num_features_subsample_mean = 0;          // number of features to subsample for each mean forest split (0 means no subsampling)
  double a_sigma2_mean = 3.0;                   // shape parameter for inverse gamma prior on mean forest leaf scale
  double b_sigma2_mean = -1.0;                  // scale parameter for inverse gamma prior on mean forest leaf scale (-1 is a sentinel value that triggers a data-informed calibration based on the variance of the outcome and the number of trees)
  double sigma2_mean_init = -1.0;               // initial value of mean forest leaf scale (-1 is a sentinel value that triggers a data-informed calibration based on the variance of the outcome and the number of trees)
  std::vector<double> var_weights_mean;         // variable weights for mean forest splits (should be same length as number of covariates in the dataset)
  std::vector<double> sigma2_leaf_mean_matrix;  // prior covariance matrix Sigma_0 for multivariate leaf regression, stored column-major (size leaf_dim_mean^2); empty = use sigma2_mean_init * I
  bool sample_sigma2_leaf_mean = false;         // whether to sample mean forest leaf scale (if false, it will be fixed at sigma2_mean_init)
  std::vector<int> sweep_update_indices_mean;   // indices of trees to update in a given sweep (should be subset of [0, num_trees - 1])
  MeanLeafModelType mean_leaf_model_type;       // leaf model type for mean forest
  int num_classes_cloglog = 0;                  // number of classes for cloglog ordinal leaf model (should be set if mean_leaf_model_type = CloglogOrdinal)
  double cloglog_leaf_prior_shape = 2.0;        // shape parameter for cloglog ordinal leaf model prior
  double cloglog_leaf_prior_scale = 2.0;        // scale parameter for cloglog ordinal leaf model prior
  double cloglog_cutpoint_0 = 0.0;              // Fixed value of the first log-scale cutpoint for the cloglog model (defaults to 0 for identifiability)

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
};

}  // namespace StochTree

#endif  // STOCHTREE_BART_H_
