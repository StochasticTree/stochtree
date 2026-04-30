/*! Copyright (c) 2026 by stochtree authors */
#include <stochtree/bcf.h>
#include <stochtree/bcf_sampler.h>
#include <stochtree/distributions.h>
#include <stochtree/leaf_model.h>
#include <stochtree/meta.h>
#include <stochtree/probit.h>
#include <stochtree/tree_sampler.h>
#include <stochtree/variance_model.h>
#include <memory>
#include <random>
#include "stochtree/data.h"
#include "stochtree/random_effects.h"

namespace StochTree {

void AddModelTermsForProbit(double* outcome_preds, ForestTracker* mu_forest_tracker, ForestTracker* tau_forest_tracker, RandomEffectsTracker* random_effects_tracker, int n) {
  // TODO: Add treatment intercept contribution when that's added to this implementation
  double* mu_preds = mu_forest_tracker->GetSumPredictions();
  double* tau_preds = tau_forest_tracker->GetSumPredictions();
  if (random_effects_tracker != nullptr) {
    double* rfx_preds = random_effects_tracker->GetPredictions();
    for (int i = 0; i < n; i++) {
      outcome_preds[i] = mu_preds[i] + tau_preds[i] + rfx_preds[i];
    }
  } else {
    for (int i = 0; i < n; i++) {
      outcome_preds[i] = mu_preds[i] + tau_preds[i];
    }
  }
}

BCFSampler::BCFSampler(BCFSamples& samples, BCFConfig& config, BCFData& data) : config_{config}, data_{data}, mu_leaf_model_(GaussianConstantLeafModel(0.0)), tau_leaf_model_(GaussianUnivariateRegressionLeafModel(0.0)), variance_leaf_model_(0.0, 0.0) {
  InitializeState(samples);
}

void BCFSampler::InitializeState(BCFSamples& samples) {
  // Validate y_train values match the expected support for discrete link functions
  if (config_.link_function == LinkFunction::Probit) {
    for (int i = 0; i < data_.n_train; i++) {
      if (data_.y_train[i] != 0.0 && data_.y_train[i] != 1.0) {
        Log::Fatal("Outcomes must be 0 or 1 for probit link function");
      }
    }
    // Initialize model_preds_ vector for probit latent outcome sampling
    model_preds_.resize(data_.n_train, 0.0);
  } else if (config_.link_function == LinkFunction::Cloglog) {
    Log::Fatal("Cloglog link function is not currently supported in BCF");
  }

  // Validate that both num_trees_mu and num_trees_tau are positive
  if (config_.num_trees_mu <= 0) {
    Log::Fatal("num_trees_mu must be >0");
  }
  if (config_.num_trees_tau <= 0) {
    Log::Fatal("num_trees_tau must be >0");
  }

  // Validate outcome type
  if (config_.outcome_type == OutcomeType::Ordinal) {
    Log::Fatal("Ordinal outcome type is not currently supported in BCF");
  }

  // Switch off treatment forest leaf scale sampling if treatment is multivariate
  if (config_.sample_sigma2_leaf_tau && config_.tau_leaf_model_type != MeanLeafModelType::GaussianUnivariateRegression) {
    Log::Info("sample_sigma2_leaf_tau can only be true when tau_leaf_model_type is GaussianUnivariateRegression, setting sample_sigma2_leaf_tau to false");
    config_.sample_sigma2_leaf_tau = false;
  }

  // Load data from BARTData object into ForestDataset object
  forest_dataset_ = std::make_unique<ForestDataset>();
  forest_dataset_->AddCovariates(data_.X_train, data_.n_train, data_.p, /*row_major=*/false);
  if (data_.treatment_train != nullptr) {
    forest_dataset_->AddBasis(data_.treatment_train, data_.n_train, data_.treatment_dim, /*row_major=*/false);
  }
  if (data_.obs_weights_train != nullptr) {
    forest_dataset_->AddVarianceWeights(data_.obs_weights_train, data_.n_train);
  }
  samples.num_train = data_.n_train;
  samples.num_test = data_.n_test;
  residual_ = std::make_unique<ColumnVector>(data_.y_train, data_.n_train);
  outcome_raw_ = std::make_unique<ColumnVector>(data_.y_train, data_.n_train);
  if (data_.X_test != nullptr) {
    forest_dataset_test_ = std::make_unique<ForestDataset>();
    forest_dataset_test_->AddCovariates(data_.X_test, data_.n_test, data_.p, /*row_major=*/false);
    if (data_.treatment_test != nullptr) {
      forest_dataset_test_->AddBasis(data_.treatment_test, data_.n_test, data_.treatment_dim, /*row_major=*/false);
    }
    if (data_.obs_weights_test != nullptr) {
      forest_dataset_test_->AddVarianceWeights(data_.obs_weights_test, data_.n_test);
    }
    has_test_ = true;
  }

  // Precompute outcome mean and variance for standardization and calibration
  double y_mean = 0.0, M2 = 0.0, y_mean_prev = 0.0;
  for (int i = 0; i < data_.n_train; i++) {
    y_mean_prev = y_mean;
    y_mean = y_mean_prev + (data_.y_train[i] - y_mean_prev) / (i + 1);
    M2 = M2 + (data_.y_train[i] - y_mean_prev) * (data_.y_train[i] - y_mean);
  }
  double y_var = M2 / data_.n_train;

  // Outcome standardization and forest initial value setup
  if (config_.link_function == LinkFunction::Probit) {
    // Initialize forests to 0, no scaling, but offset by the probit transform of the mean outcome to improve mixing
    samples.y_std = 1.0;
    samples.y_bar = norm_inv_cdf(y_mean);
    init_val_mu_ = 0.0;
    init_val_tau_ = 0.0;
    if (config_.tau_leaf_model_type == MeanLeafModelType::GaussianMultivariateRegression) {
      init_val_tau_vec_.assign(config_.leaf_dim_tau, 0.0);
    }
  } else {
    if (config_.standardize_outcome) {
      samples.y_bar = y_mean;
      samples.y_std = std::sqrt(y_var);
      init_val_mu_ = 0.0;
      init_val_tau_ = 0.0;
      if (config_.tau_leaf_model_type == MeanLeafModelType::GaussianMultivariateRegression) {
        init_val_tau_vec_.assign(config_.leaf_dim_tau, 0.0);
      }
    } else {
      samples.y_bar = 0.0;
      samples.y_std = 1.0;
      init_val_mu_ = y_mean;
      init_val_tau_ = 0.0;
      if (config_.tau_leaf_model_type == MeanLeafModelType::GaussianMultivariateRegression) {
        init_val_tau_vec_.assign(config_.leaf_dim_tau, 0.0);
      }
    }
  }

  // Calibration for mu forest
  if (config_.sigma2_mu_init < 0.0) {
    if (config_.link_function == LinkFunction::Probit) {
      config_.sigma2_mu_init = 1.0 / config_.num_trees_mu;
    } else {
      if (config_.standardize_outcome)
        config_.sigma2_mu_init = 1.0 / config_.num_trees_mu;
      else
        config_.sigma2_mu_init = y_var / config_.num_trees_mu;
    }
  }
  if (config_.sample_sigma2_leaf_mu) {
    if (config_.b_sigma2_mu <= 0.0) {
      if (config_.link_function == LinkFunction::Probit) {
        config_.b_sigma2_mu = 1.0 / (2 * config_.num_trees_mu);
      } else {
        if (config_.standardize_outcome)
          config_.sigma2_mu_init = 1.0 / (2 * config_.num_trees_mu);
        else
          config_.sigma2_mu_init = y_var / (2 * config_.num_trees_mu);
      }
    }
  }

  // Calibration for tau forest
  if (config_.sigma2_tau_init < 0.0) {
    if (config_.link_function == LinkFunction::Probit) {
      config_.sigma2_tau_init = 1.0 / config_.num_trees_tau;
    } else {
      if (config_.standardize_outcome)
        config_.sigma2_tau_init = 1.0 / config_.num_trees_tau;
      else
        config_.sigma2_tau_init = y_var / config_.num_trees_tau;
    }
  }
  if (config_.sample_sigma2_leaf_tau) {
    if (config_.b_sigma2_tau <= 0.0) {
      if (config_.link_function == LinkFunction::Probit) {
        config_.b_sigma2_tau = 1.0 / (2 * config_.num_trees_tau);
      } else {
        if (config_.standardize_outcome)
          config_.sigma2_tau_init = 1.0 / (2 * config_.num_trees_tau);
        else
          config_.sigma2_tau_init = y_var / (2 * config_.num_trees_tau);
      }
    }
  }

  // Initialize mu leaf model
  mu_leaf_model_ = GaussianConstantLeafModel(config_.sigma2_mu_init);

  // Initialize tau leaf model
  if (config_.tau_leaf_model_type == MeanLeafModelType::GaussianUnivariateRegression) {
    tau_leaf_model_ = GaussianUnivariateRegressionLeafModel(config_.sigma2_tau_init);
  } else if (config_.tau_leaf_model_type == MeanLeafModelType::GaussianConstant) {
    Eigen::MatrixXd Sigma_0;
    if (!config_.sigma2_leaf_tau_matrix.empty()) {
      if ((int)config_.sigma2_leaf_tau_matrix.size() != config_.leaf_dim_tau * config_.leaf_dim_tau) {
        Log::Fatal("sigma2_leaf_tau_matrix must have leaf_dim_tau * leaf_dim_tau = %d elements, but has %zu",
                   config_.leaf_dim_tau * config_.leaf_dim_tau, config_.sigma2_leaf_tau_matrix.size());
      }
      // Column-major interpretation matches both R and Eigen (python must be reordered before passing to C++)
      Sigma_0 = Eigen::Map<const Eigen::MatrixXd>(config_.sigma2_leaf_tau_matrix.data(), config_.leaf_dim_tau, config_.leaf_dim_tau);
    } else {
      Sigma_0 = config_.sigma2_tau_init * Eigen::MatrixXd::Identity(config_.leaf_dim_tau, config_.leaf_dim_tau);
    }
    tau_leaf_model_ = GaussianMultivariateRegressionLeafModel(Sigma_0);
  } else {
    Log::Fatal("Unsupported leaf model type for treatment forest");
  }

  // Calibration for variance forests
  if (config_.num_trees_variance > 0) {
    // NOTE: calibration only works for standardized outcomes
    if (config_.shape_variance_forest <= 0.0 || config_.scale_variance_forest <= 0.0) {
      if (config_.leaf_prior_calibration_param <= 0.0) {
        config_.leaf_prior_calibration_param = 1.5;
      }
      if (config_.shape_variance_forest <= 0.0) {
        config_.shape_variance_forest = config_.num_trees_variance / (config_.leaf_prior_calibration_param * config_.leaf_prior_calibration_param) + 0.5;
      }
      if (config_.scale_variance_forest <= 0.0) {
        config_.scale_variance_forest = config_.num_trees_variance / (config_.leaf_prior_calibration_param * config_.leaf_prior_calibration_param);
      }
    }
    if (config_.standardize_outcome) {
      init_val_variance_ = 1.0;
    } else {
      init_val_variance_ = y_var;
    }
  }

  // Standardize partial residuals in place; these are updated in each iteration but initialized to standardized outcomes
  // Works for:
  //  1. Standardized outcomes (since y_bar = mean(y) and y_std = sd(y))
  //  2. Non-standardized outcomes (since y_bar = 0 and y_std = 1, so this just transfers y_train as-is)
  //  3. Probit link (since y_bar = norm_inv_cdf(mean(y)) and y_std = 1)
  for (int i = 0; i < data_.n_train; i++) residual_->GetData()[i] = (data_.y_train[i] - samples.y_bar) / samples.y_std;

  // Initialize mean forest state
  mu_forest_ = std::make_unique<TreeEnsemble>(config_.num_trees_mu, config_.leaf_dim_mu, config_.leaf_constant_mu, config_.exponentiated_leaf_mu);
  samples.mu_forests = std::make_unique<ForestContainer>(config_.num_trees_mu, config_.leaf_dim_mu, config_.leaf_constant_mu, config_.exponentiated_leaf_mu);
  mu_forest_tracker_ = std::make_unique<ForestTracker>(forest_dataset_->GetCovariates(), config_.feature_types, config_.num_trees_mu, data_.n_train);
  tree_prior_mu_ = std::make_unique<TreePrior>(config_.alpha_mu, config_.beta_mu, config_.min_samples_leaf_mu, config_.max_depth_mu);
  mu_forest_->SetLeafValue(init_val_mu_ / config_.num_trees_mu);
  UpdateResidualEntireForest(*mu_forest_tracker_, *forest_dataset_, *residual_, mu_forest_.get(), !config_.leaf_constant_mu, std::minus<double>());
  mu_forest_tracker_->UpdatePredictions(mu_forest_.get(), *forest_dataset_.get());

  // Initialize treatment effect forest state
  if (config_.tau_leaf_model_type == MeanLeafModelType::GaussianUnivariateRegression) {
    tau_forest_ = std::make_unique<TreeEnsemble>(config_.num_trees_tau, config_.leaf_dim_tau, config_.leaf_constant_tau, config_.exponentiated_leaf_tau);
    samples.tau_forests = std::make_unique<ForestContainer>(config_.num_trees_tau, config_.leaf_dim_tau, config_.leaf_constant_tau, config_.exponentiated_leaf_tau);
    tau_forest_tracker_ = std::make_unique<ForestTracker>(forest_dataset_->GetCovariates(), config_.feature_types, config_.num_trees_tau, data_.n_train);
    tree_prior_tau_ = std::make_unique<TreePrior>(config_.alpha_tau, config_.beta_tau, config_.min_samples_leaf_tau, config_.max_depth_tau);
    tau_forest_->SetLeafValue(init_val_tau_ / config_.num_trees_tau);
    UpdateResidualEntireForest(*tau_forest_tracker_, *forest_dataset_, *residual_, tau_forest_.get(), !config_.leaf_constant_tau, std::minus<double>());
    tau_forest_tracker_->UpdatePredictions(tau_forest_.get(), *forest_dataset_.get());
  } else if (config_.tau_leaf_model_type == MeanLeafModelType::GaussianMultivariateRegression) {
    tau_forest_ = std::make_unique<TreeEnsemble>(config_.num_trees_tau, config_.leaf_dim_tau, config_.leaf_constant_tau, config_.exponentiated_leaf_tau);
    samples.tau_forests = std::make_unique<ForestContainer>(config_.num_trees_tau, config_.leaf_dim_tau, config_.leaf_constant_tau, config_.exponentiated_leaf_tau);
    tau_forest_tracker_ = std::make_unique<ForestTracker>(forest_dataset_->GetCovariates(), config_.feature_types, config_.num_trees_tau, data_.n_train);
    tree_prior_tau_ = std::make_unique<TreePrior>(config_.alpha_tau, config_.beta_tau, config_.min_samples_leaf_tau, config_.max_depth_tau);
    tau_forest_->SetLeafVector(init_val_tau_vec_);
    UpdateResidualEntireForest(*tau_forest_tracker_, *forest_dataset_, *residual_, tau_forest_.get(), true, std::minus<double>());
    tau_forest_tracker_->UpdatePredictions(tau_forest_.get(), *forest_dataset_.get());
  } else {
    Log::Fatal("Unsupported leaf model type for treatment forest");
  }

  // Initialize variance forest state (if present)
  if (config_.num_trees_variance > 0) {
    variance_leaf_model_ = LogLinearVarianceLeafModel(config_.shape_variance_forest, config_.scale_variance_forest);
    variance_forest_ = std::make_unique<TreeEnsemble>(config_.num_trees_variance, config_.leaf_dim_variance, config_.leaf_constant_variance, config_.exponentiated_leaf_variance);
    samples.variance_forests = std::make_unique<ForestContainer>(config_.num_trees_variance, config_.leaf_dim_variance, config_.leaf_constant_variance, config_.exponentiated_leaf_variance);
    variance_forest_tracker_ = std::make_unique<ForestTracker>(forest_dataset_->GetCovariates(), config_.feature_types, config_.num_trees_variance, data_.n_train);
    tree_prior_variance_ = std::make_unique<TreePrior>(config_.alpha_variance, config_.beta_variance, config_.min_samples_leaf_variance, config_.max_depth_variance);
    // Leaf values for the log-linear variance model are on the log scale; the ensemble sums
    // log(sigma^2_i) contributions, so each tree starts at log(init_val) / num_trees.
    variance_forest_->SetLeafValue(std::log(init_val_variance_) / config_.num_trees_variance);
    variance_forest_tracker_->UpdatePredictions(variance_forest_.get(), *forest_dataset_.get());
    // UpdateVarModelTree (called inside GFRSampleOneIter / MCMCSampleOneIter) unconditionally
    // reads and writes the dataset variance weight slot via VarWeightValue / SetVarWeightValue.
    // This slot tracks the cumulative per-observation variance prediction
    // (sigma^2_i = exp(sum of tree leaf values)) and is incompatible with case weights, which
    // would need to be reapplied after every per-tree update. The R/Python APIs enforce this
    // as a hard error; guard here for callers that use BARTSampler directly.
    if (forest_dataset_->HasVarWeights()) {
      Log::Fatal("observation_weights and a variance forest cannot be used together.");
    }
    std::vector<double> initial_variance_preds(data_.n_train, init_val_variance_);
    forest_dataset_->AddVarianceWeights(initial_variance_preds.data(), data_.n_train);
    has_variance_forest_ = true;
  }

  // Global error variance model
  if (config_.sample_sigma2_global) {
    var_model_ = std::make_unique<GlobalHomoskedasticVarianceModel>();
    sample_sigma2_global_ = true;
  }

  // Leaf scale models
  if (config_.sample_sigma2_leaf_mu) {
    leaf_scale_model_mu_ = std::make_unique<LeafNodeHomoskedasticVarianceModel>();
    sample_sigma2_leaf_mu_ = true;
  }
  if (config_.sample_sigma2_leaf_tau) {
    leaf_scale_model_tau_ = std::make_unique<LeafNodeHomoskedasticVarianceModel>();
    sample_sigma2_leaf_tau_ = true;
  }

  // Random effects model
  if (config_.has_random_effects) {
    random_effects_dataset_ = std::make_unique<RandomEffectsDataset>();
    random_effects_dataset_->AddGroupLabels(data_.rfx_group_ids_train, data_.n_train);
    if (data_.rfx_basis_train != nullptr) {
      random_effects_dataset_->AddBasis(data_.rfx_basis_train, data_.n_train, data_.rfx_basis_dim, /*row_major=*/false);
    } else {
      if (config_.rfx_model_spec == BCFRFXModelSpec::InterceptOnly) {
        // If no basis is provided, add an intercept basis (column of 1s)
        // TODO: do we need to do this before we determine rfx_basis_dim and initialize the RFX data structures?
        std::vector<double> intercept_basis(data_.n_train, 1.0);
        random_effects_dataset_->AddBasis(intercept_basis.data(), data_.n_train, 1, /*row_major=*/false);
        // Override rfx_basis_dim to 1 for intercept-only model the basis is a 1-dimensional vector of ones
        data_.rfx_basis_dim = 1;
      } else if (config_.rfx_model_spec == BCFRFXModelSpec::InterceptPlusTreatment) {
        // If no basis is provided, add an intercept basis (column of 1s) and the treatment variable(s) as the basis
        // TODO: do we need to do this before we determine rfx_basis_dim and initialize the RFX data structures?
        std::vector<double> intercept_basis(data_.n_train * (1 + data_.treatment_dim), 1.0);
        for (int i = 0; i < data_.n_train; i++) {
          for (int j = 0; j < data_.treatment_dim; j++) {
            intercept_basis[(j + 1) * data_.n_train + i] = data_.treatment_train[j * data_.n_train + i];
          }
        }
        random_effects_dataset_->AddBasis(intercept_basis.data(), data_.n_train, 1 + data_.treatment_dim, /*row_major=*/false);
        // Override rfx_basis_dim to 1 for intercept-only model the basis is a 1-dimensional vector of ones
        data_.rfx_basis_dim = 1 + data_.treatment_dim;
      } else {
        Log::Fatal("Random effects basis data must be provided for non-intercept-only random effects model");
      }
    }
    // Tracking data structure for random effects groups
    random_effects_tracker_ = std::make_unique<RandomEffectsTracker>(data_.rfx_group_ids_train, data_.n_train);
    // Container of random effects samples
    samples.rfx_container = std::make_unique<RandomEffectsContainer>(data_.rfx_basis_dim, data_.rfx_num_groups);
    // Mapping from RFX labels to 0-indexed group IDs for efficient lookup in the sampler; populated from the RFX dataset group labels
    samples.rfx_label_mapper = std::make_unique<LabelMapper>(random_effects_tracker_->GetLabelMap());

    // Initialize random effects model object
    random_effects_model_ = std::make_unique<MultivariateRegressionRandomEffectsModel>(data_.rfx_basis_dim, data_.rfx_num_groups);

    // Handle "working" parameter prior mean
    Eigen::VectorXd working_parameter_prior_mean;
    if (!config_.rfx_working_parameter_mean_prior.empty()) {
      if ((int)config_.rfx_working_parameter_mean_prior.size() != data_.rfx_basis_dim) {
        Log::Fatal("rfx_working_parameter_mean_prior must have rfx_basis_dim = %d elements, but has %zu",
                   data_.rfx_basis_dim, config_.rfx_working_parameter_mean_prior.size());
      }
      // Column-major interpretation matches both R and Eigen (python must be reordered before passing to C++)
      working_parameter_prior_mean = Eigen::Map<const Eigen::VectorXd>(config_.rfx_working_parameter_mean_prior.data(), data_.rfx_basis_dim);
    } else {
      working_parameter_prior_mean = Eigen::VectorXd::Zero(data_.rfx_basis_dim);
    }
    random_effects_model_->SetWorkingParameter(working_parameter_prior_mean);

    // Handle "group" parameter prior mean
    Eigen::MatrixXd group_parameter_prior_mean;
    if (!config_.rfx_group_parameter_mean_prior.empty()) {
      if ((int)config_.rfx_group_parameter_mean_prior.size() != data_.rfx_basis_dim * data_.rfx_num_groups) {
        Log::Fatal("rfx_group_parameter_mean_prior must have rfx_basis_dim * rfx_num_groups = %d elements, but has %zu",
                   data_.rfx_basis_dim * data_.rfx_num_groups, config_.rfx_group_parameter_mean_prior.size());
      }
      // Column-major interpretation matches both R and Eigen (python must be reordered before passing to C++)
      group_parameter_prior_mean = Eigen::Map<const Eigen::MatrixXd>(config_.rfx_group_parameter_mean_prior.data(), data_.rfx_basis_dim, data_.rfx_num_groups);
    } else {
      group_parameter_prior_mean = Eigen::MatrixXd::Zero(data_.rfx_basis_dim, data_.rfx_num_groups);
    }
    random_effects_model_->SetGroupParameters(group_parameter_prior_mean);

    // Handle "working" parameter prior covariance
    Eigen::MatrixXd working_parameter_prior_cov;
    if (!config_.rfx_working_parameter_cov_prior.empty()) {
      if ((int)config_.rfx_working_parameter_cov_prior.size() != data_.rfx_basis_dim * data_.rfx_basis_dim) {
        Log::Fatal("rfx_working_parameter_cov_prior must have rfx_basis_dim * rfx_basis_dim = %d elements, but has %zu",
                   data_.rfx_basis_dim * data_.rfx_basis_dim, config_.rfx_working_parameter_cov_prior.size());
      }
      // Column-major interpretation matches both R and Eigen (python must be reordered before passing to C++)
      working_parameter_prior_cov = Eigen::Map<const Eigen::MatrixXd>(config_.rfx_working_parameter_cov_prior.data(), data_.rfx_basis_dim, data_.rfx_basis_dim);
    } else {
      working_parameter_prior_cov = Eigen::MatrixXd::Identity(data_.rfx_basis_dim, data_.rfx_basis_dim);
    }
    random_effects_model_->SetWorkingParameterCovariance(working_parameter_prior_cov);

    // Handle "group" parameter prior covariance
    Eigen::MatrixXd group_parameter_prior_cov;
    if (!config_.rfx_group_parameter_cov_prior.empty()) {
      if ((int)config_.rfx_group_parameter_cov_prior.size() != data_.rfx_basis_dim * data_.rfx_basis_dim) {
        Log::Fatal("rfx_group_parameter_cov_prior must have rfx_basis_dim * rfx_basis_dim = %d elements, but has %zu",
                   data_.rfx_basis_dim * data_.rfx_basis_dim, config_.rfx_group_parameter_cov_prior.size());
      }
      // Column-major interpretation matches both R and Eigen (python must be reordered before passing to C++)
      group_parameter_prior_cov = Eigen::Map<const Eigen::MatrixXd>(config_.rfx_group_parameter_cov_prior.data(), data_.rfx_basis_dim, data_.rfx_basis_dim);
    } else {
      group_parameter_prior_cov = Eigen::MatrixXd::Identity(data_.rfx_basis_dim, data_.rfx_basis_dim);
    }
    random_effects_model_->SetGroupParameterCovariance(group_parameter_prior_cov);

    // Handle variance model priors
    if (config_.rfx_variance_prior_shape <= 0.0) {
      config_.rfx_variance_prior_shape = 1.0;
    }
    if (config_.rfx_variance_prior_scale <= 0.0) {
      config_.rfx_variance_prior_scale = 1.0;
    }
    random_effects_model_->SetVariancePriorShape(config_.rfx_variance_prior_shape);
    random_effects_model_->SetVariancePriorScale(config_.rfx_variance_prior_scale);

    // Set has_random_effects_ flag to true so that the sampler will perform random effects updates at each iteration
    has_random_effects_ = true;
  }

  // RNG
  rng_ = std::mt19937(config_.random_seed >= 0 ? config_.random_seed : std::random_device{}());

  // Other internal model state
  global_variance_ = config_.sigma2_global_init;
  leaf_scale_mu_ = config_.sigma2_mu_init;
  leaf_scale_tau_ = config_.sigma2_tau_init;
  leaf_scale_tau_multivariate_ = config_.sigma2_leaf_tau_matrix;

  tau_raw_sum_preds_.assign(data_.n_train * data_.treatment_dim, 0.0);

  initialized_ = true;
}

void BCFSampler::run_gfr(BCFSamples& samples, int num_gfr, bool keep_gfr, int num_chains) {
  // Reserve space for GFR predictions if they are to be retained
  if (keep_gfr) {
    samples.mu_forest_predictions_train.reserve(data_.n_train * num_gfr);
    samples.tau_forest_predictions_train.reserve(data_.n_train * num_gfr);
    if (has_variance_forest_) {
      samples.variance_forest_predictions_train.reserve(data_.n_train * num_gfr);
    }
  }

  // NOTE: for serial sampling (which is all we currently support), chain 1 uses the live sampler state after
  // the GFR loop (i.e. the state after GFR iteration num_gfr-1). Chains 2..N each need their own earlier
  // GFR starting point so that the chains are initialized from distinct states.
  // We save exactly num_chains-1 snapshots, one per "extra" chain:
  //   gfr_snapshots_[k] = state after GFR iteration (num_gfr - num_chains + k), for k = 0..num_chains-2.
  // The last GFR iteration (i = num_gfr-1) is NOT snapshotted because chain 1 uses the live state.
  // Chain j (1-indexed, j >= 2) uses gfr_snapshots_[num_chains-j] = state after GFR[num_gfr-j].
  // If num_chains is 1, we keep no snapshots.
  int snapshot_start = (num_chains > 1) ? std::max(0, num_gfr - num_chains) : num_gfr;
  if (num_chains > 1) {
    gfr_snapshots_.clear();
    gfr_snapshots_.reserve(num_chains - 1);
  }

  bool write_snapshot = false;
  for (int i = 0; i < num_gfr; i++) {
    // Do not snapshot the final GFR iteration: chain 1 uses the live sampler state directly.
    write_snapshot = (i >= snapshot_start) && (i < num_gfr - 1);
    RunOneIteration(samples, /*gfr=*/true, /*keep_sample=*/keep_gfr, /*write_snapshot=*/write_snapshot);
  }
}

void BCFSampler::run_mcmc(BCFSamples& samples, int num_burnin, int keep_every, int num_mcmc) {
  // Reserve space for MCMC predictions if they are to be retained
  samples.mu_forest_predictions_train.reserve(data_.n_train * num_mcmc);
  samples.tau_forest_predictions_train.reserve(data_.n_train * num_mcmc);
  if (has_test_) {
    samples.mu_forest_predictions_test.reserve(data_.n_test * num_mcmc);
    samples.tau_forest_predictions_test.reserve(data_.n_test * num_mcmc);
  }
  if (has_variance_forest_) {
    samples.variance_forest_predictions_train.reserve(data_.n_train * num_mcmc);
    if (has_test_) {
      samples.variance_forest_predictions_test.reserve(data_.n_test * num_mcmc);
    }
  }

  // Create leaf models and pass them to the RunOneIteration function; these are updated in place and will reflect the current state of the leaf scale parameters (if they are being sampled)
  bool keep_forest = false;
  for (int i = 0; i < num_burnin + keep_every * num_mcmc; i++) {
    if (i >= num_burnin && (i - num_burnin) % keep_every == 0)
      keep_forest = true;
    else
      keep_forest = false;
    RunOneIteration(samples, /*gfr=*/false, /*keep_sample=*/keep_forest, /*write_snapshot=*/false);
  }
}

void BCFSampler::run_mcmc_chains(BCFSamples& samples, int num_chains, int num_burnin, int keep_every, int num_mcmc) {
  for (int chain_idx = 0; chain_idx < num_chains; chain_idx++) {
    if (chain_idx > 0 && !gfr_snapshots_.empty()) {
      // Re-initialize the sampler state for each new chain.
      // gfr_snapshots_ holds num_chains-1 states (oldest-first): index 0 = GFR[num_gfr-num_chains],
      // index num_chains-2 = GFR[num_gfr-2].  Chain j (1-indexed, j>=2) uses index num_chains-j.
      // When num_gfr < num_chains we may not have enough distinct snapshots; in that case
      // fall back to running chains from whatever state is available (same behavior as R).
      int snapshot_idx = num_chains - 1 - chain_idx;
      if (snapshot_idx >= 0 && snapshot_idx < static_cast<int>(gfr_snapshots_.size())) {
        RestoreStateFromGFRSnapshot(samples, snapshot_idx);
      }
    }
    run_mcmc(samples, num_burnin, keep_every, num_mcmc);
  }
}

void BCFSampler::postprocess_samples(BCFSamples& samples) {
  // Unpack test set predictions for mean and variance forest
  if (has_test_) {
    std::vector<double> predictions = samples.mu_forests->Predict(*forest_dataset_test_);
    samples.mu_forest_predictions_test.insert(samples.mu_forest_predictions_test.end(),
                                              predictions.data(), predictions.data() + predictions.size());
    predictions = samples.tau_forests->PredictRaw(*forest_dataset_test_);
    samples.tau_forest_predictions_test.insert(samples.tau_forest_predictions_test.end(),
                                               predictions.data(), predictions.data() + predictions.size());
    if (has_variance_forest_) {
      std::vector<double> predictions = samples.variance_forests->Predict(*forest_dataset_test_);
      samples.variance_forest_predictions_test.insert(samples.variance_forest_predictions_test.end(),
                                                      predictions.data(), predictions.data() + predictions.size());
    }
    if (has_random_effects_) {
      RandomEffectsDataset rfx_dataset_test;
      rfx_dataset_test.AddGroupLabels(data_.rfx_group_ids_test, data_.n_test);
      if (data_.rfx_basis_test != nullptr) {
        rfx_dataset_test.AddBasis(data_.rfx_basis_test, data_.n_test, data_.rfx_basis_dim, /*row_major=*/false);
      } else {
        std::vector<double> ones(data_.n_test, 1.0);
        rfx_dataset_test.AddBasis(ones.data(), data_.n_test, 1, /*row_major=*/false);
      }
      samples.rfx_predictions_test.resize(data_.n_test * samples.num_samples);
      samples.rfx_container->Predict(rfx_dataset_test, *samples.rfx_label_mapper, samples.rfx_predictions_test);
    }

    // Compute outcome predictions on the linear (link) scale: E[eta|X,Z] = mu(X) + Z*tau(X) + rfx
    // tau_forest_predictions stores raw tau(x) (no z multiplication), so we multiply by z here.
    // Callers that need probability-scale predictions (probit, cloglog) apply the inverse link themselves.
    samples.y_hat_train.resize(data_.n_train * samples.num_samples);
    double mu_term, tau_term, y_term;
    const int treatment_dim = data_.treatment_dim;
    for (int j = 0; j < samples.num_samples; j++) {
      for (int i = 0; i < data_.n_train; i++) {
        // Data index for the two terms that are guaranteed to be univariate - mu(x) and y_hat
        const int k = j * data_.n_train + i;
        mu_term = samples.mu_forest_predictions_train[k];
        if (treatment_dim > 1) {
          tau_term = 0;
          for (int treatment_idx = 0; treatment_idx < treatment_dim; treatment_idx++) {
            // Starting data index for multivariate treatment case, where tau(x) is col-major with dimensions (n_train, treatment_dim, num_samples)
            const int k_tau = j * data_.n_train * treatment_dim + data_.n_train * treatment_idx + i;
            tau_term += samples.tau_forest_predictions_train[k_tau] * data_.treatment_train[data_.n_train * treatment_idx + i];
          }
        } else {
          tau_term = samples.tau_forest_predictions_train[k] * data_.treatment_test[i];
        }
        y_term = mu_term + tau_term;
        if (has_random_effects_) y_term += samples.rfx_predictions_train[k];
        samples.y_hat_train[k] = y_term * samples.y_std + samples.y_bar;
      }
    }

    samples.y_hat_test.resize(data_.n_test * samples.num_samples);
    for (int j = 0; j < samples.num_samples; j++) {
      for (int i = 0; i < data_.n_test; i++) {
        // Data index for the two terms that are guaranteed to be univariate - mu(x) and y_hat
        const int k = j * data_.n_test + i;
        mu_term = samples.mu_forest_predictions_test[k];
        if (treatment_dim > 1) {
          tau_term = 0;
          for (int treatment_idx = 0; treatment_idx < treatment_dim; treatment_idx++) {
            // Starting data index for multivariate treatment case, where tau(x) is col-major with dimensions (n_test, treatment_dim, num_samples)
            const int k_tau = j * data_.n_test * treatment_dim + data_.n_test * treatment_idx + i;
            tau_term += samples.tau_forest_predictions_test[k_tau] * data_.treatment_test[data_.n_test * treatment_idx + i];
          }
        } else {
          tau_term = samples.tau_forest_predictions_test[k] * data_.treatment_test[i];
        }
        y_term = mu_term + tau_term;
        if (has_random_effects_) y_term += samples.rfx_predictions_test[k];
        samples.y_hat_test[k] = y_term * samples.y_std + samples.y_bar;
      }
    }
  }
}

void BCFSampler::RunOneIteration(BCFSamples& samples, bool gfr, bool keep_sample, bool write_snapshot) {
  // mu forest
  if (gfr) {
    GFRSampleOneIter<GaussianConstantLeafModel, GaussianConstantSuffStat>(
        *mu_forest_, *mu_forest_tracker_, *samples.mu_forests, mu_leaf_model_,
        *forest_dataset_, *residual_, *tree_prior_mu_, rng_,
        config_.var_weights_mu, config_.sweep_update_indices_mu, global_variance_, config_.feature_types,
        config_.cutpoint_grid_size, /*keep_forest=*/keep_sample,
        /*pre_initialized=*/true, /*backfitting=*/true,
        /*num_features_subsample=*/config_.num_features_subsample_mu, config_.num_threads);
  } else {
    MCMCSampleOneIter<GaussianConstantLeafModel, GaussianConstantSuffStat>(
        *mu_forest_, *mu_forest_tracker_, *samples.mu_forests, mu_leaf_model_,
        *forest_dataset_, *residual_, *tree_prior_mu_, rng_,
        config_.var_weights_mu, config_.sweep_update_indices_mu, global_variance_, /*keep_forest=*/keep_sample,
        /*pre_initialized=*/true, /*backfitting=*/true,
        /*num_threads=*/config_.num_threads);
  }

  // tau forest
  if (gfr) {
    std::visit(GFROneIterationVisitorTau{*this, samples, keep_sample}, tau_leaf_model_);
  } else {
    std::visit(MCMCOneIterationVisitorTau{*this, samples, keep_sample}, tau_leaf_model_);
  }
  // Update raw tau(x): sum leaf values across trees for each dimension of the tau leaf.
  // Uses node IDs already cached in the tracker — no tree traversal needed.
  const int tau_dim = data_.treatment_dim;
  const int data_dim = data_.n_train;
  for (int i = 0; i < data_dim; i++) {
    for (int k = 0; k < tau_dim; k++) tau_raw_sum_preds_[i * tau_dim + k] = 0.0;
    for (int j = 0; j < config_.num_trees_tau; j++) {
      data_size_t leaf = tau_forest_tracker_->GetNodeId(i, j);
      for (int k = 0; k < tau_dim; k++)
        tau_raw_sum_preds_[k * data_dim + i] += tau_forest_->GetTree(j)->LeafValue(leaf, k);
    }
  }

  if (has_variance_forest_) {
    if (gfr) {
      GFRSampleOneIter<LogLinearVarianceLeafModel, LogLinearVarianceSuffStat>(
          *variance_forest_, *variance_forest_tracker_, *samples.variance_forests, variance_leaf_model_,
          *forest_dataset_, *residual_, *tree_prior_variance_, rng_,
          config_.var_weights_variance, config_.sweep_update_indices_variance, global_variance_, config_.feature_types,
          config_.cutpoint_grid_size, /*keep_forest=*/keep_sample,
          /*pre_initialized=*/true, /*backfitting=*/false,
          /*num_features_subsample=*/config_.num_features_subsample_variance, config_.num_threads);
    } else {
      MCMCSampleOneIter<LogLinearVarianceLeafModel, LogLinearVarianceSuffStat>(
          *variance_forest_, *variance_forest_tracker_, *samples.variance_forests, variance_leaf_model_,
          *forest_dataset_, *residual_, *tree_prior_variance_, rng_,
          config_.var_weights_variance, config_.sweep_update_indices_variance, global_variance_, /*keep_forest=*/keep_sample,
          /*pre_initialized=*/true, /*backfitting=*/false,
          /*num_threads=*/config_.num_threads);
    }
  }

  if (config_.link_function == LinkFunction::Probit) {
    AddModelTermsForProbit(model_preds_.data(), mu_forest_tracker_.get(), tau_forest_tracker_.get(), random_effects_tracker_.get(), data_.n_train);
    sample_probit_latent_outcome(rng_, outcome_raw_->GetData().data(), model_preds_.data(),
                                 residual_->GetData().data(), samples.y_bar, data_.n_train);
  }

  if (sample_sigma2_global_) {
    global_variance_ = var_model_->SampleVarianceParameter(
        residual_->GetData(), config_.a_sigma2_global, config_.b_sigma2_global, rng_);
  }

  if (sample_sigma2_leaf_mu_) {
    leaf_scale_mu_ = leaf_scale_model_mu_->SampleVarianceParameter(
        mu_forest_.get(), config_.a_sigma2_mu, config_.b_sigma2_mu, rng_);
    mu_leaf_model_.SetScale(leaf_scale_mu_);
  }

  // Gibbs updates for random effects model
  if (has_random_effects_) {
    random_effects_model_->SampleRandomEffects(*random_effects_dataset_, *residual_, *random_effects_tracker_, global_variance_, rng_);
    if (keep_sample) {
      samples.rfx_container->AddSample(*random_effects_model_);
      for (int i = 0; i < data_.n_train; i++) {
        samples.rfx_predictions_train.push_back(random_effects_tracker_->GetPrediction(i));
      }
    }
  }

  if (keep_sample) {
    // Add parameter and prediction samples
    samples.num_samples++;
    if (sample_sigma2_global_) samples.global_error_variance_samples.push_back(global_variance_);
    if (sample_sigma2_leaf_mu_) samples.leaf_scale_mu_samples.push_back(leaf_scale_mu_);
    if (sample_sigma2_leaf_tau_) samples.leaf_scale_tau_samples.push_back(leaf_scale_tau_);
    double* mu_forest_preds_train = mu_forest_tracker_->GetSumPredictions();
    samples.mu_forest_predictions_train.insert(samples.mu_forest_predictions_train.end(),
                                               mu_forest_preds_train,
                                               mu_forest_preds_train + samples.num_train);
    samples.tau_forest_predictions_train.insert(samples.tau_forest_predictions_train.end(),
                                                tau_raw_sum_preds_.begin(), tau_raw_sum_preds_.end());
    if (has_variance_forest_) {
      double* variance_forest_preds_train = variance_forest_tracker_->GetSumPredictions();
      samples.variance_forest_predictions_train.insert(samples.variance_forest_predictions_train.end(),
                                                       variance_forest_preds_train,
                                                       variance_forest_preds_train + samples.num_train);
    }
  }

  if (write_snapshot) {
    GFRSnapshot snap;
    snap.mu_forest = std::make_unique<TreeEnsemble>(*mu_forest_);
    snap.tau_forest = std::make_unique<TreeEnsemble>(*tau_forest_);
    if (has_variance_forest_) snap.variance_forest = std::make_unique<TreeEnsemble>(*variance_forest_);
    snap.sigma2 = global_variance_;
    snap.leaf_scale_mu = leaf_scale_mu_;
    if (config_.tau_leaf_model_type == MeanLeafModelType::GaussianMultivariateRegression) {
      snap.leaf_scale_tau_multivariate = leaf_scale_tau_multivariate_;
    } else if (config_.tau_leaf_model_type == MeanLeafModelType::GaussianUnivariateRegression) {
      snap.leaf_scale_tau = leaf_scale_tau_;
    }
    snap.residual.clear();
    snap.residual.resize(data_.n_train);
    snap.residual.assign(residual_->GetData().data(), residual_->GetData().data() + data_.n_train);
    if (has_variance_forest_) {
      snap.variance_weights.clear();
      snap.variance_weights.resize(data_.n_train);
      snap.variance_weights.assign(forest_dataset_->GetVarWeights().data(), forest_dataset_->GetVarWeights().data() + data_.n_train);
    }
    if (config_.has_random_effects) {
      snap.rfx_working_parameter = random_effects_model_->GetWorkingParameter();
      snap.rfx_group_parameters = random_effects_model_->GetGroupParameters();
      snap.rfx_group_parameter_covariance = random_effects_model_->GetGroupParameterCovariance();
      snap.rfx_working_parameter_covariance = random_effects_model_->GetWorkingParameterCovariance();
      snap.rfx_variance_prior_shape = random_effects_model_->GetVariancePriorShape();
      snap.rfx_variance_prior_scale = random_effects_model_->GetVariancePriorScale();
    }
    gfr_snapshots_.push_back(std::move(snap));
  }
}

void BCFSampler::RestoreStateFromGFRSnapshot(BCFSamples& samples, int snapshot_index) {
  GFRSnapshot& snap = gfr_snapshots_[snapshot_index];

  // Restore mean forest state (if present).
  // ReconstituteFromForest increments the residual by (prev_tree_pred - new_tree_pred) for
  // every tree, swapping the chain-N forest contribution out and the GFR-snapshot contribution
  // in.  The residual must still hold the chain-N state here so that this swap is correct.
  mu_forest_->ReconstituteFromForest(*snap.mu_forest);
  mu_forest_tracker_->ReconstituteFromForest(*snap.mu_forest, *forest_dataset_, *residual_, true);
  mu_forest_tracker_->UpdatePredictions(mu_forest_.get(), *forest_dataset_.get());
  std::visit(TauForestResetVisitor{*this, samples, *snap.tau_forest}, tau_leaf_model_);

  // Initialize variance forest state (if present)
  if (config_.num_trees_variance > 0) {
    variance_leaf_model_ = LogLinearVarianceLeafModel(config_.shape_variance_forest, config_.scale_variance_forest);
    variance_forest_ = std::make_unique<TreeEnsemble>(config_.num_trees_variance, config_.leaf_dim_variance, config_.leaf_constant_variance, config_.exponentiated_leaf_variance);
    samples.variance_forests = std::make_unique<ForestContainer>(config_.num_trees_variance, config_.leaf_dim_variance, config_.leaf_constant_variance, config_.exponentiated_leaf_variance);
    variance_forest_tracker_ = std::make_unique<ForestTracker>(forest_dataset_->GetCovariates(), config_.feature_types, config_.num_trees_variance, data_.n_train);
    tree_prior_variance_ = std::make_unique<TreePrior>(config_.alpha_variance, config_.beta_variance, config_.min_samples_leaf_variance, config_.max_depth_variance);
    // Leaf values for the log-linear variance model are on the log scale; the ensemble sums
    // log(sigma^2_i) contributions, so each tree starts at log(init_val) / num_trees.
    variance_forest_->SetLeafValue(std::log(init_val_variance_) / config_.num_trees_variance);
    variance_forest_tracker_->UpdatePredictions(variance_forest_.get(), *forest_dataset_.get());
    // UpdateVarModelTree (called inside GFRSampleOneIter / MCMCSampleOneIter) unconditionally
    // reads and writes the dataset variance weight slot via VarWeightValue / SetVarWeightValue.
    // This slot tracks the cumulative per-observation variance prediction
    // (sigma^2_i = exp(sum of tree leaf values)) and is incompatible with case weights, which
    // would need to be reapplied after every per-tree update. The R/Python APIs enforce this
    // as a hard error; guard here for callers that use BARTSampler directly.
    if (forest_dataset_->HasVarWeights()) {
      Log::Fatal("observation_weights and a variance forest cannot be used together.");
    }
    std::vector<double> initial_variance_preds(data_.n_train, init_val_variance_);
    forest_dataset_->AddVarianceWeights(initial_variance_preds.data(), data_.n_train);
    has_variance_forest_ = true;
  }

  // Random effects model
  if (config_.has_random_effects) {
    // Restore "working" parameter prior mean
    random_effects_model_->SetWorkingParameter(snap.rfx_working_parameter);

    // Restore "group" parameter prior mean
    random_effects_model_->SetGroupParameters(snap.rfx_group_parameters);

    // Restore "working" parameter prior covariance
    random_effects_model_->SetWorkingParameterCovariance(snap.rfx_working_parameter_covariance);

    // Restore "group" parameter prior covariance
    random_effects_model_->SetGroupParameterCovariance(snap.rfx_group_parameter_covariance);

    // Restore variance model priors
    random_effects_model_->SetVariancePriorShape(snap.rfx_variance_prior_shape);
    random_effects_model_->SetVariancePriorScale(snap.rfx_variance_prior_scale);

    // Swap the chain-N RFX contribution out of the residual and the GFR-snapshot
    // contribution in, exactly as the R sampler does via resetRandomEffectsTracker.
    // At this point residual_ = y - f_gfr - rfx_chain_N (forest already swapped above),
    // so ResetFromSample produces: residual_ += rfx_chain_N - rfx_gfr = y - f_gfr - rfx_gfr.
    random_effects_tracker_->ResetFromSample(*random_effects_model_, *random_effects_dataset_, *residual_);
  }

  // Other internal model state
  global_variance_ = snap.sigma2;
  leaf_scale_mu_ = snap.leaf_scale_mu;
  if (config_.tau_leaf_model_type == MeanLeafModelType::GaussianMultivariateRegression) {
    leaf_scale_tau_multivariate_ = snap.leaf_scale_tau_multivariate;
  } else if (config_.tau_leaf_model_type == MeanLeafModelType::GaussianUnivariateRegression || config_.tau_leaf_model_type == MeanLeafModelType::GaussianConstant) {
    leaf_scale_tau_ = snap.leaf_scale_tau;
  }
}

}  // namespace StochTree
