/*! Copyright (c) 2026 by stochtree authors */
#include <stochtree/bart.h>
#include <stochtree/bart_sampler.h>
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

BARTSampler::BARTSampler(BARTSamples& samples, BARTConfig& config, BARTData& data) : config_{config}, data_{data}, mean_leaf_model_(GaussianConstantLeafModel(0.0)), variance_leaf_model_(0.0, 0.0) {
  InitializeState(samples);
}

void BARTSampler::InitializeState(BARTSamples& samples) {
  // Validate y_train values match the expected support for discrete link functions
  if (config_.link_function == LinkFunction::Probit) {
    for (int i = 0; i < data_.n_train; i++) {
      if (data_.y_train[i] != 0.0 && data_.y_train[i] != 1.0) {
        Log::Fatal("Outcomes must be 0 or 1 for probit link function");
      }
    }
  } else if (config_.link_function == LinkFunction::Cloglog) {
    for (int i = 0; i < data_.n_train; i++) {
      if (std::floor(data_.y_train[i]) != data_.y_train[i]) {
        Log::Fatal("Outcomes must be integers for cloglog link function");
      }
      if (data_.y_train[i] < 0.0) {
        Log::Fatal("Outcomes must be 0-indexed for cloglog link function; remap before calling the sampler");
      }
    }
  }

  // Load data from BARTData object into ForestDataset object
  forest_dataset_ = std::make_unique<ForestDataset>();
  forest_dataset_->AddCovariates(data_.X_train, data_.n_train, data_.p, /*row_major=*/false);
  if (data_.basis_train != nullptr) {
    forest_dataset_->AddBasis(data_.basis_train, data_.n_train, data_.basis_dim, /*row_major=*/false);
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
    if (data_.basis_test != nullptr) {
      forest_dataset_test_->AddBasis(data_.basis_test, data_.n_test, data_.basis_dim, /*row_major=*/false);
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

  // Standardization, calibration, and initialization for mean forests
  if (config_.num_trees_mean > 0) {
    if (config_.link_function == LinkFunction::Probit) {
      // Initialize forests to 0, no scaling, but offset by the probit transform of the mean outcome to improve mixing
      samples.y_std = 1.0;
      samples.y_bar = norm_inv_cdf(y_mean);
      init_val_mean_ = 0.0;
    } else if (config_.link_function == LinkFunction::Cloglog) {
      // Initialize forests to 0, no scaling or location shifting of the outcome
      // Outcomes are expected to already be 0-indexed by the caller
      samples.y_std = 1.0;
      samples.y_bar = 0.0;
      init_val_mean_ = 0.0;
    } else {
      if (config_.mean_leaf_model_type == MeanLeafModelType::GaussianConstant) {
        // Case 1: Constant leaf
        if (config_.standardize_outcome) {
          samples.y_bar = y_mean;
          samples.y_std = std::sqrt(y_var);
          init_val_mean_ = 0.0;
        } else {
          samples.y_bar = 0.0;
          samples.y_std = 1.0;
          init_val_mean_ = y_mean;
        }
      } else if (config_.mean_leaf_model_type == MeanLeafModelType::GaussianUnivariateRegression) {
        // Case 2: Univariate leaf regression
        if (config_.standardize_outcome) {
          samples.y_bar = y_mean;
          samples.y_std = std::sqrt(y_var);
        } else {
          samples.y_bar = 0.0;
          samples.y_std = 1.0;
        }
        // Always map initial leaf value to zero
        // Users fitting a univariate leaf regression (with a non-centered basis) should standardize their outcomes
        // TODO: consider adding warning in R / Python if univariate regression leaf model is specified without standardization
        init_val_mean_ = 0.0;
      } else {
        // Case 3: Multivariate leaf regression
        if (config_.standardize_outcome) {
          samples.y_bar = y_mean;
          samples.y_std = std::sqrt(y_var);
        } else {
          samples.y_bar = 0.0;
          samples.y_std = 1.0;
        }
        init_val_mean_ = 0.0;
        init_val_mean_vec_.assign(config_.leaf_dim_mean, 0.0);
      }
    }

    // Calibrate leaf scale and variance model priors
    if (config_.sigma2_mean_init < 0.0) {
      if (config_.link_function == LinkFunction::Probit) {
        config_.sigma2_mean_init = 1.0 / config_.num_trees_mean;
      } else {
        config_.sigma2_mean_init = y_var / config_.num_trees_mean;
      }
    }
    if (config_.sample_sigma2_leaf_mean) {
      if (config_.b_sigma2_mean <= 0.0) {
        if (config_.link_function == LinkFunction::Probit) {
          config_.b_sigma2_mean = 1.0 / (2 * config_.num_trees_mean);
        } else {
          config_.b_sigma2_mean = y_var / (2 * config_.num_trees_mean);
        }
      }
    }

    // Initialize leaf model
    if (config_.mean_leaf_model_type == MeanLeafModelType::GaussianConstant) {
      mean_leaf_model_ = GaussianConstantLeafModel(config_.sigma2_mean_init);
    } else if (config_.mean_leaf_model_type == MeanLeafModelType::GaussianUnivariateRegression) {
      mean_leaf_model_ = GaussianUnivariateRegressionLeafModel(config_.sigma2_mean_init);
    } else if (config_.mean_leaf_model_type == MeanLeafModelType::GaussianMultivariateRegression) {
      Eigen::MatrixXd Sigma_0;
      if (!config_.sigma2_leaf_mean_matrix.empty()) {
        if ((int)config_.sigma2_leaf_mean_matrix.size() != config_.leaf_dim_mean * config_.leaf_dim_mean) {
          Log::Fatal("sigma2_leaf_mean_matrix must have leaf_dim_mean * leaf_dim_mean = %d elements, but has %zu",
                     config_.leaf_dim_mean * config_.leaf_dim_mean, config_.sigma2_leaf_mean_matrix.size());
        }
        // Column-major interpretation matches both R and Eigen (python must be reordered before passing to C++)
        Sigma_0 = Eigen::Map<const Eigen::MatrixXd>(config_.sigma2_leaf_mean_matrix.data(), config_.leaf_dim_mean, config_.leaf_dim_mean);
      } else {
        Sigma_0 = config_.sigma2_mean_init * Eigen::MatrixXd::Identity(config_.leaf_dim_mean, config_.leaf_dim_mean);
      }
      mean_leaf_model_ = GaussianMultivariateRegressionLeafModel(Sigma_0);
    } else if (config_.mean_leaf_model_type == MeanLeafModelType::CloglogOrdinal) {
      mean_leaf_model_ = CloglogOrdinalLeafModel(config_.cloglog_leaf_prior_shape, config_.cloglog_leaf_prior_scale);
    } else {
      Log::Fatal("Unsupported leaf model type for mean forest");
    }
  } else {
    // Variance-only model (num_trees_mean == 0): no mean forest, but y_bar/y_std must
    // still be valid so the residual initialisation below doesn't divide by zero.
    if (config_.standardize_outcome) {
      samples.y_bar = y_mean;
      samples.y_std = std::sqrt(y_var);
    } else {
      samples.y_bar = 0.0;
      samples.y_std = 1.0;
    }
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

  // Initialize mean forest state (if present)
  if (config_.num_trees_mean > 0) {
    std::visit(MeanForestInitVisitor{*this, samples}, mean_leaf_model_);
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

  // Leaf scale model
  if (config_.sample_sigma2_leaf_mean && config_.num_trees_mean > 0) {
    leaf_scale_model_ = std::make_unique<LeafNodeHomoskedasticVarianceModel>();
    sample_sigma2_leaf_ = true;
  }

  // Random effects model
  if (config_.has_random_effects) {
    random_effects_dataset_ = std::make_unique<RandomEffectsDataset>();
    random_effects_dataset_->AddGroupLabels(data_.rfx_group_ids_train, data_.n_train);
    if (data_.rfx_basis_train != nullptr) {
      random_effects_dataset_->AddBasis(data_.rfx_basis_train, data_.n_train, data_.rfx_basis_dim, /*row_major=*/false);
    } else {
      if (config_.rfx_model_spec == BARTRFXModelSpec::InterceptOnly) {
        // If no basis is provided, add an intercept basis (column of 1s)
        // TODO: do we need to do this before we determine rfx_basis_dim and initialize the RFX data structures?
        std::vector<double> intercept_basis(data_.n_train, 1.0);
        random_effects_dataset_->AddBasis(intercept_basis.data(), data_.n_train, 1, /*row_major=*/false);
        // Override rfx_basis_dim to 1 for intercept-only model the basis is a 1-dimensional vector of ones
        data_.rfx_basis_dim = 1;
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

  // Cloglog state
  if (config_.link_function == LinkFunction::Cloglog) {
    // Initialize the ordinal sampler
    ordinal_sampler_ = std::make_unique<OrdinalSampler>();
    // Latent variable (Z in Alam et al (2025) notation)
    forest_dataset_->AddAuxiliaryDimension(data_.n_train);
    // Forest predictions (eta in Alam et al (2025) notation)
    forest_dataset_->AddAuxiliaryDimension(data_.n_train);
    // Log-scale non-cumulative cutpoint (gamma in Alam et al (2025) notation)
    forest_dataset_->AddAuxiliaryDimension(config_.num_classes_cloglog - 1);
    // Exponentiated cumulative cutpoints (exp(c_k) in Alam et al (2025) notation)
    // This auxiliary series is designed so that the element stored at position `i`
    // corresponds to the sum of all exponentiated gamma_j values for j < i.
    // It has cloglog_num_categories elements instead of cloglog_num_categories - 1 because
    // even the largest categorical index has a valid value of sum_{j < i} exp(gamma_j)
    forest_dataset_->AddAuxiliaryDimension(config_.num_classes_cloglog);

    // Set initial values for auxiliary data
    // Initialize latent variables to zero (slot 0)
    for (int i = 0; i < data_.n_train; i++) {
      forest_dataset_->SetAuxiliaryDataValue(0, i, 0.0);
    }
    // Initialize forest predictions to zero (slot 1)
    for (int i = 0; i < data_.n_train; i++) {
      forest_dataset_->SetAuxiliaryDataValue(1, i, 0.0);
    }
    // Initialize log-scale cutpoints to 0
    for (int i = 0; i < config_.num_classes_cloglog - 1; i++) {
      forest_dataset_->SetAuxiliaryDataValue(2, i, 0.0);
    }
    // Convert to cumulative exponentiated cutpoints directly in C++
    ordinal_sampler_->UpdateCumulativeExpSums(*forest_dataset_);
  }

  // Other internal model state
  global_variance_ = config_.sigma2_global_init;
  leaf_scale_ = config_.sigma2_mean_init;
  // leaf_scale_multivariate_ = config_.sigma2_leaf_multivariate_init;

  initialized_ = true;
}

void BARTSampler::run_gfr(BARTSamples& samples, int num_gfr, bool keep_gfr, int num_chains) {
  // Reserve space for GFR predictions if they are to be retained
  if (keep_gfr) {
    if (has_mean_forest_) {
      samples.mean_forest_predictions_train.reserve(data_.n_train * num_gfr);
    }
    if (has_variance_forest_) {
      samples.variance_forest_predictions_train.reserve(data_.n_train * num_gfr);
    }
  }

  // NOTE: for serial sampling (which is all we currently support), we can use the current BARTSampler state to
  // initialize the first MCMC chain, so we only need to keep GFR snapshots for the number of chains minus one
  // (the first chain uses the final GFR state). If num_chains is 1, we keep no snapshots.
  int snapshot_start = (num_chains > 1) ? std::max(0, num_gfr - (num_chains - 1)) : num_gfr;
  if (num_chains > 1) {
    gfr_snapshots_.clear();
    gfr_snapshots_.reserve(num_chains - 1);
  }

  bool write_snapshot = false;
  for (int i = 0; i < num_gfr; i++) {
    write_snapshot = (i >= snapshot_start);
    RunOneIteration(samples, /*gfr=*/true, /*keep_sample=*/keep_gfr, /*write_snapshot=*/write_snapshot);
  }
}

void BARTSampler::run_mcmc(BARTSamples& samples, int num_burnin, int keep_every, int num_mcmc) {
  // Reserve space for MCMC predictions if they are to be retained
  if (has_mean_forest_) {
    samples.mean_forest_predictions_train.reserve(data_.n_train * num_mcmc);
    if (has_test_) {
      samples.mean_forest_predictions_test.reserve(data_.n_test * num_mcmc);
    }
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

void BARTSampler::run_mcmc_chains(BARTSamples& samples, int num_chains, int num_burnin, int keep_every, int num_mcmc) {
  for (int chain_idx = 0; chain_idx < num_chains; chain_idx++) {
    if (chain_idx > 0) {
      // Re-initialize the sampler state for each new chain.
      // Snapshots are stored oldest-first; chain 2 gets the most recent snapshot
      // (index num_chains-2), chain 3 the next-most-recent (num_chains-3), etc.
      RestoreStateFromGFRSnapshot(samples, num_chains - 1 - chain_idx);
    }
    run_mcmc(samples, num_burnin, keep_every, num_mcmc);
  }
}

void BARTSampler::postprocess_samples(BARTSamples& samples) {
  // Unpack test set predictions for mean and variance forest
  if (has_test_) {
    if (has_mean_forest_) {
      std::vector<double> predictions = samples.mean_forests->Predict(*forest_dataset_test_);
      samples.mean_forest_predictions_test.insert(samples.mean_forest_predictions_test.end(),
                                                  predictions.data(), predictions.data() + predictions.size());
    }
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
  }
}

void BARTSampler::RunOneIteration(BARTSamples& samples, bool gfr, bool keep_sample, bool write_snapshot) {
  if (has_mean_forest_) {
    if (gfr) {
      std::visit(GFROneIterationVisitor{*this, samples, keep_sample}, mean_leaf_model_);
    } else {
      std::visit(MCMCOneIterationVisitor{*this, samples, keep_sample}, mean_leaf_model_);
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
    sample_probit_latent_outcome(rng_, outcome_raw_->GetData().data(), mean_forest_tracker_->GetSumPredictions(),
                                 residual_->GetData().data(), samples.y_bar, data_.n_train);
  }

  if (sample_sigma2_global_) {
    global_variance_ = var_model_->SampleVarianceParameter(
        residual_->GetData(), config_.a_sigma2_global, config_.b_sigma2_global, rng_);
  }

  if (sample_sigma2_leaf_) {
    leaf_scale_ = leaf_scale_model_->SampleVarianceParameter(
        mean_forest_.get(), config_.a_sigma2_mean, config_.b_sigma2_mean, rng_);
    std::visit(ScaleUpdateVisitor{*this, leaf_scale_}, mean_leaf_model_);
  }

  // Gibbs updates for the cloglog model
  if (config_.link_function == LinkFunction::Cloglog) {
    // Update auxiliary data to current forest predictions
    for (int i = 0; i < data_.n_train; i++) {
      forest_dataset_->SetAuxiliaryDataValue(1, i, mean_forest_tracker_->GetSamplePrediction(i));
    }

    // Sample latent z_i's using truncated exponential
    ordinal_sampler_->UpdateLatentVariables(*forest_dataset_, residual_->GetData(), rng_);

    // Sample gamma parameters (cutpoints)
    ordinal_sampler_->UpdateGammaParams(*forest_dataset_, residual_->GetData(), config_.cloglog_leaf_prior_shape, config_.cloglog_leaf_prior_scale, config_.cloglog_cutpoint_0, rng_);

    // Update cumulative sum of exp(gamma) values
    ordinal_sampler_->UpdateCumulativeExpSums(*forest_dataset_);
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
    if (sample_sigma2_leaf_) samples.leaf_scale_samples.push_back(leaf_scale_);
    if (has_mean_forest_) {
      double* mean_forest_preds_train = mean_forest_tracker_->GetSumPredictions();
      samples.mean_forest_predictions_train.insert(samples.mean_forest_predictions_train.end(),
                                                   mean_forest_preds_train,
                                                   mean_forest_preds_train + samples.num_train);
    }
    if (has_variance_forest_) {
      double* variance_forest_preds_train = variance_forest_tracker_->GetSumPredictions();
      samples.variance_forest_predictions_train.insert(samples.variance_forest_predictions_train.end(),
                                                       variance_forest_preds_train,
                                                       variance_forest_preds_train + samples.num_train);
    }
    if (config_.link_function == LinkFunction::Cloglog) {
      // Store cutpoint samples
      std::vector<double> cloglog_cutpoints(config_.num_classes_cloglog - 1);
      for (int i = 0; i < config_.num_classes_cloglog - 1; i++) {
        cloglog_cutpoints[i] = forest_dataset_->GetAuxiliaryDataValue(2, i);
      }
      samples.cloglog_cutpoint_samples.insert(samples.cloglog_cutpoint_samples.end(),
                                              cloglog_cutpoints.data(), cloglog_cutpoints.data() + cloglog_cutpoints.size());
    }
  }

  if (write_snapshot) {
    GFRSnapshot snap;
    if (has_mean_forest_) snap.mean_forest = std::make_unique<TreeEnsemble>(*mean_forest_);
    if (has_variance_forest_) snap.variance_forest = std::make_unique<TreeEnsemble>(*variance_forest_);
    snap.sigma2 = global_variance_;
    if (has_mean_forest_) {
      if (config_.mean_leaf_model_type == MeanLeafModelType::GaussianMultivariateRegression) {
        snap.leaf_scale_multivariate = leaf_scale_multivariate_;
      } else if (config_.mean_leaf_model_type == MeanLeafModelType::GaussianUnivariateRegression || config_.mean_leaf_model_type == MeanLeafModelType::GaussianConstant) {
        snap.leaf_scale = leaf_scale_;
      } else if (config_.mean_leaf_model_type == MeanLeafModelType::CloglogOrdinal) {
        snap.cloglog_forest_preds.clear();
        snap.cloglog_forest_preds.resize(data_.n_train);
        snap.cloglog_forest_preds.assign(mean_forest_tracker_->GetSumPredictions(), mean_forest_tracker_->GetSumPredictions() + data_.n_train);
        snap.cloglog_latent_outcome.clear();
        snap.cloglog_latent_outcome.resize(data_.n_train);
        for (int i = 0; i < data_.n_train; i++) {
          snap.cloglog_latent_outcome[i] = forest_dataset_->GetAuxiliaryDataValue(0, i);
        }
        snap.cloglog_logscale_cutpoints.clear();
        snap.cloglog_logscale_cutpoints.resize(config_.num_classes_cloglog - 1);
        for (int i = 0; i < config_.num_classes_cloglog - 1; i++) {
          snap.cloglog_logscale_cutpoints[i] = forest_dataset_->GetAuxiliaryDataValue(2, i);
        }
      }
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

void BARTSampler::RestoreStateFromGFRSnapshot(BARTSamples& samples, int snapshot_index) {
  GFRSnapshot& snap = gfr_snapshots_[snapshot_index];

  // Restore residual from snapshot; this is used for initializing the variance forest state and must be done before initializing the variance forest tracker
  residual_->OverwriteData(snap.residual.data(), data_.n_train);

  // Restore mean forest state (if present)
  if (config_.num_trees_mean > 0) {
    std::visit(MeanForestResetVisitor{*this, samples, *snap.mean_forest}, mean_leaf_model_);
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

    // Set has_random_effects_ flag to true so that the sampler will perform random effects updates at each iteration
    has_random_effects_ = true;
  }

  // Cloglog state
  if (config_.link_function == LinkFunction::Cloglog) {
    // Reset auxiliary data values
    // Restore latent outcome values (slot 0)
    for (int i = 0; i < data_.n_train; i++) {
      forest_dataset_->SetAuxiliaryDataValue(0, i, snap.cloglog_latent_outcome[i]);
    }
    // Restore forest predictions (slot 1)
    for (int i = 0; i < data_.n_train; i++) {
      forest_dataset_->SetAuxiliaryDataValue(1, i, snap.cloglog_forest_preds[i]);
    }
    // Restore log-scale cutpoints
    for (int i = 0; i < config_.num_classes_cloglog - 1; i++) {
      forest_dataset_->SetAuxiliaryDataValue(2, i, snap.cloglog_logscale_cutpoints[i]);
    }
    // Convert to cumulative exponentiated cutpoints directly in C++
    ordinal_sampler_->UpdateCumulativeExpSums(*forest_dataset_);
  }

  // Other internal model state
  global_variance_ = snap.sigma2;
  if (has_mean_forest_) {
    if (config_.mean_leaf_model_type == MeanLeafModelType::GaussianMultivariateRegression) {
      leaf_scale_multivariate_ = snap.leaf_scale_multivariate;
    } else if (config_.mean_leaf_model_type == MeanLeafModelType::GaussianUnivariateRegression || config_.mean_leaf_model_type == MeanLeafModelType::GaussianConstant) {
      leaf_scale_ = snap.leaf_scale;
    }
  }
}

}  // namespace StochTree
