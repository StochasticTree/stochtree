/*! Copyright (c) 2026 by stochtree authors */
#include <stochtree/bart.h>
#include <stochtree/bart_sampler.h>
#include <stochtree/distributions.h>
#include <stochtree/leaf_model.h>
#include <stochtree/meta.h>
#include <stochtree/probit.h>
#include <stochtree/tree_sampler.h>
#include <stochtree/variance_model.h>
#include <algorithm>
#include <memory>
#include <random>

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

  // Standardization and calibration for mean forests
  if (config_.num_trees_mean > 0) {
    // Initialize leaf model
    if (config_.mean_leaf_model_type == MeanLeafModelType::GaussianConstant) {
      mean_leaf_model_ = GaussianConstantLeafModel(config_.sigma2_mean_init);
    } else if (config_.mean_leaf_model_type == MeanLeafModelType::GaussianUnivariateRegression) {
      mean_leaf_model_ = GaussianUnivariateRegressionLeafModel(config_.sigma2_mean_init);
    } else if (config_.mean_leaf_model_type == MeanLeafModelType::GaussianMultivariateRegression) {
      // TODO
      // mean_leaf_model_ = GaussianMultivariateRegressionLeafModel(...);
    } else if (config_.mean_leaf_model_type == MeanLeafModelType::CloglogOrdinal) {
      mean_leaf_model_ = CloglogOrdinalLeafModel(config_.cloglog_leaf_prior_shape, config_.cloglog_leaf_prior_scale);
    } else {
      Log::Fatal("Unsupported leaf model type for mean forest");
    }

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
        // TODO ...
      } else {
        // Case 3: Multivariate leaf regression
        // TODO ...
      }
    }
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
    variance_forest_->SetLeafValue(init_val_variance_ / config_.num_trees_variance);
    variance_forest_tracker_->UpdatePredictions(variance_forest_.get(), *forest_dataset_.get());
    has_variance_forest_ = true;
  }

  // Global error variance model
  if (config_.sample_sigma2_global) {
    var_model_ = std::make_unique<GlobalHomoskedasticVarianceModel>();
    sample_sigma2_global_ = true;
  }

  // Leaf scale model
  if (config_.sample_sigma2_leaf_mean) {
    leaf_scale_model_ = std::make_unique<LeafNodeHomoskedasticVarianceModel>();
    sample_sigma2_leaf_ = true;
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

void BARTSampler::run_gfr(BARTSamples& samples, int num_gfr, bool keep_gfr) {
  // Reserve space for GFR predictions if they are to be retained
  if (keep_gfr) {
    if (has_mean_forest_) {
      samples.mean_forest_predictions_train.reserve(data_.n_train * num_gfr);
    }
    if (has_variance_forest_) {
      samples.variance_forest_predictions_train.reserve(data_.n_train * num_gfr);
    }
  }

  // TODO: dispatch correct leaf model and variance model based on config; currently hardcoded to Gaussian constant-leaf and homoskedastic variance
  for (int i = 0; i < num_gfr; i++) {
    RunOneIteration(samples, /*gfr=*/true, /*keep_sample=*/keep_gfr);
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
    RunOneIteration(samples, /*gfr=*/false, /*keep_sample=*/keep_forest);
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
  }
}

void BARTSampler::RunOneIteration(BARTSamples& samples, bool gfr, bool keep_sample) {
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

  if (config_.link_function == LinkFunction::Cloglog) {
    // TODO
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

  if (keep_sample) {
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
}

}  // namespace StochTree
