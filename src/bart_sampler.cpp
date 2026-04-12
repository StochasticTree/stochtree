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

namespace StochTree {

BARTSampler::BARTSampler(BARTSamples& samples, BARTConfig& config, BARTData& data) : config_{config}, data_{data} {
  InitializeState(samples);
}

void BARTSampler::InitializeState(BARTSamples& samples) {
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

  // Compute outcome location and scale for standardization
  if (config_.link_function == LinkFunction::Probit) {
    samples.y_std = 1.0;
    samples.y_bar = norm_inv_cdf(y_mean);
  } else {
    if (config_.standardize_outcome) {
      samples.y_bar = y_mean;
      samples.y_std = std::sqrt(y_var);
    } else {
      samples.y_bar = 0.0;
      samples.y_std = 1.0;
    }
  }

  // Standardize partial residuals in place; these are updated in each iteration but initialized to standardized outcomes
  for (int i = 0; i < data_.n_train; i++) residual_->GetData()[i] = (data_.y_train[i] - samples.y_bar) / samples.y_std;

  // Initialize mean forest state (if present)
  if (config_.num_trees_mean > 0) {
    mean_forest_ = std::make_unique<TreeEnsemble>(config_.num_trees_mean, config_.leaf_dim_mean, config_.leaf_constant_mean, config_.exponentiated_leaf_mean);
    samples.mean_forests = std::make_unique<ForestContainer>(config_.num_trees_mean, config_.leaf_dim_mean, config_.leaf_constant_mean, config_.exponentiated_leaf_mean);
    mean_forest_tracker_ = std::make_unique<ForestTracker>(forest_dataset_->GetCovariates(), config_.feature_types, config_.num_trees_mean, data_.n_train);
    tree_prior_mean_ = std::make_unique<TreePrior>(config_.alpha_mean, config_.beta_mean, config_.min_samples_leaf_mean, config_.max_depth_mean);
    mean_forest_->SetLeafValue(0.0);
    UpdateResidualEntireForest(*mean_forest_tracker_, *forest_dataset_, *residual_, mean_forest_.get(), !config_.leaf_constant_mean, std::minus<double>());
    mean_forest_tracker_->UpdatePredictions(mean_forest_.get(), *forest_dataset_.get());
    has_mean_forest_ = true;
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

  // Initialize variance forest state (if present)
  if (config_.num_trees_variance > 0) {
    variance_forest_ = std::make_unique<TreeEnsemble>(config_.num_trees_variance, config_.leaf_dim_variance, config_.leaf_constant_variance, config_.exponentiated_leaf_variance);
    samples.variance_forests = std::make_unique<ForestContainer>(config_.num_trees_variance, config_.leaf_dim_variance, config_.leaf_constant_variance, config_.exponentiated_leaf_variance);
    variance_forest_tracker_ = std::make_unique<ForestTracker>(forest_dataset_->GetCovariates(), config_.feature_types, config_.num_trees_variance, data_.n_train);
    tree_prior_variance_ = std::make_unique<TreePrior>(config_.alpha_variance, config_.beta_variance, config_.min_samples_leaf_variance, config_.max_depth_variance);
    variance_forest_->SetLeafValue(1.0 / config_.num_trees_variance);
    variance_forest_tracker_->UpdatePredictions(variance_forest_.get(), *forest_dataset_.get());
    has_variance_forest_ = true;
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

  // Other internal model state
  global_variance_ = config_.sigma2_global_init;
  leaf_scale_ = config_.sigma2_mean_init;
  // leaf_scale_multivariate_ = config_.sigma2_leaf_multivariate_init;

  initialized_ = true;
}

void BARTSampler::run_gfr(BARTSamples& samples, int num_gfr, bool keep_gfr) {
  // TODO: dispatch correct leaf model and variance model based on config; currently hardcoded to Gaussian constant-leaf and homoskedastic variance
  std::unique_ptr<GaussianConstantLeafModel> mean_leaf_model_ptr = std::make_unique<GaussianConstantLeafModel>(leaf_scale_);
  std::unique_ptr<LogLinearVarianceLeafModel> variance_leaf_model_ptr = std::make_unique<LogLinearVarianceLeafModel>(config_.shape_variance_forest, config_.scale_variance_forest);
  for (int i = 0; i < num_gfr; i++) {
    RunOneIteration(samples, mean_leaf_model_ptr.get(), variance_leaf_model_ptr.get(), /*gfr=*/true, /*keep_sample=*/keep_gfr);
  }
}

void BARTSampler::run_mcmc(BARTSamples& samples, int num_burnin, int keep_every, int num_mcmc) {
  std::unique_ptr<GaussianConstantLeafModel> mean_leaf_model_ptr = std::make_unique<GaussianConstantLeafModel>(leaf_scale_);
  std::unique_ptr<LogLinearVarianceLeafModel> variance_leaf_model_ptr = std::make_unique<LogLinearVarianceLeafModel>(config_.shape_variance_forest, config_.scale_variance_forest);
  bool keep_forest = false;
  for (int i = 0; i < num_burnin + keep_every * num_mcmc; i++) {
    if (i >= num_burnin && (i - num_burnin) % keep_every == 0)
      keep_forest = true;
    else
      keep_forest = false;
    RunOneIteration(samples, mean_leaf_model_ptr.get(), variance_leaf_model_ptr.get(), /*gfr=*/false, /*keep_sample=*/keep_forest);
  }
}

void BARTSampler::RunOneIteration(BARTSamples& samples, GaussianConstantLeafModel* mean_leaf_model, LogLinearVarianceLeafModel* variance_leaf_model, bool gfr, bool keep_sample) {
  if (has_mean_forest_) {
    if (gfr) {
      GFRSampleOneIter<GaussianConstantLeafModel, GaussianConstantSuffStat>(
          *mean_forest_, *mean_forest_tracker_, *samples.mean_forests, *mean_leaf_model,
          *forest_dataset_, *residual_, *tree_prior_mean_, rng_,
          config_.var_weights_mean, config_.sweep_update_indices_mean, global_variance_, config_.feature_types,
          config_.cutpoint_grid_size, /*keep_forest=*/keep_sample,
          /*pre_initialized=*/true, /*backfitting=*/true,
          /*num_features_subsample=*/data_.p, config_.num_threads);
    } else {
      MCMCSampleOneIter<GaussianConstantLeafModel, GaussianConstantSuffStat>(
          *mean_forest_, *mean_forest_tracker_, *samples.mean_forests, *mean_leaf_model,
          *forest_dataset_, *residual_, *tree_prior_mean_, rng_,
          config_.var_weights_mean, config_.sweep_update_indices_mean, global_variance_, /*keep_forest=*/keep_sample,
          /*pre_initialized=*/true, /*backfitting=*/true,
          /*num_threads=*/config_.num_threads);
    }
  }

  if (has_variance_forest_) {
    if (gfr) {
      GFRSampleOneIter<LogLinearVarianceLeafModel, LogLinearVarianceSuffStat>(
          *variance_forest_, *variance_forest_tracker_, *samples.variance_forests, *variance_leaf_model,
          *forest_dataset_, *residual_, *tree_prior_variance_, rng_,
          config_.var_weights_variance, config_.sweep_update_indices_variance, global_variance_, config_.feature_types,
          config_.cutpoint_grid_size, /*keep_forest=*/keep_sample,
          /*pre_initialized=*/true, /*backfitting=*/false,
          /*num_features_subsample=*/data_.p, config_.num_threads);
    } else {
      MCMCSampleOneIter<LogLinearVarianceLeafModel, LogLinearVarianceSuffStat>(
          *variance_forest_, *variance_forest_tracker_, *samples.variance_forests, *variance_leaf_model,
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
    mean_leaf_model->SetScale(leaf_scale_);
  }

  if (keep_sample) {
    samples.num_samples++;
    if (sample_sigma2_global_) samples.global_error_variance_samples.push_back(global_variance_);
    if (sample_sigma2_leaf_) samples.leaf_scale_samples.push_back(leaf_scale_);
    if (has_mean_forest_) {
      double* mean_forest_preds_train = mean_forest_tracker_->GetSumPredictions();
      samples.mean_forest_predictions_train.insert(samples.mean_forest_predictions_train.end(),
                                                   mean_forest_preds_train, mean_forest_preds_train + samples.num_train);
      if (has_test_) {
        std::vector<double> predictions = samples.mean_forests->GetEnsemble(samples.num_samples - 1)->Predict(*forest_dataset_test_);
        samples.mean_forest_predictions_test.insert(samples.mean_forest_predictions_test.end(),
                                                    predictions.data(), predictions.data() + samples.num_test);
      }
    }
  }
}

}  // namespace StochTree
