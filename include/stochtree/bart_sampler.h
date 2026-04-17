/*!
 * Copyright (c) 2026 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_BART_SAMPLER_H_
#define STOCHTREE_BART_SAMPLER_H_

#include <stochtree/bart.h>
#include <stochtree/container.h>
#include <stochtree/data.h>
#include <stochtree/ensemble.h>
#include <stochtree/leaf_model.h>
#include <stochtree/linear_regression.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/prior.h>
#include <stochtree/probit.h>
#include <stochtree/tree_sampler.h>
#include <stochtree/variance_model.h>
#include <memory>
#include <variant>
#include <vector>
#include "stochtree/ordinal_sampler.h"

namespace StochTree {

class BARTSampler {
 public:
  BARTSampler(BARTSamples& samples, BARTConfig& config, BARTData& data);

  // Main entry point for running the BART GFR sampler
  void run_gfr(BARTSamples& samples, int num_gfr, bool keep_gfr);

  // Main entry point for running the BART MCMC sampler
  void run_mcmc(BARTSamples& samples, int num_burnin, int keep_every, int num_mcmc);

  // Post-process samples by extracting test set predictions and running any necessary transformations
  void postprocess_samples(BARTSamples& samples);

 private:
  /*! Initialize state variables */
  void InitializeState(BARTSamples& samples);
  bool initialized_ = false;

  /*! Internal sample runner function */
  void RunOneIteration(BARTSamples& samples, bool gfr, bool keep_sample);

  /*! Initialization visitor */
  struct MeanForestInitVisitor {
    BARTSampler& sampler;
    BARTSamples& samples;
    void operator()(GaussianConstantLeafModel& model) {
      sampler.mean_forest_ = std::make_unique<TreeEnsemble>(sampler.config_.num_trees_mean, sampler.config_.leaf_dim_mean, sampler.config_.leaf_constant_mean, sampler.config_.exponentiated_leaf_mean);
      samples.mean_forests = std::make_unique<ForestContainer>(sampler.config_.num_trees_mean, sampler.config_.leaf_dim_mean, sampler.config_.leaf_constant_mean, sampler.config_.exponentiated_leaf_mean);
      sampler.mean_forest_tracker_ = std::make_unique<ForestTracker>(sampler.forest_dataset_->GetCovariates(), sampler.config_.feature_types, sampler.config_.num_trees_mean, sampler.data_.n_train);
      sampler.tree_prior_mean_ = std::make_unique<TreePrior>(sampler.config_.alpha_mean, sampler.config_.beta_mean, sampler.config_.min_samples_leaf_mean, sampler.config_.max_depth_mean);
      sampler.mean_forest_->SetLeafValue(sampler.init_val_mean_ / sampler.config_.num_trees_mean);
      UpdateResidualEntireForest(*sampler.mean_forest_tracker_, *sampler.forest_dataset_, *sampler.residual_, sampler.mean_forest_.get(), !sampler.config_.leaf_constant_mean, std::minus<double>());
      sampler.mean_forest_tracker_->UpdatePredictions(sampler.mean_forest_.get(), *sampler.forest_dataset_.get());
      sampler.has_mean_forest_ = true;
    }
    void operator()(GaussianUnivariateRegressionLeafModel& model) {
      sampler.mean_forest_ = std::make_unique<TreeEnsemble>(sampler.config_.num_trees_mean, sampler.config_.leaf_dim_mean, sampler.config_.leaf_constant_mean, sampler.config_.exponentiated_leaf_mean);
      samples.mean_forests = std::make_unique<ForestContainer>(sampler.config_.num_trees_mean, sampler.config_.leaf_dim_mean, sampler.config_.leaf_constant_mean, sampler.config_.exponentiated_leaf_mean);
      sampler.mean_forest_tracker_ = std::make_unique<ForestTracker>(sampler.forest_dataset_->GetCovariates(), sampler.config_.feature_types, sampler.config_.num_trees_mean, sampler.data_.n_train);
      sampler.tree_prior_mean_ = std::make_unique<TreePrior>(sampler.config_.alpha_mean, sampler.config_.beta_mean, sampler.config_.min_samples_leaf_mean, sampler.config_.max_depth_mean);
      sampler.mean_forest_->SetLeafValue(sampler.init_val_mean_ / sampler.config_.num_trees_mean);
      UpdateResidualEntireForest(*sampler.mean_forest_tracker_, *sampler.forest_dataset_, *sampler.residual_, sampler.mean_forest_.get(), !sampler.config_.leaf_constant_mean, std::minus<double>());
      sampler.mean_forest_tracker_->UpdatePredictions(sampler.mean_forest_.get(), *sampler.forest_dataset_.get());
      sampler.has_mean_forest_ = true;
    }
    void operator()(GaussianMultivariateRegressionLeafModel& model) {
      // TODO ...
    }
    void operator()(CloglogOrdinalLeafModel& model) {
      sampler.mean_forest_ = std::make_unique<TreeEnsemble>(sampler.config_.num_trees_mean, sampler.config_.leaf_dim_mean, sampler.config_.leaf_constant_mean, sampler.config_.exponentiated_leaf_mean);
      samples.mean_forests = std::make_unique<ForestContainer>(sampler.config_.num_trees_mean, sampler.config_.leaf_dim_mean, sampler.config_.leaf_constant_mean, sampler.config_.exponentiated_leaf_mean);
      sampler.mean_forest_tracker_ = std::make_unique<ForestTracker>(sampler.forest_dataset_->GetCovariates(), sampler.config_.feature_types, sampler.config_.num_trees_mean, sampler.data_.n_train);
      sampler.tree_prior_mean_ = std::make_unique<TreePrior>(sampler.config_.alpha_mean, sampler.config_.beta_mean, sampler.config_.min_samples_leaf_mean, sampler.config_.max_depth_mean);
      sampler.mean_forest_->SetLeafValue(sampler.init_val_mean_ / sampler.config_.num_trees_mean);
      UpdateResidualEntireForest(*sampler.mean_forest_tracker_, *sampler.forest_dataset_, *sampler.residual_, sampler.mean_forest_.get(), false, std::minus<double>());
      sampler.mean_forest_tracker_->UpdatePredictions(sampler.mean_forest_.get(), *sampler.forest_dataset_.get());
      sampler.has_mean_forest_ = true;
    }
  };

  /*! GFR iteration visitor */
  struct GFROneIterationVisitor {
    BARTSampler& sampler;
    BARTSamples& samples;
    bool keep_sample;
    void operator()(GaussianConstantLeafModel& model) {
      GFRSampleOneIter<GaussianConstantLeafModel, GaussianConstantSuffStat>(
          *sampler.mean_forest_, *sampler.mean_forest_tracker_, *samples.mean_forests, model,
          *sampler.forest_dataset_, *sampler.residual_, *sampler.tree_prior_mean_, sampler.rng_,
          sampler.config_.var_weights_mean, sampler.config_.sweep_update_indices_mean, sampler.global_variance_, sampler.config_.feature_types,
          sampler.config_.cutpoint_grid_size, /*keep_forest=*/keep_sample,
          /*pre_initialized=*/true, /*backfitting=*/true,
          /*num_features_subsample=*/sampler.config_.num_features_subsample_mean, sampler.config_.num_threads);
    }
    void operator()(GaussianUnivariateRegressionLeafModel& model) {
      GFRSampleOneIter<GaussianUnivariateRegressionLeafModel, GaussianUnivariateRegressionSuffStat>(
          *sampler.mean_forest_, *sampler.mean_forest_tracker_, *samples.mean_forests, model,
          *sampler.forest_dataset_, *sampler.residual_, *sampler.tree_prior_mean_, sampler.rng_,
          sampler.config_.var_weights_mean, sampler.config_.sweep_update_indices_mean, sampler.global_variance_, sampler.config_.feature_types,
          sampler.config_.cutpoint_grid_size, /*keep_forest=*/keep_sample,
          /*pre_initialized=*/true, /*backfitting=*/true,
          /*num_features_subsample=*/sampler.config_.num_features_subsample_mean, sampler.config_.num_threads);
    }
    void operator()(GaussianMultivariateRegressionLeafModel& model) {
      // TODO ...
    }
    void operator()(CloglogOrdinalLeafModel& model) {
      GFRSampleOneIter<CloglogOrdinalLeafModel, CloglogOrdinalSuffStat>(
          *sampler.mean_forest_, *sampler.mean_forest_tracker_, *samples.mean_forests, model,
          *sampler.forest_dataset_, *sampler.residual_, *sampler.tree_prior_mean_, sampler.rng_,
          sampler.config_.var_weights_mean, sampler.config_.sweep_update_indices_mean, sampler.global_variance_, sampler.config_.feature_types,
          sampler.config_.cutpoint_grid_size, /*keep_forest=*/keep_sample,
          /*pre_initialized=*/true, /*backfitting=*/false,
          /*num_features_subsample=*/sampler.config_.num_features_subsample_mean, sampler.config_.num_threads);
    }
  };

  /*! MCMC iteration visitor */
  struct MCMCOneIterationVisitor {
    BARTSampler& sampler;
    BARTSamples& samples;
    bool keep_sample;
    void operator()(GaussianConstantLeafModel& model) {
      MCMCSampleOneIter<GaussianConstantLeafModel, GaussianConstantSuffStat>(
          *sampler.mean_forest_, *sampler.mean_forest_tracker_, *samples.mean_forests, model,
          *sampler.forest_dataset_, *sampler.residual_, *sampler.tree_prior_mean_, sampler.rng_,
          sampler.config_.var_weights_mean, sampler.config_.sweep_update_indices_mean, sampler.global_variance_, /*keep_forest=*/keep_sample,
          /*pre_initialized=*/true, /*backfitting=*/true,
          /*num_threads=*/sampler.config_.num_threads);
    }
    void operator()(GaussianUnivariateRegressionLeafModel& model) {
      MCMCSampleOneIter<GaussianUnivariateRegressionLeafModel, GaussianUnivariateRegressionSuffStat>(
          *sampler.mean_forest_, *sampler.mean_forest_tracker_, *samples.mean_forests, model,
          *sampler.forest_dataset_, *sampler.residual_, *sampler.tree_prior_mean_, sampler.rng_,
          sampler.config_.var_weights_mean, sampler.config_.sweep_update_indices_mean, sampler.global_variance_, /*keep_forest=*/keep_sample,
          /*pre_initialized=*/true, /*backfitting=*/true,
          /*num_threads=*/sampler.config_.num_threads);
    }
    void operator()(GaussianMultivariateRegressionLeafModel& model) {
      // TODO ...
    }
    void operator()(CloglogOrdinalLeafModel& model) {
      MCMCSampleOneIter<CloglogOrdinalLeafModel, CloglogOrdinalSuffStat>(
          *sampler.mean_forest_, *sampler.mean_forest_tracker_, *samples.mean_forests, model,
          *sampler.forest_dataset_, *sampler.residual_, *sampler.tree_prior_mean_, sampler.rng_,
          sampler.config_.var_weights_mean, sampler.config_.sweep_update_indices_mean, sampler.global_variance_, /*keep_forest=*/keep_sample,
          /*pre_initialized=*/true, /*backfitting=*/false,
          /*num_threads=*/sampler.config_.num_threads);
    }
  };

  /*! Mean forest leaf scale update visitor */
  struct ScaleUpdateVisitor {
    BARTSampler& sampler;
    double leaf_scale;
    void operator()(GaussianConstantLeafModel& model) {
      model.SetScale(leaf_scale);
    }
    void operator()(GaussianUnivariateRegressionLeafModel& model) {
      model.SetScale(leaf_scale);
    }
    void operator()(GaussianMultivariateRegressionLeafModel& model) {
      // No-op for multivariate regression leaf model since scale is a vector
    }
    void operator()(CloglogOrdinalLeafModel& model) {
      // No-op for cloglog ordinal leaf model since scale is not a variance parameter
    }
  };

  /*! Internal reference to config and data state */
  BARTConfig& config_;
  BARTData& data_;

  /*! Leaf model for mean and variance forests */
  std::variant<GaussianConstantLeafModel, GaussianUnivariateRegressionLeafModel, GaussianMultivariateRegressionLeafModel, CloglogOrdinalLeafModel> mean_leaf_model_;
  LogLinearVarianceLeafModel variance_leaf_model_;

  /*! Mean forest state */
  std::unique_ptr<TreeEnsemble> mean_forest_;
  std::unique_ptr<ForestTracker> mean_forest_tracker_;
  std::unique_ptr<TreePrior> tree_prior_mean_;
  bool has_mean_forest_ = false;
  double init_val_mean_;
  std::unique_ptr<OrdinalSampler> ordinal_sampler_;

  /*! Variance forest state */
  std::unique_ptr<TreeEnsemble> variance_forest_;
  std::unique_ptr<ForestTracker> variance_forest_tracker_;
  std::unique_ptr<TreePrior> tree_prior_variance_;
  bool has_variance_forest_ = false;
  double init_val_variance_;

  /*! Dataset */
  std::unique_ptr<ColumnVector> residual_;
  std::unique_ptr<ColumnVector> outcome_raw_;
  std::unique_ptr<ForestDataset> forest_dataset_;
  std::unique_ptr<ForestDataset> forest_dataset_test_;
  bool has_test_ = false;

  /*! Random number generator */
  std::mt19937 rng_;

  /*! Model parameters */
  double global_variance_;
  double leaf_scale_;
  std::vector<double> leaf_scale_multivariate_;

  // Global error scale model
  std::unique_ptr<GlobalHomoskedasticVarianceModel> var_model_;
  bool sample_sigma2_global_ = false;

  // Leaf scale model
  std::unique_ptr<LeafNodeHomoskedasticVarianceModel> leaf_scale_model_;
  bool sample_sigma2_leaf_ = false;

  /*! Random effects state */
  // TODO ...

  /*! Vector of warm-start snapshots (forests needed for MCMC chains but not retained) */
  std::vector<ForestContainer> warm_start_forests_mean_;
  std::vector<ForestContainer> warm_start_forests_variance_;
};

}  // namespace StochTree

#endif  // STOCHTREE_BART_SAMPLER_H_
