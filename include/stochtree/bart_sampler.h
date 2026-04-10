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
#include <stochtree/partition_tracker.h>
#include <stochtree/prior.h>
#include <stochtree/variance_model.h>
#include <memory>
#include <vector>

namespace StochTree {

class BARTSampler {
 public:
  BARTSampler(BARTSamples& samples, BARTConfig& config, BARTData& data);

  // Main entry point for running the BART sampler, which dispatches to GFR warmup and MCMC sampling functions
  void run_gfr(BARTSamples& samples, BARTConfig& config, BARTData& data, std::mt19937& rng, int num_gfr, bool keep_gfr);

  // Main entry point for running the BART sampler, which dispatches to GFR warmup and MCMC sampling functions
  void run_mcmc(BARTSamples& samples, BARTConfig& config, BARTData& data, std::mt19937& rng, int num_burnin, int keep_every, int num_mcmc);

 private:
  /*! Initialize state variables */
  void InitializeState(BARTSamples& samples, BARTConfig& config, BARTData& data);
  bool initialized_ = false;

  /*! Internal sample runner function */
  void RunOneIteration(BARTSamples& samples, BARTConfig& config, BARTData& data, GaussianConstantLeafModel* mean_leaf_model, LogLinearVarianceLeafModel* variance_leaf_model, std::mt19937& rng, bool gfr, bool keep_sample);

  /*! Mean forest state */
  std::unique_ptr<TreeEnsemble> mean_forest_;
  std::unique_ptr<ForestTracker> mean_forest_tracker_;
  std::unique_ptr<TreePrior> tree_prior_mean_;
  bool has_mean_forest_ = false;

  /*! Variance forest state */
  std::unique_ptr<TreeEnsemble> variance_forest_;
  std::unique_ptr<ForestTracker> variance_forest_tracker_;
  std::unique_ptr<TreePrior> tree_prior_variance_;
  bool has_variance_forest_ = false;

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
