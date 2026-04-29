/*!
 * Copyright (c) 2026 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_BCF_SAMPLER_H_
#define STOCHTREE_BCF_SAMPLER_H_

#include <stochtree/bcf.h>
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
#include "stochtree/random_effects.h"

namespace StochTree {

class BCFSampler {
 public:
  BCFSampler(BCFSamples& samples, BCFConfig& config, BCFData& data);

  // Main entry point for running the BCF GFR sampler
  // If num_chains > 0, captures snapshots of the last num_chains GFR states for fork_chains()
  void run_gfr(BCFSamples& samples, int num_gfr, bool keep_gfr, int num_chains = 0);

  // Run a single chain of the BCF MCMC sampler
  void run_mcmc(BCFSamples& samples, int num_burnin, int keep_every, int num_mcmc);

  // Run num_chains independent MCMC chains sequentually based on GFR snapshots captured by run_gfr() or re-initialized from root
  void run_mcmc_chains(BCFSamples& samples, int num_chains, int num_burnin, int keep_every, int num_mcmc);

  // Post-process samples by extracting test set predictions and running any necessary transformations
  void postprocess_samples(BCFSamples& samples);

 private:
  /*! Initialize state variables */
  void InitializeState(BCFSamples& samples);
  bool initialized_ = false;

  /*! Internal function to restore sampler state based on a GFR snapshot */
  void RestoreStateFromGFRSnapshot(BCFSamples& samples, int snapshot_index);

  /*! Internal function to restore sampler state to root / initial values */
  void RestoreStateDefault();

  /*! Internal sample runner function */
  void RunOneIteration(BCFSamples& samples, bool gfr, bool keep_sample, bool write_snapshot = false);

  /*! Internal reference to config and data state */
  BCFConfig& config_;
  BCFData& data_;

  /*! Leaf model for mean and variance forests */
  GaussianConstantLeafModel mu_leaf_model_;
  std::variant<GaussianUnivariateRegressionLeafModel, GaussianMultivariateRegressionLeafModel> tau_leaf_model_;
  LogLinearVarianceLeafModel variance_leaf_model_;

  /*! Mean forest state */
  std::unique_ptr<TreeEnsemble> mu_forest_;
  std::unique_ptr<ForestTracker> mu_forest_tracker_;
  std::unique_ptr<TreePrior> tree_prior_mu_;
  std::unique_ptr<TreeEnsemble> tau_forest_;
  std::unique_ptr<ForestTracker> tau_forest_tracker_;
  std::unique_ptr<TreePrior> tree_prior_tau_;
  double init_val_mu_;
  double init_val_tau_;
  std::vector<double> init_val_tau_vec_;

  /*! Variance forest state */
  std::unique_ptr<TreeEnsemble> variance_forest_;
  std::unique_ptr<ForestTracker> variance_forest_tracker_;
  std::unique_ptr<TreePrior> tree_prior_variance_;
  bool has_variance_forest_ = false;
  double init_val_variance_;

  /*! Random effects state */
  std::unique_ptr<MultivariateRegressionRandomEffectsModel> random_effects_model_;
  std::unique_ptr<RandomEffectsTracker> random_effects_tracker_;
  std::unique_ptr<RandomEffectsDataset> random_effects_dataset_;
  bool has_random_effects_ = false;

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
  double leaf_scale_mu_;
  double leaf_scale_tau_;
  std::vector<double> leaf_scale_tau_multivariate_;

  /*! Probit terms / helpers */
  std::vector<double> model_preds_;

  /*! Raw tau(x) predictions (sum of leaf values across trees, no z multiplication), maintained each step via leaf-lookup */
  std::vector<double> tau_raw_sum_preds_;

  // Global error scale model
  std::unique_ptr<GlobalHomoskedasticVarianceModel> var_model_;
  bool sample_sigma2_global_ = false;

  // Leaf scale models
  std::unique_ptr<LeafNodeHomoskedasticVarianceModel> leaf_scale_model_mu_;
  bool sample_sigma2_leaf_mu_ = false;
  std::unique_ptr<LeafNodeHomoskedasticVarianceModel> leaf_scale_model_tau_;
  bool sample_sigma2_leaf_tau_ = false;

  /*! GFR iteration visitor for tau forest */
  struct GFROneIterationVisitorTau {
    BCFSampler& sampler;
    BCFSamples& samples;
    bool keep_sample;
    void operator()(GaussianUnivariateRegressionLeafModel& model) {
      GFRSampleOneIter<GaussianUnivariateRegressionLeafModel, GaussianUnivariateRegressionSuffStat>(
          *sampler.tau_forest_, *sampler.tau_forest_tracker_, *samples.tau_forests, model,
          *sampler.forest_dataset_, *sampler.residual_, *sampler.tree_prior_tau_, sampler.rng_,
          sampler.config_.var_weights_tau, sampler.config_.sweep_update_indices_tau, sampler.global_variance_, sampler.config_.feature_types,
          sampler.config_.cutpoint_grid_size, /*keep_forest=*/keep_sample,
          /*pre_initialized=*/true, /*backfitting=*/true,
          /*num_features_subsample=*/sampler.config_.num_features_subsample_tau, sampler.config_.num_threads);
    }
    void operator()(GaussianMultivariateRegressionLeafModel& model) {
      GFRSampleOneIter<GaussianMultivariateRegressionLeafModel, GaussianMultivariateRegressionSuffStat, int>(
          *sampler.tau_forest_, *sampler.tau_forest_tracker_, *samples.tau_forests, model,
          *sampler.forest_dataset_, *sampler.residual_, *sampler.tree_prior_tau_, sampler.rng_,
          sampler.config_.var_weights_tau, sampler.config_.sweep_update_indices_tau, sampler.global_variance_, sampler.config_.feature_types,
          sampler.config_.cutpoint_grid_size, /*keep_forest=*/keep_sample,
          /*pre_initialized=*/true, /*backfitting=*/true,
          /*num_features_subsample=*/sampler.config_.num_features_subsample_tau, sampler.config_.num_threads,
          sampler.config_.leaf_dim_tau);
    }
  };

  /*! MCMC iteration visitor */
  struct MCMCOneIterationVisitorTau {
    BCFSampler& sampler;
    BCFSamples& samples;
    bool keep_sample;
    void operator()(GaussianUnivariateRegressionLeafModel& model) {
      MCMCSampleOneIter<GaussianUnivariateRegressionLeafModel, GaussianUnivariateRegressionSuffStat>(
          *sampler.tau_forest_, *sampler.tau_forest_tracker_, *samples.tau_forests, model,
          *sampler.forest_dataset_, *sampler.residual_, *sampler.tree_prior_tau_, sampler.rng_,
          sampler.config_.var_weights_tau, sampler.config_.sweep_update_indices_tau, sampler.global_variance_, /*keep_forest=*/keep_sample,
          /*pre_initialized=*/true, /*backfitting=*/true,
          /*num_threads=*/sampler.config_.num_threads);
    }
    void operator()(GaussianMultivariateRegressionLeafModel& model) {
      MCMCSampleOneIter<GaussianMultivariateRegressionLeafModel, GaussianMultivariateRegressionSuffStat, int>(
          *sampler.tau_forest_, *sampler.tau_forest_tracker_, *samples.tau_forests, model,
          *sampler.forest_dataset_, *sampler.residual_, *sampler.tree_prior_tau_, sampler.rng_,
          sampler.config_.var_weights_tau, sampler.config_.sweep_update_indices_tau, sampler.global_variance_, /*keep_forest=*/keep_sample,
          /*pre_initialized=*/true, /*backfitting=*/true,
          /*num_threads=*/sampler.config_.num_threads, sampler.config_.leaf_dim_tau);
    }
  };

  /*! Mu / tau forest leaf scale update visitor */
  struct ScaleUpdateVisitor {
    BCFSampler& sampler;
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

  struct TauForestResetVisitor {
    BCFSampler& sampler;
    BCFSamples& samples;
    TreeEnsemble& forest;
    void operator()(GaussianUnivariateRegressionLeafModel& model) {
      sampler.tau_forest_->ReconstituteFromForest(forest);
      sampler.tau_forest_tracker_->ReconstituteFromForest(forest, *sampler.forest_dataset_, *sampler.residual_, true);
      sampler.tau_forest_tracker_->UpdatePredictions(sampler.tau_forest_.get(), *sampler.forest_dataset_.get());
    }
    void operator()(GaussianMultivariateRegressionLeafModel& model) {
      sampler.tau_forest_->ReconstituteFromForest(forest);
      sampler.tau_forest_tracker_->ReconstituteFromForest(forest, *sampler.forest_dataset_, *sampler.residual_, true);
      sampler.tau_forest_tracker_->UpdatePredictions(sampler.tau_forest_.get(), *sampler.forest_dataset_.get());
    }
  };

  /*! Snapshot of sampler state captured at the end of a GFR iteration, used to initialize independent MCMC chains */
  struct GFRSnapshot {
    // Forest state
    std::unique_ptr<TreeEnsemble> mu_forest;
    std::unique_ptr<TreeEnsemble> tau_forest;
    std::unique_ptr<TreeEnsemble> variance_forest;  // null if no variance forest

    // Global parameters
    double sigma2;
    double leaf_scale_mu;
    double leaf_scale_tau;
    std::vector<double> leaf_scale_tau_multivariate;

    // Residual (incorporates forest + RFX contributions for a given sampler iteration)
    std::vector<double> residual;

    // Heteroskedastic variance model state
    std::vector<double> variance_weights;  // forest_dataset_ var_weights at snapshot time; only valid when variance_forest != null

    // RFX model state (only populated when has_random_effects_)
    Eigen::VectorXd rfx_working_parameter;
    Eigen::MatrixXd rfx_group_parameters;
    Eigen::MatrixXd rfx_group_parameter_covariance;
    Eigen::MatrixXd rfx_working_parameter_covariance;
    double rfx_variance_prior_shape;
    double rfx_variance_prior_scale;
  };

  /*! GFR snapshots captured during run_gfr() for use by multi-chain sampler */
  std::vector<GFRSnapshot> gfr_snapshots_;
};

}  // namespace StochTree

#endif  // STOCHTREE_BCF_SAMPLER_H_
