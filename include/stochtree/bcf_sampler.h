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
#include <stochtree/random_effects.h>
#include <stochtree/tree_sampler.h>
#include <stochtree/variance_model.h>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace StochTree {

class BCFSampler {
 public:
  // If continuation is true, the sampler warm-starts the active mu and tau forests from
  // the last retained samples already present in `samples` (rather than initializing them
  // to root), and the existing forest containers are preserved so that new samples are
  // appended to them.
  //
  // If warmstart_source is non-null (and continuation is false), this is a FRESH run whose forests /
  // scalars / tau_0 / adaptive-coding / rfx are seeded from an EXTERNAL model's samples
  // (*warmstart_source) at warmstart_sample_num (1-indexed). This backs bcf(previous_model_json=...):
  // the destination `samples` containers are fresh, but the active forests are reconstituted from the
  // previous model's retained sample. Same-scale is assumed (no leaf rescale); the previous model's
  // y_std is used to convert its stored global variance / tau_0 back to standardized space. The
  // borrowed pointer need only outlive this constructor call.
  BCFSampler(BCFSamples& samples, BCFConfig& config, BCFData& data, bool continuation = false,
             BCFSamples* warmstart_source = nullptr, int warmstart_sample_num = 0);

  // Main entry point for running the BCF GFR sampler
  // If num_chains > 0, captures snapshots of the last num_chains GFR states for fork_chains()
  void run_gfr(BCFSamples& samples, int num_gfr, bool keep_gfr, int num_chains = 0);

  // Run a single chain of the BCF MCMC sampler
  void run_mcmc(BCFSamples& samples, int num_burnin, int keep_every, int num_mcmc);

  // Run num_chains independent MCMC chains sequentually based on GFR snapshots captured by run_gfr() or re-initialized from root
  void run_mcmc_chains(BCFSamples& samples, int num_chains, int num_burnin, int keep_every, int num_mcmc);

  // Post-process samples by extracting test set predictions and running any necessary transformations
  void postprocess_samples(BCFSamples& samples, int start_sample = 0);

  // Serialize the internal RNG state to a string. std::mt19937 round-trips losslessly through
  // its stream operators, so this captures the exact position in the random stream -- used to
  // persist RNG state across a sample() / continue_sampling() boundary for bit-identical results.
  std::string GetRngState() const {
    std::ostringstream oss;
    oss << rng_;
    return oss.str();
  }

  // Restore the internal RNG state from a string produced by GetRngState(). Resumes the random
  // stream at exactly the captured position. Must be called after construction (InitializeState
  // unconditionally re-seeds rng_) and before any sampling draws.
  void SetRngState(const std::string& state) {
    std::istringstream iss(state);
    iss >> rng_;
  }

  // Regenerate the probit latent outcome for a continuation warm-start. The latent is not persisted
  // (it is re-drawn each MCMC iteration), so this draws a fresh z ~ p(z | y, mu + Z*tau + rfx + tau_0)
  // to place the residual in a valid, stationary state before the first continued draw. No-op unless
  // the model uses a probit link. MUST be called after SetRngState so the draw comes from the resumed
  // (or user-re-seeded) stream rather than the pre-seed RNG.
  void RegenerateProbitLatent(BCFSamples& samples);

 private:
  /*! Initialize state variables */
  void InitializeState(BCFSamples& samples, bool continuation = false);
  bool initialized_ = false;

  /*! Internal function to restore sampler state based on a GFR snapshot */
  void RestoreStateFromGFRSnapshot(BCFSamples& samples, int snapshot_index);

  /*! Internal function to restore sampler state to root / initial values */
  void RestoreStateDefault();

  /*! Internal sample runner function */
  void RunOneIteration(BCFSamples& samples, bool gfr, bool keep_sample, bool write_snapshot = false);

  /*! Internal function to sample parametric treatment effect "intercept" term (tau_0 in stochtree nomenclature) */
  void SampleParametricTreatmentEffect();

  /*! Internal function to sample adaptive coding parameters for binary treatment */
  void SampleAdaptiveCodingParameters();

  /*! Internal reference to config and data state */
  BCFConfig& config_;
  BCFData& data_;

  /*! External warm-start source for bcf(previous_model_json=...); null for a fresh or continuation
   *  run. Borrowed pointer, valid only during construction. warmstart_sample_num_ is 1-indexed. */
  BCFSamples* warmstart_source_ = nullptr;
  int warmstart_sample_num_ = 0;

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

  // Treatment intercept term
  double tau_0_scalar_;
  std::vector<double> tau_0_vector_;
  bool sample_tau_0_ = false;

  // Adaptive coding parameters
  double b_0_;
  double b_1_;
  bool adaptive_coding_ = false;
  std::vector<double> tau_basis_vector_train_;
  std::vector<double> tau_basis_vector_test_;

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

    // Treatment intercept
    double tau_0_scalar;
    std::vector<double> tau_0_vector;

    // Adaptive coding
    double b_0;
    double b_1;

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
