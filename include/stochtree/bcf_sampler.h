/*!
 * Copyright (c) 2026 stochtree authors. All rights reserved.
 *
 * Stateful BCF sampler (prototype).
 *
 * Mirrors the BARTSampler interface exactly:
 *
 *   BCFSampler sampler(config, data);
 *   sampler.run_gfr(n_gfr);
 *   sampler.run_mcmc(n_mcmc, result, keep_every);
 *
 * BCFSamplerFit() is a thin free-function wrapper (defined in bcf.h).
 *
 * Key structural differences from BARTSampler are documented inline and
 * summarised in src/bcf_sampler.cpp.
 */
#ifndef STOCHTREE_BCF_SAMPLER_H_
#define STOCHTREE_BCF_SAMPLER_H_

#include <stochtree/bart.h>
#include <stochtree/bcf.h>
#include <stochtree/container.h>
#include <stochtree/data.h>
#include <stochtree/ensemble.h>
#include <stochtree/leaf_model.h>
#include <stochtree/normal_sampler.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/prior.h>
#include <stochtree/random_effects.h>
#include <stochtree/variance_model.h>

#include <Eigen/Dense>

#include <memory>
#include <random>
#include <string>
#include <vector>

namespace StochTree {

struct BCFChainState;  // defined in bcf_sampler.cpp

/*!
 * \brief Stateful BCF sampler.
 *
 * Constructor performs all one-time setup.  run_gfr() and run_mcmc()
 * mirror BARTSampler exactly.
 *
 * SURPRISE vs BARTSampler: BCFSampler owns *two* complete forest stacks
 * (mu and tau), each with their own ForestDataset, ForestTracker, TreePrior,
 * and per-chain working buffers.  Most fields appear in duplicate.
 */
class BCFSampler {
 public:
  BCFSampler(const BCFConfig& config, const BCFData& data);

  BCFSampler(const BCFSampler&)            = delete;
  BCFSampler& operator=(const BCFSampler&) = delete;
  BCFSampler(BCFSampler&&)                 = delete;
  BCFSampler& operator=(BCFSampler&&)      = delete;

  ~BCFSampler() = default;

  void run_gfr(int n_gfr);
  void run_mcmc(int n_mcmc, BCFResult* result, int keep_every = 1);

 private:
  // ── Config-derived flags (set once at construction) ─────────────────
  BCFConfig config_;
  int n_train_, n_test_, p_, p_aug_, treatment_dim_;
  int num_trees_mu_, num_trees_tau_;
  // p_aug_ = p_ + 1 when propensity score is appended; p_ otherwise.
  // Both forests share this dimension; variable_weights zeros out the
  // pi_hat column for whichever forest should not use it.
  bool has_test_, has_rfx_, has_rfx_test_;
  bool adaptive_coding_, sample_intercept_;
  bool has_variance_forest_;
  int  rfx_num_groups_, rfx_num_components_;

  // ── Standardization ──────────────────────────────────────────────────
  double y_bar_, y_std_;
  std::vector<double> resid_vec_;

  // ── Calibrated prior parameters ──────────────────────────────────────
  double a_global_, b_global_, sigma2_init_;
  double a_leaf_mu_, b_leaf_mu_, leaf_scale_mu_init_;
  double a_leaf_tau_, b_leaf_tau_, leaf_scale_tau_init_;
  double a_forest_, b_forest_, var_leaf_init_;
  double tau_0_prior_var_;

  // ── Feature types (length p_aug_) ─────────────────────────────────────
  std::vector<FeatureType> feature_types_;

  // ── RNG ───────────────────────────────────────────────────────────────
  std::mt19937 rng_;

  // ── Owned augmented covariate copy ───────────────────────────────────
  // One copy of [X | π̂] (p_aug_ columns) shared by both ForestDatasets.
  // The pi_hat column is excluded from individual forests via variable_weights,
  // not by maintaining separate covariate matrices.
  std::vector<double> X_aug_train_;  ///< [X | pi_hat], n_train × p_aug_
  std::vector<double> X_aug_test_;   ///< [X | pi_hat], n_test  × p_aug_

  // ── Raw data pointer (kept for per-chain basis reconstruction) ─────────
  const double* Z_train_ptr_ = nullptr;

  // ── Variable weights and sweep indices ────────────────────────────────
  // variable_weights_mu_:  pi_hat weight = 0 unless propensity_covariate ∈ {"mu","both"}
  // variable_weights_tau_: pi_hat weight = 0 unless propensity_covariate ∈ {"tau","both"}
  std::vector<double> variable_weights_mu_;
  std::vector<double> variable_weights_tau_;
  std::vector<double> variable_weights_variance_;
  std::vector<int>    sweep_indices_mu_;
  std::vector<int>    sweep_indices_tau_;
  std::vector<int>    sweep_indices_variance_;
  int num_features_subsample_mu_  = -1;
  int num_features_subsample_tau_ = -1;

  // ── Initial leaf values ───────────────────────────────────────────────
  double init_val_mu_  = 0.0;
  double init_val_tau_ = 0.0;

  // ── Treatment basis (updated in place by adaptive_coding) ────────────
  // SURPRISE: unlike BART, the basis for the tau forest is not a fixed
  // user-supplied matrix — it is a mutable vector updated every iteration
  // by the adaptive coding step.  ForestDataset::update_basis() propagates
  // the new basis to the internal data structure, and UpdateResidualNewBasis
  // in tree_sampler.h re-derives leaf predictions from the new basis.
  std::vector<double> tau_basis_train_;
  std::vector<double> tau_basis_test_;
  double b0_, b1_;  ///< Current adaptive coding parameters

  // ── Current tau_0 (global treatment effect intercept) ────────────────
  // SURPRISE: tau_0 is a p_tau0-vector (1 for scalar treatment, treatment_dim
  // for multi-dimensional treatment).  After each tau_0 update we correct the
  // residual by subtracting (tau_basis_train ⊙ Δtau_0).
  std::vector<double> tau_0_;

  // ── Datasets (non-owning views into X_mu_train_ / X_tau_train_) ──────
  // SURPRISE: two separate ForestDataset objects are required — one for mu
  // (no basis; standard BART) and one for tau (mutable basis from Z / coding).
  ForestDataset dataset_mu_train_;
  ForestDataset dataset_mu_test_;
  ForestDataset dataset_tau_train_;
  ForestDataset dataset_tau_test_;

  // ── Residual ─────────────────────────────────────────────────────────
  ColumnVector residual_;

  // ── Mu-forest objects ─────────────────────────────────────────────────
  std::unique_ptr<TreeEnsemble>  active_forest_mu_;
  std::unique_ptr<ForestTracker> tracker_mu_;
  std::unique_ptr<TreePrior>     tree_prior_mu_;
  LeafNodeHomoskedasticVarianceModel leaf_var_model_mu_;

  // ── Tau-forest objects ────────────────────────────────────────────────
  std::unique_ptr<TreeEnsemble>  active_forest_tau_;
  std::unique_ptr<ForestTracker> tracker_tau_;
  std::unique_ptr<TreePrior>     tree_prior_tau_;
  LeafNodeHomoskedasticVarianceModel leaf_var_model_tau_;

  // ── Global variance model ─────────────────────────────────────────────
  GlobalHomoskedasticVarianceModel global_var_model_;

  // ── Variance forest objects ───────────────────────────────────────────
  std::unique_ptr<TreeEnsemble>  active_forest_variance_;
  std::unique_ptr<ForestTracker> variance_tracker_;
  std::unique_ptr<TreePrior>     variance_prior_;

  // ── Random effects ────────────────────────────────────────────────────
  std::vector<double>   rfx_ones_train_;
  std::vector<double>   rfx_ones_test_;
  std::vector<int32_t>  rfx_groups_train_vec_;
  std::vector<int32_t>  rfx_groups_test_vec_;
  std::unique_ptr<RandomEffectsDataset>                     rfx_dataset_train_;
  std::unique_ptr<RandomEffectsDataset>                     rfx_dataset_test_;
  std::unique_ptr<RandomEffectsTracker>                     rfx_tracker_;
  std::unique_ptr<MultivariateRegressionRandomEffectsModel> rfx_model_;

  // ── GFR snapshot state ────────────────────────────────────────────────
  std::unique_ptr<ForestContainer>        gfr_mu_fc_;
  std::unique_ptr<ForestContainer>        gfr_tau_fc_;
  std::unique_ptr<ForestContainer>        gfr_var_fc_;
  std::unique_ptr<RandomEffectsContainer> gfr_rfx_fc_;
  std::vector<double> gfr_sigma2_seeds_;
  std::vector<double> gfr_leaf_scale_mu_seeds_;
  std::vector<double> gfr_leaf_scale_tau_seeds_;
  // Adaptive coding / intercept seeds (one per GFR snapshot):
  std::vector<double> gfr_b0_seeds_;
  std::vector<double> gfr_b1_seeds_;
  std::vector<double> gfr_tau_0_seeds_;  ///< treatment_dim per snapshot
  int n_gfr_stored_ = 0;

  // ── Per-chain helpers ─────────────────────────────────────────────────
  std::unique_ptr<BCFChainState> make_chain_state_(int chain_idx,
                                                   bool alloc_chain_containers);

  void run_chain_iters_(BCFChainState& cs, int chain_idx,
                        int n_mcmc, int keep_every, int num_burnin,
                        int num_threads, BCFResult* result,
                        ForestContainer* mu_fc,
                        ForestContainer* tau_fc,
                        ForestContainer* var_fc,
                        RandomEffectsContainer* rfx_fc);

  void alloc_result_(BCFResult* result, int n_mcmc, int keep_every) const;

  // ── Internal sampling steps ───────────────────────────────────────────

  // Sample tau_0 from its conjugate posterior and subtract residual delta.
  // SURPRISE: partial residual = resid - mu(X) - tau(X)*basis — requires
  // PredictRaw calls on both active forests to reconstruct partial residual.
  void sample_tau_0_(BCFChainState& cs);

  // Sample (b0, b1) from their diagonal bivariate posterior and call
  // UpdateResidualNewBasis after updating ForestDataset basis.
  // SURPRISE: tau(X) needed without basis multiplication — use PredictRaw
  // on the tau TreeEnsemble (not ForestContainer).
  void sample_adaptive_coding_(BCFChainState& cs);
};

} // namespace StochTree

#endif // STOCHTREE_BCF_SAMPLER_H_
