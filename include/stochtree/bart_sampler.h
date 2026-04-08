/*!
 * Copyright (c) 2026 stochtree authors. All rights reserved.
 *
 * Stateful BART sampler (RFC 0005).
 *
 * BARTSampler owns all mutable sampling state and exposes a phased API:
 *
 *   BARTSampler sampler(config, data);
 *   sampler.run_gfr(n_gfr);                      // warm-start
 *   sampler.run_mcmc(n_mcmc, result, keep_every); // posterior samples
 *
 * BARTSamplerFit() is a thin free-function wrapper that creates a BARTSampler,
 * runs GFR and MCMC phases, and writes results into a caller-owned BARTResult.
 *
 * See .claude/stateful_sampler_design.md for the full design rationale.
 */
#ifndef STOCHTREE_BART_SAMPLER_H_
#define STOCHTREE_BART_SAMPLER_H_

#include <stochtree/bart.h>
#include <stochtree/container.h>
#include <stochtree/data.h>
#include <stochtree/ensemble.h>
#include <stochtree/leaf_model.h>
#include <stochtree/normal_sampler.h>
#include <stochtree/ordinal_sampler.h>
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

/*!
 * \brief Stateful BART sampler.
 *
 * Constructor does all one-time setup (standardization, prior calibration,
 * dataset construction, tree/tracker initialization).  The sampling phases
 * (GFR, burnin, MCMC) are then individual method calls, enabling incremental
 * extension of an existing run.
 *
 * Ownership model:
 *   - BARTSampler owns all mutable state (forests, trackers, residual, etc.)
 *   - Caller owns BARTData buffers (must remain live for the sampler's lifetime)
 *   - Caller owns BARTResult and passes it to run_mcmc()
 *
 */
struct ChainState;  // defined in bart_sampler.cpp

class BARTSampler {
 public:
  /*!
   * \brief Construct sampler, performing all one-time setup.
   *
   * \param config  Sampler configuration (pre-calibrated priors from caller).
   * \param data    Non-owning view into caller-managed data buffers.
   */
  BARTSampler(const BARTConfig& config, const BARTData& data);

  // Non-copyable; move would invalidate internal data pointers.
  BARTSampler(const BARTSampler&)            = delete;
  BARTSampler& operator=(const BARTSampler&) = delete;
  BARTSampler(BARTSampler&&)                 = delete;
  BARTSampler& operator=(BARTSampler&&)      = delete;

  ~BARTSampler() = default;

  /*!
   * \brief Run GFR warm-start iterations, storing snapshots for chain seeding.
   *
   * After run_gfr(), the sampler holds internal snapshots of the last
   * n_gfr forest states that run_mcmc() uses to seed each chain.
   *
   * \param n_gfr   Number of GFR iterations.  0 = skip GFR entirely.
   */
  void run_gfr(int n_gfr);

  /*!
   * \brief Run burnin + posterior MCMC, writing kept samples into result.
   *
   * Must be called after run_gfr() (or with n_gfr=0 in the constructor config).
   * Allocates result arrays here based on n_mcmc / keep_every / num_chains.
   * Can be called again on the same result to extend sampling.
   *
   * \param n_mcmc      Number of posterior draws to collect per chain.
   * \param result      Caller-owned output struct (allocated by caller).
   * \param keep_every  Thinning: keep one draw per keep_every iterations.
   */
  void run_mcmc(int n_mcmc, BARTResult* result, int keep_every = 1);

  /*!
   * \brief Expose the GFR snapshots directly as the result (GFR-only mode).
   *
   * Moves the internal GFR forest/RFX containers into result and computes
   * predictions for all n_gfr samples.  Must be called after run_gfr().
   * After this call the sampler's GFR containers are consumed (moved-from).
   *
   * \param result  Caller-owned output struct.
   */
  void run_gfr_result(BARTResult* result);

 private:
  // ── Config-derived flags (set once at construction) ─────────────────
  BARTConfig  config_;     // retained copy for run_mcmc reuse
  int n_train_, n_test_, p_, basis_dim_;
  int num_trees_, num_trees_variance_;
  bool has_test_, is_probit_, is_cloglog_, is_leaf_regression_, is_multivariate_;
  bool has_variance_forest_, has_mean_forest_, has_rfx_, has_rfx_test_;
  int  K_;                 // cloglog ordinal category count (0 when not cloglog)
  int  rfx_num_groups_, rfx_num_components_;
  int  cutpoint_grid_size_, num_features_subsample_;

  // ── Standardization ──────────────────────────────────────────────────
  double y_bar_, y_std_;
  std::vector<double> resid_vec_;  // standardized residuals (also holds category labels for cloglog)
  std::vector<int>    y_int_;      // probit: 0/1 binary labels

  // ── Calibrated prior parameters ──────────────────────────────────────
  double a_global_, b_global_;
  double a_leaf_, b_leaf_, leaf_scale_init_;
  double sigma2_init_;
  double a_forest_, b_forest_, var_leaf_init_;
  double init_val_;              // initial mean-forest leaf value

  // ── Feature & weight configuration ───────────────────────────────────
  std::vector<FeatureType> feature_types_;
  std::vector<double>      variable_weights_;
  std::vector<double>      variable_weights_variance_;
  std::vector<int>         sweep_indices_;
  std::vector<int>         variance_sweep_indices_;

  // ── RNG ───────────────────────────────────────────────────────────────
  std::mt19937 rng_;

  // ── Raw data pointers (for per-chain dataset reconstruction) ─────────
  const double* X_train_ptr_     = nullptr;
  const double* basis_train_ptr_ = nullptr;
  const double* weights_ptr_     = nullptr;

  // ── Datasets (non-owning views into caller data) ─────────────────────
  ForestDataset dataset_train_;
  ForestDataset dataset_test_;

  // ── Residual (internal Eigen copy of resid_vec_) ─────────────────────
  ColumnVector residual_;

  // ── Mean forest objects ───────────────────────────────────────────────
  std::unique_ptr<TreeEnsemble>  active_forest_;
  std::unique_ptr<ForestTracker> tracker_;
  std::unique_ptr<TreePrior>     tree_prior_;

  GlobalHomoskedasticVarianceModel   global_var_model_;
  LeafNodeHomoskedasticVarianceModel leaf_var_model_;

  // ── Variance forest objects ───────────────────────────────────────────
  std::unique_ptr<TreeEnsemble>  active_forest_variance_;
  std::unique_ptr<ForestTracker> variance_tracker_;
  std::unique_ptr<TreePrior>     variance_prior_;

  // ── Cloglog ───────────────────────────────────────────────────────────
  OrdinalSampler ordinal_sampler_;

  // ── Random effects ────────────────────────────────────────────────────
  std::vector<double>   rfx_ones_train_;
  std::vector<double>   rfx_ones_test_;
  std::vector<int32_t>  rfx_groups_train_vec_;
  std::vector<int32_t>  rfx_groups_test_vec_;
  std::unique_ptr<RandomEffectsDataset>                     rfx_dataset_train_;
  std::unique_ptr<RandomEffectsDataset>                     rfx_dataset_test_;
  std::unique_ptr<RandomEffectsTracker>                     rfx_tracker_;
  std::unique_ptr<MultivariateRegressionRandomEffectsModel> rfx_model_;

  // ── GFR snapshot state (populated by run_gfr(), consumed by run_mcmc()) ─
  // Stored in ForestContainers (one sample per GFR iteration) so that chain
  // seeding can use ReconstituteFromForest() directly.
  std::unique_ptr<ForestContainer>       gfr_mean_fc_;    // n_gfr mean forest samples
  std::unique_ptr<ForestContainer>       gfr_var_fc_;     // n_gfr variance forest samples
  std::unique_ptr<RandomEffectsContainer> gfr_rfx_fc_;   // n_gfr rfx samples
  std::vector<double> gfr_sigma2_seeds_;                  // n_gfr sigma2 scalars
  std::vector<double> gfr_leaf_scale_seeds_;              // n_gfr leaf_scale scalars
  std::vector<double> gfr_cloglog_cutpoint_seeds_;        // (K-1)*n_gfr cutpoint seeds
  int n_gfr_stored_ = 0;   // number of GFR iterations stored (0 = no GFR run yet)

  // ── Per-chain helpers ─────────────────────────────────────────────────

  // Allocate and seed per-chain mutable state from GFR snapshots (or from
  // root stumps when n_gfr = 0).  Called serially before the parallel loop.
  // alloc_chain_containers: true for multi-chain (each chain needs its own
  // ForestContainer buffers); false for single-chain (write directly to result).
  std::unique_ptr<ChainState> make_chain_state_(int chain_idx,
                                                bool alloc_chain_containers);

  // Run burnin + MCMC iterations for one chain.  Writes scalar samples and
  // y_hat_train directly into result (non-overlapping column offsets).
  // Forest/RFX samples are written into mean_fc / var_fc / rfx_fc, which are
  // either per-chain buffers (multi-chain) or the result containers directly
  // (single-chain, to avoid a redundant copy).
  void run_chain_iters_(ChainState& cs, int chain_idx,
                        int n_mcmc, int keep_every, int num_burnin,
                        int num_threads, BARTResult* result,
                        ForestContainer& mean_fc,
                        ForestContainer* var_fc,
                        RandomEffectsContainer* rfx_fc);

  void alloc_result_(BARTResult* result, int n_mcmc, int keep_every) const;
};

// ── Free-function entry point ─────────────────────────────────────────────

/*!
 * \brief Fit a BART model using the stateful BARTSampler.
 *
 * Creates a BARTSampler, calls run_gfr() and run_mcmc(), then writes
 * predictions and metadata into *result.
 */
void BARTSamplerFit(BARTResult*        result_ptr,
                    const BARTConfig&  config,
                    const BARTData&    data,
                    const std::string& previous_model_json = "");

} // namespace StochTree

#endif // STOCHTREE_BART_SAMPLER_H_
