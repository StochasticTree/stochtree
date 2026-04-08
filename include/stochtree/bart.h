/*!
 * Copyright (c) 2024 stochtree authors. All rights reserved.
 *
 * High-level C++ dispatch API for BART and BCF.
 *
 * BARTSamplerFit implements the full MCMC sampling loop in C++.
 * The R and Python interfaces become thin marshaling layers that:
 *   1. Validate and preprocess inputs
 *   2. Populate BARTData / BARTConfig structs
 *   3. Call BARTSamplerFit
 *   4. Unpack BARTResult into language-native objects
 *
 * See RFC 0005 (stateful_sampler_design.md) for the full design.
 */
#ifndef STOCHTREE_BART_H_
#define STOCHTREE_BART_H_

#include <stochtree/container.h>
#include <stochtree/random_effects.h>
#include <memory>
#include <string>
#include <vector>

namespace StochTree {

// ── Enumerations ───────────────────────────────────────────────────────────

enum class LinkFunction {
  Identity = 0,  ///< Continuous outcome, standard BART
  Probit   = 1,  ///< Binary outcome, Albert-Chib augmentation
  Cloglog  = 2,  ///< Binary or ordinal outcome, OrdinalSampler augmentation
};

enum class LeafModel {
  Constant             = 0,  ///< Standard BART: constant leaf value
  UnivariateRegression = 1,  ///< Linear leaf: 1-column basis
  MultivariateRegression = 2,///< Linear leaf: multi-column basis (leaf scale sampling disabled)
};

enum class RFXModelSpec {
  None          = 0,
  InterceptOnly = 1,  ///< Basis = column of ones; rfx_basis_train not required
  Custom        = 2,  ///< User supplies rfx_basis_train
};

// ── Config struct ──────────────────────────────────────────────────────────

/*!
 * \brief Configuration for BARTSamplerFit.
 *
 * Fields for disabled features (e.g. cloglog_* when link_function == Identity)
 * are ignored. The caller (R or Python marshaling layer) is responsible for:
 *   - calibrating a_global / b_global from nu / lamb
 *   - calibrating b_leaf / leaf_scale from data range
 *   - computing sigma2_init from the standardized residual variance
 * BARTSamplerFit uses whatever values are provided directly.
 */
struct BARTConfig {
  // ── Sampler counts ────────────────────────────────────────────────
  int num_trees   = 200;
  int num_gfr     = 5;
  int num_burnin  = 0;
  int num_mcmc    = 100;
  int num_chains  = 1;
  int keep_every  = 1;    ///< Thinning: retain one MCMC draw per keep_every iters
  bool keep_gfr    = false;
  bool keep_burnin = false;
  bool profile_phases = false;  ///< When true, run_mcmc() prints per-phase wall times to stderr.

  // ── Tree prior ────────────────────────────────────────────────────
  double alpha         = 0.95;
  double beta          = 2.0;
  int min_samples_leaf = 5;
  int max_depth        = 10;

  // ── Global variance prior IG(a_global, b_global) ──────────────────
  // Defaults of 0 give a flat (non-informative) prior.
  double a_global = 0.0;
  double b_global = 0.0;

  // ── Leaf scale prior IG(a_leaf, b_leaf) ───────────────────────────
  double a_leaf    = 3.0;
  double b_leaf    = -1.0;   ///< <0 = caller must set; no auto-calibration in BARTSamplerFit
  double leaf_scale = -1.0;  ///< Initial leaf scale; <0 = auto from resid_var / num_trees

  // ── Initial global variance ───────────────────────────────────────
  double sigma2_init = -1.0; ///< <0 = auto from resid_var

  // ── Variance sampling flags ───────────────────────────────────────
  bool sample_sigma2_global = true;
  bool sample_sigma2_leaf   = true;

  // ── Model type ────────────────────────────────────────────────────
  LinkFunction link_function = LinkFunction::Identity;
  LeafModel    leaf_model    = LeafModel::Constant;

  // ── Variable selection weights ────────────────────────────────────
  // Empty = uniform 1/p. Length must equal p if non-empty.
  std::vector<double> variable_weights_mean;

  // ── Variance forest ───────────────────────────────────────────────
  bool include_variance_forest    = false;
  int  num_trees_variance         = 0;
  double alpha_variance           = 0.95;
  double beta_variance            = 2.0;
  int  min_samples_leaf_variance  = 5;
  int  max_depth_variance         = 10;
  double a_forest                 = -1.0;  ///< IG shape for variance forest leaf; <0 = calibrate
  double b_forest                 = -1.0;  ///< IG scale for variance forest leaf; <0 = calibrate
  double variance_forest_leaf_init = -1.0;
  std::vector<double> variable_weights_variance;

  // ── Cloglog / ordinal ─────────────────────────────────────────────
  // Ignored when link_function != Cloglog.
  int    cloglog_num_categories = 2;   ///< 2 = binary cloglog; >2 = ordinal
  double cloglog_forest_shape   = 2.0;
  double cloglog_forest_rate    = 2.0;
  double cloglog_cutpoint_0     = 0.0;

  // ── Random effects ────────────────────────────────────────────────
  RFXModelSpec rfx_model_spec       = RFXModelSpec::None;
  int    rfx_num_components         = 1;     ///< Ignored for InterceptOnly (forced to 1)
  double rfx_alpha_init             = 1.0;   ///< Initial working parameter (scalar, all components)
  double rfx_xi_init                = 1.0;   ///< Initial group parameters (scalar, all groups/components)
  double rfx_sigma_alpha_init       = 1.0;   ///< Initial working parameter variance (diagonal)
  double rfx_sigma_xi_init          = 1.0;   ///< Initial group parameter variance (diagonal)
  double rfx_variance_prior_shape   = 1.0;   ///< IG prior shape for σ²_ξ
  double rfx_variance_prior_scale   = 1.0;   ///< IG prior scale for σ²_ξ

  // ── Standardization ───────────────────────────────────────────────
  bool standardize = true;  ///< Center and scale y; un-standardize predictions before return

  // ── Misc ──────────────────────────────────────────────────────────
  int cutpoint_grid_size  = 100; ///< Max cutpoints considered per feature per GFR iter
  int num_threads         = -1;  ///< OpenMP threads; -1 = use default
  int random_seed         = -1;  ///< -1 = non-deterministic
};

// ── Data struct ────────────────────────────────────────────────────────────

/*!
 * \brief Training and test data for BARTSamplerFit.
 *
 * All matrices are column-major (Fortran order), consistent with Eigen's
 * default. Inputs must already be preprocessed (no DataFrames or factors).
 * The caller owns all memory and must keep it live for the duration of BARTSamplerFit.
 *
 * Optional fields are nullptr when not used.
 */
struct BARTData {
  // ── Training covariates (required) ───────────────────────────────
  const double* X_train = nullptr; ///< Column-major, shape: n_train × p
  int n_train = 0;
  int p = 0;

  // ── Outcome (required) ────────────────────────────────────────────
  const double* y_train = nullptr; ///< Length n_train

  // ── Test covariates (optional) ────────────────────────────────────
  const double* X_test = nullptr;  ///< Column-major, shape: n_test × p
  int n_test = 0;

  // ── Feature types (optional) ──────────────────────────────────────
  // nullptr = all kNumeric (0). Length p if non-null.
  // Values: 0 = numeric, 1 = ordered categorical, 2 = unordered categorical
  const int* feature_types = nullptr;

  // ── Leaf regression basis (optional) ─────────────────────────────
  // Required when leaf_model != Constant.
  const double* basis_train = nullptr; ///< Column-major, n_train × basis_dim
  const double* basis_test  = nullptr; ///< Column-major, n_test × basis_dim
  int basis_dim = 0;

  // ── Observation weights (optional) ───────────────────────────────
  const double* weights = nullptr; ///< Length n_train; nullptr = uniform

  // ── Random effects (optional) ─────────────────────────────────────
  // Required when rfx_model_spec != None.
  const int*    rfx_groups      = nullptr; ///< Length n_train; arbitrary integer group labels
  const double* rfx_basis_train = nullptr; ///< Column-major, n_train × rfx_num_components (Custom only)
  const int*    rfx_groups_test  = nullptr; ///< Length n_test; test group labels (subset of train labels)
  const double* rfx_basis_test  = nullptr; ///< Column-major, n_test × rfx_num_components (Custom only)
};

// ── Result struct ──────────────────────────────────────────────────────────

/*!
 * \brief Results from BARTSamplerFit.
 *
 * Sample array layout (column-major):
 *   shape: n_obs × num_total_samples
 * where:
 *   num_total_samples = num_stored_gfr + num_chains × num_mcmc
 *   num_stored_gfr    = num_gfr if keep_gfr (or num_mcmc==0), else 0
 *
 * Column ordering:
 *   [GFR samples 0..num_gfr-1]  (only present when keep_gfr=true or num_mcmc==0)
 *   [chain 0: MCMC]
 *   [chain 1: MCMC]
 *   ...
 *
 * Scalar sample arrays (sigma2, leaf_scale) follow the same column order
 * with length num_total_samples.
 *
 * Predictions are on the original (un-standardized) scale.
 */
struct BARTResult {
  // ── Predictions ───────────────────────────────────────────────────
  std::vector<double> y_hat_train;         ///< n_train × num_total_samples
  std::vector<double> y_hat_test;          ///< n_test × num_total_samples (empty if n_test=0)
  std::vector<double> sigma2_x_hat_train;  ///< n_train × num_total_samples (variance forest only)
  std::vector<double> sigma2_x_hat_test;   ///< n_test × num_total_samples  (variance forest only)

  // ── Posterior samples (empty if not sampled) ──────────────────────
  std::vector<double> sigma2_global_samples; ///< num_total_samples (original scale)
  std::vector<double> leaf_scale_samples;    ///< num_total_samples

  // ── Cloglog cutpoints (empty if link != Cloglog) ──────────────────
  // shape: (num_categories - 1) × num_total_samples
  std::vector<double> cloglog_cutpoint_samples;

  // ── Random effects (null if rfx_model_spec == None) ──────────────
  // Container holds alpha_, xi_, beta_, sigma_xi_ sample arrays.
  // Analogous to forest_container for tree ensembles.
  std::unique_ptr<RandomEffectsContainer> rfx_container;
  std::vector<int32_t> rfx_group_ids;  ///< Sorted unique training group labels

  // ── Fitted forests ───────────────────────────────────────────────
  std::unique_ptr<ForestContainer> forest_container;          ///< Mean forest samples
  std::unique_ptr<ForestContainer> variance_forest_container; ///< Variance forest samples (empty if not used)

  // ── Metadata ─────────────────────────────────────────────────────
  int num_total_samples  = 0;
  int num_chains         = 1;
  int n_train            = 0;
  int n_test             = 0;
  int rfx_num_groups     = 0;   ///< 0 when no RFX
  int rfx_num_components = 0;   ///< 0 when no RFX
  double y_bar           = 0.0; ///< Stored for un-standardization in predict
  double y_std           = 1.0;
};

} // namespace StochTree

#endif // STOCHTREE_BART_H_
