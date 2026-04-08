/*!
 * Copyright (c) 2026 stochtree authors. All rights reserved.
 *
 * High-level C++ dispatch API for BCF.
 *
 * BCFSamplerFit implements the full MCMC sampling loop in C++.
 * The R and Python interfaces become thin marshaling layers that:
 *   1. Validate and preprocess inputs
 *   2. Populate BCFData / BCFConfig structs
 *   3. Call BCFSamplerFit
 *   4. Unpack BCFResult into language-native objects
 *
 * Design note: BCF is structurally more complex than BART in several ways —
 * see bcf_sampler.cpp for a full enumeration of surprises vs BART.
 */
#ifndef STOCHTREE_BCF_H_
#define STOCHTREE_BCF_H_

#include <stochtree/bart.h>       // LinkFunction, LeafModel, RFXModelSpec reused
#include <stochtree/container.h>
#include <stochtree/random_effects.h>

#include <memory>
#include <string>
#include <vector>

namespace StochTree {

// ── Per-forest configuration ───────────────────────────────────────────────

/*!
 * \brief Tree prior and sampling parameters for one BCF forest.
 *
 * Used for both the prognostic (mu) and treatment-effect (tau) forests.
 * Mirrors the per-forest parameter lists in the R BCF API.
 */
struct BCFForestConfig {
  // ── Tree prior ────────────────────────────────────────────────────
  int    num_trees        = 200;
  double alpha            = 0.95;
  double beta             = 2.0;
  int    min_samples_leaf = 5;
  int    max_depth        = 10;

  // ── Leaf scale prior IG(a_leaf, b_leaf) ───────────────────────────
  double a_leaf    = 3.0;
  double b_leaf    = -1.0;    ///< <0 = calibrate from data
  double leaf_scale = -1.0;  ///< Initial leaf scale; <0 = auto

  // ── Leaf scale sampling flag ──────────────────────────────────────
  bool sample_sigma2_leaf = true;

  // ── Variable selection ────────────────────────────────────────────
  // Empty = uniform 1/p_forest.  Length must equal the forest's covariate
  // dimension (p+1 for mu if propensity included; p for tau) if non-empty.
  std::vector<double> variable_weights;

  // ── Cutpoint / subsample ──────────────────────────────────────────
  int cutpoint_grid_size      = 100;
  int num_features_subsample  = -1;  ///< -1 = all features
};

// ── Top-level config ───────────────────────────────────────────────────────

/*!
 * \brief Configuration for BCFSamplerFit.
 *
 * Separates per-forest parameters into mu_forest / tau_forest sub-structs.
 * Global parameters (sampler counts, link, variance, RFX, etc.) are flat.
 */
struct BCFConfig {
  // ── Sampler counts ────────────────────────────────────────────────
  int num_gfr     = 5;
  int num_burnin  = 0;
  int num_mcmc    = 100;
  int num_chains  = 1;
  int keep_every  = 1;
  bool keep_gfr    = false;
  bool keep_burnin = false;
  bool profile_phases = false;

  // ── Global variance prior IG(a_global, b_global) ──────────────────
  double a_global    = 0.0;
  double b_global    = 0.0;
  double sigma2_init = -1.0;  ///< <0 = auto from resid variance
  bool   sample_sigma2_global = true;

  // ── Link function (identity / probit / cloglog) ───────────────────
  LinkFunction link_function = LinkFunction::Identity;

  // ── Per-forest configs ────────────────────────────────────────────
  BCFForestConfig mu_forest;   ///< Prognostic forest defaults: 250 trees
  BCFForestConfig tau_forest;  ///< Treatment-effect forest defaults: 100 trees

  // ── Propensity covariate routing ──────────────────────────────────
  // "mu"   = append π̂ to mu-forest covariates only (default BCF)
  // "tau"  = append π̂ to tau-forest covariates only
  // "both" = append π̂ to both forests
  // "none" = do not include π̂ in any forest
  std::string propensity_covariate = "mu";

  // ── Adaptive coding ───────────────────────────────────────────────
  // When true and Z is binary, the treatment basis becomes b0*(1-Z) + b1*Z
  // and (b0, b1) are sampled from independent conjugate posteriors.
  // Only valid for binary treatment; caller must validate.
  bool   adaptive_coding  = false;
  double b0_init          = -0.5;  ///< Initial control coding
  double b1_init          =  0.5;  ///< Initial treated coding
  double coding_prior_var =  0.5;  ///< N(0, coding_prior_var) prior on b0/b1

  // ── Global treatment effect intercept (tau_0) ─────────────────────
  // When true, sample a scalar (or vector for multi-dim Z) tau_0 so that
  // the full CATE is  tau_0 + τ(X).  Basis used: current tau_basis_train.
  bool   sample_intercept = true;
  double tau_0_prior_var  = -1.0;  ///< <0 = auto-calibrate from resid variance

  // ── Variance forest ───────────────────────────────────────────────
  bool   include_variance_forest       = false;
  int    num_trees_variance            = 0;
  double alpha_variance                = 0.95;
  double beta_variance                 = 2.0;
  int    min_samples_leaf_variance     = 5;
  int    max_depth_variance            = 10;
  double a_forest                      = -1.0;
  double b_forest                      = -1.0;
  double variance_forest_leaf_init     = -1.0;
  std::vector<double> variable_weights_variance;

  // ── Random effects ────────────────────────────────────────────────
  RFXModelSpec rfx_model_spec       = RFXModelSpec::None;
  int    rfx_num_components         = 1;
  double rfx_alpha_init             = 1.0;
  double rfx_xi_init                = 1.0;
  double rfx_sigma_alpha_init       = 1.0;
  double rfx_sigma_xi_init          = 1.0;
  double rfx_variance_prior_shape   = 1.0;
  double rfx_variance_prior_scale   = 1.0;

  // ── Standardization ───────────────────────────────────────────────
  bool standardize = true;

  // ── Misc ──────────────────────────────────────────────────────────
  int num_threads  = -1;
  int random_seed  = -1;
};

// ── Data struct ────────────────────────────────────────────────────────────

/*!
 * \brief Training and test data for BCFSamplerFit.
 *
 * Covariate matrices are column-major (Fortran order).
 * Z_train / Z_test may be binary (1 column) or continuous / multi-dimensional
 * (treatment_dim > 1 disables adaptive_coding).
 * The caller owns all memory and must keep it live for the sampler's lifetime.
 *
 * Internally BCFSampler builds one augmented covariate matrix [X | π̂]
 * (p_aug = p + 1 when propensity is included) and two ForestDataset objects
 * that both reference it.  The forests are distinguished by variable_weights:
 * the pi_hat column weight is zeroed out for whichever forest should not use it.
 * Two separate ForestTrackers are still required (each forest has its own
 * split decisions), and the tau-forest ForestDataset carries the treatment basis
 * while the mu-forest ForestDataset does not.
 */
struct BCFData {
  // ── Training covariates (required) ────────────────────────────────
  const double* X_train    = nullptr;  ///< Column-major, n_train × p
  int           n_train    = 0;
  int           p          = 0;

  // ── Outcome (required) ────────────────────────────────────────────
  const double* y_train    = nullptr;  ///< Length n_train

  // ── Treatment (required) ─────────────────────────────────────────
  // Binary Z → 1-column, length n_train.  Multi-dimensional treatment →
  // column-major, n_train × treatment_dim.
  const double* Z_train    = nullptr;
  int           treatment_dim = 1;

  // ── Propensity scores (optional) ─────────────────────────────────
  // When non-null and propensity_covariate != "none", appended to mu/tau
  // forest covariate matrices.  Length n_train.
  const double* pi_hat_train = nullptr;

  // ── Feature types (optional) ──────────────────────────────────────
  // nullptr = all numeric.  Length p if non-null.
  const int* feature_types   = nullptr;

  // ── Observation weights (optional) ───────────────────────────────
  const double* weights      = nullptr;  ///< Length n_train; nullptr = uniform

  // ── Test covariates (optional) ────────────────────────────────────
  const double* X_test       = nullptr;  ///< Column-major, n_test × p
  const double* Z_test       = nullptr;  ///< n_test × treatment_dim
  const double* pi_hat_test  = nullptr;  ///< Length n_test
  int           n_test       = 0;

  // ── Random effects (optional) ─────────────────────────────────────
  // Required when rfx_model_spec != None.
  const int*    rfx_groups      = nullptr;
  const double* rfx_basis_train = nullptr;
  const int*    rfx_groups_test  = nullptr;
  const double* rfx_basis_test  = nullptr;
};

// ── Result struct ──────────────────────────────────────────────────────────

/*!
 * \brief Results from BCFSamplerFit.
 *
 * All prediction arrays are on the original (un-standardized) scale.
 * Column layout (n_obs × num_total_samples) follows the same convention as
 * BARTResult: [GFR samples] [chain 0] [chain 1] ...
 *
 * SURPRISE vs BART: BCF result carries additional per-sample arrays for
 *   tau-specific outputs (tau_hat, tau_0, b0, b1) that have no BART analogue.
 */
struct BCFResult {
  // ── Predictions ───────────────────────────────────────────────────
  std::vector<double> y_hat_train;         ///< n_train × num_total_samples
  std::vector<double> y_hat_test;          ///< n_test × num_total_samples
  std::vector<double> mu_hat_train;        ///< n_train × num_total_samples (prognostic component)
  std::vector<double> mu_hat_test;         ///< n_test  × num_total_samples
  std::vector<double> tau_hat_train;       ///< n_train × num_total_samples (treatment effect component)
  std::vector<double> tau_hat_test;        ///< n_test  × num_total_samples
  std::vector<double> sigma2_x_hat_train;  ///< n_train × num_total_samples (variance forest only)
  std::vector<double> sigma2_x_hat_test;   ///< n_test  × num_total_samples

  // ── Global posterior samples ──────────────────────────────────────
  std::vector<double> sigma2_global_samples;   ///< num_total_samples
  std::vector<double> leaf_scale_mu_samples;   ///< num_total_samples
  std::vector<double> leaf_scale_tau_samples;  ///< num_total_samples

  // ── Treatment effect intercept ────────────────────────────────────
  // Shape: treatment_dim × num_total_samples (1 × S for scalar treatment)
  std::vector<double> tau_0_samples;

  // ── Adaptive coding ───────────────────────────────────────────────
  std::vector<double> b0_samples;  ///< num_total_samples (empty if !adaptive_coding)
  std::vector<double> b1_samples;  ///< num_total_samples

  // ── Cloglog cutpoints (empty if link != Cloglog) ──────────────────
  std::vector<double> cloglog_cutpoint_samples;

  // ── Random effects ────────────────────────────────────────────────
  std::unique_ptr<RandomEffectsContainer> rfx_container;
  std::vector<int32_t> rfx_group_ids;

  // ── Fitted forests ────────────────────────────────────────────────
  std::unique_ptr<ForestContainer> mu_forest_container;
  std::unique_ptr<ForestContainer> tau_forest_container;
  std::unique_ptr<ForestContainer> variance_forest_container;

  // ── Metadata ──────────────────────────────────────────────────────
  int num_total_samples  = 0;
  int num_chains         = 1;
  int n_train            = 0;
  int n_test             = 0;
  int treatment_dim      = 1;
  int rfx_num_groups     = 0;
  int rfx_num_components = 0;
  double y_bar           = 0.0;
  double y_std           = 1.0;
};

// ── Free-function entry point ─────────────────────────────────────────────

/*!
 * \brief Fit a BCF model using the stateful BCFSampler.
 *
 * Creates a BCFSampler, calls run_gfr() and run_mcmc(), then writes
 * predictions and metadata into *result.
 */
void BCFSamplerFit(BCFResult*         result_ptr,
                   const BCFConfig&   config,
                   const BCFData&     data,
                   const std::string& previous_model_json = "");

} // namespace StochTree

#endif // STOCHTREE_BCF_H_
