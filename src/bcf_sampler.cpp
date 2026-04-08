/*!
 * Copyright (c) 2026 stochtree authors. All rights reserved.
 *
 * BCFSampler implementation.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * SURPRISES vs BARTSampler
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * 1. Two ForestDatasets sharing one covariate matrix.
 *    BCF builds one augmented matrix [X | π̂] (p_aug_ = p+1 columns when
 *    propensity is included) and two ForestDatasets that both reference it.
 *    The pi_hat column is excluded from whichever forest should not split on
 *    it via a zero entry in that forest's variable_weights_. Two separate
 *    ForestTrackers are still required because each forest makes its own
 *    split decisions.
 *
 * 2. Mutable treatment basis.
 *    The tau-forest basis starts as Z_train (or b0*(1-Z) + b1*Z for
 *    adaptive_coding) and is updated every MCMC iteration:
 *      a. ForestDataset::AddBasis/UpdateBasis — stores new basis.
 *      b. UpdateResidualNewBasis(*tracker_tau, dataset_tau_train,
 *            residual, active_forest_tau.get())
 *         re-derives all tree leaf predictions under the new basis and
 *         patches the residual accordingly.
 *
 * 3. tau_0 residual correction after basis change.
 *    After UpdateResidualNewBasis (which handles only the τ(X)*Δbasis
 *    component), the tau_0 contribution must be separately corrected:
 *      residual[i] -= (new_basis[i] - old_basis[i]) * tau_0
 *
 * 4. PredictRaw needed for adaptive coding sufficient statistics.
 *    Adaptive coding regresses (y - μ(X)) on [τ(X)·(1-Z), τ(X)·Z].
 *    TreeEnsemble::PredictRaw gives τ(X) without basis multiplication.
 *
 * 5. Sampling order per MCMC iteration:
 *    mu-forest → sigma2 → leaf_scale_mu → tau_0 → tau-forest →
 *    leaf_scale_tau → adaptive_coding + UpdateResidualNewBasis →
 *    variance_forest → sigma2 (again if heteroskedastic) → RFX
 *    This matches the R BCF sampling order.
 *
 * 6. tau_0 dimensions.
 *    Scalar treatment → tau_0 is length 1, SampleBayesLinReg1D used.
 *    Multi-dimensional treatment → tau_0 is length treatment_dim,
 *    SampleBayesLinRegMulti used.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include <stochtree/bcf_sampler.h>
#include <stochtree/linear_regression.h>
#include <stochtree/leaf_model.h>
#include <stochtree/normal_sampler.h>
#include <stochtree/tree_sampler.h>
#include <stochtree/variance_model.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace StochTree {

// ── BCFChainState ────────────────────────────────────────────────────────

struct BCFChainState {
  // Per-chain forests
  std::unique_ptr<TreeEnsemble>  forest_mu;
  std::unique_ptr<ForestTracker> tracker_mu;
  std::unique_ptr<TreeEnsemble>  forest_tau;
  std::unique_ptr<ForestTracker> tracker_tau;
  std::unique_ptr<TreeEnsemble>  forest_var;
  std::unique_ptr<ForestTracker> tracker_var;

  // Per-chain datasets (each chain needs its own tau dataset so the basis
  // can be mutated independently when multi-chain)
  ForestDataset dataset_mu;
  ForestDataset dataset_tau;

  // Per-chain mutable scalars
  double sigma2        = 0.0;
  double leaf_scale_mu = 0.0;
  double leaf_scale_tau = 0.0;

  // Adaptive coding / intercept state (per-chain so chains are independent)
  double b0 = -0.5, b1 = 0.5;
  std::vector<double> tau_0;    // length treatment_dim
  std::vector<double> tau_basis; // length n_train * treatment_dim

  // Per-chain residual
  ColumnVector residual;

  // Per-chain RFX (future extension)
  // std::unique_ptr<RandomEffectsTracker> rfx_tracker;

  // Per-chain output containers (multi-chain only)
  std::unique_ptr<ForestContainer> chain_mu_fc;
  std::unique_ptr<ForestContainer> chain_tau_fc;
  std::unique_ptr<ForestContainer> chain_var_fc;

  // Per-chain RNG
  std::mt19937 rng;
};

// ══════════════════════════════════════════════════════════════════════════
// Constructor
// ══════════════════════════════════════════════════════════════════════════

BCFSampler::BCFSampler(const BCFConfig& config, const BCFData& data)
    : config_(config),
      n_train_(data.n_train),
      n_test_(data.n_test),
      p_(data.p),
      treatment_dim_(data.treatment_dim),
      has_test_(data.n_test > 0),
      has_rfx_(false),            // RFX deferred
      has_rfx_test_(false),
      adaptive_coding_(config.adaptive_coding && data.treatment_dim == 1),
      sample_intercept_(config.sample_intercept),
      has_variance_forest_(config.include_variance_forest &&
                           config.num_trees_variance > 0),
      rfx_num_groups_(0),
      rfx_num_components_(0),
      num_trees_mu_(config.mu_forest.num_trees),
      num_trees_tau_(config.tau_forest.num_trees),
      Z_train_ptr_(data.Z_train)
{
  // ── STEP 1: Augmented covariate dimension ───────────────────────────
  bool has_propensity = data.pi_hat_train != nullptr &&
                        config.propensity_covariate != "none";
  p_aug_ = p_ + (has_propensity ? 1 : 0);

  // ── STEP 2: Build one owned augmented covariate copy ────────────────
  X_aug_train_.resize(n_train_ * p_aug_);
  std::copy(data.X_train, data.X_train + n_train_ * p_, X_aug_train_.data());
  if (has_propensity) {
    std::copy(data.pi_hat_train, data.pi_hat_train + n_train_,
              X_aug_train_.data() + n_train_ * p_);
  }
  if (has_test_) {
    X_aug_test_.resize(n_test_ * p_aug_);
    std::copy(data.X_test, data.X_test + n_test_ * p_, X_aug_test_.data());
    if (has_propensity && data.pi_hat_test != nullptr) {
      std::copy(data.pi_hat_test, data.pi_hat_test + n_test_,
                X_aug_test_.data() + n_test_ * p_);
    }
  }

  // ── STEP 3: Variable weights ─────────────────────────────────────────
  // Both forests use p_aug_ columns; the pi_hat column gets weight 0 in
  // whichever forest should not split on it.
  bool pi_in_mu  = (config.propensity_covariate == "mu"  ||
                    config.propensity_covariate == "both");
  bool pi_in_tau = (config.propensity_covariate == "tau" ||
                    config.propensity_covariate == "both");

  auto build_weights = [&](const std::vector<double>& user_weights,
                           bool pi_allowed) -> std::vector<double> {
    std::vector<double> w(p_aug_, 1.0 / p_);  // uniform over X columns
    if (!user_weights.empty()) {
      // User supplied weights for X columns (length p_)
      for (int j = 0; j < p_; j++) w[j] = user_weights[j];
    }
    if (has_propensity) {
      w[p_] = pi_allowed ? w[0] : 0.0;  // same weight as first X column, or 0
    }
    // Normalise so sum = 1 for the sampler
    double s = 0.0;
    for (double v : w) s += v;
    if (s > 0) for (double& v : w) v /= s;
    return w;
  };

  variable_weights_mu_  = build_weights(config.mu_forest.variable_weights,  pi_in_mu);
  variable_weights_tau_ = build_weights(config.tau_forest.variable_weights, pi_in_tau);

  if (has_variance_forest_) {
    variable_weights_variance_ = build_weights(
        config.variable_weights_variance, /*pi_allowed=*/false);
  }

  // Sweep indices: 0..num_trees-1
  sweep_indices_mu_.resize(num_trees_mu_);
  std::iota(sweep_indices_mu_.begin(), sweep_indices_mu_.end(), 0);
  sweep_indices_tau_.resize(num_trees_tau_);
  std::iota(sweep_indices_tau_.begin(), sweep_indices_tau_.end(), 0);
  if (has_variance_forest_) {
    sweep_indices_variance_.resize(config.num_trees_variance);
    std::iota(sweep_indices_variance_.begin(), sweep_indices_variance_.end(), 0);
  }

  // -1 means "all features"; clamp to p_aug_ so the tree sampler receives a valid count
  num_features_subsample_mu_  = (config.mu_forest.num_features_subsample  > 0)
                                    ? config.mu_forest.num_features_subsample  : p_aug_;
  num_features_subsample_tau_ = (config.tau_forest.num_features_subsample > 0)
                                    ? config.tau_forest.num_features_subsample : p_aug_;

  // ── STEP 4: Build initial treatment basis ────────────────────────────
  b0_ = config.b0_init;
  b1_ = config.b1_init;
  tau_basis_train_.resize(n_train_ * treatment_dim_);
  if (adaptive_coding_) {
    for (int i = 0; i < n_train_; i++) {
      tau_basis_train_[i] = (1.0 - Z_train_ptr_[i]) * b0_ +
                             Z_train_ptr_[i]         * b1_;
    }
  } else {
    std::copy(Z_train_ptr_, Z_train_ptr_ + n_train_ * treatment_dim_,
              tau_basis_train_.data());
  }
  if (has_test_) {
    tau_basis_test_.resize(n_test_ * treatment_dim_);
    if (adaptive_coding_) {
      for (int i = 0; i < n_test_; i++) {
        tau_basis_test_[i] = (1.0 - data.Z_test[i]) * b0_ +
                              data.Z_test[i]         * b1_;
      }
    } else {
      std::copy(data.Z_test, data.Z_test + n_test_ * treatment_dim_,
                tau_basis_test_.data());
    }
  }

  // ── STEP 5: Standardize outcome ──────────────────────────────────────
  if (config.standardize) {
    double sum = 0.0;
    for (int i = 0; i < n_train_; i++) sum += data.y_train[i];
    y_bar_ = sum / n_train_;
    double ss = 0.0;
    for (int i = 0; i < n_train_; i++) {
      double d = data.y_train[i] - y_bar_;
      ss += d * d;
    }
    y_std_ = std::sqrt(ss / n_train_);
    if (y_std_ < 1e-10) y_std_ = 1.0;
  } else {
    y_bar_ = 0.0;
    y_std_ = 1.0;
  }
  resid_vec_.resize(n_train_);
  for (int i = 0; i < n_train_; i++)
    resid_vec_[i] = (data.y_train[i] - y_bar_) / y_std_;

  // ── STEP 6: Calibrate priors ──────────────────────────────────────────
  a_global_   = config.a_global;
  b_global_   = config.b_global;
  sigma2_init_ = config.sigma2_init > 0.0 ? config.sigma2_init : 1.0;

  a_leaf_mu_  = config.mu_forest.a_leaf;
  b_leaf_mu_  = config.mu_forest.b_leaf > 0.0
                    ? config.mu_forest.b_leaf
                    : 0.5 / num_trees_mu_;
  leaf_scale_mu_init_ = config.mu_forest.leaf_scale > 0.0
                            ? config.mu_forest.leaf_scale
                            : 1.0 / std::sqrt(num_trees_mu_);

  a_leaf_tau_  = config.tau_forest.a_leaf;
  b_leaf_tau_  = config.tau_forest.b_leaf > 0.0
                     ? config.tau_forest.b_leaf
                     : 0.5 / num_trees_tau_;
  leaf_scale_tau_init_ = config.tau_forest.leaf_scale > 0.0
                             ? config.tau_forest.leaf_scale
                             : 1.0 / std::sqrt(num_trees_tau_);

  tau_0_prior_var_ = config.tau_0_prior_var > 0.0 ? config.tau_0_prior_var : 1.0;
  tau_0_.assign(treatment_dim_, 0.0);

  a_forest_    = config.a_forest;
  b_forest_    = config.b_forest;
  var_leaf_init_ = config.variance_forest_leaf_init > 0.0
                       ? config.variance_forest_leaf_init : 0.0;
  // init_val_mu_: prior mean of mu; approximate as y_bar_ / y_std_ = 0 after standardization
  init_val_mu_  = 0.0;
  init_val_tau_ = 0.0;

  // ── STEP 7: Feature types (length p_aug_) ─────────────────────────────
  feature_types_.assign(p_aug_, FeatureType::kNumeric);
  if (data.feature_types != nullptr) {
    for (int j = 0; j < p_; j++)
      feature_types_[j] = static_cast<FeatureType>(data.feature_types[j]);
  }

  // ── STEP 8: Build ForestDatasets ──────────────────────────────────────
  dataset_mu_train_.AddCovariates(X_aug_train_.data(), n_train_, p_aug_, false);
  dataset_tau_train_.AddCovariates(X_aug_train_.data(), n_train_, p_aug_, false);
  dataset_tau_train_.AddBasis(tau_basis_train_.data(), n_train_,
                              treatment_dim_, false);
  if (data.weights != nullptr) {
    dataset_mu_train_.AddVarianceWeights(
        const_cast<double*>(data.weights), n_train_);
    dataset_tau_train_.AddVarianceWeights(
        const_cast<double*>(data.weights), n_train_);
  }
  if (has_test_) {
    dataset_mu_test_.AddCovariates(X_aug_test_.data(), n_test_, p_aug_, false);
    dataset_tau_test_.AddCovariates(X_aug_test_.data(), n_test_, p_aug_, false);
    dataset_tau_test_.AddBasis(tau_basis_test_.data(), n_test_,
                               treatment_dim_, false);
  }

  // ── STEP 9: Residual ──────────────────────────────────────────────────
  residual_ = ColumnVector(resid_vec_.data(), n_train_);

  // ── STEP 10: Active forests, trackers, priors ─────────────────────────
  active_forest_mu_ = std::make_unique<TreeEnsemble>(
      num_trees_mu_, 1, /*leaf_const=*/true, false);
  tracker_mu_ = std::make_unique<ForestTracker>(
      dataset_mu_train_.GetCovariates(), feature_types_, num_trees_mu_, n_train_);
  tree_prior_mu_ = std::make_unique<TreePrior>(
      config.mu_forest.alpha, config.mu_forest.beta,
      config.mu_forest.min_samples_leaf, config.mu_forest.max_depth);

  // tau-forest: leaf regression on treatment basis (treatment_dim columns)
  active_forest_tau_ = std::make_unique<TreeEnsemble>(
      num_trees_tau_, treatment_dim_, /*leaf_const=*/false, false);
  tracker_tau_ = std::make_unique<ForestTracker>(
      dataset_tau_train_.GetCovariates(), feature_types_, num_trees_tau_, n_train_);
  tree_prior_tau_ = std::make_unique<TreePrior>(
      config.tau_forest.alpha, config.tau_forest.beta,
      config.tau_forest.min_samples_leaf, config.tau_forest.max_depth);

  if (has_variance_forest_) {
    active_forest_variance_ = std::make_unique<TreeEnsemble>(
        config.num_trees_variance, 1, true, /*is_exponentiated=*/true);
    variance_tracker_ = std::make_unique<ForestTracker>(
        dataset_mu_train_.GetCovariates(), feature_types_,
        config.num_trees_variance, n_train_);
    variance_prior_ = std::make_unique<TreePrior>(
        config.alpha_variance, config.beta_variance,
        config.min_samples_leaf_variance, config.max_depth_variance);
  }

  // ── STEP 11: Initialize active forest leaf values and residuals ────────
  // mu-forest: all leaf values = init_val_mu_ = 0 after standardization;
  // residual already = y - y_bar (no mu contribution to subtract)
  active_forest_mu_->ResetRoot();
  active_forest_mu_->SetLeafValue(init_val_mu_);
  tracker_mu_->ReconstituteFromForest(
      *active_forest_mu_, dataset_mu_train_, residual_, /*update_residual=*/true);

  // tau-forest: all leaf values = 0; basis in dataset is the initial tau_basis
  active_forest_tau_->ResetRoot();
  if (treatment_dim_ == 1) {
    active_forest_tau_->SetLeafValue(0.0);
  } else {
    std::vector<double> zero_tau(treatment_dim_, 0.0);
    active_forest_tau_->SetLeafVector(zero_tau);
  }
  tracker_tau_->ReconstituteFromForest(
      *active_forest_tau_, dataset_tau_train_, residual_, /*update_residual=*/true);

  if (has_variance_forest_) {
    active_forest_variance_->ResetRoot();
    active_forest_variance_->SetLeafValue(var_leaf_init_);
    variance_tracker_->ReconstituteFromForest(
        *active_forest_variance_, dataset_mu_train_, residual_, false);
  }

  // ── STEP 12: RNG ──────────────────────────────────────────────────────
  if (config.random_seed >= 0) {
    rng_.seed(static_cast<uint32_t>(config.random_seed));
  } else {
    std::random_device rd;
    rng_.seed(rd());
  }
}

// ══════════════════════════════════════════════════════════════════════════
// Internal helpers
// ══════════════════════════════════════════════════════════════════════════

// Sample tau_0 from conjugate posterior and correct residual.
// Operates on sampler-level state (used during GFR; chain-level version below).
//
// The running backfitting residual already has mu(X), beta(X)*Z, and
// tau_0_prev*Z all subtracted.  The partial residual for the tau_0 update is
//   partial_resid = y_std - mu(X) - beta(X)*Z
//                = residual + tau_0_prev * basis
// We add back the previous tau_0 contribution here.
static void sample_tau_0_impl(
    const std::vector<double>& tau_basis,   // current treatment basis (length n)
    ColumnVector&              residual,
    std::vector<double>&       tau_0,
    int                        n_train,
    int                        treatment_dim,
    double                     sigma2,
    double                     tau_0_prior_var,
    std::mt19937&              rng)
{
  std::vector<double> partial_resid(n_train);
  for (int i = 0; i < n_train; i++) {
    // Add back old tau_0 contribution so partial_resid = y_std - mu(X) - beta(X)*Z
    double tau0_prev_contrib = 0.0;
    for (int d = 0; d < treatment_dim; d++) {
      tau0_prev_contrib += tau_0[d] * tau_basis[i * treatment_dim + d];
    }
    partial_resid[i] = residual.GetElement(i) + tau0_prev_contrib;
  }

  if (treatment_dim == 1) {
    double tau_0_new = SampleBayesLinReg1D(
        partial_resid.data(), tau_basis.data(), n_train,
        sigma2, tau_0_prior_var, 0.0, rng);
    double delta = tau_0_new - tau_0[0];
    for (int i = 0; i < n_train; i++)
      residual.SetElement(i, residual.GetElement(i) - delta * tau_basis[i]);
    tau_0[0] = tau_0_new;
  } else {
    Eigen::VectorXd xtr(treatment_dim);
    Eigen::MatrixXd xtx(treatment_dim, treatment_dim);
    xtr.setZero();  xtx.setZero();
    for (int i = 0; i < n_train; i++) {
      Eigen::VectorXd zi = Eigen::Map<const Eigen::VectorXd>(
          tau_basis.data() + i * treatment_dim, treatment_dim);
      xtr += zi * partial_resid[i];
      xtx += zi * zi.transpose();
    }
    Eigen::MatrixXd prior_prec =
        Eigen::MatrixXd::Identity(treatment_dim, treatment_dim) / tau_0_prior_var;
    Eigen::VectorXd prior_mean = Eigen::VectorXd::Zero(treatment_dim);
    Eigen::VectorXd tau_0_new =
        SampleBayesLinRegMulti(xtr, xtx, sigma2, prior_prec, prior_mean, rng);
    for (int i = 0; i < n_train; i++) {
      double delta_c = 0.0;
      for (int d = 0; d < treatment_dim; d++)
        delta_c += (tau_0_new(d) - tau_0[d]) * tau_basis[i * treatment_dim + d];
      residual.SetElement(i, residual.GetElement(i) - delta_c);
    }
    for (int d = 0; d < treatment_dim; d++) tau_0[d] = tau_0_new(d);
  }
}

// Sample adaptive coding (b0, b1) and propagate basis change.
static void sample_adaptive_coding_impl(
    TreeEnsemble&              forest_mu,
    TreeEnsemble&              forest_tau,
    ForestDataset&             dataset_mu,
    ForestDataset&             dataset_tau,
    ForestTracker&             tracker_tau,
    ColumnVector&              residual,
    const double*              Z_train,
    std::vector<double>&       tau_basis,   // updated in place
    std::vector<double>&       tau_0,
    bool                       sample_intercept,
    int                        n_train,
    double                     sigma2,
    double                     coding_prior_var,
    double&                    b0,
    double&                    b1,
    std::mt19937&              rng)
{
  // Raw mu(X) and tau(X) without basis multiplication
  std::vector<double> mu_raw  = forest_mu.PredictRaw(dataset_mu);
  std::vector<double> tau_raw = forest_tau.PredictRaw(dataset_tau);

  // partial_resid = resid - mu(X)  (residual already removes mu and tau*basis,
  // so add back tau*basis to get resid - mu)
  std::vector<double> partial_resid(n_train);
  for (int i = 0; i < n_train; i++) {
    double tau_contrib = tau_raw[i] * tau_basis[i];  // scalar treatment
    // resid = y - mu - tau*basis - tau_0*basis, so y - mu = resid + tau*basis + tau_0*basis
    partial_resid[i] = residual.GetElement(i) + tau_contrib;
    if (sample_intercept) partial_resid[i] += tau_0[0] * tau_basis[i];
  }

  // tau_total(X_i) = tau_raw[i] + tau_0  (scalar treatment only)
  double tau_0_scalar = sample_intercept ? tau_0[0] : 0.0;
  std::vector<double> x0(n_train), x1(n_train);
  for (int i = 0; i < n_train; i++) {
    double tau_total = tau_raw[i] + tau_0_scalar;
    double z_i = Z_train[i];
    x0[i] = tau_total * (1.0 - z_i);
    x1[i] = tau_total * z_i;
  }

  double b0_new, b1_new;
  SampleBayesLinReg2DDiag(partial_resid.data(), x0.data(), x1.data(),
                          n_train, sigma2, coding_prior_var,
                          b0_new, b1_new, rng);

  // Update basis
  std::vector<double> tau_basis_old = tau_basis;
  for (int i = 0; i < n_train; i++) {
    double z_i = Z_train[i];
    tau_basis[i] = (1.0 - z_i) * b0_new + z_i * b1_new;
  }
  // Tell the dataset about the new basis so trackers can re-predict
  dataset_tau.AddBasis(tau_basis.data(), n_train, 1, false);

  // Re-derive leaf predictions under new basis and patch residual
  UpdateResidualNewBasis(tracker_tau, dataset_tau, residual, &forest_tau);

  // Additionally correct tau_0 * Δbasis component
  if (sample_intercept) {
    for (int i = 0; i < n_train; i++) {
      double delta = (tau_basis[i] - tau_basis_old[i]) * tau_0[0];
      residual.SetElement(i, residual.GetElement(i) - delta);
    }
  }
  b0 = b0_new;
  b1 = b1_new;
}

// ══════════════════════════════════════════════════════════════════════════
// run_gfr
// ══════════════════════════════════════════════════════════════════════════

void BCFSampler::run_gfr(int n_gfr)
{
  if (n_gfr <= 0) return;

  if (config_.num_mcmc > 0 && config_.num_chains > n_gfr)
    Log::Fatal("BCFSampler::run_gfr: num_chains > n_gfr");

  // Allocate GFR snapshot containers
  gfr_mu_fc_  = std::make_unique<ForestContainer>(num_trees_mu_, 1, true, false);
  gfr_tau_fc_ = std::make_unique<ForestContainer>(num_trees_tau_, treatment_dim_, false, false);
  if (has_variance_forest_)
    gfr_var_fc_ = std::make_unique<ForestContainer>(
        config_.num_trees_variance, 1, true, true);

  gfr_sigma2_seeds_.assign(n_gfr, sigma2_init_);
  gfr_leaf_scale_mu_seeds_.assign(n_gfr, leaf_scale_mu_init_);
  gfr_leaf_scale_tau_seeds_.assign(n_gfr, leaf_scale_tau_init_);
  gfr_b0_seeds_.assign(n_gfr, b0_);
  gfr_b1_seeds_.assign(n_gfr, b1_);
  gfr_tau_0_seeds_.assign(static_cast<size_t>(treatment_dim_) * n_gfr, 0.0);

  double sigma2     = sigma2_init_;
  double ls_mu      = leaf_scale_mu_init_;
  double ls_tau     = leaf_scale_tau_init_;
  std::vector<double> tau_0_cur(treatment_dim_, 0.0);
  std::vector<double> tau_basis_cur = tau_basis_train_;
  double b0_cur = b0_, b1_cur = b1_;

  for (int i = 0; i < n_gfr; i++) {
    // ── mu-forest GFR ────────────────────────────────────────────────
    {
      auto lm = GaussianConstantLeafModel(ls_mu);
      GFRSampleOneIter<GaussianConstantLeafModel, GaussianConstantSuffStat>(
          *active_forest_mu_, *tracker_mu_, *gfr_mu_fc_, lm,
          dataset_mu_train_, residual_, *tree_prior_mu_, rng_,
          variable_weights_mu_, sweep_indices_mu_, sigma2,
          feature_types_, config_.mu_forest.cutpoint_grid_size,
          /*keep_forest=*/true, /*pre_initialized=*/true, /*backfitting=*/true,
          num_features_subsample_mu_, config_.num_threads);
    }

    // ── sigma2 ────────────────────────────────────────────────────────
    if (config_.sample_sigma2_global) {
      if (dataset_mu_train_.HasVarWeights()) {
        sigma2 = global_var_model_.SampleVarianceParameter(
            residual_.GetData(), dataset_mu_train_.GetVarWeights(),
            a_global_, b_global_, rng_);
      } else {
        sigma2 = global_var_model_.SampleVarianceParameter(
            residual_.GetData(), a_global_, b_global_, rng_);
      }
      gfr_sigma2_seeds_[i] = sigma2;
    }

    // ── leaf scale mu ─────────────────────────────────────────────────
    if (config_.mu_forest.sample_sigma2_leaf) {
      ls_mu = leaf_var_model_mu_.SampleVarianceParameter(
          active_forest_mu_.get(), a_leaf_mu_, b_leaf_mu_, rng_);
      gfr_leaf_scale_mu_seeds_[i] = ls_mu;
    }

    // ── tau_0 ─────────────────────────────────────────────────────────
    if (sample_intercept_) {
      sample_tau_0_impl(tau_basis_cur, residual_,
                        tau_0_cur, n_train_, treatment_dim_,
                        sigma2, tau_0_prior_var_, rng_);
      double* dst = gfr_tau_0_seeds_.data() + static_cast<size_t>(i) * treatment_dim_;
      for (int d = 0; d < treatment_dim_; d++) dst[d] = tau_0_cur[d];
    }

    // ── tau-forest GFR ────────────────────────────────────────────────
    if (treatment_dim_ == 1) {
      auto lm = GaussianUnivariateRegressionLeafModel(ls_tau);
      GFRSampleOneIter<GaussianUnivariateRegressionLeafModel,
                       GaussianUnivariateRegressionSuffStat>(
          *active_forest_tau_, *tracker_tau_, *gfr_tau_fc_, lm,
          dataset_tau_train_, residual_, *tree_prior_tau_, rng_,
          variable_weights_tau_, sweep_indices_tau_, sigma2,
          feature_types_, config_.tau_forest.cutpoint_grid_size,
          true, true, true, num_features_subsample_tau_, config_.num_threads);
    } else {
      Eigen::MatrixXd Sigma_tau =
          Eigen::MatrixXd::Identity(treatment_dim_, treatment_dim_) * ls_tau;
      auto lm = GaussianMultivariateRegressionLeafModel(Sigma_tau);
      GFRSampleOneIter<GaussianMultivariateRegressionLeafModel,
                       GaussianMultivariateRegressionSuffStat>(
          *active_forest_tau_, *tracker_tau_, *gfr_tau_fc_, lm,
          dataset_tau_train_, residual_, *tree_prior_tau_, rng_,
          variable_weights_tau_, sweep_indices_tau_, sigma2,
          feature_types_, config_.tau_forest.cutpoint_grid_size,
          true, true, true, num_features_subsample_tau_, config_.num_threads,
          treatment_dim_);
    }

    // ── leaf scale tau ────────────────────────────────────────────────
    if (config_.tau_forest.sample_sigma2_leaf && treatment_dim_ == 1) {
      ls_tau = leaf_var_model_tau_.SampleVarianceParameter(
          active_forest_tau_.get(), a_leaf_tau_, b_leaf_tau_, rng_);
      gfr_leaf_scale_tau_seeds_[i] = ls_tau;
    }

    // ── adaptive coding ───────────────────────────────────────────────
    if (adaptive_coding_) {
      sample_adaptive_coding_impl(
          *active_forest_mu_, *active_forest_tau_,
          dataset_mu_train_, dataset_tau_train_,
          *tracker_tau_, residual_,
          Z_train_ptr_, tau_basis_cur, tau_0_cur,
          sample_intercept_, n_train_, sigma2,
          config_.coding_prior_var, b0_cur, b1_cur, rng_);
      gfr_b0_seeds_[i] = b0_cur;
      gfr_b1_seeds_[i] = b1_cur;
    }

    // ── variance forest GFR ───────────────────────────────────────────
    if (has_variance_forest_) {
      LogLinearVarianceLeafModel var_lm(a_forest_, b_forest_);
      GFRSampleOneIter<LogLinearVarianceLeafModel, LogLinearVarianceSuffStat>(
          *active_forest_variance_, *variance_tracker_,
          *gfr_var_fc_, var_lm,
          dataset_mu_train_, residual_, *variance_prior_, rng_,
          variable_weights_variance_, sweep_indices_variance_, sigma2,
          feature_types_, config_.tau_forest.cutpoint_grid_size,
          true, true, false, p_aug_, config_.num_threads);
    }
  }

  // Stash final basis into the member variable so make_chain_state_ can use it
  tau_basis_train_ = tau_basis_cur;
  tau_0_          = tau_0_cur;
  b0_             = b0_cur;
  b1_             = b1_cur;
  n_gfr_stored_   = n_gfr;
}

// ══════════════════════════════════════════════════════════════════════════
// alloc_result_
// ══════════════════════════════════════════════════════════════════════════

void BCFSampler::alloc_result_(BCFResult* result, int n_mcmc,
                                int keep_every) const
{
  int num_total = config_.num_chains * n_mcmc;

  result->mu_forest_container  = std::make_unique<ForestContainer>(
      num_trees_mu_, 1, true, false);
  result->tau_forest_container = std::make_unique<ForestContainer>(
      num_trees_tau_, treatment_dim_, false, false);
  if (has_variance_forest_)
    result->variance_forest_container = std::make_unique<ForestContainer>(
        config_.num_trees_variance, 1, true, true);

  result->y_hat_train.assign(static_cast<size_t>(n_train_) * num_total, 0.0);
  result->mu_hat_train.assign(static_cast<size_t>(n_train_) * num_total, 0.0);
  result->tau_hat_train.assign(static_cast<size_t>(n_train_) * num_total, 0.0);
  if (has_test_) {
    result->y_hat_test.assign(static_cast<size_t>(n_test_) * num_total, 0.0);
    result->mu_hat_test.assign(static_cast<size_t>(n_test_) * num_total, 0.0);
    result->tau_hat_test.assign(static_cast<size_t>(n_test_) * num_total, 0.0);
  }
  if (has_variance_forest_) {
    result->sigma2_x_hat_train.assign(
        static_cast<size_t>(n_train_) * num_total, 0.0);
    if (has_test_)
      result->sigma2_x_hat_test.assign(
          static_cast<size_t>(n_test_) * num_total, 0.0);
  }
  if (config_.sample_sigma2_global)
    result->sigma2_global_samples.assign(num_total, 0.0);
  if (config_.mu_forest.sample_sigma2_leaf)
    result->leaf_scale_mu_samples.assign(num_total, 0.0);
  if (config_.tau_forest.sample_sigma2_leaf && treatment_dim_ == 1)
    result->leaf_scale_tau_samples.assign(num_total, 0.0);
  if (sample_intercept_)
    result->tau_0_samples.assign(
        static_cast<size_t>(treatment_dim_) * num_total, 0.0);
  if (adaptive_coding_) {
    result->b0_samples.assign(num_total, 0.0);
    result->b1_samples.assign(num_total, 0.0);
  }

  result->num_total_samples  = num_total;
  result->num_chains         = config_.num_chains;
  result->n_train            = n_train_;
  result->n_test             = n_test_;
  result->treatment_dim      = treatment_dim_;
  result->y_bar              = y_bar_;
  result->y_std              = y_std_;
}

// ══════════════════════════════════════════════════════════════════════════
// make_chain_state_
// ══════════════════════════════════════════════════════════════════════════

std::unique_ptr<BCFChainState>
BCFSampler::make_chain_state_(int chain_idx, bool alloc_chain_containers)
{
  auto cs = std::make_unique<BCFChainState>();

  // ── Per-chain residual ────────────────────────────────────────────
  cs->residual.LoadData(resid_vec_.data(), n_train_);

  // ── Per-chain datasets ────────────────────────────────────────────
  // tau_basis starts from the sampler-level snapshot (updated by GFR/root).
  cs->tau_basis = tau_basis_train_;

  cs->dataset_mu.AddCovariates(X_aug_train_.data(), n_train_, p_aug_, false);
  cs->dataset_tau.AddCovariates(X_aug_train_.data(), n_train_, p_aug_, false);
  cs->dataset_tau.AddBasis(cs->tau_basis.data(), n_train_, treatment_dim_, false);

  // ── Per-chain forests and trackers ────────────────────────────────
  cs->forest_mu = std::make_unique<TreeEnsemble>(num_trees_mu_, 1, true, false);
  cs->tracker_mu = std::make_unique<ForestTracker>(
      cs->dataset_mu.GetCovariates(), feature_types_, num_trees_mu_, n_train_);

  cs->forest_tau = std::make_unique<TreeEnsemble>(
      num_trees_tau_, treatment_dim_, false, false);
  cs->tracker_tau = std::make_unique<ForestTracker>(
      cs->dataset_tau.GetCovariates(), feature_types_, num_trees_tau_, n_train_);

  if (has_variance_forest_) {
    cs->forest_var = std::make_unique<TreeEnsemble>(
        config_.num_trees_variance, 1, true, true);
    cs->tracker_var = std::make_unique<ForestTracker>(
        cs->dataset_mu.GetCovariates(), feature_types_,
        config_.num_trees_variance, n_train_);
  }

  // ── Per-chain RNG ─────────────────────────────────────────────────
  unsigned base_seed = (config_.random_seed >= 0)
      ? static_cast<unsigned>(config_.random_seed)
      : std::random_device{}();
  cs->rng = std::mt19937(base_seed + static_cast<unsigned>(chain_idx) + 1u);

  // ── Seed scalars ──────────────────────────────────────────────────
  cs->sigma2        = sigma2_init_;
  cs->leaf_scale_mu = leaf_scale_mu_init_;
  cs->leaf_scale_tau = leaf_scale_tau_init_;
  cs->tau_0 = std::vector<double>(treatment_dim_, 0.0);
  cs->b0 = b0_;
  cs->b1 = b1_;

  if (n_gfr_stored_ > 0) {
    int forest_ind = n_gfr_stored_ - chain_idx - 1;

    // Reconstitute mu forest + tracker from GFR snapshot
    cs->forest_mu->ReconstituteFromForest(*gfr_mu_fc_->GetEnsemble(forest_ind));
    cs->tracker_mu->ReconstituteFromForest(
        *cs->forest_mu, cs->dataset_mu, cs->residual, /*update_residual=*/true);

    // Reconstitute tau forest from GFR snapshot (don't update residual —
    // ReconstituteFromForest for tau uses the basis already in cs->dataset_tau)
    cs->forest_tau->ReconstituteFromForest(*gfr_tau_fc_->GetEnsemble(forest_ind));
    cs->tracker_tau->ReconstituteFromForest(
        *cs->forest_tau, cs->dataset_tau, cs->residual, true);

    if (!gfr_sigma2_seeds_.empty())
      cs->sigma2 = gfr_sigma2_seeds_[forest_ind];
    if (!gfr_leaf_scale_mu_seeds_.empty())
      cs->leaf_scale_mu = gfr_leaf_scale_mu_seeds_[forest_ind];
    if (!gfr_leaf_scale_tau_seeds_.empty())
      cs->leaf_scale_tau = gfr_leaf_scale_tau_seeds_[forest_ind];

    // Restore tau_0 and coding parameters from GFR seeds
    if (sample_intercept_) {
      const double* src = gfr_tau_0_seeds_.data() +
                          static_cast<size_t>(forest_ind) * treatment_dim_;
      for (int d = 0; d < treatment_dim_; d++) cs->tau_0[d] = src[d];
    }
    if (adaptive_coding_) {
      cs->b0 = gfr_b0_seeds_[forest_ind];
      cs->b1 = gfr_b1_seeds_[forest_ind];
      // Rebuild tau_basis for this chain from restored b0/b1
      for (int i = 0; i < n_train_; i++) {
        double z_i = Z_train_ptr_[i];
        cs->tau_basis[i] = (1.0 - z_i) * cs->b0 + z_i * cs->b1;
      }
      cs->dataset_tau.AddBasis(cs->tau_basis.data(), n_train_, treatment_dim_, false);
      UpdateResidualNewBasis(*cs->tracker_tau, cs->dataset_tau,
                             cs->residual, cs->forest_tau.get());
    }

    if (has_variance_forest_) {
      cs->forest_var->ReconstituteFromForest(*gfr_var_fc_->GetEnsemble(forest_ind));
      cs->tracker_var->ReconstituteFromForest(
          *cs->forest_var, cs->dataset_mu, cs->residual, false);
    }
  } else {
    // No GFR: initialize from root stumps
    cs->forest_mu->ResetRoot();
    cs->forest_mu->SetLeafValue(init_val_mu_);
    cs->tracker_mu->ReconstituteFromForest(
        *cs->forest_mu, cs->dataset_mu, cs->residual, true);

    cs->forest_tau->ResetRoot();
    if (treatment_dim_ == 1) {
      cs->forest_tau->SetLeafValue(0.0);
    } else {
      std::vector<double> zero_tau(treatment_dim_, 0.0);
      cs->forest_tau->SetLeafVector(zero_tau);
    }
    cs->tracker_tau->ReconstituteFromForest(
        *cs->forest_tau, cs->dataset_tau, cs->residual, true);

    if (has_variance_forest_) {
      cs->forest_var->ResetRoot();
      cs->forest_var->SetLeafValue(var_leaf_init_);
      cs->tracker_var->ReconstituteFromForest(
          *cs->forest_var, cs->dataset_mu, cs->residual, false);
    }
  }

  // ── Per-chain output containers (multi-chain only) ────────────────
  if (alloc_chain_containers) {
    cs->chain_mu_fc  = std::make_unique<ForestContainer>(
        num_trees_mu_, 1, true, false);
    cs->chain_tau_fc = std::make_unique<ForestContainer>(
        num_trees_tau_, treatment_dim_, false, false);
    if (has_variance_forest_)
      cs->chain_var_fc = std::make_unique<ForestContainer>(
          config_.num_trees_variance, 1, true, true);
  }

  return cs;
}

// ══════════════════════════════════════════════════════════════════════════
// run_chain_iters_
// ══════════════════════════════════════════════════════════════════════════

void BCFSampler::run_chain_iters_(BCFChainState& cs, int chain_idx,
                                   int n_mcmc, int keep_every, int num_burnin,
                                   int num_threads, BCFResult* result,
                                   ForestContainer* mu_fc,
                                   ForestContainer* tau_fc,
                                   ForestContainer* var_fc,
                                   RandomEffectsContainer* /*rfx_fc*/)
{
  int total_iters = num_burnin + n_mcmc * keep_every;
  int mcmc_kept = 0;

  for (int iter = 0; iter < total_iters; iter++) {
    bool is_burnin = (iter < num_burnin);
    bool is_kept   = !is_burnin && ((iter - num_burnin) % keep_every == 0);

    // ── mu-forest MCMC ───────────────────────────────────────────────
    {
      auto lm = GaussianConstantLeafModel(cs.leaf_scale_mu);
      MCMCSampleOneIter<GaussianConstantLeafModel, GaussianConstantSuffStat>(
          *cs.forest_mu, *cs.tracker_mu, *mu_fc, lm,
          cs.dataset_mu, cs.residual, *tree_prior_mu_, cs.rng,
          variable_weights_mu_, sweep_indices_mu_, cs.sigma2,
          is_kept, true, true, num_threads);
    }

    // ── sigma2 ───────────────────────────────────────────────────────
    if (config_.sample_sigma2_global) {
      if (cs.dataset_mu.HasVarWeights()) {
        cs.sigma2 = global_var_model_.SampleVarianceParameter(
            cs.residual.GetData(), cs.dataset_mu.GetVarWeights(),
            a_global_, b_global_, cs.rng);
      } else {
        cs.sigma2 = global_var_model_.SampleVarianceParameter(
            cs.residual.GetData(), a_global_, b_global_, cs.rng);
      }
    }

    // ── leaf scale mu ─────────────────────────────────────────────────
    if (config_.mu_forest.sample_sigma2_leaf) {
      cs.leaf_scale_mu = leaf_var_model_mu_.SampleVarianceParameter(
          cs.forest_mu.get(), a_leaf_mu_, b_leaf_mu_, cs.rng);
    }

    // ── tau_0 ─────────────────────────────────────────────────────────
    if (sample_intercept_) {
      sample_tau_0_impl(cs.tau_basis, cs.residual,
                        cs.tau_0, n_train_, treatment_dim_,
                        cs.sigma2, tau_0_prior_var_, cs.rng);
    }

    // ── tau-forest MCMC ───────────────────────────────────────────────
    if (treatment_dim_ == 1) {
      auto lm = GaussianUnivariateRegressionLeafModel(cs.leaf_scale_tau);
      MCMCSampleOneIter<GaussianUnivariateRegressionLeafModel,
                        GaussianUnivariateRegressionSuffStat>(
          *cs.forest_tau, *cs.tracker_tau, *tau_fc, lm,
          cs.dataset_tau, cs.residual, *tree_prior_tau_, cs.rng,
          variable_weights_tau_, sweep_indices_tau_, cs.sigma2,
          is_kept, true, true, num_threads);
    } else {
      Eigen::MatrixXd Sigma_tau =
          Eigen::MatrixXd::Identity(treatment_dim_, treatment_dim_) *
          cs.leaf_scale_tau;
      auto lm = GaussianMultivariateRegressionLeafModel(Sigma_tau);
      MCMCSampleOneIter<GaussianMultivariateRegressionLeafModel,
                        GaussianMultivariateRegressionSuffStat>(
          *cs.forest_tau, *cs.tracker_tau, *tau_fc, lm,
          cs.dataset_tau, cs.residual, *tree_prior_tau_, cs.rng,
          variable_weights_tau_, sweep_indices_tau_, cs.sigma2,
          is_kept, true, true, num_threads, treatment_dim_);
    }

    // ── leaf scale tau ────────────────────────────────────────────────
    if (config_.tau_forest.sample_sigma2_leaf && treatment_dim_ == 1) {
      cs.leaf_scale_tau = leaf_var_model_tau_.SampleVarianceParameter(
          cs.forest_tau.get(), a_leaf_tau_, b_leaf_tau_, cs.rng);
    }

    // ── adaptive coding ───────────────────────────────────────────────
    if (adaptive_coding_) {
      sample_adaptive_coding_impl(
          *cs.forest_mu, *cs.forest_tau,
          cs.dataset_mu, cs.dataset_tau, *cs.tracker_tau,
          cs.residual, Z_train_ptr_,
          cs.tau_basis, cs.tau_0, sample_intercept_,
          n_train_, cs.sigma2, config_.coding_prior_var,
          cs.b0, cs.b1, cs.rng);
    }

    // ── variance forest MCMC ──────────────────────────────────────────
    if (has_variance_forest_) {
      LogLinearVarianceLeafModel var_lm(a_forest_, b_forest_);
      MCMCSampleOneIter<LogLinearVarianceLeafModel, LogLinearVarianceSuffStat>(
          *cs.forest_var, *cs.tracker_var, *var_fc, var_lm,
          cs.dataset_mu, cs.residual, *variance_prior_, cs.rng,
          variable_weights_variance_, sweep_indices_variance_, cs.sigma2,
          is_kept, true, false, num_threads);
    }

    // ── Write kept sample ─────────────────────────────────────────────
    if (is_kept) {
      int col = chain_idx * n_mcmc + mcmc_kept;

      // mu and tau predictions from tracker (already basis-weighted for tau)
      for (int j = 0; j < n_train_; j++) {
        double mu_pred  = cs.tracker_mu->GetSamplePrediction(j);
        double tau_pred = cs.tracker_tau->GetSamplePrediction(j);
        // Add tau_0 * basis contribution
        double tau0_contrib = 0.0;
        if (sample_intercept_) {
          for (int d = 0; d < treatment_dim_; d++)
            tau0_contrib += cs.tau_0[d] * cs.tau_basis[j * treatment_dim_ + d];
        }
        double mu_orig  = mu_pred * y_std_;
        double tau_orig = (tau_pred + tau0_contrib) * y_std_;
        result->mu_hat_train[j + col * n_train_]  = mu_orig;
        result->tau_hat_train[j + col * n_train_] = tau_orig;
        result->y_hat_train[j + col * n_train_]   = mu_orig + tau_orig + y_bar_;
      }

      if (has_variance_forest_) {
        for (int j = 0; j < n_train_; j++) {
          double vp = cs.tracker_var->GetSamplePrediction(j);
          result->sigma2_x_hat_train[j + col * n_train_] =
              std::exp(vp) * y_std_ * y_std_;
        }
      }

      if (config_.sample_sigma2_global && !result->sigma2_global_samples.empty())
        result->sigma2_global_samples[col] = cs.sigma2 * y_std_ * y_std_;
      if (config_.mu_forest.sample_sigma2_leaf &&
          !result->leaf_scale_mu_samples.empty())
        result->leaf_scale_mu_samples[col] = cs.leaf_scale_mu;
      if (config_.tau_forest.sample_sigma2_leaf && treatment_dim_ == 1 &&
          !result->leaf_scale_tau_samples.empty())
        result->leaf_scale_tau_samples[col] = cs.leaf_scale_tau;
      if (sample_intercept_) {
        double* dst = result->tau_0_samples.data() +
                      static_cast<size_t>(col) * treatment_dim_;
        for (int d = 0; d < treatment_dim_; d++) dst[d] = cs.tau_0[d] * y_std_;
      }
      if (adaptive_coding_) {
        result->b0_samples[col] = cs.b0;
        result->b1_samples[col] = cs.b1;
      }

      mcmc_kept++;
    }
  }
}

// ══════════════════════════════════════════════════════════════════════════
// run_mcmc
// ══════════════════════════════════════════════════════════════════════════

void BCFSampler::run_mcmc(int n_mcmc, BCFResult* result, int keep_every)
{
  if (n_mcmc <= 0) return;

  alloc_result_(result, n_mcmc, keep_every);

  int num_chains   = config_.num_chains;
  int num_burnin   = config_.num_burnin;
  bool multi_chain = (num_chains > 1);

  // Build per-chain states (parallel)
  std::vector<std::unique_ptr<BCFChainState>> chains(num_chains);
#pragma omp parallel for schedule(static, 1) num_threads(num_chains)
  for (int c = 0; c < num_chains; c++)
    chains[c] = make_chain_state_(c, multi_chain);

  // Determine per-chain thread budget
  int chain_threads = std::max(1, config_.num_threads / std::max(1, num_chains));

  // Run MCMC (parallel over chains)
#pragma omp parallel for schedule(static, 1) num_threads(num_chains)
  for (int c = 0; c < num_chains; c++) {
    ForestContainer* mu_target  = multi_chain ? chains[c]->chain_mu_fc.get()
                                              : result->mu_forest_container.get();
    ForestContainer* tau_target = multi_chain ? chains[c]->chain_tau_fc.get()
                                              : result->tau_forest_container.get();
    ForestContainer* var_target = has_variance_forest_
                                      ? (multi_chain ? chains[c]->chain_var_fc.get()
                                                     : result->variance_forest_container.get())
                                      : nullptr;
    run_chain_iters_(*chains[c], c, n_mcmc, keep_every, num_burnin,
                     chain_threads, result,
                     mu_target, tau_target, var_target, nullptr);
  }

  // Merge per-chain forest containers (multi-chain only)
  if (multi_chain) {
    for (int c = 0; c < num_chains; c++) {
      auto& cmu  = *chains[c]->chain_mu_fc;
      auto& ctau = *chains[c]->chain_tau_fc;
      for (int s = 0; s < cmu.NumSamples(); s++)
        result->mu_forest_container->AddSample(*cmu.GetEnsemble(s));
      for (int s = 0; s < ctau.NumSamples(); s++)
        result->tau_forest_container->AddSample(*ctau.GetEnsemble(s));
      if (has_variance_forest_) {
        auto& cvar = *chains[c]->chain_var_fc;
        for (int s = 0; s < cvar.NumSamples(); s++)
          result->variance_forest_container->AddSample(*cvar.GetEnsemble(s));
      }
    }
  }

  // Batch test-set predictions
  if (has_test_) {
    int num_total = result->num_total_samples;

    std::vector<double> mu_test_raw =
        result->mu_forest_container->Predict(dataset_mu_test_);
    std::vector<double> tau_test_raw =
        result->tau_forest_container->Predict(dataset_tau_test_);

    // tau_test already includes basis multiplication from Predict()
    for (int s = 0; s < num_total; s++) {
      // tau_0 contribution for test: use the tau_0 from the corresponding
      // train sample (stored in result->tau_0_samples).
      for (int j = 0; j < n_test_; j++) {
        double mu_t  = mu_test_raw[j + s * n_test_] * y_std_;
        double tau_t = tau_test_raw[j + s * n_test_] * y_std_;
        // Add scaled tau_0 * test_basis
        if (sample_intercept_ && !result->tau_0_samples.empty()) {
          for (int d = 0; d < treatment_dim_; d++) {
            double tau_0_s = result->tau_0_samples[s * treatment_dim_ + d];
            tau_t += tau_0_s * tau_basis_test_[j * treatment_dim_ + d];
          }
        }
        result->mu_hat_test[j + s * n_test_]  = mu_t;
        result->tau_hat_test[j + s * n_test_] = tau_t;
        result->y_hat_test[j + s * n_test_]   = mu_t + tau_t + y_bar_;
      }
    }

    if (has_variance_forest_) {
      std::vector<double> var_test_raw =
          result->variance_forest_container->Predict(dataset_mu_test_);
      for (int s = 0; s < num_total; s++) {
        for (int j = 0; j < n_test_; j++) {
          result->sigma2_x_hat_test[j + s * n_test_] =
              std::exp(var_test_raw[j + s * n_test_]) * y_std_ * y_std_;
        }
      }
    }
  }
}

// ══════════════════════════════════════════════════════════════════════════
// BCFSamplerFit
// ══════════════════════════════════════════════════════════════════════════

void BCFSamplerFit(BCFResult* result_ptr, const BCFConfig& config,
                   const BCFData& data, const std::string& /*previous_model_json*/)
{
  BCFSampler sampler(config, data);
  sampler.run_gfr(config.num_gfr);
  sampler.run_mcmc(config.num_mcmc, result_ptr, config.keep_every);
}

} // namespace StochTree
