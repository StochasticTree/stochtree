/*!
 * Copyright (c) 2026 stochtree authors. All rights reserved.
 *
 * Implementation of the stateful BARTSampler (RFC 0005).
 *
 * The constructor performs one-time setup (standardization, prior calibration,
 * dataset and tree initialization).  run_gfr() runs the GFR warm-start loop
 * and run_mcmc() runs the MCMC loop.  Mutable state lives in class members
 * rather than local variables, enabling phased and incremental sampling.
 */
#include <stochtree/bart_sampler.h>
#include <stochtree/log.h>
#include <stochtree/probit.h>
#include <stochtree/tree_sampler.h>

#include <chrono>
#include <iomanip>
#include <iostream>

#include <algorithm>
#include <cmath>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace StochTree {

// ── ChainState ────────────────────────────────────────────────────────────
//
// All mutable state needed by one MCMC chain.  Allocated on the heap via
// unique_ptr so the address of resid_vec never changes (safe for ColumnVector).

struct ChainState {
  // Per-chain forests and trackers
  std::unique_ptr<TreeEnsemble>  forest;
  std::unique_ptr<ForestTracker> tracker;
  std::unique_ptr<TreeEnsemble>  forest_variance;
  std::unique_ptr<ForestTracker> variance_tracker;
  // Per-chain RFX
  std::unique_ptr<MultivariateRegressionRandomEffectsModel> rfx_model;
  std::unique_ptr<RandomEffectsTracker>                     rfx_tracker;
  // Per-chain residual (owned Eigen copy)
  ColumnVector residual;
  // Per-chain dataset (needed for variance-forest and cloglog aux slots)
  ForestDataset dataset;
  // Per-chain cloglog sampler
  OrdinalSampler ordinal_sampler;
  // Per-chain RNG
  std::mt19937 rng;
  // Per-chain current scalar parameters
  double sigma2     = 0.0;
  double leaf_scale = 0.0;
  // Per-chain result containers (merged into BARTResult after parallel section)
  std::unique_ptr<ForestContainer>        chain_fc;
  std::unique_ptr<ForestContainer>        chain_var_fc;
  std::unique_ptr<RandomEffectsContainer> chain_rfx_fc;

  ChainState() = default;
  // Non-copyable, non-movable (ColumnVector owns Eigen data; ForestDataset
  // may hold raw pointers from the caller; unique_ptrs would need rebinding).
  ChainState(const ChainState&)            = delete;
  ChainState& operator=(const ChainState&) = delete;
  ChainState(ChainState&&)                 = delete;
  ChainState& operator=(ChainState&&)      = delete;
};

// ── Internal helpers ───────────────────────────────────────────────────────

static double VecMeanS(const double* data, int n) {
  double s = 0.0;
  for (int i = 0; i < n; i++) s += data[i];
  return s / static_cast<double>(n);
}

static double VecVarS(const double* data, int n, double mean) {
  double s = 0.0;
  for (int i = 0; i < n; i++) { double d = data[i] - mean; s += d * d; }
  return s / static_cast<double>(n);
}

// ── Constructor ────────────────────────────────────────────────────────────

BARTSampler::BARTSampler(const BARTConfig& config, const BARTData& data)
    : config_(config)
{
  X_train_ptr_     = data.X_train;
  basis_train_ptr_ = data.basis_train;
  weights_ptr_     = data.weights;

  // ── Feature gates (stubs for unsupported features) ────────────────
  if (config_.leaf_model == LeafModel::MultivariateRegression && config_.sample_sigma2_leaf)
    Log::Fatal("BARTSampler: leaf scale sampling is not supported for multivariate leaf regression.");
  if (config_.link_function == LinkFunction::Cloglog) {
    if (config_.num_trees <= 0)
      Log::Fatal("BARTSampler: cloglog link requires num_trees > 0.");
    if (config_.include_variance_forest)
      Log::Fatal("BARTSampler: variance forest is not supported with cloglog link.");
    if (config_.leaf_model != LeafModel::Constant)
      Log::Fatal("BARTSampler: leaf basis regression is not supported with cloglog link.");
    if (config_.cloglog_num_categories < 2)
      Log::Fatal("BARTSampler: cloglog_num_categories must be >= 2.");
  }
  if (config_.include_variance_forest && config_.num_trees_variance <= 0)
    Log::Fatal("BARTSampler: include_variance_forest=true requires num_trees_variance > 0.");
  if (config_.num_trees <= 0 && !config_.include_variance_forest
      && config_.rfx_model_spec == RFXModelSpec::None)
    Log::Fatal("BARTSampler: at least one of num_trees > 0, include_variance_forest=true, "
               "or rfx_model_spec != None must be specified.");
  if (config_.rfx_model_spec == RFXModelSpec::Custom && config_.rfx_num_components < 1)
    Log::Fatal("BARTSampler: rfx_num_components must be >= 1 for Custom RFX.");

  // ── Input validation ───────────────────────────────────────────────
  if (data.X_train == nullptr || data.y_train == nullptr)
    Log::Fatal("BARTSampler: X_train and y_train must not be null.");
  if (data.n_train <= 0 || data.p <= 0)
    Log::Fatal("BARTSampler: n_train and p must be positive.");
  if (config_.num_chains < 1)
    Log::Fatal("BARTSampler: num_chains must be >= 1.");
  if (!config_.variable_weights_mean.empty() &&
      static_cast<int>(config_.variable_weights_mean.size()) != data.p)
    Log::Fatal("BARTSampler: variable_weights_mean length must equal p.");

  // ── Derived flags ──────────────────────────────────────────────────
  n_train_       = data.n_train;
  n_test_        = data.n_test;
  p_             = data.p;
  basis_dim_     = data.basis_dim;
  num_trees_             = config_.num_trees;
  num_trees_variance_    = config_.num_trees_variance;
  has_test_              = (n_test_ > 0 && data.X_test != nullptr);
  is_probit_             = (config_.link_function == LinkFunction::Probit);
  is_cloglog_            = (config_.link_function == LinkFunction::Cloglog);
  is_leaf_regression_    = (config_.leaf_model != LeafModel::Constant);
  is_multivariate_       = (config_.leaf_model == LeafModel::MultivariateRegression);
  has_variance_forest_   = config_.include_variance_forest && num_trees_variance_ > 0;
  has_mean_forest_       = (num_trees_ > 0);
  K_                     = is_cloglog_ ? config_.cloglog_num_categories : 0;
  rfx_num_components_    = (config_.rfx_model_spec == RFXModelSpec::InterceptOnly)
                               ? 1 : config_.rfx_num_components;
  has_rfx_               = (config_.rfx_model_spec != RFXModelSpec::None);
  has_rfx_test_          = false;  // set below
  rfx_num_groups_        = 0;      // set below
  cutpoint_grid_size_    = config_.cutpoint_grid_size > 0 ? config_.cutpoint_grid_size : n_train_;
  num_features_subsample_ = p_;

  if (is_leaf_regression_) {
    if (data.basis_train == nullptr || data.basis_dim <= 0)
      Log::Fatal("BARTSampler: basis_train and basis_dim > 0 required for leaf regression.");
  }
  if (config_.rfx_model_spec != RFXModelSpec::None) {
    if (data.rfx_groups == nullptr)
      Log::Fatal("BARTSampler: rfx_groups must not be null when rfx_model_spec != None.");
    if (config_.rfx_model_spec == RFXModelSpec::Custom && data.rfx_basis_train == nullptr)
      Log::Fatal("BARTSampler: rfx_basis_train required for Custom rfx_model_spec.");
  }

  // ── Standardize y ─────────────────────────────────────────────────
  resid_vec_.resize(n_train_);
  if (is_probit_) {
    double mean_y = VecMeanS(data.y_train, n_train_);
    y_bar_ = norm_inv_cdf(mean_y);
    y_std_ = 1.0;
    for (int i = 0; i < n_train_; i++) resid_vec_[i] = data.y_train[i] - mean_y;
  } else if (is_cloglog_) {
    double y_min = *std::min_element(data.y_train, data.y_train + n_train_);
    y_bar_ = 0.0; y_std_ = 1.0;
    for (int i = 0; i < n_train_; i++) resid_vec_[i] = data.y_train[i] - y_min;
    int K_data = static_cast<int>(*std::max_element(resid_vec_.begin(), resid_vec_.end())) + 1;
    K_ = std::max(K_, K_data);
  } else if (config_.standardize) {
    y_bar_ = VecMeanS(data.y_train, n_train_);
    double v = VecVarS(data.y_train, n_train_, y_bar_);
    y_std_ = (v > 1e-14) ? std::sqrt(v) : 1.0;
    for (int i = 0; i < n_train_; i++) resid_vec_[i] = (data.y_train[i] - y_bar_) / y_std_;
  } else {
    y_bar_ = 0.0; y_std_ = 1.0;
    for (int i = 0; i < n_train_; i++) resid_vec_[i] = data.y_train[i];
  }

  // ── Prior calibration ──────────────────────────────────────────────
  double resid_var = VecVarS(resid_vec_.data(), n_train_,
                              VecMeanS(resid_vec_.data(), n_train_));

  a_global_ = config_.a_global >= 0.0 ? config_.a_global : 0.0;
  b_global_ = config_.b_global >= 0.0 ? config_.b_global : 0.0;
  a_leaf_   = config_.a_leaf;
  double leaf_ref = is_probit_ ? 1.0 : (is_cloglog_ ? 2.0 : resid_var);
  b_leaf_       = config_.b_leaf    > 0.0 ? config_.b_leaf    : leaf_ref / std::max(1, num_trees_);
  leaf_scale_init_ = config_.leaf_scale > 0.0 ? config_.leaf_scale : leaf_ref / std::max(1, num_trees_);
  sigma2_init_  = (is_probit_ || is_cloglog_) ? 1.0 :
                  (config_.sigma2_init > 0.0 ? config_.sigma2_init : resid_var);

  a_forest_ = 1.0; b_forest_ = 1.0; var_leaf_init_ = 0.0;
  if (has_variance_forest_) {
    constexpr double a_0 = 1.5;
    a_forest_ = config_.a_forest > 0.0 ? config_.a_forest
                                        : num_trees_variance_ / (a_0 * a_0) + 0.5;
    b_forest_ = config_.b_forest > 0.0 ? config_.b_forest
                                        : num_trees_variance_ / (a_0 * a_0);
    double init_sigma2 = config_.variance_forest_leaf_init > 0.0
        ? config_.variance_forest_leaf_init : 0.6 * resid_var;
    var_leaf_init_ = std::log(init_sigma2) / num_trees_variance_;
  }

  // ── Binary label vector for probit ────────────────────────────────
  if (is_probit_) {
    y_int_.resize(n_train_);
    for (int i = 0; i < n_train_; i++) y_int_[i] = (data.y_train[i] > 0.5) ? 1 : 0;
  }

  // ── Feature types ──────────────────────────────────────────────────
  feature_types_.assign(p_, FeatureType::kNumeric);
  if (data.feature_types != nullptr)
    for (int i = 0; i < p_; i++)
      feature_types_[i] = static_cast<FeatureType>(data.feature_types[i]);

  // ── Variable weights ───────────────────────────────────────────────
  if (!config_.variable_weights_mean.empty())
    variable_weights_ = config_.variable_weights_mean;
  else
    variable_weights_.assign(p_, 1.0 / p_);

  if (has_variance_forest_) {
    if (!config_.variable_weights_variance.empty())
      variable_weights_variance_ = config_.variable_weights_variance;
    else
      variable_weights_variance_.assign(p_, 1.0 / p_);
  }

  // ── Sweep indices ──────────────────────────────────────────────────
  sweep_indices_.resize(num_trees_);
  std::iota(sweep_indices_.begin(), sweep_indices_.end(), 0);
  if (has_variance_forest_) {
    variance_sweep_indices_.resize(num_trees_variance_);
    std::iota(variance_sweep_indices_.begin(), variance_sweep_indices_.end(), 0);
  }

  // ── RNG ────────────────────────────────────────────────────────────
  rng_ = std::mt19937(config_.random_seed >= 0
                      ? static_cast<unsigned>(config_.random_seed)
                      : std::random_device{}());

  // ── Forest datasets ────────────────────────────────────────────────
  dataset_train_.AddCovariates(
      const_cast<double*>(data.X_train), n_train_, p_, /*row_major=*/false);
  if (data.weights != nullptr)
    dataset_train_.AddVarianceWeights(const_cast<double*>(data.weights), n_train_);

  if (has_test_)
    dataset_test_.AddCovariates(
        const_cast<double*>(data.X_test), n_test_, p_, /*row_major=*/false);

  if (is_leaf_regression_) {
    dataset_train_.AddBasis(
        const_cast<double*>(data.basis_train), n_train_, basis_dim_, /*row_major=*/false);
    if (has_test_ && data.basis_test != nullptr)
      dataset_test_.AddBasis(
          const_cast<double*>(data.basis_test), n_test_, basis_dim_, /*row_major=*/false);
  }

  // ── Residual ───────────────────────────────────────────────────────
  residual_.LoadData(resid_vec_.data(), n_train_);

  // ── Cloglog auxiliary data ─────────────────────────────────────────
  if (is_cloglog_) {
    dataset_train_.AddAuxiliaryDimension(n_train_);  // slot 0: Z
    for (int i = 0; i < n_train_; i++) dataset_train_.SetAuxiliaryDataValue(0, i, 0.0);
    dataset_train_.AddAuxiliaryDimension(n_train_);  // slot 1: λ̂
    for (int i = 0; i < n_train_; i++) dataset_train_.SetAuxiliaryDataValue(1, i, 0.0);
    dataset_train_.AddAuxiliaryDimension(K_ - 1);   // slot 2: γ
    for (int k = 0; k < K_ - 1; k++) dataset_train_.SetAuxiliaryDataValue(2, k, 0.0);
    dataset_train_.AddAuxiliaryDimension(K_);       // slot 3: seg
    ordinal_sampler_.UpdateCumulativeExpSums(dataset_train_);
    dataset_train_.AddAuxiliaryDimension(n_train_);  // slot 4: exp(λ_minus) cache
    for (int i = 0; i < n_train_; i++) dataset_train_.SetAuxiliaryDataValue(4, i, 1.0);
  }

  // ── RFX setup ─────────────────────────────────────────────────────
  if (has_rfx_) {
    rfx_groups_train_vec_.resize(n_train_);
    for (int i = 0; i < n_train_; i++)
      rfx_groups_train_vec_[i] = static_cast<int32_t>(data.rfx_groups[i]);

    rfx_dataset_train_ = std::make_unique<RandomEffectsDataset>();
    if (config_.rfx_model_spec == RFXModelSpec::InterceptOnly) {
      rfx_ones_train_.assign(n_train_, 1.0);
      rfx_dataset_train_->AddBasis(rfx_ones_train_.data(), n_train_, 1, false);
    } else {
      rfx_dataset_train_->AddBasis(
          const_cast<double*>(data.rfx_basis_train), n_train_, rfx_num_components_, false);
    }
    rfx_dataset_train_->AddGroupLabels(rfx_groups_train_vec_);

    rfx_tracker_   = std::make_unique<RandomEffectsTracker>(rfx_groups_train_vec_);
    rfx_num_groups_ = rfx_tracker_->NumCategories();
    rfx_model_     = std::make_unique<MultivariateRegressionRandomEffectsModel>(
        rfx_num_components_, rfx_num_groups_);

    Eigen::VectorXd alpha_init = Eigen::VectorXd::Constant(rfx_num_components_, config_.rfx_alpha_init);
    rfx_model_->SetWorkingParameter(alpha_init);
    Eigen::MatrixXd xi_init = Eigen::MatrixXd::Constant(rfx_num_components_, rfx_num_groups_, config_.rfx_xi_init);
    rfx_model_->SetGroupParameters(xi_init);
    Eigen::MatrixXd Sigma_alpha = Eigen::MatrixXd::Identity(rfx_num_components_, rfx_num_components_)
                                  * config_.rfx_sigma_alpha_init;
    rfx_model_->SetWorkingParameterCovariance(Sigma_alpha);
    Eigen::MatrixXd Sigma_xi = Eigen::MatrixXd::Identity(rfx_num_components_, rfx_num_components_)
                               * config_.rfx_sigma_xi_init;
    rfx_model_->SetGroupParameterCovariance(Sigma_xi);
    rfx_model_->SetVariancePriorShape(config_.rfx_variance_prior_shape);
    rfx_model_->SetVariancePriorScale(config_.rfx_variance_prior_scale);

    has_rfx_test_ = has_test_ && (data.rfx_groups_test != nullptr);
    if (has_rfx_test_) {
      rfx_groups_test_vec_.resize(n_test_);
      for (int i = 0; i < n_test_; i++)
        rfx_groups_test_vec_[i] = static_cast<int32_t>(data.rfx_groups_test[i]);

      rfx_dataset_test_ = std::make_unique<RandomEffectsDataset>();
      if (config_.rfx_model_spec == RFXModelSpec::InterceptOnly) {
        rfx_ones_test_.assign(n_test_, 1.0);
        rfx_dataset_test_->AddBasis(rfx_ones_test_.data(), n_test_, 1, false);
      } else {
        rfx_dataset_test_->AddBasis(
            const_cast<double*>(data.rfx_basis_test), n_test_, rfx_num_components_, false);
      }
      rfx_dataset_test_->AddGroupLabels(rfx_groups_test_vec_);
    }
  }

  // ── Sampler objects ────────────────────────────────────────────────
  int mean_ctor_trees = has_mean_forest_ ? num_trees_ : 1;
  int mean_output_dim = is_multivariate_ ? basis_dim_ : 1;
  bool mean_leaf_const = !is_leaf_regression_;

  active_forest_ = std::make_unique<TreeEnsemble>(
      mean_ctor_trees, mean_output_dim, mean_leaf_const, false);
  tracker_ = std::make_unique<ForestTracker>(
      dataset_train_.GetCovariates(), feature_types_, mean_ctor_trees, n_train_);
  tree_prior_ = std::make_unique<TreePrior>(
      config_.alpha, config_.beta, config_.min_samples_leaf, config_.max_depth);

  // Variance forest
  int var_ctor_trees = has_variance_forest_ ? num_trees_variance_ : 1;
  active_forest_variance_ = std::make_unique<TreeEnsemble>(
      var_ctor_trees, 1, true, /*is_exponentiated=*/true);
  variance_tracker_ = std::make_unique<ForestTracker>(
      dataset_train_.GetCovariates(), feature_types_, var_ctor_trees, n_train_);
  variance_prior_ = std::make_unique<TreePrior>(
      config_.alpha_variance, config_.beta_variance,
      config_.min_samples_leaf_variance, config_.max_depth_variance);

  // ── Initialize mean forest ─────────────────────────────────────────
  init_val_ = (has_mean_forest_ && !is_leaf_regression_)
      ? VecMeanS(resid_vec_.data(), n_train_) / num_trees_ : 0.0;
  if (has_mean_forest_) {
    if (is_multivariate_) {
      std::vector<double> zero_leaf(basis_dim_, 0.0);
      active_forest_->SetLeafVector(zero_leaf);
    } else {
      active_forest_->SetLeafValue(init_val_);
    }
    if (!is_cloglog_) {
      UpdateResidualEntireForest(
          *tracker_, dataset_train_, residual_, active_forest_.get(),
          /*requires_basis=*/is_leaf_regression_, std::minus<double>());
    }
  }

  // ── Initialize variance forest ─────────────────────────────────────
  if (has_variance_forest_) {
    active_forest_variance_->SetLeafValue(var_leaf_init_);
    std::vector<double> ones(n_train_, 1.0);
    dataset_train_.AddVarianceWeights(ones.data(), n_train_);
    variance_tracker_->ReconstituteFromForest(
        *active_forest_variance_, dataset_train_, residual_, /*is_mean_model=*/false);
  }
}

// ── run_gfr ────────────────────────────────────────────────────────────────

void BARTSampler::run_gfr(int n_gfr)
{
  if (n_gfr <= 0) return;

  // Validate: can only seed as many chains as GFR iterations.
  if (config_.num_mcmc > 0 && config_.num_chains > n_gfr)
    Log::Fatal("BARTSampler::run_gfr: num_chains > n_gfr; not enough GFR samples "
               "to seed each chain independently.");

  // Allocate GFR snapshot containers.
  int mean_ctor_trees = has_mean_forest_ ? num_trees_ : 1;
  int mean_output_dim = is_multivariate_ ? basis_dim_ : 1;
  bool mean_leaf_const = !is_leaf_regression_;

  gfr_mean_fc_ = std::make_unique<ForestContainer>(
      mean_ctor_trees, mean_output_dim, mean_leaf_const, /*is_exponentiated=*/false);
  if (has_variance_forest_)
    gfr_var_fc_ = std::make_unique<ForestContainer>(
        num_trees_variance_, 1, true, /*is_exponentiated=*/true);
  if (has_rfx_)
    gfr_rfx_fc_ = std::make_unique<RandomEffectsContainer>(rfx_num_components_, rfx_num_groups_);

  gfr_sigma2_seeds_.assign(n_gfr, sigma2_init_);
  gfr_leaf_scale_seeds_.assign(n_gfr, leaf_scale_init_);
  if (is_cloglog_)
    gfr_cloglog_cutpoint_seeds_.assign(static_cast<size_t>(K_ - 1) * n_gfr, 0.0);

  double current_sigma2 = sigma2_init_;
  double leaf_scale     = leaf_scale_init_;

  for (int i = 0; i < n_gfr; i++) {
    // ── Mean forest GFR step ────────────────────────────────────────
    if (has_mean_forest_) {
      if (is_probit_)
        sample_probit_latent_outcome(residual_, *tracker_, y_int_.data(), n_train_, y_bar_, rng_);

      if (config_.leaf_model == LeafModel::UnivariateRegression) {
        auto lm = GaussianUnivariateRegressionLeafModel(leaf_scale);
        GFRSampleOneIter<GaussianUnivariateRegressionLeafModel,
                         GaussianUnivariateRegressionSuffStat>(
            *active_forest_, *tracker_, *gfr_mean_fc_, lm,
            dataset_train_, residual_, *tree_prior_, rng_,
            variable_weights_, sweep_indices_, current_sigma2,
            feature_types_, cutpoint_grid_size_,
            true, true, true, num_features_subsample_, config_.num_threads);
      } else if (config_.leaf_model == LeafModel::MultivariateRegression) {
        Eigen::MatrixXd Sigma_0 = Eigen::MatrixXd::Identity(basis_dim_, basis_dim_) * leaf_scale;
        auto lm = GaussianMultivariateRegressionLeafModel(Sigma_0);
        GFRSampleOneIter<GaussianMultivariateRegressionLeafModel,
                         GaussianMultivariateRegressionSuffStat>(
            *active_forest_, *tracker_, *gfr_mean_fc_, lm,
            dataset_train_, residual_, *tree_prior_, rng_,
            variable_weights_, sweep_indices_, current_sigma2,
            feature_types_, cutpoint_grid_size_,
            true, true, true, num_features_subsample_, config_.num_threads, basis_dim_);
      } else if (is_cloglog_) {
        auto lm = CloglogOrdinalLeafModel(config_.cloglog_forest_shape, config_.cloglog_forest_rate);
        GFRSampleOneIter<CloglogOrdinalLeafModel, CloglogOrdinalSuffStat>(
            *active_forest_, *tracker_, *gfr_mean_fc_, lm,
            dataset_train_, residual_, *tree_prior_, rng_,
            variable_weights_, sweep_indices_, current_sigma2,
            feature_types_, cutpoint_grid_size_,
            true, true, false, num_features_subsample_, config_.num_threads);
      } else {
        auto lm = GaussianConstantLeafModel(leaf_scale);
        GFRSampleOneIter<GaussianConstantLeafModel, GaussianConstantSuffStat>(
            *active_forest_, *tracker_, *gfr_mean_fc_, lm,
            dataset_train_, residual_, *tree_prior_, rng_,
            variable_weights_, sweep_indices_, current_sigma2,
            feature_types_, cutpoint_grid_size_,
            true, true, true, num_features_subsample_, config_.num_threads);
      }
    }

    // ── Cloglog Gibbs steps ─────────────────────────────────────────
    if (is_cloglog_) {
      for (int j = 0; j < n_train_; j++)
        dataset_train_.SetAuxiliaryDataValue(1, j, tracker_->GetSamplePrediction(j));
      ordinal_sampler_.UpdateLatentVariables(dataset_train_, residual_.GetData(), rng_);
      ordinal_sampler_.UpdateGammaParams(dataset_train_, residual_.GetData(),
          config_.cloglog_forest_shape, config_.cloglog_forest_rate,
          config_.cloglog_cutpoint_0, rng_);
      ordinal_sampler_.UpdateCumulativeExpSums(dataset_train_);
      // Store cutpoints into seed buffer.
      double* dst = gfr_cloglog_cutpoint_seeds_.data() + static_cast<size_t>(i) * (K_ - 1);
      for (int k = 0; k < K_ - 1; k++) dst[k] = dataset_train_.GetAuxiliaryDataValue(2, k);
    }

    // ── Variance forest GFR step ────────────────────────────────────
    if (has_variance_forest_) {
      LogLinearVarianceLeafModel var_lm(a_forest_, b_forest_);
      GFRSampleOneIter<LogLinearVarianceLeafModel, LogLinearVarianceSuffStat>(
          *active_forest_variance_, *variance_tracker_,
          *gfr_var_fc_, var_lm,
          dataset_train_, residual_, *variance_prior_, rng_,
          variable_weights_variance_, variance_sweep_indices_, current_sigma2,
          feature_types_, cutpoint_grid_size_,
          true, true, false, num_features_subsample_, config_.num_threads);
    }

    // ── RFX GFR step ────────────────────────────────────────────────
    if (has_rfx_) {
      rfx_model_->SampleRandomEffects(
          *rfx_dataset_train_, residual_, *rfx_tracker_, current_sigma2, rng_);
      gfr_rfx_fc_->AddSample(*rfx_model_);
    }

    // ── Sample scalar variance parameters ───────────────────────────
    if (has_mean_forest_ && !is_probit_ && !is_cloglog_ && config_.sample_sigma2_global) {
      if (dataset_train_.HasVarWeights()) {
        current_sigma2 = global_var_model_.SampleVarianceParameter(
            residual_.GetData(), dataset_train_.GetVarWeights(), a_global_, b_global_, rng_);
      } else {
        current_sigma2 = global_var_model_.SampleVarianceParameter(
            residual_.GetData(), a_global_, b_global_, rng_);
      }
      gfr_sigma2_seeds_[i] = current_sigma2;
    }
    if (has_mean_forest_ && !is_cloglog_ && config_.sample_sigma2_leaf) {
      leaf_scale = leaf_var_model_.SampleVarianceParameter(
          active_forest_.get(), a_leaf_, b_leaf_, rng_);
      gfr_leaf_scale_seeds_[i] = leaf_scale;
    }
  }

  n_gfr_stored_ = n_gfr;
}

// ── make_chain_state_ ────────────────────────────────────────────────────

std::unique_ptr<ChainState> BARTSampler::make_chain_state_(int chain_idx,
                                                             bool alloc_chain_containers)
{
  auto cs = std::make_unique<ChainState>();

  int mean_ctor_trees  = has_mean_forest_ ? num_trees_ : 1;
  int mean_output_dim  = is_multivariate_ ? basis_dim_ : 1;
  bool mean_leaf_const = !is_leaf_regression_;

  // ── Per-chain residual (deep copy into Eigen) ─────────────────────
  cs->residual.LoadData(const_cast<double*>(resid_vec_.data()), n_train_);

  // ── Per-chain dataset ─────────────────────────────────────────────
  // Reconstruct from stored raw pointers so each chain can independently
  // mutate variance weights (variance forest) and aux slots (cloglog).
  cs->dataset.AddCovariates(const_cast<double*>(X_train_ptr_), n_train_, p_, false);
  if (is_leaf_regression_ && basis_train_ptr_)
    cs->dataset.AddBasis(const_cast<double*>(basis_train_ptr_), n_train_, basis_dim_, false);
  if (weights_ptr_)
    cs->dataset.AddVarianceWeights(const_cast<double*>(weights_ptr_), n_train_);
  if (has_variance_forest_) {
    std::vector<double> ones(n_train_, 1.0);
    cs->dataset.AddVarianceWeights(ones.data(), n_train_);
  }
  if (is_cloglog_) {
    cs->dataset.AddAuxiliaryDimension(n_train_);  // slot 0: Z
    for (int i = 0; i < n_train_; i++) cs->dataset.SetAuxiliaryDataValue(0, i, 0.0);
    cs->dataset.AddAuxiliaryDimension(n_train_);  // slot 1: λ̂
    for (int i = 0; i < n_train_; i++) cs->dataset.SetAuxiliaryDataValue(1, i, 0.0);
    cs->dataset.AddAuxiliaryDimension(K_ - 1);    // slot 2: γ (cutpoints)
    for (int k = 0; k < K_ - 1; k++) cs->dataset.SetAuxiliaryDataValue(2, k, 0.0);
    cs->dataset.AddAuxiliaryDimension(K_);        // slot 3: seg
    cs->ordinal_sampler.UpdateCumulativeExpSums(cs->dataset);
    cs->dataset.AddAuxiliaryDimension(n_train_);  // slot 4: exp(λ_minus) cache
    for (int i = 0; i < n_train_; i++) cs->dataset.SetAuxiliaryDataValue(4, i, 1.0);
  }

  // ── Per-chain forests and trackers ────────────────────────────────
  cs->forest  = std::make_unique<TreeEnsemble>(mean_ctor_trees, mean_output_dim, mean_leaf_const, false);
  cs->tracker = std::make_unique<ForestTracker>(
      cs->dataset.GetCovariates(), feature_types_, mean_ctor_trees, n_train_);
  if (has_variance_forest_) {
    cs->forest_variance = std::make_unique<TreeEnsemble>(num_trees_variance_, 1, true, true);
    cs->variance_tracker = std::make_unique<ForestTracker>(
        cs->dataset.GetCovariates(), feature_types_, num_trees_variance_, n_train_);
  }

  // ── Per-chain RFX ─────────────────────────────────────────────────
  if (has_rfx_) {
    cs->rfx_model = std::make_unique<MultivariateRegressionRandomEffectsModel>(
        rfx_num_components_, rfx_num_groups_);
    Eigen::VectorXd alpha0 = Eigen::VectorXd::Constant(rfx_num_components_, config_.rfx_alpha_init);
    cs->rfx_model->SetWorkingParameter(alpha0);
    Eigen::MatrixXd xi0 = Eigen::MatrixXd::Constant(rfx_num_components_, rfx_num_groups_, config_.rfx_xi_init);
    cs->rfx_model->SetGroupParameters(xi0);
    Eigen::MatrixXd Sigma_a0 = Eigen::MatrixXd::Identity(rfx_num_components_, rfx_num_components_) * config_.rfx_sigma_alpha_init;
    cs->rfx_model->SetWorkingParameterCovariance(Sigma_a0);
    Eigen::MatrixXd Sigma_xi0 = Eigen::MatrixXd::Identity(rfx_num_components_, rfx_num_components_) * config_.rfx_sigma_xi_init;
    cs->rfx_model->SetGroupParameterCovariance(Sigma_xi0);
    cs->rfx_model->SetVariancePriorShape(config_.rfx_variance_prior_shape);
    cs->rfx_model->SetVariancePriorScale(config_.rfx_variance_prior_scale);
    cs->rfx_tracker = std::make_unique<RandomEffectsTracker>(rfx_groups_train_vec_);
  }

  // ── Per-chain RNG (different seed per chain) ──────────────────────
  unsigned base_seed = (config_.random_seed >= 0)
      ? static_cast<unsigned>(config_.random_seed)
      : std::random_device{}();
  cs->rng = std::mt19937(base_seed + static_cast<unsigned>(chain_idx) + 1u);

  // ── Seed from GFR or initialize from root stumps ──────────────────
  cs->sigma2     = sigma2_init_;
  cs->leaf_scale = leaf_scale_init_;

  if (n_gfr_stored_ > 0) {
    int forest_ind = n_gfr_stored_ - chain_idx - 1;

    if (has_mean_forest_) {
      cs->forest->ReconstituteFromForest(*gfr_mean_fc_->GetEnsemble(forest_ind));
      cs->tracker->ReconstituteFromForest(*cs->forest, cs->dataset, cs->residual, true);
      if (!gfr_sigma2_seeds_.empty())    cs->sigma2     = gfr_sigma2_seeds_[forest_ind];
      if (!gfr_leaf_scale_seeds_.empty()) cs->leaf_scale = gfr_leaf_scale_seeds_[forest_ind];
      if (is_cloglog_) {
        for (int j = 0; j < n_train_; j++) cs->residual.SetElement(j, resid_vec_[j]);
        const double* csrc = gfr_cloglog_cutpoint_seeds_.data()
                             + static_cast<size_t>(forest_ind) * (K_ - 1);
        for (int k = 0; k < K_ - 1; k++) cs->dataset.SetAuxiliaryDataValue(2, k, csrc[k]);
        cs->ordinal_sampler.UpdateCumulativeExpSums(cs->dataset);
        for (int j = 0; j < n_train_; j++)
          cs->dataset.SetAuxiliaryDataValue(1, j, cs->tracker->GetSamplePrediction(j));
        for (int j = 0; j < n_train_; j++) cs->dataset.SetAuxiliaryDataValue(0, j, 0.0);
      }
    }
    if (has_variance_forest_) {
      cs->forest_variance->ReconstituteFromForest(*gfr_var_fc_->GetEnsemble(forest_ind));
      cs->variance_tracker->ReconstituteFromForest(
          *cs->forest_variance, cs->dataset, cs->residual, false);
    }
    if (has_rfx_) {
      cs->rfx_model->ResetFromSample(*gfr_rfx_fc_, forest_ind);
      Eigen::MatrixXd Sigma_a_gfr = Eigen::MatrixXd::Identity(rfx_num_components_, rfx_num_components_)
                                    * config_.rfx_sigma_alpha_init;
      cs->rfx_model->SetWorkingParameterCovariance(Sigma_a_gfr);
      cs->rfx_tracker->ResetFromSample(*cs->rfx_model, *rfx_dataset_train_, cs->residual);
    }
  } else {
    // No GFR: initialize from root stumps.
    if (has_mean_forest_) {
      cs->forest->ResetRoot();
      if (is_multivariate_) {
        std::vector<double> zero_leaf(basis_dim_, 0.0);
        cs->forest->SetLeafVector(zero_leaf);
      } else {
        cs->forest->SetLeafValue(init_val_);
      }
      cs->tracker->ReconstituteFromForest(*cs->forest, cs->dataset, cs->residual, true);
      if (is_cloglog_) {
        for (int j = 0; j < n_train_; j++) cs->residual.SetElement(j, resid_vec_[j]);
        for (int k = 0; k < K_ - 1; k++) cs->dataset.SetAuxiliaryDataValue(2, k, 0.0);
        cs->ordinal_sampler.UpdateCumulativeExpSums(cs->dataset);
        for (int j = 0; j < n_train_; j++) cs->dataset.SetAuxiliaryDataValue(1, j, 0.0);
        for (int j = 0; j < n_train_; j++) cs->dataset.SetAuxiliaryDataValue(0, j, 0.0);
      }
    }
    if (has_variance_forest_) {
      cs->forest_variance->ResetRoot();
      cs->forest_variance->SetLeafValue(var_leaf_init_);
      cs->variance_tracker->ReconstituteFromForest(
          *cs->forest_variance, cs->dataset, cs->residual, false);
    }
    if (has_rfx_) {
      cs->rfx_tracker->RootReset(*cs->rfx_model, *rfx_dataset_train_, cs->residual);
    }
  }

  // ── Per-chain result containers (multi-chain only) ───────────────
  // For single-chain runs the caller passes the result containers directly,
  // so we skip this allocation to avoid a redundant copy on merge.
  if (alloc_chain_containers) {
    cs->chain_fc = std::make_unique<ForestContainer>(
        mean_ctor_trees, mean_output_dim, mean_leaf_const, false);
    if (has_variance_forest_)
      cs->chain_var_fc = std::make_unique<ForestContainer>(num_trees_variance_, 1, true, true);
    if (has_rfx_)
      cs->chain_rfx_fc = std::make_unique<RandomEffectsContainer>(
          rfx_num_components_, rfx_num_groups_);
  }

  return cs;
}

// ── run_chain_iters_ ─────────────────────────────────────────────────────

void BARTSampler::run_chain_iters_(ChainState& cs, int chain_idx,
                                    int n_mcmc, int keep_every, int num_burnin,
                                    int num_threads, BARTResult* result,
                                    ForestContainer& mean_fc,
                                    ForestContainer* var_fc,
                                    RandomEffectsContainer* rfx_fc)
{
  int total_iters = num_burnin + n_mcmc * keep_every;
  int mcmc_kept   = 0;

  for (int i = 0; i < total_iters; i++) {
    bool is_burnin = (i < num_burnin);
    bool is_kept   = !is_burnin && ((i - num_burnin) % keep_every == 0);

    // ── Mean forest MCMC step ────────────────────────────────────────
    if (has_mean_forest_) {
      if (is_probit_)
        sample_probit_latent_outcome(cs.residual, *cs.tracker,
                                     y_int_.data(), n_train_, y_bar_, cs.rng);

      if (config_.leaf_model == LeafModel::UnivariateRegression) {
        auto lm = GaussianUnivariateRegressionLeafModel(cs.leaf_scale);
        MCMCSampleOneIter<GaussianUnivariateRegressionLeafModel,
                          GaussianUnivariateRegressionSuffStat>(
            *cs.forest, *cs.tracker, mean_fc, lm,
            cs.dataset, cs.residual, *tree_prior_, cs.rng,
            variable_weights_, sweep_indices_, cs.sigma2,
            is_kept, true, true, num_threads);
      } else if (config_.leaf_model == LeafModel::MultivariateRegression) {
        Eigen::MatrixXd Sigma_0 =
            Eigen::MatrixXd::Identity(basis_dim_, basis_dim_) * cs.leaf_scale;
        auto lm = GaussianMultivariateRegressionLeafModel(Sigma_0);
        MCMCSampleOneIter<GaussianMultivariateRegressionLeafModel,
                          GaussianMultivariateRegressionSuffStat>(
            *cs.forest, *cs.tracker, mean_fc, lm,
            cs.dataset, cs.residual, *tree_prior_, cs.rng,
            variable_weights_, sweep_indices_, cs.sigma2,
            is_kept, true, true, num_threads, basis_dim_);
      } else if (is_cloglog_) {
        auto lm = CloglogOrdinalLeafModel(config_.cloglog_forest_shape, config_.cloglog_forest_rate);
        MCMCSampleOneIter<CloglogOrdinalLeafModel, CloglogOrdinalSuffStat>(
            *cs.forest, *cs.tracker, mean_fc, lm,
            cs.dataset, cs.residual, *tree_prior_, cs.rng,
            variable_weights_, sweep_indices_, cs.sigma2,
            is_kept, true, false, num_threads);
      } else {
        auto lm = GaussianConstantLeafModel(cs.leaf_scale);
        MCMCSampleOneIter<GaussianConstantLeafModel, GaussianConstantSuffStat>(
            *cs.forest, *cs.tracker, mean_fc, lm,
            cs.dataset, cs.residual, *tree_prior_, cs.rng,
            variable_weights_, sweep_indices_, cs.sigma2,
            is_kept, true, true, num_threads);
      }
    }

    // ── Cloglog Gibbs steps ──────────────────────────────────────────
    if (is_cloglog_) {
      for (int j = 0; j < n_train_; j++)
        cs.dataset.SetAuxiliaryDataValue(1, j, cs.tracker->GetSamplePrediction(j));
      cs.ordinal_sampler.UpdateLatentVariables(cs.dataset, cs.residual.GetData(), cs.rng);
      cs.ordinal_sampler.UpdateGammaParams(cs.dataset, cs.residual.GetData(),
          config_.cloglog_forest_shape, config_.cloglog_forest_rate,
          config_.cloglog_cutpoint_0, cs.rng);
      cs.ordinal_sampler.UpdateCumulativeExpSums(cs.dataset);
    }

    // ── Variance forest MCMC step ────────────────────────────────────
    if (has_variance_forest_) {
      LogLinearVarianceLeafModel var_lm(a_forest_, b_forest_);
      MCMCSampleOneIter<LogLinearVarianceLeafModel, LogLinearVarianceSuffStat>(
          *cs.forest_variance, *cs.variance_tracker, *var_fc, var_lm,
          cs.dataset, cs.residual, *variance_prior_, cs.rng,
          variable_weights_variance_, variance_sweep_indices_, cs.sigma2,
          is_kept, true, false, num_threads);
    }

    // ── RFX MCMC step ────────────────────────────────────────────────
    if (has_rfx_)
      cs.rfx_model->SampleRandomEffects(
          *rfx_dataset_train_, cs.residual, *cs.rfx_tracker, cs.sigma2, cs.rng);

    // ── Scalar variance sampling ──────────────────────────────────────
    if (has_mean_forest_ && !is_probit_ && !is_cloglog_ && config_.sample_sigma2_global) {
      if (cs.dataset.HasVarWeights()) {
        cs.sigma2 = global_var_model_.SampleVarianceParameter(
            cs.residual.GetData(), cs.dataset.GetVarWeights(), a_global_, b_global_, cs.rng);
      } else {
        cs.sigma2 = global_var_model_.SampleVarianceParameter(
            cs.residual.GetData(), a_global_, b_global_, cs.rng);
      }
    }
    if (has_mean_forest_ && !is_cloglog_ && config_.sample_sigma2_leaf)
      cs.leaf_scale = leaf_var_model_.SampleVarianceParameter(
          cs.forest.get(), a_leaf_, b_leaf_, cs.rng);

    // ── Write kept sample ─────────────────────────────────────────────
    if (is_kept) {
      int col = chain_idx * n_mcmc + mcmc_kept;

      // Training predictions (non-overlapping column writes → thread-safe)
      double* dst = result->y_hat_train.data() + static_cast<size_t>(col) * n_train_;
      if (has_mean_forest_ && has_rfx_) {
        for (int j = 0; j < n_train_; j++)
          dst[j] = (cs.tracker->GetSamplePrediction(j)
                    + cs.rfx_tracker->GetPrediction(j)) * y_std_ + y_bar_;
      } else if (has_mean_forest_) {
        for (int j = 0; j < n_train_; j++)
          dst[j] = cs.tracker->GetSamplePrediction(j) * y_std_ + y_bar_;
      } else if (has_rfx_) {
        for (int j = 0; j < n_train_; j++)
          dst[j] = cs.rfx_tracker->GetPrediction(j) * y_std_ + y_bar_;
      } else {
        std::fill(dst, dst + n_train_, y_bar_);
      }

      if (has_variance_forest_) {
        double* vdst = result->sigma2_x_hat_train.data() + static_cast<size_t>(col) * n_train_;
        for (int j = 0; j < n_train_; j++)
          vdst[j] = std::exp(cs.variance_tracker->GetSamplePrediction(j)) * y_std_ * y_std_;
      }
      if (has_rfx_)
        rfx_fc->AddSample(*cs.rfx_model);
      if (is_cloglog_) {
        double* cdst = result->cloglog_cutpoint_samples.data()
                       + static_cast<size_t>(col) * (K_ - 1);
        for (int k = 0; k < K_ - 1; k++)
          cdst[k] = cs.dataset.GetAuxiliaryDataValue(2, k);
      }
      if (has_mean_forest_ && !is_probit_ && !is_cloglog_ && config_.sample_sigma2_global)
        result->sigma2_global_samples[col] = cs.sigma2 * y_std_ * y_std_;
      if (has_mean_forest_ && !is_cloglog_ && config_.sample_sigma2_leaf)
        result->leaf_scale_samples[col] = cs.leaf_scale;

      mcmc_kept++;
    }
  }
}

// ── alloc_result_ ─────────────────────────────────────────────────────────

void BARTSampler::alloc_result_(BARTResult* result, int n_mcmc, int keep_every) const
{
  int num_chains          = config_.num_chains;
  int num_total           = num_chains * n_mcmc;
  int mean_ctor_trees     = has_mean_forest_ ? num_trees_ : 1;
  int mean_output_dim     = is_multivariate_ ? basis_dim_ : 1;
  bool mean_leaf_const    = !is_leaf_regression_;

  // Allocate forest containers.
  result->forest_container = std::make_unique<ForestContainer>(
      mean_ctor_trees, mean_output_dim, mean_leaf_const, false);
  if (has_variance_forest_)
    result->variance_forest_container = std::make_unique<ForestContainer>(
        num_trees_variance_, 1, true, /*is_exponentiated=*/true);
  if (has_rfx_)
    result->rfx_container = std::make_unique<RandomEffectsContainer>(
        rfx_num_components_, rfx_num_groups_);

  // Allocate prediction and scalar sample arrays.
  result->y_hat_train.assign(static_cast<size_t>(n_train_) * num_total, 0.0);
  if (has_test_)
    result->y_hat_test.assign(static_cast<size_t>(n_test_) * num_total, 0.0);
  if (!is_probit_ && !is_cloglog_ && config_.sample_sigma2_global)
    result->sigma2_global_samples.assign(num_total, 0.0);
  if (!is_cloglog_ && config_.sample_sigma2_leaf)
    result->leaf_scale_samples.assign(num_total, 0.0);
  if (is_cloglog_)
    result->cloglog_cutpoint_samples.assign(static_cast<size_t>(K_ - 1) * num_total, 0.0);
  if (has_variance_forest_) {
    result->sigma2_x_hat_train.assign(static_cast<size_t>(n_train_) * num_total, 0.0);
    if (has_test_)
      result->sigma2_x_hat_test.assign(static_cast<size_t>(n_test_) * num_total, 0.0);
  }

  // Metadata.
  result->num_total_samples = num_total;
  result->num_chains        = num_chains;
  result->n_train           = n_train_;
  result->n_test            = n_test_;
  result->y_bar             = y_bar_;
  result->y_std             = y_std_;
}

// ── run_mcmc ──────────────────────────────────────────────────────────────

void BARTSampler::run_mcmc(int n_mcmc, BARTResult* result, int keep_every)
{
  if (result == nullptr)
    Log::Fatal("BARTSampler::run_mcmc: result pointer must not be null.");
  if (n_mcmc <= 0) return;

  keep_every = std::max(1, keep_every);
  int num_chains = config_.num_chains;
  int num_burnin = config_.num_burnin;

  using Clock = std::chrono::high_resolution_clock;
  using Ms    = std::chrono::duration<double, std::milli>;
  auto ms_since = [](Clock::time_point t0) {
    return Ms(Clock::now() - t0).count();
  };
  Clock::time_point tp;

  // Allocate result arrays (forest containers start empty; filled below).
  tp = Clock::now();
  alloc_result_(result, n_mcmc, keep_every);
  double t_alloc = ms_since(tp);

  bool multi_chain = (num_chains > 1);

  // Build per-chain state in parallel. All reads are from GFR snapshots /
  // class members that are const at this point; each chain writes only into
  // its own freshly-allocated ChainState, so there are no data races.
  // For single-chain runs, skip allocating per-chain forest buffers — the
  // chain writes directly into the result containers (no merge needed).
  tp = Clock::now();
  std::vector<std::unique_ptr<ChainState>> chains(num_chains);
  #pragma omp parallel for schedule(static, 1) num_threads(num_chains)
  for (int c = 0; c < num_chains; c++)
    chains[c] = make_chain_state_(c, multi_chain);
  double t_chain_setup = ms_since(tp);

  // When running parallel chains, disable within-tree OpenMP so we don't
  // over-subscribe.  Single-chain runs keep the configured thread count.
  int chain_threads = multi_chain ? 1 : config_.num_threads;

  // ── Parallel chain loop ─────────────────────────────────────────────
  // Scalar samples and y_hat_train use non-overlapping column offsets per
  // chain — safe to write in parallel.
  // Multi-chain: each chain writes forests into its own chain_fc buffer.
  // Single-chain: write forests directly into the result containers.
  tp = Clock::now();
  #pragma omp parallel for schedule(static, 1) num_threads(num_chains)
  for (int c = 0; c < num_chains; c++) {
    ForestContainer* mean_target = multi_chain
        ? chains[c]->chain_fc.get()
        : result->forest_container.get();
    ForestContainer* var_target = (has_variance_forest_ && multi_chain)
        ? chains[c]->chain_var_fc.get()
        : result->variance_forest_container.get();
    RandomEffectsContainer* rfx_target = (has_rfx_ && multi_chain)
        ? chains[c]->chain_rfx_fc.get()
        : result->rfx_container.get();
    run_chain_iters_(*chains[c], c, n_mcmc, keep_every, num_burnin,
                     chain_threads, result,
                     *mean_target, var_target, rfx_target);
  }
  double t_chain_iters = ms_since(tp);

  // ── Merge per-chain forest containers (multi-chain only) ────────────
  tp = Clock::now();
  if (multi_chain) {
    ForestContainer& mean_fc = *result->forest_container;
    for (int c = 0; c < num_chains; c++) {
      auto& cfc = *chains[c]->chain_fc;
      for (int s = 0; s < cfc.NumSamples(); s++)
        mean_fc.AddSample(*cfc.GetEnsemble(s));
    }
    if (has_variance_forest_) {
      ForestContainer& var_fc = *result->variance_forest_container;
      for (int c = 0; c < num_chains; c++) {
        auto& cfc = *chains[c]->chain_var_fc;
        for (int s = 0; s < cfc.NumSamples(); s++)
          var_fc.AddSample(*cfc.GetEnsemble(s));
      }
    }
    if (has_rfx_) {
      // Merge by directly appending flat arrays (avoids serialisation round-trip).
      RandomEffectsContainer& dst_rfx = *result->rfx_container;
      for (int c = 0; c < num_chains; c++) {
        auto& src = *chains[c]->chain_rfx_fc;
        auto& sb = src.GetBeta();   dst_rfx.GetBeta().insert(  dst_rfx.GetBeta().end(),   sb.begin(), sb.end());
        auto& sa = src.GetAlpha();  dst_rfx.GetAlpha().insert( dst_rfx.GetAlpha().end(),  sa.begin(), sa.end());
        auto& sx = src.GetXi();     dst_rfx.GetXi().insert(    dst_rfx.GetXi().end(),     sx.begin(), sx.end());
        auto& ss = src.GetSigma();  dst_rfx.GetSigma().insert( dst_rfx.GetSigma().end(),  ss.begin(), ss.end());
        dst_rfx.SetNumSamples(dst_rfx.NumSamples() + src.NumSamples());
      }
    }
  }

  // ── Deferred test predictions (batch pass, serial) ──────────────────
  int num_total = num_chains * n_mcmc;
  if (has_test_ && num_total > 0) {
    std::vector<double> yhat_raw;
    if (has_mean_forest_)
      yhat_raw = result->forest_container->Predict(dataset_test_);

    std::vector<double> rfx_batch;
    if (has_rfx_ && has_rfx_test_) {
      // Use label map from any chain's rfx_tracker (all are identical).
      LabelMapper test_label_mapper;
      test_label_mapper.LoadFromLabelMap(chains[0]->rfx_tracker->GetLabelMap());
      rfx_batch.resize(static_cast<size_t>(n_test_) * num_total, 0.0);
      result->rfx_container->Predict(*rfx_dataset_test_, test_label_mapper, rfx_batch);
    }

    for (int col = 0; col < num_total; col++) {
      double* dst = result->y_hat_test.data() + static_cast<size_t>(col) * n_test_;
      bool have_forest = has_mean_forest_ && !yhat_raw.empty();
      bool have_rfx    = has_rfx_ && has_rfx_test_;
      if (have_forest && have_rfx) {
        const double* fc = yhat_raw.data()  + static_cast<size_t>(col) * n_test_;
        const double* rc = rfx_batch.data() + static_cast<size_t>(col) * n_test_;
        for (int j = 0; j < n_test_; j++) dst[j] = (fc[j] + rc[j]) * y_std_ + y_bar_;
      } else if (have_forest) {
        const double* fc = yhat_raw.data() + static_cast<size_t>(col) * n_test_;
        for (int j = 0; j < n_test_; j++) dst[j] = fc[j] * y_std_ + y_bar_;
      } else if (have_rfx) {
        const double* rc = rfx_batch.data() + static_cast<size_t>(col) * n_test_;
        for (int j = 0; j < n_test_; j++) dst[j] = rc[j] * y_std_ + y_bar_;
      } else {
        std::fill(dst, dst + n_test_, y_bar_);
      }
    }
    if (has_variance_forest_) {
      std::vector<double> vhat_raw = result->variance_forest_container->Predict(dataset_test_);
      for (int col = 0; col < num_total; col++) {
        const double* vc = vhat_raw.data() + static_cast<size_t>(col) * n_test_;
        double* vdst = result->sigma2_x_hat_test.data() + static_cast<size_t>(col) * n_test_;
        for (int j = 0; j < n_test_; j++) vdst[j] = vc[j] * y_std_ * y_std_;
      }
    }
  }

  double t_merge_post = ms_since(tp);

  // ── RFX result metadata ────────────────────────────────────────────
  if (has_rfx_) {
    result->rfx_num_groups     = rfx_num_groups_;
    result->rfx_num_components = rfx_num_components_;
    for (const auto& kv : chains[0]->rfx_tracker->GetLabelMap())
      result->rfx_group_ids.push_back(kv.first);
  }

  // ── Phase timing report ────────────────────────────────────────────
  if (config_.profile_phases) {
    double total = t_alloc + t_chain_setup + t_chain_iters + t_merge_post;
    std::cerr << std::fixed << std::setprecision(2)
      << "[profile run_mcmc]"
      << "  n=" << n_train_
      << "  T=" << num_trees_ << "+" << num_trees_variance_
      << "  S=" << n_mcmc << "x" << num_chains << "\n"
      << "  alloc_result:    " << std::setw(8) << t_alloc        << " ms\n"
      << "  make_chain_state:" << std::setw(8) << t_chain_setup  << " ms"
      << "  (" << std::setprecision(1) << t_chain_setup / total * 100 << "%)\n"
      << "  run_chain_iters: " << std::setprecision(2)
      <<                          std::setw(8) << t_chain_iters  << " ms"
      << "  (" << std::setprecision(1) << t_chain_iters / total * 100 << "%)\n"
      << "  merge+post:      " << std::setprecision(2)
      <<                          std::setw(8) << t_merge_post   << " ms"
      << "  (" << std::setprecision(1) << t_merge_post / total * 100 << "%)\n"
      << "  TOTAL run_mcmc:  " << std::setprecision(2)
      <<                          std::setw(8) << total          << " ms\n";
  }
}

// ── run_gfr_result ───────────────────────────────────────────────────────────

void BARTSampler::run_gfr_result(BARTResult* result)
{
  if (n_gfr_stored_ == 0)
    Log::Fatal("run_gfr_result: run_gfr() must be called first.");

  const int n_gfr = n_gfr_stored_;

  // Move GFR forest/RFX containers into result.
  result->forest_container = std::move(gfr_mean_fc_);
  if (has_variance_forest_)
    result->variance_forest_container = std::move(gfr_var_fc_);
  if (has_rfx_)
    result->rfx_container = std::move(gfr_rfx_fc_);

  // Scalar samples (rescaled to original scale where applicable).
  if (has_mean_forest_ && !is_probit_ && !is_cloglog_ && config_.sample_sigma2_global) {
    result->sigma2_global_samples.resize(n_gfr);
    for (int i = 0; i < n_gfr; i++)
      result->sigma2_global_samples[i] = gfr_sigma2_seeds_[i] * y_std_ * y_std_;
  }
  if (has_mean_forest_ && !is_cloglog_ && config_.sample_sigma2_leaf)
    result->leaf_scale_samples.assign(gfr_leaf_scale_seeds_.begin(),
                                      gfr_leaf_scale_seeds_.end());
  if (is_cloglog_)
    result->cloglog_cutpoint_samples.assign(gfr_cloglog_cutpoint_seeds_.begin(),
                                            gfr_cloglog_cutpoint_seeds_.end());

  // Helper: predict from a forest container over a dataset, returning a flat
  // n_obs × n_gfr column-major matrix.  Returns empty if no mean forest.
  auto predict_fc = [&](ForestContainer* fc, ForestDataset& ds, int n_obs)
      -> std::vector<double> {
    if (!fc || !has_mean_forest_) return {};
    return fc->Predict(ds);
  };

  // Helper: predict from rfx container over a dataset.
  auto predict_rfx = [&](RandomEffectsContainer* rc,
                          RandomEffectsDataset& ds, int n_obs)
      -> std::vector<double> {
    if (!rc) return {};
    LabelMapper lm;
    lm.LoadFromLabelMap(rfx_tracker_->GetLabelMap());
    std::vector<double> out(static_cast<size_t>(n_obs) * n_gfr, 0.0);
    rc->Predict(ds, lm, out);
    return out;
  };

  // Helper: fill one prediction column from forest + rfx raw values.
  auto fill_yhat = [&](double* dst, int n_obs,
                        const std::vector<double>& fc_raw,
                        const std::vector<double>& rfx_raw,
                        int col, bool have_fc, bool have_rfx) {
    if (have_fc && have_rfx) {
      const double* fc = fc_raw.data()  + static_cast<size_t>(col) * n_obs;
      const double* rc = rfx_raw.data() + static_cast<size_t>(col) * n_obs;
      for (int j = 0; j < n_obs; j++) dst[j] = (fc[j] + rc[j]) * y_std_ + y_bar_;
    } else if (have_fc) {
      const double* fc = fc_raw.data() + static_cast<size_t>(col) * n_obs;
      for (int j = 0; j < n_obs; j++) dst[j] = fc[j] * y_std_ + y_bar_;
    } else if (have_rfx) {
      const double* rc = rfx_raw.data() + static_cast<size_t>(col) * n_obs;
      for (int j = 0; j < n_obs; j++) dst[j] = rc[j] * y_std_ + y_bar_;
    } else {
      std::fill(dst, dst + n_obs, y_bar_);
    }
  };

  // ── Training predictions ────────────────────────────────────────────────────
  result->y_hat_train.assign(static_cast<size_t>(n_train_) * n_gfr, 0.0);
  {
    auto fc_raw  = predict_fc(result->forest_container.get(), dataset_train_, n_train_);
    auto rfx_raw = has_rfx_ ? predict_rfx(result->rfx_container.get(),
                                          *rfx_dataset_train_, n_train_)
                            : std::vector<double>{};
    bool have_fc  = has_mean_forest_ && !fc_raw.empty();
    bool have_rfx = has_rfx_ && !rfx_raw.empty();
    for (int col = 0; col < n_gfr; col++)
      fill_yhat(result->y_hat_train.data() + static_cast<size_t>(col) * n_train_,
                n_train_, fc_raw, rfx_raw, col, have_fc, have_rfx);
  }

  // Variance forest training predictions.
  if (has_variance_forest_) {
    result->sigma2_x_hat_train.assign(static_cast<size_t>(n_train_) * n_gfr, 0.0);
    std::vector<double> vraw =
        result->variance_forest_container->Predict(dataset_train_);
    for (int col = 0; col < n_gfr; col++) {
      const double* vc = vraw.data() + static_cast<size_t>(col) * n_train_;
      double* dst = result->sigma2_x_hat_train.data() + static_cast<size_t>(col) * n_train_;
      for (int j = 0; j < n_train_; j++)
        dst[j] = std::exp(vc[j]) * y_std_ * y_std_;
    }
  }

  // ── Test predictions ────────────────────────────────────────────────────────
  if (has_test_) {
    result->y_hat_test.assign(static_cast<size_t>(n_test_) * n_gfr, 0.0);
    {
      auto fc_raw  = predict_fc(result->forest_container.get(), dataset_test_, n_test_);
      auto rfx_raw = (has_rfx_ && has_rfx_test_)
          ? predict_rfx(result->rfx_container.get(), *rfx_dataset_test_, n_test_)
          : std::vector<double>{};
      bool have_fc  = has_mean_forest_ && !fc_raw.empty();
      bool have_rfx = has_rfx_ && has_rfx_test_ && !rfx_raw.empty();
      for (int col = 0; col < n_gfr; col++)
        fill_yhat(result->y_hat_test.data() + static_cast<size_t>(col) * n_test_,
                  n_test_, fc_raw, rfx_raw, col, have_fc, have_rfx);
    }

    if (has_variance_forest_) {
      result->sigma2_x_hat_test.assign(static_cast<size_t>(n_test_) * n_gfr, 0.0);
      std::vector<double> vraw =
          result->variance_forest_container->Predict(dataset_test_);
      for (int col = 0; col < n_gfr; col++) {
        const double* vc = vraw.data() + static_cast<size_t>(col) * n_test_;
        double* dst = result->sigma2_x_hat_test.data() + static_cast<size_t>(col) * n_test_;
        for (int j = 0; j < n_test_; j++)
          dst[j] = std::exp(vc[j]) * y_std_ * y_std_;
      }
    }
  }

  // ── RFX metadata ────────────────────────────────────────────────────────────
  if (has_rfx_) {
    result->rfx_num_groups     = rfx_num_groups_;
    result->rfx_num_components = rfx_num_components_;
    for (const auto& kv : rfx_tracker_->GetLabelMap())
      result->rfx_group_ids.push_back(kv.first);
  }

  // ── Result metadata ─────────────────────────────────────────────────────────
  result->n_train           = n_train_;
  result->n_test            = n_test_;
  result->num_total_samples = n_gfr;
  result->num_chains        = 1;
  result->y_bar             = y_bar_;
  result->y_std             = y_std_;
}

// ── BARTSamplerFit ────────────────────────────────────────────────────────

void BARTSamplerFit(BARTResult*        result_ptr,
                    const BARTConfig&  config,
                    const BARTData&    data,
                    const std::string& previous_model_json)
{
  if (result_ptr == nullptr)
    Log::Fatal("BARTSamplerFit: result pointer must not be null.");
  if (!previous_model_json.empty())
    Log::Fatal("BARTSamplerFit: warm-start from previous_model_json is not yet supported.");
  if (config.num_gfr == 0 && config.num_mcmc == 0)
    Log::Fatal("BARTSamplerFit: num_gfr + num_mcmc must be > 0.");
  if (config.num_mcmc > 0 && config.num_chains > config.num_gfr && config.num_gfr > 0)
    Log::Fatal("BARTSamplerFit: num_chains > num_gfr; not enough GFR samples to seed "
               "each chain independently.");

  BARTSampler sampler(config, data);
  if (config.num_gfr > 0)
    sampler.run_gfr(config.num_gfr);
  if (config.num_mcmc > 0)
    sampler.run_mcmc(config.num_mcmc, result_ptr, config.keep_every);
  else if (config.num_gfr > 0) {
    sampler.run_gfr_result(result_ptr);
  }
}

} // namespace StochTree
