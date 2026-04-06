/*!
 * Copyright (c) 2026 stochtree authors. All rights reserved.
 *
 * Implementation of the high-level C++ dispatch API (RFC 0004).
 *
 * Stage 1 supports:
 *   - Identity link, constant leaf BART (the common case)
 *   - Observation weights
 *   - GFR warm-start + MCMC sampling
 *   - Global and leaf variance sampling
 *   - Multi-chain with GFR initialization
 *
 * Stage 2 adds:
 *   - Probit link (Albert-Chib latent variable augmentation)
 *
 * Unsupported features call Log::Fatal with a clear message.
 * They will be added in subsequent stages (see RFC 0004 migration sequence).
 */
#include <stochtree/bart.h>

#include <stochtree/container.h>
#include <stochtree/data.h>
#include <stochtree/ensemble.h>
#include <stochtree/leaf_model.h>
#include <stochtree/log.h>
#include <stochtree/normal_sampler.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/prior.h>
#include <stochtree/probit.h>
#include <stochtree/tree_sampler.h>
#include <stochtree/variance_model.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace StochTree {

// ── Internal helpers ───────────────────────────────────────────────────────

static double VecMean(const double* data, int n) {
  double s = 0.0;
  for (int i = 0; i < n; i++) s += data[i];
  return s / static_cast<double>(n);
}

static double VecVar(const double* data, int n, double mean) {
  double s = 0.0;
  for (int i = 0; i < n; i++) {
    double d = data[i] - mean;
    s += d * d;
  }
  return s / static_cast<double>(n);
}

// ── BARTFit ────────────────────────────────────────────────────────────────

void BARTFit(BARTResult*        result_ptr,
             const BARTConfig&  config,
             const BARTData&    data,
             const std::string& previous_model_json)
{
  if (result_ptr == nullptr)
    Log::Fatal("BARTFit: result pointer must not be null.");

  // ── Feature gates (stubs for unsupported features) ────────────────
  if (config.link_function == LinkFunction::Cloglog)
    Log::Fatal("BARTFit: cloglog link is not yet supported in the C++ dispatch layer.");
  if (config.leaf_model == LeafModel::MultivariateRegression && config.sample_sigma2_leaf)
    Log::Fatal("BARTFit: leaf scale sampling is not supported for multivariate leaf regression.");
  if (config.include_variance_forest && config.num_trees_variance <= 0)
    Log::Fatal("BARTFit: include_variance_forest=true requires num_trees_variance > 0.");
  if (config.rfx_model_spec != RFXModelSpec::None)
    Log::Fatal("BARTFit: random effects are not yet supported in the C++ dispatch layer.");
  if (!previous_model_json.empty())
    Log::Fatal("BARTFit: warm-start from previous_model_json is not yet supported "
               "in the C++ dispatch layer.");

  // ── Input validation ───────────────────────────────────────────────
  if (data.X_train == nullptr || data.y_train == nullptr)
    Log::Fatal("BARTFit: X_train and y_train must not be null.");
  if (data.n_train <= 0 || data.p <= 0)
    Log::Fatal("BARTFit: n_train and p must be positive.");
  if (config.num_gfr == 0 && config.num_mcmc == 0)
    Log::Fatal("BARTFit: num_gfr + num_mcmc must be > 0.");
  if (config.num_chains < 1)
    Log::Fatal("BARTFit: num_chains must be >= 1.");
  if (config.num_mcmc > 0 && config.num_chains > config.num_gfr && config.num_gfr > 0)
    Log::Fatal("BARTFit: num_chains > num_gfr; not enough GFR samples to seed "
               "each chain independently.");
  if (!config.variable_weights_mean.empty() &&
      static_cast<int>(config.variable_weights_mean.size()) != data.p)
    Log::Fatal("BARTFit: variable_weights_mean length must equal p.");

  // ── Leaf regression validation ─────────────────────────────────────
  bool is_leaf_regression  = (config.leaf_model != LeafModel::Constant);
  bool is_multivariate     = (config.leaf_model == LeafModel::MultivariateRegression);
  if (is_leaf_regression) {
    if (data.basis_train == nullptr || data.basis_dim <= 0)
      Log::Fatal("BARTFit: basis_train and basis_dim > 0 required for leaf regression.");
  }

  int n_train  = data.n_train;
  int p        = data.p;
  int n_test   = data.n_test;
  int basis_dim = data.basis_dim;
  bool has_test          = (n_test > 0 && data.X_test != nullptr);
  bool is_probit         = (config.link_function == LinkFunction::Probit);
  bool has_variance_forest = config.include_variance_forest && config.num_trees_variance > 0;
  bool has_mean_forest     = (config.num_trees > 0);
  int  num_trees_variance  = config.num_trees_variance;

  // ── Standardize y (or set probit-scale intercept) ──────────────────
  double y_bar = 0.0, y_std = 1.0;
  std::vector<double> resid_vec(n_train);
  if (is_probit) {
    // Probit: y_bar = Phi^{-1}(mean(y)); y_std fixed at 1 (Albert-Chib).
    // Initial pseudo-outcome = y - mean(y); mean ≈ 0 drives init_val to 0.
    double mean_y = VecMean(data.y_train, n_train);
    y_bar = norm_inv_cdf(mean_y);
    y_std = 1.0;
    for (int i = 0; i < n_train; i++)
      resid_vec[i] = data.y_train[i] - mean_y;
  } else if (config.standardize) {
    y_bar = VecMean(data.y_train, n_train);
    double v = VecVar(data.y_train, n_train, y_bar);
    y_std = (v > 1e-14) ? std::sqrt(v) : 1.0;
    for (int i = 0; i < n_train; i++)
      resid_vec[i] = (data.y_train[i] - y_bar) / y_std;
  } else {
    for (int i = 0; i < n_train; i++)
      resid_vec[i] = data.y_train[i];
  }

  // ── Prior calibration (auto values when sentinel <0) ───────────────
  double resid_var = VecVar(resid_vec.data(), n_train, VecMean(resid_vec.data(), n_train));

  double a_global = config.a_global >= 0.0 ? config.a_global : 0.0;
  double b_global = config.b_global >= 0.0 ? config.b_global : 0.0;
  double a_leaf   = config.a_leaf;
  // Probit: sigma2 is fixed at 1 (Albert-Chib model); use 1/num_trees as the
  // leaf calibration reference instead of data-driven resid_var.
  double leaf_ref   = is_probit ? 1.0 : resid_var;
  double b_leaf     = config.b_leaf    > 0.0 ? config.b_leaf    : leaf_ref / config.num_trees;
  double leaf_scale = config.leaf_scale > 0.0 ? config.leaf_scale : leaf_ref / config.num_trees;
  double current_sigma2 = is_probit ? 1.0 :
      (config.sigma2_init > 0.0 ? config.sigma2_init : resid_var);

  // ── Variance forest prior calibration ─────────────────────────────
  // a_forest / b_forest: IG(a, b) prior on each variance-forest leaf.
  // Auto-calibrated from num_trees_variance when config values are ≤ 0.
  // leaf_init_sigma2: initial per-observation variance (sigma2 scale),
  //   stored in log scale divided across trees so that exp(sum) = init_sigma2.
  double a_forest = 1.0, b_forest = 1.0, var_leaf_init = 0.0;
  if (has_variance_forest) {
    constexpr double a_0 = 1.5;
    a_forest = config.a_forest > 0.0 ? config.a_forest
                                     : num_trees_variance / (a_0 * a_0) + 0.5;
    b_forest = config.b_forest > 0.0 ? config.b_forest
                                     : num_trees_variance / (a_0 * a_0);
    double init_sigma2 = config.variance_forest_leaf_init > 0.0
        ? config.variance_forest_leaf_init : 0.6 * resid_var;
    var_leaf_init = std::log(init_sigma2) / num_trees_variance;
  }

  // ── Binary label vector for probit ────────────────────────────────
  std::vector<int> y_int;
  if (is_probit) {
    y_int.resize(n_train);
    for (int i = 0; i < n_train; i++)
      y_int[i] = (data.y_train[i] > 0.5) ? 1 : 0;
  }

  // ── Feature types ──────────────────────────────────────────────────
  std::vector<FeatureType> feature_types(p, FeatureType::kNumeric);
  if (data.feature_types != nullptr) {
    for (int i = 0; i < p; i++)
      feature_types[i] = static_cast<FeatureType>(data.feature_types[i]);
  }

  // ── Variable weights ───────────────────────────────────────────────
  std::vector<double> variable_weights;
  if (!config.variable_weights_mean.empty()) {
    variable_weights = config.variable_weights_mean;
  } else {
    variable_weights.assign(p, 1.0 / p);
  }

  std::vector<double> variable_weights_variance;
  if (has_variance_forest) {
    if (!config.variable_weights_variance.empty()) {
      variable_weights_variance = config.variable_weights_variance;
    } else {
      variable_weights_variance.assign(p, 1.0 / p);
    }
  }

  // ── Sweep indices (update all trees each iteration) ────────────────
  std::vector<int> sweep_indices(config.num_trees);
  std::iota(sweep_indices.begin(), sweep_indices.end(), 0);

  std::vector<int> variance_sweep_indices;
  if (has_variance_forest) {
    variance_sweep_indices.resize(num_trees_variance);
    std::iota(variance_sweep_indices.begin(), variance_sweep_indices.end(), 0);
  }

  // ── RNG ────────────────────────────────────────────────────────────
  std::mt19937 rng(
      config.random_seed >= 0
          ? static_cast<unsigned>(config.random_seed)
          : std::random_device{}());

  // ── Forest dataset ─────────────────────────────────────────────────
  // AddCovariates: row_major=false means column-major, matching Eigen default.
  ForestDataset dataset_train;
  dataset_train.AddCovariates(
      const_cast<double*>(data.X_train), n_train, p, /*row_major=*/false);
  if (data.weights != nullptr)
    dataset_train.AddVarianceWeights(
        const_cast<double*>(data.weights), n_train);

  ForestDataset dataset_test;
  if (has_test)
    dataset_test.AddCovariates(
        const_cast<double*>(data.X_test), n_test, p, /*row_major=*/false);

  // ── Leaf regression basis ──────────────────────────────────────────
  if (is_leaf_regression) {
    dataset_train.AddBasis(
        const_cast<double*>(data.basis_train), n_train, basis_dim, /*row_major=*/false);
    if (has_test && data.basis_test != nullptr)
      dataset_test.AddBasis(
          const_cast<double*>(data.basis_test), n_test, basis_dim, /*row_major=*/false);
  }

  // ── Residual ───────────────────────────────────────────────────────
  ColumnVector residual(resid_vec.data(), n_train);

  // ── Sampler objects ────────────────────────────────────────────────
  int num_trees = config.num_trees;
  int cutpoint_grid_size = config.cutpoint_grid_size > 0
      ? config.cutpoint_grid_size : n_train;
  int num_features_subsample = p;  // all features; subsetting deferred

  // Use size-1 dummies when num_trees == 0 (variance-only model); the objects
  // are never touched because all mean-forest code paths are guarded by has_mean_forest.
  int mean_ctor_trees  = has_mean_forest ? num_trees : 1;
  // Leaf regression: output_dim = basis_dim (multivariate) or 1 (univariate/constant).
  int mean_output_dim  = is_multivariate ? basis_dim : 1;
  bool mean_leaf_const = !is_leaf_regression;
  TreeEnsemble  active_forest(mean_ctor_trees, mean_output_dim, mean_leaf_const, false);
  ForestTracker tracker(dataset_train.GetCovariates(), feature_types, mean_ctor_trees, n_train);
  TreePrior     tree_prior(config.alpha, config.beta, config.min_samples_leaf, config.max_depth);

  GlobalHomoskedasticVarianceModel    global_var_model;
  LeafNodeHomoskedasticVarianceModel  leaf_var_model;

  // ── Sample counts and result allocation ────────────────────────────
  int num_gfr    = config.num_gfr;
  int num_burnin = config.num_burnin;
  int num_mcmc   = config.num_mcmc;
  int num_chains = config.num_chains;
  int keep_every = std::max(1, config.keep_every);

  // When keep_gfr=false and MCMC samples exist, GFR columns are omitted from
  // the result arrays (GFR forests are still stored temporarily for chain seeding
  // and are deleted after the MCMC loop below).
  int num_mcmc_per_chain = num_mcmc;
  int num_stored_gfr = (config.keep_gfr || num_mcmc == 0) ? num_gfr : 0;
  int num_total = num_stored_gfr + num_chains * num_mcmc_per_chain;

  // Reference alias so all `result.X` accesses below are unchanged.
  BARTResult& result = *result_ptr;
  // ForestContainer size-1 dummy when num_trees == 0 (variance-only model).
  result.forest_container = std::make_unique<ForestContainer>(
      mean_ctor_trees, mean_output_dim, mean_leaf_const, /*is_exponentiated=*/false);
  ForestContainer& forest_container = *result.forest_container;
  if (has_variance_forest)
    result.variance_forest_container = std::make_unique<ForestContainer>(
        num_trees_variance, 1, /*is_leaf_constant=*/true, /*is_exponentiated=*/true);
  result.num_total_samples = num_total;
  result.num_chains = num_chains;
  result.n_train = n_train;
  result.n_test = n_test;
  result.y_bar = y_bar;
  result.y_std = y_std;

  result.y_hat_train.resize(static_cast<size_t>(n_train) * num_total, 0.0);
  if (has_test)
    result.y_hat_test.resize(static_cast<size_t>(n_test) * num_total, 0.0);
  // Probit fixes sigma2 at 1 (Albert-Chib); never allocate the sample array.
  if (!is_probit && config.sample_sigma2_global)
    result.sigma2_global_samples.resize(num_total, 0.0);
  if (config.sample_sigma2_leaf)
    result.leaf_scale_samples.resize(num_total, 0.0);
  if (has_variance_forest) {
    result.sigma2_x_hat_train.resize(static_cast<size_t>(n_train) * num_total, 0.0);
    if (has_test)
      result.sigma2_x_hat_test.resize(static_cast<size_t>(n_test) * num_total, 0.0);
  }

  // ── Helper: write predictions from tracker into result columns ─────
  // col: which sample column (0-indexed) in y_hat_train / y_hat_test.
  // When num_trees == 0 (variance-only model) the mean prediction is y_bar.
  auto cache_train_predictions = [&](int col) {
    double* dst = result.y_hat_train.data() + static_cast<size_t>(col) * n_train;
    if (has_mean_forest) {
      for (int j = 0; j < n_train; j++)
        dst[j] = tracker.GetSamplePrediction(j) * y_std + y_bar;
    } else {
      std::fill(dst, dst + n_train, y_bar);
    }
  };
  auto cache_test_predictions = [&](int col) {
    if (!has_test) return;
    double* dst = result.y_hat_test.data() + static_cast<size_t>(col) * n_test;
    if (has_mean_forest && forest_container.NumSamples() > 0) {
      int last = forest_container.NumSamples() - 1;
      std::vector<double> test_preds = forest_container.PredictRaw(dataset_test, last);
      for (int j = 0; j < n_test; j++)
        dst[j] = test_preds[j] * y_std + y_bar;
    } else {
      std::fill(dst, dst + n_test, y_bar);
    }
  };

  // ── Variance forest sampler objects ───────────────────────────────
  // Declared here (before GFR loop) so lambdas can capture them.
  TreeEnsemble active_forest_variance(num_trees_variance > 0 ? num_trees_variance : 1,
                                      1, true, /*is_exponentiated=*/true);
  ForestTracker variance_tracker(dataset_train.GetCovariates(), feature_types,
                                 num_trees_variance > 0 ? num_trees_variance : 1, n_train);
  TreePrior variance_prior(config.alpha_variance, config.beta_variance,
                           config.min_samples_leaf_variance, config.max_depth_variance);

  auto cache_sigma2x_train = [&](int col) {
    if (!has_variance_forest) return;
    double* dst = result.sigma2_x_hat_train.data() + static_cast<size_t>(col) * n_train;
    // variance_tracker stores log-scale sum; y_std^2 scales back to original.
    for (int j = 0; j < n_train; j++)
      dst[j] = std::exp(variance_tracker.GetSamplePrediction(j)) * y_std * y_std;
  };
  auto cache_sigma2x_test = [&](int col) {
    if (!has_variance_forest || !has_test) return;
    int last = result.variance_forest_container->NumSamples() - 1;
    // PredictRaw applies exp() because is_exponentiated=true on the container.
    // PredictRaw returns the log-scale sum regardless of is_exponentiated;
    // apply exp() manually to get the variance-scale prediction.
    std::vector<double> test_preds =
        result.variance_forest_container->PredictRaw(dataset_test, last);
    double* dst = result.sigma2_x_hat_test.data() + static_cast<size_t>(col) * n_test;
    for (int j = 0; j < n_test; j++)
      dst[j] = std::exp(test_preds[j]) * y_std * y_std;
  };

  // ── Initialize mean forest ─────────────────────────────────────────
  // Skipped when num_trees == 0 (variance-only model).
  // Leaf regression: initialize all params to 0 (constant uses mean/num_trees).
  double init_val = (has_mean_forest && !is_leaf_regression)
      ? VecMean(resid_vec.data(), n_train) / num_trees : 0.0;
  if (has_mean_forest) {
    if (is_multivariate) {
      std::vector<double> zero_leaf(basis_dim, 0.0);
      active_forest.SetLeafVector(zero_leaf);
    } else {
      active_forest.SetLeafValue(init_val);
    }
    UpdateResidualEntireForest(
        tracker, dataset_train, residual, &active_forest,
        /*requires_basis=*/is_leaf_regression, std::minus<double>());
  }

  // ── Initialize variance forest ─────────────────────────────────────
  // Each tree root = log(init_sigma2) / num_trees_variance so that
  // exp(sum_of_all_roots) = init_sigma2 (the initial per-obs variance).
  // We seed VarWeights to 1.0 first; ReconstituteFromForest then updates
  // them to the correct init_sigma2 value via the incremental log-weight update.
  if (has_variance_forest) {
    active_forest_variance.SetLeafValue(var_leaf_init);
    std::vector<double> ones(n_train, 1.0);
    dataset_train.AddVarianceWeights(ones.data(), n_train);
    variance_tracker.ReconstituteFromForest(
        active_forest_variance, dataset_train, residual, /*is_mean_model=*/false);
  }

  // When keep_gfr=false and there are MCMC samples, GFR forests are still
  // needed to seed the chains, but we skip writing their predictions or scalar
  // samples into the result arrays (num_stored_gfr == 0 in that case).
  bool store_gfr = (num_stored_gfr > 0);

  // ── GFR scratch buffers ────────────────────────────────────────────
  // When keep_gfr=false (store_gfr=false) and num_mcmc > 0, GFR samples are
  // only needed to seed the MCMC chains.  Rather than appending them to the
  // result containers and deleting afterwards, we route them into private
  // scratch containers that are discarded once seeding is complete.
  // This mirrors how gfr_sigma2_seeds / gfr_leaf_scale_seeds work for scalars.
  bool use_gfr_scratch = (!store_gfr && num_mcmc > 0 && num_gfr > 0);

  std::unique_ptr<ForestContainer> gfr_mean_scratch;
  std::unique_ptr<ForestContainer> gfr_var_scratch;
  if (use_gfr_scratch) {
    gfr_mean_scratch = std::make_unique<ForestContainer>(
        mean_ctor_trees, mean_output_dim, mean_leaf_const, /*is_exponentiated=*/false);
    if (has_variance_forest)
      gfr_var_scratch = std::make_unique<ForestContainer>(
          num_trees_variance, 1, /*is_leaf_constant=*/true, /*is_exponentiated=*/true);
  }

  // Reference aliases: GFR loops write to scratch; when keep_gfr=true they
  // write directly to the result containers (no scratch needed).
  ForestContainer& gfr_mean_fc = use_gfr_scratch ? *gfr_mean_scratch : forest_container;
  ForestContainer* gfr_var_fc  = has_variance_forest
      ? (use_gfr_scratch ? gfr_var_scratch.get() : result.variance_forest_container.get())
      : nullptr;

  // Scalar seed buffers: always populated during GFR when num_mcmc > 0 so that
  // chain seeding can restore the correct sigma2 / leaf_scale state even when
  // keep_gfr=false (i.e. when the result arrays have no GFR-indexed slots).
  // These are internal only and are not exposed in BARTResult.
  std::vector<double> gfr_sigma2_seeds;
  std::vector<double> gfr_leaf_scale_seeds;
  if (num_mcmc > 0 && num_gfr > 0) {
    gfr_sigma2_seeds.resize(num_gfr, current_sigma2);
    gfr_leaf_scale_seeds.resize(num_gfr, leaf_scale);
  }

  // ── GFR loop ───────────────────────────────────────────────────────
  for (int i = 0; i < num_gfr; i++) {
    if (has_mean_forest) {
      // Probit: sample latent z | forests before the forest step.
      if (is_probit)
        sample_probit_latent_outcome(residual, tracker, y_int.data(), n_train, y_bar, rng);

      if (config.leaf_model == LeafModel::UnivariateRegression) {
        auto lm = GaussianUnivariateRegressionLeafModel(leaf_scale);
        GFRSampleOneIter<GaussianUnivariateRegressionLeafModel, GaussianUnivariateRegressionSuffStat>(
            active_forest, tracker, gfr_mean_fc, lm,
            dataset_train, residual, tree_prior, rng,
            variable_weights, sweep_indices, current_sigma2,
            feature_types, cutpoint_grid_size,
            /*keep_forest=*/true, /*pre_initialized=*/true, /*backfitting=*/true,
            num_features_subsample, config.num_threads);
      } else if (config.leaf_model == LeafModel::MultivariateRegression) {
        Eigen::MatrixXd Sigma_0 = Eigen::MatrixXd::Identity(basis_dim, basis_dim) * leaf_scale;
        auto lm = GaussianMultivariateRegressionLeafModel(Sigma_0);
        GFRSampleOneIter<GaussianMultivariateRegressionLeafModel, GaussianMultivariateRegressionSuffStat>(
            active_forest, tracker, gfr_mean_fc, lm,
            dataset_train, residual, tree_prior, rng,
            variable_weights, sweep_indices, current_sigma2,
            feature_types, cutpoint_grid_size,
            /*keep_forest=*/true, /*pre_initialized=*/true, /*backfitting=*/true,
            num_features_subsample, config.num_threads, basis_dim);
      } else {
        auto lm = GaussianConstantLeafModel(leaf_scale);
        GFRSampleOneIter<GaussianConstantLeafModel, GaussianConstantSuffStat>(
            active_forest, tracker, gfr_mean_fc, lm,
            dataset_train, residual, tree_prior, rng,
            variable_weights, sweep_indices, current_sigma2,
            feature_types, cutpoint_grid_size,
            /*keep_forest=*/true, /*pre_initialized=*/true, /*backfitting=*/true,
            num_features_subsample, config.num_threads);
      }
    }

    // Cache mean predictions (column i) only when storing GFR samples.
    // For probit, predictions are on the probit scale (forest_pred + y_bar).
    // When num_trees == 0, fills y_bar (no mean forest).
    if (store_gfr) {
      cache_train_predictions(i);
      cache_test_predictions(i);
    }

    // Sample variance forest (GFR step), then update per-obs variance weights.
    if (has_variance_forest) {
      LogLinearVarianceLeafModel var_leaf_model(a_forest, b_forest);
      GFRSampleOneIter<LogLinearVarianceLeafModel, LogLinearVarianceSuffStat>(
          active_forest_variance, variance_tracker,
          *gfr_var_fc, var_leaf_model,
          dataset_train, residual, variance_prior, rng,
          variable_weights_variance, variance_sweep_indices, current_sigma2,
          feature_types, cutpoint_grid_size,
          /*keep_forest=*/true,
          /*pre_initialized=*/true,
          /*backfitting=*/false,  // variance model: update VarWeights, not residual
          num_features_subsample,
          config.num_threads);
      if (store_gfr) {
        cache_sigma2x_train(i);
        cache_sigma2x_test(i);
      }
    }

    // Sample scalar variance parameters (mean forest only).
    // Probit: sigma2 is fixed at 1 (Albert-Chib); skip global variance sampling.
    // Under the identity model y_i ~ N(mu_i, sigma^2/w_i), the posterior for
    // sigma^2 uses weighted residuals when observation weights are present.
    if (has_mean_forest && !is_probit && config.sample_sigma2_global) {
      if (dataset_train.HasVarWeights()) {
        current_sigma2 = global_var_model.SampleVarianceParameter(
            residual.GetData(), dataset_train.GetVarWeights(), a_global, b_global, rng);
      } else {
        current_sigma2 = global_var_model.SampleVarianceParameter(
            residual.GetData(), a_global, b_global, rng);
      }
      if (store_gfr)
        result.sigma2_global_samples[i] = current_sigma2 * y_std * y_std;
      if (!gfr_sigma2_seeds.empty())
        gfr_sigma2_seeds[i] = current_sigma2;
    }
    if (has_mean_forest && config.sample_sigma2_leaf) {
      leaf_scale = leaf_var_model.SampleVarianceParameter(
          &active_forest, a_leaf, b_leaf, rng);
      if (store_gfr)
        result.leaf_scale_samples[i] = leaf_scale;
      if (!gfr_leaf_scale_seeds.empty())
        gfr_leaf_scale_seeds[i] = leaf_scale;
    }
  }

  // ── MCMC loop ──────────────────────────────────────────────────────
  if (num_burnin + num_mcmc > 0) {
    for (int chain = 0; chain < num_chains; chain++) {
      double chain_sigma2     = current_sigma2;
      double chain_leaf_scale = leaf_scale;

      if (num_gfr > 0) {
        // Seed from a GFR ensemble (last num_chains GFR samples, one per chain).
        int forest_ind = num_gfr - chain - 1;
        if (has_mean_forest) {
          active_forest.ReconstituteFromForest(
              *gfr_mean_fc.GetEnsemble(forest_ind));
          tracker.ReconstituteFromForest(
              active_forest, dataset_train, residual, /*is_mean_model=*/true);
          // Restore scalar variance state from GFR seed buffers.
          if (config.sample_sigma2_global && !gfr_sigma2_seeds.empty())
            chain_sigma2 = gfr_sigma2_seeds[forest_ind];
          if (config.sample_sigma2_leaf && !gfr_leaf_scale_seeds.empty())
            chain_leaf_scale = gfr_leaf_scale_seeds[forest_ind];
        }
        // Restore variance forest state from the same GFR sample.
        if (has_variance_forest) {
          active_forest_variance.ReconstituteFromForest(
              *gfr_var_fc->GetEnsemble(forest_ind));
          variance_tracker.ReconstituteFromForest(
              active_forest_variance, dataset_train, residual, /*is_mean_model=*/false);
        }
      } else {
        // No GFR: reset every tree to a single root stump, then seed the tracker.
        if (has_mean_forest) {
          active_forest.ResetRoot();
          if (is_multivariate) {
            std::vector<double> zero_leaf(basis_dim, 0.0);
            active_forest.SetLeafVector(zero_leaf);
          } else {
            active_forest.SetLeafValue(init_val);
          }
          tracker.ReconstituteFromForest(
              active_forest, dataset_train, residual, /*is_mean_model=*/true);
        }
        if (has_variance_forest) {
          active_forest_variance.ResetRoot();
          active_forest_variance.SetLeafValue(var_leaf_init);
          variance_tracker.ReconstituteFromForest(
              active_forest_variance, dataset_train, residual, /*is_mean_model=*/false);
        }
      }

      int mcmc_kept = 0;
      int total_mcmc_iters = num_burnin + num_mcmc * keep_every;
      for (int i = 0; i < total_mcmc_iters; i++) {
        bool is_burnin = (i < num_burnin);
        bool is_kept   = !is_burnin && ((i - num_burnin) % keep_every == 0);

        if (has_mean_forest) {
          // Probit: sample latent z | forests before the forest step.
          if (is_probit)
            sample_probit_latent_outcome(residual, tracker, y_int.data(), n_train, y_bar, rng);

          if (config.leaf_model == LeafModel::UnivariateRegression) {
            auto lm = GaussianUnivariateRegressionLeafModel(chain_leaf_scale);
            MCMCSampleOneIter<GaussianUnivariateRegressionLeafModel, GaussianUnivariateRegressionSuffStat>(
                active_forest, tracker, forest_container, lm,
                dataset_train, residual, tree_prior, rng,
                variable_weights, sweep_indices, chain_sigma2,
                /*keep_forest=*/is_kept, /*pre_initialized=*/true, /*backfitting=*/true,
                config.num_threads);
          } else if (config.leaf_model == LeafModel::MultivariateRegression) {
            Eigen::MatrixXd Sigma_0 = Eigen::MatrixXd::Identity(basis_dim, basis_dim) * chain_leaf_scale;
            auto lm = GaussianMultivariateRegressionLeafModel(Sigma_0);
            MCMCSampleOneIter<GaussianMultivariateRegressionLeafModel, GaussianMultivariateRegressionSuffStat>(
                active_forest, tracker, forest_container, lm,
                dataset_train, residual, tree_prior, rng,
                variable_weights, sweep_indices, chain_sigma2,
                /*keep_forest=*/is_kept, /*pre_initialized=*/true, /*backfitting=*/true,
                config.num_threads, basis_dim);
          } else {
            auto lm = GaussianConstantLeafModel(chain_leaf_scale);
            MCMCSampleOneIter<GaussianConstantLeafModel, GaussianConstantSuffStat>(
                active_forest, tracker, forest_container, lm,
                dataset_train, residual, tree_prior, rng,
                variable_weights, sweep_indices, chain_sigma2,
                /*keep_forest=*/is_kept, /*pre_initialized=*/true, /*backfitting=*/true,
                config.num_threads);
          }
        }

        // Sample variance forest (MCMC step), then update per-obs variance weights.
        if (has_variance_forest) {
          LogLinearVarianceLeafModel var_leaf_model_mcmc(a_forest, b_forest);
          MCMCSampleOneIter<LogLinearVarianceLeafModel, LogLinearVarianceSuffStat>(
              active_forest_variance, variance_tracker,
              *result.variance_forest_container, var_leaf_model_mcmc,
              dataset_train, residual, variance_prior, rng,
              variable_weights_variance, variance_sweep_indices, chain_sigma2,
              /*keep_forest=*/is_kept,
              /*pre_initialized=*/true,
              /*backfitting=*/false,  // variance model: update VarWeights, not residual
              config.num_threads);
        }

        // Probit: sigma2 fixed at 1, skip global variance sampling.
        if (has_mean_forest && !is_probit && config.sample_sigma2_global) {
          if (dataset_train.HasVarWeights()) {
            chain_sigma2 = global_var_model.SampleVarianceParameter(
                residual.GetData(), dataset_train.GetVarWeights(), a_global, b_global, rng);
          } else {
            chain_sigma2 = global_var_model.SampleVarianceParameter(
                residual.GetData(), a_global, b_global, rng);
          }
        }
        if (has_mean_forest && config.sample_sigma2_leaf)
          chain_leaf_scale = leaf_var_model.SampleVarianceParameter(
              &active_forest, a_leaf, b_leaf, rng);

        if (is_kept) {
          int gfr_offset = store_gfr ? num_gfr : 0;
          int col = gfr_offset + chain * num_mcmc_per_chain + mcmc_kept;
          cache_train_predictions(col);
          cache_test_predictions(col);
          if (has_variance_forest) {
            cache_sigma2x_train(col);
            cache_sigma2x_test(col);
          }
          if (has_mean_forest && !is_probit && config.sample_sigma2_global)
            result.sigma2_global_samples[col] = chain_sigma2 * y_std * y_std;
          if (has_mean_forest && config.sample_sigma2_leaf)
            result.leaf_scale_samples[col] = chain_leaf_scale;
          mcmc_kept++;
        }
      }
    }
  }

}


} // namespace StochTree
