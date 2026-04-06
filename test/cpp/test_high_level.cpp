/*!
 * Tests for the high-level C++ dispatch API (RFC 0004).
 *
 * Stage 1 coverage:
 *   - Identity-link constant-leaf BART: shape checks, finite values
 *   - GFR-only (XBART mode)
 *   - MCMC-only (pure MCMC, no warm-start)
 *   - GFR + MCMC warm-start (the typical use case)
 *   - Multi-chain warm-start
 *   - With and without test set
 *   - With and without observation weights
 *   - Variance sampling on/off
 *   - Reproducibility via random seed
 */
#include <gtest/gtest.h>
#include <stochtree/bart.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

namespace {

// ── Synthetic data helpers ─────────────────────────────────────────────────

// Phi(x) = 0.5 * erfc(-x / sqrt(2)) — standard normal CDF.
static double Phi(double x) {
  return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

// Generate a simple linear DGP: y = 2*x1 - x2 + eps, eps ~ N(0, 0.5)
// X is column-major (Fortran order).
struct SyntheticData {
  int n;
  int p;
  std::vector<double> X;  // column-major, n × p
  std::vector<double> y;
};

SyntheticData MakeSyntheticData(int n, int p, unsigned seed = 42) {
  SyntheticData d;
  d.n = n;
  d.p = p;
  d.X.resize(static_cast<size_t>(n) * p);
  d.y.resize(n);

  std::mt19937 rng(seed);
  std::normal_distribution<double> norm(0.0, 1.0);

  // Fill X column-major
  for (int j = 0; j < p; j++)
    for (int i = 0; i < n; i++)
      d.X[j * n + i] = norm(rng);

  // y = 2*X[:,0] - X[:,1] + noise (first two columns)
  std::normal_distribution<double> eps(0.0, 0.5);
  for (int i = 0; i < n; i++) {
    double signal = 2.0 * d.X[0 * n + i];
    if (p > 1) signal -= d.X[1 * n + i];
    d.y[i] = signal + eps(rng);
  }
  return d;
}

// Generate a binary DGP for probit BART.
// True linear predictor: eta_i = x0_i - 0.5*x1_i
// P(y=1 | x) = Phi(eta_i)
// Returns y as 0.0/1.0 doubles (BARTFit thresholds at 0.5 to recover y_int).
SyntheticData MakeBinaryData(int n, int p, unsigned seed = 42) {
  SyntheticData d;
  d.n = n;
  d.p = p;
  d.X.resize(static_cast<size_t>(n) * p);
  d.y.resize(n);

  std::mt19937 rng(seed);
  std::normal_distribution<double> norm(0.0, 1.0);
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  for (int j = 0; j < p; j++)
    for (int i = 0; i < n; i++)
      d.X[j * n + i] = norm(rng);

  for (int i = 0; i < n; i++) {
    double eta = d.X[0 * n + i];
    if (p > 1) eta -= 0.5 * d.X[1 * n + i];
    d.y[i] = (unif(rng) < Phi(eta)) ? 1.0 : 0.0;
  }
  return d;
}

// ── Assertion helpers ──────────────────────────────────────────────────────

bool AllFinite(const std::vector<double>& v) {
  return std::all_of(v.begin(), v.end(), [](double x) { return std::isfinite(x); });
}

double VecMean(const std::vector<double>& v) {
  if (v.empty()) return 0.0;
  return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

// ── Tests ──────────────────────────────────────────────────────────────────

// Simplest smoke test: GFR-only BART, check output shape and finite values.
TEST(BARTFit, GFROnly_Shape_Finite) {
  auto d = MakeSyntheticData(100, 5);

  StochTree::BARTConfig config;
  config.num_trees  = 10;
  config.num_gfr    = 20;
  config.num_burnin = 0;
  config.num_mcmc   = 0;
  config.random_seed = 1;

  StochTree::BARTData data;
  data.X_train = d.X.data();
  data.n_train = d.n;
  data.p       = d.p;
  data.y_train = d.y.data();

  StochTree::BARTResult result;
  StochTree::BARTFit(&result, config, data);

  // num_total_samples = num_gfr + num_chains * num_mcmc = 20 + 0 = 20
  EXPECT_EQ(result.num_total_samples, 20);
  EXPECT_EQ(result.n_train, 100);
  EXPECT_EQ(result.n_test,  0);
  EXPECT_EQ(static_cast<int>(result.y_hat_train.size()), 100 * 20);
  EXPECT_TRUE(result.y_hat_test.empty());
  EXPECT_TRUE(AllFinite(result.y_hat_train));
}

// MCMC-only (no GFR warm-start): check shape and finite values.
TEST(BARTFit, MCMCOnly_Shape_Finite) {
  auto d = MakeSyntheticData(100, 5);

  StochTree::BARTConfig config;
  config.num_trees   = 10;
  config.num_gfr     = 0;
  config.num_burnin  = 10;
  config.num_mcmc    = 50;
  config.num_chains  = 1;
  config.random_seed = 2;

  StochTree::BARTData data;
  data.X_train = d.X.data();
  data.n_train = d.n;
  data.p       = d.p;
  data.y_train = d.y.data();

  StochTree::BARTResult result;
  StochTree::BARTFit(&result, config, data);

  // num_total = 0 (GFR) + 1 chain * 50 (MCMC) = 50
  EXPECT_EQ(result.num_total_samples, 50);
  EXPECT_EQ(static_cast<int>(result.y_hat_train.size()), 100 * 50);
  EXPECT_TRUE(AllFinite(result.y_hat_train));
  // Variance samples
  EXPECT_EQ(static_cast<int>(result.sigma2_global_samples.size()), 50);
  EXPECT_EQ(static_cast<int>(result.leaf_scale_samples.size()), 50);
  EXPECT_TRUE(AllFinite(result.sigma2_global_samples));
  EXPECT_TRUE(AllFinite(result.leaf_scale_samples));
}

// GFR + MCMC warm-start: the typical use case.
TEST(BARTFit, GFR_Plus_MCMC_Shape_Finite) {
  auto d = MakeSyntheticData(200, 10);

  StochTree::BARTConfig config;
  config.num_trees   = 50;
  config.num_gfr     = 5;
  config.num_burnin  = 0;
  config.num_mcmc    = 100;
  config.num_chains  = 1;
  config.random_seed = 3;

  StochTree::BARTData data;
  data.X_train = d.X.data();
  data.n_train = d.n;
  data.p       = d.p;
  data.y_train = d.y.data();

  StochTree::BARTResult result;
  StochTree::BARTFit(&result, config, data);

  // keep_gfr=false (default): num_total = 1 * 100 (MCMC only)
  EXPECT_EQ(result.num_total_samples, 100);
  EXPECT_EQ(static_cast<int>(result.y_hat_train.size()), 200 * 100);
  EXPECT_TRUE(AllFinite(result.y_hat_train));
  EXPECT_TRUE(AllFinite(result.sigma2_global_samples));
  EXPECT_TRUE(AllFinite(result.leaf_scale_samples));
}

// Multi-chain: 3 chains, each seeded from a different GFR ensemble.
TEST(BARTFit, MultiChain_Shape_Finite) {
  auto d = MakeSyntheticData(150, 5);

  StochTree::BARTConfig config;
  config.num_trees   = 20;
  config.num_gfr     = 5;   // >= num_chains
  config.num_burnin  = 5;
  config.num_mcmc    = 50;
  config.num_chains  = 3;
  config.random_seed = 4;

  StochTree::BARTData data;
  data.X_train = d.X.data();
  data.n_train = d.n;
  data.p       = d.p;
  data.y_train = d.y.data();

  StochTree::BARTResult result;
  StochTree::BARTFit(&result, config, data);

  // keep_gfr=false (default): num_total = 3 chains * 50 (MCMC only) = 150
  EXPECT_EQ(result.num_total_samples, 150);
  EXPECT_EQ(static_cast<int>(result.y_hat_train.size()), 150 * 150);
  EXPECT_TRUE(AllFinite(result.y_hat_train));
}

// With test set: y_hat_test must be populated with the right shape.
TEST(BARTFit, WithTestSet_Shape_Finite) {
  int n_train = 200, n_test = 50, p = 5;
  auto d_train = MakeSyntheticData(n_train, p, 10);
  auto d_test  = MakeSyntheticData(n_test,  p, 11);

  StochTree::BARTConfig config;
  config.num_trees   = 20;
  config.num_gfr     = 5;
  config.num_mcmc    = 50;
  config.num_chains  = 1;
  config.random_seed = 5;

  StochTree::BARTData data;
  data.X_train = d_train.X.data();
  data.n_train = n_train;
  data.p       = p;
  data.y_train = d_train.y.data();
  data.X_test  = d_test.X.data();
  data.n_test  = n_test;

  StochTree::BARTResult result;
  StochTree::BARTFit(&result, config, data);

  int num_total = 50;  // keep_gfr=false (default): MCMC only
  EXPECT_EQ(result.num_total_samples, num_total);
  EXPECT_EQ(static_cast<int>(result.y_hat_train.size()), n_train * num_total);
  EXPECT_EQ(static_cast<int>(result.y_hat_test.size()),  n_test  * num_total);
  EXPECT_TRUE(AllFinite(result.y_hat_train));
  EXPECT_TRUE(AllFinite(result.y_hat_test));
}

// Observation weights: run should complete and produce finite values.
TEST(BARTFit, ObservationWeights_Finite) {
  auto d = MakeSyntheticData(100, 5);
  std::vector<double> weights(d.n, 1.0);
  // Down-weight first 10 observations
  for (int i = 0; i < 10; i++) weights[i] = 0.1;

  StochTree::BARTConfig config;
  config.num_trees   = 10;
  config.num_gfr     = 5;
  config.num_mcmc    = 20;
  config.random_seed = 6;

  StochTree::BARTData data;
  data.X_train = d.X.data();
  data.n_train = d.n;
  data.p       = d.p;
  data.y_train = d.y.data();
  data.weights = weights.data();

  StochTree::BARTResult result;
  StochTree::BARTFit(&result, config, data);
  EXPECT_TRUE(AllFinite(result.y_hat_train));
}

// Variance sampling disabled: variance sample arrays should be empty.
TEST(BARTFit, NoVarianceSampling_EmptyArrays) {
  auto d = MakeSyntheticData(100, 5);

  StochTree::BARTConfig config;
  config.num_trees           = 10;
  config.num_gfr             = 5;
  config.num_mcmc            = 20;
  config.sample_sigma2_global = false;
  config.sample_sigma2_leaf   = false;
  config.random_seed          = 7;

  StochTree::BARTData data;
  data.X_train = d.X.data();
  data.n_train = d.n;
  data.p       = d.p;
  data.y_train = d.y.data();

  StochTree::BARTResult result;
  StochTree::BARTFit(&result, config, data);
  EXPECT_TRUE(result.sigma2_global_samples.empty());
  EXPECT_TRUE(result.leaf_scale_samples.empty());
  EXPECT_TRUE(AllFinite(result.y_hat_train));
}

// Reproducibility: same seed produces identical results.
TEST(BARTFit, Reproducibility_SameSeed) {
  auto d = MakeSyntheticData(100, 5);

  StochTree::BARTConfig config;
  config.num_trees   = 10;
  config.num_gfr     = 5;
  config.num_mcmc    = 20;
  config.random_seed = 42;

  StochTree::BARTData data;
  data.X_train = d.X.data();
  data.n_train = d.n;
  data.p       = d.p;
  data.y_train = d.y.data();

  StochTree::BARTResult r1;
  StochTree::BARTFit(&r1, config, data);
  StochTree::BARTResult r2;
  StochTree::BARTFit(&r2, config, data);

  ASSERT_EQ(r1.y_hat_train.size(), r2.y_hat_train.size());
  for (size_t i = 0; i < r1.y_hat_train.size(); i++)
    EXPECT_DOUBLE_EQ(r1.y_hat_train[i], r2.y_hat_train[i]) << "mismatch at i=" << i;
}

// Sanity: predictions on training data should correlate with the DGP signal.
// GFR posterior mean should have lower RMSE than predicting y_bar for all obs.
TEST(BARTFit, TrainingRMSE_BetterThanIntercept) {
  auto d = MakeSyntheticData(500, 5, 99);

  StochTree::BARTConfig config;
  config.num_trees   = 100;
  config.num_gfr     = 10;
  config.num_mcmc    = 0;
  config.random_seed = 100;

  StochTree::BARTData data;
  data.X_train = d.X.data();
  data.n_train = d.n;
  data.p       = d.p;
  data.y_train = d.y.data();

  StochTree::BARTResult result;
  StochTree::BARTFit(&result, config, data);

  int n = d.n, S = result.num_total_samples;
  // Posterior mean prediction: average across all GFR samples
  std::vector<double> yhat_mean(n, 0.0);
  for (int s = 0; s < S; s++)
    for (int i = 0; i < n; i++)
      yhat_mean[i] += result.y_hat_train[s * n + i] / S;

  // RMSE of posterior mean
  double rmse_bart = 0.0;
  for (int i = 0; i < n; i++) {
    double e = yhat_mean[i] - d.y[i];
    rmse_bart += e * e;
  }
  rmse_bart = std::sqrt(rmse_bart / n);

  // Null model RMSE: predict y_bar for everyone
  double y_bar = VecMean(d.y);
  double rmse_null = 0.0;
  for (int i = 0; i < n; i++) {
    double e = y_bar - d.y[i];
    rmse_null += e * e;
  }
  rmse_null = std::sqrt(rmse_null / n);

  EXPECT_LT(rmse_bart, rmse_null)
      << "BART RMSE (" << rmse_bart << ") should be less than null RMSE (" << rmse_null << ")";
}

// Error cases: verify clear error messages for unsupported configurations.
TEST(BARTFit, UnsupportedFeatures_ThrowClear) {
  auto d = MakeSyntheticData(50, 3);

  StochTree::BARTData data;
  data.X_train = d.X.data();
  data.n_train = d.n;
  data.p       = d.p;
  data.y_train = d.y.data();

  // num_gfr = 0 and num_mcmc = 0
  {
    StochTree::BARTConfig config;
    config.num_gfr  = 0;
    config.num_mcmc = 0;
    { StochTree::BARTResult _r; EXPECT_THROW(StochTree::BARTFit(&_r, config, data), std::runtime_error); }
  }

  // num_chains > num_gfr
  {
    StochTree::BARTConfig config;
    config.num_gfr    = 3;
    config.num_mcmc   = 10;
    config.num_chains = 5;
    { StochTree::BARTResult _r; EXPECT_THROW(StochTree::BARTFit(&_r, config, data), std::runtime_error); }
  }

  // MultivariateRegression + sample_sigma2_leaf is not supported
  {
    std::vector<double> basis(50, 1.0);
    std::vector<double> basis2(100, 1.0);  // 50 x 2, column-major
    StochTree::BARTConfig config;
    config.leaf_model       = StochTree::LeafModel::MultivariateRegression;
    config.sample_sigma2_leaf = true;
    StochTree::BARTData d2 = data;
    d2.basis_train = basis2.data();
    d2.basis_dim   = 2;
    { StochTree::BARTResult _r; EXPECT_THROW(StochTree::BARTFit(&_r, config, d2), std::runtime_error); }
  }

  // num_trees=0 without a variance forest: nothing to sample
  {
    StochTree::BARTConfig config;
    config.num_trees = 0;
    { StochTree::BARTResult _r; EXPECT_THROW(StochTree::BARTFit(&_r, config, data), std::runtime_error); }
  }
}

// ── Variance forest tests ──────────────────────────────────────────────────

// Synthetic heteroskedastic DGP.
// True conditional std: s(x) = {0.5, 1.0, 2.0, 3.0} * x1, partitioned on x0.
SyntheticData MakeHeteroskedData(int n, int p, unsigned seed = 42) {
  SyntheticData d;
  d.n = n; d.p = p;
  d.X.resize(static_cast<size_t>(n) * p);
  d.y.resize(n);

  std::mt19937 rng(seed);
  std::normal_distribution<double> norm(0.0, 1.0);
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  for (int j = 0; j < p; j++)
    for (int i = 0; i < n; i++)
      d.X[j * n + i] = unif(rng);

  for (int i = 0; i < n; i++) {
    double x0 = d.X[0 * n + i], x1 = (p > 1) ? d.X[1 * n + i] : 0.5;
    double s = (x0 < 0.25) ? 0.5 : (x0 < 0.5) ? 1.0 : (x0 < 0.75) ? 2.0 : 3.0;
    s *= x1;
    d.y[i] = s * norm(rng);  // zero mean, varying variance
  }
  return d;
}

// Smoke test: variance forest with GFR + MCMC.
// Checks shape and finiteness of sigma2_x_hat_train/test.
TEST(BARTFit, VarianceForest_Shape_Finite) {
  int n_train = 200, n_test = 50, p = 5;
  auto d_train = MakeHeteroskedData(n_train, p, 10);
  auto d_test  = MakeHeteroskedData(n_test,  p, 11);

  StochTree::BARTConfig config;
  config.num_trees              = 0;   // variance-only model (no mean forest)
  config.num_gfr                = 5;
  config.num_mcmc               = 20;
  config.include_variance_forest = true;
  config.num_trees_variance      = 20;
  config.sample_sigma2_global    = false;
  config.sample_sigma2_leaf      = false;
  config.random_seed             = 10;

  StochTree::BARTData data;
  data.X_train = d_train.X.data(); data.n_train = n_train; data.p = p;
  data.y_train = d_train.y.data();
  data.X_test  = d_test.X.data();  data.n_test  = n_test;

  StochTree::BARTResult result;
  StochTree::BARTFit(&result, config, data);

  int num_total = 20;  // keep_gfr=false (default): MCMC only
  EXPECT_EQ(result.num_total_samples, num_total);
  EXPECT_EQ(static_cast<int>(result.sigma2_x_hat_train.size()), n_train * num_total);
  EXPECT_EQ(static_cast<int>(result.sigma2_x_hat_test.size()),  n_test  * num_total);
  EXPECT_TRUE(AllFinite(result.sigma2_x_hat_train));
  EXPECT_TRUE(AllFinite(result.sigma2_x_hat_test));
}

// All sigma2_x_hat values must be strictly positive (they are exp(.) * y_std^2).
TEST(BARTFit, VarianceForest_Predictions_Positive) {
  auto d = MakeHeteroskedData(150, 3, 20);

  StochTree::BARTConfig config;
  config.num_trees               = 0;
  config.num_gfr                 = 5;
  config.num_mcmc                = 10;
  config.include_variance_forest  = true;
  config.num_trees_variance       = 10;
  config.sample_sigma2_global     = false;
  config.sample_sigma2_leaf       = false;
  config.random_seed              = 11;

  StochTree::BARTData data;
  data.X_train = d.X.data(); data.n_train = d.n; data.p = d.p;
  data.y_train = d.y.data();

  StochTree::BARTResult result;
  StochTree::BARTFit(&result, config, data);

  for (double v : result.sigma2_x_hat_train)
    EXPECT_GT(v, 0.0) << "sigma2_x_hat must be strictly positive";
}

// Variance forest + mean forest together (the typical heteroskedastic BART setup).
TEST(BARTFit, VarianceForest_WithMeanForest_Shape_Finite) {
  int n_train = 250, n_test = 50, p = 5;
  auto d_train = MakeHeteroskedData(n_train, p, 30);
  auto d_test  = MakeHeteroskedData(n_test,  p, 31);

  StochTree::BARTConfig config;
  config.num_trees               = 50;   // mean forest
  config.num_gfr                 = 5;
  config.num_burnin              = 5;
  config.num_mcmc                = 30;
  config.include_variance_forest  = true;
  config.num_trees_variance       = 20;
  config.sample_sigma2_global     = false;  // disabled with variance forest
  config.sample_sigma2_leaf       = true;
  config.random_seed              = 12;

  StochTree::BARTData data;
  data.X_train = d_train.X.data(); data.n_train = n_train; data.p = p;
  data.y_train = d_train.y.data();
  data.X_test  = d_test.X.data();  data.n_test  = n_test;

  StochTree::BARTResult result;
  StochTree::BARTFit(&result, config, data);

  int num_total = 30;  // keep_gfr=false (default): MCMC only
  EXPECT_EQ(result.num_total_samples, num_total);
  EXPECT_EQ(static_cast<int>(result.y_hat_train.size()),        n_train * num_total);
  EXPECT_EQ(static_cast<int>(result.y_hat_test.size()),         n_test  * num_total);
  EXPECT_EQ(static_cast<int>(result.sigma2_x_hat_train.size()), n_train * num_total);
  EXPECT_EQ(static_cast<int>(result.sigma2_x_hat_test.size()),  n_test  * num_total);
  EXPECT_TRUE(AllFinite(result.y_hat_train));
  EXPECT_TRUE(AllFinite(result.sigma2_x_hat_train));
  EXPECT_TRUE(AllFinite(result.sigma2_x_hat_test));
}

// Reproducibility: same seed → bit-identical variance predictions.
TEST(BARTFit, VarianceForest_Reproducibility) {
  auto d = MakeHeteroskedData(150, 5, 40);

  StochTree::BARTConfig config;
  config.num_trees               = 0;
  config.num_gfr                 = 5;
  config.num_mcmc                = 15;
  config.include_variance_forest  = true;
  config.num_trees_variance       = 10;
  config.sample_sigma2_global     = false;
  config.sample_sigma2_leaf       = false;
  config.random_seed              = 42;

  StochTree::BARTData data;
  data.X_train = d.X.data(); data.n_train = d.n; data.p = d.p;
  data.y_train = d.y.data();

  StochTree::BARTResult r1, r2;
  StochTree::BARTFit(&r1, config, data);
  StochTree::BARTFit(&r2, config, data);

  ASSERT_EQ(r1.sigma2_x_hat_train.size(), r2.sigma2_x_hat_train.size());
  for (size_t i = 0; i < r1.sigma2_x_hat_train.size(); i++)
    EXPECT_DOUBLE_EQ(r1.sigma2_x_hat_train[i], r2.sigma2_x_hat_train[i])
        << "variance prediction mismatch at i=" << i;
}

// ── Probit link tests ──────────────────────────────────────────────────────

// Smoke test: probit BART with GFR + MCMC.
// Checks shape, finiteness, and that sigma2_global is NOT stored (fixed at 1),
// while leaf_scale IS stored (leaf scale is still sampled under probit).
TEST(BARTFit, ProbitBART_Shape_Finite) {
  auto d = MakeBinaryData(200, 5);

  StochTree::BARTConfig config;
  config.num_trees          = 20;
  config.num_gfr            = 5;
  config.num_burnin         = 5;
  config.num_mcmc           = 30;
  config.num_chains         = 1;
  config.link_function      = StochTree::LinkFunction::Probit;
  config.sample_sigma2_global = false;  // probit: sigma2 fixed at 1
  config.sample_sigma2_leaf   = true;
  config.random_seed        = 10;

  StochTree::BARTData data;
  data.X_train = d.X.data();
  data.n_train = d.n;
  data.p       = d.p;
  data.y_train = d.y.data();

  StochTree::BARTResult result;
  StochTree::BARTFit(&result, config, data);

  // keep_gfr=false (default): num_total = 30 (MCMC only)
  EXPECT_EQ(result.num_total_samples, 30);
  EXPECT_EQ(static_cast<int>(result.y_hat_train.size()), 200 * 30);
  EXPECT_TRUE(AllFinite(result.y_hat_train));
  EXPECT_TRUE(result.y_hat_test.empty());

  // sigma2_global must be absent for probit
  EXPECT_TRUE(result.sigma2_global_samples.empty());
  // leaf_scale is still sampled
  EXPECT_EQ(static_cast<int>(result.leaf_scale_samples.size()), 30);
  EXPECT_TRUE(AllFinite(result.leaf_scale_samples));
}

// Probit predictions are on the probit/latent scale (forest_pred + y_bar).
// Verify that passing sample_sigma2_global=true under probit also yields
// an empty sigma2_global_samples — BARTFit never allocates it for probit.
TEST(BARTFit, ProbitBART_GlobalVarianceAlwaysEmpty) {
  auto d = MakeBinaryData(150, 3);

  StochTree::BARTConfig config;
  config.num_trees          = 10;
  config.num_gfr            = 5;
  config.num_mcmc           = 20;
  config.link_function      = StochTree::LinkFunction::Probit;
  config.sample_sigma2_global = true;  // caller passes true; BARTFit must ignore for probit
  config.random_seed        = 11;

  StochTree::BARTData data;
  data.X_train = d.X.data();
  data.n_train = d.n;
  data.p       = d.p;
  data.y_train = d.y.data();

  StochTree::BARTResult result;
  StochTree::BARTFit(&result, config, data);

  // Even with sample_sigma2_global=true, sigma2_global must be absent for probit.
  EXPECT_TRUE(result.sigma2_global_samples.empty());
  EXPECT_TRUE(AllFinite(result.y_hat_train));
}

// Probit with a test set: test predictions must be populated with correct shape.
TEST(BARTFit, ProbitBART_WithTestSet_Shape_Finite) {
  int n_train = 200, n_test = 50, p = 5;
  auto d_train = MakeBinaryData(n_train, p, 20);
  auto d_test  = MakeBinaryData(n_test,  p, 21);

  StochTree::BARTConfig config;
  config.num_trees          = 20;
  config.num_gfr            = 5;
  config.num_mcmc           = 30;
  config.link_function      = StochTree::LinkFunction::Probit;
  config.sample_sigma2_global = false;
  config.random_seed        = 12;

  StochTree::BARTData data;
  data.X_train = d_train.X.data();
  data.n_train = n_train;
  data.p       = p;
  data.y_train = d_train.y.data();
  data.X_test  = d_test.X.data();
  data.n_test  = n_test;

  StochTree::BARTResult result;
  StochTree::BARTFit(&result, config, data);

  int num_total = 30;  // keep_gfr=false (default): MCMC only
  EXPECT_EQ(result.num_total_samples, num_total);
  EXPECT_EQ(static_cast<int>(result.y_hat_train.size()), n_train * num_total);
  EXPECT_EQ(static_cast<int>(result.y_hat_test.size()),  n_test  * num_total);
  EXPECT_TRUE(AllFinite(result.y_hat_train));
  EXPECT_TRUE(AllFinite(result.y_hat_test));
}

// Probit reproducibility: same seed must yield bit-identical latent predictions.
TEST(BARTFit, ProbitBART_Reproducibility_SameSeed) {
  auto d = MakeBinaryData(150, 5, 30);

  StochTree::BARTConfig config;
  config.num_trees          = 10;
  config.num_gfr            = 5;
  config.num_mcmc           = 20;
  config.link_function      = StochTree::LinkFunction::Probit;
  config.sample_sigma2_global = false;
  config.random_seed        = 42;

  StochTree::BARTData data;
  data.X_train = d.X.data();
  data.n_train = d.n;
  data.p       = d.p;
  data.y_train = d.y.data();

  StochTree::BARTResult r1, r2;
  StochTree::BARTFit(&r1, config, data);
  StochTree::BARTFit(&r2, config, data);

  ASSERT_EQ(r1.y_hat_train.size(), r2.y_hat_train.size());
  for (size_t i = 0; i < r1.y_hat_train.size(); i++)
    EXPECT_DOUBLE_EQ(r1.y_hat_train[i], r2.y_hat_train[i]) << "mismatch at i=" << i;
}

// Sanity: probit posterior mean of Phi(latent_pred) should be closer to the
// true class probability than the naive intercept Phi(y_bar).
TEST(BARTFit, ProbitBART_Sanity_BetterThanIntercept) {
  auto d = MakeBinaryData(500, 5, 99);

  StochTree::BARTConfig config;
  config.num_trees          = 50;
  config.num_gfr            = 10;
  config.num_mcmc           = 0;
  config.link_function      = StochTree::LinkFunction::Probit;
  config.sample_sigma2_global = false;
  config.random_seed        = 100;

  StochTree::BARTData data;
  data.X_train = d.X.data();
  data.n_train = d.n;
  data.p       = d.p;
  data.y_train = d.y.data();

  StochTree::BARTResult result;
  StochTree::BARTFit(&result, config, data);

  int n = d.n, S = result.num_total_samples;

  // Posterior mean predicted probability: average Phi(latent) across samples
  std::vector<double> p_hat(n, 0.0);
  for (int s = 0; s < S; s++)
    for (int i = 0; i < n; i++)
      p_hat[i] += Phi(result.y_hat_train[s * n + i]);
  for (int i = 0; i < n; i++) p_hat[i] /= S;

  // Mean log-loss of BART predictions
  double ll_bart = 0.0;
  for (int i = 0; i < n; i++) {
    double pi = std::max(std::min(p_hat[i], 1.0 - 1e-10), 1e-10);
    ll_bart += (d.y[i] > 0.5) ? -std::log(pi) : -std::log(1.0 - pi);
  }
  ll_bart /= n;

  // Intercept-only log-loss: predict y_bar (mean class rate) for all observations
  double y_bar = VecMean(d.y);
  double ll_null = -(y_bar * std::log(y_bar) + (1.0 - y_bar) * std::log(1.0 - y_bar));

  EXPECT_LT(ll_bart, ll_null)
      << "Probit BART log-loss (" << ll_bart
      << ") should be less than intercept log-loss (" << ll_null << ")";
}

// ── Leaf regression helpers ────────────────────────────────────────────────

// Extends SyntheticData with a leaf-regression basis (column-major).
// DGP: y_i = sum_j(X[i,0] * B[i,j]) + eps, linking the first covariate
// to each basis column.  The exact form doesn't matter for shape tests.
struct SyntheticDataWithBasis {
  SyntheticData base;
  int basis_dim;
  std::vector<double> basis_train;  // column-major n × basis_dim
  std::vector<double> basis_test;   // column-major n_test × basis_dim (may be empty)
};

SyntheticDataWithBasis MakeLeafRegressionData(
    int n_train, int n_test, int p, int q, unsigned seed = 42) {
  SyntheticDataWithBasis d;
  d.base = MakeSyntheticData(n_train, p, seed);
  d.basis_dim = q;

  std::mt19937 rng(seed + 1000);
  std::normal_distribution<double> norm(0.0, 1.0);
  std::normal_distribution<double> eps(0.0, 0.3);

  // Basis for training set (column-major)
  d.basis_train.resize(static_cast<size_t>(n_train) * q);
  for (int j = 0; j < q; j++)
    for (int i = 0; i < n_train; i++)
      d.basis_train[j * n_train + i] = norm(rng);

  // Rewrite y to depend on the basis interaction: y_i = x0_i * b0_i + eps
  for (int i = 0; i < n_train; i++)
    d.base.y[i] = d.base.X[0 * n_train + i] * d.basis_train[0 * n_train + i] + eps(rng);

  // Basis for test set (column-major, independent of training X)
  if (n_test > 0) {
    d.basis_test.resize(static_cast<size_t>(n_test) * q);
    for (int j = 0; j < q; j++)
      for (int i = 0; i < n_test; i++)
        d.basis_test[j * n_test + i] = norm(rng);
  }

  return d;
}

// ── Leaf regression tests ──────────────────────────────────────────────────

// Univariate basis, GFR-only: verify output shape and finite values.
TEST(BARTFit, LeafRegression_Univariate_GFROnly_Shape_Finite) {
  int n = 150, p = 5, q = 1;
  auto d = MakeLeafRegressionData(n, 0, p, q);

  StochTree::BARTConfig config;
  config.num_trees           = 20;
  config.num_gfr             = 15;
  config.num_burnin          = 0;
  config.num_mcmc            = 0;
  config.leaf_model          = StochTree::LeafModel::UnivariateRegression;
  config.sample_sigma2_leaf  = false;
  config.random_seed         = 20;

  StochTree::BARTData data;
  data.X_train    = d.base.X.data();
  data.n_train    = n;
  data.p          = p;
  data.y_train    = d.base.y.data();
  data.basis_train = d.basis_train.data();
  data.basis_dim   = q;

  StochTree::BARTResult result;
  StochTree::BARTFit(&result, config, data);

  EXPECT_EQ(result.num_total_samples, 15);
  EXPECT_EQ(static_cast<int>(result.y_hat_train.size()), n * 15);
  EXPECT_TRUE(AllFinite(result.y_hat_train));
  EXPECT_TRUE(result.y_hat_test.empty());
}

// Univariate basis, GFR + MCMC warm-start: standard workflow.
TEST(BARTFit, LeafRegression_Univariate_GFR_Plus_MCMC_Shape_Finite) {
  int n = 200, p = 5, q = 1;
  auto d = MakeLeafRegressionData(n, 0, p, q, 43);

  StochTree::BARTConfig config;
  config.num_trees           = 30;
  config.num_gfr             = 5;
  config.num_burnin          = 5;
  config.num_mcmc            = 50;
  config.num_chains          = 1;
  config.leaf_model          = StochTree::LeafModel::UnivariateRegression;
  config.sample_sigma2_leaf  = false;
  config.random_seed         = 21;

  StochTree::BARTData data;
  data.X_train    = d.base.X.data();
  data.n_train    = n;
  data.p          = p;
  data.y_train    = d.base.y.data();
  data.basis_train = d.basis_train.data();
  data.basis_dim   = q;

  StochTree::BARTResult result;
  StochTree::BARTFit(&result, config, data);

  int num_total = 50;  // keep_gfr=false (default): MCMC only
  EXPECT_EQ(result.num_total_samples, num_total);
  EXPECT_EQ(static_cast<int>(result.y_hat_train.size()), n * num_total);
  EXPECT_TRUE(AllFinite(result.y_hat_train));
}

// Univariate basis, with test set: y_hat_test must have correct shape.
TEST(BARTFit, LeafRegression_Univariate_WithTestSet_Shape_Finite) {
  int n_train = 200, n_test = 50, p = 5, q = 1;
  auto d = MakeLeafRegressionData(n_train, n_test, p, q, 44);
  auto d_test_base = MakeSyntheticData(n_test, p, 45);

  StochTree::BARTConfig config;
  config.num_trees           = 20;
  config.num_gfr             = 5;
  config.num_mcmc            = 30;
  config.num_chains          = 1;
  config.leaf_model          = StochTree::LeafModel::UnivariateRegression;
  config.sample_sigma2_leaf  = false;
  config.random_seed         = 22;

  StochTree::BARTData data;
  data.X_train     = d.base.X.data();
  data.n_train     = n_train;
  data.p           = p;
  data.y_train     = d.base.y.data();
  data.basis_train = d.basis_train.data();
  data.basis_dim   = q;
  data.X_test      = d_test_base.X.data();
  data.n_test      = n_test;
  data.basis_test  = d.basis_test.data();

  StochTree::BARTResult result;
  StochTree::BARTFit(&result, config, data);

  int num_total = 30;  // keep_gfr=false (default): MCMC only
  EXPECT_EQ(result.num_total_samples, num_total);
  EXPECT_EQ(static_cast<int>(result.y_hat_train.size()), n_train * num_total);
  EXPECT_EQ(static_cast<int>(result.y_hat_test.size()),  n_test  * num_total);
  EXPECT_TRUE(AllFinite(result.y_hat_train));
  EXPECT_TRUE(AllFinite(result.y_hat_test));
}

// Univariate basis, leaf scale sampling enabled (sample_sigma2_leaf = true).
TEST(BARTFit, LeafRegression_Univariate_LeafScaleSampling) {
  int n = 150, p = 5, q = 1;
  auto d = MakeLeafRegressionData(n, 0, p, q, 46);

  StochTree::BARTConfig config;
  config.num_trees           = 20;
  config.num_gfr             = 5;
  config.num_mcmc            = 30;
  config.num_chains          = 1;
  config.leaf_model          = StochTree::LeafModel::UnivariateRegression;
  config.sample_sigma2_leaf  = true;
  config.random_seed         = 23;

  StochTree::BARTData data;
  data.X_train    = d.base.X.data();
  data.n_train    = n;
  data.p          = p;
  data.y_train    = d.base.y.data();
  data.basis_train = d.basis_train.data();
  data.basis_dim   = q;

  StochTree::BARTResult result;
  StochTree::BARTFit(&result, config, data);

  int num_total = 30;  // keep_gfr=false (default): MCMC only
  EXPECT_EQ(result.num_total_samples, num_total);
  EXPECT_EQ(static_cast<int>(result.y_hat_train.size()), n * num_total);
  EXPECT_TRUE(AllFinite(result.y_hat_train));
  // leaf_scale_samples must be populated
  EXPECT_EQ(static_cast<int>(result.leaf_scale_samples.size()), num_total);
  EXPECT_TRUE(AllFinite(result.leaf_scale_samples));
}

// Multivariate basis (q=3), GFR + MCMC: shape and finiteness.
TEST(BARTFit, LeafRegression_Multivariate_Shape_Finite) {
  int n = 200, p = 5, q = 3;
  auto d = MakeLeafRegressionData(n, 0, p, q, 47);

  StochTree::BARTConfig config;
  config.num_trees           = 20;
  config.num_gfr             = 5;
  config.num_mcmc            = 30;
  config.num_chains          = 1;
  config.leaf_model          = StochTree::LeafModel::MultivariateRegression;
  config.sample_sigma2_leaf  = false;  // not supported for multivariate
  config.random_seed         = 24;

  StochTree::BARTData data;
  data.X_train    = d.base.X.data();
  data.n_train    = n;
  data.p          = p;
  data.y_train    = d.base.y.data();
  data.basis_train = d.basis_train.data();
  data.basis_dim   = q;

  StochTree::BARTResult result;
  StochTree::BARTFit(&result, config, data);

  int num_total = 30;  // keep_gfr=false (default): MCMC only
  EXPECT_EQ(result.num_total_samples, num_total);
  EXPECT_EQ(static_cast<int>(result.y_hat_train.size()), n * num_total);
  EXPECT_TRUE(AllFinite(result.y_hat_train));
}

// Multivariate basis with test set.
TEST(BARTFit, LeafRegression_Multivariate_WithTestSet_Shape_Finite) {
  int n_train = 200, n_test = 50, p = 5, q = 2;
  auto d = MakeLeafRegressionData(n_train, n_test, p, q, 48);
  auto d_test_base = MakeSyntheticData(n_test, p, 49);

  StochTree::BARTConfig config;
  config.num_trees           = 20;
  config.num_gfr             = 5;
  config.num_mcmc            = 20;
  config.num_chains          = 1;
  config.leaf_model          = StochTree::LeafModel::MultivariateRegression;
  config.sample_sigma2_leaf  = false;
  config.random_seed         = 25;

  StochTree::BARTData data;
  data.X_train     = d.base.X.data();
  data.n_train     = n_train;
  data.p           = p;
  data.y_train     = d.base.y.data();
  data.basis_train = d.basis_train.data();
  data.basis_dim   = q;
  data.X_test      = d_test_base.X.data();
  data.n_test      = n_test;
  data.basis_test  = d.basis_test.data();

  StochTree::BARTResult result;
  StochTree::BARTFit(&result, config, data);

  int num_total = 20;  // keep_gfr=false (default): MCMC only
  EXPECT_EQ(result.num_total_samples, num_total);
  EXPECT_EQ(static_cast<int>(result.y_hat_train.size()), n_train * num_total);
  EXPECT_EQ(static_cast<int>(result.y_hat_test.size()),  n_test  * num_total);
  EXPECT_TRUE(AllFinite(result.y_hat_train));
  EXPECT_TRUE(AllFinite(result.y_hat_test));
}

// Reproducibility: leaf regression with same seed → bit-identical predictions.
TEST(BARTFit, LeafRegression_Univariate_Reproducibility) {
  int n = 150, p = 5, q = 1;
  auto d = MakeLeafRegressionData(n, 0, p, q, 50);

  StochTree::BARTConfig config;
  config.num_trees           = 20;
  config.num_gfr             = 5;
  config.num_mcmc            = 20;
  config.leaf_model          = StochTree::LeafModel::UnivariateRegression;
  config.sample_sigma2_leaf  = false;
  config.random_seed         = 42;

  StochTree::BARTData data;
  data.X_train    = d.base.X.data();
  data.n_train    = n;
  data.p          = p;
  data.y_train    = d.base.y.data();
  data.basis_train = d.basis_train.data();
  data.basis_dim   = q;

  StochTree::BARTResult r1, r2;
  StochTree::BARTFit(&r1, config, data);
  StochTree::BARTFit(&r2, config, data);

  ASSERT_EQ(r1.y_hat_train.size(), r2.y_hat_train.size());
  for (size_t i = 0; i < r1.y_hat_train.size(); i++)
    EXPECT_DOUBLE_EQ(r1.y_hat_train[i], r2.y_hat_train[i]) << "mismatch at i=" << i;
}

// ── No-mean-forest tests ───────────────────────────────────────────────────
//
// These tests verify that BARTFit works correctly when num_trees=0 (i.e. no
// mean forest).  A variance forest must be present to give the sampler
// something to do; the mean prediction is a constant y_bar for every sample.

// GFR-only with variance forest (no mean forest).
TEST(BARTFit, NoMeanForest_VarianceOnly_GFROnly) {
  int n = 150, p = 5;
  auto d = MakeHeteroskedData(n, p, 60);

  StochTree::BARTConfig config;
  config.num_trees               = 0;
  config.num_gfr                 = 15;
  config.num_burnin              = 0;
  config.num_mcmc                = 0;
  config.include_variance_forest  = true;
  config.num_trees_variance       = 15;
  config.sample_sigma2_global     = false;
  config.sample_sigma2_leaf       = false;
  config.random_seed              = 60;

  StochTree::BARTData data;
  data.X_train = d.X.data(); data.n_train = n; data.p = p;
  data.y_train = d.y.data();

  StochTree::BARTResult result;
  StochTree::BARTFit(&result, config, data);

  // GFR-only: all 15 samples kept regardless of keep_gfr.
  EXPECT_EQ(result.num_total_samples, 15);
  EXPECT_EQ(static_cast<int>(result.y_hat_train.size()),        n * 15);
  EXPECT_EQ(static_cast<int>(result.sigma2_x_hat_train.size()), n * 15);
  EXPECT_TRUE(AllFinite(result.y_hat_train));
  EXPECT_TRUE(AllFinite(result.sigma2_x_hat_train));
  // Mean predictions must all equal y_bar (no mean forest).
  double y_bar = result.y_bar;
  for (double v : result.y_hat_train)
    EXPECT_DOUBLE_EQ(v, y_bar);
}

// MCMC-only with variance forest (no GFR warm-start, no mean forest).
TEST(BARTFit, NoMeanForest_VarianceOnly_MCMCOnly) {
  int n = 150, p = 5;
  auto d = MakeHeteroskedData(n, p, 61);

  StochTree::BARTConfig config;
  config.num_trees               = 0;
  config.num_gfr                 = 0;
  config.num_burnin              = 5;
  config.num_mcmc                = 20;
  config.include_variance_forest  = true;
  config.num_trees_variance       = 15;
  config.sample_sigma2_global     = false;
  config.sample_sigma2_leaf       = false;
  config.random_seed              = 61;

  StochTree::BARTData data;
  data.X_train = d.X.data(); data.n_train = n; data.p = p;
  data.y_train = d.y.data();

  StochTree::BARTResult result;
  StochTree::BARTFit(&result, config, data);

  EXPECT_EQ(result.num_total_samples, 20);
  EXPECT_EQ(static_cast<int>(result.sigma2_x_hat_train.size()), n * 20);
  EXPECT_TRUE(AllFinite(result.sigma2_x_hat_train));
  for (double v : result.sigma2_x_hat_train)
    EXPECT_GT(v, 0.0);
}

// GFR + MCMC with variance forest, keep_gfr=true: verify GFR samples included.
TEST(BARTFit, NoMeanForest_VarianceOnly_KeepGFR) {
  int n = 150, p = 5;
  auto d = MakeHeteroskedData(n, p, 62);

  StochTree::BARTConfig config;
  config.num_trees               = 0;
  config.num_gfr                 = 5;
  config.num_mcmc                = 20;
  config.keep_gfr                = true;
  config.include_variance_forest  = true;
  config.num_trees_variance       = 15;
  config.sample_sigma2_global     = false;
  config.sample_sigma2_leaf       = false;
  config.random_seed              = 62;

  StochTree::BARTData data;
  data.X_train = d.X.data(); data.n_train = n; data.p = p;
  data.y_train = d.y.data();

  StochTree::BARTResult result;
  StochTree::BARTFit(&result, config, data);

  int num_total = 5 + 20;  // keep_gfr=true: GFR + MCMC
  EXPECT_EQ(result.num_total_samples, num_total);
  EXPECT_EQ(static_cast<int>(result.sigma2_x_hat_train.size()), n * num_total);
  EXPECT_TRUE(AllFinite(result.sigma2_x_hat_train));
}

// GFR + MCMC with variance forest, with test set (no mean forest).
TEST(BARTFit, NoMeanForest_VarianceOnly_WithTestSet) {
  int n_train = 200, n_test = 50, p = 5;
  auto d_train = MakeHeteroskedData(n_train, p, 63);
  auto d_test  = MakeHeteroskedData(n_test,  p, 64);

  StochTree::BARTConfig config;
  config.num_trees               = 0;
  config.num_gfr                 = 5;
  config.num_mcmc                = 20;
  config.include_variance_forest  = true;
  config.num_trees_variance       = 15;
  config.sample_sigma2_global     = false;
  config.sample_sigma2_leaf       = false;
  config.random_seed              = 63;

  StochTree::BARTData data;
  data.X_train = d_train.X.data(); data.n_train = n_train; data.p = p;
  data.y_train = d_train.y.data();
  data.X_test  = d_test.X.data();  data.n_test  = n_test;

  StochTree::BARTResult result;
  StochTree::BARTFit(&result, config, data);

  int num_total = 20;  // keep_gfr=false (default): MCMC only
  EXPECT_EQ(result.num_total_samples, num_total);
  EXPECT_EQ(static_cast<int>(result.sigma2_x_hat_train.size()), n_train * num_total);
  EXPECT_EQ(static_cast<int>(result.sigma2_x_hat_test.size()),  n_test  * num_total);
  EXPECT_TRUE(AllFinite(result.sigma2_x_hat_train));
  EXPECT_TRUE(AllFinite(result.sigma2_x_hat_test));
  for (double v : result.sigma2_x_hat_test)
    EXPECT_GT(v, 0.0);
  // y_hat_test must all equal y_bar
  double y_bar = result.y_bar;
  for (double v : result.y_hat_test)
    EXPECT_DOUBLE_EQ(v, y_bar);
}

// Multi-chain with variance forest (no mean forest).
TEST(BARTFit, NoMeanForest_VarianceOnly_MultiChain) {
  int n = 150, p = 5;
  auto d = MakeHeteroskedData(n, p, 65);

  StochTree::BARTConfig config;
  config.num_trees               = 0;
  config.num_gfr                 = 3;
  config.num_burnin              = 2;
  config.num_mcmc                = 15;
  config.num_chains              = 3;
  config.include_variance_forest  = true;
  config.num_trees_variance       = 10;
  config.sample_sigma2_global     = false;
  config.sample_sigma2_leaf       = false;
  config.random_seed              = 65;

  StochTree::BARTData data;
  data.X_train = d.X.data(); data.n_train = n; data.p = p;
  data.y_train = d.y.data();

  StochTree::BARTResult result;
  StochTree::BARTFit(&result, config, data);

  int num_total = 3 * 15;  // 3 chains × 15 MCMC, keep_gfr=false
  EXPECT_EQ(result.num_total_samples, num_total);
  EXPECT_EQ(static_cast<int>(result.sigma2_x_hat_train.size()), n * num_total);
  EXPECT_TRUE(AllFinite(result.sigma2_x_hat_train));
}

// ── Random effects helpers ─────────────────────────────────────────────────

struct SyntheticRFXData {
  int n_train, n_test, p, n_groups;
  std::vector<double> X_train;      // column-major, n_train × p
  std::vector<double> y_train;      // n_train
  std::vector<int>    groups_train; // n_train  (1-indexed group labels)
  std::vector<double> X_test;       // column-major, n_test × p
  std::vector<int>    groups_test;  // n_test
};

// DGP: y_i = 2*x_i0 - x_i1 + gamma_{g(i)} + eps,
//      gamma_g ~ N(0, 1) (group effects), eps ~ N(0, 0.5)
// Group labels: 1..n_groups uniformly assigned.
static SyntheticRFXData MakeRFXData(int n_train, int n_test, int p, int n_groups, unsigned seed) {
  SyntheticRFXData d;
  d.n_train  = n_train;
  d.n_test   = n_test;
  d.p        = p;
  d.n_groups = n_groups;

  std::mt19937 rng(seed);
  std::normal_distribution<double> norm(0.0, 1.0);
  std::normal_distribution<double> eps(0.0, 0.5);
  std::uniform_int_distribution<int> grp(1, n_groups);

  // Group effects
  std::vector<double> gamma(n_groups + 1);  // 1-indexed
  for (int g = 1; g <= n_groups; g++) gamma[g] = norm(rng);

  int n_total = n_train + n_test;
  std::vector<double> X_all(static_cast<size_t>(n_total) * p);
  std::vector<int>    grp_all(n_total);

  for (int j = 0; j < p; j++)
    for (int i = 0; i < n_total; i++)
      X_all[j * n_total + i] = norm(rng);

  for (int i = 0; i < n_total; i++) grp_all[i] = grp(rng);

  // Training set
  d.X_train.resize(static_cast<size_t>(n_train) * p);
  d.y_train.resize(n_train);
  d.groups_train.resize(n_train);
  for (int j = 0; j < p; j++)
    for (int i = 0; i < n_train; i++)
      d.X_train[j * n_train + i] = X_all[j * n_total + i];
  for (int i = 0; i < n_train; i++) {
    double sig = 2.0 * X_all[0 * n_total + i];
    if (p > 1) sig -= X_all[1 * n_total + i];
    d.y_train[i]     = sig + gamma[grp_all[i]] + eps(rng);
    d.groups_train[i] = grp_all[i];
  }

  // Test set (only groups already present in training)
  d.X_test.resize(static_cast<size_t>(n_test) * p);
  d.groups_test.resize(n_test);
  for (int j = 0; j < p; j++)
    for (int i = 0; i < n_test; i++)
      d.X_test[j * n_test + i] = X_all[j * n_total + n_train + i];
  for (int i = 0; i < n_test; i++) {
    // Assign test obs to training groups (modulo ensures subset constraint)
    d.groups_test[i] = ((grp_all[n_train + i] - 1) % n_groups) + 1;
  }

  return d;
}

// ── Random effects tests ──────────────────────────────────────────────────

TEST(BARTFit, RFX_InterceptOnly_GFROnly_Shape_Finite) {
  auto d = MakeRFXData(150, 0, 5, 4, 11);

  StochTree::BARTConfig config;
  config.num_trees     = 20;
  config.num_gfr       = 10;
  config.num_burnin    = 0;
  config.num_mcmc      = 0;
  config.keep_gfr      = true;
  config.rfx_model_spec = StochTree::RFXModelSpec::InterceptOnly;
  config.sample_sigma2_global = false;
  config.sample_sigma2_leaf   = false;
  config.random_seed   = 11;

  StochTree::BARTData data;
  data.X_train     = d.X_train.data(); data.n_train = d.n_train; data.p = d.p;
  data.y_train     = d.y_train.data();
  data.rfx_groups  = d.groups_train.data();

  StochTree::BARTResult result;
  StochTree::BARTFit(&result, config, data);

  int num_total = 10;  // GFR only, keep_gfr=true
  EXPECT_EQ(result.num_total_samples, num_total);
  EXPECT_EQ(static_cast<int>(result.y_hat_train.size()), d.n_train * num_total);
  EXPECT_TRUE(AllFinite(result.y_hat_train));
  EXPECT_EQ(result.rfx_num_groups,     d.n_groups);
  EXPECT_EQ(result.rfx_num_components, 1);
  EXPECT_EQ(static_cast<int>(result.rfx_group_ids.size()), d.n_groups);
  EXPECT_NE(result.rfx_container, nullptr);
  EXPECT_EQ(result.rfx_container->NumSamples(), num_total);
}

TEST(BARTFit, RFX_InterceptOnly_GFR_Plus_MCMC_Shape_Finite) {
  auto d = MakeRFXData(150, 0, 5, 4, 22);

  StochTree::BARTConfig config;
  config.num_trees      = 20;
  config.num_gfr        = 5;
  config.num_burnin     = 2;
  config.num_mcmc       = 20;
  config.num_chains     = 1;
  config.rfx_model_spec = StochTree::RFXModelSpec::InterceptOnly;
  config.sample_sigma2_global = true;
  config.sample_sigma2_leaf   = false;
  config.random_seed    = 22;

  StochTree::BARTData data;
  data.X_train    = d.X_train.data(); data.n_train = d.n_train; data.p = d.p;
  data.y_train    = d.y_train.data();
  data.rfx_groups = d.groups_train.data();

  StochTree::BARTResult result;
  StochTree::BARTFit(&result, config, data);

  int num_total = 20;  // keep_gfr=false → only MCMC samples
  EXPECT_EQ(result.num_total_samples, num_total);
  EXPECT_EQ(static_cast<int>(result.y_hat_train.size()), d.n_train * num_total);
  EXPECT_TRUE(AllFinite(result.y_hat_train));
  EXPECT_NE(result.rfx_container, nullptr);
  EXPECT_EQ(result.rfx_container->NumSamples(), num_total);
  EXPECT_EQ(static_cast<int>(result.sigma2_global_samples.size()), num_total);
  EXPECT_TRUE(AllFinite(result.sigma2_global_samples));
}

TEST(BARTFit, RFX_InterceptOnly_WithTestSet) {
  auto d = MakeRFXData(150, 50, 5, 3, 33);

  StochTree::BARTConfig config;
  config.num_trees      = 20;
  config.num_gfr        = 5;
  config.num_mcmc       = 15;
  config.num_chains     = 1;
  config.rfx_model_spec = StochTree::RFXModelSpec::InterceptOnly;
  config.sample_sigma2_global = false;
  config.sample_sigma2_leaf   = false;
  config.random_seed    = 33;

  StochTree::BARTData data;
  data.X_train       = d.X_train.data(); data.n_train = d.n_train; data.p = d.p;
  data.y_train       = d.y_train.data();
  data.X_test        = d.X_test.data();  data.n_test  = d.n_test;
  data.rfx_groups    = d.groups_train.data();
  data.rfx_groups_test = d.groups_test.data();

  StochTree::BARTResult result;
  StochTree::BARTFit(&result, config, data);

  int num_total = 15;
  EXPECT_EQ(result.num_total_samples, num_total);
  EXPECT_EQ(static_cast<int>(result.y_hat_train.size()), d.n_train * num_total);
  EXPECT_EQ(static_cast<int>(result.y_hat_test.size()),  d.n_test  * num_total);
  EXPECT_TRUE(AllFinite(result.y_hat_train));
  EXPECT_TRUE(AllFinite(result.y_hat_test));
}

TEST(BARTFit, RFX_InterceptOnly_MultiChain) {
  auto d = MakeRFXData(150, 0, 5, 5, 44);

  StochTree::BARTConfig config;
  config.num_trees      = 20;
  config.num_gfr        = 5;
  config.num_burnin     = 2;
  config.num_mcmc       = 10;
  config.num_chains     = 3;
  config.rfx_model_spec = StochTree::RFXModelSpec::InterceptOnly;
  config.sample_sigma2_global = false;
  config.sample_sigma2_leaf   = false;
  config.random_seed    = 44;

  StochTree::BARTData data;
  data.X_train    = d.X_train.data(); data.n_train = d.n_train; data.p = d.p;
  data.y_train    = d.y_train.data();
  data.rfx_groups = d.groups_train.data();

  StochTree::BARTResult result;
  StochTree::BARTFit(&result, config, data);

  int num_total = 3 * 10;
  EXPECT_EQ(result.num_total_samples, num_total);
  EXPECT_EQ(static_cast<int>(result.y_hat_train.size()), d.n_train * num_total);
  EXPECT_TRUE(AllFinite(result.y_hat_train));
  EXPECT_NE(result.rfx_container, nullptr);
  EXPECT_EQ(result.rfx_container->NumSamples(), num_total);
}

TEST(BARTFit, RFX_Custom_Basis_Shape_Finite) {
  auto d = MakeRFXData(150, 0, 5, 4, 55);
  int n = d.n_train;
  int n_components = 2;
  // Custom basis: [1, x_0] for each observation (2 components)
  std::vector<double> basis(static_cast<size_t>(n) * n_components);
  for (int i = 0; i < n; i++) {
    basis[0 * n + i] = 1.0;
    basis[1 * n + i] = d.X_train[0 * n + i];  // first covariate as second basis
  }

  StochTree::BARTConfig config;
  config.num_trees        = 20;
  config.num_gfr          = 5;
  config.num_mcmc         = 10;
  config.num_chains       = 1;
  config.rfx_model_spec   = StochTree::RFXModelSpec::Custom;
  config.rfx_num_components = n_components;
  config.sample_sigma2_global = false;
  config.sample_sigma2_leaf   = false;
  config.random_seed      = 55;

  StochTree::BARTData data;
  data.X_train         = d.X_train.data(); data.n_train = n; data.p = d.p;
  data.y_train         = d.y_train.data();
  data.rfx_groups      = d.groups_train.data();
  data.rfx_basis_train = basis.data();

  StochTree::BARTResult result;
  StochTree::BARTFit(&result, config, data);

  int num_total = 10;
  EXPECT_EQ(result.num_total_samples, num_total);
  EXPECT_EQ(static_cast<int>(result.y_hat_train.size()), n * num_total);
  EXPECT_TRUE(AllFinite(result.y_hat_train));
  EXPECT_EQ(result.rfx_num_components, n_components);
  EXPECT_NE(result.rfx_container, nullptr);
  EXPECT_EQ(result.rfx_container->NumSamples(), num_total);
}

TEST(BARTFit, RFX_Only_No_Mean_Forest) {
  // RFX with no mean forest (num_trees=0) — pure random effects model.
  auto d = MakeRFXData(150, 0, 5, 4, 66);

  StochTree::BARTConfig config;
  config.num_trees      = 0;
  config.num_gfr        = 0;
  config.num_mcmc       = 20;
  config.num_chains     = 1;
  config.rfx_model_spec = StochTree::RFXModelSpec::InterceptOnly;
  config.sample_sigma2_global = false;
  config.sample_sigma2_leaf   = false;
  config.random_seed    = 66;

  StochTree::BARTData data;
  data.X_train    = d.X_train.data(); data.n_train = d.n_train; data.p = d.p;
  data.y_train    = d.y_train.data();
  data.rfx_groups = d.groups_train.data();

  StochTree::BARTResult result;
  StochTree::BARTFit(&result, config, data);

  int num_total = 20;
  EXPECT_EQ(result.num_total_samples, num_total);
  EXPECT_EQ(static_cast<int>(result.y_hat_train.size()), d.n_train * num_total);
  EXPECT_TRUE(AllFinite(result.y_hat_train));
  EXPECT_NE(result.rfx_container, nullptr);
  EXPECT_EQ(result.rfx_container->NumSamples(), num_total);
}

TEST(BARTFit, RFX_Reproducibility_SameSeed) {
  auto d = MakeRFXData(120, 40, 4, 3, 77);

  auto run = [&]() {
    StochTree::BARTConfig config;
    config.num_trees      = 20;
    config.num_gfr        = 5;
    config.num_mcmc       = 15;
    config.num_chains     = 1;
    config.rfx_model_spec = StochTree::RFXModelSpec::InterceptOnly;
    config.sample_sigma2_global = false;
    config.sample_sigma2_leaf   = false;
    config.random_seed    = 77;

    StochTree::BARTData data;
    data.X_train         = d.X_train.data(); data.n_train = d.n_train; data.p = d.p;
    data.y_train         = d.y_train.data();
    data.X_test          = d.X_test.data();  data.n_test  = d.n_test;
    data.rfx_groups      = d.groups_train.data();
    data.rfx_groups_test = d.groups_test.data();

    StochTree::BARTResult result;
    StochTree::BARTFit(&result, config, data);
    return result;
  };

  auto r1 = run();
  auto r2 = run();

  EXPECT_EQ(r1.y_hat_train, r2.y_hat_train);
  EXPECT_EQ(r1.y_hat_test,  r2.y_hat_test);
}

TEST(BARTFit, RFX_ValidationErrors) {
  auto d = MakeRFXData(100, 0, 4, 3, 88);

  // Missing rfx_groups
  {
    StochTree::BARTConfig config;
    config.num_trees      = 20;
    config.num_gfr        = 5;
    config.num_mcmc       = 5;
    config.rfx_model_spec = StochTree::RFXModelSpec::InterceptOnly;

    StochTree::BARTData data;
    data.X_train = d.X_train.data(); data.n_train = d.n_train; data.p = d.p;
    data.y_train = d.y_train.data();
    // rfx_groups intentionally null

    StochTree::BARTResult result;
    EXPECT_THROW(StochTree::BARTFit(&result, config, data), std::runtime_error);
  }

  // Custom RFX but missing basis
  {
    StochTree::BARTConfig config;
    config.num_trees        = 20;
    config.num_gfr          = 5;
    config.num_mcmc         = 5;
    config.rfx_model_spec   = StochTree::RFXModelSpec::Custom;
    config.rfx_num_components = 1;

    StochTree::BARTData data;
    data.X_train    = d.X_train.data(); data.n_train = d.n_train; data.p = d.p;
    data.y_train    = d.y_train.data();
    data.rfx_groups = d.groups_train.data();
    // rfx_basis_train intentionally null

    StochTree::BARTResult result;
    EXPECT_THROW(StochTree::BARTFit(&result, config, data), std::runtime_error);
  }
}

// ── Cloglog tests ──────────────────────────────────────────────────────────

// Generate ordinal outcome data with K categories (labels 0..K-1).
// True linear predictor drives a cloglog model with K-1 uniform cutpoints.
struct OrdinalData {
  int n, p, K;
  std::vector<double> X;  // column-major n × p
  std::vector<double> y;  // category labels as double (0.0, 1.0, ...)
};

OrdinalData MakeOrdinalData(int n, int p, int K, unsigned seed = 99) {
  OrdinalData d;
  d.n = n; d.p = p; d.K = K;
  d.X.resize(static_cast<size_t>(n) * p);
  d.y.resize(n);

  std::mt19937 rng(seed);
  std::normal_distribution<double> norm(0.0, 1.0);
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  for (int j = 0; j < p; j++)
    for (int i = 0; i < n; i++)
      d.X[j * n + i] = norm(rng);

  // Uniform cutpoints in (-2, 2).
  std::vector<double> cuts(K - 1);
  for (int k = 0; k < K - 1; k++)
    cuts[k] = -2.0 + 4.0 * (k + 1.0) / K;

  for (int i = 0; i < n; i++) {
    double eta = d.X[0 * n + i];
    if (p > 1) eta -= 0.5 * d.X[1 * n + i];
    // cloglog F(x) = 1 - exp(-exp(x))
    double u = unif(rng);
    int cat = 0;
    for (int k = 0; k < K - 1; k++) {
      double p_cum = 1.0 - std::exp(-std::exp(eta - cuts[k]));
      if (u < p_cum) break;
      cat = k + 1;
    }
    d.y[i] = static_cast<double>(cat);
  }
  return d;
}

// Binary cloglog (K=2): GFR + MCMC, check shape and finite values.
TEST(BARTFitCloglog, Binary_GFR_MCMC_Shape_Finite) {
  auto d = MakeOrdinalData(200, 5, 2);

  StochTree::BARTConfig config;
  config.num_trees              = 20;
  config.num_gfr                = 5;
  config.num_burnin             = 0;
  config.num_mcmc               = 50;
  config.num_chains             = 1;
  config.link_function          = StochTree::LinkFunction::Cloglog;
  config.cloglog_num_categories = 2;
  config.cloglog_forest_shape   = 2.0;
  config.cloglog_forest_rate    = 2.0;
  config.random_seed            = 10;

  StochTree::BARTData data;
  data.X_train = d.X.data();
  data.n_train = d.n;
  data.p       = d.p;
  data.y_train = d.y.data();

  StochTree::BARTResult result;
  StochTree::BARTFit(&result, config, data);

  int num_total = 50;  // 0 gfr stored + 1*50 mcmc
  EXPECT_EQ(result.num_total_samples, num_total);
  EXPECT_EQ(result.n_train, 200);
  EXPECT_EQ(result.n_test,  0);
  EXPECT_EQ(static_cast<int>(result.y_hat_train.size()), 200 * num_total);
  EXPECT_TRUE(AllFinite(result.y_hat_train));

  // K=2: one cutpoint per sample
  EXPECT_EQ(static_cast<int>(result.cloglog_cutpoint_samples.size()), 1 * num_total);
  EXPECT_TRUE(AllFinite(result.cloglog_cutpoint_samples));
  // No sigma2 or leaf_scale for cloglog
  EXPECT_TRUE(result.sigma2_global_samples.empty());
  EXPECT_TRUE(result.leaf_scale_samples.empty());
  // No variance forest
  EXPECT_TRUE(result.sigma2_x_hat_train.empty());
}

// Multi-category cloglog (K=3): GFR + MCMC, check shape and finite values.
TEST(BARTFitCloglog, Ordinal3_GFR_MCMC_Shape_Finite) {
  auto d = MakeOrdinalData(200, 5, 3, 77);

  StochTree::BARTConfig config;
  config.num_trees              = 20;
  config.num_gfr                = 5;
  config.num_burnin             = 0;
  config.num_mcmc               = 50;
  config.num_chains             = 1;
  config.link_function          = StochTree::LinkFunction::Cloglog;
  config.cloglog_num_categories = 3;
  config.random_seed            = 11;

  StochTree::BARTData data;
  data.X_train = d.X.data();
  data.n_train = d.n;
  data.p       = d.p;
  data.y_train = d.y.data();

  StochTree::BARTResult result;
  StochTree::BARTFit(&result, config, data);

  int num_total = 50;
  EXPECT_EQ(result.num_total_samples, num_total);
  EXPECT_EQ(static_cast<int>(result.y_hat_train.size()), 200 * num_total);
  EXPECT_TRUE(AllFinite(result.y_hat_train));
  // K=3: two cutpoints per sample
  EXPECT_EQ(static_cast<int>(result.cloglog_cutpoint_samples.size()), 2 * num_total);
  EXPECT_TRUE(AllFinite(result.cloglog_cutpoint_samples));
  EXPECT_TRUE(result.sigma2_global_samples.empty());
  EXPECT_TRUE(result.leaf_scale_samples.empty());
}

// Cloglog with keep_gfr=true: GFR samples also stored in cutpoint_samples.
TEST(BARTFitCloglog, Binary_KeepGFR) {
  auto d = MakeOrdinalData(150, 4, 2, 55);

  StochTree::BARTConfig config;
  config.num_trees              = 20;
  config.num_gfr                = 5;
  config.num_burnin             = 0;
  config.num_mcmc               = 20;
  config.num_chains             = 1;
  config.keep_gfr               = true;
  config.link_function          = StochTree::LinkFunction::Cloglog;
  config.cloglog_num_categories = 2;
  config.random_seed            = 12;

  StochTree::BARTData data;
  data.X_train = d.X.data();
  data.n_train = d.n;
  data.p       = d.p;
  data.y_train = d.y.data();

  StochTree::BARTResult result;
  StochTree::BARTFit(&result, config, data);

  int num_total = 5 + 20;  // 5 gfr + 20 mcmc
  EXPECT_EQ(result.num_total_samples, num_total);
  EXPECT_EQ(static_cast<int>(result.y_hat_train.size()), 150 * num_total);
  EXPECT_EQ(static_cast<int>(result.cloglog_cutpoint_samples.size()), 1 * num_total);
  EXPECT_TRUE(AllFinite(result.cloglog_cutpoint_samples));
}

// Multi-chain cloglog: two chains, check total samples and finite values.
TEST(BARTFitCloglog, Binary_MultiChain) {
  auto d = MakeOrdinalData(200, 5, 2, 33);

  StochTree::BARTConfig config;
  config.num_trees              = 20;
  config.num_gfr                = 4;
  config.num_burnin             = 0;
  config.num_mcmc               = 30;
  config.num_chains             = 2;
  config.link_function          = StochTree::LinkFunction::Cloglog;
  config.cloglog_num_categories = 2;
  config.random_seed            = 13;

  StochTree::BARTData data;
  data.X_train = d.X.data();
  data.n_train = d.n;
  data.p       = d.p;
  data.y_train = d.y.data();

  StochTree::BARTResult result;
  StochTree::BARTFit(&result, config, data);

  int num_total = 0 + 2 * 30;  // 0 gfr stored + 2 chains * 30 mcmc
  EXPECT_EQ(result.num_total_samples, num_total);
  EXPECT_EQ(static_cast<int>(result.y_hat_train.size()), 200 * num_total);
  EXPECT_EQ(static_cast<int>(result.cloglog_cutpoint_samples.size()), 1 * num_total);
  EXPECT_TRUE(AllFinite(result.cloglog_cutpoint_samples));
}

// Cloglog validation: combining with variance forest should throw.
TEST(BARTFitCloglog, Validation_Errors) {
  auto d = MakeOrdinalData(100, 3, 2);

  // Cloglog + variance forest is not supported
  {
    StochTree::BARTConfig config;
    config.num_trees              = 20;
    config.num_gfr                = 5;
    config.num_mcmc               = 10;
    config.link_function          = StochTree::LinkFunction::Cloglog;
    config.cloglog_num_categories = 2;
    config.include_variance_forest = true;
    config.num_trees_variance     = 10;

    StochTree::BARTData data;
    data.X_train = d.X.data(); data.n_train = d.n; data.p = d.p;
    data.y_train = d.y.data();

    StochTree::BARTResult result;
    EXPECT_THROW(StochTree::BARTFit(&result, config, data), std::runtime_error);
  }

  // Cloglog + non-constant leaf model should throw
  {
    StochTree::BARTConfig config;
    config.num_trees              = 20;
    config.num_gfr                = 5;
    config.num_mcmc               = 10;
    config.link_function          = StochTree::LinkFunction::Cloglog;
    config.cloglog_num_categories = 2;
    config.leaf_model             = StochTree::LeafModel::UnivariateRegression;

    StochTree::BARTData data;
    data.X_train = d.X.data(); data.n_train = d.n; data.p = d.p;
    data.y_train = d.y.data();
    data.basis_dim = 1;

    StochTree::BARTResult result;
    EXPECT_THROW(StochTree::BARTFit(&result, config, data), std::runtime_error);
  }

  // Cloglog + fewer than 2 categories should throw
  {
    StochTree::BARTConfig config;
    config.num_trees              = 20;
    config.num_gfr                = 5;
    config.num_mcmc               = 10;
    config.link_function          = StochTree::LinkFunction::Cloglog;
    config.cloglog_num_categories = 1;

    StochTree::BARTData data;
    data.X_train = d.X.data(); data.n_train = d.n; data.p = d.p;
    data.y_train = d.y.data();

    StochTree::BARTResult result;
    EXPECT_THROW(StochTree::BARTFit(&result, config, data), std::runtime_error);
  }
}

} // namespace
