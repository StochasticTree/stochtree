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

} // namespace
