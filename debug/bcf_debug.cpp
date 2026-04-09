/*
 * BCF debug driver. The first CLI argument selects the scenario (default: 0).
 *
 * Usage: bcf_debug [scenario]
 *   0  Two-forest BCF: constant-leaf mu, univariate-leaf tau (Z as basis)
 *      DGP: mu(x) = 2*sin(pi*x1) + 0.5*x2
 *           tau(x) = 1 + x3
 *           z ~ Bernoulli(0.5)
 *           y = mu(x) + tau(x)*z + N(0, 0.5^2)
 *
 * Add scenarios here as the BCFSampler API develops (heteroskedastic,
 * random effects, propensity weighting, etc.).
 *
 * Algorithm overview
 * ------------------
 * Both forests share a single ColumnVector residual. Alternating GFR/MCMC
 * steps for mu and tau each run backfitting, so the residual after each
 * step correctly reflects the other forest's current contribution:
 *
 *   After mu step:  residual ≈ y - y_bar - mu_hat
 *   After tau step: residual ≈ y - y_bar - mu_hat - tau_hat*z
 *
 * The tau forest uses z as a univariate basis (AddBasis), so its prediction
 * for observation i is  tau_leaf(i) * z(i), and backfitting is z-aware.
 */

#include <stochtree/container.h>
#include <stochtree/data.h>
#include <stochtree/leaf_model.h>
#include <stochtree/log.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/tree_sampler.h>
#include <stochtree/variance_model.h>

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static constexpr double kPi = 3.14159265358979323846;

// ---- Data ------------------------------------------------------------

struct BCFDataset {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X;
  Eigen::VectorXd y;
  Eigen::VectorXd z;
  Eigen::VectorXd mu_true;
  Eigen::VectorXd tau_true;
};

BCFDataset generate_data(int n, int p, std::mt19937& rng) {
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  std::normal_distribution<double> normal(0.0, 1.0);
  std::bernoulli_distribution bern(0.5);

  BCFDataset d;
  d.X.resize(n, p);
  d.y.resize(n);
  d.z.resize(n);
  d.mu_true.resize(n);
  d.tau_true.resize(n);

  for (int i = 0; i < n; i++)
    for (int j = 0; j < p; j++)
      d.X(i, j) = unif(rng);

  for (int i = 0; i < n; i++) {
    d.z(i) = bern(rng) ? 1.0 : 0.0;
    d.mu_true(i) = 2.0 * std::sin(kPi * d.X(i, 0)) + 0.5 * d.X(i, 1);
    d.tau_true(i) = 1.0 + d.X(i, 2);
    d.y(i) = d.mu_true(i) + d.tau_true(i) * d.z(i) + 0.5 * normal(rng);
  }
  return d;
}

// ---- Scenario 0: constant-leaf mu + univariate-leaf tau (Z basis) ---

void run_scenario_0(int n, int p, int num_trees, int num_gfr, int num_mcmc) {
  constexpr int num_threads = 1;
  constexpr int cutpoint_grid_size = 100;
  std::mt19937 rng(42);

  BCFDataset data = generate_data(n, p, rng);
  double y_bar = data.y.mean();
  Eigen::VectorXd resid_vec = data.y.array() - y_bar;

  // Mu dataset: X covariates only
  StochTree::ForestDataset dataset_mu;
  dataset_mu.AddCovariates(data.X.data(), n, p, /*row_major=*/true);

  // Tau dataset: X covariates + Z as univariate basis
  StochTree::ForestDataset dataset_tau;
  dataset_tau.AddCovariates(data.X.data(), n, p, true);
  dataset_tau.AddBasis(data.z.data(), n, /*num_col=*/1, /*row_major=*/false);

  // Shared residual
  StochTree::ColumnVector residual(resid_vec.data(), n);

  std::vector<StochTree::FeatureType> feature_types(p, StochTree::FeatureType::kNumeric);
  std::vector<double> var_weights(p, 1.0 / p);
  std::vector<int> sweep_indices;

  StochTree::TreePrior tree_prior(0.95, 2.0, /*min_samples_leaf=*/5);

  // Mu forest: constant-leaf
  StochTree::ForestContainer mu_samples(num_trees, 1, /*leaf_constant=*/true, /*exponentiated=*/false);
  StochTree::TreeEnsemble mu_forest(num_trees, 1, true, false);
  StochTree::ForestTracker mu_tracker(dataset_mu.GetCovariates(), feature_types, num_trees, n);
  double mu_leaf_scale = 1.0 / num_trees;
  StochTree::GaussianConstantLeafModel mu_leaf_model(mu_leaf_scale);

  // Tau forest: univariate regression leaf (prediction = leaf_param * z)
  StochTree::ForestContainer tau_samples(num_trees, 1, /*leaf_constant=*/false, /*exponentiated=*/false);
  StochTree::TreeEnsemble tau_forest(num_trees, 1, false, false);
  StochTree::ForestTracker tau_tracker(dataset_tau.GetCovariates(), feature_types, num_trees, n);
  double tau_leaf_scale = 1.0 / num_trees;
  StochTree::GaussianUnivariateRegressionLeafModel tau_leaf_model(tau_leaf_scale);

  double global_variance = 1.0;
  constexpr double a_sigma = 0.0, b_sigma = 0.0;  // non-informative IG prior
  StochTree::GlobalHomoskedasticVarianceModel var_model;

  // GFR warmup — no samples stored
  std::cout << "[GFR]  " << num_gfr << " warmup iterations...\n";
  bool pre_mu = false, pre_tau = false;
  for (int i = 0; i < num_gfr; i++) {
    StochTree::GFRSampleOneIter<
        StochTree::GaussianConstantLeafModel,
        StochTree::GaussianConstantSuffStat>(
        mu_forest, mu_tracker, mu_samples, mu_leaf_model,
        dataset_mu, residual, tree_prior, rng,
        var_weights, sweep_indices, global_variance, feature_types,
        cutpoint_grid_size, /*keep_forest=*/false, pre_mu,
        /*backfitting=*/true, /*num_features_subsample=*/-1, num_threads);
    pre_mu = true;

    StochTree::GFRSampleOneIter<
        StochTree::GaussianUnivariateRegressionLeafModel,
        StochTree::GaussianUnivariateRegressionSuffStat>(
        tau_forest, tau_tracker, tau_samples, tau_leaf_model,
        dataset_tau, residual, tree_prior, rng,
        var_weights, sweep_indices, global_variance, feature_types,
        cutpoint_grid_size, false, pre_tau,
        true, -1, num_threads);
    pre_tau = true;

    global_variance = var_model.SampleVarianceParameter(
        residual.GetData(), a_sigma, b_sigma, rng);
  }

  // MCMC — store samples
  std::cout << "[MCMC] " << num_mcmc << " sampling iterations...\n";
  for (int i = 0; i < num_mcmc; i++) {
    StochTree::MCMCSampleOneIter<
        StochTree::GaussianConstantLeafModel,
        StochTree::GaussianConstantSuffStat>(
        mu_forest, mu_tracker, mu_samples, mu_leaf_model,
        dataset_mu, residual, tree_prior, rng,
        var_weights, sweep_indices, global_variance,
        /*keep_forest=*/true, /*pre_initialized=*/true,
        /*backfitting=*/true, num_threads);

    StochTree::MCMCSampleOneIter<
        StochTree::GaussianUnivariateRegressionLeafModel,
        StochTree::GaussianUnivariateRegressionSuffStat>(
        tau_forest, tau_tracker, tau_samples, tau_leaf_model,
        dataset_tau, residual, tree_prior, rng,
        var_weights, sweep_indices, global_variance,
        true, true, true, num_threads);

    global_variance = var_model.SampleVarianceParameter(
        residual.GetData(), a_sigma, b_sigma, rng);
  }

  // Posterior predictions
  // mu_preds[j*n + i] = mu_hat for sample j, obs i  (column-major)
  // tau_preds[j*n + i] = tau_hat(i)*z(i)  (since basis is z)
  std::vector<double> mu_preds  = mu_samples.Predict(dataset_mu);
  std::vector<double> tau_preds = tau_samples.Predict(dataset_tau);

  double mu_rmse_sum = 0.0;
  double tau_rmse_sum = 0.0;
  int n_treated = 0;

  for (int i = 0; i < n; i++) {
    double mu_hat = y_bar;
    for (int j = 0; j < num_mcmc; j++)
      mu_hat += mu_preds[static_cast<std::size_t>(j * n + i)] / num_mcmc;
    double mu_err = mu_hat - data.mu_true(i);
    mu_rmse_sum += mu_err * mu_err;

    // For z=1: tau_preds = tau_hat * 1 = tau_hat, so we can evaluate CATE
    if (data.z(i) > 0.5) {
      double tau_hat = 0.0;
      for (int j = 0; j < num_mcmc; j++)
        tau_hat += tau_preds[static_cast<std::size_t>(j * n + i)] / num_mcmc;
      double tau_err = tau_hat - data.tau_true(i);
      tau_rmse_sum += tau_err * tau_err;
      n_treated++;
    }
  }

  std::cout << "\nScenario 0 (BCF: constant mu + univariate tau with Z basis):\n"
            << "  mu RMSE:             " << std::sqrt(mu_rmse_sum / n) << "\n"
            << "  tau RMSE (treated):  "
            << (n_treated > 0 ? std::sqrt(tau_rmse_sum / n_treated) : 0.0) << "\n"
            << "  sigma (last sample): " << std::sqrt(global_variance) << "\n"
            << "  sigma (truth):       0.5\n";
}

// ---- Main -----------------------------------------------------------

int main(int argc, char** argv) {
  int scenario = 0;
  if (argc > 1) scenario = std::stoi(argv[1]);

  constexpr int n = 200, p = 5, num_trees = 200, num_gfr = 20, num_mcmc = 100;

  switch (scenario) {
    case 0:
      run_scenario_0(n, p, num_trees, num_gfr, num_mcmc);
      break;
    default:
      std::cerr << "Unknown scenario " << scenario
                << ". Available scenarios: 0 (BasicBCF)\n";
      return 1;
  }
  return 0;
}
