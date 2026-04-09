/*
 * BART debug driver. The first CLI argument selects the scenario (default: 0).
 *
 * Usage: bart_debug [scenario]
 *   0  Homoskedastic constant-leaf BART
 *      DGP: y = sin(2*pi*x1) + 0.5*x2 - 1.5*x3 + eps, eps ~ N(0,1)
 *
 * Add scenarios here as the BARTSampler API develops (heteroskedastic,
 * random effects, multivariate leaf, etc.).
 */

#include <stochtree/container.h>
#include <stochtree/data.h>
#include <stochtree/distributions.h>
#include <stochtree/leaf_model.h>
#include <stochtree/log.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/probit.h>
#include <stochtree/tree_sampler.h>
#include <stochtree/variance_model.h>

#include <Eigen/Dense>
#include <cmath>
#include <numeric>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static constexpr double kPi = 3.14159265358979323846;

// ---- Data ------------------------------------------------------------

struct RegressionDataset {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X;
  Eigen::VectorXd y;
};

struct ProbitDataset {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X;
  Eigen::VectorXd y;
  Eigen::VectorXd Z;
};

// DGP: y ~ sin(2*pi*x1) + 0.5*x2 - 1.5*x3 + N(0,1)
RegressionDataset generate_constant_leaf_regression_data(int n, int p, std::mt19937& rng) {
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  std::normal_distribution<double> normal(0.0, 1.0);
  RegressionDataset d;
  d.X.resize(n, p);
  d.y.resize(n);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < p; j++)
      d.X(i, j) = unif(rng);
  for (int i = 0; i < n; i++)
    d.y(i) = std::sin(2.0 * kPi * d.X(i, 0)) + 0.5 * d.X(i, 1) - 1.5 * d.X(i, 2) + normal(rng);
  return d;
}

// DGP
// ---
// Z ~ sin(2*pi*x1) + 0.5*x2 - 1.5*x3 + N(0,1)
// y = 1{Z > 0}
ProbitDataset generate_probit_data(int n, int p, std::mt19937& rng) {
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  std::normal_distribution<double> normal(0.0, 1.0);
  Eigen::VectorXd Z;
  ProbitDataset d;
  d.X.resize(n, p);
  d.y.resize(n);
  d.Z.resize(n);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < p; j++)
      d.X(i, j) = unif(rng);
  for (int i = 0; i < n; i++) {
    d.Z(i) = std::sin(2.0 * kPi * d.X(i, 0)) + 0.5 * d.X(i, 1) - 1.5 * d.X(i, 2) + normal(rng);
    d.y(i) = (d.Z(i) > 0) ? 1.0 : 0.0;
  }
  return d;
}

// ---- Shared sampler loop --------------------------------------------
//
// Runs GFR warmup then MCMC sampling, both using the same forest/leaf/variance
// setup.  The two scenario-specific hooks are:
//
//   post_iter(tracker, global_variance) — called after every forest sample in
//       both GFR and MCMC (e.g. sample global variance, or augment latent Z).
//
//   report_results(preds, global_variance) — called once after all samples are
//       collected; receives the flat column-major predictions matrix and the
//       final global variance value.

using PostIterFn = std::function<void(StochTree::ForestTracker&, double&)>;
using ReportFn = std::function<void(const std::vector<double>&, double)>;

void run_bart_sampler(int n, int p, int num_trees, int num_gfr, int num_mcmc,
                      StochTree::ForestDataset& dataset,
                      StochTree::ColumnVector& residual, std::mt19937& rng,
                      PostIterFn post_iter, ReportFn report_results) {
  constexpr int num_threads = 1;
  constexpr int cutpoint_grid_size = 100;

  std::vector<StochTree::FeatureType> feature_types(p, StochTree::FeatureType::kNumeric);
  std::vector<double> var_weights(p, 1.0 / p);
  std::vector<int> sweep_indices(num_trees);
  std::iota(sweep_indices.begin(), sweep_indices.end(), 0);

  StochTree::TreePrior tree_prior(0.95, 2.0, /*min_samples_leaf=*/5);
  StochTree::ForestContainer forest_samples(num_trees, /*output_dim=*/1, /*leaf_constant=*/true, /*exponentiated=*/false);
  StochTree::TreeEnsemble active_forest(num_trees, 1, true, false);
  StochTree::ForestTracker tracker(dataset.GetCovariates(), feature_types, num_trees, n);

  active_forest.SetLeafValue(0.0);
  UpdateResidualEntireForest(tracker, dataset, residual, &active_forest, false, std::minus<double>());
  tracker.UpdatePredictions(&active_forest, dataset);

  StochTree::GaussianConstantLeafModel leaf_model(1.0 / num_trees);
  double global_variance = 1.0;

  std::cout << "[GFR]  " << num_gfr << " warmup iterations...\n";
  bool pre_initialized = true;
  for (int i = 0; i < num_gfr; i++) {
    StochTree::GFRSampleOneIter<
        StochTree::GaussianConstantLeafModel,
        StochTree::GaussianConstantSuffStat>(
        active_forest, tracker, forest_samples, leaf_model,
        dataset, residual, tree_prior, rng,
        var_weights, sweep_indices, global_variance, feature_types,
        cutpoint_grid_size, /*keep_forest=*/false, pre_initialized,
        /*backfitting=*/true, /*num_features_subsample=*/p, num_threads);
    post_iter(tracker, global_variance);
  }

  std::cout << "[MCMC] " << num_mcmc << " sampling iterations...\n";
  for (int i = 0; i < num_mcmc; i++) {
    StochTree::MCMCSampleOneIter<
        StochTree::GaussianConstantLeafModel,
        StochTree::GaussianConstantSuffStat>(
        active_forest, tracker, forest_samples, leaf_model,
        dataset, residual, tree_prior, rng,
        var_weights, sweep_indices, global_variance,
        /*keep_forest=*/true, /*pre_initialized=*/true,
        /*backfitting=*/true, num_threads);
    post_iter(tracker, global_variance);
  }

  // Posterior predictions: column-major, element [j*n + i] = sample j, obs i
  report_results(forest_samples.Predict(dataset), global_variance);
}

// ---- Scenario 0: homoskedastic constant-leaf BART -------------------

void run_scenario_0(int n, int p, int num_trees, int num_gfr, int num_mcmc, int seed = 1234) {
  std::mt19937 rng(seed);

  RegressionDataset data = generate_constant_leaf_regression_data(n, p, rng);
  double y_bar = data.y.mean();
  double y_std = std::sqrt((data.y.array() - y_bar).square().sum() / (data.y.size() - 1));
  Eigen::VectorXd resid_vec = (data.y.array() - y_bar) / y_std;  // standardize

  StochTree::ForestDataset dataset;
  dataset.AddCovariates(data.X.data(), n, p, /*row_major=*/true);
  StochTree::ColumnVector residual(resid_vec.data(), n);

  constexpr double a_sigma = 0.0, b_sigma = 0.0;  // non-informative IG prior
  StochTree::GlobalHomoskedasticVarianceModel var_model;

  auto post_iter = [&](StochTree::ForestTracker&, double& global_variance) {
    global_variance = var_model.SampleVarianceParameter(residual.GetData(), a_sigma, b_sigma, rng);
  };

  auto report = [&](const std::vector<double>& preds, double global_variance) {
    double rmse_sum = 0.0;
    for (int i = 0; i < n; i++) {
      double mu_hat = 0.0;
      for (int j = 0; j < num_mcmc; j++)
        mu_hat += preds[static_cast<std::size_t>(j * n + i)] / num_mcmc;
      double err = (mu_hat * y_std + y_bar) - data.y(i);
      rmse_sum += err * err;
    }
    std::cout << "\nScenario 0 (Homoskedastic BART):\n"
              << "  RMSE:                " << std::sqrt(rmse_sum / n) << "\n"
              << "  sigma (last sample): " << std::sqrt(global_variance) * y_std << "\n"
              << "  sigma (truth):       1.0\n";
  };

  run_bart_sampler(n, p, num_trees, num_gfr, num_mcmc, dataset, residual, rng, post_iter, report);
}

// ---- Scenario 1: constant-leaf probit BART -------------------

void run_scenario_1(int n, int p, int num_trees, int num_gfr, int num_mcmc, int seed = 1234) {
  std::mt19937 rng(seed);

  ProbitDataset data = generate_probit_data(n, p, rng);
  double y_bar = StochTree::norm_cdf(data.y.mean());
  Eigen::VectorXd y_vec = data.y.array();
  Eigen::VectorXd Z_vec = (data.y.array() - y_bar);

  StochTree::ForestDataset dataset;
  dataset.AddCovariates(data.X.data(), n, p, /*row_major=*/true);
  StochTree::ColumnVector residual(Z_vec.data(), n);

  // Data augmentation: sample latent Z given current forest predictions
  auto post_iter = [&](StochTree::ForestTracker& tracker, double&) {
    StochTree::sample_probit_latent_outcome(
        rng, y_vec.data(), tracker.GetSumPredictions(), residual.GetData().data(), n);
  };

  auto report = [&](const std::vector<double>& preds, double global_variance) {
    double rmse_sum = 0.0;
    for (int i = 0; i < n; i++) {
      double mu_hat = 0.0;
      for (int j = 0; j < num_mcmc; j++)
        mu_hat += preds[static_cast<std::size_t>(j * n + i)] / num_mcmc;
      double err = (mu_hat + y_bar) - data.Z(i);
      rmse_sum += err * err;
    }
    std::cout << "\nScenario 1 (Probit BART):\n"
              << "  RMSE:                " << std::sqrt(rmse_sum / n) << "\n"
              << "  sigma (truth):       1.0\n";
  };

  run_bart_sampler(n, p, num_trees, num_gfr, num_mcmc, dataset, residual, rng, post_iter, report);
}

// ---- Main -----------------------------------------------------------

int main(int argc, char** argv) {
  int scenario = 1;
  int n = 500;
  int p = 5;
  int num_trees = 200;
  int num_gfr = 20;
  int num_mcmc = 100;
  int seed = 1234;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "--scenario" || arg == "--n" || arg == "--p" ||
         arg == "--num_trees" || arg == "--num_gfr" || arg == "--num_mcmc" || arg == "--seed") &&
        i + 1 < argc) {
      int val = std::stoi(argv[++i]);
      if (arg == "--scenario")
        scenario = val;
      else if (arg == "--n")
        n = val;
      else if (arg == "--p")
        p = val;
      else if (arg == "--num_trees")
        num_trees = val;
      else if (arg == "--num_gfr")
        num_gfr = val;
      else if (arg == "--num_mcmc")
        num_mcmc = val;
      else if (arg == "--num_mcmc")
        seed = val;
    } else {
      std::cerr << "Unknown or incomplete argument: " << arg << "\n"
                << "Usage: bart_debug [--scenario N] [--n N] [--p N]"
                   " [--num_trees N] [--num_gfr N] [--num_mcmc N]\n";
      return 1;
    }
  }

  switch (scenario) {
    case 0:
      run_scenario_0(n, p, num_trees, num_gfr, num_mcmc);
      break;
    case 1:
      run_scenario_1(n, p, num_trees, num_gfr, num_mcmc);
      break;
    default:
      std::cerr << "Unknown scenario " << scenario
                << ". Available scenarios: 0 (Homoskedastic BART), 1 (Probit BART)\n";
      return 1;
  }
  return 0;
}
