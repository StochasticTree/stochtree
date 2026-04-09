/*
 * BART debug program. The first CLI argument selects the scenario (default: 0).
 *
 * Usage: bart_debug [--scenario N] [--n N] [--n_test N] [--p N] [--num_trees N]
 *                   [--num_gfr N] [--num_mcmc N] [--seed N]
 *
 *   0  Homoskedastic constant-leaf BART
 *      DGP: y = sin(2*pi*x1) + 0.5*x2 - 1.5*x3 + eps, eps ~ N(0,1)
 *
 *   1  Homoskedastic constant-leaf probit BART
 *      DGP: Z = sin(2*pi*x1) + 0.5*x2 - 1.5*x3 + eps, eps ~ N(0,1)
 *           y = 1{Z > 0}
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

#include <cmath>
#include <numeric>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static constexpr double kPi = 3.14159265358979323846;

// ---- Data ------------------------------------------------------------

struct RegressionDataset {
  std::vector<double> X;
  std::vector<double> y;
};

struct ProbitDataset {
  std::vector<double> X;
  std::vector<double> y;
  std::vector<double> Z;
};

// DGP: y ~ sin(2*pi*x1) + 0.5*x2 - 1.5*x3 + N(0,1)
RegressionDataset generate_constant_leaf_regression_data(int n, int p, std::mt19937& rng) {
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  std::normal_distribution<double> normal(0.0, 1.0);
  RegressionDataset d;
  d.X.resize(n * p);
  d.y.resize(n);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < p; j++)
      d.X[j * n + i] = unif(rng);
  for (int i = 0; i < n; i++)
    d.y[i] = std::sin(2.0 * kPi * d.X[i]) + 0.5 * d.X[1 * n + i] - 1.5 * d.X[2 * n + i] + normal(rng);
  return d;
}

// DGP
// ---
// Z ~ sin(2*pi*x1) + 0.5*x2 - 1.5*x3 + N(0,1)
// y = 1{Z > 0}
ProbitDataset generate_probit_data(int n, int p, std::mt19937& rng) {
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  std::normal_distribution<double> normal(0.0, 1.0);
  ProbitDataset d;
  d.X.resize(n * p);
  d.y.resize(n);
  d.Z.resize(n);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < p; j++)
      d.X[j * n + i] = unif(rng);
  for (int i = 0; i < n; i++) {
    d.Z[i] = std::sin(2.0 * kPi * d.X[i]) + 0.5 * d.X[1 * n + i] - 1.5 * d.X[2 * n + i] + normal(rng);
    d.y[i] = (d.Z[i] > 0) ? 1.0 : 0.0;
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

void run_bart_sampler(int n, int n_test, int p, int num_trees, int num_gfr, int num_mcmc,
                      StochTree::ForestDataset& dataset,
                      StochTree::ColumnVector& residual, std::mt19937& rng,
                      StochTree::ForestDataset& test_dataset,
                      PostIterFn post_iter, ReportFn report_results) {
  // Single-threaded with default cutpoint grid size (for now)
  constexpr int num_threads = 1;
  constexpr int cutpoint_grid_size = 100;

  // Model parameters for split rule selection and tree sweeps
  std::vector<StochTree::FeatureType> feature_types(p, StochTree::FeatureType::kNumeric);
  std::vector<double> var_weights(p, 1.0 / p);
  std::vector<int> sweep_indices(num_trees);
  std::iota(sweep_indices.begin(), sweep_indices.end(), 0);

  // Ephemeral sampler state
  StochTree::TreePrior tree_prior(0.95, 2.0, /*min_samples_leaf=*/5);
  StochTree::ForestContainer forest_samples(num_trees, /*output_dim=*/1, /*leaf_constant=*/true, /*exponentiated=*/false);
  StochTree::TreeEnsemble active_forest(num_trees, 1, true, false);
  StochTree::ForestTracker tracker(dataset.GetCovariates(), feature_types, num_trees, n);

  // Initialize forest and tracker predictions to 0 (after standardization, this is the best initial guess)
  active_forest.SetLeafValue(0.0);
  UpdateResidualEntireForest(tracker, dataset, residual, &active_forest, false, std::minus<double>());
  tracker.UpdatePredictions(&active_forest, dataset);

  // Initialize leaf model and global variance for sampling iterations
  StochTree::GaussianConstantLeafModel leaf_model(1.0 / num_trees);
  double global_variance = 1.0;

  // Run GFR
  std::cout << "[GFR]  " << num_gfr << " warmup iterations...\n";
  bool pre_initialized = true;
  for (int i = 0; i < num_gfr; i++) {
    // Sample forest
    StochTree::GFRSampleOneIter<
        StochTree::GaussianConstantLeafModel,
        StochTree::GaussianConstantSuffStat>(
        active_forest, tracker, forest_samples, leaf_model,
        dataset, residual, tree_prior, rng,
        var_weights, sweep_indices, global_variance, feature_types,
        cutpoint_grid_size, /*keep_forest=*/false, pre_initialized,
        /*backfitting=*/true, /*num_features_subsample=*/p, num_threads);

    // Sample other model parameters (e.g. global variance, probit data augmentation, etc.)
    post_iter(tracker, global_variance);
  }

  // Run MCMC
  std::cout << "[MCMC] " << num_mcmc << " sampling iterations...\n";
  for (int i = 0; i < num_mcmc; i++) {
    // Sample forest
    StochTree::MCMCSampleOneIter<
        StochTree::GaussianConstantLeafModel,
        StochTree::GaussianConstantSuffStat>(
        active_forest, tracker, forest_samples, leaf_model,
        dataset, residual, tree_prior, rng,
        var_weights, sweep_indices, global_variance,
        /*keep_forest=*/true, /*pre_initialized=*/true,
        /*backfitting=*/true, num_threads);

    // Sample other model parameters (e.g. global variance, probit data augmentation, etc.)
    post_iter(tracker, global_variance);
  }

  // Analyze posterior predictions (column-major, element [j*n_test + i] = sample j, obs i)
  report_results(forest_samples.Predict(test_dataset), global_variance);
}

// ---- Scenario 0: homoskedastic constant-leaf BART -------------------

void run_scenario_0(int n, int n_test, int p, int num_trees, int num_gfr, int num_mcmc, int seed = 1234) {
  // Allow seed to be non-deterministic if set to sentinel value of -1
  int rng_seed;
  if (seed == -1) {
    std::random_device rd;
    rng_seed = rd();
  } else {
    rng_seed = seed;
  }
  std::mt19937 rng(rng_seed);

  // Generate data
  RegressionDataset data = generate_constant_leaf_regression_data(n, p, rng);
  double y_bar = std::accumulate(data.y.begin(), data.y.end(), 0.0) / data.y.size();
  double y_std = 0;
  for (int i = 0; i < n; i++) {
    y_std += (data.y[i] - y_bar) * (data.y[i] - y_bar);
  }
  y_std = std::sqrt(y_std / n);
  std::vector<double> resid_vec(data.y.size());
  for (std::size_t i = 0; i < data.y.size(); i++) {
    resid_vec[i] = (data.y[i] - y_bar) / y_std;
  }

  // Initialize dataset and residual vector for sampler
  StochTree::ForestDataset dataset;
  dataset.AddCovariates(data.X.data(), n, p, /*row_major=*/false);
  StochTree::ColumnVector residual(resid_vec.data(), n);

  // Initialize global error variance model
  constexpr double a_sigma = 0.0, b_sigma = 0.0;  // non-informative IG prior
  StochTree::GlobalHomoskedasticVarianceModel var_model;

  // Lambda function for sampling global error variance after each forest sample
  auto post_iter = [&](StochTree::ForestTracker&, double& global_variance) {
    global_variance = var_model.SampleVarianceParameter(residual.GetData(), a_sigma, b_sigma, rng);
  };

  // Generate test data and build test dataset
  RegressionDataset test_data = generate_constant_leaf_regression_data(n_test, p, rng);
  StochTree::ForestDataset test_dataset;
  test_dataset.AddCovariates(test_data.X.data(), n_test, p, /*row_major=*/false);

  // Lambda function for reporting test-set RMSE and last draw of global error variance model
  auto report = [&](const std::vector<double>& preds, double global_variance) {
    double rmse_sum = 0.0;
    for (int i = 0; i < n_test; i++) {
      double mu_hat = 0.0;
      for (int j = 0; j < num_mcmc; j++)
        mu_hat += preds[static_cast<std::size_t>(j * n_test + i)] / num_mcmc;
      double err = (mu_hat * y_std + y_bar) - test_data.y[i];
      rmse_sum += err * err;
    }
    std::cout << "\nScenario 0 (Homoskedastic BART):\n"
              << "  RMSE (test):         " << std::sqrt(rmse_sum / n_test) << "\n"
              << "  sigma (last sample): " << std::sqrt(global_variance) * y_std << "\n"
              << "  sigma (truth):       1.0\n";
  };

  // Dispatch BART sampler
  run_bart_sampler(n, n_test, p, num_trees, num_gfr, num_mcmc, dataset, residual, rng, test_dataset, post_iter, report);
}

// ---- Scenario 1: constant-leaf probit BART -------------------

void run_scenario_1(int n, int n_test, int p, int num_trees, int num_gfr, int num_mcmc, int seed = 1234) {
  // Allow seed to be non-deterministic if set to sentinel value of -1
  int rng_seed;
  if (seed == -1) {
    std::random_device rd;
    rng_seed = rd();
  } else {
    rng_seed = seed;
  }
  std::mt19937 rng(rng_seed);

  // Generate data
  ProbitDataset data = generate_probit_data(n, p, rng);
  double y_bar = std::accumulate(data.y.begin(), data.y.end(), 0.0) / data.y.size();
  std::vector<double> y_vec = data.y;
  std::vector<double> Z_vec(n);
  for (int i = 0; i < n; i++) {
    Z_vec[i] = data.y[i] - y_bar;
  }

  // Initialize dataset and residual vector for sampler
  StochTree::ForestDataset dataset;
  dataset.AddCovariates(data.X.data(), n, p, /*row_major=*/false);
  StochTree::ColumnVector residual(Z_vec.data(), n);

  // Lambda function for probit data augmentation sampling step (after each forest sample)
  auto post_iter = [&](StochTree::ForestTracker& tracker, double&) {
    StochTree::sample_probit_latent_outcome(
        rng, y_vec.data(), tracker.GetSumPredictions(), residual.GetData().data(), y_bar, n);
  };

  // Generate test data and build test dataset
  ProbitDataset test_data = generate_probit_data(n_test, p, rng);
  StochTree::ForestDataset test_dataset;
  test_dataset.AddCovariates(test_data.X.data(), n_test, p, /*row_major=*/false);

  // Lambda function for reporting test-set RMSE
  auto report = [&](const std::vector<double>& preds, double global_variance) {
    double rmse_sum = 0.0;
    for (int i = 0; i < n_test; i++) {
      double mu_hat = 0.0;
      for (int j = 0; j < num_mcmc; j++)
        mu_hat += preds[static_cast<std::size_t>(j * n_test + i)] / num_mcmc;
      double err = (mu_hat + y_bar) - test_data.Z[i];
      rmse_sum += err * err;
    }
    std::cout << "\nScenario 1 (Probit BART):\n"
              << "  RMSE (test):         " << std::sqrt(rmse_sum / n_test) << "\n"
              << "  sigma (truth):       1.0\n";
  };

  // Dispatch BART sampler
  run_bart_sampler(n, n_test, p, num_trees, num_gfr, num_mcmc, dataset, residual, rng, test_dataset, post_iter, report);
}

// ---- Main -----------------------------------------------------------

int main(int argc, char** argv) {
  int scenario = 1;
  int n = 500;
  int n_test = 100;
  int p = 5;
  int num_trees = 200;
  int num_gfr = 10;
  int num_mcmc = 100;
  int seed = 1234;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "--scenario" || arg == "--n" || arg == "--n_test" || arg == "--p" ||
         arg == "--num_trees" || arg == "--num_gfr" || arg == "--num_mcmc" || arg == "--seed") &&
        i + 1 < argc) {
      int val = std::stoi(argv[++i]);
      if (arg == "--scenario")
        scenario = val;
      else if (arg == "--n")
        n = val;
      else if (arg == "--n_test")
        n_test = val;
      else if (arg == "--p")
        p = val;
      else if (arg == "--num_trees")
        num_trees = val;
      else if (arg == "--num_gfr")
        num_gfr = val;
      else if (arg == "--num_mcmc")
        num_mcmc = val;
      else if (arg == "--seed")
        seed = val;
    } else {
      std::cerr << "Unknown or incomplete argument: " << arg << "\n"
                << "Usage: bart_debug [--scenario N] [--n N] [--n_test N] [--p N]"
                   " [--num_trees N] [--num_gfr N] [--num_mcmc N] [--seed N]\n";
      return 1;
    }
  }

  switch (scenario) {
    case 0:
      run_scenario_0(n, n_test, p, num_trees, num_gfr, num_mcmc, seed);
      break;
    case 1:
      run_scenario_1(n, n_test, p, num_trees, num_gfr, num_mcmc, seed);
      break;
    default:
      std::cerr << "Unknown scenario " << scenario
                << ". Available scenarios: 0 (Homoskedastic BART), 1 (Probit BART)\n";
      return 1;
  }
  return 0;
}
