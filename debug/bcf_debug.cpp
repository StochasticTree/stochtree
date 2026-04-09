/*
 * BCF debug program
 *
 * Usage: bcf_debug [--scenario N] [--n N] [--n_test N] [--p N] [--num_trees_mu N] [--num_trees_tau N]
 *                  [--num_gfr N] [--num_mcmc N] [--seed N]
 *
 *   0  Two-forest BCF: constant-leaf mu, univariate-leaf tau (Z as basis)
 *      DGP: mu(x) = 2*sin(pi*x1) + 0.5*x2
 *           tau(x) = 1 + x3
 *           z ~ Bernoulli(0.5)
 *           y = mu(x) + tau(x)*z + N(0, 0.5^2)
 *
 *   1  Two-forest BCF: constant-leaf mu, univariate-leaf tau (Z as basis)
 *      DGP: mu(x) = 2*sin(pi*x1) + 0.5*x2
 *           tau(x) = 1 + x3
 *           z ~ Bernoulli(0.5)
 *           W = mu(x) + tau(x)*z + N(0, 1)
 *           y = 1{W > 0}
 *
 * Add scenarios here as the BCFSampler API develops (heteroskedastic,
 * random effects, propensity weighting, etc.).
 */

#include <stochtree/container.h>
#include <stochtree/data.h>
#include <stochtree/leaf_model.h>
#include <stochtree/log.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/probit.h>
#include <stochtree/tree_sampler.h>
#include <stochtree/variance_model.h>

#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

static constexpr double kPi = 3.14159265358979323846;

// ---- Data ------------------------------------------------------------

struct SimpleBCFDataset {
  std::vector<double> X;
  std::vector<double> y;
  std::vector<double> z;
  std::vector<double> mu_true;
  std::vector<double> tau_true;
};

struct ProbitBCFDataset {
  std::vector<double> X;
  std::vector<double> y;
  std::vector<double> latent_outcome;
  std::vector<double> z;
  std::vector<double> mu_true;
  std::vector<double> tau_true;
};

SimpleBCFDataset generate_simple_bcf_data(int n, int p, std::mt19937& rng) {
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  std::normal_distribution<double> normal(0.0, 1.0);
  std::bernoulli_distribution bern(0.5);

  SimpleBCFDataset d;
  d.X.resize(n * p);
  d.y.resize(n);
  d.z.resize(n);
  d.mu_true.resize(n);
  d.tau_true.resize(n);

  for (int i = 0; i < n; i++)
    for (int j = 0; j < p; j++)
      d.X[j * n + i] = unif(rng);

  for (int i = 0; i < n; i++) {
    d.z[i] = bern(rng) ? 1.0 : 0.0;
    d.mu_true[i] = 2.0 * std::sin(kPi * d.X[i]) + 0.5 * d.X[1 * n + i];
    d.tau_true[i] = 1.0 + d.X[2 * n + i];
    d.y[i] = d.mu_true[i] + d.tau_true[i] * d.z[i] + 0.5 * normal(rng);
  }
  return d;
}

ProbitBCFDataset generate_probit_bcf_data(int n, int p, std::mt19937& rng) {
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  std::normal_distribution<double> normal(0.0, 1.0);
  std::bernoulli_distribution bern(0.5);

  ProbitBCFDataset d;
  d.X.resize(n * p);
  d.y.resize(n);
  d.z.resize(n);
  d.mu_true.resize(n);
  d.tau_true.resize(n);
  d.latent_outcome.resize(n);

  for (int i = 0; i < n; i++)
    for (int j = 0; j < p; j++)
      d.X[j * n + i] = unif(rng);

  for (int i = 0; i < n; i++) {
    d.z[i] = bern(rng) ? 1.0 : 0.0;
    d.mu_true[i] = 2.0 * std::sin(kPi * d.X[i]) + 0.5 * d.X[1 * n + i];
    d.tau_true[i] = 1.0 + d.X[2 * n + i];
    d.latent_outcome[i] = d.mu_true[i] + d.tau_true[i] * d.z[i] + normal(rng);
    d.y[i] = (d.latent_outcome[i] > 0.0) ? 1.0 : 0.0;
  }
  return d;
}

// ---- Shared sampler loop --------------------------------------------
//
// Runs alternating mu/tau GFR warmup then MCMC, sharing a single residual.
// The two scenario-specific hooks are:
//
//   post_iter(mu_tracker, global_variance) — called after each full mu+tau
//       iteration (e.g. sample global variance).
//
//   report_results(mu_preds, tau_preds, global_variance) — called once after
//       all samples are collected; receives column-major prediction matrices
//       and the final global variance value.

using PostIterFn = std::function<void(double*, double&)>;
using BCFReportFn = std::function<void(const std::vector<double>&, const std::vector<double>&, double)>;

void run_bcf_sampler(int n, int n_test, int p, int num_trees_mu, int num_trees_tau, int num_gfr, int num_mcmc,
                     StochTree::ForestDataset& dataset,
                     StochTree::ColumnVector& residual, std::mt19937& rng,
                     StochTree::ForestDataset& test_dataset,
                     PostIterFn post_iter, BCFReportFn report_results) {
  // Single-threaded with default cutpoint grid size (for now)
  constexpr int num_threads = 1;
  constexpr int cutpoint_grid_size = 100;

  // Model parameters for split rule selection and tree sweeps
  std::vector<StochTree::FeatureType> feature_types(p, StochTree::FeatureType::kNumeric);
  std::vector<double> var_weights(p, 1.0 / p);
  std::vector<int> sweep_indices_mu(num_trees_mu);
  std::iota(sweep_indices_mu.begin(), sweep_indices_mu.end(), 0);
  std::vector<int> sweep_indices_tau(num_trees_tau);
  std::iota(sweep_indices_tau.begin(), sweep_indices_tau.end(), 0);

  // Ephemeral sampler state
  // Mu forest: constant-leaf
  StochTree::TreePrior mu_tree_prior(0.95, 2.0, /*min_samples_leaf=*/5);
  StochTree::ForestContainer mu_samples(num_trees_mu, /*output_dim=*/1, /*leaf_constant=*/true, /*exponentiated=*/false);
  StochTree::TreeEnsemble mu_forest(num_trees_mu, 1, true, false);
  StochTree::ForestTracker mu_tracker(dataset.GetCovariates(), feature_types, num_trees_mu, n);
  StochTree::GaussianConstantLeafModel mu_leaf_model(1.0 / num_trees_mu);

  // Tau forest: univariate regression leaf (prediction = leaf_param * z)
  StochTree::TreePrior tau_tree_prior(0.5, 2.0, /*min_samples_leaf=*/5);
  StochTree::ForestContainer tau_samples(num_trees_tau, /*output_dim=*/1, /*leaf_constant=*/false, /*exponentiated=*/false);
  StochTree::TreeEnsemble tau_forest(num_trees_tau, 1, false, false);
  StochTree::ForestTracker tau_tracker(dataset.GetCovariates(), feature_types, num_trees_tau, n);
  StochTree::GaussianUnivariateRegressionLeafModel tau_leaf_model(1.0 / num_trees_tau);

  // Initialize mu forest and tracker predictions to 0
  mu_forest.SetLeafValue(0.0);
  UpdateResidualEntireForest(mu_tracker, dataset, residual, &mu_forest, false, std::minus<double>());
  mu_tracker.UpdatePredictions(&mu_forest, dataset);

  // Initial tau forest and tracker predictions to 0
  tau_forest.SetLeafValue(0.0);
  UpdateResidualEntireForest(tau_tracker, dataset, residual, &tau_forest, false, std::minus<double>());
  tau_tracker.UpdatePredictions(&tau_forest, dataset);

  // Model predictions
  std::vector<double> outcome_preds(n, 0.0);

  // Initialize global error variance to 1 (output is standardized)
  double global_variance = 1.0;

  // Run GFR
  std::cout << "[GFR]  " << num_gfr << " warmup iterations...\n";
  bool pre_initialized = true;
  for (int i = 0; i < num_gfr; i++) {
    // Sample mu forest
    StochTree::GFRSampleOneIter<
        StochTree::GaussianConstantLeafModel,
        StochTree::GaussianConstantSuffStat>(
        mu_forest, mu_tracker, mu_samples, mu_leaf_model,
        dataset, residual, mu_tree_prior, rng,
        var_weights, sweep_indices_mu, global_variance, feature_types,
        cutpoint_grid_size, /*keep_forest=*/false, pre_initialized,
        /*backfitting=*/true, /*num_features_subsample=*/p, num_threads);

    // Sample tau forest
    StochTree::GFRSampleOneIter<
        StochTree::GaussianUnivariateRegressionLeafModel,
        StochTree::GaussianUnivariateRegressionSuffStat>(
        tau_forest, tau_tracker, tau_samples, tau_leaf_model,
        dataset, residual, tau_tree_prior, rng,
        var_weights, sweep_indices_tau, global_variance, feature_types,
        cutpoint_grid_size, /*keep_forest=*/false, pre_initialized,
        /*backfitting=*/true, /*num_features_subsample=*/p, num_threads);

    // Update predictions and residual for post-iteration hook (e.g. global variance sampling, probit data augmentation, etc.)
    for (int j = 0; j < n; j++) {
      outcome_preds[j] = mu_tracker.GetSamplePrediction(j) + tau_tracker.GetSamplePrediction(j);
    }

    // Sample other model parameters (e.g. global variance, probit data augmentation, etc.)
    post_iter(outcome_preds.data(), global_variance);
  }

  // Run MCMC
  std::cout << "[MCMC] " << num_mcmc << " sampling iterations...\n";
  for (int i = 0; i < num_mcmc; i++) {
    // Sample mu forest
    StochTree::MCMCSampleOneIter<
        StochTree::GaussianConstantLeafModel,
        StochTree::GaussianConstantSuffStat>(
        mu_forest, mu_tracker, mu_samples, mu_leaf_model,
        dataset, residual, mu_tree_prior, rng,
        var_weights, sweep_indices_mu, global_variance,
        /*keep_forest=*/true, /*pre_initialized=*/true,
        /*backfitting=*/true, num_threads);

    // Sample tau forest
    StochTree::MCMCSampleOneIter<
        StochTree::GaussianUnivariateRegressionLeafModel,
        StochTree::GaussianUnivariateRegressionSuffStat>(
        tau_forest, tau_tracker, tau_samples, tau_leaf_model,
        dataset, residual, tau_tree_prior, rng,
        var_weights, sweep_indices_tau, global_variance,
        /*keep_forest=*/true, /*pre_initialized=*/true,
        /*backfitting=*/true, num_threads);

    // Update predictions and residual for post-iteration hook (e.g. global variance sampling, probit data augmentation, etc.)
    for (int j = 0; j < n; j++) {
      outcome_preds[j] = mu_tracker.GetSamplePrediction(j) + tau_tracker.GetSamplePrediction(j);
    }

    // Sample other model parameters (e.g. global variance, probit data augmentation, etc.)
    post_iter(outcome_preds.data(), global_variance);
  }

  // Analyze posterior predictions (column-major, element [j*n_test + i] = sample j, obs i)
  report_results(mu_samples.Predict(test_dataset), tau_samples.PredictRaw(test_dataset), global_variance);
}

// ---- Scenario 0: constant-leaf mu + univariate-leaf tau (Z basis) ---

void run_scenario_0(int n, int n_test, int p, int num_trees_mu, int num_trees_tau, int num_gfr, int num_mcmc, int seed = 42) {
  // Allow seed to be non-deterministic if set to sentinel value of -1
  int rng_seed;
  if (seed == -1) {
    std::random_device rd;
    rng_seed = rd();
  } else {
    rng_seed = seed;
  }
  std::mt19937 rng(rng_seed);

  // Generate data and standardize outcome
  SimpleBCFDataset data = generate_simple_bcf_data(n, p, rng);
  double y_bar = std::accumulate(data.y.begin(), data.y.end(), 0.0) / data.y.size();
  double y_std = 0;
  for (int i = 0; i < n; i++) {
    y_std += (data.y[i] - y_bar) * (data.y[i] - y_bar);
  }
  y_std = std::sqrt(y_std / n);
  std::vector<double> resid_vec(n);
  for (int i = 0; i < n; i++) {
    resid_vec[i] = (data.y[i] - y_bar) / y_std;
  }

  // Shared dataset: only tau forest uses the Z basis for leaf regression
  StochTree::ForestDataset dataset;
  dataset.AddCovariates(data.X.data(), n, p, /*row_major=*/false);
  dataset.AddBasis(data.z.data(), n, /*num_col=*/1, /*row_major=*/false);

  // Shared residual
  StochTree::ColumnVector residual(resid_vec.data(), n);

  // Global error variance model
  constexpr double a_sigma = 0.0, b_sigma = 0.0;  // non-informative IG prior
  StochTree::GlobalHomoskedasticVarianceModel var_model;

  // Lambda function for sampling global error variance after each mu+tau step
  auto post_iter = [&](double* outcome_preds, double& global_variance) {
    global_variance = var_model.SampleVarianceParameter(residual.GetData(), a_sigma, b_sigma, rng);
  };

  // Generate test data and build test datasets
  SimpleBCFDataset test_data = generate_simple_bcf_data(n_test, p, rng);

  // Test dataset: covariates + actual treatment z (for y prediction)
  StochTree::ForestDataset test_dataset;
  test_dataset.AddCovariates(test_data.X.data(), n_test, p, /*row_major=*/false);
  test_dataset.AddBasis(test_data.z.data(), n_test, /*num_col=*/1, /*row_major=*/false);

  // Lambda function for reporting mu/tau RMSE and last draw of global error variance
  auto report = [&](const std::vector<double>& mu_preds, const std::vector<double>& tau_preds,
                    double global_variance) {
    double mu_rmse_sum = 0.0, tau_rmse_sum = 0.0, y_rmse_sum = 0.0;

    for (int i = 0; i < n_test; i++) {
      double mu_hat = 0.0;
      for (int j = 0; j < num_mcmc; j++)
        mu_hat += mu_preds[static_cast<std::size_t>(j * n_test + i)] / num_mcmc;
      mu_rmse_sum += (mu_hat * y_std + y_bar - test_data.mu_true[i]) * (mu_hat * y_std + y_bar - test_data.mu_true[i]);

      // tau_preds from test_dataset_cate (z=1 basis) => raw CATE estimates
      double cate_hat = 0.0;
      for (int j = 0; j < num_mcmc; j++)
        cate_hat += tau_preds[static_cast<std::size_t>(j * n_test + i)] / num_mcmc;
      tau_rmse_sum += (cate_hat * y_std - test_data.tau_true[i]) * (cate_hat * y_std - test_data.tau_true[i]);

      double y_hat = mu_hat * y_std + y_bar + cate_hat * test_data.z[i] * y_std;
      y_rmse_sum += (y_hat - test_data.y[i]) * (y_hat - test_data.y[i]);
    }

    std::cout << "\nScenario 0 (BCF: constant mu + univariate tau with Z basis):\n"
              << "  mu RMSE (test):      " << std::sqrt(mu_rmse_sum / n_test) << "\n"
              << "  tau RMSE (test):     " << std::sqrt(tau_rmse_sum / n_test) << "\n"
              << "  y RMSE (test):       " << std::sqrt(y_rmse_sum / n_test) << "\n"
              << "  sigma (last sample): " << std::sqrt(global_variance) * y_std << "\n"
              << "  sigma (truth):       0.5\n";
  };

  // Dispatch BCF sampler
  run_bcf_sampler(n, n_test, p, num_trees_mu, num_trees_tau, num_gfr, num_mcmc,
                  dataset, residual, rng, test_dataset, post_iter, report);
}

// ---- Scenario 1: constant-leaf mu + univariate-leaf tau (Z basis) with probit link ---

void run_scenario_1(int n, int n_test, int p, int num_trees_mu, int num_trees_tau, int num_gfr, int num_mcmc, int seed = 42) {
  // Allow seed to be non-deterministic if set to sentinel value of -1
  int rng_seed;
  if (seed == -1) {
    std::random_device rd;
    rng_seed = rd();
  } else {
    rng_seed = seed;
  }
  std::mt19937 rng(rng_seed);

  // Generate data and standardize outcome
  ProbitBCFDataset data = generate_probit_bcf_data(n, p, rng);
  double y_bar = std::accumulate(data.y.begin(), data.y.end(), 0.0) / data.y.size();
  std::vector<double> Z_vec(n);
  for (int i = 0; i < n; i++) {
    Z_vec[i] = data.y[i] - y_bar;
  }
  std::vector<double> y_vec = data.y;

  // Shared dataset: only tau forest uses the Z basis for leaf regression
  StochTree::ForestDataset dataset;
  dataset.AddCovariates(data.X.data(), n, p, /*row_major=*/true);
  dataset.AddBasis(data.z.data(), n, /*num_col=*/1, /*row_major=*/false);

  // Shared residual
  StochTree::ColumnVector residual(Z_vec.data(), n);

  // Global error variance model
  constexpr double a_sigma = 0.0, b_sigma = 0.0;  // non-informative IG prior
  StochTree::GlobalHomoskedasticVarianceModel var_model;

  // Lambda function for probit data augmentation sampling step (after each forest sample)
  auto post_iter = [&](double* outcome_preds, double&) {
    StochTree::sample_probit_latent_outcome(
        rng, y_vec.data(), outcome_preds, residual.GetData().data(), y_bar, n);
  };

  // Generate test data and build test datasets
  ProbitBCFDataset test_data = generate_probit_bcf_data(n_test, p, rng);

  // Test dataset: covariates + actual treatment z (for y prediction)
  StochTree::ForestDataset test_dataset;
  test_dataset.AddCovariates(test_data.X.data(), n_test, p, /*row_major=*/true);
  test_dataset.AddBasis(test_data.z.data(), n_test, /*num_col=*/1, /*row_major=*/false);

  // Lambda function for reporting mu/tau RMSE and last draw of global error variance
  auto report = [&](const std::vector<double>& mu_preds, const std::vector<double>& tau_preds,
                    double global_variance) {
    double mu_rmse_sum = 0.0, tau_rmse_sum = 0.0, y_rmse_sum = 0.0;

    for (int i = 0; i < n_test; i++) {
      double mu_hat = 0.0;
      for (int j = 0; j < num_mcmc; j++)
        mu_hat += mu_preds[static_cast<std::size_t>(j * n_test + i)] / num_mcmc;
      mu_rmse_sum += (mu_hat + y_bar - test_data.mu_true[i]) * (mu_hat + y_bar - test_data.mu_true[i]);

      // tau_preds from test_dataset_cate (z=1 basis) => raw CATE estimates
      double cate_hat = 0.0;
      for (int j = 0; j < num_mcmc; j++)
        cate_hat += tau_preds[static_cast<std::size_t>(j * n_test + i)] / num_mcmc;
      tau_rmse_sum += (cate_hat - test_data.tau_true[i]) * (cate_hat - test_data.tau_true[i]);

      double y_hat = mu_hat + y_bar + cate_hat * test_data.z[i];
      y_rmse_sum += (y_hat - test_data.latent_outcome[i]) * (y_hat - test_data.latent_outcome[i]);
    }

    std::cout << "\nScenario 0 (BCF: constant mu + univariate tau with Z basis):\n"
              << "  mu RMSE (test):      " << std::sqrt(mu_rmse_sum / n_test) << "\n"
              << "  tau RMSE (test):     " << std::sqrt(tau_rmse_sum / n_test) << "\n"
              << "  latent outcome RMSE (test):       " << std::sqrt(y_rmse_sum / n_test) << "\n"
              << "  sigma (last sample): " << std::sqrt(global_variance) << "\n"
              << "  sigma (truth):       1\n";
  };

  // Dispatch BCF sampler
  run_bcf_sampler(n, n_test, p, num_trees_mu, num_trees_tau, num_gfr, num_mcmc,
                  dataset, residual, rng, test_dataset, post_iter, report);
}

// ---- Main -----------------------------------------------------------

int main(int argc, char** argv) {
  int scenario = 0;
  int n = 500;
  int n_test = 100;
  int p = 5;
  int num_trees_mu = 200;
  int num_trees_tau = 50;
  int num_gfr = 20;
  int num_mcmc = 100;
  int seed = 1234;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "--scenario" || arg == "--n" || arg == "--n_test" || arg == "--p" ||
         arg == "--num_trees_mu" || arg == "--num_trees_tau" || arg == "--num_gfr" || arg == "--num_mcmc" || arg == "--seed") &&
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
      else if (arg == "--num_trees_mu")
        num_trees_mu = val;
      else if (arg == "--num_trees_tau")
        num_trees_tau = val;
      else if (arg == "--num_gfr")
        num_gfr = val;
      else if (arg == "--num_mcmc")
        num_mcmc = val;
      else if (arg == "--seed")
        seed = val;
    } else {
      std::cerr << "Unknown or incomplete argument: " << arg << "\n"
                << "Usage: bcf_debug [--scenario N] [--n N] [--n_test N] [--p N]"
                   " [--num_trees_mu N] [--num_trees_tau N] [--num_gfr N] [--num_mcmc N] [--seed N]\n";
      return 1;
    }
  }

  switch (scenario) {
    case 0:
      run_scenario_0(n, n_test, p, num_trees_mu, num_trees_tau, num_gfr, num_mcmc, seed);
      break;
    case 1:
      run_scenario_1(n, n_test, p, num_trees_mu, num_trees_tau, num_gfr, num_mcmc, seed);
      break;
    default:
      std::cerr << "Unknown scenario " << scenario
                << ". Available scenarios: 0 (BCF: constant mu + univariate tau)\n";
      return 1;
  }
  return 0;
}
