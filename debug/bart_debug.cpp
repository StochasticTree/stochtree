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

#include <stochtree/bart.h>
#include <stochtree/bart_sampler.h>

#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "stochtree/meta.h"

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

// DGP: y = sin(2*pi*x1) + 0.5*x2 - 1.5*x3 + N(0,1)
static RegressionDataset generate_regression_data(int n, int p, std::mt19937& rng) {
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

// DGP:
//   Z = sin(2*pi*x1) + 0.5*x2 - 1.5*x3 + N(0,1)
//   y = 1{Z > 0}
static ProbitDataset generate_probit_data(int n, int p, std::mt19937& rng) {
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

// ---- Reporter --------------------------------------------------------
//
// Reads directly from BARTSamples (already un-standardized by BARTSamplerFit).
// test_ref is the comparison target on the original outcome scale.

static void report_bart(const StochTree::BARTSamples& samples,
                        const std::vector<double>& test_ref,
                        const char* scenario_name) {
  const int num_samples = samples.num_samples;
  const int n_test = samples.num_test;
  double rmse_sum = 0.0;
  for (int i = 0; i < n_test; i++) {
    double y_hat = 0.0;
    for (int j = 0; j < num_samples; j++)
      y_hat += samples.mean_forest_predictions_test[static_cast<std::size_t>(j * n_test + i)] / num_samples;
    double err = (y_hat * samples.y_std + samples.y_bar) - test_ref[i];
    rmse_sum += err * err;
  }
  std::cout << "\n"
            << scenario_name << ":\n"
            << "  RMSE (test):   " << std::sqrt(rmse_sum / n_test) << "\n";
  if (!samples.global_error_variance_samples.empty()) {
    std::cout << "  sigma (last):  " << std::sqrt(samples.global_error_variance_samples.back()) * samples.y_std << "\n"
              << "  sigma (truth): 1.0\n";
  }
}

// ---- Scenario 0: homoskedastic constant-leaf BART --------------------

static void run_scenario_0(int n, int n_test, int p, int num_trees, int num_gfr, int num_mcmc, int seed) {
  std::mt19937 rng(seed < 0 ? std::random_device{}() : static_cast<unsigned>(seed));
  RegressionDataset train = generate_regression_data(n, p, rng);
  RegressionDataset test = generate_regression_data(n_test, p, rng);

  StochTree::BARTData data;
  data.X_train = train.X.data();
  data.y_train = train.y.data();
  data.n_train = n;
  data.p = p;
  data.X_test = test.X.data();
  data.n_test = n_test;

  StochTree::BARTConfig config;
  config.num_trees_mean = num_trees;
  config.random_seed = seed;
  config.link_function = StochTree::LinkFunction::Identity;
  config.standardize_outcome = true;
  config.sample_sigma2_global = true;
  config.var_weights_mean = std::vector<double>(p, 1.0 / p);
  config.feature_types = std::vector<StochTree::FeatureType>(p, StochTree::FeatureType::kNumeric);
  config.sweep_update_indices_mean = std::vector<int>(num_trees, 0);
  std::iota(config.sweep_update_indices_mean.begin(), config.sweep_update_indices_mean.end(), 0);

  StochTree::BARTSamples samples;
  StochTree::BARTSampler sampler(samples, config, data);
  sampler.run_gfr(samples, num_gfr, true);
  sampler.run_mcmc(samples, 0, 1, num_mcmc);
  sampler.postprocess_samples(samples);
  report_bart(samples, test.y, "Scenario 0 (Homoskedastic BART)");
}

// ---- Scenario 1: constant-leaf probit BART ---------------------------

static void run_scenario_1(int n, int n_test, int p, int num_trees, int num_gfr, int num_mcmc, int seed) {
  std::mt19937 rng(seed < 0 ? std::random_device{}() : static_cast<unsigned>(seed));
  ProbitDataset train = generate_probit_data(n, p, rng);
  ProbitDataset test = generate_probit_data(n_test, p, rng);

  StochTree::BARTData data;
  data.X_train = train.X.data();
  data.y_train = train.y.data();
  data.n_train = n;
  data.p = p;
  data.X_test = test.X.data();
  data.n_test = n_test;

  StochTree::BARTConfig config;
  config.num_trees_mean = num_trees;
  config.random_seed = seed;
  config.mean_leaf_model_type = StochTree::MeanLeafModelType::GaussianConstant;
  config.link_function = StochTree::LinkFunction::Probit;
  config.sample_sigma2_global = false;
  config.var_weights_mean = std::vector<double>(p, 1.0 / p);
  config.feature_types = std::vector<StochTree::FeatureType>(p, StochTree::FeatureType::kNumeric);
  config.sweep_update_indices_mean = std::vector<int>(num_trees, 0);
  std::iota(config.sweep_update_indices_mean.begin(), config.sweep_update_indices_mean.end(), 0);

  StochTree::BARTSamples samples;
  StochTree::BARTSampler sampler(samples, config, data);
  sampler.run_gfr(samples, num_gfr, true);
  sampler.run_mcmc(samples, 0, 1, num_mcmc);
  // Predictions are on latent scale (= raw + y_bar); compare to true latent Z.
  report_bart(samples, test.Z, "Scenario 1 (Probit BART)");
}

// ---- Main -----------------------------------------------------------

int main(int argc, char** argv) {
  int scenario = 0;
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
                << ". Available: 0 (Homoskedastic BART), 1 (Probit BART)\n";
      return 1;
  }
  return 0;
}
