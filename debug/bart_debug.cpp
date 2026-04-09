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

struct Dataset {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X;
  Eigen::VectorXd y;
};

// DGP: y = sin(2*pi*x1) + 0.5*x2 - 1.5*x3 + N(0,1)
Dataset generate_data(int n, int p, std::mt19937& rng) {
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  std::normal_distribution<double> normal(0.0, 1.0);
  Dataset d;
  d.X.resize(n, p);
  d.y.resize(n);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < p; j++)
      d.X(i, j) = unif(rng);
  for (int i = 0; i < n; i++)
    d.y(i) = std::sin(2.0 * kPi * d.X(i, 0))
             + 0.5 * d.X(i, 1)
             - 1.5 * d.X(i, 2)
             + normal(rng);
  return d;
}

// ---- Scenario 0: homoskedastic constant-leaf BART -------------------

void run_scenario_0(int n, int p, int num_trees, int num_gfr, int num_mcmc) {
  constexpr int num_threads = 1;
  constexpr int cutpoint_grid_size = 100;
  std::mt19937 rng(42);

  Dataset data = generate_data(n, p, rng);
  double y_bar = data.y.mean();
  Eigen::VectorXd resid_vec = data.y.array() - y_bar;

  StochTree::ForestDataset dataset;
  dataset.AddCovariates(data.X.data(), n, p, /*row_major=*/true);
  StochTree::ColumnVector residual(resid_vec.data(), n);

  std::vector<StochTree::FeatureType> feature_types(p, StochTree::FeatureType::kNumeric);
  std::vector<double> var_weights(p, 1.0 / p);
  std::vector<int> sweep_indices;

  StochTree::TreePrior tree_prior(0.95, 2.0, /*min_samples_leaf=*/5);
  StochTree::ForestContainer forest_samples(num_trees, /*output_dim=*/1, /*leaf_constant=*/true, /*exponentiated=*/false);
  StochTree::TreeEnsemble active_forest(num_trees, 1, true, false);
  StochTree::ForestTracker tracker(dataset.GetCovariates(), feature_types, num_trees, n);

  double leaf_scale = 1.0 / num_trees;
  StochTree::GaussianConstantLeafModel leaf_model(leaf_scale);

  double global_variance = 1.0;
  constexpr double a_sigma = 0.0, b_sigma = 0.0;  // non-informative IG prior
  StochTree::GlobalHomoskedasticVarianceModel var_model;

  // GFR warmup — no samples stored
  std::cout << "[GFR]  " << num_gfr << " warmup iterations...\n";
  bool pre_initialized = false;
  for (int i = 0; i < num_gfr; i++) {
    StochTree::GFRSampleOneIter<
        StochTree::GaussianConstantLeafModel,
        StochTree::GaussianConstantSuffStat>(
        active_forest, tracker, forest_samples, leaf_model,
        dataset, residual, tree_prior, rng,
        var_weights, sweep_indices, global_variance, feature_types,
        cutpoint_grid_size, /*keep_forest=*/false, pre_initialized,
        /*backfitting=*/true, /*num_features_subsample=*/-1, num_threads);
    global_variance = var_model.SampleVarianceParameter(
        residual.GetData(), a_sigma, b_sigma, rng);
    pre_initialized = true;
  }

  // MCMC — store samples
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
    global_variance = var_model.SampleVarianceParameter(
        residual.GetData(), a_sigma, b_sigma, rng);
  }

  // Posterior predictions: column-major, element [j*n + i] = sample j, obs i
  std::vector<double> preds = forest_samples.Predict(dataset);
  double rmse_sum = 0.0;
  for (int i = 0; i < n; i++) {
    double mu_hat = y_bar;
    for (int j = 0; j < num_mcmc; j++)
      mu_hat += preds[static_cast<std::size_t>(j * n + i)] / num_mcmc;
    double err = mu_hat - data.y(i);
    rmse_sum += err * err;
  }

  std::cout << "\nScenario 0 (HomoskedasticBART):\n"
            << "  RMSE:                " << std::sqrt(rmse_sum / n) << "\n"
            << "  sigma (last sample): " << std::sqrt(global_variance) << "\n"
            << "  sigma (truth):       1.0\n";
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
                << ". Available scenarios: 0 (HomoskedasticBART)\n";
      return 1;
  }
  return 0;
}
