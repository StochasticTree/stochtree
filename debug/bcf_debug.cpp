/*
 * BCF debug program. The first CLI argument selects the scenario (default: 0).
 *
 * Usage: bcf_debug [--scenario N] [--n N] [--n_test N] [--p N]
 *                  [--num_trees_mu N] [--num_trees_tau N]
 *                  [--num_gfr N] [--num_mcmc N] [--seed N]
 *
 *   0  Two-forest BCF: constant-leaf mu, univariate-leaf tau (Z as basis)
 *      DGP: mu(x)  = 2*sin(pi*x1) + 0.5*x2
 *           tau(x) = 1 + x3
 *           z ~ Bernoulli(0.5)
 *           y = mu(x) + tau(x)*z + N(0, 0.5^2)
 *
 *   1  Two-forest BCF with probit link: constant-leaf mu, univariate-leaf tau
 *      DGP: mu(x)  = 2*sin(pi*x1) + 0.5*x2
 *           tau(x) = 1 + x3
 *           z ~ Bernoulli(0.5)
 *           W = mu(x) + tau(x)*z + N(0, 1)
 *           y = 1{W > 0}
 *
 * Add scenarios here as the BCFSampler API develops (propensity covariate,
 * adaptive coding, random effects, etc.).
 */

#include <stochtree/bcf.h>
#include <stochtree/bcf_sampler.h>

#include <cmath>
#include <iostream>
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

// DGP: mu(x) = 2*sin(pi*x1) + 0.5*x2, tau(x) = 1 + x3
//      z ~ Bernoulli(0.5), y = mu + tau*z + N(0, 0.25)
static SimpleBCFDataset generate_simple_bcf_data(int n, int p, std::mt19937& rng) {
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

// DGP: same mu/tau; W = mu + tau*z + N(0,1); y = 1{W > 0}
static ProbitBCFDataset generate_probit_bcf_data(int n, int p, std::mt19937& rng) {
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

// ---- Reporter --------------------------------------------------------
//
// Reads directly from BCFSamples (already un-standardized by BCFSamplerFit).
//   mu_ref  — true prognostic function (original outcome scale)
//   tau_ref — true CATE (treatment effect scale, no y_bar offset)
//   y_ref   — true outcome or latent outcome for comparison

static void report_bcf(const StochTree::BCFSamples& samples,
                       const std::vector<double>& mu_ref,
                       const std::vector<double>& tau_ref,
                       const std::vector<double>& y_ref,
                       const char* scenario_name) {
  const int num_samples = samples.num_samples;
  const int n_test = samples.num_test;
  double mu_rmse_sum = 0.0, tau_rmse_sum = 0.0, y_rmse_sum = 0.0;
  for (int i = 0; i < n_test; i++) {
    double mu_hat = 0.0, tau_hat = 0.0, y_hat = 0.0;
    for (int j = 0; j < num_samples; j++) {
      const auto k = static_cast<std::size_t>(j * n_test + i);
      mu_hat += samples.mu_forest_predictions_test[k] / num_samples;
      tau_hat += samples.tau_forest_predictions_test[k] / num_samples;
      y_hat += samples.y_hat_test[k] / num_samples;
    }
    mu_rmse_sum += (mu_hat - mu_ref[i]) * (mu_hat - mu_ref[i]);
    tau_rmse_sum += (tau_hat - tau_ref[i]) * (tau_hat - tau_ref[i]);
    y_rmse_sum += (y_hat - y_ref[i]) * (y_hat - y_ref[i]);
  }
  std::cout << "\n"
            << scenario_name << ":\n"
            << "  mu RMSE (test):  " << std::sqrt(mu_rmse_sum / n_test) << "\n"
            << "  tau RMSE (test): " << std::sqrt(tau_rmse_sum / n_test) << "\n"
            << "  y RMSE (test):   " << std::sqrt(y_rmse_sum / n_test) << "\n";
  if (!samples.global_error_variance_samples.empty()) {
    std::cout << "  sigma (last):    "
              << std::sqrt(samples.global_error_variance_samples.back()) << "\n";
  }
}

// ---- Scenario 0: constant-leaf mu + univariate-leaf tau (identity link) ---

static void run_scenario_0(int n, int n_test, int p,
                           int num_trees_mu, int num_trees_tau,
                           int num_gfr, int num_mcmc, int seed) {
  std::mt19937 rng(seed < 0 ? std::random_device{}() : static_cast<unsigned>(seed));
  SimpleBCFDataset train = generate_simple_bcf_data(n, p, rng);
  SimpleBCFDataset test = generate_simple_bcf_data(n_test, p, rng);

  StochTree::BCFData data;
  data.X_train = train.X.data();
  data.y_train = train.y.data();
  data.treatment_train = train.z.data();
  data.n_train = n;
  data.p = p;
  data.treatment_dim = 1;
  data.X_test = test.X.data();
  data.treatment_test = test.z.data();
  data.n_test = n_test;

  StochTree::BCFConfig config;
  config.num_trees_mu = num_trees_mu;
  config.num_trees_tau = num_trees_tau;
  config.random_seed = seed;
  config.link_function = StochTree::LinkFunction::Identity;
  config.standardize_outcome = true;
  config.sample_sigma2_global = true;

  StochTree::BCFSamples samples;
  StochTree::BCFSampler sampler(samples, config, data);
  sampler.run_gfr(samples, num_gfr, /*keep_gfr=*/true);
  sampler.run_mcmc(samples, /*num_burnin=*/0, /*keep_every=*/1, /*num_mcmc=*/num_mcmc);
  report_bcf(samples, test.mu_true, test.tau_true, test.y,
             "Scenario 0 (BCF: constant mu + univariate tau, identity link)");
  std::cout << "  sigma (truth):   0.5\n";
}

// ---- Scenario 1: probit BCF (constant-leaf mu + univariate-leaf tau) ----

static void run_scenario_1(int n, int n_test, int p,
                           int num_trees_mu, int num_trees_tau,
                           int num_gfr, int num_mcmc, int seed) {
  std::mt19937 rng(seed < 0 ? std::random_device{}() : static_cast<unsigned>(seed));
  ProbitBCFDataset train = generate_probit_bcf_data(n, p, rng);
  ProbitBCFDataset test = generate_probit_bcf_data(n_test, p, rng);

  StochTree::BCFData data;
  data.X_train = train.X.data();
  data.y_train = train.y.data();
  data.treatment_train = train.z.data();
  data.n_train = n;
  data.p = p;
  data.treatment_dim = 1;
  data.X_test = test.X.data();
  data.treatment_test = test.z.data();
  data.n_test = n_test;

  StochTree::BCFConfig config;
  config.num_trees_mu = num_trees_mu;
  config.num_trees_tau = num_trees_tau;
  config.random_seed = seed;
  config.link_function = StochTree::LinkFunction::Probit;
  config.standardize_outcome = true;
  config.sample_sigma2_global = false;

  StochTree::BCFSamples samples;
  StochTree::BCFSampler sampler(samples, config, data);
  sampler.run_gfr(samples, num_gfr, /*keep_gfr=*/true);
  sampler.run_mcmc(samples, /*num_burnin=*/0, /*keep_every=*/1, /*num_mcmc=*/num_mcmc);
  report_bcf(samples, test.mu_true, test.tau_true, test.y,
             "Scenario 0 (BCF: constant mu + univariate tau, probit link)");
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
         arg == "--num_trees_mu" || arg == "--num_trees_tau" || arg == "--num_gfr" ||
         arg == "--num_mcmc" || arg == "--seed") &&
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
                << ". Available: 0 (BCF: identity), 1 (BCF: probit)\n";
      return 1;
  }
  return 0;
}
