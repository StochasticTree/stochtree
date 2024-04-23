/*! Copyright (c) 2024 stochtree authors*/
#include <stochtree/container.h>
#include <stochtree/data.h>
#include <stochtree/io.h>
#include <nlohmann/json.hpp>
#include <stochtree/leaf_model.h>
#include <stochtree/log.h>
#include <stochtree/random_effects.h>
#include <stochtree/tree_sampler.h>
#include <stochtree/variance_model.h>
#include <boost/math/special_functions/gamma.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace StochTree{

enum ForestLeafModel {
    kConstant, 
    kUnivariateRegression, 
    kMultivariateRegression
};

double calibrate_lambda(ForestDataset& covariates, ColumnVector& residual, double nu, double q) {
  // Linear model of residual ~ covariates
  double n = static_cast<double>(covariates.NumObservations());
  Eigen::MatrixXd X = covariates.GetCovariates();
  Eigen::VectorXd y = residual.GetData();
  Eigen::VectorXd beta = (X.transpose() * X).inverse() * (X.transpose() * y);
  double sum_sq_resid = (y - X * beta).transpose() * (y - X * beta);
  double sigma_hat = sum_sq_resid / n;

  // Compute implied lambda
  return  (sigma_hat * boost::math::gamma_q_inv(nu, q)) / nu;
}

double std_gaussian_cdf(double x) {
  return 0.5*(1 + std::erf(x/std::sqrt(2)));
}

double g(double x1, double x2, double x3, double x4, double x5) {
  double output;
  if (std::abs(x5-0.0) < 0.001) {output = 2.0;}
  else if (std::abs(x5-1.) < 0.001) {output = -1.0;}
  else {output = -4.0;}
  return output;
}

double mu1(std::vector<double>& covariates, int n, int x_cols, int i) {
  CHECK_GE(x_cols, 5);
  CHECK_GT(n, i);
  double x1, x2, x3, x4, x5;
  x1 = covariates[i*x_cols + 0];
  x2 = covariates[i*x_cols + 1];
  x3 = covariates[i*x_cols + 2];
  x4 = covariates[i*x_cols + 3];
  x5 = covariates[i*x_cols + 4];
  return 1.0 + g(x1,x2,x3,x4,x5) + x1*x3;
}

double mu2(std::vector<double>& covariates, int n, int x_cols, int i) {
  CHECK_GE(x_cols, 5);
  CHECK_GT(n, i);
  double x1, x2, x3, x4, x5;
  x1 = covariates[i*x_cols + 0];
  x2 = covariates[i*x_cols + 1];
  x3 = covariates[i*x_cols + 2];
  x4 = covariates[i*x_cols + 3];
  x5 = covariates[i*x_cols + 4];
  return 1.0 + g(x1,x2,x3,x4,x5) + 6.0*std::abs(x3-1);
}

double tau1(std::vector<double>& covariates, int n, int x_cols, int i) {
  return 3;
}

double tau2(std::vector<double>& covariates, int n, int x_cols, int i) {
  CHECK_GE(x_cols, 5);
  CHECK_GT(n, i);
  double x1, x2, x3, x4, x5;
  x1 = covariates[i*x_cols + 0];
  x2 = covariates[i*x_cols + 1];
  x3 = covariates[i*x_cols + 2];
  x4 = covariates[i*x_cols + 3];
  x5 = covariates[i*x_cols + 4];
  return 1 + 2.0*x2*x4;
}

void GenerateRandomData(std::vector<double>& covariates, std::vector<double>& propensity, std::vector<double>& treatment, std::vector<double>& outcome, 
                        std::function<double(std::vector<double>&, int, int, int)> mu, std::function<double(std::vector<double>&, int, int, int)> tau, 
                        int n, int x_cols, double snr = 2.0, int random_seed = -1) {
  std::mt19937 gen;
  if (random_seed == 1) {
    std::random_device rd;
    gen = std::mt19937(rd());
  } else {
    gen = std::mt19937(random_seed);
  }
  std::uniform_real_distribution<double> std_uniform_dist{0.0,1.0};
  std::normal_distribution<double> std_normal_dist(0.,1.);
  std::discrete_distribution<> binary_covariate_dist({50, 50});
  std::discrete_distribution<> categorical_covariate_dist({33,33,33});
  std::vector<double> mu_x(n);
  std::vector<double> tau_x(n);
  std::vector<double> f_x_z(n);

  CHECK_GE(x_cols, 5);
  double x_val;
  for (int i = 0; i < n; i++) {
    // Covariates
    for (int j = 0; j < x_cols; j++) {
      if (j == 3) {
        x_val = categorical_covariate_dist(gen);
      } else if (j == 4) {
        x_val = binary_covariate_dist(gen);
      } else {
        x_val = std_normal_dist(gen);
      }
      covariates[i*x_cols + j] = x_val;
    }

    // Prognostic function
    mu_x[i] = mu(covariates, n, x_cols, i);

    // Treatment effect
    tau_x[i] = tau(covariates, n, x_cols, i);
  }

  // Compute mean and sd of mu_x
  double mu_sum = std::accumulate(mu_x.begin(), mu_x.end(), 0.0);
  double mu_mean = mu_sum / static_cast<double>(n);
  double mu_sum_squares = std::accumulate(mu_x.begin(), mu_x.end(), 0.0, [](double a, double b){return a + b*b;});
  double mu_stddev = std::sqrt((mu_sum_squares / static_cast<double>(n)) - mu_mean*mu_mean);

  for (int i = 0; i < n; i++) {
    // Propensity score
    propensity[i] = 0.8*std_gaussian_cdf((3*mu_x[i]/mu_stddev) - 0.5*covariates[i * x_cols + 0]) + 0.05 + std_uniform_dist(gen)/10.;

    // Treatment
    treatment[i] = (std_uniform_dist(gen) < propensity[i]) ? 1.0 : 0.0;

    // Expected outcome
    f_x_z[i] = mu_x[i] + tau_x[i] * treatment[i];
  }

  // Compute sd(E(Y | X, Z))
  double ey_sum = std::accumulate(f_x_z.begin(), f_x_z.end(), 0.0);
  double ey_mean = ey_sum / static_cast<double>(n);
  double ey_sum_squares = std::accumulate(f_x_z.begin(), f_x_z.end(), 0.0, [](double a, double b){return a + b*b;});
  double ey_stddev = std::sqrt((ey_sum_squares / static_cast<double>(n)) - ey_mean*ey_mean);

  for (int i = 0; i < n; i++) {
    // Propensity score
    outcome[i] = f_x_z[i] + (ey_stddev/snr)*std_normal_dist(gen);
  }
}

void OutcomeOffsetScale(ColumnVector& residual, double& outcome_offset, double& outcome_scale) {
  data_size_t n = residual.NumRows();
  double outcome_val = 0.0;
  double outcome_sum = 0.0;
  double outcome_sum_squares = 0.0;
  double var_y = 0.0;
  for (data_size_t i = 0; i < n; i++){
    outcome_val = residual.GetElement(i);
    outcome_sum += outcome_val;
    outcome_sum_squares += std::pow(outcome_val, 2.0);
  }
  var_y = outcome_sum_squares / static_cast<double>(n) - std::pow(outcome_sum / static_cast<double>(n), 2.0);
  outcome_scale = std::sqrt(var_y);
  outcome_offset = outcome_sum / static_cast<double>(n);
  double previous_residual;
  for (data_size_t i = 0; i < n; i++){
    previous_residual = residual.GetElement(i);
    residual.SetElement(i, (previous_residual - outcome_offset) / outcome_scale);
  }
}

void sampleGFR(ForestTracker& tracker, TreePrior& tree_prior, ForestContainer& forest_samples, ForestDataset& dataset, 
               ColumnVector& residual, std::mt19937& rng, std::vector<FeatureType>& feature_types, std::vector<double>& var_weights_vector, 
               ForestLeafModel leaf_model_type, Eigen::MatrixXd& leaf_scale_matrix, double global_variance, double leaf_scale, int cutpoint_grid_size) {
  if (leaf_model_type == ForestLeafModel::kConstant) {
    GaussianConstantLeafModel leaf_model = GaussianConstantLeafModel(leaf_scale);
    GFRForestSampler<GaussianConstantLeafModel> sampler = GFRForestSampler<GaussianConstantLeafModel>(cutpoint_grid_size);
    sampler.SampleOneIter(tracker, forest_samples, leaf_model, dataset, residual, tree_prior, rng, var_weights_vector, global_variance, feature_types);
  } else if (leaf_model_type == ForestLeafModel::kUnivariateRegression) {
    GaussianUnivariateRegressionLeafModel leaf_model = GaussianUnivariateRegressionLeafModel(leaf_scale);
    GFRForestSampler<GaussianUnivariateRegressionLeafModel> sampler = GFRForestSampler<GaussianUnivariateRegressionLeafModel>(cutpoint_grid_size);
    sampler.SampleOneIter(tracker, forest_samples, leaf_model, dataset, residual, tree_prior, rng, var_weights_vector, global_variance, feature_types);
  } else if (leaf_model_type == ForestLeafModel::kMultivariateRegression) {
    GaussianMultivariateRegressionLeafModel leaf_model = GaussianMultivariateRegressionLeafModel(leaf_scale_matrix);
    GFRForestSampler<GaussianMultivariateRegressionLeafModel> sampler = GFRForestSampler<GaussianMultivariateRegressionLeafModel>(cutpoint_grid_size);
    sampler.SampleOneIter(tracker, forest_samples, leaf_model, dataset, residual, tree_prior, rng, var_weights_vector, global_variance, feature_types);
  }
}

void sampleMCMC(ForestTracker& tracker, TreePrior& tree_prior, ForestContainer& forest_samples, ForestDataset& dataset, 
                ColumnVector& residual, std::mt19937& rng, std::vector<FeatureType>& feature_types, std::vector<double>& var_weights_vector, 
                ForestLeafModel leaf_model_type, Eigen::MatrixXd& leaf_scale_matrix, double global_variance, double leaf_scale, int cutpoint_grid_size) {
  if (leaf_model_type == ForestLeafModel::kConstant) {
    GaussianConstantLeafModel leaf_model = GaussianConstantLeafModel(leaf_scale);
    MCMCForestSampler<GaussianConstantLeafModel> sampler = MCMCForestSampler<GaussianConstantLeafModel>();
    sampler.SampleOneIter(tracker, forest_samples, leaf_model, dataset, residual, tree_prior, rng, var_weights_vector, global_variance);
  } else if (leaf_model_type == ForestLeafModel::kUnivariateRegression) {
    GaussianUnivariateRegressionLeafModel leaf_model = GaussianUnivariateRegressionLeafModel(leaf_scale);
    MCMCForestSampler<GaussianUnivariateRegressionLeafModel> sampler = MCMCForestSampler<GaussianUnivariateRegressionLeafModel>();
    sampler.SampleOneIter(tracker, forest_samples, leaf_model, dataset, residual, tree_prior, rng, var_weights_vector, global_variance);
  } else if (leaf_model_type == ForestLeafModel::kMultivariateRegression) {
    GaussianMultivariateRegressionLeafModel leaf_model = GaussianMultivariateRegressionLeafModel(leaf_scale_matrix);
    MCMCForestSampler<GaussianMultivariateRegressionLeafModel> sampler = MCMCForestSampler<GaussianMultivariateRegressionLeafModel>();
    sampler.SampleOneIter(tracker, forest_samples, leaf_model, dataset, residual, tree_prior, rng, var_weights_vector, global_variance);
  }
}

void RunAPI() {
  // Data dimensions
  int n = 1000;
  int x_cols = 5;
  
  // Declare covariates, basis and outcome
  std::vector<double> covariates_raw(n*x_cols);
  std::vector<double> propensity_raw(n);
  std::vector<double> treatment_raw(n);
  std::vector<double> outcome_raw(n);
  
  // Load the data
  GenerateRandomData(covariates_raw, propensity_raw, treatment_raw, outcome_raw, mu1, tau2, n, x_cols, 2.0, 1);
  
  // Add pi_x as a column to a covariate set used in the prognostic forest
  std::vector<double> covariates_pi(n*(x_cols+1));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < x_cols; j++) {
      covariates_pi[i*(x_cols+1) + j] = covariates_raw[i*x_cols + j];
    }
    covariates_pi[i*(x_cols+1) + x_cols] = propensity_raw[i];
  }

  // Define internal datasets
  bool row_major = true;

  // Construct datasets for training, include pi(x) as a covariate in the prognostic forest
  ForestDataset tau_dataset = ForestDataset();
  tau_dataset.AddCovariates(covariates_raw.data(), n, x_cols, row_major);
  tau_dataset.AddBasis(treatment_raw.data(), n, 1, row_major);
  ForestDataset mu_dataset = ForestDataset();
  mu_dataset.AddCovariates(covariates_pi.data(), n, x_cols+1, row_major);
  ColumnVector residual = ColumnVector(outcome_raw.data(), n);
  
  // Center and scale the data
  double outcome_offset;
  double outcome_scale;
  OutcomeOffsetScale(residual, outcome_offset, outcome_scale);

  // Initialize ensembles for prognostic and treatment forests
  int num_trees_mu = 200;
  int num_trees_tau = 50;
  ForestContainer forest_samples_mu = ForestContainer(num_trees_mu, 1, true);
  ForestContainer forest_samples_tau = ForestContainer(num_trees_tau, 1, false);

  // Initialize leaf models for mu and tau forests
  double leaf_prior_scale_mu = (outcome_scale*outcome_scale)/num_trees_mu;
  double leaf_prior_scale_tau = (outcome_scale*outcome_scale)/(2*num_trees_tau);
  GaussianConstantLeafModel leaf_model_mu = GaussianConstantLeafModel(leaf_prior_scale_mu);
  GaussianUnivariateRegressionLeafModel leaf_model_tau = GaussianUnivariateRegressionLeafModel(leaf_prior_scale_tau);

  // Initialize forest sampling machinery
  std::vector<FeatureType> feature_types_mu(x_cols + 1, FeatureType::kNumeric);
  feature_types_mu[3] = FeatureType::kOrderedCategorical;
  feature_types_mu[4] = FeatureType::kOrderedCategorical;
  std::vector<FeatureType> feature_types_tau(x_cols, FeatureType::kNumeric);
  feature_types_tau[3] = FeatureType::kOrderedCategorical;
  feature_types_tau[4] = FeatureType::kOrderedCategorical;
  double alpha_mu = 0.95;
  double alpha_tau = 0.25;
  double beta_mu = 2.0;
  double beta_tau = 3.0;
  int min_samples_leaf_mu = 5;
  int min_samples_leaf_tau = 5;
  int cutpoint_grid_size_mu = 100;
  int cutpoint_grid_size_tau = 100;
  double a_leaf_mu = 3.;
  double b_leaf_mu = leaf_prior_scale_mu;
  double a_leaf_tau = 3.;
  double b_leaf_tau = leaf_prior_scale_tau;
  double nu = 3.;
  double lamb = calibrate_lambda(tau_dataset, residual, nu, 0.9);
  ForestLeafModel leaf_model_type_mu = ForestLeafModel::kConstant;
  ForestLeafModel leaf_model_type_tau = ForestLeafModel::kUnivariateRegression;

  // Set leaf model parameters
  double leaf_scale_mu;
  double leaf_scale_tau = leaf_prior_scale_tau;
  Eigen::MatrixXd leaf_scale_matrix_mu;
  Eigen::MatrixXd leaf_scale_matrix_tau;

  // Set global variance
  double global_variance_init = 1.0;
  double global_variance;

  // Set variable weights
  double const_var_wt_mu = static_cast<double>(1/(x_cols+1));
  std::vector<double> variable_weights_mu(x_cols+1, const_var_wt_mu);
  double const_var_wt_tau = static_cast<double>(1/x_cols);
  std::vector<double> variable_weights_tau(x_cols, const_var_wt_tau);

  // Initialize tracker and tree prior
  ForestTracker mu_tracker = ForestTracker(mu_dataset.GetCovariates(), feature_types_mu, num_trees_mu, n);
  ForestTracker tau_tracker = ForestTracker(tau_dataset.GetCovariates(), feature_types_tau, num_trees_tau, n);
  TreePrior tree_prior_mu = TreePrior(alpha_mu, beta_mu, min_samples_leaf_mu);
  TreePrior tree_prior_tau = TreePrior(alpha_tau, beta_tau, min_samples_leaf_tau);

  // Initialize a random number generator
  std::random_device rd;
  std::mt19937 rng = std::mt19937(rd());
  
  // Initialize variance models
  GlobalHomoskedasticVarianceModel global_var_model = GlobalHomoskedasticVarianceModel();
  LeafNodeHomoskedasticVarianceModel leaf_var_model_mu = LeafNodeHomoskedasticVarianceModel();

  // Initialize storage for samples of variance
  std::vector<double> global_variance_samples{};
  std::vector<double> leaf_variance_samples_mu{};

  // Run the GFR sampler
  int num_gfr_samples = 10;
  for (int i = 0; i < num_gfr_samples; i++) {
    if (i == 0) {
      global_variance = global_variance_init;
      leaf_scale_mu = leaf_prior_scale_mu;
    } else {
      global_variance = global_variance_samples[i-1];
      leaf_scale_mu = leaf_variance_samples_mu[i-1];
    }

    // Sample mu ensemble
    sampleGFR(mu_tracker, tree_prior_mu, forest_samples_mu, mu_dataset, residual, rng, feature_types_mu, variable_weights_mu, 
              leaf_model_type_mu, leaf_scale_matrix_mu, global_variance, leaf_scale_mu, cutpoint_grid_size_mu);

    // Sample leaf node variance
    leaf_variance_samples_mu.push_back(leaf_var_model_mu.SampleVarianceParameter(forest_samples_mu.GetEnsemble(i), a_leaf_mu, b_leaf_mu, rng));

    // Sample global variance
    global_variance_samples.push_back(global_var_model.SampleVarianceParameter(residual.GetData(), nu, nu*lamb, rng));

    // Sample tau ensemble
    sampleGFR(tau_tracker, tree_prior_tau, forest_samples_tau, tau_dataset, residual, rng, feature_types_tau, variable_weights_tau, 
              leaf_model_type_tau, leaf_scale_matrix_tau, global_variance, leaf_scale_tau, cutpoint_grid_size_tau);

    // Sample global variance
    global_variance_samples.push_back(global_var_model.SampleVarianceParameter(residual.GetData(), nu, nu*lamb, rng));
  }

  // Run the MCMC sampler
  int num_mcmc_samples = 10000;
  for (int i = num_gfr_samples; i < num_gfr_samples + num_mcmc_samples; i++) {
    if (i == 0) {
      global_variance = global_variance_init;
      leaf_scale_mu = leaf_prior_scale_mu;
    } else {
      global_variance = global_variance_samples[i-1];
      leaf_scale_mu = leaf_variance_samples_mu[i-1];
    }

    // Sample mu ensemble
    sampleMCMC(mu_tracker, tree_prior_mu, forest_samples_mu, mu_dataset, residual, rng, feature_types_mu, variable_weights_mu, 
               leaf_model_type_mu, leaf_scale_matrix_mu, global_variance, leaf_scale_mu, cutpoint_grid_size_mu);

    // Sample leaf node variance
    leaf_variance_samples_mu.push_back(leaf_var_model_mu.SampleVarianceParameter(forest_samples_mu.GetEnsemble(i), a_leaf_mu, b_leaf_mu, rng));

    // Sample global variance
    global_variance_samples.push_back(global_var_model.SampleVarianceParameter(residual.GetData(), nu, nu*lamb, rng));

    // Sample tau ensemble
    sampleMCMC(tau_tracker, tree_prior_tau, forest_samples_tau, tau_dataset, residual, rng, feature_types_tau, variable_weights_tau, 
               leaf_model_type_tau, leaf_scale_matrix_tau, global_variance, leaf_scale_tau, cutpoint_grid_size_tau);

    // Sample global variance
    global_variance_samples.push_back(global_var_model.SampleVarianceParameter(residual.GetData(), nu, nu*lamb, rng));
  }
}

} // namespace StochTree

int main() {
  StochTree::RunAPI();
}
