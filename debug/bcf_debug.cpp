/*! Copyright (c) 2024 stochtree authors*/
#include <stochtree/container.h>
#include <stochtree/cpp_api.h>
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

double calibrate_lambda(std::vector<double>& covariates, std::vector<double>& residual, double nu, double q, int num_rows, int x_cols) {
  // Linear model of residual ~ covariates
  double n = static_cast<double>(residual.size());
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> X(covariates.data(), num_rows, x_cols);
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> y(residual.data(), num_rows);
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
  x1 = covariates[n*0 + i];
  x2 = covariates[n*1 + i];
  x3 = covariates[n*2 + i];
  x4 = covariates[n*3 + i];
  x5 = covariates[n*4 + i];
  return 1.0 + g(x1,x2,x3,x4,x5) + x1*x3;
}

double mu2(std::vector<double>& covariates, int n, int x_cols, int i) {
  CHECK_GE(x_cols, 5);
  CHECK_GT(n, i);
  double x1, x2, x3, x4, x5;
  x1 = covariates[n*0 + i];
  x2 = covariates[n*1 + i];
  x3 = covariates[n*2 + i];
  x4 = covariates[n*3 + i];
  x5 = covariates[n*4 + i];
  return 1.0 + g(x1,x2,x3,x4,x5) + 6.0*std::abs(x3-1);
}

double tau1(std::vector<double>& covariates, int n, int x_cols, int i) {
  return 3;
}

double tau2(std::vector<double>& covariates, int n, int x_cols, int i) {
  CHECK_GE(x_cols, 5);
  CHECK_GT(n, i);
  double x1, x2, x3, x4, x5;
  x1 = covariates[n*0 + i];
  x2 = covariates[n*1 + i];
  x3 = covariates[n*2 + i];
  x4 = covariates[n*3 + i];
  x5 = covariates[n*4 + i];
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
      // Store in column-major format
      covariates[j*n + i] = x_val;
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

void OutcomeOffsetScale(std::vector<double>& residual, double& outcome_offset, double& outcome_scale) {
  data_size_t n = residual.size();
  double outcome_val = 0.0;
  double outcome_sum = 0.0;
  double outcome_sum_squares = 0.0;
  double var_y = 0.0;
  for (data_size_t i = 0; i < n; i++){
    outcome_val = residual.at(i);
    outcome_sum += outcome_val;
    outcome_sum_squares += std::pow(outcome_val, 2.0);
  }
  var_y = outcome_sum_squares / static_cast<double>(n) - std::pow(outcome_sum / static_cast<double>(n), 2.0);
  outcome_scale = std::sqrt(var_y);
  outcome_offset = outcome_sum / static_cast<double>(n);
  double previous_residual;
  for (data_size_t i = 0; i < n; i++){
    previous_residual = residual.at(i);
    residual.at(i) = (previous_residual - outcome_offset) / outcome_scale;
  }
}

// void sampleGFR(ForestTracker& tracker, TreePrior& tree_prior, ForestContainer& forest_samples, ForestDataset& dataset, 
//                ColumnVector& residual, std::mt19937& rng, std::vector<FeatureType>& feature_types, std::vector<double>& var_weights_vector, 
//                ForestLeafModel leaf_model_type, Eigen::MatrixXd& leaf_scale_matrix, double global_variance, double leaf_scale, int cutpoint_grid_size) {
//   if (leaf_model_type == ForestLeafModel::kConstant) {
//     GaussianConstantLeafModel leaf_model = GaussianConstantLeafModel(leaf_scale);
//     GFRForestSampler<GaussianConstantLeafModel> sampler = GFRForestSampler<GaussianConstantLeafModel>(cutpoint_grid_size);
//     sampler.SampleOneIter(tracker, forest_samples, leaf_model, dataset, residual, tree_prior, rng, var_weights_vector, global_variance, feature_types);
//   } else if (leaf_model_type == ForestLeafModel::kUnivariateRegression) {
//     GaussianUnivariateRegressionLeafModel leaf_model = GaussianUnivariateRegressionLeafModel(leaf_scale);
//     GFRForestSampler<GaussianUnivariateRegressionLeafModel> sampler = GFRForestSampler<GaussianUnivariateRegressionLeafModel>(cutpoint_grid_size);
//     sampler.SampleOneIter(tracker, forest_samples, leaf_model, dataset, residual, tree_prior, rng, var_weights_vector, global_variance, feature_types);
//   } else if (leaf_model_type == ForestLeafModel::kMultivariateRegression) {
//     GaussianMultivariateRegressionLeafModel leaf_model = GaussianMultivariateRegressionLeafModel(leaf_scale_matrix);
//     GFRForestSampler<GaussianMultivariateRegressionLeafModel> sampler = GFRForestSampler<GaussianMultivariateRegressionLeafModel>(cutpoint_grid_size);
//     sampler.SampleOneIter(tracker, forest_samples, leaf_model, dataset, residual, tree_prior, rng, var_weights_vector, global_variance, feature_types);
//   }
// }

// void sampleMCMC(ForestTracker& tracker, TreePrior& tree_prior, ForestContainer& forest_samples, ForestDataset& dataset, 
//                 ColumnVector& residual, std::mt19937& rng, std::vector<FeatureType>& feature_types, std::vector<double>& var_weights_vector, 
//                 ForestLeafModel leaf_model_type, Eigen::MatrixXd& leaf_scale_matrix, double global_variance, double leaf_scale, int cutpoint_grid_size) {
//   if (leaf_model_type == ForestLeafModel::kConstant) {
//     GaussianConstantLeafModel leaf_model = GaussianConstantLeafModel(leaf_scale);
//     MCMCForestSampler<GaussianConstantLeafModel> sampler = MCMCForestSampler<GaussianConstantLeafModel>();
//     sampler.SampleOneIter(tracker, forest_samples, leaf_model, dataset, residual, tree_prior, rng, var_weights_vector, global_variance);
//   } else if (leaf_model_type == ForestLeafModel::kUnivariateRegression) {
//     GaussianUnivariateRegressionLeafModel leaf_model = GaussianUnivariateRegressionLeafModel(leaf_scale);
//     MCMCForestSampler<GaussianUnivariateRegressionLeafModel> sampler = MCMCForestSampler<GaussianUnivariateRegressionLeafModel>();
//     sampler.SampleOneIter(tracker, forest_samples, leaf_model, dataset, residual, tree_prior, rng, var_weights_vector, global_variance);
//   } else if (leaf_model_type == ForestLeafModel::kMultivariateRegression) {
//     GaussianMultivariateRegressionLeafModel leaf_model = GaussianMultivariateRegressionLeafModel(leaf_scale_matrix);
//     MCMCForestSampler<GaussianMultivariateRegressionLeafModel> sampler = MCMCForestSampler<GaussianMultivariateRegressionLeafModel>();
//     sampler.SampleOneIter(tracker, forest_samples, leaf_model, dataset, residual, tree_prior, rng, var_weights_vector, global_variance);
//   }
// }

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
      covariates_pi[n*j + i] = covariates_raw[n*j + i];
    }
    covariates_pi[n*x_cols + i] = propensity_raw[i];
  }

  // Define internal datasets
  bool row_major = false;

  // Center and scale the data
  double outcome_offset;
  double outcome_scale;
  OutcomeOffsetScale(outcome_raw, outcome_offset, outcome_scale);

  // Initialize ensembles for prognostic and treatment forests
  int num_trees_mu = 200;
  int num_trees_tau = 50;
  ForestContainer forest_samples_mu = ForestContainer(num_trees_mu, 1, true);
  ForestContainer forest_samples_tau = ForestContainer(num_trees_tau, 1, false);
  forest_samples_mu.InitializeRoot(0.);
  forest_samples_tau.InitializeRoot(0.);

  // Initialize leaf models for mu and tau forests
  double leaf_prior_scale_mu = (outcome_scale*outcome_scale)/num_trees_mu;
  double leaf_prior_scale_tau = (outcome_scale*outcome_scale)/(2*num_trees_tau);

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
  int cutpoint_grid_size = 100;
  double a_leaf_mu = 3.;
  double b_leaf_mu = leaf_prior_scale_mu;
  double a_leaf_tau = 3.;
  double b_leaf_tau = leaf_prior_scale_tau;
  double nu = 3.;
  double lamb = calibrate_lambda(covariates_raw, outcome_raw, nu, 0.9, n, x_cols);
  double b1 = 0.5;
  double b0 = -0.5;
  // ForestLeafModel leaf_model_type_mu = ForestLeafModel::kConstant;
  // ForestLeafModel leaf_model_type_tau = ForestLeafModel::kUnivariateRegression;
  int num_gfr_samples = 10;
  int num_mcmc_samples = 10;
  int num_samples = num_gfr_samples + num_mcmc_samples;

  // // Set leaf model parameters
  // double leaf_scale_mu;
  // double leaf_scale_tau = leaf_prior_scale_tau;
  // Eigen::MatrixXd leaf_scale_matrix_mu;
  // Eigen::MatrixXd leaf_scale_matrix_tau;

  // Set global variance
  double sigma2 = 1.0;

  // Initialize a random number generator
  std::random_device rd;
  std::mt19937 rng = std::mt19937(rd());

  // Initialize storage for samples of variance
  std::vector<double> global_variance_samples(num_samples);
  std::vector<double> leaf_variance_samples_mu(num_samples);

  // Initialize the BCF sampler
  BCFModel bcf = BCFModel<GaussianUnivariateRegressionLeafModel>();
  bcf.LoadTrain(outcome_raw.data(), n, covariates_pi.data(), x_cols+1, 
                covariates_raw.data(), x_cols, treatment_raw.data(), 1, true);
  bcf.ResetGlobalVarSamples(global_variance_samples.data(), num_samples);
  bcf.ResetPrognosticLeafVarSamples(leaf_variance_samples_mu.data(), num_samples);

  // Run the BCF sampler
  bcf.SampleBCF(&forest_samples_mu, &forest_samples_tau, &rng, cutpoint_grid_size, 
                leaf_prior_scale_mu, leaf_prior_scale_tau, alpha_mu, alpha_tau, beta_mu, beta_tau, 
                min_samples_leaf_mu, min_samples_leaf_tau, nu, lamb, a_leaf_mu, a_leaf_tau, b_leaf_mu, b_leaf_tau, 
                sigma2, num_trees_mu, num_trees_tau, b1, b0, feature_types_mu, feature_types_tau, 
                num_gfr_samples, 0, num_mcmc_samples, 0.0, 0.0);
}

} // namespace StochTree

int main() {
  StochTree::RunAPI();
}
