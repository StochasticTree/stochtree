/*! Copyright (c) 2024 stochtree authors*/
#include <stochtree/data.h>
#include <stochtree/container.h>
#include <stochtree/leaf_model.h>
#include <stochtree/random_effects.h>
#include <stochtree/tree_sampler.h>

#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace StochTree{

void GenerateRandomData(std::vector<double>& covariates, std::vector<double>& basis, std::vector<double>& outcome, std::vector<double>& rfx_basis, std::vector<int32_t>& rfx_groups, int n, int x_cols, int omega_cols, int y_cols, int rfx_basis_cols) {
  std::mt19937 gen(101);
  std::uniform_real_distribution<double> uniform_dist{0.0,1.0};
  std::normal_distribution<double> normal_dist(0.,1.);
  std::vector<double> betas{-10, -5, 5, 10};
  int num_partitions = betas.size();
  double beta = 5;
  double f_x_omega;
  double rfx;

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < x_cols; j++) {
      covariates[i*x_cols + j] = uniform_dist(gen);
    }
    
    for (int j = 0; j < omega_cols; j++) {
      basis[i*omega_cols + j] = uniform_dist(gen);
    }
    
    for (int j = 0; j < rfx_basis_cols; j++) {
      rfx_basis[i*rfx_basis_cols + j] = 1;
    }
    
    if (i % 2 == 0) {
      rfx_groups[i] = 1;
    } else {
      rfx_groups[i] = 2;
    }
    
    for (int j = 0; j < y_cols; j++) {
      if ((covariates[i * x_cols + 0] >= 0.0) && covariates[i * x_cols + 0] < 0.25) {
        f_x_omega = betas[0] * basis[i * omega_cols + 0];
      } else if ((covariates[i * x_cols + 0] >= 0.25) && covariates[i * x_cols + 0] < 0.5) {
        f_x_omega = betas[1] * basis[i * omega_cols + 0];
      } else if ((covariates[i * x_cols + 0] >= 0.5) && covariates[i * x_cols + 0] < 0.75) {
        f_x_omega = betas[2] * basis[i * omega_cols + 0];
      } else {
        f_x_omega = betas[3] * basis[i * omega_cols + 0];
      }
      if (rfx_groups[i] == 1) {
        rfx = 5.;
      } else {
        rfx = -5.;
      }
      outcome[i * y_cols + j] = f_x_omega + rfx + 0.1*normal_dist(gen);
    }
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

void RunAPI() {
  // Data dimensions
  int n = 1000;
  int x_rows = n;
  int x_cols = 2;
  int omega_rows = n;
  int omega_cols = 1;
  int y_rows = n;
  int y_cols = 1;
  int rfx_basis_rows = n;
  int rfx_basis_cols = 1;
  int num_rfx_groups = 2;
  
  // Declare covariates, basis and outcome
  std::vector<double> covariates_raw(x_rows*x_cols);
  std::vector<double> basis_raw(omega_rows*omega_cols);
  std::vector<double> rfx_basis_raw(rfx_basis_rows*rfx_basis_cols);
  std::vector<double> outcome_raw(y_rows*y_cols);
  std::vector<int32_t> rfx_groups(n);
  
  // Load the data
  GenerateRandomData(covariates_raw, basis_raw, outcome_raw, rfx_basis_raw, rfx_groups, n, x_cols, omega_cols, y_cols, rfx_basis_cols);
  
  // Define internal datasets
  bool row_major = true;

  // Construct datasets for training
  ForestDataset dataset = ForestDataset();
  dataset.AddCovariates(covariates_raw.data(), n, x_cols, row_major);
  dataset.AddBasis(basis_raw.data(), n, omega_cols, row_major);
  ColumnVector residual = ColumnVector(outcome_raw.data(), n);
  
  // Center and scale the data
  double outcome_offset;
  double outcome_scale;
  OutcomeOffsetScale(residual, outcome_offset, outcome_scale);

  // Construct a random effects dataset
  RandomEffectsDataset rfx_dataset = RandomEffectsDataset();
  rfx_dataset.AddBasis(rfx_basis_raw.data(), n, rfx_basis_cols, row_major);
  rfx_dataset.AddGroupLabels(rfx_groups);
  
  // Initialize an ensemble
  int num_trees = 100;
  int output_dimension = 1;
  bool is_leaf_constant = false;
  ForestContainer forest_samples = ForestContainer(num_trees, output_dimension, is_leaf_constant);

  // Initialize a leaf model
  double leaf_prior_mean = 0.;
  double leaf_prior_scale = 1.;
  GaussianUnivariateRegressionLeafModel leaf_model = GaussianUnivariateRegressionLeafModel(leaf_prior_scale);

  // // Check consistency
  // CHECK(ForestModelCompatible(forest_samples, leaf_model));

  // // Initialize a global variance model
  // double a_global_variance = 1.;
  // double b_global_variance = 1.;
  // HomoskedasticVarianceModel global_variance_model = HomoskedasticVarianceModel(a_global_variance, b_global_variance);
  
  // Check consistency

  // Initialize forest sampling machinery
  std::vector<FeatureType> feature_types(x_cols, FeatureType::kNumeric);
  double alpha = 0.99;
  double beta = 1.25;
  int min_samples_leaf = 10;
  int cutpoint_grid_size = 500;
  double a_rfx = 1.;
  double b_rfx = 1.;
  TreePrior tree_prior = TreePrior(alpha, beta, min_samples_leaf);
  GFRForestSampler gfr_sampler = GFRForestSampler<GaussianUnivariateRegressionLeafModel>(cutpoint_grid_size);
  MCMCForestSampler mcmc_sampler = MCMCForestSampler<GaussianUnivariateRegressionLeafModel>();
  ForestTracker tracker = ForestTracker(dataset.GetCovariates(), feature_types, num_trees, n);

  // Initialize random effect sampling machinery
  RandomEffectsTracker rfx_tracker = RandomEffectsTracker(rfx_groups);
  RandomEffectsContainer rfx_samples = RandomEffectsContainer();
  MultivariateRegressionRandomEffectsModel rfx_model = MultivariateRegressionRandomEffectsModel();
  double rfx_scale = 0.5;
  Eigen::MatrixXd working_parameter_prior_covariance = Eigen::MatrixXd::Identity(rfx_basis_cols, rfx_basis_cols);
  Eigen::MatrixXd group_parameter_prior_covariance = Eigen::MatrixXd::Identity(rfx_basis_cols, rfx_basis_cols) * rfx_scale;
  Eigen::VectorXd working_parameter_init = Eigen::VectorXd::Ones(rfx_basis_cols);
  Eigen::MatrixXd group_parameter_init = Eigen::MatrixXd::Ones(rfx_basis_cols, rfx_tracker.NumCategories());
  double group_parameter_variance_prior_shape = 1.;
  double group_parameter_variance_prior_scale = 1.;
  
  // Run a single iteration of the GFR algorithm
  int random_seed = 1;
  std::mt19937 gen = std::mt19937(random_seed);
  double global_variance = 0.1;
  
  // Run GFR ensemble sampler and other parameter samplers in a loop
  int num_gfr_samples = 10;
  for (int i = 0; i < num_gfr_samples; i++) {
    // Tree ensemble sampler
    gfr_sampler.SampleOneIter(tracker, forest_samples, leaf_model, dataset, residual, tree_prior, gen, global_variance, feature_types);

    // Random effects
    rfx_samples.AddSamples(rfx_dataset, rfx_tracker, working_parameter_init, group_parameter_init, 
                           working_parameter_prior_covariance, group_parameter_prior_covariance,
                           group_parameter_variance_prior_shape, group_parameter_variance_prior_scale, 1);
    rfx_model.SampleRandomEffects(rfx_samples.GetRandomEffectsTerm(i), rfx_dataset, residual, rfx_tracker, global_variance, gen);
  }

  // Run MCMC ensemble sampler and other parameter samplers in a loop
  int num_mcmc_samples = 10;
  for (int i = 0; i < num_mcmc_samples; i++) {
    // Tree ensemble sampler
    mcmc_sampler.SampleOneIter(tracker, forest_samples, leaf_model, dataset, residual, tree_prior, gen, global_variance);

    // Random effects
    rfx_samples.AddSamples(rfx_dataset, rfx_tracker, working_parameter_init, group_parameter_init,
                           working_parameter_prior_covariance, group_parameter_prior_covariance,
                           group_parameter_variance_prior_shape, group_parameter_variance_prior_scale, 1);
    rfx_model.SampleRandomEffects(rfx_samples.GetRandomEffectsTerm(i + num_gfr_samples), rfx_dataset, residual, rfx_tracker, global_variance, gen);
  }
}

} // namespace StochTree

int main() {
  StochTree::RunAPI();
}
