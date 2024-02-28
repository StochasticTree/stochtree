/*! Copyright (c) 2024 stochtree authors*/
#include <stochtree/data.h>
#include <stochtree/container.h>
#include <stochtree/leaf_model.h>
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
      outcome[i * y_cols + j] = f_x_omega + rfx + normal_dist(gen);
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
  int n = 100;
  int x_rows = n;
  int x_cols = 10;
  int omega_rows = n;
  int omega_cols = 1;
  int y_rows = n;
  int y_cols = 1;
  int rfx_basis_rows = n;
  int rfx_basis_cols = 1;
  
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
  
  // Initialize an ensemble
  int num_trees = 100;
  int output_dimension = 1;
  bool is_leaf_constant = false;
  ForestContainer forest_samples = ForestContainer(num_trees, output_dimension, is_leaf_constant);

  // Initialize a leaf model
  double leaf_prior_mean = 0.;
  double leaf_prior_scale = 1.;
  GaussianConstantLeafModel leaf_model = GaussianConstantLeafModel(leaf_prior_scale);

  // // Check consistency
  // CHECK(ForestModelCompatible(forest_samples, leaf_model));

  // // Initialize a global variance model
  // double a_global_variance = 1.;
  // double b_global_variance = 1.;
  // HomoskedasticVarianceModel global_variance_model = HomoskedasticVarianceModel(a_global_variance, b_global_variance);
  
  // Check consistency

  // Initialize a sampler
  std::vector<FeatureType> feature_types(x_cols, FeatureType::kNumeric);
  double alpha = 0.99;
  double beta = 1.25;
  int min_samples_leaf = 10;
  int cutpoint_grid_size = 500;
  TreePrior tree_prior = TreePrior(alpha, beta, min_samples_leaf);
  GFRForestSampler gfr_sampler = GFRForestSampler<GaussianConstantLeafModel>(cutpoint_grid_size);
  MCMCForestSampler mcmc_sampler = MCMCForestSampler<GaussianConstantLeafModel>();
  ForestTracker tracker = ForestTracker(dataset.GetCovariates(), feature_types, num_trees, n);
  
  // Run a single iteration of the GFR algorithm
  int random_seed = 1;
  std::mt19937 gen = std::mt19937(random_seed);
  double global_variance = 1.;
  gfr_sampler.SampleOneIter(tracker, forest_samples, leaf_model, dataset, residual, tree_prior, gen, global_variance, feature_types);
  mcmc_sampler.SampleOneIter(tracker, forest_samples, leaf_model, dataset, residual, tree_prior, gen, global_variance);
}

} // namespace StochTree

int main() {
  StochTree::RunAPI();
}
