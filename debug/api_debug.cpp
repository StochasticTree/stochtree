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

void GenerateDGP1(std::vector<double>& covariates, std::vector<double>& basis, std::vector<double>& outcome, std::vector<double>& rfx_basis, std::vector<int32_t>& rfx_groups, std::vector<FeatureType>& feature_types, std::mt19937& gen, int& n, int& x_cols, int& omega_cols, int& y_cols, int& rfx_basis_cols, int& num_rfx_groups, bool rfx_included, int random_seed = -1) {
  // Data dimensions
  n = 1000;
  x_cols = 2;
  omega_cols = 1;
  y_cols = 1;
  if (rfx_included) {
    num_rfx_groups = 2;
    rfx_basis_cols = 1;
  } else {
    num_rfx_groups = 0;
    rfx_basis_cols = 0;
  }

  // Resize data
  covariates.resize(n * x_cols);
  basis.resize(n * omega_cols);
  rfx_basis.resize(n * rfx_basis_cols);
  outcome.resize(n * y_cols);
  rfx_groups.resize(n);
  feature_types.resize(x_cols, FeatureType::kNumeric);
  
  // Random number generation
  std::uniform_real_distribution<double> uniform_dist{0.0,1.0};
  std::normal_distribution<double> normal_dist(0.,1.);
  
  // DGP parameters
  std::vector<double> betas{-10, -5, 5, 10};
  int num_partitions = betas.size();
  double f_x_omega;
  double rfx;
  double error;

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < x_cols; j++) {
      covariates[i*x_cols + j] = uniform_dist(gen);
    }
    
    for (int j = 0; j < omega_cols; j++) {
      basis[i*omega_cols + j] = uniform_dist(gen);
    }
    
    if (rfx_included) {
      for (int j = 0; j < rfx_basis_cols; j++) {
        rfx_basis[i * rfx_basis_cols + j] = 1;
      }

      if (i % 2 == 0) {
        rfx_groups[i] = 1;
      }
      else {
        rfx_groups[i] = 2;
      }
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
      error = 0.1 * normal_dist(gen);
      outcome[i * y_cols + j] = f_x_omega + error;
      if (rfx_included) {
        if (rfx_groups[i] == 1) {
          rfx = 5.;
        }
        else {
          rfx = -5.;
        }
        outcome[i * y_cols + j] += rfx;
      }
    }
  }
}

void int_to_binary_vector(int32_t input, std::vector<int32_t>& output, int32_t offset) {
  bool terminated = false;
  int numerator = input;
  int vec_pos = 0;
  if (numerator < 2) {
    terminated = true;
    output.at(offset + vec_pos) = numerator;
  }
  while (!terminated) {
    std::div_t div_result = std::div(numerator, 2);
    output.at(offset + vec_pos) = div_result.rem;
    if (div_result.quot == 1) {
      terminated = true;
      output.at(offset + vec_pos + 1) = 1;
    } else {
      numerator = div_result.quot;
      vec_pos += 1;
    }
  }
}

void GenerateDGP2(std::vector<double>& covariates, std::vector<double>& basis, std::vector<double>& outcome, std::vector<double>& rfx_basis, std::vector<int32_t>& rfx_groups, std::vector<FeatureType>& feature_types, std::mt19937& gen, int& n, int& x_cols, int& omega_cols, int& y_cols, int& rfx_basis_cols, int& num_rfx_groups, bool rfx_included, int random_seed = -1) {
  // Data dimensions
  int n1 = 50;
  x_cols = 10;
  int num_cells = std::pow(2, x_cols);
  int p1 = 100;
  if (p1 >= num_cells) Log::Fatal("p1 must be < 2^x_cols");
  n = n1*num_cells;
  omega_cols = 0;
  y_cols = 1;
  if (rfx_included) {
    num_rfx_groups = 2;
    rfx_basis_cols = 1;
  }
  else {
    num_rfx_groups = 0;
    rfx_basis_cols = 0;
  }

  // Resize data
  covariates.resize(n * x_cols);
  basis.resize(n * omega_cols);
  rfx_basis.resize(n * rfx_basis_cols);
  outcome.resize(n * y_cols);
  rfx_groups.resize(n);
  feature_types.resize(x_cols, FeatureType::kNumeric);

  // Random number generation
  std::normal_distribution<double> normal_dist(0., 1.);

  // Generate a sequence of integers from 0 to num_cells - 1
  std::vector<int32_t> cell_run(num_cells);
  std::iota(cell_run.begin(), cell_run.end(), 0);

  // Repeat this sequence n1 times as the "covariates"
  std::vector<int32_t> cell_vector;
  for (int i = 0; i < n1; i++) {
    std::copy(cell_run.begin(), cell_run.end(), std::back_inserter(cell_vector));
  }

  // Convert cells to binary covariate columns (row-major)
  std::vector<int> covariates_binary(n * x_cols);
  int32_t offset = 0;
  for (size_t i = 0; i < n; i++) {
    int_to_binary_vector(cell_vector.at(i), covariates_binary, offset);
    offset += x_cols;
  }

  // Add (folded) gaussian noise to the binary covariates
  // std::vector<double> covariates_numeric(n* x_cols);
  std::vector<double> noise1(n);
  std::vector<double> noise2(n);
  int switch_flip;
  for (size_t i = 0; i < n; i++) {
    noise1.at(i) = std::abs(normal_dist(gen));
    noise2.at(i) = std::abs(normal_dist(gen));
  }
  for (size_t i = 0; i < n; i++) {
    for (int j = 0; j < x_cols; j++) {
      switch_flip = covariates_binary.at(i * x_cols + j);
      covariates.at(i * x_cols + j) = switch_flip * noise1.at(i) + (1 - switch_flip) * noise2.at(i);
    }
  }

  // DGP parameters
  double intercept = 0.5;
  std::vector<double> cell_coefficients_sparse(p1-1);
  for (int i = 0; i < cell_coefficients_sparse.size(); i++) {
    cell_coefficients_sparse.at(i) = -10*std::abs(normal_dist(gen));
  }
  std::vector<double> cell_weights(num_cells, 1./num_cells);
  std::vector<double> cell_indices_sparse(p1 - 1);
  std::discrete_distribution cell_selector(cell_weights.begin(), cell_weights.end());
  for (int i = 0; i < p1-1; i++) {
    cell_indices_sparse.at(i) = cell_selector(gen);
  }
  std::vector<double> cell_coefficients_full(num_cells, 0.);
  for (int i = 0; i < p1 - 1; i++) {
    cell_coefficients_full.at(cell_indices_sparse.at(i)) = cell_coefficients_sparse.at(i);
  }
  double f_x;
  double rfx;
  double error;

  // Outcome
  for (int i = 0; i < n; i++) {
    f_x = intercept + cell_coefficients_full.at(cell_vector.at(i));

    if (rfx_included) {
      for (int j = 0; j < rfx_basis_cols; j++) {
        rfx_basis[i * rfx_basis_cols + j] = 1;
      }

      if (i % 2 == 0) {
        rfx_groups[i] = 1;
      }
      else {
        rfx_groups[i] = 2;
      }
    }

    for (int j = 0; j < y_cols; j++) {
      error = 0.1 * normal_dist(gen);
      outcome[i * y_cols + j] = f_x +  error;
      if (rfx_included) {
        if (rfx_groups[i] == 1) {
          rfx = 5.;
        }
        else {
          rfx = -5.;
        }
        outcome[i * y_cols + j] += rfx;
      }
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

void RunDebug(int dgp_num = 0, bool rfx_included = false, int num_gfr = 10, int num_mcmc = 100, int random_seed = -1) {
  // Flag the data as row-major
  bool row_major = true;

  // Random number generation
  std::mt19937 gen;
  if (random_seed == -1) {
    std::random_device rd;
    std::mt19937 gen(rd());
  }
  else {
    std::mt19937 gen(random_seed);
  }

  // Initialize dataset
  ForestDataset dataset = ForestDataset();

  // Empty data containers and dimensions (filled in by calling a specific DGP simulation function below)
  int n;
  int x_cols;
  int omega_cols;
  int y_cols;
  int num_rfx_groups;
  int rfx_basis_cols;
  std::vector<double> covariates_raw;
  std::vector<double> basis_raw;
  std::vector<double> rfx_basis_raw;
  std::vector<double> outcome_raw;
  std::vector<int32_t> rfx_groups;
  std::vector<FeatureType> feature_types;

  // Generate the data
  int output_dimension;
  bool is_leaf_constant;
  ForestLeafModel leaf_model_type;
  if (dgp_num == 0) {
    GenerateDGP1(covariates_raw, basis_raw, outcome_raw, rfx_basis_raw, rfx_groups, feature_types, gen, n, x_cols, omega_cols, y_cols, rfx_basis_cols, num_rfx_groups, rfx_included, random_seed);
    dataset.AddCovariates(covariates_raw.data(), n, x_cols, row_major);
    dataset.AddBasis(basis_raw.data(), n, omega_cols, row_major);
    output_dimension = 1;
    is_leaf_constant = false;
    leaf_model_type = ForestLeafModel::kUnivariateRegression;
  }
  else if (dgp_num == 1) {
    GenerateDGP2(covariates_raw, basis_raw, outcome_raw, rfx_basis_raw, rfx_groups, feature_types, gen, n, x_cols, omega_cols, y_cols, rfx_basis_cols, num_rfx_groups, rfx_included, random_seed);
    dataset.AddCovariates(covariates_raw.data(), n, x_cols, row_major);
    output_dimension = 1;
    is_leaf_constant = true;
    leaf_model_type = ForestLeafModel::kConstant;
  }
  else {
    Log::Fatal("Invalid dgp_num");
  }

  // Construct residual
  ColumnVector residual = ColumnVector(outcome_raw.data(), n);

  // Center and scale the data
  double outcome_offset;
  double outcome_scale;
  OutcomeOffsetScale(residual, outcome_offset, outcome_scale);

  RandomEffectsDataset rfx_dataset;
  std::vector<int> rfx_init(n, 0);
  RandomEffectsTracker rfx_tracker = RandomEffectsTracker(rfx_init);
  MultivariateRegressionRandomEffectsModel rfx_model = MultivariateRegressionRandomEffectsModel(1, 1);
  RandomEffectsContainer rfx_container;
  LabelMapper label_mapper;
  if (rfx_included) {
    // Construct a random effects dataset
    rfx_dataset = RandomEffectsDataset();
    rfx_dataset.AddBasis(rfx_basis_raw.data(), n, rfx_basis_cols, true);
    rfx_dataset.AddGroupLabels(rfx_groups);

    // Construct random effects tracker / model / container
    RandomEffectsTracker rfx_tracker = RandomEffectsTracker(rfx_groups);
    MultivariateRegressionRandomEffectsModel rfx_model = MultivariateRegressionRandomEffectsModel(rfx_basis_cols, num_rfx_groups);
    RandomEffectsContainer rfx_container = RandomEffectsContainer(rfx_basis_cols, num_rfx_groups);
    LabelMapper label_mapper = LabelMapper(rfx_tracker.GetLabelMap());

    // Set random effects model parameters
    Eigen::VectorXd working_param_init(rfx_basis_cols);
    Eigen::MatrixXd group_param_init(rfx_basis_cols, num_rfx_groups);
    Eigen::MatrixXd working_param_cov_init(rfx_basis_cols, rfx_basis_cols);
    Eigen::MatrixXd group_param_cov_init(rfx_basis_cols, rfx_basis_cols);
    double variance_prior_shape = 1.;
    double variance_prior_scale = 1.;
    working_param_init << 1.;
    group_param_init << 1., 1.;
    working_param_cov_init << 1;
    group_param_cov_init << 1;
    rfx_model.SetWorkingParameter(working_param_init);
    rfx_model.SetGroupParameters(group_param_init);
    rfx_model.SetWorkingParameterCovariance(working_param_cov_init);
    rfx_model.SetGroupParameterCovariance(group_param_cov_init);
    rfx_model.SetVariancePriorShape(variance_prior_shape);
    rfx_model.SetVariancePriorScale(variance_prior_scale);
  }

  // Initialize an ensemble
  int num_trees = 50;
  ForestContainer forest_samples = ForestContainer(num_trees, output_dimension, is_leaf_constant);

  // Initialize a leaf model
  double leaf_prior_mean = 0.;
  double leaf_prior_scale = 1.;
  
  // Initialize forest sampling machinery
  double alpha = 1;
  double beta = 0.1;
  int min_samples_leaf = 1;
  int max_depth = 10;
  int cutpoint_grid_size = 100;
  double a_rfx = 1.;
  double b_rfx = 1.;
  double a_leaf = 2.;
  double b_leaf = 0.5;
  double a_global = 4.;
  double b_global = 2.;

  // Set leaf model parameters
  double leaf_scale_init = 1.;
  Eigen::MatrixXd leaf_scale_matrix, leaf_scale_matrix_init;
  // leaf_scale_matrix_init << 1.0, 0.0, 0.0, 1.0;
  double leaf_scale;

  // Set global variance
  double global_variance_init = 1.0;
  double global_variance;

  // Set variable weights
  double const_var_wt = static_cast<double>(1. / x_cols);
  std::vector<double> variable_weights(x_cols, const_var_wt);

  // Initialize tracker and tree prior
  ForestTracker tracker = ForestTracker(dataset.GetCovariates(), feature_types, num_trees, n);
  TreePrior tree_prior = TreePrior(alpha, beta, min_samples_leaf, max_depth);

  // Initialize variance models
  GlobalHomoskedasticVarianceModel global_var_model = GlobalHomoskedasticVarianceModel();
  LeafNodeHomoskedasticVarianceModel leaf_var_model = LeafNodeHomoskedasticVarianceModel();

  // Initialize storage for samples of variance
  std::vector<double> global_variance_samples{};
  std::vector<double> leaf_variance_samples{};

  // Run the GFR sampler
  if (num_gfr > 0) {
    for (int i = 0; i < num_gfr; i++) {
      if (i == 0) {
        global_variance = global_variance_init;
        leaf_scale = leaf_scale_init;
      }
      else {
        global_variance = global_variance_samples[i - 1];
        leaf_scale = leaf_variance_samples[i - 1];
      }

      // Sample tree ensemble
      sampleGFR(tracker, tree_prior, forest_samples, dataset, residual, gen, feature_types, variable_weights,
        leaf_model_type, leaf_scale_matrix, global_variance, leaf_scale, cutpoint_grid_size);

      if (rfx_included) {
        // Sample random effects
        rfx_model.SampleRandomEffects(rfx_dataset, residual, rfx_tracker, global_variance, gen);
        rfx_container.AddSample(rfx_model);
      }

      // Sample leaf node variance
      leaf_variance_samples.push_back(leaf_var_model.SampleVarianceParameter(forest_samples.GetEnsemble(i), a_leaf, b_leaf, gen));

      // Sample global variance
      global_variance_samples.push_back(global_var_model.SampleVarianceParameter(residual.GetData(), a_global, b_global, gen));
    }
  }

  // Run the MCMC sampler
  if (num_mcmc > 0) {
    for (int i = num_gfr; i < num_gfr + num_mcmc; i++) {
      if (i == 0) {
        global_variance = global_variance_init;
        leaf_scale = leaf_scale_init;
      }
      else {
        global_variance = global_variance_samples[i - 1];
        leaf_scale = leaf_variance_samples[i - 1];
      }

      // Sample tree ensemble
      sampleMCMC(tracker, tree_prior, forest_samples, dataset, residual, gen, feature_types, variable_weights,
        leaf_model_type, leaf_scale_matrix, global_variance, leaf_scale, cutpoint_grid_size);

      if (rfx_included) {
        // Sample random effects
        rfx_model.SampleRandomEffects(rfx_dataset, residual, rfx_tracker, global_variance, gen);
        rfx_container.AddSample(rfx_model);
      }

      // Sample leaf node variance
      leaf_variance_samples.push_back(leaf_var_model.SampleVarianceParameter(forest_samples.GetEnsemble(i), a_leaf, b_leaf, gen));

      // Sample global variance
      global_variance_samples.push_back(global_var_model.SampleVarianceParameter(residual.GetData(), a_global, b_global, gen));
    }
  }

  // Predict from the tree ensemble
  int num_samples = num_gfr + num_mcmc;
  std::vector<double> pred_orig = forest_samples.Predict(dataset);

  if (rfx_included) {
    // Predict from the random effects dataset
    std::vector<double> rfx_predictions(n * num_samples);
    rfx_container.Predict(rfx_dataset, label_mapper, rfx_predictions);
  }

  // Write model to a file
  std::string filename = "model.json";
  forest_samples.SaveToJsonFile(filename);

  // Read and parse json from file
  ForestContainer forest_samples_parsed = ForestContainer(num_trees, output_dimension, is_leaf_constant);
  forest_samples_parsed.LoadFromJsonFile(filename);
  
  // Make sure we can predict from both the original (above) and parsed forest containers
  std::vector<double> pred_parsed = forest_samples_parsed.Predict(dataset);
}

} // namespace StochTree

int main(int argc, char* argv[]) {
  // Unpack command line arguments
  int dgp_num = std::stoi(argv[1]);
  if ((dgp_num != 0) && (dgp_num != 1)) {
    StochTree::Log::Fatal("The first command line argument must be 0 or 1");
  }
  int rfx_int = std::stoi(argv[2]);
  if ((rfx_int != 0) && (rfx_int != 1)) {
    StochTree::Log::Fatal("The second command line argument must be 0 or 1");
  }
  bool rfx_included = static_cast<bool>(rfx_int);
  int num_gfr = std::stoi(argv[3]);
  if (num_gfr < 0) {
    StochTree::Log::Fatal("The third command line argument must be >= 0");
  }
  int num_mcmc = std::stoi(argv[4]);
  if (num_mcmc < 0) {
    StochTree::Log::Fatal("The fourth command line argument must be >= 0");
  }
  int random_seed = std::stoi(argv[5]);
  if (random_seed < -1) {
    StochTree::Log::Fatal("The fifth command line argument must be >= -0");
  }

  // Run the debug program
  StochTree::RunDebug(dgp_num, rfx_included, num_gfr, num_mcmc);
}
