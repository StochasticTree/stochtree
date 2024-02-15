/*! Copyright (c) 2024 by stochtree authors */
#include <Eigen/Dense>
#include <stochtree/dispatcher.h>

namespace StochTree {

void Dispatcher::AddOutcome(double* outcome_data_ptr, data_size_t num_row) {
  residual_.reset(new UnivariateResidual());
  residual_->LoadFromMemory(outcome_data_ptr, num_row);
}

void Dispatcher::AddConstantLeafForest(double* covariate_data_ptr, int num_covariate, data_size_t num_row, bool is_row_major, int num_trees, double mu_bar, double tau, double alpha, double beta, int min_samples_in_leaf, ForestSampler forest_sampler, std::vector<FeatureType> feature_types, bool leaf_variance_random, double a_leaf, double b_leaf) {
  // Add dataset
  forest_datasets_.push_back(std::make_unique<ConstantLeafForestDataset>());
  forest_datasets_[num_forests_]->LoadFromMemory(covariate_data_ptr, num_covariate, num_row, is_row_major);
  forest_feature_types_.push_back(feature_types);
  ConstantLeafForestDataset* forest_dataset = dynamic_cast<ConstantLeafForestDataset*>(forest_datasets_[num_forests_].get());

  // Add forest (with no drawn samples yet ... we'll add later)
  forest_sample_containers_.push_back(std::make_unique<TreeEnsembleContainer>(0, num_trees, 1, true));

  // Add leaf prior
  forest_leaf_priors_.push_back(std::make_unique<LeafConstantGaussianPrior>(mu_bar, tau));
  forest_leaf_prior_type_.push_back(ForestLeafPriorType::kConstantLeafGaussian);

  // Add tree prior
  forest_tree_priors_.push_back(std::make_unique<TreePrior>(alpha, beta, min_samples_in_leaf));

  // Add forest sampler
  forest_sampler_type_.push_back(forest_sampler);
  if (forest_sampler == ForestSampler::kMCMC) {
    forest_samplers_.push_back(std::make_unique<MCMCTreeSampler>());
    MCMCTreeSampler* forest_sampler = dynamic_cast<MCMCTreeSampler*>(forest_samplers_[num_forests_].get());
    forest_sampler->template Initialize<ConstantLeafForestDataset, LeafConstantGaussianPrior, LeafConstantGaussianSuffStat>(forest_dataset, num_trees, num_row, feature_types);
  // } else if (forest_sampler == ForestSampler::kGFR) {
  //   forest_samplers_.push_back(std::make_unique<GFRTreeSampler>());
  //    GFRTreeSampler* forest_sampler = dynamic_cast<GFRTreeSampler*>(forest_samplers_[num_forests_].get());
  //    forest_sampler->template Initialize<RegressionLeafForestDataset>(forest_dataset, num_trees, num_row, feature_types);
  } else {
    Log::Fatal("Sampler %s not implemented", forest_sampler);
  }

  // Add forest leaf sampler
  forest_leaf_mean_samplers_.push_back(std::make_unique<LeafConstantGaussianSampler>());

  // Add tracking info on leaf scale sampling
  if (leaf_variance_random) {
    forest_leaf_variance_type_.push_back(ForestLeafVarianceType::kStochastic);
  } else {
    forest_leaf_variance_type_.push_back(ForestLeafVarianceType::kFixed);
  }
  forest_leaf_variance_samplers_.push_back(std::make_unique<LeafNodeHomoskedasticVarianceSampler>());
  forest_leaf_variance_priors_.push_back(std::make_unique<IGVariancePrior>(a_leaf, b_leaf));
  // TODO: eventually this will be refactored into a contained of "LeafScaleSample" objects, flexible enough to be scalar / matrix
  forest_leaf_variance_sample_containers_.push_back(std::vector<double>(0));

  // Increment forest count
  num_forests_++;
}

void Dispatcher::AddUnivariateRegressionLeafForest(double* covariate_data_ptr, int num_covariate, double* basis_data_ptr, int num_basis, data_size_t num_row, bool is_row_major, int num_trees, double beta_bar, double tau, double alpha, double beta, int min_samples_in_leaf, ForestSampler forest_sampler, std::vector<FeatureType> feature_types, bool leaf_variance_random, double a_leaf, double b_leaf) {
  // Add dataset
  forest_datasets_.push_back(std::make_unique<RegressionLeafForestDataset>());
  forest_datasets_[num_forests_]->LoadFromMemory(covariate_data_ptr, num_covariate, basis_data_ptr, num_basis, num_row, is_row_major);
  forest_feature_types_.push_back(feature_types);
  RegressionLeafForestDataset* forest_dataset = dynamic_cast<RegressionLeafForestDataset*>(forest_datasets_[num_forests_].get());

  // Add forest (with no drawn samples yet ... we'll add later)
  forest_sample_containers_.push_back(std::make_unique<TreeEnsembleContainer>(0, num_trees, 1, true));

  // Add leaf prior
  forest_leaf_priors_.push_back(std::make_unique<LeafUnivariateRegressionGaussianPrior>(beta_bar, tau));
  forest_leaf_prior_type_.push_back(ForestLeafPriorType::kUnivariateRegressionLeafGaussian);

  // Add tree prior
  forest_tree_priors_.push_back(std::make_unique<TreePrior>(alpha, beta, min_samples_in_leaf));

  // Add forest sampler
  if (forest_sampler == ForestSampler::kMCMC) {
    forest_samplers_.push_back(std::make_unique<MCMCTreeSampler>());
    MCMCTreeSampler* forest_sampler = dynamic_cast<MCMCTreeSampler*>(forest_samplers_[num_forests_].get());
    forest_sampler->template Initialize<RegressionLeafForestDataset, LeafUnivariateRegressionGaussianPrior, LeafUnivariateRegressionGaussianSuffStat>(forest_dataset, num_trees, num_row, feature_types);
  // } else if (forest_sampler == ForestSampler::kGFR) {
  //   forest_samplers_.push_back(std::make_unique<GFRTreeSampler>());
  //    GFRTreeSampler* forest_sampler = dynamic_cast<GFRTreeSampler*>(forest_samplers_[num_forests_].get());
  //    forest_sampler->template Initialize<RegressionLeafForestDataset>(forest_dataset, num_trees, num_row, feature_types);
  } else {
    Log::Fatal("Sampler %s not implemented", forest_sampler);
  }

  // Add forest leaf sampler
  forest_leaf_mean_samplers_.push_back(std::make_unique<LeafUnivariateRegressionGaussianSampler>());

  // Add tracking info on leaf scale sampling
  if (leaf_variance_random) {
    forest_leaf_variance_type_.push_back(ForestLeafVarianceType::kStochastic);
  } else {
    forest_leaf_variance_type_.push_back(ForestLeafVarianceType::kFixed);
  }
  forest_leaf_variance_samplers_.push_back(std::make_unique<LeafNodeHomoskedasticVarianceSampler>());
  forest_leaf_variance_priors_.push_back(std::make_unique<IGVariancePrior>(a_leaf, b_leaf));
  // TODO: eventually this will be refactored into a contained of "LeafScaleSample" objects, flexible enough to be scalar / matrix
  forest_leaf_variance_sample_containers_.push_back(std::vector<double>(0));

  // Increment forest count
  num_forests_++;
}

void Dispatcher::AddMultivariateRegressionLeafForest(double* covariate_data_ptr, int num_covariate, double* basis_data_ptr, int num_basis, data_size_t num_row, bool is_row_major, int num_trees, Eigen::VectorXd& Beta, Eigen::MatrixXd& Sigma, double alpha, double beta, int min_samples_in_leaf, ForestSampler forest_sampler, std::vector<FeatureType> feature_types, bool leaf_variance_random) {
  // Add dataset
  forest_datasets_.push_back(std::make_unique<RegressionLeafForestDataset>());
  forest_datasets_[num_forests_]->LoadFromMemory(covariate_data_ptr, num_covariate, basis_data_ptr, num_basis, num_row, is_row_major);
  forest_feature_types_.push_back(feature_types);
  RegressionLeafForestDataset* forest_dataset = dynamic_cast<RegressionLeafForestDataset*>(forest_datasets_[num_forests_].get());

  // Add forest (with no drawn samples yet ... we'll add later)
  forest_sample_containers_.push_back(std::make_unique<TreeEnsembleContainer>(0, num_trees, 1, true));

  // Add leaf prior
  forest_leaf_priors_.push_back(std::make_unique<LeafMultivariateRegressionGaussianPrior>(Beta, Sigma, num_basis));
  forest_leaf_prior_type_.push_back(ForestLeafPriorType::kMultivariateRegressionLeafGaussian);

  // Add tree prior
  forest_tree_priors_.push_back(std::make_unique<TreePrior>(alpha, beta, min_samples_in_leaf));

  // Add forest sampler
  if (forest_sampler == ForestSampler::kMCMC) {
    forest_samplers_.push_back(std::make_unique<MCMCTreeSampler>());
    MCMCTreeSampler* forest_sampler = dynamic_cast<MCMCTreeSampler*>(forest_samplers_[num_forests_].get());
    forest_sampler->template Initialize<RegressionLeafForestDataset, LeafMultivariateRegressionGaussianPrior, LeafMultivariateRegressionGaussianSuffStat>(forest_dataset, num_trees, num_row, feature_types);
  // } else if (forest_sampler == ForestSampler::kGFR) {
  //   forest_samplers_.push_back(std::make_unique<GFRTreeSampler>());
  //    GFRTreeSampler* forest_sampler = dynamic_cast<GFRTreeSampler*>(forest_samplers_[num_forests_].get());
  //    forest_sampler->template Initialize<RegressionLeafForestDataset>(forest_dataset, num_trees, num_row, feature_types);
  } else {
    Log::Fatal("Sampler %s not implemented", forest_sampler);
  }

  // Add forest leaf sampler
  forest_leaf_mean_samplers_.push_back(std::make_unique<LeafMultivariateRegressionGaussianSampler>());

  // Add tracking info on leaf scale sampling
  if (leaf_variance_random) {
    Log::Fatal("Leaf covariance sampling is not implemented for multivariate leaf node regression");
  } else {
    forest_leaf_variance_type_.push_back(ForestLeafVarianceType::kFixed);
  }
  forest_leaf_variance_samplers_.push_back(std::make_unique<LeafNodeHomoskedasticVarianceSampler>());
  forest_leaf_variance_priors_.push_back(std::make_unique<IGVariancePrior>(1., 1.));
  // TODO: eventually this will be refactored into a contained of "LeafScaleSample" objects, flexible enough to be scalar / matrix
  forest_leaf_variance_sample_containers_.push_back(std::vector<double>(0));

  // Increment forest count
  num_forests_++;
}

void Dispatcher::AddRandomEffectRegression(double* basis_data_ptr, int num_basis, data_size_t num_row, bool is_row_major, std::vector<int32_t> group_indices, double a, double b, int num_components, int num_groups) {
  // Add dataset
  rfx_datasets_.push_back(std::make_unique<RegressionRandomEffectsDataset>());
  rfx_datasets_[num_rfx_]->LoadFromMemory(basis_data_ptr, num_basis, num_row, is_row_major, group_indices);

  // Add sample container (with no drawn samples yet ... we'll add later)
  rfx_sample_containers_.push_back(std::make_unique<RandomEffectsContainer>(0));

  // Add prior
  rfx_priors_.push_back(std::make_unique<RandomEffectsRegressionGaussianPrior>(a, b, num_components, num_groups));

  // Add sampler
  rfx_samplers_.push_back(std::make_unique<RandomEffectsSampler>(rfx_datasets_[num_rfx_].get(), rfx_priors_[num_rfx_].get()));

  // Increment rfx count
  num_rfx_++;
}

void Dispatcher::AddGlobalVarianceTerm(double a, double b, double global_variance_init) {
  // Update sample container (with no drawn samples yet ... we'll add later)
  global_variance_sample_container_.resize(0);

  // Update sample container (with no drawn samples yet ... we'll add later)
  global_variance_init_ = global_variance_init;
  global_variance_prior_ = std::make_unique<IGVariancePrior>(a, b);
  global_variance_sampler_ = std::make_unique<GlobalHomoskedasticVarianceSampler>();
}

void Dispatcher::CenterScaleOutcome() {
  // Compute and store the offset / scale factors
  double var_y = 0.0;
  double outcome_sum_squares = 0.0;
  double outcome_sum = 0.0;
  double outcome_val;
  data_size_t n = residual_->residual.rows();
  for (data_size_t i = 0; i < n; i++){
    outcome_val = residual_->residual(i);
    outcome_sum += outcome_val;
    outcome_sum_squares += std::pow(outcome_val, 2.0);
  }
  var_y = outcome_sum_squares/n - std::pow(outcome_sum / n, 2.0);
  
  // Store the offset and scale factors (necessary for prediction later on)
  outcome_scale_ = std::sqrt(var_y);
  outcome_offset_ = outcome_sum / n;

  // Update the residual
  residual_->residual = residual_->residual.array() - outcome_offset_;
  residual_->residual /= outcome_scale_;
}

double Dispatcher::MeanOutcome() {
  double outcome_sum = 0.0;
  double outcome_val;
  data_size_t n = residual_->residual.rows();
  for (data_size_t i = 0; i < n; i++){
    outcome_val = residual_->residual(i);
    outcome_sum += outcome_val;
  }
  return outcome_sum / n;
}

void Dispatcher::InitializeForest(int forest_num, int sample_num, double initial_forest_pred) {
  // Compute the implied leaf value initialization for each root node
  double initial_leaf_value;
  std::vector<double> initial_leaf_values;
  int num_basis;
  int num_trees = forest_sample_containers_[forest_num]->NumTrees();
  TreeEnsembleContainer* ensemble_container = forest_sample_containers_[forest_num].get();
  UnivariateResidual* residual = residual_.get();
  if (forest_leaf_prior_type_[forest_num] == ForestLeafPriorType::kConstantLeafGaussian) {
    initial_leaf_value = initial_forest_pred / num_trees;
    num_basis = 0;
  } else if (forest_leaf_prior_type_[forest_num] == ForestLeafPriorType::kUnivariateRegressionLeafGaussian) {
    RegressionLeafForestDataset* dataset = dynamic_cast<RegressionLeafForestDataset*>(forest_datasets_[forest_num].get());
    initial_leaf_value = (initial_forest_pred / num_trees) / (dataset->basis.array().sum());
  } else if (forest_leaf_prior_type_[forest_num] == ForestLeafPriorType::kMultivariateRegressionLeafGaussian) {
    // TODO: find a heuristic initialization that yields mean_outcome as a prediction
    std::vector<double> initial_leaf_values;
    RegressionLeafForestDataset* dataset = dynamic_cast<RegressionLeafForestDataset*>(forest_datasets_[forest_num].get());
    Eigen::MatrixXd leaf_reg_solution = (dataset->basis.transpose() * dataset->basis).inverse() * dataset->basis.transpose() * residual->residual;
    int num_basis = dataset->basis.cols();
    initial_leaf_values.resize(num_basis);
    for (int i = 0; i < num_basis; i++) {
      initial_leaf_values[i] = leaf_reg_solution(i,0) / num_trees;
    }
  }

  TreeEnsemble* ensemble = ensemble_container->GetEnsemble(sample_num);
  for (int j = 0; j < num_trees; j++) {
    Tree* tree = ensemble->GetTree(j);
    if (num_basis <= 1) {
      tree->SetLeaf(0, initial_leaf_value);
    } else {
      tree->SetLeafVector(0, initial_leaf_values);
    }
    forest_samplers_[forest_num]->AssignAllSamplesToRoot(j);
  }
}

} // namespace StochTree
