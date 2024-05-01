/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include <stochtree/random_effects.h>

namespace StochTree {

RandomEffectsTracker::RandomEffectsTracker(std::vector<int32_t>& group_indices) {
  sample_category_mapper_ = std::make_unique<SampleCategoryMapper>(group_indices);
  category_sample_tracker_ = std::make_unique<CategorySampleTracker>(group_indices);
  num_categories_ = category_sample_tracker_->NumCategories();
  num_observations_ = group_indices.size();
  rfx_predictions_.resize(num_observations_, 0.);
}

void MultivariateRegressionRandomEffectsModel::SampleRandomEffects(RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& rfx_tracker, 
                                                                   double global_variance, std::mt19937& gen) {
  // Update partial residual to add back in the random effects
  AddCurrentPredictionToResidual(dataset, rfx_tracker, residual);
  
  // Sample random effects
  SampleWorkingParameter(dataset, residual, rfx_tracker, global_variance, gen);
  SampleGroupParameters(dataset, residual, rfx_tracker, global_variance, gen);
  SampleVarianceComponents(dataset, residual, rfx_tracker, global_variance, gen);

  // Update partial residual to remove the random effects
  SubtractNewPredictionFromResidual(dataset, rfx_tracker, residual);
}

void MultivariateRegressionRandomEffectsModel::SampleWorkingParameter(RandomEffectsDataset& dataset, ColumnVector& residual, 
                                                                      RandomEffectsTracker& rfx_tracker, double global_variance, std::mt19937& gen) {
  Eigen::VectorXd posterior_mean = WorkingParameterMean(dataset, residual, rfx_tracker, global_variance);
  Eigen::MatrixXd posterior_covariance = WorkingParameterVariance(dataset, residual, rfx_tracker, global_variance);
  working_parameter_ = normal_sampler_.SampleEigen(posterior_mean, posterior_covariance, gen);
}

void MultivariateRegressionRandomEffectsModel::SampleGroupParameters(RandomEffectsDataset& dataset, ColumnVector& residual, 
                                                                     RandomEffectsTracker& rfx_tracker, double global_variance, std::mt19937& gen) {
  int32_t num_groups = num_groups_;
  Eigen::VectorXd posterior_mean;
  Eigen::MatrixXd posterior_covariance;
  Eigen::VectorXd output;
  for (int i = 0; i < num_groups; i++) {
    posterior_mean = GroupParameterMean(dataset, residual, rfx_tracker, global_variance, i);
    posterior_covariance = GroupParameterVariance(dataset, residual, rfx_tracker, global_variance, i);
    group_parameters_(Eigen::all, i) = normal_sampler_.SampleEigen(posterior_mean, posterior_covariance, gen);
  }  
}

void MultivariateRegressionRandomEffectsModel::SampleVarianceComponents(RandomEffectsDataset& dataset, ColumnVector& residual, 
                                                                        RandomEffectsTracker& rfx_tracker, double global_variance, std::mt19937& gen) {
  int32_t num_components = num_components_;
  double posterior_shape;
  double posterior_scale;
  double output;
  for (int i = 0; i < num_components; i++) {
    posterior_shape = VarianceComponentShape(dataset, residual, rfx_tracker, global_variance, i);
    posterior_scale = VarianceComponentScale(dataset, residual, rfx_tracker, global_variance, i);
    group_parameter_covariance_(i, i) = ig_sampler_.Sample(posterior_shape, posterior_scale, gen);
  }
}

Eigen::VectorXd MultivariateRegressionRandomEffectsModel::WorkingParameterMean(RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& rfx_tracker, 
                                                                               double global_variance){
  int32_t num_components = num_components_;
  int32_t num_groups = num_groups_;
  std::vector<data_size_t> observation_indices;
  Eigen::MatrixXd X_group;
  Eigen::VectorXd y_group;
  Eigen::MatrixXd xi_group;
  Eigen::MatrixXd posterior_denominator = working_parameter_covariance_.inverse();
  Eigen::VectorXd posterior_numerator = Eigen::VectorXd::Zero(num_components);
  Eigen::MatrixXd X = dataset.GetBasis();
  Eigen::VectorXd y = residual.GetData();
  Eigen::MatrixXd xi = group_parameters_;
  for (int i = 0; i < num_groups; i++) {
    observation_indices = rfx_tracker.NodeIndices(i);
    X_group = X(observation_indices, Eigen::all);
    y_group = y(observation_indices, Eigen::all);
    xi_group = xi(Eigen::all, i);
    posterior_denominator += (xi_group).asDiagonal() * X_group.transpose() * X_group * (xi_group).asDiagonal();
    posterior_numerator += (xi_group).asDiagonal() * X_group.transpose() * y_group;
  }
  return posterior_denominator.inverse() * posterior_numerator;
}

Eigen::MatrixXd MultivariateRegressionRandomEffectsModel::WorkingParameterVariance(RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& rfx_tracker, double global_variance){
  int32_t num_components = num_components_;
  int32_t num_groups = num_groups_;
  std::vector<data_size_t> observation_indices;
  Eigen::MatrixXd X_group;
  Eigen::VectorXd y_group;
  Eigen::MatrixXd xi_group;
  Eigen::MatrixXd posterior_denominator = working_parameter_covariance_.inverse();
  Eigen::VectorXd posterior_numerator = Eigen::VectorXd::Zero(num_components);
  Eigen::MatrixXd X = dataset.GetBasis();
  Eigen::VectorXd y = residual.GetData();
  Eigen::MatrixXd xi = group_parameters_;
  for (int i = 0; i < num_groups; i++) {
    observation_indices = rfx_tracker.NodeIndices(i);
    X_group = X(observation_indices, Eigen::all);
    y_group = y(observation_indices, Eigen::all);
    xi_group = xi(Eigen::all, i);
    posterior_denominator += (xi_group).asDiagonal() * X_group.transpose() * X_group * (xi_group).asDiagonal();
    posterior_numerator += (xi_group).asDiagonal() * X_group.transpose() * y_group;
  }
  return posterior_denominator.inverse();
}

Eigen::VectorXd MultivariateRegressionRandomEffectsModel::GroupParameterMean(RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& rfx_tracker, double global_variance, int32_t group_id) {
  int32_t num_components = num_components_;
  int32_t num_groups = num_groups_;
  Eigen::MatrixXd X = dataset.GetBasis();
  Eigen::VectorXd y = residual.GetData();
  Eigen::VectorXd alpha = working_parameter_;
  Eigen::MatrixXd posterior_denominator = group_parameter_covariance_.inverse();
  Eigen::VectorXd posterior_numerator = Eigen::VectorXd::Zero(num_components);
  std::vector<data_size_t> observation_indices = rfx_tracker.NodeIndices(group_id);
  Eigen::MatrixXd X_group = X(observation_indices, Eigen::all);
  Eigen::VectorXd y_group = y(observation_indices, Eigen::all);
  posterior_denominator += (alpha).asDiagonal() * X_group.transpose() * X_group * (alpha).asDiagonal();
  posterior_numerator += (alpha).asDiagonal() * X_group.transpose() * y_group;
  return posterior_denominator.inverse() * posterior_numerator;
}

Eigen::MatrixXd MultivariateRegressionRandomEffectsModel::GroupParameterVariance(RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& rfx_tracker, double global_variance, int32_t group_id){
  int32_t num_components = num_components_;
  int32_t num_groups = num_groups_;
  Eigen::MatrixXd X = dataset.GetBasis();
  Eigen::VectorXd y = residual.GetData();
  Eigen::VectorXd alpha = working_parameter_;
  Eigen::MatrixXd posterior_denominator = group_parameter_covariance_.inverse();
  Eigen::VectorXd posterior_numerator = Eigen::VectorXd::Zero(num_components);
  std::vector<data_size_t> observation_indices = rfx_tracker.NodeIndices(group_id);
  Eigen::MatrixXd X_group = X(observation_indices, Eigen::all);
  Eigen::VectorXd y_group = y(observation_indices, Eigen::all);
  posterior_denominator += (alpha).asDiagonal() * X_group.transpose() * X_group * (alpha).asDiagonal();
  posterior_numerator += (alpha).asDiagonal() * X_group.transpose() * y_group;
  return posterior_denominator.inverse();
}

double MultivariateRegressionRandomEffectsModel::VarianceComponentShape(RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& rfx_tracker, double global_variance, int32_t component_id) {
  return static_cast<double>(variance_prior_shape_ + num_groups_);
}

double MultivariateRegressionRandomEffectsModel::VarianceComponentScale(RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& rfx_tracker, double global_variance, int32_t component_id) {
  int32_t num_groups = num_groups_;
  Eigen::MatrixXd xi = group_parameters_;
  double output = variance_prior_scale_;
  for (int i = 0; i < num_groups; i++) {
    output += xi(component_id, i)*xi(component_id, i);
  }
  return output;
}

}  // namespace StochTree
