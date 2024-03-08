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
}

void MultivariateRegressionRandomEffectsModel::SampleRandomEffects(RandomEffectsTerm* rfx, RandomEffectsDataset& dataset, ColumnVector& residual, 
                                                                   RandomEffectsTracker& rfx_tracker, double global_variance, std::mt19937& gen) {
  // Update partial residual to add back in the random effects
  std::vector<double> rfx_pred = rfx->Predict(dataset);
  int32_t n = dataset.GetBasis().rows();
  double resid_value;
  for (int i = 0; i < n; i++) {
    resid_value = residual.GetElement(i);
    residual.SetElement(i, resid_value + rfx_pred[i]);
  }
  
  // Sample random effects
  SampleWorkingParameter(rfx, dataset, residual, rfx_tracker, global_variance, gen);
  SampleGroupParameters(rfx, dataset, residual, rfx_tracker, global_variance, gen);
  SampleVarianceComponents(rfx, dataset, residual, rfx_tracker, global_variance, gen);

  // Update partial residual to remove the random effects
  rfx_pred = rfx->Predict(dataset);
  for (int i = 0; i < n; i++) {
    resid_value = residual.GetElement(i);
    residual.SetElement(i, resid_value + rfx_pred[i]);
  }
}

void MultivariateRegressionRandomEffectsModel::SampleWorkingParameter(RandomEffectsTerm* rfx, RandomEffectsDataset& dataset, ColumnVector& residual, 
                                                                      RandomEffectsTracker& rfx_tracker, double global_variance, std::mt19937& gen) {
  Eigen::VectorXd posterior_mean = WorkingParameterMean(dataset, residual, rfx_tracker, *rfx, global_variance);
  Eigen::MatrixXd posterior_covariance = WorkingParameterVariance(dataset, residual, rfx_tracker, *rfx, global_variance);
  Eigen::VectorXd output = normal_sampler_.SampleEigen(posterior_mean, posterior_covariance, gen);
  rfx->SetWorkingParameter(output);
}

void MultivariateRegressionRandomEffectsModel::SampleGroupParameters(RandomEffectsTerm* rfx, RandomEffectsDataset& dataset, ColumnVector& residual, 
                                                                     RandomEffectsTracker& rfx_tracker, double global_variance, std::mt19937& gen) {
  int32_t num_groups = rfx->NumGroups();
  Eigen::VectorXd posterior_mean;
  Eigen::MatrixXd posterior_covariance;
  Eigen::VectorXd output;
  for (int i = 0; i < num_groups; i++) {
    posterior_mean = GroupParameterMean(dataset, residual, rfx_tracker, *rfx, global_variance, i);
    posterior_covariance = GroupParameterVariance(dataset, residual, rfx_tracker, *rfx, global_variance, i);
    output = normal_sampler_.SampleEigen(posterior_mean, posterior_covariance, gen);
    rfx->SetGroupParameter(output, i);
  }  
}

void MultivariateRegressionRandomEffectsModel::SampleVarianceComponents(RandomEffectsTerm* rfx, RandomEffectsDataset& dataset, ColumnVector& residual, 
                                                                        RandomEffectsTracker& rfx_tracker, double global_variance, std::mt19937& gen) {
  // int32_t num_groups = rfx->NumGroups();
  int32_t num_components = rfx->NumComponents();
  double posterior_shape;
  double posterior_scale;
  double output;
  for (int i = 0; i < num_components; i++) {
    posterior_shape = VarianceComponentShape(dataset, residual, rfx_tracker, *rfx, global_variance, i);
    posterior_scale = VarianceComponentScale(dataset, residual, rfx_tracker, *rfx, global_variance, i);
    output = ig_sampler_.Sample(posterior_shape, posterior_scale, gen);
    rfx->SetGroupParameterVarianceComponent(output, i);
  }
}

Eigen::VectorXd MultivariateRegressionRandomEffectsModel::WorkingParameterMean(RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& rfx_tracker, 
                                                                               RandomEffectsTerm& rfx_term, double global_variance){
  int32_t num_components = rfx_term.NumComponents();
  int32_t num_groups = rfx_term.NumGroups();
  std::vector<data_size_t> observation_indices;
  Eigen::MatrixXd X_group;
  Eigen::VectorXd y_group;
  Eigen::MatrixXd xi_group;
  Eigen::MatrixXd posterior_denominator = rfx_term.GetWorkingParameterCovariance().inverse();
  Eigen::VectorXd posterior_numerator = Eigen::VectorXd::Zero(num_components);
  Eigen::MatrixXd X = dataset.GetBasis();
  Eigen::VectorXd y = residual.GetData();
  Eigen::MatrixXd xi = rfx_term.GetGroupParameters();
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

Eigen::MatrixXd MultivariateRegressionRandomEffectsModel::WorkingParameterVariance(RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& rfx_tracker, 
                                                                                   RandomEffectsTerm& rfx_term, double global_variance){
  int32_t num_components = rfx_term.NumComponents();
  int32_t num_groups = rfx_term.NumGroups();
  std::vector<data_size_t> observation_indices;
  Eigen::MatrixXd X_group;
  Eigen::VectorXd y_group;
  Eigen::MatrixXd xi_group;
  Eigen::MatrixXd posterior_denominator = rfx_term.GetWorkingParameterCovariance().inverse();
  Eigen::VectorXd posterior_numerator = Eigen::VectorXd::Zero(num_components);
  Eigen::MatrixXd X = dataset.GetBasis();
  Eigen::VectorXd y = residual.GetData();
  Eigen::MatrixXd xi = rfx_term.GetGroupParameters();
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

Eigen::VectorXd MultivariateRegressionRandomEffectsModel::GroupParameterMean(RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& rfx_tracker, 
                                                                             RandomEffectsTerm& rfx_term, double global_variance, int32_t group_id) {
  int32_t num_components = rfx_term.NumComponents();
  int32_t num_groups = rfx_term.NumGroups();
  Eigen::MatrixXd X = dataset.GetBasis();
  Eigen::VectorXd y = residual.GetData();
  Eigen::VectorXd alpha = rfx_term.GetWorkingParameter();
  Eigen::MatrixXd posterior_denominator = rfx_term.GetGroupParameterCovariance().inverse();
  Eigen::VectorXd posterior_numerator = Eigen::VectorXd::Zero(num_components);
  std::vector<data_size_t> observation_indices = rfx_tracker.NodeIndices(group_id);
  Eigen::MatrixXd X_group = X(observation_indices, Eigen::all);
  Eigen::VectorXd y_group = y(observation_indices, Eigen::all);
  posterior_denominator += (alpha).asDiagonal() * X_group.transpose() * X_group * (alpha).asDiagonal();
  posterior_numerator += (alpha).asDiagonal() * X_group.transpose() * y_group;
  return posterior_denominator.inverse() * posterior_numerator;
}

Eigen::MatrixXd MultivariateRegressionRandomEffectsModel::GroupParameterVariance(RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& rfx_tracker, 
                                                                                   RandomEffectsTerm& rfx_term, double global_variance, int32_t group_id){
  int32_t num_components = rfx_term.NumComponents();
  int32_t num_groups = rfx_term.NumGroups();
  Eigen::MatrixXd X = dataset.GetBasis();
  Eigen::VectorXd y = residual.GetData();
  Eigen::VectorXd alpha = rfx_term.GetWorkingParameter();
  Eigen::MatrixXd posterior_denominator = rfx_term.GetGroupParameterCovariance().inverse();
  Eigen::VectorXd posterior_numerator = Eigen::VectorXd::Zero(num_components);
  std::vector<data_size_t> observation_indices = rfx_tracker.NodeIndices(group_id);
  Eigen::MatrixXd X_group = X(observation_indices, Eigen::all);
  Eigen::VectorXd y_group = y(observation_indices, Eigen::all);
  posterior_denominator += (alpha).asDiagonal() * X_group.transpose() * X_group * (alpha).asDiagonal();
  posterior_numerator += (alpha).asDiagonal() * X_group.transpose() * y_group;
  return posterior_denominator.inverse();
}

double MultivariateRegressionRandomEffectsModel::VarianceComponentShape(RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& rfx_tracker, 
                                                                        RandomEffectsTerm& rfx_term, double global_variance, int32_t component_id) {
  return static_cast<double>(rfx_term.GetVariancePriorShape() + rfx_term.NumGroups());
}

double MultivariateRegressionRandomEffectsModel::VarianceComponentScale(RandomEffectsDataset& dataset, ColumnVector& residual, RandomEffectsTracker& rfx_tracker, 
                                                                        RandomEffectsTerm& rfx_term, double global_variance, int32_t component_id) {
  int32_t num_groups = rfx_term.NumGroups();
  Eigen::MatrixXd xi = rfx_term.GetGroupParameters();
  double output = rfx_term.GetVariancePriorScale();
  for (int i = 0; i < num_groups; i++) {
    output += xi(component_id, i)*xi(component_id, i);
  }
  return output;
}

}  // namespace StochTree
