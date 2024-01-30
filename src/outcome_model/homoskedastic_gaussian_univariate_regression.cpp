/*! Copyright (c) 2023 by randtree authors. */
#include <stochtree/cutpoint_candidates.h>
#include <stochtree/meta.h>
#include <stochtree/outcome_model.h>
#include <stochtree/sampler.h>
#include <cmath>
#include <iterator>
#include <numbers>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

namespace StochTree {

GaussianHomoskedasticUnivariateRegressionModelWrapper::GaussianHomoskedasticUnivariateRegressionModelWrapper() {
  suff_stat_map_.emplace(std::piecewise_construct, std::forward_as_tuple(NodeIndicator::SplitNode), std::forward_as_tuple());
  suff_stat_map_.emplace(std::piecewise_construct, std::forward_as_tuple(NodeIndicator::LeftNode), std::forward_as_tuple());
  suff_stat_map_.emplace(std::piecewise_construct, std::forward_as_tuple(NodeIndicator::RightNode), std::forward_as_tuple());
}

GaussianHomoskedasticUnivariateRegressionModelWrapper::~GaussianHomoskedasticUnivariateRegressionModelWrapper() {}

void GaussianHomoskedasticUnivariateRegressionModelWrapper::IncrementNodeSuffStat(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, Eigen::MatrixXd& outcome, data_size_t row_idx, NodeIndicator node_indicator) {
  suff_stat_map_[node_indicator].IncrementSuffStat(covariates, basis, outcome, row_idx);
}

void GaussianHomoskedasticUnivariateRegressionModelWrapper::ResetNodeSuffStat(NodeIndicator node_indicator) {
  suff_stat_map_[node_indicator].ResetSuffStat();
}

void GaussianHomoskedasticUnivariateRegressionModelWrapper::SubtractNodeSuffStat(NodeIndicator node_indicator, NodeIndicator lhs_node_indicator, NodeIndicator rhs_node_indicator) {
  suff_stat_map_[node_indicator].SubtractSuffStat(suff_stat_map_[lhs_node_indicator],suff_stat_map_[rhs_node_indicator]);
}

bool GaussianHomoskedasticUnivariateRegressionModelWrapper::NodeSampleGreaterThan(NodeIndicator node_indicator, data_size_t threshold) {
  return suff_stat_map_[node_indicator].SampleGreaterThan(threshold);
}

data_size_t GaussianHomoskedasticUnivariateRegressionModelWrapper::NodeSampleSize(NodeIndicator node_indicator) {
  return suff_stat_map_[node_indicator].SampleSize();
}

double GaussianHomoskedasticUnivariateRegressionModelWrapper::SplitLogMarginalLikelihood() {
  return outcome_model_.SplitLogMarginalLikelihood(suff_stat_map_[NodeIndicator::LeftNode], suff_stat_map_[NodeIndicator::RightNode], global_model_params_);
}

double GaussianHomoskedasticUnivariateRegressionModelWrapper::NoSplitLogMarginalLikelihood() {
  return outcome_model_.NoSplitLogMarginalLikelihood(suff_stat_map_[NodeIndicator::SplitNode], global_model_params_);
}

void GaussianHomoskedasticUnivariateRegressionModelWrapper::SampleLeafParameters(std::mt19937& gen, int leaf_num, Tree* tree) {
  return outcome_model_.SampleLeafParameters(suff_stat_map_[NodeIndicator::SplitNode], global_model_params_, gen, leaf_num, tree);
}

void GaussianHomoskedasticUnivariateRegressionModelWrapper::SetGlobalParameter(double param_value, GlobalParamName param_name) {
  if (param_name == GlobalParamName::GlobalVariance) {
    global_model_params_.sigma_sq = param_value;
  } else if (param_name == GlobalParamName::LeafPriorVariance) {
    global_model_params_.tau = param_value;
  } else {
    Log::Fatal("Supplied param name %d does not correspond to a parameter in the GaussianHomoskedasticUnivariateRegression model", param_name);
  }
}

double GaussianHomoskedasticUnivariateRegressionOutcomeModel::SplitLogMarginalLikelihood(GaussianHomoskedasticUnivariateRegressionSuffStat& left_stat, GaussianHomoskedasticUnivariateRegressionSuffStat& right_stat, GaussianHomoskedasticUnivariateRegressionGlobalParameters& global_params) {
  // Compute the log marginal likelihood for the left node
  double left_n = static_cast<double>(left_stat.n);
  double left_sum_yx = left_stat.sum_yx;
  double left_sum_x_squared = left_stat.sum_x_squared;
  double left_sum_y_squared = left_stat.sum_y_squared;
  double left_log_ml = (
    -(left_n*0.5)*std::log(2*M_PI) - (left_n)*std::log(std::sqrt(global_params.sigma_sq)) + 
    (0.5)*std::log(global_params.sigma_sq/(global_params.sigma_sq + global_params.tau*left_sum_x_squared)) - (left_sum_y_squared/(2.0*global_params.sigma_sq)) + 
    ((global_params.tau*std::pow(left_sum_yx, 2.0))/(2*global_params.sigma_sq*(global_params.sigma_sq + global_params.tau*left_sum_x_squared)))
  );

  // Compute the log marginal likelihood for the right node
  double right_n = static_cast<double>(right_stat.n);
  double right_sum_yx = right_stat.sum_yx;
  double right_sum_x_squared = right_stat.sum_x_squared;
  double right_sum_y_squared = right_stat.sum_y_squared;
  double right_log_ml = (
    -(right_n*0.5)*std::log(2*M_PI) - (right_n)*std::log(std::sqrt(global_params.sigma_sq)) + 
    (0.5)*std::log(global_params.sigma_sq/(global_params.sigma_sq + global_params.tau*right_sum_x_squared)) - (right_sum_y_squared/(2.0*global_params.sigma_sq)) + 
    ((global_params.tau*std::pow(right_sum_yx, 2.0))/(2*global_params.sigma_sq*(global_params.sigma_sq + global_params.tau*right_sum_x_squared)))
  );

  // Return the combined log marginal likelihood
  return left_log_ml + right_log_ml;
}

double GaussianHomoskedasticUnivariateRegressionOutcomeModel::NoSplitLogMarginalLikelihood(GaussianHomoskedasticUnivariateRegressionSuffStat& suff_stat, GaussianHomoskedasticUnivariateRegressionGlobalParameters& global_params) {
  double n = static_cast<double>(suff_stat.n);
  double sum_yx = suff_stat.sum_yx;
  double sum_x_squared = suff_stat.sum_x_squared;
  double sum_y_squared = suff_stat.sum_y_squared;
  double log_ml = (
    -(n*0.5)*std::log(2*M_PI) - (n)*std::log(std::sqrt(global_params.sigma_sq)) + 
    (0.5)*std::log(global_params.sigma_sq/(global_params.sigma_sq + global_params.tau*sum_x_squared)) - (sum_y_squared/(2.0*global_params.sigma_sq)) + 
    ((global_params.tau*std::pow(sum_yx, 2.0))/(2*global_params.sigma_sq*(global_params.sigma_sq + global_params.tau*sum_x_squared)))
  );

  return log_ml;
}

double GaussianHomoskedasticUnivariateRegressionOutcomeModel::PosteriorParameterMean(GaussianHomoskedasticUnivariateRegressionSuffStat& suff_stat, GaussianHomoskedasticUnivariateRegressionGlobalParameters& global_params) {
  double sum_yx = suff_stat.sum_yx;
  double sum_x_squared = suff_stat.sum_x_squared;
  return ((global_params.tau*sum_yx)/(global_params.sigma_sq + (global_params.tau*sum_x_squared)));
}

double GaussianHomoskedasticUnivariateRegressionOutcomeModel::PosteriorParameterVariance(GaussianHomoskedasticUnivariateRegressionSuffStat& suff_stat, GaussianHomoskedasticUnivariateRegressionGlobalParameters& global_params) {
  double sum_x_squared = suff_stat.sum_x_squared;
  return ((global_params.tau*global_params.sigma_sq)/(global_params.sigma_sq + (global_params.tau*sum_x_squared)));
}

void GaussianHomoskedasticUnivariateRegressionOutcomeModel::SampleLeafParameters(GaussianHomoskedasticUnivariateRegressionSuffStat& suff_stat, GaussianHomoskedasticUnivariateRegressionGlobalParameters& global_params, std::mt19937& gen, int leaf_num, Tree* tree) {
  double mu_post = PosteriorParameterMean(suff_stat, global_params);
  double sigma_sq_post = PosteriorParameterVariance(suff_stat, global_params);
  std::normal_distribution<double> leaf_node_dist(0.,1.);
  double result = mu_post + std::sqrt(sigma_sq_post) * leaf_node_dist(gen);
  tree->SetLeaf(leaf_num, result);
}

} // namespace StochTree
