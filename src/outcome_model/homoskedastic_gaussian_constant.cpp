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

GaussianHomoskedasticConstantModelWrapper::GaussianHomoskedasticConstantModelWrapper() {
  suff_stat_map_.emplace(std::piecewise_construct, std::forward_as_tuple(NodeIndicator::SplitNode), std::forward_as_tuple());
  suff_stat_map_.emplace(std::piecewise_construct, std::forward_as_tuple(NodeIndicator::LeftNode), std::forward_as_tuple());
  suff_stat_map_.emplace(std::piecewise_construct, std::forward_as_tuple(NodeIndicator::RightNode), std::forward_as_tuple());
}

GaussianHomoskedasticConstantModelWrapper::~GaussianHomoskedasticConstantModelWrapper() {}

void GaussianHomoskedasticConstantModelWrapper::IncrementNodeSuffStat(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, Eigen::MatrixXd& outcome, data_size_t row_idx, NodeIndicator node_indicator) {
  suff_stat_map_[node_indicator].IncrementSuffStat(covariates, basis, outcome, row_idx);
}

void GaussianHomoskedasticConstantModelWrapper::ResetNodeSuffStat(NodeIndicator node_indicator) {
  suff_stat_map_[node_indicator].ResetSuffStat();
}

void GaussianHomoskedasticConstantModelWrapper::SubtractNodeSuffStat(NodeIndicator node_indicator, NodeIndicator lhs_node_indicator, NodeIndicator rhs_node_indicator) {
  suff_stat_map_[node_indicator].SubtractSuffStat(suff_stat_map_[lhs_node_indicator],suff_stat_map_[rhs_node_indicator]);
}

bool GaussianHomoskedasticConstantModelWrapper::NodeSampleGreaterThan(NodeIndicator node_indicator, data_size_t threshold) {
  return suff_stat_map_[node_indicator].SampleGreaterThan(threshold);
}

data_size_t GaussianHomoskedasticConstantModelWrapper::NodeSampleSize(NodeIndicator node_indicator) {
  return suff_stat_map_[node_indicator].SampleSize();
}

double GaussianHomoskedasticConstantModelWrapper::SplitLogMarginalLikelihood() {
  return outcome_model_.SplitLogMarginalLikelihood(suff_stat_map_[NodeIndicator::LeftNode], suff_stat_map_[NodeIndicator::RightNode], global_model_params_);
}

double GaussianHomoskedasticConstantModelWrapper::NoSplitLogMarginalLikelihood() {
  return outcome_model_.NoSplitLogMarginalLikelihood(suff_stat_map_[NodeIndicator::SplitNode], global_model_params_);
}

void GaussianHomoskedasticConstantModelWrapper::SampleLeafParameters(std::mt19937& gen, int leaf_num, Tree* tree) {
  return outcome_model_.SampleLeafParameters(suff_stat_map_[NodeIndicator::SplitNode], global_model_params_, gen, leaf_num, tree);
}

void GaussianHomoskedasticConstantModelWrapper::SetGlobalParameter(double param_value, GlobalParamName param_name) {
  if (param_name == GlobalParamName::GlobalVariance) {
    global_model_params_.sigma_sq = param_value;
  } else if (param_name == GlobalParamName::LeafPriorVariance) {
    global_model_params_.tau = param_value;
  } else {
    Log::Fatal("Supplied param name %d does not correspond to a parameter in the GaussianHomoskedasticUnivariateRegression model", param_name);
  }
}

double GaussianHomoskedasticConstantOutcomeModel::SplitLogMarginalLikelihood(GaussianHomoskedasticConstantSuffStat& left_stat, GaussianHomoskedasticConstantSuffStat& right_stat, GaussianHomoskedasticConstantGlobalParameters& global_params) {
  // Compute the log marginal likelihood for the left node
  double left_n = static_cast<double>(left_stat.n);
  double left_sum_y = left_stat.sum_y;
  double left_sum_y_squared = left_stat.sum_y_squared;
  double left_log_ml = (
    -(left_n*0.5)*std::log(2*M_PI) - (left_n)*std::log(std::sqrt(global_params.sigma_sq)) + 
    (0.5)*std::log(global_params.sigma_sq/(global_params.sigma_sq + global_params.tau*left_n)) - (left_sum_y_squared/(2.0*global_params.sigma_sq)) + 
    ((global_params.tau*std::pow(left_sum_y, 2.0))/(2*global_params.sigma_sq*(global_params.sigma_sq + global_params.tau*left_n)))
  );

  // Compute the log marginal likelihood for the right node
  double right_n = static_cast<double>(right_stat.n);
  double right_sum_y = right_stat.sum_y;
  double right_sum_y_squared = right_stat.sum_y_squared;
  double right_log_ml = (
    -(right_n*0.5)*std::log(2*M_PI) - (right_n)*std::log(std::sqrt(global_params.sigma_sq)) + 
    (0.5)*std::log(global_params.sigma_sq/(global_params.sigma_sq + global_params.tau*right_n)) - (right_sum_y_squared/(2.0*global_params.sigma_sq)) + 
    ((global_params.tau*std::pow(right_sum_y, 2.0))/(2*global_params.sigma_sq*(global_params.sigma_sq + global_params.tau*right_n)))
  );

  // Return the combined log marginal likelihood
  return left_log_ml + right_log_ml;
}

double GaussianHomoskedasticConstantOutcomeModel::NoSplitLogMarginalLikelihood(GaussianHomoskedasticConstantSuffStat& suff_stat, GaussianHomoskedasticConstantGlobalParameters& global_params) {
  double n = static_cast<double>(suff_stat.n);
  double sum_y = suff_stat.sum_y;
  double sum_y_squared = suff_stat.sum_y_squared;
  double log_ml = (
    -(n*0.5)*std::log(2*M_PI) - (n)*std::log(std::sqrt(global_params.sigma_sq)) + 
    (0.5)*std::log(global_params.sigma_sq/(global_params.sigma_sq + global_params.tau*n)) - (sum_y_squared/(2.0*global_params.sigma_sq)) + 
    ((global_params.tau*std::pow(sum_y, 2.0))/(2*global_params.sigma_sq*(global_params.sigma_sq + global_params.tau*n)))
  );

  return log_ml;
}

double GaussianHomoskedasticConstantOutcomeModel::PosteriorParameterMean(GaussianHomoskedasticConstantSuffStat& suff_stat, GaussianHomoskedasticConstantGlobalParameters& global_params) {
  double n = static_cast<double>(suff_stat.n);
  double sum_y = suff_stat.sum_y;
  return ((global_params.tau*sum_y)/(global_params.sigma_sq + (global_params.tau*n)));
}

double GaussianHomoskedasticConstantOutcomeModel::PosteriorParameterVariance(GaussianHomoskedasticConstantSuffStat& suff_stat, GaussianHomoskedasticConstantGlobalParameters& global_params) {
  double n = static_cast<double>(suff_stat.n);
  return ((global_params.tau*global_params.sigma_sq)/(global_params.sigma_sq + (global_params.tau*n)));
}

void GaussianHomoskedasticConstantOutcomeModel::SampleLeafParameters(GaussianHomoskedasticConstantSuffStat& suff_stat, GaussianHomoskedasticConstantGlobalParameters& global_params, std::mt19937& gen, int leaf_num, Tree* tree) {
  double mu_post = PosteriorParameterMean(suff_stat, global_params);
  double sigma_sq_post = PosteriorParameterVariance(suff_stat, global_params);
  std::normal_distribution<double> leaf_node_dist(0.,1.);
  double result = mu_post + std::sqrt(sigma_sq_post) * leaf_node_dist(gen);
  tree->SetLeaf(leaf_num, result);
}

} // namespace StochTree
