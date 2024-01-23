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

GaussianHomoskedasticMultivariateRegressionModelWrapper::GaussianHomoskedasticMultivariateRegressionModelWrapper() {
  basis_dim_ = 1;
  suff_stat_map_.emplace(std::piecewise_construct, std::forward_as_tuple(NodeIndicator::SplitNode), std::forward_as_tuple());
  suff_stat_map_.emplace(std::piecewise_construct, std::forward_as_tuple(NodeIndicator::LeftNode), std::forward_as_tuple());
  suff_stat_map_.emplace(std::piecewise_construct, std::forward_as_tuple(NodeIndicator::RightNode), std::forward_as_tuple());
}

GaussianHomoskedasticMultivariateRegressionModelWrapper::GaussianHomoskedasticMultivariateRegressionModelWrapper(int basis_dim) {
  basis_dim_ = basis_dim;
  suff_stat_map_.emplace(std::piecewise_construct, std::forward_as_tuple(NodeIndicator::SplitNode), std::forward_as_tuple(basis_dim));
  suff_stat_map_.emplace(std::piecewise_construct, std::forward_as_tuple(NodeIndicator::LeftNode), std::forward_as_tuple(basis_dim));
  suff_stat_map_.emplace(std::piecewise_construct, std::forward_as_tuple(NodeIndicator::RightNode), std::forward_as_tuple(basis_dim));
}

GaussianHomoskedasticMultivariateRegressionModelWrapper::~GaussianHomoskedasticMultivariateRegressionModelWrapper() {}

void GaussianHomoskedasticMultivariateRegressionModelWrapper::IncrementNodeSuffStat(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, Eigen::MatrixXd& outcome, data_size_t row_idx, NodeIndicator node_indicator) {
  suff_stat_map_[node_indicator].IncrementSuffStat(covariates, basis, outcome, row_idx);
}

void GaussianHomoskedasticMultivariateRegressionModelWrapper::ResetNodeSuffStat(NodeIndicator node_indicator) {
  suff_stat_map_[node_indicator].ResetSuffStat();
}

void GaussianHomoskedasticMultivariateRegressionModelWrapper::SubtractNodeSuffStat(NodeIndicator node_indicator, NodeIndicator lhs_node_indicator, NodeIndicator rhs_node_indicator) {
  suff_stat_map_[node_indicator].SubtractSuffStat(suff_stat_map_[lhs_node_indicator],suff_stat_map_[rhs_node_indicator]);
}

bool GaussianHomoskedasticMultivariateRegressionModelWrapper::NodeSampleGreaterThan(NodeIndicator node_indicator, data_size_t threshold) {
  return suff_stat_map_[node_indicator].SampleGreaterThan(threshold);
}

data_size_t GaussianHomoskedasticMultivariateRegressionModelWrapper::NodeSampleSize(NodeIndicator node_indicator) {
  return suff_stat_map_[node_indicator].SampleSize();
}

double GaussianHomoskedasticMultivariateRegressionModelWrapper::SplitLogMarginalLikelihood() {
  return outcome_model_.SplitLogMarginalLikelihood(suff_stat_map_[NodeIndicator::LeftNode], suff_stat_map_[NodeIndicator::RightNode], global_model_params_);
}

double GaussianHomoskedasticMultivariateRegressionModelWrapper::NoSplitLogMarginalLikelihood() {
  return outcome_model_.NoSplitLogMarginalLikelihood(suff_stat_map_[NodeIndicator::SplitNode], global_model_params_);
}

void GaussianHomoskedasticMultivariateRegressionModelWrapper::SampleLeafParameters(std::mt19937& gen, int leaf_num, Tree* tree) {
  return outcome_model_.SampleLeafParameters(suff_stat_map_[NodeIndicator::SplitNode], global_model_params_, gen, leaf_num, tree);
}

void GaussianHomoskedasticMultivariateRegressionModelWrapper::SetGlobalParameter(double param_value, GlobalParamName param_name) {
  if (param_name == GlobalParamName::GlobalVariance) {
    global_model_params_.sigma_sq = param_value;
  } else {
    Log::Fatal("Supplied param name %d does not correspond to a parameter in the GaussianHomoskedasticUnivariateRegression model", param_name);
  }
}

double GaussianHomoskedasticMultivariateRegressionOutcomeModel::SplitLogMarginalLikelihood(GaussianHomoskedasticMultivariateRegressionSuffStat& left_stat, GaussianHomoskedasticMultivariateRegressionSuffStat& right_stat, GaussianHomoskedasticMultivariateRegressionGlobalParameters& global_params) {
  // Compute the log marginal likelihood for the left node
  int basis_dim = left_stat.XtX.rows();
  double left_n = static_cast<double>(left_stat.n);
  double left_yty = left_stat.yty;
  Eigen::MatrixXd left_XtX = left_stat.XtX;
  Eigen::MatrixXd left_Xty = left_stat.Xty;
  Eigen::MatrixXd left_inverse_posterior_var = (global_params.Sigma.inverse().array() + (left_XtX/global_params.sigma_sq).array()).inverse();
  double left_log_ml = (
    -(left_n*0.5)*std::log(2*M_PI) - (left_n)*std::log(std::sqrt(global_params.sigma_sq)) - 
    (0.5)*std::log(((Eigen::MatrixXd::Identity(basis_dim, basis_dim).array() + ((global_params.Sigma * left_XtX)/global_params.sigma_sq).array()).matrix()).determinant()) - 
    (left_yty/(2.0*global_params.sigma_sq)) + 
    ((1.0/(2.0*global_params.sigma_sq*global_params.sigma_sq))*(left_Xty.transpose() * left_inverse_posterior_var * left_Xty)(0,0))
  );

  // Compute the log marginal likelihood for the right node
  double right_n = static_cast<double>(right_stat.n);
  double right_yty = right_stat.yty;
  Eigen::MatrixXd right_XtX = right_stat.XtX;
  Eigen::MatrixXd right_Xty = right_stat.Xty;
  Eigen::MatrixXd right_inverse_posterior_var = (global_params.Sigma.inverse().array() + (right_XtX/global_params.sigma_sq).array()).inverse();
  double right_log_ml = (
    -(right_n*0.5)*std::log(2*M_PI) - (right_n)*std::log(std::sqrt(global_params.sigma_sq)) - 
    (0.5)*std::log(((Eigen::MatrixXd::Identity(basis_dim, basis_dim).array() + ((global_params.Sigma * right_XtX)/global_params.sigma_sq).array()).matrix()).determinant()) - 
    (right_yty/(2.0*global_params.sigma_sq)) + 
    ((1.0/(2.0*global_params.sigma_sq*global_params.sigma_sq))*(right_Xty.transpose() * right_inverse_posterior_var * right_Xty)(0,0))
  );

  // Return the combined log marginal likelihood
  return left_log_ml + right_log_ml;
}

double GaussianHomoskedasticMultivariateRegressionOutcomeModel::NoSplitLogMarginalLikelihood(GaussianHomoskedasticMultivariateRegressionSuffStat& suff_stat, GaussianHomoskedasticMultivariateRegressionGlobalParameters& global_params) {
  int basis_dim = suff_stat.XtX.rows();
  double n = static_cast<double>(suff_stat.n);
  double yty = suff_stat.yty;
  Eigen::MatrixXd XtX = suff_stat.XtX;
  Eigen::MatrixXd Xty = suff_stat.Xty;
  Eigen::MatrixXd inverse_posterior_var = (global_params.Sigma.inverse().array() + (suff_stat.XtX/global_params.sigma_sq).array()).inverse();
  double log_ml = (
    -(n*0.5)*std::log(2*M_PI) - (n)*std::log(std::sqrt(global_params.sigma_sq)) - 
    (0.5)*std::log(((Eigen::MatrixXd::Identity(basis_dim, basis_dim).array() + ((global_params.Sigma * XtX)/global_params.sigma_sq).array()).matrix()).determinant()) - 
    (suff_stat.yty/(2.0*global_params.sigma_sq)) + 
    ((1.0/(2.0*global_params.sigma_sq*global_params.sigma_sq))*(Xty.transpose() * inverse_posterior_var * Xty)(0,0))
  );

  return log_ml;
}

Eigen::MatrixXd GaussianHomoskedasticMultivariateRegressionOutcomeModel::PosteriorParameterMean(GaussianHomoskedasticMultivariateRegressionSuffStat& suff_stat, GaussianHomoskedasticMultivariateRegressionGlobalParameters& global_params) {
  Eigen::MatrixXd inverse_posterior_var = (global_params.Sigma.inverse().array() + (suff_stat.XtX/global_params.sigma_sq).array()).inverse();
  Eigen::MatrixXd result = inverse_posterior_var * (suff_stat.Xty / global_params.sigma_sq);
  return result;
}

Eigen::MatrixXd GaussianHomoskedasticMultivariateRegressionOutcomeModel::PosteriorParameterVariance(GaussianHomoskedasticMultivariateRegressionSuffStat& suff_stat, GaussianHomoskedasticMultivariateRegressionGlobalParameters& global_params) {
  Eigen::MatrixXd result = global_params.Sigma.inverse().array() + (suff_stat.XtX/global_params.sigma_sq).array();
  return result;
}

void GaussianHomoskedasticMultivariateRegressionOutcomeModel::SampleLeafParameters(GaussianHomoskedasticMultivariateRegressionSuffStat& suff_stat, GaussianHomoskedasticMultivariateRegressionGlobalParameters& global_params, std::mt19937& gen, int leaf_num, Tree* tree) {
  // Mean, variance, and variance cholesky decomposition
  Eigen::MatrixXd mu_post = PosteriorParameterMean(suff_stat, global_params);
  Eigen::MatrixXd Sigma_post = PosteriorParameterVariance(suff_stat, global_params);
  Eigen::LLT<Eigen::MatrixXd> decomposition(Sigma_post);
  Eigen::MatrixXd Sigma_post_chol = decomposition.matrixL();

  // Sample a vector of standard normal random variables
  std::normal_distribution<double> leaf_node_dist(0.,1.);
  int leaf_dim = Sigma_post.cols();
  Eigen::MatrixXd std_norm_vec(leaf_dim, 1);
  for (int i = 0; i < leaf_dim; i++) {
    std_norm_vec(i,0) = leaf_node_dist(gen);
  }

  // Generate the leaf parameters
  Eigen::MatrixXd leaf_values_raw = mu_post + Sigma_post_chol * std_norm_vec;
  std::vector<double> result(leaf_dim);
  for (int i = 0; i < leaf_dim; i++) {
    result[i] = leaf_values_raw(i, 0);
  }
  tree->SetLeafVector(leaf_num, result);
}

} // namespace StochTree
