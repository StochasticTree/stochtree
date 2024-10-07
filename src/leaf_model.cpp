#include <stochtree/leaf_model.h>
#include <boost/math/special_functions/gamma.hpp>

namespace StochTree {

double GaussianConstantLeafModel::SplitLogMarginalLikelihood(GaussianConstantSuffStat& left_stat, GaussianConstantSuffStat& right_stat, double global_variance) {
  double left_log_ml = (
    -0.5*std::log(1 + tau_*(left_stat.sum_w/global_variance)) + ((tau_*left_stat.sum_yw*left_stat.sum_yw)/(2.0*global_variance*(tau_*left_stat.sum_w + global_variance)))
  );

  double right_log_ml = (
    -0.5*std::log(1 + tau_*(right_stat.sum_w/global_variance)) + ((tau_*right_stat.sum_yw*right_stat.sum_yw)/(2.0*global_variance*(tau_*right_stat.sum_w + global_variance)))
  );

  return left_log_ml + right_log_ml;
}

double GaussianConstantLeafModel::NoSplitLogMarginalLikelihood(GaussianConstantSuffStat& suff_stat, double global_variance) {
  double log_ml = (
    -0.5*std::log(1 + tau_*(suff_stat.sum_w/global_variance)) + ((tau_*suff_stat.sum_yw*suff_stat.sum_yw)/(2.0*global_variance*(tau_*suff_stat.sum_w + global_variance)))
  );

  return log_ml;
}

double GaussianConstantLeafModel::PosteriorParameterMean(GaussianConstantSuffStat& suff_stat, double global_variance) {
  return (tau_*suff_stat.sum_yw) / (suff_stat.sum_w*tau_ + global_variance);
}

double GaussianConstantLeafModel::PosteriorParameterVariance(GaussianConstantSuffStat& suff_stat, double global_variance) {
  return (tau_*global_variance) / (suff_stat.sum_w*tau_ + global_variance);
}

void GaussianConstantLeafModel::SampleLeafParameters(ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual, Tree* tree, int tree_num, double global_variance, std::mt19937& gen) {
  // Vector of leaf indices for tree
  std::vector<int32_t> tree_leaves = tree->GetLeaves();
  
  // Initialize sufficient statistics
  GaussianConstantSuffStat node_suff_stat = GaussianConstantSuffStat();

  // Sample each leaf node parameter
  double node_mean;
  double node_variance;
  double node_mu;
  int32_t leaf_id;
  for (int i = 0; i < tree_leaves.size(); i++) {
    // Compute leaf node sufficient statistics
    leaf_id = tree_leaves[i];
    node_suff_stat.ResetSuffStat();
    AccumulateSingleNodeSuffStat<GaussianConstantSuffStat, false>(node_suff_stat, dataset, tracker, residual, tree_num, leaf_id);
    
    // Compute posterior mean and variance
    node_mean = PosteriorParameterMean(node_suff_stat, global_variance);
    node_variance = PosteriorParameterVariance(node_suff_stat, global_variance);
    
    // Draw from N(mean, stddev^2) and set the leaf parameter with each draw
    node_mu = normal_sampler_.Sample(node_mean, node_variance, gen);
    tree->SetLeaf(leaf_id, node_mu);
  }
}

void GaussianConstantLeafModel::SetEnsembleRootPredictedValue(ForestDataset& dataset, TreeEnsemble* ensemble, double root_pred_value) {
  int num_trees = ensemble->NumTrees();
  for (int i = 0; i < num_trees; i++) {
    Tree* tree = ensemble->GetTree(i);
    tree->SetLeaf(0, root_pred_value);
  }
}

double GaussianUnivariateRegressionLeafModel::SplitLogMarginalLikelihood(GaussianUnivariateRegressionSuffStat& left_stat, GaussianUnivariateRegressionSuffStat& right_stat, double global_variance) {
  double left_log_ml = (
    -0.5*std::log(1 + tau_*(left_stat.sum_xxw/global_variance)) + ((tau_*left_stat.sum_yxw*left_stat.sum_yxw)/(2.0*global_variance*(tau_*left_stat.sum_xxw + global_variance)))
  );

  double right_log_ml = (
    -0.5*std::log(1 + tau_*(right_stat.sum_xxw/global_variance)) + ((tau_*right_stat.sum_yxw*right_stat.sum_yxw)/(2.0*global_variance*(tau_*right_stat.sum_xxw + global_variance)))
  );

  return left_log_ml + right_log_ml;
}

double GaussianUnivariateRegressionLeafModel::NoSplitLogMarginalLikelihood(GaussianUnivariateRegressionSuffStat& suff_stat, double global_variance) {
  double log_ml = (
    -0.5*std::log(1 + tau_*(suff_stat.sum_xxw/global_variance)) + ((tau_*suff_stat.sum_yxw*suff_stat.sum_yxw)/(2.0*global_variance*(tau_*suff_stat.sum_xxw + global_variance)))
  );

  return log_ml;
}

double GaussianUnivariateRegressionLeafModel::PosteriorParameterMean(GaussianUnivariateRegressionSuffStat& suff_stat, double global_variance) {
  return (tau_*suff_stat.sum_yxw) / (suff_stat.sum_xxw*tau_ + global_variance);
}

double GaussianUnivariateRegressionLeafModel::PosteriorParameterVariance(GaussianUnivariateRegressionSuffStat& suff_stat, double global_variance) {
  return (tau_*global_variance) / (suff_stat.sum_xxw*tau_ + global_variance);
}

void GaussianUnivariateRegressionLeafModel::SampleLeafParameters(ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual, Tree* tree, int tree_num, double global_variance, std::mt19937& gen) {
  // Vector of leaf indices for tree
  std::vector<int32_t> tree_leaves = tree->GetLeaves();
  
  // Initialize sufficient statistics
  GaussianUnivariateRegressionSuffStat node_suff_stat = GaussianUnivariateRegressionSuffStat();

  // Sample each leaf node parameter
  double node_mean;
  double node_variance;
  double node_mu;
  int32_t leaf_id;
  for (int i = 0; i < tree_leaves.size(); i++) {
    // Compute leaf node sufficient statistics
    leaf_id = tree_leaves[i];
    node_suff_stat.ResetSuffStat();
    AccumulateSingleNodeSuffStat<GaussianUnivariateRegressionSuffStat, false>(node_suff_stat, dataset, tracker, residual, tree_num, leaf_id);
    
    // Compute posterior mean and variance
    node_mean = PosteriorParameterMean(node_suff_stat, global_variance);
    node_variance = PosteriorParameterVariance(node_suff_stat, global_variance);
    
    // Draw from N(mean, stddev^2) and set the leaf parameter with each draw
    node_mu = normal_sampler_.Sample(node_mean, node_variance, gen);
    tree->SetLeaf(leaf_id, node_mu);
  }
}

void GaussianUnivariateRegressionLeafModel::SetEnsembleRootPredictedValue(ForestDataset& dataset, TreeEnsemble* ensemble, double root_pred_value) {
  int num_trees = ensemble->NumTrees();
  for (int i = 0; i < num_trees; i++) {
    Tree* tree = ensemble->GetTree(i);
    tree->SetLeaf(0, root_pred_value);
  }
}

double GaussianMultivariateRegressionLeafModel::SplitLogMarginalLikelihood(GaussianMultivariateRegressionSuffStat& left_stat, GaussianMultivariateRegressionSuffStat& right_stat, double global_variance) {
  Eigen::MatrixXd I_p = Eigen::MatrixXd::Identity(left_stat.p, left_stat.p);
  double left_log_ml = (
    -0.5*std::log((I_p + (Sigma_0_ * left_stat.XtWX)/global_variance).determinant()) + 0.5*((left_stat.ytWX/global_variance) * (Sigma_0_.inverse() + (left_stat.XtWX/global_variance)).inverse() * (left_stat.ytWX/global_variance).transpose()).value()
  );

  double right_log_ml = (
    -0.5*std::log((I_p + (Sigma_0_ * right_stat.XtWX)/global_variance).determinant()) + 0.5*((right_stat.ytWX/global_variance) * (Sigma_0_.inverse() + (right_stat.XtWX/global_variance)).inverse() * (right_stat.ytWX/global_variance).transpose()).value()
  );

  return left_log_ml + right_log_ml;
}

double GaussianMultivariateRegressionLeafModel::NoSplitLogMarginalLikelihood(GaussianMultivariateRegressionSuffStat& suff_stat, double global_variance) {
  Eigen::MatrixXd I_p = Eigen::MatrixXd::Identity(suff_stat.p, suff_stat.p);
  double log_ml = (
    -0.5*std::log((I_p + (Sigma_0_ * suff_stat.XtWX)/global_variance).determinant()) + 0.5*((suff_stat.ytWX/global_variance) * (Sigma_0_.inverse() + (suff_stat.XtWX/global_variance)).inverse() * (suff_stat.ytWX/global_variance).transpose()).value()
  );

  return log_ml;
}

Eigen::VectorXd GaussianMultivariateRegressionLeafModel::PosteriorParameterMean(GaussianMultivariateRegressionSuffStat& suff_stat, double global_variance) {
  return (Sigma_0_.inverse() + (suff_stat.XtWX/global_variance)).inverse() * (suff_stat.ytWX/global_variance).transpose();
}

Eigen::MatrixXd GaussianMultivariateRegressionLeafModel::PosteriorParameterVariance(GaussianMultivariateRegressionSuffStat& suff_stat, double global_variance) {
  return (Sigma_0_.inverse() + (suff_stat.XtWX/global_variance)).inverse();
}

void GaussianMultivariateRegressionLeafModel::SampleLeafParameters(ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual, Tree* tree, int tree_num, double global_variance, std::mt19937& gen) {
  // Vector of leaf indices for tree
  std::vector<int32_t> tree_leaves = tree->GetLeaves();
  
  // Initialize sufficient statistics
  int num_basis = dataset.GetBasis().cols();
  GaussianMultivariateRegressionSuffStat node_suff_stat = GaussianMultivariateRegressionSuffStat(num_basis);

  // Sample each leaf node parameter
  Eigen::VectorXd node_mean;
  Eigen::MatrixXd node_variance;
  std::vector<double> node_mu;
  int32_t leaf_id;
  for (int i = 0; i < tree_leaves.size(); i++) {
    // Compute leaf node sufficient statistics
    leaf_id = tree_leaves[i];
    node_suff_stat.ResetSuffStat();
    AccumulateSingleNodeSuffStat<GaussianMultivariateRegressionSuffStat, false>(node_suff_stat, dataset, tracker, residual, tree_num, leaf_id);
    
    // Compute posterior mean and variance
    node_mean = PosteriorParameterMean(node_suff_stat, global_variance);
    node_variance = PosteriorParameterVariance(node_suff_stat, global_variance);
    
    // Draw from N(mean, stddev^2) and set the leaf parameter with each draw
    node_mu = multivariate_normal_sampler_.Sample(node_mean, node_variance, gen);
    tree->SetLeafVector(leaf_id, node_mu);
  }
}

void GaussianMultivariateRegressionLeafModel::SetEnsembleRootPredictedValue(ForestDataset& dataset, TreeEnsemble* ensemble, double root_pred_value) {
  int num_trees = ensemble->NumTrees();
  int num_basis = dataset.GetBasis().cols();
  
  // Check that root predicted value is close to 0
  // TODO: formalize and document this
  if ((root_pred_value < -0.1) || root_pred_value > 0.1) {
    Log::Fatal("For multivariate leaf regression, outcomes should be centered / scaled so that the root coefficients can be initialized to 0");
  }

  std::vector<double> root_pred_vector(ensemble->OutputDimension(), root_pred_value);
  for (int i = 0; i < num_trees; i++) {
    Tree* tree = ensemble->GetTree(i);
    tree->SetLeafVector(0, root_pred_vector);
  }
}

double LogLinearVarianceLeafModel::SplitLogMarginalLikelihood(LogLinearVarianceSuffStat& left_stat, LogLinearVarianceSuffStat& right_stat, double global_variance) {
  double left_log_ml = SuffStatLogMarginalLikelihood(left_stat, global_variance);
  double right_log_ml = SuffStatLogMarginalLikelihood(right_stat, global_variance);
  return left_log_ml + right_log_ml;
}

double LogLinearVarianceLeafModel::NoSplitLogMarginalLikelihood(LogLinearVarianceSuffStat& suff_stat, double global_variance) {
  return SuffStatLogMarginalLikelihood(suff_stat, global_variance);
}

double LogLinearVarianceLeafModel::SuffStatLogMarginalLikelihood(LogLinearVarianceSuffStat& suff_stat, double global_variance) {
  double prior_terms = a_ * std::log(b_) - boost::math::lgamma(a_);
  double a_term = a_ + 0.5 * suff_stat.n;
  double b_term = b_ + ((0.5 * suff_stat.weighted_sum_ei) / global_variance);
  double log_b_term = std::log(b_term);
  double lgamma_a_term = boost::math::lgamma(a_term);
  double resid_term = a_term * log_b_term;
  double log_ml = prior_terms + lgamma_a_term - resid_term;
  return log_ml;
}

double LogLinearVarianceLeafModel::PosteriorParameterShape(LogLinearVarianceSuffStat& suff_stat, double global_variance) {
  return a_ + 0.5 * suff_stat.n;
}

double LogLinearVarianceLeafModel::PosteriorParameterScale(LogLinearVarianceSuffStat& suff_stat, double global_variance) {
  return (b_ + (0.5 * suff_stat.weighted_sum_ei) / global_variance);
}

void LogLinearVarianceLeafModel::SampleLeafParameters(ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual, Tree* tree, int tree_num, double global_variance, std::mt19937& gen) {
  // Vector of leaf indices for tree
  std::vector<int32_t> tree_leaves = tree->GetLeaves();
  
  // Initialize sufficient statistics
  LogLinearVarianceSuffStat node_suff_stat = LogLinearVarianceSuffStat();

  // Sample each leaf node parameter
  double node_shape;
  double node_rate;
  double node_mu;
  int32_t leaf_id;
  for (int i = 0; i < tree_leaves.size(); i++) {
    // Compute leaf node sufficient statistics
    leaf_id = tree_leaves[i];
    node_suff_stat.ResetSuffStat();
    AccumulateSingleNodeSuffStat<LogLinearVarianceSuffStat, false>(node_suff_stat, dataset, tracker, residual, tree_num, leaf_id);
    
    // Compute posterior mean and variance
    node_shape = PosteriorParameterShape(node_suff_stat, global_variance);
    node_rate = PosteriorParameterScale(node_suff_stat, global_variance);
    
    // Draw from IG(shape, scale) and set the leaf parameter with each draw
    std::gamma_distribution<double> gamma_dist_(node_shape, 1.);
    node_mu = -std::log(gamma_dist_(gen) / node_rate);
    // node_mu = std::log(gamma_sampler_.Sample(node_shape, node_rate, gen, true));
    tree->SetLeaf(leaf_id, node_mu);
  }
}

void LogLinearVarianceLeafModel::SetEnsembleRootPredictedValue(ForestDataset& dataset, TreeEnsemble* ensemble, double root_pred_value) {
  int num_trees = ensemble->NumTrees();
  for (int i = 0; i < num_trees; i++) {
    Tree* tree = ensemble->GetTree(i);
    tree->SetLeaf(0, root_pred_value);
  }
}

} // namespace StochTree
