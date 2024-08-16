#include <stochtree/leaf_model.h>

namespace StochTree {

template<typename SuffStatType>
void AccumulateSuffStatProposed(SuffStatType& node_suff_stat, SuffStatType& left_suff_stat, SuffStatType& right_suff_stat, ForestDataset& dataset, ForestTracker& tracker, 
                                ColumnVector& residual, double global_variance, TreeSplit& split, int tree_num, int leaf_num, int split_feature) {
  // Acquire iterators
  auto node_begin_iter = tracker.UnsortedNodeBeginIterator(tree_num, leaf_num);
  auto node_end_iter = tracker.UnsortedNodeEndIterator(tree_num, leaf_num);

  // Accumulate sufficient statistics
  for (auto i = node_begin_iter; i != node_end_iter; i++) {
    auto idx = *i;
    double feature_value = dataset.CovariateValue(idx, split_feature);
    node_suff_stat.IncrementSuffStat(dataset, residual.GetData(), idx);
    if (split.SplitTrue(feature_value)) {
      left_suff_stat.IncrementSuffStat(dataset, residual.GetData(), idx);
    } else {
      right_suff_stat.IncrementSuffStat(dataset, residual.GetData(), idx);
    }
  }
}

template<typename SuffStatType>
void AccumulateSuffStatExisting(SuffStatType& node_suff_stat, SuffStatType& left_suff_stat, SuffStatType& right_suff_stat, ForestDataset& dataset, ForestTracker& tracker, 
                                ColumnVector& residual, double global_variance, int tree_num, int split_node_id, int left_node_id, int right_node_id) {
  // Acquire iterators
  auto left_node_begin_iter = tracker.UnsortedNodeBeginIterator(tree_num, left_node_id);
  auto left_node_end_iter = tracker.UnsortedNodeEndIterator(tree_num, left_node_id);
  auto right_node_begin_iter = tracker.UnsortedNodeBeginIterator(tree_num, right_node_id);
  auto right_node_end_iter = tracker.UnsortedNodeEndIterator(tree_num, right_node_id);

  // Accumulate sufficient statistics for the left and split nodes
  for (auto i = left_node_begin_iter; i != left_node_end_iter; i++) {
    auto idx = *i;
    left_suff_stat.IncrementSuffStat(dataset, residual.GetData(), idx);
    node_suff_stat.IncrementSuffStat(dataset, residual.GetData(), idx);
  }

  // Accumulate sufficient statistics for the right and split nodes
  for (auto i = right_node_begin_iter; i != right_node_end_iter; i++) {
    auto idx = *i;
    right_suff_stat.IncrementSuffStat(dataset, residual.GetData(), idx);
    node_suff_stat.IncrementSuffStat(dataset, residual.GetData(), idx);
  }
}

template<typename SuffStatType, bool sorted>
void AccumulateSingleNodeSuffStat(SuffStatType& node_suff_stat, ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual, int tree_num, int node_id) {
  // Acquire iterators
  std::vector<data_size_t>::iterator node_begin_iter;
  std::vector<data_size_t>::iterator node_end_iter;
  if (sorted) {
    // Default to the first feature if we're using the presort tracker
    node_begin_iter = tracker.SortedNodeBeginIterator(node_id, 0);
    node_end_iter = tracker.SortedNodeEndIterator(node_id, 0);
  } else {
    node_begin_iter = tracker.UnsortedNodeBeginIterator(tree_num, node_id);
    node_end_iter = tracker.UnsortedNodeEndIterator(tree_num, node_id);
  }
  
  // Accumulate sufficient statistics
  for (auto i = node_begin_iter; i != node_end_iter; i++) {
    auto idx = *i;
    node_suff_stat.IncrementSuffStat(dataset, residual.GetData(), idx);
  }
}

std::tuple<double, double, data_size_t, data_size_t> GaussianConstantLeafModel::EvaluateProposedSplit(ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual, 
                                                                                                      TreeSplit& split, int tree_num, int leaf_num, int split_feature, double global_variance) {
  // Initialize sufficient statistics
  GaussianConstantSuffStat node_suff_stat = GaussianConstantSuffStat();
  GaussianConstantSuffStat left_suff_stat = GaussianConstantSuffStat();
  GaussianConstantSuffStat right_suff_stat = GaussianConstantSuffStat();

  // Accumulate sufficient statistics
  AccumulateSuffStatProposed<GaussianConstantSuffStat>(node_suff_stat, left_suff_stat, right_suff_stat, dataset, tracker, 
                                                       residual, global_variance, split, tree_num, leaf_num, split_feature);
  data_size_t left_n = left_suff_stat.n;
  data_size_t right_n = right_suff_stat.n;

  // Evaluate split
  double split_log_ml = SplitLogMarginalLikelihood(left_suff_stat, right_suff_stat, global_variance);
  double no_split_log_ml = NoSplitLogMarginalLikelihood(node_suff_stat, global_variance);

  return std::tuple<double, double, data_size_t, data_size_t>(split_log_ml, no_split_log_ml, left_n, right_n);
}

std::tuple<double, double, data_size_t, data_size_t> GaussianConstantLeafModel::EvaluateExistingSplit(ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual, double global_variance, 
                                                                                                      int tree_num, int split_node_id, int left_node_id, int right_node_id) {
  // Initialize sufficient statistics
  GaussianConstantSuffStat node_suff_stat = GaussianConstantSuffStat();
  GaussianConstantSuffStat left_suff_stat = GaussianConstantSuffStat();
  GaussianConstantSuffStat right_suff_stat = GaussianConstantSuffStat();

  // Accumulate sufficient statistics
  AccumulateSuffStatExisting<GaussianConstantSuffStat>(node_suff_stat, left_suff_stat, right_suff_stat, dataset, tracker, 
                                                       residual, global_variance, tree_num, split_node_id, left_node_id, right_node_id);
  data_size_t left_n = left_suff_stat.n;
  data_size_t right_n = right_suff_stat.n;

  // Evaluate split
  double split_log_ml = SplitLogMarginalLikelihood(left_suff_stat, right_suff_stat, global_variance);
  double no_split_log_ml = NoSplitLogMarginalLikelihood(node_suff_stat, global_variance);

  return std::tuple<double, double, data_size_t, data_size_t>(split_log_ml, no_split_log_ml, left_n, right_n);
}

void GaussianConstantLeafModel::EvaluateAllPossibleSplits(ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual, TreePrior& tree_prior, double global_variance, int tree_num, int node_id, std::vector<double>& log_cutpoint_evaluations, 
                                                          data_size_t& valid_cutpoint_count, int cutpoint_grid_size, data_size_t node_begin, data_size_t node_end, std::vector<double>& variable_weights, std::vector<FeatureType>& feature_types) {
  // Initialize sufficient statistics
  GaussianConstantSuffStat node_suff_stat = GaussianConstantSuffStat();
  GaussianConstantSuffStat left_suff_stat = GaussianConstantSuffStat();
  GaussianConstantSuffStat right_suff_stat = GaussianConstantSuffStat();

  // Accumulate aggregate sufficient statistic for the node to be split
  AccumulateSingleNodeSuffStat<GaussianConstantSuffStat, false>(node_suff_stat, dataset, tracker, residual, tree_num, node_id);

  // Compute the "no split" log marginal likelihood
  double no_split_log_ml = NoSplitLogMarginalLikelihood(node_suff_stat, global_variance);

  // Unpack data
  Eigen::MatrixXd covariates = dataset.GetCovariates();
  Eigen::VectorXd outcome = residual.GetData();
  Eigen::VectorXd var_weights;
  bool has_weights = dataset.HasVarWeights();
  if (has_weights) var_weights = dataset.GetVarWeights();
  
  // Minimum size of newly created leaf nodes (used to rule out invalid splits)
  int32_t min_samples_in_leaf = tree_prior.GetMinSamplesLeaf();

  // Compute log ML for each possible split
  data_size_t num_cutpoints = 0;
  bool valid_split = false;
  data_size_t node_row_iter;
  data_size_t current_bin_begin, current_bin_size, next_bin_begin;
  data_size_t feature_sort_idx;
  data_size_t row_iter_idx;
  double outcome_val, outcome_val_sq;
  FeatureType feature_type;
  double feature_value = 0.0;
  double cutoff_value = 0.0;
  double log_split_eval = 0.0;
  double split_log_ml;
  int cutpoint_iter = 0;

  // Compute the "step size"
  data_size_t node_size = node_end - node_begin; 
  int step_size;
  if (cutpoint_grid_size <= node_size) step_size = node_size / cutpoint_grid_size;
  else step_size = 1;

  // Compute the number of steps
  int num_steps;
  if (step_size == 1) num_steps = node_size;
  else num_steps = cutpoint_grid_size;
  
  // Evaluate all cutpoints for features included in the sampler
  int bin_offset;
  int idx;
  for (int j = 0; j < covariates.cols(); j++) {

    if (std::abs(variable_weights.at(j)) > kEpsilon) {
      // Reset sufficient statistics
      left_suff_stat.ResetSuffStat();
      right_suff_stat.ResetSuffStat();

      for (int k = 0; k < num_steps-1; k++) {
        bin_offset = k*step_size;
        for (int i = 0; i < step_size; i++) {

          // Accumulate left node sufficient statistics
          idx = tracker.GetSortedNodeSampleTracker()->SortIndex(node_begin + bin_offset + i, j);
          left_suff_stat.IncrementSuffStat(dataset, outcome, idx);
          
          // Compute the corresponding right node sufficient statistics
          right_suff_stat.SubtractSuffStat(node_suff_stat, left_suff_stat);
        }
        split_log_ml = SplitLogMarginalLikelihood(left_suff_stat, right_suff_stat, global_variance);
        log_cutpoint_evaluations[j*(num_steps-1) + k] = split_log_ml;
        cutpoint_iter++;
      }
    }
  }

  // Update valid cutpoint count
  valid_cutpoint_count = cutpoint_iter;
}

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

std::tuple<double, double, data_size_t, data_size_t> GaussianUnivariateRegressionLeafModel::EvaluateProposedSplit(ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual,
                                                                                                      TreeSplit& split, int tree_num, int leaf_num, int split_feature, double global_variance) {
  // Initialize sufficient statistics
  GaussianUnivariateRegressionSuffStat node_suff_stat = GaussianUnivariateRegressionSuffStat();
  GaussianUnivariateRegressionSuffStat left_suff_stat = GaussianUnivariateRegressionSuffStat();
  GaussianUnivariateRegressionSuffStat right_suff_stat = GaussianUnivariateRegressionSuffStat();

  // Accumulate sufficient statistics
  AccumulateSuffStatProposed<GaussianUnivariateRegressionSuffStat>(node_suff_stat, left_suff_stat, right_suff_stat, dataset, tracker,
                                                       residual, global_variance, split, tree_num, leaf_num, split_feature);
  data_size_t left_n = left_suff_stat.n;
  data_size_t right_n = right_suff_stat.n;

  // Evaluate split
  double split_log_ml = SplitLogMarginalLikelihood(left_suff_stat, right_suff_stat, global_variance);
  double no_split_log_ml = NoSplitLogMarginalLikelihood(node_suff_stat, global_variance);

  return std::tuple<double, double, data_size_t, data_size_t>(split_log_ml, no_split_log_ml, left_n, right_n);
}

std::tuple<double, double, data_size_t, data_size_t> GaussianUnivariateRegressionLeafModel::EvaluateExistingSplit(ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual, double global_variance,
                                                                                                      int tree_num, int split_node_id, int left_node_id, int right_node_id) {
  // Initialize sufficient statistics
  GaussianUnivariateRegressionSuffStat node_suff_stat = GaussianUnivariateRegressionSuffStat();
  GaussianUnivariateRegressionSuffStat left_suff_stat = GaussianUnivariateRegressionSuffStat();
  GaussianUnivariateRegressionSuffStat right_suff_stat = GaussianUnivariateRegressionSuffStat();

  // Accumulate sufficient statistics
  AccumulateSuffStatExisting<GaussianUnivariateRegressionSuffStat>(node_suff_stat, left_suff_stat, right_suff_stat, dataset, tracker,
                                                       residual, global_variance, tree_num, split_node_id, left_node_id, right_node_id);
  data_size_t left_n = left_suff_stat.n;
  data_size_t right_n = right_suff_stat.n;

  // Evaluate split
  double split_log_ml = SplitLogMarginalLikelihood(left_suff_stat, right_suff_stat, global_variance);
  double no_split_log_ml = NoSplitLogMarginalLikelihood(node_suff_stat, global_variance);

  return std::tuple<double, double, data_size_t, data_size_t>(split_log_ml, no_split_log_ml, left_n, right_n);
}

void GaussianUnivariateRegressionLeafModel::EvaluateAllPossibleSplits(ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual, TreePrior& tree_prior, double global_variance, int tree_num, int node_id, std::vector<double>& log_cutpoint_evaluations,
                                                                      data_size_t& valid_cutpoint_count, int cutpoint_grid_size, data_size_t node_begin, data_size_t node_end, std::vector<double>& variable_weights, std::vector<FeatureType>& feature_types) {
  // Initialize sufficient statistics
  GaussianUnivariateRegressionSuffStat node_suff_stat = GaussianUnivariateRegressionSuffStat();
  GaussianUnivariateRegressionSuffStat left_suff_stat = GaussianUnivariateRegressionSuffStat();
  GaussianUnivariateRegressionSuffStat right_suff_stat = GaussianUnivariateRegressionSuffStat();

  // Accumulate aggregate sufficient statistic for the node to be split
  AccumulateSingleNodeSuffStat<GaussianUnivariateRegressionSuffStat, false>(node_suff_stat, dataset, tracker, residual, tree_num, node_id);

  // Compute the "no split" log marginal likelihood
  double no_split_log_ml = NoSplitLogMarginalLikelihood(node_suff_stat, global_variance);

  // Unpack data
  Eigen::MatrixXd covariates = dataset.GetCovariates();
  Eigen::VectorXd outcome = residual.GetData();
  Eigen::VectorXd var_weights;
  bool has_weights = dataset.HasVarWeights();
  if (has_weights) var_weights = dataset.GetVarWeights();
  
  // Minimum size of newly created leaf nodes (used to rule out invalid splits)
  int32_t min_samples_in_leaf = tree_prior.GetMinSamplesLeaf();

  // Compute log ML for each possible split
  data_size_t num_cutpoints = 0;
  bool valid_split = false;
  data_size_t node_row_iter;
  data_size_t current_bin_begin, current_bin_size, next_bin_begin;
  data_size_t feature_sort_idx;
  data_size_t row_iter_idx;
  double outcome_val, outcome_val_sq;
  FeatureType feature_type;
  double feature_value = 0.0;
  double cutoff_value = 0.0;
  double log_split_eval = 0.0;
  double split_log_ml;

  // Compute the "step size"
  data_size_t node_size = node_end - node_begin; 
  int step_size;
  if (cutpoint_grid_size <= node_size) step_size = node_size / cutpoint_grid_size;
  else step_size = 1;

  // Compute the number of steps
  int num_steps;
  if (step_size == 1) num_steps = node_size;
  else num_steps = cutpoint_grid_size;
  
  // Evaluate all cutpoints for features included in the sampler
  int bin_offset;
  
  for (int j = 0; j < covariates.cols(); j++) {

    if (std::abs(variable_weights.at(j)) > kEpsilon) {
      // Reset sufficient statistics
      left_suff_stat.ResetSuffStat();
      right_suff_stat.ResetSuffStat();

      for (int k = 0; k < num_steps-1; k++) {
        bin_offset = k*step_size;
        for (int i = 0; i < step_size; i++) {

          // Accumulate left node sufficient statistics
          left_suff_stat.IncrementSuffStat(dataset, outcome, node_begin + bin_offset + i);
          
          // Compute the corresponding right node sufficient statistics
          right_suff_stat.SubtractSuffStat(node_suff_stat, left_suff_stat);
        }
        split_log_ml = SplitLogMarginalLikelihood(left_suff_stat, right_suff_stat, global_variance);
        log_cutpoint_evaluations[j*(num_steps-1) + k] = split_log_ml;
      }
    }
  }

  // Update valid cutpoint count
  valid_cutpoint_count = num_cutpoints;
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

std::tuple<double, double, data_size_t, data_size_t> GaussianMultivariateRegressionLeafModel::EvaluateProposedSplit(ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual,
                                                                                                      TreeSplit& split, int tree_num, int leaf_num, int split_feature, double global_variance) {
  // Initialize sufficient statistics
  int num_basis = dataset.GetBasis().cols();
  GaussianMultivariateRegressionSuffStat node_suff_stat = GaussianMultivariateRegressionSuffStat(num_basis);
  GaussianMultivariateRegressionSuffStat left_suff_stat = GaussianMultivariateRegressionSuffStat(num_basis);
  GaussianMultivariateRegressionSuffStat right_suff_stat = GaussianMultivariateRegressionSuffStat(num_basis);

  // Accumulate sufficient statistics
  AccumulateSuffStatProposed<GaussianMultivariateRegressionSuffStat>(node_suff_stat, left_suff_stat, right_suff_stat, dataset, tracker,
                                                                     residual, global_variance, split, tree_num, leaf_num, split_feature);
  data_size_t left_n = left_suff_stat.n;
  data_size_t right_n = right_suff_stat.n;

  // Evaluate split
  double split_log_ml = SplitLogMarginalLikelihood(left_suff_stat, right_suff_stat, global_variance);
  double no_split_log_ml = NoSplitLogMarginalLikelihood(node_suff_stat, global_variance);

  return std::tuple<double, double, data_size_t, data_size_t>(split_log_ml, no_split_log_ml, left_n, right_n);
}

std::tuple<double, double, data_size_t, data_size_t> GaussianMultivariateRegressionLeafModel::EvaluateExistingSplit(ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual, double global_variance,
                                                                                                      int tree_num, int split_node_id, int left_node_id, int right_node_id) {
  // Initialize sufficient statistics
  int num_basis = dataset.GetBasis().cols();
  GaussianMultivariateRegressionSuffStat node_suff_stat = GaussianMultivariateRegressionSuffStat(num_basis);
  GaussianMultivariateRegressionSuffStat left_suff_stat = GaussianMultivariateRegressionSuffStat(num_basis);
  GaussianMultivariateRegressionSuffStat right_suff_stat = GaussianMultivariateRegressionSuffStat(num_basis);

  // Accumulate sufficient statistics
  AccumulateSuffStatExisting<GaussianMultivariateRegressionSuffStat>(node_suff_stat, left_suff_stat, right_suff_stat, dataset, tracker,
                                                                     residual, global_variance, tree_num, split_node_id, left_node_id, right_node_id);
  data_size_t left_n = left_suff_stat.n;
  data_size_t right_n = right_suff_stat.n;

  // Evaluate split
  double split_log_ml = SplitLogMarginalLikelihood(left_suff_stat, right_suff_stat, global_variance);
  double no_split_log_ml = NoSplitLogMarginalLikelihood(node_suff_stat, global_variance);

  return std::tuple<double, double, data_size_t, data_size_t>(split_log_ml, no_split_log_ml, left_n, right_n);
}

void GaussianMultivariateRegressionLeafModel::EvaluateAllPossibleSplits(ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual, TreePrior& tree_prior, double global_variance, int tree_num, int node_id, std::vector<double>& log_cutpoint_evaluations,
                                                                        data_size_t& valid_cutpoint_count, int cutpoint_grid_size, data_size_t node_begin, data_size_t node_end, std::vector<double>& variable_weights, std::vector<FeatureType>& feature_types) {
  // Initialize sufficient statistics
  int basis_dim = dataset.GetBasis().cols();
  GaussianMultivariateRegressionSuffStat node_suff_stat = GaussianMultivariateRegressionSuffStat(basis_dim);
  GaussianMultivariateRegressionSuffStat left_suff_stat = GaussianMultivariateRegressionSuffStat(basis_dim);
  GaussianMultivariateRegressionSuffStat right_suff_stat = GaussianMultivariateRegressionSuffStat(basis_dim);

  // Accumulate aggregate sufficient statistic for the node to be split
  AccumulateSingleNodeSuffStat<GaussianMultivariateRegressionSuffStat, false>(node_suff_stat, dataset, tracker, residual, tree_num, node_id);

  // Compute the "no split" log marginal likelihood
  double no_split_log_ml = NoSplitLogMarginalLikelihood(node_suff_stat, global_variance);

  // Unpack data
  Eigen::MatrixXd covariates = dataset.GetCovariates();
  Eigen::VectorXd outcome = residual.GetData();
  Eigen::VectorXd var_weights;
  bool has_weights = dataset.HasVarWeights();
  if (has_weights) var_weights = dataset.GetVarWeights();
  
  // Minimum size of newly created leaf nodes (used to rule out invalid splits)
  int32_t min_samples_in_leaf = tree_prior.GetMinSamplesLeaf();

  // Compute log ML for each possible split
  data_size_t num_cutpoints = 0;
  bool valid_split = false;
  data_size_t node_row_iter;
  data_size_t current_bin_begin, current_bin_size, next_bin_begin;
  data_size_t feature_sort_idx;
  data_size_t row_iter_idx;
  double outcome_val, outcome_val_sq;
  FeatureType feature_type;
  double feature_value = 0.0;
  double cutoff_value = 0.0;
  double log_split_eval = 0.0;
  double split_log_ml;

  // Compute the "step size"
  data_size_t node_size = node_end - node_begin; 
  int step_size;
  if (cutpoint_grid_size <= node_size) step_size = node_size / cutpoint_grid_size;
  else step_size = 1;

  // Compute the number of steps
  int num_steps;
  if (step_size == 1) num_steps = node_size;
  else num_steps = cutpoint_grid_size;
  
  // Evaluate all cutpoints for features included in the sampler
  int bin_offset;
  
  for (int j = 0; j < covariates.cols(); j++) {

    if (std::abs(variable_weights.at(j)) > kEpsilon) {
      // Reset sufficient statistics
      left_suff_stat.ResetSuffStat();
      right_suff_stat.ResetSuffStat();

      for (int k = 0; k < num_steps-1; k++) {
        bin_offset = k*step_size;
        for (int i = 0; i < step_size; i++) {

          // Accumulate left node sufficient statistics
          left_suff_stat.IncrementSuffStat(dataset, outcome, node_begin + bin_offset + i);
          
          // Compute the corresponding right node sufficient statistics
          right_suff_stat.SubtractSuffStat(node_suff_stat, left_suff_stat);
        }
        split_log_ml = SplitLogMarginalLikelihood(left_suff_stat, right_suff_stat, global_variance);
        log_cutpoint_evaluations[j*(num_steps-1) + k] = split_log_ml;
      }
    }
  }

  // Update valid cutpoint count
  valid_cutpoint_count = num_cutpoints;
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

} // namespace StochTree
