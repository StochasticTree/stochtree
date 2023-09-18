/*!
 * Copyright (c) 2023 by randtree authors. 
 */
#include <stochtree/meta.h>
#include <stochtree/model.h>
#include <stochtree/train_data.h>
#include <cmath>

namespace StochTree {

XBARTGaussianRegressionModel::XBARTGaussianRegressionModel() {
  config_ = Config();
  alpha_ = config_.alpha;
  beta_ = config_.beta;
  a_sigma_ = config_.a_sigma;
  a_tau_ = config_.a_tau;
  b_sigma_ = config_.b_sigma;
  b_tau_ = config_.b_tau;
  sigma_sq_ = 1;
  tau_ = 1;
  if (config_.random_seed < 0) {
    std::random_device rd;
    std::mt19937 gen(rd());
  } else {
    std::mt19937 gen(config_.random_seed);
  }
}

XBARTGaussianRegressionModel::XBARTGaussianRegressionModel(const Config& config) {
  config_ = config;
  alpha_ = config_.alpha;
  beta_ = config_.beta;
  a_sigma_ = config_.a_sigma;
  a_tau_ = config_.a_tau;
  b_sigma_ = config_.b_sigma;
  b_tau_ = config_.b_tau;
  sigma_sq_ = 1;
  tau_ = 1;
  if (config_.random_seed < 0) {
    std::random_device rd;
    std::mt19937 gen(rd());
  } else {
    std::mt19937 gen(config_.random_seed);
  }
}

void XBARTGaussianRegressionModel::InitializeGlobalParameters(TrainData* train_data) {
  if (config_.data_driven_prior) {
    double var_y = 0.0;
    double outcome_sum_squares = 0.0;
    double outcome_sum = 0.0;
    double outcome_val;
    data_size_t n = train_data->num_data();
    int num_trees = config_.num_trees;
    for (data_size_t i = 0; i < n; i++){
      outcome_val = train_data->get_outcome_value(i);
      outcome_sum += outcome_val;
      outcome_sum_squares += std::pow(outcome_val, 2.0);
    }
    var_y = outcome_sum_squares/n - std::pow(outcome_sum / n, 2.0);
    b_tau_ = 0.5 * var_y / num_trees;
    tau_ = var_y / num_trees;
  }
}

void XBARTGaussianRegressionModel::SampleTree(TrainData* train_data, Tree* tree, std::vector<std::vector<data_size_t>>& tree_observation_indices, int tree_num) {
  node_t root_id = Tree::kRoot;
  node_t curr_node_id;
  data_size_t curr_node_begin;
  data_size_t curr_node_end;
  data_size_t n = train_data->num_data();
  // Reset the mapping from node id to start and end points of sorted indices
  NodeIndexMapReset(n);
  std::pair<data_size_t, data_size_t> begin_end;
  // Add root node to the split queue
  std::deque<node_t> split_queue_;
  split_queue_.push_back(Tree::kRoot);
  // Run the "GrowFromRoot" procedure using a stack in place of recursion
  while (!split_queue_.empty()) {
    // Remove the next node from the queue
    curr_node_id = split_queue_.front();
    split_queue_.pop_front();
    // Determine the beginning and ending indices of the left and right nodes
    begin_end = node_index_map_[curr_node_id];
    curr_node_begin = begin_end.first;
    curr_node_end = begin_end.second;
    // Draw a split rule at random
    SampleSplitRule(train_data, tree, curr_node_id, curr_node_begin, curr_node_end, split_queue_, tree_observation_indices, tree_num);
  }
}

void XBARTGaussianRegressionModel::SampleLeafParameters(TrainData* train_data, Tree* tree) {
  // Vector of leaf indices for tree
  std::vector<node_t> tree_leaves = tree->GetLeaves();
  std::vector<XBARTGaussianRegressionSuffStat> leaf_suff_stats;
  std::normal_distribution<double> leaf_node_dist(0.,1.);
  // Vector of sufficient statistics for each leaf
  for (int i = 0; i < tree_leaves.size(); i++) {
    leaf_suff_stats.push_back(LeafSuffStat(train_data, tree_leaves[i]));
  }

  // Sample each leaf node parameter
  double node_mean;
  double node_stddev;
  double node_mu;
  for (int i = 0; i < tree_leaves.size(); i++) {
    // Compute posterior mean and variance
    node_mean = LeafPosteriorMean(leaf_suff_stats[i]);
    node_stddev = LeafPosteriorStddev(leaf_suff_stats[i]);
    
    // Draw from N(mean, stddev^2) and set the leaf parameter with each draw
    node_mu = leaf_node_dist(gen)*node_stddev + node_mean;
    (*tree)[tree_leaves[i]].SetLeaf(node_mu);
  }
}

XBARTGaussianRegressionSuffStat XBARTGaussianRegressionModel::LeafSuffStat(TrainData* train_data, node_t leaf_id) {
  data_size_t node_begin, node_end;
  std::pair<data_size_t, data_size_t> begin_end;
  if (node_index_map_.find(leaf_id) == node_index_map_.end()) {
    Log::Fatal("Leaf id %d is not present in the tree", leaf_id);
  }
  begin_end = node_index_map_[leaf_id];
  node_begin = begin_end.first;
  node_end = begin_end.second;
  return ComputeNodeSuffStat(train_data, node_begin, node_end, 0);
}

void XBARTGaussianRegressionModel::SampleGlobalParameters(TrainData* train_data, TreeEnsemble* tree_ensemble, std::set<std::string> update_params) {
  // Update sigma^2
  if (update_params.count("sigma_sq") > 0) {
    // Compute posterior shape and scale parameters for inverse gamma
    double ig_shape_sig = SigmaPosteriorShape(train_data);
    double ig_scale_sig = SigmaPosteriorScale(train_data);
    
    // C++ standard library provides a gamma distribution with scale
    // parameter, but the correspondence between gamma and IG is that 
    // 1 / gamma(a,b) ~ IG(a,b) when b is a __rate__ parameter.
    // Before sampling, we convert ig_scale to a gamma scale parameter by 
    // taking its multiplicative inverse.
    double gamma_scale_sig = 1./ig_scale_sig;
    residual_variance_dist_.param(
      std::gamma_distribution<double>::param_type(ig_shape_sig, gamma_scale_sig)
    );
    sigma_sq_ = (1/residual_variance_dist_(gen));
  }
  
  // Update tau
  if (update_params.count("tau") > 0) {
    // Compute posterior shape and scale parameters for inverse gamma
    double ig_shape_tau = TauPosteriorShape(tree_ensemble);
    double ig_scale_tau = TauPosteriorScale(tree_ensemble);

    double gamma_scale_tau = 1./ig_scale_tau;
    leaf_node_variance_dist_.param(
      std::gamma_distribution<double>::param_type(ig_shape_tau, gamma_scale_tau)
    );
    tau_ = (1/leaf_node_variance_dist_(gen));
  }
}

void XBARTGaussianRegressionModel::SampleSplitRule(TrainData* train_data, Tree* tree, node_t leaf_node, 
                      data_size_t node_begin, data_size_t node_end, std::deque<node_t>& split_queue, 
                      std::vector<std::vector<data_size_t>>& tree_observation_indices, int tree_num) {
  std::vector<double> log_cutpoint_evaluations;
  std::vector<int> cutpoint_features;
  std::vector<double> cutpoint_values;
  StochTree::data_size_t valid_cutpoint_count;
  Cutpoints(train_data, tree, leaf_node, node_begin, node_end, 
            log_cutpoint_evaluations, cutpoint_features, cutpoint_values, 
            valid_cutpoint_count);
  
  // Convert log marginal likelihood to marginal likelihood, normalizing by the maximum log-likelihood
  double largest_mll = *std::max_element(log_cutpoint_evaluations.begin(), log_cutpoint_evaluations.end());
  std::vector<double> cutpoint_evaluations(log_cutpoint_evaluations.size());
  for (data_size_t i = 0; i < log_cutpoint_evaluations.size(); i++){
    cutpoint_evaluations[i] = std::exp(log_cutpoint_evaluations[i] - largest_mll);
  }
  
  // Sample the split (including a "no split" option)
  std::discrete_distribution<data_size_t> split_dist(cutpoint_evaluations.begin(), cutpoint_evaluations.end());
  data_size_t split_chosen = split_dist(gen);
  if (split_chosen == valid_cutpoint_count){
    // "No split" sampled, don't split or add any nodes to split queue
    return;
  } else {
    // Split sampled
    int feature_split = cutpoint_features[split_chosen];
    double split_value = cutpoint_values[split_chosen];
    // Perform all of the relevant "split" operations in the model, tree and training dataset
    AddSplitToModel(train_data, tree, leaf_node, node_begin, node_end, 
                    feature_split, split_value, split_queue, tree_observation_indices, tree_num);
  }
}

void XBARTGaussianRegressionModel::Cutpoints(TrainData* train_data, Tree* tree, node_t leaf_node, 
                                             data_size_t node_begin, data_size_t node_end, 
                                             std::vector<double>& log_cutpoint_evaluations, 
                                             std::vector<int>& cutpoint_feature, 
                                             std::vector<double>& cutpoint_values, 
                                             data_size_t& valid_cutpoint_count) {
  // Compute sufficient statistics for the current node
  XBARTGaussianRegressionSuffStat node_suff_stat_ = ComputeNodeSuffStat(train_data, node_begin, node_end, 0);
  XBARTGaussianRegressionSuffStat left_suff_stat_;
  XBARTGaussianRegressionSuffStat right_suff_stat_;

  // Clear vectors
  log_cutpoint_evaluations.clear();
  cutpoint_feature.clear();
  cutpoint_values.clear();
  
  // Compute sufficient statistics for each possible split
  data_size_t num_cutpoints = 0;
  bool valid_split = false;
  data_size_t node_row_iter;
  double feature_value = 0.0;
  double log_split_eval = 0.0;
  for (int j = 0; j < train_data->num_variables(); j++) {
    node_row_iter = node_begin;
    ResetSuffStat(left_suff_stat_, 0, 0.0);
    ResetSuffStat(right_suff_stat_, node_suff_stat_.sample_size_, node_suff_stat_.outcome_sum_);
    while (node_row_iter < node_end) {
      feature_value = train_data->get_feature_value(train_data->get_feature_sort_index(node_row_iter, j), j);
      AccumulateRowSuffStat(train_data, left_suff_stat_, node_row_iter, j, node_row_iter);
      right_suff_stat_ = SubtractSuffStat(node_suff_stat_, left_suff_stat_);
      
      // Only include cutpoint for consideration if it defines a valid split in the training data
      valid_split = ((left_suff_stat_.sample_size_ >= config_.min_data_in_leaf) && 
                      (right_suff_stat_.sample_size_ >= config_.min_data_in_leaf));
      if (valid_split) {
        num_cutpoints++;
        // Add to split rule vector
        cutpoint_feature.push_back(j);
        cutpoint_values.push_back(feature_value);
        // Add the log marginal likelihood of the split to the split eval vector 
        log_split_eval = SplitLogMarginalLikelihood(left_suff_stat_, right_suff_stat_);
        log_cutpoint_evaluations.push_back(log_split_eval);
      }
    }
  }

  // Evaluate the "no-split" option
  double log_no_split_eval = 0.;
  int node_depth = tree->GetDepth(leaf_node);
  cutpoint_feature.push_back(-1);
  cutpoint_values.push_back(std::numeric_limits<double>::max());
  log_no_split_eval = NoSplitLogMarginalLikelihood(node_suff_stat_, node_depth, num_cutpoints);
  log_cutpoint_evaluations.push_back(log_no_split_eval);
  valid_cutpoint_count = num_cutpoints;
}

void XBARTGaussianRegressionModel::AddSplitToModel(TrainData* train_data, Tree* tree, node_t leaf_node, 
                                                   data_size_t node_begin, data_size_t node_end, 
                                                   int feature_split, double split_value, std::deque<node_t>& split_queue, 
                                                   std::vector<std::vector<data_size_t>>& tree_observation_indices, int tree_num) {
  // Compute the sufficient statistics for the new left and right node as well as the parent node being split
  XBARTGaussianRegressionSuffStat node_suff_stat_;
  XBARTGaussianRegressionSuffStat left_suff_stat_;
  XBARTGaussianRegressionSuffStat right_suff_stat_;
  ResetSuffStat(left_suff_stat_);
  node_suff_stat_ = ComputeNodeSuffStat(train_data, node_begin, node_end, feature_split);
  AccumulateSplitRule(train_data, left_suff_stat_, feature_split, split_value, node_begin, node_end, true);
  right_suff_stat_ = SubtractSuffStat(node_suff_stat_, left_suff_stat_);

  // Split the tree at leaf node
  // Use the average outcome in each leaf as a "temporary" leaf value since we sample 
  // all leaf parameters after tree sampling is complete
  double left_leaf_value = left_suff_stat_.outcome_sum_/left_suff_stat_.sample_size_;
  double right_leaf_value = right_suff_stat_.outcome_sum_/right_suff_stat_.sample_size_;
  tree->ExpandNode(leaf_node, feature_split, split_value, true, 1., left_leaf_value, right_leaf_value, -1);
  node_t left_node = tree->LeftChild(leaf_node);
  node_t right_node = tree->RightChild(leaf_node);

  // Partition the dataset according to the new split rule and 
  // determine the beginning and end of the new left and right nodes
  train_data->PartitionLeaf(node_begin, node_suff_stat_.sample_size_, 
                            feature_split, split_value);

  // Update the leaf node observation tracker
  data_size_t obs_idx;
  for (data_size_t i = node_begin; i < node_begin + left_suff_stat_.sample_size_; i++) {
    obs_idx = train_data->get_feature_sort_index(i, 0);
    tree_observation_indices[tree_num][obs_idx] = left_node;
  }
  for (data_size_t i = node_begin + left_suff_stat_.sample_size_; i < node_end; i++) {
    obs_idx = train_data->get_feature_sort_index(i, 0);
    tree_observation_indices[tree_num][obs_idx] = right_node;
  }

  // Add the begin and end indices for the new left and right nodes to node_index_map
  AddNode(left_node, node_begin, node_begin + left_suff_stat_.sample_size_);
  AddNode(right_node, node_begin + left_suff_stat_.sample_size_, node_end);

  // Add the left and right nodes to the split tracker
  split_queue.push_front(right_node);
  split_queue.push_front(left_node);
}

double XBARTGaussianRegressionModel::SplitLogMarginalLikelihood(const XBARTGaussianRegressionSuffStat& left_stat, 
                                                                const XBARTGaussianRegressionSuffStat& right_stat) {
  // Unpack node sufficient statistics
  data_size_t left_n = left_stat.sample_size_;
  data_size_t right_n = right_stat.sample_size_;
  double left_sum_y = left_stat.outcome_sum_;
  double right_sum_y = right_stat.outcome_sum_;
  double sum_y_sq = left_stat.outcome_sum_sq_ + right_stat.outcome_sum_sq_;
  
  // Compute left node contribution to log marginal likelihood
  double left_prior_contrib = 0.5 * std::log(sigma_sq_) - 0.5 * std::log(sigma_sq_ + tau_*left_n);
  double left_data_contrib = 0.5 * (tau_*std::pow(left_sum_y, 2.)) / (sigma_sq_*(sigma_sq_ + tau_*left_n));
  double left_exponent = left_prior_contrib + left_data_contrib;

  // Compute right node contribution to log marginal likelihood
  double right_prior_contrib = 0.5 * std::log(sigma_sq_) - 0.5 * std::log(sigma_sq_ + tau_*right_n);
  double right_data_contrib = 0.5 * (tau_*std::pow(right_sum_y, 2.)) / (sigma_sq_*(sigma_sq_ + tau_*right_n));
  double right_exponent = right_prior_contrib + right_data_contrib;

  // Compute the normalizing components of the log marginal likelihood
  data_size_t n = left_n + right_n;
  double normalizing_component = (-0.5)*(n)*(std::log(sigma_sq_)) - 0.5 * (sum_y_sq/(sigma_sq_));
  
  return normalizing_component + left_exponent + right_exponent;
}

double XBARTGaussianRegressionModel::SplitMarginalLikelihood(const XBARTGaussianRegressionSuffStat& left_stat, 
                                                             const XBARTGaussianRegressionSuffStat& right_stat) {
  return std::exp(SplitLogMarginalLikelihood(left_stat, right_stat));
}

double XBARTGaussianRegressionModel::NoSplitLogMarginalLikelihood(const XBARTGaussianRegressionSuffStat& node_stat, 
                                                                  int node_depth, data_size_t num_split_candidates) {
  // Unpack node sufficient statistics
  data_size_t node_n = node_stat.sample_size_;
  double node_sum_y = node_stat.outcome_sum_;
  double node_sum_y_sq = node_stat.outcome_sum_sq_;

  // Compute leaf prior contribution to log marginal likelihood
  double prior_contrib = std::log(sigma_sq_) - std::log(sigma_sq_ + tau_*node_n);

  // Compute data likelihood contribution to log marginal likelihood
  double data_contrib = ((tau_*std::pow(node_sum_y, 2.0))/(sigma_sq_*(sigma_sq_ + tau_*node_n)));
  
  // Combine the results with prior probability on tree structures
  double exponent = (prior_contrib + data_contrib) * 0.5;

  // BART prior adjustments
  // Handle the case where a node is exactly at min_data_in_leaf so there are no valid split candidates
  // In this case the no split option will be chosen by default, but taking std::log(num_split_candidates) yields -inf
  double bart_prior_no_split_adj;
  if (num_split_candidates == 0) {
    bart_prior_no_split_adj = std::log(((std::pow(1+node_depth, beta_))/alpha_) - 1.0);
  } else {
    bart_prior_no_split_adj = std::log(((std::pow(1+node_depth, beta_))/alpha_) - 1.0) + std::log(num_split_candidates);
  }

  // Compute the normalizing component of the log marginal likelihood
  double normalizing_component = - 0.5 * (node_n)*(std::log(sigma_sq_)) - 0.5 * (node_sum_y_sq/(sigma_sq_));
  
  return exponent + bart_prior_no_split_adj + normalizing_component;
}

double XBARTGaussianRegressionModel::NoSplitMarginalLikelihood(const XBARTGaussianRegressionSuffStat& node_stat, 
                                                               int node_depth, data_size_t num_split_candidates) {
  return std::exp(NoSplitLogMarginalLikelihood(node_stat, node_depth, num_split_candidates));
}

XBARTGaussianRegressionSuffStat XBARTGaussianRegressionModel::ComputeNodeSuffStat(TrainData* train_data, data_size_t node_begin, 
                                                                                  data_size_t node_end, int feature_idx) {
  XBARTGaussianRegressionSuffStat suff_stat;
  // Reset sample size and outcome sum
  data_size_t sample_size = 0;
  double outcome;
  double outcome_sum = 0.;
  double outcome_sum_sq = 0.;
  // Compute the total sufficient statistics for a node
  for (data_size_t i = node_begin; i < node_end; i++) {
    // We could also compute this as node_end - node_begin
    sample_size++;
    // Each feature has different sort indices, but since we just 
    // need a running total sample size and running total outcome 
    // per node we can use any feature's sort indices to get the 
    // corresponding outcome information
    outcome = train_data->get_residual_value(train_data->get_feature_sort_index(i, feature_idx));
    outcome_sum += outcome;
    outcome_sum_sq += std::pow(outcome, 2.0);
  }
  suff_stat.sample_size_ = sample_size;
  suff_stat.outcome_sum_ = outcome_sum;
  suff_stat.outcome_sum_sq_ = outcome_sum_sq;
  return suff_stat;
}

XBARTGaussianRegressionSuffStat XBARTGaussianRegressionModel::SubtractSuffStat(const XBARTGaussianRegressionSuffStat& first_node_suff_stat, 
                                                                               const XBARTGaussianRegressionSuffStat& second_node_suff_stat) {
  XBARTGaussianRegressionSuffStat suff_stat;
  suff_stat.sample_size_ = first_node_suff_stat.sample_size_ - second_node_suff_stat.sample_size_;
  suff_stat.outcome_sum_ = first_node_suff_stat.outcome_sum_ - second_node_suff_stat.outcome_sum_;
  suff_stat.outcome_sum_sq_ = first_node_suff_stat.outcome_sum_sq_ - second_node_suff_stat.outcome_sum_sq_;
  return suff_stat;
}

void XBARTGaussianRegressionModel::AccumulateRowSuffStat(TrainData* train_data, XBARTGaussianRegressionSuffStat& suff_stat, 
                                                         data_size_t row, int col, data_size_t& node_row_iter) {
  double outcome_value = train_data->get_residual_value(train_data->get_feature_sort_index(row, col));
  double outcome_value_sq = std::pow(outcome_value, 2.0);
  double feature_value = train_data->get_feature_value(train_data->get_feature_sort_index(row, col), col);
  data_size_t feature_stride = train_data->get_feature_stride(row, col);
  
  if (feature_stride == 1){
    // If feature value is unique, no need to handle the strides
    suff_stat.sample_size_++;
    suff_stat.outcome_sum_ += outcome_value;
    suff_stat.outcome_sum_sq_ += outcome_value_sq;
    node_row_iter++;
  } else if (feature_stride > 1) {
    // If feature value is non-unique, need to look at the outcome for every unique occurence of that value
    suff_stat.sample_size_ += feature_stride;
    node_row_iter += feature_stride;
    for (data_size_t k = 0; k < feature_stride; k++){
      outcome_value = train_data->get_residual_value(train_data->get_feature_sort_index(row, col) + k);
      outcome_value_sq = std::pow(outcome_value, 2.0);
      suff_stat.outcome_sum_ += outcome_value;
      suff_stat.outcome_sum_sq_ += outcome_value_sq;
    }
  } else {
    // This case implies an error in the training dataset loading process (feature with at least one 
    // unique value and a stride of <= 0)
    Log::Fatal("Error reading from training dataset, unique feature value %d has a stride of 0", feature_value);
  }
}

void XBARTGaussianRegressionModel::ResetSuffStat(XBARTGaussianRegressionSuffStat& suff_stat, data_size_t sample_size, double outcome_sum, double outcome_sum_sq) {
  suff_stat.sample_size_ = sample_size;
  suff_stat.outcome_sum_ = outcome_sum;
  suff_stat.outcome_sum_sq_ = outcome_sum_sq;
}

void XBARTGaussianRegressionModel::AccumulateSplitRule(TrainData* train_data, XBARTGaussianRegressionSuffStat& suff_stat, 
                                                       int split_col, double split_value, data_size_t node_begin, 
                                                       data_size_t node_end, bool is_left) {
  double feature_value;
  double outcome_value;
  double outcome_value_sq;
  suff_stat.sample_size_ = 0;
  suff_stat.outcome_sum_ = 0.0;
  suff_stat.outcome_sum_sq_ = 0.0;
  for (data_size_t i = node_begin; i < node_end; i++) {
    feature_value = train_data->get_feature_value(train_data->get_feature_sort_index(i, split_col), split_col);
    outcome_value = train_data->get_residual_value(train_data->get_feature_sort_index(i, split_col));
    outcome_value_sq = std::pow(outcome_value, 2.0);
    if (feature_value <= split_value && is_left){
      suff_stat.sample_size_++;
      suff_stat.outcome_sum_ += outcome_value;
      suff_stat.outcome_sum_sq_ += outcome_value_sq;
    } else if (feature_value > split_value && !is_left) {
      suff_stat.sample_size_++;
      suff_stat.outcome_sum_ += outcome_value;
      suff_stat.outcome_sum_sq_ += outcome_value_sq;
    }
  }
}

} // namespace StochTree
