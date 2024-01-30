/*!
 * Copyright (c) 2023 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 * 
 * Inspired by the design of the Learner, GBTreeModel, and GBTree classes in xgboost, 
 * released under the Apache license with the following copyright:
 * 
 * Copyright 2015-2023 by XGBoost Contributors
 */
#ifndef STOCHTREE_SAMPLER_H_
#define STOCHTREE_SAMPLER_H_

#include <stochtree/cutpoint_candidates.h>
#include <stochtree/ensemble.h>
#include <stochtree/outcome_model.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/tree.h>
#include <stochtree/tree_prior.h>
#include <Eigen/Dense>

#include <cmath>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace StochTree {

/*! \brief Partition the presorted feature node tracker used in the grow-from-root algorithm according to a numeric split */
void PartitionLeaf(Eigen::MatrixXd& covariates, SortedNodeSampleTracker* sorted_node_sample_tracker, int leaf_node, int split_col, double split_value);

/*! \brief Partition the presorted feature node tracker used in the grow-from-root algorithm according to a categorical split */
void PartitionLeaf(Eigen::MatrixXd& covariates, SortedNodeSampleTracker* sorted_node_sample_tracker, int leaf_node, int split_col, std::vector<std::uint32_t>& categories);

/*! \brief Whether all leaf nodes are non-constant (in at least one covariate) after a proposed numeric split */
bool NodesNonConstantAfterSplit(Eigen::MatrixXd& covariates, Tree* tree, UnsortedNodeSampleTracker* node_tracker, int leaf_split, int feature_split, double split_value, int tree_num);

/*! \brief Whether all leaf nodes are non-constant (in at least one covariate) after a proposed categorical split */
bool NodesNonConstantAfterSplit(Eigen::MatrixXd& covariates, Tree* tree, UnsortedNodeSampleTracker* node_tracker, int leaf_split, int feature_split, std::vector<std::uint32_t> split_categories, int tree_num);

/*! \brief Whether a given node is non-constant */
bool NodeNonConstant(Eigen::MatrixXd& covariates, Tree* tree, UnsortedNodeSampleTracker* node_tracker, int node_id, int tree_num);

/*! \brief Range of possible split values for a given (numeric) feature */
void VarSplitRange(Eigen::MatrixXd& covariates, Tree* tree, UnsortedNodeSampleTracker* node_tracker, int leaf_split, int feature_split, double& var_min, double& var_max, int tree_num);

/*! \brief Adding a numeric split to a model and updating all of the relevant data structures that support sampling */
void AddSplitToModel(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, Tree* tree, UnsortedNodeSampleTracker* node_tracker, SampleNodeMapper* sample_node_mapper, int leaf_node, int feature_split, double split_value, int tree_num);

/*! \brief Removing a numeric split from a model and updating all of the relevant data structures that support sampling */
void RemoveSplitFromModel(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, Tree* tree, UnsortedNodeSampleTracker* node_tracker, SampleNodeMapper* sample_node_mapper, int leaf_node, int left_node, int right_node, int feature_split, double split_value, int tree_num);

/*! \brief Compute sufficient statistics relevant to a proposed numeric split */
template <typename ModelType, typename TreePriorType>
void ComputeSplitSuffStats(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, Eigen::MatrixXd& outcome, Tree* tree, UnsortedNodeSampleTracker* node_sample_tracker, SampleNodeMapper* sample_node_mapper, int tree_num, int leaf_split, int feature_split, double split_value, ModelType& model, TreePriorType& tree_prior) {
  // Unpack shifted iterators to observations in a given node
  auto tree_node_tracker = node_sample_tracker->GetFeaturePartition(tree_num);
  data_size_t node_begin = tree_node_tracker->NodeBegin(leaf_split);
  data_size_t node_end = tree_node_tracker->NodeEnd(leaf_split);
  data_size_t idx;
  auto node_begin_iter = tree_node_tracker->indices_.begin() + node_begin;
  auto node_end_iter = tree_node_tracker->indices_.begin() + node_end;

  // Reset all sufficient statistics
  model.ResetNodeSuffStat(NodeIndicator::SplitNode);
  model.ResetNodeSuffStat(NodeIndicator::LeftNode);
  model.ResetNodeSuffStat(NodeIndicator::RightNode);

  // Iterate through every observation in the node
  double feature_value;
  for (auto i = node_begin_iter; i != node_end_iter; i++) {
    idx = *i;
    feature_value = covariates(idx, feature_split);
    // Increment sufficient statistics for the split node, regardless of covariate value
    model.IncrementNodeSuffStat(covariates, basis, outcome, idx, NodeIndicator::SplitNode);
    if (SplitTrueNumeric(feature_value, split_value)) {
      // Increment new left node sufficient statistic if split is true
      model.IncrementNodeSuffStat(covariates, basis, outcome, idx, NodeIndicator::LeftNode);
    } else {
      // Increment new left node sufficient statistic if split is false
      model.IncrementNodeSuffStat(covariates, basis, outcome, idx, NodeIndicator::RightNode);
    }
  }
}

/*! \brief Compute sufficient statistics relevant to a proposed categorical split */
template <typename ModelType, typename TreePriorType>
void ComputeSplitSuffStats(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, Eigen::MatrixXd& outcome, Tree* tree, UnsortedNodeSampleTracker* node_sample_tracker, SampleNodeMapper* sample_node_mapper, int tree_num, int leaf_split, int feature_split, std::vector<std::uint32_t>& split_categories, ModelType& model, TreePriorType& tree_prior) {
  // Unpack shifted iterators to observations in a given node
  auto tree_node_tracker = node_sample_tracker->GetFeaturePartition(tree_num);
  data_size_t node_begin = tree_node_tracker->NodeBegin(leaf_split);
  data_size_t node_end = tree_node_tracker->NodeEnd(leaf_split);
  data_size_t idx;
  auto node_begin_iter = tree_node_tracker->indices_.begin() + node_begin;
  auto node_end_iter = tree_node_tracker->indices_.begin() + node_end;

  // Reset all sufficient statistics
  model.ResetNodeSuffStat(NodeIndicator::SplitNode);
  model.ResetNodeSuffStat(NodeIndicator::LeftNode);
  model.ResetNodeSuffStat(NodeIndicator::RightNode);

  // Iterate through every observation in the node
  double feature_value;
  for (auto i = node_begin_iter; i != node_end_iter; i++) {
    idx = *i;
    feature_value = covariates(idx, feature_split);
    // Increment sufficient statistics for the split node, regardless of covariate value
    model.IncrementNodeSuffStat(covariates, basis, outcome, idx, NodeIndicator::SplitNode);
    if (SplitTrueCategorical(feature_value, split_categories)) {
      // Increment new left node sufficient statistic if split is true
      model.IncrementNodeSuffStat(covariates, basis, outcome, idx, NodeIndicator::LeftNode);
    } else {
      // Increment new left node sufficient statistic if split is false
      model.IncrementNodeSuffStat(covariates, basis, outcome, idx, NodeIndicator::RightNode);
    }
  }
}

/*! \brief Compute sufficient statistics for a given node, assuming a non-feature-specific node tracker */
template <typename ModelType, typename TreePriorType>
void ComputeNodeSuffStats(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, Eigen::MatrixXd& outcome, Tree* tree, UnsortedNodeSampleTracker* node_sample_tracker, SampleNodeMapper* sample_node_mapper, int tree_num, int leaf_split, ModelType& model, TreePriorType& tree_prior, NodeIndicator node_indicator) {
  // Unpack shifted iterators to observations in a given node
  auto tree_node_tracker = node_sample_tracker->GetFeaturePartition(tree_num);
  data_size_t node_begin = tree_node_tracker->NodeBegin(leaf_split);
  data_size_t node_end = tree_node_tracker->NodeEnd(leaf_split);
  data_size_t idx;
  auto node_begin_iter = tree_node_tracker->indices_.begin() + node_begin;
  auto node_end_iter = tree_node_tracker->indices_.begin() + node_end;

  // Reset sufficient statistic for the "SplitNode"
  model.ResetNodeSuffStat(node_indicator);
 
  // Iterate through every observation in the node
  for (auto i = node_begin_iter; i != node_end_iter; i++) {
    idx = *i;
    // Increment sufficient statistic
    model.IncrementNodeSuffStat(covariates, basis, outcome, idx, node_indicator);
  }
}

/*! \brief Compute sufficient statistics for a given node, assuming a feature-pre-sorted node tracker */
template <typename ModelType, typename TreePriorType>
void ComputeNodeSuffStats(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, Eigen::MatrixXd& outcome, Tree* tree, SortedNodeSampleTracker* sorted_node_sample_tracker, SampleNodeMapper* sample_node_mapper, data_size_t node_begin, data_size_t node_end, ModelType& model, TreePriorType& tree_prior, NodeIndicator node_indicator) {
  // Reset sufficient statistic
  model.ResetNodeSuffStat(node_indicator);

  // Compute the total sufficient statistics for a node
  data_size_t sort_idx;
  for (data_size_t i = node_begin; i < node_end; i++) {
    sort_idx = sorted_node_sample_tracker->SortIndex(i, 0);
    model.IncrementNodeSuffStat(covariates, basis, outcome, sort_idx, node_indicator);
  }
}

/*! \brief Perform one MCMC grow step */
template <typename ModelType, typename TreePriorType>
void GrowMCMC(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, Eigen::MatrixXd& outcome, Tree* tree, UnsortedNodeSampleTracker* node_sample_tracker, SampleNodeMapper* sample_node_mapper, int tree_num, std::mt19937& gen, ModelType& model, TreePriorType& tree_prior, double prob_grow_old) {
  // Extract dataset information
  data_size_t n = covariates.rows();
  int basis_dim = basis.cols();

  // Choose a leaf node at random
  int num_leaves = tree->NumLeaves();
  std::vector<int> leaves = tree->GetLeaves();
  std::vector<double> leaf_weights(num_leaves);
  std::fill(leaf_weights.begin(), leaf_weights.end(), 1.0/num_leaves);
  std::discrete_distribution<> leaf_dist(leaf_weights.begin(), leaf_weights.end());
  int leaf_chosen = leaves[leaf_dist(gen)];
  int leaf_depth = tree->GetDepth(leaf_chosen);

  // Select a split variable at random
  int p = covariates.cols();
  std::vector<double> var_weights(p);
  std::fill(var_weights.begin(), var_weights.end(), 1.0/p);
  std::discrete_distribution<> var_dist(var_weights.begin(), var_weights.end());
  int var_chosen = var_dist(gen);

  // Determine the range of possible cutpoints
  double var_min, var_max;
  VarSplitRange(covariates, tree, node_sample_tracker, leaf_chosen, var_chosen, var_min, var_max, tree_num);
  if (var_max <= var_min) {
    return;
  }
  // Split based on var_min to var_max in a given node
  std::uniform_real_distribution<double> split_point_dist(var_min, var_max);
  double split_point_chosen = split_point_dist(gen);

  // Compute sufficient statistics of the existing node and the two new nodes
  ComputeSplitSuffStats<ModelType, TreePriorType>(covariates, basis, outcome, tree, node_sample_tracker, sample_node_mapper, tree_num, leaf_chosen, var_chosen, split_point_chosen, model, tree_prior);

  // Compute the marginal likelihood
  double split_log_marginal_likelihood = model.SplitLogMarginalLikelihood();
  double no_split_log_marginal_likelihood = model.NoSplitLogMarginalLikelihood();
  
  // Determine probability of growing the split node and its two new left and right nodes
  double pg = tree_prior.alpha * std::pow(1+leaf_depth, -tree_prior.beta);
  double pgl = tree_prior.alpha * std::pow(1+leaf_depth+1, -tree_prior.beta);
  double pgr = tree_prior.alpha * std::pow(1+leaf_depth+1, -tree_prior.beta);

  // Determine whether a "grow" move is possible from the newly formed tree
  // in order to compute the probability of choosing "prune" from the new tree
  // (which is always possible by construction)
  bool non_constant = NodesNonConstantAfterSplit(covariates, tree, node_sample_tracker, leaf_chosen, var_chosen, split_point_chosen, tree_num);
  bool min_samples_left_check = model.NodeSampleGreaterThan(NodeIndicator::LeftNode, 2*tree_prior.min_samples_in_leaf);
  bool min_samples_right_check = model.NodeSampleGreaterThan(NodeIndicator::RightNode, 2*tree_prior.min_samples_in_leaf);
  double prob_prune_new;
  if (non_constant && min_samples_left_check && min_samples_right_check) {
    prob_prune_new = 0.5;
  } else {
    prob_prune_new = 1.0;
  }

  // Determine the number of leaves in the current tree and leaf parents in the proposed tree
  int num_leaf_parents = tree->NumLeafParents();
  double p_leaf = 1/num_leaves;
  double p_leaf_parent = 1/(num_leaf_parents+1);

  // Compute the final MH ratio
  double log_mh_ratio = (
    std::log(pg) + std::log(1-pgl) + std::log(1-pgr) - std::log(1-pg) + std::log(prob_prune_new) +
    std::log(p_leaf_parent) - std::log(prob_grow_old) - std::log(p_leaf) + no_split_log_marginal_likelihood - split_log_marginal_likelihood
  );
  // Threshold at 0
  if (log_mh_ratio > 1) {
    log_mh_ratio = 1;
  }

  // Draw a uniform random variable and accept/reject the proposal on this basis
  bool accept;
  std::uniform_real_distribution<double> mh_accept(0.0, 1.0);
  double log_acceptance_prob = std::log(mh_accept(gen));
  if (log_acceptance_prob <= log_mh_ratio) {
    accept = true;
    AddSplitToModel(covariates, basis, tree, node_sample_tracker, sample_node_mapper, leaf_chosen, var_chosen, split_point_chosen, tree_num);
  } else {
    accept = false;
  }
}

/*! \brief Perform one MCMC prune step */
template <typename ModelType, typename TreePriorType>
void PruneMCMC(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, Eigen::MatrixXd& outcome, Tree* tree, UnsortedNodeSampleTracker* node_sample_tracker, SampleNodeMapper* sample_node_mapper, int tree_num, std::mt19937& gen, ModelType& model, TreePriorType& tree_prior) {
  // Choose a "leaf parent" node at random
  int num_leaves = tree->NumLeaves();
  int num_leaf_parents = tree->NumLeafParents();
  std::vector<int> leaf_parents = tree->GetLeafParents();
  std::vector<double> leaf_parent_weights(num_leaf_parents);
  std::fill(leaf_parent_weights.begin(), leaf_parent_weights.end(), 1.0/num_leaf_parents);
  std::discrete_distribution<> leaf_parent_dist(leaf_parent_weights.begin(), leaf_parent_weights.end());
  int leaf_parent_chosen = leaf_parents[leaf_parent_dist(gen)];
  int leaf_parent_depth = tree->GetDepth(leaf_parent_chosen);
  int left_node = tree->LeftChild(leaf_parent_chosen);
  int right_node = tree->RightChild(leaf_parent_chosen);
  int feature_split = tree->SplitIndex(leaf_parent_chosen);
  double split_value = tree->Threshold(leaf_parent_chosen);

  // Compute sufficient statistics for the leaf parent and its left and right nodes
  ComputeNodeSuffStats<ModelType, TreePriorType>(covariates, basis, outcome, tree, node_sample_tracker, sample_node_mapper, tree_num, leaf_parent_chosen, model, tree_prior, NodeIndicator::SplitNode);
  ComputeNodeSuffStats<ModelType, TreePriorType>(covariates, basis, outcome, tree, node_sample_tracker, sample_node_mapper, tree_num, left_node, model, tree_prior, NodeIndicator::LeftNode);
  ComputeNodeSuffStats<ModelType, TreePriorType>(covariates, basis, outcome, tree, node_sample_tracker, sample_node_mapper, tree_num, right_node, model, tree_prior, NodeIndicator::RightNode);

  // Compute the marginal likelihoods
  double split_log_marginal_likelihood = model.SplitLogMarginalLikelihood();
  double no_split_log_marginal_likelihood = model.NoSplitLogMarginalLikelihood();
  
  // Determine probability of growing the split node and its two new left and right nodes
  double pg = tree_prior.alpha * std::pow(1+leaf_parent_depth, -tree_prior.beta);
  double pgl = tree_prior.alpha * std::pow(1+leaf_parent_depth+1, -tree_prior.beta);
  double pgr = tree_prior.alpha * std::pow(1+leaf_parent_depth+1, -tree_prior.beta);

  // Determine whether a "prune" move is possible from the new tree,
  // in order to compute the probability of choosing "grow" from the new tree
  // (which is always possible by construction)
  bool non_root_tree = tree->NumNodes() > 1;
  double prob_grow_new;
  if (non_root_tree) {
    prob_grow_new = 0.5;
  } else {
    prob_grow_new = 1.0;
  }

  // Determine whether a "grow" move was possible from the old tree,
  // in order to compute the probability of choosing "prune" from the old tree
  bool non_constant_left = NodeNonConstant(covariates, tree, node_sample_tracker, left_node, tree_num);
  bool non_constant_right = NodeNonConstant(covariates, tree, node_sample_tracker, right_node, tree_num);
  double prob_prune_old;
  if (non_constant_left && non_constant_right) {
    prob_prune_old = 0.5;
  } else {
    prob_prune_old = 1.0;
  }

  // Determine the number of leaves in the current tree and leaf parents in the proposed tree
  double p_leaf = 1/(num_leaves-1);
  double p_leaf_parent = 1/(num_leaf_parents);

  // Compute the final MH ratio
  double log_mh_ratio = (
    std::log(1-pg) - std::log(pg) - std::log(1-pgl) - std::log(1-pgr) + std::log(prob_prune_old) +
    std::log(p_leaf) - std::log(prob_grow_new) - std::log(p_leaf_parent) + no_split_log_marginal_likelihood - split_log_marginal_likelihood
  );
  // Threshold at 0
  if (log_mh_ratio > 0) {
    log_mh_ratio = 0;
  }

  // Draw a uniform random variable and accept/reject the proposal on this basis
  bool accept;
  std::uniform_real_distribution<double> mh_accept(0.0, 1.0);
  double log_acceptance_prob = std::log(mh_accept(gen));
  if (log_acceptance_prob <= log_mh_ratio) {
    accept = true;
    RemoveSplitFromModel(covariates, basis, tree, node_sample_tracker, sample_node_mapper, leaf_parent_chosen, left_node, right_node, feature_split, split_value, tree_num);
  } else {
    accept = false;
  }
}

/*! \brief Perform one MCMC grow-prune step */
template <typename ModelType, typename TreePriorType>
void BirthDeathMCMC(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, Eigen::MatrixXd& outcome, Tree* tree, UnsortedNodeSampleTracker* node_sample_tracker, SampleNodeMapper* sample_node_mapper, int tree_num, std::mt19937& gen, ModelType& model, TreePriorType& tree_prior) {
  // Determine whether it is possible to grow any of the leaves
  bool grow_possible = false;
  std::vector<int> leaves = tree->GetLeaves();
  for (auto& leaf: leaves) {
    if (node_sample_tracker->NodeSize(tree_num, leaf) > 2*tree_prior.min_samples_in_leaf) {
      grow_possible = true;
      break;
    }
  }

  // Determine whether it is possible to prune the tree
  bool prune_possible = false;
  if (tree->NumValidNodes() > 1) {
    prune_possible = true;
  }

  // Determine the relative probability of grow vs prune (0 = grow, 1 = prune)
  double prob_grow;
  std::vector<double> step_probs(2);
  if (grow_possible && prune_possible) {
    step_probs = {0.5, 0.5};
    prob_grow = 0.5;
  } else if (!grow_possible && prune_possible) {
    step_probs = {0.0, 1.0};
    prob_grow = 0.0;
  } else if (grow_possible && !prune_possible) {
    step_probs = {1.0, 0.0};
    prob_grow = 1.0;
  } else {
    Log::Fatal("In this tree, neither grow nor prune is possible");
  }
  std::discrete_distribution<> step_dist(step_probs.begin(), step_probs.end());

  // Draw a split rule at random
  data_size_t step_chosen = step_dist(gen);
  bool accept;
  
  if (step_chosen == 0) {
    GrowMCMC<ModelType, TreePriorType>(covariates, basis, outcome, tree, node_sample_tracker, sample_node_mapper, tree_num, gen, model, tree_prior, prob_grow);
  } else {
    PruneMCMC<ModelType, TreePriorType>(covariates, basis, outcome, tree, node_sample_tracker, sample_node_mapper, tree_num, gen, model, tree_prior);
  }
}


/*! \brief Accumulate a numeric split rule to a left node sufficient statistic */
template <typename ModelType, typename TreePriorType>
void AccumulateSplitRule(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, Eigen::MatrixXd& outcome, Tree* tree, SortedNodeSampleTracker* sorted_node_sample_tracker, 
                         data_size_t node_begin, data_size_t node_end, int feature_split, double split_value, bool is_left, ModelType& model, TreePriorType& tree_prior) {
  model.ResetNodeSuffStat(NodeIndicator::LeftNode);
  // suff_stat.ResetSuffStat(covariates, basis, outcome);
  double feature_value;
  data_size_t sort_idx;
  bool split_true;
  for (data_size_t i = node_begin; i < node_end; i++) {
    sort_idx = sorted_node_sample_tracker->SortIndex(i, feature_split);
    feature_value = covariates(sort_idx, feature_split);
    split_true = SplitTrueNumeric(feature_value, split_value);
    // Only accumulate sample sufficient statistics if either 
    // (a) the accumulated sufficient statistic is for a left node and the split rule is true, or
    // (b) the accumulated sufficient statistic is for a right node and the split rule is false
    if (split_true && is_left){
      model.IncrementNodeSuffStat(covariates, basis, outcome, i, NodeIndicator::LeftNode);
    } else if (!split_true && !is_left) {
      model.IncrementNodeSuffStat(covariates, basis, outcome, i, NodeIndicator::LeftNode);
    }
  }
}

/*! \brief Accumulate a categorical split rule to a left node sufficient statistic */
template <typename ModelType, typename TreePriorType>
void AccumulateSplitRule(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, Eigen::MatrixXd& outcome, Tree* tree, SortedNodeSampleTracker* sorted_node_sample_tracker, 
                         data_size_t node_begin, data_size_t node_end, int feature_split, std::vector<std::uint32_t> const& categorical_indices, bool is_left, ModelType& model, TreePriorType& tree_prior) {
  model.ResetNodeSuffStat(NodeIndicator::LeftNode);
  double feature_value;
  data_size_t sort_idx;
  bool split_true;
  for (data_size_t i = node_begin; i < node_end; i++) {
    sort_idx = sorted_node_sample_tracker->SortIndex(i, feature_split);
    feature_value = covariates(sort_idx, feature_split);
    split_true = SplitTrueCategorical(feature_value, categorical_indices);
    // Only accumulate sample sufficient statistics if either 
    // (a) the accumulated sufficient statistic is for a left node and the split rule is true, or
    // (b) the accumulated sufficient statistic is for a right node and the split rule is false
    if (split_true && is_left){
      model.IncrementNodeSuffStat(covariates, basis, outcome, i, NodeIndicator::LeftNode);
    } else if (!split_true && !is_left) {
      model.IncrementNodeSuffStat(covariates, basis, outcome, i, NodeIndicator::LeftNode);
    }
  }
}

/*! \brief Add a split to a model in the GFR algorithm */
template <typename ModelType, typename TreePriorType>
void AddSplitToModel(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, Eigen::MatrixXd& outcome, Tree* tree, SortedNodeSampleTracker* sorted_node_sample_tracker, 
                     FeatureType feature_type, node_t leaf_node, data_size_t node_begin, data_size_t node_end, int feature_split, double split_value, std::deque<node_t>& split_queue, 
                     SampleNodeMapper* sample_node_mapper, int tree_num, ModelType& model, TreePriorType& tree_prior, 
                     CutpointGridContainer& cutpoint_grid_container, std::unordered_map<int, std::pair<data_size_t, data_size_t>>& node_index_map) {
  // Compute the sufficient statistics for the new left and right node as well as the parent node being split
  model.ResetNodeSuffStat(NodeIndicator::SplitNode);
  // ComputeNodeSuffStats<SuffStatType, GlobalParamType, OutcomeModelType, TreePriorType>(covariates, basis, outcome, tree, sorted_node_sample_tracker, sample_node_mapper, node_begin, node_end, node_suff_stat, outcome_model, global_param, tree_prior);
  ComputeNodeSuffStats<ModelType, TreePriorType>(covariates, basis, outcome, tree, sorted_node_sample_tracker, sample_node_mapper, node_begin, node_end, model, tree_prior, NodeIndicator::SplitNode);
  data_size_t node_n = model.NodeSampleSize(NodeIndicator::SplitNode);
  
  // Actual numeric cutpoint used for ordered categorical and numeric features
  double split_value_numeric;

  // Split the tree at leaf node
  // Use 0 as a "temporary" leaf value since we sample 
  // all leaf parameters after tree sampling is complete
  int basis_dim = basis.cols();
  double left_leaf_value = 0.;
  double right_leaf_value = 0.;
  std::vector<double> left_leaf_vector, right_leaf_vector;
  if (basis_dim > 1) {
    left_leaf_vector.resize(basis_dim, 0.);
    right_leaf_vector.resize(basis_dim, 0.);
  }

  if (feature_type == FeatureType::kUnorderedCategorical) {
    // Determine the number of categories available in a categorical split and the set of categories that route observations to the left node after split
    int num_categories;
    std::vector<std::uint32_t> categories = cutpoint_grid_container.CutpointVector(static_cast<std::uint32_t>(split_value), feature_split);

    // Accumulate split rule sufficient statistics
    AccumulateSplitRule<ModelType, TreePriorType>(covariates, basis, outcome, tree, sorted_node_sample_tracker, node_begin, node_end, feature_split, categories, true, model, tree_prior);
    model.SubtractNodeSuffStat(NodeIndicator::RightNode, NodeIndicator::SplitNode, NodeIndicator::LeftNode);

    // Perform the split
    if (basis_dim == 1) {
      tree->ExpandNode(leaf_node, feature_split, categories, true, left_leaf_value, right_leaf_value);
    } else {
      tree->ExpandNode(leaf_node, feature_split, categories, true, left_leaf_vector, right_leaf_vector);
    }
    // Partition the dataset according to the new split rule and determine the beginning and end of the new left and right nodes
    PartitionLeaf(covariates, sorted_node_sample_tracker, leaf_node, feature_split, categories);
  } else if (feature_type == FeatureType::kOrderedCategorical) {
    // Convert the bin split to an actual split value
    // split_value_numeric = cutpoint_grid_container->CutpointValue(static_cast<std::uint32_t>(split_value), feature_split);
    split_value_numeric = cutpoint_grid_container.CutpointValue(static_cast<std::uint32_t>(split_value), feature_split);

    // Accumulate split rule sufficient statistics
    AccumulateSplitRule<ModelType, TreePriorType>(covariates, basis, outcome, tree, sorted_node_sample_tracker, node_begin, node_end, feature_split, split_value_numeric, true, model, tree_prior);
    model.SubtractNodeSuffStat(NodeIndicator::RightNode, NodeIndicator::SplitNode, NodeIndicator::LeftNode);
    
    if (basis_dim == 1) {
      tree->ExpandNode(leaf_node, feature_split, split_value_numeric, true, left_leaf_value, right_leaf_value);
    } else {
      tree->ExpandNode(leaf_node, feature_split, split_value_numeric, true, left_leaf_vector, right_leaf_vector);
    }
    // Partition the dataset according to the new split rule and determine the beginning and end of the new left and right nodes
    PartitionLeaf(covariates, sorted_node_sample_tracker, leaf_node, feature_split, split_value_numeric);
  } else if (feature_type == FeatureType::kNumeric) {
    // Convert the bin split to an actual split value
    split_value_numeric = cutpoint_grid_container.CutpointValue(static_cast<std::uint32_t>(split_value), feature_split);

    // Accumulate split rule sufficient statistics
    AccumulateSplitRule<ModelType, TreePriorType>(covariates, basis, outcome, tree, sorted_node_sample_tracker, node_begin, node_end, feature_split, split_value_numeric, true, model, tree_prior);
    model.SubtractNodeSuffStat(NodeIndicator::RightNode, NodeIndicator::SplitNode, NodeIndicator::LeftNode);
    
    tree->ExpandNode(leaf_node, feature_split, split_value_numeric, true, left_leaf_value, right_leaf_value);
    // Partition the dataset according to the new split rule and determine the beginning and end of the new left and right nodes
    PartitionLeaf(covariates, sorted_node_sample_tracker, leaf_node, feature_split, split_value_numeric);
  } else {
    Log::Fatal("Invalid split type");
  }
  int left_node = tree->LeftChild(leaf_node);
  int right_node = tree->RightChild(leaf_node);

  // Update the leaf node observation tracker
  sorted_node_sample_tracker->UpdateObservationMapping(left_node, tree_num, sample_node_mapper);
  sorted_node_sample_tracker->UpdateObservationMapping(right_node, tree_num, sample_node_mapper);

  // Add the begin and end indices for the new left and right nodes to node_index_map
  data_size_t left_n = model.NodeSampleSize(NodeIndicator::LeftNode);
  node_index_map.insert({left_node, std::make_pair(node_begin, node_begin + left_n)});
  node_index_map.insert({right_node, std::make_pair(node_begin + left_n, node_end)});

  // Add the left and right nodes to the split tracker
  split_queue.push_front(right_node);
  split_queue.push_front(left_node);
}

/*! \brief Evaluate potential splits for each variable */
template <typename ModelType, typename TreePriorType>
void Cutpoints(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, Eigen::MatrixXd& outcome, Tree* tree, SortedNodeSampleTracker* sorted_node_sample_tracker, SampleNodeMapper* sample_node_mapper, int tree_num, std::mt19937& gen, ModelType& model, TreePriorType& tree_prior, 
               std::vector<FeatureType>& feature_types, data_size_t node_begin, data_size_t node_end, std::deque<node_t>& split_queue, std::vector<double>& log_cutpoint_evaluations, std::vector<int>& cutpoint_feature, std::vector<double>& cutpoint_values, std::vector<FeatureType>& cutpoint_feature_types, 
               data_size_t& valid_cutpoint_count, int cutpoint_grid_size, int leaf_node, CutpointGridContainer& cutpoint_grid_container) {
  // Compute sufficient statistics for the current node
  ComputeNodeSuffStats<ModelType, TreePriorType>(covariates, basis, outcome, tree, sorted_node_sample_tracker, sample_node_mapper, node_begin, node_end, model, tree_prior, NodeIndicator::SplitNode);

  // Clear vectors
  log_cutpoint_evaluations.clear();
  cutpoint_feature.clear();
  cutpoint_values.clear();
  cutpoint_feature_types.clear();

  // Reset cutpoint grid container
  cutpoint_grid_container.Reset(covariates, outcome, cutpoint_grid_size);

  // Compute sufficient statistics for each possible split
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
  for (int j = 0; j < covariates.cols(); j++) {

    // Enumerate cutpoint strides
    cutpoint_grid_container.CalculateStrides(covariates, outcome, sorted_node_sample_tracker, leaf_node, node_begin, node_end, j, feature_types);

    // Iterate through possible cutpoints
    int32_t num_feature_cutpoints = cutpoint_grid_container.NumCutpoints(j);
    feature_type = feature_types[j];
    model.ResetNodeSuffStat(NodeIndicator::LeftNode);
    model.ResetNodeSuffStat(NodeIndicator::RightNode);
    for (data_size_t cutpoint_idx = 0; cutpoint_idx < (num_feature_cutpoints - 1); cutpoint_idx++) {
      // Unpack cutpoint details, noting that since we partition an entire cutpoint bin to the left, 
      // we must stop one bin before the total number of cutpoint bins
      current_bin_begin = cutpoint_grid_container.BinStartIndex(cutpoint_idx, j);
      current_bin_size = cutpoint_grid_container.BinLength(cutpoint_idx, j);
      next_bin_begin = cutpoint_grid_container.BinStartIndex(cutpoint_idx + 1, j);

      // Accumulate sufficient statistics
      for (data_size_t k = 0; k < current_bin_size; k++) {
        row_iter_idx = current_bin_begin + k;
        feature_sort_idx = sorted_node_sample_tracker->SortIndex(row_iter_idx, j);
        model.IncrementNodeSuffStat(covariates, basis, outcome, feature_sort_idx, NodeIndicator::LeftNode);
      }

      model.SubtractNodeSuffStat(NodeIndicator::RightNode, NodeIndicator::SplitNode, NodeIndicator::LeftNode);

      // Store the bin index as the "cutpoint value" - we can use this to query the actual split 
      // value or the set of split categories later on once a split is chose
      cutoff_value = cutpoint_idx;

      // Only include cutpoint for consideration if it defines a valid split in the training data
      valid_split = (model.NodeSampleGreaterThan(NodeIndicator::LeftNode, tree_prior.min_samples_in_leaf) && 
                     model.NodeSampleGreaterThan(NodeIndicator::RightNode, tree_prior.min_samples_in_leaf));
      if (valid_split) {
        num_cutpoints++;
        // Add to split rule vector
        cutpoint_feature_types.push_back(feature_type);
        cutpoint_feature.push_back(j);
        cutpoint_values.push_back(cutoff_value);
        // Add the log marginal likelihood of the split to the split eval vector 
        log_split_eval = model.SplitLogMarginalLikelihood();
        log_cutpoint_evaluations.push_back(log_split_eval);
      }
    }
  }

  // Evaluate the log marginal likelihood of the "no-split" option
  double log_no_split_eval;
  cutpoint_feature.push_back(-1);
  cutpoint_values.push_back(std::numeric_limits<double>::max());
  cutpoint_feature_types.push_back(FeatureType::kNumeric);
  log_no_split_eval = model.NoSplitLogMarginalLikelihood();
  
  // Compute an adjustment to reflect the no split prior probability and the number of cutpoints
  double bart_prior_no_split_adj;
  int node_depth = tree->GetDepth(leaf_node);
  if (num_cutpoints == 0) {
    bart_prior_no_split_adj = std::log(((std::pow(1+node_depth, tree_prior.beta))/tree_prior.alpha) - 1.0);
  } else {
    bart_prior_no_split_adj = std::log(((std::pow(1+node_depth, tree_prior.beta))/tree_prior.alpha) - 1.0) + std::log(num_cutpoints);
  }
  log_no_split_eval += bart_prior_no_split_adj;
  
  // Add the no split evaluation to the evaluations
  log_cutpoint_evaluations.push_back(log_no_split_eval);
  valid_cutpoint_count = num_cutpoints;
}

/*! \brief Sample a split (or no-split) rule at a given stage of the grow-from-root algorithm */
template <typename ModelType, typename TreePriorType>
void SampleSplitRule(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, Eigen::MatrixXd& outcome, Tree* tree, SortedNodeSampleTracker* sorted_node_sample_tracker, SampleNodeMapper* sample_node_mapper, int tree_num, std::mt19937& gen, ModelType& model, TreePriorType& tree_prior, std::vector<FeatureType>& feature_types, data_size_t node_begin, data_size_t node_end, std::deque<node_t>& split_queue, int cutpoint_grid_size, int leaf_node, std::unordered_map<int, std::pair<data_size_t, data_size_t>>& node_index_map) {
  std::vector<double> log_cutpoint_evaluations;
  std::vector<int> cutpoint_features;
  std::vector<double> cutpoint_values;
  std::vector<FeatureType> cutpoint_feature_types;
  StochTree::data_size_t valid_cutpoint_count;
  CutpointGridContainer cutpoint_grid_container(covariates, outcome, cutpoint_grid_size);
  Cutpoints<ModelType, TreePriorType>(covariates, basis, outcome, tree, sorted_node_sample_tracker, sample_node_mapper, tree_num, gen, model, tree_prior, feature_types, node_begin, node_end, split_queue, log_cutpoint_evaluations, cutpoint_features, cutpoint_values, cutpoint_feature_types, valid_cutpoint_count, cutpoint_grid_size, leaf_node, cutpoint_grid_container);
  
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
    FeatureType feature_type = cutpoint_feature_types[split_chosen];
    double split_value = cutpoint_values[split_chosen];
    // Perform all of the relevant "split" operations in the model, tree and training dataset
    AddSplitToModel<ModelType, TreePriorType>(covariates, basis, outcome, tree, sorted_node_sample_tracker, feature_type, leaf_node, node_begin, node_end, 
                                              feature_split, split_value, split_queue, sample_node_mapper, tree_num, model, tree_prior, cutpoint_grid_container, node_index_map);
  }
}

/*! \brief Perform one stochastic grow-from-root step */
template <typename ModelType, typename TreePriorType>
void TreeGrowFromRoot(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, Eigen::MatrixXd& outcome, Tree* tree, SortedNodeSampleTracker* sorted_node_sample_tracker, SampleNodeMapper* sample_node_mapper, int tree_num, std::mt19937& gen, ModelType& model, TreePriorType& tree_prior, std::vector<FeatureType>& feature_types, int cutpoint_grid_size) {
  int root_id = Tree::kRoot;
  int curr_node_id;
  data_size_t curr_node_begin;
  data_size_t curr_node_end;
  data_size_t n = covariates.rows();
  // Mapping from node id to start and end points of sorted indices
  std::unordered_map<int, std::pair<data_size_t, data_size_t>> node_index_map_;
  node_index_map_.insert({root_id, std::make_pair(0, n)});
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
    SampleSplitRule<ModelType, TreePriorType>(covariates, basis, outcome, tree, sorted_node_sample_tracker, sample_node_mapper, tree_num, gen, model, tree_prior, feature_types, curr_node_begin, curr_node_end, split_queue_, cutpoint_grid_size, curr_node_id, node_index_map_);
  }
}

/*! \brief Sample leaf node parameters, assuming an unsorted node tracker (i.e. for birth-death style algorithms) */
template <typename ModelType, typename TreePriorType>
void SampleLeafParameters(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, Eigen::MatrixXd& outcome, Tree* tree, UnsortedNodeSampleTracker* node_sample_tracker, SampleNodeMapper* sample_node_mapper, int tree_num, std::mt19937& gen, ModelType& model, TreePriorType& tree_prior) {
  std::vector<int> tree_leaves = tree->GetLeaves();
  int basis_dim = basis.cols();
  data_size_t node_begin, node_end;

  for (int i = 0; i < tree_leaves.size(); i++) {
    // Compute node sufficient statistics
    model.ResetNodeSuffStat(NodeIndicator::SplitNode);
    node_begin = node_sample_tracker->NodeBegin(tree_num, tree_leaves[i]);
    node_end = node_sample_tracker->NodeEnd(tree_num, tree_leaves[i]);
    ComputeNodeSuffStats<ModelType, TreePriorType>(covariates, basis, outcome, tree, node_sample_tracker, sample_node_mapper, tree_num, tree_leaves[i], model, tree_prior, NodeIndicator::SplitNode);

    // Sample leaf value / vector and place it directly in the tree leaf
    model.SampleLeafParameters(gen, tree_leaves[i], tree);
  }
}

/*! \brief Sample leaf node parameters, assuming a pre-sorted feature-specific node tracker (i.e. for recursive grow-from-root style algorithms) */
template <typename ModelType, typename TreePriorType>
void SampleLeafParameters(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, Eigen::MatrixXd& outcome, Tree* tree, SortedNodeSampleTracker* sorted_node_sample_tracker, SampleNodeMapper* sample_node_mapper, int tree_num, std::mt19937& gen, ModelType& model, TreePriorType& tree_prior) {
  std::vector<int> tree_leaves = tree->GetLeaves();
  int basis_dim = basis.cols();
  data_size_t node_begin, node_end;

  for (int i = 0; i < tree_leaves.size(); i++) {
    // Compute node sufficient statistics
    model.ResetNodeSuffStat(NodeIndicator::SplitNode);
    node_begin = sorted_node_sample_tracker->NodeBegin(tree_leaves[i], 0);
    node_end = sorted_node_sample_tracker->NodeEnd(tree_leaves[i], 0);
    ComputeNodeSuffStats<ModelType, TreePriorType>(covariates, basis, outcome, tree, sorted_node_sample_tracker, sample_node_mapper, node_begin, node_end, model, tree_prior, NodeIndicator::SplitNode);

    // Sample leaf value / vector and place it directly in the tree leaf
    model.SampleLeafParameters(gen, tree_leaves[i], tree);
  }
}

} // namespace StochTree

#endif // STOCHTREE_SAMPLER_H_
