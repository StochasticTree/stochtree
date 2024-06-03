/*! Copyright (c) 2024 stochtree authors. All rights reserved. */
#ifndef STOCHTREE_TREE_SAMPLER_H_
#define STOCHTREE_TREE_SAMPLER_H_

#include <stochtree/container.h>
#include <stochtree/cutpoint_candidates.h>
#include <stochtree/data.h>
#include <stochtree/ensemble.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/prior.h>

#include <cmath>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

namespace StochTree {

static void VarSplitRange(ForestTracker& tracker, ForestDataset& dataset, int tree_num, int leaf_split, int feature_split, double& var_min, double& var_max) {
  var_min = std::numeric_limits<double>::max();
  var_max = std::numeric_limits<double>::min();
  double feature_value;
  
  std::vector<data_size_t>::iterator node_begin_iter = tracker.UnsortedNodeBeginIterator(tree_num, leaf_split);
  std::vector<data_size_t>::iterator node_end_iter = tracker.UnsortedNodeEndIterator(tree_num, leaf_split);
  
  for (auto i = node_begin_iter; i != node_end_iter; i++) {
    auto idx = *i;
    feature_value = dataset.CovariateValue(idx, feature_split);
    if (feature_value < var_min) {
      var_min = feature_value;
    } else if (feature_value > var_max) {
      var_max = feature_value;
    }
  }
}

static bool NodesNonConstantAfterSplit(ForestDataset& dataset, ForestTracker& tracker, TreeSplit& split, int tree_num, int leaf_split, int feature_split) {
  int p = dataset.GetCovariates().cols();
  data_size_t idx;
  double feature_value;
  double split_feature_value;
  double var_max_left;
  double var_min_left;
  double var_max_right;
  double var_min_right;
  
  for (int j = 0; j < p; j++) {
    auto node_begin_iter = tracker.UnsortedNodeBeginIterator(tree_num, leaf_split);
    auto node_end_iter = tracker.UnsortedNodeEndIterator(tree_num, leaf_split);
    var_max_left = std::numeric_limits<double>::min();
    var_min_left = std::numeric_limits<double>::max();
    var_max_right = std::numeric_limits<double>::min();
    var_min_right = std::numeric_limits<double>::max();

    for (auto i = node_begin_iter; i != node_end_iter; i++) {
      auto idx = *i;
      split_feature_value = dataset.CovariateValue(idx, feature_split);
      feature_value = dataset.CovariateValue(idx, j);
      if (split.SplitTrue(split_feature_value)) {
        if (var_max_left < feature_value) {
          var_max_left = feature_value;
        } else if (var_min_left > feature_value) {
          var_min_left = feature_value;
        }
      } else {
        if (var_max_right < feature_value) {
          var_max_right = feature_value;
        } else if (var_min_right > feature_value) {
          var_min_right = feature_value;
        }
      }
    }
    if ((var_max_left > var_min_left) && (var_max_right > var_min_right)) {
      return true;
    }
  }
  return false;
}

static bool NodeNonConstant(ForestDataset& dataset, ForestTracker& tracker, int tree_num, int node_id) {
  int p = dataset.GetCovariates().cols();
  data_size_t idx;
  double feature_value;
  double var_max;
  double var_min;
  
  for (int j = 0; j < p; j++) {
    auto node_begin_iter = tracker.UnsortedNodeBeginIterator(tree_num, node_id);
    auto node_end_iter = tracker.UnsortedNodeEndIterator(tree_num, node_id);
    var_max = std::numeric_limits<double>::min();
    var_min = std::numeric_limits<double>::max();

    for (auto i = node_begin_iter; i != node_end_iter; i++) {
      auto idx = *i;
      feature_value = dataset.CovariateValue(idx, j);
      if (var_max < feature_value) {
        var_max = feature_value;
      } else if (var_min > feature_value) {
        var_min = feature_value;
      }
    }
    if (var_max > var_min) {
      return true;
    }
  }
  return false;
}

static void AddSplitToModel(ForestTracker& tracker, ForestDataset& dataset, TreePrior& tree_prior, TreeSplit& split, std::mt19937& gen, Tree* tree, int tree_num, int leaf_node, int feature_split, bool keep_sorted = false) {
  // Use zeros as a "temporary" leaf values since we draw leaf parameters after tree sampling is complete
  int basis_dim = 1;
  if (dataset.HasBasis()) {
    if (dataset.GetBasis().cols() > 1) {
      std::vector<double> temp_leaf_values(basis_dim, 0.);
      tree->ExpandNode(leaf_node, feature_split, split, temp_leaf_values, temp_leaf_values);
    } else {
      double temp_leaf_value = 0.;
      tree->ExpandNode(leaf_node, feature_split, split, temp_leaf_value, temp_leaf_value);
    }
  } else {
    double temp_leaf_value = 0.;
    tree->ExpandNode(leaf_node, feature_split, split, temp_leaf_value, temp_leaf_value);
  }
  int left_node = tree->LeftChild(leaf_node);
  int right_node = tree->RightChild(leaf_node);

  // Update the ForestTracker
  tracker.AddSplit(dataset.GetCovariates(), split, feature_split, tree_num, leaf_node, left_node, right_node, keep_sorted);
}

static void RemoveSplitFromModel(ForestTracker& tracker, ForestDataset& dataset, TreePrior& tree_prior, std::mt19937& gen, Tree* tree, int tree_num, int leaf_node, int left_node, int right_node, bool keep_sorted = false) {
  // Use zeros as a "temporary" leaf values since we draw leaf parameters after tree sampling is complete
  int basis_dim = 1;
  if (dataset.HasBasis()) {
    if (dataset.GetBasis().cols() > 1) {
      std::vector<double> temp_leaf_values(basis_dim, 0.);
      tree->CollapseToLeaf(leaf_node, temp_leaf_values);
    } else {
      double temp_leaf_value = 0.;
      tree->CollapseToLeaf(leaf_node, temp_leaf_value);
    }
  } else {
    double temp_leaf_value = 0.;
    tree->CollapseToLeaf(leaf_node, temp_leaf_value);
  }

  // Update the ForestTracker
  tracker.RemoveSplit(dataset.GetCovariates(), tree, tree_num, leaf_node, left_node, right_node, keep_sorted);
}

static double ComputeMeanOutcome(ColumnVector& residual) {
  data_size_t n = residual.NumRows();
  double total_outcome = 0.;
  for (data_size_t i = 0; i < n; i++) {
    total_outcome += residual.GetElement(i);
  }
  return total_outcome / static_cast<double>(n);
}

static void UpdateResidualEntireForest(ForestTracker& tracker, ForestDataset& dataset, ColumnVector& residual, TreeEnsemble* forest, bool requires_basis, std::function<double(double, double)> op) {
  data_size_t n = dataset.GetCovariates().rows();
  double tree_pred = 0.;
  double pred_value = 0.;
  double new_resid = 0.;
  int32_t leaf_pred;
  for (data_size_t i = 0; i < n; i++) {
    for (int j = 0; j < forest->NumTrees(); j++) {
      Tree* tree = forest->GetTree(j);
      leaf_pred = tracker.GetNodeId(i, j);
      if (requires_basis) {
        tree_pred += tree->PredictFromNode(leaf_pred, dataset.GetBasis(), i);
      } else {
        tree_pred += tree->PredictFromNode(leaf_pred);
      }
      tracker.SetTreeSamplePrediction(i, j, tree_pred);
      pred_value += tree_pred;
    }
    
    // Run op (either plus or minus) on the residual and the new prediction
    new_resid = op(residual.GetElement(i), pred_value);
    residual.SetElement(i, new_resid);
  }
}

static void UpdateResidualTree(ForestTracker& tracker, ForestDataset& dataset, ColumnVector& residual, Tree* tree, int tree_num, bool requires_basis, std::function<double(double, double)> op, bool tree_new) {
  data_size_t n = dataset.GetCovariates().rows();
  double pred_value;
  int32_t leaf_pred;
  double new_resid;
  for (data_size_t i = 0; i < n; i++) {
    if (tree_new) {
      // If the tree has been newly sampled or adjusted, we must rerun the prediction 
      // method and update the SamplePredMapper stored in tracker
      leaf_pred = tracker.GetNodeId(i, tree_num);
      if (requires_basis) {
        pred_value = tree->PredictFromNode(leaf_pred, dataset.GetBasis(), i);
      } else {
        pred_value = tree->PredictFromNode(leaf_pred);
      }
      tracker.SetTreeSamplePrediction(i, tree_num, pred_value);
    } else {
      // If the tree has not yet been modified via a sampling step, 
      // we can query its prediction directly from the SamplePredMapper stored in tracker
      pred_value = tracker.GetTreeSamplePrediction(i, tree_num);
    }
    // Run op (either plus or minus) on the residual and the new prediction
    new_resid = op(residual.GetElement(i), pred_value);
    residual.SetElement(i, new_resid);
  }
}

template <typename LeafModel>
class MCMCForestSampler {
 public:
  MCMCForestSampler() {}
  ~MCMCForestSampler() {}
  
  void SampleOneIter(ForestTracker& tracker, ForestContainer& forests, LeafModel& leaf_model, ForestDataset& dataset, 
                     ColumnVector& residual, TreePrior& tree_prior, std::mt19937& gen, std::vector<double> variable_weights,
                     std::vector<int32_t>& variable_split_counts, double global_variance, bool pre_initialized = false) {
    // Previous number of samples
    int prev_num_samples = forests.NumSamples();
    
    if ((prev_num_samples == 0) && (!pre_initialized)) {
      // Add new forest to the container
      forests.AddSamples(1);
      
      // Set initial value for each leaf in the forest
      double root_pred = ComputeMeanOutcome(residual) / static_cast<double>(forests.NumTrees());
      TreeEnsemble* ensemble = forests.GetEnsemble(0);
      leaf_model.SetEnsembleRootPredictedValue(dataset, ensemble, root_pred);
    } else if (prev_num_samples > 0) {
      // Add new forest to the container
      forests.AddSamples(1);
      
      // Copy previous forest
      forests.CopyFromPreviousSample(prev_num_samples, prev_num_samples - 1);
    } else {
      forests.IncrementSampleCount();
    }
    
    // Run the MCMC algorithm for each tree
    TreeEnsemble* ensemble = forests.GetEnsemble(prev_num_samples);
    Tree* tree;
    int num_trees = forests.NumTrees();
    for (int i = 0; i < num_trees; i++) {
      // Add tree i's predictions back to the residual (thus, training a model on the "partial residual")
      tree = ensemble->GetTree(i);
      UpdateResidualTree(tracker, dataset, residual, tree, i, leaf_model.RequiresBasis(), plus_op_, false);
      
      // Sample tree i
      tree = ensemble->GetTree(i);
      SampleTreeOneIter(tree, tracker, forests, leaf_model, dataset, residual, tree_prior, gen, variable_weights, variable_split_counts, i, global_variance);
      
      // Sample leaf parameters for tree i
      tree = ensemble->GetTree(i);
      leaf_model.SampleLeafParameters(dataset, tracker, residual, tree, i, global_variance, gen);
      
      // Subtract tree i's predictions back out of the residual
      tree = ensemble->GetTree(i);
      UpdateResidualTree(tracker, dataset, residual, tree, i, leaf_model.RequiresBasis(), minus_op_, true);
    }
  }
 
 private:
  // Function objects for element-wise addition and subtraction (used in the residual update function which takes std::function as an argument)
  std::plus<double> plus_op_;
  std::minus<double> minus_op_;
  
  void SampleTreeOneIter(Tree* tree, ForestTracker& tracker, ForestContainer& forests, LeafModel& leaf_model, ForestDataset& dataset,
                         ColumnVector& residual, TreePrior& tree_prior, std::mt19937& gen, std::vector<double> variable_weights, 
                         std::vector<int32_t>& variable_split_counts,
                         int tree_num, double global_variance) {
    // Determine whether it is possible to grow any of the leaves
    bool grow_possible = false;
    std::vector<int> leaves = tree->GetLeaves();
    for (auto& leaf: leaves) {
      if (tracker.UnsortedNodeSize(tree_num, leaf) > 2 * tree_prior.GetMinSamplesLeaf()) {
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
      GrowTreeOneIter(tree, tracker, leaf_model, dataset, residual, tree_prior, gen, tree_num, variable_weights, variable_split_counts, global_variance, prob_grow);
    } else {
      PruneTreeOneIter(tree, tracker, leaf_model, dataset, residual, tree_prior, variable_split_counts, gen, tree_num, global_variance);
    }
  }

  void GrowTreeOneIter(Tree* tree, ForestTracker& tracker, LeafModel& leaf_model, ForestDataset& dataset, ColumnVector& residual, 
                       TreePrior& tree_prior, std::mt19937& gen, int tree_num, std::vector<double> variable_weights, 
                       std::vector<int32_t>& variable_split_counts,
                       double global_variance, double prob_grow_old) {
    // Extract dataset information
    data_size_t n = dataset.GetCovariates().rows();
    int basis_dim = 1;

    // Choose a leaf node at random
    int num_leaves = tree->NumLeaves();
    std::vector<int> leaves = tree->GetLeaves();
    std::vector<double> leaf_weights(num_leaves);
    std::fill(leaf_weights.begin(), leaf_weights.end(), 1.0/num_leaves);
    std::discrete_distribution<> leaf_dist(leaf_weights.begin(), leaf_weights.end());
    int leaf_chosen = leaves[leaf_dist(gen)];
    int leaf_depth = tree->GetDepth(leaf_chosen);

    // Select a split variable at random
    int p = dataset.GetCovariates().cols();
    CHECK_EQ(variable_weights.size(), p);
    // std::vector<double> var_weights(p);
    // std::fill(var_weights.begin(), var_weights.end(), 1.0/p);
    std::discrete_distribution<> var_dist(variable_weights.begin(), variable_weights.end());
    int var_chosen = var_dist(gen);
    
    
    // Determine the range of possible cutpoints
    // TODO: specialize this for binary / ordered categorical / unordered categorical variables
    double var_min, var_max;
    VarSplitRange(tracker, dataset, tree_num, leaf_chosen, var_chosen, var_min, var_max);
    if (var_max <= var_min) {
      return;
    }
    
    // Split based on var_min to var_max in a given node
    std::uniform_real_distribution<double> split_point_dist(var_min, var_max);
    double split_point_chosen = split_point_dist(gen);

    // Create a split object
    TreeSplit split = TreeSplit(split_point_chosen);

    // Compute the marginal likelihood of split and no split, given the leaf prior
    std::tuple<double, double, int32_t, int32_t> split_eval = leaf_model.EvaluateProposedSplit(dataset, tracker, residual, split, tree_num, leaf_chosen, var_chosen, global_variance);
    double split_log_marginal_likelihood = std::get<0>(split_eval);
    double no_split_log_marginal_likelihood = std::get<1>(split_eval);
    int32_t left_n = std::get<2>(split_eval);
    int32_t right_n = std::get<3>(split_eval);
    
    // Determine probability of growing the split node and its two new left and right nodes
    double pg = tree_prior.GetAlpha() * std::pow(1+leaf_depth, -tree_prior.GetBeta());
    double pgl = tree_prior.GetAlpha() * std::pow(1+leaf_depth+1, -tree_prior.GetBeta());
    double pgr = tree_prior.GetAlpha() * std::pow(1+leaf_depth+1, -tree_prior.GetBeta());

    // Determine whether a "grow" move is possible from the newly formed tree
    // in order to compute the probability of choosing "prune" from the new tree
    // (which is always possible by construction)
    bool non_constant = NodesNonConstantAfterSplit(dataset, tracker, split, tree_num, leaf_chosen, var_chosen);
    bool min_samples_left_check = left_n >= 2*tree_prior.GetMinSamplesLeaf();
    bool min_samples_right_check = right_n >= 2*tree_prior.GetMinSamplesLeaf();
    double prob_prune_new;
    if (non_constant && (min_samples_left_check || min_samples_right_check)) {
      prob_prune_new = 0.5;
    } else {
      prob_prune_new = 1.0;
    }

    // Determine the number of leaves in the current tree and leaf parents in the proposed tree
    int num_leaf_parents = tree->NumLeafParents();
    double p_leaf = 1/static_cast<double>(num_leaves);
    double p_leaf_parent = 1/static_cast<double>(num_leaf_parents+1);

    // Compute the final MH ratio
    double log_mh_ratio = (
      std::log(pg) + std::log(1-pgl) + std::log(1-pgr) - std::log(1-pg) + std::log(prob_prune_new) +
      std::log(p_leaf_parent) - std::log(prob_grow_old) - std::log(p_leaf) - no_split_log_marginal_likelihood + split_log_marginal_likelihood
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
      variable_split_counts.at(var_chosen)++;
      AddSplitToModel(tracker, dataset, tree_prior, split, gen, tree, tree_num, leaf_chosen, var_chosen, false);
    } else {
      accept = false;
    }
  }

  void PruneTreeOneIter(Tree* tree, ForestTracker& tracker, LeafModel& leaf_model, ForestDataset& dataset, ColumnVector& residual, 
                        TreePrior& tree_prior, std::vector<int32_t>& variable_split_counts, std::mt19937& gen, int tree_num, double global_variance) {
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
    
    // Compute the marginal likelihood for the leaf parent and its left and right nodes
    std::tuple<double, double, int32_t, int32_t> split_eval = leaf_model.EvaluateExistingSplit(dataset, tracker, residual, global_variance, tree_num, leaf_parent_chosen, left_node, right_node);
    double split_log_marginal_likelihood = std::get<0>(split_eval);
    double no_split_log_marginal_likelihood = std::get<1>(split_eval);
    int32_t left_n = std::get<2>(split_eval);
    int32_t right_n = std::get<3>(split_eval);
    
    // Determine probability of growing the split node and its two new left and right nodes
    double pg = tree_prior.GetAlpha() * std::pow(1+leaf_parent_depth, -tree_prior.GetBeta());
    double pgl = tree_prior.GetAlpha() * std::pow(1+leaf_parent_depth+1, -tree_prior.GetBeta());
    double pgr = tree_prior.GetAlpha() * std::pow(1+leaf_parent_depth+1, -tree_prior.GetBeta());

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
    bool non_constant_left = NodeNonConstant(dataset, tracker, tree_num, left_node);
    bool non_constant_right = NodeNonConstant(dataset, tracker, tree_num, right_node);
    double prob_prune_old;
    if (non_constant_left && non_constant_right) {
      prob_prune_old = 0.5;
    } else {
      prob_prune_old = 1.0;
    }

    // Determine the number of leaves in the current tree and leaf parents in the proposed tree
    double p_leaf = 1/static_cast<double>(num_leaves-1);
    double p_leaf_parent = 1/static_cast<double>(num_leaf_parents);

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
      variable_split_counts.at(feature_split)--;
      RemoveSplitFromModel(tracker, dataset, tree_prior, gen, tree, tree_num, leaf_parent_chosen, left_node, right_node, false);
    } else {
      accept = false;
    }
  }
};

template <typename LeafModel>
class GFRForestSampler {
 public:
  GFRForestSampler() {cutpoint_grid_size_ = 500;}
  GFRForestSampler(int cutpoint_grid_size) {cutpoint_grid_size_ = cutpoint_grid_size;}
  ~GFRForestSampler() {}

  void SampleOneIter(ForestTracker& tracker, ForestContainer& forests, LeafModel& leaf_model, ForestDataset& dataset, 
                     ColumnVector& residual, TreePrior& tree_prior, std::mt19937& gen, std::vector<double> variable_weights, 
                     double global_variance, std::vector<FeatureType>& feature_types, bool pre_initialized = false) {
    // Previous number of samples
    int prev_num_samples = forests.NumSamples();
    
    if ((prev_num_samples == 0) && (!pre_initialized)) {
      // Add new forest to the container
      forests.AddSamples(1);
      
      // Set initial value for each leaf in the forest
      double root_pred = ComputeMeanOutcome(residual) / static_cast<double>(forests.NumTrees());
      TreeEnsemble* ensemble = forests.GetEnsemble(0);
      leaf_model.SetEnsembleRootPredictedValue(dataset, ensemble, root_pred);
    } else if (prev_num_samples > 0) {
      // Add new forest to the container
      forests.AddSamples(1);

      // NOTE: only doing this for the simplicity of the partial residual step
      // We could alternatively "reach back" to the tree predictions from a previous
      // sample (whenever there is more than one sample). This is cleaner / quicker
      // to implement during this refactor.
      forests.CopyFromPreviousSample(prev_num_samples, prev_num_samples - 1);
    } else {
      forests.IncrementSampleCount();
    }
    
    // Run the GFR algorithm for each tree
    TreeEnsemble* ensemble = forests.GetEnsemble(prev_num_samples);
    int num_trees = forests.NumTrees();
    for (int i = 0; i < num_trees; i++) {
      // Add tree i's predictions back to the residual (thus, training a model on the "partial residual")
      Tree* tree = ensemble->GetTree(i);
      UpdateResidualTree(tracker, dataset, residual, tree, i, leaf_model.RequiresBasis(), plus_op_, false);
      
      // Reset the tree and sample trackers
      ensemble->ResetInitTree(i);
      tracker.ResetRoot(dataset.GetCovariates(), feature_types, i);
      tree = ensemble->GetTree(i);
      
      // Sample tree i
      SampleTreeOneIter(tree, tracker, forests, leaf_model, dataset, residual, tree_prior, gen, i, global_variance, feature_types);
      
      // Sample leaf parameters for tree i
      tree = ensemble->GetTree(i);
      leaf_model.SampleLeafParameters(dataset, tracker, residual, tree, i, global_variance, gen);
      
      // Subtract tree i's predictions back out of the residual
      UpdateResidualTree(tracker, dataset, residual, tree, i, leaf_model.RequiresBasis(), minus_op_, true);
    }
  }

 private:
  // Maximum cutpoint grid size in the enumeration of possible splits
  int cutpoint_grid_size_;
  
  // Function objects for element-wise addition and subtraction (used in the residual update function which takes std::function as an argument)
  std::plus<double> plus_op_;
  std::minus<double> minus_op_;
  
  void SampleTreeOneIter(Tree* tree, ForestTracker& tracker, ForestContainer& forests, LeafModel& leaf_model, ForestDataset& dataset,
                         ColumnVector& residual, TreePrior& tree_prior, std::mt19937& gen, int tree_num, double global_variance, 
                         std::vector<FeatureType>& feature_types) {
    int root_id = Tree::kRoot;
    int curr_node_id;
    data_size_t curr_node_begin;
    data_size_t curr_node_end;
    data_size_t n = dataset.GetCovariates().rows();
    // Mapping from node id to start and end points of sorted indices
    std::unordered_map<int, std::pair<data_size_t, data_size_t>> node_index_map;
    node_index_map.insert({root_id, std::make_pair(0, n)});
    std::pair<data_size_t, data_size_t> begin_end;
    // Add root node to the split queue
    std::deque<node_t> split_queue;
    split_queue.push_back(Tree::kRoot);
    // Run the "GrowFromRoot" procedure using a stack in place of recursion
    while (!split_queue.empty()) {
      // Remove the next node from the queue
      curr_node_id = split_queue.front();
      split_queue.pop_front();
      // Determine the beginning and ending indices of the left and right nodes
      begin_end = node_index_map[curr_node_id];
      curr_node_begin = begin_end.first;
      curr_node_end = begin_end.second;
      // Draw a split rule at random
      SampleSplitRule(tree, tracker, leaf_model, dataset, residual, tree_prior, gen, tree_num, global_variance, cutpoint_grid_size_, 
                      node_index_map, split_queue, curr_node_id, curr_node_begin, curr_node_end, feature_types);
    }
  }

  void SampleSplitRule(Tree* tree, ForestTracker& tracker, LeafModel& leaf_model, ForestDataset& dataset, ColumnVector& residual, 
                       TreePrior& tree_prior, std::mt19937& gen, int tree_num, double global_variance, int cutpoint_grid_size, 
                       std::unordered_map<int, std::pair<data_size_t, data_size_t>>& node_index_map, std::deque<node_t>& split_queue, 
                       int node_id, data_size_t node_begin, data_size_t node_end, std::vector<FeatureType>& feature_types) {
    std::vector<double> log_cutpoint_evaluations;
    std::vector<int> cutpoint_features;
    std::vector<double> cutpoint_values;
    std::vector<FeatureType> cutpoint_feature_types;
    StochTree::data_size_t valid_cutpoint_count;
    CutpointGridContainer cutpoint_grid_container(dataset.GetCovariates(), residual.GetData(), cutpoint_grid_size);
    EvaluateCutpoints(tree, tracker, leaf_model, dataset, residual, tree_prior, gen, tree_num, global_variance,
                      cutpoint_grid_size, node_id, node_begin, node_end, log_cutpoint_evaluations, cutpoint_features, 
                      cutpoint_values, cutpoint_feature_types, valid_cutpoint_count, feature_types, cutpoint_grid_container);
    // TODO: maybe add some checks here?
    
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
      
      // Compute node sample size
      data_size_t node_n = node_end - node_begin;
      
      // Actual numeric cutpoint used for ordered categorical and numeric features
      double split_value_numeric;
      TreeSplit tree_split;
      
      // We will use these later in the model expansion
      data_size_t left_n = 0;
      data_size_t right_n = 0;
      data_size_t sort_idx;
      double feature_value;
      bool split_true;

      if (feature_type == FeatureType::kUnorderedCategorical) {
        // Determine the number of categories available in a categorical split and the set of categories that route observations to the left node after split
        int num_categories;
        std::vector<std::uint32_t> categories = cutpoint_grid_container.CutpointVector(static_cast<std::uint32_t>(split_value), feature_split);
        tree_split = TreeSplit(categories);
      } else if (feature_type == FeatureType::kOrderedCategorical) {
        // Convert the bin split to an actual split value
        split_value_numeric = cutpoint_grid_container.CutpointValue(static_cast<std::uint32_t>(split_value), feature_split);
        tree_split = TreeSplit(split_value_numeric);
      } else if (feature_type == FeatureType::kNumeric) {
        // Convert the bin split to an actual split value
        split_value_numeric = cutpoint_grid_container.CutpointValue(static_cast<std::uint32_t>(split_value), feature_split);
        tree_split = TreeSplit(split_value_numeric);
      } else {
        Log::Fatal("Invalid split type");
      }
      
      // Add split to tree and trackers
      AddSplitToModel(tracker, dataset, tree_prior, tree_split, gen, tree, tree_num, node_id, feature_split, true);

      // Determine the number of observation in the newly created left node
      int left_node = tree->LeftChild(node_id);
      int right_node = tree->RightChild(node_id);
      auto left_begin_iter = tracker.SortedNodeBeginIterator(left_node, feature_split);
      auto left_end_iter = tracker.SortedNodeEndIterator(left_node, feature_split);
      for (auto i = left_begin_iter; i < left_end_iter; i++) {
        left_n += 1;
      }

      // Add the begin and end indices for the new left and right nodes to node_index_map
      node_index_map.insert({left_node, std::make_pair(node_begin, node_begin + left_n)});
      node_index_map.insert({right_node, std::make_pair(node_begin + left_n, node_end)});

      // Add the left and right nodes to the split tracker
      split_queue.push_front(right_node);
      split_queue.push_front(left_node);      
    }
  }

  void EvaluateCutpoints(Tree* tree, ForestTracker& tracker, LeafModel& leaf_model, ForestDataset& dataset, ColumnVector& residual, TreePrior& tree_prior, 
                         std::mt19937& gen, int tree_num, double global_variance, int cutpoint_grid_size, int node_id, data_size_t node_begin, data_size_t node_end, 
                         std::vector<double>& log_cutpoint_evaluations, std::vector<int>& cutpoint_features, std::vector<double>& cutpoint_values, 
                         std::vector<FeatureType>& cutpoint_feature_types, data_size_t& valid_cutpoint_count, std::vector<FeatureType> feature_types, CutpointGridContainer& cutpoint_grid_container) {
    // Evaluate all possible cutpoints according to the leaf node model, 
    // recording their log-likelihood and other split information in a series of vectors.
    // The last element of these vectors concerns the "no-split" option.
    leaf_model.EvaluateAllPossibleSplits(dataset, tracker, residual, tree_prior, global_variance, tree_num, node_id, log_cutpoint_evaluations, 
                                         cutpoint_features, cutpoint_values, cutpoint_feature_types, valid_cutpoint_count, 
                                         cutpoint_grid_container, node_begin, node_end, feature_types);

    // Compute an adjustment to reflect the no split prior probability and the number of cutpoints
    double bart_prior_no_split_adj;
    double alpha = tree_prior.GetAlpha();
    double beta = tree_prior.GetBeta();
    int node_depth = tree->GetDepth(node_id);
    if (valid_cutpoint_count == 0) {
      bart_prior_no_split_adj = std::log(((std::pow(1+node_depth, beta))/alpha) - 1.0);
    } else {
      bart_prior_no_split_adj = std::log(((std::pow(1+node_depth, beta))/alpha) - 1.0) + std::log(valid_cutpoint_count);
    }
    log_cutpoint_evaluations[log_cutpoint_evaluations.size()-1] += bart_prior_no_split_adj;
  }

};

} // namespace StochTree

#endif // STOCHTREE_TREE_SAMPLER_H_
