/*! Copyright (c) 2024 stochtree authors. All rights reserved. */
#ifndef STOCHTREE_TREE_SAMPLER_H_
#define STOCHTREE_TREE_SAMPLER_H_

#include <stochtree/container.h>
#include <stochtree/cutpoint_candidates.h>
#include <stochtree/data.h>
#include <stochtree/ensemble.h>
#include <stochtree/leaf_model.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/prior.h>

#include <cmath>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace StochTree {

static inline void VarSplitRange(ForestTracker& tracker, ForestDataset& dataset, int tree_num, int leaf_split, int feature_split, double& var_min, double& var_max) {
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

static inline bool NodesNonConstantAfterSplit(ForestDataset& dataset, ForestTracker& tracker, TreeSplit& split, int tree_num, int leaf_split, int feature_split) {
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

static inline bool NodeNonConstant(ForestDataset& dataset, ForestTracker& tracker, int tree_num, int node_id) {
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

static inline void AddSplitToModel(ForestTracker& tracker, ForestDataset& dataset, TreePrior& tree_prior, TreeSplit& split, std::mt19937& gen, Tree* tree, 
                                   int tree_num, int leaf_node, int feature_split, bool keep_sorted = false) {
  // Use zeros as a "temporary" leaf values since we draw leaf parameters after tree sampling is complete
  if (tree->OutputDimension() > 1) {
    std::vector<double> temp_leaf_values(tree->OutputDimension(), 0.);
    tree->ExpandNode(leaf_node, feature_split, split, temp_leaf_values, temp_leaf_values);
  } else {
    double temp_leaf_value = 0.;
    tree->ExpandNode(leaf_node, feature_split, split, temp_leaf_value, temp_leaf_value);
  }
  int left_node = tree->LeftChild(leaf_node);
  int right_node = tree->RightChild(leaf_node);

  // Update the ForestTracker
  tracker.AddSplit(dataset.GetCovariates(), split, feature_split, tree_num, leaf_node, left_node, right_node, keep_sorted);
}

static inline void RemoveSplitFromModel(ForestTracker& tracker, ForestDataset& dataset, TreePrior& tree_prior, std::mt19937& gen, Tree* tree, 
                                        int tree_num, int leaf_node, int left_node, int right_node, bool keep_sorted = false) {
  // Use zeros as a "temporary" leaf values since we draw leaf parameters after tree sampling is complete
  if (tree->OutputDimension() > 1) {
    std::vector<double> temp_leaf_values(tree->OutputDimension(), 0.);
    tree->CollapseToLeaf(leaf_node, temp_leaf_values);
  } else {
    double temp_leaf_value = 0.;
    tree->CollapseToLeaf(leaf_node, temp_leaf_value);
  }

  // Update the ForestTracker
  tracker.RemoveSplit(dataset.GetCovariates(), tree, tree_num, leaf_node, left_node, right_node, keep_sorted);
}

static inline double ComputeMeanOutcome(ColumnVector& residual) {
  int n = residual.NumRows();
  double sum_y = 0.;
  double y;
  for (data_size_t i = 0; i < n; i++) {
    y = residual.GetElement(i);
    sum_y += y;
  }
  return sum_y / static_cast<double>(n);
}

static inline double ComputeVarianceOutcome(ColumnVector& residual) {
  int n = residual.NumRows();
  double sum_y = 0.;
  double sum_y_sq = 0.;
  double y;
  for (data_size_t i = 0; i < n; i++) {
    y = residual.GetElement(i);
    sum_y += y;
    sum_y_sq += y * y;
  }
  return sum_y_sq / static_cast<double>(n) - (sum_y * sum_y) / (static_cast<double>(n) * static_cast<double>(n));
}

static inline void UpdateModelVarianceForest(ForestTracker& tracker, ForestDataset& dataset, ColumnVector& residual, 
                                             TreeEnsemble* forest, bool requires_basis, std::function<double(double, double)> op) {
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
  tracker.SyncPredictions();
}

static inline void UpdateResidualNoTrackerUpdate(ForestTracker& tracker, ForestDataset& dataset, ColumnVector& residual, TreeEnsemble* forest, 
                                                 bool requires_basis, std::function<double(double, double)> op) {
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
      pred_value += tree_pred;
    }
    
    // Run op (either plus or minus) on the residual and the new prediction
    new_resid = op(residual.GetElement(i), pred_value);
    residual.SetElement(i, new_resid);
  }
}

static inline void UpdateResidualEntireForest(ForestTracker& tracker, ForestDataset& dataset, ColumnVector& residual, TreeEnsemble* forest, 
                                              bool requires_basis, std::function<double(double, double)> op) {
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
  tracker.SyncPredictions();
}

static inline void UpdateResidualNewOutcome(ForestTracker& tracker, ColumnVector& residual) {
  data_size_t n = residual.NumRows();
  double pred_value;
  double prev_outcome;
  double new_resid;
  for (data_size_t i = 0; i < n; i++) {
    prev_outcome = residual.GetElement(i);
    pred_value = tracker.GetSamplePrediction(i);
    // Run op (either plus or minus) on the residual and the new prediction
    new_resid = prev_outcome - pred_value;
    residual.SetElement(i, new_resid);
  }
}

static inline void UpdateMeanModelTree(ForestTracker& tracker, ForestDataset& dataset, ColumnVector& residual, Tree* tree, int tree_num, 
                                      bool requires_basis, std::function<double(double, double)> op, bool tree_new) {
  data_size_t n = dataset.GetCovariates().rows();
  double pred_value;
  int32_t leaf_pred;
  double new_resid;
  double pred_delta;
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
      pred_delta = pred_value - tracker.GetTreeSamplePrediction(i, tree_num);
      tracker.SetTreeSamplePrediction(i, tree_num, pred_value);
      tracker.SetSamplePrediction(i, tracker.GetSamplePrediction(i) + pred_delta);
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

static inline void UpdateResidualNewBasis(ForestTracker& tracker, ForestDataset& dataset, ColumnVector& residual, TreeEnsemble* forest) {
  CHECK(dataset.HasBasis());
  data_size_t n = dataset.GetCovariates().rows();
  int num_trees = forest->NumTrees();
  double prev_tree_pred;
  double new_tree_pred;
  int32_t leaf_pred;
  double new_resid;
  for (int tree_num = 0; tree_num < num_trees; tree_num++) {
    Tree* tree = forest->GetTree(tree_num);
    for (data_size_t i = 0; i < n; i++) {
      // Add back the currently stored tree prediction
      prev_tree_pred = tracker.GetTreeSamplePrediction(i, tree_num);
      new_resid = residual.GetElement(i) + prev_tree_pred;

      // Compute new prediction based on updated basis
      leaf_pred = tracker.GetNodeId(i, tree_num);
      new_tree_pred = tree->PredictFromNode(leaf_pred, dataset.GetBasis(), i);
      
      // Cache the new prediction in the tracker
      tracker.SetTreeSamplePrediction(i, tree_num, new_tree_pred);

      // Subtract out the updated tree prediction
      new_resid -= new_tree_pred;
      
      // Propagate the change back to the residual
      residual.SetElement(i, new_resid);
    }
  }
  tracker.SyncPredictions();
}

static inline void UpdateVarModelTree(ForestTracker& tracker, ForestDataset& dataset, ColumnVector& residual, Tree* tree, 
                                      int tree_num, bool requires_basis, std::function<double(double, double)> op, bool tree_new) {
  data_size_t n = dataset.GetCovariates().rows();
  double pred_value;
  int32_t leaf_pred;
  double new_weight;
  double pred_delta;
  double prev_tree_pred;
  double prev_pred;
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
      prev_tree_pred = tracker.GetTreeSamplePrediction(i, tree_num);
      prev_pred = tracker.GetSamplePrediction(i);
      pred_delta = pred_value - prev_tree_pred;
      tracker.SetTreeSamplePrediction(i, tree_num, pred_value);
      tracker.SetSamplePrediction(i, prev_pred + pred_delta);
      new_weight = std::log(dataset.VarWeightValue(i)) + pred_value;
      dataset.SetVarWeightValue(i, new_weight, true);
    } else {
      // If the tree has not yet been modified via a sampling step, 
      // we can query its prediction directly from the SamplePredMapper stored in tracker
      pred_value = tracker.GetTreeSamplePrediction(i, tree_num);
      new_weight = std::log(dataset.VarWeightValue(i)) - pred_value;
      dataset.SetVarWeightValue(i, new_weight, true);
    }
  }
}

template <typename LeafModel, typename LeafSuffStat, typename... LeafSuffStatConstructorArgs>
static inline std::tuple<double, double, data_size_t, data_size_t> EvaluateProposedSplit(
  ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual, LeafModel& leaf_model, 
  TreeSplit& split, int tree_num, int leaf_num, int split_feature, double global_variance, 
  LeafSuffStatConstructorArgs&... leaf_suff_stat_args
) {
  // Initialize sufficient statistics
  LeafSuffStat node_suff_stat = LeafSuffStat(leaf_suff_stat_args...);
  LeafSuffStat left_suff_stat = LeafSuffStat(leaf_suff_stat_args...);
  LeafSuffStat right_suff_stat = LeafSuffStat(leaf_suff_stat_args...);

  // Accumulate sufficient statistics
  AccumulateSuffStatProposed<LeafSuffStat>(node_suff_stat, left_suff_stat, right_suff_stat, dataset, tracker, 
                                           residual, global_variance, split, tree_num, leaf_num, split_feature);
  data_size_t left_n = left_suff_stat.n;
  data_size_t right_n = right_suff_stat.n;

  // Evaluate split
  double split_log_ml = leaf_model.SplitLogMarginalLikelihood(left_suff_stat, right_suff_stat, global_variance);
  double no_split_log_ml = leaf_model.NoSplitLogMarginalLikelihood(node_suff_stat, global_variance);

  return std::tuple<double, double, data_size_t, data_size_t>(split_log_ml, no_split_log_ml, left_n, right_n);
}

template <typename LeafModel, typename LeafSuffStat, typename... LeafSuffStatConstructorArgs>
static inline std::tuple<double, double, data_size_t, data_size_t> EvaluateExistingSplit(
  ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual, LeafModel& leaf_model, 
  double global_variance, int tree_num, int split_node_id, int left_node_id, int right_node_id, 
  LeafSuffStatConstructorArgs&... leaf_suff_stat_args
) {
  // Initialize sufficient statistics
  LeafSuffStat node_suff_stat = LeafSuffStat(leaf_suff_stat_args...);
  LeafSuffStat left_suff_stat = LeafSuffStat(leaf_suff_stat_args...);
  LeafSuffStat right_suff_stat = LeafSuffStat(leaf_suff_stat_args...);

  // Accumulate sufficient statistics
  AccumulateSuffStatExisting<LeafSuffStat>(node_suff_stat, left_suff_stat, right_suff_stat, dataset, tracker, 
                                           residual, global_variance, tree_num, split_node_id, left_node_id, right_node_id);
  data_size_t left_n = left_suff_stat.n;
  data_size_t right_n = right_suff_stat.n;

  // Evaluate split
  double split_log_ml = leaf_model.SplitLogMarginalLikelihood(left_suff_stat, right_suff_stat, global_variance);
  double no_split_log_ml = leaf_model.NoSplitLogMarginalLikelihood(node_suff_stat, global_variance);

  return std::tuple<double, double, data_size_t, data_size_t>(split_log_ml, no_split_log_ml, left_n, right_n);
}

// template <typename LeafModel>
// static inline void ModelInitialization(ForestTracker& tracker, ForestContainer& forests, LeafModel& leaf_model,
//                                        ForestDataset& dataset, ColumnVector& residual, TreePrior& tree_prior,
//                                        std::mt19937& gen, std::vector<double>& variable_weights, double global_variance,
//                                        bool pre_initialized, bool backfitting, int prev_num_samples, bool var_trees = false) {
//   if ((prev_num_samples == 0) && (!pre_initialized)) {
//     // Add new forest to the container
//     forests.AddSamples(1);
    
//     // Set initial value for each leaf in the forest
//     double leaf_value;
//     if (var_trees) {
//       leaf_value = std::log(ComputeVarianceOutcome(residual)) / static_cast<double>(forests.NumTrees());
//     } else {
//       leaf_value = ComputeMeanOutcome(residual) / static_cast<double>(forests.NumTrees());
//     }
//     TreeEnsemble* ensemble = forests.GetEnsemble(0);
//     leaf_model.SetEnsembleRootPredictedValue(dataset, ensemble, leaf_value);
//     tracker.AssignAllSamplesToConstantPrediction(leaf_value);
//   } else if (prev_num_samples > 0) {
//     // Add new forest to the container
//     forests.AddSamples(1);

//     // NOTE: only doing this for the simplicity of the partial residual step
//     // We could alternatively "reach back" to the tree predictions from a previous
//     // sample (whenever there is more than one sample). This is cleaner / quicker
//     // to implement during this refactor.
//     forests.CopyFromPreviousSample(prev_num_samples, prev_num_samples - 1);
//   } else {
//     forests.IncrementSampleCount();
//   }
// }

template <typename LeafModel>
static inline void AdjustStateBeforeTreeSampling(ForestTracker& tracker, LeafModel& leaf_model, ForestDataset& dataset, 
                                                 ColumnVector& residual, TreePrior& tree_prior, bool backfitting, Tree* tree, int tree_num) {
  if (backfitting) {
    UpdateMeanModelTree(tracker, dataset, residual, tree, tree_num, leaf_model.RequiresBasis(), std::plus<double>(), false);
  } else {
    // TODO: think about a generic way to store "state" corresponding to the other models?
    UpdateVarModelTree(tracker, dataset, residual, tree, tree_num, leaf_model.RequiresBasis(), std::minus<double>(), false);
  }
}

template <typename LeafModel>
static inline void AdjustStateAfterTreeSampling(ForestTracker& tracker, LeafModel& leaf_model, ForestDataset& dataset, 
                                                ColumnVector& residual, TreePrior& tree_prior, bool backfitting, Tree* tree, int tree_num) {
  if (backfitting) {
    UpdateMeanModelTree(tracker, dataset, residual, tree, tree_num, leaf_model.RequiresBasis(), std::minus<double>(), true);
  } else {
    // TODO: think about a generic way to store "state" corresponding to the other models?
    UpdateVarModelTree(tracker, dataset, residual, tree, tree_num, leaf_model.RequiresBasis(), std::plus<double>(), true);
  }
}

template <typename LeafModel, typename LeafSuffStat, typename... LeafSuffStatConstructorArgs>
static inline void EvaluateAllPossibleSplits(
  ForestDataset& dataset, ForestTracker& tracker, ColumnVector& residual, TreePrior& tree_prior, LeafModel& leaf_model, double global_variance, int tree_num, int split_node_id, 
  std::vector<double>& log_cutpoint_evaluations, std::vector<int>& cutpoint_features, std::vector<double>& cutpoint_values, std::vector<FeatureType>& cutpoint_feature_types, 
  data_size_t& valid_cutpoint_count, CutpointGridContainer& cutpoint_grid_container, data_size_t node_begin, data_size_t node_end, std::vector<double>& variable_weights, 
  std::vector<FeatureType>& feature_types, LeafSuffStatConstructorArgs&... leaf_suff_stat_args
) {
    // Initialize sufficient statistics
  LeafSuffStat node_suff_stat = LeafSuffStat(leaf_suff_stat_args...);
  LeafSuffStat left_suff_stat = LeafSuffStat(leaf_suff_stat_args...);
  LeafSuffStat right_suff_stat = LeafSuffStat(leaf_suff_stat_args...);

  // Accumulate aggregate sufficient statistic for the node to be split
  AccumulateSingleNodeSuffStat<LeafSuffStat, false>(node_suff_stat, dataset, tracker, residual, tree_num, split_node_id);

  // Compute the "no split" log marginal likelihood
  double no_split_log_ml = leaf_model.NoSplitLogMarginalLikelihood(node_suff_stat, global_variance);

  // Unpack data
  Eigen::MatrixXd covariates = dataset.GetCovariates();
  Eigen::VectorXd outcome = residual.GetData();
  Eigen::VectorXd var_weights;
  bool has_weights = dataset.HasVarWeights();
  if (has_weights) var_weights = dataset.GetVarWeights();
  
  // Minimum size of newly created leaf nodes (used to rule out invalid splits)
  int32_t min_samples_in_leaf = tree_prior.GetMinSamplesLeaf();

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
  double split_log_ml;
  for (int j = 0; j < covariates.cols(); j++) {

    if (std::abs(variable_weights.at(j)) > kEpsilon) {
      // Enumerate cutpoint strides
      cutpoint_grid_container.CalculateStrides(covariates, outcome, tracker.GetSortedNodeSampleTracker(), split_node_id, node_begin, node_end, j, feature_types);
      
      // Reset sufficient statistics
      left_suff_stat.ResetSuffStat();
      right_suff_stat.ResetSuffStat();

      // Iterate through possible cutpoints
      int32_t num_feature_cutpoints = cutpoint_grid_container.NumCutpoints(j);
      feature_type = feature_types[j];
      // Since we partition an entire cutpoint bin to the left, we must stop one bin before the total number of cutpoint bins
      for (data_size_t cutpoint_idx = 0; cutpoint_idx < (num_feature_cutpoints - 1); cutpoint_idx++) {
        current_bin_begin = cutpoint_grid_container.BinStartIndex(cutpoint_idx, j);
        current_bin_size = cutpoint_grid_container.BinLength(cutpoint_idx, j);
        next_bin_begin = cutpoint_grid_container.BinStartIndex(cutpoint_idx + 1, j);

        // Accumulate sufficient statistics for the left node
        AccumulateCutpointBinSuffStat<LeafSuffStat>(left_suff_stat, tracker, cutpoint_grid_container, dataset, residual,
                                                    global_variance, tree_num, split_node_id, j, cutpoint_idx);

        // Compute the corresponding right node sufficient statistics
        right_suff_stat.SubtractSuffStat(node_suff_stat, left_suff_stat);

        // Store the bin index as the "cutpoint value" - we can use this to query the actual split 
        // value or the set of split categories later on once a split is chose
        cutoff_value = cutpoint_idx;

        // Only include cutpoint for consideration if it defines a valid split in the training data
        valid_split = (left_suff_stat.SampleGreaterThanEqual(min_samples_in_leaf) && 
                      right_suff_stat.SampleGreaterThanEqual(min_samples_in_leaf));
        if (valid_split) {
          num_cutpoints++;
          // Add to split rule vector
          cutpoint_feature_types.push_back(feature_type);
          cutpoint_features.push_back(j);
          cutpoint_values.push_back(cutoff_value);
          // Add the log marginal likelihood of the split to the split eval vector 
          split_log_ml = leaf_model.SplitLogMarginalLikelihood(left_suff_stat, right_suff_stat, global_variance);
          log_cutpoint_evaluations.push_back(split_log_ml);
        }
      }
    }

  }

  // Add the log marginal likelihood of the "no-split" option (adjusted for tree prior and cutpoint size per the XBART paper)
  cutpoint_features.push_back(-1);
  cutpoint_values.push_back(std::numeric_limits<double>::max());
  cutpoint_feature_types.push_back(FeatureType::kNumeric);
  log_cutpoint_evaluations.push_back(no_split_log_ml);

  // Update valid cutpoint count
  valid_cutpoint_count = num_cutpoints;
}

template <typename LeafModel, typename LeafSuffStat, typename... LeafSuffStatConstructorArgs>
static inline void EvaluateCutpoints(Tree* tree, ForestTracker& tracker, LeafModel& leaf_model, ForestDataset& dataset, ColumnVector& residual, TreePrior& tree_prior, 
                                     std::mt19937& gen, int tree_num, double global_variance, int cutpoint_grid_size, int node_id, data_size_t node_begin, data_size_t node_end, 
                                     std::vector<double>& log_cutpoint_evaluations, std::vector<int>& cutpoint_features, std::vector<double>& cutpoint_values, 
                                     std::vector<FeatureType>& cutpoint_feature_types, data_size_t& valid_cutpoint_count, std::vector<double>& variable_weights, 
                                     std::vector<FeatureType>& feature_types, CutpointGridContainer& cutpoint_grid_container, LeafSuffStatConstructorArgs&... leaf_suff_stat_args) {
  // Evaluate all possible cutpoints according to the leaf node model, 
  // recording their log-likelihood and other split information in a series of vectors.
  // The last element of these vectors concerns the "no-split" option.
  EvaluateAllPossibleSplits<LeafModel, LeafSuffStat, LeafSuffStatConstructorArgs...>(
    dataset, tracker, residual, tree_prior, leaf_model, global_variance, tree_num, node_id, log_cutpoint_evaluations, 
    cutpoint_features, cutpoint_values, cutpoint_feature_types, valid_cutpoint_count, cutpoint_grid_container, 
    node_begin, node_end, variable_weights, feature_types, leaf_suff_stat_args...
  );
  
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

template <typename LeafModel, typename LeafSuffStat, typename... LeafSuffStatConstructorArgs>
static inline void SampleSplitRule(Tree* tree, ForestTracker& tracker, LeafModel& leaf_model, ForestDataset& dataset, ColumnVector& residual, 
                                   TreePrior& tree_prior, std::mt19937& gen, int tree_num, double global_variance, int cutpoint_grid_size, 
                                   std::unordered_map<int, std::pair<data_size_t, data_size_t>>& node_index_map, std::deque<node_t>& split_queue, 
                                   int node_id, data_size_t node_begin, data_size_t node_end, std::vector<double>& variable_weights, 
                                   std::vector<FeatureType>& feature_types, LeafSuffStatConstructorArgs&... leaf_suff_stat_args) {
  // Leaf depth
  int leaf_depth = tree->GetDepth(node_id);

  // Maximum leaf depth
  int32_t max_depth = tree_prior.GetMaxDepth();

  if ((max_depth == -1) || (leaf_depth < max_depth)) {
  
    // Cutpoint enumeration
    std::vector<double> log_cutpoint_evaluations;
    std::vector<int> cutpoint_features;
    std::vector<double> cutpoint_values;
    std::vector<FeatureType> cutpoint_feature_types;
    StochTree::data_size_t valid_cutpoint_count;
    CutpointGridContainer cutpoint_grid_container(dataset.GetCovariates(), residual.GetData(), cutpoint_grid_size);
    EvaluateCutpoints<LeafModel, LeafSuffStat, LeafSuffStatConstructorArgs...>(
      tree, tracker, leaf_model, dataset, residual, tree_prior, gen, tree_num, global_variance,
      cutpoint_grid_size, node_id, node_begin, node_end, log_cutpoint_evaluations, cutpoint_features, 
      cutpoint_values, cutpoint_feature_types, valid_cutpoint_count, variable_weights, feature_types, 
      cutpoint_grid_container, leaf_suff_stat_args...
    );
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
}

template <typename LeafModel, typename LeafSuffStat, typename... LeafSuffStatConstructorArgs>
static inline void GFRSampleTreeOneIter(Tree* tree, ForestTracker& tracker, ForestContainer& forests, LeafModel& leaf_model, ForestDataset& dataset,
                                        ColumnVector& residual, TreePrior& tree_prior, std::mt19937& gen, std::vector<double>& variable_weights, 
                                        int tree_num, double global_variance, std::vector<FeatureType>& feature_types, int cutpoint_grid_size, 
                                        LeafSuffStatConstructorArgs&... leaf_suff_stat_args) {
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
    SampleSplitRule<LeafModel, LeafSuffStat, LeafSuffStatConstructorArgs...>(
      tree, tracker, leaf_model, dataset, residual, tree_prior, gen, tree_num, global_variance, cutpoint_grid_size, 
      node_index_map, split_queue, curr_node_id, curr_node_begin, curr_node_end, variable_weights, feature_types, 
      leaf_suff_stat_args...);
  }
}

template <typename LeafModel, typename LeafSuffStat, typename... LeafSuffStatConstructorArgs>
static inline void GFRSampleOneIter(TreeEnsemble& active_forest, ForestTracker& tracker, ForestContainer& forests, LeafModel& leaf_model, ForestDataset& dataset, 
                                    ColumnVector& residual, TreePrior& tree_prior, std::mt19937& gen, std::vector<double>& variable_weights, 
                                    double global_variance, std::vector<FeatureType>& feature_types, int cutpoint_grid_size, 
                                    bool keep_forest, bool pre_initialized, bool backfitting, LeafSuffStatConstructorArgs&... leaf_suff_stat_args) {
  // // Previous number of samples
  // int prev_num_samples = forests.NumSamples();
  
  // // Handle any "initialization" of a model (trees, ForestTracker, etc...) if this is the first sample and 
  // // the model was not pre-initialized
  // bool var_trees;
  // if (std::is_same_v<LeafModel, LogLinearVarianceLeafModel>) var_trees = true;
  // else var_trees = false;
  // ModelInitialization<LeafModel>(tracker, forests, leaf_model, dataset, residual, tree_prior, gen,
  //                                variable_weights, global_variance, pre_initialized, backfitting,
  //                                prev_num_samples, var_trees);
  
  // Run the GFR algorithm for each tree
  // TreeEnsemble* ensemble = forests.GetEnsemble(prev_num_samples);
  int num_trees = forests.NumTrees();
  for (int i = 0; i < num_trees; i++) {
    // Adjust any model state needed to run a tree sampler
    // For models that involve Bayesian backfitting, this amounts to adding tree i's 
    // predictions back to the residual (thus, training a model on the "partial residual")
    // For more general "blocked MCMC" models, this might require changes to a ForestTracker or Dataset object
    Tree* tree = active_forest.GetTree(i);
    AdjustStateBeforeTreeSampling<LeafModel>(tracker, leaf_model, dataset, residual, tree_prior, backfitting, tree, i);
    
    // Reset the tree and sample trackers
    active_forest.ResetInitTree(i);
    tracker.ResetRoot(dataset.GetCovariates(), feature_types, i);
    tree = active_forest.GetTree(i);
    
    // Sample tree i
    GFRSampleTreeOneIter<LeafModel, LeafSuffStat, LeafSuffStatConstructorArgs...>(
      tree, tracker, forests, leaf_model, dataset, residual, tree_prior, gen, 
      variable_weights, i, global_variance, feature_types, cutpoint_grid_size, 
      leaf_suff_stat_args...
    );
    
    // Sample leaf parameters for tree i
    tree = active_forest.GetTree(i);
    leaf_model.SampleLeafParameters(dataset, tracker, residual, tree, i, global_variance, gen);
    
    // Adjust any model state needed to run a tree sampler
    // For models that involve Bayesian backfitting, this amounts to subtracting tree i's 
    // predictions back out of the residual (thus, using an updated "partial residual" in the following interation).
    // For more general "blocked MCMC" models, this might require changes to a ForestTracker or Dataset object
    AdjustStateAfterTreeSampling<LeafModel>(tracker, leaf_model, dataset, residual, tree_prior, backfitting, tree, i);
  }

  if (keep_forest) {
    forests.AddSample(active_forest);
  }
}

template <typename LeafModel, typename LeafSuffStat, typename... LeafSuffStatConstructorArgs>
static inline void MCMCGrowTreeOneIter(Tree* tree, ForestTracker& tracker, LeafModel& leaf_model, ForestDataset& dataset, ColumnVector& residual, 
                                       TreePrior& tree_prior, std::mt19937& gen, int tree_num, std::vector<double>& variable_weights, 
                                       double global_variance, double prob_grow_old, LeafSuffStatConstructorArgs&... leaf_suff_stat_args) {
  // Extract dataset information
  data_size_t n = dataset.GetCovariates().rows();

  // Choose a leaf node at random
  int num_leaves = tree->NumLeaves();
  std::vector<int> leaves = tree->GetLeaves();
  std::vector<double> leaf_weights(num_leaves);
  std::fill(leaf_weights.begin(), leaf_weights.end(), 1.0/num_leaves);
  std::discrete_distribution<> leaf_dist(leaf_weights.begin(), leaf_weights.end());
  int leaf_chosen = leaves[leaf_dist(gen)];
  int leaf_depth = tree->GetDepth(leaf_chosen);

  // Maximum leaf depth
  int32_t max_depth = tree_prior.GetMaxDepth();

  // Terminate early if cannot be split
  bool accept;
  if ((leaf_depth >= max_depth) && (max_depth != -1)) {
    accept = false;
  } else {

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
    std::tuple<double, double, int32_t, int32_t> split_eval = EvaluateProposedSplit<LeafModel, LeafSuffStat, LeafSuffStatConstructorArgs...>(
      dataset, tracker, residual, leaf_model, split, tree_num, leaf_chosen, var_chosen, global_variance, leaf_suff_stat_args...
    );
    double split_log_marginal_likelihood = std::get<0>(split_eval);
    double no_split_log_marginal_likelihood = std::get<1>(split_eval);
    int32_t left_n = std::get<2>(split_eval);
    int32_t right_n = std::get<3>(split_eval);

    // Reject the split if either of the left and right nodes are smaller than tree_prior.GetMinSamplesLeaf()
    bool left_node_sample_cutoff = left_n >= tree_prior.GetMinSamplesLeaf();
    bool right_node_sample_cutoff = right_n >= tree_prior.GetMinSamplesLeaf();
    if ((left_node_sample_cutoff) && (right_node_sample_cutoff)) {
      
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
      std::uniform_real_distribution<double> mh_accept(0.0, 1.0);
      double log_acceptance_prob = std::log(mh_accept(gen));
      if (log_acceptance_prob <= log_mh_ratio) {
        accept = true;
        AddSplitToModel(tracker, dataset, tree_prior, split, gen, tree, tree_num, leaf_chosen, var_chosen, false);
      } else {
        accept = false;
      }

    } else {
      accept = false;
    }
  }
}

template <typename LeafModel, typename LeafSuffStat, typename... LeafSuffStatConstructorArgs>
static inline void MCMCPruneTreeOneIter(Tree* tree, ForestTracker& tracker, LeafModel& leaf_model, ForestDataset& dataset, ColumnVector& residual, 
                                        TreePrior& tree_prior, std::mt19937& gen, int tree_num, double global_variance, LeafSuffStatConstructorArgs&... leaf_suff_stat_args) {
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
  std::tuple<double, double, int32_t, int32_t> split_eval = EvaluateExistingSplit<LeafModel, LeafSuffStat, LeafSuffStatConstructorArgs...>(
    dataset, tracker, residual, leaf_model, global_variance, tree_num, leaf_parent_chosen, left_node, right_node, leaf_suff_stat_args...
  );
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
    RemoveSplitFromModel(tracker, dataset, tree_prior, gen, tree, tree_num, leaf_parent_chosen, left_node, right_node, false);
  } else {
    accept = false;
  }
}

template <typename LeafModel, typename LeafSuffStat, typename... LeafSuffStatConstructorArgs>
static inline void MCMCSampleTreeOneIter(Tree* tree, ForestTracker& tracker, ForestContainer& forests, LeafModel& leaf_model, ForestDataset& dataset,
                                         ColumnVector& residual, TreePrior& tree_prior, std::mt19937& gen, std::vector<double>& variable_weights, 
                                         int tree_num, double global_variance, LeafSuffStatConstructorArgs&... leaf_suff_stat_args) {
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
    MCMCGrowTreeOneIter<LeafModel, LeafSuffStat, LeafSuffStatConstructorArgs...>(
      tree, tracker, leaf_model, dataset, residual, tree_prior, gen, tree_num, variable_weights, global_variance, prob_grow, leaf_suff_stat_args...
    );
  } else {
    MCMCPruneTreeOneIter<LeafModel, LeafSuffStat, LeafSuffStatConstructorArgs...>(
      tree, tracker, leaf_model, dataset, residual, tree_prior, gen, tree_num, global_variance, leaf_suff_stat_args...
    );
  }
}

template <typename LeafModel, typename LeafSuffStat, typename... LeafSuffStatConstructorArgs>
static inline void MCMCSampleOneIter(TreeEnsemble& active_forest, ForestTracker& tracker, ForestContainer& forests, LeafModel& leaf_model, ForestDataset& dataset, 
                                     ColumnVector& residual, TreePrior& tree_prior, std::mt19937& gen, std::vector<double>& variable_weights, 
                                     double global_variance, bool keep_forest, bool pre_initialized, bool backfitting, LeafSuffStatConstructorArgs&... leaf_suff_stat_args) {
  // // Previous number of samples
  // int prev_num_samples = forests.NumSamples();
  
  // // Handle any "initialization" of a model (trees, ForestTracker, etc...) if this is the first sample and 
  // // the model was not pre-initialized
  // bool var_trees;
  // if (std::is_same_v<LeafModel, LogLinearVarianceLeafModel>) var_trees = true;
  // else var_trees = false;
  // ModelInitialization<LeafModel>(tracker, forests, leaf_model, dataset, residual, tree_prior, gen,
  //                                variable_weights, global_variance, pre_initialized, backfitting,
  //                                prev_num_samples, var_trees);
  
  // Run the MCMC algorithm for each tree
  // TreeEnsemble* ensemble = forests.GetEnsemble(prev_num_samples);
  int num_trees = forests.NumTrees();
  for (int i = 0; i < num_trees; i++) {
    // Adjust any model state needed to run a tree sampler
    // For models that involve Bayesian backfitting, this amounts to adding tree i's 
    // predictions back to the residual (thus, training a model on the "partial residual")
    // For more general "blocked MCMC" models, this might require changes to a ForestTracker or Dataset object
    // Tree* tree = ensemble->GetTree(i);
    Tree* tree = active_forest.GetTree(i);
    AdjustStateBeforeTreeSampling<LeafModel>(tracker, leaf_model, dataset, residual, tree_prior, backfitting, tree, i);
    
    // Sample tree i
    tree = active_forest.GetTree(i);
    MCMCSampleTreeOneIter<LeafModel, LeafSuffStat, LeafSuffStatConstructorArgs...>(
      tree, tracker, forests, leaf_model, dataset, residual, tree_prior, gen, variable_weights, i, 
      global_variance, leaf_suff_stat_args...
    );
    
    // Sample leaf parameters for tree i
    tree = active_forest.GetTree(i);
    leaf_model.SampleLeafParameters(dataset, tracker, residual, tree, i, global_variance, gen);
    
    // Adjust any model state needed to run a tree sampler
    // For models that involve Bayesian backfitting, this amounts to subtracting tree i's 
    // predictions back out of the residual (thus, using an updated "partial residual" in the following interation).
    // For more general "blocked MCMC" models, this might require changes to a ForestTracker or Dataset object
    AdjustStateAfterTreeSampling<LeafModel>(tracker, leaf_model, dataset, residual, tree_prior, backfitting, tree, i);
  }

  if (keep_forest) {
    forests.AddSample(active_forest);
  }
}

} // namespace StochTree

#endif // STOCHTREE_TREE_SAMPLER_H_
