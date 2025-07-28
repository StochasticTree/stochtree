/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include <stochtree/io.h>
#include <stochtree/partition_tracker.h>

namespace StochTree {

ForestTracker::ForestTracker(Eigen::MatrixXd& covariates, std::vector<FeatureType>& feature_types, int num_trees, int num_observations) {
  sample_pred_mapper_ = std::make_unique<SamplePredMapper>(num_trees, num_observations);
  sample_node_mapper_ = std::make_unique<SampleNodeMapper>(num_trees, num_observations);
  unsorted_node_sample_tracker_ = std::make_unique<UnsortedNodeSampleTracker>(num_observations, num_trees);
  presort_container_ = std::make_unique<FeaturePresortRootContainer>(covariates, feature_types);
  sorted_node_sample_tracker_ = std::make_unique<SortedNodeSampleTracker>(presort_container_.get(), covariates, feature_types);
  sum_predictions_ = std::vector<double>(num_observations, 0.);

  num_trees_ = num_trees;
  num_observations_ = num_observations;
  num_features_ = feature_types.size();
  feature_types_ = feature_types;
  initialized_ = true;
}

void ForestTracker::ReconstituteFromForest(TreeEnsemble& forest, ForestDataset& dataset, ColumnVector& residual, bool is_mean_model) {
  // Together this function:
  // (1) Updates the residual by adding currently cached tree predictions and subtracting predictions from new tree
  // (2) Updates sample_node_mapper_, sample_pred_mapper_, and sum_predictions_ based on the new forest
  UpdateSampleTrackersResidual(forest, dataset, residual, is_mean_model);
  
  // Since GFR always starts over from root, this data structure can always simply be reset
  Eigen::MatrixXd& covariates = dataset.GetCovariates();
  sorted_node_sample_tracker_.reset(new SortedNodeSampleTracker(presort_container_.get(), covariates, feature_types_));
  
  // Reconstitute each of the remaining data structures in the tracker based on splits in the ensemble
  // UnsortedNodeSampleTracker
  unsorted_node_sample_tracker_->ReconstituteFromForest(forest, dataset);
  
}

void ForestTracker::ResetRoot(Eigen::MatrixXd& covariates, std::vector<FeatureType>& feature_types, int32_t tree_num) {
  AssignAllSamplesToRoot(tree_num);
  unsorted_node_sample_tracker_->ResetTreeToRoot(tree_num, covariates.rows());
  sorted_node_sample_tracker_.reset(new SortedNodeSampleTracker(presort_container_.get(), covariates, feature_types));
}

data_size_t ForestTracker::GetNodeId(int observation_num, int tree_num) {return sample_node_mapper_->GetNodeId(observation_num, tree_num);}

data_size_t ForestTracker::UnsortedNodeBegin(int tree_id, int node_id) {return unsorted_node_sample_tracker_->NodeBegin(tree_id, node_id);}

data_size_t ForestTracker::UnsortedNodeEnd(int tree_id, int node_id) {return unsorted_node_sample_tracker_->NodeEnd(tree_id, node_id);}

data_size_t ForestTracker::UnsortedNodeSize(int tree_id, int node_id) {return unsorted_node_sample_tracker_->NodeSize(tree_id, node_id);}

data_size_t ForestTracker::SortedNodeBegin(int node_id, int feature_id) {return sorted_node_sample_tracker_->NodeBegin(node_id, feature_id);}

data_size_t ForestTracker::SortedNodeEnd(int node_id, int feature_id) {return sorted_node_sample_tracker_->NodeEnd(node_id, feature_id);}

data_size_t ForestTracker::SortedNodeSize(int node_id, int feature_id) {return sorted_node_sample_tracker_->NodeSize(node_id, feature_id);}

std::vector<data_size_t>::iterator ForestTracker::UnsortedNodeBeginIterator(int tree_id, int node_id) {
  return unsorted_node_sample_tracker_->NodeBeginIterator(tree_id, node_id);
}
std::vector<data_size_t>::iterator ForestTracker::UnsortedNodeEndIterator(int tree_id, int node_id) {
  return unsorted_node_sample_tracker_->NodeEndIterator(tree_id, node_id);
}
std::vector<data_size_t>::iterator ForestTracker::SortedNodeBeginIterator(int node_id, int feature_id) {
  return sorted_node_sample_tracker_->NodeBeginIterator(node_id, feature_id);
}
std::vector<data_size_t>::iterator ForestTracker::SortedNodeEndIterator(int node_id, int feature_id) {
  return sorted_node_sample_tracker_->NodeEndIterator(node_id, feature_id);
}

void ForestTracker::AssignAllSamplesToRoot() {
  for (int i = 0; i < num_trees_; i++) {
    sample_node_mapper_->AssignAllSamplesToRoot(i);
  }
}

void ForestTracker::AssignAllSamplesToRoot(int32_t tree_num) {
  sample_node_mapper_->AssignAllSamplesToRoot(tree_num);
}

void ForestTracker::AssignAllSamplesToConstantPrediction(double value) {
  for (data_size_t i = 0; i < num_observations_; i++) {
    sum_predictions_[i] = value*num_trees_;
  }
  for (int i = 0; i < num_trees_; i++) {
    sample_pred_mapper_->AssignAllSamplesToConstantPrediction(i, value);
  }
}

void ForestTracker::AssignAllSamplesToConstantPrediction(int32_t tree_num, double value) {
  sample_pred_mapper_->AssignAllSamplesToConstantPrediction(tree_num, value);
}

void ForestTracker::UpdateSampleTrackersInternal(TreeEnsemble& forest, Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis) {
  int output_dim = basis.cols();
  double forest_pred, tree_pred;

  for (data_size_t i = 0; i < num_observations_; i++) {
    forest_pred = 0.0;
    for (int j = 0; j < num_trees_; j++) {
      tree_pred = 0.0;
      Tree* tree = forest.GetTree(j);
      std::int32_t nidx = EvaluateTree(*tree, covariates, i);
      sample_node_mapper_->SetNodeId(i, j, nidx);
      for (int32_t k = 0; k < output_dim; k++) {
        tree_pred += tree->LeafValue(nidx, k) * basis(i, k);
      }
      sample_pred_mapper_->SetPred(i, j, tree_pred);
      forest_pred += tree_pred;
    }
    sum_predictions_[i] = forest_pred;
  }
}

void ForestTracker::UpdateSampleTrackersInternal(TreeEnsemble& forest, Eigen::MatrixXd& covariates) {
  double forest_pred, tree_pred;

  for (data_size_t i = 0; i < num_observations_; i++) {
    forest_pred = 0.0;
    for (int j = 0; j < num_trees_; j++) {
      Tree* tree = forest.GetTree(j);
      std::int32_t nidx = EvaluateTree(*tree, covariates, i);
      sample_node_mapper_->SetNodeId(i, j, nidx);
      tree_pred = tree->LeafValue(nidx, 0);
      sample_pred_mapper_->SetPred(i, j, tree_pred);
      forest_pred += tree_pred;
    }
    sum_predictions_[i] = forest_pred;
  }
}

void ForestTracker::UpdateSampleTrackers(TreeEnsemble& forest, ForestDataset& dataset) {
  if (!forest.IsLeafConstant()) {
    CHECK(dataset.HasBasis());
    UpdateSampleTrackersInternal(forest, dataset.GetCovariates(), dataset.GetBasis());
  } else {
    UpdateSampleTrackersInternal(forest, dataset.GetCovariates());
  }
}

void ForestTracker::UpdateSampleTrackersResidualInternalBasis(TreeEnsemble& forest, ForestDataset& dataset, ColumnVector& residual, bool is_mean_model) {
  double new_forest_pred, new_tree_pred, prev_tree_pred, new_resid, new_weight;
  Eigen::MatrixXd& covariates = dataset.GetCovariates();
  Eigen::MatrixXd& basis = dataset.GetBasis();
  int output_dim = basis.cols();
  if (!is_mean_model) {
    CHECK(dataset.HasVarWeights());
  }

  for (data_size_t i = 0; i < num_observations_; i++) {
    new_forest_pred = 0.0;
    for (int j = 0; j < num_trees_; j++) {
      // Query the previously cached prediction for tree j, observation i
      prev_tree_pred = sample_pred_mapper_->GetPred(i, j);
      
      // Compute the new prediction for tree j, observation i
      new_tree_pred = 0.0;
      Tree* tree = forest.GetTree(j);
      std::int32_t nidx = EvaluateTree(*tree, covariates, i);
      for (int32_t k = 0; k < output_dim; k++) {
        new_tree_pred += tree->LeafValue(nidx, k) * basis(i, k);
      }
      
      if (is_mean_model) {
        // Adjust the residual by adding the previous prediction and subtracting the new prediction
        new_resid = residual.GetElement(i) - new_tree_pred + prev_tree_pred;
        residual.SetElement(i, new_resid);
      } else {
        // Adjust the variance weights by subtracting the previous prediction and adding the new prediction (in log scale) and then exponentiating
        new_weight = std::log(dataset.VarWeightValue(i)) + new_tree_pred - prev_tree_pred;
        dataset.SetVarWeightValue(i, new_weight, true);
      }

      // Update the sample node mapper and sample prediction mapper
      sample_node_mapper_->SetNodeId(i, j, nidx);
      sample_pred_mapper_->SetPred(i, j, new_tree_pred);
      new_forest_pred += new_tree_pred;
    }
    // Update the overall cached forest prediction
    sum_predictions_[i] = new_forest_pred;
  }
}

void ForestTracker::UpdateSampleTrackersResidualInternalNoBasis(TreeEnsemble& forest, ForestDataset& dataset, ColumnVector& residual, bool is_mean_model) {
  double new_forest_pred, new_tree_pred, prev_tree_pred, new_resid, new_weight;
  Eigen::MatrixXd& covariates = dataset.GetCovariates();
  if (!is_mean_model) {
    CHECK(dataset.HasVarWeights());
  }

  for (data_size_t i = 0; i < num_observations_; i++) {
    new_forest_pred = 0.0;
    for (int j = 0; j < num_trees_; j++) {
      // Query the previously cached prediction for tree j, observation i
      prev_tree_pred = sample_pred_mapper_->GetPred(i, j);

      // Compute the new prediction for tree j, observation i
      Tree* tree = forest.GetTree(j);
      std::int32_t nidx = EvaluateTree(*tree, covariates, i);
      new_tree_pred = tree->LeafValue(nidx, 0);
      
      if (is_mean_model) {
        // Adjust the residual by adding the previous prediction and subtracting the new prediction
        new_resid = residual.GetElement(i) - new_tree_pred + prev_tree_pred;
        residual.SetElement(i, new_resid);
      } else {
        new_weight = std::log(dataset.VarWeightValue(i)) + new_tree_pred - prev_tree_pred;
        dataset.SetVarWeightValue(i, new_weight, true);
      }
      
      // Update the sample node mapper and sample prediction mapper
      sample_node_mapper_->SetNodeId(i, j, nidx);
      sample_pred_mapper_->SetPred(i, j, new_tree_pred);
      new_forest_pred += new_tree_pred;
    }
    // Update the overall cached forest prediction
    sum_predictions_[i] = new_forest_pred;
  }
}

void ForestTracker::UpdateSampleTrackersResidual(TreeEnsemble& forest, ForestDataset& dataset, ColumnVector& residual, bool is_mean_model) {
  if (!forest.IsLeafConstant()) {
    CHECK(dataset.HasBasis());
    UpdateSampleTrackersResidualInternalBasis(forest, dataset, residual, is_mean_model);
  } else {
    UpdateSampleTrackersResidualInternalNoBasis(forest, dataset, residual, is_mean_model);
  }
}

void ForestTracker::UpdatePredictionsInternal(TreeEnsemble* ensemble, Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis) {
  int output_dim = basis.cols();
  double forest_pred, tree_pred;

  for (data_size_t i = 0; i < num_observations_; i++) {
    forest_pred = 0.0;
    for (int j = 0; j < num_trees_; j++) {
      tree_pred = 0.0;
      Tree* tree = ensemble->GetTree(j);
      std::int32_t nidx = EvaluateTree(*tree, covariates, i);
      for (int32_t k = 0; k < output_dim; k++) {
        tree_pred += tree->LeafValue(nidx, k) * basis(i, k);
      }
      sample_pred_mapper_->SetPred(i, j, tree_pred);
      forest_pred += tree_pred;
    }
    sum_predictions_[i] = forest_pred;
  }
}

void ForestTracker::UpdatePredictionsInternal(TreeEnsemble* ensemble, Eigen::MatrixXd& covariates) {
  double forest_pred, tree_pred;

  for (data_size_t i = 0; i < num_observations_; i++) {
    forest_pred = 0.0;
    for (int j = 0; j < num_trees_; j++) {
      Tree* tree = ensemble->GetTree(j);
      std::int32_t nidx = EvaluateTree(*tree, covariates, i);
      tree_pred = tree->LeafValue(nidx, 0);
      sample_pred_mapper_->SetPred(i, j, tree_pred);
      forest_pred += tree_pred;
    }
    sum_predictions_[i] = forest_pred;
  }
}

void ForestTracker::UpdatePredictions(TreeEnsemble* ensemble, ForestDataset& dataset) {
  if (!ensemble->IsLeafConstant()) {
    CHECK(dataset.HasBasis());
    UpdatePredictionsInternal(ensemble, dataset.GetCovariates(), dataset.GetBasis());
  } else {
    UpdatePredictionsInternal(ensemble, dataset.GetCovariates());
  }
}

void ForestTracker::AddSplit(Eigen::MatrixXd& covariates, TreeSplit& split, int32_t split_feature, int32_t tree_id, int32_t split_node_id, int32_t left_node_id, int32_t right_node_id, bool keep_sorted, int num_threads) {
  sample_node_mapper_->AddSplit(covariates, split, split_feature, tree_id, split_node_id, left_node_id, right_node_id);
  unsorted_node_sample_tracker_->PartitionTreeNode(covariates, tree_id, split_node_id, left_node_id, right_node_id, split_feature, split);
  if (keep_sorted) {
    sorted_node_sample_tracker_->PartitionNode(covariates, split_node_id, split_feature, split, num_threads);
  }
}

void ForestTracker::RemoveSplit(Eigen::MatrixXd& covariates, Tree* tree, int32_t tree_id, int32_t split_node_id, int32_t left_node_id, int32_t right_node_id, bool keep_sorted) {
  unsorted_node_sample_tracker_->PruneTreeNodeToLeaf(tree_id, split_node_id);
  unsorted_node_sample_tracker_->UpdateObservationMapping(tree, tree_id, sample_node_mapper_.get());
  // TODO: WARN if this is called from the GFR Tree Sampler
}

double ForestTracker::GetSamplePrediction(data_size_t sample_id) {
  return sum_predictions_[sample_id];
}

double ForestTracker::GetTreeSamplePrediction(data_size_t sample_id, int tree_id) {
  return sample_pred_mapper_->GetPred(sample_id, tree_id);
}

void ForestTracker::UpdateVarWeightsFromInternalPredictions(ForestDataset& dataset) {
  dataset.UpdateVarWeights(sum_predictions_.data(), num_observations_, true);
}

void ForestTracker::SetSamplePrediction(data_size_t sample_id, double value) {
  sum_predictions_[sample_id] = value;
}

void ForestTracker::SetTreeSamplePrediction(data_size_t sample_id, int tree_id, double value) {
  sample_pred_mapper_->SetPred(sample_id, tree_id, value);
}

void ForestTracker::SyncPredictions() {
  for (data_size_t i = 0; i < num_observations_; i++) {
    sum_predictions_[i] = 0.;
    for (int j = 0; j < num_trees_; j++) {
      sum_predictions_[i] += sample_pred_mapper_->GetPred(i, j);
    }
  }
}

void UnsortedNodeSampleTracker::ReconstituteFromForest(TreeEnsemble& forest, ForestDataset& dataset) {
  int n = dataset.NumObservations();
  for (int i = 0; i < num_trees_; i++) {
    Tree* tree = forest.GetTree(i);
    feature_partitions_[i].reset(new FeatureUnsortedPartition(n));
    feature_partitions_[i]->ReconstituteFromTree(*tree, dataset);
  }
}

FeatureUnsortedPartition::FeatureUnsortedPartition(data_size_t n) {
  indices_.resize(n);
  std::iota(indices_.begin(), indices_.end(), 0);
  node_begin_ = {0};
  node_length_ = {n};
  parent_nodes_ = {StochTree::Tree::kInvalidNodeId};
  left_nodes_ = {StochTree::Tree::kInvalidNodeId};
  right_nodes_ = {StochTree::Tree::kInvalidNodeId};
  num_nodes_ = 1;
  num_deleted_nodes_ = 0;
}

void FeatureUnsortedPartition::ReconstituteFromTree(Tree& tree, ForestDataset& dataset) {
  // Make sure this data structure is a root
  CHECK_EQ(num_nodes_, 1);
  CHECK_EQ(num_deleted_nodes_, 0);
  data_size_t n = dataset.NumObservations();
  CHECK_EQ(indices_.size(), n);
  
  // Extract covariates
  Eigen::MatrixXd& covariates = dataset.GetCovariates();

  // Set node counters
  num_nodes_ = tree.NumNodes();
  num_deleted_nodes_ = tree.NumDeletedNodes();
  
  // Resize tracking vectors
  node_begin_.resize(num_nodes_);
  node_length_.resize(num_nodes_);
  parent_nodes_.resize(num_nodes_);
  left_nodes_.resize(num_nodes_);
  right_nodes_.resize(num_nodes_);
  
  // Unpack tree's splits into this data structure
  bool is_deleted;
  TreeNodeType node_type;
  data_size_t node_start_idx;
  data_size_t num_node_elements;
  data_size_t num_true, num_false;
  TreeSplit split_rule;
  int split_index;
  for (int i = 0; i < num_nodes_; i++) {
    is_deleted = tree.IsDeleted(i);
    if (is_deleted) {
      deleted_nodes_.push_back(i);
    } else {
      // Node beginning and length in indices_
      if (i == 0) {
        node_start_idx = 0;
        num_node_elements = n;
      } else {
        node_start_idx = node_begin_[i];
        num_node_elements = node_length_[i];
      }
      // Tree node info
      parent_nodes_[i] = tree.Parent(i);
      node_type = tree.NodeType(i);
      left_nodes_[i] = tree.LeftChild(i);
      right_nodes_[i] = tree.RightChild(i);
      // Only update indices_, node_begin_ and node_length_ if a split is to be added
      if (node_type == TreeNodeType::kNumericalSplitNode) {
        // Extract split rule
        split_rule = TreeSplit(tree.Threshold(i));
        split_index = tree.SplitIndex(i);
      } else if (node_type == TreeNodeType::kCategoricalSplitNode) {
        std::vector<uint32_t> categories = tree.CategoryList(i);
        split_rule = TreeSplit(categories);
        split_index = tree.SplitIndex(i);
      } else {
        continue;
      }
      // Partition the node indices 
      auto node_begin = (indices_.begin() + node_begin_[i]);
      auto node_end = (indices_.begin() + node_begin_[i] + node_length_[i]);
      auto right_node_begin = std::stable_partition(node_begin, node_end, [&](int row) { return split_rule.SplitTrue(covariates(row, split_index)); });
      
      // Determine the number of true and false elements
      node_begin = (indices_.begin() + node_begin_[i]);
      num_true = std::distance(node_begin, right_node_begin);
      num_false = num_node_elements - num_true;

      // Add left node tracking information
      node_begin_[left_nodes_[i]] = node_start_idx;
      node_length_[left_nodes_[i]] = num_true;
      parent_nodes_[left_nodes_[i]] = i;
      left_nodes_[left_nodes_[i]] = StochTree::Tree::kInvalidNodeId;
      left_nodes_[right_nodes_[i]] = StochTree::Tree::kInvalidNodeId;
      
      // Add right node tracking information
      node_begin_[right_nodes_[i]] = node_start_idx + num_true;
      node_length_[right_nodes_[i]] = num_false;
      parent_nodes_[right_nodes_[i]] = i;
      right_nodes_[left_nodes_[i]] = StochTree::Tree::kInvalidNodeId;
      right_nodes_[right_nodes_[i]] = StochTree::Tree::kInvalidNodeId;
    }
  }
}

data_size_t FeatureUnsortedPartition::NodeBegin(int node_id) {
  return node_begin_[node_id];
}

data_size_t FeatureUnsortedPartition::NodeEnd(int node_id) {
  return node_begin_[node_id] + node_length_[node_id];
}

data_size_t FeatureUnsortedPartition::NodeSize(int node_id) {
  return node_length_[node_id];
}

int FeatureUnsortedPartition::Parent(int node_id) {
  return parent_nodes_[node_id];
}

int FeatureUnsortedPartition::LeftNode(int node_id) {
  return left_nodes_[node_id];
}

int FeatureUnsortedPartition::RightNode(int node_id) {
  return right_nodes_[node_id];
}

void FeatureUnsortedPartition::PartitionNode(Eigen::MatrixXd& covariates, int node_id, int left_node_id, int right_node_id, int feature_split, TreeSplit& split) {
  // Partition-related values
  data_size_t node_start_idx = node_begin_[node_id];
  data_size_t num_node_elements = node_length_[node_id];

  // Partition the node indices 
  auto node_begin = (indices_.begin() + node_begin_[node_id]);
  auto node_end = (indices_.begin() + node_begin_[node_id] + node_length_[node_id]);
  auto right_node_begin = std::stable_partition(node_begin, node_end, [&](int row) { return split.SplitTrue(covariates(row, feature_split)); });
  
  // Determine the number of true and false elements
  node_begin = (indices_.begin() + node_begin_[node_id]);
  data_size_t num_true = std::distance(node_begin, right_node_begin);
  data_size_t num_false = num_node_elements - num_true;

  // Now, update all of the node tracking machinery
  ExpandNodeTrackingVectors(node_id, left_node_id, right_node_id, node_start_idx, num_true, num_false);
}

void FeatureUnsortedPartition::PartitionNode(Eigen::MatrixXd& covariates, int node_id, int left_node_id, int right_node_id, int feature_split, double split_value) {
  // Partition-related values
  data_size_t node_start_idx = node_begin_[node_id];
  data_size_t num_node_elements = node_length_[node_id];

  // Partition the node indices 
  auto node_begin = (indices_.begin() + node_begin_[node_id]);
  auto node_end = (indices_.begin() + node_begin_[node_id] + node_length_[node_id]);
  auto right_node_begin = std::stable_partition(node_begin, node_end, [&](int row) { return RowSplitLeft(covariates, row, feature_split, split_value); });
  
  // Determine the number of true and false elements
  node_begin = (indices_.begin() + node_begin_[node_id]);
  data_size_t num_true = std::distance(node_begin, right_node_begin);
  data_size_t num_false = num_node_elements - num_true;

  // Now, update all of the node tracking machinery
  ExpandNodeTrackingVectors(node_id, left_node_id, right_node_id, node_start_idx, num_true, num_false);
}

void FeatureUnsortedPartition::PartitionNode(Eigen::MatrixXd& covariates, int node_id, int left_node_id, int right_node_id, int feature_split, std::vector<std::uint32_t> const& category_list) {
  // Partition-related values
  data_size_t node_start_idx = node_begin_[node_id];
  data_size_t num_node_elements = node_length_[node_id];

  // Partition the node indices 
  auto node_begin = (indices_.begin() + node_begin_[node_id]);
  auto node_end = (indices_.begin() + node_begin_[node_id] + node_length_[node_id]);
  auto right_node_begin = std::stable_partition(node_begin, node_end, [&](int row) { return RowSplitLeft(covariates, row, feature_split, category_list); });
  
  // Determine the number of true and false elements
  node_begin = (indices_.begin() + node_begin_[node_id]);
  data_size_t num_true = std::distance(node_begin, right_node_begin);
  data_size_t num_false = num_node_elements - num_true;

  // Now, update all of the node tracking machinery
  ExpandNodeTrackingVectors(node_id, left_node_id, right_node_id, node_start_idx, num_true, num_false);
}

void FeatureUnsortedPartition::ExpandNodeTrackingVectors(int node_id, int left_node_id, int right_node_id, data_size_t node_start_idx, data_size_t num_left, data_size_t num_right) {
  // Allocate more space if necessary
  int largest_node_id = left_node_id > right_node_id ? left_node_id : right_node_id;
  if (largest_node_id >= num_nodes_) {
    node_begin_.resize(largest_node_id + 1);
    node_length_.resize(largest_node_id + 1);
    parent_nodes_.resize(largest_node_id + 1);
    left_nodes_.resize(largest_node_id + 1);
    right_nodes_.resize(largest_node_id + 1);
    num_nodes_ = largest_node_id + 1;
  }

  // Remove left and right nodes from "deleted" tracker if they are reused
  if (!IsValidNode(left_node_id)) {
    num_deleted_nodes_--;
    deleted_nodes_.erase(std::remove(deleted_nodes_.begin(), deleted_nodes_.end(), left_node_id), deleted_nodes_.end());
  }
  if (!IsValidNode(right_node_id)) {
    num_deleted_nodes_--;
    deleted_nodes_.erase(std::remove(deleted_nodes_.begin(), deleted_nodes_.end(), right_node_id), deleted_nodes_.end());
  }

  // Add left node tracking information
  left_nodes_[node_id] = left_node_id;
  node_begin_[left_node_id] = node_start_idx;
  node_length_[left_node_id] = num_left;
  parent_nodes_[left_node_id] = node_id;
  left_nodes_[left_node_id] = StochTree::Tree::kInvalidNodeId;
  left_nodes_[right_node_id] = StochTree::Tree::kInvalidNodeId;
  
  // Add right node tracking information
  right_nodes_[node_id] = right_node_id;
  node_begin_[right_node_id] = node_start_idx + num_left;
  node_length_[right_node_id] = num_right;
  parent_nodes_[right_node_id] = node_id;
  right_nodes_[left_node_id] = StochTree::Tree::kInvalidNodeId;
  right_nodes_[right_node_id] = StochTree::Tree::kInvalidNodeId;
}

void FeatureUnsortedPartition::UpdateObservationMapping(int node_id, int tree_id, SampleNodeMapper* sample_node_mapper) {
  data_size_t node_begin = node_begin_[node_id];
  data_size_t node_size = node_length_[node_id];
  data_size_t node_end = node_begin + node_size;
  data_size_t idx;
  for (data_size_t i = node_begin; i < node_end; i++) {
    idx = indices_[i];
    sample_node_mapper->SetNodeId(idx, tree_id, node_id);
  }
}

bool FeatureUnsortedPartition::IsLeaf(int node_id) {
  return left_nodes_[node_id] == StochTree::Tree::kInvalidNodeId;
}

bool FeatureUnsortedPartition::IsValidNode(int node_id) {
  if (node_id >= num_nodes_ || node_id < 0) {
    return false;
  }
  return !(std::find(deleted_nodes_.begin(), deleted_nodes_.end(), node_id)
           != deleted_nodes_.end());
}

bool FeatureUnsortedPartition::LeftNodeIsLeaf(int node_id) {
  return left_nodes_[left_nodes_[node_id]] == StochTree::Tree::kInvalidNodeId;
}

bool FeatureUnsortedPartition::RightNodeIsLeaf(int node_id) {
  return left_nodes_[right_nodes_[node_id]] == StochTree::Tree::kInvalidNodeId;
}

void FeatureUnsortedPartition::PruneNodeToLeaf(int node_id) {
  // No need to "un-sift" the indices in the newly pruned node, we don't depend on the indices 
  // having any type of sort order, so the indices will simply be "re-sifted" if the node is later partitioned
  if (IsLeaf(node_id)) return;
  if (!LeftNodeIsLeaf(node_id)) {
    PruneNodeToLeaf(left_nodes_[node_id]);
  }
  if (!RightNodeIsLeaf(node_id)) {
    PruneNodeToLeaf(right_nodes_[node_id]);
  }
  ConvertLeafParentToLeaf(node_id);
}

void FeatureUnsortedPartition::ConvertLeafParentToLeaf(int node_id) {
  CHECK(IsLeaf(LeftNode(node_id)));
  CHECK(IsLeaf(RightNode(node_id)));
  deleted_nodes_.push_back(left_nodes_[node_id]);
  num_deleted_nodes_++;
  deleted_nodes_.push_back(right_nodes_[node_id]);
  num_deleted_nodes_++;
  left_nodes_[node_id] = StochTree::Tree::kInvalidNodeId;
  right_nodes_[node_id] = StochTree::Tree::kInvalidNodeId;
}

std::vector<data_size_t> FeatureUnsortedPartition::NodeIndices(int node_id) {
  CHECK(IsValidNode(node_id));
  data_size_t node_begin = node_begin_[node_id];
  data_size_t node_size = node_length_[node_id];
  std::vector<data_size_t> out(node_size);
  for (data_size_t i = 0; i < node_size; i++) {
    out[i] = indices_[node_begin + i];
  }
  return out;
}

void FeaturePresortPartition::AddLeftRightNodes(data_size_t left_node_begin, data_size_t left_node_size, data_size_t right_node_begin, data_size_t right_node_size) {
  // Assumes that we aren't pruning / deleting nodes, since this is for use with recursive algorithms
  
  // Add the left ("true") node to the offset size vector
  node_offset_sizes_.emplace_back(left_node_begin, left_node_size);
  // Add the right ("false") node to the offset size vector
  node_offset_sizes_.emplace_back(right_node_begin, right_node_size);
}

void FeaturePresortPartition::SplitFeature(Eigen::MatrixXd& covariates, int32_t node_id, int32_t feature_index, TreeSplit& split) {
  // Partition-related values
  data_size_t node_start_idx = NodeBegin(node_id);
  data_size_t node_end_idx = NodeEnd(node_id);
  data_size_t num_node_elements = NodeSize(node_id);

  // Partition the node indices 
  auto node_begin = (feature_sort_indices_.begin() + node_start_idx);
  auto node_end = (feature_sort_indices_.begin() + node_end_idx);
  auto right_node_begin = std::stable_partition(node_begin, node_end, [&](int row) { return split.SplitTrue(covariates(row, feature_index)); });
  
  // Add the left and right nodes to the offset size vector
  node_begin = (feature_sort_indices_.begin() + node_start_idx);
  data_size_t num_true = std::distance(node_begin, right_node_begin);
  data_size_t num_false = num_node_elements - num_true;
  AddLeftRightNodes(node_start_idx, num_true, node_start_idx + num_true, num_false);
}

void FeaturePresortPartition::SplitFeatureNumeric(Eigen::MatrixXd& covariates, int32_t node_id, int32_t feature_index, double split_value) {
  // Partition-related values
  data_size_t node_start_idx = NodeBegin(node_id);
  data_size_t node_end_idx = NodeEnd(node_id);
  data_size_t num_node_elements = NodeSize(node_id);

  // Partition the node indices 
  auto node_begin = (feature_sort_indices_.begin() + node_start_idx);
  auto node_end = (feature_sort_indices_.begin() + node_end_idx);
  auto right_node_begin = std::stable_partition(node_begin, node_end, [&](int row) { return RowSplitLeft(covariates, row, feature_index, split_value); });
  
  // Add the left and right nodes to the offset size vector
  node_begin = (feature_sort_indices_.begin() + node_start_idx);
  data_size_t num_true = std::distance(node_begin, right_node_begin);
  data_size_t num_false = num_node_elements - num_true;
  AddLeftRightNodes(node_start_idx, num_true, node_start_idx + num_true, num_false);
}

void FeaturePresortPartition::SplitFeatureCategorical(Eigen::MatrixXd& covariates, int32_t node_id, int32_t feature_index, std::vector<std::uint32_t> const& category_list) {
  // Partition-related values
  data_size_t node_start_idx = NodeBegin(node_id);
  data_size_t node_end_idx = NodeEnd(node_id);
  data_size_t num_node_elements = NodeSize(node_id);

  // Partition the node indices 
  auto node_begin = (feature_sort_indices_.begin() + node_start_idx);
  auto node_end = (feature_sort_indices_.begin() + node_end_idx);
  auto right_node_begin = std::stable_partition(node_begin, node_end, [&](int row) { return RowSplitLeft(covariates, row, feature_index, category_list); });
  
  // Add the left and right nodes to the offset size vector
  node_begin = (feature_sort_indices_.begin() + node_start_idx);
  data_size_t num_true = std::distance(node_begin, right_node_begin);
  data_size_t num_false = num_node_elements - num_true;
  AddLeftRightNodes(node_start_idx, num_true, node_start_idx + num_true, num_false);
}

void FeaturePresortPartition::UpdateObservationMapping(int node_id, int tree_id, SampleNodeMapper* sample_node_mapper) {
  data_size_t node_begin = NodeBegin(node_id);
  data_size_t node_size = NodeSize(node_id);
  data_size_t node_end = NodeEnd(node_id);
  data_size_t idx;
  for (data_size_t i = node_begin; i < node_end; i++) {
    idx = feature_sort_indices_[i];
    sample_node_mapper->SetNodeId(idx, tree_id, node_id);
  }
}

std::vector<data_size_t> FeaturePresortPartition::NodeIndices(int node_id) {
  data_size_t node_begin = NodeBegin(node_id);
  data_size_t node_size = NodeSize(node_id);
  std::vector<data_size_t> out(node_size);
  for (data_size_t i = 0; i < node_size; i++) {
    out[i] = feature_sort_indices_[node_begin + i];
  }
  return out;
}

}  // namespace StochTree
