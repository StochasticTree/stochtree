/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include <stochtree/io.h>
#include <stochtree/partition_tracker.h>

#include <chrono>
#include <cstdio>
#include <limits>
#include <sstream>
#include <unordered_map>

namespace StochTree {

ForestTracker::ForestTracker(Eigen::MatrixXd& covariates, std::vector<FeatureType>& feature_types, int num_trees, int num_observations) {
  sample_node_mapper_ = std::make_unique<SampleNodeMapper>(num_trees, num_observations);
  unsorted_node_sample_tracker_ = std::make_unique<UnsortedNodeSampleTracker>(num_observations, num_trees);
  presort_container_ = std::make_unique<FeaturePresortRootContainer>(covariates, feature_types);
  sorted_node_sample_tracker_ = std::make_unique<SortedNodeSampleTracker>(presort_container_.get(), covariates, feature_types);

  num_trees_ = num_trees;
  num_observations_ = num_observations;
  num_features_ = feature_types.size();
  feature_types_ = feature_types;
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

void ForestTracker::AddSplit(Eigen::MatrixXd& covariates, TreeSplit& split, int32_t split_feature, int32_t tree_id, int32_t split_node_id, int32_t left_node_id, int32_t right_node_id) {
  sample_node_mapper_->AddSplit(covariates, split, split_feature, tree_id, split_node_id, left_node_id, right_node_id);
  unsorted_node_sample_tracker_->PartitionTreeNode(covariates, tree_id, split_node_id, left_node_id, right_node_id, split_feature, split);
  sorted_node_sample_tracker_->PartitionNode(covariates, split_node_id, split_feature, split);
}

void ForestTracker::RemoveSplit(Eigen::MatrixXd& covariates, Tree* tree, int32_t tree_id, int32_t split_node_id, int32_t left_node_id, int32_t right_node_id) {
  unsorted_node_sample_tracker_->PruneTreeNodeToLeaf(tree_id, split_node_id);
  unsorted_node_sample_tracker_->UpdateObservationMapping(tree, tree_id, sample_node_mapper_.get());
  // TODO: WARN if this is called from the GFR Tree Sampler
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
