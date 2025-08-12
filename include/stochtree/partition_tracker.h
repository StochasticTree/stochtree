/*!
 * Copyright (c) 2024 stochtree authors.
 * 
 * Data structures used for tracking dataset through the tree building process.
 * 
 * The first category of data structure tracks observations available in nodes of a tree.
 *   a. UnsortedNodeSampleTracker tracks the observations available in every leaf of every tree in an ensemble, 
 *      in no feature-specific sort order. It is primarily designed for use in BART-based algorithms.
 *   b. SortedNodeSampleTracker tracks the observations available in a every leaf of a tree, pre-sorted 
 *      separately for each feature. It is primarily designed for use in XBART-based algorithms.
 * 
 * The second category, SampleNodeMapper, maps observations from a dataset to leaf nodes.
 * 
 * SampleNodeMapper is inspired by the design of the DataPartition class in LightGBM, 
 * released under the MIT license with the following copyright:
 * 
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 * 
 * SortedNodeSampleTracker is inspired by the "approximate" split finding method in xgboost, released 
 * under the Apache license with the following copyright:
 * 
 * Copyright 2015~2023 by XGBoost Contributors
 */
#ifndef STOCHTREE_PARTITION_TRACKER_H_
#define STOCHTREE_PARTITION_TRACKER_H_

#include <stochtree/data.h>
#include <stochtree/ensemble.h>
#include <stochtree/log.h>
#include <stochtree/openmp_utils.h>
#include <stochtree/tree.h>

#include <numeric>
#include <vector>

namespace StochTree {

/*! \brief Forward declarations of component classes */
class SampleNodeMapper;
class SamplePredMapper;
class UnsortedNodeSampleTracker;
class SortedNodeSampleTracker;
class FeaturePresortRootContainer;

/*! \brief "Superclass" wrapper around tracking data structures for forest sampling algorithms */
class ForestTracker {
 public:
  /*!
   * \brief Construct a new `ForestTracker` object
   * 
   * \param covariates Matrix of covariate data
   * \param feature_types Type of each feature (column) in `covariates`. This is represented by the enum `StochTree::FeatureType`
   * \param num_trees Number of trees in an ensemble to be sampled
   * \param num_observations Number of rows in `covariates`
   */
  ForestTracker(Eigen::MatrixXd& covariates, std::vector<FeatureType>& feature_types, int num_trees, int num_observations);
  ~ForestTracker() {}
  void ReconstituteFromForest(TreeEnsemble& forest, ForestDataset& dataset, ColumnVector& residual, bool is_mean_model);
  void AssignAllSamplesToRoot();
  void AssignAllSamplesToRoot(int32_t tree_num);
  void AssignAllSamplesToConstantPrediction(double value);
  void AssignAllSamplesToConstantPrediction(int32_t tree_num, double value);
  void UpdatePredictions(TreeEnsemble* ensemble, ForestDataset& dataset);
  void UpdateSampleTrackers(TreeEnsemble& forest, ForestDataset& dataset);
  void UpdateSampleTrackersResidual(TreeEnsemble& forest, ForestDataset& dataset, ColumnVector& residual, bool is_mean_model);
  void ResetRoot(Eigen::MatrixXd& covariates, std::vector<FeatureType>& feature_types, int32_t tree_num);
  void AddSplit(Eigen::MatrixXd& covariates, TreeSplit& split, int32_t split_feature, int32_t tree_id, int32_t split_node_id, int32_t left_node_id, int32_t right_node_id, bool keep_sorted = false, int num_threads = -1);
  void RemoveSplit(Eigen::MatrixXd& covariates, Tree* tree, int32_t tree_id, int32_t split_node_id, int32_t left_node_id, int32_t right_node_id, bool keep_sorted = false);
  double GetSamplePrediction(data_size_t sample_id);
  double GetTreeSamplePrediction(data_size_t sample_id, int tree_id);
  void UpdateVarWeightsFromInternalPredictions(ForestDataset& dataset);
  void SetSamplePrediction(data_size_t sample_id, double value);
  void SetTreeSamplePrediction(data_size_t sample_id, int tree_id, double value);
  void SyncPredictions();
  data_size_t GetNodeId(int observation_num, int tree_num);
  data_size_t UnsortedNodeBegin(int tree_id, int node_id);
  data_size_t UnsortedNodeEnd(int tree_id, int node_id);
  data_size_t UnsortedNodeSize(int tree_id, int node_id);
  data_size_t SortedNodeBegin(int node_id, int feature_id);
  data_size_t SortedNodeEnd(int node_id, int feature_id);
  data_size_t SortedNodeSize(int node_id, int feature_id);
  std::vector<data_size_t>::iterator UnsortedNodeBeginIterator(int tree_id, int node_id);
  std::vector<data_size_t>::iterator UnsortedNodeEndIterator(int tree_id, int node_id);
  std::vector<data_size_t>::iterator SortedNodeBeginIterator(int node_id, int feature_id);
  std::vector<data_size_t>::iterator SortedNodeEndIterator(int node_id, int feature_id);
  SamplePredMapper* GetSamplePredMapper() {return sample_pred_mapper_.get();}
  SampleNodeMapper* GetSampleNodeMapper() {return sample_node_mapper_.get();}
  UnsortedNodeSampleTracker* GetUnsortedNodeSampleTracker() {return unsorted_node_sample_tracker_.get();}
  SortedNodeSampleTracker* GetSortedNodeSampleTracker() {return sorted_node_sample_tracker_.get();}
  int GetNumObservations() {return num_observations_;}
  int GetNumTrees() {return num_trees_;}
  int GetNumFeatures() {return num_features_;}
  bool Initialized() {return initialized_;}

 private:
  /*! \brief Mapper from observations to predicted values summed over every tree in a forest */
  std::vector<double> sum_predictions_;
  /*! \brief Mapper from observations to predicted values for every tree in a forest */
  std::unique_ptr<SamplePredMapper> sample_pred_mapper_;
  /*! \brief Mapper from observations to leaf node indices for every tree in a forest */
  std::unique_ptr<SampleNodeMapper> sample_node_mapper_;
  /*! \brief Data structure tracking / updating observations available in each node for every tree in a forest
   *  Primarily used in MCMC algorithms
   */
  std::unique_ptr<UnsortedNodeSampleTracker> unsorted_node_sample_tracker_;
  /*! \brief Data structure tracking / updating observations available in each node for each feature (pre-sorted) for a given tree in a forest 
   *  Primarily used in GFR algorithms
   */
  std::unique_ptr<FeaturePresortRootContainer> presort_container_;
  std::unique_ptr<SortedNodeSampleTracker> sorted_node_sample_tracker_;
  std::vector<FeatureType> feature_types_;
  int num_trees_;
  int num_observations_;
  int num_features_;
  bool initialized_{false};

  void UpdatePredictionsInternal(TreeEnsemble* ensemble, Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis);
  void UpdatePredictionsInternal(TreeEnsemble* ensemble, Eigen::MatrixXd& covariates);
  void UpdateSampleTrackersInternal(TreeEnsemble& forest, Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis);
  void UpdateSampleTrackersInternal(TreeEnsemble& forest, Eigen::MatrixXd& covariates);
  void UpdateSampleTrackersResidualInternalBasis(TreeEnsemble& forest, ForestDataset& dataset, ColumnVector& residual, bool is_mean_model);
  void UpdateSampleTrackersResidualInternalNoBasis(TreeEnsemble& forest, ForestDataset& dataset, ColumnVector& residual, bool is_mean_model);
};

/*! \brief Class storing sample-prediction map for each tree in an ensemble */
class SamplePredMapper {
 public:
  SamplePredMapper(int num_trees, data_size_t num_observations) {
    num_trees_ = num_trees;
    num_observations_ = num_observations;
    // Initialize the vector of vectors of leaf indices for each tree
    tree_preds_.resize(num_trees_);
    for (int j = 0; j < num_trees_; j++) {
      tree_preds_[j].resize(num_observations_);
    }
  }

  inline double GetPred(data_size_t sample_id, int tree_id) {
    CHECK_LT(sample_id, num_observations_);
    CHECK_LT(tree_id, num_trees_);
    return tree_preds_[tree_id][sample_id];
  }

  inline void SetPred(data_size_t sample_id, int tree_id, double value) {
    CHECK_LT(sample_id, num_observations_);
    CHECK_LT(tree_id, num_trees_);
    tree_preds_[tree_id][sample_id] = value;
  }
  
  inline int NumTrees() {return num_trees_;}
  
  inline int NumObservations() {return num_observations_;}

  inline void AssignAllSamplesToConstantPrediction(int tree_id, double value) {
    for (data_size_t i = 0; i < num_observations_; i++) {
      tree_preds_[tree_id][i] = value;
    }
  }

 private:
  std::vector<std::vector<double>> tree_preds_;
  int num_trees_;
  data_size_t num_observations_;
};

/*! \brief Class storing sample-node map for each tree in an ensemble */
class SampleNodeMapper {
 public:
  SampleNodeMapper(int num_trees, data_size_t num_observations) {
    num_trees_ = num_trees;
    num_observations_ = num_observations;
    // Initialize the vector of vectors of leaf indices for each tree
    tree_observation_indices_.resize(num_trees_);
    for (int j = 0; j < num_trees_; j++) {
      tree_observation_indices_[j].resize(num_observations_);
    }
  }
  
  SampleNodeMapper(SampleNodeMapper& other){
    num_trees_ = other.NumTrees();
    num_observations_ = other.NumObservations();
    // Initialize the vector of vectors of leaf indices for each tree
    tree_observation_indices_.resize(num_trees_);
    for (int j = 0; j < num_trees_; j++) {
      tree_observation_indices_[j].resize(num_observations_);
      for (int i = 0; i < num_observations_; i++) {
        tree_observation_indices_[j][i] = other.GetNodeId(i, j);
      }
    }
  }

  void AddSplit(Eigen::MatrixXd& covariates, TreeSplit& split, int32_t split_feature, int32_t tree_id, int32_t split_node_id, int32_t left_node_id, int32_t right_node_id) {
    CHECK_EQ(num_observations_, covariates.rows());
    // Eigen::MatrixXd X = covariates.GetData();
    for (int i = 0; i < num_observations_; i++) {
      if (tree_observation_indices_[tree_id][i] == split_node_id) {
        auto fvalue = covariates(i, split_feature);
        if (split.SplitTrue(fvalue)) {
          tree_observation_indices_[tree_id][i] = left_node_id;
        } else {
          tree_observation_indices_[tree_id][i] = right_node_id;
        }
      }
    }
  }

  inline data_size_t GetNodeId(data_size_t sample_id, int tree_id) {
    CHECK_LT(sample_id, num_observations_);
    CHECK_LT(tree_id, num_trees_);
    return tree_observation_indices_[tree_id][sample_id];
  }

  inline void SetNodeId(data_size_t sample_id, int tree_id, int node_id) {
    CHECK_LT(sample_id, num_observations_);
    CHECK_LT(tree_id, num_trees_);
    tree_observation_indices_[tree_id][sample_id] = node_id;
  }
  
  inline int NumTrees() {return num_trees_;}
  
  inline int NumObservations() {return num_observations_;}

  inline void AssignAllSamplesToRoot(int tree_id) {
    for (data_size_t i = 0; i < num_observations_; i++) {
      tree_observation_indices_[tree_id][i] = 0;
    }
  }

 private:
  std::vector<std::vector<int>> tree_observation_indices_;
  int num_trees_;
  data_size_t num_observations_;
};

/*! \brief Mapping nodes to the indices they contain */
class FeatureUnsortedPartition {
 public:
  FeatureUnsortedPartition(data_size_t n);

  /*! \brief Reconstitute a tree partition tracker from root based on a tree */
  void ReconstituteFromTree(Tree& tree, ForestDataset& dataset);

  /*! \brief Partition a node based on a new split rule */
  void PartitionNode(Eigen::MatrixXd& covariates, int node_id, int left_node_id, int right_node_id, int feature_split, TreeSplit& split);

  /*! \brief Partition a node based on a new split rule */
  void PartitionNode(Eigen::MatrixXd& covariates, int node_id, int left_node_id, int right_node_id, int feature_split, double split_value);

  /*! \brief Partition a node based on a new split rule */
  void PartitionNode(Eigen::MatrixXd& covariates, int node_id, int left_node_id, int right_node_id, int feature_split, std::vector<std::uint32_t> const& category_list);

  /*! \brief Convert a (currently split) node to a leaf */
  void PruneNodeToLeaf(int node_id);

  /*! \brief Whether node_id is a leaf */
  bool IsLeaf(int node_id);

  /*! \brief Whether node_id is a valid node */
  bool IsValidNode(int node_id);

  /*! \brief Whether node_id's left child is a leaf */
  bool LeftNodeIsLeaf(int node_id);

  /*! \brief Whether node_id's right child is a leaf */
  bool RightNodeIsLeaf(int node_id);

  /*! \brief First index of data points contained in node_id */
  data_size_t NodeBegin(int node_id);

  /*! \brief One past the last index of data points contained in node_id */
  data_size_t NodeEnd(int node_id);

  /*! \brief Number of data points contained in node_id */
  data_size_t NodeSize(int node_id);

  /*! \brief Parent node_id */
  int Parent(int node_id);

  /*! \brief Left child of node_id */
  int LeftNode(int node_id);

  /*! \brief Right child of node_id */
  int RightNode(int node_id);

  /*! \brief Data indices */
  std::vector<data_size_t> indices_;

  /*! \brief Data indices for a given node */
  std::vector<data_size_t> NodeIndices(int node_id);

  /*! \brief Update SampleNodeMapper for all the observations in node_id */
  void UpdateObservationMapping(int node_id, int tree_id, SampleNodeMapper* sample_node_mapper);

 private:
  // Vectors tracking indices in each node
  std::vector<data_size_t> node_begin_;
  std::vector<data_size_t> node_length_;
  std::vector<int32_t> parent_nodes_;
  std::vector<int32_t> left_nodes_;
  std::vector<int32_t> right_nodes_;
  int num_nodes_, num_deleted_nodes_;
  std::vector<int> deleted_nodes_;

  // Private helper functions
  void ExpandNodeTrackingVectors(int node_id, int left_node_id, int right_node_id, data_size_t node_start_idx, data_size_t num_left, data_size_t num_right);
  void ConvertLeafParentToLeaf(int node_id);
};

/*! \brief Mapping nodes to the indices they contain */
class UnsortedNodeSampleTracker {
 public:
  UnsortedNodeSampleTracker(data_size_t n, int num_trees) {
    feature_partitions_.resize(num_trees);
    num_trees_ = num_trees;
    for (int i = 0; i < num_trees; i++) {
      feature_partitions_[i].reset(new FeatureUnsortedPartition(n));
    }
  }

  /*! \brief Reconstruct the node sample tracker based on the splits in a forest */
  void ReconstituteFromForest(TreeEnsemble& forest, ForestDataset& dataset);

  /*! \brief Partition a node based on a new split rule */
  void PartitionTreeNode(Eigen::MatrixXd& covariates, int tree_id, int node_id, int left_node_id, int right_node_id, int feature_split, TreeSplit& split) {
    return feature_partitions_[tree_id]->PartitionNode(covariates, node_id, left_node_id, right_node_id, feature_split, split);
  }

  /*! \brief Partition a node based on a new split rule */
  void PartitionTreeNode(Eigen::MatrixXd& covariates, int tree_id, int node_id, int left_node_id, int right_node_id, int feature_split, double split_value) {
    return feature_partitions_[tree_id]->PartitionNode(covariates, node_id, left_node_id, right_node_id, feature_split, split_value);
  }

  /*! \brief Partition a node based on a new split rule */
  void PartitionTreeNode(Eigen::MatrixXd& covariates, int tree_id, int node_id, int left_node_id, int right_node_id, int feature_split, std::vector<std::uint32_t> const& category_list) {
    return feature_partitions_[tree_id]->PartitionNode(covariates, node_id, left_node_id, right_node_id, feature_split, category_list);
  }
  
  /*! \brief Convert a tree to root */
  void ResetTreeToRoot(int tree_id, data_size_t n) {
    feature_partitions_[tree_id].reset(new FeatureUnsortedPartition(n));;
  }

  /*! \brief Convert a (currently split) node to a leaf */
  void PruneTreeNodeToLeaf(int tree_id, int node_id) {
    return feature_partitions_[tree_id]->PruneNodeToLeaf(node_id);
  }

  /*! \brief Whether node_id is a leaf */
  bool IsLeaf(int tree_id, int node_id) {
    return feature_partitions_[tree_id]->IsLeaf(node_id);
  }

  /*! \brief Whether node_id is a valid node */
  bool IsValidNode(int tree_id, int node_id) {
    return feature_partitions_[tree_id]->IsValidNode(node_id);
  }

  /*! \brief Whether node_id's left child is a leaf */
  bool LeftNodeIsLeaf(int tree_id, int node_id) {
    return feature_partitions_[tree_id]->LeftNodeIsLeaf(node_id);
  }

  /*! \brief Whether node_id's right child is a leaf */
  bool RightNodeIsLeaf(int tree_id, int node_id) {
    return feature_partitions_[tree_id]->RightNodeIsLeaf(node_id);
  }

  /*! \brief First index of data points contained in node_id */
  data_size_t NodeBegin(int tree_id, int node_id) {
    return feature_partitions_[tree_id]->NodeBegin(node_id);
  }

  /*! \brief One past the last index of data points contained in node_id */
  data_size_t NodeEnd(int tree_id, int node_id) {
    return feature_partitions_[tree_id]->NodeEnd(node_id);
  }

  std::vector<data_size_t>::iterator NodeBeginIterator(int tree_id, int node_id) {
    data_size_t node_begin = feature_partitions_[tree_id]->NodeBegin(node_id);
    auto begin_iter = feature_partitions_[tree_id]->indices_.begin();
    return begin_iter + node_begin;
  }

  std::vector<data_size_t>::iterator NodeEndIterator(int tree_id, int node_id) {
    int node_end = feature_partitions_[tree_id]->NodeEnd(node_id);
    auto begin_iter = feature_partitions_[tree_id]->indices_.begin();
    return begin_iter + node_end;
  }

  /*! \brief One past the last index of data points contained in node_id */
  data_size_t NodeSize(int tree_id, int node_id) {
    return feature_partitions_[tree_id]->NodeSize(node_id);
  }

  /*! \brief Parent node_id */
  int Parent(int tree_id, int node_id) {
    return feature_partitions_[tree_id]->Parent(node_id);
  }

  /*! \brief Left child of node_id */
  int LeftNode(int tree_id, int node_id) {
    return feature_partitions_[tree_id]->LeftNode(node_id);
  }

  /*! \brief Right child of node_id */
  int RightNode(int tree_id, int node_id) {
    return feature_partitions_[tree_id]->RightNode(node_id);
  }

  /*! \brief Data indices for a given node */
  std::vector<data_size_t> TreeNodeIndices(int tree_id, int node_id) {
    return feature_partitions_[tree_id]->NodeIndices(node_id);
  }

  /*! \brief Update SampleNodeMapper for all the observations in node_id */
  void UpdateObservationMapping(int node_id, int tree_id, SampleNodeMapper* sample_node_mapper) {
    feature_partitions_[tree_id]->UpdateObservationMapping(node_id, tree_id, sample_node_mapper);
  }

  /*! \brief Update SampleNodeMapper for all the observations in tree */
  void UpdateObservationMapping(Tree* tree, int tree_id, SampleNodeMapper* sample_node_mapper) {
    std::vector<int> leaves = tree->GetLeaves();
    int leaf;
    for (int i = 0; i < leaves.size(); i++) {
      leaf = leaves[i];
      UpdateObservationMapping(leaf, tree_id, sample_node_mapper);
    }
  }

  /*! \brief Number of trees */
  int NumTrees() { return num_trees_; }

  /*! \brief Return a pointer to the feature partition tracking tree i */
  FeatureUnsortedPartition* GetFeaturePartition(int i) { return feature_partitions_[i].get(); }

 private:
  // Vectors of feature partitions
  std::vector<std::unique_ptr<FeatureUnsortedPartition>> feature_partitions_;
  int num_trees_;
};

/*! \brief Tracking cutpoints available at a given node */
class NodeOffsetSize {
 public:
  NodeOffsetSize(data_size_t node_offset, data_size_t node_size) : node_begin_{node_offset}, node_size_{node_size}, presorted_{false} {
    node_end_ = node_begin_ + node_size_;
  }

  ~NodeOffsetSize() {}

  void SetSorted() {presorted_ = true;}

  bool IsSorted() {return presorted_;}

  data_size_t Begin() {return node_begin_;}

  data_size_t End() {return node_end_;}

  data_size_t Size() {return node_size_;}

 private:
  data_size_t node_begin_;
  data_size_t node_size_;
  data_size_t node_end_;
  bool presorted_;
};

/*! \brief Forward declaration of partition-based presort tracker */
class FeaturePresortPartition;

/*! \brief Data structure for presorting a feature by its values
 * 
 *  This class is intended to be run *once* on a dataset as it 
 *  pre-sorts each feature across the entire dataset.
 *  
 *  FeaturePresortPartition is intended for use in recursive construction
 *  of new trees, and each new tree's FeaturePresortPartition is initialized 
 *  from a FeaturePresortRoot class so that features are only arg-sorted one time.
 */
class FeaturePresortRoot {
 friend FeaturePresortPartition; 
 public:
  FeaturePresortRoot(Eigen::MatrixXd& covariates, int32_t feature_index, FeatureType feature_type) {
    feature_index_ = feature_index;
    ArgsortRoot(covariates);
  }

  ~FeaturePresortRoot() {}

  void ArgsortRoot(Eigen::MatrixXd& covariates) {
    data_size_t num_obs = covariates.rows();
    
    // Make a vector of indices from 0 to num_obs - 1
    if (feature_sort_indices_.size() != num_obs){
      feature_sort_indices_.resize(num_obs, 0);
    }
    std::iota(feature_sort_indices_.begin(), feature_sort_indices_.end(), 0);

    // Define a custom comparator to be used with stable_sort:
    // For every two indices l and r store as elements of `data_sort_indices_`, 
    // compare them for sorting purposes by indexing the covariate's raw data with both l and r
    auto comp_op = [&](size_t const &l, size_t const &r) { return std::less<double>{}(covariates(l, feature_index_), covariates(r, feature_index_)); };
    std::stable_sort(feature_sort_indices_.begin(), feature_sort_indices_.end(), comp_op);
  }

 private:
  std::vector<data_size_t> feature_sort_indices_;
  int32_t feature_index_;
};

/*! \brief Container class for FeaturePresortRoot objects stored for every feature in a dataset */
class FeaturePresortRootContainer {
 public:
  FeaturePresortRootContainer(Eigen::MatrixXd& covariates, std::vector<FeatureType>& feature_types) {
    num_features_ = covariates.cols();
    feature_presort_.resize(num_features_);
    for (int i = 0; i < num_features_; i++) {
      feature_presort_[i].reset(new FeaturePresortRoot(covariates, i, feature_types[i]));
    }
  }

  ~FeaturePresortRootContainer() {}

  FeaturePresortRoot* GetFeaturePresort(int feature_num) {return feature_presort_[feature_num].get(); }

 private:
  std::vector<std::unique_ptr<FeaturePresortRoot>> feature_presort_;
  int num_features_;
};

/*! \brief Data structure that tracks pre-sorted feature values 
 *  through a tree's split lifecycle
 * 
 *  This class is initialized from a FeaturePresortRoot which has computed the 
 *  sort indices for a given feature over the entire dataset, so that sorting 
 *  is not necessary for each new tree.
 *  
 *  When a split is made, this class handles sifting for each feature, so that 
 *  the presorted feature values available at each node are easily queried.
 */
class FeaturePresortPartition {
 public:
  FeaturePresortPartition(FeaturePresortRoot* feature_presort_root, Eigen::MatrixXd& covariates, int32_t feature_index, FeatureType feature_type) {
    // Unpack all feature details
    feature_index_ = feature_index;
    feature_type_ = feature_type;
    num_obs_ = covariates.rows();
    feature_sort_indices_ = feature_presort_root->feature_sort_indices_;

    // Initialize new tree to root
    data_size_t node_offset = 0;
    node_offset_sizes_.emplace_back(node_offset, num_obs_);
  }

  ~FeaturePresortPartition() {}

  /*! \brief Split numeric / ordered categorical feature and update sort indices */
  void SplitFeature(Eigen::MatrixXd& covariates, int32_t node_id, int32_t feature_index, TreeSplit& split);

  /*! \brief Split numeric / ordered categorical feature and update sort indices */
  void SplitFeatureNumeric(Eigen::MatrixXd& covariates, int32_t node_id, int32_t feature_index, double split_value);

  /*! \brief Split unordered categorical feature and update sort indices */
  void SplitFeatureCategorical(Eigen::MatrixXd& covariates, int32_t node_id, int32_t feature_index, std::vector<std::uint32_t> const& category_list);

  /*! \brief Start position of node indexed by node_id */
  data_size_t NodeBegin(int32_t node_id) {return node_offset_sizes_[node_id].Begin();}

  /*! \brief End position of node indexed by node_id */
  data_size_t NodeEnd(int32_t node_id) {return node_offset_sizes_[node_id].End();}

  /*! \brief Size (in observations) of node indexed by node_id */
  data_size_t NodeSize(int32_t node_id) {return node_offset_sizes_[node_id].Size();}

  /*! \brief Data indices for a given node */
  std::vector<data_size_t> NodeIndices(int node_id);

  /*! \brief Feature sort index j */
  data_size_t SortIndex(data_size_t j) {return feature_sort_indices_[j];}

  /*! \brief Feature type */
  FeatureType GetFeatureType() {return feature_type_;}

  /*! \brief Update SampleNodeMapper for all the observations in node_id */
  void UpdateObservationMapping(int node_id, int tree_id, SampleNodeMapper* sample_node_mapper);

  /*! \brief Feature sort indices */
  std::vector<data_size_t> feature_sort_indices_;
 private:
  /*! \brief Add left and right nodes */
  void AddLeftRightNodes(data_size_t left_node_begin, data_size_t left_node_size, data_size_t right_node_begin, data_size_t right_node_size);

  /*! \brief Other node tracking information */
  std::vector<NodeOffsetSize> node_offset_sizes_;
  int32_t feature_index_;
  FeatureType feature_type_;
  data_size_t num_obs_;
};

/*! \brief Data structure for tracking observations through a tree partition with each feature pre-sorted */
class SortedNodeSampleTracker {
 public:
  SortedNodeSampleTracker(FeaturePresortRootContainer* feature_presort_root_container, Eigen::MatrixXd& covariates, std::vector<FeatureType>& feature_types) {
    num_features_ = covariates.cols();
    feature_partitions_.resize(num_features_);
    FeaturePresortRoot* feature_presort_root;
    for (int i = 0; i < num_features_; i++) {
      feature_presort_root = feature_presort_root_container->GetFeaturePresort(i);
      feature_partitions_[i].reset(new FeaturePresortPartition(feature_presort_root, covariates, i, feature_types[i]));
    }
  }

  /*! \brief Partition a node based on a new split rule */
  void PartitionNode(Eigen::MatrixXd& covariates, int node_id, int feature_split, TreeSplit& split, int num_threads = -1) {
    StochTree::ParallelFor(0, num_features_, num_threads, [&](int i) {
      feature_partitions_[i]->SplitFeature(covariates, node_id, feature_split, split);
    });
  }

  /*! \brief Partition a node based on a new split rule */
  void PartitionNode(Eigen::MatrixXd& covariates, int node_id, int feature_split, double split_value, int num_threads = -1) {
    StochTree::ParallelFor(0, num_features_, num_threads, [&](int i) {
      feature_partitions_[i]->SplitFeatureNumeric(covariates, node_id, feature_split, split_value);
    });
  }

  /*! \brief Partition a node based on a new split rule */
  void PartitionNode(Eigen::MatrixXd& covariates, int node_id, int feature_split, std::vector<std::uint32_t> const& category_list, int num_threads = -1) {
    StochTree::ParallelFor(0, num_features_, num_threads, [&](int i) {
      feature_partitions_[i]->SplitFeatureCategorical(covariates, node_id, feature_split, category_list);
    });
  }

  /*! \brief First index of data points contained in node_id */
  data_size_t NodeBegin(int node_id, int feature_index) {
    return feature_partitions_[feature_index]->NodeBegin(node_id);
  }

  /*! \brief One past the last index of data points contained in node_id */
  data_size_t NodeEnd(int node_id, int feature_index) {
    return feature_partitions_[feature_index]->NodeEnd(node_id);
  }

  /*! \brief One past the last index of data points contained in node_id */
  data_size_t NodeSize(int node_id, int feature_index) {
    return feature_partitions_[feature_index]->NodeSize(node_id);
  }

  std::vector<data_size_t>::iterator NodeBeginIterator(int node_id, int feature_index) {
    data_size_t node_begin = NodeBegin(node_id, feature_index);
    auto begin_iter = feature_partitions_[feature_index]->feature_sort_indices_.begin();
    return begin_iter + node_begin;
  }

  std::vector<data_size_t>::iterator NodeEndIterator(int node_id, int feature_index) {
    data_size_t node_end = NodeEnd(node_id, feature_index);
    auto begin_iter = feature_partitions_[feature_index]->feature_sort_indices_.begin();
    return begin_iter + node_end;
  }

  /*! \brief Data indices for a given node */
  std::vector<data_size_t> NodeIndices(int node_id, int feature_index) {
    return feature_partitions_[feature_index]->NodeIndices(node_id);
  }

  /*! \brief Feature sort index j for feature_index */
  data_size_t SortIndex(data_size_t j, int feature_index) {return feature_partitions_[feature_index]->SortIndex(j); }

  /*! \brief Update SampleNodeMapper for all the observations in node_id */
  void UpdateObservationMapping(int node_id, int tree_id, SampleNodeMapper* sample_node_mapper, int feature_index = 0) {
    feature_partitions_[feature_index]->UpdateObservationMapping(node_id, tree_id, sample_node_mapper);
  }

 private:
  std::vector<std::unique_ptr<FeaturePresortPartition>> feature_partitions_;
  int num_features_;
};

} // namespace StochTree

#endif // STOCHTREE_PARTITION_TRACKER_H_
