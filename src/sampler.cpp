/*! Copyright (c) 2024 by stochtree authors. */
#include <stochtree/tree.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/sampler.h>
#include <Eigen/Dense>

namespace StochTree {

/*! \brief Partition the presorted feature node tracker used in the grow-from-root algorithm according to a numeric split */
void PartitionLeaf(Eigen::MatrixXd& covariates, SortedNodeSampleTracker* sorted_node_sample_tracker, int leaf_node, int split_col, double split_value) {
  sorted_node_sample_tracker->PartitionNode(covariates, leaf_node, split_col, split_value);
}

/*! \brief Partition the presorted feature node tracker used in the grow-from-root algorithm according to a categorical split */
void PartitionLeaf(Eigen::MatrixXd& covariates, SortedNodeSampleTracker* sorted_node_sample_tracker, int leaf_node, int split_col, std::vector<std::uint32_t>& categories) {
  sorted_node_sample_tracker->PartitionNode(covariates, leaf_node, split_col, categories);
}

/*! \brief Whether all leaf nodes are non-constant (in at least one covariate) after a proposed numeric split */
bool NodesNonConstantAfterSplit(Eigen::MatrixXd& covariates, Tree* tree, UnsortedNodeSampleTracker* node_tracker, int leaf_split, int feature_split, double split_value, int tree_num) {
  int p = covariates.cols();
  data_size_t idx;
  double feature_value;
  double split_feature_value;
  double var_max_left;
  double var_min_left;
  double var_max_right;
  double var_min_right;
  auto tree_node_tracker = node_tracker->GetFeaturePartition(tree_num);
  data_size_t node_begin = tree_node_tracker->NodeBegin(leaf_split);
  data_size_t node_end = tree_node_tracker->NodeEnd(leaf_split);

  for (int j = 0; j < p; j++) {
    var_max_left = std::numeric_limits<double>::min();
    var_min_left = std::numeric_limits<double>::max();
    var_max_right = std::numeric_limits<double>::min();
    var_min_right = std::numeric_limits<double>::max();
    auto node_begin_iter = tree_node_tracker->indices_.begin() + node_begin;
    auto node_end_iter = tree_node_tracker->indices_.begin() + node_end;
    for (auto i = node_begin_iter; i != node_end_iter; i++) {
      idx = *i;
      feature_value = covariates(idx, j);
      split_feature_value = covariates(idx, feature_split);
      if (SplitTrueNumeric(split_feature_value, split_value)) {
        if (var_max_left < feature_value) {
          var_max_left = feature_value;
        } else if (var_min_left > feature_value) {
          var_max_left = feature_value;
        }
      } else {
        if (var_max_right < feature_value) {
          var_max_right = feature_value;
        } else if (var_min_right > feature_value) {
          var_max_right = feature_value;
        }
      }
    }
    if ((var_max_left > var_min_left) && (var_max_right > var_min_right)) {
      return true;
    }
  }
  return false;
}

/*! \brief Whether all leaf nodes are non-constant (in at least one covariate) after a proposed categorical split */
bool NodesNonConstantAfterSplit(Eigen::MatrixXd& covariates, Tree* tree, UnsortedNodeSampleTracker* node_tracker, int leaf_split, int feature_split, std::vector<std::uint32_t> split_categories, int tree_num) {
  int p = covariates.cols();
  data_size_t idx;
  double feature_value;
  double split_feature_value;
  double var_max_left;
  double var_min_left;
  double var_max_right;
  double var_min_right;
  auto tree_node_tracker = node_tracker->GetFeaturePartition(tree_num);
  data_size_t node_begin = tree_node_tracker->NodeBegin(leaf_split);
  data_size_t node_end = tree_node_tracker->NodeEnd(leaf_split);

  for (int j = 0; j < p; j++) {
    var_max_left = std::numeric_limits<double>::min();
    var_min_left = std::numeric_limits<double>::max();
    var_max_right = std::numeric_limits<double>::min();
    var_min_right = std::numeric_limits<double>::max();
    auto node_begin_iter = tree_node_tracker->indices_.begin() + node_begin;
    auto node_end_iter = tree_node_tracker->indices_.begin() + node_end;
    for (auto i = node_begin_iter; i != node_end_iter; i++) {
      idx = *i;
      feature_value = covariates(idx, j);
      split_feature_value = covariates(idx, feature_split);
      if (SplitTrueCategorical(split_feature_value, split_categories)) {
        if (var_max_left < feature_value) {
          var_max_left = feature_value;
        } else if (var_min_left > feature_value) {
          var_max_left = feature_value;
        }
      } else {
        if (var_max_right < feature_value) {
          var_max_right = feature_value;
        } else if (var_min_right > feature_value) {
          var_max_right = feature_value;
        }
      }
    }
    if ((var_max_left > var_min_left) && (var_max_right > var_min_right)) {
      return true;
    }
  }
  return false;
}

/*! \brief Whether a given node is non-constant */
bool NodeNonConstant(Eigen::MatrixXd& covariates, Tree* tree, UnsortedNodeSampleTracker* node_tracker, int node_id, int tree_num) {
  int p = covariates.cols();
  double outcome_value;
  double feature_value;
  double split_feature_value;
  double var_max;
  double var_min;
  auto tree_node_tracker = node_tracker->GetFeaturePartition(tree_num);
  data_size_t node_begin = tree_node_tracker->NodeBegin(node_id);
  data_size_t node_end = tree_node_tracker->NodeEnd(node_id);
  data_size_t idx;

  for (int j = 0; j < p; j++) {
    var_max = std::numeric_limits<double>::min();
    var_min = std::numeric_limits<double>::max();
    auto node_begin_iter = tree_node_tracker->indices_.begin() + node_begin;
    auto node_end_iter = tree_node_tracker->indices_.begin() + node_end;
    for (auto i = node_begin_iter; i != node_end_iter; i++) {
      idx = *i;
      feature_value = covariates(idx, j);
      if (var_max < feature_value) {
        var_max = feature_value;
      } else if (var_min > feature_value) {
        var_max = feature_value;
      }
    }
    if (var_max > var_min) {
      return true;
    }
  }
  return false;
}

/*! \brief Range of possible split values for a given (numeric) feature */
void VarSplitRange(Eigen::MatrixXd& covariates, Tree* tree, UnsortedNodeSampleTracker* node_tracker, int leaf_split, int feature_split, double& var_min, double& var_max, int tree_num) {
  data_size_t n = covariates.rows();
  var_min = std::numeric_limits<double>::max();
  var_max = std::numeric_limits<double>::min();
  double feature_value;
  auto tree_node_tracker = node_tracker->GetFeaturePartition(tree_num);
  data_size_t node_begin = tree_node_tracker->NodeBegin(leaf_split);
  data_size_t node_end = tree_node_tracker->NodeEnd(leaf_split);
  data_size_t idx;
  auto node_begin_iter = tree_node_tracker->indices_.begin() + node_begin;
  auto node_end_iter = tree_node_tracker->indices_.begin() + node_end;
  for (auto i = node_begin_iter; i != node_end_iter; i++) {
    idx = *i;
    feature_value = covariates(idx, feature_split);
    if (feature_value < var_min) {
      var_min = feature_value;
    } else if (feature_value > var_max) {
      var_max = feature_value;
    }
  }
}

/*! \brief Adding a numeric split to a model and updating all of the relevant data structures that support sampling */
void AddSplitToModel(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, Tree* tree, UnsortedNodeSampleTracker* node_tracker, SampleNodeMapper* sample_node_mapper, int leaf_node, int feature_split, double split_value, int tree_num) {
  // Use zeros as a "temporary" leaf values since we draw leaf parameters after tree sampling is complete
  int basis_dim = basis.cols();
  if (basis_dim > 1) {
    std::vector<double> temp_leaf_values(basis_dim, 0.);
    tree->ExpandNode(leaf_node, feature_split, split_value, true, temp_leaf_values, temp_leaf_values);
  } else {
    CHECK_EQ(basis_dim, 1);
    double temp_leaf_value = 0.;
    tree->ExpandNode(leaf_node, feature_split, split_value, true, temp_leaf_value, temp_leaf_value);
  }
  int left_node = tree->LeftChild(leaf_node);
  int right_node = tree->RightChild(leaf_node);

  // Update the UnsortedNodeSampleTracker
  node_tracker->PartitionTreeNode(covariates, tree_num, leaf_node, left_node, right_node, feature_split, split_value);

  // Update the SampleNodeMapper
  node_tracker->UpdateObservationMapping(tree, tree_num, sample_node_mapper);
}

/*! \brief Removing a numeric split from a model and updating all of the relevant data structures that support sampling */
void RemoveSplitFromModel(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, Tree* tree, UnsortedNodeSampleTracker* node_tracker, SampleNodeMapper* sample_node_mapper, int leaf_node, int left_node, int right_node, int feature_split, double split_value, int tree_num) {
  // Use zeros as a "temporary" leaf values since we draw leaf parameters after tree sampling is complete
  int basis_dim = basis.cols();
  if (basis_dim > 1) {
    std::vector<double> temp_leaf_values(basis_dim, 0.);
    tree->ChangeToLeaf(leaf_node, temp_leaf_values);
  } else {
    CHECK_EQ(basis_dim, 1);
    double temp_leaf_value = 0.;
    tree->ChangeToLeaf(leaf_node, temp_leaf_value);
  }

  // Update the UnsortedNodeSampleTracker
  node_tracker->PruneTreeNodeToLeaf(tree_num, leaf_node);

  // Update the SampleNodeMapper
  node_tracker->UpdateObservationMapping(tree, tree_num, sample_node_mapper);
}

} // namespace StochTree
