#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <testutils.h>
#include <stochtree/data.h>
#include <stochtree/log.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/tree.h>
#include <iostream>
#include <memory>
#include <vector>

TEST(UnsortedNodeSampleTracker, BasicOperations) {
  // Load test data
  StochTree::TestUtils::TestDataset test_dataset;
  test_dataset = StochTree::TestUtils::LoadSmallDatasetUnivariateBasis();

  // Construct datasets
  int n = test_dataset.n;
  StochTree::ForestDataset dataset = StochTree::ForestDataset();
  dataset.AddCovariates(test_dataset.covariates.data(), n, test_dataset.x_cols, test_dataset.row_major);
  dataset.AddBasis(test_dataset.omega.data(), test_dataset.n, test_dataset.omega_cols, test_dataset.row_major);
  StochTree::ColumnVector residual = StochTree::ColumnVector(test_dataset.outcome.data(), n);

  // Construct a NodeSampleTracker
  int num_trees = 1;
  StochTree::UnsortedNodeSampleTracker node_sample_tracker(n, num_trees);

  // Construct a SampleNodeMapper
  StochTree::SampleNodeMapper sample_node_mapper = StochTree::SampleNodeMapper(num_trees, n);
  sample_node_mapper.AssignAllSamplesToRoot(0);
  for (StochTree::data_size_t i = 0; i < n; i++) {
    ASSERT_EQ(sample_node_mapper.GetNodeId(i, 0), 0);
  }

  // Check leaf node begin and node end
  ASSERT_EQ(node_sample_tracker.NodeBegin(0, 0), 0);
  ASSERT_EQ(node_sample_tracker.NodeEnd(0, 0), n);

  // Partition based on X[,0] <= 0.5
  StochTree::TreeSplit tree_split = StochTree::TreeSplit(0.5);
  node_sample_tracker.PartitionTreeNode(dataset.GetCovariates(), 0, 0, 1, 2, 0, tree_split);
  sample_node_mapper.AddSplit(dataset.GetCovariates(), tree_split, 0, 0, 0, 1, 2);
  
  // Check that node begin and node end haven't changed for root node, but that the indices have been sifted
  ASSERT_EQ(node_sample_tracker.NodeBegin(0, 0), 0);
  ASSERT_EQ(node_sample_tracker.NodeEnd(0, 0), n);
  std::vector<StochTree::data_size_t> expected_result{2, 8, 9, 0, 1, 3, 4, 5, 6, 7};
  ASSERT_EQ(node_sample_tracker.TreeNodeIndices(0, 0), expected_result);
  
  // Check that terminal nodes are updated for for every observation
  ASSERT_EQ(sample_node_mapper.GetNodeId(0, 0), 2);
  ASSERT_EQ(sample_node_mapper.GetNodeId(1, 0), 2);
  ASSERT_EQ(sample_node_mapper.GetNodeId(2, 0), 1);
  ASSERT_EQ(sample_node_mapper.GetNodeId(3, 0), 2);
  ASSERT_EQ(sample_node_mapper.GetNodeId(4, 0), 2);
  ASSERT_EQ(sample_node_mapper.GetNodeId(5, 0), 2);
  ASSERT_EQ(sample_node_mapper.GetNodeId(6, 0), 2);
  ASSERT_EQ(sample_node_mapper.GetNodeId(7, 0), 2);
  ASSERT_EQ(sample_node_mapper.GetNodeId(8, 0), 1);
  ASSERT_EQ(sample_node_mapper.GetNodeId(9, 0), 1);
  
  // Check node begin and node end for left node
  ASSERT_EQ(node_sample_tracker.NodeBegin(0, 1), 0);
  ASSERT_EQ(node_sample_tracker.NodeEnd(0, 1), 3);
  expected_result = {2, 8, 9};
  ASSERT_EQ(node_sample_tracker.TreeNodeIndices(0, 1), expected_result);
  
  // Check node begin and node end for right node
  ASSERT_EQ(node_sample_tracker.NodeBegin(0, 2), 3);
  ASSERT_EQ(node_sample_tracker.NodeEnd(0, 2), n);
  expected_result = {0, 1, 3, 4, 5, 6, 7};
  ASSERT_EQ(node_sample_tracker.TreeNodeIndices(0, 2), expected_result);

  // Partition right node based on X[,1] <= 0.5
  tree_split = StochTree::TreeSplit(0.5);
  node_sample_tracker.PartitionTreeNode(dataset.GetCovariates(), 0, 2, 3, 4, 1, tree_split);
  sample_node_mapper.AddSplit(dataset.GetCovariates(), tree_split, 1, 0, 2, 3, 4);

  // Check that node begin and node end haven't changed for old right node, but that the indices have been sifted
  ASSERT_EQ(node_sample_tracker.NodeBegin(0, 2), 3);
  ASSERT_EQ(node_sample_tracker.NodeEnd(0, 2), n);
  expected_result = {1, 6, 0, 3, 4, 5, 7};
  ASSERT_EQ(node_sample_tracker.TreeNodeIndices(0, 2), expected_result);
  
  // Check node begin and node end for new left node
  ASSERT_EQ(node_sample_tracker.NodeBegin(0, 3), 3);
  ASSERT_EQ(node_sample_tracker.NodeEnd(0, 3), 5);
  expected_result = {1, 6};
  ASSERT_EQ(node_sample_tracker.TreeNodeIndices(0, 3), expected_result);
  
  // Check node begin and node end for new right node
  ASSERT_EQ(node_sample_tracker.NodeBegin(0, 4), 5);
  ASSERT_EQ(node_sample_tracker.NodeEnd(0, 4), n);
  expected_result = {0, 3, 4, 5, 7};
  ASSERT_EQ(node_sample_tracker.TreeNodeIndices(0, 4), expected_result);

  // Prune the node 2 split
  node_sample_tracker.PruneTreeNodeToLeaf(0, 2);
  ASSERT_TRUE(node_sample_tracker.IsLeaf(0, 2));
  ASSERT_FALSE(node_sample_tracker.IsValidNode(0, 3));
  ASSERT_FALSE(node_sample_tracker.IsValidNode(0, 4));
}
