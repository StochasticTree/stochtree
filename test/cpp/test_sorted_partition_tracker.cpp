#include <gtest/gtest.h>
#include <testutils.h>
#include <stochtree/data.h>
#include <stochtree/log.h>
#include <stochtree/meta.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/tree.h>
#include <algorithm>
#include <limits>
#include <iostream>
#include <memory>

TEST(SortedNodeSampleTracker, BasicOperations) {
  // Load test data
  StochTree::TestUtils::TestDataset test_dataset;
  test_dataset = StochTree::TestUtils::LoadSmallDatasetUnivariateBasis();
  std::vector<StochTree::FeatureType> feature_types(test_dataset.x_cols, StochTree::FeatureType::kNumeric);

  // Construct datasets
  int n = test_dataset.n;
  StochTree::ForestDataset dataset = StochTree::ForestDataset();
  dataset.AddCovariates(test_dataset.covariates.data(), n, test_dataset.x_cols, test_dataset.row_major);
  dataset.AddBasis(test_dataset.omega.data(), test_dataset.n, test_dataset.omega_cols, test_dataset.row_major);
  StochTree::ColumnVector residual = StochTree::ColumnVector(test_dataset.outcome.data(), n);

  // Construct a container of presorted feature indices
  std::unique_ptr<StochTree::FeaturePresortRootContainer> presort_container = std::make_unique<StochTree::FeaturePresortRootContainer>(dataset.GetCovariates(), feature_types);

  // Construct a NodeSampleTracker
  StochTree::SortedNodeSampleTracker sorted_node_sampler_tracker(presort_container.get(), dataset.GetCovariates(), feature_types);

  // Construct a SampleNodeMapper
  int num_trees = 1;
  StochTree::SampleNodeMapper sample_node_mapper = StochTree::SampleNodeMapper(num_trees, n);
  sample_node_mapper.AssignAllSamplesToRoot(0);

  // Check leaf node begin and node end for feature 0
  ASSERT_EQ(sorted_node_sampler_tracker.NodeBegin(0, 0), 0);
  ASSERT_EQ(sorted_node_sampler_tracker.NodeEnd(0, 0), n);

  // Partition based on X[,0] <= 0.5
  StochTree::TreeSplit tree_split = StochTree::TreeSplit(0.5);
  sorted_node_sampler_tracker.PartitionNode(dataset.GetCovariates(), 0, 0, tree_split);
  sample_node_mapper.AddSplit(dataset.GetCovariates(), tree_split, 0, 0, 0, 1, 2);

  // Update the SampleNodeMapper based on the split
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
  
  // Check that node begin and node end haven't changed for root node, but that the indices have been sifted
  ASSERT_EQ(sorted_node_sampler_tracker.NodeBegin(0, 0), 0);
  ASSERT_EQ(sorted_node_sampler_tracker.NodeEnd(0, 0), n);
  std::vector<StochTree::data_size_t> expected_result{2,8,9,4,1,6,3,7,0,5};
  ASSERT_EQ(sorted_node_sampler_tracker.NodeIndices(0, 0), expected_result);
  
  // Check node begin and node end for left node
  ASSERT_EQ(sorted_node_sampler_tracker.NodeBegin(1, 0), 0);
  ASSERT_EQ(sorted_node_sampler_tracker.NodeEnd(1, 0), 3);
  expected_result = {2,8,9};
  ASSERT_EQ(sorted_node_sampler_tracker.NodeIndices(1, 0), expected_result);
  
  // Check node begin and node end for right node
  ASSERT_EQ(sorted_node_sampler_tracker.NodeBegin(2, 0), 3);
  ASSERT_EQ(sorted_node_sampler_tracker.NodeEnd(2, 0), n);
  expected_result = {4,1,6,3,7,0,5};
  ASSERT_EQ(sorted_node_sampler_tracker.NodeIndices(2, 0), expected_result);

  // Partition right node based on X[,1] <= 0.5
  sorted_node_sampler_tracker.PartitionNode(dataset.GetCovariates(), 2, 1, tree_split);

  // Check that node begin and node end haven't changed for old right node, but that the indices have been sifted
  ASSERT_EQ(sorted_node_sampler_tracker.NodeBegin(2, 0), 3);
  ASSERT_EQ(sorted_node_sampler_tracker.NodeEnd(2, 0), n);
  expected_result = {1,6,4,3,7,0,5};
  ASSERT_EQ(sorted_node_sampler_tracker.NodeIndices(2, 0), expected_result);

  // Check same indices for feature 1
  ASSERT_EQ(sorted_node_sampler_tracker.NodeBegin(2, 1), 3);
  ASSERT_EQ(sorted_node_sampler_tracker.NodeEnd(2, 1), n);
  expected_result = {6,1,3,0,7,4,5};
  ASSERT_EQ(sorted_node_sampler_tracker.NodeIndices(2, 1), expected_result);
  
  // Check node begin and node end for new left node
  ASSERT_EQ(sorted_node_sampler_tracker.NodeBegin(3, 1), 3);
  ASSERT_EQ(sorted_node_sampler_tracker.NodeEnd(3, 1), 5);
  expected_result = {6,1};
  ASSERT_EQ(sorted_node_sampler_tracker.NodeIndices(3, 1), expected_result);
  
  // Check node begin and node end for new right node
  ASSERT_EQ(sorted_node_sampler_tracker.NodeBegin(4, 1), 5);
  ASSERT_EQ(sorted_node_sampler_tracker.NodeEnd(4, 1), n);
  expected_result = {3,0,7,4,5};
  ASSERT_EQ(sorted_node_sampler_tracker.NodeIndices(4, 1), expected_result);
}

// Exercise the production presort through its public partition view.
TEST(FeaturePresortRoot, MissingValuesSortLastAcrossPermutations) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::vector<double> values{-1.0, 0.0, 1.0, nan, nan, 1.0};
  std::vector<int> permutation{0, 1, 2, 3, 4, 5};
  do {
    Eigen::MatrixXd covariates(6, 1);
    for (int i = 0; i < 6; ++i) covariates(i, 0) = values[permutation[i]];
    // Construct expected order independently; preserve ties and NaN order.
    std::vector<StochTree::data_size_t> expected;
    for (double value : {-1.0, 0.0, 1.0}) {
      for (int i = 0; i < 6; ++i) {
        if (covariates(i, 0) == value) expected.push_back(i);
      }
    }
    for (int i = 0; i < 6; ++i) {
      if (std::isnan(covariates(i, 0))) expected.push_back(i);
    }
    StochTree::FeaturePresortRoot root(covariates, 0, StochTree::FeatureType::kNumeric);
    StochTree::FeaturePresortPartition partition(&root, covariates, 0, StochTree::FeatureType::kNumeric);
    ASSERT_EQ(partition.NodeIndices(0), expected);
    // The scored sorted prefix must agree with numeric routing (NaNs go right).
    partition.SplitFeatureNumeric(covariates, 0, 0, 0.0);
    ASSERT_EQ(partition.NodeIndices(1), (std::vector<StochTree::data_size_t>{expected[0], expected[1]}));
    ASSERT_EQ(partition.NodeIndices(2), (std::vector<StochTree::data_size_t>{expected[2], expected[3], expected[4], expected[5]}));
  } while (std::next_permutation(permutation.begin(), permutation.end()));
}

TEST(FeaturePresortRoot, NonMissingOrderAndStableTiesArePreserved) {
  const double inf = std::numeric_limits<double>::infinity();
  Eigen::MatrixXd covariates(8, 1);
  covariates << 2.0, -0.0, inf, -2.0, 2.0, -inf, 0.0, -2.0;
  StochTree::FeaturePresortRoot root(covariates, 0, StochTree::FeatureType::kNumeric);
  StochTree::FeaturePresortPartition partition(&root, covariates, 0, StochTree::FeatureType::kNumeric);
  EXPECT_EQ(partition.NodeIndices(0), (std::vector<StochTree::data_size_t>{5, 3, 7, 1, 6, 0, 4, 2}));
}

TEST(FeaturePresortRoot, EmptySingletonAndAllMissingFeatures) {
  for (int n : {0, 1, 5}) {
    Eigen::MatrixXd covariates = Eigen::MatrixXd::Constant(n, 1, std::numeric_limits<double>::quiet_NaN());
    StochTree::FeaturePresortRoot root(covariates, 0, StochTree::FeatureType::kNumeric);
    StochTree::FeaturePresortPartition partition(&root, covariates, 0, StochTree::FeatureType::kNumeric);
    std::vector<StochTree::data_size_t> expected(n);
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT_EQ(partition.NodeIndices(0), expected);
  }
}
