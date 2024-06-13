#include <gtest/gtest.h>
#include <testutils.h>
#include <stochtree/cutpoint_candidates.h>
#include <stochtree/log.h>
#include <stochtree/partition_tracker.h>
#include <iostream>
#include <memory>
#include <vector>

TEST(CutpointGrid, NumericFeaturesGrid) {
  // Load test data
  StochTree::TestUtils::TestDataset test_dataset;
  test_dataset = StochTree::TestUtils::LoadSmallDatasetUnivariateBasis();
  std::vector<StochTree::FeatureType> feature_types(test_dataset.x_cols, StochTree::FeatureType::kNumeric);

  // Construct datasets
  using data_size_t = StochTree::data_size_t;
  data_size_t n = test_dataset.n;
  int p = test_dataset.x_cols;
  StochTree::ForestDataset dataset = StochTree::ForestDataset();
  dataset.AddCovariates(test_dataset.covariates.data(), n, test_dataset.x_cols, test_dataset.row_major);
  dataset.AddBasis(test_dataset.omega.data(), test_dataset.n, test_dataset.omega_cols, test_dataset.row_major);
  StochTree::ColumnVector residual = StochTree::ColumnVector(test_dataset.outcome.data(), n);

  // Construct a container of presorted feature indices
  std::unique_ptr<StochTree::FeaturePresortRootContainer> presort_container = std::make_unique<StochTree::FeaturePresortRootContainer>(dataset.GetCovariates(), feature_types);

  // Construct a NodeSampleTracker
  std::unique_ptr<StochTree::SortedNodeSampleTracker> sorted_node_sample_tracker = std::make_unique<StochTree::SortedNodeSampleTracker>(presort_container.get(), dataset.GetCovariates(), feature_types);

  // Check that indices are correctly sorted for feature 0 in the root node (i.e. entire dataset)
  ASSERT_EQ(sorted_node_sample_tracker->NodeBegin(0, 0), 0);
  ASSERT_EQ(sorted_node_sample_tracker->NodeEnd(0, 0), n);
  std::vector<data_size_t> expected_result{2, 8, 9, 4, 1, 6, 3, 7, 0, 5};
  ASSERT_EQ(sorted_node_sample_tracker->NodeIndices(0, 0), expected_result);

  // Check that indices are correctly sorted for feature 1 in the root node (i.e. entire dataset)
  ASSERT_EQ(sorted_node_sample_tracker->NodeBegin(0, 1), 0);
  ASSERT_EQ(sorted_node_sample_tracker->NodeEnd(0, 1), n);
  expected_result = {6, 2, 1, 8, 3, 9, 0, 7, 4, 5};
  ASSERT_EQ(sorted_node_sample_tracker->NodeIndices(0, 1), expected_result);

  // Enumerate cutpoint strides for each feature
  int cutpoint_grid_size = 5;
  StochTree::CutpointGridContainer cutpoint_grid_container = StochTree::CutpointGridContainer(dataset.GetCovariates(), residual.GetData(), cutpoint_grid_size);
  for (int j = 0; j < test_dataset.x_cols; j++) {
    cutpoint_grid_container.CalculateStrides(dataset.GetCovariates(), residual.GetData(), sorted_node_sample_tracker.get(), 0, 0, n, j, feature_types);
  }

  // Check cutpoint strides for feature 0
  double kDelta = 0.0001;
  ASSERT_EQ(cutpoint_grid_container.BinStartIndex(0, 0), 0);
  ASSERT_EQ(cutpoint_grid_container.BinLength(0, 0), 2);
  ASSERT_NEAR(cutpoint_grid_container.CutpointValue(0, 0), 0.3158837, kDelta);
  ASSERT_EQ(cutpoint_grid_container.BinStartIndex(1, 0), 2);
  ASSERT_EQ(cutpoint_grid_container.BinLength(1, 0), 2);
  ASSERT_NEAR(cutpoint_grid_container.CutpointValue(1, 0), 0.6181778, kDelta);
  ASSERT_EQ(cutpoint_grid_container.BinStartIndex(2, 0), 4);
  ASSERT_EQ(cutpoint_grid_container.BinLength(2, 0), 2);
  ASSERT_NEAR(cutpoint_grid_container.CutpointValue(2, 0), 0.7192248, kDelta);
  ASSERT_EQ(cutpoint_grid_container.BinStartIndex(3, 0), 6);
  ASSERT_EQ(cutpoint_grid_container.BinLength(3, 0), 2);
  ASSERT_NEAR(cutpoint_grid_container.CutpointValue(3, 0), 0.7474223, kDelta);
  ASSERT_EQ(cutpoint_grid_container.BinStartIndex(4, 0), 8);
  ASSERT_EQ(cutpoint_grid_container.BinLength(4, 0), 2);
  ASSERT_NEAR(cutpoint_grid_container.CutpointValue(4, 0), 0.8586578, kDelta);

  // Check cutpoint strides for feature 1
  ASSERT_EQ(cutpoint_grid_container.BinStartIndex(0, 1), 0);
  ASSERT_EQ(cutpoint_grid_container.BinLength(0, 1), 2);
  ASSERT_NEAR(cutpoint_grid_container.CutpointValue(0, 1), 0.1246148, kDelta);
  ASSERT_EQ(cutpoint_grid_container.BinStartIndex(1, 1), 2);
  ASSERT_EQ(cutpoint_grid_container.BinLength(1, 1), 2);
  ASSERT_NEAR(cutpoint_grid_container.CutpointValue(1, 1), 0.3925355, kDelta);
  ASSERT_EQ(cutpoint_grid_container.BinStartIndex(2, 1), 4);
  ASSERT_EQ(cutpoint_grid_container.BinLength(2, 1), 2);
  ASSERT_NEAR(cutpoint_grid_container.CutpointValue(2, 1), 0.5586495, kDelta);
  ASSERT_EQ(cutpoint_grid_container.BinStartIndex(3, 1), 6);
  ASSERT_EQ(cutpoint_grid_container.BinLength(3, 1), 2);
  ASSERT_NEAR(cutpoint_grid_container.CutpointValue(3, 1), 0.8717203, kDelta);
  ASSERT_EQ(cutpoint_grid_container.BinStartIndex(4, 1), 8);
  ASSERT_EQ(cutpoint_grid_container.BinLength(4, 1), 2);
  ASSERT_NEAR(cutpoint_grid_container.CutpointValue(4, 1), 0.9271676, kDelta);
}
