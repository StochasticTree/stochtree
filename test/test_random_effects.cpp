/*!
 * Derived from xgboost tree unit test code:
 * https://github.com/dmlc/xgboost/blob/master/tests/cpp/tree/test_tree_model.cc
 */
#include <gtest/gtest.h>
#include <testutils.h>
#include <stochtree/data.h>
#include <stochtree/log.h>
#include <stochtree/random_effects.h>
#include <stochtree/tree.h>
#include <iostream>
#include <memory>

TEST(RandomEffects, Setup) {
  // Load test data
  StochTree::TestUtils::TestDataset test_dataset;
  test_dataset = StochTree::TestUtils::LoadSmallDatasetUnivariateBasis();
  std::vector<StochTree::FeatureType> feature_types(test_dataset.x_cols, StochTree::FeatureType::kNumeric);

  // Construct dataset
  int n = test_dataset.n;
  StochTree::RandomEffectsDataset dataset = StochTree::RandomEffectsDataset();
  dataset.AddBasis(test_dataset.rfx_basis.data(), test_dataset.n, test_dataset.rfx_basis_cols, test_dataset.row_major);
  dataset.AddGroupLabels(test_dataset.rfx_groups);
  
  // Construct tracker, model state, and container
  StochTree::RandomEffectsTracker tracker = StochTree::RandomEffectsTracker(test_dataset.rfx_groups);
  StochTree::MultivariateRegressionRandomEffectsModel model = StochTree::MultivariateRegressionRandomEffectsModel(test_dataset.rfx_basis_cols, test_dataset.rfx_num_groups);
  StochTree::RandomEffectsContainer container = StochTree::RandomEffectsContainer(test_dataset.rfx_basis_cols, test_dataset.rfx_num_groups);

  // Check the internal label map of the RandomEffectsTracker
  std::map<int32_t, int32_t> label_map = tracker.GetLabelMap();
  std::map<int32_t, int32_t> expected_label_map {{1, 0}, {2, 1}};
  ASSERT_EQ(label_map, expected_label_map);
}
