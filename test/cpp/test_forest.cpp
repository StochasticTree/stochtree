/*!
 * Derived from xgboost tree unit test code:
 * https://github.com/dmlc/xgboost/blob/master/tests/cpp/tree/test_tree_model.cc
 */
#include <gtest/gtest.h>
#include <testutils.h>
#include <stochtree/data.h>
#include <stochtree/ensemble.h>
#include <stochtree/log.h>
#include <stochtree/tree.h>
#include <iostream>
#include <memory>

TEST(Forest, UnivariateForestConstruction) {
  int num_trees = 2;
  int output_dimension = 1;
  bool is_leaf_constant = true;
  StochTree::TreeEnsemble forest(num_trees, output_dimension, is_leaf_constant);
  StochTree::Tree* tree_1_ptr = forest.GetTree(0);
  StochTree::Tree* tree_2_ptr = forest.GetTree(1);
  ASSERT_EQ(tree_1_ptr->LeafValue(0), 0.);
  tree_1_ptr->ExpandNode(0, 0, 0., 0., 0.);
  ASSERT_EQ(tree_1_ptr->NumNodes(), 3);
  ASSERT_EQ(tree_1_ptr->NodeType(0), StochTree::TreeNodeType::kNumericalSplitNode);
  tree_1_ptr->CollapseToLeaf(0, 0.);
  ASSERT_EQ(tree_1_ptr->NumValidNodes(), 1);
  tree_1_ptr->ExpandNode(0, 0, 0., 0., 0.);
  ASSERT_EQ(tree_1_ptr->NumValidNodes(), 3);
  ASSERT_EQ(tree_1_ptr->NodeType(0), StochTree::TreeNodeType::kNumericalSplitNode);
  ASSERT_EQ(tree_1_ptr->NumLeaves(), 2);
  ASSERT_FALSE(tree_1_ptr->IsLeaf(0));
  ASSERT_TRUE(tree_1_ptr->IsLeaf(1));
  ASSERT_TRUE(tree_1_ptr->IsLeaf(2));
}

TEST(Forest, UnivariateForestMerge) {
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
  
  // Create a small ensemble
  int output_dim = 1;
  int num_trees = 2;
  bool is_leaf_constant = true;
  StochTree::TreeEnsemble ensemble1(num_trees, output_dim, is_leaf_constant);
  
  // Create another small ensemble
  StochTree::TreeEnsemble ensemble2(num_trees, output_dim, is_leaf_constant);

  // Simple split rules on both trees of forest 1
  auto* tree = ensemble1.GetTree(0);
  StochTree::TreeSplit tree_split = StochTree::TreeSplit(0.5);
  tree->ExpandNode(0, 0, tree_split, -5., 5.);
  tree = ensemble1.GetTree(1);
  tree->ExpandNode(0, 1, tree_split, -2.5, 2.5);

  // Run predict on the supplied covariates and check the result for the first forest
  std::vector<double> result(n*output_dim);
  std::vector<double> expected_pred = {7.5,2.5,-7.5,7.5,7.5,7.5,2.5,7.5,-7.5,-2.5};
  ensemble1.PredictInplace(dataset.GetCovariates(), result, 0);
  for (int i = 0; i < n; i++) {
    ASSERT_NEAR(expected_pred[i], result[i], 0.01);
  }

  // Simple split rules on both trees of forest 2
  tree = ensemble2.GetTree(0);
  tree_split = StochTree::TreeSplit(0.6);
  tree->ExpandNode(0, 1, tree_split, -1., 1.);
  tree = ensemble2.GetTree(1);
  tree_split = StochTree::TreeSplit(0.5);
  tree->ExpandNode(0, 2, tree_split, -0.5, 0.5);

  // Run predict on the supplied covariates and check the result for the second forest
  result = std::vector<double>(n*output_dim);
  expected_pred = std::vector<double>{1.5,-1.5,-0.5,-0.5,1.5,1.5,-1.5,1.5,-0.5,-1.5};
  ensemble2.PredictInplace(dataset.GetCovariates(), result, 0);
  for (int i = 0; i < n; i++) {
    ASSERT_NEAR(expected_pred[i], result[i], 0.01);
  }

  // Merge the second forest into the first
  ensemble1.MergeForest(ensemble2);
  result = std::vector<double>(n*output_dim);
  expected_pred = std::vector<double>{9,1,-8,7,9,9,1,9,-8,-4};
  ensemble1.PredictInplace(dataset.GetCovariates(), result, 0);
  for (int i = 0; i < n; i++) {
    ASSERT_NEAR(expected_pred[i], result[i], 0.01);
  }
}

TEST(Forest, UnivariateForestAdd) {
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
  
  // Create a small ensemble
  int output_dim = 1;
  int num_trees = 2;
  bool is_leaf_constant = true;
  StochTree::TreeEnsemble ensemble1(num_trees, output_dim, is_leaf_constant);
  
  // Create another small ensemble
  StochTree::TreeEnsemble ensemble2(num_trees, output_dim, is_leaf_constant);

  // Simple split rules on both trees of forest 1
  auto* tree = ensemble1.GetTree(0);
  StochTree::TreeSplit tree_split = StochTree::TreeSplit(0.5);
  tree->ExpandNode(0, 0, tree_split, -5., 5.);
  tree = ensemble1.GetTree(1);
  tree->ExpandNode(0, 1, tree_split, -2.5, 2.5);

  // Run predict on the supplied covariates and check the result for the first forest
  std::vector<double> result(n*output_dim);
  std::vector<double> expected_pred = {7.5,2.5,-7.5,7.5,7.5,7.5,2.5,7.5,-7.5,-2.5};
  ensemble1.PredictInplace(dataset.GetCovariates(), result, 0);
  for (int i = 0; i < n; i++) {
    ASSERT_NEAR(expected_pred[i], result[i], 0.01);
  }

  // Add 1 to every leaf of the first forest
  ensemble1.AddValueToLeaves(1.0);

  // Run predict on the supplied covariates and check the result for the first forest
  result = std::vector<double>(n*output_dim);
  expected_pred = std::vector<double>{9.5,4.5,-5.5,9.5,9.5,9.5,4.5,9.5,-5.5,-0.5};
  ensemble1.PredictInplace(dataset.GetCovariates(), result, 0);
  for (int i = 0; i < n; i++) {
    ASSERT_NEAR(expected_pred[i], result[i], 0.01);
  }

  // Simple split rules on both trees of forest 2
  tree = ensemble2.GetTree(0);
  tree_split = StochTree::TreeSplit(0.6);
  tree->ExpandNode(0, 1, tree_split, -1., 1.);
  tree = ensemble2.GetTree(1);
  tree_split = StochTree::TreeSplit(0.5);
  tree->ExpandNode(0, 2, tree_split, -0.5, 0.5);

  // Run predict on the supplied covariates and check the result for the second forest
  result = std::vector<double>(n*output_dim);
  expected_pred = std::vector<double>{1.5,-1.5,-0.5,-0.5,1.5,1.5,-1.5,1.5,-0.5,-1.5};
  ensemble2.PredictInplace(dataset.GetCovariates(), result, 0);
  for (int i = 0; i < n; i++) {
    ASSERT_NEAR(expected_pred[i], result[i], 0.01);
  }

  // Subtract 1 from every leaf of the first forest
  ensemble2.AddValueToLeaves(-1.0);

  // Run predict on the supplied covariates and check the result for the first forest
  result = std::vector<double>(n*output_dim);
  expected_pred = std::vector<double>{-0.5,-3.5,-2.5,-2.5,-0.5,-0.5,-3.5,-0.5,-2.5,-3.5};
  ensemble2.PredictInplace(dataset.GetCovariates(), result, 0);
  for (int i = 0; i < n; i++) {
    ASSERT_NEAR(expected_pred[i], result[i], 0.01);
  }

  // Merge the second forest into the first
  ensemble1.MergeForest(ensemble2);
  result = std::vector<double>(n*output_dim);
  expected_pred = std::vector<double>{9,1,-8,7,9,9,1,9,-8,-4};
  ensemble1.PredictInplace(dataset.GetCovariates(), result, 0);
  for (int i = 0; i < n; i++) {
    ASSERT_NEAR(expected_pred[i], result[i], 0.01);
  }

  // Merge the second forest into the first
  ensemble1.MultiplyLeavesByValue(2.0);
  result = std::vector<double>(n*output_dim);
  expected_pred = std::vector<double>{18,2,-16,14,18,18,2,18,-16,-8};
  ensemble1.PredictInplace(dataset.GetCovariates(), result, 0);
  for (int i = 0; i < n; i++) {
    ASSERT_NEAR(expected_pred[i], result[i], 0.01);
  }
}

