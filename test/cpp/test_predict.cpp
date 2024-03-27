/*!
 * Test of the ensemble prediction method
 */
#include <gtest/gtest.h>
#include <testutils.h>
#include <stochtree/data.h>
#include <stochtree/ensemble.h>
#include <stochtree/tree.h>
#include <Eigen/Dense>
#include <iostream>
#include <memory>

/*! \brief Test forest prediction procedures for trees with constants in leaf nodes */
TEST(Ensemble, PredictConstant) {
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
  StochTree::TreeEnsemble ensemble(num_trees, output_dim, is_leaf_constant);

  // Simple split rules on both trees
  auto* tree = ensemble.GetTree(0);
  StochTree::TreeSplit tree_split = StochTree::TreeSplit(0.5);
  tree->ExpandNode(0, 0, tree_split, true, -5., 5.);
  tree = ensemble.GetTree(1);
  tree->ExpandNode(0, 1, tree_split, true, -2.5, 2.5);;

  // Run predict on the supplied covariates and check the result
  std::vector<double> result(n*output_dim);
  std::vector<double> expected_pred = {7.5,2.5,-7.5,7.5,7.5,7.5,2.5,7.5,-7.5,-2.5};
  ensemble.PredictInplace(dataset.GetCovariates(), result, 0);
  for (int i = 0; i < n; i++) {
    ASSERT_NEAR(expected_pred[i], result[i], 0.01);
  }
}

/*! \brief Test forest prediction procedures for trees with a univariate regression in leaf nodes */
TEST(Ensemble, PredictUnivariateRegression) {
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
  bool is_leaf_constant = false;
  StochTree::TreeEnsemble ensemble(num_trees, output_dim, is_leaf_constant);

  // Simple split rules on both trees
  auto* tree = ensemble.GetTree(0);
  StochTree::TreeSplit tree_split = StochTree::TreeSplit(0.5);
  tree->ExpandNode(0, 0, tree_split, true, -5., 5.);
  tree = ensemble.GetTree(1);
  tree->ExpandNode(0, 1, tree_split, true, -2.5, 2.5);;

  // Run predict on the supplied covariates and check the result
  std::vector<double> result(n*output_dim);
  std::vector<double> expected_pred = {7.3351256, 0.8511415, -1.5396290, 5.7172741, 4.7433491, 4.5919388, 1.0123031, 2.4834167, -6.5187785, -1.4611208};
  ensemble.PredictInplace(dataset.GetCovariates(), dataset.GetBasis(), result, 0);
  for (int i = 0; i < n; i++) {
    ASSERT_NEAR(expected_pred[i], result[i], 0.01);
  }
}

TEST(Ensemble, PredictMultivariateRegression) {
  // Load test data
  StochTree::TestUtils::TestDataset test_dataset;
  test_dataset = StochTree::TestUtils::LoadSmallDatasetMultivariateBasis();
  std::vector<StochTree::FeatureType> feature_types(test_dataset.x_cols, StochTree::FeatureType::kNumeric);

  // Construct datasets
  int n = test_dataset.n;
  StochTree::ForestDataset dataset = StochTree::ForestDataset();
  dataset.AddCovariates(test_dataset.covariates.data(), n, test_dataset.x_cols, test_dataset.row_major);
  dataset.AddBasis(test_dataset.omega.data(), test_dataset.n, test_dataset.omega_cols, test_dataset.row_major);
  StochTree::ColumnVector residual = StochTree::ColumnVector(test_dataset.outcome.data(), n);
  
  // Create a small ensemble
  int output_dim = 2;
  int num_trees = 2;
  bool is_leaf_constant = false;
  StochTree::TreeEnsemble ensemble(num_trees, output_dim, is_leaf_constant);

  // Simple split rules on both trees
  auto* tree = ensemble.GetTree(0);
  StochTree::TreeSplit tree_split = StochTree::TreeSplit(0.5);
  tree->ExpandNode(0, 0, 0.5, true, std::vector<double>{-5, -2.5}, std::vector<double>{5, 2.5});
  tree = ensemble.GetTree(1);
  tree->ExpandNode(0, 1, 0.5, true, std::vector<double>{-2.5, -1.25}, std::vector<double>{2.5, 1.25});

  // Run predict on the supplied covariates and basis function and check the result
  std::vector<double> result(n);
  std::vector<double> expected_pred = {8.725310, 1.015158, -3.645055, 6.570963, 8.129593, 7.385144, 1.331030, 4.469242, -8.613009, -1.756760};
  ensemble.PredictInplace(dataset.GetCovariates(), dataset.GetBasis(), result, 0);
  for (int i = 0; i < n; i++) {
    ASSERT_NEAR(expected_pred[i], result[i], 0.01);
  }
}
