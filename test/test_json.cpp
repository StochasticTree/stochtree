/*!
 * Derived from xgboost tree unit test code:
 * https://github.com/dmlc/xgboost/blob/master/tests/cpp/tree/test_tree_model.cc
 */
#include <gtest/gtest.h>
#include <testutils.h>
#include <stochtree/log.h>
#include <stochtree/tree.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <memory>

TEST(Json, TreeUnivariateLeaf) {
  // Initialize tree
  StochTree::Tree tree;
  StochTree::TreeSplit split;
  tree.Init(1);
  
  // Perform three splits
  split = StochTree::TreeSplit(0.5);
  tree.ExpandNode(0, 0, split, 0., 0.);
  split = StochTree::TreeSplit(0.75);
  tree.ExpandNode(1, 1, split, 0., 0.);
  split = StochTree::TreeSplit(0.6);
  tree.ExpandNode(3, 2, split, 0., 0.);
  
  // Prune node 3 to a leaf
  tree.CollapseToLeaf(3, 0.);
  
  // Write to json
  nlohmann::json tree_json = tree.to_json();

  // Convert back to a tree
  StochTree::Tree tree_parsed;
  tree_parsed.from_json(tree_json);

  // Check that trees are the same
  ASSERT_EQ(tree, tree_parsed);
}

TEST(Json, TreeUnivariateLeafCategoricalSplit) {
  // Initialize tree
  StochTree::Tree tree;
  StochTree::TreeSplit split;
  tree.Init(1);
  
  // Perform three splits
  std::vector<uint32_t> split_categories_1{1,3,5,7};
  split = StochTree::TreeSplit(split_categories_1);
  tree.ExpandNode(0, 0, split, 0., 0.);
  std::vector<uint32_t> split_categories_2{2,3,5};
  split = StochTree::TreeSplit(split_categories_2);
  tree.ExpandNode(1, 1, split, 0., 0.);
  split = StochTree::TreeSplit(0.6);
  tree.ExpandNode(3, 2, split, 0., 0.);
  
  // Prune node 3 to a leaf
  tree.CollapseToLeaf(3, 0.);
  
  // Write to json
  nlohmann::json tree_json = tree.to_json();

  // Convert back to a tree
  StochTree::Tree tree_parsed;
  tree_parsed.from_json(tree_json);

  // Check that trees are the same
  ASSERT_EQ(tree, tree_parsed);
}

TEST(Json, TreeMultivariateLeaf) {
  // Initialize tree
  StochTree::Tree tree;
  StochTree::TreeSplit split;
  int tree_dim = 2;
  std::vector<double> leaf_values1(tree_dim, 0.);
  std::vector<double> leaf_values2(tree_dim, 1.5);
  std::vector<double> leaf_values3(tree_dim, -0.75);
  std::vector<double> leaf_values4(tree_dim, 0.33);
  std::vector<double> leaf_values5(tree_dim, 345235636.4);
  std::vector<double> leaf_values6(tree_dim, 10023.1);
  tree.Init(tree_dim);
  
  // Perform three splits
  split = StochTree::TreeSplit(0.5);
  tree.ExpandNode(0, 0, split, leaf_values1, leaf_values2);
  split = StochTree::TreeSplit(0.75);
  tree.ExpandNode(1, 1, split, leaf_values3, leaf_values4);
  split = StochTree::TreeSplit(0.6);
  tree.ExpandNode(1, 1, split, leaf_values5, leaf_values6);
  
  // Prune node 3 to a leaf
  tree.CollapseToLeaf(3, leaf_values3);
  
  // Write to json
  nlohmann::json tree_json = tree.to_json();

  // Convert back to a tree
  StochTree::Tree tree_parsed;
  tree_parsed.Init(tree_dim);
  tree_parsed.from_json(tree_json);

  // Check that trees are the same
  ASSERT_EQ(tree, tree_parsed);
}

TEST(Json, TreeMultivariateLeafCategoricalSplit) {
  // Initialize tree
  StochTree::Tree tree;
  StochTree::TreeSplit split;
  int tree_dim = 2;
  std::vector<double> leaf_values1(tree_dim, 0.);
  std::vector<double> leaf_values2(tree_dim, 1.5);
  std::vector<double> leaf_values3(tree_dim, -0.75);
  std::vector<double> leaf_values4(tree_dim, 0.33);
  std::vector<double> leaf_values5(tree_dim, 345235636.4);
  std::vector<double> leaf_values6(tree_dim, 10023.1);
  tree.Init(tree_dim);
  
  // Perform three splits
  std::vector<uint32_t> split_categories_1{1,3,5,7};
  split = StochTree::TreeSplit(split_categories_1);
  tree.ExpandNode(0, 0, split, leaf_values1, leaf_values2);
  std::vector<uint32_t> split_categories_2{2,3,5};
  split = StochTree::TreeSplit(split_categories_2);
  tree.ExpandNode(1, 1, split, leaf_values3, leaf_values4);
  split = StochTree::TreeSplit(0.6);
  tree.ExpandNode(1, 1, split, leaf_values5, leaf_values6);
  
  // Prune node 3 to a leaf
  tree.CollapseToLeaf(3, leaf_values3);
  
  // Write to json
  nlohmann::json tree_json = tree.to_json();

  // Convert back to a tree
  StochTree::Tree tree_parsed;
  tree_parsed.Init(tree_dim);
  tree_parsed.from_json(tree_json);

  // Check that trees are the same
  ASSERT_EQ(tree, tree_parsed);
}
