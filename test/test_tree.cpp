/*!
 * Derived from xgboost tree unit test code:
 * https://github.com/dmlc/xgboost/blob/master/tests/cpp/tree/test_tree_model.cc
 */
#include <gtest/gtest.h>
#include <testutils.h>
#include <stochtree/log.h>
#include <stochtree/tree.h>
#include <iostream>
#include <memory>

TEST(Tree, UnivariateTreeConstruction) {
  StochTree::Tree tree;
  tree.Init(1);
  ASSERT_EQ(tree.LeafValue(0), 0.);
  tree.ExpandNode(0, 0, 0., 0., 0.);
  ASSERT_EQ(tree.NumNodes(), 3);
  ASSERT_EQ(tree.NodeType(0), StochTree::TreeNodeType::kNumericalSplitNode);
  tree.CollapseToLeaf(0, 0.);
  ASSERT_EQ(tree.NumValidNodes(), 1);
  tree.ExpandNode(0, 0, 0., 0., 0.);
  ASSERT_EQ(tree.NumValidNodes(), 3);
  ASSERT_EQ(tree.NodeType(0), StochTree::TreeNodeType::kNumericalSplitNode);
  ASSERT_EQ(tree.NumLeaves(), 2);
  ASSERT_FALSE(tree.IsLeaf(0));
  ASSERT_TRUE(tree.IsLeaf(1));
  ASSERT_TRUE(tree.IsLeaf(2));
}

TEST(Tree, UnivariateTreeCopyConstruction) {
  StochTree::Tree tree_1;
  StochTree::Tree tree_2;
  StochTree::TreeSplit split;
  tree_1.Init(1);
  
  // Perform two splits
  split = StochTree::TreeSplit(0.5);
  tree_1.ExpandNode(0, 0, split, 0., 0.);
  split = StochTree::TreeSplit(0.75);
  tree_1.ExpandNode(1, 1, split, 0., 0.);
  ASSERT_EQ(tree_1.NumValidNodes(), 5);
  ASSERT_EQ(tree_1.NumLeafParents(), 1);
  
  // Check leaves
  std::vector<int32_t> leaves = tree_1.GetLeaves();
  for (int i = 0; i < leaves.size(); i++) {
    ASSERT_TRUE(tree_1.IsLeaf(leaves[i]));
  }
  // Check leaf parents
  std::vector<int32_t> leaf_parents = tree_1.GetLeafParents();
  for (int i = 0; i < leaf_parents.size(); i++) {
    ASSERT_TRUE(tree_1.IsLeafParent(leaf_parents[i]));
  }
  
  // Perform another split
  split = StochTree::TreeSplit(0.6);
  tree_1.ExpandNode(3, 2, split, 0., 0.);
  ASSERT_EQ(tree_1.NumValidNodes(), 7);
  ASSERT_EQ(tree_1.NumLeaves(), 4);
  ASSERT_EQ(tree_1.NumLeafParents(), 1);
  
  // Check leaves
  leaves = tree_1.GetLeaves();
  for (int i = 0; i < leaves.size(); i++) {
    ASSERT_TRUE(tree_1.IsLeaf(leaves[i]));
  }
  // Check leaf parents
  leaf_parents = tree_1.GetLeafParents();
  for (int i = 0; i < leaf_parents.size(); i++) {
    ASSERT_TRUE(tree_1.IsLeafParent(leaf_parents[i]));
  }

  // Prune node 3 to a leaf
  tree_1.CollapseToLeaf(3, 0.);
  ASSERT_EQ(tree_1.NumValidNodes(), 5);
  ASSERT_EQ(tree_1.NumLeaves(), 3);
  ASSERT_EQ(tree_1.NumLeafParents(), 1);
  
  // Check leaves
  leaves = tree_1.GetLeaves();
  for (int i = 0; i < leaves.size(); i++) {
    ASSERT_TRUE(tree_1.IsLeaf(leaves[i]));
  }
  // Check leaf parents
  leaf_parents = tree_1.GetLeafParents();
  for (int i = 0; i < leaf_parents.size(); i++) {
    ASSERT_TRUE(tree_1.IsLeafParent(leaf_parents[i]));
  }
}

TEST(Tree, BadInitialization) {
  StochTree::Tree tree;
  EXPECT_THROW(tree.Init(0), std::runtime_error);
  EXPECT_THROW(tree.Init(-1), std::runtime_error);
}

TEST(Tree, UnivariateTreeCategoricalSplitConstruction) {
  StochTree::Tree tree;
  tree.Init(1);
  ASSERT_EQ(tree.LeafValue(0), 0.);
  tree.ExpandNode(0, 0, std::vector<std::uint32_t>{1,4,6}, 0., 0.);
  ASSERT_EQ(tree.NumNodes(), 3);
  ASSERT_EQ(tree.NodeType(0), StochTree::TreeNodeType::kCategoricalSplitNode);
  tree.CollapseToLeaf(0, 0.);
  ASSERT_EQ(tree.NumValidNodes(), 1);
  tree.ExpandNode(0, 0, std::vector<std::uint32_t>{2,3,5}, 0., 0.);
  ASSERT_EQ(tree.NodeType(0), StochTree::TreeNodeType::kCategoricalSplitNode);
  ASSERT_EQ(tree.NumValidNodes(), 3);
  ASSERT_EQ(tree.NumLeaves(), 2);
  ASSERT_FALSE(tree.IsLeaf(0));
  ASSERT_TRUE(tree.IsLeaf(1));
  ASSERT_TRUE(tree.IsLeaf(2));
}

TEST(Tree, MultivariateTreeConstruction) {
  StochTree::Tree tree;
  int tree_dim = 2;
  std::vector<double> leaf_values(tree_dim, 0.);
  tree.Init(tree_dim);
  EXPECT_THROW(tree.ExpandNode(0, 0, 0., 0., 0.), std::runtime_error);
  ASSERT_EQ(tree.LeafVector(0), leaf_values);
  tree.ExpandNode(0, 0, 0., leaf_values, leaf_values);
  ASSERT_EQ(tree.NumNodes(), 3);
  ASSERT_EQ(tree.NodeType(0), StochTree::TreeNodeType::kNumericalSplitNode);
  EXPECT_THROW(tree.CollapseToLeaf(0, 0.);, std::runtime_error);
  tree.CollapseToLeaf(0, leaf_values);
  ASSERT_EQ(tree.NumValidNodes(), 1);
  tree.ExpandNode(0, 0, 0., leaf_values, leaf_values);
  ASSERT_EQ(tree.NumValidNodes(), 3);
  ASSERT_EQ(tree.NodeType(0), StochTree::TreeNodeType::kNumericalSplitNode);
  ASSERT_EQ(tree.NumLeaves(), 2);
  ASSERT_FALSE(tree.IsLeaf(0));
  ASSERT_TRUE(tree.IsLeaf(1));
  ASSERT_TRUE(tree.IsLeaf(2));
}

TEST(Tree, MultivariateTreeCategoricalSplitConstruction) {
  StochTree::Tree tree;
  int tree_dim = 2;
  std::vector<double> leaf_values(tree_dim, 0.);
  tree.Init(tree_dim);
  ASSERT_EQ(tree.LeafVector(0), leaf_values);
  tree.ExpandNode(0, 0, std::vector<std::uint32_t>{1,4,6}, leaf_values, leaf_values);
  ASSERT_EQ(tree.NumNodes(), 3);
  ASSERT_EQ(tree.NodeType(0), StochTree::TreeNodeType::kCategoricalSplitNode);
  tree.CollapseToLeaf(0, leaf_values);
  ASSERT_EQ(tree.NumValidNodes(), 1);
  tree.ExpandNode(0, 0, std::vector<std::uint32_t>{2,3,5}, leaf_values, leaf_values);
  ASSERT_EQ(tree.NodeType(0), StochTree::TreeNodeType::kCategoricalSplitNode);
  ASSERT_EQ(tree.NumValidNodes(), 3);
  ASSERT_EQ(tree.NumLeaves(), 2);
  ASSERT_FALSE(tree.IsLeaf(0));
  ASSERT_TRUE(tree.IsLeaf(1));
  ASSERT_TRUE(tree.IsLeaf(2));
}
