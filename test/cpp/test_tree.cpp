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

TEST(Tree, AllocateNode) {
  StochTree::Tree tree;
  tree.ExpandNode(
    0, 0, 0., true, 0., 0., 0., StochTree::Tree::kInvalidNodeId
  );
  tree.CollapseToLeaf(0, 0);
  ASSERT_EQ(tree.NumExtraNodes(), 0);

  tree.ExpandNode(
    0, 0, 0., true, 0., 0., 0., StochTree::Tree::kInvalidNodeId
  );
  ASSERT_EQ(tree.NumExtraNodes(), 2);

  auto& nodes = tree.GetNodes();
  ASSERT_FALSE(nodes.at(1).IsDeleted());
  ASSERT_TRUE(nodes.at(1).IsLeaf());
  ASSERT_TRUE(nodes.at(2).IsLeaf());
}
