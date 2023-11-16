/*!
 * Based largely on the functions in treelite's gtil library, released under the Apache license with the following copyright:
 * Copyright 2021-2023 by [treelite] Contributors
 */
#ifndef STOCHTREE_PREDICT_H_
#define STOCHTREE_PREDICT_H_

#include <stochtree/log.h>
#include <stochtree/meta.h>
#include <stochtree/train_data.h>
#include <stochtree/tree.h>

#include <cstdint>
#include <set>
#include <stack>
#include <string>

namespace StochTree {

inline int NextNode(double fvalue, double threshold, int left_child, int right_child) {
  return (fvalue <= threshold ? left_child : right_child);
}

inline int NextNodeCategorical(double fvalue, std::vector<std::uint32_t> const& category_list, int left_child, int right_child) {
  bool category_matched;
  // A valid (integer) category must satisfy two criteria:
  // 1) it must be exactly representable as double
  // 2) it must fit into uint32_t
  auto max_representable_int
      = std::min(static_cast<double>(std::numeric_limits<std::uint32_t>::max()),
          static_cast<double>(std::uint64_t(1) << std::numeric_limits<double>::digits));
  if (fvalue < 0 || std::fabs(fvalue) > max_representable_int) {
    category_matched = false;
  } else {
    auto const category_value = static_cast<std::uint32_t>(fvalue);
    category_matched = (std::find(category_list.begin(), category_list.end(), category_value)
                        != category_list.end());
  }
  return category_matched ? left_child : right_child;
}

static int EvaluateTree(Tree const& tree, TrainData* data, int row) {
  int node_id = 0;
  while (!tree.IsLeaf(node_id)) {
    auto const split_index = tree.SplitIndex(node_id);
    double const fvalue = data->get_feature_value(row, split_index);
    if (std::isnan(fvalue)) {
      node_id = tree.DefaultChild(node_id);
    } else {
      if (tree.NodeType(node_id) == StochTree::TreeNodeType::kCategoricalSplitNode) {
        node_id = NextNodeCategorical(fvalue, tree.CategoryList(node_id),
            tree.LeftChild(node_id), tree.RightChild(node_id));
      } else {
        node_id = NextNode(fvalue, tree.Threshold(node_id), tree.LeftChild(node_id), tree.RightChild(node_id));
      }
    }
  }
  return node_id;
}

} // namespace StochTree

#endif // STOCHTREE_PREDICT_H_
