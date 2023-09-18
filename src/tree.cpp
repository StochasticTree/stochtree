/*!
 * Inspired by the design of the tree in the xgboost and treelite package, both released under the Apache license 
 * with the following copyright:
 * Copyright 2015-2023 by XGBoost Contributors
 * Copyright 2017-2021 by [treelite] Contributors
 */
#include <stochtree/common.h>
#include <stochtree/tree.h>

#include <algorithm>
#include <sstream>

namespace StochTree {

constexpr node_t Tree::kInvalidNodeId;
constexpr node_t Tree::kDeletedNodeMarker;
constexpr node_t Tree::kRoot;

bool Tree::Equal(const Tree& b) const {
  if (NumExtraNodes() != b.NumExtraNodes()) {
    return false;
  }
  auto const& self = *this;
  bool ret { true };
  this->WalkTree([&self, &b, &ret](node_t nidx) {
    if (!(self.nodes_.at(nidx) == b.nodes_.at(nidx))) {
      ret = false;
      return false;
    }
    return true;
  });
  return ret;
}

node_t Tree::GetNumLeaves() const {
  node_t leaves { 0 };
  auto const& self = *this;
  this->WalkTree([&leaves, &self](node_t nidx) {
                   if (self[nidx].IsLeaf()) {
                     leaves++;
                   }
                   return true;
                 });
  return leaves;
}

node_t Tree::GetNumLeafParents() const {
  node_t leaf_parents { 0 };
  auto const& self = *this;
  this->WalkTree([&leaf_parents, &self](node_t nidx) {
                   if (!self[nidx].IsLeaf()){
                     if ((self[self[nidx].LeftChild()].IsLeaf()) && (self[self[nidx].LeftChild()].IsLeaf())){
                      leaf_parents++;
                     }
                   }
                   return true;
                 });
  return leaf_parents;
}

node_t Tree::GetNumSplitNodes() const {
  node_t splits { 0 };
  auto const& self = *this;
  this->WalkTree([&splits, &self](node_t nidx) {
                   if (!self[nidx].IsLeaf()) {
                     splits++;
                   }
                   return true;
                 });
  return splits;
}

void Tree::ExpandNode(node_t nid, unsigned split_index, double split_value,
                      bool default_left, double base_weight,
                      double left_leaf_value, double right_leaf_value, 
                      node_t leaf_right_child) {
  int pleft = this->AllocNode();
  int pright = this->AllocNode();
  auto &node = nodes_[nid];
  CHECK(node.IsLeaf());
  node.SetLeftChild(pleft);
  node.SetRightChild(pright);
  nodes_[node.LeftChild()].SetParent(nid, true);
  nodes_[node.RightChild()].SetParent(nid, false);
  node.SetSplit(split_index, split_value, default_left);

  nodes_[pleft].SetLeaf(left_leaf_value, leaf_right_child);
  nodes_[pright].SetLeaf(right_leaf_value, leaf_right_child);

  this->split_types_.at(nid) = FeatureSplitType::kNumericSplit;

  // Remove nid from leaves and add to internal nodes and leaf parents
  leaves_.erase(std::remove(leaves_.begin(), leaves_.end(), nid), leaves_.end());
  leaf_parents_.push_back(nid);
  internal_nodes_.push_back(nid);

  // Remove nid's parent node (if applicable) from leaf parents
  if (!node.IsRoot()){
    node_t parent_idx = node.Parent();
    leaf_parents_.erase(std::remove(leaf_parents_.begin(), leaf_parents_.end(), parent_idx), leaf_parents_.end());
  }

  // Add pleft and pright to leaves
  leaves_.push_back(pleft);
  leaves_.push_back(pright);
}

void Tree::InplacePredictFromNodes(std::vector<double> result, std::vector<data_size_t> node_indices) {
  if (result.size() != node_indices.size()) {
    Log::Fatal("Indices and result vector are different sizes");
  }
  data_size_t n = node_indices.size();
  for (data_size_t i = 0; i < n; i++) {
    result[i] = (*this)[node_indices[i]].LeafValue();
  }
}

std::vector<double> Tree::PredictFromNodes(std::vector<data_size_t> node_indices) {
  data_size_t n = node_indices.size();
  std::vector<double> result(n);
  for (data_size_t i = 0; i < n; i++) {
    if (!(*this)[node_indices[i]].IsLeaf()) {
      Log::Fatal("Leaf node %d indexed by observation %d is not a leaf node", node_indices[i], i);
    }
    result[i] = (*this)[node_indices[i]].LeafValue();
  }
  return result;
}

std::string Tree::ToJSON() const {
  std::stringstream str_buf;
  Common::C_stringstream(str_buf);
  str_buf << std::setprecision(std::numeric_limits<double>::digits10 + 2);
  str_buf << "\"num_nodes\":" << this->NumNodes() << "," << '\n';
  str_buf << "\"num_leaves\":" << this->GetNumLeaves() << "," << '\n';
  str_buf << "\"num_features\":" << this->NumFeatures() << "," << '\n';
  str_buf << "\"tree_structure\":" << NodeToJSON(0) << '\n';
  return str_buf.str();
}

std::string Tree::NodeToJSON(int index) const {
  bool is_leaf = this->IsLeaf(index);
  std::stringstream str_buf;
  Common::C_stringstream(str_buf);
  str_buf << std::setprecision(std::numeric_limits<double>::digits10 + 2);
  // non-leaf
  str_buf << "{" << '\n';
  str_buf << "\"node_id\":" << index << "," << '\n';
  str_buf << "\"is_leaf\":" << is_leaf << "," << '\n';
  if (is_leaf){
    str_buf << "\"leaf_value\":" << Common::AvoidInf(nodes_[index].LeafValue()) << "," << '\n';
  } else {
    // TODO(drew) handle categorical splits here
    str_buf << "\"split_feature\":" << nodes_[index].SplitIndex() << "," << '\n';
    str_buf << "\"split_value\":" << Common::AvoidInf(nodes_[index].SplitCond()) << "," << '\n';
    str_buf << "\"default_left\":" << nodes_[index].DefaultLeft() << "," << '\n';
    str_buf << "\"left_child\":" << NodeToJSON(this->LeftChild(index)) << "," << '\n';
    str_buf << "\"right_child\":" << NodeToJSON(this->RightChild(index)) << '\n';
  }
  str_buf << "}";
  return str_buf.str();
}

} // namespace StochTree
