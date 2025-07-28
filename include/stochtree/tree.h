/*!
 * Based largely on the tree classes in xgboost and treelite, both released under the Apache license with the following copyright:
 * Copyright 2015-2023 by XGBoost Contributors
 * Copyright 2017-2021 by [treelite] Contributors
 */
#ifndef STOCHTREE_TREE_H_
#define STOCHTREE_TREE_H_

#include <nlohmann/json.hpp>
#include <stochtree/data.h>
#include <stochtree/log.h>
#include <stochtree/meta.h>
#include <Eigen/Dense>

#include <cstdint>
#include <stack>
#include <string>

using json = nlohmann::json;

namespace StochTree {

/*! \brief Tree node type */
enum TreeNodeType {
  kLeafNode = 0,
  kNumericalSplitNode = 1,
  kCategoricalSplitNode = 2
};

// template<typename T>
// int enum_to_int(T& input_enum) {
//   return static_cast<int>(input_enum);
// }

// template<typename T>
// T json_to_enum(json& input_json) {
//   return static_cast<T>(input_json);
// }

/*! \brief Get string representation of TreeNodeType */
std::string TreeNodeTypeToString(TreeNodeType type);

/*! \brief Get NodeType from string */
TreeNodeType TreeNodeTypeFromString(std::string const& name);

enum FeatureSplitType {
  kNumericSplit,
  kOrderedCategoricalSplit,
  kUnorderedCategoricalSplit
};

/*! \brief Forward declaration of TreeSplit class */
class TreeSplit;

/*! 
 * \defgroup tree_group Tree API
 * 
 * \brief Classes / functions for creating and modifying decision trees.
 * 
 * \section tree_design Design
 * 
 * \{
 */

/*! \brief Decision tree data structure */
class Tree {
 public:
  static constexpr std::int32_t kInvalidNodeId{-1};
  static constexpr std::int32_t kDeletedNodeMarker = std::numeric_limits<node_t>::max();
  static constexpr std::int32_t kRoot{0};
  
  Tree() = default;
  // ~Tree() = default;
  Tree(Tree const&) = delete;
  Tree& operator=(Tree const&) = delete;
  Tree(Tree&&) noexcept = default;
  Tree& operator=(Tree&&) noexcept = default;
  /*!
   * \brief Copy the structure and parameters of another tree. If the `Tree` object calling this method already 
   * has a non-root tree structure / parameters, this will be erased and replaced with a copy of `tree`.
   * 
   * \param tree Tree to be cloned
   */
  void CloneFromTree(Tree* tree);

  std::int32_t num_nodes{0};
  std::int32_t num_deleted_nodes{0};

  /*! \brief Reset tree to empty vectors and default values of boolean / integer variables */
  void Reset();
  /*! \brief Initialize the tree with a single root node */
  void Init(int output_dimension = 1, bool is_log_scale = false);
  /*! \brief Allocate a new node and return the node's ID */
  int AllocNode();
  /*! \brief Deletes node indexed by node ID */
  void DeleteNode(std::int32_t nid);
  /*! \brief Expand a node based on a numeric split rule */
  void ExpandNode(std::int32_t nid, int split_index, double split_value, double left_value, double right_value);
  /*! \brief Expand a node based on a categorical split rule */
  void ExpandNode(std::int32_t nid, int split_index, std::vector<std::uint32_t> const& categorical_indices, double left_value, double right_value);
  /*! \brief Expand a node based on a numeric split rule */
  void ExpandNode(std::int32_t nid, int split_index, double split_value, std::vector<double> left_value_vector, std::vector<double> right_value_vector);
  /*! \brief Expand a node based on a categorical split rule */
  void ExpandNode(std::int32_t nid, int split_index, std::vector<std::uint32_t> const& categorical_indices, std::vector<double> left_value_vector, std::vector<double> right_value_vector);
    /*! \brief Expand a node based on a generic split rule */
  void ExpandNode(std::int32_t nid, int split_index, TreeSplit& split, double left_value, double right_value);
  /*! \brief Expand a node based on a generic split rule */
  void ExpandNode(std::int32_t nid, int split_index, TreeSplit& split, std::vector<double> left_value_vector, std::vector<double> right_value_vector);

  /*! \brief Whether or not a tree is a "stump" consisting of a single root node */
  inline bool IsRoot() {return leaves_.size() == 1;}
  
  /*! \brief Convert tree to JSON and return JSON in-memory */
  json to_json();
  /*! 
   * \brief Load from JSON 
   * 
   * \param tree_json In-memory json object (of type `nlohmann::json`)
   */
  void from_json(const json& tree_json);

  void ChangeToLeaf(std::int32_t nid, double value) {
    CHECK(this->IsLeaf(this->LeftChild(nid)));
    CHECK(this->IsLeaf(this->RightChild(nid)));
    this->DeleteNode(this->LeftChild(nid));
    this->DeleteNode(this->RightChild(nid));
    this->SetLeaf(nid, value);

    // Add nid to leaves and remove from internal nodes and leaf parents (if it was there)
    leaves_.push_back(nid);
    leaf_parents_.erase(std::remove(leaf_parents_.begin(), leaf_parents_.end(), nid), leaf_parents_.end());
    internal_nodes_.erase(std::remove(internal_nodes_.begin(), internal_nodes_.end(), nid), internal_nodes_.end());

    // Check if the other child of nid's parent node is also a leaf, if so, add parent back to leaf parents
    // TODO refactor and add this to the multivariate case as well
    if (!IsRoot(nid)) {
      int parent_id = Parent(nid);
      if ((IsLeaf(LeftChild(parent_id))) && (IsLeaf(RightChild(parent_id)))){
        leaf_parents_.push_back(parent_id);
      }
    }
  }

  /*!
   * \brief Collapse an internal node to a leaf node, deleting its children from the tree
   * \param nid Node id of the new leaf node
   * \param value_vector New leaf value
   */
  void CollapseToLeaf(std::int32_t nid, double value) {
    CHECK_EQ(output_dimension_, 1);
    if (this->IsLeaf(nid)) return;
    if (!this->IsLeaf(this->LeftChild(nid))) {
      CollapseToLeaf(this->LeftChild(nid), value);
    }
    if (!this->IsLeaf(this->RightChild(nid))) {
      CollapseToLeaf(this->RightChild(nid), value);
    }
    this->ChangeToLeaf(nid, value);
  }

  void ChangeToLeaf(std::int32_t nid, std::vector<double> value_vector) {
    CHECK(this->IsLeaf(this->LeftChild(nid)));
    CHECK(this->IsLeaf(this->RightChild(nid)));
    this->DeleteNode(this->LeftChild(nid));
    this->DeleteNode(this->RightChild(nid));
    this->SetLeafVector(nid, value_vector);

    // Add nid to leaves and remove from internal nodes and leaf parents (if it was there)
    leaves_.push_back(nid);
    leaf_parents_.erase(std::remove(leaf_parents_.begin(), leaf_parents_.end(), nid), leaf_parents_.end());
    internal_nodes_.erase(std::remove(internal_nodes_.begin(), internal_nodes_.end(), nid), internal_nodes_.end());

    // Check if the other child of nid's parent node is also a leaf, if so, add parent back to leaf parents
    // TODO refactor and add this to the multivariate case as well
    if (!IsRoot(nid)) {
      int parent_id = Parent(nid);
      if ((IsLeaf(LeftChild(parent_id))) && (IsLeaf(RightChild(parent_id)))){
        leaf_parents_.push_back(parent_id);
      }
    }
  }
  
  /*!
   * \brief Collapse an internal node to a leaf node, deleting its children from the tree
   * \param nid Node id of the new leaf node
   * \param value_vector New leaf vector value
   */
  void CollapseToLeaf(std::int32_t nid, std::vector<double> value_vector) {
    CHECK_GT(output_dimension_, 1);
    CHECK_EQ(output_dimension_, value_vector.size());
    if (this->IsLeaf(nid)) return;
    if (!this->IsLeaf(this->LeftChild(nid))) {
      CollapseToLeaf(this->LeftChild(nid), value_vector);
    }
    if (!this->IsLeaf(this->RightChild(nid))) {
      CollapseToLeaf(this->RightChild(nid), value_vector);
    }
    this->ChangeToLeaf(nid, value_vector);
  }

  /*!
   * \brief Add a constant value to every leaf of a tree. If leaves are multi-dimensional, `constant_value` will be added to every dimension of the leaves.
   * 
   * \param constant_value Value that will be added to every leaf of a tree
   */
  void AddValueToLeaves(double constant_value) {
    if (output_dimension_ == 1) {
      for (int j = 0; j < leaf_value_.size(); j++) {
        leaf_value_[j] += constant_value;
      }
    } else {
      for (int j = 0; j < leaf_vector_.size(); j++) {
        leaf_vector_[j] += constant_value;
      }
    }
  }

  /*!
   * \brief Multiply every leaf of a tree by a constant value. If leaves are multi-dimensional, `constant_value` will be multiplied through every dimension of the leaves.
   * 
   * \param constant_multiple Value that will be multiplied by every leaf of a tree
   */
  void MultiplyLeavesByValue(double constant_multiple) {
    if (output_dimension_ == 1) {
      for (int j = 0; j < leaf_value_.size(); j++) {
        leaf_value_[j] *= constant_multiple;
      }
    } else {
      for (int j = 0; j < leaf_vector_.size(); j++) {
        leaf_vector_[j] *= constant_multiple;
      }
    }
  }

  /*!
   * \brief Iterate through all nodes in this tree.
   * 
   * \tparam Func Function object type, must map `std::int32_t` to `bool`.
   * \param func Function that accepts a node index and returns `False` when iteration through a given branch of the tree should stop and `True` otherwise.
   */
  template <typename Func> void WalkTree(Func func) const {
    std::stack<std::int32_t> nodes;
    nodes.push(kRoot);
    auto &self = *this;
    while (!nodes.empty()) {
      auto nidx = nodes.top();
      nodes.pop();
      if (!func(nidx)) {
        return;
      }
      auto left = self.LeftChild(nidx);
      auto right = self.RightChild(nidx);
      if (left != Tree::kInvalidNodeId) {
        nodes.push(left);
      }
      if (right != Tree::kInvalidNodeId) {
        nodes.push(right);
      }
    }
  }

  std::vector<double> PredictFromNodes(std::vector<std::int32_t> node_indices);
  std::vector<double> PredictFromNodes(std::vector<std::int32_t> node_indices, Eigen::MatrixXd& basis);
  double PredictFromNode(std::int32_t node_id);
  double PredictFromNode(std::int32_t node_id, Eigen::MatrixXd& basis, int row_idx);

  /** Getters **/
  /*!
   * \brief Whether or not a tree has vector output
   */
  bool HasVectorOutput() const {
    return output_dimension_ > 1;
  }

  /*!
   * \brief Dimension of tree output
   */
  std::int32_t OutputDimension() const {
    return output_dimension_;
  }

  /*!
   * \brief Whether or not tree parameters should be exponentiated at prediction time
   */
  bool IsLogScale() const {
    return is_log_scale_;
  }
  
  /*!
   * \brief Index of the node's parent
   * \param nid ID of node being queried
   */
  std::int32_t Parent(std::int32_t nid) const {
    return parent_[nid];
  }
  
  /*!
   * \brief Index of the node's left child
   * \param nid ID of node being queried
   */
  std::int32_t LeftChild(std::int32_t nid) const {
    return cleft_[nid];
  }
  
  /*!
   * \brief Index of the node's right child
   * \param nid ID of node being queried
   */
  std::int32_t RightChild(std::int32_t nid) const {
    return cright_[nid];
  }
  
  /*!
   * \brief Index of the node's "default" child (potentially used in the case of a missing feature at prediction time)
   * \param nid ID of node being queried
   */
  std::int32_t DefaultChild(std::int32_t nid) const {
    return cleft_[nid];
  }
  
  /*!
   * \brief Feature index defining the node's split rule
   * \param nid ID of node being queried
   */
  std::int32_t SplitIndex(std::int32_t nid) const {
    return split_index_[nid];
  }
  
  /*!
   * \brief Whether the node is a leaf node
   * \param nid ID of node being queried
   */
  bool IsLeaf(std::int32_t nid) const {
    return cleft_[nid] == kInvalidNodeId;
  }
  
  /*!
   * \brief Whether the node is root
   * \param nid ID of node being queried
   */
  bool IsRoot(std::int32_t nid) const {
    return parent_[nid] == kInvalidNodeId;
  }

  /*!
   * \brief Whether the node has been deleted
   * \param nid ID of node being queried
   */
  bool IsDeleted(std::int32_t nid) const {
    return node_deleted_[nid];
  }

  /*!
   * \brief Get parameter value of a node (typically though not necessarily a leaf node)
   * \param nid ID of node being queried
   */
  double LeafValue(std::int32_t nid) const {
    return leaf_value_[nid];
  }
  
  /*!
   * \brief Get parameter value of a node (typically though not necessarily a leaf node) at a given output dimension
   * \param nid ID of node being queried
   * \param dim_id Output dimension being queried
   */
  double LeafValue(std::int32_t nid, std::int32_t dim_id) const {
    CHECK_LT(dim_id, output_dimension_);
    if (output_dimension_ == 1 && dim_id == 0) {
      return leaf_value_[nid];
    } else {
      std::size_t const offset_begin = leaf_vector_begin_[nid];
      std::size_t const offset_end = leaf_vector_end_[nid];
      if (offset_begin >= leaf_vector_.size() || offset_end > leaf_vector_.size()) {
        Log::Fatal("No leaf vector set for node nid");
      }
      return leaf_vector_[offset_begin + dim_id];
    }
  }

  /*!
   * \brief Get maximum depth of all of the leaf nodes
   */
  std::int32_t MaxLeafDepth() const {
    std::int32_t max_depth = 0;
    std::stack<std::int32_t> nodes;
    std::stack<std::int32_t> node_depths;
    nodes.push(kRoot);
    node_depths.push(0);
    auto &self = *this;
    while (!nodes.empty()) {
      auto nidx = nodes.top();
      nodes.pop();
      auto node_depth = node_depths.top();
      node_depths.pop();
      bool valid_node = !self.IsDeleted(nidx);
      if (valid_node) {
        if (node_depth > max_depth) max_depth = node_depth;
        auto left = self.LeftChild(nidx);
        auto right = self.RightChild(nidx);
        if (left != Tree::kInvalidNodeId) {
          nodes.push(left);
          node_depths.push(node_depth+1);
        }
        if (right != Tree::kInvalidNodeId) {
          nodes.push(right);
          node_depths.push(node_depth+1);
        }
      }
    }
    return max_depth;
  }

  /*!
   * \brief Get vector-valued parameters of a node (typically leaf)
   * \param nid ID of node being queried
   */
  std::vector<double> LeafVector(std::int32_t nid) const {
    std::size_t const offset_begin = leaf_vector_begin_[nid];
    std::size_t const offset_end = leaf_vector_end_[nid];
    if (offset_begin >= leaf_vector_.size() || offset_end > leaf_vector_.size()) {
      // Return empty vector, to indicate the lack of leaf vector
      return std::vector<double>();
    }
    return std::vector<double>(&leaf_vector_[offset_begin], &leaf_vector_[offset_end]);
    // Use unsafe access here, since we may need to take the address of one past the last
    // element, to follow with the range semantic of std::vector<>.
  }

  /*!
   * \brief Sum of squared parameter values for a given node (typically though not necessarily a leaf node)
   * \param nid ID of node being queried
   */
  double SumSquaredNodeValues(std::int32_t nid) const {
    if (output_dimension_ == 1) {
      return std::pow(leaf_value_[nid], 2.0);
    } else {
      double result = 0.;
      std::size_t const offset_begin = leaf_vector_begin_[nid];
      std::size_t const offset_end = leaf_vector_end_[nid];
      if (offset_begin >= leaf_vector_.size() || offset_end > leaf_vector_.size()) {
        Log::Fatal("No leaf vector set for node nid");
      }
      for (std::size_t i = offset_begin; i < offset_end; i++) {
        result += std::pow(leaf_vector_[i], 2.0);
      }
      return result;
    }
  }

  /*!
   * \brief Sum of squared values for all leaves in a tree
   */
  double SumSquaredLeafValues() const {
    double result = 0.;
    for (auto& leaf : leaves_) {
      result += SumSquaredNodeValues(leaf);
    }
    return result;
  }
  
  /*!
   * \brief Tests whether the leaf node has a non-empty leaf vector
   * \param nid ID of node being queried
   */
  bool HasLeafVector(std::int32_t nid) const {
    return leaf_vector_begin_[nid] != leaf_vector_end_[nid];
  }

  /*!
   * \brief Get split threshold of the node
   * \param nid ID of node being queried
   */
  double Threshold(std::int32_t nid) const {
    return threshold_[nid];
  }

  /*!
   * \brief Get list of all categories belonging to the left child node.
   * Categories are integers ranging from 0 to (n-1), where n is the number of categories in that particular feature. 
   * This list is assumed to be in ascending order.
   *
   * \param nid ID of node being queried
   */
  std::vector<std::uint32_t> CategoryList(std::int32_t nid) const {
    std::size_t const offset_begin = category_list_begin_[nid];
    std::size_t const offset_end = category_list_end_[nid];
    if (offset_begin >= category_list_.size() || offset_end > category_list_.size()) {
      // Return empty vector, to indicate the lack of any category list
      // The node might be a numerical split
      return {};
    }
    return std::vector<std::uint32_t>(&category_list_[offset_begin], &category_list_[offset_end]);
    // Use unsafe access here, since we may need to take the address of one past the last
    // element, to follow with the range semantic of std::vector<>.
  }

  /*!
   * \brief Get the type of a node (i.e. numeric split, categorical split, leaf)
   * \param nid ID of node being queried
   */
  TreeNodeType NodeType(std::int32_t nid) const {
    return node_type_[nid];
  }

  /*!
   * \brief Whether the node is a numeric split node
   * \param nid ID of node being queried
   */
  bool IsNumericSplitNode(std::int32_t nid) const {
    return node_type_[nid] == TreeNodeType::kNumericalSplitNode;
  }

  /*!
   * \brief Whether the node is a numeric split node
   * \param nid ID of node being queried
   */
  bool IsCategoricalSplitNode(std::int32_t nid) const {
    return node_type_[nid] == TreeNodeType::kCategoricalSplitNode;
  }

  /*!
   * \brief Query whether this tree contains any categorical splits
   */
  bool HasCategoricalSplit() const {
    return has_categorical_split_;
  }

  /* \brief Count number of leaves in tree. */
  [[nodiscard]] std::int32_t NumLeaves() const;
  [[nodiscard]] std::int32_t NumLeafParents() const;
  [[nodiscard]] std::int32_t NumSplitNodes() const;

  /* \brief Determine whether nid is leaf parent */
  [[nodiscard]] bool IsLeafParent(std::int32_t nid) const {
    // False until we deduce left and right node are
    // available and both are leaves
    bool is_left_leaf = false;
    bool is_right_leaf = false;
    // Check if node nidx is a leaf, if so, return false
    bool is_leaf = this->IsLeaf(nid);
    if (is_leaf){
      return false;
    } else {
      // If nidx is not a leaf, it must have left and right nodes
      // so we check if those are leaves
      std::int32_t left_node = LeftChild(nid);
      std::int32_t right_node = RightChild(nid);
      is_left_leaf = IsLeaf(left_node);
      is_right_leaf = IsLeaf(right_node);
    }
    return is_left_leaf && is_right_leaf;
  }

  /*!
   * \brief Get indices of all internal nodes.
   */
  [[nodiscard]] std::vector<std::int32_t> const& GetInternalNodes() const {
    return internal_nodes_;
  }

  /*!
   * \brief Get indices of all leaf nodes.
   */
  [[nodiscard]] std::vector<std::int32_t> const& GetLeaves() const {
    return leaves_;
  }

  /*!
   * \brief Get indices of all leaf parent nodes.
   */
  [[nodiscard]] std::vector<std::int32_t> const& GetLeafParents() const {
    return leaf_parents_;
  }
  
  /*!
   * \brief Get indices of all valid (non-deleted) nodes.
   */
  [[nodiscard]] std::vector<std::int32_t> GetNodes() {
    std::vector<std::int32_t> output;
    auto const& self = *this;
    this->WalkTree([&output, &self](std::int32_t nidx) {
                    if (!self.IsDeleted(nidx)) {
                      output.push_back(nidx);
                    }
                    return true;
                  });
    return output;
  }

  /*!
   * \brief Get the depth of a node
   * \param nid node id
   */
  [[nodiscard]] std::int32_t GetDepth(std::int32_t nid) const {
    int depth = 0;
    while (!IsRoot(nid)) {
      ++depth;
      nid = Parent(nid);
    }
    return depth;
  }

  /**
   * \brief Get the total number of nodes including deleted ones in this tree.
   */
  [[nodiscard]] std::int32_t NumNodes() const noexcept { return num_nodes; }
  
  /**
   * \brief Get the total number of deleted nodes in this tree.
   */
  [[nodiscard]] std::int32_t NumDeletedNodes() const noexcept { return num_deleted_nodes; }
  
  /**
   * \brief Get the total number of valid nodes in this tree.
   */
  [[nodiscard]] std::int32_t NumValidNodes() const noexcept {
    return num_nodes - num_deleted_nodes;
  }

  /** Setters **/
  /*!
   * \brief Identify left child node
   * \param nid ID of node being modified
   * \param left_child ID of the left child node
   */
  void SetLeftChild(std::int32_t nid, std::int32_t left_child) {
    cleft_[nid] = left_child;
  }

  /*!
   * \brief Identify right child node
   * \param nid ID of node being modified
   * \param right_child ID of the right child node
   */
  void SetRightChild(std::int32_t nid, std::int32_t right_child) {
    cright_[nid] = right_child;
  }

  /*!
   * \brief Identify two child nodes of the node and the corresponding parent node of the child nodes
   * \param nid ID of node being modified
   * \param left_child ID of the left child node
   * \param right_child ID of the right child node
   */
  void SetChildren(std::int32_t nid, std::int32_t left_child, std::int32_t right_child) {
    SetLeftChild(nid, left_child);
    SetRightChild(nid, right_child);
  }

  /*!
   * \brief Identify parent node
   * \param child_node ID of child node
   * \param parent_node ID of the parent node
   */
  void SetParent(std::int32_t child_node, std::int32_t parent_node) {
    parent_[child_node] = parent_node;
  }

  /*!
   * \brief Identify parent node of the left and right node ids
   * \param nid ID of parent node
   * \param left_child ID of the left child node
   * \param right_child ID of the right child node
   */
  void SetParents(std::int32_t nid, std::int32_t left_child, std::int32_t right_child) {
    SetParent(left_child, nid);
    SetParent(right_child, nid);
  }

  /*!
   * \brief Create a numerical split
   * \param nid ID of node being updated
   * \param split_index Feature index to split
   * \param threshold Threshold value
   */
  void SetNumericSplit(
      std::int32_t nid, std::int32_t split_index, double threshold);
  
  /*!
   * \brief Create a categorical split
   * \param nid ID of node being updated
   * \param split_index Feature index to split
   * \param category_list List of categories to belong to either the right child node or the left
   *                      child node. Set categories_list_right_child parameter to indicate
   *                      which node the category list should represent.
   */
  void SetCategoricalSplit(std::int32_t nid, std::int32_t split_index,
      std::vector<std::uint32_t> const& category_list);
  
  /*!
   * \brief Set the leaf value of the node
   * \param nid ID of node being updated
   * \param value Leaf value
   */
  void SetLeaf(std::int32_t nid, double value);

  /*!
   * \brief Set the leaf vector of the node; useful for multi-output trees
   * \param nid ID of node being updated
   * \param leaf_vector Leaf vector
   */
  void SetLeafVector(std::int32_t nid, std::vector<double> const& leaf_vector);

  /*!
   * \brief Obtain a 0-based leaf index for each observation in a ForestDataset.
   *        Internally, trees are stored as vectors of node information, 
   *        and the `leaves_` vector gives us node IDs for every leaf in the tree.
   *        Here, we would like to know, for every observation in a dataset, 
   *        which leaf number it is mapped to. Since the leaf numbers themselves 
   *        do not carry any information, we renumber them from `0` to `leaves_.size()-1`. 
   *
   *        Note: this is a tree-level helper function for an ensemble-level function. 
   *        It assumes the creation of: 
   *        -# a vector of column indices of size `dataset.NumObservations()` x `ensemble.NumTrees()`, stored in "tree-major" order
   *        -# a running counter of the number of tree-observations already indexed in the ensemble  
   *           (used as offsets for the leaf number computed and returned here)
   *        Users running this function for a single tree may simply pre-allocate an output vector as 
   *        `std::vector<int32_t> output(dataset->NumObservations())` and set the offset to 0.
   * \param dataset Dataset with which to predict leaf indices from the tree
   * \param output Pre-allocated output vector storing a matrix of column indices, with "rows" corresponding to observations in `dataset` and "columns" corresponding to trees in an ensemble
   * \param offset Bookkeeping index that determines where in `output` vector that column indices should be unpacked
   * \param max_leaf Largest leaf value mapped so far. (Leaf indices serve as sparse column indices, so it is important that leaf values be unique to each tree.)
   */
  void PredictLeafIndexInplace(ForestDataset* dataset, std::vector<int32_t>& output, int32_t offset, int32_t max_leaf);

  /*!
   * \brief Obtain a 0-based leaf index for each observation in a ForestDataset.
   *        Internally, trees are stored as vectors of node information, 
   *        and the `leaves_` vector gives us node IDs for every leaf in the tree.
   *        Here, we would like to know, for every observation in a dataset, 
   *        which leaf number it is mapped to. Since the leaf numbers themselves 
   *        do not carry any information, we renumber them from `0` to `leaves_.size()-1`. 
   *
   *        Note: this is a tree-level helper function for an ensemble-level function. 
   *        It assumes the creation of: 
   *        -# a vector of column indices of size `dataset.NumObservations()` x `ensemble.NumTrees()`, stored in "tree-major" order
   *        -# a running counter of the number of tree-observations already indexed in the ensemble  
   *           (used as offsets for the leaf number computed and returned here)
   *        Users running this function for a single tree may simply pre-allocate an output vector as 
   *        `std::vector<int32_t> output(dataset->NumObservations())` and set the offset to 0.
   * \param covariates Eigen matrix with which to predict leaf indices
   * \param output Pre-allocated output vector storing a matrix of column indices, with "rows" corresponding to observations in `covariates` and "columns" corresponding to trees in an ensemble
   * \param offset Bookkeeping index that determines where in `output` vector that column indices should be unpacked
   * \param max_leaf Largest leaf value mapped so far. (Leaf indices serve as sparse column indices, so it is important that leaf values be unique to each tree.)
   */
  void PredictLeafIndexInplace(Eigen::MatrixXd& covariates, std::vector<int32_t>& output, int32_t offset, int32_t max_leaf);

  /*!
   * \brief Obtain a 0-based leaf index for each observation in a ForestDataset.
   *        Internally, trees are stored as vectors of node information, 
   *        and the `leaves_` vector gives us node IDs for every leaf in the tree.
   *        Here, we would like to know, for every observation in a dataset, 
   *        which leaf number it is mapped to. Since the leaf numbers themselves 
   *        do not carry any information, we renumber them from `0` to `leaves_.size()-1`. 
   *
   *        Note: this is a tree-level helper function for an ensemble-level function. 
   *        It assumes the creation of: 
   *        -# a vector of column indices of size `dataset.NumObservations()` x `ensemble.NumTrees()`, stored in "tree-major" order
   *        -# a running counter of the number of tree-observations already indexed in the ensemble  
   *           (used as offsets for the leaf number computed and returned here)
   *        Users running this function for a single tree may simply pre-allocate an output vector as 
   *        `std::vector<int32_t> output(dataset->NumObservations())` and set the offset to 0.
   * \param covariates Eigen matrix with which to predict leaf indices
   * \param output Pre-allocated output vector storing a matrix of column indices, with "rows" corresponding to observations in `covariates` and "columns" corresponding to trees in an ensemble
   * \param offset Bookkeeping index that determines where in `output` vector that column indices should be unpacked
   * \param max_leaf Largest leaf value mapped so far. (Leaf indices serve as sparse column indices, so it is important that leaf values be unique to each tree.)
   */
  void PredictLeafIndexInplace(Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>& covariates, std::vector<int32_t>& output, int32_t offset, int32_t max_leaf);

  void PredictLeafIndexInplace(Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>& covariates, 
                               Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>& output, 
                               int column_ind, int32_t offset, int32_t max_leaf);

  // Node info
  std::vector<TreeNodeType> node_type_;
  std::vector<std::int32_t> parent_;
  std::vector<std::int32_t> cleft_;
  std::vector<std::int32_t> cright_;
  std::vector<std::int32_t> split_index_;
  std::vector<double> leaf_value_;
  std::vector<double> threshold_;
  std::vector<bool> node_deleted_;
  std::vector<std::int32_t> internal_nodes_;
  std::vector<std::int32_t> leaves_;
  std::vector<std::int32_t> leaf_parents_;
  std::vector<std::int32_t> deleted_nodes_;
  
  // Leaf vector
  std::vector<double> leaf_vector_;
  std::vector<std::uint64_t> leaf_vector_begin_;
  std::vector<std::uint64_t> leaf_vector_end_;

  // Category list
  std::vector<std::uint32_t> category_list_;
  std::vector<std::uint64_t> category_list_begin_;
  std::vector<std::uint64_t> category_list_end_;

  bool has_categorical_split_{false};
  int output_dimension_{1};
  bool is_log_scale_{false};
};

/*! \brief Comparison operator for trees */
inline bool operator==(const Tree& lhs, const Tree& rhs) {
  return (
    (lhs.has_categorical_split_ == rhs.has_categorical_split_) && 
    (lhs.output_dimension_ == rhs.output_dimension_) && 
    (lhs.is_log_scale_ == rhs.is_log_scale_) && 
    (lhs.node_type_ == rhs.node_type_) && 
    (lhs.parent_ == rhs.parent_) && 
    (lhs.cleft_ == rhs.cleft_) && 
    (lhs.cright_ == rhs.cright_) && 
    (lhs.split_index_ == rhs.split_index_) && 
    (lhs.leaf_value_ == rhs.leaf_value_) && 
    (lhs.threshold_ == rhs.threshold_) && 
    (lhs.internal_nodes_ == rhs.internal_nodes_) && 
    (lhs.leaves_ == rhs.leaves_) && 
    (lhs.leaf_parents_ == rhs.leaf_parents_) && 
    (lhs.deleted_nodes_ == rhs.deleted_nodes_) && 
    (lhs.leaf_vector_ == rhs.leaf_vector_) && 
    (lhs.leaf_vector_begin_ == rhs.leaf_vector_begin_) && 
    (lhs.leaf_vector_end_ == rhs.leaf_vector_end_) && 
    (lhs.category_list_ == rhs.category_list_) && 
    (lhs.category_list_begin_ == rhs.category_list_begin_) && 
    (lhs.category_list_end_ == rhs.category_list_end_)
  );
}

/*! \brief Determine whether an observation produces a "true" value in a numeric split node
 *  \param fvalue Value of the split feature for the observation
 *  \param threshold Value of the numeric split threshold at the node
 *  \param left_child Node id of the left child
 *  \param right_child Node id of the right child
 */
inline bool SplitTrueNumeric(double fvalue, double threshold) {
  return (fvalue <= threshold);
}

/*! \brief Determine whether an observation produces a "true" value in a categorical split node
 *  \param fvalue Value of the split feature for the observation
 *  \param category_list Category indices that route an observation to the left child
 *  \param left_child Node id of the left child
 *  \param right_child Node id of the right child
 */
inline bool SplitTrueCategorical(double fvalue, std::vector<std::uint32_t> const& category_list) {
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
  return category_matched;
}

/*! \brief Return left or right node id based on a numeric split
 *  \param fvalue Value of the split feature for the observation
 *  \param threshold Value of the numeric split threshold at the node
 *  \param left_child Node id of the left child
 *  \param right_child Node id of the right child
 */
inline int NextNodeNumeric(double fvalue, double threshold, int left_child, int right_child) {
  return (SplitTrueNumeric(fvalue, threshold) ? left_child : right_child);
}

/*! \brief Return left or right node id based on a categorical split
 *  \param fvalue Value of the split feature for the observation
 *  \param category_list Category indices that route an observation to the left child
 *  \param left_child Node id of the left child
 *  \param right_child Node id of the right child
 */
inline int NextNodeCategorical(double fvalue, std::vector<std::uint32_t> const& category_list, int left_child, int right_child) {
  return SplitTrueCategorical(fvalue, category_list) ? left_child : right_child;
}

/*! 
 * Determine the node at which a tree places a given observation
 * 
 * \param tree Tree object used for prediction
 * \param data Dataset used for prediction
 * \param row Row indexing the prediction observation
 */
inline int EvaluateTree(Tree const& tree, Eigen::MatrixXd& data, int row) {
  int node_id = 0;
  while (!tree.IsLeaf(node_id)) {
    auto const split_index = tree.SplitIndex(node_id);
    double const fvalue = data(row, split_index);
    if (std::isnan(fvalue)) {
      node_id = tree.DefaultChild(node_id);
    } else {
      if (tree.NodeType(node_id) == StochTree::TreeNodeType::kCategoricalSplitNode) {
        node_id = NextNodeCategorical(fvalue, tree.CategoryList(node_id),
            tree.LeftChild(node_id), tree.RightChild(node_id));
      } else {
        node_id = NextNodeNumeric(fvalue, tree.Threshold(node_id), tree.LeftChild(node_id), tree.RightChild(node_id));
      }
    }
  }
  return node_id;
}

/*! 
 * Determine the node at which a tree places a given observation
 * 
 * \param tree Tree object used for prediction
 * \param data Dataset used for prediction
 * \param row Row indexing the prediction observation
 */
inline int EvaluateTree(Tree const& tree, Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>& data, int row) {
  int node_id = 0;
  while (!tree.IsLeaf(node_id)) {
    auto const split_index = tree.SplitIndex(node_id);
    double const fvalue = data(row, split_index);
    if (std::isnan(fvalue)) {
      node_id = tree.DefaultChild(node_id);
    } else {
      if (tree.NodeType(node_id) == StochTree::TreeNodeType::kCategoricalSplitNode) {
        node_id = NextNodeCategorical(fvalue, tree.CategoryList(node_id),
            tree.LeftChild(node_id), tree.RightChild(node_id));
      } else {
        node_id = NextNodeNumeric(fvalue, tree.Threshold(node_id), tree.LeftChild(node_id), tree.RightChild(node_id));
      }
    }
  }
  return node_id;
}

/*! \brief Determine whether a given observation is "true" at a split proposed by split_index and split_value
 *  \param covariates Dataset used for prediction
 *  \param row Row indexing the prediction observation
 *  \param split_index Column of new split
 *  \param split_value Value defining the split
 */
inline bool RowSplitLeft(Eigen::MatrixXd& covariates, int row, int split_index, double split_value) {
  double const fvalue = covariates(row, split_index);
  return SplitTrueNumeric(fvalue, split_value);
}

/*! \brief Determine whether a given observation is "true" at a split proposed by split_index and split_value
 *  \param covariates Dataset used for prediction
 *  \param row Row indexing the prediction observation
 *  \param split_index Column of new split
 *  \param category_list Categories defining the split
 */
inline bool RowSplitLeft(Eigen::MatrixXd& covariates, int row, int split_index, std::vector<std::uint32_t> const& category_list) {
  double const fvalue = covariates(row, split_index);
  return SplitTrueCategorical(fvalue, category_list);
}

/*! \brief Representation of arbitrary tree split rules, including numeric split rules (`X[,i] <= c`) and categorical split rules (`X[,i] in {2,4,6,7}`) */
class TreeSplit {
 public:
  TreeSplit() {}
  /*!
   * \brief Construct a numeric TreeSplit
   * 
   * \param split_value Numeric cutoff defining a new split rule
   */
  TreeSplit(double split_value) {
    numeric_ = true;
    split_value_ = split_value;
    split_set_ = true;
  }
  /*!
   * \brief Construct a categorical TreeSplit
   * 
   * \param split_categories Vector of category indices defining a new (unordered) categorical split rule
   */
  TreeSplit(std::vector<std::uint32_t>& split_categories) {
    numeric_ = false;
    split_categories_ = split_categories;
    split_set_ = true;
  }
  ~TreeSplit() {}
  bool SplitSet() {return split_set_;}
  /*! \brief Whether or not a `TreeSplit` rule is numeric */
  bool NumericSplit() {return numeric_;}
  /*!
   * \brief Whether a given covariate value is `True` or `False` on the rule defined by a `TreeSplit` object
   * 
   * \param fvalue Value of the covariate
   */
  bool SplitTrue(double fvalue) {
    if (numeric_) return SplitTrueNumeric(fvalue, split_value_);
    else return SplitTrueCategorical(fvalue, split_categories_);
  }
  /*! \brief Numeric cutoff value defining a `TreeSplit` object */
  double SplitValue() {return split_value_;}
  /*! \brief Categories defining a `TreeSplit` object */
  std::vector<std::uint32_t> SplitCategories() {return split_categories_;}
 private:
  bool split_set_{false};
  bool numeric_;
  double split_value_;
  std::vector<std::uint32_t> split_categories_;
};

/*! \} */ // end of tree_group

} // namespace StochTree

#endif // STOCHTREE_TREE_H_
