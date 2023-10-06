/*!
 * Based largely on the tree classes in xgboost and treelite, both released under the Apache license with the following copyright:
 * Copyright 2015-2023 by XGBoost Contributors
 * Copyright 2017-2021 by [treelite] Contributors
 */
#ifndef STOCHTREE_TREE_H_
#define STOCHTREE_TREE_H_

#include <stochtree/log.h>
#include <stochtree/meta.h>

#include <cstdint>
#include <stack>
#include <string>

namespace StochTree {

enum FeatureSplitType {
  kNumericSplit,
  kOrderedCategoricalSplit,
  kUnorderedCategoricalSplit
};

/**
 * \brief Data structure for storing and modifying decision trees
 */
class Tree {
 public:
  static constexpr node_t kInvalidNodeId{-1};
  static constexpr node_t kDeletedNodeMarker = std::numeric_limits<node_t>::max();
  static constexpr node_t kRoot{0};

  /*! \brief tree node */
  class Node {
   public:
    Node()  {
      // assert compact alignment
      static_assert(sizeof(Node) == 4 * sizeof(int) + sizeof(Info),
                    "Node: 64 bit align");
    }
    Node(int32_t cleft, int32_t cright, int32_t parent,
         uint32_t split_ind, float split_cond, bool default_left) :
        parent_{parent}, cleft_{cleft}, cright_{cright} {
      this->SetParent(parent_);
      this->SetSplit(split_ind, split_cond, default_left);
    }

    /*! \brief index of left child */
    [[nodiscard]] int LeftChild() const { return this->cleft_; }
    /*! \brief index of right child */
    [[nodiscard]] int RightChild() const { return this->cright_; }
    /*! \brief index of default child when feature is missing */
    [[nodiscard]] int DefaultChild() const {
      return this->DefaultLeft() ? this->LeftChild() : this->RightChild();
    }
    /*! \brief feature index of split condition */
    [[nodiscard]] unsigned SplitIndex() const {
      return sindex_ & ((1U << 31) - 1U);
    }
    /*! \brief when feature is unknown, whether goes to left child */
    [[nodiscard]] bool DefaultLeft() const { return (sindex_ >> 31) != 0; }
    /*! \brief whether current node is leaf node */
    [[nodiscard]] bool IsLeaf() const { return cleft_ == kInvalidNodeId; }
    /*! \return get leaf value of leaf node */
    [[nodiscard]] float LeafValue() const { return (this->info_).leaf_value; }
    /*! \return get split condition of the node */
    [[nodiscard]] split_cond_t SplitCond() const { return (this->info_).split_cond; }
    /*! \brief get parent of the node */
    [[nodiscard]] int Parent() const { return parent_ & ((1U << 31) - 1); }
    /*! \brief whether current node is left child */
    [[nodiscard]] bool IsLeftChild() const { return (parent_ & (1U << 31)) != 0; }
    /*! \brief whether this node is deleted */
    [[nodiscard]] bool IsDeleted() const { return sindex_ == kDeletedNodeMarker; }
    /*! \brief whether current node is root */
    [[nodiscard]] bool IsRoot() const { return parent_ == kInvalidNodeId; }
    /*!
     * \brief set the left child
     * \param nid node id to right child
     */
    void SetLeftChild(int nid) {
      this->cleft_ = nid;
    }
    /*!
     * \brief set the right child
     * \param nid node id to right child
     */
    void SetRightChild(int nid) {
      this->cright_ = nid;
    }
    /*!
     * \brief set split condition of current node
     * \param split_index feature index to split
     * \param split_cond  split condition
     * \param default_left the default direction when feature is unknown
     */
    void SetSplit(unsigned split_index, split_cond_t split_cond,
                  bool default_left = false) {
      if (default_left) split_index |= (1U << 31);
      this->sindex_ = split_index;
      (this->info_).split_cond = split_cond;
    }
    /*!
     * \brief set the leaf value of the node
     * \param value leaf value
     * \param right right index, could be used to store
     *        additional information
     */
    void SetLeaf(double value, int right = kInvalidNodeId) {
      (this->info_).leaf_value = value;
      this->cleft_ = kInvalidNodeId;
      this->cright_ = right;
    }
    /*! \brief mark that this node is deleted */
    void MarkDelete() {
      this->sindex_ = kDeletedNodeMarker;
    }
    /*! \brief Reuse this deleted node. */
    void Reuse() {
      this->sindex_ = 0;
    }
    // set parent
    void SetParent(int pidx, bool is_left_child = true) {
      if (is_left_child) pidx |= (1U << 31);
      this->parent_ = pidx;
    }
    bool operator==(const Node& b) const {
      return parent_ == b.parent_ && cleft_ == b.cleft_ &&
             cright_ == b.cright_ && sindex_ == b.sindex_ &&
             info_.leaf_value == b.info_.leaf_value;
    }

   private:
    /*!
     * \brief in leaf node, we have weights, in non-leaf nodes,
     *        we have split condition
     */
    union Info{
      double leaf_value;
      split_cond_t split_cond;
    };
    // pointer to parent, highest bit is used to
    // indicate whether it's a left child or not
    int32_t parent_{kInvalidNodeId};
    // pointer to left, right
    int32_t cleft_{kInvalidNodeId}, cright_{kInvalidNodeId};
    // split feature index, left split or right split depends on the highest bit
    uint32_t sindex_{0};
    // extra info
    Info info_;
  };

  /*!
   * \brief change a non leaf node to a leaf node, delete its children
   * \param rid node id of the node
   * \param value new leaf value
   */
  void ChangeToLeaf(int rid, float value) {
    CHECK(nodes_[nodes_[rid].LeftChild()].IsLeaf());
    CHECK(nodes_[nodes_[rid].RightChild()].IsLeaf());
    this->DeleteNode(nodes_[rid].LeftChild());
    this->DeleteNode(nodes_[rid].RightChild());
    nodes_[rid].SetLeaf(value);

    // Add rid to leaves and remove from internal nodes and leaf parents (if it was there)
    leaves_.push_back(rid);
    leaf_parents_.erase(std::remove(leaf_parents_.begin(), leaf_parents_.end(), rid), leaf_parents_.end());
    internal_nodes_.erase(std::remove(internal_nodes_.begin(), internal_nodes_.end(), rid), internal_nodes_.end());
  }
  /*!
   * \brief collapse a non leaf node to a leaf node, delete its children
   * \param rid node id of the node
   * \param value new leaf value
   */
  void CollapseToLeaf(int rid, float value) {
    if (nodes_[rid].IsLeaf()) return;
    if (!nodes_[nodes_[rid].LeftChild() ].IsLeaf()) {
      CollapseToLeaf(nodes_[rid].LeftChild(), 0.0f);
    }
    if (!nodes_[nodes_[rid].RightChild() ].IsLeaf()) {
      CollapseToLeaf(nodes_[rid].RightChild(), 0.0f);
    }
    this->ChangeToLeaf(rid, value);
  }

  Tree() {
    param_ = TreeParam();
    nodes_.resize(param_.num_nodes);
    internal_nodes_ = std::vector<node_t>(0);
    leaf_parents_ = std::vector<node_t>(0);
    leaves_ = std::vector<node_t>(1, 0);
    split_types_.resize(param_.num_nodes, FeatureSplitType::kNumericSplit);
    for (int i = 0; i < param_.num_nodes; i++) {
      nodes_[i].SetLeaf(0.0f);
      nodes_[i].SetParent(kInvalidNodeId);
    }
  }

  /*! \brief Copy constructor */
  Tree(Tree& tree) {
    param_ = TreeParam();
    param_.num_nodes = tree.param_.num_nodes;
    param_.num_deleted = tree.param_.num_deleted;
    param_.num_feature = tree.param_.num_feature;
    nodes_ = tree.nodes_;
    internal_nodes_ = tree.internal_nodes_;
    leaf_parents_ = tree.leaf_parents_;
    leaves_ = tree.leaves_;
    split_types_ = tree.split_types_;
    deleted_nodes_ = tree.deleted_nodes_;
  }
  
  /**
   * \brief Constructor that initializes the tree model with shape.
   */
  explicit Tree(feature_size_t n_features) : Tree{} {
    param_.num_feature = n_features;
  }

  /*! \brief get node given nid */
  Node& operator[](int nid) {
    return nodes_[nid];
  }
  /*! \brief get node given nid */
  const Node& operator[](int nid) const {
    return nodes_[nid];
  }

  /*! \brief get const reference to nodes */
  [[nodiscard]] const std::vector<Node>& GetNodes() const { return nodes_; }

  bool operator==(const Tree& b) const {
    return nodes_ == b.nodes_ && deleted_nodes_ == b.deleted_nodes_ && 
           param_ == b.param_;
  }
  /* \brief Iterate through all nodes in this tree.
   *
   * \param Function that accepts a node index, and returns false when iteration should
   *        stop, otherwise returns true.
   */
  template <typename Func> void WalkTree(Func func) const {
    std::stack<node_t> nodes;
    nodes.push(kRoot);
    auto &self = *this;
    while (!nodes.empty()) {
      auto nidx = nodes.top();
      nodes.pop();
      if (!func(nidx)) {
        return;
      }
      auto left = self[nidx].LeftChild();
      auto right = self[nidx].RightChild();
      if (left != Tree::kInvalidNodeId) {
        nodes.push(left);
      }
      if (right != Tree::kInvalidNodeId) {
        nodes.push(right);
      }
    }
  }
  /*!
   * \brief Compares whether 2 trees are equal from a user's perspective.  The equality
   *        compares only non-deleted nodes.
   *
   * \param b The other tree.
   */
  [[nodiscard]] bool Equal(const Tree& b) const;

  /**
   * \brief Expands a leaf node into two additional leaf nodes.
   *
   * \param nid                       The node index to expand.
   * \param split_index               Feature index of the split.
   * \param split_value               The split condition.
   * \param default_left              True to default left.
   * \param left_leaf_value           The left leaf value used for prediction.
   * \param right_leaf_value          The right leaf value for prediction.
   * \param left_sample_size          The left leaf sample size.
   * \param right_sample_size         The right leaf sample size.
   * \param left_outcome_sum          The left leaf sum of outcome values.
   * \param right_outcome_sum         The right leaf sum of outcome values.
   * \param left_outcome_sum_squares  The left leaf sum of squares of outcome values.
   * \param right_outcome_sum_squares The right leaf sum of squares of outcome values.
   * \param leaf_right_child          The right child index of leaf, by default kInvalidNodeId,
   *                                  some updaters use the right child index of leaf as a marker
   */
  void ExpandNode(node_t nid, unsigned split_index, double split_value,
                  bool default_left, double base_weight,
                  double left_leaf_value, double right_leaf_value, 
                  node_t leaf_right_child);

  /*! \brief Exports tree to string. */
  void ToString();

  /**
   * \brief Get the number of features.
   */
  [[nodiscard]] feature_size_t NumFeatures() const noexcept { return param_.num_feature; }
  /**
   * \brief Get the total number of nodes including deleted ones in this tree.
   */
  [[nodiscard]] node_t NumNodes() const noexcept { return param_.num_nodes; }
  /**
   * \brief Get the total number of valid nodes in this tree.
   */
  [[nodiscard]] node_t NumValidNodes() const noexcept {
    return param_.num_nodes - param_.num_deleted;
  }
  /**
   * \brief number of extra nodes besides the root
   */
  [[nodiscard]] node_t NumExtraNodes() const noexcept {
    return param_.num_nodes - 1 - param_.num_deleted;
  }
  /* \brief Count number of leaves in tree. */
  [[nodiscard]] node_t GetNumLeaves() const;
  [[nodiscard]] node_t GetNumLeafParents() const;
  [[nodiscard]] node_t GetNumSplitNodes() const;

  /*!
   * \brief get current depth
   * \param nid node id
   */
  [[nodiscard]] std::int32_t GetDepth(node_t nid) const {
    int depth = 0;
    while (!nodes_[nid].IsRoot()) {
      ++depth;
      nid = nodes_[nid].Parent();
    }
    return depth;
  }

  /*!
   * \brief get maximum depth
   * \param nid node id
   */
  [[nodiscard]] int MaxDepth(int nid) const {
    if (nodes_[nid].IsLeaf()) return 0;
    return std::max(MaxDepth(nodes_[nid].LeftChild()) + 1, MaxDepth(nodes_[nid].RightChild()) + 1);
  }

  /*!
   * \brief get maximum depth
   */
  int MaxDepth() { return MaxDepth(0); }

  /*! \brief Serialize tree to json*/
  std::string ToJSON() const;

  /*! \brief Serialize node at `index` and all of its descendents to json*/
  std::string NodeToJSON(int index) const;

  /*!
   * \brief dump the model in the requested format as a text string
   * \param with_stats whether dump out statistics as well
   * \return the string of dumped model
   */
  [[nodiscard]] std::string DumpModel(bool with_stats) const;
  /*!
   * \brief Get split type for a node.
   * \param nidx Index of node.
   * \return The type of this split.  For leaf node it's always kNumerical.
   */
  [[nodiscard]] FeatureSplitType NodeSplitType(node_t nidx) const { return split_types_.at(nidx); }
  /*!
   * \brief Get split types for all nodes.
   */
  [[nodiscard]] std::vector<FeatureSplitType> const& GetSplitTypes() const {
    return split_types_;
  }
  /*!
   * \brief Get indices of all internal nodes.
   */
  [[nodiscard]] std::vector<node_t> const& GetInternalNodes() const {
    return internal_nodes_;
  }
  /*!
   * \brief Get indices of all leaf nodes.
   */
  [[nodiscard]] std::vector<node_t> const& GetLeaves() const {
    return leaves_;
  }
  /*!
   * \brief Get indices of all leaf parent nodes.
   */
  [[nodiscard]] std::vector<node_t> const& GetLeafParents() const {
    return leaf_parents_;
  }

  [[nodiscard]] feature_size_t SplitIndex(node_t nidx) const {
    return (*this)[nidx].SplitIndex();
  }
  [[nodiscard]] float SplitCond(node_t nidx) const {
    return (*this)[nidx].SplitCond();
  }
  [[nodiscard]] bool DefaultLeft(node_t nidx) const {
    return (*this)[nidx].DefaultLeft();
  }
  [[nodiscard]] bool IsRoot(node_t nidx) const {
    return (*this)[nidx].IsRoot();
  }
  [[nodiscard]] bool IsLeaf(node_t nidx) const {
    return (*this)[nidx].IsLeaf();
  }
  [[nodiscard]] node_t Parent(node_t nidx) const {
    return (*this)[nidx].Parent();
  }
  [[nodiscard]] node_t LeftChild(node_t nidx) const {
    return (*this)[nidx].LeftChild();
  }
  [[nodiscard]] node_t RightChild(node_t nidx) const {
    return (*this)[nidx].RightChild();
  }
  [[nodiscard]] bool IsLeftChild(node_t nidx) const {
    return (*this)[nidx].IsLeftChild();
  }
  [[nodiscard]] node_t Size() const {
    return this->nodes_.size();
  }
  [[nodiscard]] bool IsLeafParent(node_t nidx) const {
    // False until we deduce left and right node are
    // available and both are leaves
    bool is_left_leaf = false;
    bool is_right_leaf = false;
    // Check if node nidx is a leaf, if so, return false
    bool is_leaf = (*this)[nidx].IsLeaf();
    if (is_leaf){
      return false;
    } else {
      // If nidx is not a leaf, it must have left and right nodes
      // so we check if those are leaves
      node_t left_node = LeftChild(nidx);
      node_t right_node = RightChild(nidx);
      is_left_leaf = IsLeaf(left_node);
      is_right_leaf = IsLeaf(right_node);
    }
    return is_left_leaf && is_right_leaf;
  }

  void InplacePredictFromNodes(std::vector<double> result, std::vector<data_size_t> node_indices);

  std::vector<double> PredictFromNodes(std::vector<data_size_t> node_indices);

 /*! \brief meta parameters of the tree */
 struct TreeParam {
  /*! \brief total number of nodes */
  int num_nodes{1};
  /*!\brief number of deleted nodes */
  int num_deleted{0};
  /*! \brief number of features used for tree construction */
  feature_size_t num_feature{0};
  /*! \brief constructor */
  TreeParam() {
    // assert compact alignment
    static_assert(sizeof(TreeParam) == (3) * sizeof(int), "TreeParam: 64 bit align");
  }
  bool operator==(const TreeParam& b) const {
    return num_nodes == b.num_nodes && num_deleted == b.num_deleted &&
          num_feature == b.num_feature;
  }
 };

 private:
  /*! \brief model parameter */
  TreeParam param_;
  // vector of nodes
  std::vector<Node> nodes_;
  // vector of internal nodes
  std::vector<node_t> internal_nodes_;
  // vector of leaf nodes
  std::vector<node_t> leaves_;
  // vector of parents of leaf nodes
  std::vector<node_t> leaf_parents_;
  // free node space, used during training process
  std::vector<node_t> deleted_nodes_;
  // split types of internal nodes
  std::vector<FeatureSplitType> split_types_;
  // allocate a new node,
  // !!!!!! NOTE: may cause BUG here, nodes.resize
  node_t AllocNode() {
    if (param_.num_deleted != 0) {
      int nid = deleted_nodes_.back();
      deleted_nodes_.pop_back();
      nodes_[nid].Reuse();
      --param_.num_deleted;
      return nid;
    }
    int nd = param_.num_nodes++;

    CHECK_LT(param_.num_nodes, std::numeric_limits<int>::max());
    nodes_.resize(param_.num_nodes);
    split_types_.resize(param_.num_nodes, FeatureSplitType::kNumericSplit);
    return nd;
  }
  // delete a tree node, keep the parent field to allow trace back
  void DeleteNode(int nid) {
    CHECK_GE(nid, 1);
    auto pid = (*this)[nid].Parent();
    if (nid == (*this)[pid].LeftChild()) {
      (*this)[pid].SetLeftChild(kInvalidNodeId);
    } else {
      (*this)[pid].SetRightChild(kInvalidNodeId);
    }

    deleted_nodes_.push_back(nid);
    nodes_[nid].MarkDelete();
    ++param_.num_deleted;

    // Remove from vectors that track leaves, leaf parents, internal nodes, etc...
    leaves_.erase(std::remove(leaves_.begin(), leaves_.end(), nid), leaves_.end());
    leaf_parents_.erase(std::remove(leaf_parents_.begin(), leaf_parents_.end(), nid), leaf_parents_.end());
    internal_nodes_.erase(std::remove(internal_nodes_.begin(), internal_nodes_.end(), nid), internal_nodes_.end());
  }
};

} // namespace StochTree

#endif // STOCHTREE_TREE_H_
