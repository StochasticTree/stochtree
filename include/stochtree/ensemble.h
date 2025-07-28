/*!
 * Copyright (c) 2024 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 * 
 * Inspired by the design of the Learner, GBTreeModel, and GBTree classes in xgboost, 
 * released under the Apache license with the following copyright:
 * 
 * Copyright 2015-2023 by XGBoost Contributors
 */
#ifndef STOCHTREE_ENSEMBLE_H_
#define STOCHTREE_ENSEMBLE_H_

#include <stochtree/data.h>
#include <stochtree/tree.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace StochTree {

/*!
 * \defgroup forest_group Forest API
 *
 * \brief Classes / functions for creating and modifying forests (i.e. ensembles of trees).
 *
 * \{
 */

/*! \brief Class storing a "forest," or an ensemble of decision trees.
 */
class TreeEnsemble {
 public:
  /*!
   * \brief Initialize a new TreeEnsemble
   * 
   * \param num_trees Number of trees in a forest
   * \param output_dimension Dimension of the leaf node parameter
   * \param is_leaf_constant Whether or not the leaves of each tree are treated as "constant." If true, then predicting from an ensemble is simply a matter or determining which leaf node an observation falls into. If false, prediction will multiply a leaf node's parameter(s) for a given observation by a basis vector.
   * \param is_exponentiated Whether or not the leaves of each tree are stored in log scale. If true, leaf predictions are exponentiated before their prediction is returned.
   */
  TreeEnsemble(int num_trees, int output_dimension = 1, bool is_leaf_constant = true, bool is_exponentiated = false) {
    // Initialize trees in the ensemble
    trees_ = std::vector<std::unique_ptr<Tree>>(num_trees);
    for (int i = 0; i < num_trees; i++) {
      trees_[i].reset(new Tree());
      trees_[i]->Init(output_dimension, is_exponentiated);
    }
    // Store ensemble configurations
    num_trees_ = num_trees;
    output_dimension_ = output_dimension;
    is_leaf_constant_ = is_leaf_constant;
    is_exponentiated_ = is_exponentiated;
  }
  
  /*!
   * \brief Initialize an ensemble based on the state of an existing ensemble
   * 
   * \param ensemble `TreeEnsemble` used to initialize the current ensemble
   */
  TreeEnsemble(TreeEnsemble& ensemble) {
    // Unpack ensemble configurations
    num_trees_ = ensemble.num_trees_;
    output_dimension_ = ensemble.output_dimension_;
    is_leaf_constant_ = ensemble.is_leaf_constant_;
    is_exponentiated_ = ensemble.is_exponentiated_;
    // Initialize trees in the ensemble
    trees_ = std::vector<std::unique_ptr<Tree>>(num_trees_);
    for (int i = 0; i < num_trees_; i++) {
      trees_[i].reset(new Tree());
    }
    // Clone trees in the ensemble
    for (int j = 0; j < num_trees_; j++) {
      Tree* tree = ensemble.GetTree(j);
      this->CloneFromExistingTree(j, tree);
    }
  }
  
  ~TreeEnsemble() {}

  /*!
   * \brief Combine two forests into a single forest by merging their trees
   * 
   * \param ensemble Reference to another `TreeEnsemble` that will be merged into the current ensemble
   */
  void MergeForest(TreeEnsemble& ensemble) {
    // Unpack ensemble configurations
    int old_num_trees = num_trees_;
    num_trees_ += ensemble.num_trees_;
    CHECK_EQ(output_dimension_, ensemble.output_dimension_);
    CHECK_EQ(is_leaf_constant_, ensemble.is_leaf_constant_);
    CHECK_EQ(is_exponentiated_, ensemble.is_exponentiated_);
    // Resize tree vector and reset new trees
    trees_.resize(num_trees_);
    for (int i = old_num_trees; i < num_trees_; i++) {
      trees_[i].reset(new Tree());
    }
    // Clone trees in the input ensemble
    for (int j = 0; j < ensemble.num_trees_; j++) {
      Tree* tree = ensemble.GetTree(j);
      this->CloneFromExistingTree(old_num_trees + j, tree);
    }
  }

  /*!
   * \brief Add a constant value to every leaf of every tree in an ensemble. If leaves are multi-dimensional, `constant_value` will be added to every dimension of the leaves.
   * 
   * \param constant_value Value that will be added to every leaf of every tree
   */
  void AddValueToLeaves(double constant_value) {
    for (int j = 0; j < num_trees_; j++) {
      Tree* tree = GetTree(j);
      tree->AddValueToLeaves(constant_value);
    }
  }

  /*!
   * \brief Multiply every leaf of every tree by a constant value. If leaves are multi-dimensional, `constant_multiple` will be multiplied through every dimension of the leaves.
   * 
   * \param constant_multiple Value that will be multiplied by every leaf of every tree
   */
  void MultiplyLeavesByValue(double constant_multiple) {
    for (int j = 0; j < num_trees_; j++) {
      Tree* tree = GetTree(j);
      tree->MultiplyLeavesByValue(constant_multiple);
    }
  }

  /*!
   * \brief Return a pointer to a tree in the forest
   * 
   * \param i Index (0-based) of a tree to be queried
   * \return Tree* 
   */
  inline Tree* GetTree(int i) {
    return trees_[i].get();
  }

  /*!
   * \brief Reset a `TreeEnsemble` to all single-node "root" trees
   */
  inline void ResetRoot() {
    for (int i = 0; i < num_trees_; i++) {
      ResetInitTree(i);
    }
  }

  /*!
   * \brief Reset a single tree in an ensemble
   * \todo Consider refactoring this and `ResetInitTree`
   * 
   * \param i Index (0-based) of the tree to be reset
   */
  inline void ResetTree(int i) {
    trees_[i].reset(new Tree());
  }

  /*!
   * \brief Reset a single tree in an ensemble
   * \todo Consider refactoring this and `ResetTree`
   * 
   * \param i Index (0-based) of the tree to be reset
   */
  inline void ResetInitTree(int i) {
    trees_[i].reset(new Tree());
    trees_[i]->Init(output_dimension_, is_exponentiated_);
  }

  /*!
   * \brief Clone a single tree in an ensemble from an existing tree, overwriting current tree
   * 
   * \param i Index of the tree to be overwritten
   * \param tree Pointer to tree used to clone tree `i`
   */
  inline void CloneFromExistingTree(int i, Tree* tree) {
    return trees_[i]->CloneFromTree(tree);
  }

  /*!
   * \brief Reset an ensemble to clone another ensemble
   * 
   * \param ensemble Reference to an existing `TreeEnsemble`
   */
  inline void ReconstituteFromForest(TreeEnsemble& ensemble) {
    // Delete old tree pointers
    trees_.clear();
    // Unpack ensemble configurations
    num_trees_ = ensemble.num_trees_;
    output_dimension_ = ensemble.output_dimension_;
    is_leaf_constant_ = ensemble.is_leaf_constant_;
    is_exponentiated_ = ensemble.is_exponentiated_;
    // Initialize trees in the ensemble
    trees_ = std::vector<std::unique_ptr<Tree>>(num_trees_);
    for (int i = 0; i < num_trees_; i++) {
      trees_[i].reset(new Tree());
    }
    // Clone trees in the ensemble
    for (int j = 0; j < num_trees_; j++) {
      Tree* tree = ensemble.GetTree(j);
      this->CloneFromExistingTree(j, tree);
    }
  }

  std::vector<double> Predict(ForestDataset& dataset) {
    data_size_t n = dataset.NumObservations();
    std::vector<double> output(n);
    PredictInplace(dataset, output, 0);
    return output;
  }

  std::vector<double> PredictRaw(ForestDataset& dataset) {
    data_size_t n = dataset.NumObservations();
    data_size_t total_output_size = n * output_dimension_;
    std::vector<double> output(total_output_size);
    PredictRawInplace(dataset, output, 0);
    return output;
  }
  
  inline void PredictInplace(ForestDataset& dataset, std::vector<double> &output, data_size_t offset = 0) {
    PredictInplace(dataset, output, 0, trees_.size(), offset);
  }

  inline void PredictInplace(ForestDataset& dataset, std::vector<double> &output, 
                             int tree_begin, int tree_end, data_size_t offset = 0) {
    if (is_leaf_constant_) {
      PredictInplace(dataset.GetCovariates(), output, tree_begin, tree_end, offset);
    } else {
      CHECK(dataset.HasBasis());
      PredictInplace(dataset.GetCovariates(), dataset.GetBasis(), output, tree_begin, tree_end, offset);
    }
  }

  inline void PredictInplace(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, std::vector<double> &output, data_size_t offset = 0) {
    PredictInplace(covariates, basis, output, 0, trees_.size(), offset);
  }

  inline void PredictInplace(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, std::vector<double> &output, 
                             int tree_begin, int tree_end, data_size_t offset = 0) {
    double pred;
    CHECK_EQ(covariates.rows(), basis.rows());
    CHECK_EQ(output_dimension_, trees_[0]->OutputDimension());
    CHECK_EQ(output_dimension_, basis.cols());
    data_size_t n = covariates.rows();
    data_size_t total_output_size = n;
    if (output.size() < total_output_size + offset) {
      Log::Fatal("Mismatched size of prediction vector and training data");
    }
    for (data_size_t i = 0; i < n; i++) {
      pred = 0.0;
      for (size_t j = tree_begin; j < tree_end; j++) {
        auto &tree = *trees_[j];
        std::int32_t nidx = EvaluateTree(tree, covariates, i);
        for (int32_t k = 0; k < output_dimension_; k++) {
          pred += tree.LeafValue(nidx, k) * basis(i, k);
        }
      }
      if (is_exponentiated_) output[i + offset] = std::exp(pred);
      else output[i + offset] = pred;
    }
  }

  inline void PredictInplace(Eigen::MatrixXd& covariates, std::vector<double> &output, data_size_t offset = 0) {
    PredictInplace(covariates, output, 0, trees_.size(), offset);
  }

  inline void PredictInplace(Eigen::MatrixXd& covariates, std::vector<double> &output, int tree_begin, int tree_end, data_size_t offset = 0) {
    double pred;
    data_size_t n = covariates.rows();
    data_size_t total_output_size = n;
    if (output.size() < total_output_size + offset) {
      Log::Fatal("Mismatched size of prediction vector and training data");
    }
    for (data_size_t i = 0; i < n; i++) {
      pred = 0.0;
      for (size_t j = tree_begin; j < tree_end; j++) {
        auto &tree = *trees_[j];
        std::int32_t nidx = EvaluateTree(tree, covariates, i);
        pred += tree.LeafValue(nidx, 0);
      }
      if (is_exponentiated_) output[i + offset] = std::exp(pred);
      else output[i + offset] = pred;
    }
  }

  inline void PredictRawInplace(ForestDataset& dataset, std::vector<double> &output, data_size_t offset = 0) {
    PredictRawInplace(dataset, output, 0, trees_.size(), offset);
  }

  inline void PredictRawInplace(ForestDataset& dataset, std::vector<double> &output, 
                             int tree_begin, int tree_end, data_size_t offset = 0) {
    double pred;
    Eigen::MatrixXd covariates = dataset.GetCovariates();
    CHECK_EQ(output_dimension_, trees_[0]->OutputDimension());
    data_size_t n = covariates.rows();
    data_size_t total_output_size = n * output_dimension_;
    if (output.size() < total_output_size + offset) {
      Log::Fatal("Mismatched size of raw prediction vector and training data");
    }
    for (data_size_t i = 0; i < n; i++) {
      for (int32_t k = 0; k < output_dimension_; k++) {
        pred = 0.0;
        for (size_t j = tree_begin; j < tree_end; j++) {
          auto &tree = *trees_[j];
          int32_t nidx = EvaluateTree(tree, covariates, i);
          pred += tree.LeafValue(nidx, k);
        }
        output[i*output_dimension_ + k + offset] = pred;
      }
    }
  }

  inline int32_t NumTrees() {
    return num_trees_;
  }

  inline int32_t NumLeaves() {
    int32_t result = 0;
    for (int i = 0; i < num_trees_; i++) {
      result += trees_[i]->NumLeaves();
    }
    return result;
  }

  inline double SumLeafSquared() {
    double result = 0.;
    for (int i = 0; i < num_trees_; i++) {
      result += trees_[i]->SumSquaredLeafValues();
    }
    return result;
  }

  inline int32_t OutputDimension() {
    return output_dimension_;
  }

  inline bool IsLeafConstant() {
    return is_leaf_constant_;
  }

  inline bool IsExponentiated() {
    return is_exponentiated_;
  }

  inline int32_t TreeMaxDepth(int tree_num) {
    return trees_[tree_num]->MaxLeafDepth();
  }

  inline double AverageMaxDepth() {
    double numerator = 0.;
    double denominator = 0.;
    for (int i = 0; i < num_trees_; i++) {
      numerator += static_cast<double>(TreeMaxDepth(i));
      denominator += 1.;
    }
    return numerator / denominator;
  }

  inline bool AllRoots() {
    for (int i = 0; i < num_trees_; i++) {
      if (!trees_[i]->IsRoot()) {
        return false;
      }
    }
    return true;
  }

  inline void SetLeafValue(double leaf_value) {
    CHECK_EQ(output_dimension_, 1);
    for (int i = 0; i < num_trees_; i++) {
      CHECK(trees_[i]->IsRoot());
      trees_[i]->SetLeaf(0, leaf_value);
    }
  }

  inline void SetLeafVector(std::vector<double>& leaf_vector) {
    CHECK_EQ(output_dimension_, leaf_vector.size());
    for (int i = 0; i < num_trees_; i++) {
      CHECK(trees_[i]->IsRoot());
      trees_[i]->SetLeafVector(0, leaf_vector);
    }
  }

  /*!
   * \brief Obtain a 0-based "maximum" leaf index for an ensemble, which is equivalent to the sum of the 
   *        number of leaves in each tree. This is used in conjunction with `PredictLeafIndicesInplace`,
   *        which returns an observation-specific leaf index for every observation-tree pair.
   */
  int GetMaxLeafIndex() {
    int max_leaf = 0;
    for (int j = 0; j < num_trees_; j++) {
      auto &tree = *trees_[j];
      max_leaf += tree.NumLeaves();
    }
    return max_leaf;
  }

  /*!
   * \brief Obtain a 0-based leaf index for every tree in an ensemble and for each 
   *        observation in a ForestDataset. Internally, trees are stored as essentially 
   *        vectors of node information, and the leaves_ vector gives us node IDs for every 
   *        leaf in the tree. Here, we would like to know, for every observation in a dataset, 
   *        which leaf number it is mapped to. Since the leaf numbers themselves 
   *        do not carry any information, we renumber them from 0 to `leaves_.size()-1`. 
   *        We compute this at the tree-level and coordinate this computation at the 
   *        ensemble level.
   *
   *        Note: this assumes the creation of a vector of column indices of size 
   *        `dataset.NumObservations()` x `ensemble.NumTrees()`
   * \param ForestDataset Dataset with which to predict leaf indices from the tree
   * \param output Vector of length num_trees*n which stores the leaf node prediction
   * \param num_trees Number of trees in an ensemble
   * \param n Size of dataset
   */
  void PredictLeafIndicesInplace(ForestDataset* dataset, std::vector<int32_t>& output, int num_trees, data_size_t n) {
    PredictLeafIndicesInplace(dataset->GetCovariates(), output, num_trees, n);
  }

  /*!
   * \brief Obtain a 0-based leaf index for every tree in an ensemble and for each 
   *        observation in a ForestDataset. Internally, trees are stored as essentially 
   *        vectors of node information, and the leaves_ vector gives us node IDs for every 
   *        leaf in the tree. Here, we would like to know, for every observation in a dataset, 
   *        which leaf number it is mapped to. Since the leaf numbers themselves 
   *        do not carry any information, we renumber them from 0 to `leaves_.size()-1`. 
   *        We compute this at the tree-level and coordinate this computation at the 
   *        ensemble level.
   *
   *        Note: this assumes the creation of a vector of column indices of size 
   *        `dataset.NumObservations()` x `ensemble.NumTrees()`
   * \param covariates Matrix of covariates
   * \param output Vector of length num_trees*n which stores the leaf node prediction
   * \param num_trees Number of trees in an ensemble
   * \param n Size of dataset
   */
  void PredictLeafIndicesInplace(Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>& covariates, std::vector<int32_t>& output, int num_trees, data_size_t n) {
    CHECK_GE(output.size(), num_trees*n);
    int offset = 0;
    int max_leaf = 0;
    for (int j = 0; j < num_trees; j++) {
      auto &tree = *trees_[j];
      int num_leaves = tree.NumLeaves();
      tree.PredictLeafIndexInplace(covariates, output, offset, max_leaf);
      offset += n;
      max_leaf += num_leaves;
    }
  }

  /*!
   * \brief Obtain a 0-based leaf index for every tree in an ensemble and for each 
   *        observation in a ForestDataset. Internally, trees are stored as essentially 
   *        vectors of node information, and the leaves_ vector gives us node IDs for every 
   *        leaf in the tree. Here, we would like to know, for every observation in a dataset, 
   *        which leaf number it is mapped to. Since the leaf numbers themselves 
   *        do not carry any information, we renumber them from 0 to `leaves_.size()-1`. 
   *        We compute this at the tree-level and coordinate this computation at the 
   *        ensemble level.
   *
   *        Note: this assumes the creation of a matrix of column indices with `num_trees*n` rows
   *        and as many columns as forests that were requested from R / Python
   * \param covariates Matrix of covariates
   * \param output Matrix with num_trees*n rows and as many columns as forests that were requested from R / Python
   * \param column_ind Index of column in `output` into which the result should be unpacked
   * \param num_trees Number of trees in an ensemble
   * \param n Size of dataset
   */
  void PredictLeafIndicesInplace(Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>& covariates, 
                                 Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>& output, 
                                 int column_ind, int num_trees, data_size_t n) {
    CHECK_GE(output.size(), num_trees*n);
    int offset = 0;
    int max_leaf = 0;
    for (int j = 0; j < num_trees; j++) {
      auto &tree = *trees_[j];
      int num_leaves = tree.NumLeaves();
      tree.PredictLeafIndexInplace(covariates, output, column_ind, offset, max_leaf);
      offset += n;
      max_leaf += num_leaves;
    }
  }

  /*!
   * \brief Obtain a 0-based leaf index for every tree in an ensemble and for each 
   *        observation in a ForestDataset. Internally, trees are stored as essentially 
   *        vectors of node information, and the leaves_ vector gives us node IDs for every 
   *        leaf in the tree. Here, we would like to know, for every observation in a dataset, 
   *        which leaf number it is mapped to. Since the leaf numbers themselves 
   *        do not carry any information, we renumber them from 0 to `leaves_.size()-1`. 
   *        We compute this at the tree-level and coordinate this computation at the 
   *        ensemble level.
   *
   *        Note: this assumes the creation of a vector of column indices of size 
   *        `dataset.NumObservations()` x `ensemble.NumTrees()`
   * \param ForestDataset Dataset with which to predict leaf indices from the tree
   * \param output Vector of length num_trees*n which stores the leaf node prediction
   * \param num_trees Number of trees in an ensemble
   * \param n Size of dataset
   */
  void PredictLeafIndicesInplace(Eigen::MatrixXd& covariates, std::vector<int32_t>& output, int num_trees, data_size_t n) {
    CHECK_GE(output.size(), num_trees*n);
    int offset = 0;
    int max_leaf = 0;
    for (int j = 0; j < num_trees; j++) {
      auto &tree = *trees_[j];
      int num_leaves = tree.NumLeaves();
      tree.PredictLeafIndexInplace(covariates, output, offset, max_leaf);
      offset += n;
      max_leaf += num_leaves;
    }
  }

  /*!
   * \brief Same as `PredictLeafIndicesInplace` but assumes responsibility for allocating and returning output vector.
   * \param ForestDataset Dataset with which to predict leaf indices from the tree
   */
  std::vector<int32_t> PredictLeafIndices(ForestDataset* dataset) {
    int num_trees = num_trees_;
    data_size_t n = dataset->NumObservations();
    std::vector<int32_t> output(n*num_trees);
    PredictLeafIndicesInplace(dataset, output, num_trees, n);
    return output;
  }

  /*! \brief Save to JSON */
  json to_json() {
    json result_obj;
    result_obj.emplace("num_trees", this->num_trees_);
    result_obj.emplace("output_dimension", this->output_dimension_);
    result_obj.emplace("is_leaf_constant", this->is_leaf_constant_);
    result_obj.emplace("is_exponentiated", this->is_exponentiated_);

    std::string tree_label;
    for (int i = 0; i < trees_.size(); i++) {
      tree_label = "tree_" + std::to_string(i);
      result_obj.emplace(tree_label, trees_[i]->to_json());
    }
    
    return result_obj;
  }
  
  /*! \brief Load from JSON */
  void from_json(const json& ensemble_json) {
    this->num_trees_ = ensemble_json.at("num_trees");
    this->output_dimension_ = ensemble_json.at("output_dimension");
    this->is_leaf_constant_ = ensemble_json.at("is_leaf_constant");
    this->is_exponentiated_ = ensemble_json.at("is_exponentiated");

    std::string tree_label;
    trees_.clear();
    trees_.resize(this->num_trees_);
    for (int i = 0; i < this->num_trees_; i++) {
      tree_label = "tree_" + std::to_string(i);
      trees_[i] = std::make_unique<Tree>();
      trees_[i]->from_json(ensemble_json.at(tree_label));
    }
  }

 private:
  std::vector<std::unique_ptr<Tree>> trees_;
  int num_trees_;
  int output_dimension_;
  bool is_leaf_constant_;
  bool is_exponentiated_;
};

/*! \} */ // end of forest_group

} // namespace StochTree

#endif // STOCHTREE_ENSEMBLE_H_
