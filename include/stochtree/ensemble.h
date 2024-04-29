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

#include <algorithm>
#include <deque>
#include <optional>
#include <random>
#include <unordered_map>

using json = nlohmann::json;

namespace StochTree {

class TreeEnsemble {
 public:
  TreeEnsemble(int num_trees, int output_dimension = 1, bool is_leaf_constant = true) {
    // Initialize trees in the ensemble
    trees_ = std::vector<std::unique_ptr<Tree>>(num_trees);
    for (int i = 0; i < num_trees; i++) {
      trees_[i].reset(new Tree());
      trees_[i]->Init(output_dimension);
    }
    // Store ensemble configurations
    num_trees_ = num_trees;
    output_dimension_ = output_dimension;
    is_leaf_constant_ = is_leaf_constant;
  }
  TreeEnsemble(TreeEnsemble& ensemble) {
    // Unpack ensemble configurations
    num_trees_ = ensemble.num_trees_;
    output_dimension_ = ensemble.output_dimension_;
    is_leaf_constant_ = ensemble.is_leaf_constant_;
    // Initialize trees in the ensemble
    trees_ = std::vector<std::unique_ptr<Tree>>(num_trees_);
    for (int i = 0; i < num_trees_; i++) {
      trees_[i].reset(new Tree());
      // trees_[i]->Init(output_dimension);
    }
    // Clone trees in the ensemble
    // trees_ = std::vector<std::unique_ptr<Tree>>(num_trees_);
    for (int j = 0; j < num_trees_; j++) {
      Tree* tree = ensemble.GetTree(j);
      this->CloneFromExistingTree(j, tree);
    }
  }
  ~TreeEnsemble() {}

  inline Tree* GetTree(int i) {
    return trees_[i].get();
  }

  inline void ResetTree(int i) {
    trees_[i].reset(new Tree());
  }

  inline void ResetInitTree(int i) {
    trees_[i].reset(new Tree());
    trees_[i]->Init(output_dimension_);
  }

  inline void CloneFromExistingTree(int i, Tree* tree) {
    return trees_[i]->CloneFromTree(tree);
  }

  inline void PredictInplace(ForestDataset& dataset, std::vector<double> &output, data_size_t offset = 0) {
    PredictInplace(dataset, output, 0, trees_.size(), offset);
  }

  inline void PredictInplace(ForestDataset& dataset, std::vector<double> &output, 
                             int tree_begin, int tree_end, data_size_t offset = 0) {
    if (dataset.HasBasis()) {
      CHECK(!is_leaf_constant_);
      PredictInplace(dataset.GetCovariates(), dataset.GetBasis(), output, tree_begin, tree_end, offset);
    } else {
      CHECK(is_leaf_constant_);
      PredictInplace(dataset.GetCovariates(), output, tree_begin, tree_end, offset);
    }
  }

  inline void PredictInplace(MatrixMap& covariates, MatrixMap& basis, std::vector<double> &output, data_size_t offset = 0) {
    PredictInplace(covariates, basis, output, 0, trees_.size(), offset);
  }

  inline void PredictInplace(MatrixMap& covariates, MatrixMap& basis, std::vector<double> &output, 
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
      output[i + offset] = pred;
    }
  }

  inline void PredictInplace(MatrixMap& covariates, std::vector<double> &output, data_size_t offset = 0) {
    PredictInplace(covariates, output, 0, trees_.size(), offset);
  }

  inline void PredictInplace(MatrixMap& covariates, std::vector<double> &output, int tree_begin, int tree_end, data_size_t offset = 0) {
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
      output[i + offset] = pred;
    }
  }

  inline void PredictRawInplace(ForestDataset& dataset, std::vector<double> &output, data_size_t offset = 0) {
    PredictRawInplace(dataset, output, 0, trees_.size(), offset);
  }

  inline void PredictRawInplace(ForestDataset& dataset, std::vector<double> &output, 
                             int tree_begin, int tree_end, data_size_t offset = 0) {
    double pred;
    MatrixMap covariates = dataset.GetCovariates();
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

  /*! \brief Save to JSON */
  json to_json() {
    json result_obj;
    result_obj.emplace("num_trees", this->num_trees_);
    result_obj.emplace("output_dimension", this->output_dimension_);
    result_obj.emplace("is_leaf_constant", this->is_leaf_constant_);

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
};

} // namespace StochTree

#endif // STOCHTREE_ENSEMBLE_H_
