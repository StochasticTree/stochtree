/*!
 * Copyright (c) 2023 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 * 
 * Inspired by the design of the Learner, GBTreeModel, and GBTree classes in xgboost, 
 * released under the Apache license with the following copyright:
 * 
 * Copyright 2015-2023 by XGBoost Contributors
 */
#ifndef STOCHTREE_ENSEMBLE_H_
#define STOCHTREE_ENSEMBLE_H_

#include <stochtree/config.h>
#include <stochtree/data.h>
#include <stochtree/tree.h>

#include <algorithm>
#include <deque>
#include <random>
#include <unordered_map>

namespace StochTree {

class TreeEnsemble {
 public:
  TreeEnsemble(int output_dimension = 1) {
    // Set class config
    config_ = Config();
    trees_ = std::vector<std::unique_ptr<Tree>>(config_.num_trees);
    for (int i = 0; i < config_.num_trees; i++) {
      trees_[i].reset(new Tree());
      trees_[i]->Init(output_dimension);
    }
  }
  TreeEnsemble(const Config& config, int output_dimension = 1) {
    // Set class config
    config_ = config;
    trees_ = std::vector<std::unique_ptr<Tree>>(config_.num_trees);
    for (int i = 0; i < config_.num_trees; i++) {
      trees_[i].reset(new Tree());
      trees_[i]->Init(output_dimension);
    }
  }
  TreeEnsemble(int num_trees, int output_dimension = 1) {
    // Set class config
    config_ = Config();
    trees_ = std::vector<std::unique_ptr<Tree>>(num_trees);
    for (int i = 0; i < num_trees; i++) {
      trees_[i].reset(new Tree());
      trees_[i]->Init(output_dimension);
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
    trees_[i]->Init(1);
  }

  inline void CopyTree(int i, Tree* tree) {
    return trees_[i].reset(tree->Clone());
  }

  inline void CloneFromExistingTree(int i, Tree* tree) {
    return trees_[i]->CloneFromTree(tree);
  }

  inline void PredictInplace(Dataset* dataset, std::vector<double> &output, data_size_t offset = 0) {
    PredictInplace(dataset, output, 0, trees_.size(), offset);
  }

  inline void PredictInplace(Dataset* dataset, std::vector<double> &output, 
                             int tree_begin, int tree_end, data_size_t offset = 0) {
    std::vector<double> pred;
    data_size_t n = dataset->NumObservations();
    std::int32_t output_dim = trees_[0]->OutputDimension();
    data_size_t total_output_size = output_dim*n;
    if (output.size() < total_output_size + offset) {
      Log::Fatal("Mismatched size of prediction vector and training data");
    }
    for (data_size_t i = 0; i < n; i++) {
      pred = std::vector<double>(output_dim, 0.0);
      for (size_t j = tree_begin; j < tree_end; j++) {
        auto &tree = *trees_[j];
        std::int32_t nidx = EvaluateTree(tree, dataset, i);
        for (int32_t k = 0; k < output_dim; k++) {
          pred[k] += tree.LeafValue(nidx, k);
        }
      }
      for (int32_t k = 0; k < output_dim; k++) {
        output[i*output_dim + k + offset] = pred[k];
      }
    }
  }

  inline std::vector<double> Predict(Dataset* dataset) {
    return Predict(dataset, 0, trees_.size());
  }
  
  inline std::vector<double> Predict(Dataset* dataset, int tree_begin, int tree_end) {
    std::vector<double> pred;
    data_size_t n = dataset->NumObservations();
    std::int32_t output_dim = trees_[0]->OutputDimension();
    data_size_t total_output_size = output_dim*n;
    std::vector<double> output(total_output_size);
    // Predict the outcome for each observation in data
    for (data_size_t i = 0; i < dataset->NumObservations(); i++) {
      pred = std::vector<double>(output_dim, 0.0);
      for (size_t j = tree_begin; j < tree_end; ++j) {
        auto &tree = *trees_[j];
        std::int32_t nidx = EvaluateTree(tree, dataset, i);
        for (int32_t k = 0; k < output_dim; k++) {
          pred[k] += tree.LeafValue(nidx, k);
        }
      }
      for (int32_t k = 0; k < output_dim; k++) {
        output[i*output_dim + k] = pred[k];
      }
    }
    return output;
  }

  inline int32_t NumTrees() {
    return trees_.size();
  }

 private:
  std::vector<std::unique_ptr<Tree>> trees_;
  Config config_;
};

} // namespace StochTree

#endif // STOCHTREE_ENSEMBLE_H_
