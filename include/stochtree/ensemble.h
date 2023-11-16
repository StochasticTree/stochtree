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
#include <stochtree/predict.h>
#include <stochtree/train_data.h>
#include <stochtree/tree.h>

#include <algorithm>
#include <deque>
#include <random>
#include <unordered_map>

namespace StochTree {

class TreeEnsemble {
 public:
  TreeEnsemble() {
    // Set class config
    config_ = Config();
    trees_ = std::vector<std::unique_ptr<Tree>>(config_.num_trees);
    for (int i = 0; i < config_.num_trees; i++) {
      trees_[i].reset(new Tree());
      trees_[i]->Init(1);
    }
  }
  TreeEnsemble(const Config& config) {
    // Set class config
    config_ = config;
    trees_ = std::vector<std::unique_ptr<Tree>>(config_.num_trees);
    for (int i = 0; i < config_.num_trees; i++) {
      trees_[i].reset(new Tree());
      trees_[i]->Init(1);
    }
  }
  TreeEnsemble(int num_trees) {
    // Set class config
    config_ = Config();
    trees_ = std::vector<std::unique_ptr<Tree>>(num_trees);
    for (int i = 0; i < num_trees; i++) {
      trees_[i].reset(new Tree());
      trees_[i]->Init(1);
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

  inline void PredictInplace(TrainData* data, std::vector<double> &output, data_size_t offset = 0) {
    PredictInplace(data, output, 0, trees_.size(), offset);
  }

  inline void PredictInplace(TrainData* data, std::vector<double> &output, 
                             int tree_begin, int tree_end, data_size_t offset = 0) {
    double pred;
    if (output.size() < data->num_data() + offset) {
      Log::Fatal("Mismatched size of prediction vector and training data");
    }
    for (data_size_t i = 0; i < data->num_data(); i++) {
      pred = 0.0;
      for (size_t j = tree_begin; j < tree_end; j++) {
        auto &tree = *trees_[j];
        std::int32_t nidx = EvaluateTree(tree, data, i);
        pred += tree.LeafValue(nidx);
      }
      output[i + offset] = pred;
    }
  }

  inline std::vector<double> Predict(TrainData* data) {
    return Predict(data, 0, trees_.size());
  }
  
  inline std::vector<double> Predict(TrainData* data, int tree_begin, int tree_end) {
    std::vector<double> output(data->num_data());
    double pred;
    // Predict the outcome for each observation in data
    for (data_size_t i = 0; i < data->num_data(); i++) {
      pred = 0.0;
      for (size_t j = tree_begin; j < tree_end; ++j) {
        auto &tree = *trees_[j];
        std::int32_t nidx = EvaluateTree(tree, data, i);
        // TODO: handle vector-valued leaves
        pred += tree.LeafValue(nidx);
      }
      output[i] = pred;
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
