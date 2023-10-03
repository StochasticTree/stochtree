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
    }
  }
  TreeEnsemble(const Config& config) {
    // Set class config
    config_ = config;
    trees_ = std::vector<std::unique_ptr<Tree>>(config_.num_trees);
    for (int i = 0; i < config_.num_trees; i++) {
      trees_[i].reset(new Tree());
    }
  }
  TreeEnsemble(int num_trees) {
    // Set class config
    config_ = Config();
    trees_ = std::vector<std::unique_ptr<Tree>>(num_trees);
    for (int i = 0; i < num_trees; i++) {
      trees_[i].reset(new Tree());
    }
  }
  ~TreeEnsemble() {}

  inline Tree* GetTree(int i) {
    return trees_[i].get();
  }

  inline void ResetTree(int i) {
    trees_[i].reset(new Tree());
  }

  inline void CopyTree(int i, Tree* tree) {
    return trees_[i].reset(new Tree(*tree));
  }

  inline void PredictInplace(TrainData* data, std::vector<double> &output, data_size_t offset = 0) {
    PredictInplace(data, output, 0, trees_.size(), offset);
  }

  inline void PredictInplace(TrainData* data, std::vector<double> &output, 
                             int tree_begin, int tree_end, data_size_t offset = 0) {
    double pred;
    // Predict from 
    if (output.size() < data->num_data() + offset) {
      Log::Fatal("Mismatched size of prediction vector and training data");
    }
    for (data_size_t i = 0; i < data->num_data(); i++) {
      pred = 0.0;
      for (size_t j = tree_begin; j < tree_end; j++) {
        auto const &tree = *trees_[j];
        node_t nidx = 0;
        while (!tree[nidx].IsLeaf()) {
          int col = tree[nidx].SplitIndex();
          auto fvalue = data->get_feature_value(i, col);
          bool proceed_left = fvalue <= tree[nidx].SplitCond();
          if (proceed_left) {
            nidx = tree[nidx].LeftChild();
          } else {
            nidx = tree[nidx].RightChild();
          }
        }
        pred += tree[nidx].LeafValue();
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
        auto const &tree = *trees_[j];
        node_t nidx = 0;
        while (!tree[nidx].IsLeaf()) {
          int col = tree[nidx].SplitIndex();
          auto fvalue = data->get_feature_value(i, col);
          bool proceed_left = fvalue <= tree[nidx].SplitCond();
          if (proceed_left) {
            nidx = tree[nidx].LeftChild();
          } else {
            nidx = tree[nidx].RightChild();
          }
        }
        pred += tree[nidx].LeafValue();
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
