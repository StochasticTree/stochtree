/*!
 * Copyright (c) 2023 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 * 
 * Inspired by the design of the Learner, GBTreeModel, and GBTree classes in xgboost, 
 * released under the Apache license with the following copyright:
 * 
 * Copyright 2015-2023 by XGBoost Contributors
 */
#ifndef STOCHTREE_ENSEMBLE_CONTAINER_H_
#define STOCHTREE_ENSEMBLE_CONTAINER_H_

#include <stochtree/ensemble.h>
#include <stochtree/tree.h>

#include <algorithm>
#include <deque>
#include <random>
#include <unordered_map>

namespace StochTree {

class TreeEnsembleContainer {
 public:
  TreeEnsembleContainer(int num_samples, int num_trees, int output_dimension = 1, bool is_leaf_constant = true) {
    forests_ = std::vector<std::unique_ptr<TreeEnsemble>>(num_samples);
    for (auto& forest : forests_) {
      forest.reset(new TreeEnsemble(num_trees, output_dimension, is_leaf_constant));
    }
    num_samples_ = num_samples;
  }
  ~TreeEnsembleContainer() {}

  inline void PredictInplace(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, std::vector<double> &output, data_size_t offset = 0) {
    data_size_t n = covariates.rows();
    data_size_t total_output_size = n*num_samples_;
    if (output.size() < total_output_size) {
      Log::Fatal("Mismatched size of prediction vector and training data");
    }

    data_size_t offset = 0;
    for (int i = 0; i < num_samples_; i++) {
      auto num_trees = forests_[i]->NumTrees();
      forests_[i]->PredictInplace(covariates, basis, output, 0, num_trees, offset);
      offset += n;
    }
  }

  inline void PredictInplace(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, std::vector<double> &output, 
                             int tree_begin, int tree_end) {
    data_size_t n = covariates.rows();
    data_size_t total_output_size = n*num_samples_;
    if (output.size() < total_output_size) {
      Log::Fatal("Mismatched size of prediction vector and training data");
    }

    data_size_t offset = 0;
    for (int i = 0; i < num_samples_; i++) {
      forests_[i]->PredictInplace(covariates, basis, output, tree_begin, tree_end, offset);
      offset += n;
    }
  }

  inline void PredictInplace(Eigen::MatrixXd& covariates, std::vector<double> &output, data_size_t offset = 0) {
    data_size_t n = covariates.rows();
    data_size_t total_output_size = n*num_samples_;
    if (output.size() < total_output_size) {
      Log::Fatal("Mismatched size of prediction vector and training data");
    }

    data_size_t offset = 0;
    for (int i = 0; i < num_samples_; i++) {
      auto num_trees = forests_[i]->NumTrees();
      forests_[i]->PredictInplace(covariates, output, 0, num_trees, offset);
      offset += n;
    }
  }

  inline void PredictInplace(Eigen::MatrixXd& covariates, std::vector<double> &output, int tree_begin, int tree_end) {
    data_size_t n = covariates.rows();
    data_size_t total_output_size = n*num_samples_;
    if (output.size() < total_output_size) {
      Log::Fatal("Mismatched size of prediction vector and training data");
    }

    data_size_t offset = 0;
    for (int i = 0; i < num_samples_; i++) {
      forests_[i]->PredictInplace(covariates, output, tree_begin, tree_end, offset);
      offset += n;
    }
  }

  inline int32_t NumTrees(int ensemble_num) {
    return forests_[ensemble_num]->NumTrees();
  }

  inline int32_t NumLeaves(int ensemble_num) {
    return forests_[ensemble_num]->NumLeaves();
  }

  inline double SumLeafSquared(int ensemble_num) {
    return forests_[ensemble_num]->SumLeafSquared();
  }

  inline int32_t OutputDimension(int ensemble_num) {
    return forests_[ensemble_num]->OutputDimension();
  }

  inline bool IsLeafConstant(int ensemble_num) {
    return forests_[ensemble_num]->IsLeafConstant();
  }

 private:
  std::vector<std::unique_ptr<TreeEnsemble>> forests_;
  int num_samples_;
};

} // namespace StochTree

#endif // STOCHTREE_ENSEMBLE_CONTAINER_H_
