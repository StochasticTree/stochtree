/*!
 * Copyright (c) 2023 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 * 
 * Inspired by the design of the Learner, GBTreeModel, and GBTree classes in xgboost, 
 * released under the Apache license with the following copyright:
 * 
 * Copyright 2015-2023 by XGBoost Contributors
 */
#ifndef STOCHTREE_MODEL_DRAW_H_
#define STOCHTREE_MODEL_DRAW_H_

#include <stochtree/ensemble.h>
#include <stochtree/tree.h>

#include <algorithm>
#include <deque>
#include <random>
#include <unordered_map>

namespace StochTree {

class ModelDraw {
 public:
  ModelDraw(int num_trees, int output_dimension = 1, bool is_leaf_constant = true) {
    tree_ensemble_.reset(new TreeEnsemble(num_trees, output_dimension, is_leaf_constant));
  }
  ~ModelDraw() {}
  
  void SetGlobalParameters(double param_value, std::string param_name) {
    if (param_name == std::string("ybar_offset")) {
      ybar_offset_ = param_value;
    } else if (param_name == std::string("sd_scale")) {
      sd_scale_ = param_value;
    }
  }

  double GetGlobalParameter(std::string param_name) {
    if (param_name == std::string("ybar_offset")) {
      return ybar_offset_;
    } else if (param_name == std::string("sd_scale")) {
      return sd_scale_;
    }
  }

  TreeEnsemble* GetEnsemble() {
    return tree_ensemble_.get();
  }

  inline int32_t NumLeaves() {
    return tree_ensemble_->NumLeaves();
  }

  inline double SumLeafSquared() {
    return tree_ensemble_->SumLeafSquared();
  }

  void PredictInplace(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, std::vector<double> &output, data_size_t offset = 0) {
    PredictInplace(covariates, basis, output, 0, tree_ensemble_->NumTrees(), offset);
  }

  void PredictInplace(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, std::vector<double> &output, int tree_begin, int tree_end, data_size_t offset = 0) {
    tree_ensemble_->PredictInplace(covariates, basis, output, tree_begin, tree_end, offset);
    data_size_t n = covariates.rows();
    for (int i = 0; i < n; i++) {
      output[offset + i] = ybar_offset_ + sd_scale_ * output[offset + i];
    }
  }

  void PredictInplace(Eigen::MatrixXd& covariates, std::vector<double> &output, data_size_t offset = 0) {
    PredictInplace(covariates, output, 0, tree_ensemble_->NumTrees(), offset);
  }

  void PredictInplace(Eigen::MatrixXd& covariates, std::vector<double> &output, int tree_begin, int tree_end, data_size_t offset = 0) {
    tree_ensemble_->PredictInplace(covariates, output, tree_begin, tree_end, offset);
    data_size_t n = covariates.rows();
    for (int i = 0; i < n; i++) {
      output[offset + i] = ybar_offset_ + sd_scale_ * output[offset + i];
    }
  }
 
 private:
  std::unique_ptr<TreeEnsemble> tree_ensemble_;
  double ybar_offset_;
  double sd_scale_;
};

} // namespace StochTree

#endif // STOCHTREE_MODEL_DRAW_H_