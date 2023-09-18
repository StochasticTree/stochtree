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

#include <stochtree/config.h>
#include <stochtree/ensemble.h>
#include <stochtree/model.h>
#include <stochtree/train_data.h>
#include <stochtree/tree.h>

#include <algorithm>
#include <deque>
#include <random>
#include <unordered_map>

namespace StochTree {

class ModelDraw {
 public:
  ModelDraw() = default;
  ModelDraw(const Config& config) {}
  virtual ~ModelDraw() = default;
  virtual void SetGlobalParameters(Model* model, std::set<std::string> update_params) {}
  virtual TreeEnsemble* GetEnsemble() {}
  virtual void SaveModelDrawToFile(const char* filename) {}
  virtual const char* SubModelName() const {}
};

class XBARTGaussianRegressionModelDraw : public ModelDraw {
 public:
  XBARTGaussianRegressionModelDraw() {
    config_ = Config();
    tree_ensemble_.reset(new TreeEnsemble(config_));
  }
  
  XBARTGaussianRegressionModelDraw(const Config& config) {
    config_ = config;
    tree_ensemble_.reset(new TreeEnsemble(config_));
  }
  
  ~XBARTGaussianRegressionModelDraw() {}

  void SetGlobalParameters(Model* model, std::set<std::string> update_params) {
    if (update_params.count("sigma_sq") > 0) {
      sigma_sq_ = model->GetGlobalParameter("sigma_sq");
    }

    if (update_params.count("tau") > 0) {
      tau_ = model->GetGlobalParameter("tau");
    }
  }

  TreeEnsemble* GetEnsemble() {
    return tree_ensemble_.get();
  }

  /*! \brief Return the type of model being sampled */
  const char* SubModelName() const { return "XBARTGaussianRegression"; }

  void SaveModelDrawToFile(const char* filename);

  std::string SaveModelDrawToString() const;

 private:
  std::unique_ptr<TreeEnsemble> tree_ensemble_;
  double tau_;
  double sigma_sq_;
  Config config_;
};

} // namespace StochTree

#endif // STOCHTREE_MODEL_DRAW_H_