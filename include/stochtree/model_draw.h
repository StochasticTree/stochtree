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
  virtual void PredictInplace(TrainData* data, std::vector<double> &output, data_size_t offset = 0) {}
  virtual void PredictInplace(TrainData* data, std::vector<double> &output, 
                             int tree_begin, int tree_end, data_size_t offset = 0) {}
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
    } else if (update_params.count("tau") > 0) {
      tau_ = model->GetGlobalParameter("tau");
    } else if (update_params.count("ybar_offset") > 0) {
      ybar_offset_ = model->GetGlobalParameter("ybar_offset");
    } else if (update_params.count("sd_scale") > 0) {
      sd_scale_ = model->GetGlobalParameter("sd_scale");
    }
  }

  double GetGlobalParameter(std::string param_name) {
    if (param_name == std::string("sigma_sq")) {
      return sigma_sq_;
    } else if (param_name == std::string("tau")) {
      return tau_;
    } else if (param_name == std::string("ybar_offset")) {
      return ybar_offset_;
    } else if (param_name == std::string("sd_scale")) {
      return sd_scale_;
    }
  }

  TreeEnsemble* GetEnsemble() {
    return tree_ensemble_.get();
  }

  /*! \brief Return the type of model being sampled */
  const char* SubModelName() const { return "XBARTGaussianRegression"; }

  // void SaveModelDrawToFile(const char* filename);

  // std::string SaveModelDrawToString() const;

  void PredictInplace(TrainData* data, std::vector<double> &output, data_size_t offset = 0) {
    PredictInplace(data, output, 0, tree_ensemble_->NumTrees(), offset);
  }

  void PredictInplace(TrainData* data, std::vector<double> &output, 
                      int tree_begin, int tree_end, data_size_t offset = 0) {
    tree_ensemble_->PredictInplace(data, output, tree_begin, tree_end, offset);
    data_size_t n = data->num_data();
    for (int i = 0; i < n; i++) {
      output[offset + i] = ybar_offset_ + sd_scale_ * output[offset + i];
    }
  }

 private:
  std::unique_ptr<TreeEnsemble> tree_ensemble_;
  double tau_;
  double sigma_sq_;
  double ybar_offset_;
  double sd_scale_;
  Config config_;
};

class BARTGaussianRegressionModelDraw : public ModelDraw {
 public:
  BARTGaussianRegressionModelDraw() {
    config_ = Config();
    tree_ensemble_.reset(new TreeEnsemble(config_));
  }
  
  BARTGaussianRegressionModelDraw(const Config& config) {
    config_ = config;
    tree_ensemble_.reset(new TreeEnsemble(config_));
  }
  
  ~BARTGaussianRegressionModelDraw() {}

  void SetGlobalParameters(Model* model, std::set<std::string> update_params) {
    if (update_params.count("sigma_sq") > 0) {
      sigma_sq_ = model->GetGlobalParameter("sigma_sq");
    } else if (update_params.count("ybar_offset") > 0) {
      ybar_offset_ = model->GetGlobalParameter("ybar_offset");
    } else if (update_params.count("sd_scale") > 0) {
      sd_scale_ = model->GetGlobalParameter("sd_scale");
    }
  }

  double GetGlobalParameter(std::string param_name) {
    if (param_name == std::string("sigma_sq")) {
      return sigma_sq_;
    } else if (param_name == std::string("ybar_offset")) {
      return ybar_offset_;
    } else if (param_name == std::string("sd_scale")) {
      return sd_scale_;
    }
  }

  TreeEnsemble* GetEnsemble() {
    return tree_ensemble_.get();
  }

  /*! \brief Return the type of model being sampled */
  const char* SubModelName() const { return "BARTGaussianRegression"; }

  // void SaveModelDrawToFile(const char* filename);

  // std::string SaveModelDrawToString() const;

  void PredictInplace(TrainData* data, std::vector<double> &output, data_size_t offset = 0) {
    PredictInplace(data, output, 0, tree_ensemble_->NumTrees(), offset);
  }

  void PredictInplace(TrainData* data, std::vector<double> &output, 
                      int tree_begin, int tree_end, data_size_t offset = 0) {
    tree_ensemble_->PredictInplace(data, output, tree_begin, tree_end, offset);
    data_size_t n = data->num_data();
    for (int i = 0; i < n; i++) {
      output[offset + i] = ybar_offset_ + sd_scale_ * output[offset + i];
    }
  }

 private:
  std::unique_ptr<TreeEnsemble> tree_ensemble_;
  double sigma_sq_;
  double ybar_offset_;
  double sd_scale_;
  Config config_;
};

} // namespace StochTree

#endif // STOCHTREE_MODEL_DRAW_H_