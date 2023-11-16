/*!
 * Copyright (c) 2023 by randtree authors. 
 * 
 * Inspired by the C API of both lightgbm and xgboost, which carry the 
 * following respective copyrights:
 * 
 * LightGBM
 * ========
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 * 
 * xgboost
 * =======
 * Copyright 2015~2023 by XGBoost Contributors
 */

#include <stochtree/interface.h>
#include <stochtree/ensemble.h>
#include <stochtree/export.h>
#include <stochtree/log.h>
#include <stochtree/meta.h>
#include <stochtree/model.h>
#include <stochtree/model_draw.h>
#include <stochtree/train_data.h>
#include <stochtree/tree.h>

#include <iostream>
#include <memory>
#include <set>
#include <string>

namespace StochTree {

StochTreeInterface::StochTreeInterface() {
  config_ = Config();
  model_draws_ = std::vector<std::unique_ptr<ModelDraw>>(config_.num_samples);
}

StochTreeInterface::StochTreeInterface(const Config& config) {
  config_ = config;
  model_draws_ = std::vector<std::unique_ptr<ModelDraw>>(config_.num_samples);
}

StochTreeInterface::~StochTreeInterface() {}

void StochTreeInterface::LoadTrainDataFromMemory(double* matrix_data, int num_col, data_size_t num_row, bool is_row_major) {
  TrainDataLoader dataloader(config_, config_.num_class, nullptr);
  train_data_.reset(dataloader.ConstructFromMatrix(matrix_data, num_col, num_row, is_row_major));
}

void StochTreeInterface::LoadPredictionDataFromMemory(double* matrix_data, int num_col, data_size_t num_row, bool is_row_major) {
  TrainDataLoader dataloader(config_, config_.num_class, nullptr);
  prediction_data_.reset(dataloader.ConstructFromMatrix(matrix_data, num_col, num_row, is_row_major));
}

void StochTreeInterface::LoadPredictionDataFromMemory(double* matrix_data, int num_col, data_size_t num_row, bool is_row_major, const Config config) {
  TrainDataLoader dataloader(config, config.num_class, nullptr);
  prediction_data_.reset(dataloader.ConstructFromMatrix(matrix_data, num_col, num_row, is_row_major));
}

void StochTreeInterface::LoadTrainDataFromFile() {
  const char* train_filename;
  if (config_.data.size() > 0){
    train_filename = config_.data.c_str();
    Log::Info("Loading train file: %s", config_.data.c_str());
  } else {
    train_filename = nullptr;
    Log::Fatal("No training data filename provided to config");
  }
  TrainDataLoader dataloader(config_, config_.num_class, train_filename);
  train_data_.reset(dataloader.LoadFromFile(train_filename));
}

void StochTreeInterface::LoadPredictionDataFromFile() {
  const char* predict_filename;
  if (config_.prediction_data.size() > 0){
    predict_filename = config_.prediction_data.c_str();
    Log::Info("Loading train file: %s", config_.prediction_data.c_str());
  } else {
    predict_filename = nullptr;
    Log::Fatal("No prediction data filename provided to config");
  }
  TrainDataLoader dataloader(config_, config_.num_class, predict_filename);
  prediction_data_.reset(dataloader.LoadFromFile(predict_filename));
}

void StochTreeInterface::SampleModel() {
  // Reset the model pointer to the correct model subtype
  if ((config_.task == TaskType::kSupervisedLearning) &&
      (config_.outcome_type == OutcomeType::kContinuous) &&
      (config_.method_type == MethodType::kXBART)) {
    model_.reset(new XBARTGaussianRegressionModel(config_));
    SampleXBARTGaussianRegression();
  } else if ((config_.task == TaskType::kSupervisedLearning) &&
      (config_.outcome_type == OutcomeType::kContinuous) &&
      (config_.method_type == MethodType::kBART)) {
    model_.reset(new BARTGaussianRegressionModel(config_));
    SampleBARTGaussianRegression();
  } else {
    Log::Fatal("Only continuous gaussian XBART or BART is currently implemented");
  }
}

std::vector<double> StochTreeInterface::PredictSamples() {
  if (prediction_data_ == nullptr) {
    Log::Fatal("No prediction dataset available!");
  }
  data_size_t n = prediction_data_->num_data();
  data_size_t offset = 0;
  int num_samples = config_.num_samples;
  std::vector<double> result(n*num_samples);
  for (int j = 0; j < num_samples; j++) {
    if (model_draws_[j]->GetEnsemble() == nullptr) {
      Log::Fatal("Sample %d has not drawn a tree ensemble");
    }
    // Store in column-major format and handle unpacking into proper format at the R / Python layers
    model_draws_[j]->PredictInplace(prediction_data_.get(), result, offset);
    offset += n;
  }
  return result;
}

void StochTreeInterface::SampleXBARTGaussianRegression() {
  // Run the sampler
  bool file_saved;
  std::string model_draw_filename;

  // Data structure to track in-sample predictions of each sweep of the model
  data_size_t n = train_data_->num_data();
  std::vector<std::vector<double>> insample_sweep_predictions;
  insample_sweep_predictions.resize(config_.num_samples + config_.num_burnin);
  for (int i = 0; i < config_.num_samples + config_.num_burnin; i++) {
    insample_sweep_predictions[i].resize(n);
  }

  // Initialize all of the global parameters outside of the loop
  std::set<std::string> param_list;
  model_->InitializeGlobalParameters(train_data_.get());

  // Compute the mean outcome for the model
  double outcome_sum = 0;
  for (data_size_t i = 0; i < train_data_->num_data(); i++){
    outcome_sum += train_data_->get_residual_value(i);
  }
  double mean_outcome = outcome_sum / train_data_->num_data();

  // Initialize the vector of vectors of leaf indices for each tree
  tree_observation_indices_.resize(config_.num_trees);
  for (int j = 0; j < config_.num_trees; j++) {
    tree_observation_indices_[j].resize(n);
  }

  int model_iter = 0;
  int prev_model_iter = 0;
  for (int i = 0; i < config_.num_samples + config_.num_burnin; i++) {
    // The way we handle "burn-in" samples is to write them to the first 
    // element of the model draw vector until we begin retaining samples.
    // Thus, the two conditions in which we reset an entry in the model 
    // draw vector are:
    //   1. The very first iteration of the sampler (i = 0)
    //   2. The model_iter variable tracking retained samples has advanced past 0
    if ((i == 0) || (model_iter > prev_model_iter)) {
      model_draws_[model_iter].reset(new XBARTGaussianRegressionModelDraw(config_));
      param_list.clear();
      param_list.insert("ybar_offset");
      model_draws_[model_iter]->SetGlobalParameters(model_.get(), param_list);
      param_list.clear();
      param_list.insert("sd_scale");
      model_draws_[model_iter]->SetGlobalParameters(model_.get(), param_list);
      param_list.clear();
    }

    if (i == 0) {
      // Initialize the ensemble by setting all trees to a root node predicting mean(y) / num_trees
      for (int j = 0; j < config_.num_trees; j++) {
        Tree* tree = (model_draws_[model_iter]->GetEnsemble())->GetTree(j);
        // (*tree)[0].SetLeaf(mean_outcome / config_.num_trees);
        tree->SetLeaf(0, mean_outcome / config_.num_trees);
        for (data_size_t k = 0; k < n; k++) {
          tree_observation_indices_[j][k] = 0;
        }
      }

      // Subtract the predictions of the (constant) trees from the outcome to obtain initial residuals
      // train_data_->ResidualReset();
      for (int j = 0; j < config_.num_trees; j++) {
        Tree* tree = (model_draws_[model_iter]->GetEnsemble())->GetTree(j);
        train_data_->ResidualSubtract(tree->PredictFromNodes(tree_observation_indices_[j]));
      }

      // Sample sigma^2
      param_list.clear();
      param_list.insert("sigma_sq");
      model_->SampleGlobalParameters(train_data_.get(), (model_draws_[model_iter]->GetEnsemble()), param_list);
      param_list.clear();
    }

    // Sample the ensemble
    for (int j = 0; j < config_.num_trees; j++) {
      // Add the predictions from tree j in the previous sweep back to the residual
      // NOTE: in the first sweep, we treat each constant (ybar / num_trees) root tree 
      // as the result of the "previous sweep" which is why we use a special prev_model_iter
      // variable to track this
      // 
      // Similarly, we do not "store" any of the burnin draws, we just continue to overwrite 
      // draws in the first sweep, so we don't begin incrementing model_iter at an offset of 
      // 1 from prev_model_iter until burn-in is complete
      
      // Retrieve pointer to tree j from the previous draw of the model
      Tree* tree = (model_draws_[prev_model_iter]->GetEnsemble())->GetTree(j);

      // Add its prediction back to the residual to obtain a "partial" residual for fitting tree j
      train_data_->ResidualAdd(tree->PredictFromNodes(tree_observation_indices_[j]));

      // Reset training data so that features are pre-sorted based on the entire dataset
      train_data_->ResetToRaw();
      train_data_->ResetCategoricalMetadata();
      
      // Reset tree j to a constant root node
      (model_draws_[model_iter]->GetEnsemble())->ResetInitTree(j);

      // Reset the observation indices to point to node 0
      for (data_size_t k = 0; k < n; k++) {
        tree_observation_indices_[j][k] = 0;
      }
      
      // Retrieve pointer to the newly-reallocated tree j
      tree = (model_draws_[model_iter]->GetEnsemble())->GetTree(j);
      
      // Sample tree structure recursively using the "grow from root" algorithm
      model_->SampleTree(train_data_.get(), tree, tree_observation_indices_, j);

      // Sample leaf node parameters
      model_->SampleLeafParameters(train_data_.get(), tree);
      
      // Subtract tree j's predictions back out of the residual
      train_data_->ResidualSubtract(tree->PredictFromNodes(tree_observation_indices_[j]));
      
      // Sample sigma^2
      param_list.insert("sigma_sq");
      model_->SampleGlobalParameters(train_data_.get(), (model_draws_[model_iter]->GetEnsemble()), param_list);
      if (j == (config_.num_trees - 1)) {
        // Store the value of sigma^2 at the last sweep of the model
        model_draws_[model_iter]->SetGlobalParameters(model_.get(), param_list);
      }
      param_list.clear();
    }

    // Sample tau (in between "sweeps" of the ensemble)
    param_list.insert("tau");
    model_->SampleGlobalParameters(train_data_.get(), (model_draws_[model_iter]->GetEnsemble()), param_list);
    model_draws_[model_iter]->SetGlobalParameters(model_.get(), param_list);
    param_list.clear();
    
    // Store model draws in a text file if user specifies
    if (config_.save_model_draws) {
      model_draw_filename = std::string("model_" + std::to_string(i) + ".txt");
      model_draws_[model_iter]->SaveModelDrawToFile(model_draw_filename.c_str());
    }
    
    // Determine whether to advance the model_iter variable
    if (i >= config_.num_burnin) {
      prev_model_iter = model_iter;
      model_iter += 1;
    }
  }
}

void StochTreeInterface::SampleBARTGaussianRegression() {
  // Run the sampler
  bool file_saved;
  std::string model_draw_filename;

  // Data structure to track in-sample predictions for each draw of the model
  data_size_t n = train_data_->num_data();
  std::vector<std::vector<double>> insample_sweep_predictions;
  insample_sweep_predictions.resize(config_.num_samples + config_.num_burnin);
  for (int i = 0; i < config_.num_samples + config_.num_burnin; i++) {
    insample_sweep_predictions[i].resize(n);
  }

  // Initialize all of the global parameters outside of the loop
  std::set<std::string> param_list;
  model_->InitializeGlobalParameters(train_data_.get());

  // Compute the mean outcome for the model
  double outcome_sum = 0;
  for (data_size_t i = 0; i < train_data_->num_data(); i++){
    outcome_sum += train_data_->get_residual_value(i);
  }
  double mean_outcome = outcome_sum / train_data_->num_data();

  // Initialize the vector of vectors of leaf indices for each tree
  tree_observation_indices_.resize(config_.num_trees);
  for (int j = 0; j < config_.num_trees; j++) {
    tree_observation_indices_[j].resize(n);
  }

  // Initialize a vector of NodeSampleTracker
  std::vector<std::unique_ptr<NodeSampleTracker>> node_trackers(config_.num_trees);
  for (int i = 0; i < config_.num_trees; i++) {
    node_trackers[i] = std::make_unique<NodeSampleTracker>(n);
  }

  int model_iter = 0;
  int prev_model_iter = 0;
  for (int i = 0; i < config_.num_samples + config_.num_burnin; i++) {
    // The way we handle "burn-in" samples is to write them to the first 
    // element of the model draw vector until we begin retaining samples.
    // Thus, the two conditions in which we reset an entry in the model 
    // draw vector are:
    //   1. The very first iteration of the sampler (i = 0)
    //   2. The model_iter variable tracking retained samples has advanced past 0
    if ((i == 0) || (model_iter > prev_model_iter)) {
      model_draws_[model_iter].reset(new BARTGaussianRegressionModelDraw(config_));
      param_list.clear();
      param_list.insert("ybar_offset");
      model_draws_[model_iter]->SetGlobalParameters(model_.get(), param_list);
      param_list.clear();
      param_list.insert("sd_scale");
      model_draws_[model_iter]->SetGlobalParameters(model_.get(), param_list);
      param_list.clear();
    }

    if (i == 0) {
      // Initialize the ensemble by setting all trees to a root node predicting mean(y) / num_trees
      for (int j = 0; j < config_.num_trees; j++) {
        Tree* tree = (model_draws_[model_iter]->GetEnsemble())->GetTree(j);
        // (*tree)[0].SetLeaf(mean_outcome / config_.num_trees);
        tree->SetLeaf(0, mean_outcome / config_.num_trees);
        for (data_size_t k = 0; k < n; k++) {
          tree_observation_indices_[j][k] = 0;
        }
      }

      // Subtract the predictions of the (constant) trees from the outcome to obtain initial residuals
      // train_data_->ResidualReset();
      for (int j = 0; j < config_.num_trees; j++) {
        Tree* tree = (model_draws_[model_iter]->GetEnsemble())->GetTree(j);
        train_data_->ResidualSubtract(tree->PredictFromNodes(tree_observation_indices_[j]));
      }
    }

    // Sample the ensemble
    for (int j = 0; j < config_.num_trees; j++) {
      // Add the predictions from tree j in the previous sweep back to the residual
      // NOTE: in the first sweep, we treat each constant (ybar / num_trees) root tree 
      // as the result of the "previous sweep" which is why we use a special prev_model_iter
      // variable to track this
      // 
      // Similarly, we do not "store" any of the burnin draws, we just continue to overwrite 
      // draws in the first sweep, so we don't begin incrementing model_iter at an offset of 
      // 1 from prev_model_iter until burn-in is complete
      
      // Retrieve pointer to tree j from the previous draw of the model
      Tree* tree = (model_draws_[prev_model_iter]->GetEnsemble())->GetTree(j);

      // Add its prediction back to the residual to obtain a "partial" residual for fitting tree j
      train_data_->ResidualAdd(tree->PredictFromNodes(tree_observation_indices_[j]));

      // If model_iter is different from prev_model_iter, copy tree j from prev_model_iter to model_iter
      if (model_iter > prev_model_iter) {
        // (model_draws_[model_iter]->GetEnsemble())->CopyTree(j, tree);
        (model_draws_[model_iter]->GetEnsemble())->ResetTree(j);
        (model_draws_[model_iter]->GetEnsemble())->CloneFromExistingTree(j, tree);
      }
      
      // Retrieve pointer to tree j (which might be a new tree if we copied it)
      tree = (model_draws_[model_iter]->GetEnsemble())->GetTree(j);
      
      // Conduct one MCMC step of the grow/prune process
      model_->SampleTree(train_data_.get(), tree, node_trackers[j].get(), tree_observation_indices_, j);

      // Sample leaf node parameters
      model_->SampleLeafParameters(train_data_.get(), tree);
      
      // Subtract tree j's predictions back out of the residual
      train_data_->ResidualSubtract(tree->PredictFromNodes(tree_observation_indices_[j]));
    }

    // Sample sigma^2
    param_list.insert("sigma_sq");
    model_->SampleGlobalParameters(train_data_.get(), (model_draws_[model_iter]->GetEnsemble()), param_list);
    // Store the value of sigma^2 at the last sweep of the model
    model_draws_[model_iter]->SetGlobalParameters(model_.get(), param_list);
    param_list.clear();
    
    // Store model draws in a text file if user specifies
    if (config_.save_model_draws) {
      model_draw_filename = std::string("model_" + std::to_string(i) + ".txt");
      model_draws_[model_iter]->SaveModelDrawToFile(model_draw_filename.c_str());
    }
    
    // Determine whether to advance the model_iter variable
    if (i >= config_.num_burnin) {
      prev_model_iter = model_iter;
      model_iter += 1;
    }
  }
}

void StochTreeInterface::SaveSamples() {}

} // namespace StochTree
