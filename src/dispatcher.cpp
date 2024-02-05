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
#include <stochtree/dispatcher.h>
#include <stochtree/partition_tracker.h>

#include <iostream>
#include <memory>
#include <set>
#include <string>

namespace StochTree {

MCMCDispatcher::MCMCDispatcher(int num_samples, int num_burnin, int num_trees, int random_seed) {
  num_samples_ = num_samples;
  num_burnin_ = num_burnin;
  num_trees_ = num_trees;
  model_draws_ = std::vector<std::unique_ptr<ModelDraw>>(num_samples);
  if (random_seed < 0) {
    std::random_device rd;
    std::mt19937 gen_(rd());
  } else {
    std::mt19937 gen_(random_seed);
  }
}

MCMCDispatcher::~MCMCDispatcher() {}

bool MCMCDispatcher::TrainDataConsistent(){
  if (outcome_.rows() != residual_.rows()) {
    return false;
  } 
  if (outcome_.rows() != covariates_.rows()) {
    return false;
  } 
  if (outcome_.rows() != basis_.rows()) {
    return false;
  } 
  if (residual_.rows() != covariates_.rows()) {
    return false;
  } 
  if (residual_.rows() != basis_.rows()) {
    return false;
  } 
  if (covariates_.rows() != basis_.rows()) {
    return false;
  }
  if (outcome_.rows() == 0) {
    return false;
  }
  return true;
}

bool MCMCDispatcher::PredictionDataConsistent(){
  if (prediction_covariates_.rows() != prediction_basis_.rows()) {
    return false;
  }
  if (prediction_covariates_.rows() == 0) {
    return false;
  }
  return true;
}

void MCMCDispatcher::LoadData(double* data_ptr, int num_row, int num_col, bool is_row_major, Eigen::MatrixXd& data_matrix) {
  data_matrix.resize(num_row, num_col);

  // Copy data from R / Python process memory to Eigen matrix
  double temp_value;
  for (data_size_t i = 0; i < num_row; ++i) {
    for (int j = 0; j < num_col; ++j) {
      if (is_row_major){
        // Numpy 2-d arrays are stored in "row major" order
        temp_value = static_cast<double>(*(data_ptr + static_cast<data_size_t>(num_col) * i + j));
      } else {
        // R matrices are stored in "column major" order
        temp_value = static_cast<double>(*(data_ptr + static_cast<data_size_t>(num_row) * j + i));
      }
      
      data_matrix(i, j) = temp_value;
    }
  }
}

std::vector<double> MCMCDispatcher::PredictSamples(double* covariate_data_ptr, int num_covariate, double* basis_data_ptr, int num_basis, double* rfx_basis_data_ptr, int num_rfx_basis, std::vector<int32_t>& rfx_groups, data_size_t num_row, bool is_row_major) {
  // Load the data
  LoadData(covariate_data_ptr, num_row, num_covariate, is_row_major, prediction_covariates_);
  LoadData(basis_data_ptr, num_row, num_basis, is_row_major, prediction_basis_);
  LoadData(rfx_basis_data_ptr, num_row, num_rfx_basis, is_row_major, prediction_rfx_basis_);
  CHECK(PredictionDataConsistent());
  
  // Predict outcomes using supplied data and sampled ensembles
  data_size_t n = prediction_covariates_.rows();
  data_size_t offset = 0;
  int num_samples = num_samples_;
  std::vector<double> result(n*num_samples_);
  for (int j = 0; j < num_samples_; j++) {
    if (model_draws_[j]->GetEnsemble() == nullptr) {
      Log::Fatal("Sample %d has not drawn a tree ensemble");
    }
    // Store in column-major format and handle unpacking into proper format at the R / Python layers
    model_draws_[j]->PredictInplace(prediction_covariates_, prediction_basis_, prediction_rfx_basis_, rfx_groups, result, offset);
    offset += n;
  }
  return result;
}

void MCMCDispatcher::OutcomeCenterScale(Eigen::MatrixXd& outcome, double& ybar_offset, double& sd_scale){
  double var_y = 0.0;
  double outcome_sum_squares = 0.0;
  double outcome_sum = 0.0;
  double outcome_val;
  data_size_t n = outcome.rows();
  int p = outcome.cols();
  CHECK_EQ(p, 1);
  for (data_size_t i = 0; i < n; i++){
    outcome_val = outcome(i, 0);
    outcome_sum += outcome_val;
    outcome_sum_squares += std::pow(outcome_val, 2.0);
  }
  var_y = outcome_sum_squares/n - std::pow(outcome_sum / n, 2.0);
  sd_scale = std::sqrt(var_y);
  ybar_offset = outcome_sum / n;
}

GFRDispatcher::GFRDispatcher(int num_samples, int num_burnin, int num_trees, int random_seed) {
  num_samples_ = num_samples;
  num_burnin_ = num_burnin;
  num_trees_ = num_trees;
  model_draws_ = std::vector<std::unique_ptr<ModelDraw>>(num_samples);
  if (random_seed < 0) {
    std::random_device rd;
    std::mt19937 gen_(rd());
  } else {
    std::mt19937 gen_(random_seed);
  }
}

GFRDispatcher::~GFRDispatcher() {}

bool GFRDispatcher::TrainDataConsistent(){
  if (outcome_.rows() != residual_.rows()) {
    return false;
  } 
  if (outcome_.rows() != covariates_.rows()) {
    return false;
  } 
  if (outcome_.rows() != basis_.rows()) {
    return false;
  } 
  if (residual_.rows() != covariates_.rows()) {
    return false;
  } 
  if (residual_.rows() != basis_.rows()) {
    return false;
  } 
  if (covariates_.rows() != basis_.rows()) {
    return false;
  }
  if (outcome_.rows() == 0) {
    return false;
  }
  return true;
}

bool GFRDispatcher::PredictionDataConsistent(){
  if (prediction_covariates_.rows() != prediction_basis_.rows()) {
    return false;
  }
  if (prediction_covariates_.rows() == 0) {
    return false;
  }
  return true;
}

void GFRDispatcher::LoadData(double* data_ptr, int num_row, int num_col, bool is_row_major, Eigen::MatrixXd& data_matrix) {
  data_matrix.resize(num_row, num_col);

  // Copy data from R / Python process memory to Eigen matrix
  double temp_value;
  for (data_size_t i = 0; i < num_row; ++i) {
    for (int j = 0; j < num_col; ++j) {
      if (is_row_major){
        // Numpy 2-d arrays are stored in "row major" order
        temp_value = static_cast<double>(*(data_ptr + static_cast<data_size_t>(num_col) * i + j));
      } else {
        // R matrices are stored in "column major" order
        temp_value = static_cast<double>(*(data_ptr + static_cast<data_size_t>(num_row) * j + i));
      }
      
      data_matrix(i, j) = temp_value;
    }
  }
}

std::vector<double> GFRDispatcher::PredictSamples(double* covariate_data_ptr, int num_covariate, double* basis_data_ptr, int num_basis, double* rfx_basis_data_ptr, int num_rfx_basis, std::vector<int32_t>& rfx_groups, data_size_t num_row, bool is_row_major) {
  // Load the data
  LoadData(covariate_data_ptr, num_row, num_covariate, is_row_major, prediction_covariates_);
  LoadData(basis_data_ptr, num_row, num_basis, is_row_major, prediction_basis_);
  LoadData(rfx_basis_data_ptr, num_row, num_rfx_basis, is_row_major, prediction_rfx_basis_);
  CHECK(PredictionDataConsistent());
  
  // Predict outcomes using supplied data and sampled ensembles
  data_size_t n = prediction_covariates_.rows();
  data_size_t offset = 0;
  int num_samples = num_samples_;
  std::vector<double> result(n*num_samples_);
  for (int j = 0; j < num_samples_; j++) {
    if (model_draws_[j]->GetEnsemble() == nullptr) {
      Log::Fatal("Sample %d has not drawn a tree ensemble");
    }
    // Store in column-major format and handle unpacking into proper format at the R / Python layers
    model_draws_[j]->PredictInplace(prediction_covariates_, prediction_basis_, prediction_rfx_basis_, rfx_groups, result, offset);
    offset += n;
  }
  return result;
}

void GFRDispatcher::OutcomeCenterScale(Eigen::MatrixXd& outcome, double& ybar_offset, double& sd_scale){
  double var_y = 0.0;
  double outcome_sum_squares = 0.0;
  double outcome_sum = 0.0;
  double outcome_val;
  data_size_t n = outcome.rows();
  int p = outcome.cols();
  CHECK_EQ(p, 1);
  for (data_size_t i = 0; i < n; i++){
    outcome_val = outcome(i, 0);
    outcome_sum += outcome_val;
    outcome_sum_squares += std::pow(outcome_val, 2.0);
  }
  var_y = outcome_sum_squares/n - std::pow(outcome_sum / n, 2.0);
  sd_scale = std::sqrt(var_y);
  ybar_offset = outcome_sum / n;
}

} // namespace StochTree
