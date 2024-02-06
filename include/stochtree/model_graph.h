/*!
 * Copyright (c) 2023 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_MODEL_GRAPH_H_
#define STOCHTREE_MODEL_GRAPH_H_

#include <stochtree/ensemble.h>
#include <stochtree/parameter.h>

#include <map>
#include <memory>
#include <vector>

namespace StochTree {

class ModelGraph {
 public:
  ModelGraph() {}
  ModelGraph(ModelGraph& model_graph);
  ModelGraph& operator=(ModelGraph& model_graph);
  ~ModelGraph() {}
  
  void AddForest(int num_trees, int output_dimension, bool is_leaf_constant, std::string forest_name) {
    int i = ensembles_.size();
    // ensembles_.emplace_back(num_trees, output_dimension, is_leaf_constant);
    ensembles_.push_back(std::make_unique<TreeEnsemble>(num_trees, output_dimension, is_leaf_constant));
    if (is_leaf_constant) basis_dim_.push_back(0);
    else basis_dim_.push_back(output_dimension);
    forest_map_.insert({forest_name, i});
  }
  
  void AddScalarParameter(double value, std::string parameter_name) {
    int i = parameters_.size();
    parameters_.push_back(ScalarParameter(value));
    parameter_map_.insert({parameter_name, i});
  }
  
  void AddVectorParameter(Eigen::VectorXd& vector, std::string parameter_name) {
    int i = parameters_.size();
    parameters_.push_back(VectorParameter(vector));
    parameter_map_.insert({parameter_name, i});
  }
  
  void AddMatrixParameter(Eigen::MatrixXd& matrix, std::string parameter_name) {
    int i = parameters_.size();
    parameters_.push_back(MatrixParameter(matrix));
    parameter_map_.insert({parameter_name, i});
  }

  void PredictForest(std::string forest_name, Eigen::MatrixXd& covariates, std::vector<double>& output) {
    CHECK_EQ(forest_map_.count(forest_name), 1);
    int forest_num = forest_map_[forest_name];
    CHECK_LT(forest_num, ensembles_.size());
    
    if (basis_dim_[forest_num] > 0) {
      Log::Fatal("Forest %s requires a basis matrix for prediction", forest_name.c_str());
    }

    ensembles_[forest_num]->PredictInplace(covariates, output);
  }

  void PredictForest(std::string forest_name, Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, std::vector<double>& output) {
    CHECK_EQ(forest_map_.count(forest_name), 1);
    int forest_num = forest_map_[forest_name];
    CHECK_LT(forest_num, ensembles_.size());
    
    if (basis_dim_[forest_num] <= 0) {
      Log::Info("Forest %s does not requires a basis matrix, the supplied matrix will be ignored", forest_name.c_str());
      ensembles_[forest_num]->PredictInplace(covariates, output);
    } else {
      ensembles_[forest_num]->PredictInplace(covariates, basis, output);
    }
  }

 private:
  // Forest trackers
  std::vector<std::unique_ptr<TreeEnsemble>> ensembles_;
  std::vector<int> basis_dim_;
  std::map<std::string, int> forest_map_;

  // Model parameters
  std::vector<ModelParameter> parameters_;
  std::map<std::string, int> parameter_map_;
};

} // namespace StochTree

#endif // STOCHTREE_MODEL_GRAPH_H_
