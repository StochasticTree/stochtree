/*!
 * Copyright (c) 2024 stochtree authors. All rights reserved.
 * 
 * Simple container-like interfaces for samples of common models.
 */
#ifndef STOCHTREE_CONTAINER_H_
#define STOCHTREE_CONTAINER_H_

#include <stochtree/data.h>
#include <stochtree/ensemble.h>
#include <nlohmann/json.hpp>
#include <stochtree/tree.h>

#include <algorithm>
#include <deque>
#include <fstream>
#include <optional>
#include <random>
#include <unordered_map>

namespace StochTree {

class ForestContainer {
 public:
  ForestContainer(int num_trees, int output_dimension = 1, bool is_leaf_constant = true);
  ForestContainer(int num_samples, int num_trees, int output_dimension = 1, bool is_leaf_constant = true);
  ~ForestContainer() {}

  void AddSamples(int num_samples);
  void CopyFromPreviousSample(int new_sample_id, int previous_sample_id);
  std::vector<double> Predict(ForestDataset& dataset);
  std::vector<double> PredictRaw(ForestDataset& dataset);
  std::vector<double> PredictRaw(ForestDataset& dataset, int forest_num);
  
  inline TreeEnsemble* GetEnsemble(int i) {return forests_[i].get();}
  inline int32_t NumSamples() {return num_samples_;}
  inline int32_t NumTrees() {return num_trees_;}  
  inline int32_t NumTrees(int ensemble_num) {return forests_[ensemble_num]->NumTrees();}
  inline int32_t NumLeaves(int ensemble_num) {return forests_[ensemble_num]->NumLeaves();}
  inline int32_t OutputDimension() {return output_dimension_;}
  inline int32_t OutputDimension(int ensemble_num) {return forests_[ensemble_num]->OutputDimension();}
  inline bool IsLeafConstant(int ensemble_num) {return forests_[ensemble_num]->IsLeafConstant();}
  
  void SaveToJsonFile(std::string filename) {
    nlohmann::json model_json = this->to_json();
    std::ofstream output_file(filename);
    output_file << model_json << std::endl;
  }
  
  void LoadFromJsonFile(std::string filename) {
    std::ifstream f(filename);
    nlohmann::json file_tree_json = nlohmann::json::parse(f);
    this->Reset();
    this->from_json(file_tree_json);
  }

  void Reset() {
    forests_.clear();
    num_samples_ = 0;
    num_trees_ = 0;
    output_dimension_ = 0;
    is_leaf_constant_ = 0;
    initialized_ = false;
  }

  /*! \brief Save to JSON */
  nlohmann::json to_json();
  /*! \brief Load from JSON */
  void from_json(const nlohmann::json& forest_container_json);

 private:
  std::vector<std::unique_ptr<TreeEnsemble>> forests_;
  int num_samples_;
  int num_trees_;
  int output_dimension_;
  bool is_leaf_constant_;
  bool initialized_{false};
};
} // namespace StochTree

#endif // STOCHTREE_CONTAINER_H_
