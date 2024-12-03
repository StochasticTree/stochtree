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
  ForestContainer(int num_trees, int output_dimension = 1, bool is_leaf_constant = true, bool is_exponentiated = false);
  ForestContainer(int num_samples, int num_trees, int output_dimension = 1, bool is_leaf_constant = true, bool is_exponentiated = false);
  ~ForestContainer() {}

  void DeleteSample(int sample_num);
  void AddSample(TreeEnsemble& forest);
  void InitializeRoot(double leaf_value);
  void InitializeRoot(std::vector<double>& leaf_vector);
  void AddSamples(int num_samples);
  void CopyFromPreviousSample(int new_sample_id, int previous_sample_id);
  std::vector<double> Predict(ForestDataset& dataset);
  std::vector<double> PredictRaw(ForestDataset& dataset);
  std::vector<double> PredictRaw(ForestDataset& dataset, int forest_num);
  std::vector<double> PredictRawSingleTree(ForestDataset& dataset, int forest_num, int tree_num);
  void PredictInPlace(ForestDataset& dataset, std::vector<double>& output);
  void PredictRawInPlace(ForestDataset& dataset, std::vector<double>& output);
  void PredictRawInPlace(ForestDataset& dataset, int forest_num, std::vector<double>& output);
  void PredictRawSingleTreeInPlace(ForestDataset& dataset, int forest_num, int tree_num, std::vector<double>& output);
  void PredictLeafIndicesInplace(Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>& covariates, 
                                 Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>& output, 
                                 std::vector<int>& forest_indices, int num_trees, data_size_t n);

  inline TreeEnsemble* GetEnsemble(int i) {return forests_[i].get();}
  inline int32_t NumSamples() {return num_samples_;}
  inline int32_t NumTrees() {return num_trees_;}  
  inline int32_t NumTrees(int ensemble_num) {return forests_[ensemble_num]->NumTrees();}
  inline int32_t NumLeaves(int ensemble_num) {return forests_[ensemble_num]->NumLeaves();}
  inline int32_t EnsembleTreeMaxDepth(int ensemble_num, int tree_num) {return forests_[ensemble_num]->TreeMaxDepth(tree_num);}
  inline double EnsembleAverageMaxDepth(int ensemble_num) {return forests_[ensemble_num]->AverageMaxDepth();}
  inline double AverageMaxDepth() {
    double numerator = 0.;
    double denominator = 0.;
    for (int i = 0; i < num_samples_; i++) {
      for (int j = 0; j < num_trees_; j++) {
        numerator += static_cast<double>(forests_[i]->TreeMaxDepth(j));
        denominator += 1.;
      }
    }
    return numerator / denominator;
  }
  inline int32_t OutputDimension() {return output_dimension_;}
  inline int32_t OutputDimension(int ensemble_num) {return forests_[ensemble_num]->OutputDimension();}
  inline bool IsLeafConstant() {return is_leaf_constant_;}
  inline bool IsLeafConstant(int ensemble_num) {return forests_[ensemble_num]->IsLeafConstant();}
  inline bool IsExponentiated() {return is_exponentiated_;}
  inline bool IsExponentiated(int ensemble_num) {return forests_[ensemble_num]->IsExponentiated();}
  inline bool AllRoots(int ensemble_num) {return forests_[ensemble_num]->AllRoots();}
  inline void SetLeafValue(int ensemble_num, double leaf_value) {forests_[ensemble_num]->SetLeafValue(leaf_value);}
  inline void SetLeafVector(int ensemble_num, std::vector<double>& leaf_vector) {forests_[ensemble_num]->SetLeafVector(leaf_vector);}
  inline void IncrementSampleCount() {num_samples_++;}

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

  std::string DumpJsonString() {
    nlohmann::json model_json = this->to_json();
    return model_json.dump();
  }

  void LoadFromJsonString(std::string& json_string) {
    nlohmann::json file_tree_json = nlohmann::json::parse(json_string);
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
  /*! \brief Append to a forest container from JSON, requires that the ensemble already contains a nonzero number of forests */
  void append_from_json(const nlohmann::json& forest_container_json);

 private:
  std::vector<std::unique_ptr<TreeEnsemble>> forests_;
  int num_samples_;
  int num_trees_;
  int output_dimension_;
  bool is_exponentiated_{false};
  bool is_leaf_constant_;
  bool initialized_{false};
};
} // namespace StochTree

#endif // STOCHTREE_CONTAINER_H_
