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

#include <fstream>

namespace StochTree {

/*!
 * \brief Container of `TreeEnsemble` forest objects. This is the primary (in-memory) storage interface for multiple
 * "samples" of a decision tree ensemble in `stochtree`.
 * \ingroup forest_group
 */
class ForestContainer {
 public:
  /*!
   * \brief Construct a new ForestContainer object. 
   * 
   * \param num_trees Number of trees in each forest.
   * \param output_dimension Dimension of the leaf node parameter in each tree of each forest.
   * \param is_leaf_constant Whether or not the leaves of each tree are treated as "constant." If true, then predicting from an ensemble is simply a matter or determining which leaf node an observation falls into. If false, prediction will multiply a leaf node's parameter(s) for a given observation by a basis vector.
   * \param is_exponentiated Whether or not the leaves of each tree are stored in log scale. If true, leaf predictions are exponentiated before their prediction is returned.
   */
  ForestContainer(int num_trees, int output_dimension = 1, bool is_leaf_constant = true, bool is_exponentiated = false);
  /*!
   * \brief Construct a new ForestContainer object.
   * 
   * \param num_samples Initial size of a container of forest samples.
   * \param num_trees Number of trees in each forest.
   * \param output_dimension Dimension of the leaf node parameter in each tree of each forest.
   * \param is_leaf_constant Whether or not the leaves of each tree are treated as "constant." If true, then predicting from an ensemble is simply a matter or determining which leaf node an observation falls into. If false, prediction will multiply a leaf node's parameter(s) for a given observation by a basis vector.
   * \param is_exponentiated Whether or not the leaves of each tree are stored in log scale. If true, leaf predictions are exponentiated before their prediction is returned.
   */
  ForestContainer(int num_samples, int num_trees, int output_dimension = 1, bool is_leaf_constant = true, bool is_exponentiated = false);
  ~ForestContainer() {}
  /*!
   * \brief Combine two forests into a single forest by merging their trees
   * 
   * \param inbound_forest_index Index of the forest that will be appended to
   * \param outbound_forest_index Index of the forest that will be appended
   */
  void MergeForests(int inbound_forest_index, int outbound_forest_index) {
    forests_[inbound_forest_index]->MergeForest(*forests_[outbound_forest_index]);
  }
  /*!
   * \brief Add a constant value to every leaf of every tree of a specified forest
   * 
   * \param forest_index Index of forest whose leaves will be modified
   * \param constant_value Value to add to every leaf of every tree of the forest at `forest_index`
   */
  void AddToForest(int forest_index, double constant_value) {
    forests_[forest_index]->AddValueToLeaves(constant_value);
  }
  /*!
   * \brief Multiply every leaf of every tree of a specified forest by a constant value
   * 
   * \param forest_index Index of forest whose leaves will be modified
   * \param constant_multiple Value to multiply through by every leaf of every tree of the forest at `forest_index`
   */
  void MultiplyForest(int forest_index, double constant_multiple) {
    forests_[forest_index]->MultiplyLeavesByValue(constant_multiple);
  }
  /*!
   * \brief Remove a forest from a container of forest samples and delete the corresponding object, freeing its memory.
   * 
   * \param sample_num Index of forest to be deleted.
   */
  void DeleteSample(int sample_num);
  /*!
   * \brief Add a new forest to the container by copying `forest`.
   * 
   * \param forest Forest to be copied and added to the container of retained forest samples.
   */
  void AddSample(TreeEnsemble& forest);
  /*!
   * \brief Initialize a "root" forest of univariate trees as the first element of the container, setting all root node values in every tree to `leaf_value`.
   * 
   * \param leaf_value Value to assign to the root node of every tree.
   */
  void InitializeRoot(double leaf_value);
  /*!
   * \brief Initialize a "root" forest of multivariate trees as the first element of the container, setting all root node values in every tree to `leaf_vector`.
   * 
   * \param leaf_value Vector of values to assign to the root node of every tree.
   */
  void InitializeRoot(std::vector<double>& leaf_vector);
  /*!
   * \brief Pre-allocate space for `num_samples` additional forests in the container.
   * 
   * \param num_samples Number of (default-constructed) forests to allocated space for in the container.
   */
  void AddSamples(int num_samples);
  /*!
   * \brief Copy the forest stored at `previous_sample_id` to the forest stored at `new_sample_id`.
   * 
   * \param new_sample_id Index of the new forest to be copied from an earlier sample.
   * \param previous_sample_id Index of the previous forest to copy to `new_sample_id`.
   */
  void CopyFromPreviousSample(int new_sample_id, int previous_sample_id);
  /*!
   * \brief Predict from every forest in the container on every observation in the provided dataset. 
   * The resulting vector is "column-major", where every forest in a container defines the columns of a 
   * prediction matrix and every observation in the provided dataset defines the rows. The (`i`,`j`) element 
   * of this prediction matrix can be read from the `j * num_rows + i` element of the returned `std::vector<double>`, 
   * where `num_rows` is equal to the number of observations in `dataset` (i.e. `dataset.NumObservations()`).
   * 
   * \param dataset Data object containining training data, including covariates, leaf regression bases, and case weights.
   * \return std::vector<double> Vector of predictions for every forest in the container and every observation in `dataset`.
   */
  std::vector<double> Predict(ForestDataset& dataset);
  /*!
   * \brief Predict from every forest in the container on every observation in the provided dataset. 
   * The resulting vector stores a possibly three-dimensional array, where the dimensions are arranged as follows 
   * 
   * 1. Dimension of the leaf node's raw values (1 for GaussianConstantLeafModel, GaussianUnivariateRegressionLeafModel, and LogLinearVarianceLeafModel, >1 for GaussianMultivariateRegressionLeafModel)
   * 2. Observations in the provided dataset.
   * 3. Forest samples in the container.
   * 
   * If the leaf nodes have univariate values, then the "first dimension" is 1 and the resulting array has the exact same layout as in \ref Predict.
   * 
   * \param dataset Data object containining training data, including covariates, leaf regression bases, and case weights.
   * \return std::vector<double> Vector of predictions for every forest in the container and every observation in `dataset`.
   */
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
