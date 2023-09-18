/*!
 * Copyright (c) 2023 randtree authors.
 * 
 * Define a training dataset as essentially a thin wrapper around a vector of 
 * features, a class that wraps the raw data for each feature along with:
 * (1) A vector of indices that place the feature in sorted order, and 
 * (2) A vector of "strides" that report the 
 * 
 * Interface and class design inspired heavily by the Dataset and DatasetLoader 
 * classes in LightGBM, which is released under the following copyright:
 * 
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_TRAIN_DATA_H_
#define STOCHTREE_TRAIN_DATA_H_

#include <stochtree/config.h>
#include <stochtree/io.h>
#include <stochtree/meta.h>
#include <stochtree/random.h>
#include <stochtree/tree.h>

#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace StochTree {

enum FeatureType {
  kNumeric,
  kOrderedCategorical,
  kUnorderedCategorical
};

/*! \brief Class for features of various types */
class Feature {
 public:
  /*!
  * \brief Constructor
  * \param num_data Total number of data
  * \param feature_type whether the feature is numeric, ordered categorical, or unordered categorical
  */
  Feature(data_size_t num_data, FeatureType feature_type): num_data_(num_data), feature_type_(feature_type) {
    // Initialize raw data vector
    raw_data_.resize(num_data_, 0.);
    // Initialize sort indices
    data_sort_indices_.resize(num_data_, 0);
    // Initialize index stride vector
    data_index_strides_.resize(num_data_, 1);

    // Deduce (high-level) feature type and missing value coding
    if (feature_type == FeatureType::kUnorderedCategorical) {
      Log::Fatal("Unordered categorical features not yet implemented");
    } else if ((feature_type != FeatureType::kOrderedCategorical) && (feature_type != FeatureType::kNumeric)) {
      Log::Fatal("RawFeature type must be either Numeric, Ordered Categorical, or Unordered Categorical");
    }
  }

  /*! \brief Destructor */
  ~Feature() {}
  
  /*!
   * \brief Push one record to the raw data vector
   * \param idx Index of record
   * \param value feature value of record
   */
  inline void PushRawData(data_size_t idx, double value) {
    raw_data_[idx] = value;
  }

  /*!
   * \brief Calculate the sort indices and strides for a feature, 
   *        either for the first time (i.e. after the dataset has 
   *        been loaded), or to reset training for the next tree in 
   *        an ensemble.
   */
  inline void ResetFeatureMetadata() {
    // Construct sort indices from the raw data
    ArgSortFeature();
    // Construct strides vector from the raw data
    CalculateStrides();
  }

  /*!
   * \brief Convert feature values to sort indices. Adapted from xgboost code:
   * https://github.com/dmlc/xgboost/blob/master/src/common/algorithm.h#L77
   * https://github.com/dmlc/xgboost/blob/master/src/common/algorithm.h#L40
   */
  void ArgSortFeature() {
    // Make a vector of indices from 0 to num_data_ - 1
    if (data_sort_indices_.size() != num_data_){
      data_sort_indices_.resize(num_data_, 0);
    }
    std::iota(data_sort_indices_.begin(), data_sort_indices_.end(), 0);
    
    // Check whether or not the data are trivial
    auto first_data_elem = raw_data_[0];
    auto function_check = [&first_data_elem](double elem) {return elem == first_data_elem;};
    bool trivial_raw_data = std::all_of(raw_data_.begin(), raw_data_.end(), function_check);
    if (trivial_raw_data){
      Log::Info("Feature had trivial data elements, this could be because the feature is trivial or because raw_data_ has not been filled");
      return;
    }
    
    // Check that the argsort indices produced by this procedure are the same size as the raw data
    if (raw_data_.size() != data_sort_indices_.size()){
      Log::Fatal("Raw data and sort indices must be the same size");
    }

    // Define a custom comparator to be used with stable_sort:
    // For every two indices l and r store as elements of `data_sort_indices_`, 
    // compare them for sorting purposes by indexing `raw_data_` with both l and r
    auto comp_op = [&](size_t const &l, size_t const &r) { return std::less<double>{}(raw_data_[l], raw_data_[r]); };
    std::stable_sort(data_sort_indices_.begin(), data_sort_indices_.end(), comp_op);
    return;
  }

  /*!
   * \brief Determine strides for a feature for the first time
   */
  void CalculateStrides() {
    // Make a vector of all 1s of length num_data_
    if (data_index_strides_.size() != num_data_){
      data_index_strides_.resize(num_data_, 0);
    }

    // Run through the sorted data, determining the stride length 
    // of all unique values (stored for every observation, no dedupliation)
    data_size_t stride_begin = 0;
    data_size_t stride_length = 1;
    double current_val;
    double previous_val;
    for (data_size_t i = 0; i < num_data_; i++){
      // Need at least two values to have a duplicate
      if (i != 0){
        previous_val = raw_data_[data_sort_indices_[i-1]];
        current_val = raw_data_[data_sort_indices_[i]];
        // Check if current value is same as previous
        if (std::fabs(current_val - previous_val) < kEpsilon) { 
          // If so, increment stride length
          stride_length += 1;
        } else {
          // Otherwise, write the stride length for every element in the stride
          for (data_size_t j = 0; j < stride_length; j++){ 
            data_index_strides_[stride_begin + j] = stride_length;
          }
          // Reset stride at the next element
          stride_begin += stride_length;
          stride_length = 1;
        }
      } 
    }
  }

  /*!
   * \brief Re-arrange sort indices and recompute strides according to a split condition
   */
  void ReArrangeFeatureMetadata(data_size_t start_idx, data_size_t num_elements, 
                                std::vector<int>& partition_cond, data_size_t num_true) {
    // Check that the vector of split conditions is the same size as the 
    // number of elements to be rearranged by the node split
    if (partition_cond.size() != num_elements){
      Log::Fatal("Split condition vector not the same size as num_elements");
    }
    
    data_size_t num_false_ = num_elements - num_true;

    // Rearrange the sort indices based on whether partition_cond = 1 (feature <= val)
    data_size_t true_idx = 0;
    data_size_t false_idx = num_true;
    data_size_t temp_swap_idx;
    while (true_idx != num_true) {
      if (partition_cond[true_idx] == 0 && partition_cond[false_idx] == 1) {
        // Swap the indices
        temp_swap_idx = data_sort_indices_[start_idx + true_idx];
        data_sort_indices_[start_idx + true_idx] = data_sort_indices_[start_idx + false_idx];
        data_sort_indices_[start_idx + false_idx] = temp_swap_idx;
        // Increment offset indices into the "true" and "false" partitions of the node
        true_idx += 1;
        false_idx += 1;
      } else if (partition_cond[true_idx] == 1 && partition_cond[false_idx] == 1) {
        // Increment offset indices only into the "true" partition of the node, 
        // since the current offset into the "false" partition is ready to be swapped
        true_idx += 1;
      } else if (partition_cond[true_idx] == 0 && partition_cond[false_idx] == 0) {
        // Increment offset indices only into the "false" partition of the node, 
        // since the current offset into the "true" partition is ready to be swapped
        false_idx += 1;
      } else if (partition_cond[true_idx] == 1 && partition_cond[false_idx] == 0) {
        // Increment offset indices into the "true" and "false" partitions of the node, 
        // since both conditions are satisfied
        true_idx += 1;
        false_idx += 1;
      }
    }
    
    // Recompute strides in the new "right" node
    data_size_t stride_begin = 0;
    data_size_t stride_length = 1;
    double current_val;
    double previous_val;
    for (data_size_t i = 0; i < num_true; i++){
      // Need at least two values to have a duplicate
      if (i != 0){
        previous_val = raw_data_[data_sort_indices_[start_idx + i - 1]];
        current_val = raw_data_[data_sort_indices_[start_idx + i]];
        // Check if current value is same as previous
        if (std::fabs(current_val - previous_val) < kEpsilon) { 
          // If so, increment stride length
          stride_length += 1;
        } else {
          // Otherwise, write the stride length for every element in the stride
          for (data_size_t j = 0; j < stride_length; j++){ 
            data_index_strides_[start_idx + stride_begin + j] = stride_length;
          }
          // Reset stride at the next element
          stride_begin += stride_length;
          stride_length = 1;
        }
      } 
    }

    // Recompute strides in the new "left" node
    stride_begin = 0;
    stride_length = 1;
    for (data_size_t i = 0; i < num_false_; i++){
      // Need at least two values to have a duplicate
      if (i != 0){
        previous_val = raw_data_[data_sort_indices_[start_idx + num_true + i - 1]];
        current_val = raw_data_[data_sort_indices_[start_idx + num_true + i]];
        // Check if current value is same as previous
        if (std::fabs(current_val - previous_val) < kEpsilon) { 
          // If so, increment stride length
          stride_length += 1;
        } else {
          // Otherwise, write the stride length for every element in the stride
          for (data_size_t j = 0; j < stride_length; j++){ 
            data_index_strides_[start_idx + num_true + stride_begin + j] = stride_length;
          }
          // Reset stride at the next element
          stride_begin += stride_length;
          stride_length = 1;
        }
      } 
    }
  }

  /*! \brief Disable copy */
  Feature& operator=(const Feature&) = delete;
  Feature(const Feature&) = delete;

  /*!
   * \brief Retrieve raw data value (in original sort order) at index idx
   * \param idx Index of record
   */
  inline double GetRawData(data_size_t idx) {
    if (idx < num_data_) {
      return raw_data_[idx];
    } else {
      Log::Fatal("Observation %d exceeds the total number of observations (%d)", idx, num_data_);
    }
  }

  /*!
   * \brief Retrive the data sort index currently in position idx
   * \param idx Index of record
   */
  inline data_size_t GetSortIndex(data_size_t idx) {
    if (idx < num_data_) {
      return data_sort_indices_[idx];
    } else {
      Log::Fatal("Observation %d exceeds the total number of observations (%d)", idx, num_data_);
    }
  }

  /*!
   * \brief Retrive the data sort index currently in position idx
   * \param idx Index of record
   */
  inline void SetSortIndex(data_size_t row_idx, data_size_t sort_idx) {
    if ((row_idx < num_data_) && (sort_idx < num_data_)) {
      data_sort_indices_[row_idx] = sort_idx;
    } else {
      Log::Fatal("Observation %d or %d exceeds the total number of observations (%d)", row_idx, sort_idx, num_data_);
    }
  }

  /*!
   * \brief Retrive the stride of the value corresponding to the sort index in position idx
   * \param idx Index of record
   */
  inline data_size_t GetSortIndexStride(data_size_t idx) {
    if (idx < num_data_) {
      return data_index_strides_[idx];
    } else {
      Log::Fatal("Observation %d exceeds the total number of observations (%d)", idx, num_data_);
    }
  }

  /*! @brief return the number of observations */
  inline data_size_t get_num_data() {
    return num_data_;
  }

  /*! @brief return the type of the feature */
  inline FeatureType get_feature_type() {
    return feature_type_;
  }

 private:
  /*! \brief Raw data for a feature 
   */
  std::vector<double> raw_data_;
  
  /*! \brief Indices that place the feature's value in ascending order */
  std::vector<data_size_t> data_sort_indices_;
  
  /*! \brief Indices that report the stride length of any repeated indices */
  std::vector<data_size_t> data_index_strides_;
  
  /*! \brief Details about size of feature */
  data_size_t num_data_;
  
  /*! \brief Type of feature */
  FeatureType feature_type_;
};

/*! \brief forward declaration of TrainDataLoader class*/
class TrainDataLoader;

/*! \brief Dataset used for "training" (i.e. sampling) BART and XBART models */
class TrainData {
 public:
  // Give TrainDataLoader access to private members of TrainData
  friend TrainDataLoader;

  TrainData();

  TrainData(data_size_t num_data);

  void Construct(int num_total_features);

  /*! \brief Destructor */
  ~TrainData();
  
  inline void FinishOneRow(data_size_t row_idx) {
    if (is_finish_load_) { return; }
  }

  void FinishLoad();

  /*! \brief Get Number of used features */
  inline int num_variables() const { return num_variables_; }

  /*! \brief Get names of current data set */
  inline const std::vector<std::string>& variable_names() const { return variable_names_; }

  inline void set_feature_names(const std::vector<std::string>& variable_names) {
    if (variable_names.size() != static_cast<size_t>(num_variables_)) {
      Log::Fatal("Size of num_variables error, should equal with total number of variables");
    }
    variable_names_ = std::vector<std::string>(variable_names);
    std::unordered_set<std::string> variable_name_set;
    // replace ' ' in feature_names with '_'
    bool spaceInFeatureName = false;
    for (auto& variable_name : variable_names_) {
      // check JSON
      if (!Common::CheckAllowedJSON(variable_name)) {
        Log::Fatal("Do not support special JSON characters in variable name.");
      }
      if (variable_name.find(' ') != std::string::npos) {
        spaceInFeatureName = true;
        std::replace(variable_name.begin(), variable_name.end(), ' ', '_');
      }
      if (variable_name_set.count(variable_name) > 0) {
        Log::Fatal("Variable (%s) appears more than one time.", variable_name.c_str());
      }
      variable_name_set.insert(variable_name);
    }
    if (spaceInFeatureName) {
      Log::Warning("Found whitespace in variable_names, replace with underlines");
    }
  }

  /*! \brief Reset a training dataset so that each of its features are pre-sorted based on the entire dataset */
  inline void ResetToRaw(){
    // Resort all of the features and recalculate strides
    for (int i = 0; i < num_variables_; i++){
      variables_[i]->ResetFeatureMetadata();
    }
  }

  inline void Predict(const std::vector<std::unique_ptr<Tree>> &trees, 
                      std::vector<double> output, int tree_begin, int tree_end) {
    double pred;
    if (output.size() != num_data_){
      Log::Fatal("Mismatched size of prediction vector and training data");
    }
    for (data_size_t i = 0; i < num_data_; i++) {
      pred = 0.0;
      for (size_t i = tree_begin; i < tree_end; ++i) {
        auto const &tree = *(trees[i].get());
        node_t nidx = 0;
        while (!tree[nidx].IsLeaf()) {
          int col = tree[nidx].SplitIndex();
          auto fvalue = this->get_feature_value(i, col);
          bool proceed_left = fvalue < tree[nidx].SplitCond();
          if (proceed_left) {
            nidx = tree[nidx].LeftChild();
          } else {
            nidx = tree[nidx].RightChild();
          }
        }
        pred += (*trees[i])[nidx].LeafValue();
      }
      output[i] = pred;
    }
  }

  /*! \brief Subtract predictions of various trees from the residual vector */
  inline void ResidualSubtract(std::vector<std::vector<data_size_t>>& tree_data_observations, Tree* tree, int tree_num) {
    data_size_t n = tree_data_observations[0].size();
    if (n != num_data_) { 
      Log::Fatal("Training set and leaf tracker have a different number of observations");
    }
    for (data_size_t i = 0; i < num_data_; i++) {
      residuals_[i] -= (*tree)[tree_data_observations[tree_num][i]].LeafValue();
    }
  }

  /*! \brief Subtract predictions of various trees from the residual vector */
  inline void ResidualSubtract(std::vector<double>&& delta) {
    for (data_size_t i = 0; i < num_data_; i++) {
      residuals_[i] -= delta[i];
    }
  }

  /*! \brief Add predictions of various trees from the residual vector */
  inline void ResidualAdd(std::vector<std::vector<data_size_t>>& tree_data_observations, Tree* tree, int tree_num) {
    data_size_t n = tree_data_observations[0].size();
    if (n != num_data_) { 
      Log::Fatal("Training set and leaf tracker have a different number of observations");
    }
    for (data_size_t i = 0; i < num_data_; i++) {
      residuals_[i] += (*tree)[tree_data_observations[tree_num][i]].LeafValue();
    }
  }

  /*! \brief Add predictions of various trees to the residual vector */
  inline void ResidualAdd(std::vector<double>&& delta) {
    for (data_size_t i = 0; i < num_data_; i++) {
      residuals_[i] += delta[i];
    }
  }

  /*! \brief Convert all residuals to 0 */
  inline void ResidualZero() {
    for (data_size_t i = 0; i < num_data_; i++) {
      residuals_[i] = 0.;
    }
  }

  /*! \brief Reset all residuals to raw outcome values */
  inline void ResidualReset() {
    for (data_size_t i = 0; i < num_data_; i++) {
      residuals_[i] = outcome_[i];
    }
  }

  /*!
   * \brief "Partition" the sort indices in a training dataset after a node has been split, 
   * so that the samples indices are sifted into the two newly created leaf nodes.
   */
  inline void PartitionLeaf(data_size_t leaf_start_idx, data_size_t num_leaf_elements, 
                            int split_col, double split_value) {
    double feature_value;
    data_size_t feature_sort_ind;
    double first_comp_feature_value;
    double second_comp_feature_value;
    bool first_comp_true;
    bool second_comp_true;
    std::vector<data_size_t> true_vector_inds(num_leaf_elements);
    std::vector<data_size_t> false_vector_inds(num_leaf_elements);
    data_size_t num_true = 0;
    data_size_t num_false = 0;
    data_size_t true_idx = 0;
    data_size_t false_idx = 0;
    data_size_t offset = 0;

    for (int j = 0; j < num_variables_; j++) {
      // First pass through the data for feature j -- assess true / false for each observation
      num_true = 0;
      num_false = 0;
      for (data_size_t i = leaf_start_idx; i < leaf_start_idx + num_leaf_elements; i++) {
        feature_sort_ind = this->get_feature_sort_index(i, j);
        feature_value = this->get_feature_value(feature_sort_ind, split_col);
        if (feature_value <= split_value){
          true_vector_inds[num_true] = feature_sort_ind;
          num_true++;
        } else {
          false_vector_inds[num_false] = feature_sort_ind;
          num_false++;
        }
      }

      // Second pass through data -- rearrange indices for feature j
      true_idx = 0;
      false_idx = 0;
      offset = 0;
      for (data_size_t i = leaf_start_idx; i < leaf_start_idx + num_leaf_elements; i++) {
        if (offset < num_true){
          this->set_feature_sort_index(i, j, true_vector_inds[true_idx]);
          true_idx++;
        } else {
          this->set_feature_sort_index(i, j, false_vector_inds[false_idx]);
          false_idx++;
        }
        offset++;
      }
    }
  }

  /*! \brief Set the outcome index */
  inline void set_outcome_index(int idx) { outcome_index_ = idx; }

  /*! \brief Set the treatment index */
  inline void set_treatment_index(int idx) { treatment_index_ = idx; }

  /*! \brief Get number of observations in the dataset */
  inline data_size_t num_data() const { return num_data_; }

  /*! \brief Disable copy */
  TrainData& operator=(const TrainData&) = delete;
  TrainData(const TrainData&) = delete;

  /*! \brief Indexing data */
  inline double get_feature_value(data_size_t row_id, int col_id){
    if (col_id < variables_.size()){
      return variables_[col_id]->GetRawData(row_id);
    } else {
      Log::Fatal("Feature %d exceeds the total number of features %d", 
                 col_id, num_variables_);
    }
  }

  /*! \brief Access "argsort" indices for a given feature */
  inline double get_feature_sort_index(data_size_t row_id, int col_id){
    if (col_id < variables_.size()){
      return variables_[col_id]->GetSortIndex(row_id);
    } else {
      Log::Fatal("Feature %d exceeds the total number of features %d", 
                 col_id, num_variables_);
    }
  }

  /*! \brief Access "stride" for a given feature value */
  inline double get_feature_stride(data_size_t row_id, int col_id){
    if (col_id < variables_.size()){
      return variables_[col_id]->GetSortIndexStride(row_id);
    } else {
      Log::Fatal("Feature %d exceeds the total number of features %d", 
                 col_id, num_variables_);
    }
  }

  /*! \brief Indexing data */
  inline double get_outcome_value(data_size_t row_id){
    if (row_id < outcome_.size()){
      return outcome_[row_id];
    } else {
      Log::Fatal("Row %d exceeds the number of outcome observations %d", 
                 row_id, outcome_.size());
    }
  }

  /*! \brief Indexing data */
  inline double get_residual_value(data_size_t row_id){
    if (row_id < residuals_.size()){
      return residuals_[row_id];
    } else {
      Log::Fatal("Row %d exceeds the number of residual (outcome) observations %d", 
                 row_id, residuals_.size());
    }
  }

  /*! \brief Indexing data */
  inline double get_treatment_value(data_size_t row_id){
    if (treatment_index_ >= 0){
      if (row_id < treatment_.size()){
        return treatment_[row_id];
      } else {
        Log::Fatal("Row %d exceeds the number of treatment observations %d", 
                  row_id, treatment_.size());
      }
    } else {
      Log::Fatal("Treatment variable is not set");
    }
  }

  /*! \brief Swap "argsort" indices for two given features */
  inline void swap_feature_sort_indices(data_size_t first_row_id, data_size_t second_row_id, int col_id){
    if (col_id < variables_.size()){
      data_size_t temporary_idx = variables_[col_id]->GetSortIndex(first_row_id);
      variables_[col_id]->SetSortIndex(first_row_id, variables_[col_id]->GetSortIndex(second_row_id));
      variables_[col_id]->SetSortIndex(second_row_id, temporary_idx);
    } else {
      Log::Fatal("Feature %d exceeds the total number of features %d", 
                 col_id, num_variables_);
    }
  }

  /*! \brief Set "argsort" index for a given feature */
  inline void set_feature_sort_index(data_size_t row_idx, int col_id, data_size_t sort_idx){
    if (col_id < variables_.size()){
      variables_[col_id]->SetSortIndex(row_idx, sort_idx);
    } else {
      Log::Fatal("Feature %d exceeds the total number of features %d", 
                 col_id, num_variables_);
    }
  }

 private:
  std::string data_filename_;
  /*! \brief Store variables all as doubles (for now) */
  std::vector<std::unique_ptr<Feature>> variables_;
  /*! \brief Outcome variable*/
  std::vector<double> outcome_;
  /*! \brief treatment variable (if applicable)*/
  std::vector<double> treatment_;
  /*! \brief Residual variable (outcome minus predictions of each of the trees)*/
  std::vector<double> residuals_;
  
  /*! \brief Number of features*/
  int num_variables_;
  /*! \brief Number of observations*/
  data_size_t num_data_;
  /*! \brief store feature names */
  std::vector<std::string> variable_names_;
  /*! \brief serialized versions */
  bool is_finish_load_;
  /*! \brief index of the outcome variable */
  int outcome_index_;
  /*! \brief index of the treatment variable (if applicable) */
  int treatment_index_;
};

/*! \brief Training data creation class. Can build a training dataset by either:
 *     (1) Parsing CSV files (no other file types supported at present)
 *     (2) Reading contiguous-memory data from an R matrix or Numpy array
 */
class TrainDataLoader {
 public:
  TrainDataLoader(const Config& io_config, int num_class, const char* filename);

  ~TrainDataLoader();

  TrainData* LoadFromFile(const char* filename);

  TrainData* ConstructFromMatrix(double* matrix_data, int num_col, 
                                 data_size_t num_row, bool is_row_major);

  /*! \brief Disable copy */
  TrainDataLoader& operator=(const TrainDataLoader&) = delete;
  TrainDataLoader(const TrainDataLoader&) = delete;

  inline std::vector<std::string> get_variable_names(){
    return variable_names_;
  }

 private:
  void LoadHeaderFromMemory(TrainData* dataset, const char* buffer);

  void SetHeader(const char* filename);

  void SetHeader(const Config& io_config);

  void CheckDataset(const TrainData* dataset);

  std::vector<std::string> LoadTextDataToMemory(const char* filename, int* num_global_data);

  /*! \brief Extract local features from memory */
  void ExtractFeaturesFromMemory(std::vector<std::string>* text_data, const Parser* parser, TrainData* dataset);
  
  /*! \brief Config object used to drive prediction*/
  const Config& config_;
  /*! \brief number of classes */
  int num_class_;
  /*! \brief index of label column */
  int label_idx_;
  /*! \brief index of treatment column */
  int treatment_idx_;
  /*! \brief store feature names */
  std::vector<std::string> variable_names_;
};

}  // namespace StochTree

#endif   // STOCHTREE_TRAIN_DATA_H_
