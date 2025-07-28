/*!
 * Copyright (c) 2024 stochtree authors.
 * 
 * General-purpose data structures used for keeping track of categories in a training dataset.
 * 
 * SampleCategoryMapper is a simplified version of SampleNodeMapper, which is not tree-specific 
 * as it tracks categories loaded into a training dataset, and we do not expect to modify it during 
 * training.
 * 
 * SampleCategoryMapper is used in two places:
 *   1. Group random effects: mapping observations to group IDs for the purpose of computing random effects
 *   2. Heteroskedasticity based on fixed categories (as opposed to partitions as in HBART by Pratola et al 2018)
 *         - One example of this would be binary treatment causal inference with separate outcome variances 
 *           for the treated and control groups (as in Krantsevich et al 2023)
 * 
 * CategorySampleTracker is a simplified version of FeatureUnsortedPartition, which as above does 
 * not vary based on tree / partition and is not expected to change during training.
 * 
 * SampleNodeMapper is inspired by the design of the DataPartition class in LightGBM, 
 * released under the MIT license with the following copyright:
 * 
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_CATEGORY_TRACKER_H_
#define STOCHTREE_CATEGORY_TRACKER_H_

#include <Eigen/Dense>
#include <stochtree/log.h>
#include <stochtree/meta.h>

#include <map>
#include <numeric>
#include <vector>

namespace StochTree {

/*! \brief Class storing sample-node map for each tree in an ensemble
 * TODO: Add run-time checks for categories with a few observations
 */
class SampleCategoryMapper {
 public:
  SampleCategoryMapper(std::vector<int32_t>& group_indices) {
    num_observations_ = group_indices.size();
    observation_indices_ = group_indices;
  }
  
  SampleCategoryMapper(SampleCategoryMapper& other){
    num_observations_ = other.NumObservations();
    observation_indices_.resize(num_observations_);
    for (int i = 0; i < num_observations_; i++) {
      observation_indices_[i] = other.GetCategoryId(i);
    }
  }

  inline data_size_t GetCategoryId(data_size_t sample_id) {
    CHECK_LT(sample_id, num_observations_);
    return observation_indices_[sample_id];
  }

  inline void SetCategoryId(data_size_t sample_id, int category_id) {
    CHECK_LT(sample_id, num_observations_);
    observation_indices_[sample_id] = sample_id;
  }
  
  inline int NumObservations() {return num_observations_;}

 private:
  std::vector<int> observation_indices_;
  data_size_t num_observations_;
};

/*! \brief Mapping categories to the indices they contain
 * TODO: Add run-time checks for categories with a few observations
 */
class CategorySampleTracker {
 public:
  CategorySampleTracker(const std::vector<int32_t>& group_indices) {
    int n = group_indices.size();
    indices_ = std::vector<data_size_t>(n);
    std::iota(indices_.begin(), indices_.end(), 0);

    auto comp_op = [&](size_t const &l, size_t const &r) { return std::less<data_size_t>{}(group_indices[l], group_indices[r]); };
    std::stable_sort(indices_.begin(), indices_.end(), comp_op);

    category_count_ = 0;
    int observation_count = 0;
    for (int i = 0; i < n; i++) {
      bool start_cond = i == 0;
      bool end_cond = i == n-1;
      bool new_group_cond{false};
      if (i > 0) new_group_cond = group_indices[indices_[i]] != group_indices[indices_[i-1]];
      if (start_cond || new_group_cond) {
        category_id_map_.insert({group_indices[indices_[i]], category_count_});
        unique_category_ids_.push_back(group_indices[indices_[i]]);
        node_index_vector_.emplace_back();
        if (i == 0) {
          category_begin_.push_back(i);
        } else {
          category_begin_.push_back(i);
          category_length_.push_back(observation_count);
        }
        observation_count = 1;
        category_count_++;
      } else if (end_cond) {
        category_length_.push_back(observation_count+1);
      } else {
        observation_count++;
      }
      // Add the index to the category's node index vector in either case
      node_index_vector_[category_count_ - 1].emplace_back(indices_[i]);
    }
  }

  /*! \brief Zero-indexed numeric index that category_id is remapped to internally */
  inline int32_t CategoryNumber(int category_id) {
    return category_id_map_[category_id];
  }

  /*! \brief First index of data points contained in node_id */
  inline data_size_t CategoryBegin(int category_id) {return category_begin_[category_id_map_[category_id]];}

  /*! \brief One past the last index of data points contained in node_id */
  inline data_size_t CategoryEnd(int category_id) {
    int32_t id = category_id_map_[category_id];
    return category_begin_[id] + category_length_[id];
  }

  /*! \brief Number of data points contained in node_id */
  inline data_size_t CategorySize(int category_id) {
    return category_length_[category_id_map_[category_id]];
  }

  /*! \brief Number of total categories stored */
  inline data_size_t NumCategories() {return category_count_;}

  /*! \brief Data indices */
  std::vector<data_size_t> indices_;

  /*! \brief Data indices for a given node */
  std::vector<data_size_t>& NodeIndices(int category_id) {
    int32_t id = category_id_map_[category_id];
    return node_index_vector_[id];
  }
  
  /*! \brief Data indices for a given node */
  std::vector<data_size_t>& NodeIndicesInternalIndex(int internal_category_id) {
    return node_index_vector_[internal_category_id];
  }

  /*! \brief Returns label index map */
  std::map<int32_t, int32_t>& GetLabelMap() {return category_id_map_;}

  std::vector<int32_t>& GetUniqueGroupIds() {return unique_category_ids_;}

 private:
  // Vectors tracking indices in each node
  std::vector<data_size_t> category_begin_;
  std::vector<data_size_t> category_length_;
  std::map<int32_t, int32_t> category_id_map_;
  std::vector<int32_t> unique_category_ids_;
  std::vector<std::vector<data_size_t>> node_index_vector_;
  int32_t category_count_;
};

} // namespace StochTree

#endif // STOCHTREE_CATEGORY_TRACKER_H_
