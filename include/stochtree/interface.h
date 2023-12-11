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
#ifndef STOCHTREE_INTERFACE_H_
#define STOCHTREE_INTERFACE_H_

#include <stochtree/cutpoint_candidates.h>
#include <stochtree/data.h>
#include <stochtree/ensemble.h>
#include <stochtree/export.h>
#include <stochtree/log.h>
#include <stochtree/meta.h>
#include <stochtree/model.h>
#include <stochtree/model_draw.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/tree.h>

#include <memory>

namespace StochTree {

class StochTreeInterface {
 public:
  StochTreeInterface();
  StochTreeInterface(const Config& config);
  ~StochTreeInterface();
  
  void LoadTrainDataFromMemory(double* matrix_data, int num_col, data_size_t num_row, bool is_row_major);
  void LoadPredictionDataFromMemory(double* matrix_data, int num_col, data_size_t num_row, bool is_row_major);
  void LoadPredictionDataFromMemory(double* matrix_data, int num_col, data_size_t num_row, bool is_row_major, const Config config);
  
  void LoadTrainDataFromFile();
  void LoadPredictionDataFromFile();

  void SampleModel();
  std::vector<double> PredictSamples();
  std::vector<double> PredictGreedy(bool predict_train_ = false);

  void SaveSamples();
  void LoadSamples();
 private:
  /*! \brief Config */
  Config config_;
  /*! \brief Pointer to training dataset */
  std::unique_ptr<Dataset> train_dataset_;
  /*! \brief Vector of pointers to feature sort indices */
  // std::vector<std::unique_ptr<FeatureNodeSortTracker>> feature_sort_tracker_;
  std::unique_ptr<UnsortedNodeSampleTracker> unsorted_node_sample_tracker_;
  std::unique_ptr<SortedNodeSampleTracker> sorted_node_sample_tracker_;
  /*! \brief Pointer to prediction dataset */
  std::unique_ptr<Dataset> prediction_dataset_;
  /*! \brief Pointer to model */
  std::unique_ptr<Model> model_;
  /*! \brief Pointer to draws of the model */
  std::vector<std::unique_ptr<ModelDraw>> model_draws_;
  /*! \brief Leaf ids for each observation in every tree of an ensemble */
  std::vector<std::vector<data_size_t>> tree_observation_indices_;
  /*! \brief Pointer to tree ensemble grown greedily */
  std::unique_ptr<TreeEnsemble> greedy_tree_ensemble_;
  /*! \brief Sample XBART gaussian regression */
  void SampleXBARTGaussianRegression();
  /*! \brief Sample BART gaussian regression */
  void SampleBARTGaussianRegression();
};

} // namespace StochTree

#endif  // STOCHTREE_INTERFACE_H_
