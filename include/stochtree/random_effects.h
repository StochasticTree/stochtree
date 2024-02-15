/*!
 * Copyright (c) 2024 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_RANDOM_EFFECTS_H_
#define STOCHTREE_RANDOM_EFFECTS_H_

#include <stochtree/data.h>
#include <stochtree/log.h>
#include <stochtree/sampler.h>
#include <Eigen/Dense>

#include <cmath>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace StochTree {

class RandomEffectsPersisted {
 public: 
  RandomEffectsPersisted() {
    default_rfx_ = true;
  }
  RandomEffectsPersisted(RandomEffectsSampler& rfx_sampler) {
    default_rfx_ = false;
    alpha_ = rfx_sampler.alpha_;
    xi_ = rfx_sampler.xi_;
    num_components_ = rfx_sampler.num_components_;
    num_groups_ = rfx_sampler.num_groups_;
    label_map_ = rfx_sampler.label_map_;
  }
  ~RandomEffectsPersisted() {}
  
  Eigen::VectorXd Predict(RegressionRandomEffectsDataset* rfx_dataset) {
    return Predict(rfx_dataset->basis, rfx_dataset->group_indices);
  }

  void PredictInplace(RegressionRandomEffectsDataset* rfx_dataset, std::vector<double>& output, data_size_t offset = 0) {
    PredictInplaceSampled(rfx_dataset->basis, rfx_dataset->group_indices, output, offset);
  }
  
  Eigen::VectorXd Predict(Eigen::MatrixXd& X, std::vector<int32_t>& group_labels) {
    if (default_rfx_) {
      return PredictDefault(X.rows());
    } else {
      return PredictSampled(X, group_labels);
    }
  }

  void PredictInplace(Eigen::MatrixXd& X, std::vector<int32_t>& group_labels, std::vector<double>& output, data_size_t offset = 0) {
    if (default_rfx_) {
      PredictInplaceDefault(X.rows(), output, offset);
    } else {
      PredictInplaceSampled(X, group_labels, output, offset);
    }
  }
  
  Eigen::VectorXd PredictDefault(int n) {
    Eigen::VectorXd result(n);
    for (int i = 0; i < n; i++) {
      result(i) = 0.;
    }
    return result;
  }
  
  void PredictInplaceDefault(int n, std::vector<double>& output, data_size_t offset = 0) {
    CHECK_GT(output.size(), n + offset);
    for (int i = 0; i < n; i++) {
      output[i + offset] = 0.;
    }
  }
  
  Eigen::VectorXd PredictSampled(Eigen::MatrixXd& X, std::vector<int32_t>& group_labels) {
    CHECK_EQ(X.rows(), group_labels.size());
    int n = X.rows();
    Eigen::VectorXd result(n);
    Eigen::MatrixXd alpha_diag = alpha_.asDiagonal().toDenseMatrix();
    std::uint64_t group_ind;
    for (int i = 0; i < n; i++) {
      group_ind = label_map_[group_labels[i]];
      result(i) = X(i, Eigen::all) * alpha_diag * xi_(Eigen::all, group_ind);
    }
    return result;
  }
  
  void PredictInplaceSampled(Eigen::MatrixXd& X, std::vector<int32_t>& group_labels, std::vector<double>& output, data_size_t offset = 0) {
    CHECK_EQ(X.rows(), group_labels.size());
    int n = X.rows();
    CHECK_GT(output.size(), n + offset);
    Eigen::MatrixXd alpha_diag = alpha_.asDiagonal().toDenseMatrix();
    std::uint64_t group_ind;
    for (int i = 0; i < n; i++) {
      group_ind = label_map_[group_labels[i]];
      output[i + offset] = X(i, Eigen::all) * alpha_diag * xi_(Eigen::all, group_ind);
    }
  }

 private:
  bool default_rfx_;
  int num_components_;
  int num_groups_;
  Eigen::VectorXd alpha_;
  Eigen::MatrixXd xi_;
  std::map<int32_t, uint32_t> label_map_;
};

class RandomEffectsContainer {
 public:
  RandomEffectsContainer(int num_samples) {
    rfx_ = std::vector<std::unique_ptr<RandomEffectsPersisted>>(num_samples);
    num_samples_ = num_samples;
  }
  ~RandomEffectsContainer() {}

  void AddSamples(int num_new_samples) {
    int total_new_samples = num_new_samples + num_samples_;
    rfx_.resize(total_new_samples);
    for (int i = num_samples_; i < total_new_samples; i++) {
      rfx_[i].reset(new RandomEffectsPersisted());
    }
    num_samples_ = total_new_samples;
  }

  void ResetSample(RandomEffectsSampler* sampler, int sample_num) {
    rfx_[sample_num].reset(new RandomEffectsPersisted(*sampler));
  }

  inline void CopyRandomEffect(int i, RandomEffectsSampler* rfx_model) {
    return rfx_[i].reset(new RandomEffectsPersisted(*rfx_model));
  }
  
  Eigen::VectorXd Predict(int i, RegressionRandomEffectsDataset* rfx_dataset) {
    return rfx_[i]->Predict(rfx_dataset);
  }

  void PredictInplace(int i, RegressionRandomEffectsDataset* rfx_dataset, std::vector<double>& output, data_size_t offset = 0) {
    rfx_[i]->PredictInplace(rfx_dataset, output, offset);
  }

 private:
  std::vector<std::unique_ptr<RandomEffectsPersisted>> rfx_;
  int num_samples_;
};

} // namespace StochTree

#endif // STOCHTREE_RANDOM_EFFECTS_H_
