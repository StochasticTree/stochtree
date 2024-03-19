/*! Copyright (c) 2024 by stochtree authors */
#include <Eigen/Dense>
#include <stochtree/container.h>
#include <stochtree/data.h>

namespace StochTree {

ForestContainer::ForestContainer(int num_trees, int output_dimension, bool is_leaf_constant) {
  forests_ = std::vector<std::unique_ptr<TreeEnsemble>>(0);
  num_samples_ = 0;
  num_trees_ = num_trees;
  output_dimension_ = output_dimension;
  is_leaf_constant_ = is_leaf_constant;
  initialized_ = true;
}

ForestContainer::ForestContainer(int num_samples, int num_trees, int output_dimension, bool is_leaf_constant) {
  forests_ = std::vector<std::unique_ptr<TreeEnsemble>>(num_samples);
  for (auto& forest : forests_) {
    forest.reset(new TreeEnsemble(num_trees, output_dimension, is_leaf_constant));
  }
  num_samples_ = num_samples;
  num_trees_ = num_trees;
  output_dimension_ = output_dimension;
  is_leaf_constant_ = is_leaf_constant;
}

void ForestContainer::CopyFromPreviousSample(int new_sample_id, int previous_sample_id) {
  forests_[new_sample_id].reset(new TreeEnsemble(*forests_[previous_sample_id]));
}

void ForestContainer::AddSamples(int num_samples) {
  CHECK(initialized_);
  int total_new_samples = num_samples + num_samples_;
  forests_.resize(total_new_samples);
  for (int i = num_samples_; i < total_new_samples; i++) {
    forests_[i].reset(new TreeEnsemble(num_trees_, output_dimension_, is_leaf_constant_));
  }
  num_samples_ = total_new_samples;
}

std::vector<double> ForestContainer::Predict(ForestDataset& dataset) {
  data_size_t n = dataset.NumObservations();
  data_size_t total_output_size = n*num_samples_;
  std::vector<double> output(total_output_size);
  data_size_t offset = 0;
  for (int i = 0; i < num_samples_; i++) {
    auto num_trees = forests_[i]->NumTrees();
    forests_[i]->PredictInplace(dataset, output, 0, num_trees, offset);
    offset += n;
  }
  return output;
}

std::vector<double> ForestContainer::PredictRaw(ForestDataset& dataset) {
  data_size_t n = dataset.NumObservations();
  data_size_t total_output_size = n * output_dimension_ * num_samples_;
  std::vector<double> output(total_output_size);
  data_size_t offset = 0;
  for (int i = 0; i < num_samples_; i++) {
    auto num_trees = forests_[i]->NumTrees();
    forests_[i]->PredictRawInplace(dataset, output, 0, num_trees, offset);
    offset += n * output_dimension_;
  }
  return output;
}

} // namespace StochTree
