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
  initialized_ = true;
}

void ForestContainer::CopyFromPreviousSample(int new_sample_id, int previous_sample_id) {
  forests_[new_sample_id].reset(new TreeEnsemble(*forests_[previous_sample_id]));
}

void ForestContainer::InitializeRoot(double leaf_value) {
  CHECK(initialized_);
  CHECK_EQ(num_samples_, 0);
  CHECK_EQ(forests_.size(), 0);
  forests_.resize(1);
  forests_[0].reset(new TreeEnsemble(num_trees_, output_dimension_, is_leaf_constant_));
  // NOTE: not setting num_samples = 1, since we are just initializing constant root 
  // nodes and the forest still needs to be sampled by either MCMC or GFR
  num_samples_ = 0;
  SetLeafValue(0, leaf_value);
}

void ForestContainer::InitializeRoot(std::vector<double>& leaf_vector) {
  CHECK(initialized_);
  CHECK_EQ(num_samples_, 0);
  CHECK_EQ(forests_.size(), 0);
  forests_.resize(1);
  forests_[0].reset(new TreeEnsemble(num_trees_, output_dimension_, is_leaf_constant_));
  // NOTE: not setting num_samples = 1, since we are just initializing constant root 
  // nodes and the forest still needs to be sampled by either MCMC or GFR
  num_samples_ = 0;
  SetLeafVector(0, leaf_vector);
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

std::vector<double> ForestContainer::PredictRaw(ForestDataset& dataset, int forest_num) {
  data_size_t n = dataset.NumObservations();
  data_size_t total_output_size = n * output_dimension_;
  std::vector<double> output(total_output_size);
  data_size_t offset = 0;
  auto num_trees = forests_[forest_num]->NumTrees();
  forests_[forest_num]->PredictRawInplace(dataset, output, 0, num_trees, offset);
  return output;
}

void ForestContainer::PredictRawInplace(ForestDataset& dataset, VectorMap& output) {
  data_size_t n = dataset.NumObservations();
  // data_size_t total_output_size = n * output_dimension_ * num_samples_;
  // std::vector<double> output(total_output_size);
  data_size_t offset = 0;
  for (int i = 0; i < num_samples_; i++) {
    auto num_trees = forests_[i]->NumTrees();
    forests_[i]->PredictRawInplace(dataset, output, 0, num_trees, offset);
    offset += n * output_dimension_;
  }
}

void ForestContainer::PredictRawInplace(ForestDataset& dataset, VectorMap& output, int forest_num) {
  data_size_t n = dataset.NumObservations();
  // data_size_t total_output_size = n * output_dimension_;
  // std::vector<double> output(total_output_size);
  data_size_t offset = 0;
  auto num_trees = forests_[forest_num]->NumTrees();
  forests_[forest_num]->PredictRawInplace(dataset, output, 0, num_trees, offset);
}

void ForestContainer::PredictRawInplace(ForestDataset& dataset, VectorMap& output, VectorMap& scalar_multiple) {
  CHECK_EQ(scalar_multiple.size(), num_samples_);
  data_size_t n = dataset.NumObservations();
  // data_size_t total_output_size = n * output_dimension_ * num_samples_;
  // std::vector<double> output(total_output_size);
  data_size_t offset = 0;
  for (int i = 0; i < num_samples_; i++) {
    auto num_trees = forests_[i]->NumTrees();
    forests_[i]->PredictRawInplace(dataset, output, scalar_multiple(i), 0, num_trees, offset);
    offset += n * output_dimension_;
  }
}

void ForestContainer::PredictRawInplace(ForestDataset& dataset, VectorMap& output, VectorMap& scalar_multiple, int forest_num) {
  CHECK_EQ(scalar_multiple.size(), num_samples_);
  data_size_t n = dataset.NumObservations();
  // data_size_t total_output_size = n * output_dimension_;
  // std::vector<double> output(total_output_size);
  data_size_t offset = 0;
  auto num_trees = forests_[forest_num]->NumTrees();
  forests_[forest_num]->PredictRawInplace(dataset, output, scalar_multiple(forest_num), 0, num_trees, offset);
}

void ForestContainer::PredictRawInplace(ForestDataset& dataset, VectorMap& output, VectorMap& scalar_multiple_minuend, VectorMap& scalar_multiple_subtrahend) {
  CHECK_EQ(scalar_multiple_minuend.size(), num_samples_);
  CHECK_EQ(scalar_multiple_subtrahend.size(), num_samples_);
  data_size_t n = dataset.NumObservations();
  // data_size_t total_output_size = n * output_dimension_ * num_samples_;
  // std::vector<double> output(total_output_size);
  data_size_t offset = 0;
  double mult;
  for (int i = 0; i < num_samples_; i++) {
    mult = scalar_multiple_minuend(i) - scalar_multiple_subtrahend(i);
    auto num_trees = forests_[i]->NumTrees();
    forests_[i]->PredictRawInplace(dataset, output, mult, 0, num_trees, offset);
    offset += n * output_dimension_;
  }
}

void ForestContainer::PredictRawInplace(ForestDataset& dataset, VectorMap& output, VectorMap& scalar_multiple_minuend, VectorMap& scalar_multiple_subtrahend, int forest_num) {
  CHECK_EQ(scalar_multiple_minuend.size(), num_samples_);
  CHECK_EQ(scalar_multiple_subtrahend.size(), num_samples_);
  data_size_t n = dataset.NumObservations();
  // data_size_t total_output_size = n * output_dimension_;
  // std::vector<double> output(total_output_size);
  data_size_t offset = 0;
  double mult = scalar_multiple_minuend(forest_num) - scalar_multiple_subtrahend(forest_num);
  auto num_trees = forests_[forest_num]->NumTrees();
  forests_[forest_num]->PredictRawInplace(dataset, output, mult, 0, num_trees, offset);
}

/*! \brief Save to JSON */
json ForestContainer::to_json() {
  json result_obj;
  result_obj.emplace("num_samples", this->num_samples_);
  result_obj.emplace("num_trees", this->num_trees_);
  result_obj.emplace("output_dimension", this->output_dimension_);
  result_obj.emplace("is_leaf_constant", this->is_leaf_constant_);
  result_obj.emplace("initialized", this->initialized_);

  std::string forest_label;
  for (int i = 0; i < forests_.size(); i++) {
    forest_label = "forest_" + std::to_string(i);
    result_obj.emplace(forest_label, forests_[i]->to_json());
  }
  
  return result_obj;
}

/*! \brief Load from JSON */
void ForestContainer::from_json(const json& forest_container_json) {
  this->num_samples_ = forest_container_json.at("num_samples");
  this->num_trees_ = forest_container_json.at("num_trees");
  this->output_dimension_ = forest_container_json.at("output_dimension");
  this->is_leaf_constant_ = forest_container_json.at("is_leaf_constant");
  this->initialized_ = forest_container_json.at("initialized");

  std::string forest_label;
  forests_.clear();
  forests_.resize(this->num_samples_);
  for (int i = 0; i < this->num_samples_; i++) {
    forest_label = "forest_" + std::to_string(i);
    forests_[i] = std::make_unique<TreeEnsemble>(this->num_trees_, this->output_dimension_, this->is_leaf_constant_);
    forests_[i]->from_json(forest_container_json.at(forest_label));
  }
}

} // namespace StochTree
