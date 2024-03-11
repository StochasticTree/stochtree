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
  bool basis_predict = dataset.HasBasis();
  if (basis_predict) {
    return Predict(dataset.GetCovariates(), dataset.GetBasis());
  } else {
    return Predict(dataset.GetCovariates());
  }
}

std::vector<double> ForestContainer::Predict(Eigen::MatrixXd& covariates) {
  CHECK(is_leaf_constant_);
  data_size_t n = covariates.rows();
  data_size_t total_output_size = n*num_samples_;
  std::vector<double> output(total_output_size);
  data_size_t offset = 0;
  for (int i = 0; i < num_samples_; i++) {
    auto num_trees = forests_[i]->NumTrees();
    forests_[i]->PredictInplace(covariates, output, 0, num_trees, offset);
    offset += n;
  }
  return output;
}

std::vector<double> ForestContainer::Predict(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis) {
  CHECK_EQ(covariates.rows(), basis.rows());
  CHECK(!is_leaf_constant_);
  data_size_t n = covariates.rows();
  data_size_t total_output_size = n*num_samples_;
  std::vector<double> output(total_output_size);
  data_size_t offset = 0;
  for (int i = 0; i < num_samples_; i++) {
    auto num_trees = forests_[i]->NumTrees();
    forests_[i]->PredictInplace(covariates, basis, output, 0, num_trees, offset);
    offset += n;
  }
  return output;
}

} // namespace StochTree
