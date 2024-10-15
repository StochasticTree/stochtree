/*!
 * Copyright (c) 2024 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_TREE_KERNEL_H_
#define STOCHTREE_TREE_KERNEL_H_

#include <stochtree/data.h>
#include <stochtree/ensemble.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <cmath>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

namespace StochTree {

typedef Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> KernelMatrixType;

class ForestKernel {
 public:
  ForestKernel() {}
  ~ForestKernel() {}

  void ComputeLeafIndices(Eigen::MatrixXd& covariates, TreeEnsemble& forest) {
    num_train_observations_ = covariates.rows();
    num_trees_ = forest.NumTrees();
    train_leaf_index_vector_.resize(num_train_observations_*num_trees_);
    forest.PredictLeafIndicesInplace(covariates, train_leaf_index_vector_, num_trees_, num_train_observations_);
    int max_cols = *std::max_element(train_leaf_index_vector_.begin(), train_leaf_index_vector_.end());
    train_leaf_index_matrix_ = Eigen::SparseMatrix<double>(num_train_observations_,max_cols+1);
    int col_num;
    for (data_size_t i = 0; i < num_train_observations_; i++) {
        for (int j = 0; j < num_trees_; j++) {
            col_num = train_leaf_index_vector_.at(j*num_train_observations_ + i);
            train_leaf_index_matrix_.insert(i,col_num) = 1.;
        }
    }
    train_leaf_indices_stored_ = true;
  }

  void ComputeLeafIndices(KernelMatrixType& covariates, TreeEnsemble& forest) {
    num_train_observations_ = covariates.rows();
    num_trees_ = forest.NumTrees();
    train_leaf_index_vector_.resize(num_train_observations_*num_trees_);
    forest.PredictLeafIndicesInplace(covariates, train_leaf_index_vector_, num_trees_, num_train_observations_);
    int max_cols = *std::max_element(train_leaf_index_vector_.begin(), train_leaf_index_vector_.end());
    train_leaf_index_matrix_ = Eigen::SparseMatrix<double>(num_train_observations_,max_cols+1);
    int col_num;
    for (data_size_t i = 0; i < num_train_observations_; i++) {
        for (int j = 0; j < num_trees_; j++) {
            col_num = train_leaf_index_vector_.at(j*num_train_observations_ + i);
            train_leaf_index_matrix_.insert(i,col_num) = 1.;
        }
    }
    train_leaf_indices_stored_ = true;
  }

  void ComputeLeafIndices(Eigen::MatrixXd& covariates_train, Eigen::MatrixXd& covariates_test, TreeEnsemble& forest) {
    CHECK_EQ(covariates_train.cols(), covariates_test.cols());
    num_train_observations_ = covariates_train.rows();
    num_test_observations_ = covariates_test.rows();
    num_trees_ = forest.NumTrees();
    train_leaf_index_vector_.resize(num_train_observations_*num_trees_);
    test_leaf_index_vector_.resize(num_test_observations_*num_trees_);
    forest.PredictLeafIndicesInplace(covariates_train, train_leaf_index_vector_, num_trees_, num_train_observations_);
    forest.PredictLeafIndicesInplace(covariates_test, test_leaf_index_vector_, num_trees_, num_test_observations_);
    int max_cols_train = *std::max_element(train_leaf_index_vector_.begin(), train_leaf_index_vector_.end());
    int max_cols_test = *std::max_element(test_leaf_index_vector_.begin(), test_leaf_index_vector_.end());
    int max_cols = max_cols_train > max_cols_test ? max_cols_train : max_cols_test;
    train_leaf_index_matrix_ = Eigen::SparseMatrix<double>(num_train_observations_,max_cols+1);
    test_leaf_index_matrix_ = Eigen::SparseMatrix<double>(num_test_observations_,max_cols+1);
    int col_num;
    for (data_size_t i = 0; i < num_train_observations_; i++) {
        for (int j = 0; j < num_trees_; j++) {
            col_num = train_leaf_index_vector_.at(j*num_train_observations_ + i);
            train_leaf_index_matrix_.insert(i,col_num) = 1.;
        }
    }
    train_leaf_indices_stored_ = true;
    for (data_size_t i = 0; i < num_test_observations_; i++) {
        for (int j = 0; j < num_trees_; j++) {
            col_num = test_leaf_index_vector_.at(j*num_test_observations_ + i);
            test_leaf_index_matrix_.insert(i,col_num) = 1.;
        }
    }
    test_leaf_indices_stored_ = true;
  }

  void ComputeLeafIndices(KernelMatrixType& covariates_train, KernelMatrixType& covariates_test, TreeEnsemble& forest) {
    CHECK_EQ(covariates_train.cols(), covariates_test.cols());
    num_train_observations_ = covariates_train.rows();
    num_test_observations_ = covariates_test.rows();
    num_trees_ = forest.NumTrees();
    train_leaf_index_vector_.resize(num_train_observations_*num_trees_);
    test_leaf_index_vector_.resize(num_test_observations_*num_trees_);
    forest.PredictLeafIndicesInplace(covariates_train, train_leaf_index_vector_, num_trees_, num_train_observations_);
    forest.PredictLeafIndicesInplace(covariates_test, test_leaf_index_vector_, num_trees_, num_test_observations_);
    int max_cols_train = *std::max_element(train_leaf_index_vector_.begin(), train_leaf_index_vector_.end());
    int max_cols_test = *std::max_element(test_leaf_index_vector_.begin(), test_leaf_index_vector_.end());
    int max_cols = max_cols_train > max_cols_test ? max_cols_train : max_cols_test;
    train_leaf_index_matrix_ = Eigen::SparseMatrix<double>(num_train_observations_,max_cols+1);
    test_leaf_index_matrix_ = Eigen::SparseMatrix<double>(num_test_observations_,max_cols+1);
    int col_num;
    for (data_size_t i = 0; i < num_train_observations_; i++) {
        for (int j = 0; j < num_trees_; j++) {
            col_num = train_leaf_index_vector_.at(j*num_train_observations_ + i);
            train_leaf_index_matrix_.insert(i,col_num) = 1.;
        }
    }
    train_leaf_indices_stored_ = true;
    for (data_size_t i = 0; i < num_test_observations_; i++) {
        for (int j = 0; j < num_trees_; j++) {
            col_num = test_leaf_index_vector_.at(j*num_test_observations_ + i);
            test_leaf_index_matrix_.insert(i,col_num) = 1.;
        }
    }
    test_leaf_indices_stored_ = true;
  }

  void ComputeKernel(Eigen::MatrixXd& covariates, TreeEnsemble& forest) {
    ComputeLeafIndices(covariates, forest);
    tree_kernel_train_ = train_leaf_index_matrix_ * train_leaf_index_matrix_.transpose();
    train_kernel_stored_ = true;
  }

  void ComputeKernel(KernelMatrixType& covariates, TreeEnsemble& forest) {
    ComputeLeafIndices(covariates, forest);
    tree_kernel_train_ = train_leaf_index_matrix_ * train_leaf_index_matrix_.transpose();
    train_kernel_stored_ = true;
  }

  void ComputeKernelExternal(Eigen::MatrixXd& covariates, TreeEnsemble& forest, KernelMatrixType& kernel_map) {
    ComputeLeafIndices(covariates, forest);
    kernel_map = train_leaf_index_matrix_ * train_leaf_index_matrix_.transpose();
  }

  void ComputeKernelExternal(KernelMatrixType& covariates, TreeEnsemble& forest, KernelMatrixType& kernel_map) {
    ComputeLeafIndices(covariates, forest);
    kernel_map = train_leaf_index_matrix_ * train_leaf_index_matrix_.transpose();
  }

  void ComputeKernel(Eigen::MatrixXd& covariates_train, Eigen::MatrixXd& covariates_test, TreeEnsemble& forest) {
    ComputeLeafIndices(covariates_train, covariates_test, forest);
    tree_kernel_train_ = train_leaf_index_matrix_ * train_leaf_index_matrix_.transpose();
    train_kernel_stored_ = true;
    tree_kernel_test_train_ = test_leaf_index_matrix_ * train_leaf_index_matrix_.transpose();
    tree_kernel_test_ = test_leaf_index_matrix_ * test_leaf_index_matrix_.transpose();
    test_kernel_stored_ = true;
  }

  void ComputeKernel(KernelMatrixType& covariates_train, KernelMatrixType& covariates_test, TreeEnsemble& forest) {
    ComputeLeafIndices(covariates_train, covariates_test, forest);
    tree_kernel_train_ = train_leaf_index_matrix_ * train_leaf_index_matrix_.transpose();
    train_kernel_stored_ = true;
    tree_kernel_test_train_ = test_leaf_index_matrix_ * train_leaf_index_matrix_.transpose();
    tree_kernel_test_ = test_leaf_index_matrix_ * test_leaf_index_matrix_.transpose();
    test_kernel_stored_ = true;
  }

  void ComputeKernelExternal(Eigen::MatrixXd& covariates_train, Eigen::MatrixXd& covariates_test, TreeEnsemble& forest, 
                             KernelMatrixType& kernel_map_train, KernelMatrixType& kernel_map_test_train, KernelMatrixType& kernel_map_test) {
    ComputeLeafIndices(covariates_train, covariates_test, forest);
    kernel_map_train = train_leaf_index_matrix_ * train_leaf_index_matrix_.transpose();
    kernel_map_test_train = test_leaf_index_matrix_ * train_leaf_index_matrix_.transpose();
    kernel_map_test = test_leaf_index_matrix_ * test_leaf_index_matrix_.transpose();
  }

  void ComputeKernelExternal(KernelMatrixType& covariates_train, KernelMatrixType& covariates_test, TreeEnsemble& forest, 
                             KernelMatrixType& kernel_map_train, KernelMatrixType& kernel_map_test_train, KernelMatrixType& kernel_map_test) {
    ComputeLeafIndices(covariates_train, covariates_test, forest);
    kernel_map_train = train_leaf_index_matrix_ * train_leaf_index_matrix_.transpose();
    kernel_map_test_train = test_leaf_index_matrix_ * train_leaf_index_matrix_.transpose();
    kernel_map_test = test_leaf_index_matrix_ * test_leaf_index_matrix_.transpose();
  }

  std::vector<int32_t>& GetTrainLeafIndices() {
    CHECK(train_leaf_indices_stored_); 
    return train_leaf_index_vector_;
  }

  std::vector<int32_t>& GetTestLeafIndices() {
    CHECK(test_leaf_indices_stored_); 
    return test_leaf_index_vector_;
  }

  Eigen::MatrixXd& GetTrainKernel() {
    CHECK(train_kernel_stored_); 
    return tree_kernel_train_;
  }

  Eigen::MatrixXd& GetTestTrainKernel() {
    CHECK(test_kernel_stored_); 
    return tree_kernel_test_train_;
  }

  Eigen::MatrixXd& GetTestKernel() {
    CHECK(test_kernel_stored_); 
    return tree_kernel_test_;
  }

  data_size_t NumTrainObservations() {
    return num_train_observations_;
  }

  data_size_t NumTestObservations() {
    return num_test_observations_;
  }

  int NumTrees() {
    return num_trees_;
  }

  bool HasTrainLeafIndices() {
    return train_leaf_indices_stored_;
  }

  bool HasTestLeafIndices() {
    return test_leaf_indices_stored_;
  }

  bool HasTrainKernel() {
    return train_kernel_stored_;
  }

  bool HasTestKernel() {
    return test_kernel_stored_;
  }

 private:
  data_size_t num_train_observations_{0};
  data_size_t num_test_observations_{0};
  int num_trees_{0};
  std::vector<int32_t> train_leaf_index_vector_;
  std::vector<int32_t> test_leaf_index_vector_;
  Eigen::SparseMatrix<double> train_leaf_index_matrix_;
  Eigen::SparseMatrix<double> test_leaf_index_matrix_;
  Eigen::MatrixXd tree_kernel_train_;
  Eigen::MatrixXd tree_kernel_test_train_;
  Eigen::MatrixXd tree_kernel_test_;
  bool train_leaf_indices_stored_{false};
  bool test_leaf_indices_stored_{false};
  bool train_kernel_stored_{false};
  bool test_kernel_stored_{false};
};

} // namespace StochTree

#endif // STOCHTREE_TREE_KERNEL_H_