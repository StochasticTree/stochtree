/*!
 * Copyright (c) 2024 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_DATA_H_
#define STOCHTREE_DATA_H_

#include <Eigen/Dense>
#include <stochtree/meta.h>

namespace StochTree {

struct UnivariateResidual {
  Eigen::VectorXd residual;
  void LoadFromMemory(double* outcome_data_ptr, data_size_t num_row);
};

struct ForestDataset {
  virtual void LoadFromMemory(double* covariate_data_ptr, int num_covariate, data_size_t num_row, bool is_row_major) {};
  virtual void LoadFromMemory(double* covariate_data_ptr, int num_covariate, double* basis_data_ptr, int num_basis, data_size_t num_row, bool is_row_major) {};
};

struct ConstantLeafForestDataset : public ForestDataset {
  Eigen::MatrixXd covariates;
  void LoadFromMemory(double* covariate_data_ptr, int num_covariate, data_size_t num_row, bool is_row_major);
};

struct RegressionLeafForestDataset : public ForestDataset {
  Eigen::MatrixXd covariates;
  Eigen::MatrixXd basis;
  void LoadFromMemory(double* covariate_data_ptr, int num_covariate, double* basis_data_ptr, int num_basis, data_size_t num_row, bool is_row_major);
};

struct RandomEffectsDataset {
  virtual void LoadFromMemory(std::vector<int32_t>& rfx_group_indices) {};
  virtual void LoadFromMemory(double* basis_data_ptr, int num_basis, data_size_t num_row, bool is_row_major, std::vector<int32_t>& rfx_group_indices) {};
};

struct ConstantRandomEffectsDataset : public RandomEffectsDataset {
  std::vector<int32_t> group_indices;
  void LoadFromMemory(std::vector<int32_t>& rfx_group_indices);
};

struct RegressionRandomEffectsDataset : public RandomEffectsDataset {
  std::vector<int32_t> group_indices;
  Eigen::MatrixXd basis;
  void LoadFromMemory(double* basis_data_ptr, int num_basis, data_size_t num_row, bool is_row_major, std::vector<int32_t>& rfx_group_indices);
};

} // namespace StochTree

#endif // STOCHTREE_DATA_H_
