/*! Copyright (c) 2024 by stochtree authors */
#include <Eigen/Dense>
#include <stochtree/data.h>
#include <iostream>

namespace StochTree {

ColumnMatrix::ColumnMatrix(double* data_ptr, data_size_t num_row, int num_col, bool is_row_major) {
  LoadData(data_ptr, num_row, num_col, is_row_major);
}

void ColumnMatrix::LoadData(double* data_ptr, data_size_t num_row, int num_col, bool is_row_major) {
  data_.resize(num_row, num_col);

  // Copy data from R / Python process memory to Eigen matrix
  double temp_value;
  for (data_size_t i = 0; i < num_row; ++i) {
    for (int j = 0; j < num_col; ++j) {
      if (is_row_major){
        // Numpy 2-d arrays are stored in "row major" order
        temp_value = static_cast<double>(*(data_ptr + static_cast<data_size_t>(num_col) * i + j));
      } else {
        // R matrices are stored in "column major" order
        temp_value = static_cast<double>(*(data_ptr + static_cast<data_size_t>(num_row) * j + i));
      }
      data_(i, j) = temp_value;
    }
  }
}

ColumnVector::ColumnVector(double* data_ptr, data_size_t num_row) {
  LoadData(data_ptr, num_row);
}

void ColumnVector::LoadData(double* data_ptr, data_size_t num_row) {
  data_.resize(num_row);

  // Copy data from R / Python process memory to Eigen matrix
  double temp_value;
  for (data_size_t i = 0; i < num_row; ++i) {
    temp_value = static_cast<double>(*(data_ptr + i));
    data_(i) = temp_value;
  }
}

void LoadData(double* data_ptr, int num_row, int num_col, bool is_row_major, Eigen::MatrixXd& data_matrix) {
  data_matrix.resize(num_row, num_col);

  // Copy data from R / Python process memory to Eigen matrix
  double temp_value;
  for (data_size_t i = 0; i < num_row; ++i) {
    for (int j = 0; j < num_col; ++j) {
      if (is_row_major){
        // Numpy 2-d arrays are stored in "row major" order
        temp_value = static_cast<double>(*(data_ptr + static_cast<data_size_t>(num_col) * i + j));
      } else {
        // R matrices are stored in "column major" order
        temp_value = static_cast<double>(*(data_ptr + static_cast<data_size_t>(num_row) * j + i));
      }
      data_matrix(i, j) = temp_value;
    }
  }
}

void LoadData(double* data_ptr, int num_row, Eigen::VectorXd& data_vector) {
  data_vector.resize(num_row);

  // Copy data from R / Python process memory to Eigen matrix
  double temp_value;
  for (data_size_t i = 0; i < num_row; ++i) {
    temp_value = static_cast<double>(*(data_ptr + i));
    data_vector(i) = temp_value;
  }
}

void RegressionRandomEffectsDataset::LoadFromMemory(double* basis_data_ptr, int num_basis, data_size_t num_row, bool is_row_major, std::vector<int32_t>& rfx_group_indices) {
  LoadData(basis_data_ptr, num_row, num_basis, is_row_major, basis);
  group_indices = rfx_group_indices;
}

} // namespace StochTree
