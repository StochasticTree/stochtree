/*! Copyright (c) 2024 by stochtree authors */
#include <Eigen/Dense>
#include <stochtree/data.h>
#include <iostream>

namespace StochTree {

ColumnMatrix::ColumnMatrix() : data_(NULL,1,1) {
  // Eigen::Map does not have a default initializer, 
  // so we initialize the data member with a null pointer 
  // and modify later
}

ColumnMatrix::ColumnMatrix(double* data_ptr, data_size_t num_row, int num_col, bool is_row_major) : data_(NULL,1,1) {
  LoadData(data_ptr, num_row, num_col, is_row_major);
}

void ColumnMatrix::LoadData(double* data_ptr, data_size_t num_row, int num_col, bool is_row_major) {
  new (&data_) MatrixMap(data_ptr,num_row,num_col);
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

} // namespace StochTree
