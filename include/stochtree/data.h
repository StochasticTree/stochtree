/*!
 * Copyright (c) 2024 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_DATA_H_
#define STOCHTREE_DATA_H_

#include <Eigen/Dense>
#include <stochtree/log.h>
#include <stochtree/meta.h>
#include <memory>

namespace StochTree {

class ColumnMatrix {
 public:
  ColumnMatrix() {}
  ColumnMatrix(double* data_ptr, data_size_t num_row, int num_col, bool is_row_major);
  ~ColumnMatrix() {}
  double GetElement(data_size_t row_num, int32_t col_num) {return data_(row_num, col_num);}
  void SetElement(data_size_t row_num, int32_t col_num, double value) {data_(row_num, col_num) = value;}
  void LoadData(double* data_ptr, data_size_t num_row, int num_col, bool is_row_major);
  inline data_size_t NumRows() {return data_.rows();}
  inline int NumCols() {return data_.cols();}
  inline Eigen::MatrixXd& GetData() {return data_;}
 private:
  Eigen::MatrixXd data_;
};

class ColumnVector {
 public:
  ColumnVector() {}
  ColumnVector(double* data_ptr, data_size_t num_row);
  ~ColumnVector() {}
  double GetElement(data_size_t row_num) {return data_(row_num);}
  void SetElement(data_size_t row_num, double value) {data_(row_num) = value;}
  void LoadData(double* data_ptr, data_size_t num_row);
  inline data_size_t NumRows() {return data_.size();}
  inline Eigen::VectorXd& GetData() {return data_;}
 private:
  Eigen::VectorXd data_;
};

class ForestDataset {
 public:
  ForestDataset() {}
  ~ForestDataset() {}
  void AddCovariates(double* data_ptr, data_size_t num_row, int num_col, bool is_row_major) {
    covariates_ = ColumnMatrix(data_ptr, num_row, num_col, is_row_major);
    num_observations_ = num_row;
    num_covariates_ = num_col;
    has_covariates_ = true;
  }
  void AddBasis(double* data_ptr, data_size_t num_row, int num_col, bool is_row_major) {
    basis_ = ColumnMatrix(data_ptr, num_row, num_col, is_row_major);
    num_basis_ = num_col;
    has_basis_ = true;
  }
  void AddVarianceWeights(double* data_ptr, data_size_t num_row) {
    var_weights_ = ColumnVector(data_ptr, num_row);
    has_var_weights_ = true;
  }
  inline bool HasCovariates() {return has_covariates_;}
  inline bool HasBasis() {return has_basis_;}
  inline bool HasVarWeights() {return has_var_weights_;}
  inline data_size_t NumObservations() {return num_observations_;}
  inline double CovariateValue(data_size_t row, int col) {return covariates_.GetElement(row, col);}
  inline double BasisValue(data_size_t row, int col) {return basis_.GetElement(row, col);}
  inline double VarWeightValue(data_size_t row) {return var_weights_.GetElement(row);}
  inline Eigen::MatrixXd& GetCovariates() {return covariates_.GetData();}
  inline Eigen::MatrixXd& GetBasis() {return basis_.GetData();}
  inline Eigen::VectorXd& GetVarWeights() {return var_weights_.GetData();}
  void UpdateBasis(double* data_ptr, data_size_t num_row, int num_col, bool is_row_major) {
    CHECK(has_basis_);
    CHECK_EQ(num_col, num_basis_);
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
        basis_.SetElement(i, j, temp_value);
      }
    }
  }
 private:
  ColumnMatrix covariates_;
  ColumnMatrix basis_;
  ColumnVector var_weights_;
  data_size_t num_observations_;
  int num_covariates_;
  int num_basis_;
  bool has_covariates_{false};
  bool has_basis_{false};
  bool has_var_weights_{false};
};

class RandomEffectsDataset {
 public:
  RandomEffectsDataset() {}
  ~RandomEffectsDataset() {}
  void AddBasis(double* data_ptr, data_size_t num_row, int num_col, bool is_row_major) {
    basis_ = ColumnMatrix(data_ptr, num_row, num_col, is_row_major);
    has_basis_ = true;
  }
  void AddVarianceWeights(double* data_ptr, data_size_t num_row) {
    var_weights_ = ColumnVector(data_ptr, num_row);
    has_var_weights_ = true;
  }
  void AddGroupLabels(std::vector<int32_t>& group_labels) {
    group_labels_ = group_labels;
    has_group_labels_ = true;
  }
  inline bool HasBasis() {return has_basis_;}
  inline bool HasVarWeights() {return has_var_weights_;}
  inline bool HasGroupLabels() {return has_group_labels_;}
  inline double BasisValue(data_size_t row, int col) {return basis_.GetElement(row, col);}
  inline double VarWeightValue(data_size_t row) {return var_weights_.GetElement(row);}
  inline int32_t GroupId(data_size_t row) {return group_labels_[row];}
  inline Eigen::MatrixXd& GetBasis() {return basis_.GetData();}
  inline Eigen::VectorXd& GetVarWeights() {return var_weights_.GetData();}
  inline std::vector<int32_t>& GetGroupLabels() {return group_labels_;}
 private:
  ColumnMatrix basis_;
  ColumnVector var_weights_;
  std::vector<int32_t> group_labels_;
  bool has_basis_{false};
  bool has_var_weights_{false};
  bool has_group_labels_{false};
};

} // namespace StochTree

#endif // STOCHTREE_DATA_H_
