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

/*! \brief API for loading and accessing data */
class ForestDataset {
 public:
  /*!
   * \brief Default constructor. No data is loaded at construction time.
   */
  ForestDataset() {}
  ~ForestDataset() {}
  /*!
   * \brief Copy / load covariates from raw memory buffer (often pointer to data in a R matrix or numpy array)
   * 
   * \param data_ptr Pointer to first element of a contiguous array of data storing a covariate matrix
   * \param num_row Number of rows in the covariate matrix
   * \param num_col Number of columns / covariates in the covariate matrix
   * \param is_row_major Whether or not the data in `data_ptr` are organized in a row-major or column-major fashion
   */
  void AddCovariates(double* data_ptr, data_size_t num_row, int num_col, bool is_row_major) {
    covariates_ = ColumnMatrix(data_ptr, num_row, num_col, is_row_major);
    num_observations_ = num_row;
    num_covariates_ = num_col;
    has_covariates_ = true;
  }
  /*!
   * \brief Copy / load basis matrix from raw memory buffer (often pointer to data in a R matrix or numpy array)
   * 
   * \param data_ptr Pointer to first element of a contiguous array of data storing a basis matrix
   * \param num_row Number of rows in the basis matrix
   * \param num_col Number of columns in the basis matrix
   * \param is_row_major Whether or not the data in `data_ptr` are organized in a row-major or column-major fashion
   */
  void AddBasis(double* data_ptr, data_size_t num_row, int num_col, bool is_row_major) {
    basis_ = ColumnMatrix(data_ptr, num_row, num_col, is_row_major);
    num_basis_ = num_col;
    has_basis_ = true;
  }
  /*!
   * \brief Copy / load variance weights from raw memory buffer (often pointer to data in a R vector or numpy array)
   * 
   * \param data_ptr Pointer to first element of a contiguous array of data storing weights
   * \param num_row Number of rows in the weight vector
   */
  void AddVarianceWeights(double* data_ptr, data_size_t num_row) {
    var_weights_ = ColumnVector(data_ptr, num_row);
    has_var_weights_ = true;
  }
  /*! \brief Whether or not a `ForestDataset` has (yet) loaded covariate data */
  inline bool HasCovariates() {return has_covariates_;}
  /*! \brief Whether or not a `ForestDataset` has (yet) loaded basis data */
  inline bool HasBasis() {return has_basis_;}
  /*! \brief Whether or not a `ForestDataset` has (yet) loaded variance weights */
  inline bool HasVarWeights() {return has_var_weights_;}
  /*! \brief Number of observations (rows) in the dataset */
  inline data_size_t NumObservations() {return num_observations_;}
  /*! \brief Number of covariate columns in the dataset */
  inline int NumCovariates() {return num_covariates_;}
  /*! \brief Number of bases in the dataset. This is 0 if the dataset has not been provided a basis matrix. */
  inline int NumBasis() {return num_basis_;}
  /*!
   * \brief Returns a dataset's covariate value stored at (`row`, `col`)
   * 
   * \param row Row number to query in the covariate matrix
   * \param col Column number to query in the covariate matrix
   */
  inline double CovariateValue(data_size_t row, int col) {return covariates_.GetElement(row, col);}
  /*!
   * \brief Returns a dataset's basis value stored at (`row`, `col`)
   * 
   * \param row Row number to query in the basis matrix
   * \param col Column number to query in the basis matrix
   */
  inline double BasisValue(data_size_t row, int col) {return basis_.GetElement(row, col);}
  /*!
   * \brief Returns a dataset's variance weight stored at element `row`
   * 
   * \param row Index to query in the weight vector
   */
  inline double VarWeightValue(data_size_t row) {return var_weights_.GetElement(row);}
  /*!
   * \brief Return a reference to the raw `Eigen::MatrixXd` storing the covariate data
   * 
   * \return Eigen::MatrixXd& by reference
   */
  inline Eigen::MatrixXd& GetCovariates() {return covariates_.GetData();}
  /*!
   * \brief Return a reference to the raw `Eigen::MatrixXd` storing the basis data
   * 
   * \return Eigen::MatrixXd& by reference
   */
  inline Eigen::MatrixXd& GetBasis() {return basis_.GetData();}
  /*!
   * \brief Return a reference to the raw `Eigen::VectorXd` storing the variance weights
   * 
   * \return Eigen::VectorXd& by reference
   */
  inline Eigen::VectorXd& GetVarWeights() {return var_weights_.GetData();}
  /*!
   * \brief Update the data in the internal basis matrix to new values stored in a raw double array
   * 
   * \param data_ptr Pointer to first element of a contiguous array of data storing a basis matrix
   * \param num_row Number of rows in the basis matrix
   * \param num_col Number of columns in the basis matrix
   * \param is_row_major Whether or not the data in `data_ptr` are organized in a row-major or column-major fashion
   */
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
  data_size_t num_observations_{0};
  int num_covariates_{0};
  int num_basis_{0};
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
  inline data_size_t NumObservations() {return basis_.NumRows();}
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
