/*!
 * Copyright (c) 2024 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_DATA_H_
#define STOCHTREE_DATA_H_

#include <Eigen/Dense>
#include <stochtree/io.h>
#include <stochtree/log.h>
#include <stochtree/meta.h>
#include <memory>

namespace StochTree {

/*! \brief Extract local features from memory */
static inline void ExtractMultipleFeaturesFromMemory(std::vector<std::string>* text_data, const Parser* parser,
                                                     std::vector<int32_t>& column_indices, Eigen::MatrixXd& data,
                                                     data_size_t num_rows) {
  std::vector<std::pair<int, double>> oneline_features;
  auto& ref_text_data = *text_data;
  int feature_counter;
  bool column_matched;
  for (data_size_t i = 0; i < num_rows; ++i) {
    // unpack the vector of textlines read from file into a vector of (int, double) tuples
    oneline_features.clear();
    parser->ParseOneLine(ref_text_data[i].c_str(), &oneline_features);
    
    // free processed line:
    ref_text_data[i].clear();

    // unload the data from oneline_features vector into the dataset variables containers
    int feature_counter = 0;
    for (auto& inner_data : oneline_features) {
      int feature_idx = inner_data.first;
      column_matched = (std::find(column_indices.begin(), column_indices.end(), feature_idx)
                        != column_indices.end());
      if (column_matched){
        data(i, feature_counter) = inner_data.second;
        feature_counter += 1;
      }
    }
  }
  // free text data after use
  text_data->clear();
}

/*! \brief Extract local features from memory */
static inline void ExtractSingleFeatureFromMemory(std::vector<std::string>* text_data, const Parser* parser,
                                                  int32_t column_index, Eigen::VectorXd& data, data_size_t num_rows) {
  std::vector<std::pair<int, double>> oneline_features;
  auto& ref_text_data = *text_data;
  bool column_matched;
  for (data_size_t i = 0; i < num_rows; ++i) {
    // unpack the vector of textlines read from file into a vector of (int, double) tuples
    oneline_features.clear();
    parser->ParseOneLine(ref_text_data[i].c_str(), &oneline_features);
    
    // free processed line:
    ref_text_data[i].clear();

    // unload the data from oneline_features vector into the dataset variables containers
    for (auto& inner_data : oneline_features) {
      int feature_idx = inner_data.first;
      if (column_index == feature_idx){
        data(i) = inner_data.second;
      }
    }
  }
  // free text data after use
  text_data->clear();
}

static inline std::vector<std::string> LoadTextDataToMemory(const char* filename, int* num_global_data, bool header) {
  size_t file_load_progress_interval_bytes = size_t(10) * 1024 * 1024 * 1024;
  TextReader<data_size_t> text_reader(filename, header, file_load_progress_interval_bytes);
  // read all lines
  *num_global_data = text_reader.ReadAllLines();
  return std::move(text_reader.Lines());
}

static inline void FeatureUnpack(std::vector<int32_t>& categorical_variables, const char* var_id) {
  std::string var_clean = Common::RemoveQuotationSymbol(Common::Trim(var_id));
  int out;
  bool success = Common::AtoiAndCheck(var_clean.c_str(), &out);
  if (success) {
    categorical_variables.push_back(out);
  } else {
    Log::Warning("Parsed variable index %s cannot be cast to an integer", var_clean.c_str());
  }
}

static inline std::vector<int> Str2FeatureVec(const char* parameters) {
  std::vector<int> feature_vec;
  auto args = Common::Split(parameters, ",");
  for (auto arg : args) {
    FeatureUnpack(feature_vec, Common::Trim(arg).c_str());
  }
  return feature_vec;
}

class ColumnMatrix {
 public:
  ColumnMatrix() {}
  ColumnMatrix(double* data_ptr, data_size_t num_row, int num_col, bool is_row_major);
  ColumnMatrix(std::string filename, std::string column_index_string, bool header = true, bool precise_float_parser = false);
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
  ColumnVector(std::string filename, int32_t column_index, bool header = true, bool precise_float_parser = false);
  ~ColumnVector() {}
  double GetElement(data_size_t row_num) {return data_(row_num);}
  void SetElement(data_size_t row_num, double value) {data_(row_num) = value;}
  void LoadData(double* data_ptr, data_size_t num_row);
  void AddToData(double* data_ptr, data_size_t num_row);
  void SubtractFromData(double* data_ptr, data_size_t num_row);
  void OverwriteData(double* data_ptr, data_size_t num_row);
  inline data_size_t NumRows() {return data_.size();}
  inline Eigen::VectorXd& GetData() {return data_;}
 private:
  Eigen::VectorXd data_;
  void UpdateData(double* data_ptr, data_size_t num_row, std::function<double(double, double)> op);
};

/*! \brief API for loading and accessing data used to sample tree ensembles */
class ForestDataset {
 public:
  /*! \brief Default constructor. No data is loaded at construction time. */
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
  /*!
   * \brief Copy / load covariates from CSV file
   * 
   * \param filename Name of the file (including any necessary path prefixes)
   * \param column_index_string Comma-delimited string listing columns to extract into covariates matrix
   */
  void AddCovariatesFromCSV(std::string filename, std::string column_index_string, bool header = true, bool precise_float_parser = false) {
    covariates_ = ColumnMatrix(filename, column_index_string, header, precise_float_parser);
    num_observations_ = covariates_.NumRows();
    num_covariates_ = covariates_.NumCols();
    has_covariates_ = true;
  }
  /*!
   * \brief Copy / load basis matrix from CSV file
   * 
   * \param filename Name of the file (including any necessary path prefixes)
   * \param column_index_string Comma-delimited string listing columns to extract into covariates matrix
   */
  void AddBasisFromCSV(std::string filename, std::string column_index_string, bool header = true, bool precise_float_parser = false) {
    basis_ = ColumnMatrix(filename, column_index_string, header, precise_float_parser);
    num_basis_ = basis_.NumCols();
    has_basis_ = true;
  }
  /*!
   * \brief Copy / load variance / case weights from CSV file
   * 
   * \param filename Name of the file (including any necessary path prefixes)
   * \param column_index Integer index of column containing weights
   */
  void AddVarianceWeightsFromCSV(std::string filename, int32_t column_index, bool header = true, bool precise_float_parser = false) {
    var_weights_ = ColumnVector(filename, column_index, header, precise_float_parser);
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
   * \return Reference to internal Eigen::MatrixXd
   */
  inline Eigen::MatrixXd& GetCovariates() {return covariates_.GetData();}
  /*!
   * \brief Return a reference to the raw `Eigen::MatrixXd` storing the basis data
   * 
   * \return Reference to internal Eigen::MatrixXd
   */
  inline Eigen::MatrixXd& GetBasis() {return basis_.GetData();}
  /*!
   * \brief Return a reference to the raw `Eigen::VectorXd` storing the variance weights
   * 
   * \return Reference to internal Eigen::VectorXd
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
  /*!
   * \brief Update the data in the internal variance weight vector to new values stored in a raw double array
   *
   * \param data_ptr Pointer to first element of a contiguous array of data storing a weight vector
   * \param num_row Number of rows in the weight vector
   * \param exponentiate Whether or not inputs should be exponentiated before being saved to var weight vector
   */
  void UpdateVarWeights(double* data_ptr, data_size_t num_row, bool exponentiate = true) {
    CHECK(has_var_weights_);
    // Copy data from R / Python process memory to Eigen vector
    double temp_value;
    for (data_size_t i = 0; i < num_row; ++i) {
      if (exponentiate) temp_value = std::exp(static_cast<double>(*(data_ptr + i)));
      else temp_value = static_cast<double>(*(data_ptr + i));
      var_weights_.SetElement(i, temp_value);
    }
  }
  /*!
   * \brief Update an observation in the internal variance weight vector to a new value
   *
   * \param row_id Row ID in the variance weight vector to be overwritten
   * \param new_value New variance weight value
   * \param exponentiate Whether or not input should be exponentiated before being saved to var weight vector
   */
  void SetVarWeightValue(data_size_t row_id, double new_value, bool exponentiate = true) {
    CHECK(has_var_weights_);
    if (exponentiate) var_weights_.SetElement(row_id, std::exp(new_value));
    else var_weights_.SetElement(row_id, new_value);
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

/*! \brief API for loading and accessing data used to sample (additive) random effects */
class RandomEffectsDataset {
 public:
  /*! \brief Default constructor. No data is loaded at construction time. */
  RandomEffectsDataset() {}
  ~RandomEffectsDataset() {}
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
  /*!
   * \brief Copy / load group indices for random effects
   * 
   * \param group_labels Vector of integers with as many elements as `num_row` in the basis matrix, 
   * where each element corresponds to the group label for a given observation.
   */
  void AddGroupLabels(std::vector<int32_t>& group_labels) {
    group_labels_ = group_labels;
    has_group_labels_ = true;
  }
  /*! \brief Number of observations (rows) in the dataset */
  inline data_size_t NumObservations() {return basis_.NumRows();}
  /*! \brief Whether or not a `RandomEffectsDataset` has (yet) loaded basis data */
  inline bool HasBasis() {return has_basis_;}
  /*! \brief Whether or not a `RandomEffectsDataset` has (yet) loaded variance weights */
  inline bool HasVarWeights() {return has_var_weights_;}
  /*! \brief Whether or not a `RandomEffectsDataset` has (yet) loaded group labels */
  inline bool HasGroupLabels() {return has_group_labels_;}
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
   * \brief Returns a dataset's group label stored at element `row`
   * 
   * \param row Index to query in the group label vector
   */
  inline int32_t GroupId(data_size_t row) {return group_labels_[row];}
  /*!
   * \brief Return a reference to the raw `Eigen::MatrixXd` storing the basis data
   * 
   * \return Reference to internal Eigen::MatrixXd
   */
  inline Eigen::MatrixXd& GetBasis() {return basis_.GetData();}
  /*!
   * \brief Return a reference to the raw `Eigen::VectorXd` storing the variance weights
   * 
   * \return Reference to internal Eigen::VectorXd
   */
  inline Eigen::VectorXd& GetVarWeights() {return var_weights_.GetData();}
  /*!
   * \brief Return a reference to the raw `std::vector` storing the group labels
   * 
   * \return Reference to internal std::vector
   */
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