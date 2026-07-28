/*! Copyright (c) 2024 by stochtree authors */
#include <Eigen/Dense>
#include <stochtree/data.h>

namespace StochTree {

ColumnMatrix::ColumnMatrix(double* data_ptr, data_size_t num_row, int num_col, bool is_row_major) {
  LoadData(data_ptr, num_row, num_col, is_row_major);
}

ColumnMatrix::ColumnMatrix(std::string filename, std::string column_index_string, bool header, bool precise_float_parser) {
  // Convert string to vector of indices
  std::vector<int32_t> column_indices = Str2FeatureVec(column_index_string.c_str());
  
  // Set up CSV parser
  data_size_t num_global_data = 0;
  auto parser = std::unique_ptr<Parser>(Parser::CreateParser(filename.c_str(), header, 0, precise_float_parser));
  if (parser == nullptr) {
    Log::Fatal("Could not recognize data format of %s", filename.c_str());
  }
  
  // Determine number of columns in the data file
  int num_columns = parser->NumFeatures();

  // Check compatibility between column_indices and num_columns
  int32_t max_col = *std::max_element(column_indices.begin(), column_indices.end());
  if (max_col >= num_columns) Log::Fatal("Some column indices requested do not exist in the CSV file");

  // Read data to memory
  auto text_data = LoadTextDataToMemory(filename.c_str(), &num_global_data, header);
  int num_observations = static_cast<data_size_t>(text_data.size());

  // Allocate the data_ matrix
  data_ = Eigen::MatrixXd(num_observations, column_indices.size());

  // Load data
  ExtractMultipleFeaturesFromMemory(&text_data, parser.get(), column_indices, data_, num_observations);
  text_data.clear();
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

ColumnVector::ColumnVector(std::string filename, int32_t column_index, bool header, bool precise_float_parser) {
  // Set up CSV parser
  data_size_t num_global_data = 0;
  auto parser = std::unique_ptr<Parser>(Parser::CreateParser(filename.c_str(), header, 0, precise_float_parser));
  if (parser == nullptr) {
    Log::Fatal("Could not recognize data format of %s", filename.c_str());
  }
  
  // Read data to memory
  auto text_data = LoadTextDataToMemory(filename.c_str(), &num_global_data, header);
  int num_observations = static_cast<data_size_t>(text_data.size());

  // Allocate the data_ matrix
  data_ = Eigen::VectorXd(num_observations);

  // Load data
  ExtractSingleFeatureFromMemory(&text_data, parser.get(), column_index, data_, num_observations);
  text_data.clear();
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

void ColumnVector::AddToData(double* data_ptr, data_size_t num_row) {
  data_size_t num_existing_rows = NumRows();
  CHECK_EQ(num_row, num_existing_rows);
  // std::function<double(double, double)> op = std::plus<double>();
  UpdateData(data_ptr, num_row, std::plus<double>());
}

void ColumnVector::SubtractFromData(double* data_ptr, data_size_t num_row) {
  data_size_t num_existing_rows = NumRows();
  CHECK_EQ(num_row, num_existing_rows);
  // std::function<double(double, double)> op = std::minus<double>();
  UpdateData(data_ptr, num_row, std::minus<double>());
}

void ColumnVector::OverwriteData(double* data_ptr, data_size_t num_row) {
  double ptr_val;
  for (data_size_t i = 0; i < num_row; ++i) {
    ptr_val = static_cast<double>(*(data_ptr + i));
    data_(i) = ptr_val;
  }
}

void ColumnVector::UpdateData(double* data_ptr, data_size_t num_row, std::function<double(double, double)> op) {
  double ptr_val;
  double updated_val;
  for (data_size_t i = 0; i < num_row; ++i) {
    ptr_val = static_cast<double>(*(data_ptr + i));
    updated_val = op(data_(i), ptr_val);
    data_(i) = updated_val;
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

} // namespace StochTree
