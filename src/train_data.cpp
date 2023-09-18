/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include <stochtree/io.h>
#include <stochtree/train_data.h>

#include <chrono>
#include <cstdio>
#include <limits>
#include <sstream>
#include <unordered_map>

namespace StochTree {

TrainData::TrainData() {
  data_filename_ = "noname";
  num_data_ = 0;
  is_finish_load_ = false;
  treatment_index_ = NO_SPECIFIC;
  outcome_index_ = NO_SPECIFIC;
}

TrainData::TrainData(data_size_t num_data) {
  CHECK_GT(num_data, 0);
  data_filename_ = "noname";
  num_data_ = num_data;
  is_finish_load_ = false;
  treatment_index_ = NO_SPECIFIC;
  outcome_index_ = NO_SPECIFIC;
}

TrainData::~TrainData() {}

void TrainData::Construct(int num_variables) {
  num_variables_ = num_variables;
  std::vector<int> used_variables;
  for (int i = 0; i < num_variables; ++i) {
    used_variables.emplace_back(i);
  }
  if (used_variables.empty()) {
    Log::Warning(
        "There are no meaningful variables which satisfy the provided configuration. "
        "Decreasing Dataset parameters min_data_in_bin or min_data_in_leaf and re-constructing "
        "Dataset might resolve this warning.");
  }
}

void TrainData::FinishLoad() {
  if (is_finish_load_) {
    return;
  }
  if (num_variables_ > 0) {
    for (int i = 0; i < num_variables_; ++i) {
      variables_[i]->ResetFeatureMetadata();
    }
  }
  is_finish_load_ = true;
}

TrainDataLoader::TrainDataLoader(const Config& io_config, int num_class, const char* filename)
  :config_(io_config), num_class_(num_class) {
  label_idx_ = NO_SPECIFIC;
  treatment_idx_ = NO_SPECIFIC;
  SetHeader(filename);
}

TrainDataLoader::~TrainDataLoader() {
}

void TrainDataLoader::SetHeader(const char* filename) {
  std::unordered_map<std::string, int> name2idx;
  std::string name_prefix("name:");
  if (filename != nullptr) {
    TextReader<data_size_t> text_reader(filename, config_.header);

    // get column names
    if (config_.header) {
      std::string first_line = text_reader.first_line();
      variable_names_ = Common::Split(first_line.c_str(), "\t,");
    } else {
      Log::Fatal("Config header must be set to True");
    }

    // determine label index
    if (config_.label_column.size() > 0) {
      if (Common::StartsWith(config_.label_column, name_prefix)) {
        std::string name = config_.label_column.substr(name_prefix.size());
        label_idx_ = -1;
        for (int i = 0; i < static_cast<int>(variable_names_.size()); ++i) {
          if (name == variable_names_[i]) {
            label_idx_ = i;
            break;
          }
        }
        if (label_idx_ >= 0) {
          Log::Info("Using column %s as label", name.c_str());
        } else {
          Log::Fatal("Could not find label column %s in data file \n"
                     "or data file doesn't contain header", name.c_str());
        }
      } else {
        if (!Common::AtoiAndCheck(config_.label_column.c_str(), &label_idx_)) {
          Log::Fatal("label_column is not a number,\n"
                     "if you want to use a column name,\n"
                     "please add the prefix \"name:\" to the column name");
        }
        Log::Info("Using column number %d as label", label_idx_);
      }
      variable_names_.erase(variable_names_.begin() + label_idx_);
    }

    // determine treatment index (if applicable)
    if (config_.treatment_column.size() > 0) {
      if (Common::StartsWith(config_.treatment_column, name_prefix)) {
        std::string name = config_.treatment_column.substr(name_prefix.size());
        treatment_idx_ = -1;
        for (int i = 0; i < static_cast<int>(variable_names_.size()); ++i) {
          if (name == variable_names_[i]) {
            treatment_idx_ = i;
            break;
          }
        }
        if (treatment_idx_ >= 0) {
          Log::Info("Using column %s as treatment", name.c_str());
        } else {
          Log::Fatal("Could not find treatment column %s in data file \n"
                     "or data file doesn't contain header", name.c_str());
        }
      } else {
        if (!Common::AtoiAndCheck(config_.treatment_column.c_str(), &treatment_idx_)) {
          Log::Fatal("treatment_column is not a number,\n"
                     "if you want to use a column name,\n"
                     "please add the prefix \"name:\" to the column name");
        }
        Log::Info("Using column number %d as treatment", treatment_idx_);
      }
      variable_names_.erase(variable_names_.begin() + treatment_idx_);
    }
  } else {
    // Read label index and treatment column from config
    if (config_.label_column.size() > 0) {
      // Only parse by column number when passed from an in-memory matrix
      if (!Common::AtoiAndCheck(config_.label_column.c_str(), &label_idx_)) {
        Log::Fatal("label_column is not a number,\n"
                    "if you want to use a column name,\n"
                    "please add the prefix \"name:\" to the column name");
      }
      Log::Info("Using column number %d as label", label_idx_);
    }
    if (config_.treatment_column.size() > 0) {
      if (!Common::AtoiAndCheck(config_.treatment_column.c_str(), &treatment_idx_)) {
        Log::Fatal("treatment_column is not a number,\n"
                    "if you want to use a column name,\n"
                    "please add the prefix \"name:\" to the column name");
      }
      Log::Info("Using column number %d as treatment", treatment_idx_);
    }
  }
}

TrainData* TrainDataLoader::LoadFromFile(const char* filename) {
  auto dataset = std::unique_ptr<TrainData>(new TrainData());
  data_size_t num_global_data = 0;
  std::vector<data_size_t> used_data_indices;
  auto parser = std::unique_ptr<Parser>(Parser::CreateParser(filename, config_.header, 0, label_idx_,
                                                              config_.precise_float_parser));
  if (parser == nullptr) {
    Log::Fatal("Could not recognize data format of %s", filename);
  }
  
  // Determine number of columns in the data file
  int num_columns = parser->NumFeatures();

  // read data to memory
  auto text_data = LoadTextDataToMemory(filename, &num_global_data);
  dataset->num_data_ = static_cast<data_size_t>(text_data.size());
  
  // Determine the number of features (as opposed to outcome / treatment)
  int num_features = 0;
  // unpack variables from file into variable class
  for (int i = 0; i < num_columns; i++){
    if (i == label_idx_) {
      dataset->outcome_.resize(dataset->num_data_);
      dataset->residuals_.resize(dataset->num_data_);
    } else if (i == treatment_idx_) {
      dataset->treatment_.resize(dataset->num_data_);
    } else {
      num_features += 1;
    }
  }
  Log::Info("Num features = %d", num_features);

  // Log::Info("Set number of features");
  dataset->num_variables_ = num_features;
  dataset->data_filename_ = filename;
  dataset->set_feature_names(variable_names_);
  
  // check that the size of variable_names_ is num_variables_ 
  if (!variable_names_.empty()) {
    CHECK_EQ(dataset->num_variables_, static_cast<int>(variable_names_.size()));
  }
  
  dataset->Construct(dataset->num_variables_);

  // Set label and treatment indices
  dataset->set_outcome_index(label_idx_);
  dataset->set_treatment_index(treatment_idx_);
  
  // unpack variables from file into variable class
  dataset->variables_.resize(dataset->num_variables_);
  for (int i = 0; i < dataset->num_variables_; i++){
    dataset->variables_[i].reset(new Feature(dataset->num_data_, FeatureType::kNumeric));
  }
  ExtractFeaturesFromMemory(&text_data, parser.get(), dataset.get());
  text_data.clear();
  
  // Make sure an in-memory dataset was successfully created
  CheckDataset(dataset.get());

  // Release the in memory dataset pointer
  return dataset.release();
}

TrainData* TrainDataLoader::ConstructFromMatrix(double* matrix_data, int num_col, data_size_t num_row, bool is_row_major){
  auto dataset = std::unique_ptr<TrainData>(new TrainData());
  
  // Determine the number of features (as opposed to outcome / treatment)
  int num_features = 0;
  // unpack variables from file into variable class
  for (int i = 0; i < num_col; i++){
    if (i == label_idx_) {
      dataset->outcome_.resize(num_row);
      dataset->residuals_.resize(num_row);
    } else if (i == treatment_idx_) {
      dataset->treatment_.resize(num_row);
    } else {
      num_features += 1;
    }
  }
  Log::Info("Num features = {%d}", num_features);
  
  dataset->num_variables_ = num_features;
  dataset->num_data_ = num_row;
  dataset->Construct(dataset->num_variables_);
  Log::Info("Ran construct on the data");

  // Set label and treatment indices
  dataset->set_outcome_index(label_idx_);
  dataset->set_treatment_index(treatment_idx_);

  // Allocate space for variables in vector
  dataset->variables_.resize(dataset->num_variables_);
  for (int i = 0; i < dataset->num_variables_; i++){
    dataset->variables_[i].reset(new Feature(dataset->num_data_, FeatureType::kNumeric));
  }

  // set variable names as a default "Column_0" through num_col - 1
  // this can be changed later through the R and Python interfaces
  if (dataset->variable_names_.empty()) {
    for (int i = 0; i < num_features; ++i) {
      std::stringstream str_buf;
      str_buf << "Column_" << i;
      dataset->variable_names_.push_back(str_buf.str());
    }
  }

  double temp_value;
  int var_counter;
  for (data_size_t i = 0; i < dataset->num_data_; ++i) {
    var_counter = 0;
    for (int j = 0; j < num_col; ++j) {
      if (is_row_major){
        // Numpy 2-d arrays are stored in "row major" order
        temp_value = static_cast<double>(*(matrix_data + static_cast<data_size_t>(num_col) * i + j));
      } else {
        // R matrices are stored in "column major" order
        temp_value = static_cast<double>(*(matrix_data + static_cast<data_size_t>(num_row) * j + i));
      }
      if (j == label_idx_){
        dataset->outcome_[i] = temp_value;
        dataset->residuals_[i] = temp_value;
      } else if (j == treatment_idx_) {
        dataset->treatment_[i] = temp_value;
      } else {
        dataset->variables_[var_counter]->PushRawData(i, temp_value);
        var_counter += 1;
      }
    }
    dataset->FinishOneRow(i);
  }
  dataset->FinishLoad();
  
  // Make sure an in-memory dataset was successfully created
  CheckDataset(dataset.get());

  // Release the in memory dataset pointer
  return dataset.release();
}

void TrainDataLoader::CheckDataset(const TrainData* dataset) {
  if (dataset->num_data_ <= 0) {
    Log::Fatal("Data loaded wass empty");
  }
  if (dataset->variable_names_.size() != static_cast<size_t>(dataset->num_variables_)) {
    Log::Fatal("Size of feature name error, should be %d, got %d", dataset->num_variables_,
               static_cast<int>(dataset->variable_names_.size()));
  }
  if (label_idx_ >= 0) {
    if (dataset->num_data_ != dataset->outcome_.size()) {
      Log::Fatal("Dataset size (%s) not the same as unpacked outcome size (%s)", dataset->num_data_, dataset->outcome_.size());
    }
  }
  if (treatment_idx_ >= 0) {
    if (dataset->num_data_ != dataset->treatment_.size()) {
      Log::Fatal("Dataset size (%s) not the same as unpacked treatment size (%s)", dataset->num_data_, dataset->treatment_.size());
    }
  }
}

std::vector<std::string> TrainDataLoader::LoadTextDataToMemory(const char* filename, int* num_global_data) {
  TextReader<data_size_t> text_reader(filename, config_.header, config_.file_load_progress_interval_bytes);
  // read all lines
  *num_global_data = text_reader.ReadAllLines();
  return std::move(text_reader.Lines());
}

/*! \brief Extract local features from memory */
void TrainDataLoader::ExtractFeaturesFromMemory(std::vector<std::string>* text_data, const Parser* parser, TrainData* dataset) {
  std::vector<std::pair<int, double>> oneline_features;
  auto& ref_text_data = *text_data;
  std::vector<float> feature_row(dataset->num_variables_);
  int feature_counter;
  for (data_size_t i = 0; i < dataset->num_data_; ++i) {
    // unpack the vector of textlines read from file into a vector of (int, double) tuples
    oneline_features.clear();
    parser->ParseOneLine(ref_text_data[i].c_str(), &oneline_features);
    
    // free processed line:
    ref_text_data[i].clear();
    
    // unload the data from oneline_features vector into the dataset variables containers    
    feature_counter = 0;
    for (auto& inner_data : oneline_features) {
      int feature_idx = inner_data.first;
      if (feature_idx == label_idx_){
        dataset->outcome_[i] = inner_data.second;
        dataset->residuals_[i] = inner_data.second;
      } else if (feature_idx == treatment_idx_) {
        dataset->treatment_[i] = inner_data.second;
      } else {
        dataset->variables_[feature_counter]->PushRawData(i, inner_data.second);
        feature_counter += 1;
      }
    }
    dataset->FinishOneRow(i);
  }
  dataset->FinishLoad();
  // free text data after use
  text_data->clear();
}

}  // namespace StochTree
