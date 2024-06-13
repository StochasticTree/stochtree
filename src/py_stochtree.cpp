#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <nlohmann/json.hpp>
#include <stochtree/container.h>
#include <stochtree/data.h>
#include <stochtree/leaf_model.h>
#include <stochtree/meta.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/tree_sampler.h>
#include <stochtree/variance_model.h>
#include <functional>
#include <memory>
#include <string>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
using data_size_t = StochTree::data_size_t;

enum ForestLeafModel {
    kConstant, 
    kUnivariateRegression, 
    kMultivariateRegression
};

class ForestDatasetCpp {
 public:
  ForestDatasetCpp() {
    // Initialize pointer to C++ ForestDataset class
    dataset_ = std::make_unique<StochTree::ForestDataset>();
  }
  ~ForestDatasetCpp() {}

  void AddCovariates(py::array_t<double> covariate_matrix, data_size_t num_row, int num_col, bool row_major) {
    // Extract pointer to contiguous block of memory
    double* data_ptr = static_cast<double*>(covariate_matrix.mutable_data());
    
    // Load covariates
    dataset_->AddCovariates(data_ptr, num_row, num_col, row_major);
  }

  void AddBasis(py::array_t<double> basis_matrix, data_size_t num_row, int num_col, bool row_major) {
    // Extract pointer to contiguous block of memory
    double* data_ptr = static_cast<double*>(basis_matrix.mutable_data());
    
    // Load covariates
    dataset_->AddBasis(data_ptr, num_row, num_col, row_major);
  }

  void UpdateBasis(py::array_t<double> basis_matrix, data_size_t num_row, int num_col, bool row_major) {
    // Extract pointer to contiguous block of memory
    double* data_ptr = static_cast<double*>(basis_matrix.mutable_data());
    
    // Load covariates
    dataset_->UpdateBasis(data_ptr, num_row, num_col, row_major);
  }

  void AddVarianceWeights(py::array_t<double> weight_vector, data_size_t num_row) {
    // Extract pointer to contiguous block of memory
    double* data_ptr = static_cast<double*>(weight_vector.mutable_data());
    
    // Load covariates
    dataset_->AddVarianceWeights(data_ptr, num_row);
  }

  data_size_t NumRows() {
    return dataset_->NumObservations();
  }

  StochTree::ForestDataset* GetDataset() {
    return dataset_.get();
  }

 private:
  std::unique_ptr<StochTree::ForestDataset> dataset_;
};

class ResidualCpp {
 public:
  ResidualCpp(py::array_t<double> residual_array, data_size_t num_row) {
    // Extract pointer to contiguous block of memory
    double* data_ptr = static_cast<double*>(residual_array.mutable_data());
    
    // Initialize pointer to C++ ColumnVector class
    residual_ = std::make_unique<StochTree::ColumnVector>(data_ptr, num_row);
  }
  ~ResidualCpp() {}

  StochTree::ColumnVector* GetData() {
    return residual_.get();
  }

 private:
  std::unique_ptr<StochTree::ColumnVector> residual_;
};

class RngCpp {
 public:
  RngCpp(int random_seed = -1) {
    if (random_seed == -1) {
      std::random_device rd;
      rng_ = std::make_unique<std::mt19937>(rd());
    } else {
      rng_ = std::make_unique<std::mt19937>(random_seed);
    }
  }
  ~RngCpp() {}

  std::mt19937* GetRng() {
    return rng_.get();
  }

 private:
  std::unique_ptr<std::mt19937> rng_;
};

// Forward declarations
class ForestSamplerCpp;
class JsonCpp;

class ForestContainerCpp {
 public:
  ForestContainerCpp(int num_trees, int output_dimension = 1, bool is_leaf_constant = true) {
    // Initialize pointer to C++ ForestContainer class
    forest_samples_ = std::make_unique<StochTree::ForestContainer>(num_trees, output_dimension, is_leaf_constant);
    num_trees_ = num_trees;
    output_dimension_ = output_dimension;
    is_leaf_constant_ = is_leaf_constant;
  }
  ~ForestContainerCpp() {}

  int OutputDimension() {
    return forest_samples_->OutputDimension();
  }

  int NumSamples() {
    return forest_samples_->NumSamples();
  }

  py::array_t<double> Predict(ForestDatasetCpp& dataset) {
    // Predict from the forest container
    data_size_t n = dataset.NumRows();
    int num_samples = this->NumSamples();
    StochTree::ForestDataset* data_ptr = dataset.GetDataset();
    std::vector<double> output_raw = forest_samples_->Predict(*data_ptr);

    // Convert result to a matrix
    auto result = py::array_t<double>(py::detail::any_container<ssize_t>({n, num_samples}));
    auto accessor = result.mutable_unchecked<2>();
    // py::buffer_info buf = result.request();
    // double *ptr = static_cast<double *>(buf.ptr);
    for (size_t i = 0; i < n; i++) {
      for (int j = 0; j < num_samples; j++) {
        // NOTE: converting from "column-major" to "row-major" here
        accessor(i,j) = output_raw[j*n + i];
        // ptr[i*num_samples + j] = output_raw[j*n + i];
      }
    }

    return result;
  }

  py::array_t<double> PredictRaw(ForestDatasetCpp& dataset) {
    // Predict from the forest container
    data_size_t n = dataset.NumRows();
    int num_samples = this->NumSamples();
    int output_dim = this->OutputDimension();
    StochTree::ForestDataset* data_ptr = dataset.GetDataset();
    std::vector<double> output_raw = forest_samples_->PredictRaw(*data_ptr);

    // Convert result to 3 dimensional array (n x num_samples x output_dim)
    auto result = py::array_t<double>(py::detail::any_container<ssize_t>({n, num_samples, output_dim}));
    auto accessor = result.mutable_unchecked<3>();
    // py::buffer_info buf = result.request();
    // double *ptr = static_cast<double *>(buf.ptr);
    for (size_t i = 0; i < n; i++) {
      for (int j = 0; j < output_dim; j++) {
        for (int k = 0; k < num_samples; k++) {
          accessor(i,j,k) = output_raw[k*(output_dim*n) + i*output_dim + j];
          // ptr[i*(output_dim*num_samples) + j*output_dim + k] = output_raw[k*(output_dim*n) + i*output_dim + j];
        }
      }
    }

    return result;
  }

  py::array_t<double> PredictRawSingleForest(ForestDatasetCpp& dataset, int forest_num) {
    // Predict from the forest container
    data_size_t n = dataset.NumRows();
    int num_samples = this->NumSamples();
    int output_dim = this->OutputDimension();
    StochTree::ForestDataset* data_ptr = dataset.GetDataset();
    std::vector<double> output_raw = forest_samples_->PredictRaw(*data_ptr, forest_num);

    // Convert result to a matrix
    auto result = py::array_t<double>(py::detail::any_container<ssize_t>({n, output_dim}));
    auto accessor = result.mutable_unchecked<2>();
    // py::buffer_info buf = result.request();
    // double *ptr = static_cast<double *>(buf.ptr);
    for (size_t i = 0; i < n; i++) {
      for (int j = 0; j < output_dim; j++) {
        accessor(i,j) = output_raw[i*output_dim + j];
        // ptr[i*output_dim + j] = output_raw[i*output_dim + j];
      }
    }

    return result;
  }

  void SetRootValue(int forest_num, double leaf_value) {
    forest_samples_->InitializeRoot(leaf_value);
  }

  void SetRootVector(int forest_num, py::array_t<double>& leaf_vector, int leaf_size) {
    std::vector<double> leaf_vector_converted(leaf_size);
    for (int i = 0; i < leaf_size; i++) {
        leaf_vector_converted[i] = leaf_vector.at(i);
    }
    forest_samples_->InitializeRoot(leaf_vector_converted);
  }

  void UpdateResidual(ForestDatasetCpp& dataset, ResidualCpp& residual, ForestSamplerCpp& sampler, bool requires_basis, int forest_num, bool add);

  void SaveToJsonFile(std::string json_filename) {
    forest_samples_->SaveToJsonFile(json_filename);
  }

  void LoadFromJsonFile(std::string json_filename) {
    forest_samples_->LoadFromJsonFile(json_filename);
  }

  void LoadFromJson(JsonCpp& json, std::string forest_label);

  StochTree::ForestContainer* GetContainer() {
    return forest_samples_.get();
  }

  StochTree::TreeEnsemble* GetForest(int i) {
    return forest_samples_->GetEnsemble(i);
  }

  nlohmann::json ToJson() {
    return forest_samples_->to_json();
  }

 private:
  std::unique_ptr<StochTree::ForestContainer> forest_samples_;
  int num_trees_;
  int output_dimension_;
  bool is_leaf_constant_;
};

class ForestSamplerCpp {
 public:
  ForestSamplerCpp(ForestDatasetCpp& dataset, py::array_t<int> feature_types, int num_trees, data_size_t num_obs, double alpha, double beta, int min_samples_leaf) {
    // Convert vector of integers to std::vector of enum FeatureType
    std::vector<StochTree::FeatureType> feature_types_(feature_types.size());
    for (int i = 0; i < feature_types.size(); i++) {
        feature_types_[i] = static_cast<StochTree::FeatureType>(feature_types.at(i));
    }
    
    // Initialize pointer to C++ ForestTracker and TreePrior classes
    StochTree::ForestDataset* dataset_ptr = dataset.GetDataset();
    tracker_ = std::make_unique<StochTree::ForestTracker>(dataset_ptr->GetCovariates(), feature_types_, num_trees, num_obs);
    split_prior_ = std::make_unique<StochTree::TreePrior>(alpha, beta, min_samples_leaf);
  }
  ~ForestSamplerCpp() {}

  StochTree::ForestTracker* GetTracker() {return tracker_.get();}

  void SampleOneIteration(ForestContainerCpp& forest_samples, ForestDatasetCpp& dataset, ResidualCpp& residual, RngCpp& rng, 
                          py::array_t<int> feature_types, int cutpoint_grid_size, py::array_t<double> leaf_model_scale_input, 
                          py::array_t<double> variable_weights, double global_variance, int leaf_model_int, bool gfr = true, bool pre_initialized = false) {
    // Unpack feature types
    std::vector<StochTree::FeatureType> feature_types_(feature_types.size());
    for (int i = 0; i < feature_types.size(); i++) {
      feature_types_[i] = static_cast<StochTree::FeatureType>(feature_types.at(i));
    }

    // Convert leaf model type to enum
    ForestLeafModel leaf_model_enum;
    if (leaf_model_int == 0) leaf_model_enum = ForestLeafModel::kConstant;
    else if (leaf_model_int == 1) leaf_model_enum = ForestLeafModel::kUnivariateRegression;
    else if (leaf_model_int == 2) leaf_model_enum = ForestLeafModel::kMultivariateRegression;

    // Unpack leaf model parameters
    double leaf_scale;
    Eigen::MatrixXd leaf_scale_matrix;
    if ((leaf_model_enum == ForestLeafModel::kConstant) || 
        (leaf_model_enum == ForestLeafModel::kUnivariateRegression)) {
        leaf_scale = leaf_model_scale_input.at(0,0);
    } else if (leaf_model_enum == ForestLeafModel::kMultivariateRegression) {
        int num_row = leaf_model_scale_input.shape(0);
        int num_col = leaf_model_scale_input.shape(1);
        leaf_scale_matrix.resize(num_row, num_col);
        for (int i = 0; i < num_row; i++) {
            for (int j = 0; j < num_col; j++) {
                leaf_scale_matrix(i,j) = leaf_model_scale_input.at(i,j);
            }
        }
    }
    
    // Convert variable weights to std::vector
    std::vector<double> var_weights_vector(variable_weights.size());
    for (int i = 0; i < variable_weights.size(); i++) {
        var_weights_vector[i] = variable_weights.at(i);
    }

    // Run one iteration of the sampler
    StochTree::ForestContainer* forest_sample_ptr = forest_samples.GetContainer();
    StochTree::ForestDataset* forest_data_ptr = dataset.GetDataset();
    StochTree::ColumnVector* residual_data_ptr = residual.GetData();
    std::mt19937* rng_ptr = rng.GetRng();
    if (gfr) {
      InternalSampleGFR(*forest_sample_ptr, *forest_data_ptr, *residual_data_ptr, *rng_ptr, feature_types_, var_weights_vector, 
                        leaf_model_enum, leaf_scale_matrix, global_variance, leaf_scale, cutpoint_grid_size, pre_initialized);
    } else {
      InternalSampleMCMC(*forest_sample_ptr, *forest_data_ptr, *residual_data_ptr, *rng_ptr, feature_types_, var_weights_vector, 
                         leaf_model_enum, leaf_scale_matrix, global_variance, leaf_scale, cutpoint_grid_size, pre_initialized);
    }
  }

 private:
  std::unique_ptr<StochTree::ForestTracker> tracker_;
  std::unique_ptr<StochTree::TreePrior> split_prior_;

  void InternalSampleGFR(StochTree::ForestContainer& forest_samples, StochTree::ForestDataset& dataset, StochTree::ColumnVector& residual, std::mt19937& rng, 
                         std::vector<StochTree::FeatureType>& feature_types, std::vector<double>& var_weights_vector, ForestLeafModel leaf_model_enum, 
                         Eigen::MatrixXd& leaf_scale_matrix, double global_variance, double leaf_scale, int cutpoint_grid_size, bool pre_initialized) {
    if (leaf_model_enum == ForestLeafModel::kConstant) {
      StochTree::GaussianConstantLeafModel leaf_model = StochTree::GaussianConstantLeafModel(leaf_scale);
      StochTree::GFRForestSampler<StochTree::GaussianConstantLeafModel> sampler = StochTree::GFRForestSampler<StochTree::GaussianConstantLeafModel>(cutpoint_grid_size);
      sampler.SampleOneIter(*(tracker_.get()), forest_samples, leaf_model, dataset, residual, *(split_prior_.get()), rng, var_weights_vector, global_variance, feature_types, pre_initialized);
    } else if (leaf_model_enum == ForestLeafModel::kUnivariateRegression) {
      StochTree::GaussianUnivariateRegressionLeafModel leaf_model = StochTree::GaussianUnivariateRegressionLeafModel(leaf_scale);
      StochTree::GFRForestSampler<StochTree::GaussianUnivariateRegressionLeafModel> sampler = StochTree::GFRForestSampler<StochTree::GaussianUnivariateRegressionLeafModel>(cutpoint_grid_size);
      sampler.SampleOneIter(*(tracker_.get()), forest_samples, leaf_model, dataset, residual, *(split_prior_.get()), rng, var_weights_vector, global_variance, feature_types, pre_initialized);
    } else if (leaf_model_enum == ForestLeafModel::kMultivariateRegression) {
      StochTree::GaussianMultivariateRegressionLeafModel leaf_model = StochTree::GaussianMultivariateRegressionLeafModel(leaf_scale_matrix);
      StochTree::GFRForestSampler<StochTree::GaussianMultivariateRegressionLeafModel> sampler = StochTree::GFRForestSampler<StochTree::GaussianMultivariateRegressionLeafModel>(cutpoint_grid_size);
      sampler.SampleOneIter(*(tracker_.get()), forest_samples, leaf_model, dataset, residual, *(split_prior_.get()), rng, var_weights_vector, global_variance, feature_types, pre_initialized);
    }
  }

  void InternalSampleMCMC(StochTree::ForestContainer& forest_samples, StochTree::ForestDataset& dataset, StochTree::ColumnVector& residual, std::mt19937& rng, 
                          std::vector<StochTree::FeatureType>& feature_types, std::vector<double>& var_weights_vector, ForestLeafModel leaf_model_enum, 
                          Eigen::MatrixXd& leaf_scale_matrix, double global_variance, double leaf_scale, int cutpoint_grid_size, bool pre_initialized) {
    if (leaf_model_enum == ForestLeafModel::kConstant) {
      StochTree::GaussianConstantLeafModel leaf_model = StochTree::GaussianConstantLeafModel(leaf_scale);
      StochTree::MCMCForestSampler<StochTree::GaussianConstantLeafModel> sampler = StochTree::MCMCForestSampler<StochTree::GaussianConstantLeafModel>();
      sampler.SampleOneIter(*(tracker_.get()), forest_samples, leaf_model, dataset, residual, *(split_prior_.get()), rng, var_weights_vector, global_variance, pre_initialized);
    } else if (leaf_model_enum == ForestLeafModel::kUnivariateRegression) {
      StochTree::GaussianUnivariateRegressionLeafModel leaf_model = StochTree::GaussianUnivariateRegressionLeafModel(leaf_scale);
      StochTree::MCMCForestSampler<StochTree::GaussianUnivariateRegressionLeafModel> sampler = StochTree::MCMCForestSampler<StochTree::GaussianUnivariateRegressionLeafModel>();
      sampler.SampleOneIter(*(tracker_.get()), forest_samples, leaf_model, dataset, residual, *(split_prior_.get()), rng, var_weights_vector, global_variance, pre_initialized);
    } else if (leaf_model_enum == ForestLeafModel::kMultivariateRegression) {
      StochTree::GaussianMultivariateRegressionLeafModel leaf_model = StochTree::GaussianMultivariateRegressionLeafModel(leaf_scale_matrix);
      StochTree::MCMCForestSampler<StochTree::GaussianMultivariateRegressionLeafModel> sampler = StochTree::MCMCForestSampler<StochTree::GaussianMultivariateRegressionLeafModel>();
      sampler.SampleOneIter(*(tracker_.get()), forest_samples, leaf_model, dataset, residual, *(split_prior_.get()), rng, var_weights_vector, global_variance, pre_initialized);
    }
  }
};

class GlobalVarianceModelCpp {
 public:
  GlobalVarianceModelCpp() {
    var_model_ = StochTree::GlobalHomoskedasticVarianceModel();
  }
  ~GlobalVarianceModelCpp() {}

  double SampleOneIteration(ResidualCpp& residual, RngCpp& rng, double nu, double lamb) {
    StochTree::ColumnVector* residual_ptr = residual.GetData();
    std::mt19937* rng_ptr = rng.GetRng();
    return var_model_.SampleVarianceParameter(residual_ptr->GetData(), nu, lamb, *rng_ptr);
  }  

 private:
  StochTree::GlobalHomoskedasticVarianceModel var_model_;
};

class LeafVarianceModelCpp {
 public:
  LeafVarianceModelCpp() {
    var_model_ = StochTree::LeafNodeHomoskedasticVarianceModel();
  }
  ~LeafVarianceModelCpp() {}

  double SampleOneIteration(ForestContainerCpp& forest_samples, RngCpp& rng, double a, double b, int sample_num) {
    StochTree::ForestContainer* forest_sample_ptr = forest_samples.GetContainer();
    std::mt19937* rng_ptr = rng.GetRng();
    return var_model_.SampleVarianceParameter(forest_sample_ptr->GetEnsemble(sample_num), a, b, *rng_ptr);
  }

 private:
  StochTree::LeafNodeHomoskedasticVarianceModel var_model_;
};

void ForestContainerCpp::UpdateResidual(ForestDatasetCpp& dataset, ResidualCpp& residual, ForestSamplerCpp& sampler, bool requires_basis, int forest_num, bool add) {
  // Determine whether or not we are adding forest_num to the residuals
  std::function<double(double, double)> op;
  if (add) op = std::plus<double>();
  else op = std::minus<double>();
  
  // Perform the update (addition / subtraction) operation
  StochTree::UpdateResidualEntireForest(*(sampler.GetTracker()), *(dataset.GetDataset()), *(residual.GetData()), forest_samples_->GetEnsemble(forest_num), requires_basis, op);
}

class JsonCpp {
 public:
  JsonCpp() {
    // Initialize pointer to C++ nlohmann::json class
    json_ = std::make_unique<nlohmann::json>();
    nlohmann::json forests = nlohmann::json::object();
    json_->emplace("forests", forests);
    json_->emplace("num_forests", 0);
  }
  ~JsonCpp() {}

  void LoadFile(std::string filename) {
    std::ifstream f(filename);
    *json_ = nlohmann::json::parse(f);
  }

  void SaveFile(std::string filename) {
    std::ofstream output_file(filename);
    output_file << *json_ << std::endl;
  }

  std::string DumpJson() {
    return json_->dump();
  }

  std::string AddForest(ForestContainerCpp& forest_samples) {
    int forest_num = json_->at("num_forests");
    std::string forest_label = "forest_" + std::to_string(forest_num);
    nlohmann::json forest_json = forest_samples.ToJson();
    json_->at("forests").emplace(forest_label, forest_json);
    json_->at("num_forests") = forest_num + 1;
    return forest_label;
  }

  void AddDouble(std::string field_name, double field_value) {
    if (json_->contains(field_name)) {
      json_->at(field_name) = field_value;
    } else {
      json_->emplace(std::pair(field_name, field_value));
    }
  }

  void AddDoubleSubfolder(std::string subfolder_name, std::string field_name, double field_value) {
    if (json_->contains(subfolder_name)) {
      if (json_->at(subfolder_name).contains(field_name)) {
        json_->at(subfolder_name).at(field_name) = field_value;
      } else {
        json_->at(subfolder_name).emplace(std::pair(field_name, field_value));
      }
    } else {
      json_->emplace(std::pair(subfolder_name, nlohmann::json::object()));
      json_->at(subfolder_name).emplace(std::pair(field_name, field_value));
    }
  }

  void AddBool(std::string field_name, bool field_value) {
    if (json_->contains(field_name)) {
      json_->at(field_name) = field_value;
    } else {
      json_->emplace(std::pair(field_name, field_value));
    }
  }

  void AddBoolSubfolder(std::string subfolder_name, std::string field_name, bool field_value) {
    if (json_->contains(subfolder_name)) {
      if (json_->at(subfolder_name).contains(field_name)) {
        json_->at(subfolder_name).at(field_name) = field_value;
      } else {
        json_->at(subfolder_name).emplace(std::pair(field_name, field_value));
      }
    } else {
      json_->emplace(std::pair(subfolder_name, nlohmann::json::object()));
      json_->at(subfolder_name).emplace(std::pair(field_name, field_value));
    }
  }

  void AddString(std::string field_name, std::string field_value) {
    if (json_->contains(field_name)) {
      json_->at(field_name) = field_value;
    } else {
      json_->emplace(std::pair(field_name, field_value));
    }
  }

  void AddStringSubfolder(std::string subfolder_name, std::string field_name, std::string field_value) {
    if (json_->contains(subfolder_name)) {
      if (json_->at(subfolder_name).contains(field_name)) {
        json_->at(subfolder_name).at(field_name) = field_value;
      } else {
        json_->at(subfolder_name).emplace(std::pair(field_name, field_value));
      }
    } else {
      json_->emplace(std::pair(subfolder_name, nlohmann::json::object()));
      json_->at(subfolder_name).emplace(std::pair(field_name, field_value));
    }
  }

  void AddDoubleVector(std::string field_name, py::array_t<double> field_vector) {
    int vec_length = field_vector.size();
    auto accessor = field_vector.mutable_unchecked<1>();
    if (json_->contains(field_name)) {
      json_->at(field_name).clear();
      for (int i = 0; i < vec_length; i++) {
        json_->at(field_name).emplace_back(accessor(i));
      }
    } else {
      json_->emplace(std::pair(field_name, nlohmann::json::array()));
      for (int i = 0; i < vec_length; i++) {
        json_->at(field_name).emplace_back(accessor(i));
      }
    }
  }

  void AddDoubleVectorSubfolder(std::string subfolder_name, std::string field_name, py::array_t<double> field_vector) {
    int vec_length = field_vector.size();
    auto accessor = field_vector.mutable_unchecked<1>();
    if (json_->contains(subfolder_name)) {
      if (json_->at(subfolder_name).contains(field_name)) {
        json_->at(subfolder_name).at(field_name).clear();
        for (int i = 0; i < vec_length; i++) {
          json_->at(subfolder_name).at(field_name).emplace_back(accessor(i));
        }
      } else {
        json_->at(subfolder_name).emplace(std::pair(field_name, nlohmann::json::array()));
        for (int i = 0; i < vec_length; i++) {
          json_->at(subfolder_name).at(field_name).emplace_back(accessor(i));
        }
      }
    } else {
      json_->emplace(std::pair(subfolder_name, nlohmann::json::object()));
      json_->at(subfolder_name).emplace(std::pair(field_name, nlohmann::json::array()));
      for (int i = 0; i < vec_length; i++) {
        json_->at(subfolder_name).at(field_name).emplace_back(accessor(i));
      }
    }
  }

  void AddStringVector(std::string field_name, std::vector<std::string>& field_vector) {
    int vec_length = field_vector.size();
    if (json_->contains(field_name)) {
      json_->at(field_name).clear();
      for (int i = 0; i < vec_length; i++) {
        json_->at(field_name).emplace_back(field_vector.at(i));
      }
    } else {
      json_->emplace(std::pair(field_name, nlohmann::json::array()));
      for (int i = 0; i < vec_length; i++) {
        json_->at(field_name).emplace_back(field_vector.at(i));
      }
    }
  }

  void AddStringVectorSubfolder(std::string subfolder_name, std::string field_name, std::vector<std::string>& field_vector) {
    int vec_length = field_vector.size();
    if (json_->contains(subfolder_name)) {
      if (json_->at(subfolder_name).contains(field_name)) {
        json_->at(subfolder_name).at(field_name).clear();
        for (int i = 0; i < vec_length; i++) {
          json_->at(subfolder_name).at(field_name).emplace_back(field_vector.at(i));
        }
      } else {
        json_->at(subfolder_name).emplace(std::pair(field_name, nlohmann::json::array()));
        for (int i = 0; i < vec_length; i++) {
          json_->at(subfolder_name).at(field_name).emplace_back(field_vector.at(i));
        }
      }
    } else {
      json_->emplace(std::pair(subfolder_name, nlohmann::json::object()));
      json_->at(subfolder_name).emplace(std::pair(field_name, nlohmann::json::array()));
      for (int i = 0; i < vec_length; i++) {
        json_->at(subfolder_name).at(field_name).emplace_back(field_vector.at(i));
      }
    }
  }

  bool ContainsField(std::string field_name) {
    if (json_->contains(field_name)) {
      return true;
    } else {
      return false;
    }
  }

  bool ContainsFieldSubfolder(std::string subfolder_name, std::string field_name) {
    if (json_->contains(subfolder_name)) {
      if (json_->at(subfolder_name).contains(field_name)) {
        return true;
      } else {
        return false;
      }
    } else {
      return false;
    }
  }

  double ExtractDouble(std::string field_name) {
    return json_->at(field_name);
  }

  double ExtractDoubleSubfolder(std::string subfolder_name, std::string field_name) {
    return json_->at(subfolder_name).at(field_name);
  }

  bool ExtractBool(std::string field_name) {
    return json_->at(field_name);
  }

  bool ExtractBoolSubfolder(std::string subfolder_name, std::string field_name) {
    return json_->at(subfolder_name).at(field_name);
  }

  std::string ExtractString(std::string field_name) {
    return json_->at(field_name);
  }

  std::string ExtractStringSubfolder(std::string subfolder_name, std::string field_name) {
    return json_->at(subfolder_name).at(field_name);
  }

  py::array_t<double> ExtractDoubleVector(std::string field_name) {
    auto json_vec = json_->at(field_name);
    ssize_t json_vec_length = json_->at(field_name).size();
    auto result = py::array_t<double>(py::detail::any_container<ssize_t>({json_vec_length}));
    auto accessor = result.mutable_unchecked<1>();
    for (size_t i = 0; i < json_vec_length; i++) {
      accessor(i) = json_vec.at(i);
    }
    return result;
  }

  py::array_t<double> ExtractDoubleVectorSubfolder(std::string subfolder_name, std::string field_name) {
    auto json_vec = json_->at(subfolder_name).at(field_name);
    ssize_t json_vec_length = json_->at(subfolder_name).at(field_name).size();
    auto result = py::array_t<double>(py::detail::any_container<ssize_t>({json_vec_length}));
    auto accessor = result.mutable_unchecked<1>();
    for (size_t i = 0; i < json_vec_length; i++) {
      accessor(i) = json_vec.at(i);
    }
    return result;
  }

  std::vector<std::string> ExtractStringVector(std::string field_name) {
    auto json_vec = json_->at(field_name);
    ssize_t json_vec_length = json_->at(field_name).size();
    auto result = std::vector<std::string>(json_vec_length);
    for (size_t i = 0; i < json_vec_length; i++) {
      result.at(i) = json_vec.at(i);
    }
    return result;
  }

  std::vector<std::string> ExtractStringVectorSubfolder(std::string subfolder_name, std::string field_name) {
    auto json_vec = json_->at(subfolder_name).at(field_name);
    ssize_t json_vec_length = json_->at(subfolder_name).at(field_name).size();
    auto result = std::vector<std::string>(json_vec_length);
    for (size_t i = 0; i < json_vec_length; i++) {
      result.at(i) = json_vec.at(i);
    }
    return result;
  }

  nlohmann::json SubsetJsonForest(std::string forest_label) {
    return json_->at("forests").at(forest_label);
  }

 private:
  std::unique_ptr<nlohmann::json> json_;
};

void ForestContainerCpp::LoadFromJson(JsonCpp& json, std::string forest_label) {
  nlohmann::json forest_json = json.SubsetJsonForest(forest_label);
  forest_samples_->Reset();
  forest_samples_->from_json(forest_json);
}

PYBIND11_MODULE(stochtree_cpp, m) {
  py::class_<JsonCpp>(m, "JsonCpp")
    .def(py::init<>())
    .def("LoadFile", &JsonCpp::LoadFile)
    .def("SaveFile", &JsonCpp::SaveFile)
    .def("DumpJson", &JsonCpp::DumpJson)
    .def("AddDouble", &JsonCpp::AddDouble)
    .def("AddDoubleSubfolder", &JsonCpp::AddDoubleSubfolder)
    .def("AddBool", &JsonCpp::AddBool)
    .def("AddBoolSubfolder", &JsonCpp::AddBoolSubfolder)
    .def("AddString", &JsonCpp::AddString)
    .def("AddStringSubfolder", &JsonCpp::AddStringSubfolder)
    .def("AddDoubleVector", &JsonCpp::AddDoubleVector)
    .def("AddDoubleVectorSubfolder", &JsonCpp::AddDoubleVectorSubfolder)
    .def("AddStringVector", &JsonCpp::AddStringVector)
    .def("AddStringVectorSubfolder", &JsonCpp::AddStringVectorSubfolder)
    .def("AddForest", &JsonCpp::AddForest)
    .def("ContainsField", &JsonCpp::ContainsField)
    .def("ContainsFieldSubfolder", &JsonCpp::ContainsFieldSubfolder)
    .def("ExtractDouble", &JsonCpp::ExtractDouble)
    .def("ExtractDoubleSubfolder", &JsonCpp::ExtractDoubleSubfolder)
    .def("ExtractBool", &JsonCpp::ExtractBool)
    .def("ExtractBoolSubfolder", &JsonCpp::ExtractBoolSubfolder)
    .def("ExtractString", &JsonCpp::ExtractString)
    .def("ExtractStringSubfolder", &JsonCpp::ExtractStringSubfolder)
    .def("ExtractDoubleVector", &JsonCpp::ExtractDoubleVector)
    .def("ExtractDoubleVectorSubfolder", &JsonCpp::ExtractDoubleVectorSubfolder)
    .def("ExtractStringVector", &JsonCpp::ExtractStringVector)
    .def("ExtractStringVectorSubfolder", &JsonCpp::ExtractStringVectorSubfolder)
    .def("SubsetJsonForest", &JsonCpp::SubsetJsonForest);
  
  py::class_<ForestDatasetCpp>(m, "ForestDatasetCpp")
    .def(py::init<>())
    .def("AddCovariates", &ForestDatasetCpp::AddCovariates)
    .def("AddBasis", &ForestDatasetCpp::AddBasis)
    .def("UpdateBasis", &ForestDatasetCpp::UpdateBasis)
    .def("AddVarianceWeights", &ForestDatasetCpp::AddVarianceWeights)
    .def("NumRows", &ForestDatasetCpp::NumRows);

  py::class_<ResidualCpp>(m, "ResidualCpp")
    .def(py::init<py::array_t<double>,data_size_t>());

  py::class_<RngCpp>(m, "RngCpp")
    .def(py::init<int>());

  py::class_<ForestContainerCpp>(m, "ForestContainerCpp")
    .def(py::init<int,int,bool>())
    .def("OutputDimension", &ForestContainerCpp::OutputDimension)
    .def("NumSamples", &ForestContainerCpp::NumSamples)
    .def("Predict", &ForestContainerCpp::Predict)
    .def("PredictRaw", &ForestContainerCpp::PredictRaw)
    .def("PredictRawSingleForest", &ForestContainerCpp::PredictRawSingleForest)
    .def("SetRootValue", &ForestContainerCpp::SetRootValue)
    .def("SetRootVector", &ForestContainerCpp::SetRootVector)
    .def("UpdateResidual", &ForestContainerCpp::UpdateResidual)
    .def("SaveToJsonFile", &ForestContainerCpp::SaveToJsonFile)
    .def("LoadFromJsonFile", &ForestContainerCpp::LoadFromJsonFile)
    .def("LoadFromJson", &ForestContainerCpp::LoadFromJson);

  py::class_<ForestSamplerCpp>(m, "ForestSamplerCpp")
    .def(py::init<ForestDatasetCpp&, py::array_t<int>, int, data_size_t, double, double, int>())
    .def("SampleOneIteration", &ForestSamplerCpp::SampleOneIteration);

  py::class_<GlobalVarianceModelCpp>(m, "GlobalVarianceModelCpp")
    .def(py::init<>())
    .def("SampleOneIteration", &GlobalVarianceModelCpp::SampleOneIteration);

  py::class_<LeafVarianceModelCpp>(m, "LeafVarianceModelCpp")
    .def(py::init<>())
    .def("SampleOneIteration", &LeafVarianceModelCpp::SampleOneIteration);

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}