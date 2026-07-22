#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <nlohmann/json.hpp>
#include <stochtree/bart.h>
#include <stochtree/bart_sampler.h>
#include <stochtree/bcf.h>
#include <stochtree/bcf_sampler.h>
#include <stochtree/container.h>
#include <stochtree/data.h>
#include <stochtree/leaf_model.h>
#include <stochtree/meta.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/prediction.h>
#include <stochtree/random_effects.h>
#include <stochtree/tree_sampler.h>
#include <stochtree/ordinal_sampler.h>
#include <stochtree/variance_model.h>
#include <functional>
#include <memory>
#include <string>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
using data_size_t = StochTree::data_size_t;

class ForestSamplerCpp;

class ForestDatasetCpp {
 public:
  ForestDatasetCpp() {
    // Initialize pointer to C++ ForestDataset class
    dataset_ = std::make_unique<StochTree::ForestDataset>();
  }
  ~ForestDatasetCpp() {}

  void AddCovariates(py::array_t<double, py::array::forcecast> covariate_matrix, data_size_t num_row, int num_col, bool row_major) {
    // Extract pointer to contiguous block of memory
    double* data_ptr = static_cast<double*>(covariate_matrix.mutable_data());

    // Load covariates
    dataset_->AddCovariates(data_ptr, num_row, num_col, row_major);
  }

  void AddBasis(py::array_t<double, py::array::forcecast> basis_matrix, data_size_t num_row, int num_col, bool row_major) {
    // Extract pointer to contiguous block of memory
    double* data_ptr = static_cast<double*>(basis_matrix.mutable_data());

    // Load covariates
    dataset_->AddBasis(data_ptr, num_row, num_col, row_major);
  }

  void UpdateBasis(py::array_t<double, py::array::forcecast> basis_matrix, data_size_t num_row, int num_col, bool row_major) {
    // Extract pointer to contiguous block of memory
    double* data_ptr = static_cast<double*>(basis_matrix.mutable_data());

    // Load covariates
    dataset_->UpdateBasis(data_ptr, num_row, num_col, row_major);
  }

  void AddVarianceWeights(py::array_t<double, py::array::forcecast> weight_vector, data_size_t num_row) {
    // Extract pointer to contiguous block of memory
    double* data_ptr = static_cast<double*>(weight_vector.mutable_data());

    // Load covariates
    dataset_->AddVarianceWeights(data_ptr, num_row);
  }

  void UpdateVarianceWeights(py::array_t<double, py::array::forcecast> weight_vector, data_size_t num_row, bool exponentiate) {
    // Extract pointer to contiguous block of memory
    double* data_ptr = static_cast<double*>(weight_vector.mutable_data());

    // Load covariates
    dataset_->UpdateVarWeights(data_ptr, num_row, exponentiate);
  }

  py::array_t<double> GetCovariates() {
    // Initialize n x p numpy array to store the covariates
    data_size_t n = dataset_->NumObservations();
    int num_covariates = dataset_->NumCovariates();
    auto result = py::array_t<double>(py::detail::any_container<py::ssize_t>({n, num_covariates}));
    auto accessor = result.mutable_unchecked<2>();
    for (size_t i = 0; i < n; i++) {
      for (int j = 0; j < num_covariates; j++) {
        accessor(i, j) = dataset_->CovariateValue(i, j);
      }
    }

    return result;
  }

  py::array_t<double> GetBasis() {
    // Initialize n x k numpy array to store the basis
    data_size_t n = dataset_->NumObservations();
    int num_basis = dataset_->NumBasis();
    auto result = py::array_t<double>(py::detail::any_container<py::ssize_t>({n, num_basis}));
    auto accessor = result.mutable_unchecked<2>();
    for (size_t i = 0; i < n; i++) {
      for (int j = 0; j < num_basis; j++) {
        accessor(i, j) = dataset_->BasisValue(i, j);
      }
    }

    return result;
  }

  py::array_t<double> GetVarianceWeights() {
    // Initialize n x 1 numpy array to store the variance weights
    data_size_t n = dataset_->NumObservations();
    auto result = py::array_t<double>(py::detail::any_container<py::ssize_t>({n}));
    auto accessor = result.mutable_unchecked<1>();
    for (size_t i = 0; i < n; i++) {
      accessor(i) = dataset_->VarWeightValue(i);
    }

    return result;
  }

  data_size_t NumRows() {
    return dataset_->NumObservations();
  }

  int NumCovariates() {
    return dataset_->NumCovariates();
  }

  int NumBasis() {
    return dataset_->NumBasis();
  }

  bool HasBasis() {
    return dataset_->HasBasis();
  }

  bool HasVarianceWeights() {
    return dataset_->HasVarWeights();
  }

  void AddAuxiliaryDimension(int dim_size) {
    dataset_->AddAuxiliaryDimension(dim_size);
  }

  void SetAuxiliaryDataValue(int dim_idx, data_size_t element_idx, double value) {
    dataset_->SetAuxiliaryDataValue(dim_idx, element_idx, value);
  }

  double GetAuxiliaryDataValue(int dim_idx, data_size_t element_idx) {
    return dataset_->GetAuxiliaryDataValue(dim_idx, element_idx);
  }

  py::array_t<double> GetAuxiliaryDataVector(int dim_idx) {
    std::vector<double>& aux_vec = dataset_->GetAuxiliaryDataVector(dim_idx);
    data_size_t n = aux_vec.size();
    auto result = py::array_t<double>(py::detail::any_container<py::ssize_t>({n}));
    auto accessor = result.mutable_unchecked<1>();
    for (size_t i = 0; i < n; i++) {
      accessor(i) = aux_vec[i];
    }
    return result;
  }

  StochTree::ForestDataset* GetDataset() {
    return dataset_.get();
  }

 private:
  std::unique_ptr<StochTree::ForestDataset> dataset_;
};

class ResidualCpp {
 public:
  ResidualCpp(py::array_t<double, py::array::forcecast> residual_array, data_size_t num_row) {
    // Extract pointer to contiguous block of memory
    double* data_ptr = static_cast<double*>(residual_array.mutable_data());

    // Initialize pointer to C++ ColumnVector class
    residual_ = std::make_unique<StochTree::ColumnVector>(data_ptr, num_row);
  }
  ~ResidualCpp() {}

  StochTree::ColumnVector* GetData() {
    return residual_.get();
  }

  py::array_t<double> GetResidualArray() {
    // Obtain a reference to the underlying Eigen::VectorXd
    Eigen::VectorXd& resid_vector = residual_->GetData();

    // Initialize n x 1 numpy array to store the residual
    data_size_t n = residual_->NumRows();
    auto result = py::array_t<double>(py::detail::any_container<py::ssize_t>({n, 1}));
    auto accessor = result.mutable_unchecked<2>();
    for (size_t i = 0; i < n; i++) {
      accessor(i, 0) = resid_vector(i);
    }

    return result;
  }

  void ReplaceData(py::array_t<double, py::array::forcecast> new_vector, data_size_t num_row) {
    // Extract pointer to contiguous block of memory
    double* data_ptr = static_cast<double*>(new_vector.mutable_data());
    // Overwrite data in residual_
    residual_->OverwriteData(data_ptr, num_row);
  }

  void AddToData(py::array_t<double, py::array::forcecast> update_vector, data_size_t num_row) {
    // Extract pointer to contiguous block of memory
    double* data_ptr = static_cast<double*>(update_vector.mutable_data());
    // Add to data in residual_
    residual_->AddToData(data_ptr, num_row);
  }

  void SubtractFromData(py::array_t<double, py::array::forcecast> update_vector, data_size_t num_row) {
    // Extract pointer to contiguous block of memory
    double* data_ptr = static_cast<double*>(update_vector.mutable_data());
    // Subtract from data in residual_
    residual_->SubtractFromData(data_ptr, num_row);
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
  ForestContainerCpp(int num_trees, int output_dimension = 1, bool is_leaf_constant = true, bool is_exponentiated = false) {
    // Initialize pointer to C++ ForestContainer class
    forest_samples_ = std::make_unique<StochTree::ForestContainer>(num_trees, output_dimension, is_leaf_constant, is_exponentiated);
    num_trees_ = num_trees;
    output_dimension_ = output_dimension;
    is_leaf_constant_ = is_leaf_constant;
    is_exponentiated_ = is_exponentiated;
  }
  ForestContainerCpp(std::unique_ptr<StochTree::ForestContainer> forest_samples, int num_trees, int output_dimension = 1, bool is_leaf_constant = true, bool is_exponentiated = false) {
    forest_samples_ = std::move(forest_samples);
    num_trees_ = num_trees;
    output_dimension_ = output_dimension;
    is_leaf_constant_ = is_leaf_constant;
    is_exponentiated_ = is_exponentiated;
  }
  ~ForestContainerCpp() {}

  StochTree::ForestContainer* GetPtr() { return forest_samples_.get(); }

  void CombineForests(py::array_t<int> forest_inds) {
    int num_forests = forest_inds.size();
    for (int j = 1; j < num_forests; j++) {
      forest_samples_->MergeForests(forest_inds.at(0), forest_inds.at(j));
    }
  }

  void AddToForest(int forest_index, double constant_value) {
    forest_samples_->AddToForest(forest_index, constant_value);
  }

  void MultiplyForest(int forest_index, double constant_multiple) {
    forest_samples_->MultiplyForest(forest_index, constant_multiple);
  }

  int OutputDimension() {
    return forest_samples_->OutputDimension();
  }

  int NumTrees() {
    return num_trees_;
  }

  int NumSamples() {
    return forest_samples_->NumSamples();
  }

  int NumLeavesForest(int forest_num) {
    StochTree::TreeEnsemble* forest = forest_samples_->GetEnsemble(forest_num);
    return forest->NumLeaves();
  }

  double SumLeafSquared(int forest_num) {
    StochTree::TreeEnsemble* forest = forest_samples_->GetEnsemble(forest_num);
    return forest->SumLeafSquared();
  }

  void DeleteSample(int forest_num) {
    forest_samples_->DeleteSample(forest_num);
  }

  py::array_t<double> Predict(ForestDatasetCpp& dataset) {
    // Predict from the forest container
    data_size_t n = dataset.NumRows();
    int num_samples = this->NumSamples();
    StochTree::ForestDataset* data_ptr = dataset.GetDataset();
    std::vector<double> output_raw = forest_samples_->Predict(*data_ptr);

    // Convert result to a matrix
    auto result = py::array_t<double>(py::detail::any_container<py::ssize_t>({n, num_samples}));
    auto accessor = result.mutable_unchecked<2>();
    // py::buffer_info buf = result.request();
    // double *ptr = static_cast<double *>(buf.ptr);
    for (size_t i = 0; i < n; i++) {
      for (int j = 0; j < num_samples; j++) {
        // NOTE: converting from "column-major" to "row-major" here
        accessor(i, j) = output_raw[j * n + i];
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
    auto result = py::array_t<double>(py::detail::any_container<py::ssize_t>({n, num_samples, output_dim}));
    auto accessor = result.mutable_unchecked<3>();
    // py::buffer_info buf = result.request();
    // double *ptr = static_cast<double *>(buf.ptr);
    for (size_t i = 0; i < n; i++) {
      for (int j = 0; j < output_dim; j++) {
        for (int k = 0; k < num_samples; k++) {
          accessor(i, k, j) = output_raw[k * (output_dim * n) + i * output_dim + j];
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
    auto result = py::array_t<double>(py::detail::any_container<py::ssize_t>({n, output_dim}));
    auto accessor = result.mutable_unchecked<2>();
    // py::buffer_info buf = result.request();
    // double *ptr = static_cast<double *>(buf.ptr);
    for (size_t i = 0; i < n; i++) {
      for (int j = 0; j < output_dim; j++) {
        accessor(i, j) = output_raw[i * output_dim + j];
        // ptr[i*output_dim + j] = output_raw[i*output_dim + j];
      }
    }

    return result;
  }

  py::array_t<double> PredictRawSingleTree(ForestDatasetCpp& dataset, int forest_num, int tree_num) {
    // Predict from the forest container
    data_size_t n = dataset.NumRows();
    int num_samples = this->NumSamples();
    int output_dim = this->OutputDimension();
    StochTree::ForestDataset* data_ptr = dataset.GetDataset();
    std::vector<double> output_raw = forest_samples_->PredictRawSingleTree(*data_ptr, forest_num, tree_num);

    // Convert result to a matrix
    auto result = py::array_t<double>(py::detail::any_container<py::ssize_t>({n, output_dim}));
    auto accessor = result.mutable_unchecked<2>();
    // py::buffer_info buf = result.request();
    // double *ptr = static_cast<double *>(buf.ptr);
    for (size_t i = 0; i < n; i++) {
      for (int j = 0; j < output_dim; j++) {
        accessor(i, j) = output_raw[i * output_dim + j];
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

  void InitializeRootValue(double val) {
    forest_samples_->InitializeRoot(val);
  }

  void InitializeRootVector(std::vector<double> vals) {
    forest_samples_->InitializeRoot(vals);
  }

  void AdjustResidual(ForestDatasetCpp& dataset, ResidualCpp& residual, ForestSamplerCpp& sampler, bool requires_basis, int forest_num, bool add);

  void SaveToJsonFile(std::string json_filename) {
    forest_samples_->SaveToJsonFile(json_filename);
  }

  void LoadFromJsonFile(std::string json_filename) {
    forest_samples_->LoadFromJsonFile(json_filename);
  }

  void LoadFromJson(JsonCpp& json, std::string forest_label);

  void AppendFromJson(JsonCpp& json, std::string forest_label);

  std::string DumpJsonString() {
    return forest_samples_->DumpJsonString();
  }

  void LoadFromJsonString(std::string& json_string) {
    forest_samples_->LoadFromJsonString(json_string);
  }

  StochTree::ForestContainer* GetContainer() {
    return forest_samples_.get();
  }

  StochTree::TreeEnsemble* GetForest(int i) {
    return forest_samples_->GetEnsemble(i);
  }

  nlohmann::json ToJson() {
    return forest_samples_->to_json();
  }

  void AddSampleValue(double leaf_value) {
    if (forest_samples_->OutputDimension() != 1) {
      StochTree::Log::Fatal("leaf_value must match forest leaf dimension");
    }
    int num_samples = forest_samples_->NumSamples();
    forest_samples_->AddSamples(1);
    StochTree::TreeEnsemble* ensemble = forest_samples_->GetEnsemble(num_samples);
    int num_trees = ensemble->NumTrees();
    for (int i = 0; i < num_trees; i++) {
      StochTree::Tree* tree = ensemble->GetTree(i);
      tree->SetLeaf(0, leaf_value);
    }
  }

  void AddSampleVector(py::array_t<double, py::array::forcecast> leaf_vector) {
    if (forest_samples_->OutputDimension() != leaf_vector.size()) {
      StochTree::Log::Fatal("leaf_vector must match forest leaf dimension");
    }
    int num_samples = forest_samples_->NumSamples();
    std::vector<double> leaf_vector_cast(leaf_vector.size());
    for (int i = 0; i < leaf_vector.size(); i++) leaf_vector_cast.at(i) = leaf_vector.at(i);
    forest_samples_->AddSamples(1);
    StochTree::TreeEnsemble* ensemble = forest_samples_->GetEnsemble(num_samples);
    int num_trees = ensemble->NumTrees();
    for (int i = 0; i < num_trees; i++) {
      StochTree::Tree* tree = ensemble->GetTree(i);
      tree->SetLeafVector(0, leaf_vector_cast);
    }
  }

  void AddNumericSplitVector(int forest_num, int tree_num, int leaf_num, int feature_num,
                             double split_threshold, py::array_t<double, py::array::forcecast> left_leaf_vector,
                             py::array_t<double, py::array::forcecast> right_leaf_vector) {
    if (forest_samples_->OutputDimension() != left_leaf_vector.size()) {
      StochTree::Log::Fatal("left_leaf_vector must match forest leaf dimension");
    }
    if (forest_samples_->OutputDimension() != right_leaf_vector.size()) {
      StochTree::Log::Fatal("right_leaf_vector must match forest leaf dimension");
    }
    std::vector<double> left_leaf_vector_cast(left_leaf_vector.size());
    std::vector<double> right_leaf_vector_cast(right_leaf_vector.size());
    for (int i = 0; i < left_leaf_vector.size(); i++) left_leaf_vector_cast.at(i) = left_leaf_vector.at(i);
    for (int i = 0; i < right_leaf_vector.size(); i++) right_leaf_vector_cast.at(i) = right_leaf_vector.at(i);
    StochTree::TreeEnsemble* ensemble = forest_samples_->GetEnsemble(forest_num);
    StochTree::Tree* tree = ensemble->GetTree(tree_num);
    if (!tree->IsLeaf(leaf_num)) {
      StochTree::Log::Fatal("leaf_num is not a leaf");
    }
    tree->ExpandNode(leaf_num, feature_num, split_threshold, left_leaf_vector_cast, right_leaf_vector_cast);
  }

  void AddNumericSplitValue(int forest_num, int tree_num, int leaf_num, int feature_num,
                            double split_threshold, double left_leaf_value, double right_leaf_value) {
    if (forest_samples_->OutputDimension() != 1) {
      StochTree::Log::Fatal("left_leaf_value must match forest leaf dimension");
    }
    if (forest_samples_->OutputDimension() != 1) {
      StochTree::Log::Fatal("right_leaf_value must match forest leaf dimension");
    }
    StochTree::TreeEnsemble* ensemble = forest_samples_->GetEnsemble(forest_num);
    StochTree::Tree* tree = ensemble->GetTree(tree_num);
    if (!tree->IsLeaf(leaf_num)) {
      StochTree::Log::Fatal("leaf_num is not a leaf");
    }
    tree->ExpandNode(leaf_num, feature_num, split_threshold, left_leaf_value, right_leaf_value);
  }

  py::array_t<int> GetTreeLeaves(int forest_num, int tree_num) {
    StochTree::TreeEnsemble* ensemble = forest_samples_->GetEnsemble(forest_num);
    StochTree::Tree* tree = ensemble->GetTree(tree_num);
    std::vector<int32_t> leaves_raw = tree->GetLeaves();
    int num_leaves = leaves_raw.size();
    auto result = py::array_t<int>(py::detail::any_container<py::ssize_t>({num_leaves}));
    auto accessor = result.mutable_unchecked<1>();
    for (size_t i = 0; i < num_leaves; i++) {
      accessor(i) = leaves_raw.at(i);
    }
    return result;
  }

  py::array_t<int> GetTreeSplitCounts(int forest_num, int tree_num, int num_features) {
    auto result = py::array_t<int>(py::detail::any_container<py::ssize_t>({num_features}));
    auto accessor = result.mutable_unchecked<1>();
    for (size_t i = 0; i < num_features; i++) {
      accessor(i) = 0;
    }
    StochTree::TreeEnsemble* ensemble = forest_samples_->GetEnsemble(forest_num);
    StochTree::Tree* tree = ensemble->GetTree(tree_num);
    std::vector<int32_t> split_nodes = tree->GetInternalNodes();
    for (int i = 0; i < split_nodes.size(); i++) {
      auto node_id = split_nodes.at(i);
      auto split_feature = tree->SplitIndex(node_id);
      accessor(split_feature)++;
    }
    return result;
  }

  py::array_t<int> GetForestSplitCounts(int forest_num, int num_features) {
    auto result = py::array_t<int>(py::detail::any_container<py::ssize_t>({num_features}));
    auto accessor = result.mutable_unchecked<1>();
    for (size_t i = 0; i < num_features; i++) {
      accessor(i) = 0;
    }
    StochTree::TreeEnsemble* ensemble = forest_samples_->GetEnsemble(forest_num);
    int num_trees = ensemble->NumTrees();
    for (int i = 0; i < num_trees; i++) {
      StochTree::Tree* tree = ensemble->GetTree(i);
      std::vector<int32_t> split_nodes = tree->GetInternalNodes();
      for (int j = 0; j < split_nodes.size(); j++) {
        auto node_id = split_nodes.at(j);
        auto split_feature = tree->SplitIndex(node_id);
        accessor(split_feature)++;
      }
    }
    return result;
  }

  py::array_t<int> GetOverallSplitCounts(int num_features) {
    auto result = py::array_t<int>(py::detail::any_container<py::ssize_t>({num_features}));
    auto accessor = result.mutable_unchecked<1>();
    for (size_t i = 0; i < num_features; i++) {
      accessor(i) = 0;
    }
    int num_samples = forest_samples_->NumSamples();
    int num_trees = forest_samples_->NumTrees();
    for (int i = 0; i < num_samples; i++) {
      StochTree::TreeEnsemble* ensemble = forest_samples_->GetEnsemble(i);
      for (int j = 0; j < num_trees; j++) {
        StochTree::Tree* tree = ensemble->GetTree(j);
        std::vector<int32_t> split_nodes = tree->GetInternalNodes();
        for (int k = 0; k < split_nodes.size(); k++) {
          auto node_id = split_nodes.at(k);
          auto split_feature = tree->SplitIndex(node_id);
          accessor(split_feature)++;
        }
      }
    }
    return result;
  }

  py::array_t<int> GetGranularSplitCounts(int num_features) {
    int num_samples = forest_samples_->NumSamples();
    int num_trees = forest_samples_->NumTrees();
    auto result = py::array_t<int>(py::detail::any_container<py::ssize_t>({num_samples, num_trees, num_features}));
    auto accessor = result.mutable_unchecked<3>();
    for (int i = 0; i < num_samples; i++) {
      for (int j = 0; j < num_trees; j++) {
        for (int k = 0; k < num_features; k++) {
          accessor(i, j, k) = 0;
        }
      }
    }
    for (int i = 0; i < num_samples; i++) {
      StochTree::TreeEnsemble* ensemble = forest_samples_->GetEnsemble(i);
      for (int j = 0; j < num_trees; j++) {
        StochTree::Tree* tree = ensemble->GetTree(j);
        std::vector<int32_t> split_nodes = tree->GetInternalNodes();
        for (int k = 0; k < split_nodes.size(); k++) {
          auto node_id = split_nodes.at(k);
          auto split_feature = tree->SplitIndex(node_id);
          accessor(i, j, split_feature)++;
        }
      }
    }
    return result;
  }

  bool IsLeafNode(int forest_id, int tree_id, int node_id) {
    StochTree::TreeEnsemble* ensemble = forest_samples_->GetEnsemble(forest_id);
    StochTree::Tree* tree = ensemble->GetTree(tree_id);
    return tree->IsLeaf(node_id);
  }

  bool IsNumericSplitNode(int forest_id, int tree_id, int node_id) {
    StochTree::TreeEnsemble* ensemble = forest_samples_->GetEnsemble(forest_id);
    StochTree::Tree* tree = ensemble->GetTree(tree_id);
    return tree->IsNumericSplitNode(node_id);
  }

  bool IsCategoricalSplitNode(int forest_id, int tree_id, int node_id) {
    StochTree::TreeEnsemble* ensemble = forest_samples_->GetEnsemble(forest_id);
    StochTree::Tree* tree = ensemble->GetTree(tree_id);
    return tree->IsCategoricalSplitNode(node_id);
  }

  int ParentNode(int forest_id, int tree_id, int node_id) {
    StochTree::TreeEnsemble* ensemble = forest_samples_->GetEnsemble(forest_id);
    StochTree::Tree* tree = ensemble->GetTree(tree_id);
    return tree->Parent(node_id);
  }

  int LeftChildNode(int forest_id, int tree_id, int node_id) {
    StochTree::TreeEnsemble* ensemble = forest_samples_->GetEnsemble(forest_id);
    StochTree::Tree* tree = ensemble->GetTree(tree_id);
    return tree->LeftChild(node_id);
  }

  int RightChildNode(int forest_id, int tree_id, int node_id) {
    StochTree::TreeEnsemble* ensemble = forest_samples_->GetEnsemble(forest_id);
    StochTree::Tree* tree = ensemble->GetTree(tree_id);
    return tree->RightChild(node_id);
  }

  int SplitIndex(int forest_id, int tree_id, int node_id) {
    StochTree::TreeEnsemble* ensemble = forest_samples_->GetEnsemble(forest_id);
    StochTree::Tree* tree = ensemble->GetTree(tree_id);
    return tree->SplitIndex(node_id);
  }

  int NodeDepth(int forest_id, int tree_id, int node_id) {
    StochTree::TreeEnsemble* ensemble = forest_samples_->GetEnsemble(forest_id);
    StochTree::Tree* tree = ensemble->GetTree(tree_id);
    return tree->GetDepth(node_id);
  }

  double SplitThreshold(int forest_id, int tree_id, int node_id) {
    StochTree::TreeEnsemble* ensemble = forest_samples_->GetEnsemble(forest_id);
    StochTree::Tree* tree = ensemble->GetTree(tree_id);
    return tree->Threshold(node_id);
  }

  py::array_t<int> SplitCategories(int forest_id, int tree_id, int node_id) {
    StochTree::TreeEnsemble* ensemble = forest_samples_->GetEnsemble(forest_id);
    StochTree::Tree* tree = ensemble->GetTree(tree_id);
    std::vector<std::uint32_t> raw_categories = tree->CategoryList(node_id);
    int num_categories = raw_categories.size();
    auto result = py::array_t<int>(py::detail::any_container<py::ssize_t>({num_categories}));
    auto accessor = result.mutable_unchecked<1>();
    for (int i = 0; i < num_categories; i++) {
      accessor(i) = raw_categories.at(i);
    }
    return result;
  }

  py::array_t<double> NodeLeafValues(int forest_id, int tree_id, int node_id) {
    StochTree::TreeEnsemble* ensemble = forest_samples_->GetEnsemble(forest_id);
    StochTree::Tree* tree = ensemble->GetTree(tree_id);
    int num_outputs = tree->OutputDimension();
    auto result = py::array_t<double>(py::detail::any_container<py::ssize_t>({num_outputs}));
    auto accessor = result.mutable_unchecked<1>();
    for (int i = 0; i < num_outputs; i++) {
      accessor(i) = tree->LeafValue(node_id, i);
    }
    return result;
  }

  int NumNodes(int forest_id, int tree_id) {
    StochTree::TreeEnsemble* ensemble = forest_samples_->GetEnsemble(forest_id);
    StochTree::Tree* tree = ensemble->GetTree(tree_id);
    return tree->NumValidNodes();
  }

  int NumLeaves(int forest_id, int tree_id) {
    StochTree::TreeEnsemble* ensemble = forest_samples_->GetEnsemble(forest_id);
    StochTree::Tree* tree = ensemble->GetTree(tree_id);
    return tree->NumLeaves();
  }

  int NumLeafParents(int forest_id, int tree_id) {
    StochTree::TreeEnsemble* ensemble = forest_samples_->GetEnsemble(forest_id);
    StochTree::Tree* tree = ensemble->GetTree(tree_id);
    return tree->NumLeafParents();
  }

  int NumSplitNodes(int forest_id, int tree_id) {
    StochTree::TreeEnsemble* ensemble = forest_samples_->GetEnsemble(forest_id);
    StochTree::Tree* tree = ensemble->GetTree(tree_id);
    return tree->NumSplitNodes();
  }

  py::array_t<int> Nodes(int forest_id, int tree_id) {
    StochTree::TreeEnsemble* ensemble = forest_samples_->GetEnsemble(forest_id);
    StochTree::Tree* tree = ensemble->GetTree(tree_id);
    std::vector<std::int32_t> nodes = tree->GetNodes();
    int num_nodes = nodes.size();
    auto result = py::array_t<int>(py::detail::any_container<py::ssize_t>({num_nodes}));
    auto accessor = result.mutable_unchecked<1>();
    for (int i = 0; i < num_nodes; i++) {
      accessor(i) = nodes.at(i);
    }
    return result;
  }

  py::array_t<int> Leaves(int forest_id, int tree_id) {
    StochTree::TreeEnsemble* ensemble = forest_samples_->GetEnsemble(forest_id);
    StochTree::Tree* tree = ensemble->GetTree(tree_id);
    std::vector<std::int32_t> leaves = tree->GetLeaves();
    int num_leaves = leaves.size();
    auto result = py::array_t<int>(py::detail::any_container<py::ssize_t>({num_leaves}));
    auto accessor = result.mutable_unchecked<1>();
    for (int i = 0; i < num_leaves; i++) {
      accessor(i) = leaves.at(i);
    }
    return result;
  }

 private:
  std::unique_ptr<StochTree::ForestContainer> forest_samples_;
  int num_trees_;
  int output_dimension_;
  bool is_leaf_constant_;
  bool is_exponentiated_;
};

class ForestCpp {
 public:
  ForestCpp(int num_trees, int output_dimension = 1, bool is_leaf_constant = true, bool is_exponentiated = false) {
    // Initialize pointer to C++ TreeEnsemble class
    forest_ = std::make_unique<StochTree::TreeEnsemble>(num_trees, output_dimension, is_leaf_constant, is_exponentiated);
    num_trees_ = num_trees;
    output_dimension_ = output_dimension;
    is_leaf_constant_ = is_leaf_constant;
    is_exponentiated_ = is_exponentiated;
  }
  ~ForestCpp() {}

  StochTree::TreeEnsemble* GetForestPtr() { return forest_.get(); }

  void MergeForest(ForestCpp& outbound_forest) {
    forest_->MergeForest(*outbound_forest.GetForestPtr());
  }

  void AddConstant(double constant_value) {
    forest_->AddValueToLeaves(constant_value);
  }

  void MultiplyConstant(double constant_multiple) {
    forest_->MultiplyLeavesByValue(constant_multiple);
  }

  int OutputDimension() {
    return forest_->OutputDimension();
  }

  int NumTrees() {
    return num_trees_;
  }

  int NumLeavesForest() {
    return forest_->NumLeaves();
  }

  double SumLeafSquared(int forest_num) {
    return forest_->SumLeafSquared();
  }

  void ResetRoot() {
    // Reset active forest using the forest held at index forest_num
    forest_->ResetRoot();
  }

  void Reset(ForestContainerCpp& forest_container, int forest_num) {
    // Extract raw pointer to the forest held at index forest_num
    StochTree::TreeEnsemble* forest = forest_container.GetForest(forest_num);

    // Reset active forest using the forest held at index forest_num
    forest_->ReconstituteFromForest(*forest);
  }

  py::array_t<double> Predict(ForestDatasetCpp& dataset) {
    // Predict from the forest container
    data_size_t n = dataset.NumRows();
    StochTree::ForestDataset* data_ptr = dataset.GetDataset();
    std::vector<double> output_raw = forest_->Predict(*data_ptr);

    // Convert result to a matrix
    auto result = py::array_t<double>(py::detail::any_container<py::ssize_t>({n}));
    auto accessor = result.mutable_unchecked<1>();
    for (size_t i = 0; i < n; i++) {
      accessor(i) = output_raw[i];
    }

    return result;
  }

  py::array_t<double> PredictRaw(ForestDatasetCpp& dataset) {
    // Predict from the forest container
    data_size_t n = dataset.NumRows();
    int output_dim = this->OutputDimension();
    StochTree::ForestDataset* data_ptr = dataset.GetDataset();
    std::vector<double> output_raw = forest_->PredictRaw(*data_ptr);

    // Convert result to 2 dimensional array (n x output_dim)
    auto result = py::array_t<double>(py::detail::any_container<py::ssize_t>({n, output_dim}));
    auto accessor = result.mutable_unchecked<2>();
    for (size_t i = 0; i < n; i++) {
      for (int j = 0; j < output_dim; j++) {
        accessor(i, j) = output_raw[i * output_dim + j];
      }
    }

    return result;
  }

  void SetRootValue(double leaf_value) {
    forest_->SetLeafValue(leaf_value);
  }

  void SetRootVector(py::array_t<double>& leaf_vector, int leaf_size) {
    std::vector<double> leaf_vector_converted(leaf_size);
    for (int i = 0; i < leaf_size; i++) {
      leaf_vector_converted[i] = leaf_vector.at(i);
    }
    forest_->SetLeafVector(leaf_vector_converted);
  }

  void AdjustResidual(ForestDatasetCpp& dataset, ResidualCpp& residual, ForestSamplerCpp& sampler, bool requires_basis, bool add);

  StochTree::TreeEnsemble* GetEnsemble() {
    return forest_.get();
  }

  void AddNumericSplitValue(int tree_num, int leaf_num, int feature_num, double split_threshold,
                            double left_leaf_value, double right_leaf_value) {
    if (forest_->OutputDimension() != 1) {
      StochTree::Log::Fatal("left_leaf_value must match forest leaf dimension");
    }
    if (forest_->OutputDimension() != 1) {
      StochTree::Log::Fatal("right_leaf_value must match forest leaf dimension");
    }
    StochTree::TreeEnsemble* ensemble = forest_.get();
    StochTree::Tree* tree = ensemble->GetTree(tree_num);
    if (!tree->IsLeaf(leaf_num)) {
      StochTree::Log::Fatal("leaf_num is not a leaf");
    }
    tree->ExpandNode(leaf_num, feature_num, split_threshold, left_leaf_value, right_leaf_value);
  }

  void AddNumericSplitVector(int tree_num, int leaf_num, int feature_num, double split_threshold,
                             py::array_t<double, py::array::forcecast> left_leaf_vector, py::array_t<double, py::array::forcecast> right_leaf_vector) {
    if (forest_->OutputDimension() != left_leaf_vector.size()) {
      StochTree::Log::Fatal("left_leaf_vector must match forest leaf dimension");
    }
    if (forest_->OutputDimension() != right_leaf_vector.size()) {
      StochTree::Log::Fatal("right_leaf_vector must match forest leaf dimension");
    }
    std::vector<double> left_leaf_vector_cast(left_leaf_vector.size());
    std::vector<double> right_leaf_vector_cast(right_leaf_vector.size());
    for (int i = 0; i < left_leaf_vector.size(); i++) left_leaf_vector_cast.at(i) = left_leaf_vector.at(i);
    for (int i = 0; i < right_leaf_vector.size(); i++) right_leaf_vector_cast.at(i) = right_leaf_vector.at(i);
    StochTree::TreeEnsemble* ensemble = forest_.get();
    StochTree::Tree* tree = ensemble->GetTree(tree_num);
    if (!tree->IsLeaf(leaf_num)) {
      StochTree::Log::Fatal("leaf_num is not a leaf");
    }
    tree->ExpandNode(leaf_num, feature_num, split_threshold, left_leaf_vector_cast, right_leaf_vector_cast);
  }

  py::array_t<int> GetTreeLeaves(int tree_num) {
    StochTree::Tree* tree = forest_->GetTree(tree_num);
    std::vector<int32_t> leaves_raw = tree->GetLeaves();
    int num_leaves = leaves_raw.size();
    auto result = py::array_t<int>(py::detail::any_container<py::ssize_t>({num_leaves}));
    auto accessor = result.mutable_unchecked<1>();
    for (size_t i = 0; i < num_leaves; i++) {
      accessor(i) = leaves_raw.at(i);
    }
    return result;
  }

  py::array_t<int> GetTreeSplitCounts(int tree_num, int num_features) {
    auto result = py::array_t<int>(py::detail::any_container<py::ssize_t>({num_features}));
    auto accessor = result.mutable_unchecked<1>();
    for (size_t i = 0; i < num_features; i++) {
      accessor(i) = 0;
    }
    StochTree::Tree* tree = forest_->GetTree(tree_num);
    std::vector<int32_t> split_nodes = tree->GetInternalNodes();
    for (int i = 0; i < split_nodes.size(); i++) {
      auto node_id = split_nodes.at(i);
      auto split_feature = tree->SplitIndex(node_id);
      accessor(split_feature)++;
    }
    return result;
  }

  py::array_t<int> GetOverallSplitCounts(int num_features) {
    auto result = py::array_t<int>(py::detail::any_container<py::ssize_t>({num_features}));
    auto accessor = result.mutable_unchecked<1>();
    for (size_t i = 0; i < num_features; i++) {
      accessor(i) = 0;
    }
    int num_trees = forest_->NumTrees();
    for (int i = 0; i < num_trees; i++) {
      StochTree::Tree* tree = forest_->GetTree(i);
      std::vector<int32_t> split_nodes = tree->GetInternalNodes();
      for (int j = 0; j < split_nodes.size(); j++) {
        auto node_id = split_nodes.at(j);
        auto split_feature = tree->SplitIndex(node_id);
        accessor(split_feature)++;
      }
    }
    return result;
  }

  py::array_t<int> GetGranularSplitCounts(int num_features) {
    int num_trees = forest_->NumTrees();
    auto result = py::array_t<int>(py::detail::any_container<py::ssize_t>({num_trees, num_features}));
    auto accessor = result.mutable_unchecked<2>();
    for (int i = 0; i < num_trees; i++) {
      for (int j = 0; j < num_features; j++) {
        accessor(i, j) = 0;
      }
    }
    for (int i = 0; i < num_trees; i++) {
      StochTree::Tree* tree = forest_->GetTree(i);
      std::vector<int32_t> split_nodes = tree->GetInternalNodes();
      for (int j = 0; j < split_nodes.size(); j++) {
        auto node_id = split_nodes.at(i);
        auto split_feature = tree->SplitIndex(node_id);
        accessor(i, split_feature)++;
      }
    }
    return result;
  }

  bool IsLeafNode(int tree_id, int node_id) {
    StochTree::Tree* tree = forest_->GetTree(tree_id);
    return tree->IsLeaf(node_id);
  }

  bool IsNumericSplitNode(int tree_id, int node_id) {
    StochTree::Tree* tree = forest_->GetTree(tree_id);
    return tree->IsNumericSplitNode(node_id);
  }

  bool IsCategoricalSplitNode(int tree_id, int node_id) {
    StochTree::Tree* tree = forest_->GetTree(tree_id);
    return tree->IsCategoricalSplitNode(node_id);
  }

  int ParentNode(int tree_id, int node_id) {
    StochTree::Tree* tree = forest_->GetTree(tree_id);
    return tree->Parent(node_id);
  }

  int LeftChildNode(int tree_id, int node_id) {
    StochTree::Tree* tree = forest_->GetTree(tree_id);
    return tree->LeftChild(node_id);
  }

  int RightChildNode(int tree_id, int node_id) {
    StochTree::Tree* tree = forest_->GetTree(tree_id);
    return tree->RightChild(node_id);
  }

  int SplitIndex(int tree_id, int node_id) {
    StochTree::Tree* tree = forest_->GetTree(tree_id);
    return tree->SplitIndex(node_id);
  }

  int NodeDepth(int tree_id, int node_id) {
    StochTree::Tree* tree = forest_->GetTree(tree_id);
    return tree->GetDepth(node_id);
  }

  double SplitThreshold(int tree_id, int node_id) {
    StochTree::Tree* tree = forest_->GetTree(tree_id);
    return tree->Threshold(node_id);
  }

  py::array_t<int> SplitCategories(int tree_id, int node_id) {
    StochTree::Tree* tree = forest_->GetTree(tree_id);
    std::vector<std::uint32_t> raw_categories = tree->CategoryList(node_id);
    int num_categories = raw_categories.size();
    auto result = py::array_t<int>(py::detail::any_container<py::ssize_t>({num_categories}));
    auto accessor = result.mutable_unchecked<1>();
    for (int i = 0; i < num_categories; i++) {
      accessor(i) = raw_categories.at(i);
    }
    return result;
  }

  py::array_t<double> NodeLeafValues(int tree_id, int node_id) {
    StochTree::Tree* tree = forest_->GetTree(tree_id);
    int num_outputs = tree->OutputDimension();
    auto result = py::array_t<double>(py::detail::any_container<py::ssize_t>({num_outputs}));
    auto accessor = result.mutable_unchecked<1>();
    for (int i = 0; i < num_outputs; i++) {
      accessor(i) = tree->LeafValue(node_id, i);
    }
    return result;
  }

  int NumNodes(int tree_id) {
    StochTree::Tree* tree = forest_->GetTree(tree_id);
    return tree->NumValidNodes();
  }

  int NumLeaves(int tree_id) {
    StochTree::Tree* tree = forest_->GetTree(tree_id);
    return tree->NumLeaves();
  }

  int NumLeafParents(int tree_id) {
    StochTree::Tree* tree = forest_->GetTree(tree_id);
    return tree->NumLeafParents();
  }

  int NumSplitNodes(int tree_id) {
    StochTree::Tree* tree = forest_->GetTree(tree_id);
    return tree->NumSplitNodes();
  }

  py::array_t<int> Nodes(int tree_id) {
    StochTree::Tree* tree = forest_->GetTree(tree_id);
    std::vector<std::int32_t> nodes = tree->GetNodes();
    int num_nodes = nodes.size();
    auto result = py::array_t<int>(py::detail::any_container<py::ssize_t>({num_nodes}));
    auto accessor = result.mutable_unchecked<1>();
    for (int i = 0; i < num_nodes; i++) {
      accessor(i) = nodes.at(i);
    }
    return result;
  }

  py::array_t<int> Leaves(int tree_id) {
    StochTree::Tree* tree = forest_->GetTree(tree_id);
    std::vector<std::int32_t> leaves = tree->GetLeaves();
    int num_leaves = leaves.size();
    auto result = py::array_t<int>(py::detail::any_container<py::ssize_t>({num_leaves}));
    auto accessor = result.mutable_unchecked<1>();
    for (int i = 0; i < num_leaves; i++) {
      accessor(i) = leaves.at(i);
    }
    return result;
  }

 private:
  std::unique_ptr<StochTree::TreeEnsemble> forest_;
  int num_trees_;
  int output_dimension_;
  bool is_leaf_constant_;
  bool is_exponentiated_;
};

class ForestSamplerCpp {
 public:
  ForestSamplerCpp(ForestDatasetCpp& dataset, py::array_t<int> feature_types, int num_trees, data_size_t num_obs, double alpha, double beta, int min_samples_leaf, int max_depth) {
    // Convert vector of integers to std::vector of enum FeatureType
    std::vector<StochTree::FeatureType> feature_types_(feature_types.size());
    for (int i = 0; i < feature_types.size(); i++) {
      feature_types_[i] = static_cast<StochTree::FeatureType>(feature_types.at(i));
    }

    // Initialize pointer to C++ ForestTracker and TreePrior classes
    StochTree::ForestDataset* dataset_ptr = dataset.GetDataset();
    tracker_ = std::make_unique<StochTree::ForestTracker>(dataset_ptr->GetCovariates(), feature_types_, num_trees, num_obs);
    split_prior_ = std::make_unique<StochTree::TreePrior>(alpha, beta, min_samples_leaf, max_depth);
  }
  ~ForestSamplerCpp() {}

  StochTree::ForestTracker* GetTracker() { return tracker_.get(); }

  void ReconstituteTrackerFromForest(ForestCpp& forest, ForestDatasetCpp& dataset, ResidualCpp& residual, bool is_mean_model) {
    // Extract raw pointer to the forest and dataset
    StochTree::TreeEnsemble* forest_ptr = forest.GetEnsemble();
    StochTree::ForestDataset* data_ptr = dataset.GetDataset();
    StochTree::ColumnVector* residual_ptr = residual.GetData();

    // Reset forest tracker using the forest held at index forest_num
    tracker_->ReconstituteFromForest(*forest_ptr, *data_ptr, *residual_ptr, is_mean_model);
  }

  void SampleOneIteration(ForestContainerCpp& forest_samples, ForestCpp& forest, ForestDatasetCpp& dataset, ResidualCpp& residual, RngCpp& rng,
                          py::array_t<int> feature_types, py::array_t<int> sweep_update_indices, int cutpoint_grid_size, py::array_t<double, py::array::forcecast> leaf_model_scale_input,
                          py::array_t<double, py::array::forcecast> variable_weights, double a_forest, double b_forest, double global_variance,
                          int leaf_model_int, int num_features_subsample, bool keep_forest = true, bool gfr = true, int num_threads = -1) {
    // Refactoring completely out of the Python interface.
    // Intention to refactor out of the C++ interface in the future.
    bool pre_initialized = true;

    // Unpack feature types
    std::vector<StochTree::FeatureType> feature_types_(feature_types.size());
    for (int i = 0; i < feature_types.size(); i++) {
      feature_types_[i] = static_cast<StochTree::FeatureType>(feature_types.at(i));
    }

    // Unpack sweep indices
    std::vector<int> sweep_update_indices_;
    if (sweep_update_indices.size() > 0) {
      sweep_update_indices_.resize(sweep_update_indices.size());
      for (int i = 0; i < sweep_update_indices.size(); i++) {
        sweep_update_indices_[i] = sweep_update_indices.at(i);
      }
    }

    // Convert leaf model type to enum
    StochTree::ModelType model_type;
    if (leaf_model_int == 0)
      model_type = StochTree::ModelType::kConstantLeafGaussian;
    else if (leaf_model_int == 1)
      model_type = StochTree::ModelType::kUnivariateRegressionLeafGaussian;
    else if (leaf_model_int == 2)
      model_type = StochTree::ModelType::kMultivariateRegressionLeafGaussian;
    else if (leaf_model_int == 3)
      model_type = StochTree::ModelType::kLogLinearVariance;
    else if (leaf_model_int == 4)
      model_type = StochTree::ModelType::kCloglogOrdinal;

    // Unpack leaf model parameters
    double leaf_scale;
    Eigen::MatrixXd leaf_scale_matrix;
    if ((model_type == StochTree::ModelType::kConstantLeafGaussian) ||
        (model_type == StochTree::ModelType::kUnivariateRegressionLeafGaussian)) {
      leaf_scale = leaf_model_scale_input.at(0, 0);
    } else if (model_type == StochTree::ModelType::kMultivariateRegressionLeafGaussian) {
      int num_row = leaf_model_scale_input.shape(0);
      int num_col = leaf_model_scale_input.shape(1);
      leaf_scale_matrix.resize(num_row, num_col);
      for (int i = 0; i < num_row; i++) {
        for (int j = 0; j < num_col; j++) {
          leaf_scale_matrix(i, j) = leaf_model_scale_input.at(i, j);
        }
      }
    }

    // Convert variable weights to std::vector
    std::vector<double> var_weights_vector(variable_weights.size());
    for (int i = 0; i < variable_weights.size(); i++) {
      var_weights_vector[i] = variable_weights.at(i);
    }

    // Prepare the samplers
    StochTree::LeafModelVariant leaf_model = StochTree::leafModelFactory(model_type, leaf_scale, leaf_scale_matrix, a_forest, b_forest);

    // Run one iteration of the sampler
    StochTree::ForestContainer* forest_sample_ptr = forest_samples.GetContainer();
    StochTree::TreeEnsemble* active_forest_ptr = forest.GetEnsemble();
    StochTree::ForestDataset* forest_data_ptr = dataset.GetDataset();
    StochTree::ColumnVector* residual_data_ptr = residual.GetData();
    int num_basis = forest_data_ptr->NumBasis();
    std::mt19937* rng_ptr = rng.GetRng();
    if (gfr) {
      if (model_type == StochTree::ModelType::kConstantLeafGaussian) {
        StochTree::GFRSampleOneIter<StochTree::GaussianConstantLeafModel, StochTree::GaussianConstantSuffStat>(*active_forest_ptr, *(tracker_.get()), *forest_sample_ptr, std::get<StochTree::GaussianConstantLeafModel>(leaf_model), *forest_data_ptr, *residual_data_ptr, *(split_prior_.get()), *rng_ptr, var_weights_vector, sweep_update_indices_, global_variance, feature_types_, cutpoint_grid_size, keep_forest, pre_initialized, true, num_features_subsample, num_threads);
      } else if (model_type == StochTree::ModelType::kUnivariateRegressionLeafGaussian) {
        StochTree::GFRSampleOneIter<StochTree::GaussianUnivariateRegressionLeafModel, StochTree::GaussianUnivariateRegressionSuffStat>(*active_forest_ptr, *(tracker_.get()), *forest_sample_ptr, std::get<StochTree::GaussianUnivariateRegressionLeafModel>(leaf_model), *forest_data_ptr, *residual_data_ptr, *(split_prior_.get()), *rng_ptr, var_weights_vector, sweep_update_indices_, global_variance, feature_types_, cutpoint_grid_size, keep_forest, pre_initialized, true, num_features_subsample, num_threads);
      } else if (model_type == StochTree::ModelType::kMultivariateRegressionLeafGaussian) {
        StochTree::GFRSampleOneIter<StochTree::GaussianMultivariateRegressionLeafModel, StochTree::GaussianMultivariateRegressionSuffStat, int>(*active_forest_ptr, *(tracker_.get()), *forest_sample_ptr, std::get<StochTree::GaussianMultivariateRegressionLeafModel>(leaf_model), *forest_data_ptr, *residual_data_ptr, *(split_prior_.get()), *rng_ptr, var_weights_vector, sweep_update_indices_, global_variance, feature_types_, cutpoint_grid_size, keep_forest, pre_initialized, true, num_features_subsample, num_threads, num_basis);
      } else if (model_type == StochTree::ModelType::kLogLinearVariance) {
        StochTree::GFRSampleOneIter<StochTree::LogLinearVarianceLeafModel, StochTree::LogLinearVarianceSuffStat>(*active_forest_ptr, *(tracker_.get()), *forest_sample_ptr, std::get<StochTree::LogLinearVarianceLeafModel>(leaf_model), *forest_data_ptr, *residual_data_ptr, *(split_prior_.get()), *rng_ptr, var_weights_vector, sweep_update_indices_, global_variance, feature_types_, cutpoint_grid_size, keep_forest, pre_initialized, false, num_features_subsample, num_threads);
      } else if (model_type == StochTree::ModelType::kCloglogOrdinal) {
        StochTree::GFRSampleOneIter<StochTree::CloglogOrdinalLeafModel, StochTree::CloglogOrdinalSuffStat>(*active_forest_ptr, *(tracker_.get()), *forest_sample_ptr, std::get<StochTree::CloglogOrdinalLeafModel>(leaf_model), *forest_data_ptr, *residual_data_ptr, *(split_prior_.get()), *rng_ptr, var_weights_vector, sweep_update_indices_, global_variance, feature_types_, cutpoint_grid_size, keep_forest, pre_initialized, false, num_features_subsample, num_threads);
      }
    } else {
      if (model_type == StochTree::ModelType::kConstantLeafGaussian) {
        StochTree::MCMCSampleOneIter<StochTree::GaussianConstantLeafModel, StochTree::GaussianConstantSuffStat>(*active_forest_ptr, *(tracker_.get()), *forest_sample_ptr, std::get<StochTree::GaussianConstantLeafModel>(leaf_model), *forest_data_ptr, *residual_data_ptr, *(split_prior_.get()), *rng_ptr, var_weights_vector, sweep_update_indices_, global_variance, keep_forest, pre_initialized, true, num_threads);
      } else if (model_type == StochTree::ModelType::kUnivariateRegressionLeafGaussian) {
        StochTree::MCMCSampleOneIter<StochTree::GaussianUnivariateRegressionLeafModel, StochTree::GaussianUnivariateRegressionSuffStat>(*active_forest_ptr, *(tracker_.get()), *forest_sample_ptr, std::get<StochTree::GaussianUnivariateRegressionLeafModel>(leaf_model), *forest_data_ptr, *residual_data_ptr, *(split_prior_.get()), *rng_ptr, var_weights_vector, sweep_update_indices_, global_variance, keep_forest, pre_initialized, true, num_threads);
      } else if (model_type == StochTree::ModelType::kMultivariateRegressionLeafGaussian) {
        StochTree::MCMCSampleOneIter<StochTree::GaussianMultivariateRegressionLeafModel, StochTree::GaussianMultivariateRegressionSuffStat, int>(*active_forest_ptr, *(tracker_.get()), *forest_sample_ptr, std::get<StochTree::GaussianMultivariateRegressionLeafModel>(leaf_model), *forest_data_ptr, *residual_data_ptr, *(split_prior_.get()), *rng_ptr, var_weights_vector, sweep_update_indices_, global_variance, keep_forest, pre_initialized, true, num_threads, num_basis);
      } else if (model_type == StochTree::ModelType::kLogLinearVariance) {
        StochTree::MCMCSampleOneIter<StochTree::LogLinearVarianceLeafModel, StochTree::LogLinearVarianceSuffStat>(*active_forest_ptr, *(tracker_.get()), *forest_sample_ptr, std::get<StochTree::LogLinearVarianceLeafModel>(leaf_model), *forest_data_ptr, *residual_data_ptr, *(split_prior_.get()), *rng_ptr, var_weights_vector, sweep_update_indices_, global_variance, keep_forest, pre_initialized, false, num_threads);
      } else if (model_type == StochTree::ModelType::kCloglogOrdinal) {
        StochTree::MCMCSampleOneIter<StochTree::CloglogOrdinalLeafModel, StochTree::CloglogOrdinalSuffStat>(*active_forest_ptr, *(tracker_.get()), *forest_sample_ptr, std::get<StochTree::CloglogOrdinalLeafModel>(leaf_model), *forest_data_ptr, *residual_data_ptr, *(split_prior_.get()), *rng_ptr, var_weights_vector, sweep_update_indices_, global_variance, keep_forest, pre_initialized, false, num_threads);
      }
    }
  }

  void InitializeForestModel(ForestDatasetCpp& dataset, ResidualCpp& residual, ForestCpp& forest,
                             int leaf_model_int, py::array_t<double, py::array::forcecast> initial_values) {
    // Convert leaf model type to enum
    StochTree::ModelType model_type;
    if (leaf_model_int == 0)
      model_type = StochTree::ModelType::kConstantLeafGaussian;
    else if (leaf_model_int == 1)
      model_type = StochTree::ModelType::kUnivariateRegressionLeafGaussian;
    else if (leaf_model_int == 2)
      model_type = StochTree::ModelType::kMultivariateRegressionLeafGaussian;
    else if (leaf_model_int == 3)
      model_type = StochTree::ModelType::kLogLinearVariance;
    else if (leaf_model_int == 4)
      model_type = StochTree::ModelType::kCloglogOrdinal;
    else
      StochTree::Log::Fatal("Invalid model type");

    // Unpack initial value
    StochTree::TreeEnsemble* forest_ptr = forest.GetEnsemble();
    StochTree::ForestDataset* forest_data_ptr = dataset.GetDataset();
    StochTree::ColumnVector* residual_data_ptr = residual.GetData();
    int num_trees = forest_ptr->NumTrees();
    double init_val;
    std::vector<double> init_value_vector;
    if ((model_type == StochTree::ModelType::kConstantLeafGaussian) ||
        (model_type == StochTree::ModelType::kUnivariateRegressionLeafGaussian) ||
        (model_type == StochTree::ModelType::kLogLinearVariance) ||
        (model_type == StochTree::ModelType::kCloglogOrdinal)) {
      init_val = initial_values.at(0);
    } else if (model_type == StochTree::ModelType::kMultivariateRegressionLeafGaussian) {
      int leaf_dim = initial_values.size();
      init_value_vector.resize(leaf_dim);
      for (int i = 0; i < leaf_dim; i++) {
        init_value_vector[i] = initial_values.at(i) / static_cast<double>(num_trees);
      }
    }

    // Initialize the models accordingly
    double leaf_init_val;
    if (model_type == StochTree::ModelType::kConstantLeafGaussian) {
      leaf_init_val = init_val / static_cast<double>(num_trees);
      forest_ptr->SetLeafValue(leaf_init_val);
      StochTree::UpdateResidualEntireForest(*tracker_, *forest_data_ptr, *residual_data_ptr, forest_ptr, false, std::minus<double>());
      tracker_->UpdatePredictions(forest_ptr, *forest_data_ptr);
    } else if (model_type == StochTree::ModelType::kUnivariateRegressionLeafGaussian) {
      leaf_init_val = init_val / static_cast<double>(num_trees);
      forest_ptr->SetLeafValue(leaf_init_val);
      StochTree::UpdateResidualEntireForest(*tracker_, *forest_data_ptr, *residual_data_ptr, forest_ptr, true, std::minus<double>());
      tracker_->UpdatePredictions(forest_ptr, *forest_data_ptr);
    } else if (model_type == StochTree::ModelType::kMultivariateRegressionLeafGaussian) {
      forest_ptr->SetLeafVector(init_value_vector);
      StochTree::UpdateResidualEntireForest(*tracker_, *forest_data_ptr, *residual_data_ptr, forest_ptr, true, std::minus<double>());
      tracker_->UpdatePredictions(forest_ptr, *forest_data_ptr);
    } else if (model_type == StochTree::ModelType::kLogLinearVariance) {
      leaf_init_val = std::log(init_val) / static_cast<double>(num_trees);
      forest_ptr->SetLeafValue(leaf_init_val);
      tracker_->UpdatePredictions(forest_ptr, *forest_data_ptr);
      int n = forest_data_ptr->NumObservations();
      std::vector<double> initial_preds(n, init_val);
      forest_data_ptr->AddVarianceWeights(initial_preds.data(), n);
    } else if (model_type == StochTree::ModelType::kCloglogOrdinal) {
      leaf_init_val = init_val / static_cast<double>(num_trees);
      forest_ptr->SetLeafValue(leaf_init_val);
      StochTree::UpdateResidualEntireForest(*tracker_, *forest_data_ptr, *residual_data_ptr, forest_ptr, false, std::minus<double>());
      tracker_->UpdatePredictions(forest_ptr, *forest_data_ptr);
    }
  }

  py::array_t<double> GetCachedForestPredictions() {
    int n_train = tracker_->GetNumObservations();
    auto output = py::array_t<double>(py::detail::any_container<py::ssize_t>({n_train}));
    auto accessor = output.mutable_unchecked<1>();
    for (size_t i = 0; i < n_train; i++) {
      accessor(i) = tracker_->GetSamplePrediction(i);
    }
    return output;
  }

  void PropagateBasisUpdate(ForestDatasetCpp& dataset, ResidualCpp& residual, ForestCpp& forest) {
    // Perform the update operation
    StochTree::UpdateResidualNewBasis(*tracker_, *(dataset.GetDataset()), *(residual.GetData()), forest.GetEnsemble());
  }

  void PropagateResidualUpdate(ResidualCpp& residual) {
    // Extract pointer to forest tracker
    StochTree::ColumnVector* residual_ptr = residual.GetData();
    // Propagate update to the residual through the trackers
    StochTree::UpdateResidualNewOutcome(*tracker_, *residual_ptr);
  }

  void UpdateAlpha(double alpha) {
    split_prior_->SetAlpha(alpha);
  }

  void UpdateBeta(double beta) {
    split_prior_->SetBeta(beta);
  }

  void UpdateMinSamplesLeaf(int min_samples_leaf) {
    split_prior_->SetMinSamplesLeaf(min_samples_leaf);
  }

  void UpdateMaxDepth(int max_depth) {
    split_prior_->SetMaxDepth(max_depth);
  }

  double GetAlpha() {
    return split_prior_->GetAlpha();
  }

  double GetBeta() {
    return split_prior_->GetBeta();
  }

  int GetMinSamplesLeaf() {
    return split_prior_->GetMinSamplesLeaf();
  }

  int GetMaxDepth() {
    return split_prior_->GetMaxDepth();
  }

 private:
  std::unique_ptr<StochTree::ForestTracker> tracker_;
  std::unique_ptr<StochTree::TreePrior> split_prior_;
};

class GlobalVarianceModelCpp {
 public:
  GlobalVarianceModelCpp() {
    var_model_ = StochTree::GlobalHomoskedasticVarianceModel();
  }
  ~GlobalVarianceModelCpp() {}

  double SampleOneIteration(ResidualCpp& residual, RngCpp& rng, double a, double b) {
    StochTree::ColumnVector* residual_ptr = residual.GetData();
    std::mt19937* rng_ptr = rng.GetRng();
    return var_model_.SampleVarianceParameter(residual_ptr->GetData(), a, b, *rng_ptr);
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

  double SampleOneIteration(ForestCpp& forest, RngCpp& rng, double a, double b) {
    StochTree::TreeEnsemble* forest_ptr = forest.GetEnsemble();
    std::mt19937* rng_ptr = rng.GetRng();
    return var_model_.SampleVarianceParameter(forest_ptr, a, b, *rng_ptr);
  }

 private:
  StochTree::LeafNodeHomoskedasticVarianceModel var_model_;
};

class OrdinalSamplerCpp {
 public:
  OrdinalSamplerCpp() {
    ordinal_sampler_ = std::make_unique<StochTree::OrdinalSampler>();
  }
  ~OrdinalSamplerCpp() {}

  void UpdateLatentVariables(ForestDatasetCpp& dataset, ResidualCpp& outcome, RngCpp& rng) {
    StochTree::ForestDataset* dataset_ptr = dataset.GetDataset();
    StochTree::ColumnVector* outcome_ptr = outcome.GetData();
    std::mt19937* rng_ptr = rng.GetRng();
    Eigen::VectorXd& outcome_vec = outcome_ptr->GetData();
    ordinal_sampler_->UpdateLatentVariables(*dataset_ptr, outcome_vec, *rng_ptr);
  }

  void UpdateGammaParams(ForestDatasetCpp& dataset, ResidualCpp& outcome,
                         double alpha_gamma, double beta_gamma,
                         double gamma_0, RngCpp& rng) {
    StochTree::ForestDataset* dataset_ptr = dataset.GetDataset();
    StochTree::ColumnVector* outcome_ptr = outcome.GetData();
    std::mt19937* rng_ptr = rng.GetRng();
    Eigen::VectorXd& outcome_vec = outcome_ptr->GetData();
    ordinal_sampler_->UpdateGammaParams(*dataset_ptr, outcome_vec,
                                        alpha_gamma, beta_gamma, gamma_0, *rng_ptr);
  }

  void UpdateCumulativeExpSums(ForestDatasetCpp& dataset) {
    StochTree::ForestDataset* dataset_ptr = dataset.GetDataset();
    ordinal_sampler_->UpdateCumulativeExpSums(*dataset_ptr);
  }

 private:
  std::unique_ptr<StochTree::OrdinalSampler> ordinal_sampler_;
};

class RandomEffectsDatasetCpp {
 public:
  RandomEffectsDatasetCpp() {
    rfx_dataset_ = std::make_unique<StochTree::RandomEffectsDataset>();
  }
  ~RandomEffectsDatasetCpp() {}
  StochTree::RandomEffectsDataset* GetDataset() {
    return rfx_dataset_.get();
  }
  py::ssize_t NumObservations() {
    return rfx_dataset_->NumObservations();
  }
  int NumBases() {
    return rfx_dataset_->NumBases();
  }
  void AddGroupLabels(py::array_t<int> group_labels, data_size_t num_row) {
    std::vector<int> group_labels_vec(num_row);
    auto accessor = group_labels.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < num_row; i++) {
      group_labels_vec[i] = accessor(i);
    }
    rfx_dataset_->AddGroupLabels(group_labels_vec);
  }
  void AddBasis(py::array_t<double, py::array::forcecast> basis, data_size_t num_row, int num_col, bool row_major) {
    double* basis_data_ptr = static_cast<double*>(basis.mutable_data());
    rfx_dataset_->AddBasis(basis_data_ptr, num_row, num_col, row_major);
  }
  void AddVarianceWeights(py::array_t<double, py::array::forcecast> weights, data_size_t num_row) {
    double* weight_data_ptr = static_cast<double*>(weights.mutable_data());
    rfx_dataset_->AddVarianceWeights(weight_data_ptr, num_row);
  }
  void UpdateBasis(py::array_t<double, py::array::forcecast> basis, data_size_t num_row, int num_col, bool row_major) {
    double* basis_data_ptr = static_cast<double*>(basis.mutable_data());
    rfx_dataset_->UpdateBasis(basis_data_ptr, num_row, num_col, row_major);
  }
  void UpdateVarianceWeights(py::array_t<double, py::array::forcecast> weights, data_size_t num_row, bool exponentiate) {
    double* weight_data_ptr = static_cast<double*>(weights.mutable_data());
    rfx_dataset_->UpdateVarWeights(weight_data_ptr, num_row, exponentiate);
  }
  void UpdateGroupLabels(py::array_t<int> group_labels, data_size_t num_row) {
    std::vector<int> group_labels_vec(num_row);
    auto accessor = group_labels.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < num_row; i++) {
      group_labels_vec[i] = accessor(i);
    }
    rfx_dataset_->UpdateGroupLabels(group_labels_vec, num_row);
  }
  py::array_t<double> GetBasis() {
    int num_row = rfx_dataset_->NumObservations();
    int num_col = rfx_dataset_->NumBases();
    auto result = py::array_t<double>(py::detail::any_container<py::ssize_t>({num_row, num_col}));
    auto accessor = result.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < num_row; i++) {
      for (int j = 0; j < num_col; j++) {
        accessor(i, j) = rfx_dataset_->BasisValue(i, j);
      }
    }
    return result;
  }
  py::array_t<double> GetVarianceWeights() {
    int num_row = rfx_dataset_->NumObservations();
    auto result = py::array_t<double>(py::detail::any_container<py::ssize_t>({num_row}));
    auto accessor = result.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < num_row; i++) {
      accessor(i) = rfx_dataset_->VarWeightValue(i);
    }
    return result;
  }
  py::array_t<int> GetGroupLabels() {
    int num_row = rfx_dataset_->NumObservations();
    auto result = py::array_t<int>(py::detail::any_container<py::ssize_t>({num_row}));
    auto accessor = result.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < num_row; i++) {
      accessor(i) = rfx_dataset_->GroupId(i);
    }
    return result;
  }
  bool HasGroupLabels() { return rfx_dataset_->HasGroupLabels(); }
  bool HasBasis() { return rfx_dataset_->HasBasis(); }
  bool HasVarianceWeights() { return rfx_dataset_->HasVarWeights(); }

 private:
  std::unique_ptr<StochTree::RandomEffectsDataset> rfx_dataset_;
};

class RandomEffectsModelCpp;

class RandomEffectsLabelMapperCpp;

class RandomEffectsContainerCpp {
 public:
  RandomEffectsContainerCpp() {
    rfx_container_ = std::make_unique<StochTree::RandomEffectsContainer>();
  }
  explicit RandomEffectsContainerCpp(std::unique_ptr<StochTree::RandomEffectsContainer> ptr)
      : rfx_container_(std::move(ptr)) {}

  ~RandomEffectsContainerCpp() {}

  StochTree::RandomEffectsContainer* GetPtr() { return rfx_container_.get(); }

  void SetComponentsAndGroups(int num_components, int num_groups) {
    rfx_container_->SetNumComponents(num_components);
    rfx_container_->SetNumGroups(num_groups);
  }
  void AddSample(RandomEffectsModelCpp& rfx_model);
  int NumSamples() {
    return rfx_container_->NumSamples();
  }
  int NumComponents() {
    return rfx_container_->NumComponents();
  }
  int NumGroups() {
    return rfx_container_->NumGroups();
  }
  py::array_t<double> GetBeta() {
    int num_samples = rfx_container_->NumSamples();
    int num_components = rfx_container_->NumComponents();
    int num_groups = rfx_container_->NumGroups();
    std::vector<double> beta_raw = rfx_container_->GetBeta();
    auto result = py::array_t<double>(py::detail::any_container<py::ssize_t>({num_components, num_groups, num_samples}));
    auto accessor = result.mutable_unchecked<3>();
    for (int i = 0; i < num_components; i++) {
      for (int j = 0; j < num_groups; j++) {
        for (int k = 0; k < num_samples; k++) {
          accessor(i, j, k) = beta_raw[k * num_groups * num_components + j * num_components + i];
        }
      }
    }
    return result;
  }
  py::array_t<double> GetXi() {
    int num_samples = rfx_container_->NumSamples();
    int num_components = rfx_container_->NumComponents();
    int num_groups = rfx_container_->NumGroups();
    std::vector<double> xi_raw = rfx_container_->GetXi();
    auto result = py::array_t<double>(py::detail::any_container<py::ssize_t>({num_components, num_groups, num_samples}));
    auto accessor = result.mutable_unchecked<3>();
    for (int i = 0; i < num_components; i++) {
      for (int j = 0; j < num_groups; j++) {
        for (int k = 0; k < num_samples; k++) {
          accessor(i, j, k) = xi_raw[k * num_groups * num_components + j * num_components + i];
        }
      }
    }
    return result;
  }
  py::array_t<double> GetAlpha() {
    int num_samples = rfx_container_->NumSamples();
    int num_components = rfx_container_->NumComponents();
    std::vector<double> alpha_raw = rfx_container_->GetAlpha();
    auto result = py::array_t<double>(py::detail::any_container<py::ssize_t>({num_components, num_samples}));
    auto accessor = result.mutable_unchecked<2>();
    for (int i = 0; i < num_components; i++) {
      for (int j = 0; j < num_samples; j++) {
        accessor(i, j) = alpha_raw[j * num_components + i];
      }
    }
    return result;
  }
  py::array_t<double> GetSigma() {
    int num_samples = rfx_container_->NumSamples();
    int num_components = rfx_container_->NumComponents();
    std::vector<double> sigma_raw = rfx_container_->GetSigma();
    auto result = py::array_t<double>(py::detail::any_container<py::ssize_t>({num_components, num_samples}));
    auto accessor = result.mutable_unchecked<2>();
    for (int i = 0; i < num_components; i++) {
      for (int j = 0; j < num_samples; j++) {
        accessor(i, j) = sigma_raw[j * num_components + i];
      }
    }
    return result;
  }
  void DeleteSample(int sample_num) {
    rfx_container_->DeleteSample(sample_num);
  }
  py::array_t<double> Predict(RandomEffectsDatasetCpp& rfx_dataset, RandomEffectsLabelMapperCpp& label_mapper);
  void SaveToJsonFile(std::string json_filename) {
    rfx_container_->SaveToJsonFile(json_filename);
  }
  void LoadFromJsonFile(std::string json_filename) {
    rfx_container_->LoadFromJsonFile(json_filename);
  }
  std::string DumpJsonString() {
    return rfx_container_->DumpJsonString();
  }
  void LoadFromJsonString(std::string& json_string) {
    rfx_container_->LoadFromJsonString(json_string);
  }
  void LoadFromJson(JsonCpp& json, std::string rfx_container_label);
  void AppendFromJson(JsonCpp& json, std::string rfx_container_label);
  StochTree::RandomEffectsContainer* GetRandomEffectsContainer() {
    return rfx_container_.get();
  }

 private:
  std::unique_ptr<StochTree::RandomEffectsContainer> rfx_container_;
};

class RandomEffectsTrackerCpp {
 public:
  RandomEffectsTrackerCpp(py::array_t<int> group_labels) {
    int vec_size = group_labels.size();
    std::vector<int32_t> group_labels_vec(vec_size);
    for (int i = 0; i < vec_size; i++) {
      group_labels_vec[i] = group_labels.at(i);
    }
    rfx_tracker_ = std::make_unique<StochTree::RandomEffectsTracker>(group_labels_vec);
  }
  ~RandomEffectsTrackerCpp() {}
  py::array_t<int> GetUniqueGroupIds() {
    std::vector<int> output = rfx_tracker_->GetUniqueGroupIds();
    py::ssize_t output_length = output.size();
    auto result = py::array_t<int>(py::detail::any_container<py::ssize_t>({output_length}));
    auto accessor = result.mutable_unchecked<1>();
    for (size_t i = 0; i < output_length; i++) {
      accessor(i) = output.at(i);
    }
    return result;
  }
  StochTree::RandomEffectsTracker* GetTracker() {
    return rfx_tracker_.get();
  }
  void Reset(RandomEffectsModelCpp& rfx_model, RandomEffectsDatasetCpp& rfx_dataset, ResidualCpp& residual);
  void RootReset(RandomEffectsModelCpp& rfx_model, RandomEffectsDatasetCpp& rfx_dataset, ResidualCpp& residual);

 private:
  std::unique_ptr<StochTree::RandomEffectsTracker> rfx_tracker_;
};

class RandomEffectsLabelMapperCpp {
 public:
  RandomEffectsLabelMapperCpp() {
    rfx_label_mapper_ = std::make_unique<StochTree::LabelMapper>();
  }
  explicit RandomEffectsLabelMapperCpp(std::unique_ptr<StochTree::LabelMapper> ptr)
      : rfx_label_mapper_(std::move(ptr)) {}

  ~RandomEffectsLabelMapperCpp() {}

  StochTree::LabelMapper* GetPtr() { return rfx_label_mapper_.get(); }

  void LoadFromTracker(RandomEffectsTrackerCpp& rfx_tracker) {
    StochTree::RandomEffectsTracker* internal_tracker = rfx_tracker.GetTracker();
    rfx_label_mapper_->LoadFromLabelMap(internal_tracker->GetLabelMap());
  }
  void SaveToJsonFile(std::string json_filename) {
    rfx_label_mapper_->SaveToJsonFile(json_filename);
  }
  void LoadFromJsonFile(std::string json_filename) {
    rfx_label_mapper_->LoadFromJsonFile(json_filename);
  }
  std::string DumpJsonString() {
    return rfx_label_mapper_->DumpJsonString();
  }
  void LoadFromJsonString(std::string& json_string) {
    rfx_label_mapper_->LoadFromJsonString(json_string);
  }
  void LoadFromJson(JsonCpp& json, std::string rfx_label_mapper_label);
  StochTree::LabelMapper* GetLabelMapper() {
    return rfx_label_mapper_.get();
  }
  int MapGroupIdToArrayIndex(int original_label) {
    return rfx_label_mapper_->CategoryNumber(original_label);
  }
  py::array_t<int> MapMultipleGroupIdsToArrayIndices(py::array_t<int> original_labels) {
    int output_size = original_labels.size();
    auto result = py::array_t<int>(py::detail::any_container<py::ssize_t>({output_size}));
    auto accessor = result.mutable_unchecked<1>();
    for (int i = 0; i < output_size; i++) {
      accessor(i) = rfx_label_mapper_->CategoryNumber(original_labels.at(i));
    }
    return result;
  }
  py::array_t<int> GetUniqueGroupIds() {
    std::vector<int>& keys = rfx_label_mapper_->Keys();
    auto result = py::array_t<int>(py::detail::any_container<py::ssize_t>({(py::ssize_t)keys.size()}));
    auto accessor = result.mutable_unchecked<1>();
    for (int i = 0; i < (int)keys.size(); i++) accessor(i) = keys[i];
    return result;
  }

 private:
  std::unique_ptr<StochTree::LabelMapper> rfx_label_mapper_;
};

class RandomEffectsModelCpp {
 public:
  RandomEffectsModelCpp(int num_components, int num_groups) {
    rfx_model_ = std::make_unique<StochTree::MultivariateRegressionRandomEffectsModel>(num_components, num_groups);
  }
  ~RandomEffectsModelCpp() {}
  StochTree::MultivariateRegressionRandomEffectsModel* GetModel() {
    return rfx_model_.get();
  }
  void SampleRandomEffects(RandomEffectsDatasetCpp& rfx_dataset, ResidualCpp& residual,
                           RandomEffectsTrackerCpp& rfx_tracker, RandomEffectsContainerCpp& rfx_container,
                           bool keep_sample, double global_variance, RngCpp& rng);
  py::array_t<double> Predict(RandomEffectsDatasetCpp& rfx_dataset, RandomEffectsTrackerCpp& rfx_tracker) {
    std::vector<double> output = rfx_model_->Predict(*rfx_dataset.GetDataset(), *rfx_tracker.GetTracker());
    py::ssize_t output_length = output.size();
    auto result = py::array_t<double>(py::detail::any_container<py::ssize_t>({output_length}));
    auto accessor = result.mutable_unchecked<1>();
    for (size_t i = 0; i < output_length; i++) {
      accessor(i) = output.at(i);
    }
    return result;
  }
  void SetWorkingParameter(py::array_t<double>& working_param) {
    Eigen::VectorXd working_param_eigen(working_param.size());
    for (int i = 0; i < working_param.size(); i++) {
      working_param_eigen(i) = working_param.at(i);
    }
    rfx_model_->SetWorkingParameter(working_param_eigen);
  }
  void SetGroupParameters(py::array_t<double>& group_params) {
    py::ssize_t nrow = group_params.shape(0);
    py::ssize_t ncol = group_params.shape(1);
    Eigen::MatrixXd group_params_eigen(nrow, ncol);
    for (py::ssize_t i = 0; i < nrow; i++) {
      for (int j = 0; j < ncol; j++) {
        group_params_eigen(i, j) = group_params.at(i, j);
      }
    }
    rfx_model_->SetGroupParameters(group_params_eigen);
  }
  void SetWorkingParameterCovariance(py::array_t<double>& working_param_cov) {
    int nrow = working_param_cov.shape(0);
    int ncol = working_param_cov.shape(1);
    Eigen::MatrixXd working_param_cov_eigen(nrow, ncol);
    for (int i = 0; i < nrow; i++) {
      for (int j = 0; j < ncol; j++) {
        working_param_cov_eigen(i, j) = working_param_cov.at(i, j);
      }
    }
    rfx_model_->SetWorkingParameterCovariance(working_param_cov_eigen);
  }
  void SetGroupParameterCovariance(py::array_t<double>& group_param_cov) {
    int nrow = group_param_cov.shape(0);
    int ncol = group_param_cov.shape(1);
    Eigen::MatrixXd group_param_cov_eigen(nrow, ncol);
    for (int i = 0; i < nrow; i++) {
      for (int j = 0; j < ncol; j++) {
        group_param_cov_eigen(i, j) = group_param_cov.at(i, j);
      }
    }
    rfx_model_->SetGroupParameterCovariance(group_param_cov_eigen);
  }
  void SetVariancePriorShape(double shape) {
    rfx_model_->SetVariancePriorShape(shape);
  }
  void SetVariancePriorScale(double scale) {
    rfx_model_->SetVariancePriorScale(scale);
  }
  void Reset(RandomEffectsContainerCpp& rfx_container, int sample_num) {
    rfx_model_->ResetFromSample(*rfx_container.GetRandomEffectsContainer(), sample_num);
  }

 private:
  std::unique_ptr<StochTree::MultivariateRegressionRandomEffectsModel> rfx_model_;
};

class JsonCpp {
 public:
  JsonCpp() {
    // Initialize pointer to C++ nlohmann::json class
    json_ = std::make_unique<nlohmann::json>();
    nlohmann::json forests = nlohmann::json::object();
    json_->emplace("forests", forests);
    json_->emplace("num_forests", 0);
    nlohmann::json rfx = nlohmann::json::object();
    json_->emplace("random_effects", rfx);
    json_->emplace("num_random_effects", 0);
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

  void LoadFromString(std::string& json_string) {
    *json_ = nlohmann::json::parse(json_string);
  }

  std::string DumpJson() {
    return json_->dump();
  }

  std::string AddForest(ForestContainerCpp& forest_samples, std::string forest_label = "") {
    int forest_num = json_->at("num_forests");
    if (forest_label.empty()) {
      forest_label = "forest_" + std::to_string(forest_num);
    }
    // Reject a duplicate label before any mutation/serialization: nlohmann's emplace would silently
    // no-op on an existing key while num_forests still incremented, dropping the forest and desyncing
    // the count from the actual key set.
    if (json_->at("forests").contains(forest_label)) {
      StochTree::Log::Fatal("A forest with label '%s' already exists; forest labels must be unique", forest_label.c_str());
    }
    json_->at("forests")[forest_label] = forest_samples.ToJson();
    json_->at("num_forests") = forest_num + 1;
    return forest_label;
  }

  std::string AddRandomEffectsContainer(RandomEffectsContainerCpp& rfx_samples) {
    int rfx_num = json_->at("num_random_effects");
    std::string rfx_label = "random_effect_container_" + std::to_string(rfx_num);
    nlohmann::json rfx_json = rfx_samples.GetRandomEffectsContainer()->to_json();
    json_->at("random_effects").emplace(rfx_label, rfx_json);
    return rfx_label;
  }

  std::string AddRandomEffectsLabelMapper(RandomEffectsLabelMapperCpp& rfx_label_mapper) {
    int rfx_num = json_->at("num_random_effects");
    std::string rfx_label = "random_effect_label_mapper_" + std::to_string(rfx_num);
    nlohmann::json rfx_json = rfx_label_mapper.GetLabelMapper()->to_json();
    json_->at("random_effects").emplace(rfx_label, rfx_json);
    return rfx_label;
  }

  std::string AddRandomEffectsGroupIDs(py::array_t<int> rfx_group_ids) {
    int rfx_num = json_->at("num_random_effects");
    std::string rfx_label = "random_effect_groupids_" + std::to_string(rfx_num);
    nlohmann::json groupids_json = nlohmann::json::array();
    for (int i = 0; i < rfx_group_ids.size(); i++) {
      groupids_json.emplace_back(rfx_group_ids.at(i));
    }
    json_->at("random_effects").emplace(rfx_label, groupids_json);
    return rfx_label;
  }

  void IncrementRandomEffectsCount() {
    int rfx_num = json_->at("num_random_effects");
    json_->at("num_random_effects") = rfx_num + 1;
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

  void AddInteger(std::string field_name, int field_value) {
    if (json_->contains(field_name)) {
      json_->at(field_name) = field_value;
    } else {
      json_->emplace(std::pair(field_name, field_value));
    }
  }

  void AddIntegerSubfolder(std::string subfolder_name, std::string field_name, int field_value) {
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

  void AddIntegerVector(std::string field_name, py::array_t<int> field_vector) {
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

  void AddIntegerVectorSubfolder(std::string subfolder_name, std::string field_name, py::array_t<int> field_vector) {
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

  void EraseField(std::string field_name) {
    if (json_->contains(field_name)) {
      json_->erase(field_name);
    }
  }

  void EraseFieldSubfolder(std::string subfolder_name, std::string field_name) {
    if (json_->contains(subfolder_name) && json_->at(subfolder_name).contains(field_name)) {
      json_->at(subfolder_name).erase(field_name);
    }
  }

  void RenameField(std::string old_name, std::string new_name) {
    if (json_->contains(old_name)) {
      (*json_)[new_name] = json_->at(old_name);
      json_->erase(old_name);
    }
  }

  void RenameFieldSubfolder(std::string subfolder_name, std::string old_name, std::string new_name) {
    if (json_->contains(subfolder_name) && json_->at(subfolder_name).contains(old_name)) {
      json_->at(subfolder_name)[new_name] = json_->at(subfolder_name).at(old_name);
      json_->at(subfolder_name).erase(old_name);
    }
  }

  double ExtractDouble(std::string field_name) {
    return json_->at(field_name);
  }

  double ExtractDoubleSubfolder(std::string subfolder_name, std::string field_name) {
    return json_->at(subfolder_name).at(field_name);
  }

  int ExtractInteger(std::string field_name) {
    return json_->at(field_name);
  }

  int ExtractIntegerSubfolder(std::string subfolder_name, std::string field_name) {
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
    py::ssize_t json_vec_length = json_->at(field_name).size();
    auto result = py::array_t<double>(py::detail::any_container<py::ssize_t>({json_vec_length}));
    auto accessor = result.mutable_unchecked<1>();
    for (size_t i = 0; i < json_vec_length; i++) {
      accessor(i) = json_vec.at(i);
    }
    return result;
  }

  py::array_t<double> ExtractDoubleVectorSubfolder(std::string subfolder_name, std::string field_name) {
    auto json_vec = json_->at(subfolder_name).at(field_name);
    py::ssize_t json_vec_length = json_->at(subfolder_name).at(field_name).size();
    auto result = py::array_t<double>(py::detail::any_container<py::ssize_t>({json_vec_length}));
    auto accessor = result.mutable_unchecked<1>();
    for (size_t i = 0; i < json_vec_length; i++) {
      accessor(i) = json_vec.at(i);
    }
    return result;
  }

  py::array_t<int> ExtractIntegerVector(std::string field_name) {
    auto json_vec = json_->at(field_name);
    py::ssize_t json_vec_length = json_->at(field_name).size();
    auto result = py::array_t<int>(py::detail::any_container<py::ssize_t>({json_vec_length}));
    auto accessor = result.mutable_unchecked<1>();
    for (size_t i = 0; i < json_vec_length; i++) {
      accessor(i) = json_vec.at(i);
    }
    return result;
  }

  py::array_t<int> ExtractIntegerVectorSubfolder(std::string subfolder_name, std::string field_name) {
    auto json_vec = json_->at(subfolder_name).at(field_name);
    py::ssize_t json_vec_length = json_->at(subfolder_name).at(field_name).size();
    auto result = py::array_t<int>(py::detail::any_container<py::ssize_t>({json_vec_length}));
    auto accessor = result.mutable_unchecked<1>();
    for (size_t i = 0; i < json_vec_length; i++) {
      accessor(i) = json_vec.at(i);
    }
    return result;
  }

  std::vector<std::string> ExtractStringVector(std::string field_name) {
    auto json_vec = json_->at(field_name);
    py::ssize_t json_vec_length = json_->at(field_name).size();
    auto result = std::vector<std::string>(json_vec_length);
    for (size_t i = 0; i < json_vec_length; i++) {
      result.at(i) = json_vec.at(i);
    }
    return result;
  }

  std::vector<std::string> ExtractStringVectorSubfolder(std::string subfolder_name, std::string field_name) {
    auto json_vec = json_->at(subfolder_name).at(field_name);
    py::ssize_t json_vec_length = json_->at(subfolder_name).at(field_name).size();
    auto result = std::vector<std::string>(json_vec_length);
    for (size_t i = 0; i < json_vec_length; i++) {
      result.at(i) = json_vec.at(i);
    }
    return result;
  }

  nlohmann::json SubsetJsonForest(std::string forest_label) {
    return json_->at("forests").at(forest_label);
  }

  nlohmann::json SubsetJsonRFX() {
    return json_->at("random_effects");
  }

  // Direct access to the underlying nlohmann object so the samples wrappers can read/write the
  // samples-owned subtree in place (BARTSamplesCpp::LoadFromJson / AddToJson).
  nlohmann::json& GetJson() { return *json_; }

 private:
  std::unique_ptr<nlohmann::json> json_;
};

// Shared helpers for the single-owner samples wrappers (BART + BCF): marshal a C++ vector to a numpy
// array, and materialize a standalone deep copy of a forest container (via to_json/from_json) wrapped
// in ForestContainerCpp for the deprecated direct forest accessor (returns nullptr -> None if absent).
inline py::array_t<double> samples_vec_to_numpy(const std::vector<double>& v) {
  py::array_t<double> arr(static_cast<py::ssize_t>(v.size()));
  std::copy(v.begin(), v.end(), arr.mutable_data());
  return arr;
}

inline std::vector<double> numpy_to_samples_vec(py::object obj) {
  if (obj.is_none()) return {};
  auto arr = obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
  return std::vector<double>(arr.data(), arr.data() + arr.size());
}

// Deep-copy a ForestContainer sample-by-sample (matching dims), without a JSON round-trip.
inline std::unique_ptr<StochTree::ForestContainer> copy_forest_container(StochTree::ForestContainer* src) {
  auto copy = std::make_unique<StochTree::ForestContainer>(
      src->NumTrees(), src->OutputDimension(), src->IsLeafConstant(), src->IsExponentiated());
  for (int i = 0; i < src->NumSamples(); i++) copy->AddSample(*src->GetEnsemble(i));
  return copy;
}

inline std::unique_ptr<ForestContainerCpp> materialize_forest_container(
    const std::unique_ptr<StochTree::ForestContainer>& src) {
  if (src == nullptr) return nullptr;
  auto copy = copy_forest_container(src.get());
  // Read dims before moving the unique_ptr (argument evaluation order is unspecified).
  int num_trees = copy->NumTrees();
  int output_dim = copy->OutputDimension();
  bool leaf_constant = copy->IsLeafConstant();
  bool exponentiated = copy->IsExponentiated();
  return std::make_unique<ForestContainerCpp>(std::move(copy), num_trees, output_dim, leaf_constant, exponentiated);
}

// Thin owning wrapper around a single StochTree::BARTSamples, held as ONE external pointer in the
// Python model (single-owner design). It owns the unique_ptr and forwards to the core methods,
// doing only Python type marshalling: parameter traces -> numpy, and a materialize-on-demand deep
// copy of a forest container wrapped in ForestContainerCpp for the (deprecated) direct forest
// accessor. Serialization/merge logic lives in core BARTSamples; this class adds no model logic.
class BARTSamplesCpp {
 public:
  BARTSamplesCpp() { samples_ = std::make_unique<StochTree::BARTSamples>(); }
  BARTSamplesCpp(std::unique_ptr<StochTree::BARTSamples> samples) { samples_ = std::move(samples); }
  ~BARTSamplesCpp() {}

  StochTree::BARTSamples* GetPtr() { return samples_.get(); }

  // Populate the samples-owned subtree from a parsed JSON object held by a JsonCpp (envelope already
  // resolved/migrated by the per-language caller). Inverse: AddToJson merges the subtree back in.
  void LoadFromJson(JsonCpp& json) { samples_->FromJson(json.GetJson()); }
  void AddToJson(JsonCpp& json) { samples_->AppendToJson(json.GetJson()); }

  // Convenience string forms (used for isolated testing; the real save/load path uses JsonCpp).
  static std::unique_ptr<BARTSamplesCpp> FromJsonString(std::string json_string) {
    auto wrapper = std::make_unique<BARTSamplesCpp>();
    nlohmann::json obj = nlohmann::json::parse(json_string);
    wrapper->samples_->FromJson(obj);
    return wrapper;
  }
  std::string ToJsonString() { nlohmann::json obj; samples_->AppendToJson(obj); return obj.dump(); }

  // Append another chain's draws (multi-chain combine); forwards to core BARTSamples::Merge.
  void Merge(BARTSamplesCpp& other) { samples_->Merge(*other.samples_); }

  int NumSamples() { return samples_->num_samples; }
  double YBar() { return samples_->y_bar; }
  double YStd() { return samples_->y_std; }
  int NumTrain() { return samples_->num_train; }
  int NumTest() { return samples_->num_test; }

  py::array_t<double> GlobalVarSamples() { return samples_vec_to_numpy(samples_->global_error_variance_samples); }
  py::array_t<double> LeafScaleSamples() { return samples_vec_to_numpy(samples_->leaf_scale_samples); }
  py::array_t<double> CloglogCutpointSamples() { return samples_vec_to_numpy(samples_->cloglog_cutpoint_samples); }

  // Cached posterior prediction traces (flat, column-major n x num_samples). Empty when absent.
  py::array_t<double> MeanForestPredictionsTrain() { return samples_vec_to_numpy(samples_->mean_forest_predictions_train); }
  py::array_t<double> MeanForestPredictionsTest() { return samples_vec_to_numpy(samples_->mean_forest_predictions_test); }
  py::array_t<double> VarianceForestPredictionsTrain() { return samples_vec_to_numpy(samples_->variance_forest_predictions_train); }
  py::array_t<double> VarianceForestPredictionsTest() { return samples_vec_to_numpy(samples_->variance_forest_predictions_test); }
  py::array_t<double> RfxPredictionsTrain() { return samples_vec_to_numpy(samples_->rfx_predictions_train); }
  py::array_t<double> RfxPredictionsTest() { return samples_vec_to_numpy(samples_->rfx_predictions_test); }
  // Outcome (y_hat) = mean forest + rfx contributions, assembled by the core helper.
  py::array_t<double> YHatTrain() { return samples_vec_to_numpy(samples_->OutcomePredictionsTrain()); }
  py::array_t<double> YHatTest() { return samples_vec_to_numpy(samples_->OutcomePredictionsTest()); }

  bool HasMeanForest() { return samples_->mean_forests != nullptr; }
  bool HasVarianceForest() { return samples_->variance_forests != nullptr; }
  bool HasRfx() { return samples_->rfx_container != nullptr; }
  bool HasGlobalVarSamples() { return !samples_->global_error_variance_samples.empty(); }
  bool HasLeafScaleSamples() { return !samples_->leaf_scale_samples.empty(); }
  bool HasCloglogCutpointSamples() { return !samples_->cloglog_cutpoint_samples.empty(); }
  bool HasMeanForestPredictionsTrain() { return !samples_->mean_forest_predictions_train.empty(); }
  bool HasVarianceForestPredictionsTrain() { return !samples_->variance_forest_predictions_train.empty(); }
  bool HasVarianceForestPredictionsTest() { return !samples_->variance_forest_predictions_test.empty(); }

  // Materialize a standalone deep copy of a forest container, wrapped in ForestContainerCpp, for the
  // (deprecated) direct forest accessor. Returns None when the forest is absent.
  std::unique_ptr<ForestContainerCpp> MaterializeMeanForest() { return materialize_forest_container(samples_->mean_forests); }
  std::unique_ptr<ForestContainerCpp> MaterializeVarianceForest() { return materialize_forest_container(samples_->variance_forests); }

  // Standalone deep copies of the random effects container / label mapper (JSON round-trip), for the
  // rfx extraction path. Return nullptr (-> None) when absent. Mirrors R materialize_rfx.
  std::unique_ptr<RandomEffectsContainerCpp> MaterializeRfxContainer() {
    if (samples_->rfx_container == nullptr) return nullptr;
    auto c = std::make_unique<StochTree::RandomEffectsContainer>();
    c->from_json(samples_->rfx_container->to_json());
    return std::make_unique<RandomEffectsContainerCpp>(std::move(c));
  }
  std::unique_ptr<RandomEffectsLabelMapperCpp> MaterializeRfxLabelMapper() {
    if (samples_->rfx_label_mapper == nullptr) return nullptr;
    auto m = std::make_unique<StochTree::LabelMapper>();
    m->from_json(samples_->rfx_label_mapper->to_json());
    return std::make_unique<RandomEffectsLabelMapperCpp>(std::move(m));
  }

 private:
  std::unique_ptr<StochTree::BARTSamples> samples_;
};

// Thin owning wrapper around a single StochTree::BCFSamples (BCF mirror of BARTSamplesCpp). Owns the
// unique_ptr and forwards to core; adds only Python marshalling. BCF carries three forests
// (prognostic/treatment/variance), extra parameter traces (leaf_scale_mu/tau, tau_0, b0/b1), and a
// treatment_dim scalar.
class BCFSamplesCpp {
 public:
  BCFSamplesCpp() { samples_ = std::make_unique<StochTree::BCFSamples>(); }
  BCFSamplesCpp(std::unique_ptr<StochTree::BCFSamples> samples) { samples_ = std::move(samples); }
  ~BCFSamplesCpp() {}

  StochTree::BCFSamples* GetPtr() { return samples_.get(); }

  void LoadFromJson(JsonCpp& json) { samples_->FromJson(json.GetJson()); }
  void AddToJson(JsonCpp& json) { samples_->AppendToJson(json.GetJson()); }
  static std::unique_ptr<BCFSamplesCpp> FromJsonString(std::string json_string) {
    auto wrapper = std::make_unique<BCFSamplesCpp>();
    nlohmann::json obj = nlohmann::json::parse(json_string);
    wrapper->samples_->FromJson(obj);
    return wrapper;
  }
  std::string ToJsonString() { nlohmann::json obj; samples_->AppendToJson(obj); return obj.dump(); }

  void Merge(BCFSamplesCpp& other) { samples_->Merge(*other.samples_); }

  int NumSamples() { return samples_->num_samples; }
  double YBar() { return samples_->y_bar; }
  double YStd() { return samples_->y_std; }
  int NumTrain() { return samples_->num_train; }
  int NumTest() { return samples_->num_test; }
  int TreatmentDim() { return samples_->treatment_dim; }

  py::array_t<double> GlobalVarSamples() { return samples_vec_to_numpy(samples_->global_error_variance_samples); }
  py::array_t<double> LeafScaleMuSamples() { return samples_vec_to_numpy(samples_->leaf_scale_mu_samples); }
  py::array_t<double> LeafScaleTauSamples() { return samples_vec_to_numpy(samples_->leaf_scale_tau_samples); }
  py::array_t<double> Tau0Samples() { return samples_vec_to_numpy(samples_->tau_0_samples); }
  py::array_t<double> B0Samples() { return samples_vec_to_numpy(samples_->b0_samples); }
  py::array_t<double> B1Samples() { return samples_vec_to_numpy(samples_->b1_samples); }

  // Cached posterior prediction traces (flat, column-major). y_hat is stored directly by the BCF
  // sampler (already on the original outcome scale); tau/mu/variance/rfx are the individual terms.
  py::array_t<double> YHatTrain() { return samples_vec_to_numpy(samples_->y_hat_train); }
  py::array_t<double> YHatTest() { return samples_vec_to_numpy(samples_->y_hat_test); }
  py::array_t<double> MuForestPredictionsTrain() { return samples_vec_to_numpy(samples_->mu_forest_predictions_train); }
  py::array_t<double> MuForestPredictionsTest() { return samples_vec_to_numpy(samples_->mu_forest_predictions_test); }
  py::array_t<double> TauForestPredictionsTrain() { return samples_vec_to_numpy(samples_->tau_forest_predictions_train); }
  py::array_t<double> TauForestPredictionsTest() { return samples_vec_to_numpy(samples_->tau_forest_predictions_test); }
  py::array_t<double> VarianceForestPredictionsTrain() { return samples_vec_to_numpy(samples_->variance_forest_predictions_train); }
  py::array_t<double> VarianceForestPredictionsTest() { return samples_vec_to_numpy(samples_->variance_forest_predictions_test); }
  py::array_t<double> RfxPredictionsTrain() { return samples_vec_to_numpy(samples_->rfx_predictions_train); }
  py::array_t<double> RfxPredictionsTest() { return samples_vec_to_numpy(samples_->rfx_predictions_test); }

  bool HasMuForest() { return samples_->mu_forests != nullptr; }
  bool HasTauForest() { return samples_->tau_forests != nullptr; }
  bool HasVarianceForest() { return samples_->variance_forests != nullptr; }
  bool HasRfx() { return samples_->rfx_container != nullptr; }
  bool HasGlobalVarSamples() { return !samples_->global_error_variance_samples.empty(); }
  bool HasLeafScaleMuSamples() { return !samples_->leaf_scale_mu_samples.empty(); }
  bool HasLeafScaleTauSamples() { return !samples_->leaf_scale_tau_samples.empty(); }
  bool HasVarianceForestPredictionsTrain() { return !samples_->variance_forest_predictions_train.empty(); }
  bool HasVarianceForestPredictionsTest() { return !samples_->variance_forest_predictions_test.empty(); }
  bool HasYhatTest() { return !samples_->y_hat_test.empty(); }

  std::unique_ptr<ForestContainerCpp> MaterializeMuForest() { return materialize_forest_container(samples_->mu_forests); }
  std::unique_ptr<ForestContainerCpp> MaterializeTauForest() { return materialize_forest_container(samples_->tau_forests); }
  std::unique_ptr<ForestContainerCpp> MaterializeVarianceForest() { return materialize_forest_container(samples_->variance_forests); }

  // Standalone deep copies of the random effects container / label mapper (JSON round-trip).
  std::unique_ptr<RandomEffectsContainerCpp> MaterializeRfxContainer() {
    if (samples_->rfx_container == nullptr) return nullptr;
    auto c = std::make_unique<StochTree::RandomEffectsContainer>();
    c->from_json(samples_->rfx_container->to_json());
    return std::make_unique<RandomEffectsContainerCpp>(std::move(c));
  }
  std::unique_ptr<RandomEffectsLabelMapperCpp> MaterializeRfxLabelMapper() {
    if (samples_->rfx_label_mapper == nullptr) return nullptr;
    auto m = std::make_unique<StochTree::LabelMapper>();
    m->from_json(samples_->rfx_label_mapper->to_json());
    return std::make_unique<RandomEffectsLabelMapperCpp>(std::move(m));
  }

 private:
  std::unique_ptr<StochTree::BCFSamples> samples_;
};

template <typename T>
T get_config_scalar_default(py::dict& config_dict, const char* config_key, T default_value) {
  return config_dict.contains(config_key) ? config_dict[config_key].cast<T>() : default_value;
}

inline StochTree::BARTConfig convert_dict_to_bart_config(py::dict config_dict) {
  StochTree::BARTConfig output;

  // Global model parameters
  output.standardize_outcome = get_config_scalar_default<bool>(config_dict, "standardize_outcome", true);
  output.num_threads = get_config_scalar_default<int>(config_dict, "num_threads", 1);
  output.verbose = get_config_scalar_default<bool>(config_dict, "verbose", false);
  output.cutpoint_grid_size = get_config_scalar_default<int>(config_dict, "cutpoint_grid_size", 100);
  output.link_function = static_cast<StochTree::LinkFunction>(get_config_scalar_default<int>(config_dict, "link_function", 0));
  output.outcome_type = static_cast<StochTree::OutcomeType>(get_config_scalar_default<int>(config_dict, "outcome_type", 0));
  output.random_seed = get_config_scalar_default<int>(config_dict, "random_seed", 1);
  output.keep_gfr = get_config_scalar_default<bool>(config_dict, "keep_gfr", 1);
  output.keep_burnin = get_config_scalar_default<bool>(config_dict, "keep_burnin", 1);

  // Global error variance parameters
  output.a_sigma2_global = get_config_scalar_default<double>(config_dict, "a_sigma2_global", 0.0);
  output.b_sigma2_global = get_config_scalar_default<double>(config_dict, "b_sigma2_global", 0.0);
  output.sigma2_global_init = get_config_scalar_default<double>(config_dict, "sigma2_global_init", 1.0);
  output.sample_sigma2_global = get_config_scalar_default<bool>(config_dict, "sample_sigma2_global", true);

  // Mean forest parameters
  output.num_trees_mean = get_config_scalar_default<int>(config_dict, "num_trees_mean", 200);
  output.alpha_mean = get_config_scalar_default<double>(config_dict, "alpha_mean", 0.95);
  output.beta_mean = get_config_scalar_default<double>(config_dict, "beta_mean", 2.0);
  output.min_samples_leaf_mean = get_config_scalar_default<int>(config_dict, "min_samples_leaf_mean", 5);
  output.max_depth_mean = get_config_scalar_default<int>(config_dict, "max_depth_mean", -1);
  output.leaf_constant_mean = get_config_scalar_default<bool>(config_dict, "leaf_constant_mean", true);
  output.leaf_dim_mean = get_config_scalar_default<int>(config_dict, "leaf_dim_mean", 1);
  output.exponentiated_leaf_mean = get_config_scalar_default<bool>(config_dict, "exponentiated_leaf_mean", true);
  output.num_features_subsample_mean = get_config_scalar_default<int>(config_dict, "num_features_subsample_mean", 0);
  output.a_sigma2_mean = get_config_scalar_default<double>(config_dict, "a_sigma2_mean", 3.0);
  output.b_sigma2_mean = get_config_scalar_default<double>(config_dict, "b_sigma2_mean", -1.0);
  output.sigma2_mean_init = get_config_scalar_default<double>(config_dict, "sigma2_mean_init", -1.0);
  output.sample_sigma2_leaf_mean = get_config_scalar_default<bool>(config_dict, "sample_sigma2_leaf_mean", false);
  output.mean_leaf_model_type = static_cast<StochTree::MeanLeafModelType>(get_config_scalar_default<int>(config_dict, "mean_leaf_model_type", 0));
  output.num_classes_cloglog = get_config_scalar_default<int>(config_dict, "num_classes_cloglog", 0);
  output.cloglog_leaf_prior_shape = get_config_scalar_default<double>(config_dict, "cloglog_leaf_prior_shape", 2.0);
  output.cloglog_leaf_prior_scale = get_config_scalar_default<double>(config_dict, "cloglog_leaf_prior_scale", 2.0);
  output.cloglog_cutpoint_0 = get_config_scalar_default<double>(config_dict, "cloglog_cutpoint_0", 0.0);

  // Variance forest parameters
  output.num_trees_variance = get_config_scalar_default<int>(config_dict, "num_trees_variance", 0);
  output.leaf_prior_calibration_param = get_config_scalar_default<double>(config_dict, "leaf_prior_calibration_param", 1.5);
  output.shape_variance_forest = get_config_scalar_default<double>(config_dict, "shape_variance_forest", -1.0);
  output.scale_variance_forest = get_config_scalar_default<double>(config_dict, "scale_variance_forest", -1.0);
  output.variance_forest_leaf_init = get_config_scalar_default<double>(config_dict, "variance_forest_leaf_init", -1.0);
  output.alpha_variance = get_config_scalar_default<double>(config_dict, "alpha_variance", 0.5);
  output.beta_variance = get_config_scalar_default<double>(config_dict, "beta_variance", 2.0);
  output.min_samples_leaf_variance = get_config_scalar_default<int>(config_dict, "min_samples_leaf_variance", 5);
  output.max_depth_variance = get_config_scalar_default<int>(config_dict, "max_depth_variance", -1);
  output.leaf_constant_variance = get_config_scalar_default<bool>(config_dict, "leaf_constant_variance", true);
  output.leaf_dim_variance = get_config_scalar_default<int>(config_dict, "leaf_dim_variance", 1);
  output.exponentiated_leaf_variance = get_config_scalar_default<bool>(config_dict, "exponentiated_leaf_variance", true);
  output.num_features_subsample_variance = get_config_scalar_default<int>(config_dict, "num_features_subsample_variance", 0);

  // Random effects parameters
  output.has_random_effects = get_config_scalar_default<bool>(config_dict, "has_random_effects", false);
  output.rfx_model_spec = static_cast<StochTree::BARTRFXModelSpec>(get_config_scalar_default<int>(config_dict, "rfx_model_spec", 0));
  output.rfx_variance_prior_shape = get_config_scalar_default<double>(config_dict, "rfx_variance_prior_shape", 1.0);
  output.rfx_variance_prior_scale = get_config_scalar_default<double>(config_dict, "rfx_variance_prior_scale", 1.0);

  // Handle vector conversions separately
  if (config_dict.contains("feature_types")) {
    std::vector<int> feature_types_vector = config_dict["feature_types"].cast<std::vector<int>>();
    for (auto item : feature_types_vector) {
      output.feature_types.push_back(static_cast<StochTree::FeatureType>(item));
    }
  }
  if (config_dict.contains("sweep_update_indices_mean")) {
    output.sweep_update_indices_mean = config_dict["sweep_update_indices_mean"].cast<std::vector<int>>();
  }
  if (config_dict.contains("sweep_update_indices_variance")) {
    output.sweep_update_indices_variance = config_dict["sweep_update_indices_variance"].cast<std::vector<int>>();
  }
  if (config_dict.contains("var_weights_mean")) {
    output.var_weights_mean = config_dict["var_weights_mean"].cast<std::vector<double>>();
  }
  if (config_dict.contains("sigma2_leaf_mean_matrix")) {
    output.sigma2_leaf_mean_matrix = config_dict["sigma2_leaf_mean_matrix"].cast<std::vector<double>>();
  }
  if (config_dict.contains("var_weights_variance")) {
    output.var_weights_variance = config_dict["var_weights_variance"].cast<std::vector<double>>();
  }
  if (config_dict.contains("rfx_working_parameter_mean_prior")) {
    py::array_t<double, py::array::f_style | py::array::forcecast> arr =
        config_dict["rfx_working_parameter_mean_prior"].cast<py::array_t<double, py::array::f_style | py::array::forcecast>>();
    output.rfx_working_parameter_mean_prior = std::vector<double>(
        arr.data(), arr.data() + arr.size());
  }
  if (config_dict.contains("rfx_group_parameter_mean_prior")) {
    py::array_t<double, py::array::f_style | py::array::forcecast> arr =
        config_dict["rfx_group_parameter_mean_prior"].cast<py::array_t<double, py::array::f_style | py::array::forcecast>>();
    output.rfx_group_parameter_mean_prior = std::vector<double>(
        arr.data(), arr.data() + arr.size());
  }
  if (config_dict.contains("rfx_working_parameter_cov_prior")) {
    py::array_t<double, py::array::f_style | py::array::forcecast> arr =
        config_dict["rfx_working_parameter_cov_prior"].cast<py::array_t<double, py::array::f_style | py::array::forcecast>>();
    output.rfx_working_parameter_cov_prior = std::vector<double>(
        arr.data(), arr.data() + arr.size());
  }
  if (config_dict.contains("rfx_group_parameter_cov_prior")) {
    py::array_t<double, py::array::f_style | py::array::forcecast> arr =
        config_dict["rfx_group_parameter_cov_prior"].cast<py::array_t<double, py::array::f_style | py::array::forcecast>>();
    output.rfx_group_parameter_cov_prior = std::vector<double>(
        arr.data(), arr.data() + arr.size());
  }
  return output;
}

inline StochTree::BARTData convert_numpy_to_bart_data(
    py::object X_train,
    py::object y_train,
    py::object X_test,
    int n_train,
    int n_test,
    int p,
    py::object basis_train,
    py::object basis_test,
    int basis_dim,
    py::object obs_weights_train,
    py::object obs_weights_test,
    py::object rfx_group_ids_train,
    py::object rfx_group_ids_test,
    py::object rfx_basis_train,
    py::object rfx_basis_test,
    int rfx_num_groups,
    int rfx_basis_dim) {
  StochTree::BARTData output;
  if (!X_train.is_none()) {
    py::array_t<double, pybind11::array::f_style | pybind11::array::forcecast> X_train_array = X_train.cast<py::array_t<double, pybind11::array::f_style | pybind11::array::forcecast>>();
    output.X_train = static_cast<double*>(X_train_array.mutable_data());
  }
  if (!y_train.is_none()) {
    py::array_t<double, pybind11::array::f_style | pybind11::array::forcecast> y_train_array = y_train.cast<py::array_t<double, pybind11::array::f_style | pybind11::array::forcecast>>();
    output.y_train = static_cast<double*>(y_train_array.mutable_data());
  }
  if (!X_test.is_none()) {
    py::array_t<double, pybind11::array::f_style | pybind11::array::forcecast> X_test_array = X_test.cast<py::array_t<double, pybind11::array::f_style | pybind11::array::forcecast>>();
    output.X_test = static_cast<double*>(X_test_array.mutable_data());
  }
  if (!basis_train.is_none()) {
    py::array_t<double, pybind11::array::f_style | pybind11::array::forcecast> basis_train_array = basis_train.cast<py::array_t<double, pybind11::array::f_style | pybind11::array::forcecast>>();
    output.basis_train = static_cast<double*>(basis_train_array.mutable_data());
  }
  if (!basis_test.is_none()) {
    py::array_t<double, pybind11::array::f_style | pybind11::array::forcecast> basis_test_array = basis_test.cast<py::array_t<double, pybind11::array::f_style | pybind11::array::forcecast>>();
    output.basis_test = static_cast<double*>(basis_test_array.mutable_data());
  }
  if (!obs_weights_train.is_none()) {
    py::array_t<double, pybind11::array::f_style | pybind11::array::forcecast> obs_weights_train_array = obs_weights_train.cast<py::array_t<double, pybind11::array::f_style | pybind11::array::forcecast>>();
    output.obs_weights_train = static_cast<double*>(obs_weights_train_array.mutable_data());
  }
  if (!obs_weights_test.is_none()) {
    py::array_t<double, pybind11::array::f_style | pybind11::array::forcecast> obs_weights_test_array = obs_weights_test.cast<py::array_t<double, pybind11::array::f_style | pybind11::array::forcecast>>();
    output.obs_weights_test = static_cast<double*>(obs_weights_test_array.mutable_data());
  }
  if (!rfx_group_ids_train.is_none()) {
    py::array_t<int, pybind11::array::f_style | pybind11::array::forcecast> rfx_group_ids_train_array = rfx_group_ids_train.cast<py::array_t<int, pybind11::array::f_style | pybind11::array::forcecast>>();
    output.rfx_group_ids_train = static_cast<int*>(rfx_group_ids_train_array.mutable_data());
  }
  if (!rfx_group_ids_test.is_none()) {
    py::array_t<int, pybind11::array::f_style | pybind11::array::forcecast> rfx_group_ids_test_array = rfx_group_ids_test.cast<py::array_t<int, pybind11::array::f_style | pybind11::array::forcecast>>();
    output.rfx_group_ids_test = static_cast<int*>(rfx_group_ids_test_array.mutable_data());
  }
  if (!rfx_basis_train.is_none()) {
    py::array_t<double, pybind11::array::f_style | pybind11::array::forcecast> rfx_basis_train_array = rfx_basis_train.cast<py::array_t<double, pybind11::array::f_style | pybind11::array::forcecast>>();
    output.rfx_basis_train = static_cast<double*>(rfx_basis_train_array.mutable_data());
  }
  if (!rfx_basis_test.is_none()) {
    py::array_t<double, pybind11::array::f_style | pybind11::array::forcecast> rfx_basis_test_array = rfx_basis_test.cast<py::array_t<double, pybind11::array::f_style | pybind11::array::forcecast>>();
    output.rfx_basis_test = static_cast<double*>(rfx_basis_test_array.mutable_data());
  }
  output.n_train = n_train;
  output.n_test = n_test;
  output.p = p;
  output.basis_dim = basis_dim;
  output.rfx_num_groups = rfx_num_groups;
  output.rfx_basis_dim = rfx_basis_dim;
  return output;
}

inline py::dict convert_bart_results_to_dict(
    StochTree::BARTSamples& results_raw, StochTree::BARTConfig& config) {
  py::dict output;

  // Transfer ownership of mean forest pointers
  if (results_raw.mean_forests != nullptr) {
    output["forest_container_mean"] = py::cast(std::make_unique<ForestContainerCpp>(std::move(results_raw.mean_forests), config.num_trees_mean, config.leaf_dim_mean, config.leaf_constant_mean, config.exponentiated_leaf_mean));
  } else {
    output["forest_container_mean"] = py::none();
  }

  // Transfer ownership of variance forest pointers
  if (results_raw.variance_forests != nullptr) {
    output["forest_container_variance"] = py::cast(std::make_unique<ForestContainerCpp>(std::move(results_raw.variance_forests), config.num_trees_variance, config.leaf_dim_variance, config.leaf_constant_variance, config.exponentiated_leaf_variance));
  } else {
    output["forest_container_variance"] = py::none();
  }

  // Move parameter vector samples

  // Train set mean forest predictions
  if (results_raw.mean_forest_predictions_train.empty()) {
    output["mean_forest_predictions_train"] = py::none();
  } else {
    auto input_vec = results_raw.mean_forest_predictions_train;
    py::array_t<double> array(input_vec.size());
    std::copy(input_vec.begin(), input_vec.end(), array.mutable_data());
    output["mean_forest_predictions_train"] = array;
  }

  // Test set mean forest predictions
  if (results_raw.mean_forest_predictions_test.empty()) {
    output["mean_forest_predictions_test"] = py::none();
  } else {
    auto input_vec = results_raw.mean_forest_predictions_test;
    py::array_t<double> array(input_vec.size());
    std::copy(input_vec.begin(), input_vec.end(), array.mutable_data());
    output["mean_forest_predictions_test"] = array;
  }

  // Train set variance forest predictions
  if (results_raw.variance_forest_predictions_train.empty()) {
    output["variance_forest_predictions_train"] = py::none();
  } else {
    auto input_vec = results_raw.variance_forest_predictions_train;
    py::array_t<double> array(input_vec.size());
    std::copy(input_vec.begin(), input_vec.end(), array.mutable_data());
    output["variance_forest_predictions_train"] = array;
  }

  // Test set variance forest predictions
  if (results_raw.variance_forest_predictions_test.empty()) {
    output["variance_forest_predictions_test"] = py::none();
  } else {
    auto input_vec = results_raw.variance_forest_predictions_test;
    py::array_t<double> array(input_vec.size());
    std::copy(input_vec.begin(), input_vec.end(), array.mutable_data());
    output["variance_forest_predictions_test"] = array;
  }

  // Global error variance samples
  if (results_raw.global_error_variance_samples.empty()) {
    output["global_var_samples"] = py::none();
  } else {
    auto input_vec = results_raw.global_error_variance_samples;
    py::array_t<double> array(input_vec.size());
    std::copy(input_vec.begin(), input_vec.end(), array.mutable_data());
    output["global_var_samples"] = array;
  }

  // Leaf scale samples
  if (results_raw.leaf_scale_samples.empty()) {
    output["leaf_scale_samples"] = py::none();
  } else {
    auto input_vec = results_raw.leaf_scale_samples;
    py::array_t<double> array(input_vec.size());
    std::copy(input_vec.begin(), input_vec.end(), array.mutable_data());
    output["leaf_scale_samples"] = array;
  }

  // Cloglog cutpoint samples
  if (results_raw.cloglog_cutpoint_samples.empty()) {
    output["cloglog_cutpoint_samples"] = py::none();
  } else {
    auto input_vec = results_raw.cloglog_cutpoint_samples;
    py::array_t<double> array(input_vec.size());
    std::copy(input_vec.begin(), input_vec.end(), array.mutable_data());
    output["cloglog_cutpoint_samples"] = array;
  }

  // Unpack RFX predictions
  if (!results_raw.rfx_predictions_train.empty()) {
    auto& v = results_raw.rfx_predictions_train;
    py::array_t<double> array(v.size());
    std::copy(v.begin(), v.end(), array.mutable_data());
    output["rfx_predictions_train"] = array;
  } else {
    output["rfx_predictions_train"] = py::none();
  }
  if (!results_raw.rfx_predictions_test.empty()) {
    auto& v = results_raw.rfx_predictions_test;
    py::array_t<double> array(v.size());
    std::copy(v.begin(), v.end(), array.mutable_data());
    output["rfx_predictions_test"] = array;
  } else {
    output["rfx_predictions_test"] = py::none();
  }

  // Transfer ownership of random effects container pointers
  if (results_raw.rfx_container != nullptr) {
    output["rfx_container"] = py::cast(std::make_unique<RandomEffectsContainerCpp>(std::move(results_raw.rfx_container)));
  } else {
    output["rfx_container"] = py::none();
  }

  // Transfer ownership of random effects label mapper pointer
  if (results_raw.rfx_label_mapper != nullptr) {
    output["rfx_label_mapper"] = py::cast(std::make_unique<RandomEffectsLabelMapperCpp>(std::move(results_raw.rfx_label_mapper)));
  } else {
    output["rfx_label_mapper"] = py::none();
  }

  // Unpack scalars
  output["y_bar"] = results_raw.y_bar;
  output["y_std"] = results_raw.y_std;
  output["num_samples"] = results_raw.num_samples;
  output["num_train"] = results_raw.num_train;
  output["num_test"] = results_raw.num_test;

  return output;
}

void add_config_to_bart_result_dict(py::dict& result, StochTree::BARTConfig& config) {
  // Unpack more metadata about the model that was sampled
  result["sigma2_init"] = config.sigma2_global_init;
  result["sigma2_mean_init"] = config.sigma2_mean_init;
  result["b_sigma2_mean"] = config.b_sigma2_mean;
  result["shape_variance_forest"] = config.shape_variance_forest;
  result["scale_variance_forest"] = config.scale_variance_forest;
}

py::dict bart_sample_cpp(
    BARTSamplesCpp& samples,
    py::object X_train,
    py::object y_train,
    py::object X_test,
    int n_train,
    int n_test,
    int p,
    py::object basis_train,
    py::object basis_test,
    int basis_dim,
    py::object obs_weights_train,
    py::object obs_weights_test,
    py::object rfx_group_ids_train,
    py::object rfx_group_ids_test,
    py::object rfx_basis_train,
    py::object rfx_basis_test,
    int rfx_num_groups,
    int rfx_basis_dim,
    int num_gfr,
    int num_burnin,
    int keep_every,
    int num_mcmc,
    int num_chains,
    py::dict config_input) {
  // Convert config dict to BARTConfig struct
  StochTree::BARTConfig bart_config = convert_dict_to_bart_config(config_input);

  // Unpack pointers to input data to BARTData object
  StochTree::BARTData bart_data = convert_numpy_to_bart_data(X_train, y_train, X_test, n_train, n_test, p, basis_train, basis_test, basis_dim, obs_weights_train, obs_weights_test, rfx_group_ids_train, rfx_group_ids_test, rfx_basis_train, rfx_basis_test, rfx_num_groups, rfx_basis_dim);

  // Single-owner (mirrors R bart_sample_cpp): the C++ sampler populates the caller-owned BARTSamples
  // in place; this function only returns metadata that is NOT part of the samples object.
  StochTree::BARTSamples& bart_samples = *samples.GetPtr();

  // Initialize a BART sampler
  StochTree::BARTSampler bart_sampler(bart_samples, bart_config, bart_data);

  // Run the sampler
  bart_sampler.run_gfr(bart_samples, num_gfr, bart_config.keep_gfr, num_chains);
  if (num_chains > 1) {
    bart_sampler.run_mcmc_chains(bart_samples, num_chains, num_burnin, keep_every, num_mcmc);
  } else {
    bart_sampler.run_mcmc(bart_samples, num_burnin, keep_every, num_mcmc);
  }
  bart_sampler.postprocess_samples(bart_samples);

  // Metadata: config-derived init scalars and the final RNG state (for continue_sampling). Everything
  // else is read off the caller's samples object.
  py::dict metadata;
  add_config_to_bart_result_dict(metadata, bart_config);
  // Persist the final RNG state so continue_sampling() can resume the random stream (statistical
  // equivalence). Only meaningful for single-chain runs; multi-chain uses one RNG per chain.
  metadata["rng_state"] = (num_chains <= 1) ? py::object(py::cast(bart_sampler.GetRngState())) : py::object(py::none());
  return metadata;
}

// Continue (warm-start) sampling from an already-fit BART model.
//
// Reconstruct-on-demand stopgap (RFC 0005 / #408): rather than transferring
// ownership of the model's forest container, this DEEP-COPIES it into a fresh
// BARTSamples so the caller's container remains the source of truth. The scalar
// sample histories are pre-populated in STANDARDIZED space (forward sampling
// keeps them standardized until postprocess_samples), then the sampler is
// constructed in continuation mode (warm-starts the active forest from the last
// retained sample and appends new draws to the copied container).
//
// Returns the same dict shape as bart_sample_cpp, with the forest container and
// parameter arrays extended to (history + new) samples. Predictions in the
// returned dict are recomputed by the Python wrapper post-hoc.
py::dict bart_continue_sample_cpp(
    BARTSamplesCpp& samples,
    py::object X_train,
    py::object y_train,
    py::object X_test,
    int n_train,
    int n_test,
    int p,
    py::object basis_train,
    py::object basis_test,
    int basis_dim,
    py::object obs_weights_train,
    py::object obs_weights_test,
    py::object rfx_group_ids_train,
    py::object rfx_group_ids_test,
    py::object rfx_basis_train,
    py::object rfx_basis_test,
    int rfx_num_groups,
    int rfx_basis_dim,
    int num_gfr,
    int num_burnin,
    int keep_every,
    int num_mcmc,
    bool keep_gfr,
    std::string rng_state_in,
    bool override_seed,
    py::dict config_input) {
  // Convert config dict to BARTConfig struct
  StochTree::BARTConfig bart_config = convert_dict_to_bart_config(config_input);

  // Unpack pointers to (re-supplied) input data to BARTData object
  StochTree::BARTData bart_data = convert_numpy_to_bart_data(X_train, y_train, X_test, n_train, n_test, p, basis_train, basis_test, basis_dim, obs_weights_train, obs_weights_test, rfx_group_ids_train, rfx_group_ids_test, rfx_basis_train, rfx_basis_test, rfx_num_groups, rfx_basis_dim);

  // Continuation appends new MCMC draws in place onto the model's single-owner samples object,
  // mirroring the R sampler's run_mcmc(samples, ...). The warm-start reads the last retained forest +
  // scalar state directly off `samples` (its global variance history is post-processed; the sampler
  // un-scales it internally), and postprocess_samples(samples, num_history) rescales only the newly
  // appended draws, leaving the already-processed history untouched.
  StochTree::BARTSamples& bart_samples = *samples.GetPtr();
  const int num_history = bart_samples.num_samples;

  // Initialize a BART sampler in continuation mode (warm-start from last sample)
  StochTree::BARTSampler bart_sampler(bart_samples, bart_config, bart_data, /*continuation=*/true);

  // Resume the RNG stream from where the prior run left off (statistical-equivalence continuation),
  // unless the user supplied a new seed (override_seed) -- in which case the fresh seed set by
  // InitializeState stands. The warm-start init consumes no RNG draws, so the restored state is
  // positioned exactly at the next draw.
  if (!override_seed && !rng_state_in.empty()) {
    bart_sampler.SetRngState(rng_state_in);
  }

  // Probit warm-start: regenerate the (unpersisted) latent outcome now that the RNG is positioned at
  // the resumed/re-seeded stream, so the first continued draw starts from a valid stationary state.
  bart_sampler.RegenerateProbitLatent(bart_samples);

  // Optionally append GFR (grow-from-root) warm-start draws, then MCMC draws, then post-process only
  // the newly appended range. num_gfr defaults to 0 (MCMC-only append); when > 0, keep_gfr controls
  // whether those draws are retained (TRUE = extend the chain, e.g. 25 -> 40 warm-start samples;
  // FALSE = re-anneal then discard). Single-chain continuation, so num_chains = 1.
  bart_sampler.run_gfr(bart_samples, num_gfr, keep_gfr, /*num_chains=*/1);
  bart_sampler.run_mcmc(bart_samples, num_burnin, keep_every, num_mcmc);
  bart_sampler.postprocess_samples(bart_samples, num_history);

  // Metadata only; the extended samples live on the passed-in samples object.
  py::dict metadata;
  add_config_to_bart_result_dict(metadata, bart_config);
  metadata["rng_state"] = bart_sampler.GetRngState();
  return metadata;
}

inline StochTree::BCFConfig convert_dict_to_bcf_config(py::dict config_dict) {
  StochTree::BCFConfig output;

  // Global model parameters
  output.standardize_outcome = get_config_scalar_default<bool>(config_dict, "standardize_outcome", true);
  output.num_threads = get_config_scalar_default<int>(config_dict, "num_threads", 1);
  output.verbose = get_config_scalar_default<bool>(config_dict, "verbose", false);
  output.cutpoint_grid_size = get_config_scalar_default<int>(config_dict, "cutpoint_grid_size", 100);
  output.link_function = static_cast<StochTree::LinkFunction>(get_config_scalar_default<int>(config_dict, "link_function", 0));
  output.outcome_type = static_cast<StochTree::OutcomeType>(get_config_scalar_default<int>(config_dict, "outcome_type", 0));
  output.random_seed = get_config_scalar_default<int>(config_dict, "random_seed", 1);
  output.keep_gfr = get_config_scalar_default<bool>(config_dict, "keep_gfr", 0);
  output.keep_burnin = get_config_scalar_default<bool>(config_dict, "keep_burnin", 0);
  output.adaptive_coding = get_config_scalar_default<bool>(config_dict, "adaptive_coding", 0);
  output.b_0_init = get_config_scalar_default<double>(config_dict, "b_0_init", 0.0);
  output.b_1_init = get_config_scalar_default<double>(config_dict, "b_1_init", 1.0);

  // Global error variance parameters
  output.a_sigma2_global = get_config_scalar_default<double>(config_dict, "a_sigma2_global", 0.0);
  output.b_sigma2_global = get_config_scalar_default<double>(config_dict, "b_sigma2_global", 0.0);
  output.sigma2_global_init = get_config_scalar_default<double>(config_dict, "sigma2_global_init", 1.0);
  output.sample_sigma2_global = get_config_scalar_default<bool>(config_dict, "sample_sigma2_global", true);

  // Mu forest parameters
  output.num_trees_mu = get_config_scalar_default<int>(config_dict, "num_trees_mu", 200);
  output.alpha_mu = get_config_scalar_default<double>(config_dict, "alpha_mu", 0.95);
  output.beta_mu = get_config_scalar_default<double>(config_dict, "beta_mu", 2.0);
  output.min_samples_leaf_mu = get_config_scalar_default<int>(config_dict, "min_samples_leaf_mu", 5);
  output.max_depth_mu = get_config_scalar_default<int>(config_dict, "max_depth_mu", -1);
  output.leaf_constant_mu = get_config_scalar_default<bool>(config_dict, "leaf_constant_mu", true);
  output.leaf_dim_mu = get_config_scalar_default<int>(config_dict, "leaf_dim_mu", 1);
  output.exponentiated_leaf_mu = get_config_scalar_default<bool>(config_dict, "exponentiated_leaf_mu", false);
  output.num_features_subsample_mu = get_config_scalar_default<int>(config_dict, "num_features_subsample_mu", 0);
  output.a_sigma2_mu = get_config_scalar_default<double>(config_dict, "a_sigma2_mu", 3.0);
  output.b_sigma2_mu = get_config_scalar_default<double>(config_dict, "b_sigma2_mu", -1.0);
  output.sigma2_mu_init = get_config_scalar_default<double>(config_dict, "sigma2_mu_init", -1.0);
  output.sample_sigma2_leaf_mu = get_config_scalar_default<bool>(config_dict, "sample_sigma2_leaf_mu", false);

  // Tau forest parameters
  output.num_trees_tau = get_config_scalar_default<int>(config_dict, "num_trees_tau", 50);
  output.alpha_tau = get_config_scalar_default<double>(config_dict, "alpha_tau", 0.95);
  output.beta_tau = get_config_scalar_default<double>(config_dict, "beta_tau", 2.0);
  output.min_samples_leaf_tau = get_config_scalar_default<int>(config_dict, "min_samples_leaf_tau", 5);
  output.max_depth_tau = get_config_scalar_default<int>(config_dict, "max_depth_tau", -1);
  output.leaf_constant_tau = get_config_scalar_default<bool>(config_dict, "leaf_constant_tau", false);
  output.leaf_dim_tau = get_config_scalar_default<int>(config_dict, "leaf_dim_tau", 1);
  output.exponentiated_leaf_tau = get_config_scalar_default<bool>(config_dict, "exponentiated_leaf_tau", false);
  output.num_features_subsample_tau = get_config_scalar_default<int>(config_dict, "num_features_subsample_tau", 0);
  output.a_sigma2_tau = get_config_scalar_default<double>(config_dict, "a_sigma2_tau", 3.0);
  output.b_sigma2_tau = get_config_scalar_default<double>(config_dict, "b_sigma2_tau", -1.0);
  output.sigma2_tau_init = get_config_scalar_default<double>(config_dict, "sigma2_tau_init", -1.0);
  output.sample_sigma2_leaf_tau = get_config_scalar_default<bool>(config_dict, "sample_sigma2_leaf_tau", false);
  output.tau_leaf_model_type = static_cast<StochTree::MeanLeafModelType>(get_config_scalar_default<int>(config_dict, "tau_leaf_model_type", 0));
  output.sample_tau_0 = get_config_scalar_default<bool>(config_dict, "sample_tau_0", true);
  output.tau_0_prior_var_scalar = get_config_scalar_default<double>(config_dict, "tau_0_prior_var_scalar", -1.0);

  // Variance forest parameters
  output.num_trees_variance = get_config_scalar_default<int>(config_dict, "num_trees_variance", 0);
  output.leaf_prior_calibration_param = get_config_scalar_default<double>(config_dict, "leaf_prior_calibration_param", 1.5);
  output.shape_variance_forest = get_config_scalar_default<double>(config_dict, "shape_variance_forest", -1.0);
  output.scale_variance_forest = get_config_scalar_default<double>(config_dict, "scale_variance_forest", -1.0);
  output.variance_forest_leaf_init = get_config_scalar_default<double>(config_dict, "variance_forest_leaf_init", -1.0);
  output.alpha_variance = get_config_scalar_default<double>(config_dict, "alpha_variance", 0.5);
  output.beta_variance = get_config_scalar_default<double>(config_dict, "beta_variance", 2.0);
  output.min_samples_leaf_variance = get_config_scalar_default<int>(config_dict, "min_samples_leaf_variance", 5);
  output.max_depth_variance = get_config_scalar_default<int>(config_dict, "max_depth_variance", -1);
  output.leaf_constant_variance = get_config_scalar_default<bool>(config_dict, "leaf_constant_variance", true);
  output.leaf_dim_variance = get_config_scalar_default<int>(config_dict, "leaf_dim_variance", 1);
  output.exponentiated_leaf_variance = get_config_scalar_default<bool>(config_dict, "exponentiated_leaf_variance", true);
  output.num_features_subsample_variance = get_config_scalar_default<int>(config_dict, "num_features_subsample_variance", 0);

  // Random effects parameters
  output.has_random_effects = get_config_scalar_default<bool>(config_dict, "has_random_effects", false);
  output.rfx_model_spec = static_cast<StochTree::BCFRFXModelSpec>(get_config_scalar_default<int>(config_dict, "rfx_model_spec", 0));
  output.rfx_variance_prior_shape = get_config_scalar_default<double>(config_dict, "rfx_variance_prior_shape", 1.0);
  output.rfx_variance_prior_scale = get_config_scalar_default<double>(config_dict, "rfx_variance_prior_scale", 1.0);

  // Handle vector conversions separately
  if (config_dict.contains("feature_types")) {
    std::vector<int> feature_types_vector = config_dict["feature_types"].cast<std::vector<int>>();
    for (auto item : feature_types_vector) {
      output.feature_types.push_back(static_cast<StochTree::FeatureType>(item));
    }
  }
  if (config_dict.contains("sweep_update_indices_mu")) {
    output.sweep_update_indices_mu = config_dict["sweep_update_indices_mu"].cast<std::vector<int>>();
  }
  if (config_dict.contains("sweep_update_indices_tau")) {
    output.sweep_update_indices_tau = config_dict["sweep_update_indices_tau"].cast<std::vector<int>>();
  }
  if (config_dict.contains("sweep_update_indices_variance")) {
    output.sweep_update_indices_variance = config_dict["sweep_update_indices_variance"].cast<std::vector<int>>();
  }
  if (config_dict.contains("var_weights_mu")) {
    output.var_weights_mu = config_dict["var_weights_mu"].cast<std::vector<double>>();
  }
  if (config_dict.contains("var_weights_tau")) {
    output.var_weights_tau = config_dict["var_weights_tau"].cast<std::vector<double>>();
  }
  if (config_dict.contains("var_weights_variance")) {
    output.var_weights_variance = config_dict["var_weights_variance"].cast<std::vector<double>>();
  }
  if (config_dict.contains("sigma2_leaf_tau_matrix")) {
    output.sigma2_leaf_tau_matrix = config_dict["sigma2_leaf_tau_matrix"].cast<std::vector<double>>();
  }
  if (config_dict.contains("tau_0_prior_var_multivariate")) {
    output.tau_0_prior_var_multivariate = config_dict["tau_0_prior_var_multivariate"].cast<std::vector<double>>();
  }
  if (config_dict.contains("rfx_working_parameter_mean_prior")) {
    py::array_t<double, py::array::f_style | py::array::forcecast> arr =
        config_dict["rfx_working_parameter_mean_prior"].cast<py::array_t<double, py::array::f_style | py::array::forcecast>>();
    output.rfx_working_parameter_mean_prior = std::vector<double>(
        arr.data(), arr.data() + arr.size());
  }
  if (config_dict.contains("rfx_group_parameter_mean_prior")) {
    py::array_t<double, py::array::f_style | py::array::forcecast> arr =
        config_dict["rfx_group_parameter_mean_prior"].cast<py::array_t<double, py::array::f_style | py::array::forcecast>>();
    output.rfx_group_parameter_mean_prior = std::vector<double>(
        arr.data(), arr.data() + arr.size());
  }
  if (config_dict.contains("rfx_working_parameter_cov_prior")) {
    py::array_t<double, py::array::f_style | py::array::forcecast> arr =
        config_dict["rfx_working_parameter_cov_prior"].cast<py::array_t<double, py::array::f_style | py::array::forcecast>>();
    output.rfx_working_parameter_cov_prior = std::vector<double>(
        arr.data(), arr.data() + arr.size());
  }
  if (config_dict.contains("rfx_group_parameter_cov_prior")) {
    py::array_t<double, py::array::f_style | py::array::forcecast> arr =
        config_dict["rfx_group_parameter_cov_prior"].cast<py::array_t<double, py::array::f_style | py::array::forcecast>>();
    output.rfx_group_parameter_cov_prior = std::vector<double>(
        arr.data(), arr.data() + arr.size());
  }
  return output;
}

inline StochTree::BCFData convert_numpy_to_bcf_data(
    py::object X_train,
    py::object Z_train,
    py::object y_train,
    py::object X_test,
    py::object Z_test,
    int n_train,
    int n_test,
    int p,
    int treatment_dim,
    py::object obs_weights_train,
    py::object obs_weights_test,
    py::object rfx_group_ids_train,
    py::object rfx_group_ids_test,
    py::object rfx_basis_train,
    py::object rfx_basis_test,
    int rfx_num_groups,
    int rfx_basis_dim) {
  StochTree::BCFData output;
  if (!X_train.is_none()) {
    py::array_t<double, pybind11::array::f_style | pybind11::array::forcecast> X_train_array = X_train.cast<py::array_t<double, pybind11::array::f_style | pybind11::array::forcecast>>();
    output.X_train = static_cast<double*>(X_train_array.mutable_data());
  }
  if (!Z_train.is_none()) {
    py::array_t<double, pybind11::array::f_style | pybind11::array::forcecast> Z_train_array = Z_train.cast<py::array_t<double, pybind11::array::f_style | pybind11::array::forcecast>>();
    output.treatment_train = static_cast<double*>(Z_train_array.mutable_data());
  }
  if (!y_train.is_none()) {
    py::array_t<double, pybind11::array::f_style | pybind11::array::forcecast> y_train_array = y_train.cast<py::array_t<double, pybind11::array::f_style | pybind11::array::forcecast>>();
    output.y_train = static_cast<double*>(y_train_array.mutable_data());
  }
  if (!X_test.is_none()) {
    py::array_t<double, pybind11::array::f_style | pybind11::array::forcecast> X_test_array = X_test.cast<py::array_t<double, pybind11::array::f_style | pybind11::array::forcecast>>();
    output.X_test = static_cast<double*>(X_test_array.mutable_data());
  }
  if (!Z_test.is_none()) {
    py::array_t<double, pybind11::array::f_style | pybind11::array::forcecast> Z_test_array = Z_test.cast<py::array_t<double, pybind11::array::f_style | pybind11::array::forcecast>>();
    output.treatment_test = static_cast<double*>(Z_test_array.mutable_data());
  }
  if (!obs_weights_train.is_none()) {
    py::array_t<double, pybind11::array::f_style | pybind11::array::forcecast> obs_weights_train_array = obs_weights_train.cast<py::array_t<double, pybind11::array::f_style | pybind11::array::forcecast>>();
    output.obs_weights_train = static_cast<double*>(obs_weights_train_array.mutable_data());
  }
  if (!obs_weights_test.is_none()) {
    py::array_t<double, pybind11::array::f_style | pybind11::array::forcecast> obs_weights_test_array = obs_weights_test.cast<py::array_t<double, pybind11::array::f_style | pybind11::array::forcecast>>();
    output.obs_weights_test = static_cast<double*>(obs_weights_test_array.mutable_data());
  }
  if (!rfx_group_ids_train.is_none()) {
    py::array_t<int, pybind11::array::f_style | pybind11::array::forcecast> rfx_group_ids_train_array = rfx_group_ids_train.cast<py::array_t<int, pybind11::array::f_style | pybind11::array::forcecast>>();
    output.rfx_group_ids_train = static_cast<int*>(rfx_group_ids_train_array.mutable_data());
  }
  if (!rfx_group_ids_test.is_none()) {
    py::array_t<int, pybind11::array::f_style | pybind11::array::forcecast> rfx_group_ids_test_array = rfx_group_ids_test.cast<py::array_t<int, pybind11::array::f_style | pybind11::array::forcecast>>();
    output.rfx_group_ids_test = static_cast<int*>(rfx_group_ids_test_array.mutable_data());
  }
  if (!rfx_basis_train.is_none()) {
    py::array_t<double, pybind11::array::f_style | pybind11::array::forcecast> rfx_basis_train_array = rfx_basis_train.cast<py::array_t<double, pybind11::array::f_style | pybind11::array::forcecast>>();
    output.rfx_basis_train = static_cast<double*>(rfx_basis_train_array.mutable_data());
  }
  if (!rfx_basis_test.is_none()) {
    py::array_t<double, pybind11::array::f_style | pybind11::array::forcecast> rfx_basis_test_array = rfx_basis_test.cast<py::array_t<double, pybind11::array::f_style | pybind11::array::forcecast>>();
    output.rfx_basis_test = static_cast<double*>(rfx_basis_test_array.mutable_data());
  }
  output.n_train = n_train;
  output.n_test = n_test;
  output.p = p;
  output.treatment_dim = treatment_dim;
  output.rfx_num_groups = rfx_num_groups;
  output.rfx_basis_dim = rfx_basis_dim;
  return output;
}

inline py::dict convert_bcf_results_to_dict(
    StochTree::BCFSamples& results_raw, StochTree::BCFConfig& config) {
  py::dict output;

  // Transfer ownership of mean forest pointers
  if (results_raw.mu_forests != nullptr) {
    output["forest_container_mu"] = py::cast(std::make_unique<ForestContainerCpp>(std::move(results_raw.mu_forests), config.num_trees_mu, config.leaf_dim_mu, config.leaf_constant_mu, config.exponentiated_leaf_mu));
  } else {
    output["forest_container_mu"] = py::none();
  }
  if (results_raw.tau_forests != nullptr) {
    output["forest_container_tau"] = py::cast(std::make_unique<ForestContainerCpp>(std::move(results_raw.tau_forests), config.num_trees_tau, config.leaf_dim_tau, config.leaf_constant_tau, config.exponentiated_leaf_tau));
  } else {
    output["forest_container_tau"] = py::none();
  }

  // Transfer ownership of variance forest pointers
  if (results_raw.variance_forests != nullptr) {
    output["forest_container_variance"] = py::cast(std::make_unique<ForestContainerCpp>(std::move(results_raw.variance_forests), config.num_trees_variance, config.leaf_dim_variance, config.leaf_constant_variance, config.exponentiated_leaf_variance));
  } else {
    output["forest_container_variance"] = py::none();
  }

  // Move parameter vector samples

  // Train set prognostic forest predictions
  if (results_raw.mu_forest_predictions_train.empty()) {
    output["mu_forest_predictions_train"] = py::none();
  } else {
    auto input_vec = results_raw.mu_forest_predictions_train;
    py::array_t<double> array(input_vec.size());
    std::copy(input_vec.begin(), input_vec.end(), array.mutable_data());
    output["mu_forest_predictions_train"] = array;
  }

  // Test set prognostic forest predictions
  if (results_raw.mu_forest_predictions_test.empty()) {
    output["mu_forest_predictions_test"] = py::none();
  } else {
    auto input_vec = results_raw.mu_forest_predictions_test;
    py::array_t<double> array(input_vec.size());
    std::copy(input_vec.begin(), input_vec.end(), array.mutable_data());
    output["mu_forest_predictions_test"] = array;
  }

  // Train set treatment effect forest predictions
  if (results_raw.tau_forest_predictions_train.empty()) {
    output["tau_forest_predictions_train"] = py::none();
  } else {
    auto input_vec = results_raw.tau_forest_predictions_train;
    py::array_t<double> array(input_vec.size());
    std::copy(input_vec.begin(), input_vec.end(), array.mutable_data());
    output["tau_forest_predictions_train"] = array;
  }

  // Test set treatment effect forest predictions
  if (results_raw.tau_forest_predictions_test.empty()) {
    output["tau_forest_predictions_test"] = py::none();
  } else {
    auto input_vec = results_raw.tau_forest_predictions_test;
    py::array_t<double> array(input_vec.size());
    std::copy(input_vec.begin(), input_vec.end(), array.mutable_data());
    output["tau_forest_predictions_test"] = array;
  }

  // Train set outcome predictions
  if (results_raw.y_hat_train.empty()) {
    output["y_hat_train"] = py::none();
  } else {
    auto input_vec = results_raw.y_hat_train;
    py::array_t<double> array(input_vec.size());
    std::copy(input_vec.begin(), input_vec.end(), array.mutable_data());
    output["y_hat_train"] = array;
  }

  // Test set outcome predictions
  if (results_raw.y_hat_test.empty()) {
    output["y_hat_test"] = py::none();
  } else {
    auto input_vec = results_raw.y_hat_test;
    py::array_t<double> array(input_vec.size());
    std::copy(input_vec.begin(), input_vec.end(), array.mutable_data());
    output["y_hat_test"] = array;
  }

  // Train set variance forest predictions
  if (results_raw.variance_forest_predictions_train.empty()) {
    output["variance_forest_predictions_train"] = py::none();
  } else {
    auto input_vec = results_raw.variance_forest_predictions_train;
    py::array_t<double> array(input_vec.size());
    std::copy(input_vec.begin(), input_vec.end(), array.mutable_data());
    output["variance_forest_predictions_train"] = array;
  }

  // Test set variance forest predictions
  if (results_raw.variance_forest_predictions_test.empty()) {
    output["variance_forest_predictions_test"] = py::none();
  } else {
    auto input_vec = results_raw.variance_forest_predictions_test;
    py::array_t<double> array(input_vec.size());
    std::copy(input_vec.begin(), input_vec.end(), array.mutable_data());
    output["variance_forest_predictions_test"] = array;
  }

  // Global error variance samples
  if (results_raw.global_error_variance_samples.empty()) {
    output["global_var_samples"] = py::none();
  } else {
    auto input_vec = results_raw.global_error_variance_samples;
    py::array_t<double> array(input_vec.size());
    std::copy(input_vec.begin(), input_vec.end(), array.mutable_data());
    output["global_var_samples"] = array;
  }

  // Leaf scale samples -- prognostic forest
  if (results_raw.leaf_scale_mu_samples.empty()) {
    output["leaf_scale_mu_samples"] = py::none();
  } else {
    auto input_vec = results_raw.leaf_scale_mu_samples;
    py::array_t<double> array(input_vec.size());
    std::copy(input_vec.begin(), input_vec.end(), array.mutable_data());
    output["leaf_scale_mu_samples"] = array;
  }

  // Leaf scale samples -- treatment effect forest
  if (results_raw.leaf_scale_tau_samples.empty()) {
    output["leaf_scale_tau_samples"] = py::none();
  } else {
    auto input_vec = results_raw.leaf_scale_tau_samples;
    py::array_t<double> array(input_vec.size());
    std::copy(input_vec.begin(), input_vec.end(), array.mutable_data());
    output["leaf_scale_tau_samples"] = array;
  }

  // tau_0 samples
  if (results_raw.tau_0_samples.empty()) {
    output["tau_0_samples"] = py::none();
  } else {
    auto input_vec = results_raw.tau_0_samples;
    py::array_t<double> array(input_vec.size());
    std::copy(input_vec.begin(), input_vec.end(), array.mutable_data());
    output["tau_0_samples"] = array;
  }

  // adaptive coding samples
  // b0
  if (results_raw.b0_samples.empty()) {
    output["b0_samples"] = py::none();
  } else {
    auto input_vec = results_raw.b0_samples;
    py::array_t<double> array(input_vec.size());
    std::copy(input_vec.begin(), input_vec.end(), array.mutable_data());
    output["b0_samples"] = array;
  }
  // b1
  if (results_raw.b1_samples.empty()) {
    output["b1_samples"] = py::none();
  } else {
    auto input_vec = results_raw.b1_samples;
    py::array_t<double> array(input_vec.size());
    std::copy(input_vec.begin(), input_vec.end(), array.mutable_data());
    output["b1_samples"] = array;
  }

  // Unpack RFX predictions
  if (!results_raw.rfx_predictions_train.empty()) {
    auto& v = results_raw.rfx_predictions_train;
    py::array_t<double> array(v.size());
    std::copy(v.begin(), v.end(), array.mutable_data());
    output["rfx_predictions_train"] = array;
  } else {
    output["rfx_predictions_train"] = py::none();
  }
  if (!results_raw.rfx_predictions_test.empty()) {
    auto& v = results_raw.rfx_predictions_test;
    py::array_t<double> array(v.size());
    std::copy(v.begin(), v.end(), array.mutable_data());
    output["rfx_predictions_test"] = array;
  } else {
    output["rfx_predictions_test"] = py::none();
  }

  // Transfer ownership of random effects container pointers
  if (results_raw.rfx_container != nullptr) {
    output["rfx_container"] = py::cast(std::make_unique<RandomEffectsContainerCpp>(std::move(results_raw.rfx_container)));
  } else {
    output["rfx_container"] = py::none();
  }

  // Transfer ownership of random effects label mapper pointer
  if (results_raw.rfx_label_mapper != nullptr) {
    output["rfx_label_mapper"] = py::cast(std::make_unique<RandomEffectsLabelMapperCpp>(std::move(results_raw.rfx_label_mapper)));
  } else {
    output["rfx_label_mapper"] = py::none();
  }

  // Unpack scalars
  output["y_bar"] = results_raw.y_bar;
  output["y_std"] = results_raw.y_std;
  output["num_samples"] = results_raw.num_samples;
  output["num_train"] = results_raw.num_train;
  output["num_test"] = results_raw.num_test;

  return output;
}

inline py::dict convert_bart_preds_to_dict(StochTree::BARTPredictionResult& results_raw) {
  py::dict output;

  // Move prediction samples

  if (results_raw.y_hat.empty()) {
    output["y_hat"] = py::none();
  } else {
    py::array_t<double> array(results_raw.y_hat.size());
    std::copy(results_raw.y_hat.begin(), results_raw.y_hat.end(), array.mutable_data());
    output["y_hat"] = array;
  }

  if (results_raw.mean_forest_predictions.empty()) {
    output["mean_forest_predictions"] = py::none();
  } else {
    py::array_t<double> array(results_raw.mean_forest_predictions.size());
    std::copy(results_raw.mean_forest_predictions.begin(), results_raw.mean_forest_predictions.end(), array.mutable_data());
    output["mean_forest_predictions"] = array;
  }

  if (results_raw.variance_forest_predictions.empty()) {
    output["variance_forest_predictions"] = py::none();
  } else {
    py::array_t<double> array(results_raw.variance_forest_predictions.size());
    std::copy(results_raw.variance_forest_predictions.begin(), results_raw.variance_forest_predictions.end(), array.mutable_data());
    output["variance_forest_predictions"] = array;
  }

  if (results_raw.rfx_predictions.empty()) {
    output["rfx_predictions"] = py::none();
  } else {
    py::array_t<double> array(results_raw.rfx_predictions.size());
    std::copy(results_raw.rfx_predictions.begin(), results_raw.rfx_predictions.end(), array.mutable_data());
    output["rfx_predictions"] = array;
  }

  return output;
}

py::dict bart_predict_cpp(
    BARTSamplesCpp& samples,
    py::dict metadata,
    py::object X,
    py::object leaf_basis,
    int n,
    int p,
    int num_basis,
    py::object obs_weights,
    py::object rfx_group_ids,
    py::object rfx_basis,
    int rfx_num_groups,
    int rfx_basis_dim,
    bool posterior,
    int scale,
    bool predict_y_hat,
    bool predict_mean_forest,
    bool predict_variance_forest,
    bool predict_random_effects) {
  // Pre-convert all numpy inputs to F-contiguous at function scope so the raw pointers
  // stored in BARTData outlive the convert_numpy_to_bart_data call.
  using FArray = py::array_t<double, py::array::f_style | py::array::forcecast>;
  using IArray = py::array_t<int, py::array::f_style | py::array::forcecast>;
  FArray X_farr, leaf_basis_farr, obs_weights_farr, rfx_basis_farr;
  IArray rfx_group_ids_iarr;
  if (!X.is_none()) X_farr = X.cast<FArray>();
  if (!leaf_basis.is_none()) leaf_basis_farr = leaf_basis.cast<FArray>();
  if (!obs_weights.is_none()) obs_weights_farr = obs_weights.cast<FArray>();
  if (!rfx_group_ids.is_none()) rfx_group_ids_iarr = rfx_group_ids.cast<IArray>();
  if (!rfx_basis.is_none()) rfx_basis_farr = rfx_basis.cast<FArray>();

  // Build BARTData with test-only fields
  StochTree::BARTData bart_data = convert_numpy_to_bart_data(
      /*X_train=*/py::none(),
      /*y_train=*/py::none(),
      /*X_test=*/X.is_none() ? py::object(py::none()) : py::object(X_farr),
      /*n_train=*/0,
      /*n_test=*/n,
      /*p=*/p,
      /*basis_train=*/py::none(),
      /*basis_test=*/leaf_basis.is_none() ? py::object(py::none()) : py::object(leaf_basis_farr),
      /*basis_dim=*/num_basis,
      /*obs_weights_train=*/py::none(),
      /*obs_weights_test=*/obs_weights.is_none() ? py::object(py::none()) : py::object(obs_weights_farr),
      /*rfx_group_ids_train=*/py::none(),
      /*rfx_group_ids_test=*/rfx_group_ids.is_none() ? py::object(py::none()) : py::object(rfx_group_ids_iarr),
      /*rfx_basis_train=*/py::none(),
      /*rfx_basis_test=*/rfx_basis.is_none() ? py::object(py::none()) : py::object(rfx_basis_farr),
      /*rfx_num_groups=*/rfx_num_groups,
      /*rfx_basis_dim=*/rfx_basis_dim);

  // Build BARTPredictionInput from the model dict
  // Forests, parameter traces, rfx and cloglog cutpoints all live on the single-owner samples
  // object; only scalar metadata / model flags are passed separately.
  StochTree::BARTPredictionMetadata pred_metadata;
  pred_metadata.num_samples = metadata["num_samples"].cast<int>();
  pred_metadata.num_obs = n;
  pred_metadata.num_basis = num_basis;
  pred_metadata.y_bar = metadata["y_bar"].cast<double>();
  pred_metadata.y_std = metadata["y_std"].cast<double>();
  pred_metadata.has_variance_forest = metadata["include_variance_forest"].cast<bool>();
  pred_metadata.has_rfx = metadata["has_rfx"].cast<bool>();
  pred_metadata.cloglog_num_classes = metadata.contains("cloglog_num_classes") ? metadata["cloglog_num_classes"].cast<int>() : 0;
  {
    std::string rfx_spec_str = "";
    if (metadata.contains("rfx_model_spec") && !metadata["rfx_model_spec"].is_none()) {
      rfx_spec_str = metadata["rfx_model_spec"].cast<std::string>();
    }
    pred_metadata.rfx_model_spec = (rfx_spec_str == "intercept_only")
                                       ? StochTree::BARTRFXModelSpec::InterceptOnly
                                       : StochTree::BARTRFXModelSpec::Custom;
  }
  {
    std::string link_str = metadata.contains("link_function") ? metadata["link_function"].cast<std::string>() : "identity";
    if (link_str == "probit")
      pred_metadata.link_function = StochTree::LinkFunction::Probit;
    else if (link_str == "cloglog")
      pred_metadata.link_function = StochTree::LinkFunction::Cloglog;
    else
      pred_metadata.link_function = StochTree::LinkFunction::Identity;
  }
  {
    std::string outcome_str = metadata.contains("outcome_type") ? metadata["outcome_type"].cast<std::string>() : "continuous";
    if (outcome_str == "binary")
      pred_metadata.outcome_type = StochTree::OutcomeType::Binary;
    else if (outcome_str == "ordinal")
      pred_metadata.outcome_type = StochTree::OutcomeType::Ordinal;
    else
      pred_metadata.outcome_type = StochTree::OutcomeType::Continuous;
  }
  pred_metadata.pred_type = posterior ? StochTree::PredType::kPosterior : StochTree::PredType::kMean;
  if (scale == 0)
    pred_metadata.pred_scale = StochTree::PredScale::kLinear;
  else if (scale == 1)
    pred_metadata.pred_scale = StochTree::PredScale::kProbability;
  else
    pred_metadata.pred_scale = StochTree::PredScale::kClass;
  pred_metadata.pred_terms.y_hat = predict_y_hat;
  pred_metadata.pred_terms.mean_forest = predict_mean_forest;
  pred_metadata.pred_terms.variance_forest = predict_variance_forest;
  pred_metadata.pred_terms.random_effects = predict_random_effects;

  // Run prediction against the single-owner samples object.
  StochTree::BARTPredictionResult pred_results = predict_bart_model(bart_data, *samples.GetPtr(), pred_metadata);

  return convert_bart_preds_to_dict(pred_results);
}

inline py::dict convert_bcf_preds_to_dict(StochTree::BCFPredictionResult& results_raw) {
  py::dict output;

  // Move prediction samples

  if (results_raw.y_hat.empty()) {
    output["y_hat"] = py::none();
  } else {
    py::array_t<double> array(results_raw.y_hat.size());
    std::copy(results_raw.y_hat.begin(), results_raw.y_hat.end(), array.mutable_data());
    output["y_hat"] = array;
  }

  if (results_raw.mu_x.empty()) {
    output["mu_x"] = py::none();
  } else {
    py::array_t<double> array(results_raw.mu_x.size());
    std::copy(results_raw.mu_x.begin(), results_raw.mu_x.end(), array.mutable_data());
    output["mu_x"] = array;
  }

  if (results_raw.tau_x.empty()) {
    output["tau_x"] = py::none();
  } else {
    py::array_t<double> array(results_raw.tau_x.size());
    std::copy(results_raw.tau_x.begin(), results_raw.tau_x.end(), array.mutable_data());
    output["tau_x"] = array;
  }

  if (results_raw.prognostic_function.empty()) {
    output["prognostic_function"] = py::none();
  } else {
    py::array_t<double> array(results_raw.prognostic_function.size());
    std::copy(results_raw.prognostic_function.begin(), results_raw.prognostic_function.end(), array.mutable_data());
    output["prognostic_function"] = array;
  }

  if (results_raw.cate.empty()) {
    output["cate"] = py::none();
  } else {
    py::array_t<double> array(results_raw.cate.size());
    std::copy(results_raw.cate.begin(), results_raw.cate.end(), array.mutable_data());
    output["cate"] = array;
  }

  if (results_raw.conditional_variance.empty()) {
    output["conditional_variance"] = py::none();
  } else {
    py::array_t<double> array(results_raw.conditional_variance.size());
    std::copy(results_raw.conditional_variance.begin(), results_raw.conditional_variance.end(), array.mutable_data());
    output["conditional_variance"] = array;
  }

  if (results_raw.random_effects.empty()) {
    output["random_effects"] = py::none();
  } else {
    py::array_t<double> array(results_raw.random_effects.size());
    std::copy(results_raw.random_effects.begin(), results_raw.random_effects.end(), array.mutable_data());
    output["random_effects"] = array;
  }

  return output;
}

void add_config_to_bcf_result_dict(py::dict& result, StochTree::BCFConfig& config) {
  // Unpack more metadata about the model that was sampled
  result["sigma2_init"] = config.sigma2_global_init;
  result["sigma2_mu_init"] = config.sigma2_mu_init;
  result["sigma2_tau_init"] = config.sigma2_tau_init;
  result["b_sigma2_mu"] = config.b_sigma2_mu;
  result["b_sigma2_tau"] = config.b_sigma2_tau;
  result["shape_variance_forest"] = config.shape_variance_forest;
  result["scale_variance_forest"] = config.scale_variance_forest;
}

py::dict bcf_sample_cpp(
    BCFSamplesCpp& samples,
    py::object X_train,
    py::object Z_train,
    py::object y_train,
    py::object X_test,
    py::object Z_test,
    int n_train,
    int n_test,
    int p,
    int treatment_dim,
    py::object obs_weights_train,
    py::object obs_weights_test,
    py::object rfx_group_ids_train,
    py::object rfx_group_ids_test,
    py::object rfx_basis_train,
    py::object rfx_basis_test,
    int rfx_num_groups,
    int rfx_basis_dim,
    int num_gfr,
    int num_burnin,
    int keep_every,
    int num_mcmc,
    int num_chains,
    bool adaptive_coding,
    py::dict config_input) {
  // Convert config dict to BCFConfig struct
  StochTree::BCFConfig bcf_config = convert_dict_to_bcf_config(config_input);

  // Unpack pointers to input data to BCFData object
  StochTree::BCFData bcf_data = convert_numpy_to_bcf_data(X_train, Z_train, y_train, X_test, Z_test, n_train, n_test, p, treatment_dim, obs_weights_train, obs_weights_test, rfx_group_ids_train, rfx_group_ids_test, rfx_basis_train, rfx_basis_test, rfx_num_groups, rfx_basis_dim);

  // Single-owner (mirrors R bcf_sample_cpp): the C++ sampler populates the caller-owned BCFSamples in
  // place; this function only returns metadata that is NOT part of the samples object.
  StochTree::BCFSamples& bcf_samples = *samples.GetPtr();

  // Initialize a BCF sampler
  StochTree::BCFSampler bcf_sampler(bcf_samples, bcf_config, bcf_data);

  // Run the sampler
  bcf_sampler.run_gfr(bcf_samples, num_gfr, bcf_config.keep_gfr, num_chains);
  if (num_chains > 1) {
    bcf_sampler.run_mcmc_chains(bcf_samples, num_chains, num_burnin, keep_every, num_mcmc);
  } else {
    bcf_sampler.run_mcmc(bcf_samples, num_burnin, keep_every, num_mcmc);
  }
  bcf_sampler.postprocess_samples(bcf_samples);

  // Metadata: config-derived init scalars and the final RNG state (for continue_sampling). Everything
  // else is read off the caller's samples object.
  py::dict metadata;
  add_config_to_bcf_result_dict(metadata, bcf_config);
  metadata["rng_state"] = (num_chains <= 1) ? py::object(py::cast(bcf_sampler.GetRngState())) : py::object(py::none());
  return metadata;
}

// Continue (warm-start) sampling from an already-fit BCF model.
//
// Reconstruct-on-demand stopgap (RFC 0005 / #408), mirroring bart_continue_sample_cpp:
// DEEP-COPIES the model's mu and tau forest containers into a fresh BCFSamples so the caller's
// containers remain the source of truth, pre-populates the scalar sample histories in
// STANDARDIZED space, constructs the sampler in continuation mode (warm-starts the active
// forests from the last retained samples and appends new draws), then returns the same dict
// shape as bcf_sample_cpp with the containers and parameter arrays extended.
//
// Supports identity-link, univariate-treatment Gaussian BCF without a variance forest, random
// effects, or adaptive coding (the sampler hard-errors on those). Predictions in the returned
// dict are recomputed by the Python wrapper post-hoc, so postprocess_samples is skipped here
// (it would index the per-iteration prediction arrays, which only hold the new samples).
py::dict bcf_continue_sample_cpp(
    BCFSamplesCpp& samples,
    py::object X_train,
    py::object Z_train,
    py::object y_train,
    py::object X_test,
    py::object Z_test,
    int n_train,
    int n_test,
    int p,
    int treatment_dim,
    py::object obs_weights_train,
    py::object rfx_group_ids_train,
    py::object rfx_basis_train,
    py::object rfx_group_ids_test,
    py::object rfx_basis_test,
    int rfx_num_groups,
    int rfx_basis_dim,
    int num_burnin,
    int keep_every,
    int num_mcmc,
    std::string rng_state_in,
    bool override_seed,
    py::dict config_input) {
  // Convert config dict to BCFConfig struct
  StochTree::BCFConfig bcf_config = convert_dict_to_bcf_config(config_input);

  // Unpack pointers to (re-supplied) input data to BCFData object. A test set is optional on
  // continuation; when supplied, postprocess_samples recomputes the full test-prediction trace from
  // all retained forests.
  StochTree::BCFData bcf_data = convert_numpy_to_bcf_data(
      X_train, Z_train, y_train, X_test, Z_test,
      n_train, n_test, p, treatment_dim, obs_weights_train, /*obs_weights_test=*/py::none(),
      rfx_group_ids_train, rfx_group_ids_test,
      rfx_basis_train, rfx_basis_test,
      rfx_num_groups, rfx_basis_dim);

  // Continuation appends new MCMC draws in place onto the model's single-owner samples object,
  // mirroring the R sampler. The warm-start reads the last retained forest + scalar state directly
  // off `samples` (its global variance history is post-processed and tau_0 scaled by y_std; the
  // sampler un-scales them internally), and postprocess_samples(samples, num_history) rescales only
  // the newly-appended draws (train + params; no test data on continuation).
  StochTree::BCFSamples& bcf_samples = *samples.GetPtr();
  const int num_history = bcf_samples.num_samples;

  // Initialize a BCF sampler in continuation mode (warm-start from last samples)
  StochTree::BCFSampler bcf_sampler(bcf_samples, bcf_config, bcf_data, /*continuation=*/true);

  // Resume the RNG stream unless the user re-seeded (override_seed).
  if (!override_seed && !rng_state_in.empty()) {
    bcf_sampler.SetRngState(rng_state_in);
  }

  // Probit warm-start: regenerate the (unpersisted) latent outcome now that the RNG is positioned at
  // the resumed/re-seeded stream, so the first continued draw starts from a valid stationary state.
  bcf_sampler.RegenerateProbitLatent(bcf_samples);

  // Append new MCMC samples (continuation does not run GFR), then post-process only the new range.
  bcf_sampler.run_mcmc(bcf_samples, num_burnin, keep_every, num_mcmc);
  bcf_sampler.postprocess_samples(bcf_samples, num_history);

  // Metadata only; the extended samples live on the passed-in samples object.
  py::dict metadata;
  add_config_to_bcf_result_dict(metadata, bcf_config);
  metadata["rng_state"] = bcf_sampler.GetRngState();
  return metadata;
}

py::dict bcf_predict_cpp(
    BCFSamplesCpp& samples,
    py::dict metadata,
    py::object X,
    py::object Z,
    int n,
    int p,
    int treatment_dim,
    py::object obs_weights,
    py::object rfx_group_ids,
    py::object rfx_basis,
    int rfx_num_groups,
    int rfx_basis_dim,
    bool posterior,
    int scale,
    bool predict_y_hat,
    bool predict_mu_x,
    bool predict_tau_x,
    bool predict_prognostic_function,
    bool predict_cate,
    bool predict_conditional_variance,
    bool predict_random_effects) {
  // Pre-convert test data to F-contiguous at function scope so the buffers outlive bcf_data
  // and the predict_bcf_model call. convert_numpy_to_bcf_data casts inside if-blocks, so its
  // temporaries are freed before predict_bcf_model runs -- these function-scope arrays prevent that.
  using FArray = py::array_t<double, py::array::f_style | py::array::forcecast>;
  using IArray = py::array_t<int, py::array::f_style | py::array::forcecast>;
  FArray X_farr, Z_farr, obs_weights_farr, rfx_basis_farr;
  IArray rfx_group_ids_iarr;
  if (!X.is_none()) X_farr = X.cast<FArray>();
  if (!Z.is_none()) Z_farr = Z.cast<FArray>();
  if (!obs_weights.is_none()) obs_weights_farr = obs_weights.cast<FArray>();
  if (!rfx_group_ids.is_none()) rfx_group_ids_iarr = rfx_group_ids.cast<IArray>();
  if (!rfx_basis.is_none()) rfx_basis_farr = rfx_basis.cast<FArray>();

  // Unpack pointers to input data to BCFData object -- use only the "test" data fields as this is what the predict function expects
  StochTree::BCFData bcf_data = convert_numpy_to_bcf_data(
      /*X_train=*/py::none(), /*Z_train=*/py::none(), /*y_train=*/py::none(),
      /*X_test=*/X.is_none() ? py::object(py::none()) : py::object(X_farr),
      /*Z_test=*/Z.is_none() ? py::object(py::none()) : py::object(Z_farr),
      /*n_train=*/0, /*n_test=*/n, /*p=*/p, /*treatment_dim=*/treatment_dim,
      /*obs_weights_train=*/py::none(),
      /*obs_weights_test=*/obs_weights.is_none() ? py::object(py::none()) : py::object(obs_weights_farr),
      /*rfx_group_ids_train=*/py::none(),
      /*rfx_group_ids_test=*/rfx_group_ids.is_none() ? py::object(py::none()) : py::object(rfx_group_ids_iarr),
      /*rfx_basis_train=*/py::none(),
      /*rfx_basis_test=*/rfx_basis.is_none() ? py::object(py::none()) : py::object(rfx_basis_farr),
      /*rfx_num_groups=*/rfx_num_groups, /*rfx_basis_dim=*/rfx_basis_dim);

  // Load the BCF model and config from the model list
  // Forests, parameter traces, rfx all live on the single-owner samples object; only scalar
  // metadata / model flags are passed separately.
  StochTree::BCFPredictionMetadata pred_metadata;
  pred_metadata.num_samples = metadata["num_samples"].cast<int>();
  pred_metadata.num_obs = n;
  pred_metadata.treatment_dim = treatment_dim;
  pred_metadata.y_bar = metadata["y_bar"].cast<double>();
  pred_metadata.y_std = metadata["y_std"].cast<double>();
  pred_metadata.has_variance_forest = metadata["include_variance_forest"].cast<bool>();
  pred_metadata.has_rfx = metadata["has_rfx"].cast<bool>();
  std::string rfx_model_spec_str = "";
  if (metadata.contains("rfx_model_spec") && !metadata["rfx_model_spec"].is_none()) {
    rfx_model_spec_str = metadata["rfx_model_spec"].cast<std::string>();
  }
  if (rfx_model_spec_str == "intercept_only") {
    pred_metadata.rfx_model_spec = StochTree::BCFRFXModelSpec::InterceptOnly;
  } else if (rfx_model_spec_str == "intercept_plus_treatment") {
    pred_metadata.rfx_model_spec = StochTree::BCFRFXModelSpec::InterceptPlusTreatment;
  } else {
    pred_metadata.rfx_model_spec = StochTree::BCFRFXModelSpec::Custom;
  }
  pred_metadata.adaptive_coding = metadata["adaptive_coding"].cast<bool>();
  pred_metadata.sample_tau_0 = metadata["sample_tau_0"].cast<bool>();
  pred_metadata.pred_type = posterior ? StochTree::PredType::kPosterior : StochTree::PredType::kMean;
  if (scale == 0) {
    pred_metadata.pred_scale = StochTree::PredScale::kLinear;
  } else if (scale == 1) {
    pred_metadata.pred_scale = StochTree::PredScale::kProbability;
  } else {
    pred_metadata.pred_scale = StochTree::PredScale::kClass;
  }
  pred_metadata.pred_terms.y_hat = predict_y_hat;
  pred_metadata.pred_terms.mu_x = predict_mu_x;
  pred_metadata.pred_terms.tau_x = predict_tau_x;
  pred_metadata.pred_terms.prognostic_function = predict_prognostic_function;
  pred_metadata.pred_terms.cate = predict_cate;
  pred_metadata.pred_terms.conditional_variance = predict_conditional_variance;
  pred_metadata.pred_terms.random_effects = predict_random_effects;

  // Run the prediction function against the single-owner samples object.
  StochTree::BCFPredictionResult pred_results = predict_bcf_model(bcf_data, *samples.GetPtr(), pred_metadata);

  // Unpack outputs
  py::dict output = convert_bcf_preds_to_dict(pred_results);
  return output;
}

py::array_t<int> cppComputeForestContainerLeafIndices(ForestContainerCpp& forest_container, py::array_t<double>& covariates, py::array_t<int>& forest_nums) {
  // Wrap an Eigen Map around the raw data of the covariate matrix
  StochTree::data_size_t num_obs = covariates.shape(0);
  int num_covariates = covariates.shape(1);
  double* covariate_data_ptr = static_cast<double*>(covariates.mutable_data());
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> covariates_eigen(covariate_data_ptr, num_obs, num_covariates);

  // Extract other output dimensions
  int num_trees = forest_container.NumTrees();
  int num_samples = forest_nums.size();

  // Convert forest_nums to std::vector
  std::vector<int> forest_indices(num_samples);
  for (int i = 0; i < num_samples; i++) {
    forest_indices[i] = forest_nums.at(i);
  }

  // Compute leaf indices
  auto result = py::array_t<int, py::array::f_style>(py::detail::any_container<py::ssize_t>({num_obs * num_trees, num_samples}));
  int* output_data_ptr = static_cast<int*>(result.mutable_data());
  Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> output_eigen(output_data_ptr, num_obs * num_trees, num_samples);
  forest_container.GetContainer()->PredictLeafIndicesInplace(covariates_eigen, output_eigen, forest_indices, num_trees, num_obs);

  // Return matrix
  return result;
}

int cppComputeForestMaxLeafIndex(ForestContainerCpp& forest_container, int forest_num) {
  return forest_container.GetForest(forest_num)->GetMaxLeafIndex() - 1;
}

void ForestContainerCpp::LoadFromJson(JsonCpp& json, std::string forest_label) {
  nlohmann::json forest_json = json.SubsetJsonForest(forest_label);
  forest_samples_->Reset();
  forest_samples_->from_json(forest_json);
}

void ForestContainerCpp::AppendFromJson(JsonCpp& json, std::string forest_label) {
  nlohmann::json forest_json = json.SubsetJsonForest(forest_label);
  forest_samples_->append_from_json(forest_json);
}

void ForestContainerCpp::AdjustResidual(ForestDatasetCpp& dataset, ResidualCpp& residual, ForestSamplerCpp& sampler, bool requires_basis, int forest_num, bool add) {
  // Determine whether or not we are adding forest_num to the residuals
  std::function<double(double, double)> op;
  if (add)
    op = std::plus<double>();
  else
    op = std::minus<double>();

  // Perform the update (addition / subtraction) operation
  StochTree::UpdateResidualEntireForest(*(sampler.GetTracker()), *(dataset.GetDataset()), *(residual.GetData()), forest_samples_->GetEnsemble(forest_num), requires_basis, op);
}

void ForestCpp::AdjustResidual(ForestDatasetCpp& dataset, ResidualCpp& residual, ForestSamplerCpp& sampler, bool requires_basis, bool add) {
  // Determine whether or not we are adding forest predictions to the residuals
  std::function<double(double, double)> op;
  if (add)
    op = std::plus<double>();
  else
    op = std::minus<double>();

  // Perform the update (addition / subtraction) operation
  StochTree::UpdateResidualEntireForest(*(sampler.GetTracker()), *(dataset.GetDataset()), *(residual.GetData()), forest_.get(), requires_basis, op);
}

void RandomEffectsContainerCpp::LoadFromJson(JsonCpp& json, std::string rfx_container_label) {
  nlohmann::json rfx_json = json.SubsetJsonRFX().at(rfx_container_label);
  rfx_container_->Reset();
  rfx_container_->from_json(rfx_json);
}

void RandomEffectsContainerCpp::AppendFromJson(JsonCpp& json, std::string rfx_container_label) {
  nlohmann::json rfx_json = json.SubsetJsonRFX().at(rfx_container_label);
  rfx_container_->append_from_json(rfx_json);
}

void RandomEffectsContainerCpp::AddSample(RandomEffectsModelCpp& rfx_model) {
  rfx_container_->AddSample(*rfx_model.GetModel());
}

py::array_t<double> RandomEffectsContainerCpp::Predict(RandomEffectsDatasetCpp& rfx_dataset, RandomEffectsLabelMapperCpp& label_mapper) {
  py::ssize_t num_observations = rfx_dataset.NumObservations();
  int num_samples = rfx_container_->NumSamples();
  std::vector<double> output(num_observations * num_samples);
  rfx_container_->Predict(*rfx_dataset.GetDataset(), *label_mapper.GetLabelMapper(), output);
  auto result = py::array_t<double>(py::detail::any_container<py::ssize_t>({num_observations, num_samples}));
  auto accessor = result.mutable_unchecked<2>();
  for (size_t i = 0; i < num_observations; i++) {
    for (int j = 0; j < num_samples; j++) {
      accessor(i, j) = output.at(j * num_observations + i);
    }
  }
  return result;
}

void RandomEffectsLabelMapperCpp::LoadFromJson(JsonCpp& json, std::string rfx_label_mapper_label) {
  nlohmann::json rfx_json = json.SubsetJsonRFX().at(rfx_label_mapper_label);
  rfx_label_mapper_->Reset();
  rfx_label_mapper_->from_json(rfx_json);
}

void RandomEffectsModelCpp::SampleRandomEffects(RandomEffectsDatasetCpp& rfx_dataset, ResidualCpp& residual,
                                                RandomEffectsTrackerCpp& rfx_tracker, RandomEffectsContainerCpp& rfx_container,
                                                bool keep_sample, double global_variance, RngCpp& rng) {
  rfx_model_->SampleRandomEffects(*rfx_dataset.GetDataset(), *residual.GetData(),
                                  *rfx_tracker.GetTracker(), global_variance, *rng.GetRng());
  if (keep_sample) rfx_container.AddSample(*this);
}

void RandomEffectsTrackerCpp::Reset(RandomEffectsModelCpp& rfx_model, RandomEffectsDatasetCpp& rfx_dataset, ResidualCpp& residual) {
  rfx_tracker_->ResetFromSample(*rfx_model.GetModel(), *rfx_dataset.GetDataset(), *residual.GetData());
}

void RandomEffectsTrackerCpp::RootReset(RandomEffectsModelCpp& rfx_model, RandomEffectsDatasetCpp& rfx_dataset, ResidualCpp& residual) {
  rfx_tracker_->RootReset(*rfx_model.GetModel(), *rfx_dataset.GetDataset(), *residual.GetData());
}

PYBIND11_MODULE(stochtree_cpp, m) {
  m.def("cppComputeForestContainerLeafIndices", &cppComputeForestContainerLeafIndices, "Compute leaf indices of the forests in a forest container");
  m.def("cppComputeForestMaxLeafIndex", &cppComputeForestMaxLeafIndex, "Compute max leaf index of a forest in a forest container");
  m.def("bart_sample_cpp", &bart_sample_cpp, "Run BART sampler in C++ implementation",
        py::arg("samples"),
        py::arg("X_train"),
        py::arg("y_train"),
        py::arg("X_test") = py::none(),
        py::arg("n_train"),
        py::arg("n_test"),
        py::arg("p"),
        py::arg("basis_train") = py::none(),
        py::arg("basis_test") = py::none(),
        py::arg("basis_dim"),
        py::arg("obs_weights_train") = py::none(),
        py::arg("obs_weights_test") = py::none(),
        py::arg("rfx_group_ids_train") = py::none(),
        py::arg("rfx_group_ids_test") = py::none(),
        py::arg("rfx_basis_train") = py::none(),
        py::arg("rfx_basis_test") = py::none(),
        py::arg("rfx_num_groups"),
        py::arg("rfx_basis_dim"),
        py::arg("num_gfr"),
        py::arg("num_burnin"),
        py::arg("keep_every"),
        py::arg("num_mcmc"),
        py::arg("num_chains"),
        py::arg("config_input"));

  m.def("bart_continue_sample_cpp", &bart_continue_sample_cpp, "Continue (warm-start) BART sampling from an existing model in C++",
        py::arg("samples"),
        py::arg("X_train"),
        py::arg("y_train"),
        py::arg("X_test") = py::none(),
        py::arg("n_train"),
        py::arg("n_test"),
        py::arg("p"),
        py::arg("basis_train") = py::none(),
        py::arg("basis_test") = py::none(),
        py::arg("basis_dim"),
        py::arg("obs_weights_train") = py::none(),
        py::arg("obs_weights_test") = py::none(),
        py::arg("rfx_group_ids_train") = py::none(),
        py::arg("rfx_group_ids_test") = py::none(),
        py::arg("rfx_basis_train") = py::none(),
        py::arg("rfx_basis_test") = py::none(),
        py::arg("rfx_num_groups"),
        py::arg("rfx_basis_dim"),
        py::arg("num_gfr") = 0,
        py::arg("num_burnin"),
        py::arg("keep_every"),
        py::arg("num_mcmc"),
        py::arg("keep_gfr") = true,
        py::arg("rng_state_in") = std::string(),
        py::arg("override_seed") = false,
        py::arg("config_input"));

  m.def("bcf_sample_cpp", &bcf_sample_cpp, "Run BCF sampler in C++ implementation",
        py::arg("samples"),
        py::arg("X_train"),
        py::arg("Z_train"),
        py::arg("y_train"),
        py::arg("X_test") = py::none(),
        py::arg("Z_test") = py::none(),
        py::arg("n_train"),
        py::arg("n_test"),
        py::arg("p"),
        py::arg("treatment_dim"),
        py::arg("obs_weights_train") = py::none(),
        py::arg("obs_weights_test") = py::none(),
        py::arg("rfx_group_ids_train") = py::none(),
        py::arg("rfx_group_ids_test") = py::none(),
        py::arg("rfx_basis_train") = py::none(),
        py::arg("rfx_basis_test") = py::none(),
        py::arg("rfx_num_groups"),
        py::arg("rfx_basis_dim"),
        py::arg("num_gfr"),
        py::arg("num_burnin"),
        py::arg("keep_every"),
        py::arg("num_mcmc"),
        py::arg("num_chains"),
        py::arg("adaptive_coding"),
        py::arg("config_input"));

  m.def("bcf_continue_sample_cpp", &bcf_continue_sample_cpp, "Continue (warm-start) BCF sampling from an existing model in C++",
        py::arg("samples"),
        py::arg("X_train"),
        py::arg("Z_train"),
        py::arg("y_train"),
        py::arg("X_test") = py::none(),
        py::arg("Z_test") = py::none(),
        py::arg("n_train"),
        py::arg("n_test") = 0,
        py::arg("p"),
        py::arg("treatment_dim"),
        py::arg("obs_weights_train") = py::none(),
        py::arg("rfx_group_ids_train") = py::none(),
        py::arg("rfx_basis_train") = py::none(),
        py::arg("rfx_group_ids_test") = py::none(),
        py::arg("rfx_basis_test") = py::none(),
        py::arg("rfx_num_groups") = 0,
        py::arg("rfx_basis_dim") = 0,
        py::arg("num_burnin"),
        py::arg("keep_every"),
        py::arg("num_mcmc"),
        py::arg("rng_state_in") = std::string(),
        py::arg("override_seed") = false,
        py::arg("config_input"));

  m.def("bart_predict_cpp", &bart_predict_cpp, "Run BART predictions in C++",
        py::arg("samples"),
        py::arg("metadata"),
        py::arg("X"),
        py::arg("leaf_basis") = py::none(),
        py::arg("n"),
        py::arg("p"),
        py::arg("num_basis") = 0,
        py::arg("obs_weights") = py::none(),
        py::arg("rfx_group_ids") = py::none(),
        py::arg("rfx_basis") = py::none(),
        py::arg("rfx_num_groups") = 0,
        py::arg("rfx_basis_dim") = 0,
        py::arg("posterior"),
        py::arg("scale"),
        py::arg("predict_y_hat"),
        py::arg("predict_mean_forest"),
        py::arg("predict_variance_forest"),
        py::arg("predict_random_effects"));

  m.def("bcf_predict_cpp", &bcf_predict_cpp, "Run BCF predictions in C++",
        py::arg("samples"),
        py::arg("metadata"),
        py::arg("X"),
        py::arg("Z"),
        py::arg("n"),
        py::arg("p"),
        py::arg("treatment_dim"),
        py::arg("obs_weights") = py::none(),
        py::arg("rfx_group_ids") = py::none(),
        py::arg("rfx_basis") = py::none(),
        py::arg("rfx_num_groups"),
        py::arg("rfx_basis_dim"),
        py::arg("posterior"),
        py::arg("scale"),
        py::arg("predict_y_hat"),
        py::arg("predict_mu_x"),
        py::arg("predict_tau_x"),
        py::arg("predict_prognostic_function"),
        py::arg("predict_cate"),
        py::arg("predict_conditional_variance"),
        py::arg("predict_random_effects"));

  py::class_<JsonCpp>(m, "JsonCpp")
      .def(py::init<>())
      .def("LoadFile", &JsonCpp::LoadFile)
      .def("SaveFile", &JsonCpp::SaveFile)
      .def("LoadFromString", &JsonCpp::LoadFromString)
      .def("DumpJson", &JsonCpp::DumpJson)
      .def("AddDouble", &JsonCpp::AddDouble)
      .def("AddDoubleSubfolder", &JsonCpp::AddDoubleSubfolder)
      .def("AddInteger", &JsonCpp::AddInteger)
      .def("AddIntegerSubfolder", &JsonCpp::AddIntegerSubfolder)
      .def("AddBool", &JsonCpp::AddBool)
      .def("AddBoolSubfolder", &JsonCpp::AddBoolSubfolder)
      .def("AddString", &JsonCpp::AddString)
      .def("AddStringSubfolder", &JsonCpp::AddStringSubfolder)
      .def("AddDoubleVector", &JsonCpp::AddDoubleVector)
      .def("AddDoubleVectorSubfolder", &JsonCpp::AddDoubleVectorSubfolder)
      .def("AddIntegerVector", &JsonCpp::AddIntegerVector)
      .def("AddIntegerVectorSubfolder", &JsonCpp::AddIntegerVectorSubfolder)
      .def("AddStringVector", &JsonCpp::AddStringVector)
      .def("AddStringVectorSubfolder", &JsonCpp::AddStringVectorSubfolder)
      .def("AddForest", &JsonCpp::AddForest, py::arg("forest_samples"), py::arg("forest_label") = std::string(""))
      .def("AddRandomEffectsContainer", &JsonCpp::AddRandomEffectsContainer)
      .def("AddRandomEffectsLabelMapper", &JsonCpp::AddRandomEffectsLabelMapper)
      .def("AddRandomEffectsGroupIDs", &JsonCpp::AddRandomEffectsGroupIDs)
      .def("ContainsField", &JsonCpp::ContainsField)
      .def("ContainsFieldSubfolder", &JsonCpp::ContainsFieldSubfolder)
      .def("EraseField", &JsonCpp::EraseField)
      .def("EraseFieldSubfolder", &JsonCpp::EraseFieldSubfolder)
      .def("RenameField", &JsonCpp::RenameField)
      .def("RenameFieldSubfolder", &JsonCpp::RenameFieldSubfolder)
      .def("ExtractDouble", &JsonCpp::ExtractDouble)
      .def("ExtractDoubleSubfolder", &JsonCpp::ExtractDoubleSubfolder)
      .def("ExtractInteger", &JsonCpp::ExtractInteger)
      .def("ExtractIntegerSubfolder", &JsonCpp::ExtractIntegerSubfolder)
      .def("ExtractBool", &JsonCpp::ExtractBool)
      .def("ExtractBoolSubfolder", &JsonCpp::ExtractBoolSubfolder)
      .def("ExtractString", &JsonCpp::ExtractString)
      .def("ExtractStringSubfolder", &JsonCpp::ExtractStringSubfolder)
      .def("ExtractDoubleVector", &JsonCpp::ExtractDoubleVector)
      .def("ExtractDoubleVectorSubfolder", &JsonCpp::ExtractDoubleVectorSubfolder)
      .def("ExtractIntegerVector", &JsonCpp::ExtractIntegerVector)
      .def("ExtractIntegerVectorSubfolder", &JsonCpp::ExtractIntegerVectorSubfolder)
      .def("ExtractStringVector", &JsonCpp::ExtractStringVector)
      .def("ExtractStringVectorSubfolder", &JsonCpp::ExtractStringVectorSubfolder)
      .def("IncrementRandomEffectsCount", &JsonCpp::IncrementRandomEffectsCount)
      .def("SubsetJsonForest", &JsonCpp::SubsetJsonForest)
      .def("SubsetJsonRFX", &JsonCpp::SubsetJsonRFX);

  py::class_<ForestDatasetCpp>(m, "ForestDatasetCpp")
      .def(py::init<>())
      .def("AddCovariates", &ForestDatasetCpp::AddCovariates)
      .def("AddBasis", &ForestDatasetCpp::AddBasis)
      .def("UpdateBasis", &ForestDatasetCpp::UpdateBasis)
      .def("AddVarianceWeights", &ForestDatasetCpp::AddVarianceWeights)
      .def("UpdateVarianceWeights", &ForestDatasetCpp::UpdateVarianceWeights)
      .def("NumRows", &ForestDatasetCpp::NumRows)
      .def("NumCovariates", &ForestDatasetCpp::NumCovariates)
      .def("NumBasis", &ForestDatasetCpp::NumBasis)
      .def("GetCovariates", &ForestDatasetCpp::GetCovariates)
      .def("GetBasis", &ForestDatasetCpp::GetBasis)
      .def("GetVarianceWeights", &ForestDatasetCpp::GetVarianceWeights)
      .def("HasBasis", &ForestDatasetCpp::HasBasis)
      .def("HasVarianceWeights", &ForestDatasetCpp::HasVarianceWeights)
      .def("AddAuxiliaryDimension", &ForestDatasetCpp::AddAuxiliaryDimension)
      .def("SetAuxiliaryDataValue", &ForestDatasetCpp::SetAuxiliaryDataValue)
      .def("GetAuxiliaryDataValue", &ForestDatasetCpp::GetAuxiliaryDataValue)
      .def("GetAuxiliaryDataVector", &ForestDatasetCpp::GetAuxiliaryDataVector);

  py::class_<ResidualCpp>(m, "ResidualCpp")
      .def(py::init<py::array_t<double, py::array::forcecast>, data_size_t>())
      .def("GetResidualArray", &ResidualCpp::GetResidualArray)
      .def("ReplaceData", &ResidualCpp::ReplaceData)
      .def("AddToData", &ResidualCpp::AddToData)
      .def("SubtractFromData", &ResidualCpp::SubtractFromData);

  py::class_<RngCpp>(m, "RngCpp")
      .def(py::init<int>());

  py::class_<BARTSamplesCpp>(m, "BARTSamplesCpp")
      .def(py::init<>())
      .def_static("from_json_string", &BARTSamplesCpp::FromJsonString)
      .def("to_json_string", &BARTSamplesCpp::ToJsonString)
      .def("load_from_json", &BARTSamplesCpp::LoadFromJson)
      .def("add_to_json", &BARTSamplesCpp::AddToJson)
      .def("merge", &BARTSamplesCpp::Merge)
      .def("num_samples", &BARTSamplesCpp::NumSamples)
      .def("y_bar", &BARTSamplesCpp::YBar)
      .def("y_std", &BARTSamplesCpp::YStd)
      .def("num_train", &BARTSamplesCpp::NumTrain)
      .def("num_test", &BARTSamplesCpp::NumTest)
      .def("global_var_samples", &BARTSamplesCpp::GlobalVarSamples)
      .def("leaf_scale_samples", &BARTSamplesCpp::LeafScaleSamples)
      .def("cloglog_cutpoint_samples", &BARTSamplesCpp::CloglogCutpointSamples)
      .def("mean_forest_predictions_train", &BARTSamplesCpp::MeanForestPredictionsTrain)
      .def("mean_forest_predictions_test", &BARTSamplesCpp::MeanForestPredictionsTest)
      .def("variance_forest_predictions_train", &BARTSamplesCpp::VarianceForestPredictionsTrain)
      .def("variance_forest_predictions_test", &BARTSamplesCpp::VarianceForestPredictionsTest)
      .def("rfx_predictions_train", &BARTSamplesCpp::RfxPredictionsTrain)
      .def("rfx_predictions_test", &BARTSamplesCpp::RfxPredictionsTest)
      .def("y_hat_train", &BARTSamplesCpp::YHatTrain)
      .def("y_hat_test", &BARTSamplesCpp::YHatTest)
      .def("has_mean_forest", &BARTSamplesCpp::HasMeanForest)
      .def("has_variance_forest", &BARTSamplesCpp::HasVarianceForest)
      .def("has_rfx", &BARTSamplesCpp::HasRfx)
      .def("has_global_var_samples", &BARTSamplesCpp::HasGlobalVarSamples)
      .def("has_leaf_scale_samples", &BARTSamplesCpp::HasLeafScaleSamples)
      .def("has_cloglog_cutpoint_samples", &BARTSamplesCpp::HasCloglogCutpointSamples)
      .def("has_mean_forest_predictions_train", &BARTSamplesCpp::HasMeanForestPredictionsTrain)
      .def("has_variance_forest_predictions_train", &BARTSamplesCpp::HasVarianceForestPredictionsTrain)
      .def("has_variance_forest_predictions_test", &BARTSamplesCpp::HasVarianceForestPredictionsTest)
      .def("materialize_mean_forest", &BARTSamplesCpp::MaterializeMeanForest)
      .def("materialize_variance_forest", &BARTSamplesCpp::MaterializeVarianceForest)
      .def("materialize_rfx_container", &BARTSamplesCpp::MaterializeRfxContainer)
      .def("materialize_rfx_label_mapper", &BARTSamplesCpp::MaterializeRfxLabelMapper);

  py::class_<BCFSamplesCpp>(m, "BCFSamplesCpp")
      .def(py::init<>())
      .def_static("from_json_string", &BCFSamplesCpp::FromJsonString)
      .def("to_json_string", &BCFSamplesCpp::ToJsonString)
      .def("load_from_json", &BCFSamplesCpp::LoadFromJson)
      .def("add_to_json", &BCFSamplesCpp::AddToJson)
      .def("merge", &BCFSamplesCpp::Merge)
      .def("num_samples", &BCFSamplesCpp::NumSamples)
      .def("y_bar", &BCFSamplesCpp::YBar)
      .def("y_std", &BCFSamplesCpp::YStd)
      .def("num_train", &BCFSamplesCpp::NumTrain)
      .def("num_test", &BCFSamplesCpp::NumTest)
      .def("treatment_dim", &BCFSamplesCpp::TreatmentDim)
      .def("global_var_samples", &BCFSamplesCpp::GlobalVarSamples)
      .def("leaf_scale_mu_samples", &BCFSamplesCpp::LeafScaleMuSamples)
      .def("leaf_scale_tau_samples", &BCFSamplesCpp::LeafScaleTauSamples)
      .def("tau_0_samples", &BCFSamplesCpp::Tau0Samples)
      .def("b0_samples", &BCFSamplesCpp::B0Samples)
      .def("b1_samples", &BCFSamplesCpp::B1Samples)
      .def("y_hat_train", &BCFSamplesCpp::YHatTrain)
      .def("y_hat_test", &BCFSamplesCpp::YHatTest)
      .def("mu_forest_predictions_train", &BCFSamplesCpp::MuForestPredictionsTrain)
      .def("mu_forest_predictions_test", &BCFSamplesCpp::MuForestPredictionsTest)
      .def("tau_forest_predictions_train", &BCFSamplesCpp::TauForestPredictionsTrain)
      .def("tau_forest_predictions_test", &BCFSamplesCpp::TauForestPredictionsTest)
      .def("variance_forest_predictions_train", &BCFSamplesCpp::VarianceForestPredictionsTrain)
      .def("variance_forest_predictions_test", &BCFSamplesCpp::VarianceForestPredictionsTest)
      .def("rfx_predictions_train", &BCFSamplesCpp::RfxPredictionsTrain)
      .def("rfx_predictions_test", &BCFSamplesCpp::RfxPredictionsTest)
      .def("has_mu_forest", &BCFSamplesCpp::HasMuForest)
      .def("has_tau_forest", &BCFSamplesCpp::HasTauForest)
      .def("has_variance_forest", &BCFSamplesCpp::HasVarianceForest)
      .def("has_rfx", &BCFSamplesCpp::HasRfx)
      .def("has_global_var_samples", &BCFSamplesCpp::HasGlobalVarSamples)
      .def("has_leaf_scale_mu_samples", &BCFSamplesCpp::HasLeafScaleMuSamples)
      .def("has_leaf_scale_tau_samples", &BCFSamplesCpp::HasLeafScaleTauSamples)
      .def("has_variance_forest_predictions_train", &BCFSamplesCpp::HasVarianceForestPredictionsTrain)
      .def("has_variance_forest_predictions_test", &BCFSamplesCpp::HasVarianceForestPredictionsTest)
      .def("has_yhat_test", &BCFSamplesCpp::HasYhatTest)
      .def("materialize_mu_forest", &BCFSamplesCpp::MaterializeMuForest)
      .def("materialize_tau_forest", &BCFSamplesCpp::MaterializeTauForest)
      .def("materialize_variance_forest", &BCFSamplesCpp::MaterializeVarianceForest)
      .def("materialize_rfx_container", &BCFSamplesCpp::MaterializeRfxContainer)
      .def("materialize_rfx_label_mapper", &BCFSamplesCpp::MaterializeRfxLabelMapper);

  py::class_<ForestContainerCpp>(m, "ForestContainerCpp")
      .def(py::init<int, int, bool, bool>())
      .def("CombineForests", &ForestContainerCpp::CombineForests)
      .def("AddToForest", &ForestContainerCpp::AddToForest)
      .def("MultiplyForest", &ForestContainerCpp::MultiplyForest)
      .def("OutputDimension", &ForestContainerCpp::OutputDimension)
      .def("NumTrees", &ForestContainerCpp::NumTrees)
      .def("NumSamples", &ForestContainerCpp::NumSamples)
      .def("DeleteSample", &ForestContainerCpp::DeleteSample)
      .def("Predict", &ForestContainerCpp::Predict)
      .def("PredictRaw", &ForestContainerCpp::PredictRaw)
      .def("PredictRawSingleForest", &ForestContainerCpp::PredictRawSingleForest)
      .def("SetRootValue", &ForestContainerCpp::SetRootValue)
      .def("SetRootVector", &ForestContainerCpp::SetRootVector)
      .def("AdjustResidual", &ForestContainerCpp::AdjustResidual)
      .def("SaveToJsonFile", &ForestContainerCpp::SaveToJsonFile)
      .def("LoadFromJsonFile", &ForestContainerCpp::LoadFromJsonFile)
      .def("LoadFromJson", &ForestContainerCpp::LoadFromJson)
      .def("AppendFromJson", &ForestContainerCpp::AppendFromJson)
      .def("DumpJsonString", &ForestContainerCpp::DumpJsonString)
      .def("LoadFromJsonString", &ForestContainerCpp::LoadFromJsonString)
      .def("AddSampleValue", &ForestContainerCpp::AddSampleValue)
      .def("AddSampleVector", &ForestContainerCpp::AddSampleVector)
      .def("AddNumericSplitValue", &ForestContainerCpp::AddNumericSplitValue)
      .def("AddNumericSplitVector", &ForestContainerCpp::AddNumericSplitVector)
      .def("GetTreeLeaves", &ForestContainerCpp::GetTreeLeaves)
      .def("GetTreeSplitCounts", &ForestContainerCpp::GetTreeSplitCounts)
      .def("GetForestSplitCounts", &ForestContainerCpp::GetForestSplitCounts)
      .def("GetOverallSplitCounts", &ForestContainerCpp::GetOverallSplitCounts)
      .def("GetGranularSplitCounts", &ForestContainerCpp::GetGranularSplitCounts)
      .def("NumLeavesForest", &ForestContainerCpp::NumLeavesForest)
      .def("SumLeafSquared", &ForestContainerCpp::SumLeafSquared)
      .def("IsLeafNode", &ForestContainerCpp::IsLeafNode)
      .def("IsNumericSplitNode", &ForestContainerCpp::IsNumericSplitNode)
      .def("IsCategoricalSplitNode", &ForestContainerCpp::IsCategoricalSplitNode)
      .def("ParentNode", &ForestContainerCpp::ParentNode)
      .def("LeftChildNode", &ForestContainerCpp::LeftChildNode)
      .def("RightChildNode", &ForestContainerCpp::RightChildNode)
      .def("SplitIndex", &ForestContainerCpp::SplitIndex)
      .def("NodeDepth", &ForestContainerCpp::NodeDepth)
      .def("SplitThreshold", &ForestContainerCpp::SplitThreshold)
      .def("SplitCategories", &ForestContainerCpp::SplitCategories)
      .def("NodeLeafValues", &ForestContainerCpp::NodeLeafValues)
      .def("NumNodes", &ForestContainerCpp::NumNodes)
      .def("NumLeaves", &ForestContainerCpp::NumLeaves)
      .def("NumLeafParents", &ForestContainerCpp::NumLeafParents)
      .def("NumSplitNodes", &ForestContainerCpp::NumSplitNodes)
      .def("Nodes", &ForestContainerCpp::Nodes)
      .def("Leaves", &ForestContainerCpp::Leaves);

  py::class_<ForestCpp>(m, "ForestCpp")
      .def(py::init<int, int, bool, bool>())
      .def("GetForestPtr", &ForestCpp::GetForestPtr)
      .def("MergeForest", &ForestCpp::MergeForest)
      .def("AddConstant", &ForestCpp::AddConstant)
      .def("MultiplyConstant", &ForestCpp::MultiplyConstant)
      .def("OutputDimension", &ForestCpp::OutputDimension)
      .def("NumTrees", &ForestCpp::NumTrees)
      .def("NumLeavesForest", &ForestCpp::NumLeavesForest)
      .def("SumLeafSquared", &ForestCpp::SumLeafSquared)
      .def("ResetRoot", &ForestCpp::ResetRoot)
      .def("Reset", &ForestCpp::Reset)
      .def("Predict", &ForestCpp::Predict)
      .def("PredictRaw", &ForestCpp::PredictRaw)
      .def("SetRootValue", &ForestCpp::SetRootValue)
      .def("SetRootVector", &ForestCpp::SetRootVector)
      .def("AdjustResidual", &ForestCpp::AdjustResidual)
      .def("AddNumericSplitValue", &ForestCpp::AddNumericSplitValue)
      .def("AddNumericSplitVector", &ForestCpp::AddNumericSplitVector)
      .def("GetEnsemble", &ForestCpp::GetEnsemble)
      .def("GetTreeLeaves", &ForestCpp::GetTreeLeaves)
      .def("GetTreeSplitCounts", &ForestCpp::GetTreeSplitCounts)
      .def("GetOverallSplitCounts", &ForestCpp::GetOverallSplitCounts)
      .def("GetGranularSplitCounts", &ForestCpp::GetGranularSplitCounts)
      .def("NumLeavesForest", &ForestCpp::NumLeavesForest)
      .def("SumLeafSquared", &ForestCpp::SumLeafSquared)
      .def("IsLeafNode", &ForestCpp::IsLeafNode)
      .def("IsNumericSplitNode", &ForestCpp::IsNumericSplitNode)
      .def("IsCategoricalSplitNode", &ForestCpp::IsCategoricalSplitNode)
      .def("ParentNode", &ForestCpp::ParentNode)
      .def("LeftChildNode", &ForestCpp::LeftChildNode)
      .def("RightChildNode", &ForestCpp::RightChildNode)
      .def("SplitIndex", &ForestCpp::SplitIndex)
      .def("NodeDepth", &ForestCpp::NodeDepth)
      .def("SplitThreshold", &ForestCpp::SplitThreshold)
      .def("SplitCategories", &ForestCpp::SplitCategories)
      .def("NodeLeafValues", &ForestCpp::NodeLeafValues)
      .def("NumNodes", &ForestCpp::NumNodes)
      .def("NumLeaves", &ForestCpp::NumLeaves)
      .def("NumLeafParents", &ForestCpp::NumLeafParents)
      .def("NumSplitNodes", &ForestCpp::NumSplitNodes)
      .def("Nodes", &ForestCpp::Nodes)
      .def("Leaves", &ForestCpp::Leaves);

  py::class_<ForestSamplerCpp>(m, "ForestSamplerCpp")
      .def(py::init<ForestDatasetCpp&, py::array_t<int>, int, data_size_t, double, double, int, int>())
      .def("ReconstituteTrackerFromForest", &ForestSamplerCpp::ReconstituteTrackerFromForest)
      .def("SampleOneIteration", &ForestSamplerCpp::SampleOneIteration)
      .def("InitializeForestModel", &ForestSamplerCpp::InitializeForestModel)
      .def("GetCachedForestPredictions", &ForestSamplerCpp::GetCachedForestPredictions)
      .def("PropagateBasisUpdate", &ForestSamplerCpp::PropagateBasisUpdate)
      .def("PropagateResidualUpdate", &ForestSamplerCpp::PropagateResidualUpdate)
      .def("UpdateAlpha", &ForestSamplerCpp::UpdateAlpha)
      .def("UpdateBeta", &ForestSamplerCpp::UpdateBeta)
      .def("UpdateMinSamplesLeaf", &ForestSamplerCpp::UpdateMinSamplesLeaf)
      .def("UpdateMaxDepth", &ForestSamplerCpp::UpdateMaxDepth)
      .def("GetAlpha", &ForestSamplerCpp::GetAlpha)
      .def("GetBeta", &ForestSamplerCpp::GetBeta)
      .def("GetMinSamplesLeaf", &ForestSamplerCpp::GetMinSamplesLeaf)
      .def("GetMaxDepth", &ForestSamplerCpp::GetMaxDepth);

  py::class_<RandomEffectsDatasetCpp>(m, "RandomEffectsDatasetCpp")
      .def(py::init<>())
      .def("GetDataset", &RandomEffectsDatasetCpp::GetDataset)
      .def("NumObservations", &RandomEffectsDatasetCpp::NumObservations)
      .def("NumBases", &RandomEffectsDatasetCpp::NumBases)
      .def("AddGroupLabels", &RandomEffectsDatasetCpp::AddGroupLabels)
      .def("AddBasis", &RandomEffectsDatasetCpp::AddBasis)
      .def("AddVarianceWeights", &RandomEffectsDatasetCpp::AddVarianceWeights)
      .def("UpdateGroupLabels", &RandomEffectsDatasetCpp::UpdateGroupLabels)
      .def("UpdateBasis", &RandomEffectsDatasetCpp::UpdateBasis)
      .def("UpdateVarianceWeights", &RandomEffectsDatasetCpp::UpdateVarianceWeights)
      .def("GetGroupLabels", &RandomEffectsDatasetCpp::GetGroupLabels)
      .def("GetBasis", &RandomEffectsDatasetCpp::GetBasis)
      .def("GetVarianceWeights", &RandomEffectsDatasetCpp::GetVarianceWeights)
      .def("HasGroupLabels", &RandomEffectsDatasetCpp::HasGroupLabels)
      .def("HasBasis", &RandomEffectsDatasetCpp::HasBasis)
      .def("HasVarianceWeights", &RandomEffectsDatasetCpp::HasVarianceWeights);

  py::class_<RandomEffectsContainerCpp>(m, "RandomEffectsContainerCpp")
      .def(py::init<>())
      .def("SetComponentsAndGroups", &RandomEffectsContainerCpp::SetComponentsAndGroups)
      .def("AddSample", &RandomEffectsContainerCpp::AddSample)
      .def("NumSamples", &RandomEffectsContainerCpp::NumSamples)
      .def("NumComponents", &RandomEffectsContainerCpp::NumComponents)
      .def("NumGroups", &RandomEffectsContainerCpp::NumGroups)
      .def("GetBeta", &RandomEffectsContainerCpp::GetBeta)
      .def("GetXi", &RandomEffectsContainerCpp::GetXi)
      .def("GetAlpha", &RandomEffectsContainerCpp::GetAlpha)
      .def("GetSigma", &RandomEffectsContainerCpp::GetSigma)
      .def("DeleteSample", &RandomEffectsContainerCpp::DeleteSample)
      .def("Predict", &RandomEffectsContainerCpp::Predict)
      .def("SaveToJsonFile", &RandomEffectsContainerCpp::SaveToJsonFile)
      .def("LoadFromJsonFile", &RandomEffectsContainerCpp::LoadFromJsonFile)
      .def("DumpJsonString", &RandomEffectsContainerCpp::DumpJsonString)
      .def("LoadFromJsonString", &RandomEffectsContainerCpp::LoadFromJsonString)
      .def("LoadFromJson", &RandomEffectsContainerCpp::LoadFromJson)
      .def("AppendFromJson", &RandomEffectsContainerCpp::AppendFromJson)
      .def("GetRandomEffectsContainer", &RandomEffectsContainerCpp::GetRandomEffectsContainer);

  py::class_<RandomEffectsTrackerCpp>(m, "RandomEffectsTrackerCpp")
      .def(py::init<py::array_t<int>>())
      .def("GetUniqueGroupIds", &RandomEffectsTrackerCpp::GetUniqueGroupIds)
      .def("GetTracker", &RandomEffectsTrackerCpp::GetTracker)
      .def("Reset", &RandomEffectsTrackerCpp::Reset)
      .def("RootReset", &RandomEffectsTrackerCpp::RootReset);

  py::class_<RandomEffectsLabelMapperCpp>(m, "RandomEffectsLabelMapperCpp")
      .def(py::init<>())
      .def("LoadFromTracker", &RandomEffectsLabelMapperCpp::LoadFromTracker)
      .def("SaveToJsonFile", &RandomEffectsLabelMapperCpp::SaveToJsonFile)
      .def("LoadFromJsonFile", &RandomEffectsLabelMapperCpp::LoadFromJsonFile)
      .def("DumpJsonString", &RandomEffectsLabelMapperCpp::DumpJsonString)
      .def("LoadFromJsonString", &RandomEffectsLabelMapperCpp::LoadFromJsonString)
      .def("LoadFromJson", &RandomEffectsLabelMapperCpp::LoadFromJson)
      .def("GetLabelMapper", &RandomEffectsLabelMapperCpp::GetLabelMapper)
      .def("MapGroupIdToArrayIndex", &RandomEffectsLabelMapperCpp::MapGroupIdToArrayIndex)
      .def("MapMultipleGroupIdsToArrayIndices", &RandomEffectsLabelMapperCpp::MapMultipleGroupIdsToArrayIndices)
      .def("GetUniqueGroupIds", &RandomEffectsLabelMapperCpp::GetUniqueGroupIds);

  py::class_<RandomEffectsModelCpp>(m, "RandomEffectsModelCpp")
      .def(py::init<int, int>())
      .def("GetModel", &RandomEffectsModelCpp::GetModel)
      .def("SampleRandomEffects", &RandomEffectsModelCpp::SampleRandomEffects)
      .def("Predict", &RandomEffectsModelCpp::Predict)
      .def("SetWorkingParameter", &RandomEffectsModelCpp::SetWorkingParameter)
      .def("SetGroupParameters", &RandomEffectsModelCpp::SetGroupParameters)
      .def("SetWorkingParameterCovariance", &RandomEffectsModelCpp::SetWorkingParameterCovariance)
      .def("SetGroupParameterCovariance", &RandomEffectsModelCpp::SetGroupParameterCovariance)
      .def("SetVariancePriorShape", &RandomEffectsModelCpp::SetVariancePriorShape)
      .def("SetVariancePriorScale", &RandomEffectsModelCpp::SetVariancePriorScale)
      .def("Reset", &RandomEffectsModelCpp::Reset);

  py::class_<GlobalVarianceModelCpp>(m, "GlobalVarianceModelCpp")
      .def(py::init<>())
      .def("SampleOneIteration", &GlobalVarianceModelCpp::SampleOneIteration);

  py::class_<LeafVarianceModelCpp>(m, "LeafVarianceModelCpp")
      .def(py::init<>())
      .def("SampleOneIteration", &LeafVarianceModelCpp::SampleOneIteration);

  py::class_<OrdinalSamplerCpp>(m, "OrdinalSamplerCpp")
      .def(py::init<>())
      .def("UpdateLatentVariables", &OrdinalSamplerCpp::UpdateLatentVariables)
      .def("UpdateGammaParams", &OrdinalSamplerCpp::UpdateGammaParams)
      .def("UpdateCumulativeExpSums", &OrdinalSamplerCpp::UpdateCumulativeExpSums);
  ;

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}