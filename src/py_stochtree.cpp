#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <nlohmann/json.hpp>
#include <stochtree/container.h>
#include <stochtree/data.h>
#include <stochtree/leaf_model.h>
#include <stochtree/meta.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/random_effects.h>
#include <stochtree/tree_sampler.h>
#include <stochtree/variance_model.h>
#include <functional>
#include <memory>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
using data_size_t = StochTree::data_size_t;

enum ForestLeafModel {
    kConstant, 
    kUnivariateRegression, 
    kMultivariateRegression
};

class ForestSamplerCpp;

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

  void UpdateVarianceWeights(py::array_t<double> weight_vector, data_size_t num_row, bool exponentiate) {
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
        accessor(i,j) = dataset_->CovariateValue(i,j);
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
        accessor(i,j) = dataset_->BasisValue(i,j);
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

  py::array_t<double> GetResidualArray() {
    // Obtain a reference to the underlying Eigen::VectorXd
    Eigen::VectorXd& resid_vector = residual_->GetData();
    
    // Initialize n x 1 numpy array to store the residual
    data_size_t n = residual_->NumRows();
    auto result = py::array_t<double>(py::detail::any_container<py::ssize_t>({n, 1}));
    auto accessor = result.mutable_unchecked<2>();
    for (size_t i = 0; i < n; i++) {
      accessor(i,0) = resid_vector(i);
    }

    return result;
  }

  void ReplaceData(py::array_t<double> new_vector, data_size_t num_row) {
    // Extract pointer to contiguous block of memory
    double* data_ptr = static_cast<double*>(new_vector.mutable_data());
    // Overwrite data in residual_
    residual_->OverwriteData(data_ptr, num_row);
  }

  void AddToData(py::array_t<double> update_vector, data_size_t num_row) {
    // Extract pointer to contiguous block of memory
    double* data_ptr = static_cast<double*>(update_vector.mutable_data());
    // Add to data in residual_
    residual_->AddToData(data_ptr, num_row);
  }

  void SubtractFromData(py::array_t<double> update_vector, data_size_t num_row) {
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
  ~ForestContainerCpp() {}

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
    auto result = py::array_t<double>(py::detail::any_container<py::ssize_t>({n, num_samples, output_dim}));
    auto accessor = result.mutable_unchecked<3>();
    // py::buffer_info buf = result.request();
    // double *ptr = static_cast<double *>(buf.ptr);
    for (size_t i = 0; i < n; i++) {
      for (int j = 0; j < output_dim; j++) {
        for (int k = 0; k < num_samples; k++) {
          accessor(i,k,j) = output_raw[k*(output_dim*n) + i*output_dim + j];
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
        accessor(i,j) = output_raw[i*output_dim + j];
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

  void AddSampleVector(py::array_t<double> leaf_vector) {
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
                             double split_threshold, py::array_t<double> left_leaf_vector, 
                             py::array_t<double> right_leaf_vector) {
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
    auto result = py::array_t<int>(py::detail::any_container<py::ssize_t>({num_samples,num_trees,num_features}));
    auto accessor = result.mutable_unchecked<3>();
    for (int i = 0; i < num_samples; i++) {
      for (int j = 0; j < num_trees; j++) {
        for (int k = 0; k < num_features; k++) {
          accessor(i,j,k) = 0;
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
          accessor(i,j,split_feature)++;
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

  StochTree::TreeEnsemble* GetForestPtr() {return forest_.get();}

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
          accessor(i,j) = output_raw[i*output_dim + j];
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
                             py::array_t<double> left_leaf_vector, py::array_t<double> right_leaf_vector) {
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
    auto result = py::array_t<int>(py::detail::any_container<py::ssize_t>({num_trees,num_features}));
    auto accessor = result.mutable_unchecked<2>();
    for (int i = 0; i < num_trees; i++) {
      for (int j = 0; j < num_features; j++) {
        accessor(i,j) = 0;
      }
    }
    for (int i = 0; i < num_trees; i++) {
      StochTree::Tree* tree = forest_->GetTree(i);
      std::vector<int32_t> split_nodes = tree->GetInternalNodes();
      for (int j = 0; j < split_nodes.size(); j++) {
        auto node_id = split_nodes.at(i);
        auto split_feature = tree->SplitIndex(node_id);
        accessor(i,split_feature)++;
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

  StochTree::ForestTracker* GetTracker() {return tracker_.get();}

  void ReconstituteTrackerFromForest(ForestCpp& forest, ForestDatasetCpp& dataset, ResidualCpp& residual, bool is_mean_model) {
    // Extract raw pointer to the forest and dataset
    StochTree::TreeEnsemble* forest_ptr = forest.GetEnsemble();
    StochTree::ForestDataset* data_ptr = dataset.GetDataset();
    StochTree::ColumnVector* residual_ptr = residual.GetData();
    
    // Reset forest tracker using the forest held at index forest_num
    tracker_->ReconstituteFromForest(*forest_ptr, *data_ptr, *residual_ptr, is_mean_model);
  }

  void SampleOneIteration(ForestContainerCpp& forest_samples, ForestCpp& forest, ForestDatasetCpp& dataset, ResidualCpp& residual, RngCpp& rng, 
                          py::array_t<int> feature_types, py::array_t<int> sweep_update_indices, int cutpoint_grid_size, py::array_t<double> leaf_model_scale_input, 
                          py::array_t<double> variable_weights, double a_forest, double b_forest, double global_variance, 
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
    if (leaf_model_int == 0) model_type = StochTree::ModelType::kConstantLeafGaussian;
    else if (leaf_model_int == 1) model_type = StochTree::ModelType::kUnivariateRegressionLeafGaussian;
    else if (leaf_model_int == 2) model_type = StochTree::ModelType::kMultivariateRegressionLeafGaussian;
    else if (leaf_model_int == 3) model_type = StochTree::ModelType::kLogLinearVariance;

    // Unpack leaf model parameters
    double leaf_scale;
    Eigen::MatrixXd leaf_scale_matrix;
    if ((model_type == StochTree::ModelType::kConstantLeafGaussian) || 
        (model_type == StochTree::ModelType::kUnivariateRegressionLeafGaussian)) {
        leaf_scale = leaf_model_scale_input.at(0,0);
    } else if (model_type == StochTree::ModelType::kMultivariateRegressionLeafGaussian) {
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
      }
    }
  }

  void InitializeForestModel(ForestDatasetCpp& dataset, ResidualCpp& residual, ForestCpp& forest, 
                             int leaf_model_int, py::array_t<double> initial_values) {
    // Convert leaf model type to enum
    StochTree::ModelType model_type;
    if (leaf_model_int == 0) model_type = StochTree::ModelType::kConstantLeafGaussian;
    else if (leaf_model_int == 1) model_type = StochTree::ModelType::kUnivariateRegressionLeafGaussian;
    else if (leaf_model_int == 2) model_type = StochTree::ModelType::kMultivariateRegressionLeafGaussian;
    else if (leaf_model_int == 3) model_type = StochTree::ModelType::kLogLinearVariance;
    else StochTree::Log::Fatal("Invalid model type");
    
    // Unpack initial value
    StochTree::TreeEnsemble* forest_ptr = forest.GetEnsemble();
    StochTree::ForestDataset* forest_data_ptr = dataset.GetDataset();
    StochTree::ColumnVector* residual_data_ptr = residual.GetData();
    int num_trees = forest_ptr->NumTrees();
    double init_val;
    std::vector<double> init_value_vector;
    if ((model_type == StochTree::ModelType::kConstantLeafGaussian) || 
        (model_type == StochTree::ModelType::kUnivariateRegressionLeafGaussian) || 
        (model_type == StochTree::ModelType::kLogLinearVariance)) {
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
  void AddBasis(py::array_t<double> basis, data_size_t num_row, int num_col, bool row_major) {
    double* basis_data_ptr = static_cast<double*>(basis.mutable_data());
    rfx_dataset_->AddBasis(basis_data_ptr, num_row, num_col, row_major);
  }
  void AddVarianceWeights(py::array_t<double> weights, data_size_t num_row) {
    double* weight_data_ptr = static_cast<double*>(weights.mutable_data());
    rfx_dataset_->AddVarianceWeights(weight_data_ptr, num_row);
  }
  void UpdateBasis(py::array_t<double> basis, data_size_t num_row, int num_col, bool row_major) {
    double* basis_data_ptr = static_cast<double*>(basis.mutable_data());
    rfx_dataset_->UpdateBasis(basis_data_ptr, num_row, num_col, row_major);
  }
  void UpdateVarianceWeights(py::array_t<double> weights, data_size_t num_row, bool exponentiate) {
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
        accessor(i,j) = rfx_dataset_->BasisValue(i,j);
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
  bool HasGroupLabels() {return rfx_dataset_->HasGroupLabels();}
  bool HasBasis() {return rfx_dataset_->HasBasis();}
  bool HasVarianceWeights() {return rfx_dataset_->HasVarWeights();}

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
  ~RandomEffectsContainerCpp() {}
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
          accessor(i,j,k) = beta_raw[k*num_groups*num_components + j*num_components + i];
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
          accessor(i,j,k) = xi_raw[k*num_groups*num_components + j*num_components + i];
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
        accessor(i,j) = alpha_raw[j*num_components + i];
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
        accessor(i,j) = sigma_raw[j*num_components + i];
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
  ~RandomEffectsLabelMapperCpp() {}
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
        group_params_eigen(i,j) = group_params.at(i,j);
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
        working_param_cov_eigen(i,j) = working_param_cov.at(i,j);
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
        group_param_cov_eigen(i,j) = group_param_cov.at(i,j);
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

  std::string AddForest(ForestContainerCpp& forest_samples) {
    int forest_num = json_->at("num_forests");
    std::string forest_label = "forest_" + std::to_string(forest_num);
    nlohmann::json forest_json = forest_samples.ToJson();
    json_->at("forests").emplace(forest_label, forest_json);
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

 private:
  std::unique_ptr<nlohmann::json> json_;
};

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
  auto result = py::array_t<int, py::array::f_style>(py::detail::any_container<py::ssize_t>({num_obs*num_trees, num_samples}));
  int* output_data_ptr = static_cast<int*>(result.mutable_data());
  Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> output_eigen(output_data_ptr, num_obs*num_trees, num_samples);
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
  if (add) op = std::plus<double>();
  else op = std::minus<double>();
  
  // Perform the update (addition / subtraction) operation
  StochTree::UpdateResidualEntireForest(*(sampler.GetTracker()), *(dataset.GetDataset()), *(residual.GetData()), forest_samples_->GetEnsemble(forest_num), requires_basis, op);
}

void ForestCpp::AdjustResidual(ForestDatasetCpp& dataset, ResidualCpp& residual, ForestSamplerCpp& sampler, bool requires_basis, bool add) {
  // Determine whether or not we are adding forest predictions to the residuals
  std::function<double(double, double)> op;
  if (add) op = std::plus<double>();
  else op = std::minus<double>();
  
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
  std::vector<double> output(num_observations*num_samples);
  rfx_container_->Predict(*rfx_dataset.GetDataset(), *label_mapper.GetLabelMapper(), output);
  auto result = py::array_t<double>(py::detail::any_container<py::ssize_t>({num_observations, num_samples}));
  auto accessor = result.mutable_unchecked<2>();
  for (size_t i = 0; i < num_observations; i++) {
    for (int j = 0; j < num_samples; j++) {
      accessor(i, j) = output.at(j*num_observations + i);
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
    .def("AddForest", &JsonCpp::AddForest)
    .def("AddRandomEffectsContainer", &JsonCpp::AddRandomEffectsContainer)
    .def("AddRandomEffectsLabelMapper", &JsonCpp::AddRandomEffectsLabelMapper)
    .def("AddRandomEffectsGroupIDs", &JsonCpp::AddRandomEffectsGroupIDs)
    .def("ContainsField", &JsonCpp::ContainsField)
    .def("ContainsFieldSubfolder", &JsonCpp::ContainsFieldSubfolder)
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
    .def("HasVarianceWeights", &ForestDatasetCpp::HasVarianceWeights);

  py::class_<ResidualCpp>(m, "ResidualCpp")
    .def(py::init<py::array_t<double>,data_size_t>())
    .def("GetResidualArray", &ResidualCpp::GetResidualArray)
    .def("ReplaceData", &ResidualCpp::ReplaceData)
    .def("AddToData", &ResidualCpp::AddToData)
    .def("SubtractFromData", &ResidualCpp::SubtractFromData);

  py::class_<RngCpp>(m, "RngCpp")
    .def(py::init<int>());
  
  py::class_<ForestContainerCpp>(m, "ForestContainerCpp")
    .def(py::init<int,int,bool,bool>())
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
    .def(py::init<int,int,bool,bool>())
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
    .def("MapMultipleGroupIdsToArrayIndices", &RandomEffectsLabelMapperCpp::MapMultipleGroupIdsToArrayIndices);

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

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}