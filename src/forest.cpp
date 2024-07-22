#include <cpp11.hpp>
#include "stochtree_types.h"
#include <stochtree/container.h>
#include <stochtree/data.h>
#include <stochtree/leaf_model.h>
#include <stochtree/meta.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/tree_sampler.h>
#include <stochtree/variance_model.h>
#include <functional>
#include <memory>
#include <vector>

[[cpp11::register]]
cpp11::external_pointer<StochTree::ForestContainer> forest_container_cpp(int num_trees, int output_dimension = 1, bool is_leaf_constant = true) {
    // Create smart pointer to newly allocated object
    std::unique_ptr<StochTree::ForestContainer> forest_sample_ptr_ = std::make_unique<StochTree::ForestContainer>(num_trees, output_dimension, is_leaf_constant);
    
    // Release management of the pointer to R session
    return cpp11::external_pointer<StochTree::ForestContainer>(forest_sample_ptr_.release());
}

[[cpp11::register]]
cpp11::external_pointer<StochTree::ForestContainer> forest_container_from_json_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string forest_label) {
    // Create smart pointer to newly allocated object
    std::unique_ptr<StochTree::ForestContainer> forest_sample_ptr_ = std::make_unique<StochTree::ForestContainer>(0, 1, true);
    
    // Extract the forest's json
    nlohmann::json forest_json = json_ptr->at("forests").at(forest_label);
    
    // Reset the forest sample container using the json
    forest_sample_ptr_->Reset();
    forest_sample_ptr_->from_json(forest_json);
    
    // Release management of the pointer to R session
    return cpp11::external_pointer<StochTree::ForestContainer>(forest_sample_ptr_.release());
}

[[cpp11::register]]
int num_samples_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples) {
    return forest_samples->NumSamples();
}

[[cpp11::register]]
int num_trees_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples) {
    return forest_samples->NumTrees();
}

[[cpp11::register]]
void json_save_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, std::string json_filename) {
    forest_samples->SaveToJsonFile(json_filename);
}

[[cpp11::register]]
void json_load_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, std::string json_filename) {
    forest_samples->LoadFromJsonFile(json_filename);
}

[[cpp11::register]]
int output_dimension_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples) {
    return forest_samples->OutputDimension();
}

[[cpp11::register]]
int is_leaf_constant_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples) {
    return forest_samples->IsLeafConstant();
}

[[cpp11::register]]
bool all_roots_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, int forest_num) {
    return forest_samples->AllRoots(forest_num);
}

[[cpp11::register]]
void add_sample_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples) {
    forest_samples->AddSamples(1);
}

[[cpp11::register]]
void set_leaf_value_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, double leaf_value) {
    forest_samples->InitializeRoot(leaf_value);
}

[[cpp11::register]]
void set_leaf_vector_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, cpp11::doubles leaf_vector) {
    std::vector<double> leaf_vector_converted(leaf_vector.size());
    for (int i = 0; i < leaf_vector.size(); i++) {
        leaf_vector_converted[i] = leaf_vector[i];
    }
    forest_samples->InitializeRoot(leaf_vector_converted);
}

[[cpp11::register]]
void adjust_residual_forest_container_cpp(cpp11::external_pointer<StochTree::ForestDataset> data, 
                                          cpp11::external_pointer<StochTree::ColumnVector> residual, 
                                          cpp11::external_pointer<StochTree::ForestContainer> forest_samples, 
                                          cpp11::external_pointer<StochTree::ForestTracker> tracker, 
                                          bool requires_basis, int forest_num, bool add) {
    // Determine whether or not we are adding forest_num to the residuals
    std::function<double(double, double)> op;
    if (add) op = std::plus<double>();
    else op = std::minus<double>();
    
    // Perform the update (addition / subtraction) operation
    StochTree::UpdateResidualEntireForest(*tracker, *data, *residual, forest_samples->GetEnsemble(forest_num), requires_basis, op);
}

[[cpp11::register]]
void update_residual_forest_container_cpp(cpp11::external_pointer<StochTree::ForestDataset> data, 
                                          cpp11::external_pointer<StochTree::ColumnVector> residual, 
                                          cpp11::external_pointer<StochTree::ForestContainer> forest_samples, 
                                          cpp11::external_pointer<StochTree::ForestTracker> tracker, 
                                          int forest_num) {
    // Perform the update (addition / subtraction) operation
    StochTree::UpdateResidualNewBasis(*tracker, *data, *residual, forest_samples->GetEnsemble(forest_num));
}

[[cpp11::register]]
cpp11::writable::doubles_matrix<> predict_forest_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, cpp11::external_pointer<StochTree::ForestDataset> dataset) {
    // Predict from the sampled forests
    std::vector<double> output_raw = forest_samples->Predict(*dataset);
    
    // Convert result to a matrix
    int n = dataset->GetCovariates().rows();
    int num_samples = forest_samples->NumSamples();
    cpp11::writable::doubles_matrix<> output(n, num_samples);
    for (size_t i = 0; i < n; i++) {
        for (int j = 0; j < num_samples; j++) {
            output(i, j) = output_raw[n*j + i];
        }
    }
    
    return output;
}

[[cpp11::register]]
cpp11::writable::doubles_matrix<> predict_forest_raw_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, cpp11::external_pointer<StochTree::ForestDataset> dataset) {
    // Predict from the sampled forests
    std::vector<double> output_raw = forest_samples->PredictRaw(*dataset);
    
    // Convert result to a matrix
    int n = dataset->GetCovariates().rows();
    int num_samples = forest_samples->NumSamples();
    int output_dimension = forest_samples->OutputDimension();
    int num_rows = n * output_dimension;
    cpp11::writable::doubles_matrix<> output(num_rows, num_samples);
    for (size_t i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_samples; j++) {
            output(i, j) = output_raw[num_rows*j + i];
        }
    }
    
    return output;
}

[[cpp11::register]]
cpp11::writable::doubles_matrix<> predict_forest_raw_single_forest_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, cpp11::external_pointer<StochTree::ForestDataset> dataset, int forest_num) {
    // Predict from the sampled forests
    std::vector<double> output_raw = forest_samples->PredictRaw(*dataset, forest_num);
    
    // Convert result to a matrix
    int n = dataset->GetCovariates().rows();
    int output_dimension = forest_samples->OutputDimension();
    cpp11::writable::doubles_matrix<> output(n, output_dimension);
    for (size_t i = 0; i < n; i++) {
        for (int j = 0; j < output_dimension; j++) {
            output(i, j) = output_raw[i*output_dimension + j];
        }
    }
    
    return output;
}
