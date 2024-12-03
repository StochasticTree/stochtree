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
cpp11::external_pointer<StochTree::TreeEnsemble> active_forest_cpp(int num_trees, int output_dimension = 1, bool is_leaf_constant = true, bool is_exponentiated = false) {
    // Create smart pointer to newly allocated object
    std::unique_ptr<StochTree::TreeEnsemble> forest_ptr_ = std::make_unique<StochTree::TreeEnsemble>(num_trees, output_dimension, is_leaf_constant, is_exponentiated);
    
    // Release management of the pointer to R session
    return cpp11::external_pointer<StochTree::TreeEnsemble>(forest_ptr_.release());
}

[[cpp11::register]]
cpp11::external_pointer<StochTree::ForestContainer> forest_container_cpp(int num_trees, int output_dimension = 1, bool is_leaf_constant = true, bool is_exponentiated = false) {
    // Create smart pointer to newly allocated object
    std::unique_ptr<StochTree::ForestContainer> forest_sample_ptr_ = std::make_unique<StochTree::ForestContainer>(num_trees, output_dimension, is_leaf_constant, is_exponentiated);
    
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
void forest_container_append_from_json_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_sample_ptr, cpp11::external_pointer<nlohmann::json> json_ptr, std::string forest_label) {
    // Extract the forest's json
    nlohmann::json forest_json = json_ptr->at("forests").at(forest_label);
    
    // Append to the forest sample container using the json
    forest_sample_ptr->append_from_json(forest_json);
}

[[cpp11::register]]
cpp11::external_pointer<StochTree::ForestContainer> forest_container_from_json_string_cpp(std::string json_string, std::string forest_label) {
    // Create smart pointer to newly allocated object
    std::unique_ptr<StochTree::ForestContainer> forest_sample_ptr_ = std::make_unique<StochTree::ForestContainer>(0, 1, true);
    
    // Create a nlohmann::json object from the string
    nlohmann::json json_object = nlohmann::json::parse(json_string);
    
    // Extract the forest's json
    nlohmann::json forest_json = json_object.at("forests").at(forest_label);
    
    // Reset the forest sample container using the json
    forest_sample_ptr_->Reset();
    forest_sample_ptr_->from_json(forest_json);
    
    // Release management of the pointer to R session
    return cpp11::external_pointer<StochTree::ForestContainer>(forest_sample_ptr_.release());
}

[[cpp11::register]]
void forest_container_append_from_json_string_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_sample_ptr, std::string json_string, std::string forest_label) {
    // Create a nlohmann::json object from the string
    nlohmann::json json_object = nlohmann::json::parse(json_string);
    
    // Extract the forest's json
    nlohmann::json forest_json = json_object.at("forests").at(forest_label);
    
    // Append to the forest sample container using the json
    forest_sample_ptr->append_from_json(forest_json);
}

[[cpp11::register]]
int num_samples_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples) {
    return forest_samples->NumSamples();
}

[[cpp11::register]]
int ensemble_tree_max_depth_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, int ensemble_num, int tree_num) {
    return forest_samples->EnsembleTreeMaxDepth(ensemble_num, tree_num);
}

[[cpp11::register]]
double ensemble_average_max_depth_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, int ensemble_num) {
    return forest_samples->EnsembleAverageMaxDepth(ensemble_num);
}

[[cpp11::register]]
double average_max_depth_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples) {
    return forest_samples->AverageMaxDepth();
}

[[cpp11::register]]
int num_leaves_ensemble_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, int forest_num) {
    StochTree::TreeEnsemble* forest = forest_samples->GetEnsemble(forest_num);
    return forest->NumLeaves();
}

[[cpp11::register]]
double sum_leaves_squared_ensemble_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, int forest_num) {
    StochTree::TreeEnsemble* forest = forest_samples->GetEnsemble(forest_num);
    return forest->SumLeafSquared();
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
int is_exponentiated_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples) {
    return forest_samples->IsExponentiated();
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
void add_sample_value_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, double leaf_value) {
    if (forest_samples->OutputDimension() != 1) {
        cpp11::stop("leaf_value must match forest leaf dimension");
    }
    int num_samples = forest_samples->NumSamples();
    forest_samples->AddSamples(1);
    StochTree::TreeEnsemble* ensemble = forest_samples->GetEnsemble(num_samples);
    int num_trees = ensemble->NumTrees();
    for (int i = 0; i < num_trees; i++) {
        StochTree::Tree* tree = ensemble->GetTree(i);
        tree->SetLeaf(0, leaf_value);
    }
}

[[cpp11::register]]
void add_sample_vector_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, cpp11::doubles leaf_vector) {
    if (forest_samples->OutputDimension() != leaf_vector.size()) {
        cpp11::stop("leaf_vector must match forest leaf dimension");
    }
    int num_samples = forest_samples->NumSamples();
    forest_samples->AddSamples(1);
    StochTree::TreeEnsemble* ensemble = forest_samples->GetEnsemble(num_samples);
    int num_trees = ensemble->NumTrees();
    std::vector<double> leaf_vector_cast(leaf_vector.begin(), leaf_vector.end());
    for (int i = 0; i < num_trees; i++) {
        StochTree::Tree* tree = ensemble->GetTree(i);
        tree->SetLeafVector(0, leaf_vector_cast);
    }
}

[[cpp11::register]]
void add_numeric_split_tree_value_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, int forest_num, int tree_num, int leaf_num, int feature_num, double split_threshold, double left_leaf_value, double right_leaf_value) {
    if (forest_samples->OutputDimension() != 1) {
        cpp11::stop("leaf_vector must match forest leaf dimension");
    }
    StochTree::TreeEnsemble* ensemble = forest_samples->GetEnsemble(forest_num);
    StochTree::Tree* tree = ensemble->GetTree(tree_num);
    if (!tree->IsLeaf(leaf_num)) {
        cpp11::stop("leaf_num is not a leaf");
    }
    tree->ExpandNode(leaf_num, feature_num, split_threshold, left_leaf_value, right_leaf_value);
}

[[cpp11::register]]
void add_numeric_split_tree_vector_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, int forest_num, int tree_num, int leaf_num, int feature_num, double split_threshold, cpp11::doubles left_leaf_vector, cpp11::doubles right_leaf_vector) {
    if (forest_samples->OutputDimension() != left_leaf_vector.size()) {
        cpp11::stop("left_leaf_vector must match forest leaf dimension");
    }
    if (forest_samples->OutputDimension() != right_leaf_vector.size()) {
        cpp11::stop("right_leaf_vector must match forest leaf dimension");
    }
    StochTree::TreeEnsemble* ensemble = forest_samples->GetEnsemble(forest_num);
    std::vector<double> left_leaf_vector_cast(left_leaf_vector.begin(), left_leaf_vector.end());
    std::vector<double> right_leaf_vector_cast(right_leaf_vector.begin(), right_leaf_vector.end());
    StochTree::Tree* tree = ensemble->GetTree(tree_num);
    if (!tree->IsLeaf(leaf_num)) {
        cpp11::stop("leaf_num is not a leaf");
    }
    tree->ExpandNode(leaf_num, feature_num, split_threshold, left_leaf_vector_cast, right_leaf_vector_cast);
}

[[cpp11::register]]
cpp11::writable::integers get_tree_leaves_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, int forest_num, int tree_num) {
    StochTree::TreeEnsemble* ensemble = forest_samples->GetEnsemble(forest_num);
    StochTree::Tree* tree = ensemble->GetTree(tree_num);
    std::vector<int32_t> leaves_raw = tree->GetLeaves();
    cpp11::writable::integers leaves(leaves_raw.begin(), leaves_raw.end());
    return leaves;
}

[[cpp11::register]]
cpp11::writable::integers get_tree_split_counts_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, int forest_num, int tree_num, int num_features) {
    cpp11::writable::integers output(num_features);
    for (int i = 0; i < output.size(); i++) output.at(i) = 0;
    StochTree::TreeEnsemble* ensemble = forest_samples->GetEnsemble(forest_num);
    StochTree::Tree* tree = ensemble->GetTree(tree_num);
    std::vector<int32_t> split_nodes = tree->GetInternalNodes();
    for (int i = 0; i < split_nodes.size(); i++) {
        auto node_id = split_nodes.at(i);
        auto split_feature = tree->SplitIndex(node_id);
        output.at(split_feature)++;
    }
    return output;
}

[[cpp11::register]]
cpp11::writable::integers get_forest_split_counts_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, int forest_num, int num_features) {
    cpp11::writable::integers output(num_features);
    for (int i = 0; i < output.size(); i++) output.at(i) = 0;
    StochTree::TreeEnsemble* ensemble = forest_samples->GetEnsemble(forest_num);
    int num_trees = ensemble->NumTrees();
    for (int i = 0; i < num_trees; i++) {
        StochTree::Tree* tree = ensemble->GetTree(i);
        std::vector<int32_t> split_nodes = tree->GetInternalNodes();
        for (int j = 0; j < split_nodes.size(); j++) {
            auto node_id = split_nodes.at(j);
            auto split_feature = tree->SplitIndex(node_id);
            output.at(split_feature)++;
        }
    }
    return output;
}

[[cpp11::register]]
cpp11::writable::integers get_overall_split_counts_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, int num_features) {
    cpp11::writable::integers output(num_features);
    for (int i = 0; i < output.size(); i++) output.at(i) = 0;
    int num_samples = forest_samples->NumSamples();
    int num_trees = forest_samples->NumTrees();
    for (int i = 0; i < num_samples; i++) {
        StochTree::TreeEnsemble* ensemble = forest_samples->GetEnsemble(i);
        for (int j = 0; j < num_trees; j++) {
            StochTree::Tree* tree = ensemble->GetTree(j);
            std::vector<int32_t> split_nodes = tree->GetInternalNodes();
            for (int k = 0; k < split_nodes.size(); k++) {
                auto node_id = split_nodes.at(k);
                auto split_feature = tree->SplitIndex(node_id);
                output.at(split_feature)++;
            }
        }
    }
    return output;
}

[[cpp11::register]]
cpp11::writable::integers get_granular_split_count_array_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, int num_features) {
    int num_samples = forest_samples->NumSamples();
    int num_trees = forest_samples->NumTrees();
    cpp11::writable::integers output(num_features*num_samples*num_trees);
    for (int elem = 0; elem < output.size(); elem++) output.at(elem) = 0;
    for (int i = 0; i < num_samples; i++) {
        StochTree::TreeEnsemble* ensemble = forest_samples->GetEnsemble(i);
        for (int j = 0; j < num_trees; j++) {
            StochTree::Tree* tree = ensemble->GetTree(j);
            std::vector<int32_t> split_nodes = tree->GetInternalNodes();
            for (int k = 0; k < split_nodes.size(); k++) {
                auto node_id = split_nodes.at(k);
                auto split_feature = tree->SplitIndex(node_id);
                output.at(split_feature*num_samples*num_trees + j*num_samples + i)++;
            }
        }
    }
    return output;
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
bool is_leaf_node_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, int forest_num, int tree_num, int node_id) {
    StochTree::TreeEnsemble* ensemble = forest_samples->GetEnsemble(forest_num);
    StochTree::Tree* tree = ensemble->GetTree(tree_num);
    return tree->IsLeaf(node_id);
}

[[cpp11::register]]
bool is_numeric_split_node_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, int forest_num, int tree_num, int node_id) {
    StochTree::TreeEnsemble* ensemble = forest_samples->GetEnsemble(forest_num);
    StochTree::Tree* tree = ensemble->GetTree(tree_num);
    return tree->IsNumericSplitNode(node_id);
}

[[cpp11::register]]
bool is_categorical_split_node_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, int forest_num, int tree_num, int node_id) {
    StochTree::TreeEnsemble* ensemble = forest_samples->GetEnsemble(forest_num);
    StochTree::Tree* tree = ensemble->GetTree(tree_num);
    return tree->IsCategoricalSplitNode(node_id);
}

[[cpp11::register]]
int parent_node_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, int forest_num, int tree_num, int node_id) {
    StochTree::TreeEnsemble* ensemble = forest_samples->GetEnsemble(forest_num);
    StochTree::Tree* tree = ensemble->GetTree(tree_num);
    return tree->Parent(node_id);
}

[[cpp11::register]]
int left_child_node_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, int forest_num, int tree_num, int node_id) {
    StochTree::TreeEnsemble* ensemble = forest_samples->GetEnsemble(forest_num);
    StochTree::Tree* tree = ensemble->GetTree(tree_num);
    return tree->LeftChild(node_id);
}

[[cpp11::register]]
int right_child_node_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, int forest_num, int tree_num, int node_id) {
    StochTree::TreeEnsemble* ensemble = forest_samples->GetEnsemble(forest_num);
    StochTree::Tree* tree = ensemble->GetTree(tree_num);
    return tree->RightChild(node_id);
}

[[cpp11::register]]
int node_depth_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, int forest_num, int tree_num, int node_id) {
    StochTree::TreeEnsemble* ensemble = forest_samples->GetEnsemble(forest_num);
    StochTree::Tree* tree = ensemble->GetTree(tree_num);
    return tree->GetDepth(node_id);
}

[[cpp11::register]]
int split_index_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, int forest_num, int tree_num, int node_id) {
    StochTree::TreeEnsemble* ensemble = forest_samples->GetEnsemble(forest_num);
    StochTree::Tree* tree = ensemble->GetTree(tree_num);
    return tree->SplitIndex(node_id);
}

[[cpp11::register]]
double split_theshold_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, int forest_num, int tree_num, int node_id) {
    StochTree::TreeEnsemble* ensemble = forest_samples->GetEnsemble(forest_num);
    StochTree::Tree* tree = ensemble->GetTree(tree_num);
    return tree->Threshold(node_id);
}

[[cpp11::register]]
cpp11::writable::integers split_categories_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, int forest_num, int tree_num, int node_id) {
    StochTree::TreeEnsemble* ensemble = forest_samples->GetEnsemble(forest_num);
    StochTree::Tree* tree = ensemble->GetTree(tree_num);
    std::vector<std::uint32_t> raw_categories = tree->CategoryList(node_id);
    cpp11::writable::integers output(raw_categories.begin(), raw_categories.end());
    return output;
}

[[cpp11::register]]
cpp11::writable::doubles leaf_values_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, int forest_num, int tree_num, int node_id) {
    StochTree::TreeEnsemble* ensemble = forest_samples->GetEnsemble(forest_num);
    StochTree::Tree* tree = ensemble->GetTree(tree_num);
    int num_outputs = tree->OutputDimension();
    cpp11::writable::doubles output(num_outputs);
    for (int i = 0; i < num_outputs; i++) {
        output[i] = tree->LeafValue(node_id, i);
    }
    return output;
}

[[cpp11::register]]
int num_nodes_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, int forest_num, int tree_num) {
    StochTree::TreeEnsemble* ensemble = forest_samples->GetEnsemble(forest_num);
    StochTree::Tree* tree = ensemble->GetTree(tree_num);
    return tree->NumValidNodes();
}

[[cpp11::register]]
int num_leaves_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, int forest_num, int tree_num) {
    StochTree::TreeEnsemble* ensemble = forest_samples->GetEnsemble(forest_num);
    StochTree::Tree* tree = ensemble->GetTree(tree_num);
    return tree->NumLeaves();
}

[[cpp11::register]]
int num_leaf_parents_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, int forest_num, int tree_num) {
    StochTree::TreeEnsemble* ensemble = forest_samples->GetEnsemble(forest_num);
    StochTree::Tree* tree = ensemble->GetTree(tree_num);
    return tree->NumLeafParents();
}

[[cpp11::register]]
int num_split_nodes_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, int forest_num, int tree_num) {
    StochTree::TreeEnsemble* ensemble = forest_samples->GetEnsemble(forest_num);
    StochTree::Tree* tree = ensemble->GetTree(tree_num);
    return tree->NumSplitNodes();
}

[[cpp11::register]]
cpp11::writable::integers nodes_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, int forest_num, int tree_num) {
    StochTree::TreeEnsemble* ensemble = forest_samples->GetEnsemble(forest_num);
    StochTree::Tree* tree = ensemble->GetTree(tree_num);
    std::vector<std::int32_t> leaves = tree->GetNodes();
    cpp11::writable::integers output(leaves.begin(), leaves.end());
    return output;
}

[[cpp11::register]]
cpp11::writable::integers leaves_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, int forest_num, int tree_num) {
    StochTree::TreeEnsemble* ensemble = forest_samples->GetEnsemble(forest_num);
    StochTree::Tree* tree = ensemble->GetTree(tree_num);
    std::vector<std::int32_t> leaves = tree->GetLeaves();
    cpp11::writable::integers output(leaves.begin(), leaves.end());
    return output;
}

[[cpp11::register]]
void initialize_forest_model_cpp(cpp11::external_pointer<StochTree::ForestDataset> data, 
                                 cpp11::external_pointer<StochTree::ColumnVector> residual, 
                                 cpp11::external_pointer<StochTree::ForestContainer> forest_samples, 
                                 cpp11::external_pointer<StochTree::ForestTracker> tracker, 
                                 cpp11::doubles init_values, int leaf_model_int){
    // Convert leaf model type to enum
    StochTree::ModelType model_type;
    if (leaf_model_int == 0) model_type = StochTree::ModelType::kConstantLeafGaussian;
    else if (leaf_model_int == 1) model_type = StochTree::ModelType::kUnivariateRegressionLeafGaussian;
    else if (leaf_model_int == 2) model_type = StochTree::ModelType::kMultivariateRegressionLeafGaussian;
    else if (leaf_model_int == 3) model_type = StochTree::ModelType::kLogLinearVariance;
    else StochTree::Log::Fatal("Invalid model type");
    
    // Unpack initial value
    int num_trees = forest_samples->NumTrees();
    double init_val;
    std::vector<double> init_value_vector;
    if ((model_type == StochTree::ModelType::kConstantLeafGaussian) || 
        (model_type == StochTree::ModelType::kUnivariateRegressionLeafGaussian) || 
        (model_type == StochTree::ModelType::kLogLinearVariance)) {
        init_val = init_values.at(0);
    } else if (model_type == StochTree::ModelType::kMultivariateRegressionLeafGaussian) {
        int leaf_dim = init_values.size();
        init_value_vector.resize(leaf_dim);
        for (int i = 0; i < leaf_dim; i++) {
            init_value_vector[i] = init_values[i] / static_cast<double>(num_trees);
        }
    }
    
    // Initialize the models accordingly
    if (model_type == StochTree::ModelType::kConstantLeafGaussian) {
        forest_samples->InitializeRoot(init_val / static_cast<double>(num_trees));
        UpdateResidualEntireForest(*tracker, *data, *residual, forest_samples->GetEnsemble(0), false, std::minus<double>());
        tracker->UpdatePredictions(forest_samples->GetEnsemble(0), *data);
    } else if (model_type == StochTree::ModelType::kUnivariateRegressionLeafGaussian) {
        forest_samples->InitializeRoot(init_val / static_cast<double>(num_trees));
        UpdateResidualEntireForest(*tracker, *data, *residual, forest_samples->GetEnsemble(0), true, std::minus<double>());
        tracker->UpdatePredictions(forest_samples->GetEnsemble(0), *data);
    } else if (model_type == StochTree::ModelType::kMultivariateRegressionLeafGaussian) {
        forest_samples->InitializeRoot(init_value_vector);
        UpdateResidualEntireForest(*tracker, *data, *residual, forest_samples->GetEnsemble(0), true, std::minus<double>());
        tracker->UpdatePredictions(forest_samples->GetEnsemble(0), *data);
    } else if (model_type == StochTree::ModelType::kLogLinearVariance) {
        forest_samples->InitializeRoot(std::log(init_val) / static_cast<double>(num_trees));
        tracker->UpdatePredictions(forest_samples->GetEnsemble(0), *data);
        int n = data->NumObservations();
        std::vector<double> initial_preds(n, init_val);
        data->AddVarianceWeights(initial_preds.data(), n);
    }
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
void propagate_basis_update_forest_container_cpp(cpp11::external_pointer<StochTree::ForestDataset> data, 
                                                 cpp11::external_pointer<StochTree::ColumnVector> residual, 
                                                 cpp11::external_pointer<StochTree::ForestContainer> forest_samples, 
                                                 cpp11::external_pointer<StochTree::ForestTracker> tracker, 
                                                 int forest_num) {
    // Perform the update (addition / subtraction) operation
    StochTree::UpdateResidualNewBasis(*tracker, *data, *residual, forest_samples->GetEnsemble(forest_num));
}

[[cpp11::register]]
void remove_sample_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, 
                                        int forest_num) {
    forest_samples->DeleteSample(forest_num);
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
cpp11::writable::doubles predict_forest_raw_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, cpp11::external_pointer<StochTree::ForestDataset> dataset) {
    // Predict from the sampled forests
    std::vector<double> output_raw = forest_samples->PredictRaw(*dataset);
    
    // Unpack / re-arrange results
    int n = dataset->GetCovariates().rows();
    int num_samples = forest_samples->NumSamples();
    int output_dimension = forest_samples->OutputDimension();
    cpp11::writable::doubles output(n*output_dimension*num_samples);
    for (size_t i = 0; i < n; i++) {
        for (int j = 0; j < output_dimension; j++) {
            for (int k = 0; k < num_samples; k++) {
                // Convert from idiosyncratic C++ storage to "column-major" --- first dimension is data row, second is output column, third is sample number
                output.at(k*output_dimension*n + j*n + i) = output_raw[k*output_dimension*n + i*output_dimension + j];
            }
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

[[cpp11::register]]
cpp11::writable::doubles_matrix<> predict_forest_raw_single_tree_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, cpp11::external_pointer<StochTree::ForestDataset> dataset, int forest_num, int tree_num) {
    // Predict from the sampled forests
    std::vector<double> output_raw = forest_samples->PredictRawSingleTree(*dataset, forest_num, tree_num);
    
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

[[cpp11::register]]
cpp11::writable::doubles predict_active_forest_cpp(cpp11::external_pointer<StochTree::TreeEnsemble> active_forest, cpp11::external_pointer<StochTree::ForestDataset> dataset) {
    int n = dataset->GetCovariates().rows();
    std::vector<double> output(n);
    active_forest->PredictInplace(*dataset, output, 0);
    return output;
}

[[cpp11::register]]
cpp11::writable::doubles predict_raw_active_forest_cpp(cpp11::external_pointer<StochTree::TreeEnsemble> active_forest, cpp11::external_pointer<StochTree::ForestDataset> dataset) {
    int n = dataset->GetCovariates().rows();
    int output_dimension = active_forest->OutputDimension();
    std::vector<double> output_raw(n*output_dimension);
    active_forest->PredictRawInplace(*dataset, output_raw, 0);
    
    cpp11::writable::doubles output(n*output_dimension);
    for (size_t i = 0; i < n; i++) {
        for (int j = 0; j < output_dimension; j++) {
            // Convert from row-major to column-major
            output.at(j*n + i) = output_raw[i*output_dimension + j];
        }
    }
    
    return output;
}

[[cpp11::register]]
int output_dimension_active_forest_cpp(cpp11::external_pointer<StochTree::TreeEnsemble> active_forest) {
    return active_forest->OutputDimension();
}

[[cpp11::register]]
double average_max_depth_active_forest_cpp(cpp11::external_pointer<StochTree::TreeEnsemble> active_forest) {
    return active_forest->AverageMaxDepth();
}

[[cpp11::register]]
int num_trees_active_forest_cpp(cpp11::external_pointer<StochTree::TreeEnsemble> active_forest) {
    return active_forest->NumTrees();
}

[[cpp11::register]]
int ensemble_tree_max_depth_active_forest_cpp(cpp11::external_pointer<StochTree::TreeEnsemble> active_forest, int tree_num) {
    return active_forest->TreeMaxDepth(tree_num);
}

[[cpp11::register]]
int is_leaf_constant_active_forest_cpp(cpp11::external_pointer<StochTree::TreeEnsemble> active_forest) {
    return active_forest->IsLeafConstant();
}

[[cpp11::register]]
int is_exponentiated_active_forest_cpp(cpp11::external_pointer<StochTree::TreeEnsemble> active_forest) {
    return active_forest->IsExponentiated();
}

[[cpp11::register]]
bool all_roots_active_forest_cpp(cpp11::external_pointer<StochTree::TreeEnsemble> active_forest) {
    return active_forest->AllRoots();
}

[[cpp11::register]]
void set_leaf_value_active_forest_cpp(cpp11::external_pointer<StochTree::TreeEnsemble> active_forest, double leaf_value) {
    active_forest->SetLeafValue(leaf_value);
}

[[cpp11::register]]
void set_leaf_vector_active_forest_cpp(cpp11::external_pointer<StochTree::TreeEnsemble> active_forest, cpp11::doubles leaf_vector) {
    std::vector<double> leaf_vector_cast(leaf_vector.begin(), leaf_vector.end());
    active_forest->SetLeafVector(leaf_vector_cast);
}

[[cpp11::register]]
void add_numeric_split_tree_value_active_forest_cpp(cpp11::external_pointer<StochTree::TreeEnsemble> active_forest, int tree_num, int leaf_num, int feature_num, double split_threshold, double left_leaf_value, double right_leaf_value) {
    if (active_forest->OutputDimension() != 1) {
        cpp11::stop("leaf_vector must match forest leaf dimension");
    }
    StochTree::Tree* tree = active_forest->GetTree(tree_num);
    if (!tree->IsLeaf(leaf_num)) {
        cpp11::stop("leaf_num is not a leaf");
    }
    tree->ExpandNode(leaf_num, feature_num, split_threshold, left_leaf_value, right_leaf_value);
}

[[cpp11::register]]
void add_numeric_split_tree_vector_active_forest_cpp(cpp11::external_pointer<StochTree::TreeEnsemble> active_forest, int tree_num, int leaf_num, int feature_num, double split_threshold, cpp11::doubles left_leaf_vector, cpp11::doubles right_leaf_vector) {
    if (active_forest->OutputDimension() != left_leaf_vector.size()) {
        cpp11::stop("left_leaf_vector must match forest leaf dimension");
    }
    if (active_forest->OutputDimension() != right_leaf_vector.size()) {
        cpp11::stop("right_leaf_vector must match forest leaf dimension");
    }
    std::vector<double> left_leaf_vector_cast(left_leaf_vector.begin(), left_leaf_vector.end());
    std::vector<double> right_leaf_vector_cast(right_leaf_vector.begin(), right_leaf_vector.end());
    StochTree::Tree* tree = active_forest->GetTree(tree_num);
    if (!tree->IsLeaf(leaf_num)) {
        cpp11::stop("leaf_num is not a leaf");
    }
    tree->ExpandNode(leaf_num, feature_num, split_threshold, left_leaf_vector_cast, right_leaf_vector_cast);
}

[[cpp11::register]]
cpp11::writable::integers get_tree_leaves_active_forest_cpp(cpp11::external_pointer<StochTree::TreeEnsemble> active_forest, int tree_num) {
    StochTree::Tree* tree = active_forest->GetTree(tree_num);
    std::vector<int32_t> leaves_raw = tree->GetLeaves();
    cpp11::writable::integers leaves(leaves_raw.begin(), leaves_raw.end());
    return leaves;
}

[[cpp11::register]]
cpp11::writable::integers get_tree_split_counts_active_forest_cpp(cpp11::external_pointer<StochTree::TreeEnsemble> active_forest, int tree_num, int num_features) {
    cpp11::writable::integers output(num_features);
    for (int i = 0; i < output.size(); i++) output.at(i) = 0;
    StochTree::Tree* tree = active_forest->GetTree(tree_num);
    std::vector<int32_t> split_nodes = tree->GetInternalNodes();
    for (int i = 0; i < split_nodes.size(); i++) {
        auto split_feature = split_nodes.at(i);
        output.at(split_feature)++;
    }
    return output;
}

[[cpp11::register]]
cpp11::writable::integers get_overall_split_counts_active_forest_cpp(cpp11::external_pointer<StochTree::TreeEnsemble> active_forest, int num_features) {
    cpp11::writable::integers output(num_features);
    for (int i = 0; i < output.size(); i++) output.at(i) = 0;
    int num_trees = active_forest->NumTrees();
    for (int i = 0; i < num_trees; i++) {
        StochTree::Tree* tree = active_forest->GetTree(i);
        std::vector<int32_t> split_nodes = tree->GetInternalNodes();
        for (int j = 0; j < split_nodes.size(); j++) {
            auto split_feature = split_nodes.at(j);
            output.at(split_feature)++;
        }
    }
    return output;
}

[[cpp11::register]]
cpp11::writable::integers get_granular_split_count_array_active_forest_cpp(cpp11::external_pointer<StochTree::TreeEnsemble> active_forest, int num_features) {
    int num_trees = active_forest->NumTrees();
    cpp11::writable::integers output(num_features*num_trees);
    for (int elem = 0; elem < output.size(); elem++) output.at(elem) = 0;
    for (int i = 0; i < num_trees; i++) {
        StochTree::Tree* tree = active_forest->GetTree(i);
        std::vector<int32_t> split_nodes = tree->GetInternalNodes();
        for (int j = 0; j < split_nodes.size(); j++) {
            auto split_feature = split_nodes.at(j);
            output.at(split_feature*num_trees + i)++;
        }
    }
    return output;
}

[[cpp11::register]]
void initialize_forest_model_active_forest_cpp(cpp11::external_pointer<StochTree::ForestDataset> data, 
                                               cpp11::external_pointer<StochTree::ColumnVector> residual, 
                                               cpp11::external_pointer<StochTree::TreeEnsemble> active_forest, 
                                               cpp11::external_pointer<StochTree::ForestTracker> tracker, 
                                               cpp11::doubles init_values, int leaf_model_int){
    // Convert leaf model type to enum
    StochTree::ModelType model_type;
    if (leaf_model_int == 0) model_type = StochTree::ModelType::kConstantLeafGaussian;
    else if (leaf_model_int == 1) model_type = StochTree::ModelType::kUnivariateRegressionLeafGaussian;
    else if (leaf_model_int == 2) model_type = StochTree::ModelType::kMultivariateRegressionLeafGaussian;
    else if (leaf_model_int == 3) model_type = StochTree::ModelType::kLogLinearVariance;
    else StochTree::Log::Fatal("Invalid model type");
    
    // Unpack initial value
    int num_trees = active_forest->NumTrees();
    double init_val;
    std::vector<double> init_value_vector;
    if ((model_type == StochTree::ModelType::kConstantLeafGaussian) || 
        (model_type == StochTree::ModelType::kUnivariateRegressionLeafGaussian) || 
        (model_type == StochTree::ModelType::kLogLinearVariance)) {
        init_val = init_values.at(0);
    } else if (model_type == StochTree::ModelType::kMultivariateRegressionLeafGaussian) {
        int leaf_dim = init_values.size();
        init_value_vector.resize(leaf_dim);
        for (int i = 0; i < leaf_dim; i++) {
            init_value_vector[i] = init_values[i] / static_cast<double>(num_trees);
        }
    }
    
    // Initialize the models accordingly
    double leaf_init_val;
    if (model_type == StochTree::ModelType::kConstantLeafGaussian) {
        leaf_init_val = init_val / static_cast<double>(num_trees);
        active_forest->SetLeafValue(leaf_init_val);
        UpdateResidualEntireForest(*tracker, *data, *residual, active_forest.get(), false, std::minus<double>());
        tracker->UpdatePredictions(active_forest.get(), *data);
    } else if (model_type == StochTree::ModelType::kUnivariateRegressionLeafGaussian) {
        leaf_init_val = init_val / static_cast<double>(num_trees);
        active_forest->SetLeafValue(leaf_init_val);
        UpdateResidualEntireForest(*tracker, *data, *residual, active_forest.get(), true, std::minus<double>());
        tracker->UpdatePredictions(active_forest.get(), *data);
    } else if (model_type == StochTree::ModelType::kMultivariateRegressionLeafGaussian) {
        active_forest->SetLeafVector(init_value_vector);
        UpdateResidualEntireForest(*tracker, *data, *residual, active_forest.get(), true, std::minus<double>());
        tracker->UpdatePredictions(active_forest.get(), *data);
    } else if (model_type == StochTree::ModelType::kLogLinearVariance) {
        leaf_init_val = std::log(init_val) / static_cast<double>(num_trees);
        active_forest->SetLeafValue(leaf_init_val);
        tracker->UpdatePredictions(active_forest.get(), *data);
        int n = data->NumObservations();
        std::vector<double> initial_preds(n, init_val);
        data->AddVarianceWeights(initial_preds.data(), n);
    }
}

[[cpp11::register]]
void adjust_residual_active_forest_cpp(cpp11::external_pointer<StochTree::ForestDataset> data, 
                                       cpp11::external_pointer<StochTree::ColumnVector> residual, 
                                       cpp11::external_pointer<StochTree::TreeEnsemble> active_forest, 
                                       cpp11::external_pointer<StochTree::ForestTracker> tracker, 
                                       bool requires_basis, bool add) {
    // Determine whether or not we are adding forest predictions to the residuals
    std::function<double(double, double)> op;
    if (add) op = std::plus<double>();
    else op = std::minus<double>();
    
    // Perform the update (addition / subtraction) operation
    StochTree::UpdateResidualEntireForest(*tracker, *data, *residual, active_forest.get(), requires_basis, op);
}

[[cpp11::register]]
void propagate_basis_update_active_forest_cpp(cpp11::external_pointer<StochTree::ForestDataset> data, 
                                              cpp11::external_pointer<StochTree::ColumnVector> residual, 
                                              cpp11::external_pointer<StochTree::TreeEnsemble> active_forest, 
                                              cpp11::external_pointer<StochTree::ForestTracker> tracker) {
    // Perform the update (addition / subtraction) operation
    StochTree::UpdateResidualNewBasis(*tracker, *data, *residual, active_forest.get());
}

[[cpp11::register]]
void reset_active_forest_cpp(cpp11::external_pointer<StochTree::TreeEnsemble> active_forest, 
                             cpp11::external_pointer<StochTree::ForestContainer> forest_samples, 
                             int forest_num) {
    // Extract raw pointer to the forest held at index forest_num
    StochTree::TreeEnsemble* forest = forest_samples->GetEnsemble(forest_num);

    // Reset active forest using the forest held at index forest_num
    active_forest->ReconstituteFromForest(*forest);
}

[[cpp11::register]]
void reset_forest_model_cpp(cpp11::external_pointer<StochTree::ForestTracker> forest_tracker, 
                            cpp11::external_pointer<StochTree::TreeEnsemble> forest, 
                            cpp11::external_pointer<StochTree::ForestDataset> data, 
                            cpp11::external_pointer<StochTree::ColumnVector> residual, 
                            bool is_mean_model) {
    // Reset forest tracker using the forest held at index forest_num
    forest_tracker->ReconstituteFromForest(*forest, *data, *residual, is_mean_model);
}

[[cpp11::register]]
void root_reset_active_forest_cpp(cpp11::external_pointer<StochTree::TreeEnsemble> active_forest) {
    // Reset active forest to root
    active_forest->ResetRoot();
}
