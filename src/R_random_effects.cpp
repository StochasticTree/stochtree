#include <cpp11.hpp>
#include "stochtree_types.h"
#include <stochtree/container.h>
#include <stochtree/leaf_model.h>
#include <stochtree/meta.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/random_effects.h>
#include <stochtree/tree_sampler.h>
#include <stochtree/variance_model.h>
#include <functional>
#include <memory>
#include <vector>

[[cpp11::register]]
cpp11::external_pointer<StochTree::RandomEffectsContainer> rfx_container_cpp(int num_components, int num_groups) {
    // Create smart pointer to newly allocated object
    std::unique_ptr<StochTree::RandomEffectsContainer> rfx_container_ptr_ = std::make_unique<StochTree::RandomEffectsContainer>(num_components, num_groups);
    
    // Release management of the pointer to R session
    return cpp11::external_pointer<StochTree::RandomEffectsContainer>(rfx_container_ptr_.release());
}

[[cpp11::register]]
cpp11::external_pointer<StochTree::RandomEffectsContainer> rfx_container_from_json_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string rfx_label) {
    // Create smart pointer to newly allocated object
    std::unique_ptr<StochTree::RandomEffectsContainer> rfx_container_ptr_ = std::make_unique<StochTree::RandomEffectsContainer>();
    
    // Extract the random effect container's json
    nlohmann::json rfx_json = json_ptr->at("random_effects").at(rfx_label);
    
    // Reset the forest sample container using the json
    rfx_container_ptr_->Reset();
    rfx_container_ptr_->from_json(rfx_json);
    
    // Release management of the pointer to R session
    return cpp11::external_pointer<StochTree::RandomEffectsContainer>(rfx_container_ptr_.release());
}

[[cpp11::register]]
cpp11::external_pointer<StochTree::LabelMapper> rfx_label_mapper_from_json_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string rfx_label) {
    // Create smart pointer to newly allocated object
    std::unique_ptr<StochTree::LabelMapper> label_mapper_ptr_ = std::make_unique<StochTree::LabelMapper>();
    
    // Extract the label mapper's json
    nlohmann::json rfx_json = json_ptr->at("random_effects").at(rfx_label);
    
    // Reset the label mapper using the json
    label_mapper_ptr_->Reset();
    label_mapper_ptr_->from_json(rfx_json);
    
    // Release management of the pointer to R session
    return cpp11::external_pointer<StochTree::LabelMapper>(label_mapper_ptr_.release());
}

[[cpp11::register]]
cpp11::writable::integers rfx_group_ids_from_json_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string rfx_label) {
    // Create smart pointer to newly allocated object
    cpp11::writable::integers output;
    
    // Extract the groupids' json
    nlohmann::json rfx_json = json_ptr->at("random_effects").at(rfx_label);
    
    // Reset the forest sample container using the json
    int num_groups = rfx_json.size();
    for (int i = 0; i < num_groups; i++) {
        output.push_back(rfx_json.at(i));
    }
    
    return output;
}

[[cpp11::register]]
void rfx_container_append_from_json_cpp(cpp11::external_pointer<StochTree::RandomEffectsContainer> rfx_container_ptr, cpp11::external_pointer<nlohmann::json> json_ptr, std::string rfx_label) {
    // Extract the random effect container's json
    nlohmann::json rfx_json = json_ptr->at("random_effects").at(rfx_label);
    
    // Reset the forest sample container using the json
    rfx_container_ptr->append_from_json(rfx_json);
}

[[cpp11::register]]
cpp11::external_pointer<StochTree::RandomEffectsContainer> rfx_container_from_json_string_cpp(std::string json_string, std::string rfx_label) {
    // Create smart pointer to newly allocated object
    std::unique_ptr<StochTree::RandomEffectsContainer> rfx_container_ptr_ = std::make_unique<StochTree::RandomEffectsContainer>();
    
    // Create a nlohmann::json object from the string
    nlohmann::json json_object = nlohmann::json::parse(json_string);
    
    // Extract the random effect container's json
    nlohmann::json rfx_json = json_object.at("random_effects").at(rfx_label);
    
    // Reset the forest sample container using the json
    rfx_container_ptr_->Reset();
    rfx_container_ptr_->from_json(rfx_json);
    
    // Release management of the pointer to R session
    return cpp11::external_pointer<StochTree::RandomEffectsContainer>(rfx_container_ptr_.release());
}

[[cpp11::register]]
cpp11::external_pointer<StochTree::LabelMapper> rfx_label_mapper_from_json_string_cpp(std::string json_string, std::string rfx_label) {
    // Create smart pointer to newly allocated object
    std::unique_ptr<StochTree::LabelMapper> label_mapper_ptr_ = std::make_unique<StochTree::LabelMapper>();
    
    // Create a nlohmann::json object from the string
    nlohmann::json json_object = nlohmann::json::parse(json_string);
    
    // Extract the label mapper's json
    nlohmann::json rfx_json = json_object.at("random_effects").at(rfx_label);
    
    // Reset the label mapper using the json
    label_mapper_ptr_->Reset();
    label_mapper_ptr_->from_json(rfx_json);
    
    // Release management of the pointer to R session
    return cpp11::external_pointer<StochTree::LabelMapper>(label_mapper_ptr_.release());
}

[[cpp11::register]]
cpp11::writable::integers rfx_group_ids_from_json_string_cpp(std::string json_string, std::string rfx_label) {
    // Create smart pointer to newly allocated object
    cpp11::writable::integers output;
    
    // Create a nlohmann::json object from the string
    nlohmann::json json_object = nlohmann::json::parse(json_string);
    
    // Extract the groupids' json
    nlohmann::json rfx_json = json_object.at("random_effects").at(rfx_label);
    
    // Reset the forest sample container using the json
    int num_groups = rfx_json.size();
    for (int i = 0; i < num_groups; i++) {
        output.push_back(rfx_json.at(i));
    }
    
    return output;
}

[[cpp11::register]]
void rfx_container_append_from_json_string_cpp(cpp11::external_pointer<StochTree::RandomEffectsContainer> rfx_container_ptr, std::string json_string, std::string rfx_label) {
    // Create a nlohmann::json object from the string
    nlohmann::json json_object = nlohmann::json::parse(json_string);
    
    // Extract the random effect container's json
    nlohmann::json rfx_json = json_object.at("random_effects").at(rfx_label);
    
    // Reset the forest sample container using the json
    rfx_container_ptr->append_from_json(rfx_json);
}

[[cpp11::register]]
cpp11::external_pointer<StochTree::MultivariateRegressionRandomEffectsModel> rfx_model_cpp(int num_components, int num_groups) {
    // Create smart pointer to newly allocated object
    std::unique_ptr<StochTree::MultivariateRegressionRandomEffectsModel> rfx_model_ptr_ = std::make_unique<StochTree::MultivariateRegressionRandomEffectsModel>(num_components, num_groups);
    
    // Release management of the pointer to R session
    return cpp11::external_pointer<StochTree::MultivariateRegressionRandomEffectsModel>(rfx_model_ptr_.release());
}

[[cpp11::register]]
cpp11::external_pointer<StochTree::RandomEffectsTracker> rfx_tracker_cpp(cpp11::integers group_labels) {
    // Convert group_labels to a std::vector<int32_t>
    std::vector<int32_t> group_labels_vec(group_labels.begin(), group_labels.end());
    
    // Create smart pointer to newly allocated object
    std::unique_ptr<StochTree::RandomEffectsTracker> rfx_tracker_ptr_ = std::make_unique<StochTree::RandomEffectsTracker>(group_labels_vec);
    
    // Release management of the pointer to R session
    return cpp11::external_pointer<StochTree::RandomEffectsTracker>(rfx_tracker_ptr_.release());
}

[[cpp11::register]]
cpp11::external_pointer<StochTree::LabelMapper> rfx_label_mapper_cpp(cpp11::external_pointer<StochTree::RandomEffectsTracker> rfx_tracker) {
    // Create smart pointer to newly allocated object
    std::unique_ptr<StochTree::LabelMapper> rfx_label_mapper_ptr_ = std::make_unique<StochTree::LabelMapper>(rfx_tracker->GetLabelMap());
    
    // Release management of the pointer to R session
    return cpp11::external_pointer<StochTree::LabelMapper>(rfx_label_mapper_ptr_.release());
}

[[cpp11::register]]
void rfx_model_sample_random_effects_cpp(cpp11::external_pointer<StochTree::MultivariateRegressionRandomEffectsModel> rfx_model, cpp11::external_pointer<StochTree::RandomEffectsDataset> rfx_dataset, 
                                         cpp11::external_pointer<StochTree::ColumnVector> residual, cpp11::external_pointer<StochTree::RandomEffectsTracker> rfx_tracker, 
                                         cpp11::external_pointer<StochTree::RandomEffectsContainer> rfx_container, bool keep_sample, double global_variance, cpp11::external_pointer<std::mt19937> rng) {
    rfx_model->SampleRandomEffects(*rfx_dataset, *residual, *rfx_tracker, global_variance, *rng);
    if (keep_sample) rfx_container->AddSample(*rfx_model);
}

[[cpp11::register]]
cpp11::writable::doubles rfx_model_predict_cpp(cpp11::external_pointer<StochTree::MultivariateRegressionRandomEffectsModel> rfx_model, 
                                               cpp11::external_pointer<StochTree::RandomEffectsDataset> rfx_dataset, 
                                               cpp11::external_pointer<StochTree::RandomEffectsTracker> rfx_tracker) {
    std::vector<double> output = rfx_model->Predict(*rfx_dataset, *rfx_tracker);
    return output;
}

[[cpp11::register]]
cpp11::writable::doubles rfx_container_predict_cpp(cpp11::external_pointer<StochTree::RandomEffectsContainer> rfx_container, 
                                                   cpp11::external_pointer<StochTree::RandomEffectsDataset> rfx_dataset, 
                                                   cpp11::external_pointer<StochTree::LabelMapper> label_mapper) {
    int num_observations = rfx_dataset->NumObservations();
    int num_samples = rfx_container->NumSamples();
    std::vector<double> output(num_observations*num_samples);
    rfx_container->Predict(*rfx_dataset, *label_mapper, output);
    return output;
}

[[cpp11::register]]
int rfx_container_num_samples_cpp(cpp11::external_pointer<StochTree::RandomEffectsContainer> rfx_container) {
    return rfx_container->NumSamples();
}

[[cpp11::register]]
int rfx_container_num_components_cpp(cpp11::external_pointer<StochTree::RandomEffectsContainer> rfx_container) {
    return rfx_container->NumComponents();
}

[[cpp11::register]]
int rfx_container_num_groups_cpp(cpp11::external_pointer<StochTree::RandomEffectsContainer> rfx_container) {
    return rfx_container->NumGroups();
}

[[cpp11::register]]
void rfx_container_delete_sample_cpp(cpp11::external_pointer<StochTree::RandomEffectsContainer> rfx_container, int sample_num) {
    rfx_container->DeleteSample(sample_num);
}

[[cpp11::register]]
void rfx_model_set_working_parameter_cpp(cpp11::external_pointer<StochTree::MultivariateRegressionRandomEffectsModel> rfx_model, cpp11::doubles working_param_init) {
    Eigen::VectorXd working_param_eigen(working_param_init.size());
    for (int i = 0; i < working_param_init.size(); i++) {
        working_param_eigen(i) = working_param_init.at(i);
    }
    rfx_model->SetWorkingParameter(working_param_eigen);
}

[[cpp11::register]]
void rfx_model_set_group_parameters_cpp(cpp11::external_pointer<StochTree::MultivariateRegressionRandomEffectsModel> rfx_model, cpp11::doubles_matrix<> group_params_init) {
    Eigen::MatrixXd group_params_eigen(group_params_init.nrow(), group_params_init.ncol());
    for (int i = 0; i < group_params_init.nrow(); i++) {
        for (int j = 0; j < group_params_init.ncol(); j++) {
            group_params_eigen(i,j) = group_params_init(i,j);
        }
    }
    rfx_model->SetGroupParameters(group_params_eigen);
}

[[cpp11::register]]
void rfx_model_set_working_parameter_covariance_cpp(cpp11::external_pointer<StochTree::MultivariateRegressionRandomEffectsModel> rfx_model, cpp11::doubles_matrix<> working_param_cov_init) {
    Eigen::MatrixXd working_param_cov_eigen(working_param_cov_init.nrow(), working_param_cov_init.ncol());
    for (int i = 0; i < working_param_cov_init.nrow(); i++) {
        for (int j = 0; j < working_param_cov_init.ncol(); j++) {
            working_param_cov_eigen(i,j) = working_param_cov_init(i,j);
        }
    }
    rfx_model->SetWorkingParameterCovariance(working_param_cov_eigen);
}

[[cpp11::register]]
void rfx_model_set_group_parameter_covariance_cpp(cpp11::external_pointer<StochTree::MultivariateRegressionRandomEffectsModel> rfx_model, cpp11::doubles_matrix<> group_param_cov_init) {
    Eigen::MatrixXd group_param_cov_eigen(group_param_cov_init.nrow(), group_param_cov_init.ncol());
    for (int i = 0; i < group_param_cov_init.nrow(); i++) {
        for (int j = 0; j < group_param_cov_init.ncol(); j++) {
            group_param_cov_eigen(i,j) = group_param_cov_init(i,j);
        }
    }
    rfx_model->SetGroupParameterCovariance(group_param_cov_eigen);
}

[[cpp11::register]]
void rfx_model_set_variance_prior_shape_cpp(cpp11::external_pointer<StochTree::MultivariateRegressionRandomEffectsModel> rfx_model, double shape) {
    rfx_model->SetVariancePriorShape(shape);
}

[[cpp11::register]]
void rfx_model_set_variance_prior_scale_cpp(cpp11::external_pointer<StochTree::MultivariateRegressionRandomEffectsModel> rfx_model, double scale) {
    rfx_model->SetVariancePriorScale(scale);
}

[[cpp11::register]]
cpp11::writable::integers rfx_tracker_get_unique_group_ids_cpp(cpp11::external_pointer<StochTree::RandomEffectsTracker> rfx_tracker) {
    std::vector<int32_t> output = rfx_tracker->GetUniqueGroupIds();
    return output;
}

[[cpp11::register]]
cpp11::writable::doubles rfx_container_get_beta_cpp(cpp11::external_pointer<StochTree::RandomEffectsContainer> rfx_container_ptr) {
    return rfx_container_ptr->GetBeta();
}

[[cpp11::register]]
cpp11::writable::doubles rfx_container_get_alpha_cpp(cpp11::external_pointer<StochTree::RandomEffectsContainer> rfx_container_ptr) {
    return rfx_container_ptr->GetAlpha();
}

[[cpp11::register]]
cpp11::writable::doubles rfx_container_get_xi_cpp(cpp11::external_pointer<StochTree::RandomEffectsContainer> rfx_container_ptr) {
    return rfx_container_ptr->GetXi();
}

[[cpp11::register]]
cpp11::writable::doubles rfx_container_get_sigma_cpp(cpp11::external_pointer<StochTree::RandomEffectsContainer> rfx_container_ptr) {
    return rfx_container_ptr->GetSigma();
}

[[cpp11::register]]
cpp11::list rfx_label_mapper_to_list_cpp(cpp11::external_pointer<StochTree::LabelMapper> label_mapper_ptr) {
    cpp11::writable::integers keys;
    cpp11::writable::integers values;
    std::map<int32_t, int32_t> label_map = label_mapper_ptr->Map();
    for (const auto& [key, value] : label_map) {
        keys.push_back(key);
        values.push_back(value);
    }
    
    cpp11::writable::list output;
    output.push_back(keys);
    output.push_back(values);
    return output;
}

[[cpp11::register]]
void reset_rfx_model_cpp(cpp11::external_pointer<StochTree::MultivariateRegressionRandomEffectsModel> rfx_model, 
                         cpp11::external_pointer<StochTree::RandomEffectsContainer> rfx_container, 
                         int sample_num) {
    // Reet the RFX tracker
    rfx_model->ResetFromSample(*rfx_container, sample_num);
}

[[cpp11::register]]
void reset_rfx_tracker_cpp(cpp11::external_pointer<StochTree::RandomEffectsTracker> tracker, 
                           cpp11::external_pointer<StochTree::RandomEffectsDataset> dataset, 
                           cpp11::external_pointer<StochTree::ColumnVector> residual, 
                           cpp11::external_pointer<StochTree::MultivariateRegressionRandomEffectsModel> rfx_model) {
    // Reset the RFX tracker
    tracker->ResetFromSample(*rfx_model, *dataset, *residual);
}

[[cpp11::register]]
void root_reset_rfx_tracker_cpp(cpp11::external_pointer<StochTree::RandomEffectsTracker> tracker, 
                                cpp11::external_pointer<StochTree::RandomEffectsDataset> dataset, 
                                cpp11::external_pointer<StochTree::ColumnVector> residual, 
                                cpp11::external_pointer<StochTree::MultivariateRegressionRandomEffectsModel> rfx_model) {
    // Reset the RFX tracker
    tracker->RootReset(*rfx_model, *dataset, *residual);
}
