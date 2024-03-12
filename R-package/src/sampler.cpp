#include <cpp11.hpp>
#include "stochtree_types.h"
#include <stochtree/container.h>
#include <stochtree/leaf_model.h>
#include <stochtree/meta.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/tree_sampler.h>
#include <stochtree/variance_model.h>
#include <memory>
#include <vector>
using namespace cpp11;

[[cpp11::register]]
void sample_gfr_one_iteration_cpp(cpp11::external_pointer<StochTree::ForestDataset> data, 
                                  cpp11::external_pointer<StochTree::ColumnVector> residual, 
                                  cpp11::external_pointer<StochTree::ForestContainer> forest_samples, 
                                  cpp11::external_pointer<StochTree::ForestTracker> tracker, 
                                  cpp11::external_pointer<StochTree::TreePrior> split_prior, 
                                  cpp11::external_pointer<std::mt19937> rng, 
                                  cpp11::integers feature_types, int cutpoint_grid_size, 
                                  cpp11::doubles_matrix<> leaf_model_scale_input, 
                                  cpp11::doubles variable_weights, 
                                  double global_variance, int leaf_model_int
) {
    // Unpack feature types
    std::vector<StochTree::FeatureType> feature_types_(feature_types.size());
    for (int i = 0; i < feature_types.size(); i++) {
        feature_types_[i] = static_cast<StochTree::FeatureType>(feature_types[i]);
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
        leaf_scale = leaf_model_scale_input(0,0);
    } else if (leaf_model_enum == ForestLeafModel::kMultivariateRegression) {
        int num_row = leaf_model_scale_input.nrow();
        int num_col = leaf_model_scale_input.ncol();
        leaf_scale_matrix.resize(num_row, num_col);
        for (int i = 0; i < num_row; i++) {
            for (int j = 0; j < num_col; j++) {
                leaf_scale_matrix(i,j) = leaf_model_scale_input(i,j);
            }
        }
    }
    
    // Convert variable weights to std::vector
    std::vector<double> var_weights_vector(variable_weights.size());
    for (int i = 0; i < variable_weights.size(); i++) {
        var_weights_vector[i] = variable_weights[i];
    }
    
    // Run one iteration of the sampler
    if (leaf_model_enum == ForestLeafModel::kConstant) {
        StochTree::GaussianConstantLeafModel leaf_model = StochTree::GaussianConstantLeafModel(leaf_scale);
        StochTree::GFRForestSampler<StochTree::GaussianConstantLeafModel> sampler = StochTree::GFRForestSampler<StochTree::GaussianConstantLeafModel>(cutpoint_grid_size);
        sampler.SampleOneIter(*tracker, *forest_samples, leaf_model, *data, *residual, *split_prior, *rng, var_weights_vector, global_variance, feature_types_);
    } else if (leaf_model_enum == ForestLeafModel::kUnivariateRegression) {
        StochTree::GaussianUnivariateRegressionLeafModel leaf_model = StochTree::GaussianUnivariateRegressionLeafModel(leaf_scale);
        StochTree::GFRForestSampler<StochTree::GaussianUnivariateRegressionLeafModel> sampler = StochTree::GFRForestSampler<StochTree::GaussianUnivariateRegressionLeafModel>(cutpoint_grid_size);
        sampler.SampleOneIter(*tracker, *forest_samples, leaf_model, *data, *residual, *split_prior, *rng, var_weights_vector, global_variance, feature_types_);
    } else if (leaf_model_enum == ForestLeafModel::kMultivariateRegression) {
        StochTree::GaussianMultivariateRegressionLeafModel leaf_model = StochTree::GaussianMultivariateRegressionLeafModel(leaf_scale_matrix);
        StochTree::GFRForestSampler<StochTree::GaussianMultivariateRegressionLeafModel> sampler = StochTree::GFRForestSampler<StochTree::GaussianMultivariateRegressionLeafModel>(cutpoint_grid_size);
        sampler.SampleOneIter(*tracker, *forest_samples, leaf_model, *data, *residual, *split_prior, *rng, var_weights_vector, global_variance, feature_types_);
    }
}

[[cpp11::register]]
void sample_mcmc_one_iteration_cpp(cpp11::external_pointer<StochTree::ForestDataset> data, 
                                   cpp11::external_pointer<StochTree::ColumnVector> residual, 
                                   cpp11::external_pointer<StochTree::ForestContainer> forest_samples, 
                                   cpp11::external_pointer<StochTree::ForestTracker> tracker, 
                                   cpp11::external_pointer<StochTree::TreePrior> split_prior, 
                                   cpp11::external_pointer<std::mt19937> rng, 
                                   cpp11::integers feature_types, int cutpoint_grid_size, 
                                   cpp11::doubles_matrix<> leaf_model_scale_input, 
                                   cpp11::doubles variable_weights, 
                                   double global_variance, int leaf_model_int
) {
    // Unpack feature types
    std::vector<StochTree::FeatureType> feature_types_(feature_types.size());
    for (int i = 0; i < feature_types.size(); i++) {
        feature_types_[i] = static_cast<StochTree::FeatureType>(feature_types[i]);
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
        leaf_scale = leaf_model_scale_input(0,0);
    } else if (leaf_model_enum == ForestLeafModel::kMultivariateRegression) {
        int num_row = leaf_model_scale_input.nrow();
        int num_col = leaf_model_scale_input.ncol();
        leaf_scale_matrix.resize(num_row, num_col);
        for (int i = 0; i < num_row; i++) {
            for (int j = 0; j < num_col; j++) {
                leaf_scale_matrix(i,j) = leaf_model_scale_input(i,j);
            }
        }
    }
    
    // Convert variable weights to std::vector
    std::vector<double> var_weights_vector(variable_weights.size());
    for (int i = 0; i < variable_weights.size(); i++) {
        var_weights_vector[i] = variable_weights[i];
    }
    
    // Run one iteration of the sampler
    if (leaf_model_enum == ForestLeafModel::kConstant) {
        StochTree::GaussianConstantLeafModel leaf_model = StochTree::GaussianConstantLeafModel(leaf_scale);
        StochTree::MCMCForestSampler<StochTree::GaussianConstantLeafModel> sampler = StochTree::MCMCForestSampler<StochTree::GaussianConstantLeafModel>();
        sampler.SampleOneIter(*tracker, *forest_samples, leaf_model, *data, *residual, *split_prior, *rng, var_weights_vector, global_variance);
    } else if (leaf_model_enum == ForestLeafModel::kUnivariateRegression) {
        StochTree::GaussianUnivariateRegressionLeafModel leaf_model = StochTree::GaussianUnivariateRegressionLeafModel(leaf_scale);
        StochTree::MCMCForestSampler<StochTree::GaussianUnivariateRegressionLeafModel> sampler = StochTree::MCMCForestSampler<StochTree::GaussianUnivariateRegressionLeafModel>();
        sampler.SampleOneIter(*tracker, *forest_samples, leaf_model, *data, *residual, *split_prior, *rng, var_weights_vector, global_variance);
    } else if (leaf_model_enum == ForestLeafModel::kMultivariateRegression) {
        StochTree::GaussianMultivariateRegressionLeafModel leaf_model = StochTree::GaussianMultivariateRegressionLeafModel(leaf_scale_matrix);
        StochTree::MCMCForestSampler<StochTree::GaussianMultivariateRegressionLeafModel> sampler = StochTree::MCMCForestSampler<StochTree::GaussianMultivariateRegressionLeafModel>();
        sampler.SampleOneIter(*tracker, *forest_samples, leaf_model, *data, *residual, *split_prior, *rng, var_weights_vector, global_variance);
    }
}

[[cpp11::register]]
double sample_sigma2_one_iteration_cpp(cpp11::external_pointer<StochTree::ColumnVector> residual, 
                                       cpp11::external_pointer<std::mt19937> rng, 
                                       double nu, double lambda
) {
    // Run one iteration of the sampler
    StochTree::GlobalHomoskedasticVarianceModel var_model = StochTree::GlobalHomoskedasticVarianceModel();
    return var_model.SampleVarianceParameter(residual->GetData(), nu, lambda, *rng);
}

[[cpp11::register]]
double sample_tau_one_iteration_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, 
                                    cpp11::external_pointer<std::mt19937> rng, 
                                    double a, double b, int sample_num
) {
    // Run one iteration of the sampler
    StochTree::LeafNodeHomoskedasticVarianceModel var_model = StochTree::LeafNodeHomoskedasticVarianceModel();
    return var_model.SampleVarianceParameter(forest_samples->GetEnsemble(sample_num), a, b, *rng);
}

[[cpp11::register]]
cpp11::external_pointer<std::mt19937> rng_cpp(int random_seed = -1) {
    std::unique_ptr<std::mt19937> rng_;
    if (random_seed == -1) {
        std::random_device rd;
        rng_ = std::make_unique<std::mt19937>(rd());
    } else {
        rng_ = std::make_unique<std::mt19937>(random_seed);
    }
    
    // Release management of the pointer to R session
    return cpp11::external_pointer<std::mt19937>(rng_.release());
}

[[cpp11::register]]
cpp11::external_pointer<StochTree::ForestContainer> forest_container_cpp(int num_trees, int output_dimension = 1, bool is_leaf_constant = true) {
    // Create smart pointer to newly allocated object
    std::unique_ptr<StochTree::ForestContainer> forest_sample_ptr_ = std::make_unique<StochTree::ForestContainer>(num_trees, output_dimension, is_leaf_constant);

    // Release management of the pointer to R session
    return cpp11::external_pointer<StochTree::ForestContainer>(forest_sample_ptr_.release());
}

[[cpp11::register]]
cpp11::external_pointer<StochTree::TreePrior> tree_prior_cpp(double alpha, double beta, int min_samples_leaf) {
    // Create smart pointer to newly allocated object
    std::unique_ptr<StochTree::TreePrior> prior_ptr_ = std::make_unique<StochTree::TreePrior>(alpha, beta, min_samples_leaf);
    
    // Release management of the pointer to R session
    return cpp11::external_pointer<StochTree::TreePrior>(prior_ptr_.release());
}

[[cpp11::register]]
cpp11::external_pointer<StochTree::ForestTracker> forest_tracker_cpp(cpp11::external_pointer<StochTree::ForestDataset> data, cpp11::integers feature_types, int num_trees, StochTree::data_size_t n) {
    // Convert vector of integers to std::vector of enum FeatureType
    std::vector<StochTree::FeatureType> feature_types_(feature_types.size());
    for (int i = 0; i < feature_types.size(); i++) {
        feature_types_[i] = static_cast<StochTree::FeatureType>(feature_types[i]);
    }
    
    // Create smart pointer to newly allocated object
    std::unique_ptr<StochTree::ForestTracker> tracker_ptr_ = std::make_unique<StochTree::ForestTracker>(data->GetCovariates(), feature_types_, num_trees, n);
    
    // Release management of the pointer to R session
    return cpp11::external_pointer<StochTree::ForestTracker>(tracker_ptr_.release());
}