#include <cpp11.hpp>
#include "stochtree_types.h"
#include <stochtree/container.h>
#include <stochtree/leaf_model.h>
#include <stochtree/meta.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/tree_sampler.h>
#include <stochtree/variance_model.h>
#include <functional>
#include <memory>
#include <variant>
#include <vector>

[[cpp11::register]]
void sample_gfr_one_iteration_cpp(cpp11::external_pointer<StochTree::ForestDataset> data, 
                                  cpp11::external_pointer<StochTree::ColumnVector> residual, 
                                  cpp11::external_pointer<StochTree::ForestContainer> forest_samples, 
                                  cpp11::external_pointer<StochTree::TreeEnsemble> active_forest, 
                                  cpp11::external_pointer<StochTree::ForestTracker> tracker, 
                                  cpp11::external_pointer<StochTree::TreePrior> split_prior, 
                                  cpp11::external_pointer<std::mt19937> rng, 
                                  cpp11::integers feature_types, int cutpoint_grid_size, 
                                  cpp11::doubles_matrix<> leaf_model_scale_input, 
                                  cpp11::doubles variable_weights, 
                                  double a_forest, double b_forest,
                                  double global_variance, int leaf_model_int, 
                                  bool keep_forest,
                                  bool pre_initialized = false
) {
    // Unpack feature types
    std::vector<StochTree::FeatureType> feature_types_(feature_types.size());
    for (int i = 0; i < feature_types.size(); i++) {
        feature_types_[i] = static_cast<StochTree::FeatureType>(feature_types[i]);
    }
    
    // Convert leaf model type to enum
    StochTree::ModelType model_type;
    if (leaf_model_int == 0) model_type = StochTree::ModelType::kConstantLeafGaussian;
    else if (leaf_model_int == 1) model_type = StochTree::ModelType::kUnivariateRegressionLeafGaussian;
    else if (leaf_model_int == 2) model_type = StochTree::ModelType::kMultivariateRegressionLeafGaussian;
    else if (leaf_model_int == 3) model_type = StochTree::ModelType::kLogLinearVariance;
    else StochTree::Log::Fatal("Invalid model type");
    
    // Unpack leaf model parameters
    double leaf_scale;
    Eigen::MatrixXd leaf_scale_matrix;
    if ((model_type == StochTree::ModelType::kConstantLeafGaussian) || 
        (model_type == StochTree::ModelType::kUnivariateRegressionLeafGaussian)) {
        leaf_scale = leaf_model_scale_input(0,0);
    } else if (model_type == StochTree::ModelType::kMultivariateRegressionLeafGaussian) {
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
    
    // Prepare the samplers
    StochTree::LeafModelVariant leaf_model = StochTree::leafModelFactory(model_type, leaf_scale, leaf_scale_matrix, a_forest, b_forest);
    int num_basis = data->NumBasis();
    
    // Run one iteration of the sampler
    if (model_type == StochTree::ModelType::kConstantLeafGaussian) {
        StochTree::GFRSampleOneIter<StochTree::GaussianConstantLeafModel, StochTree::GaussianConstantSuffStat>(*active_forest, *tracker, *forest_samples, std::get<StochTree::GaussianConstantLeafModel>(leaf_model), *data, *residual, *split_prior, *rng, var_weights_vector, global_variance, feature_types_, cutpoint_grid_size, keep_forest, pre_initialized, true);
    } else if (model_type == StochTree::ModelType::kUnivariateRegressionLeafGaussian) {
        StochTree::GFRSampleOneIter<StochTree::GaussianUnivariateRegressionLeafModel, StochTree::GaussianUnivariateRegressionSuffStat>(*active_forest, *tracker, *forest_samples, std::get<StochTree::GaussianUnivariateRegressionLeafModel>(leaf_model), *data, *residual, *split_prior, *rng, var_weights_vector, global_variance, feature_types_, cutpoint_grid_size, keep_forest, pre_initialized, true);
    } else if (model_type == StochTree::ModelType::kMultivariateRegressionLeafGaussian) {
        StochTree::GFRSampleOneIter<StochTree::GaussianMultivariateRegressionLeafModel, StochTree::GaussianMultivariateRegressionSuffStat, int>(*active_forest, *tracker, *forest_samples, std::get<StochTree::GaussianMultivariateRegressionLeafModel>(leaf_model), *data, *residual, *split_prior, *rng, var_weights_vector, global_variance, feature_types_, cutpoint_grid_size, keep_forest, pre_initialized, true, num_basis);
    } else if (model_type == StochTree::ModelType::kLogLinearVariance) {
        StochTree::GFRSampleOneIter<StochTree::LogLinearVarianceLeafModel, StochTree::LogLinearVarianceSuffStat>(*active_forest, *tracker, *forest_samples, std::get<StochTree::LogLinearVarianceLeafModel>(leaf_model), *data, *residual, *split_prior, *rng, var_weights_vector, global_variance, feature_types_, cutpoint_grid_size, keep_forest, pre_initialized, false);
    }
}

[[cpp11::register]]
void sample_mcmc_one_iteration_cpp(cpp11::external_pointer<StochTree::ForestDataset> data, 
                                   cpp11::external_pointer<StochTree::ColumnVector> residual, 
                                   cpp11::external_pointer<StochTree::ForestContainer> forest_samples, 
                                   cpp11::external_pointer<StochTree::TreeEnsemble> active_forest, 
                                   cpp11::external_pointer<StochTree::ForestTracker> tracker, 
                                   cpp11::external_pointer<StochTree::TreePrior> split_prior, 
                                   cpp11::external_pointer<std::mt19937> rng, 
                                   cpp11::integers feature_types, int cutpoint_grid_size, 
                                   cpp11::doubles_matrix<> leaf_model_scale_input, 
                                   cpp11::doubles variable_weights, 
                                   double a_forest, double b_forest,
                                   double global_variance, int leaf_model_int, 
                                   bool keep_forest, 
                                   bool pre_initialized = false
) {
    // Unpack feature types
    std::vector<StochTree::FeatureType> feature_types_(feature_types.size());
    for (int i = 0; i < feature_types.size(); i++) {
        feature_types_[i] = static_cast<StochTree::FeatureType>(feature_types[i]);
    }
    
    // Convert leaf model type to enum
    StochTree::ModelType model_type;
    if (leaf_model_int == 0) model_type = StochTree::ModelType::kConstantLeafGaussian;
    else if (leaf_model_int == 1) model_type = StochTree::ModelType::kUnivariateRegressionLeafGaussian;
    else if (leaf_model_int == 2) model_type = StochTree::ModelType::kMultivariateRegressionLeafGaussian;
    else if (leaf_model_int == 3) model_type = StochTree::ModelType::kLogLinearVariance;
    else StochTree::Log::Fatal("Invalid model type");
    
    // Unpack leaf model parameters
    double leaf_scale;
    Eigen::MatrixXd leaf_scale_matrix;
    if ((model_type == StochTree::ModelType::kConstantLeafGaussian) || 
        (model_type == StochTree::ModelType::kUnivariateRegressionLeafGaussian)) {
        leaf_scale = leaf_model_scale_input(0,0);
    } else if (model_type == StochTree::ModelType::kMultivariateRegressionLeafGaussian) {
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
    
    // Prepare the samplers
    StochTree::LeafModelVariant leaf_model = StochTree::leafModelFactory(model_type, leaf_scale, leaf_scale_matrix, a_forest, b_forest);
    int num_basis = data->NumBasis();
    
    // Run one iteration of the sampler
    if (model_type == StochTree::ModelType::kConstantLeafGaussian) {
        StochTree::MCMCSampleOneIter<StochTree::GaussianConstantLeafModel, StochTree::GaussianConstantSuffStat>(*active_forest, *tracker, *forest_samples, std::get<StochTree::GaussianConstantLeafModel>(leaf_model), *data, *residual, *split_prior, *rng, var_weights_vector, global_variance, keep_forest, pre_initialized, true);
    } else if (model_type == StochTree::ModelType::kUnivariateRegressionLeafGaussian) {
        StochTree::MCMCSampleOneIter<StochTree::GaussianUnivariateRegressionLeafModel, StochTree::GaussianUnivariateRegressionSuffStat>(*active_forest, *tracker, *forest_samples, std::get<StochTree::GaussianUnivariateRegressionLeafModel>(leaf_model), *data, *residual, *split_prior, *rng, var_weights_vector, global_variance, keep_forest, pre_initialized, true);
    } else if (model_type == StochTree::ModelType::kMultivariateRegressionLeafGaussian) {
        StochTree::MCMCSampleOneIter<StochTree::GaussianMultivariateRegressionLeafModel, StochTree::GaussianMultivariateRegressionSuffStat, int>(*active_forest, *tracker, *forest_samples, std::get<StochTree::GaussianMultivariateRegressionLeafModel>(leaf_model), *data, *residual, *split_prior, *rng, var_weights_vector, global_variance, keep_forest, pre_initialized, true, num_basis);
    } else if (model_type == StochTree::ModelType::kLogLinearVariance) {
        StochTree::MCMCSampleOneIter<StochTree::LogLinearVarianceLeafModel, StochTree::LogLinearVarianceSuffStat>(*active_forest, *tracker, *forest_samples, std::get<StochTree::LogLinearVarianceLeafModel>(leaf_model), *data, *residual, *split_prior, *rng, var_weights_vector, global_variance, keep_forest, pre_initialized, false);
    }
}

[[cpp11::register]]
double sample_sigma2_one_iteration_cpp(cpp11::external_pointer<StochTree::ColumnVector> residual, 
                                       cpp11::external_pointer<StochTree::ForestDataset> dataset, 
                                       cpp11::external_pointer<std::mt19937> rng, 
                                       double a, double b
) {
    // Run one iteration of the sampler
    StochTree::GlobalHomoskedasticVarianceModel var_model = StochTree::GlobalHomoskedasticVarianceModel();
    if (dataset->HasVarWeights()) {
        return var_model.SampleVarianceParameter(residual->GetData(), dataset->GetVarWeights(), a, b, *rng);
    } else {
        return var_model.SampleVarianceParameter(residual->GetData(), a, b, *rng);
    }
}

[[cpp11::register]]
double sample_tau_one_iteration_cpp(cpp11::external_pointer<StochTree::TreeEnsemble> active_forest, 
                                    cpp11::external_pointer<std::mt19937> rng, 
                                    double a, double b
) {
    // Run one iteration of the sampler
    StochTree::LeafNodeHomoskedasticVarianceModel var_model = StochTree::LeafNodeHomoskedasticVarianceModel();
    return var_model.SampleVarianceParameter(active_forest.get(), a, b, *rng);
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
cpp11::external_pointer<StochTree::TreePrior> tree_prior_cpp(double alpha, double beta, int min_samples_leaf, int max_depth = -1) {
    // Create smart pointer to newly allocated object
    std::unique_ptr<StochTree::TreePrior> prior_ptr_ = std::make_unique<StochTree::TreePrior>(alpha, beta, min_samples_leaf, max_depth);
    
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