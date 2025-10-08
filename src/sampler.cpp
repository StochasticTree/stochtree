#include <cpp11.hpp>
#include "stochtree_types.h"
#include <stochtree/container.h>
#include <stochtree/discrete_sampler.h>
#include <stochtree/leaf_model.h>
#include <stochtree/meta.h>
#include <stochtree/ordinal_sampler.h>
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
                                  cpp11::integers sweep_indices,
                                  cpp11::integers feature_types, int cutpoint_grid_size,
                                  cpp11::doubles_matrix<> leaf_model_scale_input,
                                  cpp11::doubles variable_weights,
                                  double a_forest, double b_forest,
                                  double global_variance, int leaf_model_int,
                                  bool keep_forest, int num_features_subsample
) {
    // Refactoring completely out of the R interface.
    // Intention to refactor out of the C++ interface in the future.
    bool pre_initialized = true;

    // Unpack feature types
    std::vector<StochTree::FeatureType> feature_types_(feature_types.size());
    for (int i = 0; i < feature_types.size(); i++) {
        feature_types_[i] = static_cast<StochTree::FeatureType>(feature_types[i]);
    }

    // Unpack sweep indices
    std::vector<int> sweep_indices_(sweep_indices.size());
    // if (sweep_indices.size() > 0) {
        // sweep_indices_.resize(sweep_indices.size());
    for (int i = 0; i < sweep_indices.size(); i++) {
        sweep_indices_[i] = sweep_indices[i];
    }
    // }

    // Convert leaf model type to enum
    StochTree::ModelType model_type;
    if (leaf_model_int == 0) model_type = StochTree::ModelType::kConstantLeafGaussian;
    else if (leaf_model_int == 1) model_type = StochTree::ModelType::kUnivariateRegressionLeafGaussian;
    else if (leaf_model_int == 2) model_type = StochTree::ModelType::kMultivariateRegressionLeafGaussian;
    else if (leaf_model_int == 3) model_type = StochTree::ModelType::kLogLinearVariance;
    else if (leaf_model_int == 4) StochTree::Log::Fatal("Ordinal models only support MCMC sampling, not GFR");
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
        StochTree::GFRSampleOneIter<StochTree::GaussianConstantLeafModel, StochTree::GaussianConstantSuffStat>(*active_forest, *tracker, *forest_samples, std::get<StochTree::GaussianConstantLeafModel>(leaf_model), *data, *residual, *split_prior, *rng, var_weights_vector, sweep_indices_, global_variance, feature_types_, cutpoint_grid_size, keep_forest, pre_initialized, true, num_features_subsample);
    } else if (model_type == StochTree::ModelType::kUnivariateRegressionLeafGaussian) {
        StochTree::GFRSampleOneIter<StochTree::GaussianUnivariateRegressionLeafModel, StochTree::GaussianUnivariateRegressionSuffStat>(*active_forest, *tracker, *forest_samples, std::get<StochTree::GaussianUnivariateRegressionLeafModel>(leaf_model), *data, *residual, *split_prior, *rng, var_weights_vector, sweep_indices_, global_variance, feature_types_, cutpoint_grid_size, keep_forest, pre_initialized, true, num_features_subsample);
    } else if (model_type == StochTree::ModelType::kMultivariateRegressionLeafGaussian) {
        StochTree::GFRSampleOneIter<StochTree::GaussianMultivariateRegressionLeafModel, StochTree::GaussianMultivariateRegressionSuffStat, int>(*active_forest, *tracker, *forest_samples, std::get<StochTree::GaussianMultivariateRegressionLeafModel>(leaf_model), *data, *residual, *split_prior, *rng, var_weights_vector, sweep_indices_, global_variance, feature_types_, cutpoint_grid_size, keep_forest, pre_initialized, true, num_features_subsample, num_basis);
    } else if (model_type == StochTree::ModelType::kLogLinearVariance) {
        StochTree::GFRSampleOneIter<StochTree::LogLinearVarianceLeafModel, StochTree::LogLinearVarianceSuffStat>(*active_forest, *tracker, *forest_samples, std::get<StochTree::LogLinearVarianceLeafModel>(leaf_model), *data, *residual, *split_prior, *rng, var_weights_vector, sweep_indices_, global_variance, feature_types_, cutpoint_grid_size, keep_forest, pre_initialized, false, num_features_subsample);
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
                                   cpp11::integers sweep_indices,
                                   cpp11::integers feature_types, int cutpoint_grid_size,
                                   cpp11::doubles_matrix<> leaf_model_scale_input,
                                   cpp11::doubles variable_weights,
                                   double a_forest, double b_forest,
                                   double global_variance, int leaf_model_int,
                                   bool keep_forest
) {
    // Refactoring completely out of the R interface.
    // Intention to refactor out of the C++ interface in the future.
    bool pre_initialized = true;

    // Unpack feature types
    std::vector<StochTree::FeatureType> feature_types_(feature_types.size());
    for (int i = 0; i < feature_types.size(); i++) {
        feature_types_[i] = static_cast<StochTree::FeatureType>(feature_types[i]);
    }

    // Unpack sweep indices
    std::vector<int> sweep_indices_;
    if (sweep_indices.size() > 0) {
        sweep_indices_.resize(sweep_indices.size());
        for (int i = 0; i < sweep_indices.size(); i++) {
            sweep_indices_[i] = sweep_indices[i];
        }
    }

    // Convert leaf model type to enum
    StochTree::ModelType model_type;
    if (leaf_model_int == 0) model_type = StochTree::ModelType::kConstantLeafGaussian;
    else if (leaf_model_int == 1) model_type = StochTree::ModelType::kUnivariateRegressionLeafGaussian;
    else if (leaf_model_int == 2) model_type = StochTree::ModelType::kMultivariateRegressionLeafGaussian;
    else if (leaf_model_int == 3) model_type = StochTree::ModelType::kLogLinearVariance;
    else if (leaf_model_int == 4) model_type = StochTree::ModelType::kCloglogOrdinal;
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
        StochTree::MCMCSampleOneIter<StochTree::GaussianConstantLeafModel, StochTree::GaussianConstantSuffStat>(*active_forest, *tracker, *forest_samples, std::get<StochTree::GaussianConstantLeafModel>(leaf_model), *data, *residual, *split_prior, *rng, var_weights_vector, sweep_indices_, global_variance, keep_forest, pre_initialized, true);
    } else if (model_type == StochTree::ModelType::kUnivariateRegressionLeafGaussian) {
        StochTree::MCMCSampleOneIter<StochTree::GaussianUnivariateRegressionLeafModel, StochTree::GaussianUnivariateRegressionSuffStat>(*active_forest, *tracker, *forest_samples, std::get<StochTree::GaussianUnivariateRegressionLeafModel>(leaf_model), *data, *residual, *split_prior, *rng, var_weights_vector, sweep_indices_, global_variance, keep_forest, pre_initialized, true);
    } else if (model_type == StochTree::ModelType::kMultivariateRegressionLeafGaussian) {
        StochTree::MCMCSampleOneIter<StochTree::GaussianMultivariateRegressionLeafModel, StochTree::GaussianMultivariateRegressionSuffStat, int>(*active_forest, *tracker, *forest_samples, std::get<StochTree::GaussianMultivariateRegressionLeafModel>(leaf_model), *data, *residual, *split_prior, *rng, var_weights_vector, sweep_indices_, global_variance, keep_forest, pre_initialized, true, num_basis);
    } else if (model_type == StochTree::ModelType::kLogLinearVariance) {
        StochTree::MCMCSampleOneIter<StochTree::LogLinearVarianceLeafModel, StochTree::LogLinearVarianceSuffStat>(*active_forest, *tracker, *forest_samples, std::get<StochTree::LogLinearVarianceLeafModel>(leaf_model), *data, *residual, *split_prior, *rng, var_weights_vector, sweep_indices_, global_variance, keep_forest, pre_initialized, false);
    } else if (model_type == StochTree::ModelType::kCloglogOrdinal) {
        StochTree::MCMCSampleOneIter<StochTree::CloglogOrdinalLeafModel, StochTree::CloglogOrdinalSuffStat>(*active_forest, *tracker, *forest_samples, std::get<StochTree::CloglogOrdinalLeafModel>(leaf_model), *data, *residual, *split_prior, *rng, var_weights_vector, sweep_indices_, global_variance, keep_forest, pre_initialized, false);
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
void update_alpha_tree_prior_cpp(cpp11::external_pointer<StochTree::TreePrior> tree_prior_ptr, double alpha) {
    // Update alpha
    tree_prior_ptr->SetAlpha(alpha);
}

[[cpp11::register]]
void update_beta_tree_prior_cpp(cpp11::external_pointer<StochTree::TreePrior> tree_prior_ptr, double beta) {
    // Update beta
    tree_prior_ptr->SetBeta(beta);
}

[[cpp11::register]]
void update_min_samples_leaf_tree_prior_cpp(cpp11::external_pointer<StochTree::TreePrior> tree_prior_ptr, int min_samples_leaf) {
    // Update min_samples_leaf
    tree_prior_ptr->SetMinSamplesLeaf(min_samples_leaf);
}

[[cpp11::register]]
void update_max_depth_tree_prior_cpp(cpp11::external_pointer<StochTree::TreePrior> tree_prior_ptr, int max_depth) {
    // Update max_depth
    tree_prior_ptr->SetMaxDepth(max_depth);
}

[[cpp11::register]]
double get_alpha_tree_prior_cpp(cpp11::external_pointer<StochTree::TreePrior> tree_prior_ptr) {
    return tree_prior_ptr->GetAlpha();
}

[[cpp11::register]]
double get_beta_tree_prior_cpp(cpp11::external_pointer<StochTree::TreePrior> tree_prior_ptr) {
    // Update beta
    return tree_prior_ptr->GetBeta();
}

[[cpp11::register]]
int get_min_samples_leaf_tree_prior_cpp(cpp11::external_pointer<StochTree::TreePrior> tree_prior_ptr) {
    return tree_prior_ptr->GetMinSamplesLeaf();
}

[[cpp11::register]]
int get_max_depth_tree_prior_cpp(cpp11::external_pointer<StochTree::TreePrior> tree_prior_ptr) {
    return tree_prior_ptr->GetMaxDepth();
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

[[cpp11::register]]
cpp11::writable::doubles get_cached_forest_predictions_cpp(cpp11::external_pointer<StochTree::ForestTracker> tracker_ptr) {
    int n_train = tracker_ptr->GetNumObservations();
    cpp11::writable::doubles output(n_train);
    for (int i = 0; i < n_train; i++) {
        output[i] = tracker_ptr->GetSamplePrediction(i);
    }
    return output;
}

[[cpp11::register]]
cpp11::writable::integers sample_without_replacement_integer_cpp(
    cpp11::integers population_vector,
    cpp11::doubles sampling_probs,
    int sample_size
) {
    // Unpack pointer to population vector
    int population_size = population_vector.size();
    int* population_vector_ptr = INTEGER(PROTECT(population_vector));

    // Unpack pointer to sampling probabilities
    double* sampling_probs_ptr = REAL(PROTECT(sampling_probs));

    // Create output vector
    cpp11::writable::integers output(sample_size);

    // Unpack pointer to output vector
    int* output_ptr = INTEGER(PROTECT(output));

    // Create C++ RNG
    std::random_device rd;
    std::mt19937 gen(rd());

    // Run the sampler
    StochTree::sample_without_replacement<int, double>(
        output_ptr, sampling_probs_ptr, population_vector_ptr, population_size, sample_size, gen
    );

    // Unprotect raw pointers
    UNPROTECT(3);

    // Return result
    return(output);
}

// ============================================================================
// ORDINAL AUXILIARY DATA FUNCTIONS
// ============================================================================

[[cpp11::register]]
void ordinal_aux_data_initialize_cpp(cpp11::external_pointer<StochTree::ForestTracker> tracker_ptr, StochTree::data_size_t num_observations, int n_levels) {
    tracker_ptr->InitializeOrdinalAuxData(num_observations, n_levels);
}

[[cpp11::register]]
double ordinal_aux_data_get_cpp(cpp11::external_pointer<StochTree::ForestTracker> tracker_ptr, int type_idx, StochTree::data_size_t obs_idx) {
    return tracker_ptr->GetOrdinalAuxData(type_idx, obs_idx);
}

[[cpp11::register]]
void ordinal_aux_data_set_cpp(cpp11::external_pointer<StochTree::ForestTracker> tracker_ptr, int type_idx, StochTree::data_size_t obs_idx, double value) {
    tracker_ptr->SetOrdinalAuxData(type_idx, obs_idx, value);
}

[[cpp11::register]]
cpp11::writable::doubles ordinal_aux_data_get_vector_cpp(cpp11::external_pointer<StochTree::ForestTracker> tracker_ptr, int type_idx) {
    const std::vector<double>& aux_vec = tracker_ptr->GetOrdinalAuxDataVector(type_idx);
    cpp11::writable::doubles output(aux_vec.size());
    for (size_t i = 0; i < aux_vec.size(); i++) {
        output[i] = aux_vec[i];
    }
    return output;
}

[[cpp11::register]]
void ordinal_aux_data_set_vector_cpp(cpp11::external_pointer<StochTree::ForestTracker> tracker_ptr, int type_idx, cpp11::doubles values) {
    std::vector<double>& aux_vec = tracker_ptr->GetOrdinalAuxDataVector(type_idx);
    if (aux_vec.size() != values.size()) {
        cpp11::stop("Size mismatch between auxiliary data vector and input values");
    }
    for (size_t i = 0; i < values.size(); i++) {
        aux_vec[i] = values[i];
    }
}

[[cpp11::register]]
void ordinal_aux_data_update_cumsum_exp_cpp(cpp11::external_pointer<StochTree::ForestTracker> tracker_ptr) {
    // Get auxiliary data vectors
    const std::vector<double>& gamma = tracker_ptr->GetOrdinalAuxDataVector(2);  // cutpoints gamma
    std::vector<double>& seg = tracker_ptr->GetOrdinalAuxDataVector(3);    // cumsum exp gamma

    // Update seg (cumulative sum of exp(gamma))
    for (size_t j = 0; j < seg.size(); j++) {
        if (j == 0) {
            seg[j] = 0.0;
        } else {
            seg[j] = seg[j - 1] + std::exp(gamma[j - 1]);
        }
    }
}

// ============================================================================
// ORDINAL SAMPLER FUNCTIONS
// ============================================================================

[[cpp11::register]]
cpp11::external_pointer<StochTree::OrdinalSampler> ordinal_sampler_cpp() {
    std::unique_ptr<StochTree::OrdinalSampler> sampler_ptr = std::make_unique<StochTree::OrdinalSampler>();
    return cpp11::external_pointer<StochTree::OrdinalSampler>(sampler_ptr.release());
}

[[cpp11::register]]
void ordinal_sampler_update_latent_variables_cpp(
    cpp11::external_pointer<StochTree::OrdinalSampler> sampler_ptr,
    cpp11::external_pointer<StochTree::ForestDataset> data_ptr,
    cpp11::external_pointer<StochTree::ColumnVector> outcome_ptr,
    cpp11::external_pointer<StochTree::ForestTracker> tracker_ptr,
    cpp11::external_pointer<std::mt19937> rng_ptr
) {
    sampler_ptr->UpdateLatentVariables(*data_ptr, outcome_ptr->GetData(), *tracker_ptr, *rng_ptr);
}

[[cpp11::register]]
void ordinal_sampler_update_gamma_params_cpp(
    cpp11::external_pointer<StochTree::OrdinalSampler> sampler_ptr,
    cpp11::external_pointer<StochTree::ForestDataset> data_ptr,
    cpp11::external_pointer<StochTree::ColumnVector> outcome_ptr,
    cpp11::external_pointer<StochTree::ForestTracker> tracker_ptr,
    double alpha_gamma,
    double beta_gamma,
    double gamma_0,
    cpp11::external_pointer<std::mt19937> rng_ptr
) {
    sampler_ptr->UpdateGammaParams(*data_ptr, outcome_ptr->GetData(), *tracker_ptr, alpha_gamma, beta_gamma, gamma_0, *rng_ptr);
}

[[cpp11::register]]
void ordinal_sampler_update_cumsum_exp_cpp(
    cpp11::external_pointer<StochTree::OrdinalSampler> sampler_ptr,
    cpp11::external_pointer<StochTree::ForestTracker> tracker_ptr
) {
    sampler_ptr->UpdateCumulativeExpSums(*tracker_ptr);
}


