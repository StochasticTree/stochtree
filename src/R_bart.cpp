#include <cpp11.hpp>
#include "stochtree_types.h"
#include <stochtree/bart.h>
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
cpp11::external_pointer<StochTree::BARTResult> run_bart_cpp(
    cpp11::doubles covariates, cpp11::doubles outcome, cpp11::integers feature_types, 
    cpp11::doubles variable_weights, int num_rows, int num_covariates, int num_trees, 
    int output_dimension, bool is_leaf_constant, double alpha, double beta, 
    int min_samples_leaf, int cutpoint_grid_size, double a_leaf, double b_leaf, 
    double nu, double lamb, double leaf_variance_init, double global_variance_init, 
    int num_gfr, int num_burnin, int num_mcmc, int random_seed
) {
    // Create smart pointer to newly allocated object
    std::unique_ptr<StochTree::BARTResult> bart_result_ptr_ = std::make_unique<StochTree::BARTResult>(num_trees, output_dimension, is_leaf_constant);
    
    // Convert variable weights to std::vector
    std::vector<double> var_weights_vector(variable_weights.size());
    for (int i = 0; i < variable_weights.size(); i++) {
        var_weights_vector[i] = variable_weights[i];
    }
    
    // Convert feature types to std::vector
    std::vector<StochTree::FeatureType> feature_types_vector(feature_types.size());
    for (int i = 0; i < feature_types.size(); i++) {
        feature_types_vector[i] = static_cast<StochTree::FeatureType>(feature_types[i]);
    }
    
    // Create BART dispatcher and add data
    StochTree::BARTDispatcher bart_dispatcher{};
    double* covariate_data_ptr = REAL(PROTECT(covariates));
    double* outcome_data_ptr = REAL(PROTECT(outcome));
    bart_dispatcher.AddDataset(covariate_data_ptr, num_rows, num_covariates, false, true);
    bart_dispatcher.AddTrainOutcome(outcome_data_ptr, num_rows);
    
    // Run the BART sampling loop
    bart_dispatcher.RunSampler(
        *bart_result_ptr_.get(), feature_types_vector, var_weights_vector, 
        num_trees, num_gfr, num_burnin, num_mcmc, global_variance_init, leaf_variance_init, 
        alpha, beta, nu, lamb, a_leaf, b_leaf, min_samples_leaf, cutpoint_grid_size
    );
    
    // Unprotect pointers to R data
    UNPROTECT(2);
    
    // Release management of the pointer to R session
    return cpp11::external_pointer<StochTree::BARTResult>(bart_result_ptr_.release());
}

