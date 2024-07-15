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
    double nu, double lamb, cpp11::doubles_matrix<> leaf_cov_init, double global_variance_init, 
    int num_gfr, int num_burnin, int num_mcmc, int random_seed, int leaf_model_int
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
    
    // Convert leaf covariance to Eigen::MatrixXd
    int leaf_dim = leaf_cov_init.nrow();
    Eigen::MatrixXd leaf_cov(leaf_cov_init.nrow(), leaf_cov_init.ncol());
    for (int i = 0; i < leaf_cov_init.nrow(); i++) {
        leaf_cov(i,i) = leaf_cov_init(i,i);
        for (int j = 0; j < i; j++) {
            leaf_cov(i,j) = leaf_cov_init(i,j);
            leaf_cov(j,i) = leaf_cov_init(j,i);
        }
    }
    
    // Create BART dispatcher and add data
    double* covariate_data_ptr = REAL(PROTECT(covariates));
    double* outcome_data_ptr = REAL(PROTECT(outcome));
    if (leaf_model_int == 0) {
        // Create the dispatcher and load the data
        StochTree::BARTDispatcher<StochTree::GaussianConstantLeafModel> bart_dispatcher{};
        bart_dispatcher.AddDataset(covariate_data_ptr, num_rows, num_covariates, false, true);
        bart_dispatcher.AddTrainOutcome(outcome_data_ptr, num_rows);
        // Run the sampling loop
        bart_dispatcher.RunSampler(
            *bart_result_ptr_.get(), feature_types_vector, var_weights_vector, 
            num_trees, num_gfr, num_burnin, num_mcmc, global_variance_init, leaf_cov, 
            alpha, beta, nu, lamb, a_leaf, b_leaf, min_samples_leaf, cutpoint_grid_size,
            true, false, -1
        );
    } else if (leaf_model_int == 1) {
        // Create the dispatcher and load the data
        StochTree::BARTDispatcher<StochTree::GaussianUnivariateRegressionLeafModel> bart_dispatcher{};
        bart_dispatcher.AddDataset(covariate_data_ptr, num_rows, num_covariates, false, true);
        bart_dispatcher.AddTrainOutcome(outcome_data_ptr, num_rows);
        // Run the sampling loop
        bart_dispatcher.RunSampler(
            *bart_result_ptr_.get(), feature_types_vector, var_weights_vector, 
            num_trees, num_gfr, num_burnin, num_mcmc, global_variance_init, leaf_cov, 
            alpha, beta, nu, lamb, a_leaf, b_leaf, min_samples_leaf, cutpoint_grid_size, 
            true, false, -1
        );
    }
    // // TODO: Figure out dispatch here
    // else {
    //     // Create the dispatcher and load the data
    //     StochTree::BARTDispatcher<StochTree::GaussianMultivariateRegressionLeafModel> bart_dispatcher{};
    //     bart_dispatcher.AddDataset(covariate_data_ptr, num_rows, num_covariates, false, true);
    //     bart_dispatcher.AddTrainOutcome(outcome_data_ptr, num_rows);
    //     // Run the sampling loop
    //     bart_dispatcher.RunSampler(
    //         *bart_result_ptr_.get(), feature_types_vector, var_weights_vector, 
    //         num_trees, num_gfr, num_burnin, num_mcmc, global_variance_init, leaf_cov, 
    //         alpha, beta, nu, lamb, a_leaf, b_leaf, min_samples_leaf, cutpoint_grid_size, 
    //         true, false, -1
    //     );
    // }
    
    // Unprotect pointers to R data
    UNPROTECT(2);
    
    // Release management of the pointer to R session
    return cpp11::external_pointer<StochTree::BARTResult>(bart_result_ptr_.release());
}
