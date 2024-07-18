#include <cpp11.hpp>
#include "stochtree_types.h"
#include <stochtree/bart.h>
#include <stochtree/container.h>
#include <stochtree/leaf_model.h>
#include <stochtree/log.h>
#include <stochtree/meta.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/random_effects.h>
#include <stochtree/tree_sampler.h>
#include <stochtree/variance_model.h>
#include <functional>
#include <memory>
#include <vector>

[[cpp11::register]]
cpp11::external_pointer<StochTree::BARTResult> run_bart_cpp_basis_test_rfx(
    cpp11::doubles covariates_train, cpp11::doubles basis_train, cpp11::doubles outcome_train, 
    int num_rows_train, int num_covariates_train, int num_basis_train, 
    cpp11::doubles covariates_test, cpp11::doubles basis_test, 
    int num_rows_test, int num_covariates_test, int num_basis_test, 
    cpp11::doubles rfx_basis_train, cpp11::integers rfx_group_labels_train, 
    int num_rfx_basis_train, int num_rfx_groups_train,  
    cpp11::doubles rfx_basis_test, cpp11::integers rfx_group_labels_test, 
    int num_rfx_basis_test, int num_rfx_groups_test, cpp11::integers feature_types, 
    cpp11::doubles variable_weights, int num_trees, int output_dimension, bool is_leaf_constant, 
    double alpha, double beta, double a_leaf, double b_leaf, double nu, double lamb, 
    int min_samples_leaf, int cutpoint_grid_size, cpp11::doubles_matrix<> leaf_cov_init, 
    double global_variance_init, int num_gfr, int num_burnin, int num_mcmc, int random_seed, 
    int leaf_model_int, bool sample_global_var, bool sample_leaf_var, 
    cpp11::doubles rfx_alpha_init, cpp11::doubles_matrix<> rfx_xi_init, 
    cpp11::doubles_matrix<> rfx_sigma_alpha_init, cpp11::doubles_matrix<> rfx_sigma_xi_init, 
    double rfx_sigma_xi_shape, double rfx_sigma_xi_scale
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
    
    // Check inputs
    if (num_covariates_train != num_covariates_test) {
        StochTree::Log::Fatal("num_covariates_train must equal num_covariates_test");
    }
    if (num_basis_train != num_basis_test) {
        StochTree::Log::Fatal("num_basis_train must equal num_basis_test");
    }
    if (num_rfx_basis_train != num_rfx_basis_test) {
        StochTree::Log::Fatal("num_rfx_basis_train must equal num_rfx_basis_test");
    }
    if (num_rfx_groups_train != num_rfx_groups_test) {
        StochTree::Log::Fatal("num_rfx_groups_train must equal num_rfx_groups_test");
    }
    // if ((leaf_model_int == 1) || (leaf_model_int == 2)) {
    //     StochTree::Log::Fatal("Must provide basis for leaf regression");
    // }
    
    // Convert rfx group IDs to std::vector
    std::vector<int> rfx_group_labels_train_cpp;
    std::vector<int> rfx_group_labels_test_cpp;
    rfx_group_labels_train_cpp.resize(rfx_group_labels_train.size());
    for (int i = 0; i < rfx_group_labels_train.size(); i++) {
        rfx_group_labels_train_cpp.at(i) = rfx_group_labels_train.at(i);
    }
    rfx_group_labels_test_cpp.resize(rfx_group_labels_test.size());
    for (int i = 0; i < rfx_group_labels_test.size(); i++) {
        rfx_group_labels_test_cpp.at(i) = rfx_group_labels_test.at(i);
    }

    // Unpack RFX terms
    Eigen::VectorXd alpha_init;
    Eigen::MatrixXd xi_init;
    Eigen::MatrixXd sigma_alpha_init;
    Eigen::MatrixXd sigma_xi_init;
    double sigma_xi_shape;
    double sigma_xi_scale;
    alpha_init.resize(rfx_alpha_init.size());
    xi_init.resize(rfx_xi_init.nrow(), rfx_xi_init.ncol());
    sigma_alpha_init.resize(rfx_sigma_alpha_init.nrow(), rfx_sigma_alpha_init.ncol());
    sigma_xi_init.resize(rfx_sigma_xi_init.nrow(), rfx_sigma_xi_init.ncol());
    for (int i = 0; i < rfx_alpha_init.size(); i++) {
        alpha_init(i) = rfx_alpha_init.at(i);
    }
    for (int i = 0; i < rfx_xi_init.nrow(); i++) {
        for (int j = 0; j < rfx_xi_init.ncol(); j++) {
            xi_init(i,j) = rfx_xi_init(i,j);
        }
    }
    for (int i = 0; i < rfx_sigma_alpha_init.nrow(); i++) {
        for (int j = 0; j < rfx_sigma_alpha_init.ncol(); j++) {
            sigma_alpha_init(i,j) = rfx_sigma_alpha_init(i,j);
        }
    }
    for (int i = 0; i < rfx_sigma_xi_init.nrow(); i++) {
        for (int j = 0; j < rfx_sigma_xi_init.ncol(); j++) {
            sigma_xi_init(i,j) = rfx_sigma_xi_init(i,j);
        }
    }
    sigma_xi_shape = rfx_sigma_xi_shape;
    sigma_xi_scale = rfx_sigma_xi_scale;

    // Create BART dispatcher and add data
    double* train_covariate_data_ptr = REAL(PROTECT(covariates_train));
    double* train_basis_data_ptr = REAL(PROTECT(basis_train));
    double* train_outcome_data_ptr = REAL(PROTECT(outcome_train));
    double* test_covariate_data_ptr = REAL(PROTECT(covariates_test));
    double* test_basis_data_ptr = REAL(PROTECT(basis_test));
    double* train_rfx_basis_data_ptr = REAL(PROTECT(rfx_basis_train));
    double* test_rfx_basis_data_ptr = REAL(PROTECT(rfx_basis_test));
    if (leaf_model_int == 0) {
        // Create the dispatcher and load the data
        StochTree::BARTDispatcher<StochTree::GaussianConstantLeafModel> bart_dispatcher{};
        // Load training data
        bart_dispatcher.AddDataset(train_covariate_data_ptr, train_basis_data_ptr, num_rows_train, num_covariates_train, num_basis_train, false, true);
        bart_dispatcher.AddTrainOutcome(train_outcome_data_ptr, num_rows_train);
        // Load test data
        bart_dispatcher.AddDataset(test_covariate_data_ptr, test_basis_data_ptr, num_rows_test, num_covariates_test, num_basis_test, false, false);
        // Load rfx data
        bart_dispatcher.AddRFXTerm(train_rfx_basis_data_ptr, rfx_group_labels_train_cpp, num_rows_train, 
                                   num_rfx_groups_train, num_rfx_basis_train, false, true, alpha_init, 
                                   xi_init, sigma_alpha_init, sigma_xi_init, sigma_xi_shape, sigma_xi_scale);
        bart_dispatcher.AddRFXTerm(test_rfx_basis_data_ptr, rfx_group_labels_test_cpp, num_rows_test, 
                                   num_rfx_groups_test, num_rfx_basis_test, false, true, alpha_init, 
                                   xi_init, sigma_alpha_init, sigma_xi_init, sigma_xi_shape, sigma_xi_scale);
        // Run the sampling loop
        bart_dispatcher.RunSampler(
            *bart_result_ptr_.get(), feature_types_vector, var_weights_vector, 
            num_trees, num_gfr, num_burnin, num_mcmc, global_variance_init, leaf_cov, 
            alpha, beta, nu, lamb, a_leaf, b_leaf, min_samples_leaf, cutpoint_grid_size,
            sample_global_var, sample_leaf_var, random_seed
        );
    } else if (leaf_model_int == 1) {
        // Create the dispatcher and load the data
        StochTree::BARTDispatcher<StochTree::GaussianUnivariateRegressionLeafModel> bart_dispatcher{};
        // Load training data
        bart_dispatcher.AddDataset(train_covariate_data_ptr, train_basis_data_ptr, num_rows_train, num_covariates_train, num_basis_train, false, true);
        bart_dispatcher.AddTrainOutcome(train_outcome_data_ptr, num_rows_train);
        // Load test data
        bart_dispatcher.AddDataset(test_covariate_data_ptr, test_basis_data_ptr, num_rows_test, num_covariates_test, num_basis_test, false, false);
        // Load rfx data
        bart_dispatcher.AddRFXTerm(train_rfx_basis_data_ptr, rfx_group_labels_train_cpp, num_rows_train, 
                                   num_rfx_groups_train, num_rfx_basis_train, false, true, alpha_init, 
                                   xi_init, sigma_alpha_init, sigma_xi_init, sigma_xi_shape, sigma_xi_scale);
        bart_dispatcher.AddRFXTerm(test_rfx_basis_data_ptr, rfx_group_labels_test_cpp, num_rows_test, 
                                   num_rfx_groups_test, num_rfx_basis_test, false, true, alpha_init, 
                                   xi_init, sigma_alpha_init, sigma_xi_init, sigma_xi_shape, sigma_xi_scale);
        // Run the sampling loop
        bart_dispatcher.RunSampler(
            *bart_result_ptr_.get(), feature_types_vector, var_weights_vector, 
            num_trees, num_gfr, num_burnin, num_mcmc, global_variance_init, leaf_cov, 
            alpha, beta, nu, lamb, a_leaf, b_leaf, min_samples_leaf, cutpoint_grid_size,
            sample_global_var, sample_leaf_var, random_seed
        );
    } else {
        // Create the dispatcher and load the data
        StochTree::BARTDispatcher<StochTree::GaussianMultivariateRegressionLeafModel> bart_dispatcher{};
        // Load training data
        bart_dispatcher.AddDataset(train_covariate_data_ptr, train_basis_data_ptr, num_rows_train, num_covariates_train, num_basis_train, false, true);
        bart_dispatcher.AddTrainOutcome(train_outcome_data_ptr, num_rows_train);
        // Load test data
        bart_dispatcher.AddDataset(test_covariate_data_ptr, test_basis_data_ptr, num_rows_test, num_covariates_test, num_basis_test, false, false);
        // Load rfx data
        bart_dispatcher.AddRFXTerm(train_rfx_basis_data_ptr, rfx_group_labels_train_cpp, num_rows_train, 
                                   num_rfx_groups_train, num_rfx_basis_train, false, true, alpha_init, 
                                   xi_init, sigma_alpha_init, sigma_xi_init, sigma_xi_shape, sigma_xi_scale);
        bart_dispatcher.AddRFXTerm(test_rfx_basis_data_ptr, rfx_group_labels_test_cpp, num_rows_test, 
                                   num_rfx_groups_test, num_rfx_basis_test, false, true, alpha_init, 
                                   xi_init, sigma_alpha_init, sigma_xi_init, sigma_xi_shape, sigma_xi_scale);
        // Run the sampling loop
        bart_dispatcher.RunSampler(
            *bart_result_ptr_.get(), feature_types_vector, var_weights_vector, 
            num_trees, num_gfr, num_burnin, num_mcmc, global_variance_init, leaf_cov, 
            alpha, beta, nu, lamb, a_leaf, b_leaf, min_samples_leaf, cutpoint_grid_size,
            sample_global_var, sample_leaf_var, random_seed
        );
    }
    
    // Unprotect pointers to R data
    UNPROTECT(7);
    
    // Release management of the pointer to R session
    return cpp11::external_pointer<StochTree::BARTResult>(bart_result_ptr_.release());
}

[[cpp11::register]]
cpp11::external_pointer<StochTree::BARTResult> run_bart_cpp_basis_test_norfx(
        cpp11::doubles covariates_train, cpp11::doubles basis_train, cpp11::doubles outcome_train, 
        int num_rows_train, int num_covariates_train, int num_basis_train, 
        cpp11::doubles covariates_test, cpp11::doubles basis_test, 
        int num_rows_test, int num_covariates_test, int num_basis_test, 
        cpp11::integers feature_types, cpp11::doubles variable_weights, 
        int num_trees, int output_dimension, bool is_leaf_constant, 
        double alpha, double beta, double a_leaf, double b_leaf, double nu, double lamb, 
        int min_samples_leaf, int cutpoint_grid_size, cpp11::doubles_matrix<> leaf_cov_init, 
        double global_variance_init, int num_gfr, int num_burnin, int num_mcmc, int random_seed, 
        int leaf_model_int, bool sample_global_var, bool sample_leaf_var
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
    
    // Check inputs
    if (num_covariates_train != num_covariates_test) {
        StochTree::Log::Fatal("num_covariates_train must equal num_covariates_test");
    }
    if (num_basis_train != num_basis_test) {
        StochTree::Log::Fatal("num_basis_train must equal num_basis_test");
    }
    // if ((leaf_model_int == 1) || (leaf_model_int == 2)) {
    //     StochTree::Log::Fatal("Must provide basis for leaf regression");
    // }
    
    // Create BART dispatcher and add data
    double* train_covariate_data_ptr = REAL(PROTECT(covariates_train));
    double* train_basis_data_ptr = REAL(PROTECT(basis_train));
    double* train_outcome_data_ptr = REAL(PROTECT(outcome_train));
    double* test_covariate_data_ptr = REAL(PROTECT(covariates_test));
    double* test_basis_data_ptr = REAL(PROTECT(basis_test));
    if (leaf_model_int == 0) {
        // Create the dispatcher and load the data
        StochTree::BARTDispatcher<StochTree::GaussianConstantLeafModel> bart_dispatcher{};
        // Load training data
        bart_dispatcher.AddDataset(train_covariate_data_ptr, train_basis_data_ptr, num_rows_train, num_covariates_train, num_basis_train, false, true);
        bart_dispatcher.AddTrainOutcome(train_outcome_data_ptr, num_rows_train);
        // Load test data
        bart_dispatcher.AddDataset(test_covariate_data_ptr, test_basis_data_ptr, num_rows_test, num_covariates_test, num_basis_test, false, false);
        // Run the sampling loop
        bart_dispatcher.RunSampler(
            *bart_result_ptr_.get(), feature_types_vector, var_weights_vector, 
            num_trees, num_gfr, num_burnin, num_mcmc, global_variance_init, leaf_cov, 
            alpha, beta, nu, lamb, a_leaf, b_leaf, min_samples_leaf, cutpoint_grid_size,
            sample_global_var, sample_leaf_var, random_seed
        );
    } else if (leaf_model_int == 1) {
        // Create the dispatcher and load the data
        StochTree::BARTDispatcher<StochTree::GaussianUnivariateRegressionLeafModel> bart_dispatcher{};
        // Load training data
        bart_dispatcher.AddDataset(train_covariate_data_ptr, train_basis_data_ptr, num_rows_train, num_covariates_train, num_basis_train, false, true);
        bart_dispatcher.AddTrainOutcome(train_outcome_data_ptr, num_rows_train);
        // Load test data
        bart_dispatcher.AddDataset(test_covariate_data_ptr, test_basis_data_ptr, num_rows_test, num_covariates_test, num_basis_test, false, false);
        // Run the sampling loop
        bart_dispatcher.RunSampler(
            *bart_result_ptr_.get(), feature_types_vector, var_weights_vector, 
            num_trees, num_gfr, num_burnin, num_mcmc, global_variance_init, leaf_cov, 
            alpha, beta, nu, lamb, a_leaf, b_leaf, min_samples_leaf, cutpoint_grid_size,
            sample_global_var, sample_leaf_var, random_seed
        );
    } else {
        // Create the dispatcher and load the data
        StochTree::BARTDispatcher<StochTree::GaussianMultivariateRegressionLeafModel> bart_dispatcher{};
        // Load training data
        bart_dispatcher.AddDataset(train_covariate_data_ptr, train_basis_data_ptr, num_rows_train, num_covariates_train, num_basis_train, false, true);
        bart_dispatcher.AddTrainOutcome(train_outcome_data_ptr, num_rows_train);
        // Load test data
        bart_dispatcher.AddDataset(test_covariate_data_ptr, test_basis_data_ptr, num_rows_test, num_covariates_test, num_basis_test, false, false);
        // Run the sampling loop
        bart_dispatcher.RunSampler(
            *bart_result_ptr_.get(), feature_types_vector, var_weights_vector, 
            num_trees, num_gfr, num_burnin, num_mcmc, global_variance_init, leaf_cov, 
            alpha, beta, nu, lamb, a_leaf, b_leaf, min_samples_leaf, cutpoint_grid_size,
            sample_global_var, sample_leaf_var, random_seed
        );
    }
    
    // Unprotect pointers to R data
    UNPROTECT(5);
    
    // Release management of the pointer to R session
    return cpp11::external_pointer<StochTree::BARTResult>(bart_result_ptr_.release());
}

[[cpp11::register]]
cpp11::external_pointer<StochTree::BARTResult> run_bart_cpp_basis_notest_rfx(
        cpp11::doubles covariates_train, cpp11::doubles basis_train, cpp11::doubles outcome_train, 
        int num_rows_train, int num_covariates_train, int num_basis_train, 
        cpp11::doubles rfx_basis_train, cpp11::integers rfx_group_labels_train, 
        int num_rfx_basis_train, int num_rfx_groups_train,  
        cpp11::integers feature_types, cpp11::doubles variable_weights, 
        int num_trees, int output_dimension, bool is_leaf_constant, 
        double alpha, double beta, double a_leaf, double b_leaf, double nu, double lamb, 
        int min_samples_leaf, int cutpoint_grid_size, cpp11::doubles_matrix<> leaf_cov_init, 
        double global_variance_init, int num_gfr, int num_burnin, int num_mcmc, int random_seed, 
        int leaf_model_int, bool sample_global_var, bool sample_leaf_var, 
        cpp11::doubles rfx_alpha_init, cpp11::doubles_matrix<> rfx_xi_init, 
        cpp11::doubles_matrix<> rfx_sigma_alpha_init, cpp11::doubles_matrix<> rfx_sigma_xi_init, 
        double rfx_sigma_xi_shape, double rfx_sigma_xi_scale
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
    
    // Check inputs
    // if ((leaf_model_int == 1) || (leaf_model_int == 2)) {
    //     StochTree::Log::Fatal("Must provide basis for leaf regression");
    // }
    
    // Convert rfx group IDs to std::vector
    std::vector<int> rfx_group_labels_train_cpp;
    rfx_group_labels_train_cpp.resize(rfx_group_labels_train.size());
    for (int i = 0; i < rfx_group_labels_train.size(); i++) {
        rfx_group_labels_train_cpp.at(i) = rfx_group_labels_train.at(i);
    }

    // Unpack RFX terms
    Eigen::VectorXd alpha_init;
    Eigen::MatrixXd xi_init;
    Eigen::MatrixXd sigma_alpha_init;
    Eigen::MatrixXd sigma_xi_init;
    double sigma_xi_shape;
    double sigma_xi_scale;
    alpha_init.resize(rfx_alpha_init.size());
    xi_init.resize(rfx_xi_init.nrow(), rfx_xi_init.ncol());
    sigma_alpha_init.resize(rfx_sigma_alpha_init.nrow(), rfx_sigma_alpha_init.ncol());
    sigma_xi_init.resize(rfx_sigma_xi_init.nrow(), rfx_sigma_xi_init.ncol());
    for (int i = 0; i < rfx_alpha_init.size(); i++) {
        alpha_init(i) = rfx_alpha_init.at(i);
    }
    for (int i = 0; i < rfx_xi_init.nrow(); i++) {
        for (int j = 0; j < rfx_xi_init.ncol(); j++) {
            xi_init(i,j) = rfx_xi_init(i,j);
        }
    }
    for (int i = 0; i < rfx_sigma_alpha_init.nrow(); i++) {
        for (int j = 0; j < rfx_sigma_alpha_init.ncol(); j++) {
            sigma_alpha_init(i,j) = rfx_sigma_alpha_init(i,j);
        }
    }
    for (int i = 0; i < rfx_sigma_xi_init.nrow(); i++) {
        for (int j = 0; j < rfx_sigma_xi_init.ncol(); j++) {
            sigma_xi_init(i,j) = rfx_sigma_xi_init(i,j);
        }
    }
    sigma_xi_shape = rfx_sigma_xi_shape;
    sigma_xi_scale = rfx_sigma_xi_scale;
    
    // Create BART dispatcher and add data
    double* train_covariate_data_ptr = REAL(PROTECT(covariates_train));
    double* train_basis_data_ptr = REAL(PROTECT(basis_train));
    double* train_outcome_data_ptr = REAL(PROTECT(outcome_train));
    double* train_rfx_basis_data_ptr = REAL(PROTECT(rfx_basis_train));
    if (leaf_model_int == 0) {
        // Create the dispatcher and load the data
        StochTree::BARTDispatcher<StochTree::GaussianConstantLeafModel> bart_dispatcher{};
        // Load training data
        bart_dispatcher.AddDataset(train_covariate_data_ptr, train_basis_data_ptr, num_rows_train, num_covariates_train, num_basis_train, false, true);
        bart_dispatcher.AddTrainOutcome(train_outcome_data_ptr, num_rows_train);
        // Load rfx data
        bart_dispatcher.AddRFXTerm(train_rfx_basis_data_ptr, rfx_group_labels_train_cpp, num_rows_train, 
                                   num_rfx_groups_train, num_rfx_basis_train, false, true, alpha_init, 
                                   xi_init, sigma_alpha_init, sigma_xi_init, sigma_xi_shape, sigma_xi_scale);
        // Run the sampling loop
        bart_dispatcher.RunSampler(
            *bart_result_ptr_.get(), feature_types_vector, var_weights_vector, 
            num_trees, num_gfr, num_burnin, num_mcmc, global_variance_init, leaf_cov, 
            alpha, beta, nu, lamb, a_leaf, b_leaf, min_samples_leaf, cutpoint_grid_size,
            sample_global_var, sample_leaf_var, random_seed
        );
    } else if (leaf_model_int == 1) {
        // Create the dispatcher and load the data
        StochTree::BARTDispatcher<StochTree::GaussianUnivariateRegressionLeafModel> bart_dispatcher{};
        // Load training data
        bart_dispatcher.AddDataset(train_covariate_data_ptr, train_basis_data_ptr, num_rows_train, num_covariates_train, num_basis_train, false, true);
        bart_dispatcher.AddTrainOutcome(train_outcome_data_ptr, num_rows_train);
        // Load rfx data
        bart_dispatcher.AddRFXTerm(train_rfx_basis_data_ptr, rfx_group_labels_train_cpp, num_rows_train, 
                                   num_rfx_groups_train, num_rfx_basis_train, false, true, alpha_init, 
                                   xi_init, sigma_alpha_init, sigma_xi_init, sigma_xi_shape, sigma_xi_scale);
        // Run the sampling loop
        bart_dispatcher.RunSampler(
            *bart_result_ptr_.get(), feature_types_vector, var_weights_vector, 
            num_trees, num_gfr, num_burnin, num_mcmc, global_variance_init, leaf_cov, 
            alpha, beta, nu, lamb, a_leaf, b_leaf, min_samples_leaf, cutpoint_grid_size,
            sample_global_var, sample_leaf_var, random_seed
        );
    } else {
        // Create the dispatcher and load the data
        StochTree::BARTDispatcher<StochTree::GaussianMultivariateRegressionLeafModel> bart_dispatcher{};
        // Load training data
        bart_dispatcher.AddDataset(train_covariate_data_ptr, train_basis_data_ptr, num_rows_train, num_covariates_train, num_basis_train, false, true);
        bart_dispatcher.AddTrainOutcome(train_outcome_data_ptr, num_rows_train);
        // Load rfx data
        bart_dispatcher.AddRFXTerm(train_rfx_basis_data_ptr, rfx_group_labels_train_cpp, num_rows_train, 
                                   num_rfx_groups_train, num_rfx_basis_train, false, true, alpha_init, 
                                   xi_init, sigma_alpha_init, sigma_xi_init, sigma_xi_shape, sigma_xi_scale);
        // Run the sampling loop
        bart_dispatcher.RunSampler(
            *bart_result_ptr_.get(), feature_types_vector, var_weights_vector, 
            num_trees, num_gfr, num_burnin, num_mcmc, global_variance_init, leaf_cov, 
            alpha, beta, nu, lamb, a_leaf, b_leaf, min_samples_leaf, cutpoint_grid_size,
            sample_global_var, sample_leaf_var, random_seed
        );
    }
    
    // Unprotect pointers to R data
    UNPROTECT(4);
    
    // Release management of the pointer to R session
    return cpp11::external_pointer<StochTree::BARTResult>(bart_result_ptr_.release());
}

[[cpp11::register]]
cpp11::external_pointer<StochTree::BARTResult> run_bart_cpp_basis_notest_norfx(
        cpp11::doubles covariates_train, cpp11::doubles basis_train, cpp11::doubles outcome_train, 
        int num_rows_train, int num_covariates_train, int num_basis_train, 
        cpp11::integers feature_types, cpp11::doubles variable_weights, 
        int num_trees, int output_dimension, bool is_leaf_constant, 
        double alpha, double beta, double a_leaf, double b_leaf, double nu, double lamb, 
        int min_samples_leaf, int cutpoint_grid_size, cpp11::doubles_matrix<> leaf_cov_init, 
        double global_variance_init, int num_gfr, int num_burnin, int num_mcmc, int random_seed, 
        int leaf_model_int, bool sample_global_var, bool sample_leaf_var
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
    
    // Check inputs
    // if ((leaf_model_int == 1) || (leaf_model_int == 2)) {
    //     StochTree::Log::Fatal("Must provide basis for leaf regression");
    // }
    
    // Create BART dispatcher and add data
    double* train_covariate_data_ptr = REAL(PROTECT(covariates_train));
    double* train_basis_data_ptr = REAL(PROTECT(basis_train));
    double* train_outcome_data_ptr = REAL(PROTECT(outcome_train));
    if (leaf_model_int == 0) {
        // Create the dispatcher and load the data
        StochTree::BARTDispatcher<StochTree::GaussianConstantLeafModel> bart_dispatcher{};
        // Load training data
        bart_dispatcher.AddDataset(train_covariate_data_ptr, train_basis_data_ptr, num_rows_train, num_covariates_train, num_basis_train, false, true);
        bart_dispatcher.AddTrainOutcome(train_outcome_data_ptr, num_rows_train);
        // Run the sampling loop
        bart_dispatcher.RunSampler(
            *bart_result_ptr_.get(), feature_types_vector, var_weights_vector, 
            num_trees, num_gfr, num_burnin, num_mcmc, global_variance_init, leaf_cov, 
            alpha, beta, nu, lamb, a_leaf, b_leaf, min_samples_leaf, cutpoint_grid_size,
            sample_global_var, sample_leaf_var, random_seed
        );
    } else if (leaf_model_int == 1) {
        // Create the dispatcher and load the data
        StochTree::BARTDispatcher<StochTree::GaussianUnivariateRegressionLeafModel> bart_dispatcher{};
        // Load training data
        bart_dispatcher.AddDataset(train_covariate_data_ptr, train_basis_data_ptr, num_rows_train, num_covariates_train, num_basis_train, false, true);
        bart_dispatcher.AddTrainOutcome(train_outcome_data_ptr, num_rows_train);
        // Run the sampling loop
        bart_dispatcher.RunSampler(
            *bart_result_ptr_.get(), feature_types_vector, var_weights_vector, 
            num_trees, num_gfr, num_burnin, num_mcmc, global_variance_init, leaf_cov, 
            alpha, beta, nu, lamb, a_leaf, b_leaf, min_samples_leaf, cutpoint_grid_size,
            sample_global_var, sample_leaf_var, random_seed
        );
    } else {
        // Create the dispatcher and load the data
        StochTree::BARTDispatcher<StochTree::GaussianMultivariateRegressionLeafModel> bart_dispatcher{};
        // Load training data
        bart_dispatcher.AddDataset(train_covariate_data_ptr, train_basis_data_ptr, num_rows_train, num_covariates_train, num_basis_train, false, true);
        bart_dispatcher.AddTrainOutcome(train_outcome_data_ptr, num_rows_train);
        // Run the sampling loop
        bart_dispatcher.RunSampler(
            *bart_result_ptr_.get(), feature_types_vector, var_weights_vector, 
            num_trees, num_gfr, num_burnin, num_mcmc, global_variance_init, leaf_cov, 
            alpha, beta, nu, lamb, a_leaf, b_leaf, min_samples_leaf, cutpoint_grid_size,
            sample_global_var, sample_leaf_var, random_seed
        );
    }
    
    // Unprotect pointers to R data
    UNPROTECT(3);
    
    // Release management of the pointer to R session
    return cpp11::external_pointer<StochTree::BARTResult>(bart_result_ptr_.release());
}

[[cpp11::register]]
cpp11::external_pointer<StochTree::BARTResult> run_bart_cpp_nobasis_test_rfx(
        cpp11::doubles covariates_train, cpp11::doubles outcome_train, 
        int num_rows_train, int num_covariates_train, 
        cpp11::doubles covariates_test, 
        int num_rows_test, int num_covariates_test, 
        cpp11::doubles rfx_basis_train, cpp11::integers rfx_group_labels_train, 
        int num_rfx_basis_train, int num_rfx_groups_train,  
        cpp11::doubles rfx_basis_test, cpp11::integers rfx_group_labels_test, 
        int num_rfx_basis_test, int num_rfx_groups_test, cpp11::integers feature_types, 
        cpp11::doubles variable_weights, int num_trees, int output_dimension, bool is_leaf_constant, 
        double alpha, double beta, double a_leaf, double b_leaf, double nu, double lamb, 
        int min_samples_leaf, int cutpoint_grid_size, cpp11::doubles_matrix<> leaf_cov_init, 
        double global_variance_init, int num_gfr, int num_burnin, int num_mcmc, int random_seed, 
        int leaf_model_int, bool sample_global_var, bool sample_leaf_var, 
        cpp11::doubles rfx_alpha_init, cpp11::doubles_matrix<> rfx_xi_init, 
        cpp11::doubles_matrix<> rfx_sigma_alpha_init, cpp11::doubles_matrix<> rfx_sigma_xi_init, 
        double rfx_sigma_xi_shape, double rfx_sigma_xi_scale
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
    
    // Check inputs
    if (num_covariates_train != num_covariates_test) {
        StochTree::Log::Fatal("num_covariates_train must equal num_covariates_test");
    }
    if (num_rfx_basis_train != num_rfx_basis_test) {
        StochTree::Log::Fatal("num_rfx_basis_train must equal num_rfx_basis_test");
    }
    if (num_rfx_groups_train != num_rfx_groups_test) {
        StochTree::Log::Fatal("num_rfx_groups_train must equal num_rfx_groups_test");
    }
    // if ((leaf_model_int == 1) || (leaf_model_int == 2)) {
    //     StochTree::Log::Fatal("Must provide basis for leaf regression");
    // }
    
    // Convert rfx group IDs to std::vector
    std::vector<int> rfx_group_labels_train_cpp;
    std::vector<int> rfx_group_labels_test_cpp;
    rfx_group_labels_train_cpp.resize(rfx_group_labels_train.size());
    for (int i = 0; i < rfx_group_labels_train.size(); i++) {
        rfx_group_labels_train_cpp.at(i) = rfx_group_labels_train.at(i);
    }
    rfx_group_labels_test_cpp.resize(rfx_group_labels_test.size());
    for (int i = 0; i < rfx_group_labels_test.size(); i++) {
        rfx_group_labels_test_cpp.at(i) = rfx_group_labels_test.at(i);
    }
    
    // Unpack RFX terms
    Eigen::VectorXd alpha_init;
    Eigen::MatrixXd xi_init;
    Eigen::MatrixXd sigma_alpha_init;
    Eigen::MatrixXd sigma_xi_init;
    double sigma_xi_shape;
    double sigma_xi_scale;
    alpha_init.resize(rfx_alpha_init.size());
    xi_init.resize(rfx_xi_init.nrow(), rfx_xi_init.ncol());
    sigma_alpha_init.resize(rfx_sigma_alpha_init.nrow(), rfx_sigma_alpha_init.ncol());
    sigma_xi_init.resize(rfx_sigma_xi_init.nrow(), rfx_sigma_xi_init.ncol());
    for (int i = 0; i < rfx_alpha_init.size(); i++) {
        alpha_init(i) = rfx_alpha_init.at(i);
    }
    for (int i = 0; i < rfx_xi_init.nrow(); i++) {
        for (int j = 0; j < rfx_xi_init.ncol(); j++) {
            xi_init(i,j) = rfx_xi_init(i,j);
        }
    }
    for (int i = 0; i < rfx_sigma_alpha_init.nrow(); i++) {
        for (int j = 0; j < rfx_sigma_alpha_init.ncol(); j++) {
            sigma_alpha_init(i,j) = rfx_sigma_alpha_init(i,j);
        }
    }
    for (int i = 0; i < rfx_sigma_xi_init.nrow(); i++) {
        for (int j = 0; j < rfx_sigma_xi_init.ncol(); j++) {
            sigma_xi_init(i,j) = rfx_sigma_xi_init(i,j);
        }
    }
    sigma_xi_shape = rfx_sigma_xi_shape;
    sigma_xi_scale = rfx_sigma_xi_scale;
    
    // Create BART dispatcher and add data
    double* train_covariate_data_ptr = REAL(PROTECT(covariates_train));
    double* train_outcome_data_ptr = REAL(PROTECT(outcome_train));
    double* test_covariate_data_ptr = REAL(PROTECT(covariates_test));
    double* train_rfx_basis_data_ptr = REAL(PROTECT(rfx_basis_train));
    double* test_rfx_basis_data_ptr = REAL(PROTECT(rfx_basis_test));
    if (leaf_model_int == 0) {
        // Create the dispatcher and load the data
        StochTree::BARTDispatcher<StochTree::GaussianConstantLeafModel> bart_dispatcher{};
        // Load training data
        bart_dispatcher.AddDataset(train_covariate_data_ptr, num_rows_train, num_covariates_train, false, true);
        bart_dispatcher.AddTrainOutcome(train_outcome_data_ptr, num_rows_train);
        // Load test data
        bart_dispatcher.AddDataset(test_covariate_data_ptr, num_rows_test, num_covariates_test, false, false);
        // Load rfx data
        bart_dispatcher.AddRFXTerm(train_rfx_basis_data_ptr, rfx_group_labels_train_cpp, num_rows_train, 
                                   num_rfx_groups_train, num_rfx_basis_train, false, true, alpha_init, 
                                   xi_init, sigma_alpha_init, sigma_xi_init, sigma_xi_shape, sigma_xi_scale);
        bart_dispatcher.AddRFXTerm(test_rfx_basis_data_ptr, rfx_group_labels_test_cpp, num_rows_test, 
                                   num_rfx_groups_test, num_rfx_basis_test, false, true, alpha_init, 
                                   xi_init, sigma_alpha_init, sigma_xi_init, sigma_xi_shape, sigma_xi_scale);
        // Run the sampling loop
        bart_dispatcher.RunSampler(
            *bart_result_ptr_.get(), feature_types_vector, var_weights_vector, 
            num_trees, num_gfr, num_burnin, num_mcmc, global_variance_init, leaf_cov, 
            alpha, beta, nu, lamb, a_leaf, b_leaf, min_samples_leaf, cutpoint_grid_size,
            sample_global_var, sample_leaf_var, random_seed
        );
    } else if (leaf_model_int == 1) {
        // Create the dispatcher and load the data
        StochTree::BARTDispatcher<StochTree::GaussianUnivariateRegressionLeafModel> bart_dispatcher{};
        // Load training data
        bart_dispatcher.AddDataset(train_covariate_data_ptr, num_rows_train, num_covariates_train, false, true);
        bart_dispatcher.AddTrainOutcome(train_outcome_data_ptr, num_rows_train);
        // Load test data
        bart_dispatcher.AddDataset(test_covariate_data_ptr, num_rows_test, num_covariates_test, false, false);
        // Load rfx data
        bart_dispatcher.AddRFXTerm(train_rfx_basis_data_ptr, rfx_group_labels_train_cpp, num_rows_train, 
                                   num_rfx_groups_train, num_rfx_basis_train, false, true, alpha_init, 
                                   xi_init, sigma_alpha_init, sigma_xi_init, sigma_xi_shape, sigma_xi_scale);
        bart_dispatcher.AddRFXTerm(test_rfx_basis_data_ptr, rfx_group_labels_test_cpp, num_rows_test, 
                                   num_rfx_groups_test, num_rfx_basis_test, false, true, alpha_init, 
                                   xi_init, sigma_alpha_init, sigma_xi_init, sigma_xi_shape, sigma_xi_scale);
        // Run the sampling loop
        bart_dispatcher.RunSampler(
            *bart_result_ptr_.get(), feature_types_vector, var_weights_vector, 
            num_trees, num_gfr, num_burnin, num_mcmc, global_variance_init, leaf_cov, 
            alpha, beta, nu, lamb, a_leaf, b_leaf, min_samples_leaf, cutpoint_grid_size,
            sample_global_var, sample_leaf_var, random_seed
        );
    } else {
        // Create the dispatcher and load the data
        StochTree::BARTDispatcher<StochTree::GaussianMultivariateRegressionLeafModel> bart_dispatcher{};
        // Load training data
        bart_dispatcher.AddDataset(train_covariate_data_ptr, num_rows_train, num_covariates_train, false, true);
        bart_dispatcher.AddTrainOutcome(train_outcome_data_ptr, num_rows_train);
        // Load test data
        bart_dispatcher.AddDataset(test_covariate_data_ptr, num_rows_test, num_covariates_test, false, false);
        // Load rfx data
        bart_dispatcher.AddRFXTerm(train_rfx_basis_data_ptr, rfx_group_labels_train_cpp, num_rows_train, 
                                   num_rfx_groups_train, num_rfx_basis_train, false, true, alpha_init, 
                                   xi_init, sigma_alpha_init, sigma_xi_init, sigma_xi_shape, sigma_xi_scale);
        bart_dispatcher.AddRFXTerm(test_rfx_basis_data_ptr, rfx_group_labels_test_cpp, num_rows_test, 
                                   num_rfx_groups_test, num_rfx_basis_test, false, true, alpha_init, 
                                   xi_init, sigma_alpha_init, sigma_xi_init, sigma_xi_shape, sigma_xi_scale);
        // Run the sampling loop
        bart_dispatcher.RunSampler(
            *bart_result_ptr_.get(), feature_types_vector, var_weights_vector, 
            num_trees, num_gfr, num_burnin, num_mcmc, global_variance_init, leaf_cov, 
            alpha, beta, nu, lamb, a_leaf, b_leaf, min_samples_leaf, cutpoint_grid_size,
            sample_global_var, sample_leaf_var, random_seed
        );
    }
    
    // Unprotect pointers to R data
    UNPROTECT(5);
    
    // Release management of the pointer to R session
    return cpp11::external_pointer<StochTree::BARTResult>(bart_result_ptr_.release());
}

[[cpp11::register]]
cpp11::external_pointer<StochTree::BARTResult> run_bart_cpp_nobasis_test_norfx(
        cpp11::doubles covariates_train, cpp11::doubles outcome_train, 
        int num_rows_train, int num_covariates_train, 
        cpp11::doubles covariates_test, 
        int num_rows_test, int num_covariates_test, 
        cpp11::integers feature_types, cpp11::doubles variable_weights, 
        int num_trees, int output_dimension, bool is_leaf_constant, 
        double alpha, double beta, double a_leaf, double b_leaf, double nu, double lamb, 
        int min_samples_leaf, int cutpoint_grid_size, cpp11::doubles_matrix<> leaf_cov_init, 
        double global_variance_init, int num_gfr, int num_burnin, int num_mcmc, int random_seed, 
        int leaf_model_int, bool sample_global_var, bool sample_leaf_var
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
    
    // Check inputs
    if (num_covariates_train != num_covariates_test) {
        StochTree::Log::Fatal("num_covariates_train must equal num_covariates_test");
    }
    // if ((leaf_model_int == 1) || (leaf_model_int == 2)) {
    //     StochTree::Log::Fatal("Must provide basis for leaf regression");
    // }
    
    // Create BART dispatcher and add data
    double* train_covariate_data_ptr = REAL(PROTECT(covariates_train));
    double* train_outcome_data_ptr = REAL(PROTECT(outcome_train));
    double* test_covariate_data_ptr = REAL(PROTECT(covariates_test));
    if (leaf_model_int == 0) {
        // Create the dispatcher and load the data
        StochTree::BARTDispatcher<StochTree::GaussianConstantLeafModel> bart_dispatcher{};
        // Load training data
        bart_dispatcher.AddDataset(train_covariate_data_ptr, num_rows_train, num_covariates_train, false, true);
        bart_dispatcher.AddTrainOutcome(train_outcome_data_ptr, num_rows_train);
        // Load test data
        bart_dispatcher.AddDataset(test_covariate_data_ptr, num_rows_test, num_covariates_test, false, false);
        // Run the sampling loop
        bart_dispatcher.RunSampler(
            *bart_result_ptr_.get(), feature_types_vector, var_weights_vector, 
            num_trees, num_gfr, num_burnin, num_mcmc, global_variance_init, leaf_cov, 
            alpha, beta, nu, lamb, a_leaf, b_leaf, min_samples_leaf, cutpoint_grid_size,
            sample_global_var, sample_leaf_var, random_seed
        );
    } else if (leaf_model_int == 1) {
        // Create the dispatcher and load the data
        StochTree::BARTDispatcher<StochTree::GaussianUnivariateRegressionLeafModel> bart_dispatcher{};
        // Load training data
        bart_dispatcher.AddDataset(train_covariate_data_ptr, num_rows_train, num_covariates_train, false, true);
        bart_dispatcher.AddTrainOutcome(train_outcome_data_ptr, num_rows_train);
        // Load test data
        bart_dispatcher.AddDataset(test_covariate_data_ptr, num_rows_test, num_covariates_test, false, false);
        // Run the sampling loop
        bart_dispatcher.RunSampler(
            *bart_result_ptr_.get(), feature_types_vector, var_weights_vector, 
            num_trees, num_gfr, num_burnin, num_mcmc, global_variance_init, leaf_cov, 
            alpha, beta, nu, lamb, a_leaf, b_leaf, min_samples_leaf, cutpoint_grid_size,
            sample_global_var, sample_leaf_var, random_seed
        );
    } else {
        // Create the dispatcher and load the data
        StochTree::BARTDispatcher<StochTree::GaussianMultivariateRegressionLeafModel> bart_dispatcher{};
        // Load training data
        bart_dispatcher.AddDataset(train_covariate_data_ptr, num_rows_train, num_covariates_train, false, true);
        bart_dispatcher.AddTrainOutcome(train_outcome_data_ptr, num_rows_train);
        // Load test data
        bart_dispatcher.AddDataset(test_covariate_data_ptr, num_rows_test, num_covariates_test, false, false);
        // Run the sampling loop
        bart_dispatcher.RunSampler(
            *bart_result_ptr_.get(), feature_types_vector, var_weights_vector, 
            num_trees, num_gfr, num_burnin, num_mcmc, global_variance_init, leaf_cov, 
            alpha, beta, nu, lamb, a_leaf, b_leaf, min_samples_leaf, cutpoint_grid_size,
            sample_global_var, sample_leaf_var, random_seed
        );
    }
    
    // Unprotect pointers to R data
    UNPROTECT(3);
    
    // Release management of the pointer to R session
    return cpp11::external_pointer<StochTree::BARTResult>(bart_result_ptr_.release());
}

[[cpp11::register]]
cpp11::external_pointer<StochTree::BARTResult> run_bart_cpp_nobasis_notest_rfx(
        cpp11::doubles covariates_train, cpp11::doubles outcome_train, 
        int num_rows_train, int num_covariates_train, 
        cpp11::doubles rfx_basis_train, cpp11::integers rfx_group_labels_train, 
        int num_rfx_basis_train, int num_rfx_groups_train,  
        cpp11::integers feature_types, cpp11::doubles variable_weights, 
        int num_trees, int output_dimension, bool is_leaf_constant, 
        double alpha, double beta, double a_leaf, double b_leaf, double nu, double lamb, 
        int min_samples_leaf, int cutpoint_grid_size, cpp11::doubles_matrix<> leaf_cov_init, 
        double global_variance_init, int num_gfr, int num_burnin, int num_mcmc, int random_seed, 
        int leaf_model_int, bool sample_global_var, bool sample_leaf_var, 
        cpp11::doubles rfx_alpha_init, cpp11::doubles_matrix<> rfx_xi_init, 
        cpp11::doubles_matrix<> rfx_sigma_alpha_init, cpp11::doubles_matrix<> rfx_sigma_xi_init, 
        double rfx_sigma_xi_shape, double rfx_sigma_xi_scale
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
    
    // Check inputs
    // if ((leaf_model_int == 1) || (leaf_model_int == 2)) {
    //     StochTree::Log::Fatal("Must provide basis for leaf regression");
    // }
    
    // Convert rfx group IDs to std::vector
    std::vector<int> rfx_group_labels_train_cpp;
    rfx_group_labels_train_cpp.resize(rfx_group_labels_train.size());
    for (int i = 0; i < rfx_group_labels_train.size(); i++) {
        rfx_group_labels_train_cpp.at(i) = rfx_group_labels_train.at(i);
    }
    
    // Unpack RFX terms
    Eigen::VectorXd alpha_init;
    Eigen::MatrixXd xi_init;
    Eigen::MatrixXd sigma_alpha_init;
    Eigen::MatrixXd sigma_xi_init;
    double sigma_xi_shape;
    double sigma_xi_scale;
    alpha_init.resize(rfx_alpha_init.size());
    xi_init.resize(rfx_xi_init.nrow(), rfx_xi_init.ncol());
    sigma_alpha_init.resize(rfx_sigma_alpha_init.nrow(), rfx_sigma_alpha_init.ncol());
    sigma_xi_init.resize(rfx_sigma_xi_init.nrow(), rfx_sigma_xi_init.ncol());
    for (int i = 0; i < rfx_alpha_init.size(); i++) {
        alpha_init(i) = rfx_alpha_init.at(i);
    }
    for (int i = 0; i < rfx_xi_init.nrow(); i++) {
        for (int j = 0; j < rfx_xi_init.ncol(); j++) {
            xi_init(i,j) = rfx_xi_init(i,j);
        }
    }
    for (int i = 0; i < rfx_sigma_alpha_init.nrow(); i++) {
        for (int j = 0; j < rfx_sigma_alpha_init.ncol(); j++) {
            sigma_alpha_init(i,j) = rfx_sigma_alpha_init(i,j);
        }
    }
    for (int i = 0; i < rfx_sigma_xi_init.nrow(); i++) {
        for (int j = 0; j < rfx_sigma_xi_init.ncol(); j++) {
            sigma_xi_init(i,j) = rfx_sigma_xi_init(i,j);
        }
    }
    sigma_xi_shape = rfx_sigma_xi_shape;
    sigma_xi_scale = rfx_sigma_xi_scale;
    
    // Create BART dispatcher and add data
    double* train_covariate_data_ptr = REAL(PROTECT(covariates_train));
    double* train_outcome_data_ptr = REAL(PROTECT(outcome_train));
    double* train_rfx_basis_data_ptr = REAL(PROTECT(rfx_basis_train));
    if (leaf_model_int == 0) {
        // Create the dispatcher and load the data
        StochTree::BARTDispatcher<StochTree::GaussianConstantLeafModel> bart_dispatcher{};
        // Load training data
        bart_dispatcher.AddDataset(train_covariate_data_ptr, num_rows_train, num_covariates_train, false, true);
        bart_dispatcher.AddTrainOutcome(train_outcome_data_ptr, num_rows_train);
        // Load rfx data
        bart_dispatcher.AddRFXTerm(train_rfx_basis_data_ptr, rfx_group_labels_train_cpp, num_rows_train, 
                                   num_rfx_groups_train, num_rfx_basis_train, false, true, alpha_init, 
                                   xi_init, sigma_alpha_init, sigma_xi_init, sigma_xi_shape, sigma_xi_scale);
        // Run the sampling loop
        bart_dispatcher.RunSampler(
            *bart_result_ptr_.get(), feature_types_vector, var_weights_vector, 
            num_trees, num_gfr, num_burnin, num_mcmc, global_variance_init, leaf_cov, 
            alpha, beta, nu, lamb, a_leaf, b_leaf, min_samples_leaf, cutpoint_grid_size,
            sample_global_var, sample_leaf_var, random_seed
        );
    } else if (leaf_model_int == 1) {
        // Create the dispatcher and load the data
        StochTree::BARTDispatcher<StochTree::GaussianUnivariateRegressionLeafModel> bart_dispatcher{};
        // Load training data
        bart_dispatcher.AddDataset(train_covariate_data_ptr, num_rows_train, num_covariates_train, false, true);
        bart_dispatcher.AddTrainOutcome(train_outcome_data_ptr, num_rows_train);
        // Load rfx data
        bart_dispatcher.AddRFXTerm(train_rfx_basis_data_ptr, rfx_group_labels_train_cpp, num_rows_train, 
                                   num_rfx_groups_train, num_rfx_basis_train, false, true, alpha_init, 
                                   xi_init, sigma_alpha_init, sigma_xi_init, sigma_xi_shape, sigma_xi_scale);
        // Run the sampling loop
        bart_dispatcher.RunSampler(
            *bart_result_ptr_.get(), feature_types_vector, var_weights_vector, 
            num_trees, num_gfr, num_burnin, num_mcmc, global_variance_init, leaf_cov, 
            alpha, beta, nu, lamb, a_leaf, b_leaf, min_samples_leaf, cutpoint_grid_size,
            sample_global_var, sample_leaf_var, random_seed
        );
    } else {
        // Create the dispatcher and load the data
        StochTree::BARTDispatcher<StochTree::GaussianMultivariateRegressionLeafModel> bart_dispatcher{};
        // Load training data
        bart_dispatcher.AddDataset(train_covariate_data_ptr, num_rows_train, num_covariates_train, false, true);
        bart_dispatcher.AddTrainOutcome(train_outcome_data_ptr, num_rows_train);
        // Load rfx data
        bart_dispatcher.AddRFXTerm(train_rfx_basis_data_ptr, rfx_group_labels_train_cpp, num_rows_train, 
                                   num_rfx_groups_train, num_rfx_basis_train, false, true, alpha_init, 
                                   xi_init, sigma_alpha_init, sigma_xi_init, sigma_xi_shape, sigma_xi_scale);
        // Run the sampling loop
        bart_dispatcher.RunSampler(
            *bart_result_ptr_.get(), feature_types_vector, var_weights_vector, 
            num_trees, num_gfr, num_burnin, num_mcmc, global_variance_init, leaf_cov, 
            alpha, beta, nu, lamb, a_leaf, b_leaf, min_samples_leaf, cutpoint_grid_size,
            sample_global_var, sample_leaf_var, random_seed
        );
    }
    
    // Unprotect pointers to R data
    UNPROTECT(3);
    
    // Release management of the pointer to R session
    return cpp11::external_pointer<StochTree::BARTResult>(bart_result_ptr_.release());
}

[[cpp11::register]]
cpp11::external_pointer<StochTree::BARTResult> run_bart_cpp_nobasis_notest_norfx(
        cpp11::doubles covariates_train, cpp11::doubles outcome_train, 
        int num_rows_train, int num_covariates_train, 
        cpp11::integers feature_types, cpp11::doubles variable_weights, 
        int num_trees, int output_dimension, bool is_leaf_constant, 
        double alpha, double beta, double a_leaf, double b_leaf, double nu, double lamb, 
        int min_samples_leaf, int cutpoint_grid_size, cpp11::doubles_matrix<> leaf_cov_init, 
        double global_variance_init, int num_gfr, int num_burnin, int num_mcmc, int random_seed, 
        int leaf_model_int, bool sample_global_var, bool sample_leaf_var
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
    
    // Check inputs
    // if ((leaf_model_int == 1) || (leaf_model_int == 2)) {
    //     StochTree::Log::Fatal("Must provide basis for leaf regression");
    // }
    
    // Create BART dispatcher and add data
    double* train_covariate_data_ptr = REAL(PROTECT(covariates_train));
    double* train_outcome_data_ptr = REAL(PROTECT(outcome_train));
    if (leaf_model_int == 0) {
        // Create the dispatcher and load the data
        StochTree::BARTDispatcher<StochTree::GaussianConstantLeafModel> bart_dispatcher{};
        // Load training data
        bart_dispatcher.AddDataset(train_covariate_data_ptr, num_rows_train, num_covariates_train, false, true);
        bart_dispatcher.AddTrainOutcome(train_outcome_data_ptr, num_rows_train);
        // Run the sampling loop
        bart_dispatcher.RunSampler(
            *bart_result_ptr_.get(), feature_types_vector, var_weights_vector, 
            num_trees, num_gfr, num_burnin, num_mcmc, global_variance_init, leaf_cov, 
            alpha, beta, nu, lamb, a_leaf, b_leaf, min_samples_leaf, cutpoint_grid_size,
            sample_global_var, sample_leaf_var, random_seed
        );
    } else if (leaf_model_int == 1) {
        // Create the dispatcher and load the data
        StochTree::BARTDispatcher<StochTree::GaussianUnivariateRegressionLeafModel> bart_dispatcher{};
        // Load training data
        bart_dispatcher.AddDataset(train_covariate_data_ptr, num_rows_train, num_covariates_train, false, true);
        bart_dispatcher.AddTrainOutcome(train_outcome_data_ptr, num_rows_train);
        // Run the sampling loop
        bart_dispatcher.RunSampler(
            *bart_result_ptr_.get(), feature_types_vector, var_weights_vector, 
            num_trees, num_gfr, num_burnin, num_mcmc, global_variance_init, leaf_cov, 
            alpha, beta, nu, lamb, a_leaf, b_leaf, min_samples_leaf, cutpoint_grid_size,
            sample_global_var, sample_leaf_var, random_seed
        );
    } else {
        // Create the dispatcher and load the data
        StochTree::BARTDispatcher<StochTree::GaussianMultivariateRegressionLeafModel> bart_dispatcher{};
        // Load training data
        bart_dispatcher.AddDataset(train_covariate_data_ptr, num_rows_train, num_covariates_train, false, true);
        bart_dispatcher.AddTrainOutcome(train_outcome_data_ptr, num_rows_train);
        // Run the sampling loop
        bart_dispatcher.RunSampler(
            *bart_result_ptr_.get(), feature_types_vector, var_weights_vector, 
            num_trees, num_gfr, num_burnin, num_mcmc, global_variance_init, leaf_cov, 
            alpha, beta, nu, lamb, a_leaf, b_leaf, min_samples_leaf, cutpoint_grid_size,
            sample_global_var, sample_leaf_var, random_seed
        );
    }
    
    // Unprotect pointers to R data
    UNPROTECT(2);
    
    // Release management of the pointer to R session
    return cpp11::external_pointer<StochTree::BARTResult>(bart_result_ptr_.release());
}
