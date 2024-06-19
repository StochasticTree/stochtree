#include <cpp11.hpp>
#include "stochtree_types.h"
#include <stochtree/kernel.h>
#include <stochtree/log.h>
#include <Eigen/Dense>
#include <Eigen/Dense>
#include <memory>
#include <vector>

typedef Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> KernelMatrixType;

[[cpp11::register]]
cpp11::external_pointer<StochTree::ForestKernel> forest_kernel_cpp() {
    // Create smart pointer to newly allocated object
    std::unique_ptr<StochTree::ForestKernel> forest_kernel_ptr_ = std::make_unique<StochTree::ForestKernel>();
    
    // Release management of the pointer to R session
    return cpp11::external_pointer<StochTree::ForestKernel>(forest_kernel_ptr_.release());
}

[[cpp11::register]]
void forest_kernel_compute_leaf_indices_train_cpp(cpp11::external_pointer<StochTree::ForestKernel> forest_kernel, cpp11::doubles_matrix<> covariates_train, 
                                                  cpp11::external_pointer<StochTree::ForestContainer> forest_container, int forest_num) {
    // Wrap an Eigen Map around the raw data of the covariate_train matrix
    StochTree::data_size_t n_train = covariates_train.nrow();
    int num_covariates_train = covariates_train.ncol();
    double* covariate_train_data_ptr = REAL(PROTECT(covariates_train));
    KernelMatrixType covariates_train_eigen(covariate_train_data_ptr, n_train, num_covariates_train);
    
    // Compute leaf indices
    forest_kernel->ComputeLeafIndices(covariates_train_eigen, *(forest_container->GetEnsemble(forest_num)));
    
    // Unprotect pointers to R data
    UNPROTECT(1);
}

[[cpp11::register]]
void forest_kernel_compute_leaf_indices_train_test_cpp(cpp11::external_pointer<StochTree::ForestKernel> forest_kernel, cpp11::doubles_matrix<> covariates_train, cpp11::doubles_matrix<> covariates_test, 
                                                       cpp11::external_pointer<StochTree::ForestContainer> forest_container, int forest_num) {
    // Wrap an Eigen Map around the raw data of the covariate_train matrix
    StochTree::data_size_t n_train = covariates_train.nrow();
    int num_covariates_train = covariates_train.ncol();
    double* covariate_train_data_ptr = REAL(PROTECT(covariates_train));
    KernelMatrixType covariates_train_eigen(covariate_train_data_ptr, n_train, num_covariates_train);
    
    // Wrap an Eigen Map around the raw data of the covariate_test matrix
    StochTree::data_size_t n_test = covariates_test.nrow();
    int num_covariates_test = covariates_test.ncol();
    double* covariate_test_data_ptr = REAL(PROTECT(covariates_test));
    KernelMatrixType covariates_test_eigen(covariate_test_data_ptr, n_test, num_covariates_test);
    
    // Compute leaf indices
    forest_kernel->ComputeLeafIndices(covariates_train_eigen, covariates_test_eigen, *(forest_container->GetEnsemble(forest_num)));
    
    // Unprotect pointers to R data
    UNPROTECT(2);
}

[[cpp11::register]]
cpp11::writable::integers forest_kernel_get_train_leaf_indices_cpp(cpp11::external_pointer<StochTree::ForestKernel> forest_kernel) {
    if (!forest_kernel->HasTrainLeafIndices()) {
        cpp11::writable::integers output;
        return output;
    }
    std::vector<int32_t> train_indices = forest_kernel->GetTrainLeafIndices();
    cpp11::writable::integers output(train_indices.begin(), train_indices.end());
    return output;
}

[[cpp11::register]]
cpp11::writable::integers forest_kernel_get_test_leaf_indices_cpp(cpp11::external_pointer<StochTree::ForestKernel> forest_kernel) {
    if (!forest_kernel->HasTestLeafIndices()) {
        cpp11::writable::integers output;
        return output;
    }
    std::vector<int32_t> test_indices = forest_kernel->GetTestLeafIndices();
    cpp11::writable::integers output(test_indices.begin(), test_indices.end());
    return output;
}

[[cpp11::register]]
cpp11::list forest_kernel_compute_kernel_train_cpp(
    cpp11::external_pointer<StochTree::ForestKernel> forest_kernel, cpp11::doubles_matrix<> covariates_train, 
    cpp11::external_pointer<StochTree::ForestContainer> forest_container, int forest_num
) {
    // Wrap an Eigen Map around the raw data of the covariate_train matrix
    StochTree::data_size_t n_train = covariates_train.nrow();
    int num_covariates_train = covariates_train.ncol();
    double* covariate_train_data_ptr = REAL(PROTECT(covariates_train));
    KernelMatrixType covariates_train_eigen(covariate_train_data_ptr, n_train, num_covariates_train);
    
    // Declare outputs
    cpp11::writable::doubles_matrix<> kernel_train(n_train, n_train);
    
    // Wrap Eigen Maps around kernel and kernel inverse matrices
    double* kernel_data_ptr = REAL(PROTECT(kernel_train));
    KernelMatrixType kernel_eigen(kernel_data_ptr, n_train, n_train);

    // Compute kernel terms
    forest_kernel->ComputeKernelExternal(covariates_train_eigen, *(forest_container->GetEnsemble(forest_num)), kernel_eigen);
    
    // Unprotect pointers to R data
    UNPROTECT(2);
    
    // Return list of vectors
    cpp11::writable::list result;
    result.push_back(kernel_train);
    return result;
}

[[cpp11::register]]
cpp11::list forest_kernel_compute_kernel_train_test_cpp(
    cpp11::external_pointer<StochTree::ForestKernel> forest_kernel, cpp11::doubles_matrix<> covariates_train, 
    cpp11::doubles_matrix<> covariates_test, cpp11::external_pointer<StochTree::ForestContainer> forest_container, int forest_num
) {
    // Wrap an Eigen Map around the raw data of the covariate_train matrix
    StochTree::data_size_t n_train = covariates_train.nrow();
    int num_covariates_train = covariates_train.ncol();
    double* covariate_train_data_ptr = REAL(PROTECT(covariates_train));
    KernelMatrixType covariates_train_eigen(covariate_train_data_ptr, n_train, num_covariates_train);
    
    // Wrap an Eigen Map around the raw data of the covariate_test matrix
    StochTree::data_size_t n_test = covariates_test.nrow();
    int num_covariates_test = covariates_test.ncol();
    double* covariate_test_data_ptr = REAL(PROTECT(covariates_test));
    KernelMatrixType covariates_test_eigen(covariate_test_data_ptr, n_test, num_covariates_test);
    
    // Declare outputs
    cpp11::writable::doubles_matrix<> kernel_train(n_train, n_train);
    cpp11::writable::doubles_matrix<> kernel_test_train(n_test, n_train);
    cpp11::writable::doubles_matrix<> kernel_test(n_test, n_test);
    
    // Wrap Eigen Maps around kernel and kernel inverse matrices
    double* kernel_data_ptr = REAL(PROTECT(kernel_train));
    double* kernel_test_train_data_ptr = REAL(PROTECT(kernel_test_train));
    double* kernel_test_data_ptr = REAL(PROTECT(kernel_test));
    KernelMatrixType kernel_train_eigen(kernel_data_ptr, n_train, n_train);
    KernelMatrixType kernel_test_train_eigen(kernel_test_train_data_ptr, n_test, n_train);
    KernelMatrixType kernel_test_eigen(kernel_test_data_ptr, n_test, n_test);
    
    // Compute kernel terms
    forest_kernel->ComputeKernelExternal(covariates_train_eigen, covariates_test_eigen, *(forest_container->GetEnsemble(forest_num)), kernel_train_eigen, kernel_test_train_eigen, kernel_test_eigen);
    
    // Unprotect pointers to R data
    UNPROTECT(5);
    
    // Return list of vectors
    cpp11::writable::list result;
    result.push_back(kernel_train);
    result.push_back(kernel_test_train);
    result.push_back(kernel_test);
    return result;
}
