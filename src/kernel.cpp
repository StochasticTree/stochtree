#include <cpp11.hpp>
#include "stochtree_types.h"
#include <stochtree/log.h>
#include <Eigen/Dense>
#include <Eigen/Dense>
#include <memory>
#include <vector>

typedef Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> DoubleMatrixType;
typedef Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> IntMatrixType;

[[cpp11::register]]
int forest_container_get_max_leaf_index_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_container, int forest_num) {
    return forest_container->GetEnsemble(forest_num)->GetMaxLeafIndex();
}

[[cpp11::register]]
cpp11::writable::integers_matrix<> compute_leaf_indices_cpp(
        cpp11::external_pointer<StochTree::ForestContainer> forest_container, 
        cpp11::doubles_matrix<> covariates, cpp11::integers forest_nums
) {
    // Wrap an Eigen Map around the raw data of the covariate matrix
    StochTree::data_size_t num_obs = covariates.nrow();
    int num_covariates = covariates.ncol();
    double* covariate_data_ptr = REAL(PROTECT(covariates));
    DoubleMatrixType covariates_eigen(covariate_data_ptr, num_obs, num_covariates);
    
    // Extract other output dimensions
    int num_trees = forest_container->NumTrees();
    int num_samples = forest_nums.size();

    // Declare outputs
    cpp11::writable::integers_matrix<> output_matrix(num_obs*num_trees, num_samples);

    // Wrap Eigen Maps around kernel and kernel inverse matrices
    int* output_data_ptr = INTEGER(PROTECT(output_matrix));
    IntMatrixType output_eigen(output_data_ptr, num_obs*num_trees, num_samples);
    
    // Compute leaf indices
    std::vector<int> forest_indices(forest_nums.begin(), forest_nums.end());
    forest_container->PredictLeafIndicesInplace(covariates_eigen, output_eigen, forest_indices, num_trees, num_obs);

    // Unprotect pointers to R data
    UNPROTECT(2);
    
    // Return matrix
    return output_matrix;
}
