#include <cpp11.hpp>
#include <stochtree/data.h>
#include <memory>
#include <vector>
using namespace cpp11;

[[cpp11::register]]
cpp11::external_pointer<StochTree::ForestDataset> create_forest_dataset_cpp() {
    // Create smart pointer to newly allocated object
    std::unique_ptr<StochTree::ForestDataset> dataset_ptr_ = std::make_unique<StochTree::ForestDataset>();
    
    // Release management of the pointer to R session
    return cpp11::external_pointer<StochTree::ForestDataset>(dataset_ptr_.release());
}

[[cpp11::register]]
int num_dataset_rows(cpp11::external_pointer<StochTree::ForestDataset> dataset) {
    return dataset->NumObservations();
}

[[cpp11::register]]
void forest_dataset_add_covariates_cpp(cpp11::external_pointer<StochTree::ForestDataset> dataset_ptr, cpp11::doubles_matrix<> covariates) {
    // TODO: add handling code on the R side to ensure matrices are column-major
    bool row_major{false};

    // Add covariates
    StochTree::data_size_t n = covariates.nrow();
    int num_covariates = covariates.ncol();
    double* covariate_data_ptr = REAL(PROTECT(covariates));
    dataset_ptr->AddCovariates(covariate_data_ptr, n, num_covariates, row_major);
    
    // Unprotect pointers to R data
    UNPROTECT(1);
}

[[cpp11::register]]
void forest_dataset_add_basis_cpp(cpp11::external_pointer<StochTree::ForestDataset> dataset_ptr, cpp11::doubles_matrix<> basis) {
    // TODO: add handling code on the R side to ensure matrices are column-major
    bool row_major{false};

    // Add basis
    StochTree::data_size_t n = basis.nrow();
    int num_basis = basis.ncol();
    double* basis_data_ptr = REAL(PROTECT(basis));
    dataset_ptr->AddBasis(basis_data_ptr, n, num_basis, row_major);
    
    // Unprotect pointers to R data
    UNPROTECT(1);
}

[[cpp11::register]]
void forest_dataset_add_weights_cpp(cpp11::external_pointer<StochTree::ForestDataset> dataset_ptr, cpp11::doubles weights) {
    // Add weights
    StochTree::data_size_t n = weights.size();
    double* weight_data_ptr = REAL(PROTECT(weights));
    dataset_ptr->AddVarianceWeights(weight_data_ptr, n);

    // Unprotect pointers to R data
    UNPROTECT(1);
}

[[cpp11::register]]
cpp11::external_pointer<StochTree::ColumnVector> create_column_vector_cpp(cpp11::doubles outcome) {
    // Unpack pointers to data and dimensions
    StochTree::data_size_t n = outcome.size();
    double* outcome_data_ptr = REAL(PROTECT(outcome));

    // Create smart pointer
    std::unique_ptr<StochTree::ColumnVector> vector_ptr_ = std::make_unique<StochTree::ColumnVector>(outcome_data_ptr, n);
    
    // Unprotect pointers to R data
    UNPROTECT(1);
    
    // Release management of the pointer to R session
    return cpp11::external_pointer<StochTree::ColumnVector>(vector_ptr_.release());
}
