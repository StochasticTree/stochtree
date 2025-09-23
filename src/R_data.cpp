#include "cpp11/integers.hpp"
#include <cpp11.hpp>
#include <stochtree/container.h>
#include <stochtree/data.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/tree_sampler.h>
#include <memory>

[[cpp11::register]]
cpp11::external_pointer<StochTree::ForestDataset> create_forest_dataset_cpp() {
    // Create smart pointer to newly allocated object
    std::unique_ptr<StochTree::ForestDataset> dataset_ptr_ = std::make_unique<StochTree::ForestDataset>();
    
    // Release management of the pointer to R session
    return cpp11::external_pointer<StochTree::ForestDataset>(dataset_ptr_.release());
}

[[cpp11::register]]
int dataset_num_rows_cpp(cpp11::external_pointer<StochTree::ForestDataset> dataset) {
    return dataset->NumObservations();
}

[[cpp11::register]]
int dataset_num_covariates_cpp(cpp11::external_pointer<StochTree::ForestDataset> dataset) {
    return dataset->NumCovariates();
}

[[cpp11::register]]
int dataset_num_basis_cpp(cpp11::external_pointer<StochTree::ForestDataset> dataset) {
    return dataset->NumBasis();
}

[[cpp11::register]]
bool dataset_has_basis_cpp(cpp11::external_pointer<StochTree::ForestDataset> dataset) {
    return dataset->HasBasis();
}

[[cpp11::register]]
bool dataset_has_variance_weights_cpp(cpp11::external_pointer<StochTree::ForestDataset> dataset) {
    return dataset->HasVarWeights();
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
void forest_dataset_update_basis_cpp(cpp11::external_pointer<StochTree::ForestDataset> dataset_ptr, cpp11::doubles_matrix<> basis) {
    // TODO: add handling code on the R side to ensure matrices are column-major
    bool row_major{false};
    
    // Add basis
    StochTree::data_size_t n = basis.nrow();
    int num_basis = basis.ncol();
    double* basis_data_ptr = REAL(PROTECT(basis));
    dataset_ptr->UpdateBasis(basis_data_ptr, n, num_basis, row_major);
    
    // Unprotect pointers to R data
    UNPROTECT(1);
}

[[cpp11::register]]
void forest_dataset_update_var_weights_cpp(cpp11::external_pointer<StochTree::ForestDataset> dataset_ptr, cpp11::doubles weights, bool exponentiate) {
    // Add weights
    StochTree::data_size_t n = weights.size();
    double* weight_data_ptr = REAL(PROTECT(weights));
    dataset_ptr->UpdateVarWeights(weight_data_ptr, n, exponentiate);
    
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
cpp11::writable::doubles_matrix<> forest_dataset_get_covariates_cpp(cpp11::external_pointer<StochTree::ForestDataset> dataset_ptr) {
    // Initialize output matrix
    int num_row = dataset_ptr->NumObservations();
    int num_col = dataset_ptr->NumCovariates();
    cpp11::writable::doubles_matrix<> output(num_row, num_col);

    for (int i = 0; i < num_row; i++) {
        for (int j = 0; j < num_col; j++) {
            output(i, j) = dataset_ptr->CovariateValue(i, j);
        }
    }

    return output;
}

[[cpp11::register]]
cpp11::writable::doubles_matrix<> forest_dataset_get_basis_cpp(cpp11::external_pointer<StochTree::ForestDataset> dataset_ptr) {
    // Initialize output matrix
    int num_row = dataset_ptr->NumObservations();
    int num_col = dataset_ptr->NumBasis();
    cpp11::writable::doubles_matrix<> output(num_row, num_col);
    for (int i = 0; i < num_row; i++) {
        for (int j = 0; j < num_col; j++) {
            output(i, j) = dataset_ptr->BasisValue(i, j);
        }
    }   
    return output;
}

[[cpp11::register]]
cpp11::writable::doubles forest_dataset_get_variance_weights_cpp(cpp11::external_pointer<StochTree::ForestDataset> dataset_ptr) {
    // Initialize output vector
    int num_row = dataset_ptr->NumObservations();
    cpp11::writable::doubles output(num_row);
    for (int i = 0; i < num_row; i++) {
        output.at(i) = dataset_ptr->VarWeightValue(i);
    }
    return output;
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

[[cpp11::register]]
void add_to_column_vector_cpp(cpp11::external_pointer<StochTree::ColumnVector> outcome, cpp11::doubles update_vector) {
    // Unpack pointers to data and dimensions
    StochTree::data_size_t n = update_vector.size();
    double* update_data_ptr = REAL(PROTECT(update_vector));
    
    // Add to the outcome data using the C++ API
    outcome->AddToData(update_data_ptr, n);
    
    // Unprotect pointers to R data
    UNPROTECT(1);
}

[[cpp11::register]]
void subtract_from_column_vector_cpp(cpp11::external_pointer<StochTree::ColumnVector> outcome, cpp11::doubles update_vector) {
    // Unpack pointers to data and dimensions
    StochTree::data_size_t n = update_vector.size();
    double* update_data_ptr = REAL(PROTECT(update_vector));
    
    // Add to the outcome data using the C++ API
    outcome->SubtractFromData(update_data_ptr, n);
    
    // Unprotect pointers to R data
    UNPROTECT(1);
}

[[cpp11::register]]
void overwrite_column_vector_cpp(cpp11::external_pointer<StochTree::ColumnVector> outcome, cpp11::doubles new_vector) {
    // Unpack pointers to data and dimensions
    StochTree::data_size_t n = new_vector.size();
    double* update_data_ptr = REAL(PROTECT(new_vector));
    
    // Add to the outcome data using the C++ API
    outcome->OverwriteData(update_data_ptr, n);
    
    // Unprotect pointers to R data
    UNPROTECT(1);
}

[[cpp11::register]]
void propagate_trees_column_vector_cpp(cpp11::external_pointer<StochTree::ForestTracker> tracker, 
                                       cpp11::external_pointer<StochTree::ColumnVector> residual) {
    StochTree::UpdateResidualNewOutcome(*tracker, *residual);
}

[[cpp11::register]]
cpp11::writable::doubles get_residual_cpp(cpp11::external_pointer<StochTree::ColumnVector> vector_ptr) {
    // Initialize output vector
    StochTree::data_size_t n = vector_ptr->NumRows();
    cpp11::writable::doubles output(n);
    
    // Unpack data
    for (StochTree::data_size_t i = 0; i < n; i++) {
        output.at(i) = vector_ptr->GetElement(i);
    }
    
    // Release management of the pointer to R session
    return output;
}

[[cpp11::register]]
cpp11::external_pointer<StochTree::RandomEffectsDataset> create_rfx_dataset_cpp() {
    // Create smart pointer to newly allocated object
    std::unique_ptr<StochTree::RandomEffectsDataset> dataset_ptr_ = std::make_unique<StochTree::RandomEffectsDataset>();
    
    // Release management of the pointer to R session
    return cpp11::external_pointer<StochTree::RandomEffectsDataset>(dataset_ptr_.release());
}

[[cpp11::register]]
void rfx_dataset_update_basis_cpp(cpp11::external_pointer<StochTree::RandomEffectsDataset> dataset_ptr, cpp11::doubles_matrix<> basis) {
    // TODO: add handling code on the R side to ensure matrices are column-major
    bool row_major{false};
    
    // Add basis
    StochTree::data_size_t n = basis.nrow();
    int num_basis = basis.ncol();
    double* basis_data_ptr = REAL(PROTECT(basis));
    dataset_ptr->UpdateBasis(basis_data_ptr, n, num_basis, row_major);
    
    // Unprotect pointers to R data
    UNPROTECT(1);
}

[[cpp11::register]]
void rfx_dataset_update_var_weights_cpp(cpp11::external_pointer<StochTree::RandomEffectsDataset> dataset_ptr, cpp11::doubles weights, bool exponentiate) {
    // Add weights
    StochTree::data_size_t n = weights.size();
    double* weight_data_ptr = REAL(PROTECT(weights));
    dataset_ptr->UpdateVarWeights(weight_data_ptr, n, exponentiate);
    
    // Unprotect pointers to R data
    UNPROTECT(1);
}

[[cpp11::register]]
void rfx_dataset_update_group_labels_cpp(cpp11::external_pointer<StochTree::RandomEffectsDataset> dataset_ptr, cpp11::integers group_labels) {
    // Update group labels
    int n = group_labels.size();
    std::vector<int32_t> group_labels_vec(group_labels.begin(), group_labels.end());
    dataset_ptr->UpdateGroupLabels(group_labels_vec, n);
}

[[cpp11::register]]
int rfx_dataset_num_basis_cpp(cpp11::external_pointer<StochTree::RandomEffectsDataset> dataset) {
    return dataset->NumBases();
}

[[cpp11::register]]
int rfx_dataset_num_rows_cpp(cpp11::external_pointer<StochTree::RandomEffectsDataset> dataset) {
    return dataset->NumObservations();
}

[[cpp11::register]]
bool rfx_dataset_has_group_labels_cpp(cpp11::external_pointer<StochTree::RandomEffectsDataset> dataset) {
    return dataset->HasGroupLabels();
}

[[cpp11::register]]
bool rfx_dataset_has_basis_cpp(cpp11::external_pointer<StochTree::RandomEffectsDataset> dataset) {
    return dataset->HasBasis();
}

[[cpp11::register]]
bool rfx_dataset_has_variance_weights_cpp(cpp11::external_pointer<StochTree::RandomEffectsDataset> dataset) {
    return dataset->HasVarWeights();
}

[[cpp11::register]]
void rfx_dataset_add_group_labels_cpp(cpp11::external_pointer<StochTree::RandomEffectsDataset> dataset_ptr, cpp11::integers group_labels) {
    // Add group labels
    std::vector<int32_t> group_labels_vec(group_labels.begin(), group_labels.end());
    dataset_ptr->AddGroupLabels(group_labels_vec);
}

[[cpp11::register]]
void rfx_dataset_add_basis_cpp(cpp11::external_pointer<StochTree::RandomEffectsDataset> dataset_ptr, cpp11::doubles_matrix<> basis) {
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
void rfx_dataset_add_weights_cpp(cpp11::external_pointer<StochTree::RandomEffectsDataset> dataset_ptr, cpp11::doubles weights) {
    // Add weights
    StochTree::data_size_t n = weights.size();
    double* weight_data_ptr = REAL(PROTECT(weights));
    dataset_ptr->AddVarianceWeights(weight_data_ptr, n);
    
    // Unprotect pointers to R data
    UNPROTECT(1);
}

[[cpp11::register]]
cpp11::writable::integers rfx_dataset_get_group_labels_cpp(cpp11::external_pointer<StochTree::RandomEffectsDataset> dataset_ptr) {
    int num_row = dataset_ptr->NumObservations();
    cpp11::writable::integers output(num_row);
    for (int i = 0; i < num_row; i++) {
        output.at(i) = dataset_ptr->GroupId(i);
    }
    return output;
}

[[cpp11::register]]
cpp11::writable::doubles_matrix<> rfx_dataset_get_basis_cpp(cpp11::external_pointer<StochTree::RandomEffectsDataset> dataset_ptr) {
    int num_row = dataset_ptr->NumObservations();
    int num_col = dataset_ptr->NumBases();
    cpp11::writable::doubles_matrix<> output(num_row, num_col);
    for (int i = 0; i < num_row; i++) {
        for (int j = 0; j < num_col; j++) {
            output(i, j) = dataset_ptr->BasisValue(i, j);
        }
    }   
    return output;
}

[[cpp11::register]]
cpp11::writable::doubles rfx_dataset_get_variance_weights_cpp(cpp11::external_pointer<StochTree::RandomEffectsDataset> dataset_ptr) {
    int num_row = dataset_ptr->NumObservations();
    cpp11::writable::doubles output(num_row);
    for (int i = 0; i < num_row; i++) {
        output.at(i) = dataset_ptr->VarWeightValue(i);
    }
    return output;
}
