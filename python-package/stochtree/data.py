"""
Python classes wrapping C++ data objects
"""
import numpy as np
from stochtree_cpp import ForestDatasetCpp, ResidualCpp

class Dataset:
    def __init__(self) -> None:
        # Initialize a ForestDatasetCpp object
        self.dataset_cpp = ForestDatasetCpp()
    
    def add_covariates(self, covariates: np.array):
        """
        Add covariates to a dataset
        """
        n, p = covariates.shape
        covariates_rowmajor = np.ascontiguousarray(covariates)
        self.dataset_cpp.AddCovariates(covariates_rowmajor, n, p, True)
    
    def add_basis(self, basis: np.array):
        """
        Add basis matrix to a dataset
        """
        n, p = basis.shape
        basis_rowmajor = np.ascontiguousarray(basis)
        self.dataset_cpp.AddBasis(basis_rowmajor, n, p, True)
    
    def add_variance_weights(self, variance_weights: np.array):
        """
        Add variance weights to a dataset
        """
        n = variance_weights.size
        self.dataset_cpp.AddVarianceWeights(variance_weights, n)


class Residual:
    def __init__(self, residual: np.array) -> None:
        # Initialize a ResidualCpp object
        n = residual.size
        self.residual_cpp = ResidualCpp(residual, n)
