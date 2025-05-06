import numpy as np
from stochtree_cpp import ForestDatasetCpp, ResidualCpp


class Dataset:
    """
    Wrapper around a C++ class that stores all of the non-outcome data used in `stochtree`. This includes:

    1. Features used for partitioning (also referred to as "covariates" in many places in these docs).
    2. Basis vectors used to define non-constant leaf models. This is optional but may be included via the `add_basis` method.
    3. Variance weights used to define heteroskedastic or otherwise weighted models. This is optional but may be included via the `add_variance_weights` method.
    """

    def __init__(self) -> None:
        """
        Initialize a `Dataset` object
        """
        self.dataset_cpp = ForestDatasetCpp()

    def add_covariates(self, covariates: np.array):
        """
        Add covariates to a dataset

        Parameters
        ----------
        covariates : np.array
            Numpy array of covariates. If data contain categorical, string, time series, or other columns in a
            dataframe, please first preprocess using the `CovariateTransformer`.
        """
        covariates_ = (
            np.expand_dims(covariates, 1) if np.ndim(covariates) == 1 else covariates
        )
        n, p = covariates_.shape
        covariates_rowmajor = np.ascontiguousarray(covariates)
        self.dataset_cpp.AddCovariates(covariates_rowmajor, n, p, True)

    def add_basis(self, basis: np.array):
        """
        Add basis matrix to a dataset

        Parameters
        ----------
        basis : np.array
            Numpy array of basis vectors.
        """
        basis_ = np.expand_dims(basis, 1) if np.ndim(basis) == 1 else basis
        n, p = basis_.shape
        basis_rowmajor = np.ascontiguousarray(basis_)
        self.dataset_cpp.AddBasis(basis_rowmajor, n, p, True)

    def update_basis(self, basis: np.array):
        """
        Update basis matrix in a dataset. Allows users to build an ensemble whose leaves
        regress on bases that are updated throughout the sampler.

        Parameters
        ----------
        basis : np.array
            Numpy array of basis vectors.
        """
        basis_ = np.expand_dims(basis, 1) if np.ndim(basis) == 1 else basis
        n, p = basis_.shape
        basis_rowmajor = np.ascontiguousarray(basis_)
        self.dataset_cpp.UpdateBasis(basis_rowmajor, n, p, True)

    def add_variance_weights(self, variance_weights: np.array):
        """
        Add variance weights to a dataset

        Parameters
        ----------
        variance_weights : np.array
            Univariate numpy array of variance weights.
        """
        n = variance_weights.size
        self.dataset_cpp.AddVarianceWeights(variance_weights, n)

    def num_observations(self) -> int:
        """
        Query the number of observations in a dataset

        Returns
        -------
        int
            Number of observations in the dataset
        """
        return self.dataset_cpp.NumRows()

    def num_covariates(self) -> int:
        """
        Query the number of covariates (features) in a dataset

        Returns
        -------
        int
            Number of covariates in the dataset
        """
        return self.dataset_cpp.NumCovariates()

    def num_basis(self) -> int:
        """
        Query the dimension of the basis vector in a dataset

        Returns
        -------
        int
            Dimension of the basis vector in the dataset, returning 0 if the dataset does not have a basis
        """
        return self.dataset_cpp.NumBasis()

    def has_basis(self) -> bool:
        """
        Whether or not a dataset has a basis vector (for leaf regression)

        Returns
        -------
        bool
            `True` if the dataset has a basis, `False` otherwise
        """
        return self.dataset_cpp.HasBasis()

    def has_variance_weights(self) -> bool:
        """
        Whether or not a dataset has variance weights

        Returns
        -------
        bool
            `True` if the dataset has variance weights, `False` otherwise
        """
        return self.dataset_cpp.HasVarianceWeights()


class Residual:
    """
    Wrapper around a C++ class that stores residual data used in `stochtree`.
    This object becomes part of the real-time model "state" in that its contents
    always contain a full or partial residual, depending on the state of the sampler.

    Typically this object is initialized with the original outcome and then "residualized"
    by subtracting out the initial prediction value of every tree in every forest term
    (as well as the predictions of any other model term).
    """

    def __init__(self, residual: np.array) -> None:
        """
        Initialize a `Residual` object

        Parameters
        ----------
        residual : np.array
            Univariate numpy array of residual values.
        """
        n = residual.size
        self.residual_cpp = ResidualCpp(residual, n)

    def get_residual(self) -> np.array:
        """
        Extract the current values of the residual as a numpy array

        Returns
        -------
        np.array
            Current values of the residual (which may be net of any forest / other model terms)
        """
        return self.residual_cpp.GetResidualArray()

    def update_data(self, new_vector: np.array) -> None:
        """
        Update the current state of the outcome (i.e. partial residual) data by replacing each element with the elements of `new_vector`

        Parameters
        ----------
        new_vector : np.array
            Univariate numpy array of new residual values.
        """
        n = new_vector.size
        self.residual_cpp.ReplaceData(new_vector, n)
