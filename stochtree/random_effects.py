from typing import Union
import numpy as np
from stochtree_cpp import (
    RandomEffectsContainerCpp,
    RandomEffectsDatasetCpp,
    RandomEffectsLabelMapperCpp,
    RandomEffectsModelCpp,
    RandomEffectsTrackerCpp,
)

from .data import Residual
from .sampler import RNG


class RandomEffectsDataset:
    """
    Wrapper around a C++ class that stores all of the data needed to fit a group random effects model in `stochtree`. This includes:

    1. Labels that define random effects groups.
    2. Basis vectors used to define non-constant leaf models. This is optional but may be included via the `add_basis` method.
    3. Variance weights used to define heteroskedastic or otherwise weighted models. This is optional but may be included via the `add_variance_weights` method.
    """

    def __init__(self) -> None:
        self.rfx_dataset_cpp = RandomEffectsDatasetCpp()

    def add_group_labels(self, group_labels: np.array):
        """
        Add group labels to a dataset

        Parameters
        ----------
        group_labels : np.array
            One-dimensional numpy array of group labels.
        """
        group_labels_ = np.squeeze(group_labels)
        if group_labels_.ndim > 1:
            raise ValueError(
                "group_labels must be a one-dimensional numpy array of group indices"
            )
        n = group_labels_.shape[0]
        self.rfx_dataset_cpp.AddGroupLabels(group_labels_, n)

    def update_group_labels(self, group_labels: np.array):
        """
        Update group labels in a dataset

        Parameters
        ----------
        group_labels : np.array
            One-dimensional numpy array of group labels.
        """
        group_labels_ = np.squeeze(group_labels)
        if group_labels_.ndim > 1:
            raise ValueError(
                "group_labels must be a one-dimensional numpy array of group indices"
            )
        n = group_labels_.shape[0]
        self.rfx_dataset_cpp.UpdateGroupLabels(group_labels_, n)

    def add_basis(self, basis: np.array):
        """
        Add basis matrix to a dataset

        Parameters
        ----------
        basis : np.array
            Two-dimensional numpy array of basis vectors.
        """
        basis_ = np.expand_dims(basis, 1) if np.ndim(basis) == 1 else basis
        if basis_.ndim != 2:
            raise ValueError(
                "basis must be a one-or-two-dimensional numpy array of random effect bases"
            )
        n, p = basis_.shape
        basis_rowmajor = np.ascontiguousarray(basis_)
        self.rfx_dataset_cpp.AddBasis(basis_rowmajor, n, p, True)

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
        if basis_.ndim != 2:
            raise ValueError(
                "basis must be a one-or-two-dimensional numpy array of random effect bases"
            )
        n, p = basis_.shape
        basis_rowmajor = np.ascontiguousarray(basis_)
        self.rfx_dataset_cpp.UpdateBasis(basis_rowmajor, n, p, True)

    def add_variance_weights(self, variance_weights: np.array):
        """
        Add variance weights to a dataset

        Parameters
        ----------
        variance_weights : np.array
            Univariate numpy array of variance weights.
        """
        variance_weights_ = np.squeeze(variance_weights)
        if variance_weights_.ndim > 1:
            raise ValueError(
                "variance_weights must be a one-dimensional numpy array of group indices"
            )
        n = variance_weights_.shape[0]
        self.rfx_dataset_cpp.AddVarianceWeights(variance_weights_, n)

    def update_variance_weights(
        self, variance_weights: np.array, exponentiate: bool = False
    ):
        """
        Update variance weights in a dataset. Allows users to build an ensemble that depends on
        variance weights that are updated throughout the sampler.

        Parameters
        ----------
        variance_weights : np.array
            Univariate numpy array of variance weights.
        exponentiate : bool
            Whether to exponentiate the variance weights before storing them in the dataset.
        """
        if not self.has_variance_weights():
            raise ValueError(
                "This dataset does not have variance weights to update. Please use `add_variance_weights` to create and initialize the values in the Dataset's variance weight vector."
            )
        if not isinstance(variance_weights, np.ndarray):
            raise ValueError("variance_weights must be a numpy array.")
        variance_weights_ = np.squeeze(variance_weights)
        if variance_weights_.ndim > 1:
            raise ValueError(
                "variance_weights must be a one-dimensional numpy array of group indices"
            )
        n = variance_weights_.shape[0]
        if self.num_observations() != n:
            raise ValueError(
                f"The number of rows in the new variance_weights vector ({n}) must match the number of rows in the existing vector ({self.num_observations()})."
            )
        self.rfx_dataset_cpp.UpdateVarianceWeights(variance_weights, n, exponentiate)

    def get_group_labels(self) -> np.array:
        """
        Return the group labels in a RandomEffectsDataset as a numpy array

        Returns
        -------
        np.array
            One-dimensional numpy array of group labels.
        """
        return self.rfx_dataset_cpp.GetGroupLabels()

    def get_basis(self) -> np.array:
        """
        Return the bases in a RandomEffectsDataset as a numpy array

        Returns
        -------
        np.array
            Two-dimensional numpy array of basis vectors.
        """
        return self.rfx_dataset_cpp.GetBasis()

    def get_variance_weights(self) -> np.array:
        """
        Return the variance weights in a RandomEffectsDataset as a numpy array

        Returns
        -------
        np.array
            One-dimensional numpy array of variance weights.
        """
        return self.rfx_dataset_cpp.GetVarianceWeights()

    def num_observations(self) -> int:
        """
        Query the number of observations in a dataset

        Returns
        -------
        int
            Number of observations in the dataset
        """
        return self.rfx_dataset_cpp.NumObservations()

    def num_basis(self) -> int:
        """
        Query the number of bases in a dataset

        Returns
        -------
        int
            Number of bases in the dataset
        """
        return self.rfx_dataset_cpp.NumBases()

    def has_group_labels(self) -> bool:
        """
        Whether or not a dataset has group labels

        Returns
        -------
        bool
            `True` if the dataset has group labels, `False` otherwise
        """
        return self.rfx_dataset_cpp.HasGroupLabels()

    def has_basis(self) -> bool:
        """
        Whether or not a dataset has a basis vector (for leaf regression)

        Returns
        -------
        bool
            `True` if the dataset has a basis, `False` otherwise
        """
        return self.rfx_dataset_cpp.HasBasis()

    def has_variance_weights(self) -> bool:
        """
        Whether or not a dataset has variance weights

        Returns
        -------
        bool
            `True` if the dataset has variance weights, `False` otherwise
        """
        return self.rfx_dataset_cpp.HasVarianceWeights()


class RandomEffectsTracker:
    """
    Class that defines a "tracker" for random effects models, most notably
    storing the data indices available in each group for quicker posterior
    computation and sampling of random effects terms.

    Parameters
    ----------
    group_indices : np.ndarray
        Integer indices indicating groups used to define random effects
    """

    def __init__(self, group_indices: np.ndarray) -> None:
        self.rfx_tracker_cpp = RandomEffectsTrackerCpp(group_indices)
    
    def reset(self, rfx_model, rfx_dataset, residual, rfx_container) -> None:
        """
        Reset the random effects tracker to an existing parameter state
        """
        self.rfx_tracker_cpp.Reset(
            rfx_model.rfx_model_cpp,
            rfx_dataset.rfx_dataset_cpp,
            residual.residual_cpp
        )
    
    def root_reset(self, rfx_model, rfx_dataset, residual, rfx_container) -> None:
        """
        Reset the random effects tracker to its initial state
        """
        self.rfx_tracker_cpp.RootReset(
            rfx_model.rfx_model_cpp,
            rfx_dataset.rfx_dataset_cpp,
            residual.residual_cpp
        )


class RandomEffectsContainer:
    """
    Wrapper around the "persistent" aspects of a C++ random effects model. This includes
    draws of the parameters and a map from the original label indices to the
    0-indexed label numbers used to place group samples in memory (i.e. the
    first label is stored in column 0 of the sample matrix, the second label
    is store in column 1 of the sample matrix, etc...).
    """

    def __init__(self) -> None:
        pass

    def load_new_container(
        self, num_components: int, num_groups: int, rfx_tracker: RandomEffectsTracker
    ) -> None:
        """
        Initializes internal data structures for an "empty" random effects container to be sampled and populated.

        Parameters
        ----------
        num_components : int
            Number of components (bases) in a random effects model. For the simplest random effects model,
            in which each group has a different random intercept, this is 1, and the basis is a trivial
            "dummy" intercept vector.
        num_groups : int
            Number of groups in a random effects model.
        rfx_tracker : RandomEffectsTracker
            Tracking data structures for random effects models.
        """
        self.rfx_container_cpp = RandomEffectsContainerCpp()
        self.rfx_container_cpp.SetComponentsAndGroups(num_components, num_groups)
        self.rfx_label_mapper_cpp = RandomEffectsLabelMapperCpp()
        self.rfx_label_mapper_cpp.LoadFromTracker(rfx_tracker.rfx_tracker_cpp)
        self.rfx_group_ids = rfx_tracker.rfx_tracker_cpp.GetUniqueGroupIds()

    def load_from_json(self, json, rfx_num: int) -> None:
        """
        Initializes internal data structures for an "empty" random effects container to be sampled and populated.

        Parameters
        ----------
        json : JSONSerializer
            Python object wrapping a C++ `json` object.
        rfx_num : int
            Integer index of the RFX term in a JSON model. In practice, this is typically 0 (most models don't contain two RFX terms).
        """
        rfx_container_key = f"random_effect_container_{rfx_num:d}"
        rfx_label_mapper_key = f"random_effect_label_mapper_{rfx_num:d}"
        rfx_group_ids_key = f"random_effect_groupids_{rfx_num:d}"
        self.rfx_container_cpp = RandomEffectsContainerCpp()
        self.rfx_container_cpp.LoadFromJson(json.json_cpp, rfx_container_key)
        self.rfx_label_mapper_cpp = RandomEffectsLabelMapperCpp()
        self.rfx_label_mapper_cpp.LoadFromJson(json.json_cpp, rfx_label_mapper_key)
        self.rfx_group_ids = json.get_integer_vector(
            rfx_group_ids_key, "random_effects"
        )

    def append_from_json(self, json, rfx_num: int) -> None:
        """
        Initializes internal data structures for an "empty" random effects container to be sampled and populated.

        Parameters
        ----------
        json : JSONSerializer
            Python object wrapping a C++ `json` object.
        rfx_num : int
            Integer index of the RFX term in a JSON model. In practice, this is typically 0 (most models don't contain two RFX terms).
        """
        rfx_container_key = f"random_effect_container_{rfx_num:d}"
        self.rfx_container_cpp.AppendFromJson(json.json_cpp, rfx_container_key)

    def num_samples(self) -> int:
        return self.rfx_container_cpp.NumSamples()

    def num_components(self) -> int:
        return self.rfx_container_cpp.NumComponents()

    def num_groups(self) -> int:
        return self.rfx_container_cpp.NumGroups()

    def delete_sample(self, sample_num: int) -> None:
        self.rfx_container_cpp.DeleteSample(sample_num)

    def load_from_json_string(self, json_string: str) -> None:
        """
        Reload a random effects container from an in-memory JSON string.

        Parameters
        ----------
        json_string : str
            In-memory string containing state of a random effects container.
        """
        self.rfx_container_cpp.LoadFromJsonString(json_string)
        # TODO: re-initialize label mapper

    def predict(self, group_labels: np.array, basis: np.array) -> np.ndarray:
        """
        Predict random effects for each observation implied by `group_labels` and `basis`.
        If a random effects model is "intercept-only", `basis` will be an array of ones of size `group_labels.shape[0]`.

        Parameters
        ----------
        group_labels : np.ndarray
            Indices of random effects groups in a prediction set
        basis : np.ndarray
            Basis used for random effects prediction

        Returns
        -------
        result : np.ndarray
            Numpy array with as many rows as observations in `group_labels` and as many columns as samples in the container
        """
        # TODO: add more runtime checks to handle group labels
        rfx_dataset = RandomEffectsDataset()
        rfx_dataset.add_group_labels(group_labels)
        rfx_dataset.add_basis(basis)
        return self.rfx_container_cpp.Predict(
            rfx_dataset.rfx_dataset_cpp, self.rfx_label_mapper_cpp
        )

    def extract_parameter_samples(self) -> dict[str, np.ndarray]:
        """
        Extract the random effects parameters sampled. With the "redundant parameterization" of Gelman et al (2008),
        this includes four parameters: alpha (the "working parameter" shared across every group), xi
        (the "group parameter" sampled separately for each group), beta (the product of alpha and xi,
        which corresponds to the overall group-level random effects), and sigma (group-independent prior
        variance for each component of xi).

        Returns
        -------
        dict[str, np.ndarray]
            dict of arrays. The alpha array has dimension (`num_components`, `num_samples`) and is simply a vector if `num_components = 1`.
            The xi and beta arrays have dimension (`num_components`, `num_groups`, `num_samples`) and are simply matrices if `num_components = 1`.
            The sigma array has dimension (`num_components`, `num_samples`) and is simply a vector if `num_components = 1`.
        """
        # num_samples = self.rfx_container_cpp.NumSamples()
        # num_components = self.rfx_container_cpp.NumComponents()
        # num_groups = self.rfx_container_cpp.NumGroups()
        beta_samples = np.squeeze(self.rfx_container_cpp.GetBeta())
        xi_samples = np.squeeze(self.rfx_container_cpp.GetXi())
        alpha_samples = np.squeeze(self.rfx_container_cpp.GetAlpha())
        sigma_samples = np.squeeze(self.rfx_container_cpp.GetSigma())
        output = {
            "beta_samples": beta_samples,
            "xi_samples": xi_samples,
            "alpha_samples": alpha_samples,
            "sigma_samples": sigma_samples,
        }
        return output
    
    def map_group_id_to_array_index(self, group_id: int) -> int:
        """
        Map an integer-valued random effects group ID to its group's corresponding position in the arrays that store random effects parameter samples.

        Parameters
        ----------
        group_id : int
            Group identifier to be converted to an array position.

        Returns
        -------
        int
            The position of `group_id` in the parameter sample arrays underlying the random effects container.
        """
        return self.rfx_label_mapper_cpp.MapGroupIdToArrayIndex(group_id)
    
    def map_group_ids_to_array_indices(self, group_ids: np.ndarray) -> np.ndarray:
        """
        Map an array of integer-valued random effects group IDs to their groups' corresponding positions in the arrays that store random effects parameter samples.

        Parameters
        ----------
        group_ids : np.ndarray
            Array of group identifiers (integer-valued) to be converted to an array position.

        Returns
        -------
        np.ndarray
            Numpy array of the position of `group_id` in the parameter sample arrays underlying the random effects container.
        """
        return self.rfx_label_mapper_cpp.MapMultipleGroupIdsToArrayIndices(group_ids)


class RandomEffectsModel:
    """
    Class that stores current model state, prior parameters, and procedures for sampling from the conditional posterior of each parameter.

    Parameters
    ----------
    num_components : int
        Number of "components," or bases, defining the random effects regression
    num_groups : int
        Number of random effects groups
    """

    def __init__(self, num_components: int, num_groups: int) -> None:
        self.rfx_model_cpp = RandomEffectsModelCpp(num_components, num_groups)
        self.num_components = num_components
        self.num_groups = num_groups

    def sample(
        self,
        rfx_dataset: RandomEffectsDataset,
        residual: Residual,
        rfx_tracker: RandomEffectsTracker,
        rfx_container: RandomEffectsContainer,
        keep_sample: bool,
        global_variance: float,
        rng: RNG,
    ) -> None:
        """
        Sample from random effects model

        Parameters
        ----------
        rfx_dataset: RandomEffectsDataset
            Object of type `RandomEffectsDataset`
        residual: Residual
            Object of type `Residual`
        rfx_tracker: RandomEffectsTracker
            Object of type `RandomEffectsTracker`
        rfx_samples: RandomEffectsContainer
            Object of type `RandomEffectsContainer`
        keep_sample: bool
            Whether sample should be retained in `rfx_samples`. If `FALSE`, the state of `rfx_tracker` will be updated, but the parameter values will not be added to the sample container. Samples are commonly discarded due to burn-in or thinning.
        global_variance: float
            Scalar global variance parameter
        rng: RNG
            Object of type `RNG`
        """
        self.rfx_model_cpp.SampleRandomEffects(
            rfx_dataset.rfx_dataset_cpp,
            residual.residual_cpp,
            rfx_tracker.rfx_tracker_cpp,
            rfx_container.rfx_container_cpp,
            keep_sample,
            global_variance,
            rng.rng_cpp,
        )

    def predict(
        self, rfx_dataset: RandomEffectsDataset, rfx_tracker: RandomEffectsTracker
    ) -> np.ndarray:
        """
        Predict random effects for each observation in `rfx_dataset`

        Parameters
        ----------
        rfx_dataset: RandomEffectsDataset
            Object of type `RandomEffectsDataset`
        rfx_tracker: RandomEffectsTracker
            Object of type `RandomEffectsTracker`

        Returns
        -------
        np.ndarray
            Numpy array with as many rows as observations in `rfx_dataset` and as many columns as samples in the container
        """
        return self.rfx_model_cpp.Predict(
            rfx_dataset.rfx_dataset_cpp, rfx_tracker.rfx_tracker_cpp
        )

    def set_working_parameter(self, working_parameter: np.ndarray) -> None:
        """
        Set values for the "working parameter." This is typically used for initialization,
        but could also be used to interrupt or override the sampler.

        Parameters
        ----------
        working_parameter: np.ndarray
            Working parameter initial values. Must have the same dimension as the basis in the random effects model.
        """
        if not isinstance(working_parameter, np.ndarray):
            raise ValueError("working_parameter must be a numpy array")
        working_parameter_ = (
            np.squeeze(working_parameter)
            if working_parameter.ndim > 1
            else working_parameter
        )
        if working_parameter_.ndim != 1:
            raise ValueError(
                "working_parameter must be a 1d numpy array with as many elements as bases in the random effects model"
            )
        if working_parameter_.shape[0] != self.num_components:
            raise ValueError(
                "working_parameter must be a 1d numpy array with as many elements as bases in the random effects model"
            )
        self.rfx_model_cpp.SetWorkingParameter(working_parameter)

    def set_group_parameters(self, group_parameters: np.ndarray) -> None:
        """
        Set values for the "group parameters." This is typically used for initialization,
        but could also be used to interrupt or override the sampler.

        Parameters
        ----------
        group_parameters: np.ndarray
            Group parameter initial values. Must have as many rows as bases in the random effects model and as
            many columns as groups in the random effects model.
        """
        if not isinstance(group_parameters, np.ndarray):
            raise ValueError("group_parameters must be a numpy array")
        group_parameters_ = (
            np.squeeze(group_parameters)
            if group_parameters.ndim > 2
            else group_parameters
        )
        if group_parameters_.ndim != 2:
            raise ValueError(
                "group_parameters must be a 2d numpy array with as many rows as bases and as many columns as groups in the random effects model"
            )
        if group_parameters_.shape[0] != self.num_components:
            raise ValueError(
                "group_parameters must be a 2d numpy array with as many rows as bases and as many columns as groups in the random effects model"
            )
        if group_parameters_.shape[1] != self.num_groups:
            raise ValueError(
                "group_parameters must be a 2d numpy array with as many rows as bases and as many columns as groups in the random effects model"
            )
        self.rfx_model_cpp.SetGroupParameters(group_parameters)

    def set_working_parameter_covariance(self, covariance: np.ndarray) -> None:
        """
        Set values for the working parameter covariance. This is typically used for initialization,
        but could also be used to interrupt or override the sampler.

        Parameters
        ----------
        covariance: np.ndarray
            Working parameter covariance initial values. Must have as many rows and columns as bases in the random effects model.
        """
        if not isinstance(covariance, np.ndarray):
            raise ValueError("covariance must be a numpy array")
        covariance_ = np.squeeze(covariance) if covariance.ndim > 2 else covariance
        if covariance_.ndim != 2:
            raise ValueError(
                "covariance must be a 2d numpy array with as many rows and columns as bases in the random effects model"
            )
        if covariance_.shape[0] != self.num_components:
            raise ValueError(
                "covariance must be a 2d numpy array with as many rows and columns as bases in the random effects model"
            )
        if covariance_.shape[1] != self.num_components:
            raise ValueError(
                "covariance must be a 2d numpy array with as many rows and columns as bases in the random effects model"
            )
        self.rfx_model_cpp.SetWorkingParameterCovariance(covariance_)

    def set_group_parameter_covariance(self, covariance: np.ndarray) -> None:
        """
        Set values for the group parameter covariance. This is typically used for initialization,
        but could also be used to interrupt or override the sampler.

        Parameters
        ----------
        covariance: np.ndarray
            Group parameter covariance initial values. Must have as many rows and columns as bases in the random effects model.
        """
        if not isinstance(covariance, np.ndarray):
            raise ValueError("covariance must be a numpy array")
        covariance_ = np.squeeze(covariance) if covariance.ndim > 2 else covariance
        if covariance_.ndim != 2:
            raise ValueError(
                "covariance must be a 2d numpy array with as many rows and columns as bases in the random effects model"
            )
        if covariance_.shape[0] != self.num_components:
            raise ValueError(
                "covariance must be a 2d numpy array with as many rows and columns as bases in the random effects model"
            )
        if covariance_.shape[1] != self.num_components:
            raise ValueError(
                "covariance must be a 2d numpy array with as many rows and columns as bases in the random effects model"
            )
        self.rfx_model_cpp.SetGroupParameterCovariance(covariance_)

    def set_variance_prior_shape(self, shape: float) -> None:
        """
        Set shape parameter for the group parameter variance prior. This is typically used for initialization,
        but could also be used to interrupt or override the sampler.

        Parameters
        ----------
        shape: float
            Shape parameter for the group parameter variance prior. Must be positive.
        """
        if not isinstance(shape, (int, float)):
            raise ValueError("shape must a positive scalar")
        if shape <= 0:
            raise ValueError("shape must a positive scalar")
        self.rfx_model_cpp.SetVariancePriorShape(shape)

    def set_variance_prior_scale(self, scale: float) -> None:
        """
        Set scale parameter for the group parameter variance prior. This is typically used for initialization,
        but could also be used to interrupt or override the sampler.

        Parameters
        ----------
        scale: float
            Scale parameter for the group parameter variance prior. Must be positive.
        """
        if not isinstance(scale, (int, float)):
            raise ValueError("scale must a positive scalar")
        if scale <= 0:
            raise ValueError("scale must a positive scalar")
        self.rfx_model_cpp.SetVariancePriorScale(scale)
    
    def reset(self, rfx_container: RandomEffectsContainer, sample_num: int, sigma_alpha_init: np.array) -> None:
        """
        Reset the random effects model to a previous sample state.
        """
        if not isinstance(sigma_alpha_init, np.ndarray):
            raise ValueError("sigma_alpha_init must be a numpy array")
        if sigma_alpha_init.ndim != 2:
            raise ValueError(
                "sigma_alpha_init must be a 2d square numpy array with as many rows / columns as bases in the random effects model"
            )
        if sigma_alpha_init.shape[0] != sigma_alpha_init.shape[1]:
            raise ValueError(
                "sigma_alpha_init must be a 2d square numpy array with as many rows / columns as bases in the random effects model"
            )
        if sigma_alpha_init.shape[0] != self.num_components:
            raise ValueError(
                "sigma_alpha_init must be a 2d square numpy array with as many rows / columns as bases in the random effects model"
            )
        self.rfx_model_cpp.Reset(
            rfx_container.rfx_container_cpp, sample_num
        )
        self.set_working_parameter_covariance(sigma_alpha_init)
    
    def root_reset(self, alpha_init: np.array, xi_init: np.array, sigma_alpha_init: np.array, sigma_xi_init: np.array, sigma_xi_shape: float, sigma_xi_scale: float) -> None:
        """
        Reset the random effects model to its initial state.
        """
        self.set_working_parameter(alpha_init)
        self.set_group_parameters(xi_init)
        self.set_working_parameter_cov(sigma_alpha_init)
        self.set_group_parameter_cov(sigma_xi_init)
        self.set_variance_prior_shape(sigma_xi_shape)
        self.set_variance_prior_scale(sigma_xi_scale)
