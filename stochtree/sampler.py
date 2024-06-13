"""
Python classes wrapping C++ sampler objects
"""
import numpy as np
from .data import Dataset, Residual
from .forest import ForestContainer
from stochtree_cpp import RngCpp, ForestSamplerCpp, GlobalVarianceModelCpp, LeafVarianceModelCpp

class RNG:
    def __init__(self, random_seed: int) -> None:
        # Initialize a ForestDatasetCpp object
        self.rng_cpp = RngCpp(random_seed)


class ForestSampler:
    def __init__(self, dataset: Dataset, feature_types: np.array, num_trees: int, num_obs: int, alpha: float, beta: float, min_samples_leaf: int) -> None:
        # Initialize a ForestDatasetCpp object
        self.forest_sampler_cpp = ForestSamplerCpp(dataset.dataset_cpp, feature_types, num_trees, num_obs, alpha, beta, min_samples_leaf)
    
    def sample_one_iteration(self, forest_container: ForestContainer, dataset: Dataset, residual: Residual, rng: RNG, 
                             feature_types: np.array, cutpoint_grid_size: int, leaf_model_scale_input: np.array, 
                             variable_weights: np.array, global_variance: float, leaf_model_int: int, gfr: bool, pre_initialized: bool):
        """
        Sample one iteration of a forest using the specified model and tree sampling algorithm
        """
        self.forest_sampler_cpp.SampleOneIteration(forest_container.forest_container_cpp, dataset.dataset_cpp, residual.residual_cpp, rng.rng_cpp, 
                                                   feature_types, cutpoint_grid_size, leaf_model_scale_input, variable_weights, 
                                                   global_variance, leaf_model_int, gfr, pre_initialized)
    
    def update_residual(self, dataset: Dataset, residual: Residual, forest_container: ForestContainer, requires_basis: bool, forest_num: int, add: bool) -> None:
        forest_container.forest_container_cpp.UpdateResidual(dataset.dataset_cpp, residual.residual_cpp, self.forest_sampler_cpp, requires_basis, forest_num, add)


class GlobalVarianceModel:
    def __init__(self) -> None:
        # Initialize a GlobalVarianceModelCpp object
        self.variance_model_cpp = GlobalVarianceModelCpp()
    
    def sample_one_iteration(self, residual: Residual, rng: RNG, nu: float, lamb: float) -> float:
        """
        Sample one iteration of a forest using the specified model and tree sampling algorithm
        """
        return self.variance_model_cpp.SampleOneIteration(residual.residual_cpp, rng.rng_cpp, nu, lamb)


class LeafVarianceModel:
    def __init__(self) -> None:
        # Initialize a LeafVarianceModelCpp object
        self.variance_model_cpp = LeafVarianceModelCpp()
    
    def sample_one_iteration(self, forest_container: ForestContainer, rng: RNG, a: float, b: float, sample_num: int) -> float:
        """
        Sample one iteration of a forest using the specified model and tree sampling algorithm
        """
        return self.variance_model_cpp.SampleOneIteration(forest_container.forest_container_cpp, rng.rng_cpp, a, b, sample_num)
