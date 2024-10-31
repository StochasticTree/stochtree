"""
Python classes wrapping C++ sampler objects
"""
import numpy as np
from .data import Dataset, Residual
from .forest import ForestContainer, Forest
from stochtree_cpp import RngCpp, ForestSamplerCpp, GlobalVarianceModelCpp, LeafVarianceModelCpp
from typing import Union

class RNG:
    def __init__(self, random_seed: int) -> None:
        # Initialize a ForestDatasetCpp object
        self.rng_cpp = RngCpp(random_seed)


class ForestSampler:
    def __init__(self, dataset: Dataset, feature_types: np.array, num_trees: int, num_obs: int, alpha: float, beta: float, min_samples_leaf: int, max_depth: int = -1) -> None:
        # Initialize a ForestDatasetCpp object
        self.forest_sampler_cpp = ForestSamplerCpp(dataset.dataset_cpp, feature_types, num_trees, num_obs, alpha, beta, min_samples_leaf, max_depth)
    
    def sample_one_iteration(self, forest_container: ForestContainer, forest: Forest, dataset: Dataset, 
                             residual: Residual, rng: RNG, feature_types: np.array, cutpoint_grid_size: int, 
                             leaf_model_scale_input: np.array, variable_weights: np.array, a_forest: float, b_forest: float, 
                             global_variance: float, leaf_model_int: int, keep_forest: bool, gfr: bool, pre_initialized: bool):
        """
        Sample one iteration of a forest using the specified model and tree sampling algorithm

        Parameters
        ----------
        forest_container : :obj:`ForestContainer`
            Stochtree object storing tree ensembles
        forest : :obj:`Forest`
            Stochtree object storing the "active" forest being sampled
        dataset : :obj:`Dataset`
            Stochtree dataset object storing covariates / bases / weights
        residual : :obj:`Residual`
            Stochtree object storing continuously updated partial / full residual
        rng : :obj:`RNG`
            Stochtree object storing C++ random number generator to be used sampling algorithm
        feature_types : :obj:`np.array`
            Array of integer-coded feature types (0 = numeric, 1 = ordered categorical, 2 = unordered categorical)
        cutpoint_grid_size : :obj:`int`
            Maximum size of a grid of available cutpoints (which thins the number of possible splits, particularly useful in the grow-from-root algorithm)
        leaf_model_scale_input : :obj:`np.array`
            Numpy array containing leaf model scale parameter (if the leaf model is univariate, this is essentially a scalar which is used as such in the C++ source, but stored as a numpy array)
        variable_weights : :obj:`np.array`
            Numpy array containing sampling probabilities for each feature
        a_forest : :obj:`float`
            Scale parameter for the inverse gamma outcome model for heteroskedasticity forest
        b_forest : :obj:`float`
            Scale parameter for the inverse gamma outcome model for heteroskedasticity forest
        global_variance : :obj:`float`
            Current value of the global error variance parameter
        leaf_model_int : :obj:`int`
            Integer encoding the leaf model type (0 = constant Gaussian leaf mean model, 1 = univariate Gaussian leaf regression mean model, 2 = multivariate Gaussian leaf regression mean model, 3 = univariate Inverse Gamma constant leaf variance model)
        keep_forest : :obj:`bool`
            Whether or not the resulting forest should be retained in ``forest_container`` or discarded (due to burnin or thinning for example)
        gfr : :obj:`bool`
            Whether or not the "grow-from-root" (GFR) sampler is run (if this is ``True`` and ``leaf_model_int=0`` this is equivalent to XBART, if this is ``FALSE`` and ``leaf_model_int=0`` this is equivalent to the original BART)
        pre_initialized : :obj:`bool`
            Whether or not the forest being sampled has already been initialized
        """
        self.forest_sampler_cpp.SampleOneIteration(forest_container.forest_container_cpp, forest.forest_cpp, dataset.dataset_cpp, residual.residual_cpp, rng.rng_cpp, 
                                                   feature_types, cutpoint_grid_size, leaf_model_scale_input, variable_weights, 
                                                   a_forest, b_forest, global_variance, leaf_model_int, keep_forest, gfr, pre_initialized)
    
    def prepare_for_sampler(self, dataset: Dataset, residual: Residual, forest: Forest, leaf_model: int, initial_values: np.array):
        """
        Initialize forest and tracking data structures with constant root values before running a sampler

        Parameters
        ----------
        dataset : :obj:`Dataset`
            Stochtree dataset object storing covariates / bases / weights
        residual : :obj:`Residual`
            Stochtree object storing continuously updated partial / full residual
        forest : :obj:`Forest`
            Stochtree object storing the "active" forest being sampled
        leaf_model : :obj:`int`
            Integer encoding the leaf model type
        initial_values : :obj:`np.array`
            Constant root node value(s) at which to initialize forest prediction (internally, it is divided by the number of trees and typically it is 0 for mean models and 1 for variance models).
        """
        self.forest_sampler_cpp.InitializeForestModel(dataset.dataset_cpp, residual.residual_cpp, forest.forest_cpp, leaf_model, initial_values)
    
    def adjust_residual(self, dataset: Dataset, residual: Residual, forest: Forest, requires_basis: bool, add: bool) -> None:
        """
        Method that "adjusts" the residual used for training tree ensembles by either adding or subtracting the prediction of each tree to the existing residual. 
        
        This is typically run just once at the beginning of a forest sampling algorithm --- after trees are initialized with constant root node predictions, their 
        root predictions are subtracted out of the residual.

        Parameters
        ----------
        dataset : :obj:`Dataset`
            Stochtree dataset object storing covariates / bases / weights
        residual : :obj:`Residual`
            Stochtree object storing continuously updated partial / full residual
        forest : :obj:`Forest`
            Stochtree object storing the "active" forest being sampled
        requires_basis : :obj:`bool`
            Whether or not the forest requires a basis dot product when predicting
        add : :obj:`bool`
            Whether the predictions of each tree are added (if ``add=True``) or subtracted (``add=False``) from the outcome to form the new residual
        """
        forest.forest_cpp.AdjustResidual(dataset.dataset_cpp, residual.residual_cpp, self.forest_sampler_cpp, requires_basis, add)
    
    def propagate_basis_update(self, dataset: Dataset, residual: Residual, forest: Forest) -> None:
        """
        Propagates basis update through to the (full/partial) residual by iteratively (a) adding back in the previous prediction of each tree, (b) recomputing predictions 
        for each tree (caching on the C++ side), (c) subtracting the new predictions from the residual.

        This is useful in cases where a basis (for e.g. leaf regression) is updated outside of a tree sampler (as with e.g. adaptive coding for binary treatment BCF). 
        Once a basis has been updated, the overall "function" represented by a tree model has changed and this should be reflected through to the residual before the 
        next sampling loop is run.

        Parameters
        ----------
        dataset : :obj:`Dataset`
            Stochtree dataset object storing covariates / bases / weights
        residual : :obj:`Residual`
            Stochtree object storing continuously updated partial / full residual
        forest : :obj:`Forest`
            Stochtree object storing the "active" forest being sampled
        """
        self.forest_sampler_cpp.PropagateBasisUpdate(dataset.dataset_cpp, residual.residual_cpp, forest.forest_cpp)
    
    def propagate_residual_update(self, residual: Residual) -> None:
        self.forest_sampler_cpp.PropagateResidualUpdate(residual.residual_cpp)


class GlobalVarianceModel:
    def __init__(self) -> None:
        # Initialize a GlobalVarianceModelCpp object
        self.variance_model_cpp = GlobalVarianceModelCpp()
    
    def sample_one_iteration(self, residual: Residual, rng: RNG, a: float, b: float) -> float:
        """
        Sample one iteration of a global error variance parameter
        """
        return self.variance_model_cpp.SampleOneIteration(residual.residual_cpp, rng.rng_cpp, a, b)


class LeafVarianceModel:
    def __init__(self) -> None:
        # Initialize a LeafVarianceModelCpp object
        self.variance_model_cpp = LeafVarianceModelCpp()
    
    def sample_one_iteration(self, forest: Forest, rng: RNG, a: float, b: float, sample_num: int) -> float:
        """
        Sample one iteration of a forest leaf model's variance parameter (assuming a location-scale leaf model, most commonly ``N(0, tau)``)
        """
        return self.variance_model_cpp.SampleOneIteration(forest.forest_cpp, rng.rng_cpp, a, b, sample_num)
