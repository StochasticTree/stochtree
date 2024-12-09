"""
Bayesian Additive Regression Trees (BART) module
"""
from numbers import Number, Integral
from math import log
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from .data import Dataset, Residual
from .forest import ForestContainer, Forest
from .preprocessing import CovariateTransformer, _preprocess_bart_params
from .sampler import ForestSampler, RNG, GlobalVarianceModel, LeafVarianceModel
from .serialization import JSONSerializer
from .utils import NotSampledError

class BARTModel:
    """Class that handles sampling, storage, and serialization of stochastic forest models like BART, XBART, and Warm-Start BART
    """

    def __init__(self) -> None:
        # Internal flag for whether the sample() method has been run
        self.sampled = False
        self.rng = np.random.default_rng()
    
    def is_sampled(self) -> bool:
        return self.sampled
    
    def sample(self, X_train: np.array, y_train: np.array, basis_train: np.array = None, X_test: np.array = None, basis_test: np.array = None, 
               num_gfr: int = 5, num_burnin: int = 0, num_mcmc: int = 100, params: Optional[Dict[str, Any]] = None) -> None:
        """Runs a BART sampler on provided training set. Predictions will be cached for the training set and (if provided) the test set. 
        Does not require a leaf regression basis. 

        Parameters
        ----------
        X_train : :obj:`np.array`
            Training set covariates on which trees may be partitioned.
        y_train : :obj:`np.array`
            Training set outcome.
        basis_train : :obj:`np.array`, optional
            Optional training set basis vector used to define a regression to be run in the leaves of each tree.
        X_test : :obj:`np.array`, optional
            Optional test set covariates.
        basis_test : :obj:`np.array`, optional
            Optional test set basis vector used to define a regression to be run in the leaves of each tree. 
            Must be included / omitted consistently (i.e. if basis_train is provided, then basis_test must be provided alongside X_test).
        num_gfr : :obj:`int`, optional
            Number of "warm-start" iterations run using the grow-from-root algorithm (He and Hahn, 2021). Defaults to ``5``.
        num_burnin : :obj:`int`, optional
            Number of "burn-in" iterations of the MCMC sampler. Defaults to ``0``. Ignored if ``num_gfr > 0``.
        num_mcmc : :obj:`int`, optional
            Number of "retained" iterations of the MCMC sampler. Defaults to ``100``. If this is set to 0, GFR (XBART) samples will be retained.
        params : :obj:`dict`, optional
            Dictionary of model parameters, each of which has a default value.

            * ``cutpoint_grid_size`` (``int``): Maximum number of cutpoints to consider for each feature. Defaults to ``100``.
            * ``sigma_leaf`` (``float``): Scale parameter on the (conditional mean) leaf node regression model.
            * ``alpha_mean`` (``float``): Prior probability of splitting for a tree of depth 0 in the conditional mean model. Tree split prior combines ``alpha_mean`` and ``beta_mean`` via ``alpha_mean*(1+node_depth)^-beta_mean``.
            * ``beta_mean`` (``float``): Exponent that decreases split probabilities for nodes of depth > 0 in the conditional mean model. Tree split prior combines ``alpha_mean`` and ``beta_mean`` via ``alpha_mean*(1+node_depth)^-beta_mean``.
            * ``min_samples_leaf_mean`` (``int``): Minimum allowable size of a leaf, in terms of training samples, in the conditional mean model. Defaults to ``5``.
            * ``max_depth_mean`` (``int``): Maximum depth of any tree in the ensemble in the conditional mean model. Defaults to ``10``. Can be overriden with ``-1`` which does not enforce any depth limits on trees.
            * ``alpha_variance`` (``float``): Prior probability of splitting for a tree of depth 0 in the conditional variance model. Tree split prior combines ``alpha_variance`` and ``beta_variance`` via ``alpha_variance*(1+node_depth)^-beta_variance``.
            * ``beta_variance`` (``float``): Exponent that decreases split probabilities for nodes of depth > 0 in the conditional variance model. Tree split prior combines ``alpha_variance`` and ``beta_variance`` via ``alpha_variance*(1+node_depth)^-beta_variance``.
            * ``min_samples_leaf_variance`` (``int``): Minimum allowable size of a leaf, in terms of training samples in the conditional variance model. Defaults to ``5``.
            * ``max_depth_variance`` (``int``): Maximum depth of any tree in the ensemble in the conditional variance model. Defaults to ``10``. Can be overriden with ``-1`` which does not enforce any depth limits on trees.
            * ``a_global`` (``float``): Shape parameter in the ``IG(a_global, b_global)`` global error variance model. Defaults to ``0``.
            * ``b_global`` (``float``): Scale parameter in the ``IG(a_global, b_global)`` global error variance prior. Defaults to ``0``.
            * ``a_leaf`` (``float``): Shape parameter in the ``IG(a_leaf, b_leaf)`` leaf node parameter variance model. Defaults to ``3``.
            * ``b_leaf`` (``float``): Scale parameter in the ``IG(a_leaf, b_leaf)`` leaf node parameter variance model. Calibrated internally as ``0.5/num_trees_mean`` if not set here.
            * ``a_forest`` (``float``): Shape parameter in the [optional] ``IG(a_forest, b_forest)`` conditional error variance forest (which is only sampled if ``num_trees_variance > 0``). Calibrated internally as ``num_trees_variance / 1.5^2 + 0.5`` if not set here.
            * ``b_forest`` (``float``): Scale parameter in the [optional] ``IG(a_forest, b_forest)`` conditional error variance forest (which is only sampled if ``num_trees_variance > 0``). Calibrated internally as ``num_trees_variance / 1.5^2`` if not set here.
            * ``sigma2_init`` (``float``): Starting value of global variance parameter. Set internally as a percentage of the standardized outcome variance if not set here.
            * ``variance_forest_leaf_init`` (``float``): Starting value of root forest prediction in conditional (heteroskedastic) error variance model. Calibrated internally as ``np.log(pct_var_variance_forest_init*np.var((y-np.mean(y))/np.std(y)))/num_trees_variance`` if not set.
            * ``pct_var_sigma2_init`` (``float``): Percentage of standardized outcome variance used to initialize global error variance parameter. Superseded by ``sigma2``. Defaults to ``1``.
            * ``pct_var_variance_forest_init`` (``float``): Percentage of standardized outcome variance used to initialize global error variance parameter. Default: ``1``. Superseded by ``variance_forest_init``.
            * ``variance_scale`` (``float``): Variance after the data have been scaled. Default: ``1``.
            * ``variable_weights_mean`` (``np.array``): Numeric weights reflecting the relative probability of splitting on each variable in the mean forest. Does not need to sum to 1 but cannot be negative. Defaults to uniform over the columns of ``X_train`` if not provided.
            * ``variable_weights_forest`` (``np.array``): Numeric weights reflecting the relative probability of splitting on each variable in the variance forest. Does not need to sum to 1 but cannot be negative. Defaults to uniform over the columns of ``X_train`` if not provided.
            * ``num_trees_mean`` (``int``): Number of trees in the ensemble for the conditional mean model. Defaults to ``200``. If ``num_trees_mean = 0``, the conditional mean will not be modeled using a forest and the function will only proceed if ``num_trees_variance > 0``.
            * ``num_trees_variance`` (``int``): Number of trees in the ensemble for the conditional variance model. Defaults to ``0``. Variance is only modeled using a tree / forest if ``num_trees_variance > 0``.
            * ``sample_sigma_global`` (``bool``): Whether or not to update the ``sigma^2`` global error variance parameter based on ``IG(a_global, b_global)``. Defaults to ``True``.
            * ``sample_sigma_leaf`` (``bool``): Whether or not to update the ``tau`` leaf scale variance parameter based on ``IG(a_leaf, b_leaf)``. Cannot (currently) be set to true if ``basis_train`` has more than one column. Defaults to ``False``.
            * ``random_seed`` (``int``): Integer parameterizing the C++ random number generator. If not specified, the C++ random number generator is seeded according to ``std::random_device``.
            * ``keep_burnin`` (``bool``): Whether or not "burnin" samples should be included in predictions. Defaults to ``False``. Ignored if ``num_mcmc == 0``.
            * ``keep_gfr`` (``bool``): Whether or not "warm-start" / grow-from-root samples should be included in predictions. Defaults to ``False``. Ignored if ``num_mcmc == 0``.
            * ``keep_every`` (``int``): How many iterations of the burned-in MCMC sampler should be run before forests and parameters are retained. Defaults to ``1``. Setting ``keep_every = k`` for some ``k > 1`` will "thin" the MCMC samples by retaining every ``k``-th sample, rather than simply every sample. This can reduce the autocorrelation of the MCMC samples.

        Returns
        -------
        self : BARTModel
            Sampled BART Model.
        """
        # Unpack parameters
        bart_params = _preprocess_bart_params(params)
        cutpoint_grid_size = bart_params['cutpoint_grid_size']
        sigma_leaf = bart_params['sigma_leaf']
        alpha_mean = bart_params['alpha_mean']
        beta_mean = bart_params['beta_mean']
        min_samples_leaf_mean = bart_params['min_samples_leaf_mean']
        max_depth_mean = bart_params['max_depth_mean']
        alpha_variance = bart_params['alpha_variance']
        beta_variance = bart_params['beta_variance']
        min_samples_leaf_variance = bart_params['min_samples_leaf_variance']
        max_depth_variance = bart_params['max_depth_variance']
        a_global = bart_params['a_global']
        b_global = bart_params['b_global']
        a_leaf = bart_params['a_leaf']
        b_leaf = bart_params['b_leaf']
        a_forest = bart_params['a_forest']
        b_forest = bart_params['b_forest']
        sigma2_init = bart_params['sigma2_init']
        variance_forest_leaf_init = bart_params['variance_forest_leaf_init']
        pct_var_sigma2_init = bart_params['pct_var_sigma2_init']
        pct_var_variance_forest_init = bart_params['pct_var_variance_forest_init']
        variance_scale = bart_params['variance_scale']
        variable_weights_mean = bart_params['variable_weights_mean']
        variable_weights_variance = bart_params['variable_weights_variance']
        num_trees_mean = bart_params['num_trees_mean']
        num_trees_variance = bart_params['num_trees_variance']
        sample_sigma_global = bart_params['sample_sigma_global']
        sample_sigma_leaf = bart_params['sample_sigma_leaf']
        random_seed = bart_params['random_seed']
        keep_burnin = bart_params['keep_burnin']
        keep_gfr = bart_params['keep_gfr']
        self.standardize = bart_params['standardize']
        num_chains = bart_params['num_chains']
        keep_every = bart_params['keep_every']
    
        # Check that num_chains >= 1
        if not isinstance(num_chains, Integral) or num_chains < 1:
            raise ValueError("num_chains must be an integer greater than 0")

        # Check if there are enough GFR samples to seed num_chains samplers
        if num_gfr > 0:
            if num_chains > num_gfr:
                raise ValueError("num_chains > num_gfr, meaning we do not have enough GFR samples to seed num_chains distinct MCMC chains")
        
        # Determine which models (conditional mean, conditional variance, or both) we will fit
        self.include_mean_forest = True if num_trees_mean > 0 else False
        self.include_variance_forest = True if num_trees_variance > 0 else False
        
        # Check data inputs
        if not isinstance(X_train, pd.DataFrame) and not isinstance(X_train, np.ndarray):
            raise ValueError("X_train must be a pandas dataframe or numpy array")
        if X_test is not None:
            if not isinstance(X_test, pd.DataFrame) and not isinstance(X_test, np.ndarray):
                raise ValueError("X_test must be a pandas dataframe or numpy array")
        if not isinstance(y_train, np.ndarray):
            raise ValueError("y_train must be a numpy array")
        if basis_train is not None:
            if not isinstance(basis_train, np.ndarray):
                raise ValueError("basis_train must be a numpy array")
        if basis_test is not None:
            if not isinstance(basis_test, np.ndarray):
                raise ValueError("X_test must be a numpy array")
        
        # Convert everything to standard shape (2-dimensional)
        if isinstance(X_train, np.ndarray):
            if X_train.ndim == 1:
                X_train = np.expand_dims(X_train, 1)
        if basis_train is not None:
            if basis_train.ndim == 1:
                basis_train = np.expand_dims(basis_train, 1)
        if y_train.ndim == 1:
            y_train = np.expand_dims(y_train, 1)
        if X_test is not None:
            if isinstance(X_test, np.ndarray):
                if X_test.ndim == 1:
                    X_test = np.expand_dims(X_test, 1)
        if basis_test is not None:
            if basis_test.ndim == 1:
                basis_test = np.expand_dims(basis_test, 1)
        
        # Data checks
        if X_test is not None:
            if X_test.shape[1] != X_train.shape[1]:
                raise ValueError("X_train and X_test must have the same number of columns")
        if basis_test is not None:
            if basis_train is not None:
                if basis_test.shape[1] != basis_train.shape[1]:
                    raise ValueError("basis_train and basis_test must have the same number of columns")
            else:
                raise ValueError("basis_test provided but basis_train was not")
        if basis_train is not None:
            if basis_train.shape[0] != X_train.shape[0]:
                raise ValueError("basis_train and Z_train must have the same number of rows")
        if y_train.shape[0] != X_train.shape[0]:
            raise ValueError("X_train and y_train must have the same number of rows")
        if X_test is not None and basis_test is not None:
            if X_test.shape[0] != basis_test.shape[0]:
                raise ValueError("X_test and basis_test must have the same number of rows")

        # Compute variable weights
        p = X_train.shape[1]
        if self.include_mean_forest:
            if not variable_weights_mean:
                variable_weights_mean = np.repeat(1.0/p, p)
        if self.include_variance_forest:
            if not variable_weights_variance:
                variable_weights_variance = np.repeat(1.0/p, p)
        
        # Covariate preprocessing
        self._covariate_transformer = CovariateTransformer()
        self._covariate_transformer.fit(X_train)
        X_train_processed = self._covariate_transformer.transform(X_train)
        if X_test is not None:
            X_test_processed = self._covariate_transformer.transform(X_test)
        feature_types = np.asarray(self._covariate_transformer._processed_feature_types)

        # Determine whether a test set is provided
        self.has_test = X_test is not None

        # Determine whether a basis is provided
        self.has_basis = basis_train is not None

        # Unpack data dimensions
        self.n_train = y_train.shape[0]
        self.n_test = X_test_processed.shape[0] if self.has_test else 0
        self.num_covariates = X_train_processed.shape[1]
        self.num_basis = basis_train.shape[1] if self.has_basis else 0
        
        # Update variable weights if the covariates have been resized (by e.g. one-hot encoding)
        if X_train_processed.shape[1] != X_train.shape[1]:
            original_var_indices = self._covariate_transformer._original_feature_indices
            variable_weights_adj = np.array([1/np.sum(original_var_indices==i) for i in original_var_indices])
            if self.include_mean_forest:
                variable_weights_mean = variable_weights_mean[original_var_indices]*variable_weights_adj
            if self.include_variance_forest:
                variable_weights_variance = variable_weights_variance[original_var_indices]*variable_weights_adj

        # Scale outcome
        if self.standardize:
            self.y_bar = np.squeeze(np.mean(y_train))
            self.y_std = np.squeeze(np.std(y_train))
        else:
            self.y_bar = 0
            self.y_std = 1
        if variance_scale > 0:
            self.variance_scale = variance_scale
        else:
            raise ValueError("variance_scale must be positive")
        resid_train = (y_train-self.y_bar)/self.y_std
        resid_train = resid_train*np.sqrt(self.variance_scale)

        # Calibrate priors for global sigma^2 and sigma_leaf (don't use regression initializer for warm-start or XBART)
        if not sigma2_init:
            sigma2_init = pct_var_sigma2_init*np.var(resid_train)
        if not variance_forest_leaf_init:
            variance_forest_leaf_init = pct_var_variance_forest_init*np.var(resid_train)
        current_sigma2 = sigma2_init
        self.sigma2_init = sigma2_init
        if self.include_mean_forest:
            b_leaf = np.squeeze(np.var(resid_train)) / num_trees_mean if b_leaf is None else b_leaf
            sigma_leaf = np.squeeze(np.var(resid_train)) / num_trees_mean if sigma_leaf is None else sigma_leaf
            current_leaf_scale = np.array([[sigma_leaf]])
        else:
            current_leaf_scale = np.array([[1.]])
        if self.include_variance_forest:
            if not a_forest:
                a_forest = num_trees_variance / 1.5**2 + 0.5
            if not b_forest:
                b_forest = num_trees_variance / 1.5**2
        else:
            if not a_forest:
                a_forest = 1.
            if not b_forest:
                b_forest = 1.

        # Container of variance parameter samples
        self.num_gfr = num_gfr
        self.num_burnin = num_burnin
        self.num_mcmc = num_mcmc
        num_temp_samples = num_gfr + num_burnin + num_mcmc * keep_every
        num_retained_samples = num_mcmc * num_chains
        # Delete GFR samples from these containers after the fact if desired
        # if keep_gfr:
        #     num_retained_samples += num_gfr
        num_retained_samples += num_gfr
        if keep_burnin:
            num_retained_samples += num_burnin * num_chains
        self.num_samples = num_retained_samples
        self.sample_sigma_global = sample_sigma_global
        self.sample_sigma_leaf = sample_sigma_leaf
        if sample_sigma_global:
            self.global_var_samples = np.empty(self.num_samples, dtype = np.float64)
        if sample_sigma_leaf:
            self.leaf_scale_samples = np.empty(self.num_samples, dtype = np.float64)
        sample_counter = -1
        
        # Forest Dataset (covariates and optional basis)
        forest_dataset_train = Dataset()
        forest_dataset_train.add_covariates(X_train_processed)
        if self.has_basis:
            forest_dataset_train.add_basis(basis_train)
        if self.has_test:
            forest_dataset_test = Dataset()
            forest_dataset_test.add_covariates(X_test_processed)
            if self.has_basis:
                forest_dataset_test.add_basis(basis_test)

        # Residual
        residual_train = Residual(resid_train)

        # C++ random number generator
        if random_seed is None: 
            cpp_rng = RNG(-1)
        else:
            cpp_rng = RNG(random_seed)
        
        # Sampling data structures
        if self.include_mean_forest:
            forest_sampler_mean = ForestSampler(forest_dataset_train, feature_types, num_trees_mean, self.n_train, alpha_mean, beta_mean, min_samples_leaf_mean, max_depth_mean)
        if self.include_variance_forest:
            forest_sampler_variance = ForestSampler(forest_dataset_train, feature_types, num_trees_variance, self.n_train, alpha_variance, beta_variance, min_samples_leaf_variance, max_depth_variance)

        # Set variance leaf model type (currently only one option)
        leaf_model_variance_forest = 3

        # Determine the mean forest leaf model type
        if not self.has_basis:
            leaf_model_mean_forest = 0
        elif self.num_basis == 1:
            leaf_model_mean_forest = 1
        else:
            leaf_model_mean_forest = 2

        # Container of forest samples
        if self.include_mean_forest:
            self.forest_container_mean = ForestContainer(num_trees_mean, 1, True, False) if not self.has_basis else ForestContainer(num_trees_mean, self.num_basis, False, False)
            active_forest_mean = Forest(num_trees_mean, 1, True, False) if not self.has_basis else Forest(num_trees_mean, self.num_basis, False, False)
        if self.include_variance_forest:
            self.forest_container_variance = ForestContainer(num_trees_variance, 1, True, True)
            active_forest_variance = Forest(num_trees_variance, 1, True, True)
        
        # Variance samplers
        if self.sample_sigma_global:
            global_var_model = GlobalVarianceModel()
        if self.sample_sigma_leaf:
            leaf_var_model = LeafVarianceModel()

        # Initialize the leaves of each tree in the mean forest
        if self.include_mean_forest:
            if self.has_basis:
                init_val_mean = np.repeat(0., basis_train.shape[1])
            else:
                init_val_mean = np.array([0.])
            forest_sampler_mean.prepare_for_sampler(forest_dataset_train, residual_train, active_forest_mean, leaf_model_mean_forest, init_val_mean)

        # Initialize the leaves of each tree in the variance forest
        if self.include_variance_forest:
            init_val_variance = np.array([variance_forest_leaf_init])
            forest_sampler_variance.prepare_for_sampler(forest_dataset_train, residual_train, active_forest_variance, leaf_model_variance_forest, init_val_variance)

        # Run GFR (warm start) if specified
        if self.num_gfr > 0:
            for i in range(self.num_gfr):
                # Keep all GFR samples at this stage -- remove from ForestSamples after MCMC
                # keep_sample = keep_gfr
                keep_sample = True
                if keep_sample:
                    sample_counter += 1
                # Sample the mean forest
                if self.include_mean_forest:
                    forest_sampler_mean.sample_one_iteration(
                        self.forest_container_mean, active_forest_mean, forest_dataset_train, residual_train, 
                        cpp_rng, feature_types, cutpoint_grid_size, current_leaf_scale, variable_weights_mean, a_forest, b_forest, 
                        current_sigma2, leaf_model_mean_forest, keep_sample, True, True
                    )
                
                # Sample the variance forest
                if self.include_variance_forest:
                    forest_sampler_variance.sample_one_iteration(
                        self.forest_container_variance, active_forest_variance, forest_dataset_train, residual_train, 
                        cpp_rng, feature_types, cutpoint_grid_size, current_leaf_scale, variable_weights_variance, a_forest, b_forest, 
                        current_sigma2, leaf_model_variance_forest, keep_sample, True, True
                    )

                # Sample variance parameters (if requested)
                if self.sample_sigma_global:
                    current_sigma2 = global_var_model.sample_one_iteration(residual_train, cpp_rng, a_global, b_global)
                    if keep_sample:
                        self.global_var_samples[sample_counter] = current_sigma2
                if self.sample_sigma_leaf:
                    current_leaf_scale[0,0] = leaf_var_model.sample_one_iteration(active_forest_mean, cpp_rng, a_leaf, b_leaf)
                    if keep_sample:
                        self.leaf_scale_samples[sample_counter] = current_leaf_scale[0,0]
        
        # Run MCMC
        if self.num_burnin + self.num_mcmc > 0:
            for chain_num in range(num_chains):
                if num_gfr > 0:
                    forest_ind = num_gfr - chain_num - 1
                    if self.include_mean_forest:
                        active_forest_mean.reset(self.forest_container_mean, forest_ind)
                        forest_sampler_mean.reconstitute_from_forest(active_forest_mean, forest_dataset_train, residual_train, True)
                    if self.include_variance_forest:
                        active_forest_variance.reset(self.forest_container_variance, forest_ind)
                        forest_sampler_variance.reconstitute_from_forest(active_forest_variance, forest_dataset_train, residual_train, False)
                    if sample_sigma_global:
                        current_sigma2 = self.global_var_samples[forest_ind]
                else:
                    if self.include_mean_forest:
                        active_forest_mean.reset_root()
                        if init_val_mean.shape[0] == 1:
                            active_forest_mean.set_root_leaves(init_val_mean[0] / num_trees_mean)
                        else:
                            active_forest_mean.set_root_leaves(init_val_mean / num_trees_mean)
                        forest_sampler_mean.reconstitute_from_forest(active_forest_mean, forest_dataset_train, residual_train, True)
                    if self.include_variance_forest:
                        active_forest_variance.reset_root()
                        active_forest_variance.set_root_leaves(log(variance_forest_leaf_init) / num_trees_mean)
                        forest_sampler_variance.reconstitute_from_forest(active_forest_variance, forest_dataset_train, residual_train, False)
            
                for i in range(self.num_gfr, num_temp_samples):
                    is_mcmc = i + 1 > num_gfr + num_burnin
                    if is_mcmc:
                        mcmc_counter = i - num_gfr - num_burnin + 1
                        if (mcmc_counter % keep_every == 0):
                            keep_sample = True
                        else:
                            keep_sample = False
                    else:
                        if keep_burnin:
                            keep_sample = True
                        else:
                            keep_sample = False
                    if keep_sample:
                        sample_counter += 1
                    # Sample the mean forest
                    if self.include_mean_forest:
                        forest_sampler_mean.sample_one_iteration(
                            self.forest_container_mean, active_forest_mean, forest_dataset_train, residual_train, 
                            cpp_rng, feature_types, cutpoint_grid_size, current_leaf_scale, variable_weights_mean, a_forest, b_forest, 
                            current_sigma2, leaf_model_mean_forest, keep_sample, False, True
                        )
                    
                    # Sample the variance forest
                    if self.include_variance_forest:
                        forest_sampler_variance.sample_one_iteration(
                            self.forest_container_variance, active_forest_variance, forest_dataset_train, residual_train, 
                            cpp_rng, feature_types, cutpoint_grid_size, current_leaf_scale, variable_weights_variance, a_forest, b_forest, 
                            current_sigma2, leaf_model_variance_forest, keep_sample, False, True
                        )

                    # Sample variance parameters (if requested)
                    if self.sample_sigma_global:
                        current_sigma2 = global_var_model.sample_one_iteration(residual_train, cpp_rng, a_global, b_global)
                        if keep_sample:
                            self.global_var_samples[sample_counter] = current_sigma2
                    if self.sample_sigma_leaf:
                        current_leaf_scale[0,0] = leaf_var_model.sample_one_iteration(active_forest_mean, cpp_rng, a_leaf, b_leaf)
                        if keep_sample:
                            self.leaf_scale_samples[sample_counter] = current_leaf_scale[0,0]
        
        # Mark the model as sampled
        self.sampled = True

        # Remove GFR samples if they are not to be retained
        if not keep_gfr and num_gfr > 0:
            for i in range(num_gfr):
                if self.include_mean_forest:
                    self.forest_container_mean.delete_sample(i)
                if self.include_variance_forest:
                    self.forest_container_variance.delete_sample(i)
            if self.sample_sigma_global:
                self.global_var_samples = self.global_var_samples[num_gfr:]
            if self.sample_sigma_leaf:
                self.leaf_scale_samples = self.leaf_scale_samples[num_gfr:]
            self.num_samples -= num_gfr

        # Store predictions
        if self.sample_sigma_global:
            self.global_var_samples = self.global_var_samples*self.y_std*self.y_std/self.variance_scale

        if self.sample_sigma_leaf:
            self.leaf_scale_samples = self.leaf_scale_samples
        
        if self.include_mean_forest:
            yhat_train_raw = self.forest_container_mean.forest_container_cpp.Predict(forest_dataset_train.dataset_cpp)
            self.y_hat_train = yhat_train_raw*self.y_std/np.sqrt(self.variance_scale) + self.y_bar
            if self.has_test:
                yhat_test_raw = self.forest_container_mean.forest_container_cpp.Predict(forest_dataset_test.dataset_cpp)
                self.y_hat_test = yhat_test_raw*self.y_std/np.sqrt(self.variance_scale) + self.y_bar
        
        if self.include_variance_forest:
            sigma_x_train_raw = self.forest_container_variance.forest_container_cpp.Predict(forest_dataset_train.dataset_cpp)
            if self.sample_sigma_global:
                self.sigma_x_train = sigma_x_train_raw
                for i in range(self.num_samples):
                    self.sigma_x_train[:,i] = np.sqrt(sigma_x_train_raw[:,i]*self.global_var_samples[i])
            else:
                self.sigma_x_train = np.sqrt(sigma_x_train_raw*self.sigma2_init)*self.y_std/np.sqrt(self.variance_scale)
            if self.has_test:
                sigma_x_test_raw = self.forest_container_variance.forest_container_cpp.Predict(forest_dataset_test.dataset_cpp)
                if self.sample_sigma_global:
                    self.sigma_x_test = sigma_x_test_raw
                    for i in range(self.num_samples):
                        self.sigma_x_test[:,i] = np.sqrt(sigma_x_test_raw[:,i]*self.global_var_samples[i])
                else:
                    self.sigma_x_test = np.sqrt(sigma_x_test_raw*self.sigma2_init)*self.y_std/np.sqrt(self.variance_scale)

    def predict(self, covariates: np.array, basis: np.array = None) -> np.array:
        """Return predictions from every forest sampled (either / both of mean and variance)

        Parameters
        ----------
        covariates : :obj:`np.array`
            Test set covariates.
        basis_train : :obj:`np.array`, optional
            Optional test set basis vector, must be provided if the model was trained with a leaf regression basis.
        
        Returns
        -------
        tuple of :obj:`np.array`
            Tuple of arrays of predictions corresponding to each forest (mean and variance, depending on whether either / both was included). Each array will contain as many rows as in ``covariates`` and as many columns as retained samples of the algorithm.
        """
        if not self.is_sampled():
            msg = (
                "This BARTModel instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this model."
            )
            raise NotSampledError(msg)
        
        # Convert everything to standard shape (2-dimensional)
        if covariates.ndim == 1:
            covariates = np.expand_dims(covariates, 1)
        if basis is not None:
            if basis.ndim == 1:
                basis = np.expand_dims(basis, 1)
        
        # Data checks
        if basis is not None:
            if basis.shape[0] != covariates.shape[0]:
                raise ValueError("covariates and basis must have the same number of rows")

        pred_dataset = Dataset()
        pred_dataset.add_covariates(covariates)
        if basis is not None:
            pred_dataset.add_basis(basis)
        if self.include_mean_forest:
            mean_pred_raw = self.forest_container_mean.forest_container_cpp.Predict(pred_dataset.dataset_cpp)
            mean_pred = mean_pred_raw*self.y_std/np.sqrt(self.variance_scale) + self.y_bar
        if self.include_variance_forest:
            variance_pred_raw = self.forest_container_variance.forest_container_cpp.Predict(pred_dataset.dataset_cpp)
            if self.sample_sigma_global:
                variance_pred = variance_pred_raw
                for i in range(self.num_samples):
                    variance_pred[:,i] = np.sqrt(variance_pred_raw[:,i]*self.global_var_samples[i])
            else:
                variance_pred = np.sqrt(variance_pred_raw*self.sigma2_init)*self.y_std/np.sqrt(self.variance_scale)

        if self.include_mean_forest and self.include_variance_forest:
            return (mean_pred, variance_pred)
        elif self.include_mean_forest and not self.include_variance_forest:
            return (mean_pred)
        elif not self.include_mean_forest and self.include_variance_forest:
            return (variance_pred)

    def predict_mean(self, covariates: np.array, basis: np.array = None) -> np.array:
        """Predict expected conditional outcome from a BART model.

        Parameters
        ----------
        covariates : :obj:`np.array`
            Test set covariates.
        basis_train : :obj:`np.array`, optional
            Optional test set basis vector, must be provided if the model was trained with a leaf regression basis.
        
        Returns
        -------
        tuple of :obj:`np.array`
            Tuple of arrays of predictions corresponding to each forest (mean and variance, depending on whether either / both was included). Each array will contain as many rows as in ``covariates`` and as many columns as retained samples of the algorithm.
        """
        if not self.is_sampled():
            msg = (
                "This BARTModel instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this model."
            )
            raise NotSampledError(msg)

        if not self.include_mean_forest:
            msg = (
                "This BARTModel instance was not sampled with a mean forest. "
                "Call 'fit' with appropriate arguments before using this model."
            )
            raise NotSampledError(msg)
        
        # Convert everything to standard shape (2-dimensional)
        if covariates.ndim == 1:
            covariates = np.expand_dims(covariates, 1)
        if basis is not None:
            if basis.ndim == 1:
                basis = np.expand_dims(basis, 1)
        
        # Data checks
        if basis is not None:
            if basis.shape[0] != covariates.shape[0]:
                raise ValueError("covariates and basis must have the same number of rows")

        pred_dataset = Dataset()
        pred_dataset.add_covariates(covariates)
        if basis is not None:
            pred_dataset.add_basis(basis)
        mean_pred_raw = self.forest_container_mean.forest_container_cpp.Predict(pred_dataset.dataset_cpp)
        mean_pred = mean_pred_raw*self.y_std/np.sqrt(self.variance_scale) + self.y_bar

        return mean_pred

    def predict_variance(self, covariates: np.array, basis: np.array = None) -> np.array:
        """Predict expected conditional variance from a BART model.

        Parameters
        ----------
        covariates : np.array
            Test set covariates.
        basis_train : :obj:`np.array`, optional
            Optional test set basis vector, must be provided if the model was trained with a leaf regression basis.
        
        Returns
        -------
        tuple of :obj:`np.array`
            Tuple of arrays of predictions corresponding to each forest (mean and variance, depending on whether either / both was included). Each array will contain as many rows as in ``covariates`` and as many columns as retained samples of the algorithm.
        """
        if not self.is_sampled():
            msg = (
                "This BARTModel instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this model."
            )
            raise NotSampledError(msg)

        if not self.include_variance_forest:
            msg = (
                "This BARTModel instance was not sampled with a variance forest. "
                "Call 'fit' with appropriate arguments before using this model."
            )
            raise NotSampledError(msg)
        
        # Convert everything to standard shape (2-dimensional)
        if covariates.ndim == 1:
            covariates = np.expand_dims(covariates, 1)
        if basis is not None:
            if basis.ndim == 1:
                basis = np.expand_dims(basis, 1)
        
        # Data checks
        if basis is not None:
            if basis.shape[0] != covariates.shape[0]:
                raise ValueError("covariates and basis must have the same number of rows")

        pred_dataset = Dataset()
        pred_dataset.add_covariates(covariates)
        # if basis is not None:
        #     pred_dataset.add_basis(basis)
        variance_pred_raw = self.forest_container_variance.forest_container_cpp.Predict(pred_dataset.dataset_cpp)
        if self.sample_sigma_global:
            variance_pred = variance_pred_raw
            for i in range(self.num_samples):
                variance_pred[:,i] = np.sqrt(variance_pred_raw[:,i]*self.global_var_samples[i])
        else:
            variance_pred = np.sqrt(variance_pred_raw*self.sigma2_init)*self.y_std/np.sqrt(self.variance_scale)

        return variance_pred
    
    def to_json(self) -> str:
        """
        Converts a sampled BART model to JSON string representation (which can then be saved to a file or 
        processed using the ``json`` library)

        Returns
        -------
        :obj:`str`
            JSON string representing model metadata (hyperparameters), sampled parameters, and sampled forests
        """
        if not self.is_sampled:
            msg = (
                "This BARTModel instance has not yet been sampled. "
                "Call 'fit' with appropriate arguments before using this model."
            )
            raise NotSampledError(msg)
        
        # Initialize JSONSerializer object
        bart_json = JSONSerializer()
        
        # Add the forests
        if self.include_mean_forest:
            bart_json.add_forest(self.forest_container_mean)
        if self.include_variance_forest:
            bart_json.add_forest(self.forest_container_variance)
        
        # Add global parameters
        bart_json.add_scalar("variance_scale", self.variance_scale)
        bart_json.add_scalar("outcome_scale", self.y_std)
        bart_json.add_scalar("outcome_mean", self.y_bar)
        bart_json.add_boolean("standardize", self.standardize)
        bart_json.add_scalar("sigma2_init", self.sigma2_init)
        bart_json.add_boolean("sample_sigma_global", self.sample_sigma_global)
        bart_json.add_boolean("sample_sigma_leaf", self.sample_sigma_leaf)
        bart_json.add_boolean("include_mean_forest", self.include_mean_forest)
        bart_json.add_boolean("include_variance_forest", self.include_variance_forest)
        bart_json.add_scalar("num_gfr", self.num_gfr)
        bart_json.add_scalar("num_burnin", self.num_burnin)
        bart_json.add_scalar("num_mcmc", self.num_mcmc)
        bart_json.add_scalar("num_samples", self.num_samples)
        bart_json.add_scalar("num_basis", self.num_basis)
        bart_json.add_boolean("requires_basis", self.has_basis)
        
        # Add parameter samples
        if self.sample_sigma_global:
            bart_json.add_numeric_vector("sigma2_global_samples", self.global_var_samples, "parameters")
        if self.sample_sigma_leaf:
            bart_json.add_numeric_vector("sigma2_leaf_samples", self.leaf_scale_samples, "parameters")
        
        return bart_json.return_json_string()

    def from_json(self, json_string: str) -> None:
        """
        Converts a JSON string to an in-memory BART model.

        Parameters
        ----------
        json_string : :obj:`str`
            JSON string representing model metadata (hyperparameters), sampled parameters, and sampled forests
        """
        # Parse string to a JSON object in C++
        bart_json = JSONSerializer()
        bart_json.load_from_json_string(json_string)
        
        # Unpack forests
        self.include_mean_forest = bart_json.get_boolean("include_mean_forest")
        self.include_variance_forest = bart_json.get_boolean("include_variance_forest")
        if self.include_mean_forest:
            # TODO: don't just make this a placeholder that we overwrite
            self.forest_container_mean = ForestContainer(0, 0, False, False)
            self.forest_container_mean.forest_container_cpp.LoadFromJson(bart_json.json_cpp, "forest_0")
            if self.include_variance_forest:
                # TODO: don't just make this a placeholder that we overwrite
                self.forest_container_variance = ForestContainer(0, 0, False, False)
                self.forest_container_variance.forest_container_cpp.LoadFromJson(bart_json.json_cpp, "forest_1")
        else:
            # TODO: don't just make this a placeholder that we overwrite
            self.forest_container_variance = ForestContainer(0, 0, False, False)
            self.forest_container_variance.forest_container_cpp.LoadFromJson(bart_json.json_cpp, "forest_0")
        
        # Unpack global parameters
        self.variance_scale = bart_json.get_scalar("variance_scale")
        self.y_std = bart_json.get_scalar("outcome_scale")
        self.y_bar = bart_json.get_scalar("outcome_mean")
        self.standardize = bart_json.get_boolean("standardize")
        self.sigma2_init = bart_json.get_scalar("sigma2_init")
        self.sample_sigma_global = bart_json.get_boolean("sample_sigma_global")
        self.sample_sigma_leaf = bart_json.get_boolean("sample_sigma_leaf")
        self.num_gfr = bart_json.get_scalar("num_gfr")
        self.num_burnin = bart_json.get_scalar("num_burnin")
        self.num_mcmc = bart_json.get_scalar("num_mcmc")
        self.num_samples = bart_json.get_scalar("num_samples")
        self.num_basis = bart_json.get_scalar("num_basis")
        self.has_basis = bart_json.get_boolean("requires_basis")

        # Unpack parameter samples
        if self.sample_sigma_global:
            self.global_var_samples = bart_json.get_numeric_vector("sigma2_global_samples", "parameters")
        if self.sample_sigma_leaf:
            self.leaf_scale_samples = bart_json.get_numeric_vector("sigma2_leaf_samples", "parameters")
        
        # Mark the deserialized model as "sampled"
        self.sampled = True
