"""
Bayesian Additive Regression Trees (BART) module
"""
import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from scipy.stats import gamma
from .data import Dataset, Residual
from .forest import ForestContainer
from .preprocessing import CovariateTransformer
from .sampler import ForestSampler, RNG, GlobalVarianceModel, LeafVarianceModel
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
               cutpoint_grid_size = 100, sigma_leaf: float = None, alpha: float = 0.95, beta: float = 2.0, min_samples_leaf: int = 5, 
               nu: float = 3, lamb: float = None, a_leaf: float = 3, b_leaf: float = None, q: float = 0.9, sigma2: float = None, 
               num_trees: int = 200, num_gfr: int = 5, num_burnin: int = 0, num_mcmc: int = 100, sample_sigma_global: bool = True, 
               sample_sigma_leaf: bool = True, random_seed: int = -1, keep_burnin: bool = False, keep_gfr: bool = False) -> None:
        """Runs a BART sampler on provided training set. Predictions will be cached for the training set and (if provided) the test set. 
        Does not require a leaf regression basis. 

        Parameters
        ----------
        X_train : np.array
            Training set covariates on which trees may be partitioned.
        y_train : np.array
            Training set outcome.
        basis_train : :obj:`np.array`, optional
            Optional training set basis vector used to define a regression to be run in the leaves of each tree.
        X_test : :obj:`np.array`, optional
            Optional test set covariates.
        basis_test : :obj:`np.array`, optional
            Optional test set basis vector used to define a regression to be run in the leaves of each tree. 
            Must be included / omitted consistently (i.e. if basis_train is provided, then basis_test must be provided alongside X_test).
        cutpoint_grid_size : :obj:`int`, optional
            Maximum number of cutpoints to consider for each feature. Defaults to ``100``.
        sigma_leaf : :obj:`float`, optional
            Scale parameter on the leaf node regression model.
        alpha : :obj:`float`, optional
            Prior probability of splitting for a tree of depth 0. 
            Tree split prior combines ``alpha`` and ``beta`` via ``alpha*(1+node_depth)^-beta``.
        beta : :obj:`float`, optional
            Exponent that decreases split probabilities for nodes of depth > 0. 
            Tree split prior combines ``alpha`` and ``beta`` via ``alpha*(1+node_depth)^-beta``.
        min_samples_leaf : :obj:`int`, optional
            Minimum allowable size of a leaf, in terms of training samples. Defaults to ``5``.
        nu : :obj:`float`, optional
            Shape parameter in the ``IG(nu, nu*lamb)`` global error variance model. Defaults to ``3``.
        lamb : :obj:`float`, optional
            Component of the scale parameter in the ``IG(nu, nu*lambda)`` global error variance prior. If not specified, this is calibrated as in Sparapani et al (2021).
        a_leaf : :obj:`float`, optional
            Shape parameter in the ``IG(a_leaf, b_leaf)`` leaf node parameter variance model. Defaults to ``3``.
        b_leaf : :obj:`float`, optional
            Scale parameter in the ``IG(a_leaf, b_leaf)`` leaf node parameter variance model. Calibrated internally as ``0.5/num_trees`` if not set here.
        q : :obj:`float`, optional
            Quantile used to calibrated ``lamb`` as in Sparapani et al (2021). Defaults to ``0.9``.
        sigma2 : :obj:`float`, optional
            Starting value of global variance parameter. Calibrated internally as in Sparapani et al (2021) if not set here.
        num_trees : :obj:`int`, optional
            Number of trees in the ensemble. Defaults to ``200``.
        num_gfr : :obj:`int`, optional
            Number of "warm-start" iterations run using the grow-from-root algorithm (He and Hahn, 2021). Defaults to ``5``.
        num_burnin : :obj:`int`, optional
            Number of "burn-in" iterations of the MCMC sampler. Defaults to ``0``. Ignored if ``num_gfr > 0``.
        num_mcmc : :obj:`int`, optional
            Number of "retained" iterations of the MCMC sampler. Defaults to ``100``. If this is set to 0, GFR (XBART) samples will be retained.
        sample_sigma_global : :obj:`bool`, optional
            Whether or not to update the ``sigma^2`` global error variance parameter based on ``IG(nu, nu*lambda)``. Defaults to ``True``.
        sample_sigma_leaf : :obj:`bool`, optional
            Whether or not to update the ``tau`` leaf scale variance parameter based on ``IG(a_leaf, b_leaf)``. Cannot (currently) be set to true if ``basis_train`` has more than one column. Defaults to ``True``.
        random_seed : :obj:`int`, optional
            Integer parameterizing the C++ random number generator. If not specified, the C++ random number generator is seeded according to ``std::random_device``.
        keep_burnin : :obj:`bool`, optional
            Whether or not "burnin" samples should be included in predictions. Defaults to ``False``. Ignored if ``num_mcmc == 0``.
        keep_gfr : :obj:`bool`, optional
            Whether or not "warm-start" / grow-from-root samples should be included in predictions. Defaults to ``False``. Ignored if ``num_mcmc == 0``.
        
        Returns
        -------
        self : BARTModel
            Sampled BART Model.
        """
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
        if X_train.ndim == 1:
            X_train = np.expand_dims(X_train, 1)
        if basis_train is not None:
            if basis_train.ndim == 1:
                basis_train = np.expand_dims(basis_train, 1)
        if y_train.ndim == 1:
            y_train = np.expand_dims(y_train, 1)
        if X_test is not None:
            if X_train.ndim == 1:
                X_train = np.expand_dims(X_train, 1)
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
        
        # Set variable weights for the prognostic and treatment effect forests
        variable_weights = np.repeat(1.0/self.num_covariates, self.num_covariates)

        # Scale outcome
        self.y_bar = np.squeeze(np.mean(y_train))
        self.y_std = np.squeeze(np.std(y_train))
        resid_train = (y_train-self.y_bar)/self.y_std

        # Calibrate priors for global sigma^2 and sigma_leaf
        if lamb is None:
            reg_basis = np.c_[np.ones(self.n_train),X_train_processed]
            reg_soln = lstsq(reg_basis, np.squeeze(resid_train))
            sigma2hat = reg_soln[1] / self.n_train
            quantile_cutoff = q
            lamb = (sigma2hat*gamma.ppf(1-quantile_cutoff,nu))/nu
        sigma2 = sigma2hat if sigma2 is None else sigma2
        b_leaf = np.squeeze(np.var(resid_train)) / num_trees if b_leaf is None else b_leaf
        sigma_leaf = np.squeeze(np.var(resid_train)) / num_trees if sigma_leaf is None else sigma_leaf
        current_sigma2 = sigma2
        current_leaf_scale = np.array([[sigma_leaf]])

        # Container of variance parameter samples
        self.num_gfr = num_gfr
        self.num_burnin = num_burnin
        self.num_mcmc = num_mcmc
        self.num_samples = num_gfr + num_burnin + num_mcmc
        self.sample_sigma_global = sample_sigma_global
        self.sample_sigma_leaf = sample_sigma_leaf
        if sample_sigma_global:
            self.global_var_samples = np.zeros(self.num_samples)
        if sample_sigma_leaf:
            self.leaf_scale_samples = np.zeros(self.num_samples)
        
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
        forest_sampler = ForestSampler(forest_dataset_train, feature_types, num_trees, self.n_train, alpha, beta, min_samples_leaf)

        # Determine the leaf model
        if not self.has_basis:
            leaf_model_int = 0
        elif self.num_basis == 1:
            leaf_model_int = 1
        else:
            leaf_model_int = 2
        
        # Container of forest samples
        self.forest_container = ForestContainer(num_trees, 1, True) if not self.has_basis else ForestContainer(num_trees, self.num_basis, False)
        
        # Variance samplers
        if self.sample_sigma_global:
            global_var_model = GlobalVarianceModel()
        if self.sample_sigma_leaf:
            leaf_var_model = LeafVarianceModel()

        # Initialize the leaves of each tree in the prognostic forest
        init_root = np.squeeze(np.mean(resid_train)) / num_trees
        self.forest_container.set_root_leaves(0, init_root)
        forest_sampler.update_residual(forest_dataset_train, residual_train, self.forest_container, False, 0, True)

        # Run GFR (warm start) if specified
        if self.num_gfr > 0:
            gfr_indices = np.arange(self.num_gfr)
            for i in range(self.num_gfr):
                # Sample the forest
                forest_sampler.sample_one_iteration(
                    self.forest_container, forest_dataset_train, residual_train, cpp_rng, feature_types, 
                    cutpoint_grid_size, current_leaf_scale, variable_weights, current_sigma2, leaf_model_int, True, True
                )

                # Sample variance parameters (if requested)
                if self.sample_sigma_global:
                    current_sigma2 = global_var_model.sample_one_iteration(residual_train, cpp_rng, nu, lamb)
                    self.global_var_samples[i] = current_sigma2*self.y_std*self.y_std
                if self.sample_sigma_leaf:
                    self.leaf_scale_samples[i] = leaf_var_model.sample_one_iteration(self.forest_container, cpp_rng, a_leaf, b_leaf, i)
                    current_leaf_scale[0,0] = self.leaf_scale_samples[i]
        
        # Run MCMC
        if self.num_burnin + self.num_mcmc > 0:
            if self.num_burnin > 0:
                burnin_indices = np.arange(self.num_gfr, self.num_gfr + self.num_burnin)
            if self.num_mcmc > 0:
                mcmc_indices = np.arange(self.num_gfr + self.num_burnin, self.num_gfr + self.num_burnin + self.num_mcmc)
            for i in range(self.num_gfr, self.num_samples):
                # Sample the forest
                forest_sampler.sample_one_iteration(
                    self.forest_container, forest_dataset_train, residual_train, cpp_rng, feature_types, 
                    cutpoint_grid_size, current_leaf_scale, variable_weights, current_sigma2, leaf_model_int, False, True
                )

                # Sample variance parameters (if requested)
                if self.sample_sigma_global:
                    current_sigma2 = global_var_model.sample_one_iteration(residual_train, cpp_rng, nu, lamb)
                    self.global_var_samples[i] = current_sigma2*self.y_std*self.y_std
                if self.sample_sigma_leaf:
                    self.leaf_scale_samples[i] = leaf_var_model.sample_one_iteration(self.forest_container, cpp_rng, a_leaf, b_leaf, i)
                    current_leaf_scale[0,0] = self.leaf_scale_samples[i]
        
        # Mark the model as sampled
        self.sampled = True

        # Prediction indices to be stored
        if self.num_mcmc > 0:
            self.keep_indices = mcmc_indices
            if keep_gfr:
                self.keep_indices = np.concatenate((gfr_indices, self.keep_indices))
            else:
                # Don't retain both GFR and burnin samples
                if keep_burnin:
                    self.keep_indices = np.concatenate((burnin_indices, self.keep_indices))
        else:
            if self.num_gfr > 0 and self.num_burnin > 0:
                # Override keep_gfr = False since there are no MCMC samples
                # Don't retain both GFR and burnin samples
                self.keep_indices = gfr_indices
            elif self.num_gfr <= 0 and self.num_burnin > 0:
                self.keep_indices = burnin_indices
            elif self.num_gfr > 0 and self.num_burnin <= 0:
                self.keep_indices = gfr_indices
            else:
                raise RuntimeError("There are no samples to retain!")
        
        # Store predictions
        yhat_train_raw = self.forest_container.forest_container_cpp.Predict(forest_dataset_train.dataset_cpp)[:,self.keep_indices]
        self.y_hat_train = yhat_train_raw*self.y_std + self.y_bar
        if self.has_test:
            yhat_test_raw = self.forest_container.forest_container_cpp.Predict(forest_dataset_test.dataset_cpp)[:,self.keep_indices]
            self.y_hat_test = yhat_test_raw*self.y_std + self.y_bar
    
    def predict(self, covariates: np.array, basis: np.array = None) -> np.array:
        """Predict outcome from every retained forest of a BART sampler.

        Parameters
        ----------
        covariates : np.array
            Test set covariates.
        basis_train : :obj:`np.array`, optional
            Optional test set basis vector, must be provided if the model was trained with a leaf regression basis.
        
        Returns
        -------
        np.array
            Array of predictions with as many rows as in ``covariates`` and as many columns as retained samples of the algorithm.
        """
        if not self.is_sampled():
            msg = (
                "This BCFModel instance is not fitted yet. Call 'fit' with "
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
        pred_raw = self.forest_container.forest_container_cpp.Predict(pred_dataset.dataset_cpp)
        return pred_raw[:,self.keep_indices]*self.y_std + self.y_bar
