"""
Bayesian Causal Forests (BCF) module
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold
from typing import Optional
from scipy.linalg import lstsq
from scipy.stats import gamma
from .bart import BARTModel
from .data import Dataset, Residual
from .forest import ForestContainer
from .preprocessing import CovariateTransformer
from .sampler import ForestSampler, RNG, GlobalVarianceModel, LeafVarianceModel
from .utils import NotSampledError

class BCFModel:
    """Class that handles sampling, storage, and serialization of causal BART models like BCF, XBCF, and Warm-Start BCF
    """

    def __init__(self) -> None:
        # Internal flag for whether the sample() method has been run
        self.sampled = False
        self.rng = np.random.default_rng()
    
    def is_sampled(self) -> bool:
        return self.sampled
    
    def sample(self, X_train: np.array, Z_train: np.array, y_train: np.array, pi_train: np.array = None, 
               X_test: np.array = None, Z_test: np.array = None, pi_test: np.array = None, 
               cutpoint_grid_size = 100, sigma_leaf_mu: float = None, sigma_leaf_tau: float = None, 
               alpha_mu: float = 0.95, alpha_tau: float = 0.25, beta_mu: float = 2.0, beta_tau: float = 3.0, 
               min_samples_leaf_mu: int = 5, min_samples_leaf_tau: int = 5, nu: float = 3, lamb: float = None, 
               a_leaf_mu: float = 3, a_leaf_tau: float = 3, b_leaf_mu: float = None, b_leaf_tau: float = None, 
               q: float = 0.9, sigma2: float = None, num_trees_mu: int = 200, num_trees_tau: int = 50, 
               num_gfr: int = 5, num_burnin: int = 0, num_mcmc: int = 100, sample_sigma_global: bool = True, 
               sample_sigma_leaf_mu: bool = True, sample_sigma_leaf_tau: bool = False, propensity_covariate: str = "mu", 
               adaptive_coding: bool = True, b_0: float = -0.5, b_1: float = 0.5, random_seed: int = -1, 
               keep_burnin: bool = False, keep_gfr: bool = False) -> None:
        """Runs a BCF sampler on provided training set. Outcome predictions and estimates of the prognostic and treatment effect functions 
        will be cached for the training set and (if provided) the test set.

        Parameters
        ----------
        X_train : np.array or pd.DataFrame
            Covariates used to split trees in the ensemble. Can be passed as either a matrix or dataframe.
        Z_train : np.array
            Array of (continuous or binary) treatment assignments.
        y_train : np.array
            Outcome to be modeled by the ensemble.
        pi_train : np.array
            Optional vector of propensity scores. If not provided, this will be estimated from the data.
        X_test : :obj:`np.array`, optional
            Optional test set of covariates used to define "out of sample" evaluation data.
        Z_test : :obj:`np.array`, optional
            Optional test set of (continuous or binary) treatment assignments.
            Must be provided if ``X_test`` is provided.
        pi_test : :obj:`np.array`, optional
            Optional test set vector of propensity scores. If not provided (but ``X_test`` and ``Z_test`` are), this will be estimated from the data.
        cutpoint_grid_size : :obj:`int`, optional
            Maximum number of cutpoints to consider for each feature. Defaults to ``100``.
        sigma_leaf_mu : :obj:`float`, optional
            Starting value of leaf node scale parameter for the prognostic forest. Calibrated internally as ``2/num_trees_mu`` if not set here.
        sigma_leaf_tau : :obj:`float`, optional
            Starting value of leaf node scale parameter for the treatment effect forest. Calibrated internally as ``1/num_trees_mu`` if not set here.
        alpha_mu : :obj:`float`, optional
            Prior probability of splitting for a tree of depth 0 for the prognostic forest. 
            Tree split prior combines ``alpha`` and ``beta`` via ``alpha*(1+node_depth)^-beta``.
        alpha_tau : :obj:`float`, optional
            Prior probability of splitting for a tree of depth 0 for the treatment effect forest. 
            Tree split prior combines ``alpha`` and ``beta`` via ``alpha*(1+node_depth)^-beta``.
        beta_mu : :obj:`float`, optional
            Exponent that decreases split probabilities for nodes of depth > 0 for the prognostic forest. 
            Tree split prior combines ``alpha`` and ``beta`` via ``alpha*(1+node_depth)^-beta``.
        beta_tau : :obj:`float`, optional
            Exponent that decreases split probabilities for nodes of depth > 0 for the treatment effect forest. 
            Tree split prior combines ``alpha`` and ``beta`` via ``alpha*(1+node_depth)^-beta``.
        min_samples_leaf_mu : :obj:`int`, optional
            Minimum allowable size of a leaf, in terms of training samples, for the prognostic forest. Defaults to ``5``.
        min_samples_leaf_tau : :obj:`int`, optional
            Minimum allowable size of a leaf, in terms of training samples, for the treatment effect forest. Defaults to ``5``.
        nu : :obj:`float`, optional
            Shape parameter in the ``IG(nu, nu*lamb)`` global error variance model. Defaults to ``3``.
        lamb : :obj:`float`, optional
            Component of the scale parameter in the ``IG(nu, nu*lambda)`` global error variance prior. If not specified, this is calibrated as in Sparapani et al (2021).
        a_leaf_mu : :obj:`float`, optional
            Shape parameter in the ``IG(a_leaf, b_leaf)`` leaf node parameter variance model for the prognostic forest. Defaults to ``3``.
        a_leaf_tau : :obj:`float`, optional
            Shape parameter in the ``IG(a_leaf, b_leaf)`` leaf node parameter variance model for the treatment effect forest. Defaults to ``3``.
        b_leaf_mu : :obj:`float`, optional
            Scale parameter in the ``IG(a_leaf, b_leaf)`` leaf node parameter variance model for the prognostic forest. Calibrated internally as ``0.5/num_trees`` if not set here.
        b_leaf_tau : :obj:`float`, optional
            Scale parameter in the ``IG(a_leaf, b_leaf)`` leaf node parameter variance model for the treatment effect forest. Calibrated internally as ``0.5/num_trees`` if not set here.
        q : :obj:`float`, optional
            Quantile used to calibrated ``lamb`` as in Sparapani et al (2021). Defaults to ``0.9``.
        sigma2 : :obj:`float`, optional
            Starting value of global variance parameter. Calibrated internally as in Sparapani et al (2021) if not set here.
        num_trees_mu : :obj:`int`, optional
            Number of trees in the prognostic forest. Defaults to ``200``.
        num_trees_tau : :obj:`int`, optional
            Number of trees in the treatment effect forest. Defaults to ``50``.
        num_gfr : :obj:`int`, optional
            Number of "warm-start" iterations run using the grow-from-root algorithm (He and Hahn, 2021). Defaults to ``5``.
        num_burnin : :obj:`int`, optional
            Number of "burn-in" iterations of the MCMC sampler. Defaults to ``0``. Ignored if ``num_gfr > 0``.
        num_mcmc : :obj:`int`, optional
            Number of "retained" iterations of the MCMC sampler. Defaults to ``100``. If this is set to 0, GFR (XBART) samples will be retained.
        sample_sigma_global : :obj:`bool`, optional
            Whether or not to update the ``sigma^2`` global error variance parameter based on ``IG(nu, nu*lambda)``. Defaults to ``True``.
        sample_sigma_leaf_mu : :obj:`bool`, optional
            Whether or not to update the ``tau`` leaf scale variance parameter based on ``IG(a_leaf, b_leaf)`` for the prognostic forest. 
            Cannot (currently) be set to true if ``basis_train`` has more than one column. Defaults to ``True``.
        sample_sigma_leaf_tau : :obj:`bool`, optional
            Whether or not to update the ``tau`` leaf scale variance parameter based on ``IG(a_leaf, b_leaf)`` for the treatment effect forest. 
            Cannot (currently) be set to true if ``basis_train`` has more than one column. Defaults to ``True``.
        propensity_covariate : :obj:`str`, optional
            Whether to include the propensity score as a covariate in either or both of the forests. Enter ``"none"`` for neither, ``"mu"`` for the prognostic forest, ``"tau"`` for the treatment forest, and ``"both"`` for both forests. 
            If this is not ``"none"`` and a propensity score is not provided, it will be estimated from (``X_train``, ``Z_train``) using ``BARTModel``. Defaults to ``"mu"``.
        adaptive_coding : :obj:`bool`, optional
            Whether or not to use an "adaptive coding" scheme in which a binary treatment variable is not coded manually as (0,1) or (-1,1) but learned via 
            parameters ``b_0`` and ``b_1`` that attach to the outcome model ``[b_0 (1-Z) + b_1 Z] tau(X)``. This is ignored when Z is not binary. Defaults to True.
        b_0 : :obj:`float`, optional
            Initial value of the "control" group coding parameter. This is ignored when ``Z`` is not binary. Default: ``-0.5``.
        b_1 : :obj:`float`, optional
            Initial value of the "treated" group coding parameter. This is ignored when ``Z`` is not binary. Default: ``0.5``.
        random_seed : :obj:`int`, optional
            Integer parameterizing the C++ random number generator. If not specified, the C++ random number generator is seeded according to ``std::random_device``.
        keep_burnin : :obj:`bool`, optional
            Whether or not "burnin" samples should be included in predictions. Defaults to ``False``. Ignored if ``num_mcmc == 0``.
        keep_gfr : :obj:`bool`, optional
            Whether or not "warm-start" / grow-from-root samples should be included in predictions. Defaults to ``False``. Ignored if ``num_mcmc == 0``.
        
        Returns
        -------
        self : BCFModel
            Sampled BCF Model.
        """
        # Convert everything to standard shape (2-dimensional)
        if X_train.ndim == 1:
            X_train = np.expand_dims(X_train, 1)
        if Z_train.ndim == 1:
            Z_train = np.expand_dims(Z_train, 1)
        if y_train.ndim == 1:
            y_train = np.expand_dims(y_train, 1)
        if pi_train is not None:
            if pi_train.ndim == 1:
                pi_train = np.expand_dims(pi_train, 1)
        if X_test is not None:
            if X_train.ndim == 1:
                X_train = np.expand_dims(X_train, 1)
        if Z_test is not None:
            if Z_test.ndim == 1:
                Z_test = np.expand_dims(Z_test, 1)
        if pi_test is not None:
            if pi_test.ndim == 1:
                pi_test = np.expand_dims(pi_test, 1)
        
        # Data checks
        if X_test is not None:
            if X_test.shape[1] != X_train.shape[1]:
                raise ValueError("X_train and X_test must have the same number of columns")
        if Z_test is not None:
            if Z_test.shape[1] != Z_train.shape[1]:
                raise ValueError("Z_train and Z_test must have the same number of columns")
        if Z_train.shape[0] != X_train.shape[0]:
            raise ValueError("X_train and Z_train must have the same number of rows")
        if y_train.shape[0] != X_train.shape[0]:
            raise ValueError("X_train and y_train must have the same number of rows")
        if pi_train is not None:
            if pi_train.shape[0] != X_train.shape[0]:
                raise ValueError("X_train and pi_train must have the same number of rows")
        if X_test is not None and Z_test is not None:
            if X_test.shape[0] != Z_test.shape[0]:
                raise ValueError("X_test and Z_test must have the same number of rows")
        if X_test is not None and pi_test is not None:
            if X_test.shape[0] != pi_test.shape[0]:
                raise ValueError("X_test and pi_test must have the same number of rows")
        
        # Covariate preprocessing
        self._covariate_transformer = CovariateTransformer()
        self._covariate_transformer.fit(X_train)
        X_train_processed = self._covariate_transformer.transform(X_train)
        if X_test is not None:
            X_test_processed = self._covariate_transformer.transform(X_test)
        feature_types = np.asarray(self._covariate_transformer._processed_feature_types)

        # Determine whether a test set is provided
        self.has_test = X_test is not None
        
        # Unpack data dimensions
        self.n_train = y_train.shape[0]
        self.n_test = X_test.shape[0] if self.has_test else 0
        self.p_x = X_train_processed.shape[1]

        # Check whether treatment is binary
        self.binary_treatment = np.unique(Z_train).size == 2

        # Adaptive coding will be ignored for continuous / ordered categorical treatments
        self.adaptive_coding = adaptive_coding
        if adaptive_coding and not self.binary_treatment:
            self.adaptive_coding = False

        # Check if user has provided propensities that are needed in the model
        if pi_train is None and propensity_covariate != "none":
            self.bart_propensity_model = BARTModel()
            if self.has_test:
                pi_test = np.mean(self.bart_propensity_model.y_hat_test, axis = 1, keepdims = True)
                self.bart_propensity_model.sample(X_train=X_train_processed, y_train=Z_train, X_test=X_test_processed, num_gfr=10, num_mcmc=10)
                pi_train = np.mean(self.bart_propensity_model.y_hat_train, axis = 1, keepdims = True)
                pi_test = np.mean(self.bart_propensity_model.y_hat_test, axis = 1, keepdims = True)
            else:
                self.bart_propensity_model.sample(X_train=X_train_processed, y_train=Z_train, num_gfr=10, num_mcmc=10)
                pi_train = np.mean(self.bart_propensity_model.y_hat_train, axis = 1, keepdims = True)
            self.internal_propensity_model = True
        else:
            self.internal_propensity_model = False
        
        # Update covariates to include propensities if requested
        if propensity_covariate == "mu":
            feature_types_mu = np.append(feature_types, 0).astype('int')
            feature_types_tau = feature_types.astype('int')
            X_train_mu = np.c_[X_train_processed, pi_train]
            X_train_tau = X_train_processed
            if self.has_test:
                X_test_mu = np.c_[X_test, pi_test]
                X_test_tau = X_test
        elif propensity_covariate == "tau":
            feature_types_tau = np.append(feature_types, 0).astype('int')
            feature_types_mu = feature_types.astype('int')
            X_train_tau = np.c_[X_train_processed, pi_train]
            X_train_mu = X_train_processed
            if self.has_test:
                X_test_tau = np.c_[X_test, pi_test]
                X_test_mu = X_test
        elif propensity_covariate == "both":
            feature_types_tau = np.append(feature_types, 0).astype('int')
            feature_types_mu = np.append(feature_types, 0).astype('int')
            X_train_tau = np.c_[X_train_processed, pi_train]
            X_train_mu = np.c_[X_train_processed, pi_train]
            if self.has_test:
                X_test_tau = np.c_[X_test, pi_test]
                X_test_mu = np.c_[X_test, pi_test]
        elif propensity_covariate == "none":
            feature_types_tau = feature_types.astype('int')
            feature_types_mu = feature_types.astype('int')
            X_train_tau = X_train_processed
            X_train_mu = X_train_processed
            if self.has_test:
                X_test_tau = X_test
                X_test_mu = X_test
        else:
            raise ValueError("propensity_covariate must be one of 'mu', 'tau', 'both', or 'none'")
        
        # Store propensity score requirements of the BCF forests
        self.propensity_covariate = propensity_covariate
        
        # Set variable weights for the prognostic and treatment effect forests
        variable_weights_mu = np.repeat(1.0/X_train_mu.shape[1], X_train_mu.shape[1])
        variable_weights_tau = np.repeat(1.0/X_train_tau.shape[1], X_train_tau.shape[1])

        # Scale outcome
        self.y_bar = np.squeeze(np.mean(y_train))
        self.y_std = np.squeeze(np.std(y_train))
        resid_train = (y_train-self.y_bar)/self.y_std

        # Calibrate priors for global sigma^2 and sigma_leaf_mu / sigma_leaf_tau
        if lamb is None:
            reg_basis = np.c_[np.ones(self.n_train),X_train_processed]
            reg_soln = lstsq(reg_basis, np.squeeze(resid_train))
            sigma2hat = reg_soln[1] / self.n_train
            quantile_cutoff = q
            lamb = (sigma2hat*gamma.ppf(1-quantile_cutoff,nu))/nu
        sigma2 = sigma2hat if sigma2 is None else sigma2
        b_leaf_mu = np.squeeze(np.var(resid_train)) / num_trees_mu if b_leaf_mu is None else b_leaf_mu
        b_leaf_tau = np.squeeze(np.var(resid_train)) / (2*num_trees_tau) if b_leaf_tau is None else b_leaf_tau
        sigma_leaf_mu = np.squeeze(np.var(resid_train)) / num_trees_mu if sigma_leaf_mu is None else sigma_leaf_mu
        sigma_leaf_tau = np.squeeze(np.var(resid_train)) / (2*num_trees_tau) if sigma_leaf_tau is None else sigma_leaf_tau
        current_sigma2 = sigma2
        current_leaf_scale_mu = np.array([[sigma_leaf_mu]])
        current_leaf_scale_tau = np.array([[sigma_leaf_tau]])

        # Container of variance parameter samples
        self.num_gfr = num_gfr
        self.num_burnin = num_burnin
        self.num_mcmc = num_mcmc
        self.num_samples = num_gfr + num_burnin + num_mcmc
        self.sample_sigma_global = sample_sigma_global
        self.sample_sigma_leaf_mu = sample_sigma_leaf_mu
        self.sample_sigma_leaf_tau = sample_sigma_leaf_tau
        if sample_sigma_global:
            self.global_var_samples = np.zeros(self.num_samples)
        if sample_sigma_leaf_mu:
            self.leaf_scale_mu_samples = np.zeros(self.num_samples)
        if sample_sigma_leaf_tau:
            self.leaf_scale_tau_samples = np.zeros(self.num_samples)
        
        # Prepare adaptive coding structure
        if self.adaptive_coding:
            if np.size(b_0) > 1 or np.size(b_1) > 1:
                raise ValueError("b_0 and b_1 must be single numeric values")
            if not (isinstance(b_0, (int, float)) or isinstance(b_1, (int, float))):
                raise ValueError("b_0 and b_1 must be numeric values")
            self.b0_samples = np.zeros(self.num_samples)
            self.b1_samples = np.zeros(self.num_samples)
            current_b_0 = b_0
            current_b_1 = b_1
            tau_basis_train = (1-Z_train)*current_b_0 + Z_train*current_b_1
            if self.has_test:
                tau_basis_test = (1-Z_test)*current_b_0 + Z_test*current_b_1
        else:
            tau_basis_train = Z_train
            if self.has_test:
                tau_basis_test = Z_test

        # Prognostic Forest Dataset (covariates)
        forest_dataset_mu_train = Dataset()
        forest_dataset_mu_train.add_covariates(X_train_mu)
        if self.has_test:
            forest_dataset_mu_test = Dataset()
            forest_dataset_mu_test.add_covariates(X_test_mu)

        # Treatment Forest Dataset (covariates and treatment variable)
        forest_dataset_tau_train = Dataset()
        forest_dataset_tau_train.add_covariates(X_train_tau)
        forest_dataset_tau_train.add_basis(tau_basis_train)
        if self.has_test:
            forest_dataset_tau_test = Dataset()
            forest_dataset_tau_test.add_covariates(X_test_tau)
            forest_dataset_tau_test.add_basis(tau_basis_test)

        # Residual
        residual_train = Residual(resid_train)

        # C++ random number generator
        if random_seed is None: 
            cpp_rng = RNG(-1)
        else:
            cpp_rng = RNG(random_seed)
        
        # Sampling data structures
        forest_sampler_mu = ForestSampler(forest_dataset_mu_train, feature_types_mu, num_trees_mu, self.n_train, alpha_mu, beta_mu, min_samples_leaf_mu)
        forest_sampler_tau = ForestSampler(forest_dataset_tau_train, feature_types_tau, num_trees_tau, self.n_train, alpha_tau, beta_tau, min_samples_leaf_tau)

        # Container of forest samples
        self.forest_container_mu = ForestContainer(num_trees_mu, 1, True)
        self.forest_container_tau = ForestContainer(num_trees_tau, Z_train.shape[1], False)
        
        # Variance samplers
        if self.sample_sigma_global:
            global_var_model = GlobalVarianceModel()
        if self.sample_sigma_leaf_mu:
            leaf_var_model_mu = LeafVarianceModel()
        if self.sample_sigma_leaf_tau:
            leaf_var_model_tau = LeafVarianceModel()

        # Initialize the leaves of each tree in the prognostic forest
        init_mu = np.squeeze(np.mean(resid_train)) / num_trees_mu
        self.forest_container_mu.set_root_leaves(0, init_mu)
        forest_sampler_mu.update_residual(forest_dataset_mu_train, residual_train, self.forest_container_mu, False, 0, True)

        # Initialize the leaves of each tree in the treatment forest
        self.forest_container_tau.set_root_leaves(0, 0.)
        forest_sampler_tau.update_residual(forest_dataset_tau_train, residual_train, self.forest_container_tau, True, 0, True)

        # Run GFR (warm start) if specified
        if self.num_gfr > 0:
            gfr_indices = np.arange(self.num_gfr)
            for i in range(self.num_gfr):
                # Sample the prognostic forest
                forest_sampler_mu.sample_one_iteration(
                    self.forest_container_mu, forest_dataset_mu_train, residual_train, cpp_rng, feature_types_mu, 
                    cutpoint_grid_size, current_leaf_scale_mu, variable_weights_mu, current_sigma2, 0, True, True
                )

                # Sample variance parameters (if requested)
                if self.sample_sigma_global:
                    current_sigma2 = global_var_model.sample_one_iteration(residual_train, cpp_rng, nu, lamb)
                    self.global_var_samples[i] = current_sigma2*self.y_std*self.y_std
                if self.sample_sigma_leaf_mu:
                    self.leaf_scale_mu_samples[i] = leaf_var_model_mu.sample_one_iteration(self.forest_container_mu, cpp_rng, a_leaf_mu, b_leaf_mu, i)
                    current_leaf_scale_mu[0,0] = self.leaf_scale_mu_samples[i]
                
                # Sample the treatment forest
                forest_sampler_tau.sample_one_iteration(
                    self.forest_container_tau, forest_dataset_tau_train, residual_train, cpp_rng, feature_types_tau, 
                    cutpoint_grid_size, current_leaf_scale_tau, variable_weights_tau, current_sigma2, 1, True, True
                )
                
                # Sample variance parameters (if requested)
                if self.sample_sigma_global:
                    current_sigma2 = global_var_model.sample_one_iteration(residual_train, cpp_rng, nu, lamb)
                    self.global_var_samples[i] = current_sigma2*self.y_std*self.y_std
                if self.sample_sigma_leaf_tau:
                    self.leaf_scale_tau_samples[i] = leaf_var_model_tau.sample_one_iteration(self.forest_container_tau, cpp_rng, a_leaf_tau, b_leaf_tau, i)
                    current_leaf_scale_tau[0,0] = self.leaf_scale_tau_samples[i]
                
                # Sample coding parameters (if requested)
                if self.adaptive_coding:
                    mu_x = self.forest_container_mu.predict_raw_single_forest(forest_dataset_mu_train, i)
                    tau_x = np.squeeze(self.forest_container_tau.predict_raw_single_forest(forest_dataset_tau_train, i))
                    s_tt0 = np.sum(tau_x*tau_x*(np.squeeze(Z_train)==0))
                    s_tt1 = np.sum(tau_x*tau_x*(np.squeeze(Z_train)==1))
                    partial_resid_mu = np.squeeze(resid_train - mu_x)
                    s_ty0 = np.sum(tau_x*partial_resid_mu*(np.squeeze(Z_train)==0))
                    s_ty1 = np.sum(tau_x*partial_resid_mu*(np.squeeze(Z_train)==1))
                    current_b_0 = self.rng.normal(loc = (s_ty0/(s_tt0 + 2*current_sigma2)), 
                                             scale = np.sqrt(current_sigma2/(s_tt0 + 2*current_sigma2)), size = 1)
                    current_b_1 = self.rng.normal(loc = (s_ty1/(s_tt1 + 2*current_sigma2)), 
                                             scale = np.sqrt(current_sigma2/(s_tt1 + 2*current_sigma2)), size = 1)
                    tau_basis_train = (1-np.squeeze(Z_train))*current_b_0 + np.squeeze(Z_train)*current_b_1
                    forest_dataset_tau_train.update_basis(tau_basis_train)
                    if self.has_test:
                        tau_basis_test = (1-np.squeeze(Z_test))*current_b_0 + np.squeeze(Z_test)*current_b_1
                        forest_dataset_tau_test.update_basis(tau_basis_test)
                    self.b0_samples[i] = current_b_0
                    self.b1_samples[i] = current_b_1
        
        # Run MCMC
        if self.num_burnin + self.num_mcmc > 0:
            if self.num_burnin > 0:
                burnin_indices = np.arange(self.num_gfr, self.num_gfr + self.num_burnin)
            if self.num_mcmc > 0:
                mcmc_indices = np.arange(self.num_gfr + self.num_burnin, self.num_gfr + self.num_burnin + self.num_mcmc)
            for i in range(self.num_gfr, self.num_samples):
                # Sample the prognostic forest
                forest_sampler_mu.sample_one_iteration(
                    self.forest_container_mu, forest_dataset_mu_train, residual_train, cpp_rng, feature_types_mu, 
                    cutpoint_grid_size, current_leaf_scale_mu, variable_weights_mu, current_sigma2, 0, False, True
                )

                # Sample variance parameters (if requested)
                if self.sample_sigma_global:
                    current_sigma2 = global_var_model.sample_one_iteration(residual_train, cpp_rng, nu, lamb)
                    self.global_var_samples[i] = current_sigma2*self.y_std*self.y_std
                if self.sample_sigma_leaf_mu:
                    self.leaf_scale_mu_samples[i] = leaf_var_model_mu.sample_one_iteration(self.forest_container_mu, cpp_rng, a_leaf_mu, b_leaf_mu, i)
                    current_leaf_scale_mu[0,0] = self.leaf_scale_mu_samples[i]
                
                # Sample the treatment forest
                forest_sampler_tau.sample_one_iteration(
                    self.forest_container_tau, forest_dataset_tau_train, residual_train, cpp_rng, feature_types_tau, 
                    cutpoint_grid_size, current_leaf_scale_tau, variable_weights_tau, current_sigma2, 1, False, True
                )
                
                # Sample variance parameters (if requested)
                if self.sample_sigma_global:
                    current_sigma2 = global_var_model.sample_one_iteration(residual_train, cpp_rng, nu, lamb)
                    self.global_var_samples[i] = current_sigma2*self.y_std*self.y_std
                if self.sample_sigma_leaf_tau:
                    self.leaf_scale_tau_samples[i] = leaf_var_model_tau.sample_one_iteration(self.forest_container_tau, cpp_rng, a_leaf_tau, b_leaf_tau, i)
                    current_leaf_scale_tau[0,0] = self.leaf_scale_tau_samples[i]                
                
                # Sample coding parameters (if requested)
                if self.adaptive_coding:
                    mu_x = self.forest_container_mu.predict_raw_single_forest(forest_dataset_mu_train, i)
                    tau_x = np.squeeze(self.forest_container_tau.predict_raw_single_forest(forest_dataset_tau_train, i))
                    s_tt0 = np.sum(tau_x*tau_x*(np.squeeze(Z_train)==0))
                    s_tt1 = np.sum(tau_x*tau_x*(np.squeeze(Z_train)==1))
                    partial_resid_mu = np.squeeze(resid_train - mu_x)
                    s_ty0 = np.sum(tau_x*partial_resid_mu*(np.squeeze(Z_train)==0))
                    s_ty1 = np.sum(tau_x*partial_resid_mu*(np.squeeze(Z_train)==1))
                    current_b_0 = self.rng.normal(loc = (s_ty0/(s_tt0 + 2*current_sigma2)), 
                                             scale = np.sqrt(current_sigma2/(s_tt0 + 2*current_sigma2)), size = 1)
                    current_b_1 = self.rng.normal(loc = (s_ty1/(s_tt1 + 2*current_sigma2)), 
                                             scale = np.sqrt(current_sigma2/(s_tt1 + 2*current_sigma2)), size = 1)
                    tau_basis_train = (1-np.squeeze(Z_train))*current_b_0 + np.squeeze(Z_train)*current_b_1
                    forest_dataset_tau_train.update_basis(tau_basis_train)
                    if self.has_test:
                        tau_basis_test = (1-np.squeeze(Z_test))*current_b_0 + np.squeeze(Z_test)*current_b_1
                        forest_dataset_tau_test.update_basis(tau_basis_test)
                    self.b0_samples[i] = current_b_0
                    self.b1_samples[i] = current_b_1
        
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
        mu_raw = self.forest_container_mu.forest_container_cpp.Predict(forest_dataset_mu_train.dataset_cpp)
        self.mu_hat_train = mu_raw[:,self.keep_indices]*self.y_std + self.y_bar
        tau_raw = self.forest_container_tau.forest_container_cpp.PredictRaw(forest_dataset_tau_train.dataset_cpp)
        self.tau_hat_train = tau_raw[:,self.keep_indices]*self.y_std
        if self.adaptive_coding:
            adaptive_coding_weights = np.expand_dims(self.b1_samples[self.keep_indices] - self.b0_samples[self.keep_indices], axis=(0,2))
            self.tau_hat_train = self.tau_hat_train*adaptive_coding_weights
        self.y_hat_train = self.mu_hat_train + Z_train*np.squeeze(self.tau_hat_train)
        if self.has_test:
            mu_raw_test = self.forest_container_mu.forest_container_cpp.Predict(forest_dataset_mu_test.dataset_cpp)
            self.mu_hat_test = mu_raw_test[:,self.keep_indices]*self.y_std + self.y_bar
            tau_raw_test = self.forest_container_tau.forest_container_cpp.PredictRaw(forest_dataset_tau_test.dataset_cpp)
            self.tau_hat_test = tau_raw_test[:,self.keep_indices]*self.y_std
            if self.adaptive_coding:
                adaptive_coding_weights_test = np.expand_dims(self.b1_samples[self.keep_indices] - self.b0_samples[self.keep_indices], axis=(0,2))
                self.tau_hat_test = self.tau_hat_test*adaptive_coding_weights_test
            self.y_hat_test = self.mu_hat_test + Z_test*np.squeeze(self.tau_hat_test)
    
    def predict_tau(self, X: np.array, Z: np.array, propensity: np.array = None) -> np.array:
        """Predict CATE function for every provided observation.

        Parameters
        ----------
        X : np.array or pd.DataFrame
            Test set covariates.
        Z : np.array
            Test set treatment indicators.
        propensity : :obj:`np.array`, optional
            Optional test set propensities. Must be provided if propensities were provided when the model was sampled.
        
        Returns
        -------
        np.array
            Array with as many rows as in ``X`` and as many columns as retained samples of the algorithm.
        """
        if not self.is_sampled():
            msg = (
                "This BCFModel instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this model."
            )
            raise NotSampledError(msg)
        
        # Convert everything to standard shape (2-dimensional)
        if X.ndim == 1:
            X = np.expand_dims(X, 1)
        if Z.ndim == 1:
            Z = np.expand_dims(Z, 1)
        if propensity is not None:
            if propensity.ndim == 1:
                propensity = np.expand_dims(propensity, 1)
        
        # Data checks
        if Z.shape[0] != X.shape[0]:
            raise ValueError("X and Z must have the same number of rows")
        if propensity is not None:
            if propensity.shape[0] != X.shape[0]:
                raise ValueError("X and propensity must have the same number of rows")
        else:
            if self.propensity_covariate == "tau":
                if not self.internal_propensity_model:
                    raise ValueError("Propensity scores not provided, but no propensity model was trained during sampling")
                else:
                    propensity = np.mean(self.bart_propensity_model.predict(X), axis=1, keepdims=True)
        
        # Update covariates to include propensities if requested
        if self.propensity_covariate == "tau":
            X_tau = np.c_[X, propensity]
        else:
            X_tau = X
        
        # Treatment Forest Dataset (covariates and treatment variable)
        forest_dataset_tau = Dataset()
        forest_dataset_tau.add_covariates(X_tau)
        forest_dataset_tau.add_basis(Z)
        
        # Estimate treatment effect
        tau_raw = self.forest_container_tau.forest_container_cpp.PredictRaw(forest_dataset_tau.dataset_cpp)
        tau_raw = tau_raw*self.y_std
        if self.adaptive_coding:
            tau_raw = tau_raw*np.expand_dims(self.b1_samples - self.b0_samples, axis=(0,2))
        tau_x = tau_raw[:,self.keep_indices]

        # Return result matrices as a tuple
        return tau_x
    
    def predict(self, X: np.array, Z: np.array, propensity: np.array = None) -> np.array:
        """Predict outcome model components (CATE function and prognostic function) as well as overall outcome for every provided observation. 
        Predicted outcomes are computed as ``yhat = mu_x + Z*tau_x`` where mu_x is a sample of the prognostic function and tau_x is a sample of the treatment effect (CATE) function.

        Parameters
        ----------
        X : np.array or pd.DataFrame
            Test set covariates.
        Z : np.array
            Test set treatment indicators.
        propensity : :obj:`np.array`, optional
            Optional test set propensities. Must be provided if propensities were provided when the model was sampled.
        
        Returns
        -------
        tuple of np.array
            Tuple of arrays with as many rows as in ``X`` and as many columns as retained samples of the algorithm. 
            The first entry of the tuple contains conditional average treatment effect (CATE) samples, 
            the second entry contains prognostic effect samples, and the third entry contains outcome prediction samples
        """
        if not self.is_sampled():
            msg = (
                "This BCFModel instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this model."
            )
            raise NotSampledError(msg)
        
        # Convert everything to standard shape (2-dimensional)
        if X.ndim == 1:
            X = np.expand_dims(X, 1)
        if Z.ndim == 1:
            Z = np.expand_dims(Z, 1)
        if propensity is not None:
            if propensity.ndim == 1:
                propensity = np.expand_dims(propensity, 1)
        
        # Data checks
        if Z.shape[0] != X.shape[0]:
            raise ValueError("X and Z must have the same number of rows")
        if propensity is not None:
            if propensity.shape[0] != X.shape[0]:
                raise ValueError("X and propensity must have the same number of rows")
        else:
            if self.propensity_covariate != "none":
                if not self.internal_propensity_model:
                    raise ValueError("Propensity scores not provided, but no propensity model was trained during sampling")
                else:
                    propensity = np.mean(self.bart_propensity_model.predict(X), axis=1, keepdims=True)
        
        # Update covariates to include propensities if requested
        if self.propensity_covariate == "mu":
            X_mu = np.c_[X, propensity]
            X_tau = X
        elif self.propensity_covariate == "tau":
            X_mu = X
            X_tau = np.c_[X, propensity]
        elif self.propensity_covariate == "both":
            X_mu = np.c_[X, propensity]
            X_tau = np.c_[X, propensity]
        elif self.propensity_covariate == "none":
            X_mu = X
            X_tau = X
        
        # Prognostic Forest Dataset (covariates)
        forest_dataset_mu = Dataset()
        forest_dataset_mu.add_covariates(X_mu)

        # Treatment Forest Dataset (covariates and treatment variable)
        forest_dataset_tau = Dataset()
        forest_dataset_tau.add_covariates(X_tau)
        forest_dataset_tau.add_basis(Z)
        
        # Estimate prognostic term
        mu_raw = self.forest_container_mu.forest_container_cpp.Predict(forest_dataset_mu.dataset_cpp)
        mu_x = mu_raw[:,self.keep_indices]*self.y_std + self.y_bar
        
        # Estimate treatment effect
        tau_raw = self.forest_container_tau.forest_container_cpp.PredictRaw(forest_dataset_tau.dataset_cpp)
        tau_raw = tau_raw*self.y_std
        if self.adaptive_coding:
            tau_raw = tau_raw*np.expand_dims(self.b1_samples - self.b0_samples, axis=(0,2))
        tau_x = tau_raw[:,self.keep_indices]

        # Outcome predictions
        yhat_x = mu_x + Z*tau_x

        # Return result matrices as a tuple
        return (tau_x, mu_x, yhat_x)
