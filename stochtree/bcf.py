"""
Bayesian Causal Forests (BCF) module
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.utils import check_scalar
from typing import Optional, Union, Dict, Any
from scipy.linalg import lstsq
from scipy.stats import gamma
from .bart import BARTModel
from .calibration import calibrate_global_error_variance
from .data import Dataset, Residual
from .forest import ForestContainer
from .preprocessing import CovariateTransformer, _preprocess_bcf_params
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
    
    def sample(self, X_train: Union[pd.DataFrame, np.array], Z_train: np.array, y_train: np.array, pi_train: np.array = None, 
               X_test: Union[pd.DataFrame, np.array] = None, Z_test: np.array = None, pi_test: np.array = None, 
               num_gfr: int = 5, num_burnin: int = 0, num_mcmc: int = 100, params: Optional[Dict[str, Any]] = None) -> None:
        """Runs a BCF sampler on provided training set. Outcome predictions and estimates of the prognostic and treatment effect functions 
        will be cached for the training set and (if provided) the test set.

        Parameters
        ----------
        X_train : :obj:`np.array` or :obj:`pd.DataFrame`
            Covariates used to split trees in the ensemble. Can be passed as either a matrix or dataframe.
        Z_train : :obj:`np.array`
            Array of (continuous or binary; univariate or multivariate) treatment assignments.
        y_train : :obj:`np.array`
            Outcome to be modeled by the ensemble.
        pi_train : :obj:`np.array`
            Optional vector of propensity scores. If not provided, this will be estimated from the data.
        X_test : :obj:`np.array`, optional
            Optional test set of covariates used to define "out of sample" evaluation data.
        Z_test : :obj:`np.array`, optional
            Optional test set of (continuous or binary) treatment assignments.
            Must be provided if ``X_test`` is provided.
        pi_test : :obj:`np.array`, optional
            Optional test set vector of propensity scores. If not provided (but ``X_test`` and ``Z_test`` are), this will be estimated from the data.
        num_gfr : :obj:`int`, optional
            Number of "warm-start" iterations run using the grow-from-root algorithm (He and Hahn, 2021). Defaults to ``5``.
        num_burnin : :obj:`int`, optional
            Number of "burn-in" iterations of the MCMC sampler. Defaults to ``0``. Ignored if ``num_gfr > 0``.
        num_mcmc : :obj:`int`, optional
            Number of "retained" iterations of the MCMC sampler. Defaults to ``100``. If this is set to 0, GFR (XBART) samples will be retained.
        params : :obj:`dict`, optional
            Dictionary of model parameters, each of which has a default value.

            * ``cutpoint_grid_size`` (``int``): Maximum number of cutpoints to consider for each feature. Defaults to ``100``.
            * ``sigma_leaf_mu`` (``float``): Starting value of leaf node scale parameter for the prognostic forest. Calibrated internally as ``2/num_trees_mu`` if not set here.
            * ``sigma_leaf_tau`` (``float`` or ``np.array``): Starting value of leaf node scale parameter for the treatment effect forest. 
                When treatment (``Z_train``) is multivariate, this can be either a ``float`` or a square 2-dimensional ``np.array`` 
                with ``sigma_leaf_tau.shape[0] == Z_train.shape[1]`` and ``sigma_leaf_tau.shape[1] == Z_train.shape[1]``.
                If ``sigma_leaf_tau`` is provided as a float for multivariate treatment, the leaf scale term will be set as a 
                diagonal matrix with ``sigma_leaf_tau`` on every diagonal. If not passed as an argument, this parameter is 
                calibrated internally as ``1/num_trees_tau`` (and propagated to a diagonal matrix if necessary).
            * ``alpha_mu`` (``float``): Prior probability of splitting for a tree of depth 0 for the prognostic forest. 
                Tree split prior combines ``alpha`` and ``beta`` via ``alpha*(1+node_depth)^-beta``.
            * ``alpha_tau`` (``float``): Prior probability of splitting for a tree of depth 0 for the treatment effect forest. 
                Tree split prior combines ``alpha`` and ``beta`` via ``alpha*(1+node_depth)^-beta``.
            * ``beta_mu`` (``float``): Exponent that decreases split probabilities for nodes of depth > 0 for the prognostic forest. 
                Tree split prior combines ``alpha`` and ``beta`` via ``alpha*(1+node_depth)^-beta``.
            * ``beta_tau`` (``float``): Exponent that decreases split probabilities for nodes of depth > 0 for the treatment effect forest. 
                Tree split prior combines ``alpha`` and ``beta`` via ``alpha*(1+node_depth)^-beta``.
            * ``min_samples_leaf_mu`` (``int``): Minimum allowable size of a leaf, in terms of training samples, for the prognostic forest. Defaults to ``5``.
            * ``min_samples_leaf_tau`` (``int``): Minimum allowable size of a leaf, in terms of training samples, for the treatment effect forest. Defaults to ``5``.
            * ``max_depth_mu`` (``int``): Maximum depth of any tree in the mu ensemble. Defaults to ``10``. Can be overriden with ``-1`` which does not enforce any depth limits on trees.
            * ``max_depth_tau`` (``int``): Maximum depth of any tree in the tau ensemble. Defaults to ``5``. Can be overriden with ``-1`` which does not enforce any depth limits on trees.
            * ``a_global`` (``float``): Shape parameter in the ``IG(a_global, b_global)`` global error variance model. Defaults to ``0``.
            * ``b_global`` (``float``): Component of the scale parameter in the ``IG(a_global, b_global)`` global error variance prior. Defaults to ``0``.
            * ``a_leaf_mu`` (``float``): Shape parameter in the ``IG(a_leaf, b_leaf)`` leaf node parameter variance model for the prognostic forest. Defaults to ``3``.
            * ``a_leaf_tau`` (``float``): Shape parameter in the ``IG(a_leaf, b_leaf)`` leaf node parameter variance model for the treatment effect forest. Defaults to ``3``.
            * ``b_leaf_mu`` (``float``): Scale parameter in the ``IG(a_leaf, b_leaf)`` leaf node parameter variance model for the prognostic forest. Calibrated internally as ``0.5/num_trees`` if not set here.
            * ``b_leaf_tau`` (``float``): Scale parameter in the ``IG(a_leaf, b_leaf)`` leaf node parameter variance model for the treatment effect forest. Calibrated internally as ``0.5/num_trees`` if not set here.
            * ``sigma2`` (``float``): Starting value of global variance parameter. Calibrated internally as in Sparapani et al (2021) if not set here.
            * ``pct_var_sigma2_init`` (``float``): Percentage of standardized outcome variance used to initialize global error variance parameter. Superseded by ``sigma2``. Defaults to ``0.25``.
            * ``variable_weights`` (`np.`array``): Numeric weights reflecting the relative probability of splitting on each variable. Does not need to sum to 1 but cannot be negative. Defaults to ``np.repeat(1/X_train.shape[1], X_train.shape[1])`` if not set here. Note that if the propensity score is included as a covariate in either forest, its weight will default to ``1/X_train.shape[1]``. A workaround if you wish to provide a custom weight for the propensity score is to include it as a column in ``X_train`` and then set ``propensity_covariate`` to ``'none'`` and adjust ``keep_vars_mu`` and ``keep_vars_tau`` accordingly.
            * ``keep_vars_mu`` (``list`` or ``np.array``): Vector of variable names or column indices denoting variables that should be included in the prognostic (``mu(X)``) forest. Defaults to ``None``.
            * ``drop_vars_mu`` (``list`` or ``np.array``): Vector of variable names or column indices denoting variables that should be excluded from the prognostic (``mu(X)``) forest. Defaults to ``None``. If both ``drop_vars_mu`` and ``keep_vars_mu`` are set, ``drop_vars_mu`` will be ignored.
            * ``keep_vars_tau`` (``list`` or ``np.array``): Vector of variable names or column indices denoting variables that should be included in the treatment effect (``tau(X)``) forest. Defaults to ``None``.
            * ``drop_vars_tau`` (``list`` or ``np.array``): Vector of variable names or column indices denoting variables that should be excluded from the treatment effect (``tau(X)``) forest. Defaults to ``None``. If both ``drop_vars_tau`` and ``keep_vars_tau`` are set, ``drop_vars_tau`` will be ignored.
            * ``num_trees_mu`` (``int``): Number of trees in the prognostic forest. Defaults to ``200``.
            * ``num_trees_tau`` (``int``): Number of trees in the treatment effect forest. Defaults to ``50``.
            * ``sample_sigma_global`` (``bool``): Whether or not to update the ``sigma^2`` global error variance parameter based on ``IG(a_global, b_global)``. Defaults to ``True``.
            * ``sample_sigma_leaf_mu`` (``bool``): Whether or not to update the ``tau`` leaf scale variance parameter based on ``IG(a_leaf, b_leaf)`` for the prognostic forest. 
                Cannot (currently) be set to true if ``basis_train`` has more than one column. Defaults to ``True``.
            * ``sample_sigma_leaf_tau`` (``bool``): Whether or not to update the ``tau`` leaf scale variance parameter based on ``IG(a_leaf, b_leaf)`` for the treatment effect forest. 
                Cannot (currently) be set to true if ``basis_train`` has more than one column. Defaults to ``True``.
            * ``propensity_covariate`` (``str``): Whether to include the propensity score as a covariate in either or both of the forests. Enter ``"none"`` for neither, ``"mu"`` for the prognostic forest, ``"tau"`` for the treatment forest, and ``"both"`` for both forests. 
                If this is not ``"none"`` and a propensity score is not provided, it will be estimated from (``X_train``, ``Z_train``) using ``BARTModel``. Defaults to ``"mu"``.
            * ``adaptive_coding`` (``bool``): Whether or not to use an "adaptive coding" scheme in which a binary treatment variable is not coded manually as (0,1) or (-1,1) but learned via 
                parameters ``b_0`` and ``b_1`` that attach to the outcome model ``[b_0 (1-Z) + b_1 Z] tau(X)``. This is ignored when Z is not binary. Defaults to True.
            * ``b_0`` (``float``): Initial value of the "control" group coding parameter. This is ignored when ``Z`` is not binary. Default: ``-0.5``.
            * ``b_1`` (``float``): Initial value of the "treated" group coding parameter. This is ignored when ``Z`` is not binary. Default: ``0.5``.
            * ``random_seed`` (``int``): Integer parameterizing the C++ random number generator. If not specified, the C++ random number generator is seeded according to ``std::random_device``.
            * ``keep_burnin`` (``bool``): Whether or not "burnin" samples should be included in predictions. Defaults to ``False``. Ignored if ``num_mcmc == 0``.
            * ``keep_gfr`` (``bool``): Whether or not "warm-start" / grow-from-root samples should be included in predictions. Defaults to ``False``. Ignored if ``num_mcmc == 0``.
        
        Returns
        -------
        self : BCFModel
            Sampled BCF Model.
        """
        # Unpack parameters
        bcf_params = _preprocess_bcf_params(params)
        cutpoint_grid_size = bcf_params['cutpoint_grid_size']
        sigma_leaf_mu = bcf_params['sigma_leaf_mu']
        sigma_leaf_tau = bcf_params['sigma_leaf_tau']
        alpha_mu = bcf_params['alpha_mu']
        alpha_tau = bcf_params['alpha_tau']
        beta_mu = bcf_params['beta_mu']
        beta_tau = bcf_params['beta_tau']
        min_samples_leaf_mu = bcf_params['min_samples_leaf_mu']
        min_samples_leaf_tau = bcf_params['min_samples_leaf_tau']
        max_depth_mu = bcf_params['max_depth_mu']
        max_depth_tau = bcf_params['max_depth_tau']
        a_global = bcf_params['a_global']
        b_global = bcf_params['b_global']
        a_leaf_mu = bcf_params['a_leaf_mu']
        a_leaf_tau = bcf_params['a_leaf_tau']
        b_leaf_mu = bcf_params['b_leaf_mu']
        b_leaf_tau = bcf_params['b_leaf_tau']
        sigma2 = bcf_params['sigma2']
        pct_var_sigma2_init = bcf_params['pct_var_sigma2_init']
        variable_weights = bcf_params['variable_weights']
        keep_vars_mu = bcf_params['keep_vars_mu']
        drop_vars_mu = bcf_params['drop_vars_mu']
        keep_vars_tau = bcf_params['keep_vars_tau']
        drop_vars_tau = bcf_params['drop_vars_tau']
        num_trees_mu = bcf_params['num_trees_mu']
        num_trees_tau = bcf_params['num_trees_tau']
        sample_sigma_global = bcf_params['sample_sigma_global']
        sample_sigma_leaf_mu = bcf_params['sample_sigma_leaf_mu']
        sample_sigma_leaf_tau = bcf_params['sample_sigma_leaf_tau']
        propensity_covariate = bcf_params['propensity_covariate']
        adaptive_coding = bcf_params['adaptive_coding']
        b_0 = bcf_params['b_0']
        b_1 = bcf_params['b_1']
        random_seed = bcf_params['random_seed']
        keep_burnin = bcf_params['keep_burnin']
        keep_gfr = bcf_params['keep_gfr']
        
        # Variable weight preprocessing (and initialization if necessary)
        if variable_weights is None:
            if X_train.ndim > 1:
                variable_weights = np.repeat(1/X_train.shape[1], X_train.shape[1])
            else:
                variable_weights = np.repeat(1., 1)
        if np.any(variable_weights < 0):
            raise ValueError("variable_weights cannot have any negative weights")
        
        # Check data inputs
        if not isinstance(X_train, pd.DataFrame) and not isinstance(X_train, np.ndarray):
            raise ValueError("X_train must be a pandas dataframe or numpy array")
        if not isinstance(Z_train, np.ndarray):
            raise ValueError("Z_train must be a numpy array")
        if pi_train is not None:
            if not isinstance(pi_train, np.ndarray):
                raise ValueError("pi_train must be a numpy array")
        if not isinstance(y_train, np.ndarray):
            raise ValueError("y_train must be a numpy array")
        if X_test is not None:
            if not isinstance(X_test, pd.DataFrame) and not isinstance(X_test, np.ndarray):
                raise ValueError("X_test must be a pandas dataframe or numpy array")
        if Z_test is not None:
            if not isinstance(Z_test, np.ndarray):
                raise ValueError("Z_test must be a numpy array")
        if pi_test is not None:
            if not isinstance(pi_test, np.ndarray):
                raise ValueError("pi_test must be a numpy array")

        
        # Convert everything to standard shape (2-dimensional)
        if isinstance(X_train, np.ndarray):
            if X_train.ndim == 1:
                X_train = np.expand_dims(X_train, 1)
        if Z_train is not None:
            if Z_train.ndim == 1:
                Z_train = np.expand_dims(Z_train, 1)
        if pi_train is not None:
            if pi_train.ndim == 1:
                pi_train = np.expand_dims(pi_train, 1)
        if y_train.ndim == 1:
            y_train = np.expand_dims(y_train, 1)
        if X_test is not None:
            if isinstance(X_test, np.ndarray):
                if X_test.ndim == 1:
                    X_test = np.expand_dims(X_test, 1)
        if Z_test is not None:
            if Z_test.ndim == 1:
                Z_test = np.expand_dims(Z_test, 1)
        if pi_test is not None:
            if pi_test.ndim == 1:
                pi_test = np.expand_dims(pi_test, 1)

        # Original number of covariates
        num_cov_orig = X_train.shape[1]
        
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
        
        # Treatment details
        self.treatment_dim = Z_train.shape[1]
        self.multivariate_treatment = True if self.treatment_dim > 1 else False
        treatment_leaf_model = 2 if self.multivariate_treatment else 1
        
        # Check parameters
        if sigma_leaf_tau is not None:
            if not isinstance(sigma_leaf_tau, float) and not isinstance(sigma_leaf_tau, np.ndarray):
                raise ValueError("sigma_leaf_tau must be a float or numpy array")
            if self.multivariate_treatment:
                if sigma_leaf_tau is not None:
                    if isinstance(sigma_leaf_tau, np.ndarray):
                        if sigma_leaf_tau.ndim != 2:
                            raise ValueError("sigma_leaf_tau must be 2-dimensional if passed as a np.array")
                        if self.treatment_dim != sigma_leaf_tau.shape[0] or self.treatment_dim != sigma_leaf_tau.shape[1]:
                            raise ValueError("sigma_leaf_tau must have the same number of rows and columns, which must match Z_train.shape[1]")
        if sigma_leaf_mu is not None:
            sigma_leaf_mu = check_scalar(x=sigma_leaf_mu, name="sigma_leaf_mu", target_type=float, 
                                        min_val=0., max_val=None, include_boundaries="neither")
        if cutpoint_grid_size is not None:
            cutpoint_grid_size = check_scalar(x=cutpoint_grid_size, name="cutpoint_grid_size", target_type=int, 
                                            min_val=1, max_val=None, include_boundaries="left")
        if min_samples_leaf_mu is not None:
            min_samples_leaf_mu = check_scalar(x=min_samples_leaf_mu, name="min_samples_leaf_mu", target_type=int, 
                                               min_val=1, max_val=None, include_boundaries="left")
        if min_samples_leaf_tau is not None:
            min_samples_leaf_tau = check_scalar(x=min_samples_leaf_tau, name="min_samples_leaf_tau", target_type=int, 
                                                min_val=1, max_val=None, include_boundaries="left")
        if num_trees_mu is not None:
            num_trees_mu = check_scalar(x=num_trees_mu, name="num_trees_mu", target_type=int, 
                                        min_val=1, max_val=None, include_boundaries="left")
        if num_trees_tau is not None:
            num_trees_tau = check_scalar(x=num_trees_tau, name="num_trees_tau", target_type=int, 
                                         min_val=1, max_val=None, include_boundaries="left")
        num_gfr = check_scalar(x=num_gfr, name="num_gfr", target_type=int, 
                                min_val=0, max_val=None, include_boundaries="left")
        num_burnin = check_scalar(x=num_burnin, name="num_burnin", target_type=int, 
                                min_val=0, max_val=None, include_boundaries="left")
        num_mcmc = check_scalar(x=num_mcmc, name="num_mcmc", target_type=int, 
                                min_val=0, max_val=None, include_boundaries="left")
        num_samples = num_gfr + num_burnin + num_mcmc
        num_samples = check_scalar(x=num_samples, name="num_samples", target_type=int, 
                                   min_val=1, max_val=None, include_boundaries="left")
        if random_seed is not None:
            random_seed = check_scalar(x=random_seed, name="random_seed", target_type=int, 
                                       min_val=-1, max_val=None, include_boundaries="left")
        if alpha_mu is not None:
            alpha_mu = check_scalar(x=alpha_mu, name="alpha_mu", target_type=(float,int), 
                                    min_val=0, max_val=1, include_boundaries="neither")
        if alpha_tau is not None:
            alpha_tau = check_scalar(x=alpha_tau, name="alpha_tau", target_type=(float,int), 
                                    min_val=0, max_val=1, include_boundaries="neither")
        if beta_mu is not None:
            beta_mu = check_scalar(x=beta_mu, name="beta_mu", target_type=(float,int), 
                                min_val=1, max_val=None, include_boundaries="left")
        if beta_tau is not None:
            beta_tau = check_scalar(x=beta_tau, name="beta_tau", target_type=(float,int), 
                                    min_val=1, max_val=None, include_boundaries="left")
        if a_global is not None:
            a_global = check_scalar(x=a_global, name="a_global", target_type=(float,int), 
                            min_val=0, max_val=None, include_boundaries="left")
        if b_global is not None:
            b_global = check_scalar(x=b_global, name="b_global", target_type=(float,int), 
                                min_val=0, max_val=None, include_boundaries="left")
        if a_leaf_mu is not None:
            a_leaf_mu = check_scalar(x=a_leaf_mu, name="a_leaf_mu", target_type=(float,int), 
                                    min_val=0, max_val=None, include_boundaries="left")
        if a_leaf_tau is not None:
            a_leaf_tau = check_scalar(x=a_leaf_tau, name="a_leaf_tau", target_type=(float,int), 
                                    min_val=0, max_val=None, include_boundaries="left")
        if b_leaf_mu is not None:
            b_leaf_mu = check_scalar(x=b_leaf_mu, name="b_leaf_mu", target_type=(float,int), 
                                    min_val=0, max_val=None, include_boundaries="left")
        if b_leaf_tau is not None:
            b_leaf_tau = check_scalar(x=b_leaf_tau, name="b_leaf_tau", target_type=(float,int), 
                                    min_val=0, max_val=None, include_boundaries="left")
        if sigma2 is not None:
            sigma2 = check_scalar(x=sigma2, name="sigma2", target_type=(float,int), 
                                min_val=0, max_val=None, include_boundaries="neither")
        if sample_sigma_leaf_mu is not None:
            if not isinstance(sample_sigma_leaf_mu, bool):
                raise ValueError("sample_sigma_leaf_mu must be a bool")
        if sample_sigma_leaf_tau is not None:
            if not isinstance(sample_sigma_leaf_tau, bool):
                raise ValueError("sample_sigma_leaf_tau must be a bool")
        if propensity_covariate is not None:
            if propensity_covariate not in ["mu", "tau", "both", "none"]:
                raise ValueError("propensity_covariate must be one of 'mu', 'tau', 'both', or 'none'")
        if b_0 is not None:
            b_0 = check_scalar(x=b_0, name="b_0", target_type=(float,int), 
                            min_val=None, max_val=None, include_boundaries="neither")
        if b_1 is not None:
            b_1 = check_scalar(x=b_1, name="b_1", target_type=(float,int), 
                            min_val=None, max_val=None, include_boundaries="neither")
        if keep_burnin is not None:
            if not isinstance(keep_burnin, bool):
                raise ValueError("keep_burnin must be a bool")
        if keep_gfr is not None:
            if not isinstance(keep_gfr, bool):
                raise ValueError("keep_gfr must be a bool")
        
        # Standardize the keep variable lists to numeric indices
        if keep_vars_mu is not None:
            if isinstance(keep_vars_mu, list):
                if all(isinstance(i, str) for i in keep_vars_mu):
                    if not np.all(np.isin(keep_vars_mu, X_train.columns)):
                        raise ValueError("keep_vars_mu includes some variable names that are not in X_train")
                    variable_subset_mu = [i for i in X_train.shape[1] if keep_vars_mu.count(X_train.columns.array[i]) > 0]
                elif all(isinstance(i, int) for i in keep_vars_mu):
                    if any(i >= X_train.shape[1] for i in keep_vars_mu):
                        raise ValueError("keep_vars_mu includes some variable indices that exceed the number of columns in X_train")
                    if any(i < 0 for i in keep_vars_mu):
                        raise ValueError("keep_vars_mu includes some negative variable indices")
                    variable_subset_mu = keep_vars_mu
                else:
                    raise ValueError("keep_vars_mu must be a list of variable names (str) or column indices (int)")
            elif isinstance(keep_vars_mu, np.ndarray):
                if keep_vars_mu.dtype == np.str_:
                    if not np.all(np.isin(keep_vars_mu, X_train.columns)):
                        raise ValueError("keep_vars_mu includes some variable names that are not in X_train")
                    variable_subset_mu = [i for i in X_train.shape[1] if keep_vars_mu.count(X_train.columns.array[i]) > 0]
                else:
                    if np.any(keep_vars_mu >= X_train.shape[1]):
                        raise ValueError("keep_vars_mu includes some variable indices that exceed the number of columns in X_train")
                    if np.any(keep_vars_mu < 0):
                        raise ValueError("keep_vars_mu includes some negative variable indices")
                    variable_subset_mu = [i for i in keep_vars_mu]
            else:
                raise ValueError("keep_vars_mu must be a list or np.array")
        elif keep_vars_mu is None and drop_vars_mu is not None:
            if isinstance(drop_vars_mu, list):
                if all(isinstance(i, str) for i in drop_vars_mu):
                    if not np.all(np.isin(drop_vars_mu, X_train.columns)):
                        raise ValueError("drop_vars_mu includes some variable names that are not in X_train")
                    variable_subset_mu = [i for i in range(X_train.shape[1]) if drop_vars_mu.count(X_train.columns.array[i]) == 0]
                elif all(isinstance(i, int) for i in drop_vars_mu):
                    if any(i >= X_train.shape[1] for i in drop_vars_mu):
                        raise ValueError("drop_vars_mu includes some variable indices that exceed the number of columns in X_train")
                    if any(i < 0 for i in drop_vars_mu):
                        raise ValueError("drop_vars_mu includes some negative variable indices")
                    variable_subset_mu = [i for i in range(X_train.shape[1]) if drop_vars_mu.count(i) == 0]
                else:
                    raise ValueError("drop_vars_mu must be a list of variable names (str) or column indices (int)")
            elif isinstance(drop_vars_mu, np.ndarray):
                if drop_vars_mu.dtype == np.str_:
                    if not np.all(np.isin(drop_vars_mu, X_train.columns)):
                        raise ValueError("drop_vars_mu includes some variable names that are not in X_train")
                    keep_inds = ~np.isin(X_train.columns.array, drop_vars_mu)
                    variable_subset_mu = [i for i in keep_inds]
                else:
                    if np.any(drop_vars_mu >= X_train.shape[1]):
                        raise ValueError("drop_vars_mu includes some variable indices that exceed the number of columns in X_train")
                    if np.any(drop_vars_mu < 0):
                        raise ValueError("drop_vars_mu includes some negative variable indices")
                    keep_inds = ~np.isin(np.arange(X_train.shape[1]), drop_vars_mu)
                    variable_subset_mu = [i for i in keep_inds]
            else:
                raise ValueError("drop_vars_mu must be a list or np.array")
        else:
            variable_subset_mu = [i for i in range(X_train.shape[1])]
        if keep_vars_tau is not None:
            if isinstance(keep_vars_tau, list):
                if all(isinstance(i, str) for i in keep_vars_tau):
                    if not np.all(np.isin(keep_vars_tau, X_train.columns)):
                        raise ValueError("keep_vars_tau includes some variable names that are not in X_train")
                    variable_subset_tau = [i for i in range(X_train.shape[1]) if keep_vars_tau.count(X_train.columns.array[i]) > 0]
                elif all(isinstance(i, int) for i in keep_vars_tau):
                    if any(i >= X_train.shape[1] for i in keep_vars_tau):
                        raise ValueError("keep_vars_tau includes some variable indices that exceed the number of columns in X_train")
                    if any(i < 0 for i in keep_vars_tau):
                        raise ValueError("keep_vars_tau includes some negative variable indices")
                    variable_subset_tau = keep_vars_tau
                else:
                    raise ValueError("keep_vars_tau must be a list of variable names (str) or column indices (int)")
            elif isinstance(keep_vars_tau, np.ndarray):
                if keep_vars_tau.dtype == np.str_:
                    if not np.all(np.isin(keep_vars_tau, X_train.columns)):
                        raise ValueError("keep_vars_tau includes some variable names that are not in X_train")
                    variable_subset_tau = [i for i in range(X_train.shape[1]) if keep_vars_tau.count(X_train.columns.array[i]) > 0]
                else:
                    if np.any(keep_vars_tau >= X_train.shape[1]):
                        raise ValueError("keep_vars_tau includes some variable indices that exceed the number of columns in X_train")
                    if np.any(keep_vars_tau < 0):
                        raise ValueError("keep_vars_tau includes some negative variable indices")
                    variable_subset_tau = [i for i in keep_vars_tau]
            else:
                raise ValueError("keep_vars_tau must be a list or np.array")
        elif keep_vars_tau is None and drop_vars_tau is not None:
            if isinstance(drop_vars_tau, list):
                if all(isinstance(i, str) for i in drop_vars_tau):
                    if not np.all(np.isin(drop_vars_tau, X_train.columns)):
                        raise ValueError("drop_vars_tau includes some variable names that are not in X_train")
                    variable_subset_tau = [i for i in range(X_train.shape[1]) if drop_vars_tau.count(X_train.columns.array[i]) == 0]
                elif all(isinstance(i, int) for i in drop_vars_tau):
                    if any(i >= X_train.shape[1] for i in drop_vars_tau):
                        raise ValueError("drop_vars_tau includes some variable indices that exceed the number of columns in X_train")
                    if any(i < 0 for i in drop_vars_tau):
                        raise ValueError("drop_vars_tau includes some negative variable indices")
                    variable_subset_tau = [i for i in range(X_train.shape[1]) if drop_vars_tau.count(i) == 0]
                else:
                    raise ValueError("drop_vars_tau must be a list of variable names (str) or column indices (int)")
            elif isinstance(drop_vars_tau, np.ndarray):
                if drop_vars_tau.dtype == np.str_:
                    if not np.all(np.isin(drop_vars_tau, X_train.columns)):
                        raise ValueError("drop_vars_tau includes some variable names that are not in X_train")
                    keep_inds = ~np.isin(X_train.columns.array, drop_vars_tau)
                    variable_subset_tau = [i for i in keep_inds]
                else:
                    if np.any(drop_vars_tau >= X_train.shape[1]):
                        raise ValueError("drop_vars_tau includes some variable indices that exceed the number of columns in X_train")
                    if np.any(drop_vars_tau < 0):
                        raise ValueError("drop_vars_tau includes some negative variable indices")
                    keep_inds = ~np.isin(np.arange(X_train.shape[1]), drop_vars_tau)
                    variable_subset_tau = [i for i in keep_inds]
            else:
                raise ValueError("drop_vars_tau must be a list or np.array")
        else:
            variable_subset_tau = [i for i in range(X_train.shape[1])]
        
        # Covariate preprocessing
        self._covariate_transformer = CovariateTransformer()
        self._covariate_transformer.fit(X_train)
        X_train_processed = self._covariate_transformer.transform(X_train)
        if X_test is not None:
            X_test_processed = self._covariate_transformer.transform(X_test)
        feature_types = np.asarray(self._covariate_transformer._processed_feature_types)
        original_var_indices = self._covariate_transformer.fetch_original_feature_indices()

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
        if adaptive_coding and self.multivariate_treatment:
            self.adaptive_coding = False

        # Sampling sigma_leaf_tau will be ignored for multivariate treatments
        if sample_sigma_leaf_tau and self.multivariate_treatment:
            sample_sigma_leaf_tau = False

        # Check if user has provided propensities that are needed in the model
        if pi_train is None and propensity_covariate != "none":
            if self.multivariate_treatment:
                raise ValueError("Propensities must be provided (via pi_train and / or pi_test parameters) or omitted by setting propensity_covariate = 'none' for multivariate treatments")
            else:
                self.bart_propensity_model = BARTModel()
                if self.has_test:
                    self.bart_propensity_model.sample(X_train=X_train_processed, y_train=Z_train, X_test=X_test_processed, num_gfr=10, num_mcmc=10)
                    pi_train = np.mean(self.bart_propensity_model.y_hat_train, axis = 1, keepdims = True)
                    pi_test = np.mean(self.bart_propensity_model.y_hat_test, axis = 1, keepdims = True)
                else:
                    self.bart_propensity_model.sample(X_train=X_train_processed, y_train=Z_train, num_gfr=10, num_mcmc=10)
                    pi_train = np.mean(self.bart_propensity_model.y_hat_train, axis = 1, keepdims = True)
                self.internal_propensity_model = True
        else:
            self.internal_propensity_model = False

        # Scale outcome
        self.y_bar = np.squeeze(np.mean(y_train))
        self.y_std = np.squeeze(np.std(y_train))
        resid_train = (y_train-self.y_bar)/self.y_std

        # Calibrate priors for global sigma^2 and sigma_leaf_mu / sigma_leaf_tau (don't use regression initializer for warm-start or XBART)
        if not sigma2:
            sigma2 = pct_var_sigma2_init*np.var(resid_train)
        b_leaf_mu = np.squeeze(np.var(resid_train)) / num_trees_mu if b_leaf_mu is None else b_leaf_mu
        b_leaf_tau = np.squeeze(np.var(resid_train)) / (2*num_trees_tau) if b_leaf_tau is None else b_leaf_tau
        sigma_leaf_mu = np.squeeze(np.var(resid_train)) / num_trees_mu if sigma_leaf_mu is None else sigma_leaf_mu
        sigma_leaf_tau = np.squeeze(np.var(resid_train)) / (2*num_trees_tau) if sigma_leaf_tau is None else sigma_leaf_tau
        if self.multivariate_treatment:
            if not isinstance(sigma_leaf_tau, np.ndarray):
                sigma_leaf_tau = np.diagflat(np.repeat(sigma_leaf_tau, self.treatment_dim))
        current_sigma2 = sigma2
        current_leaf_scale_mu = np.array([[sigma_leaf_mu]])
        if not isinstance(sigma_leaf_tau, np.ndarray):
            current_leaf_scale_tau = np.array([[sigma_leaf_tau]])
        else:
            current_leaf_scale_tau = sigma_leaf_tau

        # Update variable weights
        variable_counts = [original_var_indices.count(i) for i in original_var_indices]
        variable_weights_adj = [1/i for i in variable_counts]
        variable_weights = variable_weights[original_var_indices]*variable_weights_adj

        # Create mu and tau specific variable weights with weights zeroed out for excluded variables
        variable_weights_tau = variable_weights
        variable_weights_mu = variable_weights
        variable_weights_mu[[variable_subset_mu.count(i) == 0 for i in original_var_indices]] = 0
        variable_weights_tau[[variable_subset_tau.count(i) == 0 for i in original_var_indices]] <- 0
        
        # Update covariates to include propensities if requested
        if propensity_covariate not in ["none", "mu", "tau", "both"]:
            raise ValueError("propensity_covariate must equal one of 'none', 'mu', 'tau', or 'both'")
        if propensity_covariate != "none":
            feature_types = np.append(feature_types, 0).astype('int')
            X_train_processed = np.c_[X_train_processed, pi_train]
            if self.has_test:
                X_test_processed = np.c_[X_test_processed, pi_test]
            if propensity_covariate == "mu":
                variable_weights_mu = np.append(variable_weights_mu, np.repeat(1/num_cov_orig, pi_train.shape[1]))
                variable_weights_tau = np.append(variable_weights_tau, np.repeat(0., pi_train.shape[1]))
            elif propensity_covariate == "tau":
                variable_weights_mu = np.append(variable_weights_mu, np.repeat(0., pi_train.shape[1]))
                variable_weights_tau = np.append(variable_weights_tau, np.repeat(1/num_cov_orig, pi_train.shape[1]))
            elif propensity_covariate == "both":
                variable_weights_mu = np.append(variable_weights_mu, np.repeat(1/num_cov_orig, pi_train.shape[1]))
                variable_weights_tau = np.append(variable_weights_tau, np.repeat(1/num_cov_orig, pi_train.shape[1]))
        
        # Renormalize variable weights
        variable_weights_mu = variable_weights_mu / np.sum(variable_weights_mu)
        variable_weights_tau = variable_weights_tau / np.sum(variable_weights_tau)
        
        # Store propensity score requirements of the BCF forests
        self.propensity_covariate = propensity_covariate

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
        forest_dataset_train = Dataset()
        forest_dataset_train.add_covariates(X_train_processed)
        forest_dataset_train.add_basis(tau_basis_train)
        if self.has_test:
            forest_dataset_test = Dataset()
            forest_dataset_test.add_covariates(X_test_processed)
            forest_dataset_test.add_basis(tau_basis_test)

        # Residual
        residual_train = Residual(resid_train)

        # C++ random number generator
        if random_seed is None: 
            cpp_rng = RNG(-1)
        else:
            cpp_rng = RNG(random_seed)
        
        # TODO Placeholder: expose the heteroskedasticity interface through the function signature, as in R
        a_forest = 1
        b_forest = 1
        
        # Sampling data structures
        forest_sampler_mu = ForestSampler(forest_dataset_train, feature_types, num_trees_mu, self.n_train, alpha_mu, beta_mu, min_samples_leaf_mu, max_depth_mu)
        forest_sampler_tau = ForestSampler(forest_dataset_train, feature_types, num_trees_tau, self.n_train, alpha_tau, beta_tau, min_samples_leaf_tau, max_depth_tau)

        # Container of forest samples
        self.forest_container_mu = ForestContainer(num_trees_mu, 1, True, False)
        self.forest_container_tau = ForestContainer(num_trees_tau, Z_train.shape[1], False, False)
        
        # Variance samplers
        if self.sample_sigma_global:
            global_var_model = GlobalVarianceModel()
        if self.sample_sigma_leaf_mu:
            leaf_var_model_mu = LeafVarianceModel()
        if self.sample_sigma_leaf_tau:
            leaf_var_model_tau = LeafVarianceModel()

        # Initialize the leaves of each tree in the prognostic forest
        init_mu = np.array([np.squeeze(np.mean(resid_train))])
        forest_sampler_mu.prepare_for_sampler(forest_dataset_train, residual_train, self.forest_container_mu, 0, init_mu)

        # Initialize the leaves of each tree in the treatment forest
        if self.multivariate_treatment:
            init_tau = np.zeros(Z_train.shape[1])
        else:
            init_tau = np.array([0.])
        forest_sampler_tau.prepare_for_sampler(forest_dataset_train, residual_train, self.forest_container_tau, treatment_leaf_model, init_tau)

        # Run GFR (warm start) if specified
        if self.num_gfr > 0:
            gfr_indices = np.arange(self.num_gfr)
            for i in range(self.num_gfr):
                # Sample the prognostic forest
                forest_sampler_mu.sample_one_iteration(
                    self.forest_container_mu, forest_dataset_train, residual_train, cpp_rng, feature_types, 
                    cutpoint_grid_size, current_leaf_scale_mu, variable_weights_mu, a_forest, b_forest, 
                    current_sigma2, 0, True, True
                )

                # Sample variance parameters (if requested)
                if self.sample_sigma_global:
                    current_sigma2 = global_var_model.sample_one_iteration(residual_train, cpp_rng, a_global, b_global)
                    self.global_var_samples[i] = current_sigma2*self.y_std*self.y_std
                if self.sample_sigma_leaf_mu:
                    self.leaf_scale_mu_samples[i] = leaf_var_model_mu.sample_one_iteration(self.forest_container_mu, cpp_rng, a_leaf_mu, b_leaf_mu, i)
                    current_leaf_scale_mu[0,0] = self.leaf_scale_mu_samples[i]
                
                # Sample the treatment forest
                forest_sampler_tau.sample_one_iteration(
                    self.forest_container_tau, forest_dataset_train, residual_train, cpp_rng, feature_types, 
                    cutpoint_grid_size, current_leaf_scale_tau, variable_weights_tau, a_forest, b_forest, 
                    current_sigma2, treatment_leaf_model, True, True
                )
                
                # Sample variance parameters (if requested)
                if self.sample_sigma_global:
                    current_sigma2 = global_var_model.sample_one_iteration(residual_train, cpp_rng, a_global, b_global)
                    self.global_var_samples[i] = current_sigma2*self.y_std*self.y_std
                if self.sample_sigma_leaf_tau:
                    self.leaf_scale_tau_samples[i] = leaf_var_model_tau.sample_one_iteration(self.forest_container_tau, cpp_rng, a_leaf_tau, b_leaf_tau, i)
                    current_leaf_scale_tau[0,0] = self.leaf_scale_tau_samples[i]
                
                # Sample coding parameters (if requested)
                if self.adaptive_coding:
                    mu_x = self.forest_container_mu.predict_raw_single_forest(forest_dataset_train, i)
                    tau_x = np.squeeze(self.forest_container_tau.predict_raw_single_forest(forest_dataset_train, i))
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
                    forest_dataset_train.update_basis(tau_basis_train)
                    if self.has_test:
                        tau_basis_test = (1-np.squeeze(Z_test))*current_b_0 + np.squeeze(Z_test)*current_b_1
                        forest_dataset_test.update_basis(tau_basis_test)
                    self.b0_samples[i] = current_b_0
                    self.b1_samples[i] = current_b_1

                    # Update residual to reflect adjusted basis
                    forest_sampler_tau.update_residual(forest_dataset_train, residual_train, self.forest_container_tau, i)
        
        # Run MCMC
        if self.num_burnin + self.num_mcmc > 0:
            if self.num_burnin > 0:
                burnin_indices = np.arange(self.num_gfr, self.num_gfr + self.num_burnin)
            if self.num_mcmc > 0:
                mcmc_indices = np.arange(self.num_gfr + self.num_burnin, self.num_gfr + self.num_burnin + self.num_mcmc)
            for i in range(self.num_gfr, self.num_samples):
                # Sample the prognostic forest
                forest_sampler_mu.sample_one_iteration(
                    self.forest_container_mu, forest_dataset_train, residual_train, cpp_rng, feature_types, 
                    cutpoint_grid_size, current_leaf_scale_mu, variable_weights_mu, a_forest, b_forest, 
                    current_sigma2, 0, False, True
                )

                # Sample variance parameters (if requested)
                if self.sample_sigma_global:
                    current_sigma2 = global_var_model.sample_one_iteration(residual_train, cpp_rng, a_global, b_global)
                    self.global_var_samples[i] = current_sigma2*self.y_std*self.y_std
                if self.sample_sigma_leaf_mu:
                    self.leaf_scale_mu_samples[i] = leaf_var_model_mu.sample_one_iteration(self.forest_container_mu, cpp_rng, a_leaf_mu, b_leaf_mu, i)
                    current_leaf_scale_mu[0,0] = self.leaf_scale_mu_samples[i]
                
                # Sample the treatment forest
                forest_sampler_tau.sample_one_iteration(
                    self.forest_container_tau, forest_dataset_train, residual_train, cpp_rng, feature_types, 
                    cutpoint_grid_size, current_leaf_scale_tau, variable_weights_tau, a_forest, b_forest, 
                    current_sigma2, treatment_leaf_model, False, True
                )
                
                # Sample variance parameters (if requested)
                if self.sample_sigma_global:
                    current_sigma2 = global_var_model.sample_one_iteration(residual_train, cpp_rng, a_global, b_global)
                    self.global_var_samples[i] = current_sigma2*self.y_std*self.y_std
                if self.sample_sigma_leaf_tau:
                    self.leaf_scale_tau_samples[i] = leaf_var_model_tau.sample_one_iteration(self.forest_container_tau, cpp_rng, a_leaf_tau, b_leaf_tau, i)
                    current_leaf_scale_tau[0,0] = self.leaf_scale_tau_samples[i]                
                
                # Sample coding parameters (if requested)
                if self.adaptive_coding:
                    mu_x = self.forest_container_mu.predict_raw_single_forest(forest_dataset_train, i)
                    tau_x = np.squeeze(self.forest_container_tau.predict_raw_single_forest(forest_dataset_train, i))
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
                    forest_dataset_train.update_basis(tau_basis_train)
                    if self.has_test:
                        tau_basis_test = (1-np.squeeze(Z_test))*current_b_0 + np.squeeze(Z_test)*current_b_1
                        forest_dataset_test.update_basis(tau_basis_test)
                    self.b0_samples[i] = current_b_0
                    self.b1_samples[i] = current_b_1

                    # Update residual to reflect adjusted basis
                    forest_sampler_tau.update_residual(forest_dataset_train, residual_train, self.forest_container_tau, i)
        
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
        mu_raw = self.forest_container_mu.forest_container_cpp.Predict(forest_dataset_train.dataset_cpp)
        self.mu_hat_train = mu_raw[:,self.keep_indices]*self.y_std + self.y_bar
        tau_raw_train = self.forest_container_tau.forest_container_cpp.PredictRaw(forest_dataset_train.dataset_cpp)
        self.tau_hat_train = tau_raw_train[:,self.keep_indices,:]
        if self.adaptive_coding:
            adaptive_coding_weights = np.expand_dims(self.b1_samples[self.keep_indices] - self.b0_samples[self.keep_indices], axis=(0,2))
            self.tau_hat_train = self.tau_hat_train*adaptive_coding_weights
        self.tau_hat_train = np.squeeze(self.tau_hat_train*self.y_std)
        if self.multivariate_treatment:
            treatment_term_train = np.multiply(np.atleast_3d(Z_train).swapaxes(1,2),self.tau_hat_train).sum(axis=2)
        else:
            treatment_term_train = Z_train*np.squeeze(self.tau_hat_train)
        self.y_hat_train = self.mu_hat_train + treatment_term_train
        if self.has_test:
            mu_raw_test = self.forest_container_mu.forest_container_cpp.Predict(forest_dataset_test.dataset_cpp)
            self.mu_hat_test = mu_raw_test[:,self.keep_indices]*self.y_std + self.y_bar
            tau_raw_test = self.forest_container_tau.forest_container_cpp.PredictRaw(forest_dataset_test.dataset_cpp)
            self.tau_hat_test = tau_raw_test[:,self.keep_indices,:]
            if self.adaptive_coding:
                adaptive_coding_weights_test = np.expand_dims(self.b1_samples[self.keep_indices] - self.b0_samples[self.keep_indices], axis=(0,2))
                self.tau_hat_test = self.tau_hat_test*adaptive_coding_weights_test
            self.tau_hat_test = np.squeeze(self.tau_hat_test*self.y_std)
            if self.multivariate_treatment:
                treatment_term_test = np.multiply(np.atleast_3d(Z_test).swapaxes(1,2),self.tau_hat_test).sum(axis=2)
            else:
                treatment_term_test = Z_test*np.squeeze(self.tau_hat_test)
            self.y_hat_test = self.mu_hat_test + treatment_term_test
    
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
        else:
            if Z.ndim != 2:
                raise ValueError("treatment must have 1 or 2 dimensions")
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

        tau_raw = self.forest_container_tau.forest_container_cpp.PredictRaw(forest_dataset_tau.dataset_cpp)
        tau_raw = tau_raw[:,self.keep_indices,:]
        if self.adaptive_coding:
            adaptive_coding_weights = np.expand_dims(self.b1_samples[self.keep_indices] - self.b0_samples[self.keep_indices], axis=(0,2))
            tau_raw = tau_raw*adaptive_coding_weights
        tau_x = np.squeeze(tau_raw*self.y_std)

        # Return result matrix
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
        else:
            if Z.ndim != 2:
                raise ValueError("treatment must have 1 or 2 dimensions")
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
        
        # Compute predicted outcome and decomposed outcome model terms
        mu_raw = self.forest_container_mu.forest_container_cpp.Predict(forest_dataset_tau.dataset_cpp)
        mu_x = mu_raw[:,self.keep_indices]*self.y_std + self.y_bar
        tau_raw = self.forest_container_tau.forest_container_cpp.PredictRaw(forest_dataset_tau.dataset_cpp)
        tau_raw = tau_raw[:,self.keep_indices,:]
        if self.adaptive_coding:
            adaptive_coding_weights = np.expand_dims(self.b1_samples[self.keep_indices] - self.b0_samples[self.keep_indices], axis=(0,2))
            tau_raw = tau_raw*adaptive_coding_weights
        tau_x = np.squeeze(tau_raw*self.y_std)
        if Z.shape[1] > 1:
            treatment_term = np.multiply(np.atleast_3d(Z).swapaxes(1,2),tau_x).sum(axis=2)
        else:
            treatment_term = Z*np.squeeze(tau_x)
        yhat_x = mu_x + treatment_term
        
        # Return result matrices as a tuple
        return (tau_x, mu_x, yhat_x)
