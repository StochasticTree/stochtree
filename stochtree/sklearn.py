import numpy as np
from scipy.stats import norm
from .bart import BARTModel
from .utils import OutcomeModel
from sklearn.utils._array_api import (
    get_namespace,
    indexing_dtype,
)
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, _fit_context
from sklearn.metrics import (
    accuracy_score,
    r2_score,
)
from sklearn.utils import check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data




class StochTreeBARTRegressor(RegressorMixin, BaseEstimator):
    """A scikit-learn-compatible estimator that implements a BART regression model.

    Parameters
    ----------
    num_gfr : int, default=10
        The number of grow-from-root (GFR) iterations to run of the BART model.
    num_burnin : int, default=0
        The number of MCMC iterations of the BART model that will be discarded as "burn-in" samples.
    num_mcmc : int, default=100
        The number of retained MCMC iterations to run of the BART model.
    general_params : dict, default=None
        General parameters for the BART model.
    mean_forest_params : dict, default=None
        Parameters for the mean forest.
    variance_forest_params : dict, default=None
        Parameters for the variance forest.
    rfx_params : dict, default=None
        Parameters for the random effects.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The covariates (or features) used to define tree partitions.

    y_ : ndarray, shape (n_samples,)
        The outcome variable (or labels) used to evaluate tree partitions.

    leaf_regression_basis_ : ndarray, shape (n_samples, n_bases)
        The basis functions used for leaf regression model if requested.

    rfx_group_ids_ : ndarray, shape (n_samples,)
        The group IDs for random effects if requested.

    rfx_basis_ : ndarray, shape (n_samples, n_rfx_bases)
        The basis functions used for random effects if requested.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    Examples
    --------
    >>> from sklearn.datasets import load_boston
    >>> from stochtree import StochTreeBARTRegressor
    >>> data = load_boston()
    >>> X = data.data
    >>> y = data.target
    >>> reg = StochTreeBARTRegressor()
    >>> reg.fit(X, y)
    >>> reg.predict(X)
    """

    # Define the type of each parameter
    _parameter_constraints = {
        "num_gfr": [int],
        "num_burnin": [int],
        "num_mcmc": [int],
        "general_params": [dict, None],
        "mean_forest_params": [dict, None],
        "variance_forest_params": [dict, None],
        "rfx_params": [dict, None],
    }

    def __init__(self, num_gfr = 10, num_burnin = 0, num_mcmc = 100, 
                 general_params = None, mean_forest_params = None, 
                 variance_forest_params = None, rfx_params = None):
        self.num_gfr = num_gfr
        self.num_burnin = num_burnin
        self.num_mcmc = num_mcmc
        self.general_params = general_params
        self.mean_forest_params = mean_forest_params
        self.variance_forest_params = variance_forest_params
        self.rfx_params = rfx_params

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, leaf_regression_basis=None, rfx_group_ids=None, rfx_basis=None):
        """Fit a BART regressor by sampling from its posterior.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The covariates used to train a BART forest.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The continuous outcomes used to train a BART forest.

        leaf_regression_basis : optional array-like, (n_samples, n_bases)
            The basis functions to use for leaf regression model, if requested.

        rfx_group_ids : optional array-like, (n_samples,)
            The group IDs for random effects, if requested.

        rfx_basis : optional array-like, (n_samples, n_rfx_bases)
            The basis functions to use for random effects, if requested.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X, y = validate_data(self, X, y, force_writeable=True, order="C", copy=True)
        if leaf_regression_basis is not None:
            leaf_regression_basis = check_array(leaf_regression_basis, force_writeable=True, order="C", copy=True)
        if rfx_group_ids is not None:
            rfx_group_ids = check_array(rfx_group_ids, force_writeable=True, order="C", copy=True)
        if rfx_basis is not None:
            rfx_basis = check_array(rfx_basis, force_writeable=True, order="C", copy=True)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns if hasattr(X, "columns") else None
        if X.shape[0] < 2:
            raise ValueError("n_samples=1")

        # Store the training data to predict later
        self.X_ = X
        self.y_ = y
        self.leaf_regression_basis_ = leaf_regression_basis
        self.rfx_group_ids_ = rfx_group_ids
        self.rfx_basis_ = rfx_basis

        # Parameter validation
        general_params = {
          **(self.general_params or {}),
          "outcome_model": OutcomeModel(outcome="continuous", link="identity")
        }

        # Initialize and sample a BART model
        self.model_ = BARTModel()
        self.model_.sample(X_train=X, y_train=y, leaf_basis_train=leaf_regression_basis,
                           rfx_group_ids_train=rfx_group_ids, rfx_basis_train=rfx_basis,
                           num_gfr=self.num_gfr, num_burnin=self.num_burnin, num_mcmc=self.num_mcmc,
                           general_params=general_params, mean_forest_params=self.mean_forest_params,
                           variance_forest_params=self.variance_forest_params, 
                           random_effects_params=self.rfx_params)

        # Return the estimator
        return self

    def predict(self, X, leaf_regression_basis=None, rfx_group_ids=None, rfx_basis=None):
        """Predict the outcome based on the provided test data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The covariates used to predict from a BART forest.

        leaf_regression_basis : optional array-like, (n_samples, n_bases)
            The basis functions to use for leaf regression model if requested.

        rfx_group_ids : optional array-like, (n_samples,)
            The group IDs for random effects if requested.

        rfx_basis : optional array-like, (n_samples, n_rfx_bases)
            The basis functions to use for random effects if requested.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of predicted target values.
        """
        # Check if fit had been called
        check_is_fitted(self)

        # Input validation
        X = validate_data(self, X, reset=False, force_writeable=True)
        if leaf_regression_basis is not None:
            leaf_regression_basis = check_array(leaf_regression_basis, force_writeable=True)
        if rfx_group_ids is not None:
            rfx_group_ids = check_array(rfx_group_ids, force_writeable=True)
        if rfx_basis is not None:
            rfx_basis = check_array(rfx_basis, force_writeable=True)

        # Compute and return predictions
        return self.model_.predict(X, leaf_basis=leaf_regression_basis, 
                                   rfx_group_ids=rfx_group_ids, rfx_basis=rfx_basis, 
                                   type = "mean", terms = "y_hat", scale = "linear")

    def score(self, X, y, leaf_regression_basis=None, rfx_group_ids=None, rfx_basis=None):
        """Compute and return the R2 for a BART regression model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The covariates used to train a BART forest.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The continuous outcomes used to train a BART forest.

        leaf_regression_basis : optional array-like, (n_samples, n_bases)
            The basis functions to use for leaf regression model, if requested.

        rfx_group_ids : optional array-like, (n_samples,)
            The group IDs for random effects, if requested.

        rfx_basis : optional array-like, (n_samples, n_rfx_bases)
            The basis functions to use for random effects, if requested.

        Returns
        -------
        score : float
            R^2 of `self.predict(X, leaf_regression_basis, rfx_group_ids, rfx_basis)` with respect to `y`.
        """
        # Check if fit had been called
        check_is_fitted(self)

        # Predict target values
        preds = self.predict(X, leaf_regression_basis=leaf_regression_basis,
                             rfx_group_ids=rfx_group_ids, rfx_basis=rfx_basis)

        # Compute R2
        return r2_score(y, preds)
    
    def __getstate__(self):
        """Prepare the estimator for pickling.
        
        We convert the BART model to its JSON representation.
        """
        state = self.__dict__.copy()
        
        # If the model has been fitted, serialize it to JSON
        if hasattr(self, 'model_') and self.model_ is not None:
            # Convert BARTModel to JSON
            state['_model_json'] = self.model_.to_json()
            # Remove the raw model object (which contains pointers to C++ objects)
            del state['model_']
        
        return state

    def __setstate__(self, state):
        """Restore the estimator from a pickled state.
        
        We reconstruct a BART model object from its JSON representation.
        """
        # If there's a serialized model, reconstruct it
        if '_model_json' in state:
            model_json = state.pop('_model_json')
            self.__dict__.update(state)
            self.model_ = BARTModel()
            self.model_.from_json(model_json)
        else:
            self.__dict__.update(state)


class StochTreeBARTBinaryClassifier(ClassifierMixin, BaseEstimator):
    """A scikit-learn-compatible estimator that implements a binary probit BART classifier.

    Parameters
    ----------
    num_gfr : int, default=10
        The number of grow-from-root (GFR) iterations to run of the BART model.
    num_burnin : int, default=0
        The number of MCMC iterations of the BART model that will be discarded as "burn-in" samples.
    num_mcmc : int, default=100
        The number of retained MCMC iterations to run of the BART model.
    general_params : dict, default=None
        General parameters for the BART model.
    mean_forest_params : dict, default=None
        Parameters for the mean forest.
    variance_forest_params : dict, default=None
        Parameters for the variance forest.
    rfx_params : dict, default=None
        Parameters for the random effects.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The covariates (or features) used to define tree partitions.

    y_ : ndarray, shape (n_samples,)
        The outcome variable (or labels) used to evaluate tree partitions.

    leaf_regression_basis_ : ndarray, shape (n_samples, n_bases)
        The basis functions used for leaf regression model if requested.

    rfx_group_ids_ : ndarray, shape (n_samples,)
        The group IDs for random effects if requested.

    rfx_basis_ : ndarray, shape (n_samples, n_rfx_bases)
        The basis functions used for random effects if requested.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    Examples
    --------
    >>> from sklearn.datasets import load_wine
    >>> from stochtree import StochTreeBARTBinaryClassifier
    >>> data = load_wine()
    >>> X = data.data
    >>> y = data.target
    >>> clf = StochTreeBARTBinaryClassifier()
    >>> clf.fit(X, y)
    >>> clf.predict(X)
    """

    # Define the type of each parameter
    _parameter_constraints = {
        "num_gfr": [int],
        "num_burnin": [int],
        "num_mcmc": [int],
        "general_params": [dict, None],
        "mean_forest_params": [dict, None],
        "variance_forest_params": [dict, None],
        "rfx_params": [dict, None],
    }

    def __init__(self, num_gfr = 10, num_burnin = 0, num_mcmc = 100, 
                 general_params = None, mean_forest_params = None, 
                 variance_forest_params = None, rfx_params = None):
        self.num_gfr = num_gfr
        self.num_burnin = num_burnin
        self.num_mcmc = num_mcmc
        self.general_params = general_params
        self.mean_forest_params = mean_forest_params
        self.variance_forest_params = variance_forest_params
        self.rfx_params = rfx_params

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, leaf_regression_basis=None, rfx_group_ids=None, rfx_basis=None):
        """Fit a BART classifier by sampling from its posterior.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The covariates used to train a BART forest.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The continuous outcomes used to train a BART forest.

        leaf_regression_basis : optional array-like, (n_samples, n_bases)
            The basis functions to use for leaf regression model, if requested.

        rfx_group_ids : optional array-like, (n_samples,)
            The group IDs for random effects, if requested.

        rfx_basis : optional array-like, (n_samples, n_rfx_bases)
            The basis functions to use for random effects, if requested.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X, y = validate_data(self, X, y, force_writeable=True, order="C", copy=True)
        check_classification_targets(y)
        if leaf_regression_basis is not None:
            leaf_regression_basis = check_array(leaf_regression_basis, force_writeable=True, order="C", copy=True)
        if rfx_group_ids is not None:
            rfx_group_ids = check_array(rfx_group_ids, force_writeable=True, order="C", copy=True)
        if rfx_basis is not None:
            rfx_basis = check_array(rfx_basis, force_writeable=True, order="C", copy=True)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns if hasattr(X, "columns") else None
        if X.shape[0] < 2:
            raise ValueError("n_samples=1")

        # Store the training data to predict later
        self.X_ = X
        self.y_ = y
        self.leaf_regression_basis_ = leaf_regression_basis
        self.rfx_group_ids_ = rfx_group_ids
        self.rfx_basis_ = rfx_basis

        # Initialize and sample a BART model
        general_params = {
          **(self.general_params or {}),
          "outcome_model": OutcomeModel(outcome="binary", link="probit"),
          "sample_sigma2_global": False,
        }
        self.model_ = BARTModel()
        self.model_.sample(X_train=X, y_train=y, leaf_basis_train=leaf_regression_basis,
                           rfx_group_ids_train=rfx_group_ids, rfx_basis_train=rfx_basis,
                           num_gfr=self.num_gfr, num_burnin=self.num_burnin, num_mcmc=self.num_mcmc,
                           general_params=general_params, mean_forest_params=self.mean_forest_params, 
                           variance_forest_params=self.variance_forest_params, 
                           random_effects_params=self.rfx_params)

        # Return the classifier
        return self

    def decision_function(self, X, leaf_regression_basis=None, rfx_group_ids=None, rfx_basis=None):
        """Evaluate the (linear-scale) decision function for the given input samples.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The covariates used to predict a BART forest.

        leaf_regression_basis : optional array-like, (n_samples, n_bases)
            The basis functions to use for leaf regression model, if requested.

        rfx_group_ids : optional array-like, (n_samples,)
            The group IDs for random effects, if requested.

        rfx_basis : optional array-like, (n_samples, n_rfx_bases)
            The basis functions to use for random effects, if requested.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of predicted target values.
        """
        # Check if fit had been called
        check_is_fitted(self)

        # Input validation
        X = validate_data(self, X, reset=False, force_writeable=True)
        if leaf_regression_basis is not None:
            leaf_regression_basis = check_array(leaf_regression_basis, reset=False, force_writeable=True)
        if rfx_group_ids is not None:
            rfx_group_ids = check_array(rfx_group_ids, reset=False, force_writeable=True)
        if rfx_basis is not None:
            rfx_basis = check_array(rfx_basis, reset=False, force_writeable=True)

        # Compute and return predicted probabilities
        return self.model_.predict(X, leaf_basis=leaf_regression_basis,
                                   rfx_group_ids=rfx_group_ids, rfx_basis=rfx_basis,
                                   type="mean", terms="y_hat", scale="linear")

    def predict(self, X, leaf_regression_basis=None, rfx_group_ids=None, rfx_basis=None):
        """Predict the target classes for the given input samples.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The covariates used to predict a BART forest.

        leaf_regression_basis : optional array-like, (n_samples, n_bases)
            The basis functions to use for leaf regression model, if requested.

        rfx_group_ids : optional array-like, (n_samples,)
            The group IDs for random effects, if requested.

        rfx_basis : optional array-like, (n_samples, n_rfx_bases)
            The basis functions to use for random effects, if requested.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of predicted target values.
        """
        # Check if fit had been called
        check_is_fitted(self)

        # Determine the class with the largest predicted score
        # From https://github.com/scikit-learn/scikit-learn/blob/3c5f668eb1131499e3db2fc50c1f99ee0b670756/sklearn/linear_model/_base.py#L372
        xp, _ = get_namespace(X)
        scores = self.decision_function(X, leaf_regression_basis=leaf_regression_basis, 
                                        rfx_group_ids=rfx_group_ids, rfx_basis=rfx_basis)
        if len(scores.shape) == 1:
            indices = xp.astype(scores > 0, indexing_dtype(xp))
        else:
            indices = xp.argmax(scores, axis=1)

        # Return the associated class labels
        return xp.take(self.classes_, indices, axis=0)

    def predict_proba(self, X, leaf_regression_basis=None, rfx_group_ids=None, rfx_basis=None):
        """Predict the target probabilities for the given input samples.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The covariates used to predict a BART forest.

        leaf_regression_basis : optional array-like, (n_samples, n_bases)
            The basis functions to use for leaf regression model, if requested.

        rfx_group_ids : optional array-like, (n_samples,)
            The group IDs for random effects, if requested.

        rfx_basis : optional array-like, (n_samples, n_rfx_bases)
            The basis functions to use for random effects, if requested.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of predicted target values.
        """
        # Check if fit had been called
        check_is_fitted(self)

        # Input validation
        X = validate_data(self, X, reset=False, force_writeable=True, order="C", copy=True)
        if leaf_regression_basis is not None:
            leaf_regression_basis = check_array(leaf_regression_basis, reset=False, force_writeable=True, order="C", copy=True)
        if rfx_group_ids is not None:
            rfx_group_ids = check_array(rfx_group_ids, reset=False, force_writeable=True, order="C", copy=True)
        if rfx_basis is not None:
            rfx_basis = check_array(rfx_basis, reset=False, force_writeable=True, order="C", copy=True)

        # Compute and return predicted class probabilities
        scores = self.model_.predict(X, leaf_basis=leaf_regression_basis,
                                     rfx_group_ids=rfx_group_ids, rfx_basis=rfx_basis,
                                     type="mean", terms="y_hat", scale="linear")
        probs = norm.cdf(scores)
        return np.vstack([1 - probs, probs]).T

    def score(self, X, y, leaf_regression_basis=None, rfx_group_ids=None, rfx_basis=None):
        """A reference implementation of a scoring function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The covariates used to train a BART forest.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The continuous outcomes used to train a BART forest.

        leaf_regression_basis : optional array-like, (n_samples, n_bases)
            The basis functions to use for leaf regression model, if requested.

        rfx_group_ids : optional array-like, (n_samples,)
            The group IDs for random effects, if requested.

        rfx_basis : optional array-like, (n_samples, n_rfx_bases)
            The basis functions to use for random effects, if requested.

        Returns
        -------
        score : float
            R^2 of `self.predict(X, leaf_regression_basis, rfx_group_ids, rfx_basis)` with respect to `y`.
        """
        # Check if fit had been called
        check_is_fitted(self)

        # Predict target values
        preds = self.predict(X, leaf_regression_basis=leaf_regression_basis,
                             rfx_group_ids=rfx_group_ids, rfx_basis=rfx_basis)

        # Compute R2
        return accuracy_score(y, preds)
    
    def __getstate__(self):
        """Prepare the estimator for pickling.
        
        We convert the BART model to its JSON representation.
        """
        state = self.__dict__.copy()
        
        # If the model has been fitted, serialize it to JSON
        if hasattr(self, 'model_') and self.model_ is not None:
            # Convert BARTModel to JSON
            state['_model_json'] = self.model_.to_json()
            # Remove the raw model object (which contains pointers to C++ objects)
            del state['model_']
        
        return state

    def __setstate__(self, state):
        """Restore the estimator from a pickled state.
        
        We reconstruct a BART model object from its JSON representation.
        """
        # If there's a serialized model, reconstruct it
        if '_model_json' in state:
            model_json = state.pop('_model_json')
            self.__dict__.update(state)
            self.model_ = BARTModel()
            self.model_.from_json(model_json)
        else:
            self.__dict__.update(state)
