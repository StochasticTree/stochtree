import warnings
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from scipy.stats import gamma


def calibrate_global_error_variance(X: np.array, y: np.array, sigma2: float = None, nu: float = 3, lamb: float = None, q: float = 0.9, lm_calibrate: bool = True) -> tuple:
    """Calibrates global error variance model by setting an initial value of sigma^2 (the parameter itself) and setting a value of lambda, part of the scale parameter in the 
    ``sigma2 ~ IG(nu/2, (nu*lambda)/2)`` prior.

    Parameters
    ----------
    X : :obj:`np.array`
        Covariates to be used as split candidates for constructing trees.
    y : :obj:`np.array`
        Outcome to be used as target for constructing trees.
    sigma2 : :obj:`float`, optional
        Starting value of global variance parameter. Calibrated internally if not set here.
    nu : :obj:`float`, optional
        Shape parameter in the ``IG(nu, nu*lamb)`` global error variance model. Defaults to ``3``.
    lamb : :obj:`float`, optional
        Component of the scale parameter in the ``IG(nu, nu*lambda)`` global error variance prior. If not specified, this is calibrated as in Sparapani et al (2021).
    q : :obj:`float`, optional
        Quantile used to calibrated ``lamb`` as in Sparapani et al (2021). Defaults to ``0.9``.
    lm_calibrate : :obj:`bool`, optional
        Whether or not to calibrate sigma2 based on a linear model of ``y`` given ``X``. If ``True``, uses the linear model calibration technique in Sparapani et al (2021), otherwise uses `np.var(y)`. Defaults to ``True``.
    
    Returns
    -------
    (sigma2, lamb) : :obj:`tuple` of :obj:`float`
        Tuple containing an initial value of sigma^2 (global error variance) and lambda (part of scale parameter of global error variance model)
    """
    # Initialize sigma if no initial value is provided
    var_y = np.var(y)
    if sigma2 is None:
        if lm_calibrate:
            # Convert X and y to expected dimensions
            if X.ndim == 2:
                X_processed = X
            elif X.ndim == 1:
                X_processed = np.expand_dims(X, 1)
            else: 
                raise ValueError("X must be a 1 or 2 dimensional numpy array")
            n, p = X_processed.shape

            if y.ndim == 2:
                y_processed = np.squeeze(y)
            elif y.ndim == 1:
                y_processed = y
            else: 
                raise ValueError("y must be a 1 or 2 dimensional numpy array")
            
            # Fit a linear model of y ~ X 
            lm_calibrator = linear_model.LinearRegression()
            lm_calibrator.fit(X_processed, y_processed)
            
            # Compute MSE
            y_hat_processed = lm_calibrator.predict(X_processed)
            mse = mean_squared_error(y_processed, y_hat_processed)
            
            # Check for overdetermination, revert to variance of y if model is overdetermined
            eps = np.finfo("double").eps
            if _is_model_overdetermined(lm_calibrator, n, mse, eps):
                sigma2 = var_y
                warnings.warn("Default calibration method for global error variance failed; covariate dimension exceeds number of samples. "
                              "Initializing global error variance based on the variance of the standardized outcome.")
            else:
              sigma2 = mse
              if _is_model_rank_deficient(lm_calibrator, p):
                  warnings.warn("Default calibration method for global error variance detected rank deficiency in covariate matrix. "
                                "This should not impact the calibrated values, but may indicate the presence of duplicated covariates.")
        else:
            sigma2 = var_y
    
    # Calibrate lamb if no initial value is provided
    if lamb is None:
        lamb = (sigma2*gamma.ppf(1-q,nu))/nu
    
    return (sigma2, lamb)

def _is_model_overdetermined(reg_model: linear_model.LinearRegression, n: int, mse: float, eps: float) -> bool:
    
    if reg_model.rank_ == n:
        return True
    elif np.abs(mse) < eps:
        return True
    else:
        return False

def _is_model_rank_deficient(reg_model: linear_model.LinearRegression, p: int) -> bool:
    if reg_model.rank_ < p:
        return True
    else:
        return False
