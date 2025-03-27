import warnings

import numpy as np
from scipy.stats import gamma
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


def calibrate_global_error_variance(
    X: np.array, y: np.array, nu: float = 3, q: float = 0.9, standardize: bool = True
) -> float:
    """Calibrates scale parameter of global error variance model as in Chipman et al (2010) by setting a value of lambda,
    part of the scale parameter in the `sigma2 ~ IG(nu/2, (nu*lambda)/2)` prior.

    Parameters
    ----------
    X : np.array
        Covariates to be used as split candidates for constructing trees.
    y : np.array
        Outcome to be used as target for constructing trees.
    nu : float, optional
        Shape parameter in the `IG(nu, nu*lamb)` global error variance model. Defaults to `3`.
    q : float, optional
        Quantile used to calibrated `lamb` as in Sparapani et al (2021). Defaults to `0.9`.
    standardize : bool, optional
        Whether or not `y` should be standardized before calibration. Defaults to `True`.

    Returns
    -------
    float
        Part of scale parameter of global error variance model
    """
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

    # Standardize outcome if necessary
    var_y = np.var(y)
    sd_y = np.std(y)
    mean_y = np.mean(y)
    if standardize:
        y_processed = (y_processed - mean_y) / sd_y

    # Fit a linear model of y ~ X
    lm_calibrator = linear_model.LinearRegression()
    lm_calibrator.fit(X_processed, y_processed)

    # Compute MSE
    y_hat_processed = lm_calibrator.predict(X_processed)
    mse = mean_squared_error(y_processed, y_hat_processed)

    # Check for overdetermination, revert to variance of y if model is overdetermined
    eps = np.finfo("double").eps
    if _is_model_overdetermined(lm_calibrator, n, mse, eps):
        sigma2hat = var_y
        warnings.warn(
            "Default calibration method for global error variance failed; covariate dimension exceeds number of samples. "
            "Initializing global error variance scale parameter based on the variance of the standardized outcome.",
            UserWarning,
        )
    else:
        sigma2hat = mse
        if _is_model_rank_deficient(lm_calibrator, p):
            warnings.warn(
                "Default calibration method for global error variance detected rank deficiency in covariate matrix. "
                "This should not impact the calibrated values, but may indicate the presence of duplicated covariates.",
                UserWarning,
            )

    # Calibrate lamb if no initial value is provided
    lamb = (sigma2hat * gamma.ppf(1 - q, nu)) / nu

    return lamb


def _is_model_overdetermined(
    reg_model: linear_model.LinearRegression, n: int, mse: float, eps: float
) -> bool:
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
