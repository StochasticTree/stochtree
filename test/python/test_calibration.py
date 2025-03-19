import numpy as np
import pytest
from scipy.stats import gamma
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from stochtree import calibrate_global_error_variance


class TestCalibration:
    def test_full_rank(self):
        n = 100
        p = 5
        nu = 3
        q = 0.9
        X = np.random.uniform(size=(n, p))
        y = 1 + X[:, 0] * 0.1 - X[:, 1] * 0.2 + np.random.normal(size=n)
        y_std = (y - np.mean(y)) / np.std(y)
        reg_model = linear_model.LinearRegression()
        reg_model.fit(X, y_std)
        mse = mean_squared_error(y_std, reg_model.predict(X))
        lamb = calibrate_global_error_variance(X=X, y=y, nu=nu, q=q, standardize=True)
        assert lamb == pytest.approx((mse * gamma.ppf(1 - q, nu)) / nu)

    def test_rank_deficient(self):
        n = 100
        p = 5
        nu = 3
        q = 0.9
        X = np.random.uniform(size=(n, p))
        X[:, 4] = X[:, 2]
        y = 1 + X[:, 0] * 0.1 - X[:, 1] * 0.2 + np.random.normal(size=n)
        y_std = (y - np.mean(y)) / np.std(y)
        reg_model = linear_model.LinearRegression()
        reg_model.fit(X, y_std)
        mse = mean_squared_error(y_std, reg_model.predict(X))
        if reg_model.rank_ < p:
            with pytest.warns(UserWarning):
                lamb = calibrate_global_error_variance(
                    X=X, y=y, nu=nu, q=q, standardize=True
                )
        else:
            lamb = calibrate_global_error_variance(
                X=X, y=y, nu=nu, q=q, standardize=True
            )
        assert lamb == pytest.approx((mse * gamma.ppf(1 - q, nu)) / nu)

    def test_overdetermined(self):
        n = 100
        p = 101
        nu = 3
        q = 0.9
        X = np.random.uniform(size=(n, p))
        y = 1 + X[:, 0] * 0.1 - X[:, 1] * 0.2 + np.random.normal(size=n)
        y_std = (y - np.mean(y)) / np.std(y)
        reg_model = linear_model.LinearRegression()
        reg_model.fit(X, y_std)
        with pytest.warns(UserWarning):
            lamb = calibrate_global_error_variance(
                X=X, y=y, nu=nu, q=q, standardize=True
            )
        assert lamb == pytest.approx(np.var(y) * (gamma.ppf(1 - q, nu)) / nu)
