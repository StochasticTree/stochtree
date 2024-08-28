import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from scipy.stats import gamma
from stochtree import CovariateTransformer
from stochtree import calibrate_global_error_variance
import pytest

class TestCalibration:
    def test_full_rank(self):
        n = 100
        p = 5
        nu = 3
        q = 0.9
        X = np.random.uniform(size=(n,p))
        y = 1 + X[:,0]*0.1 - X[:,1]*0.2 + np.random.normal(size=n)
        reg_model = linear_model.LinearRegression()
        reg_model.fit(X, y)
        mse = mean_squared_error(y, reg_model.predict(X))
        sigma2, lamb = calibrate_global_error_variance(X, y, None, nu, None, q, True)
        assert sigma2 == pytest.approx(mse)
        assert lamb == pytest.approx((mse*gamma.ppf(1-q,nu))/nu)

    def test_rank_deficient(self):
        n = 100
        p = 5
        nu = 3
        q = 0.9
        X = np.random.uniform(size=(n,p))
        X[:,4] = X[:,2]
        y = 1 + X[:,0]*0.1 - X[:,1]*0.2 + np.random.normal(size=n)
        reg_model = linear_model.LinearRegression()
        reg_model.fit(X, y)
        mse = mean_squared_error(y, reg_model.predict(X))
        if reg_model.rank_ < p:
          with pytest.warns(UserWarning):
            sigma2, lamb = calibrate_global_error_variance(X, y, None, nu, None, q, True)
        else:
          sigma2, lamb = calibrate_global_error_variance(X, y, None, nu, None, q, True)
        assert sigma2 == pytest.approx(mse)
        assert lamb == pytest.approx((mse*gamma.ppf(1-q,nu))/nu)

    def test_overdetermined(self):
        n = 100
        p = 101
        nu = 3
        q = 0.9
        X = np.random.uniform(size=(n,p))
        y = 1 + X[:,0]*0.1 - X[:,1]*0.2 + np.random.normal(size=n)
        reg_model = linear_model.LinearRegression()
        reg_model.fit(X, y)
        mse = mean_squared_error(y, reg_model.predict(X))
        with pytest.warns(UserWarning):
          sigma2, lamb = calibrate_global_error_variance(X, y, None, nu, None, q, True)
        assert sigma2 == pytest.approx(np.var(y))
        assert lamb == pytest.approx(np.var(y)*(gamma.ppf(1-q,nu))/nu)
