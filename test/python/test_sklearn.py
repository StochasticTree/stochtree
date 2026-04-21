import pickle
import numpy as np
import pytest
from scipy import sparse
from sklearn.model_selection import cross_val_score

from stochtree import StochTreeBARTRegressor, StochTreeBARTBinaryClassifier


@pytest.fixture
def regression_test_data():
    rng = np.random.default_rng(42)
    n, p = 200, 5
    X = rng.uniform(low=0.,high=1.,size=(n,p))
    y = X[:,0] * 2 + rng.standard_normal(n)
    return X, y


@pytest.fixture
def binary_classification_test_data():
    rng = np.random.default_rng(42)
    n, p = 200, 5
    X = rng.uniform(low=0.,high=1.,size=(n,p))
    Z = X[:,0] * 2 - 1 + rng.standard_normal(n)
    y = 1 * (Z > 0)
    return X, y


@pytest.fixture
def multiclass_classification_test_data():
    rng = np.random.default_rng(42)
    n, p = 200, 5
    X = rng.uniform(low=0.,high=1.,size=(n,p))
    Z = X[:,0] * 2 - 1 + rng.standard_normal(n)
    y = 1 * (Z > -0.333) + 1 * (Z > 0.333)
    return X, y


class TestSklearnRegressor:
    def test_fit_predict(self, regression_test_data):
        X, y = regression_test_data
        est = StochTreeBARTRegressor()
        est.fit(X, y)
        preds = est.predict(X)
        assert preds.shape == y.shape
        with pytest.raises(ValueError):
            est.fit(X, y[:10])
        with pytest.raises(TypeError):
            est.fit(sparse.csr_matrix(X), y)

    def test_cv(self, regression_test_data):
        X, y = regression_test_data
        est = StochTreeBARTRegressor()
        cross_val_score(est, X, y, cv=3)

    def test_serialization(self, regression_test_data):
        X, y = regression_test_data
        est = StochTreeBARTRegressor()
        est.fit(X, y)
        preds = est.predict(X)
        buffer = pickle.dumps(est)
        est_reload = pickle.loads(buffer)
        preds_reload = est_reload.predict(X)
        np.testing.assert_allclose(preds, preds_reload)


class TestSklearnClassifier:
    def test_fit_predict(self, binary_classification_test_data, multiclass_classification_test_data):
        X_bin, y_bin = binary_classification_test_data
        X_mult, y_mult = multiclass_classification_test_data
        est = StochTreeBARTBinaryClassifier()
        est.fit(X_bin, y_bin)
        dec = est.decision_function(X_bin)
        probs = est.predict_proba(X_bin)
        preds = est.predict(X_bin)
        assert dec.shape == y_bin.shape
        assert probs.shape == (y_bin.shape[0], 2)
        assert preds.shape == y_bin.shape
        with pytest.raises(ValueError):
            est.fit(X_mult, y_mult)
        with pytest.raises(TypeError):
            est.fit(sparse.csr_matrix(X_bin), y_bin)

    def test_cv(self, binary_classification_test_data):
        X, y = binary_classification_test_data
        est = StochTreeBARTBinaryClassifier()
        cross_val_score(est, X, y, cv=3)

    def test_serialization(self, binary_classification_test_data):
        X, y = binary_classification_test_data
        est = StochTreeBARTBinaryClassifier()
        est.fit(X, y)
        preds = est.predict(X)
        buffer = pickle.dumps(est)
        est_reload = pickle.loads(buffer)
        preds_reload = est_reload.predict(X)
        np.testing.assert_allclose(preds, preds_reload)
